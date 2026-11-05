# The -Inf softmax sink is a constant for each local head count. Create it
# once on its owning device, completing initialization before sharing streams.
const _RIPPLE_SINK_LOCK=ReentrantLock()
const _RIPPLE_SINKS=Dict{CUDA.CuContext,Dict{Int,CUDA.CuVector{Float32}}}()
function LifeAI._ripple_sinks(input::CUDA.CuArray{Float32},heads)
    context=CUDA.context()
    CUDA.context(input)==context || throw(ArgumentError("Ripple constants require the activation CUDA context"))
    lock(_RIPPLE_SINK_LOCK) do
        cache=get!(_RIPPLE_SINKS,context) do
            Dict{Int,CUDA.CuVector{Float32}}()
        end
        get!(cache,Int(heads)) do
            value=CUDA.fill(-Inf32,Int(heads));CUDA.synchronize();value
        end
    end
end

# Original BF16 values with direct Float32-accumulating decode. This dense
# wrapper is distinct from quantized weights and never changes activation dtype.
struct RippleBF16Projection{A}
    weight::A
end
LifeAI._ripple_projection_array(weight::RippleBF16Projection)=weight.weight
LifeAI._ripple_bf16_projection(weight::CUDA.CuArray{BFloat16,2})=RippleBF16Projection(weight)
LifeAI.quantized_parameter_bytes(weight::RippleBF16Projection)=LifeAI.quantized_parameter_bytes(weight.weight)

# Column-major rows are contiguous. A warp owns 32 output rows and the eight
# warps in a block partition the input columns. No expanded weight matrix exists.
function _ripple_bf16_gemv_kernel!(output,weight,input)
    thread=CUDA.threadIdx().x
    lane=(thread-Int32(1))%Int32(32)+Int32(1)
    part=(thread-Int32(1))÷Int32(32)
    row=(CUDA.blockIdx().x-Int32(1))*Int32(32)+lane
    total=0f0
    if row<=size(weight,1)
        column=part+Int32(1)
        while column<=size(weight,2)
            @inbounds total=muladd(Float32(weight[row,column]),input[column,1],total)
            column+=Int32(8)
        end
    end
    partials=CUDA.CuStaticSharedArray(Float32,256)
    @inbounds partials[thread]=total
    CUDA.sync_threads()
    if thread<=32 && row<=size(weight,1)
        value=0f0
        for offset in Int32(0):Int32(7)
            @inbounds value+=partials[thread+Int32(32)*offset]
        end
        @inbounds output[row,1]=value
    end
    return
end

function _ripple_bf16_splitk_kernel!(partials,weight,input,tile)
    thread=CUDA.threadIdx().x
    lane=(thread-Int32(1))%Int32(32)+Int32(1)
    part=(thread-Int32(1))÷Int32(32)
    row=(CUDA.blockIdx().x-Int32(1))*Int32(32)+lane
    split=CUDA.blockIdx().y
    first=(split-Int32(1))*tile
    total=0f0
    if row<=size(weight,1)
        column=first+part+Int32(1)
        stop=min(first+tile,Int32(size(weight,2)))
        while column<=stop
            @inbounds total=muladd(Float32(weight[row,column]),input[column,1],total)
            column+=Int32(8)
        end
    end
    shared=CUDA.CuStaticSharedArray(Float32,256)
    @inbounds shared[thread]=total
    CUDA.sync_threads()
    if thread<=32 && row<=size(weight,1)
        value=0f0
        for offset in Int32(0):Int32(7)
            @inbounds value+=shared[thread+Int32(32)*offset]
        end
        @inbounds partials[row,split]=value
    end
    return
end
function _ripple_bf16_splitk_merge_kernel!(output,partials)
    row=(CUDA.blockIdx().x-Int32(1))*CUDA.blockDim().x+CUDA.threadIdx().x
    if row<=size(output,1)
        value=0f0
        for split in Int32(1):Int32(size(partials,2))
            @inbounds value+=partials[row,split]
        end
        @inbounds output[row,1]=value
    end
    return
end

function LifeAI._ripple_linear(projection::RippleBF16Projection,input::CUDA.CuArray{Float32,2})
    weight=projection.weight
    size(input,1)==size(weight,2) || throw(DimensionMismatch("Ripple BF16 projection input width mismatch"))
    if size(input,2)==1
        output=similar(input,Float32,(size(weight,1),1))
        if size(weight,2)<=256
            CUDA.@cuda threads=256 blocks=cld(size(weight,1),32) _ripple_bf16_gemv_kernel!(output,weight,input)
        else
            splits=cld(size(weight,2),256)
            partials=similar(input,Float32,(size(weight,1),splits))
            CUDA.@cuda threads=256 blocks=(cld(size(weight,1),32),splits) _ripple_bf16_splitk_kernel!(partials,weight,input,Int32(256))
            CUDA.@cuda threads=256 blocks=cld(size(weight,1),256) _ripple_bf16_splitk_merge_kernel!(output,partials)
            CUDA.unsafe_free!(partials)
        end
        return output
    end
    # Prefill keeps the validated FP32 GEMM. Explicit lifetime ends on this
    # stream after GEMM, avoiding GC-retained copies of all model projections.
    converted=Float32.(weight)
    try
        return converted*input
    finally
        CUDA.unsafe_free!(converted)
    end
end
