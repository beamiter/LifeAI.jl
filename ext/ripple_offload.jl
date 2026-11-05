LifeAI._ripple_release_weight!(weight::CUDA.CuArray) = (CUDA.unsafe_free!(weight); nothing)
LifeAI._ripple_synchronize_weight(weight::CUDA.CuArray) = CUDA.synchronize()

# Single-token packed INT4 / Float32 GEMV: neighboring lanes read neighboring output rows,
# while eight warps accumulate disjoint input-column pairs for each row.
# Packed weights/scales are consumed directly; no Float32 weight matrix exists.
function _ripple_int4_gemv_kernel!(output,packed,scales,input,group)
    thread=CUDA.threadIdx().x
    row_lane=(thread-Int32(1))%Int32(32)+Int32(1)
    column_lane=(thread-Int32(1))÷Int32(32)
    row=(CUDA.blockIdx().x-Int32(1))*Int32(32)+row_lane
    total=0f0
    if row<=size(packed,1)
        pair=column_lane+Int32(1)
        while pair<=size(packed,2)
            @inbounds begin
                code=packed[row,pair]
                scale=scales[row,(Int32(2)*pair-Int32(2))÷group+Int32(1)]
                low=(Float32(code&UInt8(15))-8f0)*scale
                high=(Float32(code>>4)-8f0)*scale
                total=muladd(low,input[Int32(2)*pair-Int32(1),1],total)
                total=muladd(high,input[Int32(2)*pair,1],total)
            end
            pair+=Int32(8)
        end
    end
    partials=CUDA.CuStaticSharedArray(Float32,256)
    @inbounds partials[thread]=total
    CUDA.sync_threads()
    if thread<=32 && row<=size(packed,1)
        sum=0f0
        for lane in Int32(0):Int32(7)
            @inbounds sum+=partials[thread+Int32(32)*lane]
        end
        @inbounds output[row,1]=sum
    end
    return
end

function LifeAI._ripple_linear(weight::LifeAI.Int4GroupWeight{P,S},input::CUDA.CuArray{Float32,2}) where {P<:CUDA.CuArray{UInt8,2},S<:CUDA.CuArray{Float32,2}}
    size(input,2)==1 || return LifeAI._flint_linear(weight,input)
    size(input,1)==weight.in_dim || throw(DimensionMismatch("Ripple INT4 GEMV input width mismatch"))
    output=similar(input,Float32,(size(weight.packed,1),1))
    CUDA.@cuda threads=256 blocks=cld(size(weight.packed,1),32) _ripple_int4_gemv_kernel!(
        output,weight.packed,weight.scale,input,Int32(weight.group))
    return output
end

# Safetensors is row-major. Upload the original contiguous bytes and reverse
# dimensions on GPU instead of transposing multi-GB matrices on one host core.
function LifeAI._ripple_read_device_tensor(reader::LifeAI.HFSafetensorsReader,name,
        dtype,::typeof(CUDA.cu))
    location=reader.locations[name]
    source_type=location.dtype=="BF16" ? BFloat16 : location.dtype=="F32" ? Float32 :
        throw(ArgumentError("unsupported Ripple source dtype $(location.dtype)"))
    raw=open(location.path,"r") do io
        seek(io,location.data_base+location.data_start)
        read(io,location.data_stop-location.data_start)
    end
    length(raw)==location.data_stop-location.data_start || throw(ArgumentError("truncated Ripple tensor $name"))
    shape=Tuple(reverse(location.shape))
    uploaded=GC.@preserve raw begin
        host=unsafe_wrap(Array,Ptr{source_type}(pointer(raw)),shape;own=false)
        result=CUDA.CuArray(host)
        CUDA.synchronize()
        result
    end
    converted=dtype===source_type ? uploaded : dtype.(uploaded)
    output=length(shape)>1 ? permutedims(converted,Tuple(reverse(1:length(shape)))) : converted
    CUDA.synchronize()
    output===converted || CUDA.unsafe_free!(converted)
    converted===uploaded || CUDA.unsafe_free!(uploaded)
    return output
end
