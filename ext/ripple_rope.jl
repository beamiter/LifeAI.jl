# Independent from Flint's legacy RoPE cache: Ripple uses the inverse-frequency
# arithmetic of its own published forward, including at long absolute positions.
const _RIPPLE_ROPE_WINDOW=4096
const _RIPPLE_ROPE_LOCK=ReentrantLock()
const _RIPPLE_ROPE_TABLES=Dict{CUDA.CuContext,Dict{Tuple{Int,Float32,Int},Any}}()
const _RIPPLE_ROPE_ORDER=Dict{CUDA.CuContext,Vector{Tuple{Int,Float32,Int}}}()

# PyTorch CUDA and CPU pow differ by one ULP for several frequencies. At1Mi
# that becomes a visible phase error, so calculate power/reciprocal/trig on the
# same CUDA backend as the published forward rather than uploading CPU tables.
function _ripple_rope_device_frequencies(dim,theta)
    !(theta isa Bool) || throw(ArgumentError("Ripple RoPE theta cannot be Bool"))
    base=Float32(theta)
    isfinite(base) && base>0f0 || throw(ArgumentError("Ripple RoPE theta must be finite and positive"))
    dim>0 && iseven(dim) || throw(ArgumentError("Ripple RoPE head dimension must be positive and even"))
    exponents=CUDA.CuArray(Float32.(0:2:Int(dim)-2)./Float32(dim))
    frequencies=1f0 ./ (base .^ exponents)
    CUDA.synchronize()
    CUDA.unsafe_free!(exponents)
    return frequencies
end
function LifeAI._ripple_rope_frequencies(dim::Integer,theta::Real,input::CUDA.CuArray{Float32})
    CUDA.context(input)==CUDA.context() || throw(ArgumentError("Ripple RoPE coefficients require the active CUDA context"))
    return _ripple_rope_device_frequencies(dim,theta)
end
function _ripple_rope_device_tables(dim,theta,positions)
    half=dim÷2
    frequencies=_ripple_rope_device_frequencies(dim,theta)
    device_positions=CUDA.CuArray(Float32.(positions))
    angles=reshape(frequencies,half,1).*reshape(device_positions,1,:)
    cosine=cos.(angles);sine=sin.(angles)
    CUDA.synchronize()
    CUDA.unsafe_free!(frequencies);CUDA.unsafe_free!(device_positions);CUDA.unsafe_free!(angles)
    return cosine,sine
end

function _ripple_rope_tables(dim,theta,base)
    context=CUDA.context();key=(Int(dim),Float32(theta),Int(base))
    lock(_RIPPLE_ROPE_LOCK) do
        tables=get!(_RIPPLE_ROPE_TABLES,context) do
            Dict{Tuple{Int,Float32,Int},Any}()
        end
        order=get!(_RIPPLE_ROPE_ORDER,context) do
            Tuple{Int,Float32,Int}[]
        end
        get!(tables,key) do
            if length(tables)>=8
                # Tables may be shared by different streams on this device.
                CUDA.device_synchronize()
                delete!(tables,popfirst!(order))
            end
            cosine,sine=_ripple_rope_device_tables(dim,theta,base:base+_RIPPLE_ROPE_WINDOW-1)
            push!(order,key)
            (cosine,sine)
        end
    end
end

function _ripple_rope_kernel!(output,input,cosine,sine,first_position,base)
    index=(CUDA.blockIdx().x-Int32(1))*CUDA.blockDim().x+CUDA.threadIdx().x
    stride=CUDA.blockDim().x*CUDA.gridDim().x
    half=Int32(size(input,1)÷2);heads=Int32(size(input,2))
    total=half*heads*Int32(size(input,3))
    while index<=total
        row=(index-Int32(1))%half+Int32(1)
        head=(index-Int32(1))÷half%heads+Int32(1)
        token=(index-Int32(1))÷(half*heads)
        position=first_position+token-base+Int32(1)
        @inbounds begin
            c,s=cosine[row,position],sine[row,position]
            a,b=input[row,head,token+Int32(1)],input[row+half,head,token+Int32(1)]
            output[row,head,token+Int32(1)]=a*c-b*s
            output[row+half,head,token+Int32(1)]=b*c+a*s
        end
        index+=stride
    end
    return
end

function LifeAI._ripple_rope(input::CUDA.CuArray{Float32,3},positions::AbstractUnitRange{<:Integer},theta)
    LifeAI._ripple_rope_positions(input,positions)
    isempty(positions) && return copy(input)
    CUDA.context(input)==CUDA.context() || throw(ArgumentError("Ripple RoPE input must belong to the active CUDA context"))
    dim,heads,tokens=size(input)
    first_position,last_position=Int(first(positions)),Int(last(positions))
    base=fld(first_position,_RIPPLE_ROPE_WINDOW)*_RIPPLE_ROPE_WINDOW
    # Cross-window prefill chunks use exact-range device coefficients. Never
    # fall back to the CPU power backend at a table boundary.
    if last_position>=base+_RIPPLE_ROPE_WINDOW
        cosine,sine=_ripple_rope_device_tables(dim,theta,positions)
        base=first_position
    else
        cosine,sine=_ripple_rope_tables(dim,theta,base)
    end
    (dim÷2)*heads*tokens<=typemax(Int32) || throw(ArgumentError("Ripple RoPE chunk exceeds Int32 kernel indexing"))
    output=similar(input);pairs=(dim÷2)*heads*tokens
    GC.@preserve cosine sine begin
        CUDA.@cuda threads=256 blocks=cld(pairs,256) _ripple_rope_kernel!(output,input,cosine,sine,Int32(first_position),Int32(base))
    end
    return output
end
