# Fused Float32 kernels used by Flint's CUDA decode path. FP4 projections stay
# in flint_nvfp4_native.jl; these operations do not change the weight format.
const _FLINT_DECODE_KERNEL_LOCK = ReentrantLock()
const _FLINT_DECODE_KERNEL_CONTEXTS = Dict{Any,Any}()
const _FLINT_ROPE_WINDOW = 2048
const _FlintDecodeDeviceMemory = parentmodule(CUDA.CuArray).DeviceMemory
const _FlintRoPETable = Tuple{CUDA.CuArray{Float32,2,_FlintDecodeDeviceMemory},
    CUDA.CuArray{Float32,2,_FlintDecodeDeviceMemory}}

struct _FlintGreedyScratch
    values::CUDA.CuArray{Float32,1,_FlintDecodeDeviceMemory}
    indices::CUDA.CuArray{Int32,1,_FlintDecodeDeviceMemory}
    token::CUDA.CuArray{Int32,1,_FlintDecodeDeviceMemory}
end

mutable struct _FlintDecodeKernelContext
    rope_tables::Dict{Tuple{Int,Float32,Int},_FlintRoPETable}
    rope_order::Vector{Tuple{Int,Float32,Int}}
    greedy::Union{Nothing,_FlintGreedyScratch}
end

function _flint_decode_kernel_context()::_FlintDecodeKernelContext
    return get!(_FLINT_DECODE_KERNEL_CONTEXTS, CUDA.context()) do
        _FlintDecodeKernelContext(
            Dict{Tuple{Int,Float32,Int},_FlintRoPETable}(),
            Tuple{Int,Float32,Int}[], nothing)
    end
end

function _flint_norm_kernel!(output, input, weight, epsilon, rows)
    thread = CUDA.threadIdx().x
    column = CUDA.blockIdx().x - 1
    offset = column * rows
    total = 0f0
    row = thread
    while row <= rows
        @inbounds value = input[offset+row]
        total += value * value
        row += CUDA.blockDim().x
    end
    shared = CUDA.CuStaticSharedArray(Float32, 256)
    @inbounds shared[thread] = total
    CUDA.sync_threads()
    width = 128
    while width > 0
        if thread <= width
            @inbounds shared[thread] += shared[thread+width]
        end
        CUDA.sync_threads()
        width ÷= 2
    end
    @inbounds factor = inv(sqrt(shared[1] / Float32(rows) + epsilon))
    row = thread
    while row <= rows
        @inbounds output[offset+row] = (input[offset+row] * factor) * weight[row]
        row += CUDA.blockDim().x
    end
    return
end

function LifeAI._flint_norm(input::CUDA.CuArray{Float32,N}, weight::CUDA.CuVector{Float32}, epsilon) where N
    rows = size(input,1)
    rows > 0 || throw(ArgumentError("Flint normalization axis must be nonempty"))
    length(weight) == rows || throw(DimensionMismatch("Flint normalization weight width mismatch"))
    CUDA.context(input) == CUDA.context(weight) == CUDA.context() ||
        throw(ArgumentError("Flint normalization arrays must belong to the active CUDA context"))
    output = similar(input)
    isempty(input) && return output
    CUDA.@cuda threads=256 blocks=length(input)÷rows _flint_norm_kernel!(
        output, input, weight, Float32(epsilon), rows)
    return output
end

function _flint_rope_tables(d, theta, base)::_FlintRoPETable
    context = _flint_decode_kernel_context()
    key = (d, Float32(theta), base)
    return get!(context.rope_tables, key) do
        if length(context.rope_tables) >= 8
            # A cache miss occurs only once per position window. Complete all
            # context streams before evicting a table another task may read.
            CUDA.device_synchronize()
            delete!(context.rope_tables, popfirst!(context.rope_order))
        end
        half = d ÷ 2
        # Match the existing CPU frequency/angle/trigonometric computation.
        # Preparing one window avoids GPU pow/sin/cos approximation differences
        # and replaces repeated per-layer tiny host-to-device transfers.
        frequencies = Float32(theta) .^ (-Float32.(0:half-1) .* (2f0 / d))
        angles = reshape(frequencies, half, 1) .*
            reshape(Float32.(base:base+_FLINT_ROPE_WINDOW-1), 1, :)
        cosine = CUDA.CuArray{Float32,2,_FlintDecodeDeviceMemory}(cos.(angles))
        sine = CUDA.CuArray{Float32,2,_FlintDecodeDeviceMemory}(sin.(angles))
        CUDA.synchronize()
        push!(context.rope_order, key)
        (cosine, sine)
    end
end

function _flint_rope_kernel!(output, input, cosine, sine, first_position, base)
    index = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    half = size(input,1) ÷ 2
    heads = size(input,2)
    total = half * heads * size(input,3)
    while index <= total
        row = (index-1) % half + 1
        head = ((index-1) ÷ half) % heads + 1
        token = (index-1) ÷ (half*heads)
        position = first_position + token - base + 1
        @inbounds begin
            c, s = cosine[row,position], sine[row,position]
            a, b = input[row,head,token+1], input[row+half,head,token+1]
            output[row,head,token+1] = a*c - b*s
            output[row+half,head,token+1] = b*c + a*s
        end
        index += stride
    end
    return
end

function LifeAI._flint_rope(input::CUDA.CuArray{Float32,3}, positions::AbstractUnitRange{<:Integer}, theta)
    d, heads, tokens = size(input)
    # The ordinary path handles arbitrary ranges and chunks spanning windows.
    # Decode and normal short prefills stay in the fused cache-backed path.
    if isodd(d) || d == 0 || length(positions) != tokens || tokens == 0
        return invoke(LifeAI._flint_rope, Tuple{Any,Any,Any}, input, positions, theta)
    end
    first_position, last_position = Int(first(positions)), Int(last(positions))
    base = fld(first_position, _FLINT_ROPE_WINDOW) * _FLINT_ROPE_WINDOW
    if last_position >= base + _FLINT_ROPE_WINDOW
        return invoke(LifeAI._flint_rope, Tuple{Any,Any,Any}, input, positions, theta)
    end
    CUDA.context(input) == CUDA.context() ||
        throw(ArgumentError("Flint RoPE input must belong to the active CUDA context"))
    cosine, sine = (lock(_FLINT_DECODE_KERNEL_LOCK) do
        _flint_rope_tables(d, theta, base)
    end)::_FlintRoPETable
    output = similar(input)
    pairs = (d ÷ 2) * heads * tokens
    GC.@preserve cosine sine begin
        CUDA.@cuda threads=256 blocks=cld(pairs,256) _flint_rope_kernel!(
            output, input, cosine, sine, first_position, base)
    end
    return output
end

@inline function _flint_argmax_merge(value, index, other_value, other_index)
    if index == 0 || other_index == 0
        return 0f0, Int32(0)
    end
    # Julia argmax uses total floating-point order: +0 wins over -0; exact
    # ties retain the first index. Nonfinite values are rejected separately.
    return isless(value, other_value) || (isequal(value,other_value) && other_index < index) ?
        (other_value,other_index) : (value,index)
end

function _flint_greedy_partials_kernel!(values, indices, logits)
    thread = CUDA.threadIdx().x
    index = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + thread
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    best_value, best_index = -Inf32, typemax(Int32)
    invalid = false
    while index <= length(logits)
        @inbounds value = logits[index]
        if isfinite(value)
            best_value, best_index = _flint_argmax_merge(best_value,best_index,value,Int32(index))
        else
            invalid = true
        end
        index += stride
    end
    invalid && (best_index = Int32(0))
    shared_values = CUDA.CuStaticSharedArray(Float32,256)
    shared_indices = CUDA.CuStaticSharedArray(Int32,256)
    @inbounds shared_values[thread], shared_indices[thread] = best_value, best_index
    CUDA.sync_threads()
    width = 128
    while width > 0
        if thread <= width
            @inbounds shared_values[thread],shared_indices[thread] = _flint_argmax_merge(
                shared_values[thread],shared_indices[thread],
                shared_values[thread+width],shared_indices[thread+width])
        end
        CUDA.sync_threads()
        width ÷= 2
    end
    if thread == 1
        @inbounds values[CUDA.blockIdx().x],indices[CUDA.blockIdx().x] = shared_values[1],shared_indices[1]
    end
    return
end

function _flint_greedy_final_kernel!(token, values, indices, count)
    thread = CUDA.threadIdx().x
    shared_values = CUDA.CuStaticSharedArray(Float32,256)
    shared_indices = CUDA.CuStaticSharedArray(Int32,256)
    @inbounds begin
        shared_values[thread] = thread <= count ? values[thread] : -Inf32
        shared_indices[thread] = thread <= count ? indices[thread] : typemax(Int32)
    end
    CUDA.sync_threads()
    width = 128
    while width > 0
        if thread <= width
            @inbounds shared_values[thread],shared_indices[thread] = _flint_argmax_merge(
                shared_values[thread],shared_indices[thread],
                shared_values[thread+width],shared_indices[thread+width])
        end
        CUDA.sync_threads()
        width ÷= 2
    end
    thread == 1 && (@inbounds token[1] = shared_indices[1])
    return
end

function LifeAI._flint_sample_token(logits::CUDA.CuArray{Float32,2}, rng::LifeAI.AbstractRNG;
        temperature::Real=1f0, top_k=nothing, top_p=nothing, sample_uniform=nothing)
    LifeAI._sampling_length(logits,"logits")
    temperature isa Bool && throw(ArgumentError("`temperature` must not be Bool"))
    temperature >= 0 || throw(ArgumentError("`temperature` must be non-negative"))
    if temperature != 0
        return LifeAI._sample_token(vec(Array(logits)),rng;temperature,top_k,top_p,sample_uniform)
    end
    CUDA.context(logits) == CUDA.context() ||
        throw(ArgumentError("Flint sampling input must belong to the active CUDA context"))
    return lock(_FLINT_DECODE_KERNEL_LOCK) do
        context = _flint_decode_kernel_context()
        if context.greedy === nothing
            context.greedy = _FlintGreedyScratch(CUDA.CuArray{Float32,1,_FlintDecodeDeviceMemory}(undef,256),
                CUDA.CuArray{Int32,1,_FlintDecodeDeviceMemory}(undef,256),
                CUDA.CuArray{Int32,1,_FlintDecodeDeviceMemory}(undef,1))
        end
        scratch = context.greedy::_FlintGreedyScratch
        blocks = min(cld(length(logits),1024),256)
        CUDA.@cuda threads=256 blocks=blocks _flint_greedy_partials_kernel!(
            scratch.values,scratch.indices,logits)
        CUDA.@cuda threads=256 blocks=1 _flint_greedy_final_kernel!(
            scratch.token,scratch.values,scratch.indices,blocks)
        token = Int(only(Array(scratch.token)))
        token != 0 || throw(ArgumentError("`logits` contains non-finite values"))
        return token
    end
end
