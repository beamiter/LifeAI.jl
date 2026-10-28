# Native block-scaled FP4 GEMM through cuBLASLt. No floating-point expansion
# of weights occurs on this path. A=weight' (TN), B=activation, D=A'*B.
const _FP4_BLAS = CUDA.CUBLAS
const _FP4_DATA_TYPE = _FP4_BLAS.cudaDataType_t(33) # CUDA_R_4F_E2M1
const _FP4_FLOAT_TYPE = _FP4_BLAS.cudaDataType_t(0) # CUDA_R_32F
const _FP4_GLOBAL_LOCK = ReentrantLock()
const _FP4_CONTEXTS = Dict{Any,Any}()

mutable struct _NativeFP4Context
    handle::_FP4_BLAS.cublasLtHandle_t
    workspace::CUDA.CuVector{UInt8}
    stream::CUDA.CuStream
    decode_buffers::Dict{Int,Any}
end

struct _NativeFP4DecodeBuffers
    packed::CUDA.CuMatrix{UInt8}
    scales::CUDA.CuVector{UInt8}
    # alpha, beta, activation global scale, and nonfinite-input marker.
    scalars::CUDA.CuVector{Float32}
    # One alpha per projection for shared-input groups (up to 16).
    multi_alpha::CUDA.CuVector{Float32}
end

mutable struct _NativeFP4Plan
    n::Int
    operation::_FP4_BLAS.cublasLtMatmulDesc_t
    a::_FP4_BLAS.cublasLtMatrixLayout_t
    b::_FP4_BLAS.cublasLtMatrixLayout_t
    c::_FP4_BLAS.cublasLtMatrixLayout_t
    algorithm::_FP4_BLAS.cublasLtMatmulAlgo_t
    bscale_pointer::CUDA.CuPtr{UInt8}
end

mutable struct CUDANVFP4Weight{W,P,S,C} <: LifeAI.AbstractNativeNVFP4Weight
    weight::W
    packed::P
    scales::S
    context::C
    padded_m::Int
    padded_k::Int
    plan::Union{Nothing,_NativeFP4Plan}
    decode_plan::Union{Nothing,_NativeFP4Plan}
    native_calls::Int
    last_n::Int
    closed::Bool
end

function LifeAI.native_nvfp4_supported()
    try
        return CUDA.functional() && CUDA.capability(CUDA.device()) >= v"10.0" &&
            _FP4_BLAS.cublasLtGetVersion() >= 120800
    catch
        return false
    end
end

function _native_fp4_context()::_NativeFP4Context
    key = CUDA.context()
    return get!(_FP4_CONTEXTS, key) do
        handle = Ref{_FP4_BLAS.cublasLtHandle_t}()
        _FP4_BLAS.cublasLtCreate(handle)
        _NativeFP4Context(handle[], CUDA.CuArray{UInt8}(undef, 8 * 1024 * 1024),
            CUDA.stream(), Dict{Int,Any}())
    end
end

# Host-side submission is serialized by _FP4_GLOBAL_LOCK. Calls in the same
# CUDA stream already execute in order and may safely reuse this workspace.
# Different Julia tasks normally own different streams; finish the previous
# stream before transferring the shared scratch lease to another stream.
function _native_fp4_acquire_stream!(context::_NativeFP4Context)
    stream = CUDA.stream()
    if context.stream != stream
        CUDA.synchronize(context.stream)
        context.stream = stream
    end
    return stream
end

function _native_fp4_decode_buffers(context::_NativeFP4Context, padded_k::Int)::_NativeFP4DecodeBuffers
    return get!(context.decode_buffers, padded_k) do
        # Keep this shape cache bounded, independently of the number of experts.
        if length(context.decode_buffers) >= 8
            CUDA.synchronize(context.stream)
            empty!(context.decode_buffers)
        end
        block_width = cld(padded_k ÷ 16, 4)*4
        _NativeFP4DecodeBuffers(CUDA.CuArray{UInt8}(undef, padded_k ÷ 2, 1),
            CUDA.zeros(UInt8, 128*block_width), CUDA.CuArray{Float32}(undef, 4),
            CUDA.CuArray{Float32}(undef, 16))
    end
end

@inline function _native_fp4_decode_amax(x)
    thread = CUDA.threadIdx().x
    amax = 0f0
    index = thread
    while index <= length(x)
        @inbounds value = abs(x[index])
        # Inf also represents NaN during the max reduction, so validation does
        # not depend on hardware min/max NaN propagation rules.
        amax = isfinite(value) ? max(amax, value) : Float32(Inf)
        index += CUDA.blockDim().x
    end
    shared = CUDA.CuStaticSharedArray(Float32, 256)
    @inbounds shared[thread] = amax
    CUDA.sync_threads()
    width = 128
    while width > 0
        if thread <= width
            @inbounds shared[thread] = max(shared[thread], shared[thread+width])
        end
        CUDA.sync_threads()
        width ÷= 2
    end
    @inbounds return shared[1]
end

@inline function _native_fp4_decode_set_scalars!(scalars, maximum_value, weight_scale)
    valid = isfinite(maximum_value)
    global_scale = iszero(maximum_value) || !valid ? 1f0 :
        max(maximum_value / 2688f0, reinterpret(Float32, UInt32(1)))
    if !isfinite(2688f0 * global_scale)
        global_scale = reinterpret(Float32, reinterpret(UInt32, global_scale)-UInt32(1))
    end
    @inbounds begin
        scalars[1] = valid ? weight_scale * global_scale : Float32(NaN)
        scalars[2] = 0f0
        scalars[3] = global_scale
        scalars[4] = valid ? 0f0 : 1f0
    end
    return global_scale
end

function _native_fp4_decode_amax_kernel!(scalars, x, weight_scale)
    maximum_value = _native_fp4_decode_amax(x)
    if CUDA.threadIdx().x == 1
        _native_fp4_decode_set_scalars!(scalars, maximum_value, weight_scale)
    end
    return
end

function _native_fp4_decode_multi_amax_kernel!(scalars, alphas, x, weight_scales::NTuple{N,Float32}) where {N}
    maximum_value = _native_fp4_decode_amax(x)
    if CUDA.threadIdx().x == 1
        global_scale = _native_fp4_decode_set_scalars!(scalars, maximum_value, weight_scales[1])
        valid = isfinite(maximum_value)
        for index in 1:N
            @inbounds alphas[index] = valid ? weight_scales[index] * global_scale : Float32(NaN)
        end
    end
    return
end

# A decode token has one activation row. Write the cuBLASLt packed payload
# and swizzled scales directly, preserving the portable quantizer's exact
# division order, nearest-even codebooks, zero handling and K padding.
function _native_fp4_decode_pack_kernel!(packed, scales, scalars, x, padded_k)
    block = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x - 1
    block >= padded_k ÷ 16 && return
    input = size(x,1)
    first = 16block + 1
    block_amax = 0f0
    for offset in 0:15
        column = first + offset
        @inbounds value = column <= input ? abs(x[column]) : 0f0
        value = isfinite(value) ? value : 0f0
        block_amax = max(block_amax, value)
    end
    @inbounds global_scale = scalars[3]
    scale = LifeAI._nvfp4_encode_e4m3((block_amax / 6f0) / global_scale)
    decoded = LifeAI._nvfp4_decode_e4m3(scale)
    denominator = iszero(decoded) ? 1f0 : decoded
    block_width = cld(padded_k ÷ 16, 4)*4
    @inbounds scales[_native_fp4_scale_offset(0, block, block_width)] = scale
    for pair in 0:7
        column = first + 2pair
        @inbounds low_value = column <= input ? x[column] : 0f0
        @inbounds high_value = column+1 <= input ? x[column+1] : 0f0
        low_value = isfinite(low_value) ? low_value : 0f0
        high_value = isfinite(high_value) ? high_value : 0f0
        low = iszero(decoded) ? UInt8(0) : LifeAI._nvfp4_encode_e2m1((low_value / global_scale) / denominator)
        high = iszero(decoded) ? UInt8(0) : LifeAI._nvfp4_encode_e2m1((high_value / global_scale) / denominator)
        @inbounds packed[8block+pair+1] = low | (high << UInt8(4))
    end
    return
end

@inline function _native_fp4_scale_offset(row, block, block_width)
    return (row ÷ 128) * block_width * 128 + (block ÷ 4) * 512 +
        (row % 32) * 16 + ((row % 128) ÷ 32) * 4 + block % 4 + 1
end

function _native_fp4_pack_kernel!(destination, source, rows, columns)
    index = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    while index <= length(destination)
        col = (index-1) % size(destination,1) + 1
        row = (index-1) ÷ size(destination,1) + 1
        @inbounds destination[index] = row <= rows && col <= columns ? source[row,col] : UInt8(0)
        index += stride
    end
    return
end

function _native_fp4_scale_kernel!(destination, source, block_width)
    index = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    while index <= length(source)
        row = (index-1) % size(source,1)
        block = (index-1) ÷ size(source,1)
        offset = _native_fp4_scale_offset(row, block, block_width)
        @inbounds destination[offset] = source[index]
        index += stride
    end
    return
end

function _native_fp4_repack(weight, padded_outer, padded_k)
    packed = CUDA.CuArray{UInt8}(undef, padded_k ÷ 2, padded_outer)
    CUDA.@cuda threads=256 blocks=min(cld(length(packed),256),65535) _native_fp4_pack_kernel!(
        packed, weight.packed, size(weight,1), size(weight.packed,2))
    block_width = cld(padded_k ÷ 16, 4)*4
    scales = CUDA.zeros(UInt8, cld(padded_outer,128)*128*block_width)
    CUDA.@cuda threads=256 blocks=min(cld(length(weight.block_scales),256),65535) _native_fp4_scale_kernel!(
        scales, weight.block_scales, block_width)
    return packed, scales
end

function _native_fp4_attribute!(descriptor, attribute, value)
    value_ref = Ref(value)
    _FP4_BLAS.cublasLtMatmulDescSetAttribute(descriptor, attribute, value_ref, sizeof(value))
end

function _native_fp4_layout(dtype, rows, cols, ld)
    descriptor = Ref{_FP4_BLAS.cublasLtMatrixLayout_t}()
    _FP4_BLAS.cublasLtMatrixLayoutCreate(descriptor, dtype, rows, cols, ld)
    return descriptor[]
end

function _native_fp4_destroy_plan!(plan::_NativeFP4Plan)
    _FP4_BLAS.cublasLtMatmulDescDestroy(plan.operation)
    _FP4_BLAS.cublasLtMatrixLayoutDestroy(plan.a)
    _FP4_BLAS.cublasLtMatrixLayoutDestroy(plan.b)
    _FP4_BLAS.cublasLtMatrixLayoutDestroy(plan.c)
    return nothing
end

function Base.close(prepared::CUDANVFP4Weight)
    lock(_FP4_GLOBAL_LOCK) do
        if !prepared.closed
            prepared.plan === nothing || _native_fp4_destroy_plan!(prepared.plan)
            prepared.plan = nothing
            prepared.decode_plan === nothing || _native_fp4_destroy_plan!(prepared.decode_plan)
            prepared.decode_plan = nothing
            prepared.closed = true
        end
    end
    return nothing
end

function LifeAI.prepare_nvfp4_gemm(weight::LifeAI.NVFP4Weight{P,S}; kwargs...) where {
    P<:CUDA.CuArray{UInt8,2},S<:CUDA.CuArray{UInt8,2}}
    isempty(kwargs) || throw(ArgumentError("native NVFP4 preparation does not accept extra options"))
    LifeAI.native_nvfp4_supported() || throw(ArgumentError("native NVFP4 requires Blackwell and cuBLASLt >=12.8"))
    CUDA.context(weight.packed) == CUDA.context(weight.block_scales) == CUDA.context() ||
        throw(ArgumentError("native NVFP4 storage must belong to the active CUDA context"))
    padded_m = cld(size(weight,1),4)*4
    padded_k = cld(size(weight,2),32)*32
    return lock(_FP4_GLOBAL_LOCK) do
        context = _native_fp4_context()
        packed, scales = _native_fp4_repack(weight, padded_m, padded_k)
        # Preparation returns fully initialized immutable packed data.
        CUDA.synchronize()
        prepared = CUDANVFP4Weight(weight, packed, scales, CUDA.context(),
            padded_m, padded_k, nothing, nothing, 0, 0, false)
        finalizer(prepared) do item
            if !item.closed
                for plan in (item.plan, item.decode_plan)
                    if plan !== nothing
                        try
                            _native_fp4_destroy_plan!(plan)
                        catch
                        end
                    end
                end
                item.plan = nothing
                item.decode_plan = nothing
            end
        end
        return prepared
    end
end

LifeAI.nvfp4_weight(weight::CUDANVFP4Weight) = weight.weight

function _native_fp4_plan(prepared, n, context, bscales)
    op_ref = Ref{_FP4_BLAS.cublasLtMatmulDesc_t}()
    _FP4_BLAS.cublasLtMatmulDescCreate(op_ref, _FP4_BLAS.CUBLAS_COMPUTE_32F, _FP4_FLOAT_TYPE)
    op = op_ref[]
    a = b = c = nothing
    preference = nothing
    try
        _native_fp4_attribute!(op, _FP4_BLAS.CUBLASLT_MATMUL_DESC_TRANSA, _FP4_BLAS.CUBLAS_OP_T)
        _native_fp4_attribute!(op, _FP4_BLAS.CUBLASLT_MATMUL_DESC_TRANSB, _FP4_BLAS.CUBLAS_OP_N)
        if n == 1
            _native_fp4_attribute!(op, _FP4_BLAS.CUBLASLT_MATMUL_DESC_POINTER_MODE, _FP4_BLAS.CUBLASLT_POINTER_MODE_DEVICE)
        end
        _native_fp4_attribute!(op, _FP4_BLAS.CUBLASLT_MATMUL_DESC_A_SCALE_MODE, _FP4_BLAS.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
        _native_fp4_attribute!(op, _FP4_BLAS.CUBLASLT_MATMUL_DESC_B_SCALE_MODE, _FP4_BLAS.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
        _native_fp4_attribute!(op, _FP4_BLAS.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, pointer(prepared.scales))
        _native_fp4_attribute!(op, _FP4_BLAS.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, pointer(bscales))
        a = _native_fp4_layout(_FP4_DATA_TYPE, prepared.padded_k, prepared.padded_m, prepared.padded_k)
        b = _native_fp4_layout(_FP4_DATA_TYPE, prepared.padded_k, n, prepared.padded_k)
        c = _native_fp4_layout(_FP4_FLOAT_TYPE, prepared.padded_m, n, prepared.padded_m)
        pref_ref = Ref{_FP4_BLAS.cublasLtMatmulPreference_t}()
        _FP4_BLAS.cublasLtMatmulPreferenceCreate(pref_ref)
        preference = pref_ref[]
        bytes_ref = Ref{Csize_t}(length(context.workspace))
        _FP4_BLAS.cublasLtMatmulPreferenceSetAttribute(preference,
            _FP4_BLAS.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, bytes_ref, sizeof(Csize_t))
        algorithms = Vector{_FP4_BLAS.cublasLtMatmulHeuristicResult_t}(undef, 16)
        count = Ref{Cint}(0)
        _FP4_BLAS.cublasLtMatmulAlgoGetHeuristic(context.handle, op, a, b, c, c,
            preference, length(algorithms), algorithms, count)
        count[] > 0 || throw(ArgumentError("cuBLASLt found no native NVFP4 algorithm for padded shape $(prepared.padded_m)x$n x$(prepared.padded_k)"))
        successful = findfirst(i -> algorithms[i].state == _FP4_BLAS.CUBLAS_STATUS_SUCCESS, 1:count[])
        successful === nothing && error("cuBLASLt returned no successful NVFP4 algorithm")
        return _NativeFP4Plan(n, op, a, b, c, algorithms[successful].algo, pointer(bscales))
    catch
        _FP4_BLAS.cublasLtMatmulDescDestroy(op)
        a === nothing || _FP4_BLAS.cublasLtMatrixLayoutDestroy(a)
        b === nothing || _FP4_BLAS.cublasLtMatrixLayoutDestroy(b)
        c === nothing || _FP4_BLAS.cublasLtMatrixLayoutDestroy(c)
        rethrow()
    finally
        preference === nothing || _FP4_BLAS.cublasLtMatmulPreferenceDestroy(preference)
    end
end

# Keep the decode descriptor resident across a prefill call. Only the latest
# multi-token shape is cached alongside it, so descriptors stay bounded at two.
function _native_fp4_get_plan!(prepared::CUDANVFP4Weight, n::Int, context::_NativeFP4Context, bscales)
    plan = if n == 1
        if prepared.decode_plan === nothing
            prepared.decode_plan = _native_fp4_plan(prepared, n, context, bscales)
        end
        prepared.decode_plan
    else
        if prepared.plan === nothing || prepared.plan.n != n
            prepared.plan === nothing || _native_fp4_destroy_plan!(prepared.plan)
            prepared.plan = nothing
            prepared.plan = _native_fp4_plan(prepared, n, context, bscales)
        end
        prepared.plan
    end
    bscale_pointer = pointer(bscales)
    if plan.bscale_pointer != bscale_pointer
        _native_fp4_attribute!(plan.operation, _FP4_BLAS.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, bscale_pointer)
        plan.bscale_pointer = bscale_pointer
    end
    return plan::_NativeFP4Plan
end

function LifeAI.native_nvfp4_gemm(prepared::CUDANVFP4Weight, x::CUDA.CuArray{Float32,2}; check_finite::Bool=true)
    prepared.closed && throw(ArgumentError("native NVFP4 weight is closed"))
    size(x,1) == size(prepared.weight,2) || throw(DimensionMismatch("native NVFP4 input width mismatch"))
    size(x,2) > 0 || throw(ArgumentError("native NVFP4 input must contain tokens"))
    CUDA.context() == prepared.context == CUDA.context(x) || throw(ArgumentError("native NVFP4 input and weight must belong to the active CUDA context"))
    return lock(_FP4_GLOBAL_LOCK) do
        prepared.closed && throw(ArgumentError("native NVFP4 weight is closed"))
        context = _native_fp4_context()
        stream = _native_fp4_acquire_stream!(context)
        n = size(x,2)
        padded_n = n
        activation = nothing
        decode = nothing
        if n == 1
            decode = _native_fp4_decode_buffers(context, prepared.padded_k)
            CUDA.@cuda threads=256 blocks=1 _native_fp4_decode_amax_kernel!(decode.scalars, x, prepared.weight.global_scale)
            CUDA.@cuda threads=128 blocks=cld(prepared.padded_k ÷ 16,128) _native_fp4_decode_pack_kernel!(
                decode.packed, decode.scales, decode.scalars, x, prepared.padded_k)
            bpacked, bscales = decode.packed, decode.scales
        else
            activation = LifeAI.quantize_nvfp4(permutedims(x))
            bpacked, bscales = _native_fp4_repack(activation, padded_n, prepared.padded_k)
        end
        plan = _native_fp4_get_plan!(prepared, padded_n, context, bscales)
        output = CUDA.CuArray{Float32}(undef, prepared.padded_m, padded_n)
        alpha = n == 1 ? pointer(decode.scalars,1) : Ref{Float32}(prepared.weight.global_scale * activation.global_scale)
        beta = n == 1 ? pointer(decode.scalars,2) : Ref{Float32}(0f0)
        algorithm = Ref(plan.algorithm)
        GC.@preserve prepared activation decode bpacked bscales output alpha beta algorithm begin
            _FP4_BLAS.cublasLtMatmul(context.handle, plan.operation, alpha,
                prepared.packed, plan.a, bpacked, plan.b, beta, output, plan.c,
                output, plan.c, algorithm, context.workspace, length(context.workspace), stream)
            if n == 1
                # Public calls validate finite inputs. The model's internal
                # decode path can keep this scalar entirely on-device and
                # validate the final logits instead of every projection.
                if check_finite
                    Array(decode.scalars)[4] == 0f0 || throw(ArgumentError("NVFP4 activations must be finite"))
                end
            else
                # The multi-token path retains temporary activation arrays.
                CUDA.synchronize(stream)
            end
        end
        prepared.native_calls += 1
        prepared.last_n = n
        return size(output,1) == size(prepared.weight,1) ? output : output[1:size(prepared.weight,1), 1:n]
    end
end

function LifeAI.native_nvfp4_gemm_multi(weights::Tuple{Vararg{CUDANVFP4Weight,N}},
        x::CUDA.CuArray{Float32,2}; check_finite::Bool=true) where {N}
    1 <= N <= 16 || throw(ArgumentError("native NVFP4 multi-projection accepts one to sixteen weights"))
    all(weight -> !weight.closed, weights) || throw(ArgumentError("native NVFP4 weight is closed"))
    all(weight -> size(x,1) == size(weight,2), weights) || throw(DimensionMismatch("native NVFP4 multi-projection input width mismatch"))
    size(x,2) > 0 || throw(ArgumentError("native NVFP4 input must contain tokens"))
    context_id = CUDA.context()
    all(weight -> weight.context == context_id, weights) && CUDA.context(x) == context_id ||
        throw(ArgumentError("native NVFP4 input and weights must belong to the active CUDA context"))
    N == 1 && return (LifeAI.native_nvfp4_gemm(weights[1],x;check_finite),)
    # Prefill retains its existing arithmetic and validation. The common decode
    # input can be quantized once for Q/K/V or all selected expert gate/up projections.
    size(x,2) == 1 || return map(weight -> LifeAI.native_nvfp4_gemm(weight,x;check_finite), weights)
    return lock(_FP4_GLOBAL_LOCK) do
        all(weight -> !weight.closed, weights) || throw(ArgumentError("native NVFP4 weight is closed"))
        context = _native_fp4_context()
        stream = _native_fp4_acquire_stream!(context)
        padded_k = weights[1].padded_k
        decode = _native_fp4_decode_buffers(context,padded_k)
        weight_scales = map(weight -> weight.weight.global_scale, weights)
        CUDA.@cuda threads=256 blocks=1 _native_fp4_decode_multi_amax_kernel!(
            decode.scalars, decode.multi_alpha, x, weight_scales)
        CUDA.@cuda threads=128 blocks=cld(padded_k ÷ 16,128) _native_fp4_decode_pack_kernel!(
            decode.packed, decode.scales, decode.scalars, x, padded_k)
        outputs = map(weight -> CUDA.CuArray{Float32}(undef,weight.padded_m,1), weights)
        GC.@preserve weights decode outputs begin
            for index in 1:N
                weight = weights[index]
                plan = _native_fp4_get_plan!(weight,1,context,decode.scales)
                alpha, beta = pointer(decode.multi_alpha,index), pointer(decode.scalars,2)
                output = outputs[index]
                algorithm = Ref(plan.algorithm)
                GC.@preserve algorithm begin
                    _FP4_BLAS.cublasLtMatmul(context.handle,plan.operation,alpha,
                        weight.packed,plan.a,decode.packed,plan.b,beta,output,plan.c,
                        output,plan.c,algorithm,context.workspace,length(context.workspace),stream)
                end
                weight.native_calls += 1
                weight.last_n = 1
            end
            if check_finite
                Array(decode.scalars)[4] == 0f0 || throw(ArgumentError("NVFP4 activations must be finite"))
            end
        end
        return map(weights,outputs) do weight,output
            size(output,1) == size(weight,1) ? output : output[1:size(weight,1),1:1]
        end
    end
end

function LifeAI.native_nvfp4_stats(prepared::CUDANVFP4Weight)
    return (backend=:cublaslt_nvfp4, native_calls=prepared.native_calls,
        padded_m=prepared.padded_m, padded_k=prepared.padded_k,
        cached_shapes=Int(prepared.plan !== nothing) + Int(prepared.decode_plan !== nothing),
        workspace_bytes=8*1024*1024, shared_workspace=true,
        prepared_weight_bytes=sizeof(prepared.packed)+sizeof(prepared.scales),
        last_tokens=prepared.last_n, activation_quantized=true, closed=prepared.closed)
end
