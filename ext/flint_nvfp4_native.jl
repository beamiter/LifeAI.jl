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
end

mutable struct _NativeFP4Plan
    n::Int
    operation::_FP4_BLAS.cublasLtMatmulDesc_t
    a::_FP4_BLAS.cublasLtMatrixLayout_t
    b::_FP4_BLAS.cublasLtMatrixLayout_t
    c::_FP4_BLAS.cublasLtMatrixLayout_t
    algorithm::_FP4_BLAS.cublasLtMatmulAlgo_t
end

mutable struct CUDANVFP4Weight{W,P,S,C} <: LifeAI.AbstractNativeNVFP4Weight
    weight::W
    packed::P
    scales::S
    context::C
    padded_m::Int
    padded_k::Int
    plan::Union{Nothing,_NativeFP4Plan}
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

function _native_fp4_context()
    key = CUDA.context()
    return get!(_FP4_CONTEXTS, key) do
        handle = Ref{_FP4_BLAS.cublasLtHandle_t}()
        _FP4_BLAS.cublasLtCreate(handle)
        _NativeFP4Context(handle[], CUDA.CuArray{UInt8}(undef, 8 * 1024 * 1024))
    end
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
            padded_m, padded_k, nothing, 0, 0, false)
        finalizer(prepared) do item
            if !item.closed && item.plan !== nothing
                try
                    _native_fp4_destroy_plan!(item.plan)
                catch
                end
                item.plan = nothing
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
        return _NativeFP4Plan(n, op, a, b, c, algorithms[successful].algo)
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

function LifeAI.native_nvfp4_gemm(prepared::CUDANVFP4Weight, x::CUDA.CuArray{Float32,2})
    prepared.closed && throw(ArgumentError("native NVFP4 weight is closed"))
    size(x,1) == size(prepared.weight,2) || throw(DimensionMismatch("native NVFP4 input width mismatch"))
    size(x,2) > 0 || throw(ArgumentError("native NVFP4 input must contain tokens"))
    CUDA.context() == prepared.context == CUDA.context(x) || throw(ArgumentError("native NVFP4 input and weight must belong to the active CUDA context"))
    return lock(_FP4_GLOBAL_LOCK) do
        prepared.closed && throw(ArgumentError("native NVFP4 weight is closed"))
        context = _native_fp4_context()
        n = size(x,2)
        padded_n = n
        # CUDA broadcasts and reductions keep payloads on-device; only amax,
        # needed as the host FP32 alpha, is synchronized back to the host.
        activation = LifeAI.quantize_nvfp4(permutedims(x))
        bpacked, bscales = _native_fp4_repack(activation, padded_n, prepared.padded_k)
        if prepared.plan === nothing || prepared.plan.n != padded_n
            prepared.plan === nothing || _native_fp4_destroy_plan!(prepared.plan)
            prepared.plan = nothing
            prepared.plan = _native_fp4_plan(prepared, padded_n, context, bscales)
        end
        plan = prepared.plan
        _native_fp4_attribute!(plan.operation, _FP4_BLAS.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, pointer(bscales))
        output = CUDA.CuArray{Float32}(undef, prepared.padded_m, padded_n)
        alpha = Ref{Float32}(prepared.weight.global_scale * activation.global_scale)
        beta = Ref{Float32}(0f0)
        algorithm = Ref(plan.algorithm)
        GC.@preserve prepared activation bpacked bscales output alpha beta algorithm begin
            _FP4_BLAS.cublasLtMatmul(context.handle, plan.operation, alpha,
                prepared.packed, plan.a, bpacked, plan.b, beta, output, plan.c,
                output, plan.c, algorithm, context.workspace, length(context.workspace), CUDA.stream())
            # Shared workspace is leased until completion; no per-layer GEMM
            # workspace is retained and concurrent tasks cannot race its use.
            CUDA.synchronize()
        end
        prepared.native_calls += 1
        prepared.last_n = n
        return output[1:size(prepared.weight,1), 1:n]
    end
end

function LifeAI.native_nvfp4_stats(prepared::CUDANVFP4Weight)
    return (backend=:cublaslt_nvfp4, native_calls=prepared.native_calls,
        padded_m=prepared.padded_m, padded_k=prepared.padded_k,
        cached_shapes=prepared.plan === nothing ? 0 : 1,
        workspace_bytes=8*1024*1024, shared_workspace=true,
        prepared_weight_bytes=sizeof(prepared.packed)+sizeof(prepared.scales),
        last_tokens=prepared.last_n, activation_quantized=true, closed=prepared.closed)
end
