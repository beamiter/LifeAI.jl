# NVFP4 1D weight-only storage: E2M1 payloads, one E4M3FN scale per
# 16 input values, and one multiplicative Float32 scale for the whole tensor.
# https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html
const NVFP4_BLOCK_SIZE = 16

"""
    NVFP4Weight(packed, block_scales, global_scale, in_dim)

Rowwise NVFP4 storage with two E2M1 values per byte (low nibble first),
positive E4M3FN block-scale bytes, and a multiplicative Float32 tensor scale.
Each output row is padded to a multiple of 16 input values. Scales use the
logical `(output, input_block)` layout, not a vendor-specific GEMM swizzle.
"""
struct NVFP4Weight{P<:AbstractMatrix{UInt8},S<:AbstractMatrix{UInt8}}
    packed::P
    block_scales::S
    global_scale::Float32
    in_dim::Int
    function NVFP4Weight(packed::P, block_scales::S, global_scale::Real, in_dim::Integer) where {P<:AbstractMatrix{UInt8},S<:AbstractMatrix{UInt8}}
        input = _strict_host_int(in_dim, "NVFP4 input dimension")
        input > 0 || throw(ArgumentError("NVFP4 input dimension must be positive"))
        rows = size(packed, 1)
        rows > 0 || throw(ArgumentError("NVFP4 output dimension must be positive"))
        blocks = cld(input, NVFP4_BLOCK_SIZE)
        size(packed, 2) == 8blocks || throw(DimensionMismatch("NVFP4 packed width does not match padded input dimension"))
        size(block_scales) == (rows, blocks) || throw(DimensionMismatch("NVFP4 block-scale shape mismatch"))
        global_scale isa Bool && throw(ArgumentError("NVFP4 global scale cannot be Bool"))
        global_value = Float32(global_scale)
        isfinite(global_value) && global_value > 0 || throw(ArgumentError("NVFP4 global scale must be finite and positive in Float32"))
        all(x -> x <= 0x7e, block_scales) || throw(ArgumentError("NVFP4 block scales must encode nonnegative finite E4M3FN values"))
        return new{P,S}(packed, block_scales, global_value, input)
    end
end

Base.size(w::NVFP4Weight) = (size(w.packed, 1), w.in_dim)
Base.size(w::NVFP4Weight, d::Integer) = d == 1 ? size(w.packed, 1) : d == 2 ? w.in_dim : d > 2 ? 1 : throw(ArgumentError("dimension must be positive"))

# Nearest representable value, ties to an even encoding significand.
@inline function _nvfp4_encode_e2m1(value::Float32)
    a = abs(value)
    magnitude = if a <= 0.25f0
        UInt8(0)
    elseif a < 0.75f0
        UInt8(1)
    elseif a <= 1.25f0
        UInt8(2)
    elseif a < 1.75f0
        UInt8(3)
    elseif a <= 2.5f0
        UInt8(4)
    elseif a < 3.5f0
        UInt8(5)
    elseif a <= 5f0
        UInt8(6)
    else
        UInt8(7)
    end
    return magnitude | (signbit(value) ? UInt8(8) : UInt8(0))
end

@inline function _nvfp4_decode_e2m1(code::UInt8)
    magnitude = code & UInt8(7)
    exponent = (magnitude >> 1) & UInt8(3)
    mantissa = magnitude & UInt8(1)
    value = exponent == 0 ? 0.5f0 * Float32(mantissa) :
        (1f0 + 0.5f0 * Float32(mantissa)) * Float32(UInt32(1) << (exponent - UInt8(1)))
    return (code & UInt8(8)) == 0 ? value : -value
end

@inline function _nvfp4_encode_e4m3(value::Float32)
    a = clamp(value, 0f0, 448f0)
    if a < 0.015625f0  # 2^-6: subnormals have spacing 2^-9.
        return UInt8(round(Int32, a * 512f0))
    end
    exponent = Int32((reinterpret(UInt32, a) >> 23) & UInt32(0xff)) - Int32(127)
    power = reinterpret(Float32, UInt32(exponent + Int32(127)) << 23)
    fraction = round(Int32, (a / power - 1f0) * 8f0)
    return UInt8(min((exponent + Int32(7)) * Int32(8) + fraction, Int32(126)))
end

@inline function _nvfp4_decode_e4m3(code::UInt8)
    exponent = UInt32(code >> 3)
    mantissa = Float32(code & UInt8(7))
    return exponent == 0 ? mantissa / 512f0 :
        (1f0 + mantissa / 8f0) * reinterpret(Float32, (exponent + UInt32(120)) << 23)
end

"""
    quantize_nvfp4(weight::AbstractMatrix)

Deterministic nearest-even NVFP4 1D weight quantization. Preserves CPU/CUDA
storage placement. Nonfinite inputs are rejected. The global scale is
`maximum(abs, weight)/(448*6)`, with representable guards for all-zero and
subnormal tensors. Partial final blocks are padded with zeros. No activation
quantization, random rotation, or stochastic rounding is applied.
"""
function quantize_nvfp4(weight::AbstractMatrix{<:Real})
    eltype(weight) <: Bool && throw(ArgumentError("NVFP4 weights cannot be Bool"))
    rows, input = size(weight)
    rows > 0 && input > 0 || throw(ArgumentError("NVFP4 weight dimensions must be positive"))
    values = Float32.(weight)
    all(isfinite, values) || throw(ArgumentError("NVFP4 weights must be finite and representable in Float32"))
    amax = maximum(abs, values)
    global_scale = iszero(amax) ? 1f0 : max(amax / 2688f0, nextfloat(0f0))
    !isfinite(2688f0 * global_scale) && (global_scale = prevfloat(global_scale))
    blocks = cld(input, NVFP4_BLOCK_SIZE)
    padded = similar(values, Float32, (rows, blocks * NVFP4_BLOCK_SIZE))
    fill!(padded, 0f0)
    padded[:, 1:input] = values
    grouped = reshape(padded, rows, NVFP4_BLOCK_SIZE, blocks)
    block_amax = dropdims(maximum(abs, grouped; dims=2); dims=2)
    block_scales = _nvfp4_encode_e4m3.((block_amax ./ 6f0) ./ global_scale)
    decoded = _nvfp4_decode_e4m3.(block_scales)
    denominator = reshape(ifelse.(iszero.(decoded), 1f0, decoded), rows, 1, blocks)
    normalized = (grouped ./ global_scale) ./ denominator
    # Blocks whose scale underflowed to zero reconstruct to zero. Use canonical
    # positive zero encodings there, including padded elements.
    codes = ifelse.(reshape(iszero.(decoded), rows, 1, blocks), UInt8(0), _nvfp4_encode_e2m1.(normalized))
    flat = reshape(codes, rows, blocks * NVFP4_BLOCK_SIZE)
    packed = flat[:, 1:2:end] .| (flat[:, 2:2:end] .<< UInt8(4))
    return NVFP4Weight(packed, block_scales, global_scale, input)
end

"""Reconstruct a NVFP4 matrix to Float32 on the same device, dropping padding."""
function dequantize_nvfp4(weight::NVFP4Weight)
    rows, half = size(weight.packed)
    blocks = size(weight.block_scales, 2)
    low = _nvfp4_decode_e2m1.(weight.packed .& UInt8(15))
    high = _nvfp4_decode_e2m1.(weight.packed .>> UInt8(4))
    flat = reshape(cat(reshape(low, rows, 1, half), reshape(high, rows, 1, half); dims=2), rows, 2half)
    grouped = reshape(flat, rows, NVFP4_BLOCK_SIZE, blocks)
    scales = reshape(_nvfp4_decode_e4m3.(weight.block_scales), rows, 1, blocks)
    result = reshape((grouped .* scales) .* weight.global_scale, rows, 2half)
    return result[:, 1:weight.in_dim]
end

_flint_dequantize(weight::NVFP4Weight) = dequantize_nvfp4(weight)
function _quant_row_slice(weight::NVFP4Weight, rows)
    rows = _quantized_row_range(rows, size(weight, 1))
    return NVFP4Weight(weight.packed[rows, :], weight.block_scales[rows, :], weight.global_scale, weight.in_dim)
end

quantized_parameter_bytes(weight::NVFP4Weight) =
    _quantized_byte_sum((weight.packed, weight.block_scales)) + sizeof(Float32)
Adapt.adapt_structure(to, weight::NVFP4Weight) = NVFP4Weight(
    Adapt.adapt(to, weight.packed), Adapt.adapt(to, weight.block_scales), weight.global_scale, weight.in_dim,
)

"""Prepared native FP4 GEMM weights. Concrete implementations live in device extensions."""
abstract type AbstractNativeNVFP4Weight end

"""Prepare a CUDA NVFP4 weight for native cuBLASLt GEMM without expanding its values to floating point."""
function prepare_nvfp4_gemm end
prepare_nvfp4_gemm(weight::NVFP4Weight; kwargs...) = throw(ArgumentError(
    "native NVFP4 GEMM requires a CUDA-resident weight and the LifeAI CUDA extension",
))

"""
    native_nvfp4_gemm(weight, activations; check_finite=true)

Multiply prepared CUDA weights by NVFP4-quantized activations using native
FP4 GEMM. Single-token inputs use a fused device quantizer and device alpha.
The default checks for nonfinite activations. Inference internals can pass
`check_finite=false` to keep decode asynchronous when final logits are checked
before sampling. Scratch space is shared within a CUDA context: same-stream
calls execute in order and transfers to another stream wait for prior work.
"""
function native_nvfp4_gemm end
"""
    native_nvfp4_gemm_multi(weights::Tuple, activations; check_finite=true)

Apply one to sixteen prepared native FP4 projections to the same input and return
an output tuple. CUDA single-token groups quantize the shared
activation once, while retaining a separate device alpha for each weight.
Other input shapes use the existing individual native GEMM calls.
"""
function native_nvfp4_gemm_multi end
function native_nvfp4_gemm_multi(weights::Tuple, activations; check_finite::Bool=true)
    1 <= length(weights) <= 16 || throw(ArgumentError("native NVFP4 multi-projection accepts one to sixteen weights"))
    all(weight -> weight isa AbstractNativeNVFP4Weight,weights) ||
        throw(ArgumentError("native NVFP4 multi-projection requires prepared native weights"))
    return map(weight -> native_nvfp4_gemm(weight,activations;check_finite), weights)
end
"""Return the original portable NVFP4 weight from a prepared native weight."""
function nvfp4_weight end
"""Report successful native GEMM calls and backend resource usage."""
function native_nvfp4_stats end
"""Test whether the current CUDA device/runtime can execute native NVFP4 GEMM. Load CUDA first."""
function native_nvfp4_supported end

Base.size(weight::AbstractNativeNVFP4Weight) = size(nvfp4_weight(weight))
Base.size(weight::AbstractNativeNVFP4Weight, dim::Integer) = size(nvfp4_weight(weight), dim)
quantized_parameter_bytes(weight::AbstractNativeNVFP4Weight) = quantized_parameter_bytes(nvfp4_weight(weight))
