using Test
using BFloat16s: BFloat16
using JSON3
using SHA: sha256
import MLDataDevices
using LifeAI:
    Int4GroupWeight,
    Int8ChannelWeight,
    LinearQuantizationSpec,
    Qwen3DenseSpec,
    QuantizationPlan,
    estimate_qwen3_quantized_bytes,
    load_hf_qwen3_model,
    load_hf_qwen3_quantized,
    quantization_spec,
    quantize_bf16_parameters,
    quantized_parameter_bytes,
    qwen3_dense_spec

isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tiny_model_fixture.jl"))
isdefined(@__MODULE__, :quantized_parameters_equal) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_quantization_test_utils.jl"))

const _QWEN3_CALIBRATED_QUANTIZATION_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_calibrated_int4",
    "assets.json",
)

function _quantization_argument_error_message(f)
    try
        f()
    catch exception
        exception isa ArgumentError || rethrow()
        return exception.msg
    end
    return nothing
end

struct _Ch17ZeroBasedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

_Ch17ZeroBasedArray(data::A) where {T,N,A<:AbstractArray{T,N}} =
    _Ch17ZeroBasedArray{T,N,A}(data)
Base.size(values::_Ch17ZeroBasedArray) = size(values.data)
Base.axes(values::_Ch17ZeroBasedArray) = ntuple(
    dimension -> 0:(size(values, dimension) - 1),
    ndims(values),
)
Base.IndexStyle(::Type{<:_Ch17ZeroBasedArray}) = IndexCartesian()
Base.getindex(values::_Ch17ZeroBasedArray, indices::Vararg{Int,N}) where {N} =
    getindex(values.data, map(index -> index + 1, indices)...)

struct _Ch17ForeignDeviceVector{T,A<:AbstractVector{T}} <: AbstractVector{T}
    data::A
end
struct _Ch17ForeignDevice <: MLDataDevices.AbstractDevice end
Base.size(values::_Ch17ForeignDeviceVector) = size(values.data)
Base.axes(values::_Ch17ForeignDeviceVector) = axes(values.data)
Base.IndexStyle(::Type{<:_Ch17ForeignDeviceVector}) = IndexLinear()
Base.getindex(values::_Ch17ForeignDeviceVector, index::Int) = values.data[index]
MLDataDevices.get_device(::_Ch17ForeignDeviceVector) = _Ch17ForeignDevice()

struct _Ch17ForeignDeviceMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::A
end
Base.size(values::_Ch17ForeignDeviceMatrix) = size(values.data)
Base.axes(values::_Ch17ForeignDeviceMatrix) = axes(values.data)
Base.IndexStyle(::Type{<:_Ch17ForeignDeviceMatrix}) = IndexCartesian()
Base.getindex(
    values::_Ch17ForeignDeviceMatrix,
    row::Int,
    column::Int,
) = values.data[row, column]
MLDataDevices.get_device(::_Ch17ForeignDeviceMatrix) = _Ch17ForeignDevice()

@testset "INT8 weight tensors are strict" begin
    quantized_storage = reshape(Int8.(1:12), 3, 4)
    scale_storage = Float32[0.25, 0.5, 0.75]
    quantized = @view quantized_storage[1:2, :]
    scales = @view scale_storage[1:2]
    weight = Int8ChannelWeight(quantized, scales)
    @test weight.q === quantized
    @test weight.scale === scales
    @test_throws MethodError Int8ChannelWeight{
        typeof(quantized),
        typeof(scales),
    }(quantized, scales)

    for (values, message) in (
        (Int8[1, 2], "INT8 quantized values must be a matrix"),
        (
            reshape(Int8[1, 2], 1, 1, 2),
            "INT8 quantized values must be a matrix",
        ),
        (reshape(Int16[1, 2], 1, 2), "INT8 quantized values must contain Int8 values"),
        (
            _Ch17ZeroBasedArray(reshape(Int8[1, 2], 1, 2)),
            "INT8 quantized values must use one-based axes",
        ),
        (zeros(Int8, 0, 2), "INT8 quantized values must have positive dimensions"),
        (zeros(Int8, 2, 0), "INT8 quantized values must have positive dimensions"),
    )
        @test _quantization_argument_error_message() do
            Int8ChannelWeight(values, Float32[1])
        end == message
    end
    for (values, message) in (
        (reshape(Float32[1], 1, 1), "INT8 scales must be a vector"),
        (Float64[1], "INT8 scales must contain Float32 values"),
        (
            _Ch17ZeroBasedArray(Float32[1]),
            "INT8 scales must use one-based axes",
        ),
    )
        @test _quantization_argument_error_message() do
            Int8ChannelWeight(reshape(Int8[1], 1, 1), values)
        end == message
    end
    for invalid_scale in (0.0f0, -1.0f0, NaN32, Inf32, -Inf32)
        @test _quantization_argument_error_message() do
            Int8ChannelWeight(
                reshape(Int8[1], 1, 1),
                Float32[invalid_scale],
            )
        end == "INT8 scales must contain only finite positive values"
    end
    for invalid_scales in (Float32[1], Float32[1, 1, 1])
        @test_throws DimensionMismatch Int8ChannelWeight(
            reshape(Int8[1, 2, 3, 4], 2, 2),
            invalid_scales,
        )
    end
    @test _quantization_argument_error_message() do
        Int8ChannelWeight(
            reshape(Int8[1], 1, 1),
            _Ch17ForeignDeviceVector(Float32[1]),
        )
    end == "INT8 quantized values and scales must reside on the same device"
end

@testset "INT4 weight tensors are strict" begin
    packed_storage = reshape(UInt8.(1:12), 3, 4)
    scale_storage = reshape(Float32.(1:6), 3, 2)
    packed = @view packed_storage[1:2, :]
    scales = @view scale_storage[1:2, :]
    weight = Int4GroupWeight(packed, scales, 4, 8)
    @test weight.packed === packed
    @test weight.scale === scales
    @test_throws MethodError Int4GroupWeight{
        typeof(packed),
        typeof(scales),
    }(packed, scales, 4, 8)

    for (values, message) in (
        (UInt8[1, 2], "INT4 packed values must be a matrix"),
        (
            reshape(UInt8[1, 2, 3, 4], 1, 2, 2),
            "INT4 packed values must be a matrix",
        ),
        (
            reshape(UInt16[1, 2, 3, 4], 1, 4),
            "INT4 packed values must contain UInt8 values",
        ),
        (
            _Ch17ZeroBasedArray(reshape(UInt8[1, 2, 3, 4], 1, 4)),
            "INT4 packed values must use one-based axes",
        ),
        (
            zeros(UInt8, 0, 4),
            "INT4 packed values must have a positive output dimension",
        ),
    )
        @test _quantization_argument_error_message() do
            Int4GroupWeight(values, ones(Float32, 1, 2), 4, 8)
        end == message
    end

    valid_packed = reshape(UInt8.(1:8), 2, 4)
    for (values, message) in (
        (Float32[1, 2], "INT4 scales must be a matrix"),
        (
            reshape(Float32[1, 2, 3, 4], 1, 2, 2),
            "INT4 scales must be a matrix",
        ),
        (
            ones(Float64, 2, 2),
            "INT4 scales must contain Float32 values",
        ),
        (
            _Ch17ZeroBasedArray(ones(Float32, 2, 2)),
            "INT4 scales must use one-based axes",
        ),
    )
        @test _quantization_argument_error_message() do
            Int4GroupWeight(valid_packed, values, 4, 8)
        end == message
    end
    for invalid_scale in (0.0f0, -1.0f0, NaN32, Inf32, -Inf32)
        @test _quantization_argument_error_message() do
            Int4GroupWeight(
                valid_packed,
                fill(invalid_scale, 2, 2),
                4,
                8,
            )
        end == "INT4 scales must contain only finite positive values"
    end

    for invalid_packed in (zeros(UInt8, 2, 3), zeros(UInt8, 2, 5))
        @test_throws DimensionMismatch Int4GroupWeight(
            invalid_packed,
            ones(Float32, 2, 2),
            4,
            8,
        )
    end
    for invalid_scales in (
        zeros(Float32, 1, 2),
        zeros(Float32, 3, 2),
        zeros(Float32, 2, 1),
        zeros(Float32, 2, 3),
    )
        @test_throws DimensionMismatch Int4GroupWeight(
            valid_packed,
            invalid_scales,
            4,
            8,
        )
    end
    @test _quantization_argument_error_message() do
        Int4GroupWeight(
            valid_packed,
            _Ch17ForeignDeviceMatrix(ones(Float32, 2, 2)),
            4,
            8,
        )
    end == "INT4 packed values and scales must reside on the same device"
end

@testset "quantized row slicing is strict" begin
    int8 = Int8ChannelWeight(
        reshape(Int8.(1:12), 3, 4),
        Float32[0.25, 0.5, 0.75],
    )
    int4 = Int4GroupWeight(
        reshape(UInt8.(1:12), 3, 4),
        reshape(Float32.(1:6), 3, 2),
        4,
        8,
    )
    int8_slice = LifeAI._quant_row_slice(int8, 2:3)
    int4_slice = LifeAI._quant_row_slice(int4, 2:3)
    @test LifeAI._dequantize_bf16(int8_slice) ==
        LifeAI._dequantize_bf16(int8)[2:3, :]
    @test LifeAI._dequantize_bf16(int4_slice) ==
        LifeAI._dequantize_bf16(int4)[2:3, :]

    for weight in (int8, int4)
        for rows in (1, true, Int[1], Int32(1):Int32(1), 1:2:3, Colon(), 1:0)
            @test _quantization_argument_error_message() do
                LifeAI._quant_row_slice(weight, rows)
            end == "quantized row selection must be a non-empty UnitRange{Int}"
        end
        for rows in (0:1, 2:4)
            @test _quantization_argument_error_message() do
                LifeAI._quant_row_slice(weight, rows)
            end == "quantized row selection must lie within 1:3"
        end
    end
end

@testset "trace-safe quantized dequantization preserves host values" begin
    int8_values = reshape(Int8[-128, -1, 0, 127], 1, :)
    int8_scale = Float32[0.25]
    int8_weight = Int8ChannelWeight(int8_values, int8_scale)
    @test LifeAI._dequantize_bf16(int8_weight) ==
        BFloat16.(Float32.(int8_values) .* reshape(int8_scale, :, 1))

    packed_values = reshape(UInt8.(0:255), 1, :)
    int4_weight = Int4GroupWeight(
        packed_values,
        ones(Float32, 1, 256),
        2,
        512,
    )
    expected = BFloat16[
        isodd(column) ?
            Int(packed_values[cld(column, 2)] % UInt8(16)) - 8 :
            Int(packed_values[cld(column, 2)] ÷ UInt8(16)) - 8
        for column in 1:512
    ]
    @test vec(LifeAI._dequantize_bf16(int4_weight)) == expected
end

@testset "INT4 reconstruction-MSE calibration" begin
    # Each group has one outlier and many unit-scale values. Candidate 0.9
    # clips the outlier slightly but lowers total reconstruction error.
    row = Float32[10; fill(1, 15)]
    weight = BFloat16.(vcat(
        reshape(row, 1, :),
        reshape(-row, 1, :),
        reshape(reverse(row), 1, :),
    ))
    maxabs = LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:maxabs,
    )
    calibrated = LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:mse,
    )
    maxabs_reconstructed = Float32.(LifeAI._dequantize_bf16(maxabs))
    calibrated_reconstructed = Float32.(LifeAI._dequantize_bf16(calibrated))
    source = Float32.(weight)
    maxabs_error = sum(abs2, maxabs_reconstructed .- source)
    calibrated_error = sum(abs2, calibrated_reconstructed .- source)
    @test calibrated_error < maxabs_error
    @test any(calibrated.scale .< maxabs.scale)

    # Including 1.0 in the candidates makes MSE calibration no worse than the
    # max-abs baseline independently for every row/group.
    matrix = BFloat16.(reshape(
        Float32[sin(i / 7) * (iszero(i % 29) ? 8 : 1) for i in 1:(4 * 64)],
        4,
        64,
    ))
    baseline = LifeAI._quantize_int4_group(matrix; group=16)
    mse = LifeAI._quantize_int4_group(
        matrix;
        group=16,
        calibration=:mse,
    )
    baseline_dequant = Float32.(LifeAI._dequantize_bf16(baseline))
    mse_dequant = Float32.(LifeAI._dequantize_bf16(mse))
    for row_index in axes(matrix, 1), group_index in 1:4
        columns = ((group_index - 1) * 16 + 1):(group_index * 16)
        source_group = Float32.(matrix[row_index, columns])
        @test sum(abs2, mse_dequant[row_index, columns] .- source_group) <=
            sum(abs2, baseline_dequant[row_index, columns] .- source_group)
    end

    # Qwen3 XLA decode and quantization's default call remains exactly max-abs RTN.
    legacy = LifeAI._quantize_int4_group(matrix; group=16)
    explicit = LifeAI._quantize_int4_group(
        matrix;
        group=16,
        calibration=:maxabs,
    )
    @test legacy.packed == explicit.packed
    @test legacy.scale == explicit.scale
end

@testset "quantized byte accounting rejects host overflow" begin
    maximum_count = typemax(Int)
    overflow_spec = Qwen3DenseSpec(
        :overflow,
        "overflow",
        "overflow",
        "overflow",
        1,
        1,
        1,
        1,
        1,
        1,
        maximum_count,
        1.0f-6,
        1.0f4,
        1,
        true,
    )
    bf16_plan = QuantizationPlan(
        default=LinearQuantizationSpec(:bf16),
    )
    @test_throws ArgumentError estimate_qwen3_quantized_bytes(
        overflow_spec,
        bf16_plan,
    )

    huge_depth_spec = Qwen3DenseSpec(
        :huge_depth,
        "huge-depth",
        "huge-depth",
        "huge-depth",
        1,
        2,
        2,
        maximum_count,
        1,
        1,
        2,
        1.0f-6,
        1.0f4,
        1,
        true,
    )
    @test _quantization_argument_error_message() do
        estimate_qwen3_quantized_bytes(huge_depth_spec, bf16_plan)
    end == "Qwen3 quantized parameter byte estimate exceeds the host integer range"

    two_layer_spec = Qwen3DenseSpec(
        :two_layer,
        "two-layer",
        "two-layer",
        "two-layer",
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        1.0f-6,
        1.0f4,
        2,
        true,
    )
    fully_overridden = QuantizationPlan(
        default=LinearQuantizationSpec(:bf16),
        projection_overrides=Dict(
            :q_proj => LinearQuantizationSpec(:int4; group=4),
        ),
        layer_overrides=Dict(
            (1, :q_proj) => LinearQuantizationSpec(:bf16),
            (2, :q_proj) => LinearQuantizationSpec(:bf16),
        ),
    )
    @test estimate_qwen3_quantized_bytes(two_layer_spec, fully_overridden) ==
        estimate_qwen3_quantized_bytes(two_layer_spec, bf16_plan)

    single_length = 2 * (maximum_count ÷ sizeof(Int)) + 3
    @test_throws ArgumentError quantized_parameter_bytes(
        Base.OneTo(single_length),
    )
    aggregate_leaf = Base.OneTo(maximum_count ÷ sizeof(Int))
    @test_throws ArgumentError quantized_parameter_bytes(
        (aggregate_leaf, aggregate_leaf, aggregate_leaf),
    )
    @test quantized_parameter_bytes(Bool[true, false]) == 2
    @test quantized_parameter_bytes(ComplexF32[1 + 2im]) == 8
    for values in (
        BigInt[1, big(2)^1000],
        Complex{BigInt}[1 + 2im],
        Any[1, 2],
        ["a", "b"],
    )
        @test _quantization_argument_error_message() do
            quantized_parameter_bytes(values)
        end ==
              "quantized tensor storage requires a fixed-size isbits element type"
    end
end

@testset "INT4 weight metadata is strict" begin
    packed = zeros(UInt8, 2, 4)
    scale = ones(Float32, 2, 2)
    weight = Int4GroupWeight(packed, scale, Int128(4), big(8))
    @test weight.group === 4
    @test weight.in_dim === 8

    too_large = big(typemax(Int)) + 1
    for (group, message) in (
        (true, "INT4 weight group must be an integer"),
        (1.0, "INT4 weight group must be an integer"),
        (1 + 0im, "INT4 weight group must be an integer"),
        (
            too_large,
            "INT4 weight group is outside the host integer range",
        ),
        (0, "INT4 weight group must be positive"),
        (-2, "INT4 weight group must be positive"),
        (3, "INT4 weight group must be even"),
        (6, "INT4 weight input dimension must be divisible by its group"),
    )
        @test _quantization_argument_error_message() do
            Int4GroupWeight(packed, scale, group, 8)
        end == message
    end
    for (in_dim, message) in (
        (true, "INT4 weight input dimension must be an integer"),
        (1.0, "INT4 weight input dimension must be an integer"),
        (1 + 0im, "INT4 weight input dimension must be an integer"),
        (
            too_large,
            "INT4 weight input dimension is outside the host integer range",
        ),
        (0, "INT4 weight input dimension must be positive"),
        (-2, "INT4 weight input dimension must be positive"),
        (7, "INT4 weight input dimension must be even"),
    )
        @test _quantization_argument_error_message() do
            Int4GroupWeight(packed, scale, 4, in_dim)
        end == message
    end
end

@testset "quantization plan validation and precedence" begin
    int4_mse = LinearQuantizationSpec(:int4; group=128, calibration=:mse)
    int8 = LinearQuantizationSpec(:int8)
    bf16 = LinearQuantizationSpec(:bf16)
    @test LinearQuantizationSpec(:int8; group=Int128(64)).group == 64
    @test _quantization_argument_error_message() do
        LinearQuantizationSpec(:int8; group=true)
    end == "quantization group must be an integer"
    @test _quantization_argument_error_message() do
        LinearQuantizationSpec(:int8; group=big(typemax(Int)) + 1)
    end == "quantization group is outside the host integer range"
    widened_ratios = LinearQuantizationSpec(
        :int4;
        calibration=:mse,
        clip_ratios=BigFloat[1, 0.9],
    ).clip_ratios
    @test widened_ratios === (1.0f0, 0.9f0)
    mixed_ratios = LinearQuantizationSpec(
        :int4;
        calibration=:mse,
        clip_ratios=Any[1, 1 // 2, 0.25],
    ).clip_ratios
    @test mixed_ratios === (1.0f0, 0.5f0, 0.25f0)
    for value in (true, 0.9 + 0im, "0.9", nothing)
        @test _quantization_argument_error_message() do
            LinearQuantizationSpec(
                :int4;
                calibration=:mse,
                clip_ratios=(1.0, value),
            )
        end == "INT4 clipping ratios must contain real numbers other than Bool"
    end
    for invalid_container in (nothing, Set((1.0, 0.9)))
        @test _quantization_argument_error_message() do
            LinearQuantizationSpec(
                :int4;
                calibration=:mse,
                clip_ratios=invalid_container,
            )
        end == "INT4 clipping ratios must be a tuple or vector"
    end
    @test _quantization_argument_error_message() do
        LinearQuantizationSpec(
            :int4;
            calibration=:mse,
            clip_ratios=(1.0, nextfloat(1.0)),
        )
    end == "INT4 clipping ratios must be finite in (0, 1]"
    @test _quantization_argument_error_message() do
        LinearQuantizationSpec(
            :int4;
            calibration=:mse,
            clip_ratios=(1.0, BigFloat("1e-1000")),
        )
    end ==
          "INT4 clipping ratios must be finite in (0, 1] at Float32 precision"
    plan = QuantizationPlan(
        default=int4_mse,
        projection_overrides=Dict(:q_proj => int8, :lm_head => bf16),
        layer_overrides=Dict((2, :q_proj) => bf16),
    )
    @test quantization_spec(plan, :k_proj; layer=2) === int4_mse
    @test quantization_spec(plan, :q_proj; layer=1) === int8
    @test quantization_spec(plan, :q_proj; layer=2) === bf16
    @test quantization_spec(plan, :lm_head) === bf16

    owned_projection_overrides =
        Dict{Symbol,LinearQuantizationSpec}(:q_proj => int8)
    owned_layer_overrides =
        Dict{Tuple{Int,Symbol},LinearQuantizationSpec}((2, :q_proj) => bf16)
    owned_plan = QuantizationPlan(
        int4_mse,
        owned_projection_overrides,
        owned_layer_overrides,
    )
    owned_projection_overrides[:q_proj] = bf16
    empty!(owned_layer_overrides)
    @test quantization_spec(owned_plan, :q_proj; layer=1) === int8
    @test quantization_spec(owned_plan, :q_proj; layer=2) === bf16
    projection_snapshot = owned_plan.projection_overrides
    layer_snapshot = owned_plan.layer_overrides
    projection_snapshot[:q_proj] = bf16
    empty!(layer_snapshot)
    @test quantization_spec(owned_plan, :q_proj; layer=1) === int8
    @test quantization_spec(owned_plan, :q_proj; layer=2) === bf16

    wide_plan = QuantizationPlan(
        int4_mse,
        Dict(:q_proj => int8),
        Dict((Int128(2), :q_proj) => bf16),
    )
    @test quantization_spec(wide_plan, :q_proj; layer=Int128(2)) === bf16

    too_large = big(typemax(Int)) + 1
    for (value, message) in (
        (true, "quantization override layer must be an integer"),
        (1.0, "quantization override layer must be an integer"),
        (
            too_large,
            "quantization override layer is outside the host integer range",
        ),
    )
        @test _quantization_argument_error_message() do
            QuantizationPlan(
                layer_overrides=Dict((value, :q_proj) => int8),
            )
        end == message
    end
    for (value, message) in (
        (true, "quantization layer must be an integer"),
        (1.0, "quantization layer must be an integer"),
        (too_large, "quantization layer is outside the host integer range"),
    )
        @test _quantization_argument_error_message() do
            quantization_spec(plan, :q_proj; layer=value)
        end == message
    end

    @test _quantization_argument_error_message() do
        QuantizationPlan(
            int4_mse,
            Dict{Symbol,LinearQuantizationSpec}(),
            Dict{Tuple{Int,Symbol},LinearQuantizationSpec}(
                (0, :q_proj) => int8,
            ),
        )
    end == "quantization override layers must be positive one-based integers"

    @test_throws ArgumentError LinearQuantizationSpec(:int3)
    @test_throws ArgumentError LinearQuantizationSpec(:int4; group=3)
    @test_throws ArgumentError LinearQuantizationSpec(:int8; calibration=:mse)
    @test_throws ArgumentError LinearQuantizationSpec(
        :int4;
        calibration=:mse,
        clip_ratios=(0.9f0, 0.8f0),
    )
    @test_throws ArgumentError LinearQuantizationSpec(
        :int4;
        calibration=:mse,
        clip_ratios=(1.0f0, 0.0f0),
    )
    @test_throws ArgumentError QuantizationPlan(
        projection_overrides=Dict(:unknown => int8),
    )
    @test_throws ArgumentError QuantizationPlan(
        layer_overrides=Dict((0, :q_proj) => int8),
    )
    @test_throws ArgumentError QuantizationPlan(
        layer_overrides=Dict((1, :lm_head) => int8),
    )
    @test_throws ArgumentError quantization_spec(plan, :q_proj; layer=0)
    @test_throws ArgumentError quantization_spec(plan, :lm_head; layer=1)
end

@testset "one plan drives in-memory and streamed quantization" begin
    mktempdir() do directory
        model = _qwen3_tiny_model_fixture_dir(directory; tie=false)
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        int4_mse = LinearQuantizationSpec(
            :int4;
            group=4,
            calibration=:mse,
            clip_ratios=(1.0f0, 0.9f0, 0.8f0),
        )
        int8 = LinearQuantizationSpec(:int8; group=4)
        bf16 = LinearQuantizationSpec(:bf16; group=4)
        plan = QuantizationPlan(
            default=int4_mse,
            projection_overrides=Dict(
                :q_proj => int8,
                :down_proj => bf16,
                :lm_head => int8,
            ),
            layer_overrides=Dict(
                (1, :q_proj) => bf16,
                (2, :down_proj) => int8,
            ),
        )

        in_memory = quantize_bf16_parameters(loaded.parameters; plan)
        streamed = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan,
        )
        @test quantized_parameters_equal(in_memory, streamed.parameters)
        @test in_memory.blocks.layer_1.attn.q_proj.weight isa Matrix{BFloat16}
        @test in_memory.blocks.layer_2.attn.q_proj.weight isa Int8ChannelWeight
        @test in_memory.blocks.layer_1.mlp.down_proj.weight isa Matrix{BFloat16}
        @test in_memory.blocks.layer_2.mlp.down_proj.weight isa Int8ChannelWeight
        @test in_memory.blocks.layer_1.mlp.gate_proj.weight isa Int4GroupWeight
        @test in_memory.lm_head.weight isa Int8ChannelWeight

        actual_bytes = quantized_parameter_bytes(in_memory)
        @test actual_bytes == estimate_qwen3_quantized_bytes(model, plan)
        out_of_range = QuantizationPlan(
            layer_overrides=Dict((model.num_layers + 1, :q_proj) => int8),
        )
        @test_throws ArgumentError estimate_qwen3_quantized_bytes(
            model,
            out_of_range,
        )
        @test_throws ArgumentError quantize_bf16_parameters(
            loaded.parameters;
            plan=out_of_range,
        )

        # Legacy Qwen3 XLA decode and quantization arguments resolve to the same plan and tensor values.
        legacy = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            scheme=:int4,
            group=4,
            int8_projections=(:q_proj, :lm_head),
        )
        equivalent_plan = QuantizationPlan(
            default=LinearQuantizationSpec(:int4; group=4),
            projection_overrides=Dict(
                :q_proj => LinearQuantizationSpec(:int8; group=4),
                :lm_head => LinearQuantizationSpec(:int8; group=4),
            ),
        )
        planned = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan=equivalent_plan,
        )
        @test quantized_parameters_equal(legacy.parameters, planned.parameters)
    end

    mktempdir() do directory
        model = _qwen3_tiny_model_fixture_dir(directory; tie=true)
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        plan = QuantizationPlan(
            default=LinearQuantizationSpec(:int4; group=4),
            projection_overrides=Dict(
                :lm_head => LinearQuantizationSpec(:bf16; group=4),
            ),
        )
        in_memory = quantize_bf16_parameters(loaded.parameters; plan)
        streamed = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan,
        )
        @test isempty(in_memory.lm_head)
        @test quantized_parameters_equal(in_memory, streamed.parameters)
        @test quantized_parameter_bytes(in_memory) ==
            estimate_qwen3_quantized_bytes(model, plan)
    end
end

@testset "Qwen3-14B frozen tensor-byte budgets" begin
    fixture = JSON3.read(read(_QWEN3_CALIBRATED_QUANTIZATION_ASSETS_PATH, String))
    for (filename, expected) in pairs(fixture["plan_sha256"])
        path = joinpath(dirname(_QWEN3_CALIBRATED_QUANTIZATION_ASSETS_PATH), String(filename))
        @test bytes2hex(sha256(read(path))) == String(expected)
    end
    qwen = fixture["qwen3_14b"]
    spec = qwen3_dense_spec(:qwen3_14b)
    pure_int4 = QuantizationPlan(
        default=LinearQuantizationSpec(
            :int4;
            group=128,
            calibration=:mse,
        ),
    )
    pure_int8 = QuantizationPlan(
        default=LinearQuantizationSpec(:int8),
    )
    mixed_head = QuantizationPlan(
        default=pure_int4.default,
        projection_overrides=Dict(
            :lm_head => LinearQuantizationSpec(:int8),
        ),
    )
    mixed_24g_overrides = Dict(
        projection => LinearQuantizationSpec(:int8)
        for projection in (:q_proj, :k_proj, :v_proj, :o_proj, :down_proj)
    )
    mixed_24g_overrides[:lm_head] = LinearQuantizationSpec(:bf16)
    mixed_24g = QuantizationPlan(
        default=pure_int4.default,
        projection_overrides=mixed_24g_overrides,
    )
    @test estimate_qwen3_quantized_bytes(spec, pure_int8) ==
        Int(qwen["int8_tensor_bytes"])
    @test estimate_qwen3_quantized_bytes(spec, pure_int4) ==
        Int(qwen["pure_int4_g128_tensor_bytes"])
    @test estimate_qwen3_quantized_bytes(spec, mixed_head) ==
        Int(qwen["int4_g128_int8_lm_head_tensor_bytes"])
    @test estimate_qwen3_quantized_bytes(spec, mixed_head) >
        estimate_qwen3_quantized_bytes(spec, pure_int4)
    @test estimate_qwen3_quantized_bytes(spec, mixed_24g) ==
        Int(qwen["mixed_24g_tensor_bytes"])

    hardware = fixture["hardware_validation"]
    @test String(hardware["status"]) == "complete"
    @test Bool(hardware["model_matches_frozen_huggingface_revision"])
    @test String(hardware["gpu"]) == "NVIDIA GeForce RTX 4090 D"
    runs = hardware["runs"]
    int8_run = runs["int8"]
    mse_run = runs["mixed_mse"]
    rtn_run = runs["mixed_rtn"]
    for run in (int8_run, mse_run, rtn_run)
        @test Int(run["estimated_tree_bytes"]) == Int(run["host_tree_bytes"])
        @test Int(run["gpu_tree_bytes"]) >= Int(run["host_tree_bytes"])
        @test Int(run["vram_used_bytes"]) <= Int(hardware["gpu_total_bytes"])
        @test Bool(run["logits_argmax_equal"])
        @test Bool(run["decode_argmax_equal"])
    end
    @test Int(int8_run["estimated_tree_bytes"]) ==
        Int(qwen["int8_tensor_bytes"])
    @test Int(mse_run["estimated_tree_bytes"]) ==
        Int(qwen["mixed_24g_tensor_bytes"])
    @test Int(rtn_run["estimated_tree_bytes"]) ==
        Int(qwen["mixed_24g_tensor_bytes"])
    @test Int(int8_run["greedy_agreement"]) == 16
    @test Int(rtn_run["greedy_agreement"]) == 16
    @test Int(mse_run["greedy_agreement"]) == 4
    @test Int(mse_run["greedy_first_divergence"]) == 5
    @test Float64(mse_run["logits_max_abs"]) <
        Float64(rtn_run["logits_max_abs"])
    @test Float64(mse_run["logits_mean_abs"]) <
        Float64(rtn_run["logits_mean_abs"])
end
