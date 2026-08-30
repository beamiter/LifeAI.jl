using Test
using LifeAI: Qwen3VLProcessorSpec,
    Qwen3VLVisionInput,
    Qwen3VLVisionSpec,
    qwen3_vl_image_grid,
    qwen3_vl_image_token_count,
    qwen3_vl_patchify,
    qwen3_vl_processor_spec,
    qwen3_vl_smart_resize

function _ch43_processor_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected Qwen3-VL processor specification to fail")
end

mutable struct _Ch43ReadCountingImage{T,N} <: AbstractArray{T,N}
    dimensions::NTuple{N,Int}
    reads::Int
end

Base.size(image::_Ch43ReadCountingImage) = image.dimensions
Base.IndexStyle(::Type{<:_Ch43ReadCountingImage}) = IndexLinear()

function Base.getindex(image::_Ch43ReadCountingImage{T}, index::Int) where {T}
    image.reads += 1
    return zero(T)
end

@testset "Qwen3-VL processor specifications are strict" begin
    valid = (
        SubString("xhash", 2),
        SubString("xprocessor", 2),
        SubString("ximage", 2),
        big(1),
        Int32(1_000),
        UInt8(2),
        big(2),
        Int128(2),
        (Float64(0.25), Int32(0), Float32(-0.25)),
        (Float64(0.5), Int32(1), Float32(2)),
    )
    spec = Qwen3VLProcessorSpec(valid...)
    @test spec.preprocessor_config_sha256 === "hash"
    @test spec.processor_class === "processor"
    @test spec.min_pixels === 1
    @test spec.image_mean === (0.25f0, 0.0f0, -0.25f0)
    @test spec.image_std === (0.5f0, 1.0f0, 2.0f0)

    names = fieldnames(Qwen3VLProcessorSpec)
    for index in 4:8
        label = names[index]
        for (value, message) in (
            (true, "must be an integer"),
            (0, "must be positive"),
        )
            failure = _ch43_processor_error() do
                Qwen3VLProcessorSpec(Base.setindex(valid, value, index)...)
            end
            @test failure isa ArgumentError
            @test sprint(showerror, failure) ==
                "ArgumentError: Qwen3-VL processor $label $message"
        end
    end

    too_large = big(typemax(Int)) + 1
    for (index, value, message) in (
        (4, 1.0, "min_pixels must be an integer"),
        (4, too_large, "min_pixels is outside the host integer range"),
        (1, :hash, "preprocessor_config_sha256 must be a string"),
        (2, :processor, "processor_class must be a string"),
        (3, :image, "image_processor_type must be a string"),
        (1, "", "preprocessor_config_sha256 must not be empty"),
        (2, "", "processor_class must not be empty"),
        (3, "", "image_processor_type must not be empty"),
    )
        failure = _ch43_processor_error() do
            Qwen3VLProcessorSpec(Base.setindex(valid, value, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL processor $message"
    end

    for (index, values, message) in (
        (9, [0.0, 0.0, 0.0], "image_mean must be a tuple of three numbers"),
        (9, (0.0, 0.0), "image_mean must be a tuple of three numbers"),
        (9, (true, 0.0, 0.0), "image_mean[1] must be a real number"),
        (
            9,
            (big(10)^1_000, 0.0, 0.0),
            "image_mean[1] must be finite at Float32 precision",
        ),
        (
            10,
            (0.0, 1.0, 1.0),
            "image_std[1] must be positive and finite at Float32 precision",
        ),
        (
            10,
            (-1.0, 1.0, 1.0),
            "image_std[1] must be positive and finite at Float32 precision",
        ),
    )
        failure = _ch43_processor_error() do
            Qwen3VLProcessorSpec(Base.setindex(valid, values, index)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL processor $message"
    end

    for (values, message) in (
        (
            Base.setindex(valid, 1_001, 4),
            "min_pixels must not exceed max_pixels",
        ),
        (
            Base.setindex(valid, 15, 5),
            "max_pixels must be at least resize factor squared (16)",
        ),
        (
            Base.setindex(valid, typemax(Int), 6),
            "resize factor exceeds the host integer range",
        ),
    )
        failure = _ch43_processor_error() do
            Qwen3VLProcessorSpec(values...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL processor $message"
    end
end

@testset "Qwen3-VL smart resize and image grid" begin
    spec = qwen3_vl_processor_spec()
    @test qwen3_vl_smart_resize(256, 256) == (256, 256)
    @test qwen3_vl_smart_resize(32, 32) == (256, 256)
    @test qwen3_vl_smart_resize(100, 200) == (192, 384)
    @test qwen3_vl_smart_resize(8_192, 8_192) == (4_096, 4_096)
    @test qwen3_vl_smart_resize(1, 200) == (32, 3_648)

    # Python round and Julia RoundNearest both use ties-to-even.
    @test qwen3_vl_smart_resize(
        48,
        80;
        factor=32,
        min_pixels=1,
        max_pixels=1_000_000,
    ) == (64, 64)
    @test qwen3_vl_smart_resize(
        80,
        144;
        factor=32,
        min_pixels=1,
        max_pixels=1_000_000,
    ) == (64, 128)
    @test qwen3_vl_smart_resize(
        typemax(Int),
        typemax(Int);
        factor=1,
        min_pixels=1,
        max_pixels=1_000,
    ) == (31, 31)
    @test qwen3_vl_smart_resize(
        typemax(Int),
        typemax(Int);
        factor=2,
        min_pixels=1,
        max_pixels=1_000,
    ) == (30, 30)
    @test qwen3_vl_smart_resize(typemax(Int), typemax(Int)) == (4_096, 4_096)

    impossible_factor = _ch43_processor_error() do
        qwen3_vl_smart_resize(
            1,
            1;
            factor=32,
            min_pixels=1,
            max_pixels=1_000,
        )
    end
    @test impossible_factor isa ArgumentError
    @test sprint(showerror, impossible_factor) ==
        "ArgumentError: max_pixels must be at least factor squared (1024)"
    for (dimensions, produced) in (
        ((1, 1), "64 × 64 = 4096"),
        ((1_000, 1_000), "32 × 32 = 1024"),
    )
        failure = _ch43_processor_error() do
            qwen3_vl_smart_resize(
                dimensions...;
                factor=32,
                min_pixels=2_048,
                max_pixels=2_048,
            )
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: resize policy cannot satisfy pixel budget " *
            "[2048, 2048] at factor 32; produced $produced pixels"
    end

    @test qwen3_vl_image_grid(256, 256) == (1, 16, 16)
    @test qwen3_vl_image_grid((100, 200)) == (1, 12, 24)
    @test qwen3_vl_image_token_count((1, 16, 16)) == 64
    @test qwen3_vl_image_token_count(256, 256) == 64
    @test qwen3_vl_image_token_count(100, 200) == 72

    @test_throws ArgumentError qwen3_vl_smart_resize(0, 10)
    @test_throws ArgumentError qwen3_vl_smart_resize(true, 10)
    @test_throws ArgumentError qwen3_vl_smart_resize(1, 201)
    @test_throws ArgumentError qwen3_vl_smart_resize(
        10,
        10;
        min_pixels=100,
        max_pixels=99,
    )
    @test_throws ArgumentError qwen3_vl_image_token_count((2, 16, 16))
    @test_throws ArgumentError qwen3_vl_image_token_count((1, 15, 16))
    @test_throws ArgumentError qwen3_vl_image_token_count((1, 16))

    @test spec.patch_size * spec.merge_size == 32
end

@testset "Qwen3-VL patchify hand oracle" begin
    compact = Qwen3VLProcessorSpec(
        "test-only",
        "Qwen3VLProcessor",
        "Qwen2VLImageProcessorFast",
        1,
        1_000,
        1,
        2,
        2,
        (0.5f0, 0.5f0, 0.5f0),
        (0.5f0, 0.5f0, 0.5f0),
    )
    image = Array{Float32}(undef, 3, 2, 4)
    image[1, :, :] = Float32[11 12 13 14; 21 22 23 24]
    image[2, :, :] = Float32[111 112 113 114; 121 122 123 124]
    image[3, :, :] = Float32[211 212 213 214; 221 222 223 224]

    # Literal independent oracle: columns are merge-group ordered, and each
    # channel value is broadcast to both temporal slots.
    expected = Float32[
        11 12 21 22 13 14 23 24
        11 12 21 22 13 14 23 24
        111 112 121 122 113 114 123 124
        111 112 121 122 113 114 123 124
        211 212 221 222 213 214 223 224
        211 212 221 222 213 214 223 224
    ]
    @test qwen3_vl_patchify(image; spec=compact) == expected

    patch2 = Qwen3VLProcessorSpec(
        "test-only",
        "Qwen3VLProcessor",
        "Qwen2VLImageProcessorFast",
        1,
        1_000,
        2,
        2,
        2,
        (0.5f0, 0.5f0, 0.5f0),
        (0.5f0, 0.5f0, 0.5f0),
    )
    patch2_image = Array{Float32}(undef, 3, 4, 4)
    for channel in 1:3, height in 1:4, width in 1:4
        patch2_image[channel, height, width] =
            Float32(100 * channel + 10 * height + width)
    end
    patch2_values = qwen3_vl_patchify(patch2_image; spec=patch2)
    @test size(patch2_values) == (24, 4)
    @test patch2_values[1:4, :] == Float32[
        111 113 131 133
        112 114 132 134
        121 123 141 143
        122 124 142 144
    ]
    @test patch2_values[5:8, :] == patch2_values[1:4, :]

    official = qwen3_vl_processor_spec()
    official_patches = qwen3_vl_patchify(zeros(Float32, 3, 32, 32))
    @test size(official_patches) == (1_536, 4)
    @test all(iszero, official_patches)
    @test eltype(qwen3_vl_patchify(zeros(Float64, 3, 32, 32))) == Float64
    @test_throws ArgumentError qwen3_vl_patchify(zeros(Float32, 1, 32, 32))
    @test_throws ArgumentError qwen3_vl_patchify(zeros(Float32, 3, 16, 32))
    invalid_channels = _Ch43ReadCountingImage{Float32,3}((1, 32, 32), 0)
    invalid_channels_error = _ch43_processor_error() do
        qwen3_vl_patchify(invalid_channels)
    end
    @test invalid_channels_error isa ArgumentError
    @test sprint(showerror, invalid_channels_error) ==
        "ArgumentError: Qwen3-VL patchify expects 3 channels; got 1"
    @test invalid_channels.reads == 0
    nonfinite = zeros(Float32, 3, 32, 32)
    nonfinite[1] = NaN32
    @test_throws ArgumentError qwen3_vl_patchify(nonfinite; spec=official)
end

@testset "Qwen3-VL vision input validation" begin
    pixels = zeros(Float32, 1_536, 4)
    grid = reshape(Int[1, 2, 2], 3, 1)
    input = Qwen3VLVisionInput(pixels, grid)
    @test input.pixel_values === pixels
    @test input.grid_thw === grid
    narrow_grid = reshape(Int32[1, 2, 2], 3, 1)
    @test Qwen3VLVisionInput(pixels, narrow_grid).grid_thw === narrow_grid

    float_grid = Float64.(grid)
    float_grid_error = _ch43_processor_error() do
        Qwen3VLVisionInput(pixels, float_grid)
    end
    @test float_grid_error isa ArgumentError
    @test sprint(showerror, float_grid_error) ==
        "ArgumentError: Qwen3-VL grid_thw must contain non-Boolean integers"
    bool_grid_error = _ch43_processor_error() do
        Qwen3VLVisionInput(pixels, trues(3, 1))
    end
    @test bool_grid_error isa ArgumentError
    @test sprint(showerror, bool_grid_error) ==
        "ArgumentError: Qwen3-VL grid_thw must contain non-Boolean integers"
    @test_throws MethodError Qwen3VLVisionInput{
        typeof(pixels),
        typeof(float_grid),
    }(pixels, float_grid)

    too_large = big(typemax(Int)) + 1
    @test_throws ArgumentError Qwen3VLVisionInput(
        pixels,
        reshape(BigInt[1, 2, too_large], 3, 1),
    )
    @test_throws ArgumentError Qwen3VLVisionInput(
        pixels,
        reshape(Int[typemax(Int), 2, 2], 3, 1),
    )
    nearly_full = typemax(Int) ÷ 4
    @test_throws ArgumentError Qwen3VLVisionInput(
        zeros(Float32, 1_536, 8),
        Int[nearly_full 1; 2 2; 2 2],
    )

    overflowing_width_spec = Qwen3VLVisionSpec(
        1,
        4,
        8,
        1,
        Int(1) << 62,
        1,
        4,
        1,
        4,
        4,
        (0, 0, 0),
        "gelu",
    )
    @test_throws ArgumentError Qwen3VLVisionInput(
        zeros(Float32, 0, 1),
        reshape(Int[1, 1, 1], 3, 1);
        spec=overflowing_width_spec,
    )

    @test_throws DimensionMismatch Qwen3VLVisionInput(
        zeros(Float32, 1_535, 4),
        grid,
    )
    @test_throws DimensionMismatch Qwen3VLVisionInput(
        zeros(Float32, 1_536, 3),
        grid,
    )
    @test_throws DimensionMismatch Qwen3VLVisionInput(
        pixels,
        reshape(Int[1, 2], 2, 1),
    )
    @test_throws ArgumentError Qwen3VLVisionInput(
        zeros(Float64, 1_536, 4),
        grid,
    )
    @test_throws ArgumentError Qwen3VLVisionInput(
        pixels,
        reshape(Int[1, 1, 4], 3, 1),
    )
    @test_throws ArgumentError Qwen3VLVisionInput(
        pixels,
        reshape(Int[0, 2, 2], 3, 1),
    )
    @test_throws ArgumentError Qwen3VLVisionInput(
        pixels,
        Matrix{Int}(undef, 3, 0),
    )
    nan_pixels = copy(pixels)
    nan_pixels[1] = NaN32
    @test_throws ArgumentError Qwen3VLVisionInput(nan_pixels, grid)
    inf_pixels = copy(pixels)
    inf_pixels[end] = Inf32
    @test_throws ArgumentError Qwen3VLVisionInput(inf_pixels, grid)
end
