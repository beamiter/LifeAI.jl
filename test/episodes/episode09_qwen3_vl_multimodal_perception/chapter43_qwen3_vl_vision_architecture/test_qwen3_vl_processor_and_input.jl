using Test
using LifeAI: Qwen3VLProcessorSpec,
    Qwen3VLVisionInput,
    qwen3_vl_image_grid,
    qwen3_vl_image_token_count,
    qwen3_vl_patchify,
    qwen3_vl_processor_spec,
    qwen3_vl_smart_resize

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
