using SHA: sha256
using Test
using ColorTypes: Gray, RGBA
using FileIO: save
import LifeAI
using LifeAI: qwen3_vl_image_grid,
    qwen3_vl_image_token_count,
    qwen3_vl_process_image,
    qwen3_vl_smart_resize,
    qwen3_vl_processor_spec,
    Qwen3VLProcessorSpec

function _ch44_patterned_rgb(height::Int, width::Int)
    image = Array{UInt8}(undef, height, width, 3)
    @inbounds for row in 1:height, column in 1:width
        y = row - 1
        x = column - 1
        image[row, column, 1] = UInt8(mod(3 * x + 5 * y + 17, 256))
        image[row, column, 2] = UInt8(mod(11 * x + 7 * y + 29, 256))
        image[row, column, 3] = UInt8(mod(13 * x + 19 * y + 43, 256))
    end
    return image
end

function _ch44_processor_with_normalization(image_mean, image_std)
    frozen = qwen3_vl_processor_spec()
    return Qwen3VLProcessorSpec(
        frozen.preprocessor_config_sha256,
        frozen.processor_class,
        frozen.image_processor_type,
        frozen.min_pixels,
        frozen.max_pixels,
        frozen.patch_size,
        frozen.temporal_patch_size,
        frozen.merge_size,
        image_mean,
        image_std,
    )
end

function _ch44_f32_sha256(values)
    bytes = collect(reinterpret(UInt8, vec(Array{Float32}(values))))
    return bytes2hex(sha256(bytes))
end

function _ch44_resize_stress_rgb(height::Int, width::Int)
    image = Array{UInt8}(undef, height, width, 3)
    @inbounds for row in 1:height, column in 1:width
        y = row - 1
        x = column - 1
        image[row, column, 1] = UInt8(mod(
            37 * x + 73 * y + 19 * x * y + 17,
            256,
        ))
        image[row, column, 2] = UInt8(mod(
            151 * x + 43 * y + 7 * x * y + 29,
            256,
        ))
        image[row, column, 3] = UInt8(mod(
            97 * xor(x, y) + 13 * x + 31 * y + 43,
            256,
        ))
    end
    return image
end

function _ch44_chw_u8_sha256(values)
    bytes = vec(permutedims(values, (2, 1, 3)))
    return bytes2hex(sha256(bytes))
end

@testset "Chapter 44 — exact fast raw RGB processing" begin
    image = _ch44_patterned_rgb(256, 256)
    processed = qwen3_vl_process_image(image; layout=:hwc)

    @test processed.original_size == (256, 256)
    @test processed.resized_size == (256, 256)
    @test processed.grid_thw == reshape(Int[1, 16, 16], 3, 1)
    @test size(processed.pixel_values) == (1_536, 256)
    @test eltype(processed.pixel_values) == Float32
    @test _ch44_f32_sha256(processed.pixel_values) ==
        "6fd28a78fe1139197976baad0ab36044625016214eca93082c64cf355dbb2b3f"

    # HWC and CHW are input-layout choices only; the vision tensor is identical.
    chw = permutedims(image, (3, 1, 2))
    processed_chw = qwen3_vl_process_image(chw; layout=:chw)
    @test processed_chw.pixel_values == processed.pixel_values
    @test processed_chw.grid_thw == processed.grid_thw
end

@testset "Chapter 44 — explicit normalization contract" begin
    image = _ch44_patterned_rgb(256, 256)
    identity_normalization = _ch44_processor_with_normalization(
        (0.0f0, 0.0f0, 0.0f0),
        (1.0f0, 1.0f0, 1.0f0),
    )
    processed = qwen3_vl_process_image(image; spec=identity_normalization)
    @test processed.pixel_values[1, 1] == Float32(image[1, 1, 1]) / 255.0f0

    invalid_std = _ch44_processor_with_normalization(
        (0.5f0, 0.5f0, 0.5f0),
        (0.5f0, 0.0f0, 0.5f0),
    )
    @test_throws ArgumentError qwen3_vl_process_image(image; spec=invalid_std)
end

@testset "Chapter 44 — grayscale and alpha image decode" begin
    mktempdir() do directory
        gray_path = joinpath(directory, "gray.png")
        gray = Gray{Float32}[
            Gray{Float32}(0.0) Gray{Float32}(1.0)
            Gray{Float32}(1.0) Gray{Float32}(0.0)
        ]
        save(gray_path, gray)
        decoded_gray = LifeAI._qwen3_vl_decode_rgb(gray_path)
        @test size(decoded_gray) == (2, 2, 3)
        @test decoded_gray[:, :, 1] == decoded_gray[:, :, 2] ==
            decoded_gray[:, :, 3]
        @test decoded_gray[:, :, 1] == UInt8[0 255; 255 0]

        alpha_path = joinpath(directory, "alpha.png")
        alpha = RGBA{Float32}[
            RGBA{Float32}(1, 0, 0, 0) RGBA{Float32}(0, 1, 0, 0.5)
        ]
        save(alpha_path, alpha)
        decoded_alpha = LifeAI._qwen3_vl_decode_rgb(alpha_path)
        # The processor converts to RGB by dropping alpha rather than
        # compositing against an implicit background.
        @test decoded_alpha[1, 1, :] == UInt8[255, 0, 0]
        @test decoded_alpha[1, 2, :] == UInt8[0, 255, 0]
    end
end

@testset "Chapter 44 — quantized antialias resize stress oracles" begin
    # Dual-axis downsampling exercises widened antialias support and dynamic
    # Int16 coefficient precision (p=17 on both axes).
    downsampled = LifeAI._qwen3_vl_resize_bicubic(
        _ch44_resize_stress_rgb(33, 47),
        7,
        11,
    )
    @test _ch44_chw_u8_sha256(downsampled) ==
        "fdd0d1beee27890e5b727a23ac543206d4cb3cd73769a71475e7ebb040543175"

    # Mixed down/up scaling locks the official horizontal-then-vertical UInt8
    # pass order; a single Float32 intermediate produces a different digest.
    mixed = LifeAI._qwen3_vl_resize_bicubic(
        _ch44_resize_stress_rgb(19, 301),
        224,
        96,
    )
    @test _ch44_chw_u8_sha256(mixed) ==
        "3472ad13bf9ea5d938651a22bc07a528fe4d74153b693ee24a8d5152b4badde3"

    identity = _ch44_patterned_rgb(256, 256)
    @test LifeAI._qwen3_vl_resize_bicubic(identity, 256, 256) == identity
end

@testset "Chapter 44 — exact torchvision uint8 smart resize" begin
    image = _ch44_patterned_rgb(17, 31)
    processed = qwen3_vl_process_image(image)

    @test qwen3_vl_smart_resize(17, 31) == (192, 352)
    @test qwen3_vl_image_grid(17, 31) == (1, 12, 22)
    @test qwen3_vl_image_token_count(17, 31) == 66
    @test processed.original_size == (17, 31)
    @test processed.resized_size == (192, 352)
    @test processed.grid_thw == reshape(Int[1, 12, 22], 3, 1)
    @test size(processed.pixel_values) == (1_536, 264)
    @test _ch44_f32_sha256(processed.pixel_values) ==
        "3935253cabd26201ff0021b8b87ecf63308aa12cfd797e65427e6c2c87d08e72"

    # Keep both unchanged and min-pixel expansion geometry at this boundary.
    @test qwen3_vl_smart_resize(256, 256) == (256, 256)
    @test qwen3_vl_image_grid(256, 256) == (1, 16, 16)
    @test qwen3_vl_image_token_count(256, 256) == 64
end
