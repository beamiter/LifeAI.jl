using JSON3
using SHA: sha256
using FileIO
using ImageIO
using ColorTypes: Colorant, RGB, red, green, blue

"""
    Qwen3VLProcessorSpec

Frozen image-processor configuration for the official Qwen3-VL-2B checkpoint.
The same geometry and normalization fields drive both the raw-image path and
the already-normalized patchify boundary.
"""
struct Qwen3VLProcessorSpec
    preprocessor_config_sha256::String
    processor_class::String
    image_processor_type::String
    min_pixels::Int
    max_pixels::Int
    patch_size::Int
    temporal_patch_size::Int
    merge_size::Int
    image_mean::NTuple{3,Float32}
    image_std::NTuple{3,Float32}
end

const _QWEN3_VL_PROCESSOR_SPEC = Qwen3VLProcessorSpec(
    "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "Qwen3VLProcessor",
    "Qwen2VLImageProcessorFast",
    65_536,
    16_777_216,
    16,
    2,
    2,
    (0.5f0, 0.5f0, 0.5f0),
    (0.5f0, 0.5f0, 0.5f0),
)

"""Return the immutable Qwen3-VL image-processor boundary specification."""
qwen3_vl_processor_spec() = _QWEN3_VL_PROCESSOR_SPEC

_qwen3_vl_processor_sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function _qwen3_vl_exact_json_keys(value, expected, path, label)
    value isa JSON3.Object || throw(ArgumentError(
        "`$label` must be an object in $path",
    ))
    actual = Set(String(key) for key in keys(value))
    required = Set(String(key) for key in expected)
    actual == required || begin
        missing = sort!(collect(setdiff(required, actual)))
        extra = sort!(collect(setdiff(actual, required)))
        details = String[]
        isempty(missing) || push!(details, "missing: " * join(missing, ", "))
        isempty(extra) || push!(details, "unexpected: " * join(extra, ", "))
        throw(ArgumentError(
            "invalid fields in `$label` in $path ($(join(details, "; ")))",
        ))
    end
    return value
end

function _qwen3_vl_processor_int(value, label, path)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "`$label` must be an integer in $path",
    ))
    result = try
        Int(value)
    catch
        throw(ArgumentError("`$label` is outside the host integer range in $path"))
    end
    result > 0 || throw(ArgumentError("`$label` must be positive in $path"))
    return result
end

function _qwen3_vl_processor_string(value, label, path)
    value isa AbstractString || throw(ArgumentError(
        "`$label` must be a string in $path",
    ))
    return String(value)
end

function _qwen3_vl_processor_triplet(value, label, path)
    value isa JSON3.Array || throw(ArgumentError(
        "`$label` must be an array in $path",
    ))
    length(value) == 3 || throw(ArgumentError(
        "`$label` must contain exactly three channel values in $path",
    ))
    return ntuple(3) do index
        entry = value[index]
        entry isa Real && !(entry isa Bool) || throw(ArgumentError(
            "`$label[$index]` must be numeric in $path",
        ))
        result = Float32(entry)
        isfinite(result) || throw(ArgumentError(
            "`$label[$index]` must be finite in $path",
        ))
        result
    end
end

function _qwen3_vl_processor_matches_vision(
    processor::Qwen3VLProcessorSpec,
    vision::Qwen3VLVisionSpec,
)
    return vision.in_channels == length(processor.image_mean) &&
        vision.patch_size == processor.patch_size &&
        vision.temporal_patch_size == processor.temporal_patch_size &&
        vision.spatial_merge_size == processor.merge_size
end

"""
    load_hf_qwen3_vl_processor_config(path, vision_spec)

Strictly parse the frozen `preprocessor_config.json`, verify its SHA-256 and
all supported fields, and require its patch geometry to match
`Qwen3VLVisionSpec`. The returned value is the immutable
`Qwen3VLProcessorSpec`; this function does not construct a PNG/JPEG decoder,
resizer, or chat processor.
"""
function load_hf_qwen3_vl_processor_config(
    path::AbstractString,
    vision_spec::Qwen3VLVisionSpec,
)
    config = _json_object(path)
    _qwen3_vl_exact_json_keys(
        config,
        (
            "size",
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "image_mean",
            "image_std",
            "processor_class",
            "image_processor_type",
        ),
        path,
        "preprocessor config",
    )

    size_config = _json_required(config, "size", path)
    _qwen3_vl_exact_json_keys(
        size_config,
        ("shortest_edge", "longest_edge"),
        path,
        "size",
    )

    parsed = Qwen3VLProcessorSpec(
        qwen3_vl_processor_spec().preprocessor_config_sha256,
        _qwen3_vl_processor_string(
            _json_required(config, "processor_class", path),
            "processor_class",
            path,
        ),
        _qwen3_vl_processor_string(
            _json_required(config, "image_processor_type", path),
            "image_processor_type",
            path,
        ),
        _qwen3_vl_processor_int(
            _json_required(size_config, "shortest_edge", path),
            "size.shortest_edge",
            path,
        ),
        _qwen3_vl_processor_int(
            _json_required(size_config, "longest_edge", path),
            "size.longest_edge",
            path,
        ),
        _qwen3_vl_processor_int(
            _json_required(config, "patch_size", path),
            "patch_size",
            path,
        ),
        _qwen3_vl_processor_int(
            _json_required(config, "temporal_patch_size", path),
            "temporal_patch_size",
            path,
        ),
        _qwen3_vl_processor_int(
            _json_required(config, "merge_size", path),
            "merge_size",
            path,
        ),
        _qwen3_vl_processor_triplet(
            _json_required(config, "image_mean", path),
            "image_mean",
            path,
        ),
        _qwen3_vl_processor_triplet(
            _json_required(config, "image_std", path),
            "image_std",
            path,
        ),
    )

    frozen = qwen3_vl_processor_spec()
    parsed.processor_class == frozen.processor_class || throw(ArgumentError(
        "unsupported Qwen3-VL processor_class $(repr(parsed.processor_class)); " *
        "expected $(repr(frozen.processor_class))",
    ))
    parsed.image_processor_type == frozen.image_processor_type || throw(ArgumentError(
        "unsupported Qwen3-VL image_processor_type " *
        "$(repr(parsed.image_processor_type)); expected " *
        "$(repr(frozen.image_processor_type))",
    ))
    parsed.min_pixels == frozen.min_pixels || throw(ArgumentError(
        "Qwen3-VL shortest_edge must be $(frozen.min_pixels); got $(parsed.min_pixels)",
    ))
    parsed.max_pixels == frozen.max_pixels || throw(ArgumentError(
        "Qwen3-VL longest_edge must be $(frozen.max_pixels); got $(parsed.max_pixels)",
    ))
    parsed.patch_size == frozen.patch_size || throw(ArgumentError(
        "Qwen3-VL patch_size must be $(frozen.patch_size); got $(parsed.patch_size)",
    ))
    parsed.temporal_patch_size == frozen.temporal_patch_size || throw(ArgumentError(
        "Qwen3-VL temporal_patch_size must be $(frozen.temporal_patch_size); " *
        "got $(parsed.temporal_patch_size)",
    ))
    parsed.merge_size == frozen.merge_size || throw(ArgumentError(
        "Qwen3-VL merge_size must be $(frozen.merge_size); got $(parsed.merge_size)",
    ))
    parsed.image_mean == frozen.image_mean || throw(ArgumentError(
        "Qwen3-VL image_mean must be $(frozen.image_mean); got $(parsed.image_mean)",
    ))
    parsed.image_std == frozen.image_std || throw(ArgumentError(
        "Qwen3-VL image_std must be $(frozen.image_std); got $(parsed.image_std)",
    ))
    _qwen3_vl_processor_matches_vision(parsed, vision_spec) || throw(ArgumentError(
        "Qwen3-VL processor patch geometry does not match the vision architecture",
    ))

    actual_sha256 = _qwen3_vl_processor_sha256_file(path)
    actual_sha256 == frozen.preprocessor_config_sha256 || throw(ArgumentError(
        "Qwen3-VL preprocessor config checksum mismatch: expected " *
        "$(frozen.preprocessor_config_sha256), computed $actual_sha256",
    ))
    return frozen
end

load_hf_qwen3_vl_processor_config(
    path::AbstractString;
    vision_spec::Qwen3VLVisionSpec=qwen3_vl_checkpoint_spec().vision,
) = load_hf_qwen3_vl_processor_config(path, vision_spec)

function _qwen3_vl_positive_host_int(value, label)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "$label must be an integer",
    ))
    result = try
        Int(value)
    catch
        throw(ArgumentError("$label is outside the host integer range"))
    end
    result > 0 || throw(ArgumentError("$label must be positive"))
    return result
end

function _qwen3_vl_checked_mul(left::Int, right::Int, label)
    return try
        Base.Checked.checked_mul(left, right)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("$label exceeds the host integer range"))
    end
end

"""
    qwen3_vl_smart_resize(height, width; spec=qwen3_vl_processor_spec())

Return the `(height, width)` selected by the official Qwen3-VL image resize
policy. Rounding to the patch/merge factor uses round-to-nearest with ties to
even, matching Python's `round`; over-budget dimensions use `floor` and
under-budget dimensions use `ceil`.
"""
function qwen3_vl_smart_resize(
    height,
    width;
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
    factor=nothing,
    min_pixels=nothing,
    max_pixels=nothing,
)
    height_int = _qwen3_vl_positive_host_int(height, "height")
    width_int = _qwen3_vl_positive_host_int(width, "width")
    factor_int = _qwen3_vl_positive_host_int(
        factor === nothing ?
            _qwen3_vl_checked_mul(spec.patch_size, spec.merge_size, "resize factor") :
            factor,
        "factor",
    )
    min_pixels_int = _qwen3_vl_positive_host_int(
        min_pixels === nothing ? spec.min_pixels : min_pixels,
        "min_pixels",
    )
    max_pixels_int = _qwen3_vl_positive_host_int(
        max_pixels === nothing ? spec.max_pixels : max_pixels,
        "max_pixels",
    )
    min_pixels_int <= max_pixels_int || throw(ArgumentError(
        "min_pixels must not exceed max_pixels",
    ))

    longer = max(height_int, width_int)
    shorter = min(height_int, width_int)
    widemul(shorter, 200) >= longer || throw(ArgumentError(
        "absolute aspect ratio must not exceed 200; got " *
        "$(Float64(longer) / Float64(shorter))",
    ))

    # `RoundNearest` is Julia's ties-to-even rounding mode, matching Python's
    # integer `round` used by the reference processor.
    height_quanta = round(
        Int,
        Float64(height_int) / Float64(factor_int),
        RoundNearest,
    )
    width_quanta = round(
        Int,
        Float64(width_int) / Float64(factor_int),
        RoundNearest,
    )
    resized_height = _qwen3_vl_checked_mul(
        height_quanta,
        factor_int,
        "resized height",
    )
    resized_width = _qwen3_vl_checked_mul(
        width_quanta,
        factor_int,
        "resized width",
    )

    rounded_pixels = widemul(resized_height, resized_width)
    original_pixels = Float64(height_int) * Float64(width_int)
    if rounded_pixels > max_pixels_int
        beta = sqrt(original_pixels / Float64(max_pixels_int))
        height_quanta = floor(
            Int,
            Float64(height_int) / beta / Float64(factor_int),
        )
        width_quanta = floor(
            Int,
            Float64(width_int) / beta / Float64(factor_int),
        )
        resized_height = max(
            factor_int,
            _qwen3_vl_checked_mul(height_quanta, factor_int, "resized height"),
        )
        resized_width = max(
            factor_int,
            _qwen3_vl_checked_mul(width_quanta, factor_int, "resized width"),
        )
    elseif rounded_pixels < min_pixels_int
        beta = sqrt(Float64(min_pixels_int) / original_pixels)
        height_quanta = ceil(
            Int,
            Float64(height_int) * beta / Float64(factor_int),
        )
        width_quanta = ceil(
            Int,
            Float64(width_int) * beta / Float64(factor_int),
        )
        resized_height = _qwen3_vl_checked_mul(
            height_quanta,
            factor_int,
            "resized height",
        )
        resized_width = _qwen3_vl_checked_mul(
            width_quanta,
            factor_int,
            "resized width",
        )
    end

    (resized_height > 0 && resized_width > 0) || throw(ArgumentError(
        "resize policy produced a non-positive dimension",
    ))
    return resized_height, resized_width
end

qwen3_vl_smart_resize(
    height,
    width,
    spec::Qwen3VLProcessorSpec,
) = qwen3_vl_smart_resize(height, width; spec)

"""
    qwen3_vl_image_grid(height, width; spec=qwen3_vl_processor_spec())

Apply the frozen smart-resize policy and return the image patch grid as
`(grid_t, grid_h, grid_w)`. A single image is temporally broadcast to one
temporal patch, so `grid_t == 1`.
"""
function qwen3_vl_image_grid(
    height,
    width;
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
)
    resized_height, resized_width = qwen3_vl_smart_resize(
        height,
        width;
        spec,
    )
    resized_height % spec.patch_size == 0 || throw(ArgumentError(
        "resized height must be divisible by patch_size",
    ))
    resized_width % spec.patch_size == 0 || throw(ArgumentError(
        "resized width must be divisible by patch_size",
    ))
    grid_h = resized_height ÷ spec.patch_size
    grid_w = resized_width ÷ spec.patch_size
    (grid_h % spec.merge_size == 0 && grid_w % spec.merge_size == 0) ||
        throw(ArgumentError(
            "image patch grid must be divisible by merge_size",
        ))
    return (1, grid_h, grid_w)
end

qwen3_vl_image_grid(
    height,
    width,
    spec::Qwen3VLProcessorSpec,
) = qwen3_vl_image_grid(height, width; spec)

qwen3_vl_image_grid(
    dimensions::Tuple{Any,Any};
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
) = qwen3_vl_image_grid(dimensions[1], dimensions[2]; spec)

function _qwen3_vl_image_grid_tuple(grid)
    (grid isa Tuple || grid isa AbstractVector) || throw(ArgumentError(
        "image grid must be a three-element tuple or vector",
    ))
    length(grid) == 3 || throw(ArgumentError(
        "image grid must contain exactly (grid_t, grid_h, grid_w)",
    ))
    return ntuple(index ->
        _qwen3_vl_positive_host_int(grid[index], "image grid[$index]"), 3)
end

"""
    qwen3_vl_image_token_count(grid; spec=qwen3_vl_processor_spec())

Return the number of spatially merged visual tokens represented by a strict
single-image `(grid_t, grid_h, grid_w)` patch grid.
"""
function qwen3_vl_image_token_count(
    grid;
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
)
    grid_t, grid_h, grid_w = _qwen3_vl_image_grid_tuple(grid)
    grid_t == 1 || throw(ArgumentError(
        "an image grid must have grid_t == 1; got $grid_t",
    ))
    (grid_h % spec.merge_size == 0 && grid_w % spec.merge_size == 0) ||
        throw(ArgumentError(
            "image grid height and width must be divisible by merge_size",
        ))
    merged_h = grid_h ÷ spec.merge_size
    merged_w = grid_w ÷ spec.merge_size
    return _qwen3_vl_checked_mul(merged_h, merged_w, "image token count")
end

qwen3_vl_image_token_count(
    grid,
    spec::Qwen3VLProcessorSpec,
) = qwen3_vl_image_token_count(grid; spec)

function qwen3_vl_image_token_count(
    height,
    width;
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
)
    return qwen3_vl_image_token_count(
        qwen3_vl_image_grid(height, width; spec);
        spec,
    )
end

qwen3_vl_image_token_count(
    height,
    width,
    spec::Qwen3VLProcessorSpec,
) = qwen3_vl_image_token_count(height, width; spec)

"""
    qwen3_vl_patchify(image; spec=qwen3_vl_processor_spec())

Patchify an already resized and normalized, one-based `C×H×W` floating-point
image. The result has LifeAI layout
`(C * temporal_patch_size * patch_size^2, total_patches)`.

Columns follow the official Qwen processor's merge-group order: outer merge
group height/width first, then the within-group height/width coordinates.
Feature rows follow channel, temporal, patch-height, patch-width order. The
single image is broadcast across the temporal patch exactly as in the
reference image processor.
"""
function qwen3_vl_patchify(
    image::AbstractArray;
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
)
    ndims(image) == 3 || throw(ArgumentError(
        "Qwen3-VL patchify expects a C×H×W image; got $(ndims(image)) dimensions",
    ))
    all(dimension -> axes(image, dimension) == Base.OneTo(size(image, dimension)), 1:3) ||
        throw(ArgumentError("Qwen3-VL patchify requires one-based image axes"))
    eltype(image) <: AbstractFloat || throw(ArgumentError(
        "Qwen3-VL patchify expects an already normalized floating-point image",
    ))
    all(isfinite, image) || throw(ArgumentError(
        "Qwen3-VL patchify input must contain only finite values",
    ))

    channels, height, width = size(image)
    channels == length(spec.image_mean) || throw(ArgumentError(
        "Qwen3-VL patchify expects $(length(spec.image_mean)) channels; got $channels",
    ))
    factor = _qwen3_vl_checked_mul(
        spec.patch_size,
        spec.merge_size,
        "patch/merge factor",
    )
    (height > 0 && width > 0) || throw(ArgumentError(
        "Qwen3-VL patchify image dimensions must be positive",
    ))
    height % spec.patch_size == 0 || throw(ArgumentError(
        "image height must be divisible by patch_size=$(spec.patch_size)",
    ))
    width % spec.patch_size == 0 || throw(ArgumentError(
        "image width must be divisible by patch_size=$(spec.patch_size)",
    ))
    (height % factor == 0 && width % factor == 0) || throw(ArgumentError(
        "image height and width must be divisible by " *
        "patch_size * merge_size = $factor",
    ))

    grid_h = height ÷ spec.patch_size
    grid_w = width ÷ spec.patch_size
    group_grid_h = grid_h ÷ spec.merge_size
    group_grid_w = grid_w ÷ spec.merge_size
    total_patches = _qwen3_vl_checked_mul(grid_h, grid_w, "total patch count")
    patch_area = _qwen3_vl_checked_mul(
        spec.patch_size,
        spec.patch_size,
        "patch area",
    )
    temporal_patch_area = _qwen3_vl_checked_mul(
        spec.temporal_patch_size,
        patch_area,
        "temporal patch area",
    )
    feature_width = _qwen3_vl_checked_mul(
        channels,
        temporal_patch_area,
        "flattened patch width",
    )
    flattened = Matrix{eltype(image)}(undef, feature_width, total_patches)

    patch_index = 0
    @inbounds for group_h in 0:(group_grid_h - 1)
        for group_w in 0:(group_grid_w - 1)
            for merge_h in 0:(spec.merge_size - 1)
                for merge_w in 0:(spec.merge_size - 1)
                    patch_index += 1
                    patch_h = group_h * spec.merge_size + merge_h
                    patch_w = group_w * spec.merge_size + merge_w
                    feature_index = 0
                    for channel in 1:channels
                        for _ in 1:spec.temporal_patch_size
                            for inner_h in 1:spec.patch_size
                                source_h = patch_h * spec.patch_size + inner_h
                                for inner_w in 1:spec.patch_size
                                    source_w = patch_w * spec.patch_size + inner_w
                                    feature_index += 1
                                    flattened[feature_index, patch_index] =
                                        image[channel, source_h, source_w]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    patch_index == total_patches || error(
        "internal Qwen3-VL patch ordering error: emitted $patch_index of $total_patches patches",
    )
    return flattened
end

qwen3_vl_patchify(
    image::AbstractArray,
    spec::Qwen3VLProcessorSpec,
) = qwen3_vl_patchify(image; spec)

"""Decoded RGB image plus the exact tensors consumed by the vision tower."""
struct Qwen3VLProcessedImage{P,G}
    pixel_values::P
    grid_thw::G
    original_size::Tuple{Int,Int}
    resized_size::Tuple{Int,Int}
end

function _qwen3_vl_rgb_hwc(image::AbstractArray{UInt8,3}, layout::Symbol)
    layout in (:hwc, :chw) || throw(ArgumentError(
        "Qwen3-VL image layout must be :hwc or :chw",
    ))
    if layout === :hwc
        size(image, 3) == 3 || throw(DimensionMismatch(
            "Qwen3-VL HWC image must contain exactly three RGB channels",
        ))
        return Array(image)
    end
    size(image, 1) == 3 || throw(DimensionMismatch(
        "Qwen3-VL CHW image must contain exactly three RGB channels",
    ))
    return permutedims(image, (2, 3, 1))
end

function _qwen3_vl_decoded_rgb_channels(pixel)
    if pixel isa Colorant
        converted = convert(RGB{Float32}, pixel)
        return (red(converted), green(converted), blue(converted))
    elseif pixel isa Bool
        value = pixel ? 1.0f0 : 0.0f0
        return (value, value, value)
    elseif pixel isa Unsigned
        value = Float32(Float64(pixel) / Float64(typemax(typeof(pixel))))
        return (value, value, value)
    elseif pixel isa Real && !(pixel isa Integer)
        value = Float32(pixel)
        return (value, value, value)
    end
    throw(ArgumentError(
        "decoded image pixels must be colorants or normalized grayscale values",
    ))
end

function _qwen3_vl_decode_rgb(path::AbstractString)
    isfile(path) || throw(ArgumentError("image file does not exist: $path"))
    decoded = FileIO.load(path)
    ndims(decoded) == 2 || throw(ArgumentError(
        "decoded Qwen3-VL image must be a two-dimensional pixel array",
    ))
    height, width = size(decoded)
    (height > 0 && width > 0) || throw(ArgumentError(
        "decoded Qwen3-VL image must not be empty",
    ))
    output = Array{UInt8}(undef, height, width, 3)
    @inbounds for row in 1:height, column in 1:width
        # ColorTypes conversion duplicates gray channels and deliberately drops
        # alpha, matching PIL's RGB conversion used by the frozen processor.
        channels = _qwen3_vl_decoded_rgb_channels(decoded[row, column])
        for channel in 1:3
            value = Float32(channels[channel])
            isfinite(value) && 0.0f0 <= value <= 1.0f0 || throw(ArgumentError(
                "decoded image channels must be finite values in [0, 1]",
            ))
            output[row, column, channel] = UInt8(clamp(
                round(Int, 255.0f0 * value, RoundNearest),
                0,
                255,
            ))
        end
    end
    return output
end

@inline function _qwen3_vl_tv_aa_cubic(value::Float64)
    distance = abs(value)
    if distance < 1.0
        return ((1.5 * distance - 2.5) * distance) * distance + 1.0
    elseif distance < 2.0
        return ((-0.5 * distance + 2.5) * distance - 4.0) * distance + 2.0
    end
    return 0.0
end

# Reproduce ATen 2.7.1's PIL-SIMD-derived antialiased uint8 interpolation
# table. Source indices remain zero-based until the scalar sampling loop.
function _qwen3_vl_tv_uint8_table(input_size::Int, output_size::Int)
    scale = Float64(input_size) / Float64(output_size)
    filter_scale = max(scale, 1.0)
    support = 2.0 * filter_scale
    inverse_filter_scale = inv(filter_scale)
    maximum_interpolation_size = 2 * ceil(Int, support) + 1
    float_rows = Vector{Tuple{Int,Vector{Float64}}}(undef, output_size)
    maximum_weight = 0.0

    for output_index in 0:(output_size - 1)
        center = scale * (Float64(output_index) + 0.5)
        first_source = max(trunc(Int, center - support + 0.5), 0)
        last_source = min(trunc(Int, center + support + 0.5), input_size)
        count = min(last_source - first_source, maximum_interpolation_size)
        weights = Vector{Float64}(undef, count)
        total = 0.0
        @inbounds for offset in 0:(count - 1)
            weight = _qwen3_vl_tv_aa_cubic(
                (Float64(first_source + offset) - center + 0.5) *
                inverse_filter_scale,
            )
            weights[offset + 1] = weight
            total += weight
        end
        @inbounds for index in eachindex(weights)
            weights[index] /= total
            maximum_weight = max(maximum_weight, weights[index])
        end
        float_rows[output_index + 1] = (first_source, weights)
    end

    precision = 0
    while precision < 22 && trunc(
        Int,
        maximum_weight * Float64(1 << (precision + 1)) + 0.5,
    ) <= typemax(Int16)
        precision += 1
    end
    multiplier = Float64(1 << precision)
    rows = Vector{Tuple{Int,Vector{Int16}}}(undef, output_size)
    for index in eachindex(float_rows)
        first_source, weights = float_rows[index]
        quantized = Vector{Int16}(undef, length(weights))
        @inbounds for weight_index in eachindex(weights)
            scaled = weights[weight_index] * multiplier
            quantized[weight_index] = Int16(trunc(
                Int,
                scaled + (scaled < 0.0 ? -0.5 : 0.5),
            ))
        end
        rows[index] = (first_source, quantized)
    end
    return rows, precision
end

function _qwen3_vl_tv_uint8_axis(
    image::Array{UInt8,3},
    output_size::Int,
    dimension::Int,
)
    dimension in (1, 2) || throw(ArgumentError(
        "Qwen3-VL resize dimension must be one or two",
    ))
    input_size = size(image, dimension)
    input_size == output_size && return copy(image)
    rows, precision = _qwen3_vl_tv_uint8_table(input_size, output_size)
    rounding_bias = Int64(1) << (precision - 1)
    input_height, input_width, channels = size(image)

    if dimension == 2
        output = Array{UInt8}(undef, input_height, output_size, channels)
        @inbounds for channel in 1:channels, row in 1:input_height,
                      output_column in 1:output_size
            first_source, weights = rows[output_column]
            accumulator = rounding_bias
            for index in eachindex(weights)
                accumulator += Int64(weights[index]) *
                    Int64(image[row, first_source + index, channel])
            end
            output[row, output_column, channel] = UInt8(clamp(
                accumulator >> precision,
                0,
                255,
            ))
        end
        return output
    end

    output = Array{UInt8}(undef, output_size, input_width, channels)
    @inbounds for channel in 1:channels, output_row in 1:output_size,
                  column in 1:input_width
        first_source, weights = rows[output_row]
        accumulator = rounding_bias
        for index in eachindex(weights)
            accumulator += Int64(weights[index]) *
                Int64(image[first_source + index, column, channel])
        end
        output[output_row, column, channel] = UInt8(clamp(
            accumulator >> precision,
            0,
            255,
        ))
    end
    return output
end

# Torchvision 0.22.1 performs a quantized UInt8 horizontal pass followed by a
# quantized UInt8 vertical pass. Keeping that order is required for exact fast
# image-processor parity.
function _qwen3_vl_resize_bicubic(
    image::Array{UInt8,3},
    output_height::Int,
    output_width::Int,
)
    input_height, input_width, _ = size(image)
    (input_height, input_width) == (output_height, output_width) && return copy(image)
    horizontal = input_width == output_width ? image :
        _qwen3_vl_tv_uint8_axis(image, output_width, 2)
    return input_height == output_height ? horizontal :
        _qwen3_vl_tv_uint8_axis(horizontal, output_height, 1)
end

"""
    qwen3_vl_process_image(image; layout=:hwc, spec=...)
    qwen3_vl_process_image(path; spec=...)

Decode (for the path method), convert to RGB, apply the Qwen3-VL fast
processor's fused `(pixel - 255mean) / (255std)` normalization, and patchify
a single static image. The returned `pixel_values` uses LifeAI's
`(1536, patches)` convention and `grid_thw` is a `3×1` host matrix.

Smart resize reproduces the quantized two-pass CPU uint8 bicubic path used by
the frozen Torch 2.7.1 / torchvision 0.22.1 reference processor.
"""
function qwen3_vl_process_image(
    image::AbstractArray{UInt8,3};
    layout::Symbol=:hwc,
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
)
    @inbounds for channel in 1:3
        isfinite(spec.image_mean[channel]) || throw(ArgumentError(
            "Qwen3-VL image_mean entries must be finite",
        ))
        isfinite(spec.image_std[channel]) && spec.image_std[channel] > 0.0f0 ||
            throw(ArgumentError(
                "Qwen3-VL image_std entries must be finite and positive",
            ))
    end
    rgb = _qwen3_vl_rgb_hwc(image, layout)
    original_height, original_width, _ = size(rgb)
    resized_height, resized_width = qwen3_vl_smart_resize(
        original_height,
        original_width;
        spec,
    )
    resized = _qwen3_vl_resize_bicubic(rgb, resized_height, resized_width)
    normalized = Array{Float32}(undef, 3, resized_height, resized_width)
    @inbounds for channel in 1:3
        # Transformers fast processing fuses rescale and normalize by scaling
        # mean/std to the uint8 domain before one Float32 normalization.
        mean_u8 = 255.0f0 * spec.image_mean[channel]
        std_u8 = 255.0f0 * spec.image_std[channel]
        for row in 1:resized_height, column in 1:resized_width
            normalized[channel, row, column] =
                (Float32(resized[row, column, channel]) - mean_u8) / std_u8
        end
    end
    grid = qwen3_vl_image_grid(original_height, original_width; spec)
    grid_thw = reshape(Int[grid...], 3, 1)
    pixels = qwen3_vl_patchify(normalized; spec)
    return Qwen3VLProcessedImage(
        pixels,
        grid_thw,
        (original_height, original_width),
        (resized_height, resized_width),
    )
end

function qwen3_vl_process_image(
    path::AbstractString;
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
)
    return qwen3_vl_process_image(_qwen3_vl_decode_rgb(path); spec)
end

function _qwen3_vl_content_value(item, name::Symbol; default=nothing)
    return _hf_message_value(item, name; default)
end

function _qwen3_vl_content_has(item, name::Symbol)
    if item isa NamedTuple
        return hasproperty(item, name)
    elseif item isa AbstractDict
        return haskey(item, name) || haskey(item, String(name))
    end
    throw(ArgumentError(
        "Qwen3-VL content items must be NamedTuples or dictionaries",
    ))
end

function _qwen3_vl_content_list(raw, label::AbstractString)
    raw isa AbstractString && return nothing
    raw === nothing && return Any[]
    raw isa AbstractVector || raw isa Tuple || throw(ArgumentError(
        "$label content must be a string or a list of content objects",
    ))
    return collect(Any, raw)
end

function _qwen3_vl_render_content!(
    output::IO,
    raw_content,
    image_count::Base.RefValue{Int},
    video_count::Base.RefValue{Int};
    add_vision_id::Bool,
    assistant::Bool=false,
    system::Bool=false,
)
    if raw_content isa AbstractString
        print(output, raw_content)
        return
    end
    items = _qwen3_vl_content_list(raw_content, "Qwen3-VL chat")
    for item in items
        text = _qwen3_vl_content_value(item, :text; default=nothing)
        has_text = _qwen3_vl_content_has(item, :text)
        if assistant || system
            has_text || continue
            text isa AbstractString || throw(ArgumentError(
                "Qwen3-VL text content must be a non-null string",
            ))
            print(output, text)
            continue
        end
        content_type = _qwen3_vl_content_value(item, :type; default=nothing)
        has_image = content_type == "image" ||
            _qwen3_vl_content_has(item, :image) ||
            _qwen3_vl_content_has(item, :image_url)
        has_video = content_type == "video" ||
            _qwen3_vl_content_has(item, :video)
        if has_image
            image_count[] += 1
            add_vision_id && print(output, "Picture ", image_count[], ": ")
            print(output, "<|vision_start|><|image_pad|><|vision_end|>")
        elseif has_video
            video_count[] += 1
            add_vision_id && print(output, "Video ", video_count[], ": ")
            print(output, "<|vision_start|><|video_pad|><|vision_end|>")
        elseif has_text
            text isa AbstractString || throw(ArgumentError(
                "Qwen3-VL text content must be a non-null string",
            ))
            print(output, text)
        end
    end
end

function _qwen3_vl_content_truthy(raw_content)
    raw_content === nothing && return false
    raw_content isa AbstractString && return !isempty(raw_content)
    return !isempty(raw_content)
end

"""
    apply_qwen3_vl_chat_template(messages; tools=nothing,
                                 add_generation_prompt=true,
                                 add_vision_id=false)
    apply_qwen3_vl_chat_template(tokenizer, messages; kwargs...)

Render the frozen Qwen3-VL content-list chat template byte-for-byte. Image
items initially emit one image placeholder; expand it after image geometry is
known with [`qwen3_vl_expand_image_placeholders`](@ref). The tokenizer overload
additionally verifies the frozen tokenizer profile and chat-template identity.
"""
function _qwen3_vl_render_chat_template(
    messages;
    tools=nothing,
    add_generation_prompt::Bool=true,
    add_vision_id::Bool=false,
)
    message_list = collect(Any, messages)
    isempty(message_list) && throw(ArgumentError("chat messages must not be empty"))
    roles = String[]
    for message in message_list
        role = _hf_message_value(message, :role; required=true)
        role isa AbstractString || throw(ArgumentError("chat role must be a string"))
        String(role) in ("system", "user", "assistant", "tool") ||
            throw(ArgumentError("unsupported Qwen3-VL chat role $(repr(role))"))
        push!(roles, String(role))
    end
    (tools isa NamedTuple || tools isa AbstractDict) && throw(ArgumentError(
        "tools must be a list of tool declarations; wrap one tool in a vector",
    ))
    tool_list = tools === nothing ? Any[] : collect(Any, tools)
    output = IOBuffer()
    leading_system = roles[1] == "system"
    if !isempty(tool_list)
        print(output, "<|im_start|>system\n")
        if leading_system
            raw = _hf_message_value(message_list[1], :content; default=nothing)
            _qwen3_vl_render_content!(
                output, raw, Ref(0), Ref(0);
                add_vision_id, system=true,
            )
            print(output, "\n\n")
        end
        print(output, _QWEN3_TOOL_HEADER)
        for tool in tool_list
            print(output, "\n")
            _python_json(output, tool)
        end
        print(output, _QWEN3_TOOL_FOOTER)
    elseif leading_system
        print(output, "<|im_start|>system\n")
        raw = _hf_message_value(message_list[1], :content; default=nothing)
        _qwen3_vl_render_content!(
            output, raw, Ref(0), Ref(0);
            add_vision_id, system=true,
        )
        print(output, "<|im_end|>\n")
    end

    image_count = Ref(0)
    video_count = Ref(0)
    for index in eachindex(message_list)
        message = message_list[index]
        role = roles[index]
        raw = _hf_message_value(message, :content; default=nothing)
        if role == "user"
            print(output, "<|im_start|>user\n")
            _qwen3_vl_render_content!(
                output, raw, image_count, video_count; add_vision_id,
            )
            print(output, "<|im_end|>\n")
        elseif role == "assistant"
            print(output, "<|im_start|>assistant\n")
            _qwen3_vl_render_content!(
                output, raw, image_count, video_count;
                add_vision_id, assistant=true,
            )
            raw_calls = _hf_message_value(message, :tool_calls; default=nothing)
            calls = raw_calls === nothing ? Any[] : collect(Any, raw_calls)
            for (position, call) in enumerate(calls)
                (position > 1 || _qwen3_vl_content_truthy(raw)) && print(output, "\n")
                name, arguments = _qwen3_tool_call_fields(call)
                print(output, "<tool_call>\n{\"name\": \"", name, "\", \"arguments\": ")
                arguments isa AbstractString ? print(output, arguments) :
                    _python_json(output, arguments)
                print(output, "}\n</tool_call>")
            end
            print(output, "<|im_end|>\n")
        elseif role == "tool"
            (index == 1 || roles[index - 1] != "tool") &&
                print(output, "<|im_start|>user")
            print(output, "\n<tool_response>\n")
            _qwen3_vl_render_content!(
                output, raw, image_count, video_count; add_vision_id,
            )
            print(output, "\n</tool_response>")
            (index == lastindex(message_list) || roles[index + 1] != "tool") &&
                print(output, "<|im_end|>\n")
        end
    end
    add_generation_prompt && print(output, "<|im_start|>assistant\n")
    return String(take!(output))
end

apply_qwen3_vl_chat_template(messages; kwargs...) =
    _qwen3_vl_render_chat_template(messages; kwargs...)

function apply_qwen3_vl_chat_template(
    tokenizer::HFQwen3Tokenizer,
    messages;
    kwargs...,
)
    tokenizer.profile === :qwen3_vl_generation || throw(ArgumentError(
        "Qwen3-VL chat rendering requires a Qwen3-VL tokenizer",
    ))
    _sha256_hex(tokenizer.chat_template) == _QWEN3_VL_CHAT_TEMPLATE_SHA256 ||
        throw(ArgumentError("Qwen3-VL chat template revision is not frozen"))
    return _qwen3_vl_render_chat_template(messages; kwargs...)
end

function _qwen3_vl_grid_columns(grids)
    if grids isa AbstractMatrix
        size(grids, 1) == 3 || throw(DimensionMismatch(
            "Qwen3-VL image grids must have shape (3, image_count)",
        ))
        return [Tuple(Int.(grids[:, index])) for index in axes(grids, 2)]
    end
    return [_qwen3_vl_image_grid_tuple(grid) for grid in grids]
end

"""Expand each image placeholder exactly once using its corresponding grid."""
function qwen3_vl_expand_image_placeholders(
    text::AbstractString,
    grids;
    spec::Qwen3VLProcessorSpec=qwen3_vl_processor_spec(),
)
    image_token = "<|image_pad|>"
    pieces = split(String(text), image_token; keepempty=true)
    grid_values = _qwen3_vl_grid_columns(grids)
    length(pieces) == length(grid_values) + 1 || throw(ArgumentError(
        "Qwen3-VL image placeholder count $(length(pieces) - 1) does not " *
        "match image grid count $(length(grid_values))",
    ))
    output = IOBuffer()
    for index in eachindex(grid_values)
        print(output, pieces[index])
        print(output, repeat(
            image_token,
            qwen3_vl_image_token_count(grid_values[index]; spec),
        ))
    end
    print(output, pieces[end])
    return String(take!(output))
end

"""LifeAI-layout multimodal position ids, decode delta, and prompt masks."""
struct Qwen3VLRopeLayout{P,D,M,A}
    position_ids::P
    rope_deltas::D
    visual_mask::M
    attention_mask::A
end

Qwen3VLRopeLayout(position_ids, rope_deltas, visual_mask) = Qwen3VLRopeLayout(
    position_ids,
    rope_deltas,
    visual_mask,
    trues(size(visual_mask)),
)

function _qwen3_vl_token_matrix(input_ids)
    input_ids isa AbstractVector && return reshape(Int.(input_ids), :, 1)
    input_ids isa AbstractMatrix || throw(ArgumentError(
        "Qwen3-VL input_ids must be a vector or (sequence, batch) matrix",
    ))
    return Int.(input_ids)
end

function _qwen3_vl_attention_matrix(attention_mask, dimensions)
    attention_mask === nothing && return trues(dimensions)
    normalized = attention_mask isa AbstractVector ?
        reshape(attention_mask, :, 1) : attention_mask
    normalized isa AbstractMatrix || throw(ArgumentError(
        "Qwen3-VL attention_mask must be a vector or matrix",
    ))
    size(normalized) == dimensions || throw(DimensionMismatch(
        "Qwen3-VL attention_mask must match input_ids",
    ))
    all(value -> value in (false, true, 0, 1), normalized) ||
        throw(ArgumentError("Qwen3-VL attention_mask values must be boolean"))
    return Bool.(normalized)
end

function _qwen3_vl_append_text_positions!(
    columns::Vector{NTuple{3,Int}},
    length::Int,
    start::Int,
)
    for offset in 0:(length - 1)
        value = start + offset
        push!(columns, (value, value, value))
    end
    return columns
end

"""
    qwen3_vl_rope_layout(input_ids, image_grid_thw; attention_mask=nothing)

Construct official Qwen3-VL temporal/height/width position ids for public
one-based LifeAI token ids. Output axes are `(3, sequence, 1)`; values remain
zero-based because they are RoPE coordinates, not token ids. Multiple images
in the single prompt are supported, while batches and video are deliberately
rejected in the Chapter 44 prefill boundary.
"""
function qwen3_vl_rope_layout(
    input_ids,
    image_grid_thw=nothing;
    attention_mask=nothing,
    checkpoint::Qwen3VLCheckpointSpec=qwen3_vl_checkpoint_spec(),
)
    tokens = _qwen3_vl_token_matrix(input_ids)
    sequence_length, batch_size = size(tokens)
    batch_size == 1 || throw(ArgumentError(
        "Chapter 44 Qwen3-VL mRoPE supports batch size one",
    ))
    sequence_length > 0 || throw(ArgumentError(
        "Qwen3-VL input_ids must contain at least one token",
    ))
    all(id -> 1 <= id <= checkpoint.text.vocab_size, tokens) ||
        throw(ArgumentError("Qwen3-VL input_ids contain an out-of-vocabulary id"))
    mask = _qwen3_vl_attention_matrix(attention_mask, size(tokens))
    all(any(view(mask, :, batch)) for batch in 1:batch_size) ||
        throw(ArgumentError("every Qwen3-VL batch item must contain a valid token"))

    grids = image_grid_thw === nothing ? NTuple{3,Int}[] :
        _qwen3_vl_grid_columns(image_grid_thw)
    position_ids = ones(Int, 3, sequence_length, batch_size)
    visual_mask = falses(sequence_length, batch_size)
    deltas = Matrix{Int}(undef, batch_size, 1)
    image_token = checkpoint.image_token_id + 1
    video_token = checkpoint.video_token_id + 1
    vision_start = checkpoint.vision_start_token_id + 1
    vision_end = checkpoint.vision_end_token_id + 1
    merge_size = checkpoint.vision.spatial_merge_size
    grid_index = 0

    for batch in 1:batch_size
        valid_positions = findall(view(mask, :, batch))
        filtered = tokens[valid_positions, batch]
        video_token in filtered && throw(ArgumentError(
            "Qwen3-VL video mRoPE requires the timestamp video processor",
        ))
        image_starts = Int[]
        for index in eachindex(filtered)
            filtered[index] == vision_start || continue
            index < length(filtered) || throw(ArgumentError(
                "Qwen3-VL vision_start token cannot terminate the prompt",
            ))
            filtered[index + 1] == image_token && push!(image_starts, index + 1)
        end

        if isempty(image_starts)
            isempty(grids) || throw(ArgumentError(
                "Qwen3-VL image grids were provided without image placeholders",
            ))
            image_token in filtered && throw(ArgumentError(
                "Qwen3-VL image placeholder is not preceded by vision_start",
            ))
            for (offset, source_position) in enumerate(valid_positions)
                value = offset - 1
                position_ids[:, source_position, batch] .= value
            end
            # Padding positions retain the HF sentinel value `1`, so taking a
            # maximum over the full storage is wrong for a single valid token.
            # Valid text coordinates are exactly 0:(nvalid - 1).
            deltas[batch, 1] = length(valid_positions) - sequence_length
            continue
        end

        columns = NTuple{3,Int}[]
        cursor = 1
        for image_start in image_starts
            grid_index += 1
            grid_index <= length(grids) || throw(ArgumentError(
                "Qwen3-VL prompt contains more image placeholders than grids",
            ))
            grid_t, grid_h, grid_w = grids[grid_index]
            grid_t == 1 || throw(ArgumentError(
                "Qwen3-VL image grids must have grid_t == 1",
            ))
            (grid_h % merge_size == 0 && grid_w % merge_size == 0) ||
                throw(ArgumentError(
                    "Qwen3-VL image grid is not divisible by spatial merge size",
                ))
            merged_h = grid_h ÷ merge_size
            merged_w = grid_w ÷ merge_size
            visual_length = grid_t * merged_h * merged_w
            image_start >= cursor || throw(ArgumentError(
                "Qwen3-VL image placeholders overlap",
            ))
            text_length = image_start - cursor
            base = isempty(columns) ? 0 : maximum(maximum, columns) + 1
            _qwen3_vl_append_text_positions!(columns, text_length, base)
            visual_base = base + text_length
            for temporal in 0:(grid_t - 1), height in 0:(merged_h - 1),
                width in 0:(merged_w - 1)
                push!(columns, (
                    visual_base + temporal,
                    visual_base + height,
                    visual_base + width,
                ))
            end
            last_visual = image_start + visual_length - 1
            last_visual <= length(filtered) || throw(ArgumentError(
                "Qwen3-VL image placeholder run exceeds the prompt",
            ))
            all(==(image_token), filtered[image_start:last_visual]) ||
                throw(ArgumentError(
                    "Qwen3-VL image placeholder run is not contiguous or has the wrong length",
                ))
            last_visual < length(filtered) || throw(ArgumentError(
                "Qwen3-VL image placeholder run must be followed by vision_end",
            ))
            filtered[last_visual + 1] == vision_end || throw(ArgumentError(
                "Qwen3-VL image placeholder run must be followed by vision_end",
            ))
            for filtered_index in image_start:last_visual
                visual_mask[valid_positions[filtered_index], batch] = true
            end
            cursor = last_visual + 1
        end
        if cursor <= length(filtered)
            base = isempty(columns) ? 0 : maximum(maximum, columns) + 1
            _qwen3_vl_append_text_positions!(
                columns,
                length(filtered) - cursor + 1,
                base,
            )
        end
        length(columns) == length(filtered) || error(
            "internal Qwen3-VL mRoPE position length mismatch",
        )
        count(==(image_token), filtered) == count(view(visual_mask, :, batch)) ||
            throw(ArgumentError(
                "Qwen3-VL prompt contains an orphan or extra image placeholder",
            ))
        for (index, source_position) in enumerate(valid_positions)
            position_ids[:, source_position, batch] .= columns[index]
        end
        deltas[batch, 1] = maximum(maximum, columns) + 1 - sequence_length
    end
    grid_index == length(grids) || throw(ArgumentError(
        "Qwen3-VL image grid count exceeds prompt image placeholders",
    ))
    return Qwen3VLRopeLayout(position_ids, deltas, visual_mask, mask)
end
