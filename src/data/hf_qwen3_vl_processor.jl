using JSON3
using SHA: sha256

"""
    Qwen3VLProcessorSpec

Frozen image-processor boundary for the official Qwen3-VL-2B checkpoint. This
contract starts after image decoding, resizing, rescaling, and normalization;
it deliberately does not define any raw image or multimodal-chat behavior.
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
