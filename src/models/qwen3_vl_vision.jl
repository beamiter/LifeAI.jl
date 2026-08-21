using BFloat16s: BFloat16
using LinearAlgebra: mul!
using NNlib: gather
using SpecialFunctions: erf

const _QWEN3_VL_GELU_TANH_COEFFICIENT = Float32(
    sqrt(2.0 / Float64(pi)),
)

# Chapter 43: the Qwen3-VL vision tower.  This boundary intentionally starts
# after image decoding, resize, normalization, and patchification.  It accepts
# the same flattened patch values and grid metadata consumed by the official
# vision model, without claiming that multimodal text generation is available.

"""
    Qwen3VLVisionInput(pixel_values, grid_thw; spec)

Validated, already-preprocessed vision input.  LifeAI stores `pixel_values` as
`(patch_width, total_patches)` and `grid_thw` as `(3, media_count)`.  The latter
is the transpose of Transformers' public `(media_count, 3)` tensor and remains
small, one-based host metadata even when `pixel_values` resides on a device.
"""
struct Qwen3VLVisionInput{P,G}
    pixel_values::P
    grid_thw::G
end

"""
Outputs of the standalone Qwen3-VL vision tower and its mergers.

`patch_hidden_state` retains one hidden vector per unmerged vision patch.
`visual_embeddings` is the main merger output consumed by the language model;
each entry in `deepstack` is likewise spatially merged.
"""
struct Qwen3VLVisionFeatures{H,V,D,C}
    patch_hidden_state::H
    visual_embeddings::V
    deepstack::D
    checkpoints::C
end

function Qwen3VLVisionInput(
    pixel_values::AbstractMatrix,
    grid_thw::AbstractMatrix{<:Integer};
    spec::Qwen3VLVisionSpec=qwen3_vl_checkpoint_spec().vision,
)
    all(
        dimension -> axes(pixel_values, dimension) ==
            Base.OneTo(size(pixel_values, dimension)),
        1:2,
    ) || throw(ArgumentError(
        "Qwen3-VL pixel_values must use one-based axes",
    ))
    grid_thw isa StridedMatrix || throw(ArgumentError(
        "Qwen3-VL grid_thw must be a host-resident StridedMatrix",
    ))
    all(
        dimension -> axes(grid_thw, dimension) ==
            Base.OneTo(size(grid_thw, dimension)),
        1:2,
    ) || throw(ArgumentError(
        "Qwen3-VL grid_thw must use one-based axes",
    ))
    expected_width = spec.in_channels * spec.temporal_patch_size *
        spec.patch_size * spec.patch_size
    size(pixel_values, 1) == expected_width || throw(DimensionMismatch(
        "Qwen3-VL pixel_values first dimension must be $expected_width; " *
        "got $(size(pixel_values, 1))",
    ))
    size(grid_thw, 1) == 3 || throw(DimensionMismatch(
        "Qwen3-VL grid_thw must have shape (3, media_count)",
    ))
    size(grid_thw, 2) > 0 || throw(ArgumentError(
        "Qwen3-VL grid_thw must describe at least one image or video",
    ))
    eltype(pixel_values) in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL pixel_values must contain Float32 or BFloat16 values",
    ))

    total_patches = 0
    for media in axes(grid_thw, 2)
        t = Int(grid_thw[1, media])
        h = Int(grid_thw[2, media])
        w = Int(grid_thw[3, media])
        t > 0 && h > 0 && w > 0 || throw(ArgumentError(
            "every Qwen3-VL grid dimension must be positive",
        ))
        h % spec.spatial_merge_size == 0 || throw(ArgumentError(
            "Qwen3-VL grid height $h is not divisible by spatial merge " *
            "size $(spec.spatial_merge_size)",
        ))
        w % spec.spatial_merge_size == 0 || throw(ArgumentError(
            "Qwen3-VL grid width $w is not divisible by spatial merge " *
            "size $(spec.spatial_merge_size)",
        ))
        total_patches = Base.Checked.checked_add(
            total_patches,
            Base.Checked.checked_mul(t, Base.Checked.checked_mul(h, w)),
        )
    end
    total_patches == size(pixel_values, 2) || throw(DimensionMismatch(
        "sum(t*h*w) in Qwen3-VL grid_thw is $total_patches, but " *
        "pixel_values contains $(size(pixel_values, 2)) patches",
    ))
    all(isfinite, pixel_values) || throw(ArgumentError(
        "Qwen3-VL pixel_values must be finite",
    ))
    return Qwen3VLVisionInput{typeof(pixel_values),typeof(grid_thw)}(
        pixel_values,
        grid_thw,
    )
end

_qwen3_vl_f32(x) = 1.0f0 .* x

function _qwen3_vl_cast_like(reference::AbstractArray, values)
    return eltype(reference) === BFloat16 ? BFloat16.(values) : Float32.(values)
end

function _qwen3_vl_linear(weight, bias, x::AbstractMatrix)
    output = similar(weight, Float32, size(weight, 1), size(x, 2))
    output .= reshape(_qwen3_vl_f32(bias), :, 1)
    if weight isa Array{BFloat16,2} && x isa Array{BFloat16,2}
        mul!(output, Float32.(weight), Float32.(x), 1.0f0, 1.0f0)
    else
        # CUDA's mixed GEMM supports BF16 inputs with a Float32 destination.
        # Seeding that destination with bias and using beta=1 mirrors the
        # single-rounding accumulator+bias contract of `torch.nn.Linear`.
        mul!(output, weight, x, 1.0f0, 1.0f0)
    end
    return _qwen3_vl_cast_like(x, output)
end

function _qwen3_vl_layernorm(x, scale, bias; epsilon::Float32=1.0f-6)
    xf = _qwen3_vl_f32(x)
    mean = sum(xf; dims=1) ./ Float32(size(x, 1))
    centered = xf .- mean
    variance = sum(abs2, centered; dims=1) ./ Float32(size(x, 1))
    normalized = centered ./ sqrt.(variance .+ epsilon)
    output = normalized .* reshape(_qwen3_vl_f32(scale), :, 1) .+
        reshape(_qwen3_vl_f32(bias), :, 1)
    return _qwen3_vl_cast_like(x, output)
end

function _qwen3_vl_gelu_tanh(x)
    xf = _qwen3_vl_f32(x)
    output = 0.5f0 .* xf .* (
        1.0f0 .+ tanh.(
            _QWEN3_VL_GELU_TANH_COEFFICIENT .*
                (xf .+ 0.044715f0 .* xf .* xf .* xf),
        )
    )
    return _qwen3_vl_cast_like(x, output)
end

function _qwen3_vl_gelu_exact(x)
    xf = _qwen3_vl_f32(x)
    output = 0.5f0 .* xf .* (1.0f0 .+ erf.(xf ./ sqrt(2.0f0)))
    return _qwen3_vl_cast_like(x, output)
end

function _qwen3_vl_read_parameter(
    reader::HFSafetensorsReader,
    name::String,
    shape::Tuple,
    target_dtype::Type,
    to_device,
)
    tensor = read_safetensors_tensor(reader, name; target_dtype)
    _expect_tensor(tensor, name, shape)
    return to_device(tensor)
end

function _qwen3_vl_load_norm(
    reader,
    prefix::String,
    width::Int,
    target_dtype,
    to_device,
)
    readp(name, shape) = _qwen3_vl_read_parameter(
        reader, name, shape, target_dtype, to_device,
    )
    return (;
        scale=readp("$prefix.weight", (width,)),
        bias=readp("$prefix.bias", (width,)),
    )
end

function _qwen3_vl_load_merger(
    reader,
    prefix::String,
    spec::Qwen3VLVisionSpec,
    target_dtype,
    to_device;
    postshuffle_norm::Bool,
)
    merged_width = spec.hidden_size * spec.spatial_merge_size^2
    norm_width = postshuffle_norm ? merged_width : spec.hidden_size
    readp(name, shape) = _qwen3_vl_read_parameter(
        reader, name, shape, target_dtype, to_device,
    )
    return (;
        postshuffle_norm,
        norm=_qwen3_vl_load_norm(
            reader, "$prefix.norm", norm_width, target_dtype, to_device,
        ),
        fc1_weight=readp(
            "$prefix.linear_fc1.weight", (merged_width, merged_width),
        ),
        fc1_bias=readp("$prefix.linear_fc1.bias", (merged_width,)),
        fc2_weight=readp(
            "$prefix.linear_fc2.weight", (spec.out_hidden_size, merged_width),
        ),
        fc2_bias=readp("$prefix.linear_fc2.bias", (spec.out_hidden_size,)),
    )
end

"""
    load_hf_qwen3_vl_vision_parameters(model_dir; target_dtype=BFloat16,
                                        to_device=identity, spec=...)

Read only the vision tower, main merger, and DeepStack merger tensors from a
local Qwen3-VL checkpoint.  The full language model is never materialized.
`to_device` is applied to each completed semantic array, permitting bounded
host-to-device loading.
"""
function load_hf_qwen3_vl_vision_parameters(
    model_dir::AbstractString;
    target_dtype::Type=BFloat16,
    to_device=identity,
    spec::Qwen3VLVisionSpec=qwen3_vl_checkpoint_spec().vision,
)
    target_dtype in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL vision loading supports Float32 or BFloat16",
    ))
    isdir(model_dir) || throw(ArgumentError(
        "Qwen3-VL model directory does not exist: $model_dir",
    ))
    loaded_config = load_hf_qwen3_vl_config(
        joinpath(model_dir, "config.json"),
    )
    loaded_config.vision == spec || throw(ArgumentError(
        "Qwen3-VL checkpoint vision config does not match the requested spec",
    ))
    reader = open_safetensors_reader(model_dir)
    readp(name, shape) = _qwen3_vl_read_parameter(
        reader, name, shape, target_dtype, to_device,
    )

    raw_patch_weight = read_safetensors_tensor(
        reader,
        "model.visual.patch_embed.proj.weight";
        target_dtype,
    )
    patch_shape = (
        spec.hidden_size,
        spec.in_channels,
        spec.temporal_patch_size,
        spec.patch_size,
        spec.patch_size,
    )
    _expect_tensor(
        raw_patch_weight,
        "model.visual.patch_embed.proj.weight",
        patch_shape,
    )
    # Safetensors arrays have semantic HF axes.  Explicitly reorder the four
    # Conv3D input axes so matrix column f follows C,T,H,W row-major order.
    patch_weight = to_device(reshape(
        permutedims(raw_patch_weight, (1, 5, 4, 3, 2)),
        spec.hidden_size,
        :,
    ))
    patch_bias = readp(
        "model.visual.patch_embed.proj.bias", (spec.hidden_size,),
    )
    raw_pos = read_safetensors_tensor(
        reader,
        "model.visual.pos_embed.weight";
        target_dtype,
    )
    _expect_tensor(
        raw_pos,
        "model.visual.pos_embed.weight",
        (spec.num_position_embeddings, spec.hidden_size),
    )
    pos_embedding = to_device(permutedims(raw_pos, (2, 1)))

    blocks = ntuple(spec.depth) do julia_layer
        layer = julia_layer - 1
        prefix = "model.visual.blocks.$layer"
        return (;
            norm1=_qwen3_vl_load_norm(
                reader, "$prefix.norm1", spec.hidden_size,
                target_dtype, to_device,
            ),
            norm2=_qwen3_vl_load_norm(
                reader, "$prefix.norm2", spec.hidden_size,
                target_dtype, to_device,
            ),
            qkv_weight=readp(
                "$prefix.attn.qkv.weight",
                (3 * spec.hidden_size, spec.hidden_size),
            ),
            qkv_bias=readp(
                "$prefix.attn.qkv.bias", (3 * spec.hidden_size,),
            ),
            proj_weight=readp(
                "$prefix.attn.proj.weight",
                (spec.hidden_size, spec.hidden_size),
            ),
            proj_bias=readp(
                "$prefix.attn.proj.bias", (spec.hidden_size,),
            ),
            fc1_weight=readp(
                "$prefix.mlp.linear_fc1.weight",
                (spec.intermediate_size, spec.hidden_size),
            ),
            fc1_bias=readp(
                "$prefix.mlp.linear_fc1.bias", (spec.intermediate_size,),
            ),
            fc2_weight=readp(
                "$prefix.mlp.linear_fc2.weight",
                (spec.hidden_size, spec.intermediate_size),
            ),
            fc2_bias=readp(
                "$prefix.mlp.linear_fc2.bias", (spec.hidden_size,),
            ),
        )
    end

    merger = _qwen3_vl_load_merger(
        reader,
        "model.visual.merger",
        spec,
        target_dtype,
        to_device;
        postshuffle_norm=false,
    )
    deepstack_mergers = ntuple(length(spec.deepstack_visual_indexes)) do index
        _qwen3_vl_load_merger(
            reader,
            "model.visual.deepstack_merger_list.$(index - 1)",
            spec,
            target_dtype,
            to_device;
            postshuffle_norm=true,
        )
    end
    return (;
        patch_weight,
        patch_bias,
        pos_embedding,
        blocks,
        merger,
        deepstack_mergers,
        spec,
        source=abspath(model_dir),
    )
end

function _qwen3_vl_patch_order(grid_thw, merge_size::Int)
    total = sum(
        Int(grid_thw[1, media]) * Int(grid_thw[2, media]) *
        Int(grid_thw[3, media]) for media in axes(grid_thw, 2)
    )
    frame_count = sum(Int(grid_thw[1, media]) for media in axes(grid_thw, 2))
    frame_ranges = Vector{UnitRange{Int}}(undef, frame_count)
    heights = Vector{Int}(undef, total)
    widths = Vector{Int}(undef, total)
    cursor = 1
    frame_index = 0
    for media in axes(grid_thw, 2)
        t = Int(grid_thw[1, media])
        h = Int(grid_thw[2, media])
        w = Int(grid_thw[3, media])
        for _ in 1:t
            frame_index += 1
            first_index = cursor
            for block_h in 0:(h ÷ merge_size - 1)
                for block_w in 0:(w ÷ merge_size - 1)
                    for within_h in 0:(merge_size - 1)
                        for within_w in 0:(merge_size - 1)
                            heights[cursor] = block_h * merge_size + within_h
                            widths[cursor] = block_w * merge_size + within_w
                            cursor += 1
                        end
                    end
                end
            end
            frame_ranges[frame_index] = first_index:(cursor - 1)
        end
    end
    cursor == total + 1 || error("internal Qwen3-VL patch-order mismatch")
    frame_index == frame_count || error(
        "internal Qwen3-VL temporal frame-count mismatch",
    )
    return heights, widths, frame_ranges
end

function _qwen3_vl_interpolated_positions(spec, pos_embedding, grid_thw)
    side = isqrt(spec.num_position_embeddings)
    side^2 == spec.num_position_embeddings || throw(ArgumentError(
        "Qwen3-VL learned position count must be a perfect square",
    ))
    heights, widths, _ = _qwen3_vl_patch_order(
        grid_thw, spec.spatial_merge_size,
    )
    indices = Matrix{Int}(undef, 4, length(heights))
    weights = Matrix{Float32}(undef, 4, length(heights))
    cursor = 1
    for media in axes(grid_thw, 2)
        t = Int(grid_thw[1, media])
        h = Int(grid_thw[2, media])
        w = Int(grid_thw[3, media])
        patches_per_frame = h * w
        for _ in 1:t
            for local_index in 1:patches_per_frame
                y = heights[cursor]
                x = widths[cursor]
                source_y = h == 1 ? 0.0f0 :
                    Float32(y) * Float32(side - 1) / Float32(h - 1)
                source_x = w == 1 ? 0.0f0 :
                    Float32(x) * Float32(side - 1) / Float32(w - 1)
                y0 = clamp(floor(Int, source_y), 0, side - 1)
                x0 = clamp(floor(Int, source_x), 0, side - 1)
                y1 = min(y0 + 1, side - 1)
                x1 = min(x0 + 1, side - 1)
                wy = source_y - Float32(y0)
                wx = source_x - Float32(x0)
                indices[1, cursor] = y0 * side + x0 + 1
                indices[2, cursor] = y0 * side + x1 + 1
                indices[3, cursor] = y1 * side + x0 + 1
                indices[4, cursor] = y1 * side + x1 + 1
                weights[1, cursor] = (1.0f0 - wy) * (1.0f0 - wx)
                weights[2, cursor] = (1.0f0 - wy) * wx
                weights[3, cursor] = wy * (1.0f0 - wx)
                weights[4, cursor] = wy * wx
                cursor += 1
            end
        end
    end
    to_device = _bf16a_device_mover(pos_embedding)
    device_indices = to_device(indices)
    device_weights = _qwen3_vl_cast_like(
        pos_embedding,
        to_device(weights),
    )
    gathered = gather(pos_embedding, device_indices)
    weighted = _qwen3_vl_cast_like(
        pos_embedding,
        _qwen3_vl_f32(gathered) .* reshape(
            _qwen3_vl_f32(device_weights),
            1,
            4,
            length(heights),
        ),
    )
    interpolated = reshape(weighted[:, 1, :], size(pos_embedding, 1), :)
    for corner in 2:4
        term = reshape(weighted[:, corner, :], size(pos_embedding, 1), :)
        interpolated = _qwen3_vl_cast_like(
            pos_embedding,
            _qwen3_vl_f32(interpolated) .+ _qwen3_vl_f32(term),
        )
    end
    return interpolated
end

function _qwen3_vl_vision_rope(spec, reference, grid_thw)
    heights, widths, frame_ranges = _qwen3_vl_patch_order(
        grid_thw, spec.spatial_merge_size,
    )
    head_dim = spec.hidden_size ÷ spec.num_heads
    rotary_dim = head_dim ÷ 2
    iseven(rotary_dim) || throw(ArgumentError(
        "Qwen3-VL vision rotary dimension must be even",
    ))
    inv_frequency = _qwen3_vl_vision_inv_frequency(rotary_dim)
    positions = vcat(
        reshape(inv_frequency, :, 1) .* reshape(Float32.(heights), 1, :),
        reshape(inv_frequency, :, 1) .* reshape(Float32.(widths), 1, :),
    )
    embedding = vcat(positions, positions)
    to_device = _bf16a_device_mover(reference)
    return to_device(cos.(embedding)), to_device(sin.(embedding)), frame_ranges
end

function _qwen3_vl_vision_inv_frequency(rotary_dim::Int)
    rotary_dim > 0 || throw(ArgumentError(
        "Qwen3-VL vision rotary dimension must be positive",
    ))
    return Float32[
        1.0f0 / (10_000.0f0 ^ (Float32(index) / Float32(rotary_dim)))
        for index in 0:2:(rotary_dim - 1)
    ]
end

function _qwen3_vl_apply_vision_rope(x, cos_values, sin_values)
    head_dim = size(x, 1)
    half = head_dim ÷ 2
    xf1 = _qwen3_vl_f32(x[1:half, :, :])
    xf2 = _qwen3_vl_f32(x[(half + 1):end, :, :])
    cos_view = reshape(cos_values, head_dim, 1, size(x, 3))
    sin_view = reshape(sin_values, head_dim, 1, size(x, 3))
    rotated = vcat(.-xf2, xf1)
    output = _qwen3_vl_f32(x) .* cos_view .+ rotated .* sin_view
    return _qwen3_vl_cast_like(x, output)
end

function _qwen3_vl_attention_core(
    queries,
    keys,
    values;
    scaling::Float32,
)
    if eltype(queries) === BFloat16
        return _bf16a_attention(
            queries,
            keys,
            values;
            scaling,
            mask=nothing,
        )
    end

    head_dim, num_heads, query_tokens, batch_size = size(queries)
    size(keys) == (head_dim, num_heads, size(keys, 3), batch_size) ||
        throw(DimensionMismatch("Qwen3-VL vision key shape mismatch"))
    size(values) == size(keys) || throw(DimensionMismatch(
        "Qwen3-VL vision value shape mismatch",
    ))
    key_tokens = size(keys, 3)
    q3 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens,
        head_dim,
        num_heads * batch_size,
    )
    k3 = reshape(
        permutedims(keys, (1, 3, 2, 4)),
        head_dim,
        key_tokens,
        num_heads * batch_size,
    )
    scores = _bf16a_batched_mul(q3, k3) .* scaling
    maxima = maximum(scores; dims=2)
    exponentials = exp.(scores .- maxima)
    weights = exponentials ./ sum(exponentials; dims=2)
    v3 = reshape(
        permutedims(values, (1, 3, 2, 4)),
        head_dim,
        key_tokens,
        num_heads * batch_size,
    )
    context3 = _bf16a_batched_mul(v3, permutedims(weights, (2, 1, 3)))
    return reshape(
        permutedims(
            reshape(
                context3,
                head_dim,
                query_tokens,
                num_heads,
                batch_size,
            ),
            (1, 3, 2, 4),
        ),
        head_dim,
        num_heads,
        query_tokens,
        batch_size,
    )
end

function _qwen3_vl_vision_attention(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    frame_ranges,
)
    qkv = _qwen3_vl_linear(block.qkv_weight, block.qkv_bias, x)
    hidden = spec.hidden_size
    head_dim = hidden ÷ spec.num_heads
    tokens = size(x, 2)
    queries = reshape(qkv[1:hidden, :], head_dim, spec.num_heads, tokens)
    keys = reshape(qkv[(hidden + 1):(2 * hidden), :], head_dim, spec.num_heads, tokens)
    values = reshape(qkv[(2 * hidden + 1):(3 * hidden), :], head_dim, spec.num_heads, tokens)
    queries = _qwen3_vl_apply_vision_rope(
        queries, cos_values, sin_values,
    )
    keys = _qwen3_vl_apply_vision_rope(keys, cos_values, sin_values)

    chunks = map(frame_ranges) do range
        q = reshape(queries[:, :, range], head_dim, spec.num_heads, length(range), 1)
        k = reshape(keys[:, :, range], head_dim, spec.num_heads, length(range), 1)
        v = reshape(values[:, :, range], head_dim, spec.num_heads, length(range), 1)
        context = _qwen3_vl_attention_core(
            q,
            k,
            v;
            scaling=1.0f0 / sqrt(Float32(head_dim)),
        )
        reshape(context, hidden, length(range))
    end
    context = length(chunks) == 1 ? only(chunks) : hcat(chunks...)
    return _qwen3_vl_linear(
        block.proj_weight, block.proj_bias, context,
    )
end

function _qwen3_vl_vision_block(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    frame_ranges,
)
    normed = _qwen3_vl_layernorm(
        x, block.norm1.scale, block.norm1.bias,
    )
    attention = _qwen3_vl_vision_attention(
        spec, block, normed, cos_values, sin_values, frame_ranges,
    )
    x = _qwen3_vl_cast_like(x, _qwen3_vl_f32(x) .+ _qwen3_vl_f32(attention))
    normed = _qwen3_vl_layernorm(
        x, block.norm2.scale, block.norm2.bias,
    )
    hidden = _qwen3_vl_linear(block.fc1_weight, block.fc1_bias, normed)
    hidden = _qwen3_vl_gelu_tanh(hidden)
    mlp = _qwen3_vl_linear(block.fc2_weight, block.fc2_bias, hidden)
    return _qwen3_vl_cast_like(x, _qwen3_vl_f32(x) .+ _qwen3_vl_f32(mlp))
end

function _qwen3_vl_merge(spec, merger, x)
    merge_unit = spec.spatial_merge_size^2
    size(x, 2) % merge_unit == 0 || throw(DimensionMismatch(
        "Qwen3-VL vision tokens are not divisible by the merge unit $merge_unit",
    ))
    merged_width = spec.hidden_size * merge_unit
    hidden = if merger.postshuffle_norm
        grouped = reshape(x, merged_width, :)
        _qwen3_vl_layernorm(
            grouped, merger.norm.scale, merger.norm.bias,
        )
    else
        normalized = _qwen3_vl_layernorm(
            x, merger.norm.scale, merger.norm.bias,
        )
        reshape(normalized, merged_width, :)
    end
    hidden = _qwen3_vl_linear(
        merger.fc1_weight, merger.fc1_bias, hidden,
    )
    hidden = _qwen3_vl_gelu_exact(hidden)
    return _qwen3_vl_linear(
        merger.fc2_weight, merger.fc2_bias, hidden,
    )
end

"""
    hf_qwen3_vl_vision_forward(parameters, input; capture_layers=())

Run the standalone Qwen3-VL vision tower, main merger, and the three
checkpoint-defined DeepStack mergers.  `capture_layers` uses official 0-based
block indices and is intended for independently frozen parity boundaries.
This function does not execute the language decoder or generation.
"""
function hf_qwen3_vl_vision_forward(
    parameters,
    input::Qwen3VLVisionInput;
    capture_layers=(),
)
    spec = parameters.spec
    # Revalidate even explicitly constructed field values; callers must not be
    # able to bypass the public constructor's grid/patch safety boundary.
    validated = Qwen3VLVisionInput(
        input.pixel_values, input.grid_thw; spec,
    )
    requested = Set(Int.(collect(capture_layers)))
    all(layer -> 0 <= layer < spec.depth, requested) || throw(ArgumentError(
        "Qwen3-VL capture layer is outside 0:$(spec.depth - 1)",
    ))
    eltype(validated.pixel_values) == eltype(parameters.patch_weight) ||
        throw(ArgumentError(
            "Qwen3-VL pixel_values dtype must match loaded vision weights",
        ))

    x = _qwen3_vl_linear(
        parameters.patch_weight,
        parameters.patch_bias,
        validated.pixel_values,
    )
    positions = _qwen3_vl_interpolated_positions(
        spec, parameters.pos_embedding, validated.grid_thw,
    )
    x = _qwen3_vl_cast_like(x, _qwen3_vl_f32(x) .+ _qwen3_vl_f32(positions))
    cos_values, sin_values, frame_ranges = _qwen3_vl_vision_rope(
        spec, x, validated.grid_thw,
    )

    deepstack = Vector{Any}()
    checkpoints = Dict{Int,Any}()
    deepstack_lookup = Dict(
        layer => index for (index, layer) in
        enumerate(spec.deepstack_visual_indexes)
    )
    for julia_layer in 1:spec.depth
        layer = julia_layer - 1
        x = _qwen3_vl_vision_block(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            frame_ranges,
        )
        layer in requested && (checkpoints[layer] = x)
        merger_index = get(deepstack_lookup, layer, 0)
        merger_index > 0 && push!(
            deepstack,
            _qwen3_vl_merge(
                spec,
                parameters.deepstack_mergers[merger_index],
                x,
            ),
        )
    end
    visual_embeddings = _qwen3_vl_merge(spec, parameters.merger, x)
    return Qwen3VLVisionFeatures(
        x,
        visual_embeddings,
        Tuple(deepstack),
        checkpoints,
    )
end
