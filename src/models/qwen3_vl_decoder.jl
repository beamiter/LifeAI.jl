using BFloat16s: BFloat16
using MLDataDevices: get_device
using NNlib: batched_mul, gather
import NNlib

"""Captured states from one image-conditioned Qwen3-VL decoder prefill."""
struct Qwen3VLTextPrefill{E,B,L,F,O,R}
    input_embeddings::E
    block_outputs::B
    layer_outputs::L
    final_hidden::F
    logits::O
    rope_layout::R

    function Qwen3VLTextPrefill(
        input_embeddings::Union{Nothing,AbstractArray},
        block_outputs::AbstractDict,
        layer_outputs::AbstractDict,
        final_hidden::Union{Nothing,AbstractArray},
        logits::AbstractArray,
        rope_layout::Qwen3VLRopeLayout,
    )
        _validate_qwen3_vl_text_prefill(
            input_embeddings,
            block_outputs,
            layer_outputs,
            final_hidden,
            logits,
            rope_layout,
        )
        return new{
            typeof(input_embeddings),
            typeof(block_outputs),
            typeof(layer_outputs),
            typeof(final_hidden),
            typeof(logits),
            typeof(rope_layout),
        }(
            input_embeddings,
            block_outputs,
            layer_outputs,
            final_hidden,
            logits,
            rope_layout,
        )
    end
end

function _qwen3_vl_text_prefill_state_width(
    state,
    label::AbstractString,
    sequence_length::Int,
    batch_size::Int,
    value_type,
)
    state isa AbstractArray || throw(ArgumentError("$label must be an array"))
    ndims(state) == 3 || throw(DimensionMismatch(
        "$label must have shape (hidden, sequence, batch)",
    ))
    all(
        dimension -> axes(state, dimension) ==
            Base.OneTo(size(state, dimension)),
        1:3,
    ) || throw(ArgumentError("$label must use one-based axes"))
    size(state, 1) > 0 || throw(ArgumentError(
        "$label hidden width must be positive",
    ))
    size(state, 2) == sequence_length && size(state, 3) == batch_size ||
        throw(DimensionMismatch(
            "$label sequence and batch dimensions must match rope_layout",
        ))
    eltype(state) == value_type || throw(ArgumentError(
        "$label dtype must match logits",
    ))
    return size(state, 1)
end

function _qwen3_vl_text_prefill_capture_layers(
    states,
    label::AbstractString,
    sequence_length::Int,
    batch_size::Int,
    value_type,
    hidden_widths::Vector{Int},
)
    layers = Set{Int}()
    for (raw_layer, state) in pairs(states)
        layer = _strict_host_int(raw_layer, "$label layer")
        layer >= 0 || throw(ArgumentError(
            "$label layer must be non-negative",
        ))
        layer in layers && throw(ArgumentError(
            "$label contains duplicate normalized layer ids",
        ))
        push!(layers, layer)
        push!(hidden_widths, _qwen3_vl_text_prefill_state_width(
            state,
            "$label[$layer]",
            sequence_length,
            batch_size,
            value_type,
        ))
    end
    return layers
end

function _validate_qwen3_vl_text_prefill(
    input_embeddings,
    block_outputs,
    layer_outputs,
    final_hidden,
    logits,
    rope_layout,
)
    sequence_length, batch_size = size(rope_layout.visual_mask)
    sequence_length > 0 && batch_size > 0 || throw(ArgumentError(
        "Qwen3-VL text prefill rope layout must be non-empty",
    ))
    ndims(logits) == 3 || throw(DimensionMismatch(
        "Qwen3-VL text prefill logits must have shape (vocab, kept, batch)",
    ))
    all(
        dimension -> axes(logits, dimension) ==
            Base.OneTo(size(logits, dimension)),
        1:3,
    ) || throw(ArgumentError(
        "Qwen3-VL text prefill logits must use one-based axes",
    ))
    size(logits, 1) > 0 || throw(ArgumentError(
        "Qwen3-VL text prefill vocabulary width must be positive",
    ))
    1 <= size(logits, 2) <= sequence_length || throw(DimensionMismatch(
        "Qwen3-VL text prefill kept logits must be in 1:$sequence_length",
    ))
    size(logits, 3) == batch_size || throw(DimensionMismatch(
        "Qwen3-VL text prefill logits batch must match rope_layout",
    ))
    value_type = eltype(logits)
    value_type in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL text prefill arrays must contain Float32 or BFloat16 values",
    ))

    hidden_widths = Int[]
    input_embeddings === nothing || push!(
        hidden_widths,
        _qwen3_vl_text_prefill_state_width(
            input_embeddings,
            "Qwen3-VL input_embeddings",
            sequence_length,
            batch_size,
            value_type,
        ),
    )
    final_hidden === nothing || push!(
        hidden_widths,
        _qwen3_vl_text_prefill_state_width(
            final_hidden,
            "Qwen3-VL final_hidden",
            sequence_length,
            batch_size,
            value_type,
        ),
    )
    block_layers = _qwen3_vl_text_prefill_capture_layers(
        block_outputs,
        "Qwen3-VL block_outputs",
        sequence_length,
        batch_size,
        value_type,
        hidden_widths,
    )
    layer_layers = _qwen3_vl_text_prefill_capture_layers(
        layer_outputs,
        "Qwen3-VL layer_outputs",
        sequence_length,
        batch_size,
        value_type,
        hidden_widths,
    )
    block_layers == layer_layers || throw(DimensionMismatch(
        "Qwen3-VL block_outputs and layer_outputs must capture the same layers",
    ))
    isempty(hidden_widths) || all(==(first(hidden_widths)), hidden_widths) ||
        throw(DimensionMismatch(
            "Qwen3-VL text prefill hidden widths must match",
        ))
    return nothing
end

function _qwen3_vl_text_read_parameter(
    reader,
    name::String,
    shape::Tuple,
    target_dtype,
    to_device,
)
    value = read_safetensors_tensor(reader, name; target_dtype)
    _expect_tensor(value, name, shape)
    return to_device(value)
end

"""
    load_hf_qwen3_vl_text_parameters(model_dir; target_dtype=BFloat16,
                                      to_device=identity)

Stream only the Qwen3-VL language tower into a compact parameter tree.  Each
tensor is transferred by `to_device` immediately after it is decoded, so the
complete 2B state dict is never materialized on the host.  The tied output
projection shares the single embedding matrix.
"""
function load_hf_qwen3_vl_text_parameters(
    model_dir::AbstractString;
    target_dtype::Type=BFloat16,
    to_device=identity,
    checkpoint::Qwen3VLCheckpointSpec=qwen3_vl_checkpoint_spec(),
)
    target_dtype in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL text loading supports Float32 or BFloat16",
    ))
    isdir(model_dir) || throw(ArgumentError(
        "Qwen3-VL model directory does not exist: $model_dir",
    ))
    loaded = load_hf_qwen3_vl_config(joinpath(model_dir, "config.json"))
    loaded.text == checkpoint.text || throw(ArgumentError(
        "Qwen3-VL checkpoint text config does not match the requested spec",
    ))
    reader = open_safetensors_reader(model_dir)
    expected = qwen3_vl_expected_tensor_shapes(checkpoint)
    Set(String.(keys(reader))) == Set(keys(expected)) || throw(ArgumentError(
        "Qwen3-VL safetensors names do not match the frozen checkpoint",
    ))
    text = checkpoint.text
    prefix = "model.language_model"
    readp(name, shape) = _qwen3_vl_text_read_parameter(
        reader,
        name,
        shape,
        target_dtype,
        to_device,
    )

    raw_embedding = read_safetensors_tensor(
        reader,
        "$prefix.embed_tokens.weight";
        target_dtype,
    )
    _expect_tensor(
        raw_embedding,
        "$prefix.embed_tokens.weight",
        (text.vocab_size, text.hidden_size),
    )
    embedding = to_device(permutedims(raw_embedding, (2, 1)))
    raw_embedding = nothing
    blocks = ntuple(text.num_hidden_layers) do julia_layer
        layer = julia_layer - 1
        layer_prefix = "$prefix.layers.$layer"
        return (;
            norm1=readp(
                "$layer_prefix.input_layernorm.weight",
                (text.hidden_size,),
            ),
            q_weight=readp(
                "$layer_prefix.self_attn.q_proj.weight",
                (text.num_attention_heads * text.head_dim, text.hidden_size),
            ),
            k_weight=readp(
                "$layer_prefix.self_attn.k_proj.weight",
                (text.num_key_value_heads * text.head_dim, text.hidden_size),
            ),
            v_weight=readp(
                "$layer_prefix.self_attn.v_proj.weight",
                (text.num_key_value_heads * text.head_dim, text.hidden_size),
            ),
            o_weight=readp(
                "$layer_prefix.self_attn.o_proj.weight",
                (text.hidden_size, text.num_attention_heads * text.head_dim),
            ),
            q_norm=readp(
                "$layer_prefix.self_attn.q_norm.weight",
                (text.head_dim,),
            ),
            k_norm=readp(
                "$layer_prefix.self_attn.k_norm.weight",
                (text.head_dim,),
            ),
            norm2=readp(
                "$layer_prefix.post_attention_layernorm.weight",
                (text.hidden_size,),
            ),
            gate_weight=readp(
                "$layer_prefix.mlp.gate_proj.weight",
                (text.intermediate_size, text.hidden_size),
            ),
            up_weight=readp(
                "$layer_prefix.mlp.up_proj.weight",
                (text.intermediate_size, text.hidden_size),
            ),
            down_weight=readp(
                "$layer_prefix.mlp.down_proj.weight",
                (text.hidden_size, text.intermediate_size),
            ),
        )
    end
    final_norm = readp("$prefix.norm.weight", (text.hidden_size,))
    return (;
        embedding,
        blocks,
        final_norm,
        spec=text,
        checkpoint,
        source=abspath(model_dir),
    )
end

function _qwen3_vl_text_rmsnorm(x, scale, epsilon::Real)
    epsilon_f32 = Float32(epsilon)
    eltype(x) === BFloat16 && return _bf16a_rmsnorm(x, scale, epsilon_f32)
    xf = Float32.(x)
    mean_square = sum(abs2, xf; dims=1) ./ Float32(size(x, 1))
    shape = ntuple(index -> index == 1 ? size(x, 1) : 1, ndims(x))
    return xf ./ sqrt.(mean_square .+ epsilon_f32) .* reshape(Float32.(scale), shape)
end

function _qwen3_vl_text_residual(x, value)
    eltype(x) === BFloat16 && return BFloat16.(
        _bf16a_f32(x) .+ _bf16a_f32(value),
    )
    return x .+ value
end

function _qwen3_vl_text_mrope(
    spec::Qwen3VLTextSpec,
    reference,
    position_ids::AbstractArray{<:Integer,3},
)
    size(position_ids, 1) == 3 || throw(DimensionMismatch(
        "Qwen3-VL position_ids must have three T/H/W rows",
    ))
    half = spec.head_dim ÷ 2
    sum(BigInt, spec.mrope_section) == BigInt(half) || throw(ArgumentError(
        "Qwen3-VL mRoPE sections do not partition half the head dimension",
    ))
    inv_frequency = Float32[
        1.0f0 / Float32(spec.rope_theta ^ (Float64(index) / Float64(half)))
        for index in 0:(half - 1)
    ]
    _, sequence_length, batch_size = size(position_ids)
    frequencies = Array{Float32}(undef, half, sequence_length, batch_size)
    @inbounds for batch in 1:batch_size, token in 1:sequence_length,
                  lane in 1:half
        section_lane = cld(lane, 3)
        axis = if section_lane <= spec.mrope_section[2] && lane % 3 == 2
            2
        elseif section_lane <= spec.mrope_section[3] && lane % 3 == 0
            3
        else
            1
        end
        frequencies[lane, token, batch] = inv_frequency[lane] *
            Float32(position_ids[axis, token, batch])
    end
    to_device = _bf16a_device_mover(reference)
    if eltype(reference) === BFloat16
        return (
            to_device(BFloat16.(cos.(frequencies))),
            to_device(BFloat16.(sin.(frequencies))),
        )
    end
    return to_device(cos.(frequencies)), to_device(sin.(frequencies))
end

function _qwen3_vl_text_apply_rope(x, cos_values, sin_values)
    size(cos_values, 3) == 1 || throw(ArgumentError(
        "Chapter 44 Qwen3-VL decoder prefill supports batch size one",
    ))
    cos_slice = reshape(cos_values, size(cos_values, 1), size(cos_values, 2))
    sin_slice = reshape(sin_values, size(sin_values, 1), size(sin_values, 2))
    eltype(x) === BFloat16 && return _bf16a_apply_rope(x, cos_slice, sin_slice)
    half = size(x, 1) ÷ 2
    cosine = reshape(cos_slice, half, 1, size(x, 3), 1)
    sine = reshape(sin_slice, half, 1, size(x, 3), 1)
    first_half = x[1:half, :, :, :]
    second_half = x[(half + 1):end, :, :, :]
    return cat(
        first_half .* cosine .- second_half .* sine,
        second_half .* cosine .+ first_half .* sine;
        dims=1,
    )
end

function _qwen3_vl_text_f32_attention(
    queries,
    keys,
    values,
    mask;
    scaling::Float32,
)
    head_dim, num_heads, query_tokens, batch_size = size(queries)
    _, num_kv_heads, key_tokens, _ = size(keys)
    num_heads % num_kv_heads == 0 || throw(DimensionMismatch(
        "Qwen3-VL query heads must be divisible by KV heads",
    ))
    groups = num_heads ÷ num_kv_heads
    repeated_keys = repeat(keys; inner=(1, groups, 1, 1))
    repeated_values = repeat(values; inner=(1, groups, 1, 1))
    head_batches = num_heads * batch_size
    q3 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens, head_dim, head_batches,
    )
    k3 = reshape(
        permutedims(repeated_keys, (1, 3, 2, 4)),
        head_dim, key_tokens, head_batches,
    )
    scores = batched_mul(q3, k3) .* scaling
    if mask !== nothing
        scores = scores .+ reshape(mask, query_tokens, key_tokens, 1)
    end
    maxima = maximum(scores; dims=2)
    exponentials = exp.(scores .- maxima)
    weights = exponentials ./ sum(exponentials; dims=2)
    v3 = reshape(
        permutedims(repeated_values, (1, 3, 2, 4)),
        head_dim, key_tokens, head_batches,
    )
    context3 = batched_mul(v3, permutedims(weights, (2, 1, 3)))
    context4 = permutedims(
        reshape(context3, head_dim, query_tokens, num_heads, batch_size),
        (1, 3, 2, 4),
    )
    return reshape(context4, head_dim, num_heads, query_tokens, batch_size)
end

function _qwen3_vl_text_block(spec, block, x, cos_values, sin_values, mask)
    head_dim = spec.head_dim
    sequence_length, batch_size = size(x, 2), size(x, 3)
    normed = _qwen3_vl_text_rmsnorm(x, block.norm1, spec.rms_norm_eps)
    queries = reshape(
        _bf16a_linear(block.q_weight, normed),
        head_dim, spec.num_attention_heads, sequence_length, batch_size,
    )
    keys = reshape(
        _bf16a_linear(block.k_weight, normed),
        head_dim, spec.num_key_value_heads, sequence_length, batch_size,
    )
    values = reshape(
        _bf16a_linear(block.v_weight, normed),
        head_dim, spec.num_key_value_heads, sequence_length, batch_size,
    )
    queries = _qwen3_vl_text_rmsnorm(
        queries, block.q_norm, spec.rms_norm_eps,
    )
    keys = _qwen3_vl_text_rmsnorm(keys, block.k_norm, spec.rms_norm_eps)
    queries = _qwen3_vl_text_apply_rope(queries, cos_values, sin_values)
    keys = _qwen3_vl_text_apply_rope(keys, cos_values, sin_values)
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    context = if eltype(x) === BFloat16
        _bf16a_attention(queries, keys, values; scaling, mask)
    else
        _qwen3_vl_text_f32_attention(
            queries, keys, values, mask; scaling,
        )
    end
    attention = _bf16a_linear(
        block.o_weight,
        reshape(context, spec.hidden_size, sequence_length, batch_size),
    )
    x = _qwen3_vl_text_residual(x, attention)
    normed = _qwen3_vl_text_rmsnorm(x, block.norm2, spec.rms_norm_eps)
    gate = _bf16a_linear(block.gate_weight, normed)
    up = _bf16a_linear(block.up_weight, normed)
    hidden = if eltype(x) === BFloat16
        gate_f32 = _bf16a_f32(gate)
        activated = BFloat16.(gate_f32 ./ (1.0f0 .+ exp.(.-gate_f32)))
        BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
    else
        activated = gate ./ (1.0f0 .+ exp.(.-gate))
        activated .* up
    end
    mlp = _bf16a_linear(block.down_weight, hidden)
    return _qwen3_vl_text_residual(x, mlp)
end

function _qwen3_vl_scatter_visual(features, indices, sequence_length::Int)
    if isempty(indices)
        output = similar(features, size(features, 1), sequence_length)
        fill!(output, zero(eltype(output)))
        return output
    end
    to_device = _bf16a_device_mover(features)
    device_indices = to_device(Int32.(indices))
    return NNlib.scatter(
        +,
        features,
        device_indices;
        init=zero(eltype(features)),
        dstsize=(size(features, 1), sequence_length),
    )
end

function _qwen3_vl_replace_visual_embeddings(x, features, visual_mask)
    sequence_length, batch_size = size(visual_mask)
    batch_size == 1 || throw(ArgumentError(
        "Chapter 44 Qwen3-VL decoder prefill supports batch size one",
    ))
    indices = findall(view(visual_mask, :, 1))
    size(features, 2) == length(indices) || throw(DimensionMismatch(
        "Qwen3-VL main visual feature count does not match image placeholders",
    ))
    size(features, 1) == size(x, 1) || throw(DimensionMismatch(
        "Qwen3-VL visual feature width does not match text hidden size",
    ))
    eltype(features) == eltype(x) || throw(ArgumentError(
        "Qwen3-VL vision and text parameter dtypes must match",
    ))
    scattered = _qwen3_vl_scatter_visual(features, indices, sequence_length)
    to_device = _bf16a_device_mover(x)
    keep = to_device(reshape(eltype(x).(.!visual_mask), 1, sequence_length, 1))
    return x .* keep .+ reshape(scattered, size(x, 1), sequence_length, 1)
end

"""Validate only the vision features the text tower will actually consume."""
function _validate_qwen3_vl_text_feature_residency(
    parameters,
    vision_features,
)
    embedding = parameters.embedding
    expected_dtype = eltype(embedding)
    visual_embeddings = vision_features.visual_embeddings
    eltype(visual_embeddings) == expected_dtype || throw(ArgumentError(
        "Qwen3-VL visual_embeddings dtype must match text parameter embedding",
    ))

    consumed_deepstack = min(3, parameters.spec.num_hidden_layers)
    for index in 1:consumed_deepstack
        eltype(vision_features.deepstack[index]) == expected_dtype ||
            throw(ArgumentError(
                "Qwen3-VL deepstack[$index] dtype must match text parameter embedding",
            ))
    end

    expected_device = get_device(embedding)
    get_device(visual_embeddings) == expected_device || throw(ArgumentError(
        "Qwen3-VL visual_embeddings device must match text parameter embedding",
    ))
    for index in 1:consumed_deepstack
        get_device(vision_features.deepstack[index]) == expected_device ||
            throw(ArgumentError(
                "Qwen3-VL deepstack[$index] device must match text parameter embedding",
            ))
    end
    return nothing
end

function _qwen3_vl_add_deepstack(x, features, visual_mask)
    sequence_length, batch_size = size(visual_mask)
    batch_size == 1 || throw(ArgumentError(
        "Chapter 44 Qwen3-VL decoder prefill supports batch size one",
    ))
    indices = findall(view(visual_mask, :, 1))
    size(features) == (size(x, 1), length(indices)) || throw(DimensionMismatch(
        "Qwen3-VL DeepStack feature shape does not match visual positions",
    ))
    eltype(features) == eltype(x) || throw(ArgumentError(
        "Qwen3-VL DeepStack and text parameter dtypes must match",
    ))
    scattered = reshape(
        _qwen3_vl_scatter_visual(features, indices, sequence_length),
        size(x, 1), sequence_length, 1,
    )
    return _qwen3_vl_text_residual(x, scattered)
end

function _qwen3_vl_causal_mask(reference, attention_mask::AbstractMatrix{Bool})
    sequence_length, batch_size = size(attention_mask)
    batch_size == 1 || throw(ArgumentError(
        "Chapter 44 Qwen3-VL decoder prefill supports batch size one",
    ))
    values = zeros(Float32, sequence_length, sequence_length)
    minimum_value = eltype(reference) === BFloat16 ?
        Float32(_BF16_MASK_MIN) : -floatmax(Float32)
    @inbounds for query in 1:sequence_length, key in 1:sequence_length
        if key > query || !attention_mask[key, 1]
            values[query, key] = minimum_value
        end
    end
    to_device = _bf16a_device_mover(reference)
    return eltype(reference) === BFloat16 ?
        to_device(BFloat16.(values)) : to_device(values)
end

function _qwen3_vl_project_tied(embedding, hidden)
    hidden_size, token_count, batch_size = size(hidden)
    matrix = transpose(embedding) * reshape(hidden, hidden_size, :)
    return reshape(matrix, size(embedding, 2), token_count, batch_size)
end

function _qwen3_vl_prefill_options(
    logits_to_keep;
    max_prefill_tokens=nothing,
)
    keep = _strict_host_int(logits_to_keep, "logits_to_keep")
    keep >= 0 || throw(ArgumentError(
        "logits_to_keep must be between zero and the prefill length",
    ))
    limit = if max_prefill_tokens === nothing
        nothing
    else
        requested = _strict_host_int(
            max_prefill_tokens,
            "max_prefill_tokens",
        )
        requested > 0 || throw(ArgumentError(
            "max_prefill_tokens must be positive",
        ))
        requested
    end
    return (; logits_to_keep=keep, max_prefill_tokens=limit)
end

function _qwen3_vl_prompt_rope_deltas(
    spec,
    rope_layout::Qwen3VLRopeLayout,
    sequence_length::Int,
    batch_size::Int,
)
    position_ids = rope_layout.position_ids
    size(position_ids) == (3, sequence_length, batch_size) ||
        throw(DimensionMismatch(
            "Qwen3-VL rope layout does not match input_ids",
        ))
    normalized_positions = _strict_host_int_array(
        position_ids,
        "Qwen3-VL position_ids",
    )
    all(position -> 0 <= position < spec.max_position_embeddings,
        normalized_positions) || throw(ArgumentError(
        "Qwen3-VL prompt mRoPE coordinates are outside the decoder context",
    ))

    attention_mask = rope_layout.attention_mask
    size(attention_mask) == (sequence_length, batch_size) ||
        throw(DimensionMismatch(
            "Qwen3-VL attention mask does not match input_ids",
        ))
    eltype(attention_mask) <: Bool || throw(ArgumentError(
        "Qwen3-VL attention_mask must contain Bool values",
    ))

    rope_deltas = rope_layout.rope_deltas
    size(rope_deltas) == (batch_size, 1) || throw(DimensionMismatch(
        "Qwen3-VL rope_deltas must have shape (batch, 1)",
    ))
    normalized_deltas = _strict_host_int_array(
        rope_deltas,
        "Qwen3-VL rope_delta",
    )
    for batch in 1:batch_size
        valid_positions = findall(view(attention_mask, :, batch))
        isempty(valid_positions) && throw(ArgumentError(
            "every Qwen3-VL batch item must contain a valid token",
        ))
        maximum_coordinate = maximum(view(
            normalized_positions,
            :,
            valid_positions,
            batch,
        ))
        expected_delta = _strict_host_int(
            BigInt(maximum_coordinate) + 1 - BigInt(sequence_length),
            "Qwen3-VL expected rope_delta",
        )
        normalized_deltas[batch, 1] == expected_delta ||
            throw(ArgumentError(
                "Qwen3-VL rope_delta is inconsistent with the prompt " *
                "mRoPE positions",
            ))
    end
    return vec(normalized_deltas)
end

function _qwen3_vl_prompt_visual_token_contract(
    parameters,
    tokens,
    visual_mask,
    attention_mask,
)
    hasproperty(parameters, :checkpoint) || return nothing
    size(visual_mask) == size(tokens) || throw(DimensionMismatch(
        "Qwen3-VL visual mask does not match input_ids",
    ))
    size(attention_mask) == size(tokens) || throw(DimensionMismatch(
        "Qwen3-VL attention mask does not match input_ids",
    ))
    checkpoint = parameters.checkpoint
    hasproperty(checkpoint, :image_token_id) || throw(ArgumentError(
        "Qwen3-VL checkpoint must contain image_token_id",
    ))
    raw_image_token = _strict_host_int(
        checkpoint.image_token_id,
        "Qwen3-VL checkpoint image_token_id",
    )
    vocab_size = parameters.spec.vocab_size
    0 <= raw_image_token < vocab_size || throw(ArgumentError(
        "Qwen3-VL checkpoint image_token_id must be in 0:$(vocab_size - 1)",
    ))
    image_token = _strict_host_int(
        BigInt(raw_image_token) + 1,
        "Qwen3-VL one-based image token id",
    )
    hasproperty(checkpoint, :video_token_id) || throw(ArgumentError(
        "Qwen3-VL checkpoint must contain video_token_id",
    ))
    raw_video_token = _strict_host_int(
        checkpoint.video_token_id,
        "Qwen3-VL checkpoint video_token_id",
    )
    0 <= raw_video_token < vocab_size || throw(ArgumentError(
        "Qwen3-VL checkpoint video_token_id must be in 0:$(vocab_size - 1)",
    ))
    video_token = _strict_host_int(
        BigInt(raw_video_token) + 1,
        "Qwen3-VL one-based video token id",
    )
    any((tokens .== video_token) .& attention_mask) && throw(ArgumentError(
        "Qwen3-VL video placeholders are not supported by image-only text prefill",
    ))
    expected_visual_mask = tokens .== image_token
    all((.!expected_visual_mask) .| visual_mask) || throw(ArgumentError(
        "Qwen3-VL checkpoint image tokens must be marked by visual_mask",
    ))
    all((.!visual_mask) .| expected_visual_mask) || throw(ArgumentError(
        "Qwen3-VL visual_mask must not mark non-image input tokens",
    ))
    return nothing
end

function _qwen3_vl_cache_free_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    max_prefill_tokens::Int,
)
    spec = parameters.spec
    sequence_length, batch_size = size(tokens)
    max_prefill_tokens > 0 || throw(ArgumentError(
        "max_prefill_tokens must be positive",
    ))
    effective_limit = min(max_prefill_tokens, spec.max_position_embeddings)
    0 < sequence_length <= effective_limit || throw(ArgumentError(
        "Qwen3-VL prefill length must be in 1:$effective_limit",
    ))

    visual_mask = rope_layout.visual_mask
    size(visual_mask) == size(tokens) || throw(DimensionMismatch(
        "Qwen3-VL visual mask does not match input_ids",
    ))
    eltype(visual_mask) <: Bool || throw(ArgumentError(
        "Qwen3-VL visual_mask must contain Bool values",
    ))
    _qwen3_vl_prompt_visual_token_contract(
        parameters,
        tokens,
        visual_mask,
        rope_layout.attention_mask,
    )

    _qwen3_vl_prompt_rope_deltas(
        spec,
        rope_layout,
        sequence_length,
        batch_size,
    )
    attention_mask = rope_layout.attention_mask
    all((.!visual_mask) .| attention_mask) || throw(ArgumentError(
        "Qwen3-VL visual positions must be valid attention positions",
    ))
    return nothing
end

"""
    hf_qwen3_vl_text_prefill(parameters, input_ids, rope_layout;
                             vision_features=nothing, logits_to_keep=1,
                             capture_layers=(), max_prefill_tokens=2048)

Run a cache-free Qwen3-VL decoder prefill.  Main vision embeddings replace
image-token embeddings before layer 0; DeepStack features are added after
decoder layers 0, 1, and 2.  `capture_layers` uses official zero-based layer
numbers and stores both the raw block output and the post-DeepStack layer
output. Cached dynamic and bounded-static generation use their dedicated
prefill entry points.
"""
function hf_qwen3_vl_text_prefill(
    parameters,
    input_ids,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    logits_to_keep::Integer=1,
    capture_layers=(),
    capture_input_embeddings::Bool=false,
    max_prefill_tokens::Integer=2_048,
)
    options = _qwen3_vl_prefill_options(
        logits_to_keep;
        max_prefill_tokens,
    )
    requested = Set(_strict_host_int_array(
        capture_layers,
        "Qwen3-VL capture layer",
    ))
    spec = parameters.spec
    all(layer -> 0 <= layer < spec.num_hidden_layers, requested) ||
        throw(ArgumentError("Qwen3-VL capture layer is outside the decoder"))
    tokens = _qwen3_vl_token_matrix(input_ids)
    sequence_length, batch_size = size(tokens)
    batch_size == 1 || throw(ArgumentError(
        "Chapter 44 Qwen3-VL decoder prefill supports batch size one",
    ))
    length(parameters.blocks) == spec.num_hidden_layers || throw(DimensionMismatch(
        "Qwen3-VL decoder parameter layer count is invalid",
    ))
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL input_ids contain an out-of-vocabulary id",
    ))
    _qwen3_vl_cache_free_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        options.max_prefill_tokens,
    )
    options.logits_to_keep <= sequence_length || throw(ArgumentError(
        "logits_to_keep must be between zero and the prefill length",
    ))
    if vision_features === nothing
        any(rope_layout.visual_mask) && throw(ArgumentError(
            "Qwen3-VL visual placeholders require vision features",
        ))
    else
        length(vision_features.deepstack) == 3 || throw(DimensionMismatch(
            "Qwen3-VL prefill requires exactly three DeepStack features",
        ))
        _validate_qwen3_vl_text_feature_residency(
            parameters,
            vision_features,
        )
    end
    x = reshape(
        gather(parameters.embedding, tokens),
        spec.hidden_size, sequence_length, batch_size,
    )
    if vision_features !== nothing
        x = _qwen3_vl_replace_visual_embeddings(
            x,
            vision_features.visual_embeddings,
            rope_layout.visual_mask,
        )
    end
    captured_input = capture_input_embeddings ? x : nothing
    cos_values, sin_values = _qwen3_vl_text_mrope(
        spec,
        parameters.embedding,
        rope_layout.position_ids,
    )
    mask = _qwen3_vl_causal_mask(
        parameters.embedding,
        Bool.(rope_layout.attention_mask),
    )
    block_outputs = Dict{Int,Any}()
    layer_outputs = Dict{Int,Any}()
    for julia_layer in 1:spec.num_hidden_layers
        layer = julia_layer - 1
        x = _qwen3_vl_text_block(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            mask,
        )
        layer in requested && (block_outputs[layer] = x)
        if vision_features !== nothing && layer < 3
            x = _qwen3_vl_add_deepstack(
                x,
                vision_features.deepstack[layer + 1],
                rope_layout.visual_mask,
            )
        end
        layer in requested && (layer_outputs[layer] = x)
    end
    final_hidden = _qwen3_vl_text_rmsnorm(
        x,
        parameters.final_norm,
        spec.rms_norm_eps,
    )
    keep = options.logits_to_keep
    projection = keep == 0 ? final_hidden :
        final_hidden[:, (sequence_length - keep + 1):end, :]
    logits = _qwen3_vl_project_tied(parameters.embedding, projection)
    return Qwen3VLTextPrefill(
        captured_input,
        block_outputs,
        layer_outputs,
        final_hidden,
        logits,
        rope_layout,
    )
end

"""Run the standalone vision tower followed by cache-free decoder prefill."""
function hf_qwen3_vl_prefill(
    vision_parameters,
    text_parameters,
    vision_input::Qwen3VLVisionInput,
    input_ids;
    rope_layout=nothing,
    logits_to_keep::Integer=1,
    max_prefill_tokens::Integer=2_048,
    capture_layers=(),
    kwargs...,
)
    options = _qwen3_vl_prefill_options(
        logits_to_keep;
        max_prefill_tokens,
    )
    requested = Set(_strict_host_int_array(
        capture_layers,
        "Qwen3-VL capture layer",
    ))
    text_spec = text_parameters.spec
    all(layer -> 0 <= layer < text_spec.num_hidden_layers, requested) ||
        throw(ArgumentError("Qwen3-VL capture layer is outside the decoder"))
    resolved_rope_layout = rope_layout === nothing ?
        qwen3_vl_rope_layout(
            input_ids,
            vision_input.grid_thw;
            checkpoint=text_parameters.checkpoint,
        ) : rope_layout
    if hasproperty(text_parameters, :checkpoint)
        tokens = _qwen3_vl_token_matrix(input_ids)
        _qwen3_vl_prompt_visual_token_contract(
            text_parameters,
            tokens,
            resolved_rope_layout.visual_mask,
            resolved_rope_layout.attention_mask,
        )
    end
    features = hf_qwen3_vl_vision_forward(vision_parameters, vision_input)
    text = hf_qwen3_vl_text_prefill(
        text_parameters,
        input_ids,
        resolved_rope_layout;
        vision_features=features,
        logits_to_keep=options.logits_to_keep,
        max_prefill_tokens=options.max_prefill_tokens,
        capture_layers=requested,
        kwargs...,
    )
    return (; vision=features, text)
end
