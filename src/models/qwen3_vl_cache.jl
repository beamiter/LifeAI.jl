"""
    Qwen3VLKVCache(layers, position, rope_delta, batch_size)

Per-request dynamic KV cache for the Qwen3-VL text decoder.  Every layer uses
`LayerKVCache`, whose key and value tensors have layout
`(head_dim, num_kv_heads, cached_tokens, batch)`.  Keys are stored after
QK-Norm and multimodal RoPE; values are stored directly after projection.  GQA
expansion is deliberately confined to attention and is never persisted.

`position` is the physical number of tokens already processed.  `rope_delta`
is the request-local offset produced by the multimodal prompt, so the next
decode token uses the zero-based T/H/W coordinate
`position + rope_delta`.
"""
struct Qwen3VLKVCache{C}
    layers::C
    position::Int
    rope_delta::Int
    batch_size::Int
end

Base.length(cache::Qwen3VLKVCache) = cache.position
Base.isempty(cache::Qwen3VLKVCache) = cache.position == 0

function _qwen3_vl_cache_spec(parameters)
    hasproperty(parameters, :spec) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain a decoder spec",
    ))
    hasproperty(parameters, :blocks) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain decoder blocks",
    ))
    hasproperty(parameters, :embedding) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain token embeddings",
    ))
    hasproperty(parameters, :final_norm) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain a final norm",
    ))
    return parameters.spec
end

"""
    init_qwen3_vl_kv_cache(text_parameters; batch_size=1)

Create an empty dynamic cache for one Qwen3-VL generation request.  Layer
storage is adopted lazily from the first cached prefill, preserving the text
parameters' dtype and CPU/CUDA residency.  Chapter 45 supports batch size one.
"""
function init_qwen3_vl_kv_cache(text_parameters; batch_size::Int=1)
    batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL dynamic generation currently supports batch size one",
    ))
    spec = _qwen3_vl_cache_spec(text_parameters)
    length(text_parameters.blocks) == spec.num_hidden_layers ||
        throw(DimensionMismatch(
            "Qwen3-VL decoder parameter layer count is invalid",
        ))
    layers = ntuple(_ -> LayerKVCache(), spec.num_hidden_layers)
    return Qwen3VLKVCache(layers, 0, 0, batch_size)
end

function _validate_qwen3_vl_kv_cache(parameters, cache::Qwen3VLKVCache)
    spec = _qwen3_vl_cache_spec(parameters)
    cache.batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL dynamic generation currently supports batch size one",
    ))
    length(cache.layers) == spec.num_hidden_layers || throw(DimensionMismatch(
        "Qwen3-VL KV cache layer count does not match the decoder",
    ))
    0 <= cache.position <= spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL KV cache position is outside the decoder context",
    ))
    cache.position == 0 && cache.rope_delta != 0 && throw(ArgumentError(
        "an empty Qwen3-VL KV cache must have rope_delta == 0",
    ))

    expected_shape = (
        spec.head_dim,
        spec.num_key_value_heads,
        cache.position,
        cache.batch_size,
    )
    parameter_dtype = eltype(parameters.embedding)
    for (layer, layer_cache) in enumerate(cache.layers)
        (layer_cache.keys === nothing) == (layer_cache.values === nothing) ||
            throw(ArgumentError(
                "Qwen3-VL layer $layer cache must contain both keys and values",
            ))
        if cache.position == 0
            layer_cache.keys === nothing || throw(DimensionMismatch(
                "empty Qwen3-VL cache contains layer storage",
            ))
            continue
        end
        layer_cache.keys === nothing && throw(DimensionMismatch(
            "populated Qwen3-VL cache is missing layer $layer storage",
        ))
        size(layer_cache.keys) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL layer $layer key cache has the wrong shape",
        ))
        size(layer_cache.values) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL layer $layer value cache has the wrong shape",
        ))
        eltype(layer_cache.keys) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL cached keys do not match the text parameter dtype",
        ))
        eltype(layer_cache.values) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL cached values do not match the text parameter dtype",
        ))
    end
    return nothing
end

function _qwen3_vl_cache_block(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    mask,
    layer_cache::LayerKVCache,
)
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
        queries,
        block.q_norm,
        spec.rms_norm_eps,
    )
    keys = _qwen3_vl_text_rmsnorm(keys, block.k_norm, spec.rms_norm_eps)
    queries = _qwen3_vl_text_apply_rope(queries, cos_values, sin_values)
    keys = _qwen3_vl_text_apply_rope(keys, cos_values, sin_values)

    updated_layer_cache = _append_kv(layer_cache, keys, values)
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    context = if eltype(x) === BFloat16
        _bf16a_attention(
            queries,
            updated_layer_cache.keys,
            updated_layer_cache.values;
            scaling,
            mask,
        )
    else
        _qwen3_vl_text_f32_attention(
            queries,
            updated_layer_cache.keys,
            updated_layer_cache.values,
            mask;
            scaling,
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
    return _qwen3_vl_text_residual(x, mlp), updated_layer_cache
end

function _qwen3_vl_cached_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    cache::Qwen3VLKVCache,
    logits_to_keep::Int,
)
    spec = _qwen3_vl_cache_spec(parameters)
    sequence_length, batch_size = size(tokens)
    _validate_qwen3_vl_kv_cache(parameters, cache)
    isempty(cache) || throw(ArgumentError(
        "Qwen3-VL cached prefill requires an empty cache",
    ))
    batch_size == cache.batch_size || throw(DimensionMismatch(
        "Qwen3-VL prompt batch size does not match the cache",
    ))
    0 < sequence_length <= spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL prompt length is outside the decoder context",
    ))
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL input_ids contain an out-of-vocabulary id",
    ))
    size(rope_layout.position_ids) == (3, sequence_length, batch_size) ||
        throw(DimensionMismatch(
            "Qwen3-VL rope layout does not match input_ids",
        ))
    size(rope_layout.visual_mask) == size(tokens) || throw(DimensionMismatch(
        "Qwen3-VL visual mask does not match input_ids",
    ))
    size(rope_layout.attention_mask) == size(tokens) || throw(DimensionMismatch(
        "Qwen3-VL attention mask does not match input_ids",
    ))
    all(rope_layout.attention_mask) || throw(ArgumentError(
        "Qwen3-VL cached generation currently requires an all-ones attention mask",
    ))
    size(rope_layout.rope_deltas) == (batch_size, 1) ||
        throw(DimensionMismatch(
            "Qwen3-VL rope_deltas must have shape (batch, 1)",
        ))
    all(position -> 0 <= position < spec.max_position_embeddings,
        rope_layout.position_ids) || throw(ArgumentError(
        "Qwen3-VL prompt mRoPE coordinates are outside the decoder context",
    ))
    rope_delta = Int(only(rope_layout.rope_deltas))
    expected_rope_delta = maximum(rope_layout.position_ids) + 1 - sequence_length
    rope_delta == expected_rope_delta || throw(ArgumentError(
        "Qwen3-VL rope_delta is inconsistent with the prompt mRoPE positions",
    ))
    0 <= logits_to_keep <= sequence_length || throw(ArgumentError(
        "logits_to_keep must be between zero and the prefill length",
    ))
    return spec, rope_delta
end

"""
    hf_qwen3_vl_text_prefill_cached(parameters, input_ids, rope_layout;
                                    vision_features=nothing, cache,
                                    logits_to_keep=1)

Run one batch-1, all-ones-mask Qwen3-VL prompt and populate a dynamic KV
cache.  Main visual embeddings replace image-token embeddings before layer 0,
and the three DeepStack tensors are added after decoder layers 0, 1, and 2.

Returns `(prefill_result, updated_cache)`.  The result uses the same
`Qwen3VLTextPrefill` container as cache-free prefill; cached prefill does not
capture intermediate layers, so its capture dictionaries are empty.
"""
function hf_qwen3_vl_text_prefill_cached(
    parameters,
    input_ids,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    cache::Qwen3VLKVCache,
    logits_to_keep::Int=1,
)
    tokens = _qwen3_vl_token_matrix(input_ids)
    spec, rope_delta = _qwen3_vl_cached_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        cache,
        logits_to_keep,
    )
    sequence_length, batch_size = size(tokens)

    x = reshape(
        gather(parameters.embedding, tokens),
        spec.hidden_size,
        sequence_length,
        batch_size,
    )
    if vision_features === nothing
        any(rope_layout.visual_mask) && throw(ArgumentError(
            "Qwen3-VL visual placeholders require vision features",
        ))
    else
        length(vision_features.deepstack) == 3 || throw(DimensionMismatch(
            "Qwen3-VL cached prefill requires exactly three DeepStack features",
        ))
        x = _qwen3_vl_replace_visual_embeddings(
            x,
            vision_features.visual_embeddings,
            rope_layout.visual_mask,
        )
    end
    input_embeddings = x
    cos_values, sin_values = _qwen3_vl_text_mrope(
        spec,
        parameters.embedding,
        rope_layout.position_ids,
    )
    mask = _qwen3_vl_causal_mask(
        parameters.embedding,
        Bool.(rope_layout.attention_mask),
    )

    layer_caches = Vector{Any}(undef, spec.num_hidden_layers)
    for julia_layer in 1:spec.num_hidden_layers
        layer = julia_layer - 1
        x, layer_caches[julia_layer] = _qwen3_vl_cache_block(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            mask,
            cache.layers[julia_layer],
        )
        if vision_features !== nothing && layer < 3
            x = _qwen3_vl_add_deepstack(
                x,
                vision_features.deepstack[layer + 1],
                rope_layout.visual_mask,
            )
        end
    end

    final_hidden = _qwen3_vl_text_rmsnorm(
        x,
        parameters.final_norm,
        spec.rms_norm_eps,
    )
    projection = logits_to_keep == 0 ? final_hidden :
        final_hidden[:, (sequence_length - logits_to_keep + 1):end, :]
    logits = _qwen3_vl_project_tied(parameters.embedding, projection)
    updated_cache = Qwen3VLKVCache(
        Tuple(layer_caches),
        sequence_length,
        rope_delta,
        batch_size,
    )
    _validate_qwen3_vl_kv_cache(parameters, updated_cache)
    result = Qwen3VLTextPrefill(
        input_embeddings,
        Dict{Int,Any}(),
        Dict{Int,Any}(),
        final_hidden,
        logits,
        rope_layout,
    )
    return result, updated_cache
end

function _qwen3_vl_decode_token_matrix(token, batch_size::Int)
    tokens = if token isa Integer && !(token isa Bool)
        fill(Int(token), 1, 1)
    else
        _qwen3_vl_token_matrix(token)
    end
    size(tokens) == (1, batch_size) || throw(DimensionMismatch(
        "Qwen3-VL decode_step expects exactly one token per batch item",
    ))
    return tokens
end

"""
    hf_qwen3_vl_text_decode_step(parameters, token, cache)

Append one token to a populated Qwen3-VL dynamic cache and return
`(logits, updated_cache)`.  The logits shape is `(vocab_size, 1, 1)`.  Decode
uses the request-local coordinate `cache.position + cache.rope_delta` on all
three mRoPE axes and never reruns vision replacement or DeepStack injection.
"""
function hf_qwen3_vl_text_decode_step(
    parameters,
    token,
    cache::Qwen3VLKVCache,
)
    spec = _qwen3_vl_cache_spec(parameters)
    _validate_qwen3_vl_kv_cache(parameters, cache)
    isempty(cache) && throw(ArgumentError(
        "call hf_qwen3_vl_text_prefill_cached before decode_step",
    ))
    cache.position < spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL KV cache has reached the decoder context limit",
    ))
    coordinate = cache.position + cache.rope_delta
    0 <= coordinate < spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL decode mRoPE coordinate is outside the decoder context",
    ))
    tokens = _qwen3_vl_decode_token_matrix(token, cache.batch_size)
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL decode token is outside the vocabulary",
    ))

    x = reshape(
        gather(parameters.embedding, tokens),
        spec.hidden_size,
        1,
        cache.batch_size,
    )
    position_ids = fill(coordinate, 3, 1, cache.batch_size)
    cos_values, sin_values = _qwen3_vl_text_mrope(
        spec,
        parameters.embedding,
        position_ids,
    )
    layer_caches = Vector{Any}(undef, spec.num_hidden_layers)
    for julia_layer in 1:spec.num_hidden_layers
        x, layer_caches[julia_layer] = _qwen3_vl_cache_block(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            nothing,
            cache.layers[julia_layer],
        )
    end
    final_hidden = _qwen3_vl_text_rmsnorm(
        x,
        parameters.final_norm,
        spec.rms_norm_eps,
    )
    logits = _qwen3_vl_project_tied(parameters.embedding, final_hidden)
    updated_cache = Qwen3VLKVCache(
        Tuple(layer_caches),
        cache.position + 1,
        cache.rope_delta,
        cache.batch_size,
    )
    _validate_qwen3_vl_kv_cache(parameters, updated_cache)
    return logits, updated_cache
end
