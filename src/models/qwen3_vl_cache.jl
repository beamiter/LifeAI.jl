using MLDataDevices: get_device

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

"""
    Qwen3VLStaticLayerKVCache(keys, values)

Fixed-capacity key/value storage for one Qwen3-VL decoder layer. Both tensors
use `(head_dim, num_kv_heads, capacity, batch)` layout. Only the prefix ending
at `Qwen3VLStaticKVCache.position` is logically valid.
"""
struct Qwen3VLStaticLayerKVCache{K,V}
    keys::K
    values::V
end

"""
    Qwen3VLStaticKVCache(layers, position, rope_delta, batch_size, capacity)

Mutable request state backed by fixed-capacity per-layer K/V tensors.
`position` remains the physical number of processed tokens and `rope_delta`
remains the request-local multimodal RoPE offset. Prefill, decode, and reset
mutate this request object in place without replacing its layer storage.
"""
mutable struct Qwen3VLStaticKVCache{C}
    layers::C
    position::Int
    rope_delta::Int
    batch_size::Int
    capacity::Int
end

Base.length(cache::Qwen3VLStaticKVCache) = cache.position
Base.isempty(cache::Qwen3VLStaticKVCache) = cache.position == 0

"""
    init_qwen3_vl_static_kv_cache(text_parameters; capacity, batch_size=1)

Allocate bounded Qwen3-VL K/V storage on the same device and with the same
element type as the text embedding. `capacity` is required deliberately: the
official context limit is large enough that silently allocating the maximum
would consume excessive host or accelerator memory.
"""
function init_qwen3_vl_static_kv_cache(
    text_parameters;
    capacity::Int,
    batch_size::Int=1,
)
    batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL static generation currently supports batch size one",
    ))
    spec = _qwen3_vl_cache_spec(text_parameters)
    length(text_parameters.blocks) == spec.num_hidden_layers ||
        throw(DimensionMismatch(
            "Qwen3-VL decoder parameter layer count is invalid",
        ))
    0 < capacity <= spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL static cache capacity must be in 1:max_position_embeddings",
    ))
    dtype = eltype(text_parameters.embedding)
    dtype in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL static cache supports Float32 or BFloat16 parameters",
    ))

    shape = (
        spec.head_dim,
        spec.num_key_value_heads,
        capacity,
        batch_size,
    )
    layers = ntuple(spec.num_hidden_layers) do _
        keys = similar(text_parameters.embedding, dtype, shape)
        values = similar(text_parameters.embedding, dtype, shape)
        fill!(keys, zero(dtype))
        fill!(values, zero(dtype))
        Qwen3VLStaticLayerKVCache(keys, values)
    end
    cache = Qwen3VLStaticKVCache(layers, 0, 0, batch_size, capacity)
    _validate_qwen3_vl_static_kv_cache(text_parameters, cache)
    return cache
end

function _validate_qwen3_vl_static_kv_cache(
    parameters,
    cache::Qwen3VLStaticKVCache,
)
    spec = _qwen3_vl_cache_spec(parameters)
    cache.batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL static generation currently supports batch size one",
    ))
    length(cache.layers) == spec.num_hidden_layers || throw(DimensionMismatch(
        "Qwen3-VL static KV cache layer count does not match the decoder",
    ))
    0 < cache.capacity <= spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL static cache capacity is outside the decoder context",
    ))
    0 <= cache.position <= cache.capacity || throw(ArgumentError(
        "Qwen3-VL static cache position is outside its capacity",
    ))
    cache.position == 0 && cache.rope_delta != 0 && throw(ArgumentError(
        "an empty Qwen3-VL static cache must have rope_delta == 0",
    ))

    expected_shape = (
        spec.head_dim,
        spec.num_key_value_heads,
        cache.capacity,
        cache.batch_size,
    )
    parameter_dtype = eltype(parameters.embedding)
    parameter_dtype in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL static cache supports Float32 or BFloat16 parameters",
    ))
    parameter_device = get_device(parameters.embedding)
    for (layer, layer_cache) in enumerate(cache.layers)
        size(layer_cache.keys) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL static layer $layer key cache has the wrong shape",
        ))
        size(layer_cache.values) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL static layer $layer value cache has the wrong shape",
        ))
        eltype(layer_cache.keys) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL static cached keys do not match the text parameter dtype",
        ))
        eltype(layer_cache.values) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL static cached values do not match the text parameter dtype",
        ))
        get_device(layer_cache.keys) == parameter_device || throw(ArgumentError(
            "Qwen3-VL static cached keys do not match the text parameter device",
        ))
        get_device(layer_cache.values) == parameter_device || throw(ArgumentError(
            "Qwen3-VL static cached values do not match the text parameter device",
        ))
    end
    return nothing
end

"""
    reset_qwen3_vl_static_kv_cache!(cache; clear=false)

Reset a bounded cache for another request while retaining all allocations.
With `clear=false`, stale tail bytes remain physically present but are never
visible to attention because `position` returns to zero. `clear=true` also
zeroes every K/V buffer.
"""
function reset_qwen3_vl_static_kv_cache!(
    cache::Qwen3VLStaticKVCache;
    clear::Bool=false,
)
    if clear
        for layer_cache in cache.layers
            fill!(layer_cache.keys, zero(eltype(layer_cache.keys)))
            fill!(layer_cache.values, zero(eltype(layer_cache.values)))
        end
    end
    cache.position = 0
    cache.rope_delta = 0
    return cache
end

@inline function _qwen3_vl_profile_stage(
    thunk,
    runner,
    stage::Symbol,
    layer_index::Int,
)
    runner === nothing && return thunk()
    return runner(stage, layer_index, thunk)
end

function _qwen3_vl_cache_block_core(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    mask,
    store,
    profile_runner=nothing,
    layer_index::Int=0,
)
    head_dim = spec.head_dim
    sequence_length, batch_size = size(x, 2), size(x, 3)
    normed = _qwen3_vl_profile_stage(
        profile_runner,
        :pre_attention_norm,
        layer_index,
    ) do
        _qwen3_vl_text_rmsnorm(x, block.norm1, spec.rms_norm_eps)
    end
    queries, keys, values = _qwen3_vl_profile_stage(
        profile_runner,
        :qkv_projection,
        layer_index,
    ) do
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
        (queries, keys, values)
    end
    queries, keys = _qwen3_vl_profile_stage(
        profile_runner,
        :qk_norm,
        layer_index,
    ) do
        queries = _qwen3_vl_text_rmsnorm(
            queries,
            block.q_norm,
            spec.rms_norm_eps,
        )
        keys = _qwen3_vl_text_rmsnorm(
            keys,
            block.k_norm,
            spec.rms_norm_eps,
        )
        (queries, keys)
    end
    queries, keys = _qwen3_vl_profile_stage(
        profile_runner,
        :qk_rope,
        layer_index,
    ) do
        (
            _qwen3_vl_text_apply_rope(queries, cos_values, sin_values),
            _qwen3_vl_text_apply_rope(keys, cos_values, sin_values),
        )
    end

    attention_keys, attention_values, updated_layer_cache =
        _qwen3_vl_profile_stage(
            profile_runner,
            :kv_write,
            layer_index,
        ) do
            store(keys, values)
        end
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    context = _qwen3_vl_profile_stage(
        profile_runner,
        :attention,
        layer_index,
    ) do
        if eltype(x) === BFloat16
            _bf16a_attention(
                queries,
                attention_keys,
                attention_values;
                scaling,
                mask,
            )
        else
            _qwen3_vl_text_f32_attention(
                queries,
                attention_keys,
                attention_values,
                mask;
                scaling,
            )
        end
    end
    x = _qwen3_vl_profile_stage(
        profile_runner,
        :attention_output_projection_residual,
        layer_index,
    ) do
        attention = _bf16a_linear(
            block.o_weight,
            reshape(context, spec.hidden_size, sequence_length, batch_size),
        )
        _qwen3_vl_text_residual(x, attention)
    end
    normed = _qwen3_vl_profile_stage(
        profile_runner,
        :post_attention_norm,
        layer_index,
    ) do
        _qwen3_vl_text_rmsnorm(x, block.norm2, spec.rms_norm_eps)
    end
    gate, up = _qwen3_vl_profile_stage(
        profile_runner,
        :mlp_gate_up_projection,
        layer_index,
    ) do
        (
            _bf16a_linear(block.gate_weight, normed),
            _bf16a_linear(block.up_weight, normed),
        )
    end
    hidden = _qwen3_vl_profile_stage(
        profile_runner,
        :mlp_activation,
        layer_index,
    ) do
        if eltype(x) === BFloat16
            gate_f32 = _bf16a_f32(gate)
            activated = BFloat16.(gate_f32 ./ (1.0f0 .+ exp.(.-gate_f32)))
            BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
        else
            activated = gate ./ (1.0f0 .+ exp.(.-gate))
            activated .* up
        end
    end
    x = _qwen3_vl_profile_stage(
        profile_runner,
        :mlp_down_projection_residual,
        layer_index,
    ) do
        mlp = _bf16a_linear(block.down_weight, hidden)
        _qwen3_vl_text_residual(x, mlp)
    end
    return x, updated_layer_cache
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
    store = function (keys, values)
        updated = _append_kv(layer_cache, keys, values)
        return updated.keys, updated.values, updated
    end
    return _qwen3_vl_cache_block_core(
        spec,
        block,
        x,
        cos_values,
        sin_values,
        mask,
        store,
    )
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

function _qwen3_vl_static_cache_block!(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    mask,
    layer_cache::Qwen3VLStaticLayerKVCache,
    write_start::Int,
    profile_runner=nothing,
    layer_index::Int=0,
)
    sequence_length = size(x, 2)
    write_stop = write_start + sequence_length - 1
    1 <= write_start <= write_stop <= size(layer_cache.keys, 3) ||
        throw(ArgumentError(
            "Qwen3-VL static cache write would exceed its capacity",
        ))

    store = function (keys, values)
        expected = (
            spec.head_dim,
            spec.num_key_value_heads,
            sequence_length,
            size(x, 3),
        )
        size(keys) == expected || throw(DimensionMismatch(
            "Qwen3-VL projected keys do not match the static cache layout",
        ))
        size(values) == expected || throw(DimensionMismatch(
            "Qwen3-VL projected values do not match the static cache layout",
        ))
        eltype(keys) == eltype(layer_cache.keys) || throw(ArgumentError(
            "Qwen3-VL projected keys do not match the static cache dtype",
        ))
        eltype(values) == eltype(layer_cache.values) || throw(ArgumentError(
            "Qwen3-VL projected values do not match the static cache dtype",
        ))
        get_device(keys) == get_device(layer_cache.keys) || throw(ArgumentError(
            "Qwen3-VL projected keys do not match the static cache device",
        ))
        get_device(values) == get_device(layer_cache.values) || throw(ArgumentError(
            "Qwen3-VL projected values do not match the static cache device",
        ))

        write_range = write_start:write_stop
        copyto!(view(layer_cache.keys, :, :, write_range, :), keys)
        copyto!(view(layer_cache.values, :, :, write_range, :), values)
        valid_range = 1:write_stop
        return (
            view(layer_cache.keys, :, :, valid_range, :),
            view(layer_cache.values, :, :, valid_range, :),
            layer_cache,
        )
    end
    return _qwen3_vl_cache_block_core(
        spec,
        block,
        x,
        cos_values,
        sin_values,
        mask,
        store,
        profile_runner,
        layer_index,
    )
end

function _qwen3_vl_static_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    cache::Qwen3VLStaticKVCache,
    logits_to_keep::Int,
)
    spec = _qwen3_vl_cache_spec(parameters)
    sequence_length, batch_size = size(tokens)
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    isempty(cache) || throw(ArgumentError(
        "Qwen3-VL static cached prefill requires an empty cache",
    ))
    batch_size == cache.batch_size || throw(DimensionMismatch(
        "Qwen3-VL prompt batch size does not match the static cache",
    ))
    0 < sequence_length <= cache.capacity || throw(ArgumentError(
        "Qwen3-VL prompt length exceeds the static cache capacity",
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
        "Qwen3-VL static generation currently requires an all-ones attention mask",
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
    hf_qwen3_vl_text_prefill_static(parameters, input_ids, rope_layout;
                                    vision_features=nothing, cache,
                                    logits_to_keep=1)

Run one batch-1 Qwen3-VL prompt and write every layer's K/V tensors into the
fixed `1:prompt_length` prefix of `cache`. The returned cache is the same
mutable object passed by the caller, and all layer storage identities remain
unchanged.
"""
function hf_qwen3_vl_text_prefill_static(
    parameters,
    input_ids,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    cache::Qwen3VLStaticKVCache,
    logits_to_keep::Int=1,
)
    tokens = _qwen3_vl_token_matrix(input_ids)
    spec, rope_delta = _qwen3_vl_static_prompt_contract(
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
            "Qwen3-VL static prefill requires exactly three DeepStack features",
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

    for julia_layer in 1:spec.num_hidden_layers
        layer = julia_layer - 1
        x, _ = _qwen3_vl_static_cache_block!(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            mask,
            cache.layers[julia_layer],
            1,
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
    cache.position = sequence_length
    cache.rope_delta = rope_delta
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    result = Qwen3VLTextPrefill(
        input_embeddings,
        Dict{Int,Any}(),
        Dict{Int,Any}(),
        final_hidden,
        logits,
        rope_layout,
    )
    return result, cache
end

function _qwen3_vl_text_decode_step_static_impl(
    parameters,
    token,
    cache::Qwen3VLStaticKVCache,
    profile_runner,
)
    spec = _qwen3_vl_cache_spec(parameters)
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    isempty(cache) && throw(ArgumentError(
        "call hf_qwen3_vl_text_prefill_static before decode_step",
    ))
    cache.position < cache.capacity || throw(ArgumentError(
        "Qwen3-VL static KV cache has reached its capacity",
    ))
    coordinate = cache.position + cache.rope_delta
    0 <= coordinate < spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL decode mRoPE coordinate is outside the decoder context",
    ))
    tokens = _qwen3_vl_decode_token_matrix(token, cache.batch_size)
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL decode token is outside the vocabulary",
    ))

    x = _qwen3_vl_profile_stage(
        profile_runner,
        :token_embedding,
        0,
    ) do
        reshape(
            gather(parameters.embedding, tokens),
            spec.hidden_size,
            1,
            cache.batch_size,
        )
    end
    cos_values, sin_values = _qwen3_vl_profile_stage(
        profile_runner,
        :mrope_prepare,
        0,
    ) do
        position_ids = fill(coordinate, 3, 1, cache.batch_size)
        _qwen3_vl_text_mrope(
            spec,
            parameters.embedding,
            position_ids,
        )
    end
    write_position = cache.position + 1
    for julia_layer in 1:spec.num_hidden_layers
        x, _ = _qwen3_vl_static_cache_block!(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            nothing,
            cache.layers[julia_layer],
            write_position,
            profile_runner,
            julia_layer,
        )
    end
    final_hidden = _qwen3_vl_profile_stage(
        profile_runner,
        :final_norm,
        0,
    ) do
        _qwen3_vl_text_rmsnorm(
            x,
            parameters.final_norm,
            spec.rms_norm_eps,
        )
    end
    logits = _qwen3_vl_profile_stage(
        profile_runner,
        :vocab_logits,
        0,
    ) do
        _qwen3_vl_project_tied(parameters.embedding, final_hidden)
    end
    cache.position = write_position
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    return logits, cache
end

"""
    hf_qwen3_vl_text_decode_step_static(parameters, token, cache)

Write one token in place at physical slot `cache.position + 1`, attend only to
the valid `1:cache.position+1` prefix, and return `(logits, cache)`. The next
mRoPE coordinate remains `cache.position + cache.rope_delta` before the
position increment.
"""
function hf_qwen3_vl_text_decode_step_static(
    parameters,
    token,
    cache::Qwen3VLStaticKVCache,
)
    return _qwen3_vl_text_decode_step_static_impl(
        parameters,
        token,
        cache,
        nothing,
    )
end

"""
    _profile_qwen3_vl_text_decode_step_static(parameters, token, cache, runner)

Run one bounded-static Qwen3-VL decode step while routing each decoder stage
through `runner(stage, layer_index, thunk)`. Request-level stages use layer
index zero; decoder blocks use one-based layer indices. This diagnostic entry
point preserves the public decode path's validation, cache mutation, and
numerical operations. It is intended for allocation attribution, not latency
measurement, because a runner may synchronize between stages.
"""
function _profile_qwen3_vl_text_decode_step_static(
    parameters,
    token,
    cache::Qwen3VLStaticKVCache,
    runner,
)
    runner === nothing && throw(ArgumentError(
        "Qwen3-VL profiling requires a stage runner",
    ))
    return _qwen3_vl_text_decode_step_static_impl(
        parameters,
        token,
        cache,
        runner,
    )
end
