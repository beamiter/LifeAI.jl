using MLDataDevices: cpu_device, get_device
using Random: AbstractRNG, default_rng

"""
    LayerKVCache(keys, values)

Cached key and value tensors for one Transformer layer.

Both tensors use the layout:

    (head_dim, num_kv_heads, cached_tokens, batch)

Under GQA `num_kv_heads < num_heads`, so the cache is proportionally smaller
than the query-head count suggests.

An empty cache stores `nothing` for both tensors. The first prefill call adopts
its element type and device from the projected keys and values.
"""
struct LayerKVCache{K,V}
    keys::K
    values::V
end

LayerKVCache() = LayerKVCache(nothing, nothing)

function Base.length(cache::LayerKVCache)
    cache.keys === nothing && return 0
    return size(cache.keys, 3)
end

Base.isempty(cache::LayerKVCache) = length(cache) == 0

"""
    GPTKVCache(layers, position, batch_size)

Per-request KV cache for a decoder-only GPT model.

`position` is the number of tokens already processed. KV data is deliberately
kept separate from Lux model state because it belongs to one generation
request, not to the model itself.
"""
struct GPTKVCache{C}
    layers::C
    position::Int
    batch_size::Int
end

Base.length(cache::GPTKVCache) = cache.position
Base.isempty(cache::GPTKVCache) = cache.position == 0

"""
    init_kv_cache(model; batch_size=1)

Create an empty cache for `model`. Storage is allocated lazily during `prefill`
so its dtype and device always match the model projections.
"""
function init_kv_cache(model::GPTModel; batch_size::Int=1)
    batch_size > 0 || throw(ArgumentError("`batch_size` must be positive"))
    layers = ntuple(_ -> LayerKVCache(), model.num_layers)
    return GPTKVCache(layers, 0, batch_size)
end

function _validate_kv_cache(model::GPTModel, cache::GPTKVCache)
    length(cache.layers) == model.num_layers ||
        throw(DimensionMismatch("cache layer count does not match model.num_layers"))
    0 <= cache.position <= model.max_seq_len ||
        throw(ArgumentError("cache position is outside 0:model.max_seq_len"))
    cache.batch_size > 0 || throw(ArgumentError("cache batch size must be positive"))

    for layer_cache in cache.layers
        length(layer_cache) == cache.position ||
            throw(DimensionMismatch("all layer caches must match cache.position"))

        if !isempty(layer_cache)
            size(layer_cache.keys) == size(layer_cache.values) ||
                throw(DimensionMismatch("cached keys and values must have matching shapes"))
            size(layer_cache.keys, 4) == cache.batch_size ||
                throw(DimensionMismatch("layer cache batch size does not match cache.batch_size"))
        end
    end

    return nothing
end

function _append_kv(cache::LayerKVCache, keys, values)
    size(keys) == size(values) ||
        throw(DimensionMismatch("new keys and values must have matching shapes"))

    if isempty(cache)
        return LayerKVCache(keys, values)
    end

    size(cache.keys, 1) == size(keys, 1) ||
        throw(DimensionMismatch("cached and new key head dimensions do not match"))
    size(cache.keys, 2) == size(keys, 2) ||
        throw(DimensionMismatch("cached and new key head counts do not match"))
    size(cache.keys, 4) == size(keys, 4) ||
        throw(DimensionMismatch("cached and new key batch sizes do not match"))

    return LayerKVCache(
        cat(cache.keys, keys; dims=3),
        cat(cache.values, values; dims=3),
    )
end

function _attention_with_kv_cache(
    attn::MultiHeadAttention,
    x,
    ps,
    st::NamedTuple,
    cache::LayerKVCache;
    start_pos::Int,
)
    _, num_tokens, batch_size = size(x)
    cached_tokens = length(cache)

    cached_tokens == start_pos - 1 ||
        throw(ArgumentError("`start_pos` must immediately follow the cached sequence"))
    cached_tokens > 0 && num_tokens != 1 &&
        throw(ArgumentError("incremental decoding accepts exactly one new token"))

    queries, st_q_proj = attn.q_proj(x, ps.q_proj, st.q_proj)
    keys, st_k_proj = attn.k_proj(x, ps.k_proj, st.k_proj)
    values, st_v_proj = attn.v_proj(x, ps.v_proj, st.v_proj)

    queries = reshape(queries, attn.head_dim, attn.num_heads, num_tokens, batch_size)
    keys = reshape(keys, attn.head_dim, attn.num_kv_heads, num_tokens, batch_size)
    values = reshape(values, attn.head_dim, attn.num_kv_heads, num_tokens, batch_size)

    if attn.use_qk_norm
        queries = _apply_qk_norm(queries, ps.q_norm.scale, attn.qk_norm_epsilon)
        keys = _apply_qk_norm(keys, ps.k_norm.scale, attn.qk_norm_epsilon)
    end

    if attn.use_rope
        queries = apply_rope(
            queries,
            st.rope_cos_cache,
            st.rope_sin_cache;
            start_pos,
            rope_style=attn.rope_style,
        )
        keys = apply_rope(
            keys,
            st.rope_cos_cache,
            st.rope_sin_cache;
            start_pos,
            rope_style=attn.rope_style,
        )
    end

    # Under GQA the cache stores only num_kv_heads heads per layer.
    new_cache = _append_kv(cache, keys, values)

    # Prefill computes several queries at once and therefore needs the causal
    # mask. During decode, the cache contains only past/current positions, so a
    # one-token query can attend to every cached key without an additional mask.
    use_causal_mask = attn.is_causal && cached_tokens == 0
    context, _ = batched_scaled_dot_product_attention(
        queries,
        new_cache.keys,
        new_cache.values;
        is_causal=use_causal_mask,
    )

    context = reshape(context, attn.d_out, num_tokens, batch_size)
    y, st_o_proj = attn.o_proj(context, ps.o_proj, st.o_proj)

    return (
        y,
        (;
            q_proj=st_q_proj,
            k_proj=st_k_proj,
            v_proj=st_v_proj,
            o_proj=st_o_proj,
            rope_cos_cache=st.rope_cos_cache,
            rope_sin_cache=st.rope_sin_cache,
        ),
        new_cache,
    )
end

function _block_with_kv_cache(
    block::TransformerBlock,
    x,
    ps,
    st::NamedTuple,
    cache::LayerKVCache;
    start_pos::Int,
)
    x_norm1, st_norm1 = block.norm1(x, ps.norm1, st.norm1)
    attn_out, st_attn, new_cache = _attention_with_kv_cache(
        block.attn,
        x_norm1,
        ps.attn,
        st.attn,
        cache;
        start_pos,
    )
    x = x .+ attn_out

    x_norm2, st_norm2 = block.norm2(x, ps.norm2, st.norm2)
    mlp_out, st_mlp = block.mlp(x_norm2, ps.mlp, st.mlp)
    y = x .+ mlp_out

    return (
        y,
        (;
            norm1=st_norm1,
            attn=st_attn,
            norm2=st_norm2,
            mlp=st_mlp,
        ),
        new_cache,
    )
end

function _gpt_with_kv_cache(
    model::GPTModel,
    tokens,
    ps,
    st::NamedTuple,
    cache::GPTKVCache;
    start_pos::Int,
)
    _validate_kv_cache(model, cache)

    ndims(tokens) == 2 ||
        throw(DimensionMismatch("`tokens` must have shape (seq_len, batch)"))
    seq_len, batch_size = size(tokens)
    seq_len > 0 || throw(ArgumentError("`tokens` must contain at least one token"))
    batch_size == cache.batch_size ||
        throw(DimensionMismatch("token batch size does not match cache.batch_size"))
    start_pos == cache.position + 1 ||
        throw(ArgumentError("`start_pos` must equal cache.position + 1"))
    start_pos + seq_len - 1 <= model.max_seq_len ||
        throw(ArgumentError("cached sequence exceeds model.max_seq_len"))

    x, st_token_embedding = model.token_embedding(
        tokens,
        ps.token_embedding,
        st.token_embedding,
    )

    blocks = Tuple(values(model.blocks.layers))
    block_parameters = Tuple(values(ps.blocks))
    block_states = Tuple(values(st.blocks))
    new_block_states = Vector{Any}(undef, model.num_layers)
    new_layer_caches = Vector{Any}(undef, model.num_layers)

    for index in 1:model.num_layers
        x, new_block_states[index], new_layer_caches[index] = _block_with_kv_cache(
            blocks[index],
            x,
            block_parameters[index],
            block_states[index],
            cache.layers[index];
            start_pos,
        )
    end

    st_blocks = NamedTuple{keys(st.blocks)}(Tuple(new_block_states))
    x, st_final_norm = model.final_norm(x, ps.final_norm, st.final_norm)
    logits, st_lm_head = _project_logits(model, x, ps, st.lm_head)

    new_state = (;
        token_embedding=st_token_embedding,
        blocks=st_blocks,
        final_norm=st_final_norm,
        lm_head=st_lm_head,
    )
    new_cache = GPTKVCache(
        Tuple(new_layer_caches),
        cache.position + seq_len,
        cache.batch_size,
    )

    return logits, new_cache, new_state
end

# Small positional wrappers used by the XLA benchmark harness. Dynamic cache
# shapes are intentionally preserved: each cached length produces a distinct
# executable, making recompilation costs visible instead of hiding them behind
# the fixed-shape cache path.
function _dynamic_gpt_prefill_kernel(model, tokens, ps, st, cache)
    return _gpt_with_kv_cache(
        model,
        tokens,
        ps,
        st,
        cache;
        start_pos=1,
    )
end

function _dynamic_gpt_decode_kernel(model, tokens, ps, st, cache)
    return _gpt_with_kv_cache(
        model,
        tokens,
        ps,
        st,
        cache;
        start_pos=cache.position + 1,
    )
end

function _prefill_token_matrix(prompt_tokens)
    if prompt_tokens isa AbstractVector
        return reshape(Int.(collect(prompt_tokens)), :, 1)
    elseif prompt_tokens isa AbstractMatrix
        return Int.(collect(prompt_tokens))
    end

    throw(DimensionMismatch(
        "`prompt_tokens` must be a vector or a (seq_len, batch) matrix",
    ))
end

function _decode_token_matrix(token, batch_size::Int)
    if token isa Integer
        batch_size == 1 ||
            throw(DimensionMismatch("a scalar token is only valid for batch_size=1"))
        return reshape([Int(token)], 1, 1)
    elseif token isa AbstractVector
        length(token) == batch_size ||
            throw(DimensionMismatch("token vector length must equal cache.batch_size"))
        return reshape(Int.(collect(token)), 1, batch_size)
    elseif token isa AbstractMatrix
        size(token) == (1, batch_size) ||
            throw(DimensionMismatch("token matrix must have shape (1, cache.batch_size)"))
        return Int.(collect(token))
    end

    throw(DimensionMismatch("`token` must be an integer, vector, or one-row matrix"))
end

function _validate_generation_ids(tokens, vocab_size::Int)
    all(id -> 1 <= id <= vocab_size, tokens) ||
        throw(ArgumentError("token id is outside 1:$vocab_size"))
    return nothing
end

"""
    prefill(model, ps, st, prompt_tokens, cache; device=get_device(ps))

Process a complete prompt and populate every layer's KV cache.

Returns `(logits, updated_cache, updated_state)`. `cache` must be empty; later
single-token updates should use `decode_step`.
"""
function prefill(
    model::GPTModel,
    ps,
    st::NamedTuple,
    prompt_tokens,
    cache::GPTKVCache;
    device=get_device(ps),
)
    _validate_kv_cache(model, cache)
    isempty(cache) || throw(ArgumentError("`prefill` requires an empty cache"))

    tokens = _prefill_token_matrix(prompt_tokens)
    size(tokens, 2) == cache.batch_size ||
        throw(DimensionMismatch("prompt batch size does not match cache.batch_size"))
    _validate_generation_ids(tokens, model.vocab_size)

    return _gpt_with_kv_cache(
        model,
        device(tokens),
        ps,
        st,
        cache;
        start_pos=1,
    )
end

"""
    decode_step(model, ps, st, token, cache; device=get_device(ps))

Append one token per batch item and compute logits for that new position.

Returns `(logits, updated_cache, updated_state)`. The logits shape is
`(vocab_size, 1, batch)`.
"""
function decode_step(
    model::GPTModel,
    ps,
    st::NamedTuple,
    token,
    cache::GPTKVCache;
    device=get_device(ps),
)
    _validate_kv_cache(model, cache)
    isempty(cache) && throw(ArgumentError("call `prefill` before `decode_step`"))
    cache.position < model.max_seq_len ||
        throw(ArgumentError("KV cache has reached model.max_seq_len"))

    tokens = _decode_token_matrix(token, cache.batch_size)
    _validate_generation_ids(tokens, model.vocab_size)

    return _gpt_with_kv_cache(
        model,
        device(tokens),
        ps,
        st,
        cache;
        start_pos=cache.position + 1,
    )
end

"""
    generate_cached(model, ps, st, prompt_tokens; kwargs...)

Autoregressively generate token ids using prompt prefill followed by one-token
KV-cached decoding. The return value matches `generate`:
`(generated_tokens, updated_state)`.

Unlike the sliding-window eager `generate`, this first implementation requires
the complete processed context to fit in `model.max_seq_len`.
"""
function generate_cached(
    model::GPTModel,
    ps,
    st::NamedTuple,
    prompt_tokens;
    max_new_tokens::Int=100,
    temperature::Real=1.0f0,
    top_k=nothing,
    rng::AbstractRNG=default_rng(),
    device=get_device(ps),
)
    max_new_tokens >= 0 ||
        throw(ArgumentError("`max_new_tokens` must be non-negative"))

    generated = Int.(collect(prompt_tokens))
    isempty(generated) && throw(ArgumentError("`prompt_tokens` must not be empty"))
    _validate_generation_ids(generated, model.vocab_size)

    max_new_tokens == 0 && return generated, st

    # The final sampled token does not need to be fed back into the model.
    processed_tokens = length(generated) + max_new_tokens - 1
    processed_tokens <= model.max_seq_len || throw(ArgumentError(
        "prompt plus generated context exceeds model.max_seq_len; " *
        "reduce `max_new_tokens` or use `generate` with sliding context",
    ))

    cache = init_kv_cache(model; batch_size=1)
    logits, cache, st_current = prefill(
        model,
        ps,
        st,
        generated,
        cache;
        device,
    )
    host = cpu_device()

    for step in 1:max_new_tokens
        last_logits = vec(host(@view(logits[:, end, 1])))
        next_id = _sample_token(
            last_logits,
            rng;
            temperature,
            top_k,
        )
        push!(generated, next_id)

        if step < max_new_tokens
            logits, cache, st_current = decode_step(
                model,
                ps,
                st_current,
                next_id,
                cache;
                device,
            )
        end
    end

    return generated, st_current
end

"""
    generate_cached(model, ps, st, tokenizer, prompt; kwargs...)

String convenience overload for KV-cached generation.
"""
function generate_cached(
    model::GPTModel,
    ps,
    st::NamedTuple,
    tokenizer::Tokenizer,
    prompt::AbstractString;
    kwargs...
)
    generated_tokens, st_new = generate_cached(
        model,
        ps,
        st,
        encode(tokenizer, prompt);
        kwargs...
    )
    return decode(tokenizer, generated_tokens), st_new
end
