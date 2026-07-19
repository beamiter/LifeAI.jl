using Lux
using MLDataDevices: cpu_device, get_device
using NNlib: batched_mul, softmax
using Random: AbstractRNG, default_rng
import Reactant

"""
    StaticLayerKVCache(keys, values)

Preallocated key/value storage for one Transformer layer. Both tensors keep a
fixed physical shape:

    (head_dim, num_heads, max_seq_len, batch)

Only the prefix selected by `StaticGPTKVCache.position` is logically valid.
"""
struct StaticLayerKVCache{K,V}
    keys::K
    values::V
end

"""
    StaticGPTKVCache(layers, position, batch_size, max_seq_len)

Fixed-shape per-request KV cache. Unlike `GPTKVCache`, decoding never grows the
underlying arrays, so one XLA executable can be reused for every token position.

`position` stores the number of processed tokens. On ordinary CPU/GPU paths it
is an `Int32`; XLA uses a tracked `Reactant.ConcreteRNumber{Int32}` so the value
can change without changing the compiled program.
"""
struct StaticGPTKVCache{C,P}
    layers::C
    position::P
    batch_size::Int
    max_seq_len::Int
end

function Base.length(cache::StaticGPTKVCache)
    return Int(cache.position)
end

Base.isempty(cache::StaticGPTKVCache) = length(cache) == 0

function _model_blocks(model::GPTModel)
    return Tuple(values(model.blocks.layers))
end

function _parameter_cache_eltype(ps)
    first_block = first(Tuple(values(ps.blocks)))
    return eltype(first_block.attn.k_proj.weight)
end

"""
    init_static_kv_cache(model; batch_size=1, dtype=Float32,
                         device=Lux.cpu_device())

Allocate fixed-capacity K/V buffers for every Transformer layer. This cache can
be used eagerly through the normal `prefill` and `decode_step` functions, or as
the backing storage for `XLAKVDecoder`.
"""
function init_static_kv_cache(
    model::GPTModel;
    batch_size::Int=1,
    dtype::Type{<:AbstractFloat}=Float32,
    device=Lux.cpu_device(),
)
    batch_size > 0 || throw(ArgumentError("`batch_size` must be positive"))

    layers = map(_model_blocks(model)) do block
        shape = (
            block.attn.head_dim,
            block.attn.num_heads,
            model.max_seq_len,
            batch_size,
        )
        return StaticLayerKVCache(
            device(zeros(dtype, shape)),
            device(zeros(dtype, shape)),
        )
    end

    return StaticGPTKVCache(
        layers,
        Int32(0),
        batch_size,
        model.max_seq_len,
    )
end

function _validate_static_kv_cache(model::GPTModel, cache::StaticGPTKVCache)
    length(cache.layers) == model.num_layers ||
        throw(DimensionMismatch("cache layer count does not match model.num_layers"))
    cache.max_seq_len == model.max_seq_len ||
        throw(DimensionMismatch("cache capacity does not match model.max_seq_len"))
    cache.batch_size > 0 || throw(ArgumentError("cache batch size must be positive"))

    position = length(cache)
    0 <= position <= cache.max_seq_len ||
        throw(ArgumentError("cache position is outside 0:max_seq_len"))

    for (block, layer_cache) in zip(_model_blocks(model), cache.layers)
        expected_shape = (
            block.attn.head_dim,
            block.attn.num_heads,
            cache.max_seq_len,
            cache.batch_size,
        )
        size(layer_cache.keys) == expected_shape ||
            throw(DimensionMismatch("static key cache has an invalid shape"))
        size(layer_cache.values) == expected_shape ||
            throw(DimensionMismatch("static value cache has an invalid shape"))
    end

    return nothing
end

function _apply_rope_single_position(
    x,
    cos_cache,
    sin_cache,
    position,
)
    D, H, _, B = size(x)
    half_dim = D ÷ 2

    x_pairs = reshape(x, 2, half_dim, H, 1, B)
    x1 = selectdim(x_pairs, 1, 1)
    x2 = selectdim(x_pairs, 1, 2)

    cos_values = reshape(eltype(x).(cos_cache[:, position]), half_dim, 1, 1, 1)
    sin_values = reshape(eltype(x).(sin_cache[:, position]), half_dim, 1, 1, 1)

    y1 = x1 .* cos_values .- x2 .* sin_values
    y2 = x1 .* sin_values .+ x2 .* cos_values
    y_pairs = cat(
        reshape(y1, 1, half_dim, H, 1, B),
        reshape(y2, 1, half_dim, H, 1, B);
        dims=1,
    )

    return reshape(y_pairs, D, H, 1, B)
end

function _scaled_dot_product_attention_valid_prefix(
    q,
    k,
    v,
    valid_length,
)
    D, H, Tq, B = size(q)
    _, _, Tk, _ = size(k)
    HB = H * B

    q3 = reshape(permutedims(Float32.(q), (3, 1, 2, 4)), Tq, D, HB)
    k3 = reshape(permutedims(Float32.(k), (1, 3, 2, 4)), D, Tk, HB)
    v3 = reshape(permutedims(Float32.(v), (3, 1, 2, 4)), Tk, D, HB)

    scores = batched_mul(q3, k3) .* inv(sqrt(Float32(D)))

    # `valid_length` is an XLA-tracked scalar during compiled decoding. The
    # physical cache shape stays constant while this mask exposes only the
    # already-written prefix.
    key_positions_host = reshape(Int32.(collect(1:Tk)), 1, Tk, 1)
    key_positions = similar(scores, Int32, 1, Tk, 1)
    copyto!(key_positions, key_positions_host)
    visible = key_positions .<= valid_length
    scores = ifelse.(visible, scores, -Inf32)

    weights = softmax(scores; dims=2)
    context3 = batched_mul(weights, v3)
    context = permutedims(reshape(context3, Tq, D, H, B), (2, 3, 1, 4))

    return eltype(q).(context)
end

function _static_attention_prefill!(
    attn::MultiHeadAttention,
    x,
    ps,
    st::NamedTuple,
    cache::StaticLayerKVCache,
)
    _, num_tokens, batch_size = size(x)

    queries, st_q_proj = attn.q_proj(x, ps.q_proj, st.q_proj)
    keys, st_k_proj = attn.k_proj(x, ps.k_proj, st.k_proj)
    values, st_v_proj = attn.v_proj(x, ps.v_proj, st.v_proj)

    queries = reshape(queries, attn.head_dim, attn.num_heads, num_tokens, batch_size)
    keys = reshape(keys, attn.head_dim, attn.num_heads, num_tokens, batch_size)
    values = reshape(values, attn.head_dim, attn.num_heads, num_tokens, batch_size)

    if attn.use_rope
        queries = apply_rope(
            queries,
            st.rope_cos_cache,
            st.rope_sin_cache;
            start_pos=1,
        )
        keys = apply_rope(
            keys,
            st.rope_cos_cache,
            st.rope_sin_cache;
            start_pos=1,
        )
    end

    cache.keys[:, :, 1:num_tokens, :] .= keys
    cache.values[:, :, 1:num_tokens, :] .= values

    context, _ = batched_scaled_dot_product_attention(
        queries,
        keys,
        values;
        is_causal=attn.is_causal,
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
        cache,
    )
end

function _static_attention_decode!(
    attn::MultiHeadAttention,
    x,
    ps,
    st::NamedTuple,
    cache::StaticLayerKVCache,
    cached_tokens,
)
    _, _, batch_size = size(x)
    write_position = cached_tokens + one(cached_tokens)

    queries, st_q_proj = attn.q_proj(x, ps.q_proj, st.q_proj)
    keys, st_k_proj = attn.k_proj(x, ps.k_proj, st.k_proj)
    values, st_v_proj = attn.v_proj(x, ps.v_proj, st.v_proj)

    queries = reshape(queries, attn.head_dim, attn.num_heads, 1, batch_size)
    keys = reshape(keys, attn.head_dim, attn.num_heads, 1, batch_size)
    values = reshape(values, attn.head_dim, attn.num_heads, 1, batch_size)

    if attn.use_rope
        queries = _apply_rope_single_position(
            queries,
            st.rope_cos_cache,
            st.rope_sin_cache,
            write_position,
        )
        keys = _apply_rope_single_position(
            keys,
            st.rope_cos_cache,
            st.rope_sin_cache,
            write_position,
        )
    end

    cache.keys[:, :, write_position, :] = dropdims(keys; dims=3)
    cache.values[:, :, write_position, :] = dropdims(values; dims=3)

    context = _scaled_dot_product_attention_valid_prefix(
        queries,
        cache.keys,
        cache.values,
        write_position,
    )
    context = reshape(context, attn.d_out, 1, batch_size)
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
        cache,
    )
end

function _static_block_prefill!(
    block::TransformerBlock,
    x,
    ps,
    st::NamedTuple,
    cache::StaticLayerKVCache,
)
    x_norm1, st_norm1 = block.norm1(x, ps.norm1, st.norm1)
    attn_out, st_attn, cache = _static_attention_prefill!(
        block.attn,
        x_norm1,
        ps.attn,
        st.attn,
        cache,
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
        cache,
    )
end

function _static_block_decode!(
    block::TransformerBlock,
    x,
    ps,
    st::NamedTuple,
    cache::StaticLayerKVCache,
    cached_tokens,
)
    x_norm1, st_norm1 = block.norm1(x, ps.norm1, st.norm1)
    attn_out, st_attn, cache = _static_attention_decode!(
        block.attn,
        x_norm1,
        ps.attn,
        st.attn,
        cache,
        cached_tokens,
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
        cache,
    )
end

function _static_blocks_prefill!(
    ::Tuple{},
    x,
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
)
    return x, (), ()
end

function _static_blocks_prefill!(blocks::Tuple, x, ps::Tuple, st::Tuple, caches::Tuple)
    y, st_head, cache_head = _static_block_prefill!(
        first(blocks),
        x,
        first(ps),
        first(st),
        first(caches),
    )
    y, st_tail, cache_tail = _static_blocks_prefill!(
        Base.tail(blocks),
        y,
        Base.tail(ps),
        Base.tail(st),
        Base.tail(caches),
    )
    return y, (st_head, st_tail...), (cache_head, cache_tail...)
end

function _static_blocks_decode!(
    ::Tuple{},
    x,
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    cached_tokens,
)
    return x, (), ()
end

function _static_blocks_decode!(
    blocks::Tuple,
    x,
    ps::Tuple,
    st::Tuple,
    caches::Tuple,
    cached_tokens,
)
    y, st_head, cache_head = _static_block_decode!(
        first(blocks),
        x,
        first(ps),
        first(st),
        first(caches),
        cached_tokens,
    )
    y, st_tail, cache_tail = _static_blocks_decode!(
        Base.tail(blocks),
        y,
        Base.tail(ps),
        Base.tail(st),
        Base.tail(caches),
        cached_tokens,
    )
    return y, (st_head, st_tail...), (cache_head, cache_tail...)
end

function _static_gpt_prefill_kernel!(
    model::GPTModel,
    tokens,
    ps,
    st::NamedTuple,
    cache::StaticGPTKVCache,
)
    seq_len, _ = size(tokens)
    x, st_token_embedding = model.token_embedding(
        tokens,
        ps.token_embedding,
        st.token_embedding,
    )

    x, st_blocks_tuple, layer_caches = _static_blocks_prefill!(
        _model_blocks(model),
        x,
        Tuple(values(ps.blocks)),
        Tuple(values(st.blocks)),
        cache.layers,
    )
    st_blocks = NamedTuple{keys(st.blocks)}(st_blocks_tuple)

    x, st_final_norm = model.final_norm(x, ps.final_norm, st.final_norm)
    logits, st_lm_head = _project_logits(model, x, ps, st.lm_head)

    new_state = (;
        token_embedding=st_token_embedding,
        blocks=st_blocks,
        final_norm=st_final_norm,
        lm_head=st_lm_head,
    )
    new_cache = StaticGPTKVCache(
        layer_caches,
        cache.position + Int32(seq_len),
        cache.batch_size,
        cache.max_seq_len,
    )

    return logits, new_cache, new_state
end

function _static_gpt_decode_kernel!(
    model::GPTModel,
    tokens,
    ps,
    st::NamedTuple,
    cache::StaticGPTKVCache,
)
    x, st_token_embedding = model.token_embedding(
        tokens,
        ps.token_embedding,
        st.token_embedding,
    )

    x, st_blocks_tuple, layer_caches = _static_blocks_decode!(
        _model_blocks(model),
        x,
        Tuple(values(ps.blocks)),
        Tuple(values(st.blocks)),
        cache.layers,
        cache.position,
    )
    st_blocks = NamedTuple{keys(st.blocks)}(st_blocks_tuple)

    x, st_final_norm = model.final_norm(x, ps.final_norm, st.final_norm)
    logits, st_lm_head = _project_logits(model, x, ps, st.lm_head)

    new_state = (;
        token_embedding=st_token_embedding,
        blocks=st_blocks,
        final_norm=st_final_norm,
        lm_head=st_lm_head,
    )
    new_cache = StaticGPTKVCache(
        layer_caches,
        cache.position + one(cache.position),
        cache.batch_size,
        cache.max_seq_len,
    )

    return logits, new_cache, new_state
end

function prefill(
    model::GPTModel,
    ps,
    st::NamedTuple,
    prompt_tokens,
    cache::StaticGPTKVCache;
    device=get_device(ps),
)
    _validate_static_kv_cache(model, cache)
    isempty(cache) || throw(ArgumentError("`prefill` requires an empty cache"))

    tokens = _prefill_token_matrix(prompt_tokens)
    size(tokens, 2) == cache.batch_size ||
        throw(DimensionMismatch("prompt batch size does not match cache.batch_size"))
    size(tokens, 1) <= cache.max_seq_len ||
        throw(ArgumentError("prompt exceeds cache.max_seq_len"))
    _validate_generation_ids(tokens, model.vocab_size)

    return _static_gpt_prefill_kernel!(
        model,
        device(tokens),
        ps,
        st,
        cache,
    )
end

function decode_step(
    model::GPTModel,
    ps,
    st::NamedTuple,
    token,
    cache::StaticGPTKVCache;
    device=get_device(ps),
)
    _validate_static_kv_cache(model, cache)
    isempty(cache) && throw(ArgumentError("call `prefill` before `decode_step`"))
    length(cache) < cache.max_seq_len ||
        throw(ArgumentError("KV cache has reached model.max_seq_len"))

    tokens = _decode_token_matrix(token, cache.batch_size)
    _validate_generation_ids(tokens, model.vocab_size)

    return _static_gpt_decode_kernel!(
        model,
        device(tokens),
        ps,
        st,
        cache,
    )
end

"""
    XLAKVDecoder(model, ps, st; batch_size=1, xla_backend="gpu",
                 cache_eltype=nothing)

Stateful Reactant/XLA decoder with:

- one prefill executable cached per prompt shape;
- one fixed-shape decode executable reused at every token position;
- preallocated K/V buffers with no per-token `cat` or shape change.

The first call for a new prompt shape compiles prefill. The first decode call
compiles the single-token step. Later calls reuse those executables.
"""
mutable struct XLAKVDecoder{M,P,S,C,D}
    model::M
    parameters::P
    state::S
    cache::C
    device::D
    xla_backend::String
    cache_eltype::DataType
    prefill_thunks::Dict{Tuple{Int,Int},Any}
    decode_thunk::Any
    host_position::Int
end

function _tracked_zero_position()
    return Reactant.to_rarray(Int32(0); track_numbers=true)
end

function _xla_static_cache(
    model::GPTModel,
    batch_size::Int,
    dtype::Type{<:AbstractFloat},
    device,
)
    cache = init_static_kv_cache(
        model;
        batch_size,
        dtype,
        device,
    )
    return StaticGPTKVCache(
        cache.layers,
        _tracked_zero_position(),
        cache.batch_size,
        cache.max_seq_len,
    )
end

function XLAKVDecoder(
    model::GPTModel,
    ps,
    st::NamedTuple;
    batch_size::Int=1,
    xla_backend::AbstractString="gpu",
    cache_eltype=nothing,
)
    xla_backend in ("cpu", "gpu", "tpu") ||
        throw(ArgumentError("`xla_backend` must be \"cpu\", \"gpu\", or \"tpu\""))
    batch_size > 0 || throw(ArgumentError("`batch_size` must be positive"))

    Reactant.set_default_backend(String(xla_backend))
    device = Lux.reactant_device(; force=true)
    ps_xla, st_xla = device((ps, Lux.testmode(st)))

    dtype = cache_eltype === nothing ? _parameter_cache_eltype(ps_xla) : cache_eltype
    (dtype isa DataType && dtype <: AbstractFloat) ||
        throw(ArgumentError("`cache_eltype` must be an AbstractFloat type"))

    cache = _xla_static_cache(model, batch_size, dtype, device)

    return XLAKVDecoder(
        model,
        ps_xla,
        st_xla,
        cache,
        device,
        String(xla_backend),
        dtype,
        Dict{Tuple{Int,Int},Any}(),
        nothing,
        0,
    )
end

function _reset_xla_cache!(decoder::XLAKVDecoder)
    Reactant.set_default_backend(decoder.xla_backend)
    decoder.cache = StaticGPTKVCache(
        decoder.cache.layers,
        _tracked_zero_position(),
        decoder.cache.batch_size,
        decoder.cache.max_seq_len,
    )
    decoder.host_position = 0
    return decoder
end

"""
    xla_prefill!(decoder, prompt_tokens)

Compile or reuse a prompt-shape-specific XLA prefill executable, reset the
logical cache, and process the complete prompt.
"""
function xla_prefill!(decoder::XLAKVDecoder, prompt_tokens)
    tokens = _prefill_token_matrix(prompt_tokens)
    size(tokens, 2) == decoder.cache.batch_size ||
        throw(DimensionMismatch("prompt batch size does not match decoder batch size"))
    size(tokens, 1) > 0 || throw(ArgumentError("prompt must contain at least one token"))
    size(tokens, 1) <= decoder.cache.max_seq_len ||
        throw(ArgumentError("prompt exceeds decoder max_seq_len"))
    _validate_generation_ids(tokens, decoder.model.vocab_size)

    _reset_xla_cache!(decoder)
    tokens_xla = decoder.device(tokens)
    shape_key = size(tokens)

    if !haskey(decoder.prefill_thunks, shape_key)
        compiled_prefill = Reactant.@compile _static_gpt_prefill_kernel!(
            decoder.model,
            tokens_xla,
            decoder.parameters,
            decoder.state,
            decoder.cache,
        )
        decoder.prefill_thunks[shape_key] = compiled_prefill
    end

    logits, cache, state = decoder.prefill_thunks[shape_key](
        decoder.model,
        tokens_xla,
        decoder.parameters,
        decoder.state,
        decoder.cache,
    )
    decoder.cache = cache
    decoder.state = state
    decoder.host_position = size(tokens, 1)

    return logits, decoder.cache, decoder.state
end

"""
    xla_decode_step!(decoder, token)

Append one token per batch item. The first invocation compiles the fixed-shape
decode step; every later position reuses the same executable.
"""
function xla_decode_step!(decoder::XLAKVDecoder, token)
    decoder.host_position > 0 ||
        throw(ArgumentError("call `xla_prefill!` before `xla_decode_step!`"))
    decoder.host_position < decoder.cache.max_seq_len ||
        throw(ArgumentError("KV cache has reached model.max_seq_len"))

    tokens = _decode_token_matrix(token, decoder.cache.batch_size)
    _validate_generation_ids(tokens, decoder.model.vocab_size)
    tokens_xla = decoder.device(tokens)

    if decoder.decode_thunk === nothing
        compiled_decode = Reactant.@compile _static_gpt_decode_kernel!(
            decoder.model,
            tokens_xla,
            decoder.parameters,
            decoder.state,
            decoder.cache,
        )
        decoder.decode_thunk = compiled_decode
    end

    logits, cache, state = decoder.decode_thunk(
        decoder.model,
        tokens_xla,
        decoder.parameters,
        decoder.state,
        decoder.cache,
    )
    decoder.cache = cache
    decoder.state = state
    decoder.host_position += 1

    return logits, decoder.cache, decoder.state
end

"""
    generate_xla_cached!(decoder, prompt_tokens; kwargs...)

Autoregressively generate one sequence with compiled XLA prefill and decode
steps. The decoder retains compiled thunks, so it can be reused for later
prompts without recompiling matching shapes.
"""
function generate_xla_cached!(
    decoder::XLAKVDecoder,
    prompt_tokens;
    max_new_tokens::Int=100,
    temperature::Real=1.0f0,
    top_k=nothing,
    rng::AbstractRNG=default_rng(),
)
    decoder.cache.batch_size == 1 ||
        throw(ArgumentError("`generate_xla_cached!` currently requires batch_size=1"))
    max_new_tokens >= 0 ||
        throw(ArgumentError("`max_new_tokens` must be non-negative"))

    generated = Int.(collect(prompt_tokens))
    isempty(generated) && throw(ArgumentError("`prompt_tokens` must not be empty"))
    _validate_generation_ids(generated, decoder.model.vocab_size)

    max_new_tokens == 0 && return generated, decoder.state

    processed_tokens = length(generated) + max_new_tokens - 1
    processed_tokens <= decoder.cache.max_seq_len || throw(ArgumentError(
        "prompt plus generated context exceeds model.max_seq_len",
    ))

    logits, _, _ = xla_prefill!(decoder, generated)
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
            logits, _, _ = xla_decode_step!(decoder, next_id)
        end
    end

    return generated, decoder.state
end

"""
    generate_xla_cached!(decoder, tokenizer, prompt; kwargs...)

String convenience overload for compiled KV-cached generation.
"""
function generate_xla_cached!(
    decoder::XLAKVDecoder,
    tokenizer::Tokenizer,
    prompt::AbstractString;
    kwargs...
)
    generated_tokens, state = generate_xla_cached!(
        decoder,
        encode(tokenizer, prompt);
        kwargs...
    )
    return decode(tokenizer, generated_tokens), state
end
