# Native Ripple inference. Shared device-neutral projection, RMSNorm, RoPE and
# bounded CUDA attention/cache primitives are reused from the Flint backend;
# the decoder equations and checkpoint names below are Ricursive's architecture.
_ripple_linear(weight,x) = _flint_linear(weight,x)
_ripple_linear_multi(weights::Tuple,x) = map(weight->_ripple_linear(weight,x),weights)

mutable struct RippleKVCache
    model::RippleConfig
    layers::Vector{Any}
    tokens::Int
end
RippleKVCache(model::RippleConfig) = RippleKVCache(model, Any[nothing for _ in 1:model.num_hidden_layers], 0)

function reset_ripple_cache!(cache::RippleKVCache)
    fill!(cache.layers, nothing)
    cache.tokens = 0
    return cache
end

_ripple_sinks(x,heads)=_flint_like(x,fill(-Inf32,heads))

function _ripple_attention(c, parameters, p, x, seed, layer, past, position; cache_capacity=0,reuse_constants::Bool=false)
    d, heads, kvheads, n = c.head_dim, c.num_attention_heads, c.num_key_value_heads, size(x, 2)
    q0, k0 = _ripple_linear_multi((parameters[p*"q_proj.weight"], parameters[p*"k_proj.weight"]), x)
    q = _flint_norm(reshape(q0, d, heads, n), parameters[p*"q_norm.weight"], c.rms_norm_eps)
    k = _flint_norm(reshape(k0, d, kvheads, n), parameters[p*"k_norm.weight"], c.rms_norm_eps)
    # The residual seed is the scaled original embedding, before any layer norm.
    # Blend BEFORE v_proj, independently at every layer (including layer zero).
    value_input = if c.use_valres
        blend = 1f0 ./ (1f0 .+ exp.(-reshape(parameters[p*"value_res_lambda"], 1, 1)))
        seed .+ blend .* (x .- seed)
    else
        x
    end
    v = reshape(_ripple_linear(parameters[p*"v_proj.weight"], value_input), d, kvheads, n)
    local_attention = c.layer_types[layer] == "sliding_attention"
    if local_attention
        q = _flint_rope(q, position:position+n-1, c.rope_theta)
        k = _flint_rope(k, position:position+n-1, c.rope_theta)
    end
    window = local_attention ? c.sliding_window : 0
    appended = _flint_kv_append(k, v, past, position, window; capacity_hint=cache_capacity)
    # Ripple has ordinary softmax, without Flint's learned attention sinks.
    # A -Inf sink has exactly zero probability in the shared attention primitive.
    sinks = reuse_constants ? _ripple_sinks(x,heads) : _flint_like(x,fill(-Inf32,heads))
    context = _flint_attention_long(q, appended.key, appended.value, sinks,
        appended.first_position, position, window)
    output = reshape(context, d*heads, n)
    if c.use_gated_attention
        gate = _ripple_linear(parameters[p*"g_proj.weight"], x)
        output = output ./ (1f0 .+ exp.(-gate))
    end
    return _ripple_linear(parameters[p*"o_proj.weight"], output), appended.updated
end

function _ripple_gate_product(gate,up,limit)
    if limit===nothing
        return (gate ./ (1f0 .+ exp.(-gate))) .* up
    end
    # Min/clamp are selections of existing Float32 values. Fuse them with the
    # activation instead of materializing two additional device matrices.
    return @. (min(gate,limit)/(1f0+exp(-min(gate,limit))))*clamp(up,-limit,limit)
end

function _ripple_mlp(c, parameters, p, x;fuse::Bool=false)
    gate, up = _ripple_linear_multi((parameters[p*"gate_proj.weight"], parameters[p*"up_proj.weight"]), x)
    if fuse
        return _ripple_linear(parameters[p*"down_proj.weight"],_ripple_gate_product(gate,up,c.ffn_clamp_threshold))
    end
    if c.ffn_clamp_threshold !== nothing
        # The SiLU gate is clipped only above; the up projection on both sides.
        limit = c.ffn_clamp_threshold
        gate = min.(gate, limit)
        up = clamp.(up, -limit, limit)
    end
    return _ripple_linear(parameters[p*"down_proj.weight"], (gate ./ (1f0 .+ exp.(-gate))) .* up)
end

"""
    hf_ripple_forward(model, parameters, token_ids; cache=nothing, last_only=false,
                      cache_capacity=0, output_logits=true)

Native Float32 Ripple inference on CPU/CUDA. IDs are a nonempty one-based
vector; logits have shape (vocabulary, tokens). Supports token and chunked
KV caching, local-window eviction, and optional packed LifeAI projections.
"""
function hf_ripple_forward(model::RippleConfig, parameters, token_ids::AbstractVector{<:Integer};
        cache::Union{Nothing,RippleKVCache}=nothing, last_only::Bool=false,
        cache_capacity::Integer=0, output_logits::Bool=true,on_hidden=nothing)
    isempty(token_ids) && throw(ArgumentError("Ripple input must not be empty"))
    eltype(token_ids) <: Bool && throw(ArgumentError("Ripple token IDs cannot be Bool"))
    ids = Int.(collect(token_ids))
    all(id -> 1 <= id <= model.vocab_size, ids) || throw(ArgumentError("Ripple token IDs are out of range"))
    position = cache === nothing ? 0 : cache.tokens
    cache === nothing || cache.model === model || throw(ArgumentError("cache belongs to another Ripple model"))
    length(ids) <= model.max_position_embeddings-position || throw(ArgumentError("Ripple context limit exceeded"))
    !(cache_capacity isa Bool) && 0 <= cache_capacity <= model.max_position_embeddings ||
        throw(ArgumentError("invalid Ripple KV reservation"))
    _ripple_activate!(parameters)
    x = permutedims(Float32.(parameters["model.embed_tokens.weight"][ids, :]))
    model.embed_scale === nothing || (x = x .* model.embed_scale)
    seed = x
    on_hidden===nothing || on_hidden(0,x)
    updates = Any[]
    for layer in 1:model.num_hidden_layers
        _ripple_activate!(parameters)
        p = "model.layers.$(layer-1)."
        normalized = _flint_norm(x, parameters[p*"input_layernorm.weight"], model.rms_norm_eps)
        past = cache === nothing ? nothing : cache.layers[layer]
        attention, updated = _ripple_attention(model, parameters, p*"self_attn.", normalized,
            seed, layer, past, position; cache_capacity=Int(cache_capacity))
        x = x .+ _flint_norm(attention, parameters[p*"post_attention_layernorm.weight"], model.rms_norm_eps)
        normalized = _flint_norm(x, parameters[p*"pre_feedforward_layernorm.weight"], model.rms_norm_eps)
        feedforward = _ripple_mlp(model, parameters, p*"mlp.", normalized)
        x = x .+ _flint_norm(feedforward, parameters[p*"post_feedforward_layernorm.weight"], model.rms_norm_eps)
        on_hidden===nothing || on_hidden(layer,x)
        push!(updates, updated)
    end
    logits = if output_logits
        _ripple_activate!(parameters)
        last_only && (x = x[:, end:end])
        x = _flint_norm(x, parameters["model.norm.weight"], model.rms_norm_eps)
        head = model.tie_word_embeddings ? parameters["model.embed_tokens.weight"] : parameters["lm_head.weight"]
        _ripple_linear(head, x)
    else
        nothing
    end
    if cache !== nothing
        cache.layers = updates
        cache.tokens += length(ids)
    end
    _ripple_activate!(parameters)
    return logits
end

"""Cached generation of one-based IDs. Callbacks may return false to cancel."""
function generate_ripple_tokens(bundle, prompt::AbstractVector{<:Integer}; max_new_tokens::Integer=128,
        temperature::Real=1f0, top_k=nothing, top_p=1f0, rng=default_rng(),
        eos_ids=bundle.model.eos_token_id .+ 1, on_token=nothing,
        prefill_chunk_size::Integer=1024, on_prefill=nothing)
    !(max_new_tokens isa Bool) && max_new_tokens >= 0 || throw(ArgumentError("invalid Ripple max_new_tokens"))
    !(prefill_chunk_size isa Bool) && prefill_chunk_size > 0 || throw(ArgumentError("invalid Ripple prefill_chunk_size"))
    !isempty(prompt) && !(eltype(prompt) <: Bool) && all(id -> 1 <= id <= bundle.model.vocab_size, prompt) ||
        throw(ArgumentError("invalid Ripple prompt IDs"))
    max_new_tokens <= bundle.model.max_position_embeddings-length(prompt) ||
        throw(ArgumentError("requested Ripple generation exceeds context limit"))
    generated = Int[]
    max_new_tokens == 0 && return generated
    cache = RippleKVCache(bundle.model)
    capacity = length(prompt)+Int(max_new_tokens)
    logits = nothing
    for first in 1:Int(min(prefill_chunk_size, length(prompt))):length(prompt)
        last = min(length(prompt), first+prefill_chunk_size-1)
        logits = hf_ripple_forward(bundle.model, bundle.parameters, prompt[first:last]; cache,
            last_only=true, cache_capacity=capacity, output_logits=(last == length(prompt)))
        on_prefill === nothing || on_prefill(last, length(prompt)) !== false || return generated
    end
    for step in 1:max_new_tokens
        token = _flint_sample_token(logits, rng; temperature, top_k, top_p)
        push!(generated, token)
        on_token === nothing || on_token(token) !== false || break
        (token in eos_ids || step == max_new_tokens) && break
        logits = hf_ripple_forward(bundle.model, bundle.parameters, [token]; cache,
            last_only=true, cache_capacity=capacity)
    end
    return generated
end

function generate_ripple(bundle, tokenizer, prompt; tools=nothing, clear_thinking::Bool=false, kwargs...)
    text = prompt isa AbstractString ? String(prompt) :
        apply_ripple_chat_template(tokenizer, prompt; tools, clear_thinking)
    output = generate_ripple_tokens(bundle, encode(tokenizer, text); kwargs...)
    return (text=decode(tokenizer, output; skip_special_tokens=true, errors=:replace), token_ids=output)
end
