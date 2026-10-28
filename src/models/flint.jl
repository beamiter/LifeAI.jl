# Native Flint reference inference. Tensor names/layout match RecursiveSI/flint.
# Activations, attention scores and router scores use Float32 on CPU or CUDA.
using Random: default_rng

"""Incremental Flint KV cache. Token positions are zero-based internally."""
mutable struct FlintKVCache
    model::FlintConfig
    layers::Vector{Any}
    tokens::Int
end
FlintKVCache(model::FlintConfig) = FlintKVCache(model, Any[nothing for _ in 1:model.num_hidden_layers], 0)
function reset_flint_cache!(cache::FlintKVCache)
    fill!(cache.layers, nothing)
    cache.tokens = 0
    return cache
end

# Copy small host constants to the activation device without depending on CUDA.
function _flint_like(x, values)
    result = similar(x, Float32, size(values))
    copyto!(result, Float32.(values))
    return result
end

_flint_dequantize(w::AbstractMatrix) = Float32.(w)
_flint_dequantize(w::Int8ChannelWeight) = Float32.(w.q) .* reshape(w.scale, :, 1)
function _flint_dequantize(w::Int4GroupWeight)
    rows, half = size(w.packed)
    groups = ((2 .* (1:half) .- 2) .÷ w.group) .+ 1
    scales = w.scale[:, groups]
    low = (Float32.(w.packed .& UInt8(15)) .- 8f0) .* scales
    high = (Float32.(w.packed .>> 4) .- 8f0) .* scales
    return reshape(cat(reshape(low, rows, 1, half), reshape(high, rows, 1, half); dims=2), rows, w.in_dim)
end
function _flint_linear(w, x)
    # Bound dequantization memory for the vocabulary projection and large MLPs.
    if w isa Union{Int8ChannelWeight,Int4GroupWeight,NVFP4Weight}
        out_dim = _flint_weight_shape(w)[1]
        result = similar(x, Float32, (out_dim, size(x, 2)))
        for first in 1:4096:out_dim
            rows = first:min(first + 4095, out_dim)
            result[rows, :] = _flint_dequantize(_quant_row_slice(w, rows)) * x
        end
        return result
    end
    return (eltype(w) === Float32 ? w : Float32.(w)) * x
end

_flint_linear(weight::AbstractNativeNVFP4Weight, x) = native_nvfp4_gemm(weight, x)

"""
    prepare_flint_nvfp4(bundle; kwargs...)

Prepare all CUDA NVFP4 projections for native FP4 GEMM. The returned bundle
shares floating-point parameters with the input. Native GEMM quantizes each
activation matrix too (W4A4), so results can differ from reference W4A16 mode.
Saving the prepared bundle preserves the original portable NVFP4 weights.
"""
function prepare_flint_nvfp4(bundle; kwargs...)
    parameters = Dict{String,Any}(
        name => (weight isa NVFP4Weight ? prepare_nvfp4_gemm(weight; kwargs...) : weight)
        for (name, weight) in bundle.parameters
    )
    return merge(bundle, (parameters=parameters, nvfp4_backend=:native))
end

"""Aggregate evidence of native GEMM use in a prepared Flint bundle."""
function flint_native_fp4_stats(bundle)
    weights = [weight for weight in values(bundle.parameters) if weight isa AbstractNativeNVFP4Weight]
    stats = native_nvfp4_stats.(weights)
    return (
        backend=isempty(weights) ? :reference : :cublaslt_nvfp4,
        prepared_weights=length(weights),
        native_calls=sum((get(stat, :native_calls, 0) for stat in stats); init=0),
        prepared_weight_bytes=sum((get(stat, :prepared_weight_bytes, 0) for stat in stats); init=0),
        shared_workspace_bytes=maximum((get(stat, :workspace_bytes, 0) for stat in stats); init=0),
    )
end

function _flint_norm(x, w, epsilon)
    return x .* inv.(sqrt.(sum(abs2, x; dims=1) ./ size(x, 1) .+ Float32(epsilon))) .* reshape(Float32.(w), :, ntuple(_ -> 1, ndims(x)-1)...)
end

function _flint_rope(x, positions, theta)
    d = size(x, 1)
    half = d ÷ 2
    frequencies = Float32.(theta) .^ (-Float32.(0:half-1) .* (2f0 / d))
    angles = reshape(frequencies, half, 1) .* reshape(Float32.(positions), 1, :)
    cosine = reshape(_flint_like(x, cos.(angles)), half, 1, :)
    sine = reshape(_flint_like(x, sin.(angles)), half, 1, :)
    a, b = x[1:half, :, :], x[half+1:end, :, :]
    return cat(a .* cosine .- b .* sine, b .* cosine .+ a .* sine; dims=1)
end

function _flint_attention(model, parameters, prefix, x, layer, past, position)
    d, heads, kvheads = model.head_dim, model.num_attention_heads, model.num_key_value_heads
    n = size(x, 2)
    q = _flint_norm(reshape(_flint_linear(parameters[prefix * "q_proj.weight"], x), d, heads, n), parameters[prefix * "q_norm.weight"], model.rms_norm_eps)
    k = _flint_norm(reshape(_flint_linear(parameters[prefix * "k_proj.weight"], x), d, kvheads, n), parameters[prefix * "k_norm.weight"], model.rms_norm_eps)
    v = reshape(_flint_linear(parameters[prefix * "v_proj.weight"], x), d, kvheads, n)
    local_attention = model.layer_types[layer] == "sliding_attention"
    if local_attention
        q = _flint_rope(q, position:position+n-1, model.rope_theta)
        k = _flint_rope(k, position:position+n-1, model.rope_theta)
    end
    first_position = position
    if past !== nothing
        k = cat(past.key, k; dims=3)
        v = cat(past.value, v; dims=3)
        first_position = past.first_position
    end
    key_positions = first_position:position+n-1
    query_positions = position:position+n-1
    allowed = [kp <= qp && (!local_attention || kp > qp-model.sliding_window) for kp in key_positions, qp in query_positions]
    mask = _flint_like(x, ifelse.(allowed, 0f0, -Inf32))
    output = similar(x, Float32, (d, heads, n))
    sinks = parameters[prefix * "softmax_offset"]
    for h in 1:heads
        kh = (h-1) ÷ (heads ÷ kvheads) + 1
        scores = transpose(k[:, kh, :]) * q[:, h, :] ./ sqrt(Float32(d)) .+ mask
        sink = reshape(Float32.(sinks[h:h]), 1, 1)
        maximum_score = max.(maximum(scores; dims=1), sink)
        weights = exp.(scores .- maximum_score)
        weights ./= sum(weights; dims=1) .+ exp.(sink .- maximum_score)
        output[:, h, :] = v[:, kh, :] * weights
    end
    # A local layer only needs the preceding W-1 entries at the next call.
    keep = local_attention ? min(size(k, 3), model.sliding_window-1) : size(k, 3)
    start = size(k, 3)-keep+1
    updated = (key=k[:, :, start:end], value=v[:, :, start:end], first_position=position+n-keep)
    return _flint_linear(parameters[prefix * "o_proj.weight"], reshape(output, d*heads, n)), updated
end

function _flint_mlp(parameters, prefix, x)
    gate = _flint_linear(parameters[prefix * "gate_proj.weight"], x)
    up = _flint_linear(parameters[prefix * "up_proj.weight"], x)
    return _flint_linear(parameters[prefix * "down_proj.weight"], (gate ./ (1f0 .+ exp.(-gate))) .* up)
end

"""Sigmoid top-k routing. Returns one-based expert IDs and normalized scores."""
function flint_topk_routing(logits::AbstractMatrix, k::Integer; normalize::Bool=true)
    0 < k <= size(logits, 1) || throw(ArgumentError("invalid Flint expert top-k"))
    # Host routing is deliberate: the reference path dispatches only used experts.
    host_logits = Array(Float32.(logits))
    all(isfinite, host_logits) || throw(ArgumentError("Flint router logits must be finite"))
    scores = 1f0 ./ (1f0 .+ exp.(-host_logits))
    ids = Matrix{Int}(undef, k, size(scores, 2))
    weights = Matrix{Float32}(undef, size(ids))
    for t in axes(scores, 2)
        order = partialsortperm(view(scores, :, t), 1:k; rev=true)
        ids[:, t] = order
        selected = scores[order, t]
        if normalize
            total = sum(selected)
            # Compute stable relative sigmoid scores for extreme negative logits.
            if total == 0f0
                values = host_logits[order, t]
                selected = exp.(values .- maximum(values))
                total = sum(selected)
            end
            selected ./= total
        end
        weights[:, t] = selected
    end
    return (expert_ids=ids, weights=weights)
end

function _flint_feedforward(model, parameters, prefix, x, layer)
    sparse = !(layer-1 in model.mlp_only_layers) && model.num_experts > 0 && layer % model.decoder_sparse_step == 0
    sparse || return _flint_mlp(parameters, prefix, x)
    logits = _flint_linear(parameters[prefix * "gate.weight"], x)
    routes = flint_topk_routing(logits, model.num_experts_per_tok; normalize=model.norm_topk_prob)
    result = zero(x)
    for expert in sort!(unique(vec(routes.expert_ids)))
        matches = findall(==(expert), routes.expert_ids)
        columns = [index[2] for index in matches]
        factors = _flint_like(x, reshape([routes.weights[index] for index in matches], 1, :))
        values = _flint_mlp(parameters, prefix * "experts.$(expert-1).", x[:, columns]) .* factors
        result[:, columns] .+= values
    end
    return result .+ _flint_mlp(parameters, prefix * "shared_expert.", x)
end

"""
    hf_flint_forward(model, parameters, token_ids; cache=nothing, last_only=false)

Native Float32 Flint reference inference with optional incremental KV caching.
`token_ids` is a nonempty vector of LifeAI's **one-based** IDs. Result is a
`(vocabulary, tokens)` logits matrix (one column if `last_only=true`). Parameters
may contain packed INT4/INT8 matrices. CPU and CUDA arrays are supported; the
reference CUDA path dequantizes only the current projection and routes on CPU.
"""
function hf_flint_forward(model::FlintConfig, parameters, token_ids::AbstractVector{<:Integer}; cache::Union{Nothing,FlintKVCache}=nothing, last_only::Bool=false)
    isempty(token_ids) && throw(ArgumentError("Flint input must not be empty"))
    ids = Int.(collect(token_ids))
    eltype(token_ids) <: Bool && throw(ArgumentError("Flint token IDs cannot be Bool"))
    all(id -> 1 <= id <= model.vocab_size, ids) || throw(ArgumentError("Flint token IDs are out of range"))
    position = cache === nothing ? 0 : cache.tokens
    cache === nothing || cache.model === model || throw(ArgumentError("cache belongs to another Flint model"))
    position + length(ids) <= model.max_position_embeddings || throw(ArgumentError("Flint context limit exceeded"))
    x = permutedims(Float32.(parameters["model.embed_tokens.weight"][ids, :]))
    updates = Any[]
    for layer in 1:model.num_hidden_layers
        prefix = "model.layers.$(layer-1)."
        normalized = _flint_norm(x, parameters[prefix * "input_layernorm.weight"], model.rms_norm_eps)
        past = cache === nothing ? nothing : cache.layers[layer]
        attention, updated = _flint_attention(model, parameters, prefix * "self_attn.", normalized, layer, past, position)
        x = x .+ attention
        normalized = _flint_norm(x, parameters[prefix * "post_attention_layernorm.weight"], model.rms_norm_eps)
        x = x .+ _flint_feedforward(model, parameters, prefix * "mlp.", normalized, layer)
        push!(updates, updated)
    end
    x = _flint_norm(x, parameters["model.norm.weight"], model.rms_norm_eps)
    last_only && (x = x[:, end:end])
    head = model.tie_word_embeddings ? parameters["model.embed_tokens.weight"] : parameters["lm_head.weight"]
    logits = _flint_linear(head, x)
    if cache !== nothing
        cache.layers = updates
        cache.tokens += length(ids)
    end
    return logits
end

"""Generate one-based tokens with cached prefill/decode. Temperature zero is greedy."""
function generate_flint_tokens(bundle, prompt::AbstractVector{<:Integer}; max_new_tokens::Integer=128, temperature::Real=1f0, top_k=20, top_p=0.95f0, rng=default_rng(), eos_ids=bundle.model.eos_token_id .+ 1)
    max_new_tokens isa Bool && throw(ArgumentError("max_new_tokens cannot be Bool"))
    max_new_tokens >= 0 || throw(ArgumentError("max_new_tokens must be nonnegative"))
    !isempty(prompt) || throw(ArgumentError("Flint prompt must not be empty"))
    length(prompt) + max_new_tokens <= bundle.model.max_position_embeddings || throw(ArgumentError("requested Flint generation exceeds context limit"))
    generated = Int[]
    max_new_tokens == 0 && return generated
    cache = FlintKVCache(bundle.model)
    logits = hf_flint_forward(bundle.model, bundle.parameters, prompt; cache, last_only=true)
    for step in 1:max_new_tokens
        token = _sample_token(vec(Array(logits)), rng; temperature, top_k, top_p)
        push!(generated, token)
        (token in eos_ids || step == max_new_tokens) && break
        logits = hf_flint_forward(bundle.model, bundle.parameters, [token]; cache, last_only=true)
    end
    return generated
end

"""Generate text from a raw prompt or a vector of chat messages using the Flint tokenizer."""
function generate_flint(bundle, tokenizer, prompt; kwargs...)
    text = prompt isa AbstractString ? String(prompt) : apply_flint_chat_template(tokenizer, prompt)
    ids = encode(tokenizer, text)
    output = generate_flint_tokens(bundle, ids; kwargs...)
    return (text=decode(tokenizer, output; skip_special_tokens=true, errors=:replace), token_ids=output)
end
