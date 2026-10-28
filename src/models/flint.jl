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

_flint_f32(x) = eltype(x) === Float32 ? x : Float32.(x)

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

# Intermediate model activations are checked at routing and final sampling.
# Keep internal decode GEMMs asynchronous instead of reading a finite flag
# back to the host for every expert projection. The public GEMM keeps checks.
_flint_linear(weight::AbstractNativeNVFP4Weight, x) = native_nvfp4_gemm(weight, x; check_finite=false)

_flint_linear_multi(weights::Tuple, x) = map(weight -> _flint_linear(weight, x), weights)
_flint_linear_multi(weights::NTuple{N,AbstractNativeNVFP4Weight}, x) where {N} =
    native_nvfp4_gemm_multi(weights, x; check_finite=false)

"""
    prepare_flint_nvfp4(bundle; kwargs...)

Prepare all CUDA NVFP4 projections for native FP4 GEMM. The returned bundle
shares floating-point parameters with the input. Native GEMM quantizes each
activation matrix too (W4A4), so results can differ from reference W4A16 mode.
Saving the prepared bundle preserves the original portable NVFP4 weights.
"""
function prepare_flint_nvfp4(bundle; on_progress=nothing, kwargs...)
    parameters = Dict{String,Any}()
    total = length(bundle.parameters)
    for (index, (name, weight)) in enumerate(bundle.parameters)
        parameters[name] = weight isa NVFP4Weight ? prepare_nvfp4_gemm(weight; kwargs...) : weight
        on_progress === nothing || on_progress(name, index, total)
    end
    return merge(bundle, (parameters=parameters, nvfp4_backend=:native,
        nvfp4_activation_scaling=get(kwargs,:activation_scaling,:global)))
end

"""Aggregate evidence of native GEMM use in a prepared Flint bundle."""
function flint_native_fp4_stats(bundle)
    prepared_weights = native_calls = prepared_weight_bytes = shared_workspace_bytes = 0
    gpu_weight_bytes = host_portable_weight_bytes = 0
    for weight in values(bundle.parameters)
        weight isa AbstractNativeNVFP4Weight || continue
        stats = native_nvfp4_stats(weight)
        prepared_weights += 1
        native_calls += Int(get(stats, :native_calls, 0))
        prepared_weight_bytes += Int(get(stats, :prepared_weight_bytes, 0))
        gpu_weight_bytes += Int(get(stats, :gpu_weight_bytes, 0))
        get(stats,:portable_device,:cuda) === :cpu &&
            (host_portable_weight_bytes += Int(get(stats,:portable_weight_bytes,0)))
        shared_workspace_bytes = max(shared_workspace_bytes, Int(get(stats, :workspace_bytes, 0)))
    end
    return (
        backend=iszero(prepared_weights) ? :reference : :cublaslt_nvfp4,
        prepared_weights=prepared_weights, native_calls=native_calls,
        prepared_weight_bytes=prepared_weight_bytes, shared_workspace_bytes=shared_workspace_bytes,
        gpu_weight_bytes=gpu_weight_bytes,host_portable_weight_bytes=host_portable_weight_bytes,
    )
end

function _flint_norm(x, w, epsilon)
    return x .* inv.(sqrt.(sum(abs2, x; dims=1) ./ size(x, 1) .+ Float32(epsilon))) .* reshape(_flint_f32(w), :, ntuple(_ -> 1, ndims(x)-1)...)
end

function _flint_rope(x, positions, theta)
    d = size(x, 1)
    half = d ÷ 2
    frequencies = Float32.(theta) .^ (-Float32.(0:half-1) .* (2f0 / d))
    angles = reshape(frequencies, half, 1) .* reshape(Float32.(positions), 1, :)
    cosine = reshape(_flint_like(x, cos.(angles)), half, 1, :)
    sine = reshape(_flint_like(x, sin.(angles)), half, 1, :)
    a, b = view(x, 1:half, :, :), view(x, half+1:d, :, :)
    return cat(a .* cosine .- b .* sine, b .* cosine .+ a .* sine; dims=1)
end

"""Grouped-query attention without one GEMM and softmax launch per query head."""
function _flint_attention_context(q, k, v, sinks, mask)
    d, heads, n = size(q)
    kvheads, key_tokens = size(k, 2), size(k, 3)
    group = heads ÷ kvheads
    # Fold the query heads sharing each KV head into the matrix columns. This
    # needs only kvheads GEMM batches and never duplicates the KV cache eightfold.
    grouped_q = n == 1 ? reshape(q, d, group, kvheads) :
        reshape(permutedims(reshape(q, d, group, kvheads, n), (1, 2, 4, 3)), d, group*n, kvheads)
    keys = permutedims(k, (3, 1, 2))
    scores = reshape(batched_mul(keys, grouped_q), key_tokens, group, n, kvheads)
    scores .*= inv(sqrt(Float32(d)))
    mask === nothing || (scores .+= reshape(mask, key_tokens, 1, n, 1))
    sink = reshape(_flint_f32(sinks), 1, group, 1, kvheads)
    maxima = max.(maximum(scores; dims=1), sink)
    weights = exp.(scores .- maxima)
    weights ./= sum(weights; dims=1) .+ exp.(sink .- maxima)
    context = batched_mul(permutedims(v, (1, 3, 2)), reshape(weights, key_tokens, group*n, kvheads))
    return n == 1 ? reshape(context, d, heads, 1) :
        reshape(permutedims(reshape(context, d, group, n, kvheads), (1, 2, 4, 3)), d, heads, n)
end

function _flint_attention_long(q, k, v, sinks, first_key_position, query_position, window)
    keys = first_key_position:first_key_position+size(k,3)-1
    queries = query_position:query_position+size(q,3)-1
    allowed = [kp <= qp && (window == 0 || kp > qp-window) for kp in keys, qp in queries]
    mask = all(allowed) ? nothing : _flint_like(q, ifelse.(allowed, 0f0, -Inf32))
    return _flint_attention_context(q,k,v,sinks,mask)
end

# Device extensions can append in reusable storage while preserving these
# logical views and positions. The reference keeps ordinary immutable records.
function _flint_kv_append_reference(k, v, past, position, window::Int; capacity_hint::Int=0)
    n = size(k, 3)
    first_position = position
    if past !== nothing
        k = cat(past.key, k; dims=3)
        v = cat(past.value, v; dims=3)
        first_position = past.first_position
    end
    keep = window > 0 ? min(size(k, 3), window-1) : size(k, 3)
    start = size(k, 3)-keep+1
    updated = keep == size(k, 3) ? (key=k, value=v, first_position=first_position) :
        (key=k[:, :, start:end], value=v[:, :, start:end], first_position=position+n-keep)
    return (key=k, value=v, first_position=first_position, updated=updated)
end
_flint_kv_append(k, v, past, position, window::Int; capacity_hint::Int=0) =
    _flint_kv_append_reference(k, v, past, position, window;capacity_hint)

function _flint_attention(model, parameters, prefix, x, layer, past, position; cache_capacity::Int=0)
    d, heads, kvheads = model.head_dim, model.num_attention_heads, model.num_key_value_heads
    n = size(x, 2)
    q_projection, k_projection, v_projection = _flint_linear_multi((
        parameters[prefix * "q_proj.weight"], parameters[prefix * "k_proj.weight"],
        parameters[prefix * "v_proj.weight"]), x)
    q = _flint_norm(reshape(q_projection, d, heads, n), parameters[prefix * "q_norm.weight"], model.rms_norm_eps)
    k = _flint_norm(reshape(k_projection, d, kvheads, n), parameters[prefix * "k_norm.weight"], model.rms_norm_eps)
    v = reshape(v_projection, d, kvheads, n)
    local_attention = model.layer_types[layer] == "sliding_attention"
    if local_attention
        q = _flint_rope(q, position:position+n-1, model.rope_theta)
        k = _flint_rope(k, position:position+n-1, model.rope_theta)
    end
    window = local_attention ? model.sliding_window : 0
    appended = _flint_kv_append(k, v, past, position, window;capacity_hint=cache_capacity)
    k, v, first_position = appended.key, appended.value, appended.first_position
    sinks = parameters[prefix * "softmax_offset"]
    output = _flint_attention_long(q,k,v,sinks,first_position,position,window)
    return _flint_linear(parameters[prefix * "o_proj.weight"], reshape(output, d*heads, n)), appended.updated
end

function _flint_mlp(parameters, prefix, x)
    gate, up = _flint_linear_multi((parameters[prefix * "gate_proj.weight"],
        parameters[prefix * "up_proj.weight"]), x)
    return _flint_linear(parameters[prefix * "down_proj.weight"], (gate ./ (1f0 .+ exp.(-gate))) .* up)
end

_flint_sample_token(logits, rng; kwargs...) = _sample_token(vec(Array(logits)), rng; kwargs...)

"""Sigmoid top-k routing. Returns one-based expert IDs and normalized scores."""
function flint_topk_routing(logits::AbstractMatrix, k::Integer; normalize::Bool=true)
    0 < k <= size(logits, 1) || throw(ArgumentError("invalid Flint expert top-k"))
    # Host routing is deliberate: the reference path dispatches only used experts.
    host_logits = Array(_flint_f32(logits))
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

function _flint_expert_gate_up_batch(parameters, prefix, x, expert_ids)
    # The routed and shared experts all consume this same normalized token.
    # Keep the portable path for other weight types and unusually large top-k.
    length(expert_ids) <= 7 || return nothing
    prefixes = (map(expert -> prefix * "experts.$(expert-1).", Tuple(expert_ids))...,
        prefix * "shared_expert.")
    projections = map(prefixes) do expert_prefix
        (parameters[expert_prefix * "gate_proj.weight"],
            parameters[expert_prefix * "up_proj.weight"],
            parameters[expert_prefix * "down_proj.weight"])
    end
    all(weights -> all(weight -> weight isa AbstractNativeNVFP4Weight, weights), projections) ||
        return nothing
    shared_inputs = ntuple(2length(projections)) do index
        projections[cld(index,2)][isodd(index) ? 1 : 2]
    end
    return (weights=projections, values=_flint_linear_multi(shared_inputs,x))
end

function _flint_feedforward(model, parameters, prefix, x, layer)
    sparse = !(layer-1 in model.mlp_only_layers) && model.num_experts > 0 && layer % model.decoder_sparse_step == 0
    sparse || return _flint_mlp(parameters, prefix, x)
    logits = _flint_linear(parameters[prefix * "gate.weight"], x)
    routes = flint_topk_routing(logits, model.num_experts_per_tok; normalize=model.norm_topk_prob)
    result = zero(x)
    if size(x, 2) == 1
        # Decode routes contain one column: every selected expert consumes the
        # same activation. Avoid gathering x, uploading one-element factors,
        # and gathering/scattering the sole result column for each expert.
        order = sortperm(vec(routes.expert_ids))
        batched = _flint_expert_gate_up_batch(parameters,prefix,x,routes.expert_ids[order])
        if batched !== nothing
            for (index,route) in enumerate(order)
                gate,up = batched.values[2index-1],batched.values[2index]
                values = _flint_linear(batched.weights[index][3],
                    (gate ./ (1f0 .+ exp.(-gate))) .* up)
                result .+= values .* routes.weights[route]
            end
            index = length(order)+1
            gate,up = batched.values[2index-1],batched.values[2index]
            return result .+ _flint_linear(batched.weights[index][3],
                (gate ./ (1f0 .+ exp.(-gate))) .* up)
        end
        for route in order
            expert = routes.expert_ids[route]
            values = _flint_mlp(parameters, prefix * "experts.$(expert-1).", x)
            result .+= values .* routes.weights[route]
        end
        return result .+ _flint_mlp(parameters, prefix * "shared_expert.", x)
    end
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
    hf_flint_forward(model, parameters, token_ids; cache=nothing, last_only=false,
                     cache_capacity=0, output_logits=true)

Native Float32 Flint reference inference with optional incremental KV caching.
`token_ids` is a nonempty vector of LifeAI's **one-based** IDs. Result is a
`(vocabulary, tokens)` logits matrix (one column if `last_only=true`). Parameters
may contain packed INT4/INT8 matrices. CPU and CUDA arrays are supported; the
reference CUDA path dequantizes only the current projection and routes on CPU.
`cache_capacity` reserves CUDA KV storage without extending the model limit.
`output_logits=false` updates the cache without computing an unused vocabulary
projection, as used by intermediate prefill chunks.
"""
function hf_flint_forward(model::FlintConfig, parameters, token_ids::AbstractVector{<:Integer}; cache::Union{Nothing,FlintKVCache}=nothing, last_only::Bool=false,cache_capacity::Integer=0,output_logits::Bool=true)
    isempty(token_ids) && throw(ArgumentError("Flint input must not be empty"))
    ids = Int.(collect(token_ids))
    eltype(token_ids) <: Bool && throw(ArgumentError("Flint token IDs cannot be Bool"))
    all(id -> 1 <= id <= model.vocab_size, ids) || throw(ArgumentError("Flint token IDs are out of range"))
    position = cache === nothing ? 0 : cache.tokens
    cache === nothing || cache.model === model || throw(ArgumentError("cache belongs to another Flint model"))
    position + length(ids) <= model.max_position_embeddings || throw(ArgumentError("Flint context limit exceeded"))
    !(cache_capacity isa Bool) && 0 <= cache_capacity <= model.max_position_embeddings ||
        throw(ArgumentError("invalid Flint KV reservation"))
    x = permutedims(Float32.(parameters["model.embed_tokens.weight"][ids, :]))
    updates = Any[]
    for layer in 1:model.num_hidden_layers
        prefix = "model.layers.$(layer-1)."
        normalized = _flint_norm(x, parameters[prefix * "input_layernorm.weight"], model.rms_norm_eps)
        past = cache === nothing ? nothing : cache.layers[layer]
        attention, updated = _flint_attention(model, parameters, prefix * "self_attn.", normalized, layer, past, position;cache_capacity=Int(cache_capacity))
        x = x .+ attention
        normalized = _flint_norm(x, parameters[prefix * "post_attention_layernorm.weight"], model.rms_norm_eps)
        x = x .+ _flint_feedforward(model, parameters, prefix * "mlp.", normalized, layer)
        push!(updates, updated)
    end
    if !output_logits
        if cache !== nothing
            cache.layers = updates
            cache.tokens += length(ids)
        end
        return nothing
    end
    last_only && (x = x[:, end:end])
    x = _flint_norm(x, parameters["model.norm.weight"], model.rms_norm_eps)
    head = model.tie_word_embeddings ? parameters["model.embed_tokens.weight"] : parameters["lm_head.weight"]
    logits = _flint_linear(head, x)
    if cache !== nothing
        cache.layers = updates
        cache.tokens += length(ids)
    end
    return logits
end

"""
Generate one-based tokens with cached prefill/decode. Temperature zero is greedy.
`on_token(token_id)` is called immediately after each sampled token (including
EOS). Return `false` to stop early; callback exceptions propagate so disconnected
streaming clients can cancel before the next decode step.
Prefill is processed in bounded `prefill_chunk_size` batches. After each batch,
`on_prefill(processed,total)` reports progress; returning false cancels before
sampling. Its exceptions propagate to stop disconnected streaming requests.
"""
function generate_flint_tokens(bundle, prompt::AbstractVector{<:Integer}; max_new_tokens::Integer=128, temperature::Real=1f0, top_k=20, top_p=0.95f0, rng=default_rng(), eos_ids=bundle.model.eos_token_id .+ 1, on_token=nothing,prefill_chunk_size::Integer=4096,on_prefill=nothing)
    max_new_tokens isa Bool && throw(ArgumentError("max_new_tokens cannot be Bool"))
    max_new_tokens >= 0 || throw(ArgumentError("max_new_tokens must be nonnegative"))
    !isempty(prompt) || throw(ArgumentError("Flint prompt must not be empty"))
    max_new_tokens <= bundle.model.max_position_embeddings-length(prompt) || throw(ArgumentError("requested Flint generation exceeds context limit"))
    !(prefill_chunk_size isa Bool) && prefill_chunk_size > 0 ||
        throw(ArgumentError("prefill_chunk_size must be positive"))
    eltype(prompt) <: Bool && throw(ArgumentError("Flint token IDs cannot be Bool"))
    all(id -> 1 <= id <= bundle.model.vocab_size,prompt) ||
        throw(ArgumentError("Flint token IDs are out of range"))
    generated = Int[]
    max_new_tokens == 0 && return generated
    cache = FlintKVCache(bundle.model)
    capacity = length(prompt)+Int(max_new_tokens)
    logits = nothing
    chunk_size = Int(min(prefill_chunk_size,length(prompt)))
    for first in 1:chunk_size:length(prompt)
        last = min(length(prompt),first+chunk_size-1)
        logits = hf_flint_forward(bundle.model,bundle.parameters,prompt[first:last];cache,
            last_only=true,cache_capacity=capacity,output_logits=(last == length(prompt)))
        on_prefill === nothing || on_prefill(last,length(prompt)) !== false || return generated
    end
    for step in 1:max_new_tokens
        token = _flint_sample_token(logits, rng; temperature, top_k, top_p)
        push!(generated, token)
        on_token === nothing || on_token(token) !== false || break
        (token in eos_ids || step == max_new_tokens) && break
        logits = hf_flint_forward(bundle.model, bundle.parameters, [token]; cache, last_only=true,cache_capacity=capacity)
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
