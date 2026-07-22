using MLDataDevices: cpu_device, get_device
using Random: AbstractRNG, default_rng, rand

function _hf_greedy_choice(logits, host, capture_logits::Bool)
    values = vec(host(@view(logits[:, end, 1])))
    count = min(2, length(values))
    top_ids = partialsortperm(values, 1:count; rev=true)
    token_id = first(top_ids)
    top_logit = Float32(values[token_id])
    second_id = count == 2 ? top_ids[2] : token_id
    second_logit = Float32(values[second_id])
    return token_id, (;
        token_id,
        hf_token_id=token_id - 1,
        top_logit,
        second_token_id=second_id,
        second_hf_token_id=second_id - 1,
        second_logit,
        margin=top_logit - second_logit,
        logits=capture_logits ? Float32.(collect(values)) : nothing,
    )
end

function _hf_sample_choice(
    logits,
    host,
    rng::AbstractRNG;
    temperature::Real,
    top_k,
    top_p,
    sample_uniform=nothing,
    capture_logits::Bool=false,
    capture_distribution::Bool=false,
)
    values = vec(host(@view(logits[:, end, 1])))
    filtered_logits, probabilities = _sampling_distribution(
        values;
        temperature,
        top_k,
        top_p,
    )
    uniform = sample_uniform === nothing ? rand(rng, Float32) : Float32(sample_uniform)
    token_id = _sample_categorical(probabilities, uniform)
    count = min(2, length(values))
    top_ids = partialsortperm(values, 1:count; rev=true)
    second_id = count == 2 ? top_ids[2] : token_id
    candidate_ids = findall(isfinite, filtered_logits)
    distribution = capture_distribution ? (;
        token_ids=copy(candidate_ids),
        hf_token_ids=candidate_ids .- 1,
        logits=Float32.(filtered_logits[candidate_ids]),
        probabilities=Float32.(probabilities[candidate_ids]),
    ) : nothing
    return token_id, (;
        token_id,
        hf_token_id=token_id - 1,
        top_logit=Float32(values[top_ids[1]]),
        second_token_id=second_id,
        second_hf_token_id=second_id - 1,
        second_logit=Float32(values[second_id]),
        margin=Float32(values[top_ids[1]] - values[second_id]),
        logits=capture_logits ? Float32.(collect(values)) : nothing,
        sample_uniform=uniform,
        sampled_probability=Float32(probabilities[token_id]),
        candidate_count=length(candidate_ids),
        temperature=Float32(temperature),
        top_k=Int(top_k),
        top_p=Float32(top_p),
        distribution,
    )
end

function _hf_generation_limits(model::GPTModel, prompt_length::Int, max_new_tokens::Int)
    max_new_tokens >= 0 || throw(ArgumentError("max_new_tokens must be non-negative"))
    prompt_length > 0 || throw(ArgumentError("prompt must encode to at least one token"))
    processed = prompt_length + max(0, max_new_tokens - 1)
    processed <= model.max_seq_len || throw(ArgumentError(
        "prompt plus generated context exceeds model.max_seq_len",
    ))
    return nothing
end

"""
    generate_hf_text(bundle, prompt; cache=:dynamic, max_new_tokens=32, ...)

Generate from a bundle returned by `load_hf_qwen3_bundle`. `strategy=:config`
uses the validated official sampling settings; `sample_uniforms` can freeze the
categorical draws independently of the Julia/PyTorch RNG implementation. The
result contains prompt/new/all token ids, decoded text, stop reason, and a
per-step trace.
"""
function generate_hf_text(
    bundle,
    prompt::AbstractString;
    cache::Symbol=:dynamic,
    max_new_tokens::Int=32,
    strategy::Symbol=:greedy,
    temperature=nothing,
    top_k=nothing,
    top_p=nothing,
    rng::AbstractRNG=default_rng(),
    sample_uniforms=nothing,
    stop_token_ids=nothing,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    capture_logits::Bool=false,
    capture_distribution::Bool=false,
    device=get_device(bundle.parameters),
)
    strategy in (:greedy, :sample, :config) || throw(ArgumentError(
        "strategy must be :greedy, :sample, or :config",
    ))
    cache in (:full, :dynamic, :static) || throw(ArgumentError(
        "cache must be :full, :dynamic, or :static",
    ))
    model = bundle.model
    tokenizer = bundle.tokenizer
    tokenizer isa HFQwen3Tokenizer || throw(ArgumentError(
        "generate_hf_text requires an HFQwen3Tokenizer bundle",
    ))
    generation_config = hasproperty(bundle, :generation_config) ?
        bundle.generation_config : hf_generation_config(tokenizer)
    resolved_strategy = strategy === :config ?
        (generation_config.do_sample ? :sample : :greedy) : strategy
    sampling_overrides = temperature !== nothing || top_k !== nothing || top_p !== nothing
    resolved_strategy === :sample || (!sampling_overrides && sample_uniforms === nothing) ||
        throw(ArgumentError("sampling parameters require strategy=:sample or :config"))
    resolved_strategy === :sample || !capture_distribution || throw(ArgumentError(
        "capture_distribution requires sampled generation",
    ))
    resolved_temperature = temperature === nothing ? generation_config.temperature : temperature
    resolved_top_k = top_k === nothing ? generation_config.top_k : top_k
    resolved_top_p = top_p === nothing ? generation_config.top_p : top_p
    if resolved_strategy === :sample
        resolved_temperature isa Real && isfinite(resolved_temperature) &&
            resolved_temperature > 0 || throw(ArgumentError(
                "temperature must be a finite positive number",
            ))
        resolved_top_k isa Integer && resolved_top_k > 0 || throw(ArgumentError(
            "top_k must be a positive integer",
        ))
        resolved_top_p isa Real && isfinite(resolved_top_p) &&
            0 < resolved_top_p <= 1 || throw(ArgumentError(
                "top_p must be finite and in (0, 1]",
            ))
    end
    uniforms = sample_uniforms === nothing ? nothing : collect(sample_uniforms)
    uniforms === nothing || length(uniforms) >= max_new_tokens || throw(ArgumentError(
        "sample_uniforms must contain at least max_new_tokens values",
    ))
    uniforms === nothing || all(
        value -> value isa Real && isfinite(value) && 0 <= value < 1,
        uniforms,
    ) || throw(ArgumentError("sample_uniforms values must be finite and in [0, 1)"))
    prompt_ids = encode(tokenizer, prompt; add_special_tokens=false)
    _hf_generation_limits(model, length(prompt_ids), max_new_tokens)
    stops = stop_token_ids === nothing ? Set(tokenizer.eos_ids) : Set(Int.(stop_token_ids))
    all(id -> 1 <= id <= model.vocab_size, stops) || throw(ArgumentError(
        "stop token id is outside the model vocabulary",
    ))

    generated = copy(prompt_ids)
    new_ids = Int[]
    trace = NamedTuple[]
    state = bundle.states
    cache_state = nothing
    stop_reason = :length
    max_new_tokens == 0 && return (;
        prompt_ids,
        generated_ids=new_ids,
        token_ids=generated,
        completion="",
        text=decode(tokenizer, generated; errors=decode_errors, skip_special_tokens),
        stop_reason,
        strategy=resolved_strategy,
        trace=Tuple(trace),
        states=state,
        cache=cache_state,
    )

    host = cpu_device()
    logits = nothing
    if cache === :dynamic
        cache_state = init_kv_cache(model; batch_size=1)
        logits, cache_state, state = prefill(
            model,
            bundle.parameters,
            state,
            generated,
            cache_state;
            device,
        )
    elseif cache === :static
        cache_state = init_static_kv_cache(
            model;
            batch_size=1,
            dtype=eltype(bundle.parameters.token_embedding.weight),
            device,
        )
        logits, cache_state, state = prefill(
            model,
            bundle.parameters,
            state,
            generated,
            cache_state;
            device,
        )
    end

    for step in 1:max_new_tokens
        if cache === :full
            input = device(reshape(generated, length(generated), 1))
            logits, state = model(input, bundle.parameters, state)
        end
        token_id, token_trace = if resolved_strategy === :greedy
            _hf_greedy_choice(logits, host, capture_logits)
        else
            _hf_sample_choice(
                logits,
                host,
                rng;
                temperature=resolved_temperature,
                top_k=resolved_top_k,
                top_p=resolved_top_p,
                sample_uniform=uniforms === nothing ? nothing : uniforms[step],
                capture_logits,
                capture_distribution,
            )
        end
        push!(generated, token_id)
        push!(new_ids, token_id)
        push!(trace, merge((; step), token_trace))
        if token_id in stops
            stop_reason = :eos
            break
        end
        if step < max_new_tokens && cache !== :full
            logits, cache_state, state = decode_step(
                model,
                bundle.parameters,
                state,
                token_id,
                cache_state;
                device,
            )
        end
    end

    return (;
        prompt_ids,
        generated_ids=new_ids,
        token_ids=generated,
        completion=decode(
            tokenizer,
            new_ids;
            errors=decode_errors,
            skip_special_tokens,
        ),
        text=decode(
            tokenizer,
            generated;
            errors=decode_errors,
            skip_special_tokens,
        ),
        stop_reason,
        strategy=resolved_strategy,
        trace=Tuple(trace),
        states=state,
        cache=cache_state,
    )
end
