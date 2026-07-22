using MLDataDevices: cpu_device, get_device

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

Greedily generate from a bundle returned by `load_hf_qwen3_bundle`. The result
contains prompt/new/all token ids, decoded completion/full text, stop reason,
and a compact top-two logit trace for every generated token.
"""
function generate_hf_text(
    bundle,
    prompt::AbstractString;
    cache::Symbol=:dynamic,
    max_new_tokens::Int=32,
    strategy::Symbol=:greedy,
    stop_token_ids=nothing,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    capture_logits::Bool=false,
    device=get_device(bundle.parameters),
)
    strategy === :greedy || throw(ArgumentError("Week 08 only supports strategy=:greedy"))
    cache in (:full, :dynamic, :static) || throw(ArgumentError(
        "cache must be :full, :dynamic, or :static",
    ))
    model = bundle.model
    tokenizer = bundle.tokenizer
    tokenizer isa HFQwen3Tokenizer || throw(ArgumentError(
        "generate_hf_text requires an HFQwen3Tokenizer bundle",
    ))
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
        token_id, token_trace = _hf_greedy_choice(logits, host, capture_logits)
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
        trace=Tuple(trace),
        states=state,
        cache=cache_state,
    )
end
