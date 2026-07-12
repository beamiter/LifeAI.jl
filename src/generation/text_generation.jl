using MLDataDevices: cpu_device, get_device
using NNlib: softmax
using Random: AbstractRNG, default_rng, rand

function _sample_categorical(rng::AbstractRNG, probabilities)
    threshold = rand(rng, Float32)
    cumulative = 0.0f0

    for index in eachindex(probabilities)
        cumulative += Float32(probabilities[index])
        threshold <= cumulative && return index
    end

    # Protect against the final cumulative value being 0.99999994 because of
    # floating-point rounding.
    return lastindex(probabilities)
end

function _sample_token(
    logits::AbstractVector,
    rng::AbstractRNG;
    temperature::Real=1.0f0,
    top_k=nothing,
)
    isempty(logits) && throw(ArgumentError("`logits` must not be empty"))
    temperature >= 0 || throw(ArgumentError("`temperature` must be non-negative"))
    all(isfinite, logits) || throw(ArgumentError("`logits` contains non-finite values"))

    if temperature == 0
        return argmax(logits)
    end

    scores = Float32.(logits) ./ Float32(temperature)

    if top_k !== nothing
        top_k isa Integer || throw(ArgumentError("`top_k` must be an integer or nothing"))
        top_k > 0 || throw(ArgumentError("`top_k` must be positive"))

        k = min(Int(top_k), length(scores))
        candidate_ids = partialsortperm(scores, 1:k; rev=true)
        probabilities = softmax(scores[candidate_ids])

        return candidate_ids[_sample_categorical(rng, probabilities)]
    end

    probabilities = softmax(scores)
    return _sample_categorical(rng, probabilities)
end

"""
    generate(model, ps, st, prompt_tokens; kwargs...)

Autoregressively generate token ids.

Keyword arguments:

- `max_new_tokens=100`
- `temperature=1f0`; use `0` for greedy decoding
- `top_k=nothing`
- `rng=Random.default_rng()`
- `device=get_device(ps)`

The returned token vector includes the prompt. The second return value is the
updated Lux model state.
"""
function generate(
    model,
    ps,
    st,
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
    all(id -> 1 <= id <= model.vocab_size, generated) ||
        throw(ArgumentError("prompt token id is outside 1:$(model.vocab_size)"))

    st_current = st
    host = cpu_device()

    for _ in 1:max_new_tokens
        context_start = max(1, length(generated) - model.max_seq_len + 1)
        context = generated[context_start:end]
        input_tokens = device(reshape(context, length(context), 1))

        logits, st_current = model(
            input_tokens,
            ps,
            st_current,
        )

        last_logits = vec(host(@view(logits[:, end, 1])))
        next_id = _sample_token(
            last_logits,
            rng;
            temperature,
            top_k,
        )

        push!(generated, next_id)
    end

    return generated, st_current
end

"""
    generate(model, ps, st, tokenizer, prompt; kwargs...)

String convenience overload. Returns `(generated_text, updated_state)`.
"""
function generate(
    model,
    ps,
    st,
    tokenizer::Tokenizer,
    prompt::AbstractString;
    kwargs...
)
    prompt_tokens = encode(tokenizer, prompt)
    generated_tokens, st_new = generate(
        model,
        ps,
        st,
        prompt_tokens;
        kwargs...
    )

    return decode(tokenizer, generated_tokens), st_new
end
