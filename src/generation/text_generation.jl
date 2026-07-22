using MLDataDevices: cpu_device, get_device
using NNlib: softmax
using Random: AbstractRNG, default_rng, rand

function _sample_categorical(rng::AbstractRNG, probabilities)
    return _sample_categorical(probabilities, rand(rng, Float32))
end

function _sample_categorical(probabilities, uniform::Real)
    isfinite(uniform) && 0 <= uniform < 1 || throw(ArgumentError(
        "`uniform` must be finite and in [0, 1)",
    ))
    threshold = Float32(uniform)
    cumulative = 0.0f0
    last_positive = nothing

    for index in eachindex(probabilities)
        probability = Float32(probabilities[index])
        cumulative += probability
        if probability > 0
            last_positive = index
            threshold <= cumulative && return index
        end
    end

    # Protect against the final cumulative value being 0.99999994 because of
    # floating-point rounding.
    last_positive === nothing && throw(ArgumentError(
        "`probabilities` must contain positive mass",
    ))
    return last_positive
end

function _sampling_distribution(
    logits::AbstractVector;
    temperature::Real=1.0f0,
    top_k=nothing,
    top_p=nothing,
)
    isempty(logits) && throw(ArgumentError("`logits` must not be empty"))
    temperature > 0 || throw(ArgumentError("`temperature` must be positive for sampling"))
    isfinite(temperature) || throw(ArgumentError("`temperature` must be finite"))
    all(isfinite, logits) || throw(ArgumentError("`logits` contains non-finite values"))

    scores = Float32.(logits) ./ Float32(temperature)
    if top_k !== nothing
        top_k isa Integer || throw(ArgumentError("`top_k` must be an integer or nothing"))
        top_k > 0 || throw(ArgumentError("`top_k` must be positive"))
        k = min(Int(top_k), length(scores))
        threshold = partialsort(scores, k; rev=true)
        scores[scores .< threshold] .= Float32(-Inf)
    end

    if top_p !== nothing
        top_p isa Real || throw(ArgumentError("`top_p` must be numeric or nothing"))
        isfinite(top_p) && 0 < top_p <= 1 || throw(ArgumentError(
            "`top_p` must be finite and in (0, 1]",
        ))
        sorted_ids = sortperm(scores; alg=Base.Sort.MergeSort)
        sorted_probabilities = softmax(scores[sorted_ids])
        cumulative = cumsum(sorted_probabilities)
        remove = cumulative .<= (1.0f0 - Float32(top_p))
        remove[end] = false
        scores[sorted_ids[remove]] .= Float32(-Inf)
    end

    probabilities = softmax(scores)
    all(isfinite, probabilities) || throw(ArgumentError(
        "sampling filters removed every candidate token",
    ))
    return scores, probabilities
end

function _sample_token(
    logits::AbstractVector,
    rng::AbstractRNG;
    temperature::Real=1.0f0,
    top_k=nothing,
    top_p=nothing,
    sample_uniform=nothing,
)
    isempty(logits) && throw(ArgumentError("`logits` must not be empty"))
    temperature >= 0 || throw(ArgumentError("`temperature` must be non-negative"))
    all(isfinite, logits) || throw(ArgumentError("`logits` contains non-finite values"))

    if temperature == 0
        return argmax(logits)
    end

    _, probabilities = _sampling_distribution(logits; temperature, top_k, top_p)
    return sample_uniform === nothing ?
        _sample_categorical(rng, probabilities) :
        _sample_categorical(probabilities, sample_uniform)
end

"""
    generate(model, ps, st, prompt_tokens; kwargs...)

Autoregressively generate token ids. The returned token vector includes the prompt.
"""
function generate(
    model,
    ps,
    st,
    prompt_tokens;
    max_new_tokens::Int=100,
    temperature::Real=1.0f0,
    top_k=nothing,
    top_p=nothing,
    rng::AbstractRNG=default_rng(),
    device=get_device(ps),
)
    max_new_tokens >= 0 || throw(ArgumentError(
        "`max_new_tokens` must be non-negative",
    ))

    generated = Int.(collect(prompt_tokens))
    isempty(generated) && throw(ArgumentError("`prompt_tokens` must not be empty"))
    all(id -> 1 <= id <= model.vocab_size, generated) || throw(ArgumentError(
        "prompt token id is outside 1:$(model.vocab_size)",
    ))

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
            top_p,
        )

        push!(generated, next_id)
    end

    return generated, st_current
end

"""
    generate(model, ps, st, tokenizer, prompt; kwargs...)

String convenience overload for every `AbstractTokenizer`. Byte-based generation may
end on an incomplete UTF-8 sequence, so display decoding defaults to `:replace` while
the underlying token ids and `decode_bytes` remain lossless.
"""
function generate(
    model,
    ps,
    st,
    tokenizer::AbstractTokenizer,
    prompt::AbstractString;
    add_special_tokens::Bool=false,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    kwargs...,
)
    prompt_tokens = encode(tokenizer, prompt; add_special_tokens)
    generated_tokens, st_new = generate(
        model,
        ps,
        st,
        prompt_tokens;
        kwargs...,
    )

    return decode(
        tokenizer,
        generated_tokens;
        errors=decode_errors,
        skip_special_tokens,
    ), st_new
end

"""String convenience overload for KV-cached generation with any tokenizer."""
function generate_cached(
    model::GPTModel,
    ps,
    st::NamedTuple,
    tokenizer::AbstractTokenizer,
    prompt::AbstractString;
    add_special_tokens::Bool=false,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    kwargs...,
)
    generated_tokens, st_new = generate_cached(
        model,
        ps,
        st,
        encode(tokenizer, prompt; add_special_tokens);
        kwargs...,
    )
    return decode(
        tokenizer,
        generated_tokens;
        errors=decode_errors,
        skip_special_tokens,
    ), st_new
end

"""String convenience overload for compiled KV-cached generation with any tokenizer."""
function generate_xla_cached!(
    decoder,
    tokenizer::AbstractTokenizer,
    prompt::AbstractString;
    add_special_tokens::Bool=false,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    kwargs...,
)
    generated_tokens, state = generate_xla_cached!(
        decoder,
        encode(tokenizer, prompt; add_special_tokens);
        kwargs...,
    )
    return decode(
        tokenizer,
        generated_tokens;
        errors=decode_errors,
        skip_special_tokens,
    ), state
end
