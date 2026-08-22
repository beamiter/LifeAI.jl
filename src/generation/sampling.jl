using NNlib: softmax
using Random: AbstractRNG, rand

const MAX_SAMPLING_VOCAB_SIZE = 1_000_000

function _sampling_length(values, label::AbstractString)
    isempty(values) && throw(ArgumentError("`$label` must not be empty"))
    length(values) <= MAX_SAMPLING_VOCAB_SIZE || throw(ArgumentError(
        "`$label` exceeds the supported vocabulary limit of $MAX_SAMPLING_VOCAB_SIZE",
    ))
    return length(values)
end

function _sample_categorical(rng::AbstractRNG, probabilities)
    return _sample_categorical(probabilities, rand(rng, Float32))
end

function _sample_categorical(probabilities, uniform::Real)
    uniform isa Bool && throw(ArgumentError("`uniform` must not be Bool"))
    isfinite(uniform) && 0 <= uniform < 1 || throw(ArgumentError(
        "`uniform` must be finite and in [0, 1)",
    ))
    _sampling_length(probabilities, "probabilities")
    all(probability -> isfinite(probability) && probability >= 0, probabilities) ||
        throw(ArgumentError("`probabilities` must be finite and non-negative"))
    total = sum(Float64, probabilities)
    isapprox(total, 1.0; atol=32eps(Float32), rtol=32eps(Float32)) ||
        throw(ArgumentError("`probabilities` must sum to one"))

    # Keep the explicit-uniform path at Float64 precision. Converting a caller's
    # threshold and Float64 probabilities to Float32 can move a boundary by
    # many representable Float64 values and select the wrong token.
    threshold = Float64(uniform) * total
    cumulative = 0.0
    compensation = 0.0
    last_positive = nothing

    for index in eachindex(probabilities)
        probability = Float64(probabilities[index])
        # Neumaier compensation keeps long, highly skewed distributions from
        # losing small positive masses while preserving one linear pass.
        next = cumulative + probability
        compensation += abs(cumulative) >= abs(probability) ?
            (cumulative - next) + probability :
            (probability - next) + cumulative
        cumulative = next
        if probability > 0
            last_positive = index
            threshold < cumulative + compensation && return index
        end
    end

    # Protect against the final cumulative value being 0.99999994 because of
    # floating-point rounding.
    last_positive === nothing && throw(ArgumentError("`probabilities` must contain positive mass"))
    return last_positive
end

function _sampling_distribution(
    logits::AbstractVector;
    temperature::Real=1.0f0,
    top_k=nothing,
    top_p=nothing,
)
    _sampling_length(logits, "logits")
    temperature isa Bool && throw(ArgumentError("`temperature` must not be Bool"))
    temperature > 0 || throw(ArgumentError("`temperature` must be positive for sampling"))
    isfinite(temperature) || throw(ArgumentError("`temperature` must be finite"))
    all(isfinite, logits) || throw(ArgumentError("`logits` contains non-finite values"))

    resolved_temperature = Float32(temperature)
    isfinite(resolved_temperature) && resolved_temperature > 0 || throw(ArgumentError(
        "`temperature` must remain finite and positive at Float32 sampling precision",
    ))
    scores = Float32.(logits)
    all(isfinite, scores) || throw(ArgumentError(
        "`logits` must remain finite at Float32 sampling precision",
    ))
    scores ./= resolved_temperature
    all(isfinite, scores) || throw(ArgumentError(
        "temperature scaling produced non-finite sampling scores",
    ))
    if top_k !== nothing
        top_k isa Integer && !(top_k isa Bool) || throw(ArgumentError(
            "`top_k` must be an integer other than Bool, or nothing",
        ))
        top_k > 0 || throw(ArgumentError("`top_k` must be positive"))
        top_k <= typemax(Int) || throw(ArgumentError("`top_k` is too large"))
        k = min(Int(top_k), length(scores))
        threshold = partialsort(scores, k; rev=true)
        keep = scores .> threshold
        remaining = k - count(keep)
        if remaining > 0
            for index in eachindex(scores)
                if scores[index] == threshold
                    keep[index] = true
                    remaining -= 1
                    remaining == 0 && break
                end
            end
        end
        scores[.!keep] .= Float32(-Inf)
    end

    if top_p !== nothing
        top_p isa Real && !(top_p isa Bool) || throw(ArgumentError(
            "`top_p` must be a real number other than Bool, or nothing",
        ))
        isfinite(top_p) && 0 < top_p <= 1 || throw(ArgumentError(
            "`top_p` must be finite and in (0, 1]",
        ))
        resolved_top_p = Float32(top_p)
        isfinite(resolved_top_p) && 0 < resolved_top_p <= 1 || throw(ArgumentError(
            "`top_p` must remain finite and positive at Float32 sampling precision",
        ))
        # Stable descending rank plus the first cumulative cutoff implements
        # the smallest nucleus whose probability mass reaches `top_p`.
        # MergeSort makes exact-score ties deterministic by original index.
        sorted_ids = sortperm(scores; rev=true, alg=Base.Sort.MergeSort)
        sorted_probabilities = softmax(scores[sorted_ids])
        cumulative = cumsum(sorted_probabilities)
        cutoff = something(findfirst(>=(resolved_top_p), cumulative), length(cumulative))
        if cutoff < length(sorted_ids)
            scores[sorted_ids[(cutoff + 1):end]] .= Float32(-Inf)
        end
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
    _sampling_length(logits, "logits")
    temperature isa Bool && throw(ArgumentError("`temperature` must not be Bool"))
    temperature >= 0 || throw(ArgumentError("`temperature` must be non-negative"))
    all(isfinite, logits) || throw(ArgumentError("`logits` contains non-finite values"))

    temperature == 0 && return argmax(logits)

    _, probabilities = _sampling_distribution(logits; temperature, top_k, top_p)
    return sample_uniform === nothing ?
        _sample_categorical(rng, probabilities) :
        _sample_categorical(probabilities, sample_uniform)
end
