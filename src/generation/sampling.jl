using NNlib: softmax
using Random: AbstractRNG, rand

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

    temperature == 0 && return argmax(logits)

    _, probabilities = _sampling_distribution(logits; temperature, top_k, top_p)
    return sample_uniform === nothing ?
        _sample_categorical(rng, probabilities) :
        _sample_categorical(probabilities, sample_uniform)
end
