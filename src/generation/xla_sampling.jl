"""
Device-resident sampling policy.

Week 16 moved greedy `argmax` inside the compiled executable because pulling a
full logits vector back per token costs more than the forward pass itself
(measured 115 ms host round-trip vs 3.4 ms execution on Qwen3-0.6B). Sampling
kept the round-trip: every step transferred all 151,936 logits to the host.

The functions below express the same temperature / top-k / top-p / inverse-CDF
policy as `_sampling_distribution` + `_sample_categorical` using only
elementwise ops, reductions and scalar arithmetic, so one implementation runs
both on ordinary host arrays (reference and tests) and inside a traced XLA
program (deployment).

Deliberate contract differences from the host policy, which follows
HuggingFace and keeps every score tied with the k-th largest:

  * the device policy keeps exactly `top_k` candidates and breaks exact ties by
    the smallest vocabulary index;
  * probability sums run in descending-score order instead of index order, so
    results can differ by float rounding at an exact decision boundary.

Both differences are characterized by the Week 23 tests instead of assumed
away. Randomness stays on the host: the caller passes one uniform per step, so
RNG semantics, seeding and replay are unchanged.
"""

using Random: AbstractRNG, default_rng

"""
    _device_sampling_scalar(x)

Accept a scalar or a one-element (device) array. `sum` is used instead of
indexing because it traces to a reduction without host synchronization.
"""
_device_sampling_scalar(x::Number) = x
_device_sampling_scalar(x::AbstractArray) = sum(x)

_device_sampling_f32(x::AbstractArray) =
    eltype(x) === Float32 ? x : Float32.(x)

"""
    _validate_device_sampling_options(; temperature, top_k, top_p, vocab_size)

Host-side gate for every value that cannot be checked once tracing starts.
`temperature == 0` is rejected instead of silently falling back to greedy: the
device kernel has no branch, and a caller asking for sampling at temperature
zero should choose `:greedy` explicitly.
"""
function _validate_device_sampling_options(;
    temperature,
    top_k,
    top_p,
    vocab_size=nothing,
)
    temperature isa Real || throw(ArgumentError(
        "temperature must be a real number for device sampling",
    ))
    isfinite(temperature) && temperature > 0 || throw(ArgumentError(
        "temperature must be finite and positive for device sampling",
    ))
    top_k isa Integer || throw(ArgumentError(
        "top_k must be an integer for device sampling",
    ))
    top_k > 0 || throw(ArgumentError(
        "top_k must be positive for device sampling",
    ))
    top_p isa Real || throw(ArgumentError(
        "top_p must be a real number for device sampling",
    ))
    isfinite(top_p) && 0 < top_p <= 1 || throw(ArgumentError(
        "top_p must be finite and in (0, 1]",
    ))
    if vocab_size !== nothing
        top_k <= vocab_size || throw(ArgumentError(
            "top_k exceeds the model vocabulary size",
        ))
    end
    return (;
        temperature=Float32(temperature),
        top_k=Int(top_k),
        top_p=Float32(top_p),
    )
end

function _validate_device_sampling_uniform(uniform)
    uniform isa Real || throw(ArgumentError("sample uniform must be a real number"))
    isfinite(uniform) && 0 <= uniform < 1 || throw(ArgumentError(
        "sample uniform must be finite and in [0, 1)",
    ))
    return Float32(uniform)
end

"""
    _device_top_k_candidates(scores, top_k)

Extract the `top_k` largest scores without sorting the vocabulary: each pass is
one `findmax` plus one masked write, which XLA lowers to a reduction and an
elementwise select. A full sort of 151,936 entries per token is avoided
entirely; the cost is `top_k` linear passes.

Returns the descending values, their one-hot masks and their indices.
"""
function _device_top_k_candidates(scores, index_column, top_k::Int)
    values = Vector{Any}(undef, top_k)
    masks = Vector{Any}(undef, top_k)
    indices = Vector{Any}(undef, top_k)
    work = scores
    for slot in 1:top_k
        value = maximum(work)
        # HuggingFace ranks with a stable ascending sort and then reads it
        # backwards, so the highest index wins an exact tie. `findmax` would
        # return the lowest index and silently pick a different token whenever
        # two logits are bit-identical, which BF16 logits make possible.
        index = maximum(ifelse.(work .== value, index_column, 0))
        mask = index_column .== index
        values[slot] = value
        masks[slot] = mask
        indices[slot] = index
        slot == top_k && break
        work = ifelse.(mask, -Inf32, work)
    end
    return values, masks, indices
end

"""
    _device_sample_next_index(logits, uniform, temperature, top_p, top_k)

One traced sampling step. `logits` is the last-token logits vector (any float
element type), `uniform` / `temperature` / `top_p` are scalars or one-element
device arrays, and `top_k` is a compile-time constant because it fixes the
number of extraction passes.

The returned value is the selected 1-based vocabulary index.

Nucleus filtering is expressed without a sort: a candidate survives when the
probability mass ranked strictly above it is below `top_p`, which is the same
rule as removing an ascending-cumulative prefix at `1 - top_p`. The final
inverse-CDF walk runs in vocabulary index order, matching
`_sample_categorical`; only the at most `top_k` surviving candidates can carry
mass, so the walk needs no full-length scan.
"""
function _device_sample_next_index(
    logits,
    uniform,
    temperature,
    top_p,
    top_k::Integer,
)
    scores_input = _device_sampling_f32(vec(logits))
    vocab = length(scores_input)
    vocab > 0 || throw(ArgumentError("logits must not be empty"))
    k = min(Int(top_k), vocab)
    k > 0 || throw(ArgumentError("top_k must be positive"))

    threshold = _device_sampling_scalar(uniform)
    inverse_temperature = _device_sampling_scalar(temperature)
    nucleus = _device_sampling_scalar(top_p)

    scores = scores_input ./ inverse_temperature
    index_column = collect(1:vocab)
    values, _, indices = _device_top_k_candidates(scores, index_column, k)

    # Softmax over the surviving candidates only. Entries outside the top-k set
    # would contribute exp(-Inf) = 0 to the host implementation's sum, so the
    # normalizer is the same up to summation order.
    top_value = values[1]
    weights = Vector{Any}(undef, k)
    for slot in 1:k
        weights[slot] = exp(values[slot] - top_value)
    end
    total = weights[1]
    for slot in 2:k
        total = total + weights[slot]
    end

    # Nucleus rule: keep a candidate while the mass ranked strictly above it is
    # below `top_p`. The largest candidate is always kept, mirroring
    # `remove[end] = false`.
    kept = Vector{Any}(undef, k)
    kept[1] = true
    above = zero(total)
    for slot in 2:k
        above = above + weights[slot - 1] / total
        kept[slot] = above < nucleus
    end

    # Renormalize over the kept candidates, mirroring the host's second
    # `softmax` over the filtered scores.
    kept_weights = Vector{Any}(undef, k)
    for slot in 1:k
        kept_weights[slot] = ifelse(kept[slot], weights[slot], zero(total))
    end
    kept_total = kept_weights[1]
    for slot in 2:k
        kept_total = kept_total + kept_weights[slot]
    end
    probabilities = Vector{Any}(undef, k)
    for slot in 1:k
        probabilities[slot] = kept_weights[slot] / kept_total
    end

    # Inverse-CDF walk in vocabulary index order. Removed candidates carry zero
    # probability, which is exactly the `probability > 0` guard in
    # `_sample_categorical`; no plain/traced boolean mixing is introduced.
    first_hit = vocab + 1
    last_positive = 0
    for slot in 1:k
        cumulative = zero(total)
        for other in 1:k
            cumulative = cumulative + ifelse(
                indices[other] <= indices[slot],
                probabilities[other],
                zero(total),
            )
        end
        positive = probabilities[slot] > zero(total)
        hit = ifelse(positive, threshold <= cumulative, positive)
        first_hit = ifelse(
            ifelse(hit, indices[slot] < first_hit, hit),
            indices[slot],
            first_hit,
        )
        last_positive = ifelse(
            ifelse(positive, indices[slot] > last_positive, positive),
            indices[slot],
            last_positive,
        )
    end

    # A hit always lies at or before the last positive candidate, so this
    # minimum doubles as the "no cumulative reached the threshold" fallback
    # that `_sample_categorical` implements with `last_positive`.
    return ifelse(first_hit <= last_positive, first_hit, last_positive)
end

"""
    device_sample_token(logits, uniform; temperature, top_k, top_p)

Host-side reference entry point for the device sampling policy: validates the
options, then runs the same code path that is traced into the XLA executable.
Used by tests and by callers that want the device policy without a compiled
session.
"""
function device_sample_token(
    logits,
    uniform;
    temperature,
    top_k,
    top_p,
)
    vector = vec(collect(logits))
    isempty(vector) && throw(ArgumentError("logits must not be empty"))
    all(isfinite, vector) || throw(ArgumentError(
        "logits contains non-finite values",
    ))
    options = _validate_device_sampling_options(;
        temperature,
        top_k,
        top_p,
    )
    checked_uniform = _validate_device_sampling_uniform(uniform)
    return _device_sample_next_index(
        Float32.(vector),
        checked_uniform,
        options.temperature,
        options.top_p,
        min(options.top_k, length(vector)),
    )
end

"""
    device_sample_token(logits, rng::AbstractRNG; kwargs...)

Draw the uniform from `rng` exactly like the host sampling path, so switching
strategies does not change how many random numbers a generation consumes.
"""
function device_sample_token(
    logits,
    rng::AbstractRNG;
    temperature,
    top_k,
    top_p,
)
    return device_sample_token(
        logits,
        rand(rng, Float32);
        temperature,
        top_k,
        top_p,
    )
end
