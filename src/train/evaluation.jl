using Lux
using MLDataDevices: cpu_device, get_device

function _evaluation_positive_int(value, label::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "`$label` must be an integer",
    ))
    converted = try
        Int(value)
    catch error
        error isa InexactError || error isa OverflowError ||
            error isa DomainError || error isa MethodError || rethrow()
        throw(ArgumentError("`$label` is outside the supported integer range"))
    end
    converted > 0 || throw(ArgumentError("`$label` must be positive"))
    return converted
end

function _evaluation_nll(value)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "`total_nll` must be a real number",
    ))
    is_negative = try
        value < 0
    catch error
        error isa MethodError || error isa DomainError ||
            error isa ArgumentError || rethrow()
        throw(ArgumentError(
            "`total_nll` must support a finite sign comparison",
        ))
    end
    is_negative isa Bool || throw(ArgumentError(
        "`total_nll` sign comparison must return Bool",
    ))
    is_negative && throw(ArgumentError(
        "`total_nll` must be non-negative",
    ))
    converted = try
        Float64(value)
    catch error
        error isa InexactError || error isa OverflowError ||
            error isa DomainError || error isa MethodError || rethrow()
        throw(ArgumentError("`total_nll` must be representable as Float64"))
    end
    isfinite(converted) && converted >= 0 || throw(ArgumentError(
        "`total_nll` must be finite and non-negative",
    ))
    if iszero(converted)
        is_nonzero = try
            value != 0
        catch error
            error isa MethodError || error isa DomainError ||
                error isa ArgumentError || rethrow()
            throw(ArgumentError(
                "`total_nll` must support a finite zero comparison",
            ))
        end
        is_nonzero isa Bool || throw(ArgumentError(
            "`total_nll` zero comparison must return Bool",
        ))
        is_nonzero && throw(OverflowError("`total_nll` underflows Float64"))
    end
    return converted
end

function _evaluation_metric_float32(value::Real, label::AbstractString)
    converted = try
        Float32(value)
    catch error
        error isa InexactError || error isa OverflowError ||
            error isa DomainError || error isa MethodError || rethrow()
        throw(ArgumentError(
            "evaluation $label must be representable as Float32",
        ))
    end
    isfinite(converted) || throw(OverflowError(
        "evaluation $label is not representable as finite Float32",
    ))
    if iszero(converted)
        is_nonzero = try
            value != 0
        catch error
            error isa MethodError || error isa DomainError ||
                error isa ArgumentError || rethrow()
            throw(ArgumentError(
                "evaluation $label must support a finite zero comparison",
            ))
        end
        is_nonzero isa Bool || throw(ArgumentError(
            "evaluation $label zero comparison must return Bool",
        ))
        is_nonzero && throw(OverflowError(
            "evaluation $label underflows Float32",
        ))
    end
    return converted
end

function _checked_evaluation_ratio(
    numerator::Float64,
    denominator::Real,
    label::AbstractString,
)
    ratio = numerator / denominator
    isfinite(ratio) || throw(OverflowError(
        "evaluation $label overflowed Float64",
    ))
    iszero(ratio) && !iszero(numerator) && throw(OverflowError(
        "evaluation $label underflows Float64",
    ))
    return ratio
end

"""Convert aggregate negative log likelihood into bits per raw byte."""
function bits_per_byte(total_nll::Real, bytes::Integer)
    resolved_nll = _evaluation_nll(total_nll)
    resolved_bytes = _evaluation_positive_int(bytes, "bytes")
    return _checked_evaluation_ratio(
        resolved_nll,
        log(2.0) * resolved_bytes,
        "bits per byte",
    )
end

"""
    evaluate_gpt(model, ps, st, loader; device=get_device(ps), byte_count=nothing)

Run a no-gradient evaluation loop and aggregate negative log likelihood by target
token count. For `DocumentDatasetLoader`, the exact emitted target-byte denominator is
used automatically; callers of other loader types may pass `byte_count` explicitly.
Public scalar metrics remain Float32 for compatibility; a loss, perplexity, or
byte-normalized value that overflows or underflows that representation raises
`OverflowError` instead of silently becoming `Inf` or zero.
"""
function evaluate_gpt(
    model,
    ps,
    st,
    loader;
    device=get_device(ps),
    byte_count=nothing,
)
    explicit_byte_count = byte_count === nothing ? nothing :
        _evaluation_positive_int(byte_count, "byte_count")
    evaluation_state = Lux.testmode(st)
    total_nll = 0.0
    nll_compensation = 0.0
    total_tokens = 0
    host = cpu_device()

    for (x, targets) in loader
        _validate_token_ids(x, model.vocab_size)
        _validate_target_ids(targets, model.vocab_size)

        x_device, targets_device = device((x, targets))
        logits, evaluation_state = model(
            x_device,
            ps,
            evaluation_state,
        )
        batch_nll = next_token_nll_sum(logits, targets_device)
        batch_tokens = length(targets)

        batch_nll_value = _evaluation_nll(host(batch_nll))
        adjusted_nll = batch_nll_value - nll_compensation
        updated_total_nll = total_nll + adjusted_nll
        isfinite(updated_total_nll) || throw(ArgumentError(
            "evaluation negative log likelihood overflowed Float64",
        ))
        nll_compensation = (updated_total_nll - total_nll) - adjusted_nll
        total_nll = updated_total_nll
        batch_tokens <= typemax(Int) - total_tokens || throw(OverflowError(
            "evaluation target-token count overflow",
        ))
        total_tokens += batch_tokens
    end

    total_tokens > 0 || throw(ArgumentError(
        "evaluation loader produced no target tokens",
    ))
    resolved_byte_count = if explicit_byte_count !== nothing
        explicit_byte_count
    elseif loader isa DocumentDatasetLoader
        automatic_byte_count = target_byte_count(loader)
        automatic_byte_count === nothing ? nothing : _evaluation_positive_int(
            automatic_byte_count,
            "evaluation byte count",
        )
    else
        nothing
    end

    mean_nll = _checked_evaluation_ratio(
        total_nll,
        total_tokens,
        "mean NLL",
    )
    nll_per_byte = resolved_byte_count === nothing ? nothing :
        _checked_evaluation_ratio(
            total_nll,
            resolved_byte_count,
            "NLL per byte",
        )
    bpb = resolved_byte_count === nothing ? nothing : bits_per_byte(total_nll, resolved_byte_count)
    tokens_per_byte = resolved_byte_count === nothing ? nothing : total_tokens / resolved_byte_count
    loss = _evaluation_metric_float32(mean_nll, "loss")
    perplexity = _evaluation_metric_float32(exp(mean_nll), "perplexity")
    metrics = (;
        loss,
        mean_nll=loss,
        perplexity,
        total_nll,
        tokens=total_tokens,
        bytes=resolved_byte_count,
        nll_per_byte=nll_per_byte === nothing ? nothing :
            _evaluation_metric_float32(nll_per_byte, "NLL per byte"),
        bits_per_byte=bpb === nothing ? nothing :
            _evaluation_metric_float32(bpb, "bits per byte"),
        tokens_per_byte=tokens_per_byte === nothing ? nothing :
            _evaluation_metric_float32(tokens_per_byte, "tokens per byte"),
    )

    return metrics, evaluation_state
end
