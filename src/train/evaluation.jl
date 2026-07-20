using Lux
using MLDataDevices: cpu_device, get_device

"""Convert aggregate negative log likelihood into bits per raw byte."""
function bits_per_byte(total_nll::Real, bytes::Integer)
    bytes > 0 || throw(ArgumentError("`bytes` must be positive"))
    return Float64(total_nll) / (log(2.0) * Int(bytes))
end

"""
    evaluate_gpt(model, ps, st, loader; device=get_device(ps), byte_count=nothing)

Run a no-gradient evaluation loop and aggregate negative log likelihood by target
token count. For `DocumentDatasetLoader`, the exact emitted target-byte denominator is
used automatically; callers of other loader types may pass `byte_count` explicitly.
"""
function evaluate_gpt(
    model,
    ps,
    st,
    loader;
    device=get_device(ps),
    byte_count=nothing,
)
    evaluation_state = Lux.testmode(st)
    total_nll = 0.0
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

        total_nll += Float64(host(batch_nll))
        total_tokens += batch_tokens
    end

    total_tokens > 0 || throw(ArgumentError(
        "evaluation loader produced no target tokens",
    ))
    resolved_byte_count = if byte_count === nothing && loader isa DocumentDatasetLoader
        target_byte_count(loader)
    elseif byte_count === nothing
        nothing
    else
        byte_count isa Integer || throw(ArgumentError(
            "`byte_count` must be an integer or nothing",
        ))
        Int(byte_count)
    end
    if resolved_byte_count !== nothing
        resolved_byte_count > 0 || throw(ArgumentError(
            "evaluation byte count must be positive",
        ))
    end

    mean_nll = total_nll / total_tokens
    nll_per_byte = resolved_byte_count === nothing ? nothing : total_nll / resolved_byte_count
    bpb = resolved_byte_count === nothing ? nothing : bits_per_byte(total_nll, resolved_byte_count)
    tokens_per_byte = resolved_byte_count === nothing ? nothing : total_tokens / resolved_byte_count
    metrics = (;
        loss=Float32(mean_nll),
        mean_nll=Float32(mean_nll),
        perplexity=Float32(exp(mean_nll)),
        total_nll,
        tokens=total_tokens,
        bytes=resolved_byte_count,
        nll_per_byte=nll_per_byte === nothing ? nothing : Float32(nll_per_byte),
        bits_per_byte=bpb === nothing ? nothing : Float32(bpb),
        tokens_per_byte=tokens_per_byte === nothing ? nothing : Float32(tokens_per_byte),
    )

    return metrics, evaluation_state
end
