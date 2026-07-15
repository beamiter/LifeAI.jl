using Lux
using MLDataDevices: cpu_device, get_device

"""
    evaluate_gpt(model, ps, st, loader; device=get_device(ps))

Run a no-gradient evaluation loop and aggregate negative log likelihood by token
count rather than by batch count.

Returns `(metrics, evaluation_state)`, where `metrics` contains:

- `loss` / `mean_nll`: token-weighted mean negative log likelihood;
- `perplexity = exp(mean_nll)`;
- `total_nll`;
- `tokens`.
"""
function evaluate_gpt(
    model,
    ps,
    st,
    loader;
    device=get_device(ps),
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

    total_tokens > 0 ||
        throw(ArgumentError("evaluation loader produced no target tokens"))

    mean_nll = total_nll / total_tokens
    metrics = (;
        loss=Float32(mean_nll),
        mean_nll=Float32(mean_nll),
        perplexity=Float32(exp(mean_nll)),
        total_nll,
        tokens=total_tokens,
    )

    return metrics, evaluation_state
end
