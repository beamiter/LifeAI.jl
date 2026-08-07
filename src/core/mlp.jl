using Lux
using ConcreteStructs
using NNlib: softmax, swish
using Random: AbstractRNG

"""
    SwiGLU(d_model, hidden_dim; use_bias=false)

Gated feed-forward layer:

```text
gate = W_gate * x
up   = W_up * x
y    = W_down * (SiLU(gate) .* up)
```

`hidden_dim` is the shared width of the gate and up projections.
"""
@concrete struct SwiGLU <: AbstractLuxContainerLayer{(
    :gate_proj,
    :up_proj,
    :down_proj,
)}
    gate_proj
    up_proj
    down_proj

    d_model::Int
    hidden_dim::Int
    use_bias::Bool
end

function SwiGLU(
    d_model::Int,
    hidden_dim::Int;
    use_bias::Bool=false,
)
    @assert d_model > 0 "`d_model` must be positive"
    @assert hidden_dim > 0 "`hidden_dim` must be positive"

    return SwiGLU(
        Dense(d_model, hidden_dim; use_bias),
        Dense(d_model, hidden_dim; use_bias),
        Dense(hidden_dim, d_model; use_bias),
        d_model,
        hidden_dim,
        use_bias,
    )
end

function (mlp::SwiGLU)(x, ps, st::NamedTuple)
    gate, st_gate = mlp.gate_proj(x, ps.gate_proj, st.gate_proj)
    up, st_up = mlp.up_proj(x, ps.up_proj, st.up_proj)
    hidden = swish.(gate) .* up
    y, st_down = mlp.down_proj(hidden, ps.down_proj, st.down_proj)

    return (
        y,
        (;
            gate_proj=st_gate,
            up_proj=st_up,
            down_proj=st_down,
        ),
    )
end

"""
    qwen3_topk_routing(router_logits, experts_per_token; normalize=true)

Apply the Qwen3 MoE routing contract to a `(num_experts, num_tokens)` logits
matrix. Softmax is evaluated in Float32, exactly `experts_per_token` routes are
kept for every token, and the selected probabilities are optionally
renormalized. The returned dense routing matrix contains zero for experts that
were not selected.

This is the correctness-first host implementation. It deliberately makes the
top-k boundary explicit; accelerator-specific dispatch can replace it without
changing the expert or checkpoint parameter contract.
"""
function qwen3_topk_routing(
    router_logits::AbstractMatrix,
    experts_per_token::Int;
    normalize::Bool=true,
)
    num_experts, num_tokens = size(router_logits)
    num_experts > 0 || throw(ArgumentError("router must contain at least one expert"))
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "experts_per_token must be in 1:num_experts",
    ))

    probabilities = softmax(Float32.(router_logits); dims=1)
    routing = zeros(Float32, num_experts, num_tokens)
    for token in 1:num_tokens
        selected = partialsortperm(
            view(probabilities, :, token),
            1:experts_per_token;
            rev=true,
        )
        @inbounds routing[selected, token] .= probabilities[selected, token]
        if normalize
            @inbounds routing[selected, token] ./= sum(routing[selected, token])
        end
    end
    return routing
end

"""
    Qwen3SparseMoE(d_model, hidden_dim, num_experts, experts_per_token;
                   normalize_routing=true)

Reference Qwen3 sparse Mixture-of-Experts feed-forward layer. Parameters use a
checkpoint-oriented layout:

```text
gate.weight                 (num_experts, d_model)
experts.gate_proj           (hidden_dim, d_model, num_experts)
experts.up_proj             (hidden_dim, d_model, num_experts)
experts.down_proj           (d_model, hidden_dim, num_experts)
```

The forward pass evaluates all experts and masks unselected outputs. This is
numerically equivalent to sparse token dispatch and keeps the first
implementation simple enough to serve as a parity oracle. A fused/sparse
accelerator dispatch path is intentionally left as a separate optimization.
"""
struct Qwen3SparseMoE <: AbstractLuxLayer
    d_model::Int
    hidden_dim::Int
    num_experts::Int
    experts_per_token::Int
    normalize_routing::Bool
end

function Qwen3SparseMoE(
    d_model::Int,
    hidden_dim::Int,
    num_experts::Int,
    experts_per_token::Int;
    normalize_routing::Bool=true,
)
    d_model > 0 || throw(ArgumentError("d_model must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    num_experts > 0 || throw(ArgumentError("num_experts must be positive"))
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "experts_per_token must be in 1:num_experts",
    ))
    return Qwen3SparseMoE(
        d_model,
        hidden_dim,
        num_experts,
        experts_per_token,
        normalize_routing,
    )
end

function _stack_dense_weights(
    rng::AbstractRNG,
    input_dim::Int,
    output_dim::Int,
    count::Int,
)
    weights = ntuple(count) do _
        parameters = LuxCore.initialparameters(
            rng,
            Dense(input_dim, output_dim; use_bias=false),
        )
        return reshape(parameters.weight, output_dim, input_dim, 1)
    end
    return cat(weights...; dims=3)
end

function LuxCore.initialparameters(rng::AbstractRNG, moe::Qwen3SparseMoE)
    gate = LuxCore.initialparameters(
        rng,
        Dense(moe.d_model, moe.num_experts; use_bias=false),
    )
    experts = (;
        gate_proj=_stack_dense_weights(
            rng,
            moe.d_model,
            moe.hidden_dim,
            moe.num_experts,
        ),
        up_proj=_stack_dense_weights(
            rng,
            moe.d_model,
            moe.hidden_dim,
            moe.num_experts,
        ),
        down_proj=_stack_dense_weights(
            rng,
            moe.hidden_dim,
            moe.d_model,
            moe.num_experts,
        ),
    )
    return (; gate, experts)
end

LuxCore.initialstates(::AbstractRNG, ::Qwen3SparseMoE) = (;)

function LuxCore.parameterlength(moe::Qwen3SparseMoE)
    router = moe.num_experts * moe.d_model
    experts = moe.num_experts * 3 * moe.d_model * moe.hidden_dim
    return router + experts
end

function (moe::Qwen3SparseMoE)(x, ps, st::NamedTuple)
    ndims(x) == 3 || throw(DimensionMismatch(
        "Qwen3SparseMoE input must have shape (d_model, seq_len, batch)",
    ))
    size(x, 1) == moe.d_model || throw(DimensionMismatch(
        "Qwen3SparseMoE input d_model does not match the layer",
    ))

    _, seq_len, batch_size = size(x)
    tokens = reshape(x, moe.d_model, :)
    router_logits = ps.gate.weight * tokens
    routing = qwen3_topk_routing(
        router_logits,
        moe.experts_per_token;
        normalize=moe.normalize_routing,
    )

    output = similar(tokens)
    fill!(output, zero(eltype(output)))
    for expert in 1:moe.num_experts
        gate = view(ps.experts.gate_proj, :, :, expert) * tokens
        up = view(ps.experts.up_proj, :, :, expert) * tokens
        hidden = swish.(gate) .* up
        expert_output = view(ps.experts.down_proj, :, :, expert) * hidden
        weights = reshape(convert.(eltype(expert_output), routing[expert, :]), 1, :)
        output .+= expert_output .* weights
    end

    return reshape(output, moe.d_model, seq_len, batch_size), st
end
