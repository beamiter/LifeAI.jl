using Lux
using ConcreteStructs
using NNlib: swish

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
