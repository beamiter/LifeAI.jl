using Lux
using Random: AbstractRNG

"""
    RMSNormLayer(d_model; epsilon=1.0f-5)

Root mean square normalization over the model/channel dimension.

For an input `x` with shape `(d_model, seq_len, batch)`:

```text
y = x / sqrt(mean(abs2, x; dims=1) + epsilon) .* scale
```

The layer deliberately has a learned scale but no bias, matching the RMSNorm
variant commonly used by decoder-only language models.
"""
struct RMSNormLayer <: AbstractLuxLayer
    d_model::Int
    epsilon::Float32
end

function RMSNormLayer(d_model::Int; epsilon::Real=1.0f-5)
    @assert d_model > 0 "`d_model` must be positive"
    @assert epsilon > 0 "`epsilon` must be positive"
    return RMSNormLayer(d_model, Float32(epsilon))
end

function LuxCore.initialparameters(::AbstractRNG, norm::RMSNormLayer)
    return (; scale=ones(Float32, norm.d_model, 1, 1))
end

LuxCore.parameterlength(norm::RMSNormLayer) = norm.d_model

function (norm::RMSNormLayer)(x, ps, st::NamedTuple)
    @assert ndims(x) == 3 "`x` must have shape (d_model, seq_len, batch)"
    @assert size(x, 1) == norm.d_model "input d_model does not match RMSNormLayer"

    value_type = eltype(x)
    mean_square = sum(abs2, x; dims=1) ./ convert(value_type, norm.d_model)
    inverse_rms = one(value_type) ./ sqrt.(
        mean_square .+ convert(value_type, norm.epsilon),
    )
    y = x .* inverse_rms .* ps.scale

    return y, st
end
