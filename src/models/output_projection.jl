using Lux
using Random: AbstractRNG

"""
    TiedOutputProjection(vocab_size, d_model; use_bias=false)

Language-model output projection that reuses the token embedding kernel.

The layer owns no output kernel. Its input is `(hidden, embedding_weight)`, and
the logits are computed with `transpose(embedding_weight) * hidden`. An
independent output bias is optional.
"""
struct TiedOutputProjection <: AbstractLuxLayer
    vocab_size::Int
    d_model::Int
    use_bias::Bool
end

function TiedOutputProjection(
    vocab_size::Int,
    d_model::Int;
    use_bias::Bool=false,
)
    @assert vocab_size > 0 "`vocab_size` must be positive"
    @assert d_model > 0 "`d_model` must be positive"
    return TiedOutputProjection(vocab_size, d_model, use_bias)
end

function LuxCore.initialparameters(::AbstractRNG, head::TiedOutputProjection)
    head.use_bias || return (;)
    return (; bias=zeros(Float32, head.vocab_size))
end

LuxCore.parameterlength(head::TiedOutputProjection) =
    head.use_bias ? head.vocab_size : 0

function (head::TiedOutputProjection)(inputs::Tuple, ps, st::NamedTuple)
    hidden, embedding_weight = inputs

    @assert ndims(hidden) == 3 "`hidden` must have shape (d_model, seq_len, batch)"
    @assert size(hidden, 1) == head.d_model "hidden d_model does not match output head"
    embedding_shape_matches =
        size(embedding_weight) == (head.d_model, head.vocab_size)
    @assert embedding_shape_matches "embedding weight must have shape (d_model, vocab_size)"

    seq_len, batch_size = size(hidden, 2), size(hidden, 3)
    hidden_matrix = reshape(hidden, head.d_model, :)
    logits_matrix = transpose(embedding_weight) * hidden_matrix
    logits = reshape(logits_matrix, head.vocab_size, seq_len, batch_size)

    if head.use_bias
        logits = logits .+ reshape(ps.bias, :, 1, 1)
    end

    return logits, st
end
