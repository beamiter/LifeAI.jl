using Lux
using ConcreteStructs
using NNlib: gelu

"""
    TransformerBlock(d_model, num_heads; kwargs...)

A minimal GPT-style pre-norm Transformer block.

Input tensor convention:

    x: (d_model, seq_len, batch)

Structure:

    x -> x + MultiHeadAttention(LayerNorm(x))
      -> x + MLP(LayerNorm(x))

The attention layer follows the existing `MultiHeadAttention` implementation and can
optionally enable RoPE on Q/K.
"""
@concrete struct TransformerBlock <: AbstractLuxContainerLayer{(
    :norm1,
    :attn,
    :norm2,
    :mlp,
)}
    norm1
    attn
    norm2
    mlp

    d_model::Int
    num_heads::Int
    mlp_hidden_dim::Int
    is_causal::Bool
    use_rope::Bool
end

function TransformerBlock(
    d_model::Int,
    num_heads::Int;
    head_dim=nothing,
    mlp_ratio::Real=4,
    mlp_hidden_dim=nothing,
    use_bias::Bool=false,
    is_causal::Bool=true,
    use_rope::Bool=false,
    max_seq_len::Int=2048,
    rope_theta::Real=10000.0,
    norm_epsilon::Real=1.0f-5,
)
    @assert d_model > 0 "`d_model` must be positive"
    @assert num_heads > 0 "`num_heads` must be positive"
    @assert mlp_ratio > 0 "`mlp_ratio` must be positive"
    @assert norm_epsilon > 0 "`norm_epsilon` must be positive"

    if mlp_hidden_dim === nothing
        mlp_hidden_dim = Int(round(d_model * mlp_ratio))
    end

    @assert mlp_hidden_dim > 0 "`mlp_hidden_dim` must be positive"

    # GPT-style pre-norm: normalize over the channel/model dimension only.
    #
    # For x: (d_model, seq_len, batch), dims=1 means each token is normalized
    # independently across its feature dimension. In the current Lux/LuxLib version,
    # parameter arrays must have the same rank as x when dims is specified.
    # Therefore we use (d_model, 1), so scale/bias broadcast over seq_len.
    norm_shape = (d_model, 1)
    norm1 = LayerNorm(norm_shape; epsilon=Float32(norm_epsilon), dims=1)
    norm2 = LayerNorm(norm_shape; epsilon=Float32(norm_epsilon), dims=1)

    attn = MultiHeadAttention(
        d_model,
        num_heads;
        head_dim,
        use_bias,
        is_causal,
        use_rope,
        max_seq_len,
        rope_theta,
    )

    mlp = Chain(
        Dense(d_model, mlp_hidden_dim, gelu; use_bias),
        Dense(mlp_hidden_dim, d_model; use_bias),
    )

    return TransformerBlock(
        norm1,
        attn,
        norm2,
        mlp,
        d_model,
        num_heads,
        mlp_hidden_dim,
        is_causal,
        use_rope,
    )
end

function (block::TransformerBlock)(x, ps, st::NamedTuple)
    @assert ndims(x) == 3 "`x` must have shape (d_model, seq_len, batch)"
    @assert size(x, 1) == block.d_model "input d_model does not match block.d_model"

    # 1. Attention branch: x + Attention(LN(x))
    x_norm1, st_norm1 = block.norm1(x, ps.norm1, st.norm1)
    attn_out, st_attn = block.attn(x_norm1, ps.attn, st.attn)
    x = x .+ attn_out

    # 2. MLP branch: x + MLP(LN(x))
    x_norm2, st_norm2 = block.norm2(x, ps.norm2, st.norm2)
    mlp_out, st_mlp = block.mlp(x_norm2, ps.mlp, st.mlp)
    y = x .+ mlp_out

    return (
        y,
        (;
            norm1=st_norm1,
            attn=st_attn,
            norm2=st_norm2,
            mlp=st_mlp,
        ),
    )
end
