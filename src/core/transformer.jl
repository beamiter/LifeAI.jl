using Lux
using ConcreteStructs
using NNlib: gelu

"""
    TransformerBlock(d_model, num_heads; kwargs...)

A configurable GPT-style pre-norm Transformer block.

Input tensor convention:

    x: (d_model, seq_len, batch)

Structure:

    x -> x + MultiHeadAttention(Norm(x))
      -> x + MLP(Norm(x))

The attention layer follows the existing `MultiHeadAttention` implementation and can
optionally enable RoPE on Q/K. `norm_type` independently selects LayerNorm or
RMSNorm, while `mlp_type` selects GELU or SwiGLU.
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
    num_kv_heads::Int
    mlp_hidden_dim::Int
    is_causal::Bool
    use_rope::Bool
    rope_style::Symbol
    use_qk_norm::Bool
    qk_norm_epsilon::Float32
    norm_type::Symbol
    mlp_type::Symbol
    norm_epsilon::Float32
end

function _validate_norm_type(norm_type::Symbol)
    norm_type in (:layernorm, :rmsnorm) || throw(ArgumentError(
        "`norm_type` must be `:layernorm` or `:rmsnorm`; got $(repr(norm_type))",
    ))
    return norm_type
end

function _validate_mlp_type(mlp_type::Symbol)
    mlp_type in (:gelu, :swiglu) || throw(ArgumentError(
        "`mlp_type` must be `:gelu` or `:swiglu`; got $(repr(mlp_type))",
    ))
    return mlp_type
end

function _resolve_mlp_hidden_dim(
    d_model::Int,
    mlp_type::Symbol,
    mlp_ratio,
    mlp_hidden_dim,
)
    _validate_mlp_type(mlp_type)

    if mlp_ratio !== nothing
        @assert mlp_ratio > 0 "`mlp_ratio` must be positive"
    end

    if mlp_hidden_dim !== nothing
        resolved = Int(mlp_hidden_dim)
        @assert resolved > 0 "`mlp_hidden_dim` must be positive"
        return resolved
    end

    ratio = if mlp_ratio === nothing
        mlp_type === :gelu ? 4 : 8 / 3
    else
        mlp_ratio
    end
    resolved = Int(round(d_model * ratio))
    @assert resolved > 0 "`mlp_hidden_dim` must be positive"
    return resolved
end

function _make_norm(
    d_model::Int,
    norm_type::Symbol,
    norm_epsilon::Real,
)
    _validate_norm_type(norm_type)
    norm_type === :rmsnorm && return RMSNormLayer(d_model; epsilon=norm_epsilon)

    return LayerNorm(
        (d_model, 1);
        epsilon=Float32(norm_epsilon),
        dims=1,
    )
end

function _make_mlp(
    d_model::Int,
    mlp_hidden_dim::Int,
    mlp_type::Symbol,
    use_bias::Bool,
)
    _validate_mlp_type(mlp_type)
    mlp_type === :swiglu && return SwiGLU(
        d_model,
        mlp_hidden_dim;
        use_bias,
    )

    return Chain(
        Dense(d_model, mlp_hidden_dim, gelu; use_bias),
        Dense(mlp_hidden_dim, d_model; use_bias),
    )
end

function TransformerBlock(
    d_model::Int,
    num_heads::Int;
    num_kv_heads::Int=num_heads,
    head_dim=nothing,
    mlp_ratio=nothing,
    mlp_hidden_dim=nothing,
    use_bias::Bool=false,
    is_causal::Bool=true,
    use_rope::Bool=false,
    use_qk_norm::Bool=false,
    qk_norm_epsilon::Real=1.0f-6,
    max_seq_len::Int=2048,
    rope_theta::Real=10000.0,
    rope_style::Symbol=:interleaved,
    norm_epsilon::Real=1.0f-5,
    norm_type::Symbol=:layernorm,
    mlp_type::Symbol=:gelu,
)
    @assert d_model > 0 "`d_model` must be positive"
    @assert num_heads > 0 "`num_heads` must be positive"
    @assert norm_epsilon > 0 "`norm_epsilon` must be positive"
    _validate_norm_type(norm_type)
    _validate_mlp_type(mlp_type)

    resolved_mlp_hidden_dim = _resolve_mlp_hidden_dim(
        d_model,
        mlp_type,
        mlp_ratio,
        mlp_hidden_dim,
    )

    # GPT-style pre-norm: normalize each token independently over the model
    # dimension. Both normalization layers use the same selectable semantics.
    norm1 = _make_norm(d_model, norm_type, norm_epsilon)
    norm2 = _make_norm(d_model, norm_type, norm_epsilon)

    attn = MultiHeadAttention(
        d_model,
        num_heads;
        num_kv_heads,
        head_dim,
        use_bias,
        is_causal,
        use_rope,
        use_qk_norm,
        qk_norm_epsilon,
        max_seq_len,
        rope_theta,
        rope_style,
    )

    mlp = _make_mlp(
        d_model,
        resolved_mlp_hidden_dim,
        mlp_type,
        use_bias,
    )

    return TransformerBlock(
        norm1,
        attn,
        norm2,
        mlp,
        d_model,
        num_heads,
        num_kv_heads,
        resolved_mlp_hidden_dim,
        is_causal,
        use_rope,
        rope_style,
        use_qk_norm,
        Float32(qk_norm_epsilon),
        norm_type,
        mlp_type,
        Float32(norm_epsilon),
    )
end

function (block::TransformerBlock)(x, ps, st::NamedTuple)
    @assert ndims(x) == 3 "`x` must have shape (d_model, seq_len, batch)"
    @assert size(x, 1) == block.d_model "input d_model does not match block.d_model"

    # 1. Attention branch: x + Attention(Norm(x))
    x_norm1, st_norm1 = block.norm1(x, ps.norm1, st.norm1)
    attn_out, st_attn = block.attn(x_norm1, ps.attn, st.attn)
    x = x .+ attn_out

    # 2. MLP branch: x + MLP(Norm(x))
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
