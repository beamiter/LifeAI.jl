using Lux
using ConcreteStructs
using NNlib: gelu

"""
    gelu_new(x)

GPT-2's exact tanh-approximate GELU (`NewGELUActivation` in Transformers).
The constants and operation order intentionally follow the reference formula.
"""
function gelu_new(x)
    return 0.5f0 * x * (
        1.0f0 + tanh(
            sqrt(2.0f0 / Float32(pi)) * (x + 0.044715f0 * x * x * x),
        )
    )
end

gelu_new(x::AbstractArray) = gelu_new.(x)

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
RMSNorm, while `mlp_type` selects GELU, GPT-2 GELU-New, SwiGLU, or Qwen3
sparse MoE.
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
    num_experts::Int
    experts_per_token::Int
    normalize_routing::Bool
end

function _validate_norm_type(norm_type::Symbol)
    norm_type in (:layernorm, :rmsnorm) || throw(ArgumentError(
        "`norm_type` must be `:layernorm` or `:rmsnorm`; got $(repr(norm_type))",
    ))
    return norm_type
end

function _validate_mlp_type(mlp_type::Symbol)
    mlp_type in (:gelu, :gelu_new, :swiglu, :qwen3_moe) || throw(ArgumentError(
        "`mlp_type` must be `:gelu`, `:gelu_new`, `:swiglu`, or `:qwen3_moe`; " *
        "got $(repr(mlp_type))",
    ))
    return mlp_type
end

function _transformer_positive_host_int(value, label::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "`$label` must be an integer",
    ))
    resolved = try
        Int(value)
    catch error
        error isa Union{InexactError,OverflowError,DomainError,MethodError} ||
            rethrow()
        throw(ArgumentError("`$label` is outside the host integer range"))
    end
    resolved > 0 || throw(ArgumentError("`$label` must be positive"))
    return resolved
end

function _validate_mlp_ratio(mlp_ratio)
    mlp_ratio isa Real && !(mlp_ratio isa Bool) &&
        isfinite(mlp_ratio) && mlp_ratio > 0 || throw(ArgumentError(
            "`mlp_ratio` must be a finite positive real number",
        ))
    return mlp_ratio
end

function _transformer_mlp_width(d_model::Int, ratio::Real)
    count = if ratio isa Integer
        BigInt(d_model) * BigInt(ratio)
    elseif ratio isa Rational
        exact_ratio = BigInt(numerator(ratio)) // BigInt(denominator(ratio))
        round(BigInt, BigInt(d_model) * exact_ratio)
    elseif ratio isa AbstractFloat
        product = d_model * ratio
        isfinite(product) || throw(ArgumentError(
            "`mlp_ratio` produces a non-finite MLP width",
        ))
        round(BigInt, product)
    else
        product = BigFloat(d_model) * BigFloat(ratio)
        isfinite(product) || throw(ArgumentError(
            "`mlp_ratio` produces a non-finite MLP width",
        ))
        round(BigInt, product)
    end
    0 < count <= typemax(Int) || throw(ArgumentError(
        "`mlp_hidden_dim` is outside the host integer range",
    ))
    return Int(count)
end

function _resolve_mlp_hidden_dim(
    d_model::Int,
    mlp_type::Symbol,
    mlp_ratio,
    mlp_hidden_dim,
)
    _validate_mlp_type(mlp_type)

    mlp_ratio === nothing || _validate_mlp_ratio(mlp_ratio)

    if mlp_hidden_dim !== nothing
        return _transformer_positive_host_int(
            mlp_hidden_dim,
            "mlp_hidden_dim",
        )
    end

    ratio = if mlp_ratio === nothing
        mlp_type in (:gelu, :gelu_new) ? 4 : 8 // 3
    else
        mlp_ratio
    end
    return _transformer_mlp_width(d_model, ratio)
end

function _transformer_block_parameter_count_int(
    d_model::Int,
    num_heads::Int,
    num_kv_heads::Int,
    head_dim::Int,
    mlp_hidden_dim::Int;
    use_bias::Bool,
    use_qk_norm::Bool,
    norm_type::Symbol,
    mlp_type::Symbol,
    num_experts::Int,
)
    query_dim = _attention_dimension_int(
        BigInt(num_heads) * head_dim,
        "TransformerBlock query dimension",
    )
    kv_dim = _attention_dimension_int(
        BigInt(num_kv_heads) * head_dim,
        "TransformerBlock key/value dimension",
    )
    attention = BigInt(_attention_parameter_count_int(
        d_model,
        query_dim,
        kv_dim,
        head_dim,
        use_bias,
        use_qk_norm,
    ))

    model_width = BigInt(d_model)
    hidden_width = BigInt(mlp_hidden_dim)
    mlp = if mlp_type === :swiglu
        count = 3 * model_width * hidden_width
        use_bias && (count += 2 * hidden_width + model_width)
        count
    elseif mlp_type === :qwen3_moe
        experts = BigInt(num_experts)
        experts * model_width +
            3 * experts * model_width * hidden_width
    else
        count = 2 * model_width * hidden_width
        use_bias && (count += hidden_width + model_width)
        count
    end
    norms = norm_type === :rmsnorm ? 2 * model_width : 4 * model_width
    total = attention + mlp + norms
    0 < total <= typemax(Int) || throw(ArgumentError(
        "TransformerBlock parameter count exceeds the host integer range",
    ))
    return Int(total)
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
    num_experts::Int,
    experts_per_token::Int,
    normalize_routing::Bool,
)
    _validate_mlp_type(mlp_type)
    mlp_type === :qwen3_moe && return Qwen3SparseMoE(
        d_model,
        mlp_hidden_dim,
        num_experts,
        experts_per_token;
        normalize_routing,
    )
    mlp_type === :swiglu && return SwiGLU(
        d_model,
        mlp_hidden_dim;
        use_bias,
    )

    activation = mlp_type === :gelu_new ? gelu_new : gelu
    return Chain(
        Dense(d_model, mlp_hidden_dim, activation; use_bias),
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
    num_experts::Int=0,
    experts_per_token::Int=0,
    normalize_routing::Bool=true,
)
    @assert d_model > 0 "`d_model` must be positive"
    @assert num_heads > 0 "`num_heads` must be positive"
    @assert num_kv_heads > 0 "`num_kv_heads` must be positive"
    @assert num_heads % num_kv_heads == 0 "`num_heads` must be divisible by `num_kv_heads`"
    @assert norm_epsilon > 0 "`norm_epsilon` must be positive"
    _validate_norm_type(norm_type)
    _validate_mlp_type(mlp_type)
    if mlp_type === :qwen3_moe
        num_experts > 0 || throw(ArgumentError("Qwen3 MoE requires num_experts > 0"))
        1 <= experts_per_token <= num_experts || throw(ArgumentError(
            "Qwen3 MoE experts_per_token must be in 1:num_experts",
        ))
        use_bias && throw(ArgumentError("Qwen3 MoE experts are bias-free"))
    else
        num_experts == 0 || throw(ArgumentError(
            "num_experts is only valid with mlp_type=:qwen3_moe",
        ))
        experts_per_token == 0 || throw(ArgumentError(
            "experts_per_token is only valid with mlp_type=:qwen3_moe",
        ))
    end

    resolved_head_dim = if head_dim === nothing
        @assert d_model % num_heads == 0 "`d_model` must be divisible by `num_heads`"
        d_model ÷ num_heads
    else
        _attention_positive_host_int(head_dim, "head_dim")
    end
    resolved_mlp_hidden_dim = _resolve_mlp_hidden_dim(
        d_model,
        mlp_type,
        mlp_ratio,
        mlp_hidden_dim,
    )
    _transformer_block_parameter_count_int(
        d_model,
        num_heads,
        num_kv_heads,
        resolved_head_dim,
        resolved_mlp_hidden_dim;
        use_bias,
        use_qk_norm,
        norm_type,
        mlp_type,
        num_experts,
    )

    # GPT-style pre-norm: normalize each token independently over the model
    # dimension. Both normalization layers use the same selectable semantics.
    norm1 = _make_norm(d_model, norm_type, norm_epsilon)
    norm2 = _make_norm(d_model, norm_type, norm_epsilon)

    attn = MultiHeadAttention(
        d_model,
        num_heads;
        num_kv_heads,
        head_dim=resolved_head_dim,
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
        num_experts,
        experts_per_token,
        normalize_routing,
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
        num_experts,
        experts_per_token,
        normalize_routing,
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
