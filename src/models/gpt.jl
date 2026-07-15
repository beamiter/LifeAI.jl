using Lux
using ConcreteStructs

"""
    GPTModel(vocab_size, d_model, num_heads, num_layers; kwargs...)

A minimal decoder-only GPT model.

Input tensor convention:

    tokens: (seq_len, batch)

Output tensor convention:

    logits: (vocab_size, seq_len, batch)

Structure:

    token ids
      -> token embedding
      -> N × TransformerBlock
      -> final LayerNorm
      -> language-model head
      -> logits

When `use_rope=true`, position information is injected inside every attention
layer, so no separate learned positional embedding is used.
"""
@concrete struct GPTModel <: AbstractLuxContainerLayer{(
    :token_embedding,
    :blocks,
    :final_norm,
    :lm_head,
)}
    token_embedding
    blocks
    final_norm
    lm_head

    vocab_size::Int
    d_model::Int
    num_heads::Int
    num_layers::Int
    max_seq_len::Int
    use_rope::Bool

    # Keep every constructor option needed to reproduce the exact architecture.
    head_dim::Int
    mlp_hidden_dim::Int
    use_bias::Bool
    is_causal::Bool
    rope_theta::Float32
    norm_epsilon::Float32
end

function GPTModel(
    vocab_size::Int,
    d_model::Int,
    num_heads::Int,
    num_layers::Int;
    head_dim=nothing,
    mlp_ratio::Real=4,
    mlp_hidden_dim=nothing,
    use_bias::Bool=false,
    is_causal::Bool=true,
    use_rope::Bool=true,
    max_seq_len::Int=2048,
    rope_theta::Real=10000.0,
    norm_epsilon::Real=1.0f-5,
)
    @assert vocab_size > 0 "`vocab_size` must be positive"
    @assert d_model > 0 "`d_model` must be positive"
    @assert num_heads > 0 "`num_heads` must be positive"
    @assert num_layers > 0 "`num_layers` must be positive"
    @assert max_seq_len > 0 "`max_seq_len` must be positive"
    @assert mlp_ratio > 0 "`mlp_ratio` must be positive"
    @assert norm_epsilon > 0 "`norm_epsilon` must be positive"

    resolved_head_dim = if head_dim === nothing
        @assert d_model % num_heads == 0 "`d_model` must be divisible by `num_heads`"
        d_model ÷ num_heads
    else
        Int(head_dim)
    end
    @assert resolved_head_dim > 0 "`head_dim` must be positive"

    resolved_mlp_hidden_dim = if mlp_hidden_dim === nothing
        Int(round(d_model * mlp_ratio))
    else
        Int(mlp_hidden_dim)
    end
    @assert resolved_mlp_hidden_dim > 0 "`mlp_hidden_dim` must be positive"

    token_embedding = Embedding(vocab_size => d_model)

    blocks = Chain(ntuple(
        _ -> TransformerBlock(
            d_model,
            num_heads;
            head_dim=resolved_head_dim,
            mlp_hidden_dim=resolved_mlp_hidden_dim,
            use_bias,
            is_causal,
            use_rope,
            max_seq_len,
            rope_theta,
            norm_epsilon,
        ),
        num_layers,
    )...)

    # Normalize each token independently over the model/channel dimension.
    final_norm = LayerNorm(
        (d_model, 1);
        epsilon=Float32(norm_epsilon),
        dims=1,
    )

    lm_head = Dense(d_model, vocab_size; use_bias)

    return GPTModel(
        token_embedding,
        blocks,
        final_norm,
        lm_head,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        max_seq_len,
        use_rope,
        resolved_head_dim,
        resolved_mlp_hidden_dim,
        use_bias,
        is_causal,
        Float32(rope_theta),
        Float32(norm_epsilon),
    )
end

"""
    gpt_config(model)

Return the version-stable constructor configuration needed to rebuild `model`.
The returned named tuple intentionally contains architecture only; parameters
and Lux states belong to the checkpoint payload.
"""
function gpt_config(model::GPTModel)
    return (;
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        num_heads=model.num_heads,
        num_layers=model.num_layers,
        head_dim=model.head_dim,
        mlp_hidden_dim=model.mlp_hidden_dim,
        use_bias=model.use_bias,
        is_causal=model.is_causal,
        use_rope=model.use_rope,
        max_seq_len=model.max_seq_len,
        rope_theta=model.rope_theta,
        norm_epsilon=model.norm_epsilon,
    )
end

"""Rebuild a `GPTModel` from a configuration returned by [`gpt_config`](@ref)."""
function GPTModel(config::NamedTuple)
    return GPTModel(
        config.vocab_size,
        config.d_model,
        config.num_heads,
        config.num_layers;
        head_dim=config.head_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        use_bias=config.use_bias,
        is_causal=config.is_causal,
        use_rope=config.use_rope,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        norm_epsilon=config.norm_epsilon,
    )
end

function _validate_token_ids(tokens::Array{T,N}, vocab_size::Int) where {T<:Integer,N}
    @assert all(id -> 1 <= id <= vocab_size, tokens) "token id is outside 1:vocab_size"
    return nothing
end

function _validate_token_ids(tokens::Array, vocab_size::Int)
    @assert false "`tokens` must contain integer token ids"
end

# Device arrays are validated on the host before they enter a compiled train
# step. Avoid data-dependent Julia control flow while Reactant is tracing.
_validate_token_ids(tokens, vocab_size::Int) = nothing

function (model::GPTModel)(tokens, ps, st::NamedTuple)
    @assert ndims(tokens) == 2 "`tokens` must have shape (seq_len, batch)"

    seq_len, _ = size(tokens)

    @assert seq_len > 0 "`tokens` must contain at least one token"
    @assert seq_len <= model.max_seq_len "sequence length exceeds model.max_seq_len"
    _validate_token_ids(tokens, model.vocab_size)

    # Embedding maps (seq_len, batch) -> (d_model, seq_len, batch).
    x, st_token_embedding = model.token_embedding(
        tokens,
        ps.token_embedding,
        st.token_embedding,
    )

    x, st_blocks = model.blocks(x, ps.blocks, st.blocks)
    x, st_final_norm = model.final_norm(x, ps.final_norm, st.final_norm)
    logits, st_lm_head = model.lm_head(x, ps.lm_head, st.lm_head)

    return (
        logits,
        (;
            token_embedding=st_token_embedding,
            blocks=st_blocks,
            final_norm=st_final_norm,
            lm_head=st_lm_head,
        ),
    )
end
