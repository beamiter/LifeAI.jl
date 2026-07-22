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
      -> final normalization
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
    num_kv_heads::Int
    head_dim::Int
    mlp_hidden_dim::Int
    use_bias::Bool
    is_causal::Bool
    rope_theta::Float32
    norm_epsilon::Float32
    norm_type::Symbol
    mlp_type::Symbol
    tie_embeddings::Bool
    use_qk_norm::Bool
    qk_norm_epsilon::Float32
end

function GPTModel(
    vocab_size::Int,
    d_model::Int,
    num_heads::Int,
    num_layers::Int;
    num_kv_heads::Int=num_heads,
    head_dim=nothing,
    mlp_ratio=nothing,
    mlp_hidden_dim=nothing,
    use_bias::Bool=false,
    is_causal::Bool=true,
    use_rope::Bool=true,
    use_qk_norm::Bool=false,
    qk_norm_epsilon::Real=1.0f-6,
    max_seq_len::Int=2048,
    rope_theta::Real=10000.0,
    norm_epsilon::Real=1.0f-5,
    norm_type::Symbol=:layernorm,
    mlp_type::Symbol=:gelu,
    tie_embeddings::Bool=false,
)
    @assert vocab_size > 0 "`vocab_size` must be positive"
    @assert d_model > 0 "`d_model` must be positive"
    @assert num_heads > 0 "`num_heads` must be positive"
    @assert num_kv_heads > 0 "`num_kv_heads` must be positive"
    @assert num_heads % num_kv_heads == 0 "`num_heads` must be divisible by `num_kv_heads`"
    @assert num_layers > 0 "`num_layers` must be positive"
    @assert max_seq_len > 0 "`max_seq_len` must be positive"
    @assert norm_epsilon > 0 "`norm_epsilon` must be positive"
    @assert qk_norm_epsilon > 0 "`qk_norm_epsilon` must be positive"
    _validate_norm_type(norm_type)
    _validate_mlp_type(mlp_type)

    if mlp_ratio !== nothing
        @assert mlp_ratio > 0 "`mlp_ratio` must be positive"
    end

    resolved_head_dim = if head_dim === nothing
        @assert d_model % num_heads == 0 "`d_model` must be divisible by `num_heads`"
        d_model ÷ num_heads
    else
        Int(head_dim)
    end
    @assert resolved_head_dim > 0 "`head_dim` must be positive"

    resolved_mlp_hidden_dim = _resolve_mlp_hidden_dim(
        d_model,
        mlp_type,
        mlp_ratio,
        mlp_hidden_dim,
    )

    token_embedding = Embedding(vocab_size => d_model)

    blocks = Chain(ntuple(
        _ -> TransformerBlock(
            d_model,
            num_heads;
            num_kv_heads,
            head_dim=resolved_head_dim,
            mlp_hidden_dim=resolved_mlp_hidden_dim,
            use_bias,
            is_causal,
            use_rope,
            use_qk_norm,
            qk_norm_epsilon,
            max_seq_len,
            rope_theta,
            norm_epsilon,
            norm_type,
            mlp_type,
        ),
        num_layers,
    )...)

    # Normalize each token independently over the model/channel dimension.
    final_norm = _make_norm(d_model, norm_type, norm_epsilon)

    lm_head = if tie_embeddings
        TiedOutputProjection(vocab_size, d_model; use_bias)
    else
        Dense(d_model, vocab_size; use_bias)
    end

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
        num_kv_heads,
        resolved_head_dim,
        resolved_mlp_hidden_dim,
        use_bias,
        is_causal,
        Float32(rope_theta),
        Float32(norm_epsilon),
        norm_type,
        mlp_type,
        tie_embeddings,
        use_qk_norm,
        Float32(qk_norm_epsilon),
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
        num_kv_heads=model.num_kv_heads,
        num_layers=model.num_layers,
        head_dim=model.head_dim,
        mlp_hidden_dim=model.mlp_hidden_dim,
        use_bias=model.use_bias,
        is_causal=model.is_causal,
        use_rope=model.use_rope,
        use_qk_norm=model.use_qk_norm,
        qk_norm_epsilon=model.qk_norm_epsilon,
        max_seq_len=model.max_seq_len,
        rope_theta=model.rope_theta,
        norm_epsilon=model.norm_epsilon,
        norm_type=model.norm_type,
        mlp_type=model.mlp_type,
        tie_embeddings=model.tie_embeddings,
    )
end

"""Rebuild a `GPTModel` from a configuration returned by [`gpt_config`](@ref)."""
function GPTModel(config::NamedTuple)
    norm_type = hasproperty(config, :norm_type) ? config.norm_type : :layernorm
    mlp_type = hasproperty(config, :mlp_type) ? config.mlp_type : :gelu
    tie_embeddings = hasproperty(config, :tie_embeddings) ?
        config.tie_embeddings : false

    # Pre-Week-06 configs carry no GQA / QK-norm fields; the defaults reproduce
    # the exact legacy architecture (full KV heads, no QK normalization).
    num_kv_heads = hasproperty(config, :num_kv_heads) ?
        Int(config.num_kv_heads) : Int(config.num_heads)
    use_qk_norm = hasproperty(config, :use_qk_norm) ? config.use_qk_norm : false
    qk_norm_epsilon = hasproperty(config, :qk_norm_epsilon) ?
        config.qk_norm_epsilon : 1.0f-6

    return GPTModel(
        config.vocab_size,
        config.d_model,
        config.num_heads,
        config.num_layers;
        num_kv_heads,
        head_dim=config.head_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        use_bias=config.use_bias,
        is_causal=config.is_causal,
        use_rope=config.use_rope,
        use_qk_norm,
        qk_norm_epsilon,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        norm_epsilon=config.norm_epsilon,
        norm_type,
        mlp_type,
        tie_embeddings,
    )
end

function _project_logits(model::GPTModel, hidden, ps, st_lm_head::NamedTuple)
    if model.tie_embeddings
        return model.lm_head(
            (hidden, ps.token_embedding.weight),
            ps.lm_head,
            st_lm_head,
        )
    end

    return model.lm_head(hidden, ps.lm_head, st_lm_head)
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
    logits, st_lm_head = _project_logits(model, x, ps, st.lm_head)

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
