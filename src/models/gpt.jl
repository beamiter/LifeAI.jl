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
    @assert norm_epsilon > 0 "`norm_epsilon` must be positive"

    token_embedding = Embedding(vocab_size => d_model)

    blocks = Chain(ntuple(
        _ -> TransformerBlock(
            d_model,
            num_heads;
            head_dim,
            mlp_ratio,
            mlp_hidden_dim,
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
    )
end

function (model::GPTModel)(tokens, ps, st::NamedTuple)
    @assert ndims(tokens) == 2 "`tokens` must have shape (seq_len, batch)"

    seq_len, _ = size(tokens)

    @assert seq_len > 0 "`tokens` must contain at least one token"
    @assert seq_len <= model.max_seq_len "sequence length exceeds model.max_seq_len"
    @assert eltype(tokens) <: Integer "`tokens` must contain integer token ids"
    @assert all(id -> 1 <= id <= model.vocab_size, tokens) "token id is outside 1:vocab_size"

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
