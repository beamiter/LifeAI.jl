using BFloat16s: BFloat16

# Week 16: BF16 static-cache decode designed for Reactant/XLA compilation.
# Mirrors the proven F32 XLA pattern — traced position, dynamic single-column
# RoPE gather, dynamic cache writes, valid-prefix masking over a fixed-shape
# cache — with the Week 14/15 BF16 mixed-precision contract. Batch size 1.
# These functions contain no scalar loops over tensors and mutate the cache
# arrays they are given, so one compiled executable serves every step.

function _bf16a_rope_single(x, cos_table, sin_table, position)
    half = size(x, 1) ÷ 2
    cos_slice = reshape(cos_table[:, position], half, 1)
    sin_slice = reshape(sin_table[:, position], half, 1)
    return _bf16a_apply_rope(x, cos_slice, sin_slice)
end

function _bf16a_static_attention(
    queries,
    keys_cache,
    values_cache,
    key_positions,
    valid_length;
    scaling::Float32,
)
    head_dim, num_heads, query_tokens, batch_size = size(queries)
    _, num_kv_heads, max_len, _ = size(keys_cache)
    groups = num_heads ÷ num_kv_heads
    head_map = ((0:(num_heads - 1)) .÷ groups) .+ 1
    keys_full = keys_cache[:, head_map, :, :]
    values_full = values_cache[:, head_map, :, :]

    q3 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens, head_dim, num_heads * batch_size,
    )
    k3 = reshape(
        permutedims(keys_full, (1, 3, 2, 4)),
        head_dim, max_len, num_heads * batch_size,
    )
    scores = _bf16a_batched_mul(q3, k3)
    scaled = _bf16a_f32(BFloat16.(_bf16a_f32(scores) .* scaling))
    visible = reshape(key_positions, 1, max_len, 1) .<= valid_length
    masked = ifelse.(visible, scaled, _BF16_MASK_MIN)
    maxima = maximum(masked; dims=2)
    exponents = exp.(masked .- maxima)
    weights = BFloat16.(exponents ./ sum(exponents; dims=2))

    v3 = reshape(
        permutedims(values_full, (1, 3, 2, 4)),
        head_dim, max_len, num_heads * batch_size,
    )
    context = _bf16a_batched_mul(v3, permutedims(weights, (2, 1, 3)))
    return permutedims(
        reshape(context, head_dim, query_tokens, num_heads, batch_size),
        (1, 3, 2, 4),
    )
end

function _bf16a_static_decode_step(
    model::GPTModel,
    ps,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
)
    head_dim = model.head_dim
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    # `position` arrives as a 1-element Int32 array so the same compiled
    # executable serves every step and the greedy fast path can feed its
    # returned position straight back with an identical concrete type.
    # `sum` extracts the traced scalar without scalar indexing.
    write_position = sum(position) + one(Int32)

    x = reshape(ps.token_embedding.weight[:, token], model.d_model, 1, 1)
    block_parameters = Tuple(values(ps.blocks))
    for index in 1:model.num_layers
        ps_block = block_parameters[index]
        normed = _bf16a_rmsnorm(x, ps_block.norm1.scale, model.norm_epsilon)
        queries = reshape(
            _bf16a_linear(ps_block.attn.q_proj.weight, normed),
            head_dim, model.num_heads, 1, 1,
        )
        keys = reshape(
            _bf16a_linear(ps_block.attn.k_proj.weight, normed),
            head_dim, model.num_kv_heads, 1, 1,
        )
        values = reshape(
            _bf16a_linear(ps_block.attn.v_proj.weight, normed),
            head_dim, model.num_kv_heads, 1, 1,
        )
        queries = _bf16a_rmsnorm(
            queries, ps_block.attn.q_norm.scale, model.qk_norm_epsilon,
        )
        keys = _bf16a_rmsnorm(
            keys, ps_block.attn.k_norm.scale, model.qk_norm_epsilon,
        )
        queries = _bf16a_rope_single(queries, cos_table, sin_table, write_position)
        keys = _bf16a_rope_single(keys, cos_table, sin_table, write_position)

        key_caches[index][:, :, write_position, :] = dropdims(keys; dims=3)
        value_caches[index][:, :, write_position, :] = dropdims(values; dims=3)

        context = _bf16a_static_attention(
            queries,
            key_caches[index],
            value_caches[index],
            key_positions,
            write_position;
            scaling,
        )
        attn_out = _bf16a_linear(
            ps_block.attn.o_proj.weight,
            reshape(context, head_dim * model.num_heads, 1, 1),
        )
        x = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(attn_out))

        normed2 = _bf16a_rmsnorm(x, ps_block.norm2.scale, model.norm_epsilon)
        gate = _bf16a_linear(ps_block.mlp.gate_proj.weight, normed2)
        up = _bf16a_linear(ps_block.mlp.up_proj.weight, normed2)
        gate_f = _bf16a_f32(gate)
        activated = BFloat16.(gate_f ./ (1.0f0 .+ exp.(.-gate_f)))
        hidden = BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
        mlp_out = _bf16a_linear(ps_block.mlp.down_proj.weight, hidden)
        x = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(mlp_out))
    end
    final_hidden = _bf16a_rmsnorm(x, ps.final_norm.scale, model.norm_epsilon)
    logits_weight = model.tie_embeddings ?
        permutedims(ps.token_embedding.weight, (2, 1)) : ps.lm_head.weight
    return _bf16a_linear(logits_weight, final_hidden)
end

# Device-resident greedy step: argmax runs inside the executable and the next
# token / position feed straight back as inputs, so the host loop only pulls
# one integer per token. Fetching the full logits vector per step costs more
# than the entire forward pass (measured 115 ms vs 3.4 ms on 0.6B).
function _bf16a_static_decode_greedy_step(
    model::GPTModel,
    ps,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
)
    logits = _bf16a_static_decode_step(
        model, ps, token, key_caches, value_caches,
        position, cos_table, sin_table, key_positions,
    )
    next_token = argmax(vec(_bf16a_f32(logits)))
    token_next = token .* 0 .+ next_token
    position_next = position .+ one(Int32)
    return token_next, position_next
end

function _bf16a_static_prefill(
    model::GPTModel,
    ps,
    tokens::AbstractMatrix{Int},
    key_caches,
    value_caches,
    cos_table,
    sin_table,
    mask,
)
    seq_len = size(tokens, 1)
    caches = Vector{Any}(undef, model.num_layers)
    fill!(caches, (nothing, nothing))
    result = _bf16a_forward_pass(
        model, ps, tokens, caches, cos_table, sin_table, mask;
        start_pos=1,
    )
    for index in 1:model.num_layers
        keys, values = caches[index]
        key_caches[index][:, :, 1:seq_len, :] = keys
        value_caches[index][:, :, 1:seq_len, :] = values
    end
    return result.logits
end
