using BFloat16s: BFloat16
using Reactant: @allowscalar, @opcall, @trace

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
    kv_batches = num_kv_heads * batch_size
    q5 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens, head_dim, groups, num_kv_heads, batch_size,
    )
    q3 = reshape(
        permutedims(q5, (1, 3, 2, 4, 5)),
        query_tokens * groups, head_dim, kv_batches,
    )
    k3 = reshape(
        permutedims(keys_cache, (1, 3, 2, 4)),
        head_dim, max_len, kv_batches,
    )
    scores = _bf16a_batched_mul(q3, k3)
    scaled = _bf16a_f32(BFloat16.(_bf16a_f32(scores) .* scaling))
    query_limits = query_tokens == 1 ?
        valid_length : repeat(valid_length; outer=groups)
    visible = reshape(key_positions, 1, max_len, 1) .<= query_limits
    masked = ifelse.(visible, scaled, _BF16_MASK_MIN)
    maxima = maximum(masked; dims=2)
    exponents = exp.(masked .- maxima)
    weights = BFloat16.(exponents ./ sum(exponents; dims=2))

    v3 = reshape(
        permutedims(values_cache, (1, 3, 2, 4)),
        head_dim, max_len, kv_batches,
    )
    context3 = _bf16a_batched_mul(v3, permutedims(weights, (2, 1, 3)))
    context5 = permutedims(
        reshape(
            context3,
            head_dim, query_tokens, groups, num_kv_heads, batch_size,
        ),
        (1, 3, 4, 2, 5),
    )
    return reshape(context5, head_dim, num_heads, query_tokens, batch_size)
end

"""
    _bf16a_pack_decode_projections(ps)

Build host-side QKV and gate/up projection matrices for low-latency decode.
Packing happens once before device transfer; concatenating these weights
inside a traced decode graph would copy hundreds of MiB on every token.
"""
function _bf16a_pack_decode_projections(ps)
    block_names = keys(ps.blocks)
    packed_blocks = map(Tuple(values(ps.blocks))) do ps_block
        (;
            qkv_weight=vcat(
                ps_block.attn.q_proj.weight,
                ps_block.attn.k_proj.weight,
                ps_block.attn.v_proj.weight,
            ),
            gate_up_weight=vcat(
                ps_block.mlp.gate_proj.weight,
                ps_block.mlp.up_proj.weight,
            ),
        )
    end
    logits_weight = hasproperty(ps.lm_head, :weight) ?
        ps.lm_head.weight : permutedims(ps.token_embedding.weight, (2, 1))
    return (;
        blocks=NamedTuple{block_names}(packed_blocks),
        logits_weight,
    )
end

"""
    _bf16a_compact_decode_parameters(ps, packed_projections)

Build the compact parameter tree consumed by packed prefill and decode
executables. Unchanged weights remain references to the already-transferred
model parameters, while the unused separate Q/K/V and gate/up matrices are
omitted from the executable signature. This cuts host/PJRT buffer-handle
processing without duplicating the unchanged device weights.
"""
function _bf16a_compact_decode_parameters(ps, packed_projections)
    block_names = keys(ps.blocks)
    compact_blocks = map(
        Tuple(values(ps.blocks)),
        Tuple(values(packed_projections.blocks)),
    ) do ps_block, packed_block
        (;
            norm1=ps_block.norm1,
            qkv_weight=packed_block.qkv_weight,
            attn=(;
                o_proj=ps_block.attn.o_proj,
                q_norm=ps_block.attn.q_norm,
                k_norm=ps_block.attn.k_norm,
            ),
            norm2=ps_block.norm2,
            gate_up_weight=packed_block.gate_up_weight,
            mlp=(;
                down_proj=ps_block.mlp.down_proj,
            ),
        )
    end
    return (;
        token_embedding=ps.token_embedding,
        blocks=NamedTuple{block_names}(compact_blocks),
        final_norm=ps.final_norm,
        logits_weight=packed_projections.logits_weight,
    )
end

"""
    _bf16a_compact_parameters(ps)

Replace the separate Q/K/V and gate/up projection leaves in a host parameter
tree with their packed equivalents. The returned tree is complete: callers can
transfer it directly and do not need to retain or transfer `ps` or a second
packed-projection tree.
"""
function _bf16a_compact_parameters(ps)
    return _bf16a_compact_decode_parameters(
        ps,
        _bf16a_pack_decode_projections(ps),
    )
end

function _bf16a_static_decode_step_core(
    model::GPTModel,
    ps,
    packed_projections,
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
    packed_blocks = packed_projections === nothing ?
        nothing : Tuple(values(packed_projections.blocks))
    query_dim = head_dim * model.num_heads
    kv_dim = head_dim * model.num_kv_heads
    for index in 1:model.num_layers
        ps_block = block_parameters[index]
        normed = _bf16a_rmsnorm(x, ps_block.norm1.scale, model.norm_epsilon)
        queries, keys, values = if packed_blocks === nothing
            (
                reshape(
                    _bf16a_linear(ps_block.attn.q_proj.weight, normed),
                    head_dim, model.num_heads, 1, 1,
                ),
                reshape(
                    _bf16a_linear(ps_block.attn.k_proj.weight, normed),
                    head_dim, model.num_kv_heads, 1, 1,
                ),
                reshape(
                    _bf16a_linear(ps_block.attn.v_proj.weight, normed),
                    head_dim, model.num_kv_heads, 1, 1,
                ),
            )
        else
            qkv = _bf16a_linear(packed_blocks[index].qkv_weight, normed)
            (
                reshape(
                    qkv[1:query_dim, :, :],
                    head_dim, model.num_heads, 1, 1,
                ),
                reshape(
                    qkv[(query_dim + 1):(query_dim + kv_dim), :, :],
                    head_dim, model.num_kv_heads, 1, 1,
                ),
                reshape(
                    qkv[(query_dim + kv_dim + 1):end, :, :],
                    head_dim, model.num_kv_heads, 1, 1,
                ),
            )
        end
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
        gate, up = if packed_blocks === nothing
            (
                _bf16a_linear(ps_block.mlp.gate_proj.weight, normed2),
                _bf16a_linear(ps_block.mlp.up_proj.weight, normed2),
            )
        else
            gate_up = _bf16a_linear(
                packed_blocks[index].gate_up_weight,
                normed2,
            )
            (
                gate_up[1:model.mlp_hidden_dim, :, :],
                gate_up[(model.mlp_hidden_dim + 1):end, :, :],
            )
        end
        gate_f = _bf16a_f32(gate)
        activated = BFloat16.(gate_f ./ (1.0f0 .+ exp.(.-gate_f)))
        hidden = BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
        mlp_out = _bf16a_linear(ps_block.mlp.down_proj.weight, hidden)
        x = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(mlp_out))
    end
    final_hidden = _bf16a_rmsnorm(x, ps.final_norm.scale, model.norm_epsilon)
    logits_weight = packed_projections === nothing ?
        (
            model.tie_embeddings ?
                permutedims(ps.token_embedding.weight, (2, 1)) :
                ps.lm_head.weight
        ) :
        packed_projections.logits_weight
    return _bf16a_linear(logits_weight, final_hidden)
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
    return _bf16a_static_decode_step_core(
        model,
        ps,
        nothing,
        token,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
    )
end

function _bf16a_static_decode_step_packed(
    model::GPTModel,
    packed_parameters,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
)
    return _bf16a_static_decode_step_core(
        model,
        packed_parameters,
        packed_parameters,
        token,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
    )
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
    return _bf16a_static_decode_greedy_step_core(
        model,
        ps,
        nothing,
        token,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
    )
end

function _bf16a_static_decode_greedy_step_core(
    model::GPTModel,
    ps,
    packed_projections,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
)
    logits = _bf16a_static_decode_step_core(
        model,
        ps,
        packed_projections,
        token,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
    )
    next_token = argmax(vec(_bf16a_f32(logits)))
    token_next = token .* 0 .+ next_token
    position_next = position .+ one(Int32)
    return token_next, position_next
end

function _bf16a_static_decode_greedy_step_packed(
    model::GPTModel,
    packed_parameters,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
)
    return _bf16a_static_decode_greedy_step_core(
        model,
        packed_parameters,
        packed_parameters,
        token,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
    )
end

"""
    _bf16a_static_generate_greedy!(
        model, ps, token, key_caches, value_caches, position,
        cos_table, sin_table, key_positions, generated,
    )

Run the post-prefill greedy decode segment in one PJRT invocation containing a
StableHLO loop. `generated` includes the first token selected by prefill, so a
single host transfer retrieves the entire completion after the loop. This
avoids one PJRT dispatch and synchronization per generated token while
retaining the single-step kernel for streaming callers.
"""
function _bf16a_static_generate_greedy_core!(
    model::GPTModel,
    ps,
    packed_projections,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
    generated,
)
    generated[1:1] = token
    @trace for step in 2:length(generated)
        token, position = _bf16a_static_decode_greedy_step_core(
            model,
            ps,
            packed_projections,
            token,
            key_caches,
            value_caches,
            position,
            cos_table,
            sin_table,
            key_positions,
        )
        @allowscalar generated[step] = sum(token)
    end
    return generated
end

function _bf16a_static_generate_greedy!(
    model::GPTModel,
    ps,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
    generated,
)
    return _bf16a_static_generate_greedy_core!(
        model,
        ps,
        nothing,
        token,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
        generated,
    )
end

function _bf16a_static_generate_greedy_packed!(
    model::GPTModel,
    packed_parameters,
    token,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
    generated,
)
    return _bf16a_static_generate_greedy_core!(
        model,
        packed_parameters,
        packed_parameters,
        token,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
        generated,
    )
end

function _bf16a_static_prefill(
    model::GPTModel,
    ps,
    tokens,
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
        project_last_token_only=true,
    )
    for index in 1:model.num_layers
        keys, values = caches[index]
        key_caches[index][:, :, 1:seq_len, :] = keys
        value_caches[index][:, :, 1:seq_len, :] = values
    end
    return result.logits
end

function _bf16a_update_cache_chunk!(cache, update, first_position)
    first_index = Int(first_position)
    last_index = first_index + size(update, 3) - 1
    copyto!(
        @view(cache[:, :, first_index:last_index, :]),
        update,
    )
    return cache
end

function _bf16a_update_cache_chunk!(
    cache::Reactant.TracedRArray,
    update,
    first_position,
)
    # Reshape/view operations can leave `update` as a lazy traced array wrapper.
    # StableHLO's dynamic_update_slice requires a materialized traced tensor.
    materialized_update = Reactant.materialize_traced_array(update)
    updated = @opcall dynamic_update_slice(
        cache,
        materialized_update,
        Any[1, 1, first_position, 1],
    )
    # Preserve the mutating cache contract expected by compiled callers.
    # Reactant aliases this copy during lowering; the resulting HLO contains
    # one dynamic_update_slice and no stablehlo.copy.
    copyto!(cache, updated)
    return cache
end

"""
    _bf16a_static_prefill_chunk_core(
        model, packed_parameters, tokens, key_caches, value_caches, position,
        cos_table, sin_table, key_positions,
    )

Run one fixed-shape packed prefill chunk against a full-size static cache.
`position` is a one-element Int32 device array containing the number of
physical cache slots already written. `key_positions` contains the visible
position for every cache slot; setting left-padding entries to `typemax(Int32)`
keeps them out of attention while allowing one compiled chunk shape to serve
arbitrary prompt lengths.

The function returns last-token logits and the updated device position. Cache
updates are in-place, matching the single-token static decode kernels.
"""
function _bf16a_static_prefill_chunk_core(
    model::GPTModel,
    packed_parameters,
    tokens,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
)
    seq_len, batch_size = size(tokens)
    batch_size == 1 || throw(ArgumentError(
        "static XLA prefill supports batch size 1 only",
    ))
    head_dim = model.head_dim
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    first_position = sum(position) + one(Int32)
    query_positions = position .+ Int32.(collect(1:seq_len))

    x = reshape(
        gather(packed_parameters.token_embedding.weight, tokens),
        model.d_model,
        seq_len,
        batch_size,
    )
    cos_slice = cos_table[:, query_positions]
    sin_slice = sin_table[:, query_positions]
    block_parameters = Tuple(values(packed_parameters.blocks))
    query_dim = head_dim * model.num_heads
    kv_dim = head_dim * model.num_kv_heads

    for index in 1:model.num_layers
        ps_block = block_parameters[index]
        normed = _bf16a_rmsnorm(
            x,
            ps_block.norm1.scale,
            model.norm_epsilon,
        )
        qkv = _bf16a_linear(ps_block.qkv_weight, normed)
        queries = reshape(
            qkv[1:query_dim, :, :],
            head_dim,
            model.num_heads,
            seq_len,
            batch_size,
        )
        keys = reshape(
            qkv[(query_dim + 1):(query_dim + kv_dim), :, :],
            head_dim,
            model.num_kv_heads,
            seq_len,
            batch_size,
        )
        values = reshape(
            qkv[(query_dim + kv_dim + 1):end, :, :],
            head_dim,
            model.num_kv_heads,
            seq_len,
            batch_size,
        )
        queries = _bf16a_rmsnorm(
            queries,
            ps_block.attn.q_norm.scale,
            model.qk_norm_epsilon,
        )
        keys = _bf16a_rmsnorm(
            keys,
            ps_block.attn.k_norm.scale,
            model.qk_norm_epsilon,
        )
        queries = _bf16a_apply_rope(queries, cos_slice, sin_slice)
        keys = _bf16a_apply_rope(keys, cos_slice, sin_slice)

        _bf16a_update_cache_chunk!(
            key_caches[index],
            keys,
            first_position,
        )
        _bf16a_update_cache_chunk!(
            value_caches[index],
            values,
            first_position,
        )

        context = _bf16a_static_attention(
            queries,
            key_caches[index],
            value_caches[index],
            key_positions,
            query_positions;
            scaling,
        )
        attn_out = _bf16a_linear(
            ps_block.attn.o_proj.weight,
            reshape(
                context,
                head_dim * model.num_heads,
                seq_len,
                batch_size,
            ),
        )
        x = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(attn_out))

        normed2 = _bf16a_rmsnorm(
            x,
            ps_block.norm2.scale,
            model.norm_epsilon,
        )
        gate_up = _bf16a_linear(ps_block.gate_up_weight, normed2)
        gate = gate_up[1:model.mlp_hidden_dim, :, :]
        up = gate_up[(model.mlp_hidden_dim + 1):end, :, :]
        gate_f = _bf16a_f32(gate)
        activated = BFloat16.(gate_f ./ (1.0f0 .+ exp.(.-gate_f)))
        hidden = BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
        mlp_out = _bf16a_linear(ps_block.mlp.down_proj.weight, hidden)
        x = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(mlp_out))
    end

    final_hidden = _bf16a_rmsnorm(
        x[:, end:end, :],
        packed_parameters.final_norm.scale,
        model.norm_epsilon,
    )
    logits = _bf16a_linear(
        packed_parameters.logits_weight,
        final_hidden,
    )
    return logits, position .+ Int32(seq_len)
end

function _bf16a_static_prefill_chunk_greedy(
    model::GPTModel,
    packed_parameters,
    tokens,
    key_caches,
    value_caches,
    position,
    cos_table,
    sin_table,
    key_positions,
)
    logits, next_position = _bf16a_static_prefill_chunk_core(
        model,
        packed_parameters,
        tokens,
        key_caches,
        value_caches,
        position,
        cos_table,
        sin_table,
        key_positions,
    )
    next_token = argmax(vec(_bf16a_f32(logits)))
    token_state = tokens[1:1, 1] .* 0 .+ next_token
    return token_state, next_position
end
