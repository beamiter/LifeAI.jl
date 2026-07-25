using BFloat16s: BFloat16
using NNlib: batched_mul
import MLDataDevices

# Week 15: device-generic vectorized BF16 inference. Same mixed-precision
# contract as `bf16_inference.jl`, expressed only through broadcasts,
# `batched_mul`, gather indexing and reductions, so one implementation serves
# CPU arrays, CuArray (CUBLAS BF16 gemm, F32 accumulation on tensor cores)
# and Reactant tracing. Precision casts use spellings that trace under
# Reactant: `1.0f0 .* x` promotes BF16 -> F32; `BFloat16.(x)` rounds back.

_bf16a_f32(x) = 1.0f0 .* x

function _bf16a_linear(weight::AbstractMatrix, x::AbstractArray{<:Any,3})
    in_dim, num_tokens, batch_size = size(x)
    y = weight * reshape(x, in_dim, :)
    return reshape(y, size(weight, 1), num_tokens, batch_size)
end

# Plain CPU arrays fall back to the Week 14 chunked F32-accumulation kernel:
# the generic BFloat16 matmul would neither match the contract nor be fast.
_bf16a_linear(weight::Matrix{BFloat16}, x::Array{BFloat16,3}) =
    _bf16_linear(weight, x)

# GPU BF16 batched matmul accumulates in Float32 on tensor cores; the CPU
# generic fallback would accumulate in BFloat16 and break the contract, so
# plain arrays route through an explicit Float32 gemm with BF16 rounding.
_bf16a_batched_mul(a::AbstractArray{<:Any,3}, b::AbstractArray{<:Any,3}) =
    batched_mul(a, b)
_bf16a_batched_mul(a::Array{BFloat16,3}, b::Array{BFloat16,3}) =
    BFloat16.(batched_mul(Float32.(a), Float32.(b)))

function _bf16a_rmsnorm(x::AbstractArray, scale, epsilon::Float32)
    xf = _bf16a_f32(x)
    mean_square = sum(abs2, xf; dims=1) ./ Float32(size(x, 1))
    normalized = BFloat16.(xf ./ sqrt.(mean_square .+ epsilon))
    shape = ntuple(i -> i == 1 ? size(x, 1) : 1, ndims(x))
    scale_f = reshape(_bf16a_f32(vec(scale)), shape)
    return BFloat16.(scale_f .* _bf16a_f32(normalized))
end

function _bf16a_apply_rope(
    x::AbstractArray{<:Any,4},
    cos_slice::AbstractMatrix,
    sin_slice::AbstractMatrix,
)
    head_dim, _, num_tokens, _ = size(x)
    half = head_dim ÷ 2
    x1 = _bf16a_f32(x[1:half, :, :, :])
    x2 = _bf16a_f32(x[(half + 1):head_dim, :, :, :])
    cos_b = reshape(_bf16a_f32(cos_slice), half, 1, num_tokens, 1)
    sin_b = reshape(_bf16a_f32(sin_slice), half, 1, num_tokens, 1)
    upper = _bf16a_f32(BFloat16.(x1 .* cos_b)) .+ _bf16a_f32(BFloat16.(.-x2 .* sin_b))
    lower = _bf16a_f32(BFloat16.(x2 .* cos_b)) .+ _bf16a_f32(BFloat16.(x1 .* sin_b))
    return cat(BFloat16.(upper), BFloat16.(lower); dims=1)
end

function _bf16a_attention(
    queries::AbstractArray{<:Any,4},
    keys::AbstractArray{<:Any,4},
    values::AbstractArray{<:Any,4};
    scaling::Float32,
    mask,
)
    head_dim, num_heads, query_tokens, batch_size = size(queries)
    _, num_kv_heads, key_tokens, _ = size(keys)
    groups = num_heads ÷ num_kv_heads
    head_map = ((0:(num_heads - 1)) .÷ groups) .+ 1
    keys_full = keys[:, head_map, :, :]
    values_full = values[:, head_map, :, :]

    q3 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens, head_dim, num_heads * batch_size,
    )
    k3 = reshape(
        permutedims(keys_full, (1, 3, 2, 4)),
        head_dim, key_tokens, num_heads * batch_size,
    )
    scores = _bf16a_batched_mul(q3, k3)
    scaled = BFloat16.(_bf16a_f32(scores) .* scaling)
    if mask !== nothing
        scaled = BFloat16.(_bf16a_f32(scaled) .+ _bf16a_f32(reshape(
            mask, query_tokens, key_tokens, 1,
        )))
    end
    scores_f = _bf16a_f32(scaled)
    maxima = maximum(scores_f; dims=2)
    exponents = exp.(scores_f .- maxima)
    weights = BFloat16.(exponents ./ sum(exponents; dims=2))

    v3 = reshape(
        permutedims(values_full, (1, 3, 2, 4)),
        head_dim, key_tokens, num_heads * batch_size,
    )
    w3 = permutedims(weights, (2, 1, 3))
    context = _bf16a_batched_mul(v3, w3)
    return permutedims(
        reshape(context, head_dim, query_tokens, num_heads, batch_size),
        (1, 3, 2, 4),
    )
end

function _bf16a_block(
    model::GPTModel,
    ps_block,
    x::AbstractArray{<:Any,3},
    cos_slice,
    sin_slice,
    cache;
    mask,
)
    head_dim = model.head_dim
    num_tokens, batch_size = size(x, 2), size(x, 3)
    scaling = 1.0f0 / sqrt(Float32(head_dim))

    normed = _bf16a_rmsnorm(x, ps_block.norm1.scale, model.norm_epsilon)
    queries = reshape(
        _bf16a_linear(ps_block.attn.q_proj.weight, normed),
        head_dim, model.num_heads, num_tokens, batch_size,
    )
    keys = reshape(
        _bf16a_linear(ps_block.attn.k_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    values = reshape(
        _bf16a_linear(ps_block.attn.v_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    queries = _bf16a_rmsnorm(queries, ps_block.attn.q_norm.scale, model.qk_norm_epsilon)
    keys = _bf16a_rmsnorm(keys, ps_block.attn.k_norm.scale, model.qk_norm_epsilon)
    queries = _bf16a_apply_rope(queries, cos_slice, sin_slice)
    keys = _bf16a_apply_rope(keys, cos_slice, sin_slice)

    cached_keys, cached_values = cache
    all_keys = cached_keys === nothing ? keys : cat(cached_keys, keys; dims=3)
    all_values = cached_values === nothing ? values : cat(cached_values, values; dims=3)

    context = _bf16a_attention(queries, all_keys, all_values; scaling, mask)
    attn_out = _bf16a_linear(
        ps_block.attn.o_proj.weight,
        reshape(context, head_dim * model.num_heads, num_tokens, batch_size),
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
    return x, (all_keys, all_values)
end

function _bf16a_causal_mask(query_tokens::Int, key_tokens::Int)
    mask = zeros(Float32, query_tokens, key_tokens)
    offset = key_tokens - query_tokens
    for query_index in 1:query_tokens, key_index in 1:key_tokens
        key_index > query_index + offset &&
            (mask[query_index, key_index] = _BF16_MASK_MIN)
    end
    return BFloat16.(mask)
end

function _bf16a_forward_pass(
    model::GPTModel,
    ps,
    tokens::AbstractMatrix{Int},
    caches,
    cos_table,
    sin_table,
    mask;
    start_pos::Int,
)
    seq_len, batch_size = size(tokens)
    x = reshape(
        ps.token_embedding.weight[:, vec(tokens)],
        model.d_model, seq_len, batch_size,
    )
    embedding = x
    positions = start_pos:(start_pos + seq_len - 1)
    cos_slice = cos_table[:, positions]
    sin_slice = sin_table[:, positions]
    block_parameters = Tuple(values(ps.blocks))
    block_outputs = Vector{Any}(undef, model.num_layers)
    for index in 1:model.num_layers
        x, caches[index] = _bf16a_block(
            model,
            block_parameters[index],
            x,
            cos_slice,
            sin_slice,
            caches[index];
            mask,
        )
        block_outputs[index] = x
    end
    final_hidden = _bf16a_rmsnorm(x, ps.final_norm.scale, model.norm_epsilon)
    logits_weight = model.tie_embeddings ?
        permutedims(ps.token_embedding.weight, (2, 1)) : ps.lm_head.weight
    logits = _bf16a_linear(logits_weight, final_hidden)
    return (; embedding, block_outputs, final_hidden, logits, caches)
end

_bf16a_last_token(logits) = argmax(vec(Array(_bf16a_f32(
    logits[:, end:end, 1:1],
))))

"""
    hf_qwen3_bf16_accel_forward(
        model,
        ps,
        tokens;
        decode_token=nothing,
        greedy_steps=0,
    )

Vectorized, device-generic BF16 forward with the Week 14 mixed-precision
contract. `ps` is a BF16 parameter tree that may live on the CPU or on a
CUDA device (move it with `MLDataDevices.gpu_device()`); all compute happens
on the tree's device and only argmax reduction results are copied back.
Returns the same trace/decode/greedy structure as `hf_qwen3_bf16_forward`.
"""
function hf_qwen3_bf16_accel_forward(
    model::GPTModel,
    ps,
    tokens::AbstractMatrix{<:Integer};
    decode_token=nothing,
    greedy_steps::Int=0,
)
    _qwen3_validate_semantics(model)
    eltype(ps.token_embedding.weight) === BFloat16 || throw(ArgumentError(
        "hf_qwen3_bf16_accel_forward requires a BFloat16 parameter tree",
    ))
    greedy_steps >= 0 || throw(ArgumentError("`greedy_steps` must be non-negative"))
    seq_len, batch_size = size(tokens)
    seq_len > 0 || throw(ArgumentError("`tokens` must contain at least one token"))
    greedy_steps > 0 && batch_size != 1 && throw(ArgumentError(
        "greedy generation supports batch == 1 only",
    ))
    total_extra = max(greedy_steps > 0 ? greedy_steps - 1 : 0, decode_token === nothing ? 0 : 1)
    seq_len + total_extra <= model.max_seq_len || throw(ArgumentError(
        "prompt plus generated context exceeds model.max_seq_len",
    ))

    token_matrix = Int.(collect(tokens))
    _validate_generation_ids(token_matrix, model.vocab_size)
    rope = first(values(model.blocks.layers)).attn.rope
    to_device = _bf16a_device_mover(ps.token_embedding.weight)
    cos_table = to_device(BFloat16.(rope.cos_cache))
    sin_table = to_device(BFloat16.(rope.sin_cache))
    prefill_mask = to_device(_bf16a_causal_mask(seq_len, seq_len))

    caches = Vector{Any}(undef, model.num_layers)
    fill!(caches, (nothing, nothing))
    prefill = _bf16a_forward_pass(
        model, ps, token_matrix, caches, cos_table, sin_table, prefill_mask;
        start_pos=1,
    )

    decode_logits = nothing
    if decode_token !== nothing
        decode_matrix = _decode_token_matrix(decode_token, batch_size)
        _validate_generation_ids(decode_matrix, model.vocab_size)
        decode_caches = Vector{Any}(undef, model.num_layers)
        for index in 1:model.num_layers
            keys, values = prefill.caches[index]
            decode_caches[index] = (copy(keys), copy(values))
        end
        decode = _bf16a_forward_pass(
            model, ps, decode_matrix, decode_caches, cos_table, sin_table,
            nothing;
            start_pos=seq_len + 1,
        )
        decode_logits = decode.logits
    end

    greedy_tokens = Int[]
    if greedy_steps > 0
        logits = prefill.logits
        position = seq_len
        for _ in 1:greedy_steps
            next_token = _bf16a_last_token(logits)
            push!(greedy_tokens, next_token)
            length(greedy_tokens) == greedy_steps && break
            step = _bf16a_forward_pass(
                model,
                ps,
                reshape([next_token], 1, 1),
                prefill.caches,
                cos_table,
                sin_table,
                nothing;
                start_pos=position + 1,
            )
            logits = step.logits
            position += 1
        end
    end

    return (;
        embedding=prefill.embedding,
        blocks=Tuple(prefill.block_outputs),
        final_hidden=prefill.final_hidden,
        logits=prefill.logits,
        decode_logits,
        greedy_tokens,
    )
end

# Move host-side constants (RoPE tables, masks) to wherever the parameters
# live; plain CPU arrays stay put.
_bf16a_device_mover(::Array) = identity
function _bf16a_device_mover(reference::AbstractArray)
    device = MLDataDevices.get_device(reference)
    device === nothing && return identity
    return array -> device(array)
end
