using BFloat16s: BFloat16
using NNlib: batched_mul, gather
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
    kv_batches = num_kv_heads * batch_size

    # Keep the GQA replication factor in the query-row dimension instead of
    # gathering K/V from `num_kv_heads` to `num_heads`. This preserves adjacent
    # query-head grouping while avoiding two expanded cache tensors.
    q5 = reshape(
        permutedims(queries, (3, 1, 2, 4)),
        query_tokens, head_dim, groups, num_kv_heads, batch_size,
    )
    q3 = reshape(
        permutedims(q5, (1, 3, 2, 4, 5)),
        query_tokens * groups, head_dim, kv_batches,
    )
    k3 = reshape(
        permutedims(keys, (1, 3, 2, 4)),
        head_dim, key_tokens, kv_batches,
    )
    scores = _bf16a_batched_mul(q3, k3)
    scaled = BFloat16.(_bf16a_f32(scores) .* scaling)
    if mask !== nothing
        scaled_grouped = reshape(
            scaled,
            query_tokens, groups, key_tokens, kv_batches,
        )
        scaled_grouped = BFloat16.(
            _bf16a_f32(scaled_grouped) .+
            _bf16a_f32(reshape(mask, query_tokens, 1, key_tokens, 1))
        )
        scaled = reshape(
            scaled_grouped,
            query_tokens * groups, key_tokens, kv_batches,
        )
    end
    scores_f = _bf16a_f32(scaled)
    maxima = maximum(scores_f; dims=2)
    exponents = exp.(scores_f .- maxima)
    weights = BFloat16.(exponents ./ sum(exponents; dims=2))

    v3 = reshape(
        permutedims(values, (1, 3, 2, 4)),
        head_dim, key_tokens, kv_batches,
    )
    w3 = permutedims(weights, (2, 1, 3))
    context3 = _bf16a_batched_mul(v3, w3)
    context5 = permutedims(
        reshape(
            context3,
            head_dim, query_tokens, groups, num_kv_heads, batch_size,
        ),
        (1, 3, 4, 2, 5),
    )
    return reshape(context5, head_dim, num_heads, query_tokens, batch_size)
end

function _bf16a_input_second_moment(x::AbstractArray{<:Any,3})
    samples = size(x, 2) * size(x, 3)
    samples > 0 || throw(ArgumentError(
        "activation calibration requires at least one token",
    ))
    return vec(sum(abs2, _bf16a_f32(x); dims=(2, 3))) ./ Float32(samples)
end

function _bf16a_block_core(
    model::GPTModel,
    ps_block,
    x::AbstractArray{<:Any,3},
    cos_slice,
    sin_slice,
    cache;
    mask,
    capture_activation_moments::Bool,
)
    head_dim = model.head_dim
    num_tokens, batch_size = size(x, 2), size(x, 3)
    scaling = 1.0f0 / sqrt(Float32(head_dim))

    normed = _bf16a_rmsnorm(x, ps_block.norm1.scale, model.norm_epsilon)
    attention_input_moment = capture_activation_moments ?
        _bf16a_input_second_moment(normed) : nothing
    query_dim = head_dim * model.num_heads
    kv_dim = head_dim * model.num_kv_heads
    queries, keys, values = if hasproperty(ps_block, :qkv_weight)
        qkv = _bf16a_linear(ps_block.qkv_weight, normed)
        (
            reshape(
                qkv[1:query_dim, :, :],
                head_dim, model.num_heads, num_tokens, batch_size,
            ),
            reshape(
                qkv[(query_dim + 1):(query_dim + kv_dim), :, :],
                head_dim, model.num_kv_heads, num_tokens, batch_size,
            ),
            reshape(
                qkv[(query_dim + kv_dim + 1):end, :, :],
                head_dim, model.num_kv_heads, num_tokens, batch_size,
            ),
        )
    else
        (
            reshape(
                _bf16a_linear(ps_block.attn.q_proj.weight, normed),
                head_dim, model.num_heads, num_tokens, batch_size,
            ),
            reshape(
                _bf16a_linear(ps_block.attn.k_proj.weight, normed),
                head_dim, model.num_kv_heads, num_tokens, batch_size,
            ),
            reshape(
                _bf16a_linear(ps_block.attn.v_proj.weight, normed),
                head_dim, model.num_kv_heads, num_tokens, batch_size,
            ),
        )
    end
    queries = _bf16a_rmsnorm(queries, ps_block.attn.q_norm.scale, model.qk_norm_epsilon)
    keys = _bf16a_rmsnorm(keys, ps_block.attn.k_norm.scale, model.qk_norm_epsilon)
    queries = _bf16a_apply_rope(queries, cos_slice, sin_slice)
    keys = _bf16a_apply_rope(keys, cos_slice, sin_slice)

    cached_keys, cached_values = cache
    all_keys = cached_keys === nothing ? keys : cat(cached_keys, keys; dims=3)
    all_values = cached_values === nothing ? values : cat(cached_values, values; dims=3)

    context = _bf16a_attention(queries, all_keys, all_values; scaling, mask)
    output_input = reshape(
        context,
        head_dim * model.num_heads,
        num_tokens,
        batch_size,
    )
    output_input_moment = capture_activation_moments ?
        _bf16a_input_second_moment(output_input) : nothing
    attn_out = _bf16a_linear(
        ps_block.attn.o_proj.weight,
        output_input,
    )
    x = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(attn_out))

    normed2 = _bf16a_rmsnorm(x, ps_block.norm2.scale, model.norm_epsilon)
    mlp_input_moment = capture_activation_moments ?
        _bf16a_input_second_moment(normed2) : nothing
    gate, up = if hasproperty(ps_block, :gate_up_weight)
        gate_up = _bf16a_linear(ps_block.gate_up_weight, normed2)
        (
            gate_up[1:model.mlp_hidden_dim, :, :],
            gate_up[(model.mlp_hidden_dim + 1):end, :, :],
        )
    else
        (
            _bf16a_linear(ps_block.mlp.gate_proj.weight, normed2),
            _bf16a_linear(ps_block.mlp.up_proj.weight, normed2),
        )
    end
    gate_f = _bf16a_f32(gate)
    activated = BFloat16.(gate_f ./ (1.0f0 .+ exp.(.-gate_f)))
    hidden = BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
    down_input_moment = capture_activation_moments ?
        _bf16a_input_second_moment(hidden) : nothing
    mlp_out = _bf16a_linear(ps_block.mlp.down_proj.weight, hidden)
    x = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(mlp_out))
    moments = if capture_activation_moments
        (;
            q_proj=attention_input_moment,
            k_proj=attention_input_moment,
            v_proj=attention_input_moment,
            o_proj=output_input_moment,
            gate_proj=mlp_input_moment,
            up_proj=mlp_input_moment,
            down_proj=down_input_moment,
        )
    else
        nothing
    end
    return x, (all_keys, all_values), moments
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
    output, updated_cache, _ = _bf16a_block_core(
        model,
        ps_block,
        x,
        cos_slice,
        sin_slice,
        cache;
        mask,
        capture_activation_moments=false,
    )
    return output, updated_cache
end

function _bf16a_block_activation_moments(
    model::GPTModel,
    ps_block,
    x::AbstractArray{<:Any,3},
    cos_slice,
    sin_slice;
    mask,
)
    return _bf16a_block_core(
        model,
        ps_block,
        x,
        cos_slice,
        sin_slice,
        (nothing, nothing);
        mask,
        capture_activation_moments=true,
    )
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
    tokens,
    caches,
    cos_table,
    sin_table,
    mask;
    start_pos::Int,
    project_last_token_only::Bool=false,
)
    seq_len, batch_size = size(tokens)
    x = reshape(
        gather(ps.token_embedding.weight, tokens),
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
    logits_weight = if hasproperty(ps, :logits_weight)
        ps.logits_weight
    elseif model.tie_embeddings
        permutedims(ps.token_embedding.weight, (2, 1))
    else
        ps.lm_head.weight
    end
    projection_input = project_last_token_only ?
        final_hidden[:, end:end, :] : final_hidden
    logits = _bf16a_linear(logits_weight, projection_input)
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
