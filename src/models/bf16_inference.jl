using BFloat16s: BFloat16
using LinearAlgebra

# Week 14: native BF16 mixed-precision inference. This is a deliberately
# separate path from the Float32 forward/KV-cache/XLA code: it mirrors the
# Transformers 4.51.0 Qwen3 BF16 semantics operator by operator —
#   * RMSNorm / QK-Norm normalize in Float32, cast back, multiply in BF16;
#   * linears keep BF16 operands, accumulate in Float32, round the output;
#   * attention scores stay BF16, softmax runs in Float32, is cast back;
#   * RoPE tables are Float32, cast to BF16, applied elementwise in BF16.
# Elementwise BF16 arithmetic rounds after every operation, matching torch
# opmath behaviour, so promotion surprises are avoided by writing each
# rounding step explicitly.

# torch.finfo(torch.bfloat16).min — additive causal mask value used by HF.
const _BF16_MASK_MIN = -3.3895314f38

_bf16_matmul(a::AbstractMatrix, b::AbstractMatrix) =
    BFloat16.(Float32.(a) * Float32.(b))

"""
    _bf16_linear(weight, x)

BF16 linear projection over `(in, tokens, batch)` activations with Float32
accumulation and BF16 output rounding. The weight is upconverted in row
chunks so the Float32 transient stays tens of MiB even for multi-GiB
matrices — with the parameter tree already occupying most of the RAM
budget, a whole-matrix Float32 copy is what pushed 8B past the OOM line.
"""
function _bf16_linear(weight::AbstractMatrix{BFloat16}, x::AbstractArray{BFloat16,3})
    in_dim, num_tokens, batch_size = size(x)
    out_dim = size(weight, 1)
    size(weight, 2) == in_dim || throw(DimensionMismatch(
        "linear input dimension does not match weight",
    ))
    x_f32 = reshape(Float32.(x), in_dim, :)
    y = Matrix{Float32}(undef, out_dim, size(x_f32, 2))
    rows_per_chunk = clamp(16 * 1024 * 1024 ÷ max(in_dim, 1), 128, out_dim)
    for row_start in 1:rows_per_chunk:out_dim
        row_stop = min(row_start + rows_per_chunk - 1, out_dim)
        LinearAlgebra.mul!(
            view(y, row_start:row_stop, :),
            Float32.(view(weight, row_start:row_stop, :)),
            x_f32,
        )
    end
    return reshape(BFloat16.(y), out_dim, num_tokens, batch_size)
end

# HF Qwen3RMSNorm: normalize in Float32, cast to BF16, then multiply by the
# BF16 weight elementwise (one more rounding step).
function _bf16_rmsnorm(x::AbstractArray{BFloat16}, scale, epsilon::Float32)
    xf = Float32.(x)
    mean_square = sum(abs2, xf; dims=1) ./ Float32(size(x, 1))
    normalized = BFloat16.(xf ./ sqrt.(mean_square .+ epsilon))
    shape = ntuple(i -> i == 1 ? size(x, 1) : 1, ndims(x))
    scale_f = reshape(Float32.(vec(scale)), shape)
    return BFloat16.(scale_f .* Float32.(normalized))
end

"""
    _bf16_apply_rope(x, cos_table, sin_table; start_pos)

Rotate-half RoPE in BF16: `x1*cos - x2*sin` / `x2*cos + x1*sin` with a BF16
rounding after every elementwise product and sum, using BF16-cast Float32
tables — the same dtype flow as `apply_rotary_pos_emb` under BF16.
"""
function _bf16_apply_rope(
    x::AbstractArray{BFloat16,4},
    cos_table::AbstractMatrix{BFloat16},
    sin_table::AbstractMatrix{BFloat16};
    start_pos::Int,
)
    head_dim, num_heads, num_tokens, batch_size = size(x)
    half = head_dim ÷ 2
    x1 = Float32.(view(x, 1:half, :, :, :))
    x2 = Float32.(view(x, (half + 1):head_dim, :, :, :))
    positions = start_pos:(start_pos + num_tokens - 1)
    cos_f = Float32.(view(cos_table, :, positions))
    sin_f = Float32.(view(sin_table, :, positions))
    cos_b = reshape(cos_f, half, 1, num_tokens, 1)
    sin_b = reshape(sin_f, half, 1, num_tokens, 1)
    # HF computes rotate_half(q) .* sin as one product; negation is exact.
    upper = Float32.(BFloat16.(x1 .* cos_b)) .+ Float32.(BFloat16.(.-x2 .* sin_b))
    lower = Float32.(BFloat16.(x2 .* cos_b)) .+ Float32.(BFloat16.(x1 .* sin_b))
    y = similar(x)
    y[1:half, :, :, :] = BFloat16.(upper)
    y[(half + 1):head_dim, :, :, :] = BFloat16.(lower)
    return y
end

# Eager HF attention under BF16: BF16 score matmul (F32 accumulation), BF16
# scaling, additive BF16-min causal mask, Float32 softmax cast back to BF16,
# BF16 context matmul.
function _bf16_attention(
    queries::AbstractArray{BFloat16,4},
    keys::AbstractArray{BFloat16,4},
    values::AbstractArray{BFloat16,4};
    scaling::Float32,
    causal::Bool,
)
    head_dim, num_heads, query_tokens, batch_size = size(queries)
    _, num_kv_heads, key_tokens, _ = size(keys)
    groups = num_heads ÷ num_kv_heads
    context = Array{BFloat16,4}(undef, head_dim, num_heads, query_tokens, batch_size)
    for batch in 1:batch_size, head in 1:num_heads
        kv_head = (head - 1) ÷ groups + 1
        q = Float32.(view(queries, :, head, :, batch))
        k = Float32.(view(keys, :, kv_head, :, batch))
        v = Float32.(view(values, :, kv_head, :, batch))
        scores = BFloat16.(transpose(q) * k)
        scaled = Float32.(BFloat16.(Float32.(scores) .* scaling))
        if causal
            for query_index in 1:query_tokens, key_index in 1:key_tokens
                key_index > query_index + (key_tokens - query_tokens) &&
                    (scaled[query_index, key_index] = Float32(BFloat16(
                        scaled[query_index, key_index] + _BF16_MASK_MIN,
                    )))
            end
        end
        maxima = maximum(scaled; dims=2)
        exponents = exp.(scaled .- maxima)
        weights = Float32.(BFloat16.(exponents ./ sum(exponents; dims=2)))
        context[:, head, :, batch] = BFloat16.(v * transpose(weights))
    end
    return context
end

function _bf16_swiglu(ps_mlp, x::AbstractArray{BFloat16,3})
    gate = _bf16_linear(ps_mlp.gate_proj.weight, x)
    up = _bf16_linear(ps_mlp.up_proj.weight, x)
    gate_f = Float32.(gate)
    activated = BFloat16.(gate_f ./ (1.0f0 .+ exp.(.-gate_f)))
    hidden = BFloat16.(Float32.(activated) .* Float32.(up))
    return _bf16_linear(ps_mlp.down_proj.weight, hidden)
end

_bf16_residual(x, delta) = BFloat16.(Float32.(x) .+ Float32.(delta))

function _bf16_input_second_moment(x::AbstractArray{BFloat16,3})
    samples = size(x, 2) * size(x, 3)
    samples > 0 || throw(ArgumentError(
        "activation calibration requires at least one token",
    ))
    return vec(sum(abs2, Float32.(x); dims=(2, 3))) ./ Float32(samples)
end

function _bf16_block_core(
    model::GPTModel,
    ps_block,
    x::AbstractArray{BFloat16,3},
    cos_table,
    sin_table,
    cache;
    start_pos::Int,
    capture_activation_moments::Bool,
)
    head_dim = model.head_dim
    num_tokens, batch_size = size(x, 2), size(x, 3)
    scaling = 1.0f0 / sqrt(Float32(head_dim))

    normed = _bf16_rmsnorm(x, ps_block.norm1.scale, model.norm_epsilon)
    attention_input_moment = capture_activation_moments ?
        _bf16_input_second_moment(normed) : nothing
    queries = reshape(
        _bf16_linear(ps_block.attn.q_proj.weight, normed),
        head_dim, model.num_heads, num_tokens, batch_size,
    )
    keys = reshape(
        _bf16_linear(ps_block.attn.k_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    values = reshape(
        _bf16_linear(ps_block.attn.v_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    queries = _bf16_rmsnorm(queries, ps_block.attn.q_norm.scale, model.qk_norm_epsilon)
    keys = _bf16_rmsnorm(keys, ps_block.attn.k_norm.scale, model.qk_norm_epsilon)
    queries = _bf16_apply_rope(queries, cos_table, sin_table; start_pos)
    keys = _bf16_apply_rope(keys, cos_table, sin_table; start_pos)

    cached_keys, cached_values = cache
    all_keys = cached_keys === nothing ? keys : cat(cached_keys, keys; dims=3)
    all_values = cached_values === nothing ? values : cat(cached_values, values; dims=3)

    context = _bf16_attention(
        queries,
        all_keys,
        all_values;
        scaling,
        causal=cached_keys === nothing,
    )
    output_input = reshape(
        context,
        head_dim * model.num_heads,
        num_tokens,
        batch_size,
    )
    output_input_moment = capture_activation_moments ?
        _bf16_input_second_moment(output_input) : nothing
    attn_out = _bf16_linear(
        ps_block.attn.o_proj.weight,
        output_input,
    )
    x = _bf16_residual(x, attn_out)

    normed2 = _bf16_rmsnorm(x, ps_block.norm2.scale, model.norm_epsilon)
    mlp_input_moment = capture_activation_moments ?
        _bf16_input_second_moment(normed2) : nothing
    gate = _bf16_linear(ps_block.mlp.gate_proj.weight, normed2)
    up = _bf16_linear(ps_block.mlp.up_proj.weight, normed2)
    gate_f = Float32.(gate)
    activated = BFloat16.(gate_f ./ (1.0f0 .+ exp.(.-gate_f)))
    hidden = BFloat16.(Float32.(activated) .* Float32.(up))
    down_input_moment = capture_activation_moments ?
        _bf16_input_second_moment(hidden) : nothing
    x = _bf16_residual(x, _bf16_linear(ps_block.mlp.down_proj.weight, hidden))

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

function _bf16_block(
    model::GPTModel,
    ps_block,
    x::AbstractArray{BFloat16,3},
    cos_table,
    sin_table,
    cache;
    start_pos::Int,
)
    output, updated_cache, _ = _bf16_block_core(
        model,
        ps_block,
        x,
        cos_table,
        sin_table,
        cache;
        start_pos,
        capture_activation_moments=false,
    )
    return output, updated_cache
end

function _bf16_block_activation_moments(
    model::GPTModel,
    ps_block,
    x::AbstractArray{BFloat16,3},
    cos_table,
    sin_table;
    start_pos::Int,
)
    return _bf16_block_core(
        model,
        ps_block,
        x,
        cos_table,
        sin_table,
        (nothing, nothing);
        start_pos,
        capture_activation_moments=true,
    )
end

function _bf16_embed(model::GPTModel, ps, tokens::AbstractMatrix{Int})
    _validate_generation_ids(tokens, model.vocab_size)
    seq_len, batch_size = size(tokens)
    x = Array{BFloat16,3}(undef, model.d_model, seq_len, batch_size)
    for batch in 1:batch_size, position in 1:seq_len
        x[:, position, batch] = view(ps.token_embedding.weight, :, tokens[position, batch])
    end
    return x
end

function _bf16_logits(model::GPTModel, ps, hidden::AbstractArray{BFloat16,3})
    weight = model.tie_embeddings ?
        permutedims(ps.token_embedding.weight, (2, 1)) : ps.lm_head.weight
    return _bf16_linear(weight, hidden)
end

function _bf16_rope_tables(model::GPTModel)
    rope = first(values(model.blocks.layers)).attn.rope
    return BFloat16.(rope.cos_cache), BFloat16.(rope.sin_cache)
end

function _bf16_forward_pass(model, ps, tokens, caches, cos_table, sin_table; start_pos)
    x = _bf16_embed(model, ps, tokens)
    embedding = x
    block_parameters = Tuple(values(ps.blocks))
    block_outputs = Vector{Any}(undef, model.num_layers)
    for index in 1:model.num_layers
        x, caches[index] = _bf16_block(
            model,
            block_parameters[index],
            x,
            cos_table,
            sin_table,
            caches[index];
            start_pos,
        )
        block_outputs[index] = x
        # 每层线性会产生权重体量级的 Float32 临时数组；参数树本身可能已
        # 占据大半 RAM，必须逐层回收年轻代，否则大尺寸下 RSS 越界 OOM。
        GC.gc(false)
    end
    final_hidden = _bf16_rmsnorm(x, ps.final_norm.scale, model.norm_epsilon)
    logits = _bf16_logits(model, ps, final_hidden)
    GC.gc(false)
    return (; embedding, block_outputs, final_hidden, logits, caches)
end

"""
    hf_qwen3_bf16_forward(
        model,
        ps,
        tokens;
        decode_token=nothing,
        greedy_steps=0,
    )

Run the Qwen3 forward trace in native BF16 mixed precision: embedding, every
block output, final hidden state and logits, plus an optional cached decode
step and an optional greedy continuation. `ps` must be a BF16 parameter tree
from `load_hf_qwen3_model(...; weight_dtype=BFloat16)`. `tokens` is a
`(seq_len, batch)` matrix of 1-based ids; `decode_token` gets a fresh copy of
the prefill cache, so the greedy continuation is unaffected by it.
`greedy_steps` currently requires `batch == 1` and returns 1-based token ids.
"""
function hf_qwen3_bf16_forward(
    model::GPTModel,
    ps,
    tokens::AbstractMatrix{<:Integer};
    decode_token=nothing,
    greedy_steps::Int=0,
)
    _qwen3_validate_semantics(model)
    eltype(ps.token_embedding.weight) === BFloat16 || throw(ArgumentError(
        "hf_qwen3_bf16_forward requires a BFloat16 parameter tree; " *
        "load with weight_dtype=BFloat16",
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
    cos_table, sin_table = _bf16_rope_tables(model)
    caches = Vector{Any}(undef, model.num_layers)
    fill!(caches, (nothing, nothing))

    prefill = _bf16_forward_pass(
        model, ps, token_matrix, caches, cos_table, sin_table;
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
        decode = _bf16_forward_pass(
            model, ps, decode_matrix, decode_caches, cos_table, sin_table;
            start_pos=seq_len + 1,
        )
        decode_logits = decode.logits
    end

    greedy_tokens = Int[]
    if greedy_steps > 0
        logits = prefill.logits
        position = seq_len
        for _ in 1:greedy_steps
            next_token = argmax(vec(view(logits, :, size(logits, 2), 1)))
            push!(greedy_tokens, next_token)
            length(greedy_tokens) == greedy_steps && break
            step = _bf16_forward_pass(
                model,
                ps,
                reshape([next_token], 1, 1),
                prefill.caches,
                cos_table,
                sin_table;
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
