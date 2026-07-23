using Lux
using ConcreteStructs
using MLDataDevices: get_device
using NNlib: batched_mul, make_causal_mask, softmax
using Random: AbstractRNG

@concrete struct MultiHeadAttention <: AbstractLuxContainerLayer{(
    :q_proj, :k_proj, :v_proj, :o_proj
)}
    q_proj
    k_proj
    v_proj
    o_proj

    d_in::Int
    num_heads::Int
    num_kv_heads::Int
    head_dim::Int
    d_out::Int
    kv_dim::Int
    is_causal::Bool

    use_rope::Bool
    rope_style::Symbol
    rope

    use_qk_norm::Bool
    qk_norm_epsilon::Float32
end

function MultiHeadAttention(
    d_in::Int,
    num_heads::Int;
    num_kv_heads::Int=num_heads,
    head_dim=nothing,
    use_bias::Bool=false,
    is_causal::Bool=true,
    use_rope::Bool=false,
    use_qk_norm::Bool=false,
    qk_norm_epsilon::Real=1.0f-6,
    max_seq_len::Int=2048,
    rope_theta::Real=10000.0,
    rope_style::Symbol=:interleaved,
)
    @assert num_kv_heads > 0 "`num_kv_heads` must be positive"
    @assert num_heads % num_kv_heads == 0 "`num_heads` must be divisible by `num_kv_heads`"
    @assert qk_norm_epsilon > 0 "`qk_norm_epsilon` must be positive"
    _validate_rope_style(rope_style)

    if head_dim === nothing
        @assert d_in % num_heads == 0 "`d_in` must be divisible by `num_heads`"
        head_dim = d_in ÷ num_heads
    end

    d_out = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    if use_rope
        @assert iseven(head_dim) "`head_dim` must be even when using RoPE"
        rope = RoPE(
            head_dim;
            max_seq_len,
            theta=rope_theta,
            style=rope_style,
        )
    else
        rope = nothing
    end

    return MultiHeadAttention(
        Dense(d_in, d_out; use_bias),
        Dense(d_in, kv_dim; use_bias),
        Dense(d_in, kv_dim; use_bias),
        Dense(d_out, d_in; use_bias),
        d_in,
        num_heads,
        num_kv_heads,
        head_dim,
        d_out,
        kv_dim,
        is_causal,
        use_rope,
        rope_style,
        rope,
        use_qk_norm,
        Float32(qk_norm_epsilon),
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, attn::MultiHeadAttention)
    # Replicate the container default field-by-field so the RNG stream for
    # existing configurations stays identical, then append the optional QK-norm
    # scales. Disabled QK-norm keeps the legacy parameter tree unchanged.
    ps = (;
        q_proj=LuxCore.initialparameters(rng, attn.q_proj),
        k_proj=LuxCore.initialparameters(rng, attn.k_proj),
        v_proj=LuxCore.initialparameters(rng, attn.v_proj),
        o_proj=LuxCore.initialparameters(rng, attn.o_proj),
    )
    attn.use_qk_norm || return ps

    return merge(ps, (;
        q_norm=(; scale=ones(Float32, attn.head_dim)),
        k_norm=(; scale=ones(Float32, attn.head_dim)),
    ))
end

function LuxCore.parameterlength(attn::MultiHeadAttention)
    projection_count =
        LuxCore.parameterlength(attn.q_proj) +
        LuxCore.parameterlength(attn.k_proj) +
        LuxCore.parameterlength(attn.v_proj) +
        LuxCore.parameterlength(attn.o_proj)
    qk_norm_count = attn.use_qk_norm ? 2 * attn.head_dim : 0
    return projection_count + qk_norm_count
end

"""
    _apply_qk_norm(x, scale, epsilon)

Per-head RMS normalization over the head dimension, matching Qwen3's `q_norm` /
`k_norm`. `x` has shape `(head_dim, num_heads, num_tokens, batch)`; `scale` is a
learned `(head_dim,)` vector shared across heads. Applied before RoPE.
"""
function _apply_qk_norm(x, scale, epsilon::Float32)
    value_type = eltype(x)
    head_dim = size(x, 1)
    mean_square = sum(abs2, x; dims=1) ./ convert(value_type, head_dim)
    inverse_rms = one(value_type) ./ sqrt.(
        mean_square .+ convert(value_type, epsilon),
    )
    return x .* inverse_rms .* reshape(value_type.(scale), :, 1, 1, 1)
end

"""
    repeat_kv(kv, groups)

Reference expansion of grouped K/V heads to the full query-head count. Each KV
head is repeated `groups` times contiguously, so query head `h` maps to KV head
`(h - 1) ÷ groups + 1` — the same layout HuggingFace `repeat_kv` produces.
"""
function repeat_kv(kv, groups::Int)
    groups >= 1 || throw(ArgumentError("`groups` must be >= 1"))
    groups == 1 && return kv
    return repeat(kv; inner=(1, groups, 1, 1))
end

function LuxCore.initialstates(rng::AbstractRNG, attn::MultiHeadAttention)
    return (;
        q_proj=LuxCore.initialstates(rng, attn.q_proj),
        k_proj=LuxCore.initialstates(rng, attn.k_proj),
        v_proj=LuxCore.initialstates(rng, attn.v_proj),
        o_proj=LuxCore.initialstates(rng, attn.o_proj),
        rope_cos_cache=attn.use_rope ? attn.rope.cos_cache : nothing,
        rope_sin_cache=attn.use_rope ? attn.rope.sin_cache : nothing,
    )
end

function (attn::MultiHeadAttention)(x, ps, st::NamedTuple)
    # x: (d_in, num_tokens, batch)
    _, num_tokens, B = size(x)

    # 1. Q/K/V projection
    queries, st_q_proj = attn.q_proj(x, ps.q_proj, st.q_proj)
    keys, st_k_proj = attn.k_proj(x, ps.k_proj, st.k_proj)
    values, st_v_proj = attn.v_proj(x, ps.v_proj, st.v_proj)

    # 2. reshape to:
    #    (head_dim, num_heads, num_tokens, batch)
    #    K/V keep only num_kv_heads heads under GQA.
    queries = reshape(queries, attn.head_dim, attn.num_heads, num_tokens, B)
    keys = reshape(keys, attn.head_dim, attn.num_kv_heads, num_tokens, B)
    values = reshape(values, attn.head_dim, attn.num_kv_heads, num_tokens, B)

    # 2.4 QK-Norm（Qwen3 语义）：
    #     在 head reshape 之后、RoPE 之前对 Q/K 做 per-head RMSNorm。
    if attn.use_qk_norm
        queries = _apply_qk_norm(queries, ps.q_norm.scale, attn.qk_norm_epsilon)
        keys = _apply_qk_norm(keys, ps.k_norm.scale, attn.qk_norm_epsilon)
    end

    # 2.5 RoPE:
    #     只作用在 Q/K 上，不作用在 V 上。
    if attn.use_rope
        queries = apply_rope(
            queries,
            st.rope_cos_cache,
            st.rope_sin_cache;
            rope_style=attn.rope_style,
        )
        keys = apply_rope(
            keys,
            st.rope_cos_cache,
            st.rope_sin_cache;
            rope_style=attn.rope_style,
        )
    end

    # 3. scaled dot-product attention
    context, attn_weights = batched_scaled_dot_product_attention(
        queries,
        keys,
        values;
        is_causal=attn.is_causal,
    )

    # 4. merge heads
    context = reshape(context, attn.d_out, num_tokens, B)

    # 5. output projection
    y, st_o_proj = attn.o_proj(context, ps.o_proj, st.o_proj)

    return (
        y,
        (;
            q_proj=st_q_proj,
            k_proj=st_k_proj,
            v_proj=st_v_proj,
            o_proj=st_o_proj,
            rope_cos_cache=st.rope_cos_cache,
            rope_sin_cache=st.rope_sin_cache,
        ),
    )
end

# q shape:    (head_dim, num_heads, num_tokens, batch)
# k, v shape: (head_dim, num_kv_heads, num_tokens, batch)
#
# num_kv_heads may divide num_heads (GQA/MQA); query head h reads KV head
# (h - 1) ÷ groups + 1.
#
# return:
#   context: (head_dim, num_heads, num_tokens, batch)
#   attn:    (num_tokens, num_tokens, num_heads, batch)

function manual_scaled_dot_product_attention(
    q,
    k,
    v;
    is_causal::Bool=true,
)
    D, H, Tq, B = size(q)
    Dk, Hk, Tk, Bk = size(k)
    Dv, Hv, Tv, Bv = size(v)

    @assert D == Dk == Dv
    @assert Hk == Hv
    @assert H % Hk == 0
    @assert B == Bk == Bv
    @assert Tk == Tv

    groups = H ÷ Hk
    inv_sqrt_d = inv(sqrt(Float32(D)))

    # 用 Float32 做 attention 分数和 softmax，更稳定
    attn = similar(q, Float32, Tq, Tk, H, B)
    out = similar(q, Float32, D, H, Tq, B)

    @inbounds for b in 1:B
        for h in 1:H
            kv = (h - 1) ÷ groups + 1

            for tq in 1:Tq

                # 1. 计算 scores = q ⋅ k / sqrt(D)
                max_score = -Inf32

                for tk in 1:Tk
                    if is_causal && tk > tq
                        attn[tq, tk, h, b] = -Inf32
                        continue
                    end

                    score = 0.0f0
                    for d in 1:D
                        score += Float32(q[d, h, tq, b]) * Float32(k[d, kv, tk, b])
                    end
                    score *= inv_sqrt_d

                    attn[tq, tk, h, b] = score
                    max_score = max(max_score, score)
                end

                # 2. stable softmax
                denom = 0.0f0

                for tk in 1:Tk
                    if is_causal && tk > tq
                        attn[tq, tk, h, b] = 0.0f0
                    else
                        w = exp(attn[tq, tk, h, b] - max_score)
                        attn[tq, tk, h, b] = w
                        denom += w
                    end
                end

                for tk in 1:Tk
                    attn[tq, tk, h, b] /= denom
                end

                # 3. context = softmax(scores) * V
                for d in 1:D
                    acc = 0.0f0
                    for tk in 1:Tk
                        acc += attn[tq, tk, h, b] * Float32(v[d, kv, tk, b])
                    end
                    out[d, h, tq, b] = acc
                end
            end
        end
    end

    return eltype(q).(out), attn
end

function batched_scaled_dot_product_attention(
    q,
    k,
    v;
    is_causal::Bool=true,
)
    # q:    (D, H,  T, B)
    # k, v: (D, Hk, T, B) with Hk dividing H (GQA/MQA)
    D, H, Tq, B = size(q)
    Dk, Hk, Tk, Bk = size(k)
    Dv, Hv, Tv, Bv = size(v)

    @assert D == Dk == Dv
    @assert Hk == Hv
    @assert H % Hk == 0
    @assert B == Bk == Bv
    @assert Tk == Tv

    # Grouped-query path: fewer KV heads shared by query-head groups. The
    # existing full-head path below stays byte-identical for Hk == H.
    H == Hk || return _grouped_scaled_dot_product_attention(q, k, v; is_causal)

    HB = H * B
    inv_sqrt_d = inv(sqrt(Float32(D)))

    # 合并 head 和 batch：
    #
    # q3: (Tq, D,  H*B)
    # k3: (D,  Tk, H*B)
    # v3: (Tk, D,  H*B)
    q3 = reshape(permutedims(Float32.(q), (3, 1, 2, 4)), Tq, D, HB)
    k3 = reshape(permutedims(Float32.(k), (1, 3, 2, 4)), D, Tk, HB)
    v3 = reshape(permutedims(Float32.(v), (3, 1, 2, 4)), Tk, D, HB)

    # scores: (Tq, Tk, H*B)
    scores = batched_mul(q3, k3) .* inv_sqrt_d

    if is_causal
        # Build the mask from `scores` so it is created directly on the active
        # backend. Calling `get_device` while Reactant traces is unsupported.
        visible = if Tq == Tk
            # NNlib's mask is upper triangular; transpose it because scores are
            # laid out as (query, key, batch-head).
            reshape(permutedims(make_causal_mask(scores; dims=2)), Tq, Tk, 1)
        else
            # Cross-attention is only used outside the compiled GPT path.
            get_device(scores)(reshape((1:Tk)' .<= (1:Tq), Tq, Tk, 1))
        end
        scores = ifelse.(visible, scores, -Inf32)
    end

    # attention weights: 对 key 维度做 softmax
    # weights: (Tq, Tk, H*B)
    weights = softmax(scores; dims=2)

    # context3: (Tq, D, H*B)
    context3 = batched_mul(weights, v3)

    # 还原为 (D, H, Tq, B)
    context = permutedims(reshape(context3, Tq, D, H, B), (2, 3, 1, 4))

    # attn: (Tq, Tk, H, B)
    attn = reshape(weights, Tq, Tk, H, B)

    return eltype(q).(context), attn
end

# GQA/MQA attention without materializing repeated K/V. Query heads are folded
# into the row dimension so each KV head serves its whole group through one
# batched matmul:
#
#   q3: (Tq*groups, D,  Hk*B)
#   k3: (D,         Tk, Hk*B)
#   v3: (Tk,        D,  Hk*B)
#
# Row r = tq + (gi - 1) * Tq, and query head h = (kv - 1) * groups + gi, which
# matches the contiguous `repeat_kv` layout used as the reference.
function _grouped_scaled_dot_product_attention(
    q,
    k,
    v;
    is_causal::Bool=true,
)
    D, H, Tq, B = size(q)
    _, Hk, Tk, _ = size(k)
    groups = H ÷ Hk
    HkB = Hk * B
    inv_sqrt_d = inv(sqrt(Float32(D)))

    qp = permutedims(Float32.(q), (3, 1, 2, 4))
    q5 = reshape(qp, Tq, D, groups, Hk, B)
    q3 = reshape(permutedims(q5, (1, 3, 2, 4, 5)), Tq * groups, D, HkB)

    k3 = reshape(permutedims(Float32.(k), (1, 3, 2, 4)), D, Tk, HkB)
    v3 = reshape(permutedims(Float32.(v), (3, 1, 2, 4)), Tk, D, HkB)

    # scores: (Tq*groups, Tk, Hk*B)
    scores = batched_mul(q3, k3) .* inv_sqrt_d

    if is_causal
        # The mask depends only on tq, so expose the (Tq, groups) split and
        # broadcast a (Tq, 1, Tk, 1) visibility mask over every group.
        visible = if Tq == Tk
            reshape(permutedims(make_causal_mask(scores; dims=2)), Tq, 1, Tk, 1)
        else
            get_device(scores)(reshape((1:Tk)' .<= (1:Tq), Tq, 1, Tk, 1))
        end
        scores4 = reshape(scores, Tq, groups, Tk, HkB)
        scores = reshape(ifelse.(visible, scores4, -Inf32), Tq * groups, Tk, HkB)
    end

    weights = softmax(scores; dims=2)

    # context3: (Tq*groups, D, Hk*B)
    context3 = batched_mul(weights, v3)

    context5 = permutedims(reshape(context3, Tq, groups, D, Hk, B), (3, 2, 4, 1, 5))
    context = reshape(context5, D, H, Tq, B)

    attn5 = permutedims(reshape(weights, Tq, groups, Tk, Hk, B), (1, 3, 2, 4, 5))
    attn = reshape(attn5, Tq, Tk, H, B)

    return eltype(q).(context), attn
end
