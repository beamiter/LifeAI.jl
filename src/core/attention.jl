using Lux
using ConcreteStructs
using NNlib: batched_mul, softmax

@concrete struct MultiHeadAttention <: AbstractLuxContainerLayer{(
    :q_proj, :k_proj, :v_proj, :o_proj
)}
    q_proj
    k_proj
    v_proj
    o_proj

    d_in::Int
    num_heads::Int
    head_dim::Int
    d_out::Int
    is_causal::Bool

    use_rope::Bool
    rope
end

function MultiHeadAttention(
    d_in::Int,
    num_heads::Int;
    head_dim=nothing,
    use_bias::Bool=false,
    is_causal::Bool=true,
    use_rope::Bool=false,
    max_seq_len::Int=2048,
    rope_theta::Real=10000.0,
)
    if head_dim === nothing
        @assert d_in % num_heads == 0 "`d_in` must be divisible by `num_heads`"
        head_dim = d_in ÷ num_heads
    end

    d_out = num_heads * head_dim

    if use_rope
        @assert iseven(head_dim) "`head_dim` must be even when using RoPE"
        rope = RoPE(
            head_dim;
            max_seq_len,
            theta=rope_theta,
        )
    else
        rope = nothing
    end

    return MultiHeadAttention(
        Dense(d_in, d_out; use_bias),
        Dense(d_in, d_out; use_bias),
        Dense(d_in, d_out; use_bias),
        Dense(d_out, d_in; use_bias),
        d_in,
        num_heads,
        head_dim,
        d_out,
        is_causal,
        use_rope,
        rope,
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
    queries = reshape(queries, attn.head_dim, attn.num_heads, num_tokens, B)
    keys = reshape(keys, attn.head_dim, attn.num_heads, num_tokens, B)
    values = reshape(values, attn.head_dim, attn.num_heads, num_tokens, B)

    # 2.5 RoPE:
    #     只作用在 Q/K 上，不作用在 V 上。
    if attn.use_rope
        queries = apply_rope(queries, attn.rope)
        keys = apply_rope(keys, attn.rope)
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
        ),
    )
end

# q, k, v shape:
#   (head_dim, num_heads, num_tokens, batch)
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
    @assert H == Hk == Hv
    @assert B == Bk == Bv
    @assert Tk == Tv

    inv_sqrt_d = inv(sqrt(Float32(D)))

    # 用 Float32 做 attention 分数和 softmax，更稳定
    attn = similar(q, Float32, Tq, Tk, H, B)
    out = similar(q, Float32, D, H, Tq, B)

    @inbounds for b in 1:B
        for h in 1:H
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
                        score += Float32(q[d, h, tq, b]) * Float32(k[d, h, tk, b])
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
                        acc += attn[tq, tk, h, b] * Float32(v[d, h, tk, b])
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
    # q, k, v:
    # (D, H, T, B)
    D, H, Tq, B = size(q)
    Dk, Hk, Tk, Bk = size(k)
    Dv, Hv, Tv, Bv = size(v)

    @assert D == Dk == Dv
    @assert H == Hk == Hv
    @assert B == Bk == Bv
    @assert Tk == Tv

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
        # mask: (Tq, Tk, 1)
        # tk > tq 的位置不可见
        mask = reshape((1:Tk)' .> (1:Tq), Tq, Tk, 1)
        scores = ifelse.(mask, -Inf32, scores)
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
