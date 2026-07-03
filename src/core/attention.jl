using Lux
using ConcreteStructs

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
end

function MultiHeadAttention(
    d_in::Int,
    num_heads::Int;
    head_dim=nothing,
    use_bias::Bool=false,
    is_causal::Bool=true,
)
    if head_dim === nothing
        @assert d_in % num_heads == 0 "`d_in` must be divisible by `num_heads`"
        head_dim = d_in ÷ num_heads
    end

    d_out = num_heads * head_dim

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

    # 3. scaled dot-product attention
    context, attn_weights = manual_scaled_dot_product_attention(
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
