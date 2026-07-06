"""
    RoPE(head_dim; max_seq_len=2048, theta=10000.0)

Rotary Positional Embedding.

输入张量约定：

    x: (head_dim, num_heads, seq_len, batch)

RoPE 会对每两个相邻维度做一次二维旋转：

    (x₁, x₂) -> (x₁ cosθ - x₂ sinθ, x₁ sinθ + x₂ cosθ)

其中旋转角度由 token position 和频率共同决定。
"""
struct RoPE
    head_dim::Int
    max_seq_len::Int
    theta::Float32
    inv_freq::Vector{Float32}
end

function RoPE(
    head_dim::Int;
    max_seq_len::Int=2048,
    theta::Real=10000.0,
)
    @assert iseven(head_dim) "`head_dim` must be even for RoPE"
    @assert max_seq_len > 0 "`max_seq_len` must be positive"
    @assert theta > 0 "`theta` must be positive"

    theta32 = Float32(theta)

    # 对应维度 pair:
    # pair 1 -> dim 1,2 -> exponent 0 / head_dim
    # pair 2 -> dim 3,4 -> exponent 2 / head_dim
    # pair 3 -> dim 5,6 -> exponent 4 / head_dim
    inv_freq = Float32[
        inv(theta32 ^ (Float32(i - 1) / Float32(head_dim)))
        for i in 1:2:head_dim
    ]

    return RoPE(
        head_dim,
        max_seq_len,
        theta32,
        inv_freq,
    )
end

"""
    apply_rope(x, rope; start_pos=1)

对输入 x 应用 RoPE。

输入：

    x: (head_dim, num_heads, seq_len, batch)

输出：

    y: same shape as x

`start_pos` 是 1-based 的 token 起始位置。

例如：

    start_pos = 1

表示当前序列的第一个 token 使用 position 0，因此不会被旋转。

    start_pos = 5

表示当前序列的第一个 token 使用 position 4，常用于以后接 KV-cache。
"""
function apply_rope(
    x,
    rope::RoPE;
    start_pos::Int=1,
)
    D, H, T, B = size(x)

    @assert D == rope.head_dim "`x` head_dim does not match rope.head_dim"
    @assert iseven(D) "`head_dim` must be even for RoPE"
    @assert start_pos >= 1 "`start_pos` must be >= 1"
    @assert start_pos + T - 1 <= rope.max_seq_len "`x` exceeds rope.max_seq_len"

    y = similar(x)

    @inbounds for b in 1:B
        for t in 1:T
            # RoPE 位置一般从 0 开始。
            # Julia 下 start_pos 是 1-based，所以这里减 1。
            pos = Float32(start_pos + t - 2)

            for h in 1:H
                for pair in 1:(D ÷ 2)
                    i = 2pair - 1

                    angle = pos * rope.inv_freq[pair]
                    c = cos(angle)
                    s = sin(angle)

                    x1 = Float32(x[i, h, t, b])
                    x2 = Float32(x[i + 1, h, t, b])

                    y[i, h, t, b] = x1 * c - x2 * s
                    y[i + 1, h, t, b] = x1 * s + x2 * c
                end
            end
        end
    end

    return y
end