"""
    RoPE(head_dim; max_seq_len=2048, theta=10000.0)

Rotary Positional Embedding.

输入张量约定：

    x: (head_dim, num_heads, seq_len, batch)

内部预计算：

    cos_cache: (head_dim ÷ 2, max_seq_len)
    sin_cache: (head_dim ÷ 2, max_seq_len)

其中第 `pos_idx` 列对应 position = pos_idx - 1。
"""
struct RoPE
    head_dim::Int
    max_seq_len::Int
    theta::Float32
    inv_freq::Vector{Float32}
    cos_cache::Matrix{Float32}
    sin_cache::Matrix{Float32}
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
    half_dim = head_dim ÷ 2

    inv_freq = Vector{Float32}(undef, half_dim)

    @inbounds for pair in 1:half_dim
        # pair = 1 -> dim index 0
        # pair = 2 -> dim index 2
        # pair = 3 -> dim index 4
        dim_index = 2 * (pair - 1)
        inv_freq[pair] = inv(theta32 ^ (Float32(dim_index) / Float32(head_dim)))
    end

    cos_cache = Matrix{Float32}(undef, half_dim, max_seq_len)
    sin_cache = Matrix{Float32}(undef, half_dim, max_seq_len)

    @inbounds for pos_idx in 1:max_seq_len
        pos = Float32(pos_idx - 1)

        for pair in 1:half_dim
            angle = pos * inv_freq[pair]
            cos_cache[pair, pos_idx] = cos(angle)
            sin_cache[pair, pos_idx] = sin(angle)
        end
    end

    return RoPE(
        head_dim,
        max_seq_len,
        theta32,
        inv_freq,
        cos_cache,
        sin_cache,
    )
end

"""
    apply_rope!(y, x, rope; start_pos=1)

把 RoPE 应用到 x，并写入 y。

输入：

    x: (head_dim, num_heads, seq_len, batch)
    y: same shape as x

`start_pos` 是 1-based 的 token 起始位置。

例如：

    start_pos = 1

表示当前序列的第一个 token 使用 position 0。

    start_pos = 5

表示当前序列的第一个 token 使用 position 4，常用于 KV-cache。
"""
function apply_rope!(
    y,
    x,
    rope::RoPE;
    start_pos::Int=1,
)
    D, H, T, B = size(x)

    @assert size(y) == size(x) "`y` and `x` must have the same shape"
    @assert D == rope.head_dim "`x` head_dim does not match rope.head_dim"
    @assert iseven(D) "`head_dim` must be even for RoPE"
    @assert start_pos >= 1 "`start_pos` must be >= 1"
    @assert start_pos + T - 1 <= rope.max_seq_len "`x` exceeds rope.max_seq_len"

    half_dim = D ÷ 2
    cos_cache = rope.cos_cache
    sin_cache = rope.sin_cache

    @inbounds for b in 1:B
        for t in 1:T
            pos_idx = start_pos + t - 1

            for h in 1:H
                for pair in 1:half_dim
                    i = 2 * pair - 1

                    c = cos_cache[pair, pos_idx]
                    s = sin_cache[pair, pos_idx]

                    x1 = x[i, h, t, b]
                    x2 = x[i + 1, h, t, b]

                    y[i, h, t, b]     = x1 * c - x2 * s
                    y[i + 1, h, t, b] = x1 * s + x2 * c
                end
            end
        end
    end

    return y
end

function apply_rope(
    x,
    rope::RoPE;
    start_pos::Int=1,
)
    y = similar(x)
    return apply_rope!(y, x, rope; start_pos)
end

function apply_rope_threaded!(
    y,
    x,
    rope::RoPE;
    start_pos::Int=1,
)
    D, H, T, B = size(x)

    @assert size(y) == size(x)
    @assert D == rope.head_dim
    @assert iseven(D)
    @assert start_pos >= 1
    @assert start_pos + T - 1 <= rope.max_seq_len

    half_dim = D ÷ 2
    cos_cache = rope.cos_cache
    sin_cache = rope.sin_cache

    Threads.@threads for b in 1:B
        @inbounds for t in 1:T
            pos_idx = start_pos + t - 1

            for h in 1:H
                for pair in 1:half_dim
                    i = 2 * pair - 1

                    c = cos_cache[pair, pos_idx]
                    s = sin_cache[pair, pos_idx]

                    x1 = x[i, h, t, b]
                    x2 = x[i + 1, h, t, b]

                    y[i, h, t, b]     = x1 * c - x2 * s
                    y[i + 1, h, t, b] = x1 * s + x2 * c
                end
            end
        end
    end

    return y
end