using MLDataDevices: get_device

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
    style::Symbol
    inv_freq::Vector{Float32}
    cos_cache::Matrix{Float32}
    sin_cache::Matrix{Float32}
end

function _rope_positive_host_int(value, label::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "`$label` must be an integer",
    ))
    resolved = try
        Int(value)
    catch error
        error isa Union{InexactError,OverflowError,DomainError,MethodError} ||
            rethrow()
        throw(ArgumentError("`$label` is outside the host integer range"))
    end
    resolved > 0 || throw(ArgumentError("`$label` must be positive"))
    return resolved
end

function _rope_positive_float32(value, label::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "`$label` must be a real number",
    ))
    resolved = try
        Float32(value)
    catch error
        error isa Union{InexactError,OverflowError,DomainError,MethodError} ||
            rethrow()
        throw(ArgumentError("`$label` is not representable as Float32"))
    end
    isfinite(resolved) && resolved > 0 || throw(ArgumentError(
        "`$label` must be positive and finite at Float32 precision",
    ))
    return resolved
end

function _preflight_rope_cache_size(head_dim::Int, max_seq_len::Int)
    half_dim = BigInt(head_dim ÷ 2)
    elements = half_dim + 2 * half_dim * BigInt(max_seq_len)
    bytes = elements * sizeof(Float32)
    bytes <= typemax(Int) || throw(ArgumentError(
        "RoPE cache byte count exceeds the host integer range",
    ))
    return nothing
end

function RoPE(
    head_dim;
    max_seq_len=2048,
    theta=10000.0,
    style::Symbol=:interleaved,
)
    resolved_head_dim = _rope_positive_host_int(head_dim, "head_dim")
    resolved_max_seq_len = _rope_positive_host_int(max_seq_len, "max_seq_len")
    iseven(resolved_head_dim) || throw(ArgumentError(
        "`head_dim` must be even for RoPE",
    ))
    theta32 = _rope_positive_float32(theta, "theta")
    _validate_rope_style(style)
    _preflight_rope_cache_size(resolved_head_dim, resolved_max_seq_len)

    half_dim = resolved_head_dim ÷ 2

    inv_freq = Vector{Float32}(undef, half_dim)

    @inbounds for pair in 1:half_dim
        # pair = 1 -> dim index 0
        # pair = 2 -> dim index 2
        # pair = 3 -> dim index 4
        dim_index = 2 * (pair - 1)
        frequency = inv(
            theta32 ^ (Float32(dim_index) / Float32(resolved_head_dim)),
        )
        isfinite(frequency) || throw(ArgumentError(
            "theta produces non-finite RoPE frequencies at Float32 precision",
        ))
        inv_freq[pair] = frequency
    end

    cos_cache = Matrix{Float32}(undef, half_dim, resolved_max_seq_len)
    sin_cache = Matrix{Float32}(undef, half_dim, resolved_max_seq_len)

    @inbounds for pos_idx in 1:resolved_max_seq_len
        pos = Float32(pos_idx - 1)

        for pair in 1:half_dim
            angle = pos * inv_freq[pair]
            cos_cache[pair, pos_idx] = cos(angle)
            sin_cache[pair, pos_idx] = sin(angle)
        end
    end

    return RoPE(
        resolved_head_dim,
        resolved_max_seq_len,
        theta32,
        style,
        inv_freq,
        cos_cache,
        sin_cache,
    )
end

function _validate_rope_style(style::Symbol)
    style in (:interleaved, :rotate_half) || throw(ArgumentError(
        "`rope_style` must be `:interleaved` or `:rotate_half`; got $(repr(style))",
    ))
    return style
end

function _rope_position_bounds(
    start_pos,
    token_count::Int,
    cache_length::Int,
    bounds_message::AbstractString,
)
    start = _rope_positive_host_int(start_pos, "start_pos")
    fits = if iszero(token_count)
        start - 1 <= cache_length
    else
        token_count <= cache_length &&
            start <= cache_length - token_count + 1
    end
    fits || throw(AssertionError(bounds_message))
    stop = iszero(token_count) ? start - 1 : start + token_count - 1
    return start, stop
end

function _rope_tensor_shape(x, label::AbstractString)
    ndims(x) == 4 || throw(DimensionMismatch(
        "$label must have shape (head_dim, num_heads, seq_len, batch)",
    ))
    Base.require_one_based_indexing(x)
    return size(x)
end

function _rope_cache_shape(cache, label::AbstractString)
    ndims(cache) == 2 || throw(DimensionMismatch(
        "$label must be a (half_head_dim, max_seq_len) matrix",
    ))
    Base.require_one_based_indexing(cache)
    return size(cache)
end

function _validate_rope_storage(rope::RoPE)
    rope.head_dim > 0 && iseven(rope.head_dim) || throw(ArgumentError(
        "RoPE storage requires a positive even head_dim",
    ))
    rope.max_seq_len > 0 || throw(ArgumentError(
        "RoPE storage requires a positive max_seq_len",
    ))
    _validate_rope_style(rope.style)

    half_dim = rope.head_dim ÷ 2
    length(rope.inv_freq) == half_dim || throw(ArgumentError(
        "RoPE frequency storage does not match head_dim",
    ))
    expected_cache_shape = (half_dim, rope.max_seq_len)
    size(rope.cos_cache) == expected_cache_shape || throw(ArgumentError(
        "RoPE cosine cache storage does not match its declared dimensions",
    ))
    size(rope.sin_cache) == expected_cache_shape || throw(ArgumentError(
        "RoPE sine cache storage does not match its declared dimensions",
    ))
    return nothing
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
    start_pos=1,
)
    D, H, T, B = _rope_tensor_shape(x, "`x`")
    _rope_tensor_shape(y, "`y`")

    @assert size(y) == size(x) "`y` and `x` must have the same shape"
    @assert D == rope.head_dim "`x` head_dim does not match rope.head_dim"
    @assert iseven(D) "`head_dim` must be even for RoPE"
    _validate_rope_storage(rope)
    resolved_start, _ = _rope_position_bounds(
        start_pos,
        T,
        rope.max_seq_len,
        "`x` exceeds rope.max_seq_len",
    )

    half_dim = D ÷ 2
    cos_cache = rope.cos_cache
    sin_cache = rope.sin_cache

    @inbounds for b in 1:B
        for t in 1:T
            pos_idx = resolved_start + t - 1

            for h in 1:H
                for pair in 1:half_dim
                    i = rope.style === :interleaved ? 2 * pair - 1 : pair
                    j = rope.style === :interleaved ? i + 1 : pair + half_dim

                    c = cos_cache[pair, pos_idx]
                    s = sin_cache[pair, pos_idx]

                    x1 = x[i, h, t, b]
                    x2 = x[j, h, t, b]

                    y[i, h, t, b]     = x1 * c - x2 * s
                    y[j, h, t, b] = x1 * s + x2 * c
                end
            end
        end
    end

    return y
end

function apply_rope(
    x,
    cos_cache,
    sin_cache;
    start_pos=1,
    rope_style::Symbol=:interleaved,
)
    D, H, T, B = _rope_tensor_shape(x, "`x`")
    _rope_cache_shape(cos_cache, "`cos_cache`")
    _rope_cache_shape(sin_cache, "`sin_cache`")

    D > 0 || throw(ArgumentError("`head_dim` must be positive for RoPE"))
    @assert iseven(D) "`head_dim` must be even for RoPE"
    @assert size(cos_cache) == size(sin_cache) "RoPE cache shapes must match"
    @assert size(cos_cache, 1) == D ÷ 2 "RoPE cache head_dim does not match input"
    resolved_start, stop = _rope_position_bounds(
        start_pos,
        T,
        size(cos_cache, 2),
        "`x` exceeds RoPE cache length",
    )
    _validate_rope_style(rope_style)

    half_dim = D ÷ 2

    # The caches are model states, so Reactant moves them onto the XLA device
    # before tracing. This avoids host-to-device transfers inside the compiled
    # train step and mirrors the cache handling used by the Lux Qwen example.
    x1, x2 = if rope_style === :interleaved
        x_pairs = reshape(x, 2, half_dim, H, T, B)
        selectdim(x_pairs, 1, 1), selectdim(x_pairs, 1, 2)
    else
        x_halves = reshape(x, half_dim, 2, H, T, B)
        selectdim(x_halves, 2, 1), selectdim(x_halves, 2, 2)
    end

    positions = resolved_start:stop
    cos_values = reshape(eltype(x).(cos_cache[:, positions]), half_dim, 1, T, 1)
    sin_values = reshape(eltype(x).(sin_cache[:, positions]), half_dim, 1, T, 1)

    y1 = x1 .* cos_values .- x2 .* sin_values
    y2 = x1 .* sin_values .+ x2 .* cos_values

    if rope_style === :interleaved
        y_pairs = cat(
            reshape(y1, 1, half_dim, H, T, B),
            reshape(y2, 1, half_dim, H, T, B);
            dims=1,
        )
        return reshape(y_pairs, D, H, T, B)
    end

    return cat(y1, y2; dims=1)
end

function apply_rope(
    x,
    rope::RoPE;
    start_pos=1,
)
    _rope_tensor_shape(x, "`x`")
    @assert size(x, 1) == rope.head_dim "`x` head_dim does not match rope.head_dim"
    _validate_rope_storage(rope)
    resolved_start, _ = _rope_position_bounds(
        start_pos,
        size(x, 3),
        rope.max_seq_len,
        "`x` exceeds rope.max_seq_len",
    )

    device = get_device(x)
    cos_cache = device(eltype(x).(rope.cos_cache))
    sin_cache = device(eltype(x).(rope.sin_cache))

    return apply_rope(
        x,
        cos_cache,
        sin_cache;
        start_pos=resolved_start,
        rope_style=rope.style,
    )
end

function apply_rope_threaded!(
    y,
    x,
    rope::RoPE;
    start_pos=1,
)
    D, H, T, B = _rope_tensor_shape(x, "`x`")
    _rope_tensor_shape(y, "`y`")

    @assert size(y) == size(x)
    @assert D == rope.head_dim
    @assert iseven(D)
    _validate_rope_storage(rope)
    resolved_start, _ = _rope_position_bounds(
        start_pos,
        T,
        rope.max_seq_len,
        "`x` exceeds rope.max_seq_len",
    )

    half_dim = D ÷ 2
    cos_cache = rope.cos_cache
    sin_cache = rope.sin_cache

    Threads.@threads for b in 1:B
        @inbounds for t in 1:T
            pos_idx = resolved_start + t - 1

            for h in 1:H
                for pair in 1:half_dim
                    i = rope.style === :interleaved ? 2 * pair - 1 : pair
                    j = rope.style === :interleaved ? i + 1 : pair + half_dim

                    c = cos_cache[pair, pos_idx]
                    s = sin_cache[pair, pos_idx]

                    x1 = x[i, h, t, b]
                    x2 = x[j, h, t, b]

                    y[i, h, t, b]     = x1 * c - x2 * s
                    y[j, h, t, b] = x1 * s + x2 * c
                end
            end
        end
    end

    return y
end
