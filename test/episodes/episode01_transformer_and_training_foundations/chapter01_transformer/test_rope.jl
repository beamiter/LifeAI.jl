using Test
using Random
using LifeAI: RoPE, apply_rope, apply_rope!, apply_rope_threaded!

struct _Chapter01OffsetArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    offsets::NTuple{N,Int}
end

Base.size(array::_Chapter01OffsetArray) = size(array.parent)
Base.axes(array::_Chapter01OffsetArray{T,N}) where {T,N} = ntuple(N) do dimension
    parent_axis = axes(array.parent, dimension)
    offset = array.offsets[dimension]
    (first(parent_axis) + offset):(last(parent_axis) + offset)
end
Base.IndexStyle(::Type{<:_Chapter01OffsetArray}) = IndexCartesian()
function Base.getindex(
    array::_Chapter01OffsetArray{T,N},
    indices::Vararg{Int,N},
) where {T,N}
    parent_indices = ntuple(
        dimension -> indices[dimension] - array.offsets[dimension],
        N,
    )
    return getindex(array.parent, parent_indices...)
end
function Base.setindex!(
    array::_Chapter01OffsetArray{T,N},
    value,
    indices::Vararg{Int,N},
) where {T,N}
    parent_indices = ntuple(
        dimension -> indices[dimension] - array.offsets[dimension],
        N,
    )
    setindex!(array.parent, value, parent_indices...)
    return value
end

@testset "RoPE" begin
    D = 8
    H = 2
    T = 5
    B = 3

    rng = MersenneTwister(2026)

    x = randn(rng, Float32, D, H, T, B)

    rope = RoPE(
        D;
        max_seq_len=16,
        theta=10000.0,
    )

    wide_integer_rope = RoPE(
        Int32(D);
        max_seq_len=big(16),
        theta=BigFloat(10000),
    )
    @test wide_integer_rope.head_dim === D
    @test wide_integer_rope.max_seq_len === 16
    @test wide_integer_rope.theta === 10000.0f0
    @test size(wide_integer_rope.cos_cache) == (D ÷ 2, 16)
    @test size(wide_integer_rope.sin_cache) == (D ÷ 2, 16)
    @test all(isfinite, wide_integer_rope.inv_freq)
    @test all(isfinite, wide_integer_rope.cos_cache)
    @test all(isfinite, wide_integer_rope.sin_cache)

    y = apply_rope(x, rope)

    @test size(y) == size(x)
    @test eltype(y) == eltype(x)
    @test all(isfinite, y)

    # 第一个 token 的 position = 0
    # angle = 0，所以应该不发生变化。
    @test isapprox(
        y[:, :, 1, :],
        x[:, :, 1, :];
        atol=1.0f-6,
        rtol=1.0f-6,
    )

    # RoPE 是二维旋转，每一对维度的 L2 norm 应该保持不变。
    for b in 1:B
        for t in 1:T
            for h in 1:H
                for pair in 1:(D ÷ 2)
                    i = 2pair - 1

                    norm_before = x[i, h, t, b]^2 + x[i + 1, h, t, b]^2
                    norm_after = y[i, h, t, b]^2 + y[i + 1, h, t, b]^2

                    @test isapprox(
                        norm_after,
                        norm_before;
                        atol=1.0f-5,
                        rtol=1.0f-5,
                    )
                end
            end
        end
    end

    # start_pos > 1 时，第一个 token 不再是 position 0，
    # 因此通常会发生旋转。
    y_offset = apply_rope(x, rope; start_pos=Int128(4))

    @test !isapprox(
        y_offset[:, :, 1, :],
        x[:, :, 1, :];
        atol=1.0f-6,
        rtol=1.0f-6,
    )

    y_cache = apply_rope(
        x,
        rope.cos_cache,
        rope.sin_cache;
        start_pos=big(4),
    )
    y_inplace = similar(x)
    y_threaded = similar(x)
    apply_rope!(y_inplace, x, rope; start_pos=Int32(4))
    apply_rope_threaded!(y_threaded, x, rope; start_pos=big(4))
    @test y_cache == y_offset
    @test y_inplace == y_offset
    @test y_threaded == y_offset

    invalid_start_positions = (true, 0, big(typemax(Int)) + 1)
    for invalid_start in invalid_start_positions
        for operation in (
            () -> apply_rope(x, rope; start_pos=invalid_start),
            () -> apply_rope(
                x,
                rope.cos_cache,
                rope.sin_cache;
                start_pos=invalid_start,
            ),
            () -> apply_rope!(similar(x), x, rope; start_pos=invalid_start),
            () -> apply_rope_threaded!(
                similar(x),
                x,
                rope;
                start_pos=invalid_start,
            ),
        )
            error = try
                operation()
                nothing
            catch caught
                caught
            end
            @test error isa ArgumentError
            @test occursin("start_pos", sprint(showerror, error))
        end
    end

    for operation in (
        () -> apply_rope(x, rope; start_pos=typemax(Int)),
        () -> apply_rope(
            x,
            rope.cos_cache,
            rope.sin_cache;
            start_pos=typemax(Int),
        ),
        () -> apply_rope!(similar(x), x, rope; start_pos=typemax(Int)),
        () -> apply_rope_threaded!(
            similar(x),
            x,
            rope;
            start_pos=typemax(Int),
        ),
    )
        @test_throws AssertionError operation()
    end

    x_3d = reshape(copy(x), D, H, :)
    x_5d = reshape(copy(x), D, H, T, B, 1)
    for invalid_x in (x_3d, x_5d)
        for operation in (
            () -> apply_rope(invalid_x, rope),
            () -> apply_rope(
                invalid_x,
                rope.cos_cache,
                rope.sin_cache,
            ),
            () -> apply_rope!(similar(invalid_x), invalid_x, rope),
            () -> apply_rope_threaded!(
                similar(invalid_x),
                invalid_x,
                rope,
            ),
        )
            @test_throws DimensionMismatch operation()
        end
    end

    cache_vector = vec(rope.cos_cache)
    cache_3d = reshape(rope.cos_cache, size(rope.cos_cache)..., 1)
    @test_throws DimensionMismatch apply_rope(
        x,
        cache_vector,
        cache_vector,
    )
    @test_throws DimensionMismatch apply_rope(
        x,
        rope.cos_cache,
        cache_vector,
    )
    @test_throws DimensionMismatch apply_rope(x, cache_3d, cache_3d)
    @test_throws DimensionMismatch apply_rope(
        x,
        rope.cos_cache,
        cache_3d,
    )
    empty_head = zeros(Float32, 0, H, T, B)
    empty_cache = zeros(Float32, 0, 16)
    @test_throws ArgumentError apply_rope(
        empty_head,
        empty_cache,
        empty_cache,
    )

    offset_x = _Chapter01OffsetArray(x, (1, 0, 0, 0))
    offset_y = _Chapter01OffsetArray(similar(x), (1, 0, 0, 0))
    offset_cache = _Chapter01OffsetArray(rope.cos_cache, (1, 0))
    @test_throws ArgumentError apply_rope(offset_x, rope)
    @test_throws ArgumentError apply_rope(
        offset_x,
        rope.cos_cache,
        rope.sin_cache,
    )
    @test_throws ArgumentError apply_rope!(similar(x), offset_x, rope)
    @test_throws ArgumentError apply_rope_threaded!(similar(x), offset_x, rope)
    @test_throws ArgumentError apply_rope!(offset_y, x, rope)
    @test_throws ArgumentError apply_rope_threaded!(offset_y, x, rope)
    @test_throws ArgumentError apply_rope(x, offset_cache, offset_cache)
    @test_throws ArgumentError apply_rope(
        x,
        rope.cos_cache,
        offset_cache,
    )

    y_3d = reshape(similar(x), D, H, :)
    y_5d = reshape(similar(x), D, H, T, B, 1)
    for invalid_y in (y_3d, y_5d)
        @test_throws DimensionMismatch apply_rope!(invalid_y, x, rope)
        @test_throws DimensionMismatch apply_rope_threaded!(invalid_y, x, rope)
    end

    malformed_rope = RoPE(
        D,
        16,
        10000.0f0,
        :interleaved,
        ones(Float32, D ÷ 2),
        zeros(Float32, 1, 1),
        zeros(Float32, 1, 1),
    )
    @test_throws ArgumentError apply_rope(x, malformed_rope)
    @test_throws ArgumentError apply_rope!(similar(x), x, malformed_rope)
    @test_throws ArgumentError apply_rope_threaded!(
        similar(x),
        x,
        malformed_rope,
    )

    # 构造参数在任何大表分配之前严格规范到宿主表示。
    for invalid_head_dim in (true, 0, big(typemax(Int)) + 1)
        error = try
            RoPE(invalid_head_dim)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("head_dim", sprint(showerror, error))
    end
    @test_throws ArgumentError RoPE(7)

    for invalid_max_seq_len in (true, 0, big(typemax(Int)) + 1)
        error = try
            RoPE(D; max_seq_len=invalid_max_seq_len)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("max_seq_len", sprint(showerror, error))
    end

    for invalid_theta in (true, 0.0, NaN, Inf, 1.0e300, 1.0e-300)
        error = try
            RoPE(D; theta=invalid_theta)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("theta", sprint(showerror, error))
    end

    frequency_error = try
        RoPE(128; max_seq_len=1, theta=nextfloat(0.0f0))
        nothing
    catch caught
        caught
    end
    @test frequency_error isa ArgumentError
    @test occursin("non-finite RoPE frequencies", sprint(showerror, frequency_error))

    allocation_error = try
        RoPE(2; max_seq_len=typemax(Int))
        nothing
    catch caught
        caught
    end
    @test allocation_error isa ArgumentError
    @test occursin("cache byte count", sprint(showerror, allocation_error))

    # 输入 head_dim 和 rope.head_dim 不匹配时应该报错。
    x_bad = randn(rng, Float32, D + 2, H, T, B)

    @test_throws AssertionError apply_rope(x_bad, rope)
end
