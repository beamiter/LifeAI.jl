using Test
using Random
using LifeAI: RoPE, apply_rope

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
    y_offset = apply_rope(x, rope; start_pos=4)

    @test !isapprox(
        y_offset[:, :, 1, :],
        x[:, :, 1, :];
        atol=1.0f-6,
        rtol=1.0f-6,
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
