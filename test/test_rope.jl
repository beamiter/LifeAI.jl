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

    # 非偶数 head_dim 不合法。
    @test_throws AssertionError RoPE(7)

    # 输入 head_dim 和 rope.head_dim 不匹配时应该报错。
    x_bad = randn(rng, Float32, D + 2, H, T, B)

    @test_throws AssertionError apply_rope(x_bad, rope)
end