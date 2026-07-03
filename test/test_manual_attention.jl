using Test
using Random
using Lux

@testset "manual scaled dot-product attention" begin
    D = 8
    H = 2
    T = 4
    B = 3

    rng = MersenneTwister(1234)

    q = randn(rng, Float32, D, H, T, B)
    k = randn(rng, Float32, D, H, T, B)
    v = randn(rng, Float32, D, H, T, B)

    context, attn = manual_scaled_dot_product_attention(
        q,
        k,
        v;
        is_causal=true,
    )

    @test size(context) == size(q)
    @test size(attn) == (T, T, H, B)

    @test all(isfinite, context)
    @test all(isfinite, attn)

    # 每一行 attention weights 应该加和为 1
    for b in 1:B
        for h in 1:H
            for tq in 1:T
                @test isapprox(
                    sum(attn[tq, :, h, b]),
                    1.0f0;
                    atol=1.0f-5,
                )
            end
        end
    end

    # causal mask: 未来 token 的 attention 权重应该为 0
    for b in 1:B
        for h in 1:H
            for tq in 1:T
                for tk in (tq + 1):T
                    @test attn[tq, tk, h, b] == 0.0f0
                end
            end
        end
    end
end
