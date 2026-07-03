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

@testset "batched scaled dot-product attention" begin
    D = 8
    H = 2
    T = 4
    B = 3

    rng = MersenneTwister(1234)

    q = randn(rng, Float32, D, H, T, B)
    k = randn(rng, Float32, D, H, T, B)
    v = randn(rng, Float32, D, H, T, B)

    for is_causal in (true, false)
        context_manual, attn_manual = manual_scaled_dot_product_attention(
            q,
            k,
            v;
            is_causal,
        )

        context_batched, attn_batched = batched_scaled_dot_product_attention(
            q,
            k,
            v;
            is_causal,
        )

        @test size(context_batched) == size(q)
        @test size(attn_batched) == (T, T, H, B)

        @test all(isfinite, context_batched)
        @test all(isfinite, attn_batched)

        # batched 版本应该和 manual 版本数值接近
        @test isapprox(
            context_batched,
            context_manual;
            atol=1.0f-5,
            rtol=1.0f-5,
        )

        @test isapprox(
            attn_batched,
            attn_manual;
            atol=1.0f-5,
            rtol=1.0f-5,
        )

        # 每一行 attention weights 应该加和为 1
        for b in 1:B
            for h in 1:H
                for tq in 1:T
                    @test isapprox(
                        sum(attn_batched[tq, :, h, b]),
                        1.0f0;
                        atol=1.0f-5,
                    )
                end
            end
        end

        if is_causal
            # causal mask: 未来 token 的 attention 权重应该为 0
            for b in 1:B
                for h in 1:H
                    for tq in 1:T
                        for tk in (tq + 1):T
                            @test attn_batched[tq, tk, h, b] == 0.0f0
                        end
                    end
                end
            end
        end
    end
end

@testset "batched scaled dot-product attention cross attention" begin
    D = 8
    H = 2
    Tq = 3
    Tk = 5
    B = 3

    rng = MersenneTwister(5678)

    q = randn(rng, Float32, D, H, Tq, B)
    k = randn(rng, Float32, D, H, Tk, B)
    v = randn(rng, Float32, D, H, Tk, B)

    for is_causal in (true, false)
        context_manual, attn_manual = manual_scaled_dot_product_attention(
            q,
            k,
            v;
            is_causal,
        )

        context_batched, attn_batched = batched_scaled_dot_product_attention(
            q,
            k,
            v;
            is_causal,
        )

        @test size(context_batched) == size(q)
        @test size(attn_batched) == (Tq, Tk, H, B)

        @test all(isfinite, context_batched)
        @test all(isfinite, attn_batched)

        @test isapprox(
            context_batched,
            context_manual;
            atol=1.0f-5,
            rtol=1.0f-5,
        )

        @test isapprox(
            attn_batched,
            attn_manual;
            atol=1.0f-5,
            rtol=1.0f-5,
        )

        for b in 1:B
            for h in 1:H
                for tq in 1:Tq
                    @test isapprox(
                        sum(attn_batched[tq, :, h, b]),
                        1.0f0;
                        atol=1.0f-5,
                    )
                end
            end
        end

        if is_causal
            for b in 1:B
                for h in 1:H
                    for tq in 1:Tq
                        for tk in (tq + 1):Tk
                            @test attn_batched[tq, tk, h, b] == 0.0f0
                        end
                    end
                end
            end
        end
    end
end