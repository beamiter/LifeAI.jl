using Test
using NNlib: softmax
using LifeAI: Qwen3SparseMoE, qwen3_topk_routing

@testset "Qwen3 MoE top-k router selection and normalization" begin
    logits = Float32[
        4.0  -1.0   0.2;
        1.0   3.0   0.1;
        2.0   2.0   5.0;
       -3.0   0.5  -2.0
    ]
    probabilities = softmax(logits; dims=1)
    routing = qwen3_topk_routing(logits, 2)

    @test size(routing) == size(logits)
    @test vec(sum(routing; dims=1)) ≈ ones(Float32, 3)
    @test vec(sum(routing .> 0; dims=1)) == [2, 2, 2]
    @test findall(!iszero, routing[:, 1]) == [1, 3]
    @test findall(!iszero, routing[:, 2]) == [2, 3]
    @test findall(!iszero, routing[:, 3]) == [1, 3]
    @test routing[[1, 3], 1] ≈ probabilities[[1, 3], 1] ./
        sum(probabilities[[1, 3], 1])

    unnormalized = qwen3_topk_routing(logits, 2; normalize=false)
    @test unnormalized[routing .> 0] ≈ probabilities[routing .> 0]
    @test all(iszero, unnormalized[routing .== 0])

    @test_throws ArgumentError qwen3_topk_routing(logits, 0)
    @test_throws ArgumentError qwen3_topk_routing(logits, 5)
    for invalid_logits in (
        reshape(Float32[NaN, 0, 0, 0], :, 1),
        fill(-Inf32, 4, 1),
        reshape(Float32[Inf, 0, 0, 0], :, 1),
        reshape(Float64[1.0e300, 0, 0, 0], :, 1),
    )
        @test_throws ArgumentError qwen3_topk_routing(invalid_logits, 2)
    end
    @test_throws ArgumentError Qwen3SparseMoE(4, 3, 0, 1)
    @test_throws ArgumentError Qwen3SparseMoE(4, 3, 4, 5)
end

@testset "Qwen3 host router controls are strict host values" begin
    logits = reshape(Float32[4, 3, 2, 1], :, 1)
    @test qwen3_topk_routing(logits, Int32(2)) ==
        qwen3_topk_routing(logits, 2)
    @test qwen3_topk_routing(logits, big(2); normalize=false) ==
        qwen3_topk_routing(logits, 2; normalize=false)

    oversized = big(typemax(Int)) + 1
    for invalid_count in (true, 2.0, oversized)
        @test_throws ArgumentError qwen3_topk_routing(logits, invalid_count)
    end
    for invalid_normalize in (0, 1, :yes, nothing)
        @test_throws ArgumentError qwen3_topk_routing(
            logits,
            2;
            normalize=invalid_normalize,
        )
    end
end

@testset "Qwen3 sparse MoE construction validates every entry point" begin
    layer = Qwen3SparseMoE(
        Int32(4),
        big(3),
        UInt8(4),
        Int128(2);
        normalize_routing=false,
    )
    @test layer.d_model === 4
    @test layer.hidden_dim === 3
    @test layer.num_experts === 4
    @test layer.experts_per_token === 2
    @test layer.normalize_routing === false

    positional = Qwen3SparseMoE(4, 3, 4, 2, true)
    @test positional == Qwen3SparseMoE(4, 3, 4, 2)

    oversized = big(typemax(Int)) + 1
    for invalid_dimension in (true, 4.0, oversized)
        @test_throws ArgumentError Qwen3SparseMoE(
            invalid_dimension,
            3,
            4,
            2,
            false,
        )
        @test_throws ArgumentError Qwen3SparseMoE(
            4,
            invalid_dimension,
            4,
            2;
            normalize_routing=false,
        )
        @test_throws ArgumentError Qwen3SparseMoE(4, 3, invalid_dimension, 2)
        @test_throws ArgumentError Qwen3SparseMoE(4, 3, 4, invalid_dimension)
    end

    for invalid_normalize in (0, 1, :yes, nothing)
        @test_throws ArgumentError Qwen3SparseMoE(
            4,
            3,
            4,
            2,
            invalid_normalize,
        )
        @test_throws ArgumentError Qwen3SparseMoE(
            4,
            3,
            4,
            2;
            normalize_routing=invalid_normalize,
        )
    end

    @test_throws ArgumentError Qwen3SparseMoE(0, 3, 4, 2, false)
    @test_throws ArgumentError Qwen3SparseMoE(4, 0, 4, 2, false)
    @test_throws ArgumentError Qwen3SparseMoE(4, 3, 0, 2, false)
    @test_throws ArgumentError Qwen3SparseMoE(4, 3, 4, 0, false)
    @test_throws ArgumentError Qwen3SparseMoE(4, 3, 4, 5, false)
end
