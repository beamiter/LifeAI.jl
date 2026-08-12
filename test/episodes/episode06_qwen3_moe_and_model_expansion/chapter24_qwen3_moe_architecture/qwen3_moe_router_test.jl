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
    @test_throws ArgumentError Qwen3SparseMoE(4, 3, 0, 1)
    @test_throws ArgumentError Qwen3SparseMoE(4, 3, 4, 5)
end
