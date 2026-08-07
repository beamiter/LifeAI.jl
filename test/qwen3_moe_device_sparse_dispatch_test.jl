using Test
using Lux
using Random: Xoshiro
using LifeAI:
    Qwen3SparseMoE,
    qwen3_dense_expert_reference,
    qwen3_device_sparse_expert_dispatch,
    qwen3_device_topk_routing,
    qwen3_moe_device_forward,
    qwen3_topk_routing

@testset "Qwen3 MoE compact device routing matches the dense route contract" begin
    rng = Xoshiro(20260811)
    logits = randn(rng, Float32, 8, 11)
    compact = qwen3_device_topk_routing(logits, 2)
    dense = qwen3_topk_routing(logits, 2)
    reconstructed = zeros(Float32, size(dense))
    for token in axes(logits, 2), slot in 1:2
        expert = compact.expert_indices[slot, token]
        reconstructed[expert, token] = compact.routing_weights[slot, token]
    end

    @test size(compact.expert_indices) == (2, 11)
    @test size(compact.routing_weights) == (2, 11)
    @test reconstructed ≈ dense atol = 1.0f-7 rtol = 1.0f-6
    @test all(sum(compact.routing_weights; dims=1) .≈ 1.0f0)
    @test all(1 .<= compact.expert_indices .<= 8)

    tied = qwen3_device_topk_routing(zeros(Float32, 4, 1), 3)
    @test vec(tied.expert_indices) == Int32[4, 3, 2]
    @test vec(tied.routing_weights) == fill(1.0f0 / 3.0f0, 3)
end

@testset "Qwen3 MoE route-major expert compute matches the all-expert oracle" begin
    layer = Qwen3SparseMoE(6, 5, 8, 2)
    parameters, _ = Lux.setup(Xoshiro(20260812), layer)
    x = randn(Xoshiro(20260813), Float32, 6, 7, 2)
    result = qwen3_moe_device_forward(layer, x, parameters)
    tokens = reshape(x, layer.d_model, :)
    dense_routing = zeros(Float32, layer.num_experts, size(tokens, 2))
    for token in axes(tokens, 2), slot in 1:layer.experts_per_token
        expert = result.expert_indices[slot, token]
        dense_routing[expert, token] = result.routing_weights[slot, token]
    end
    dense = qwen3_dense_expert_reference(tokens, dense_routing, parameters.experts)

    @test result.output ≈ reshape(dense, size(x)) atol = 2.0f-7 rtol = 2.0f-6
    @test length(result.expert_indices) == 2 * 14
    @test length(result.routing_weights) == 2 * 14

    @test_throws DimensionMismatch qwen3_device_sparse_expert_dispatch(
        tokens,
        result.expert_indices[:, 1:end-1],
        result.routing_weights,
        parameters.experts,
    )
end

@testset "Qwen3 MoE device dispatch never evaluates unselected experts" begin
    layer = Qwen3SparseMoE(4, 3, 4, 1)
    initialized, _ = Lux.setup(Xoshiro(20260814), layer)
    gate_proj = copy(initialized.experts.gate_proj)
    up_proj = copy(initialized.experts.up_proj)
    down_proj = copy(initialized.experts.down_proj)
    gate_proj[:, :, 2:4] .= Float32(NaN)
    up_proj[:, :, 2:4] .= Float32(NaN)
    down_proj[:, :, 2:4] .= Float32(NaN)
    parameters = (;
        gate=(; weight=Float32[
             2  2  2  2;
             1  1  1  1;
             0  0  0  0;
            -1 -1 -1 -1
        ]),
        experts=(; gate_proj, up_proj, down_proj),
    )
    result = qwen3_moe_device_forward(layer, ones(Float32, 4, 5, 1), parameters)

    @test all(result.expert_indices .== 1)
    @test all(isfinite, result.output)
    @test length(result.expert_indices) == 5
end
