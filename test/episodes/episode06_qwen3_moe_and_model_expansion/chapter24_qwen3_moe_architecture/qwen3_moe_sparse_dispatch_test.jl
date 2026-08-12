using Test
using Lux
using Random: Xoshiro
using LifeAI:
    Qwen3SparseMoE,
    qwen3_dense_expert_reference,
    qwen3_moe_forward_with_stats,
    qwen3_sparse_expert_dispatch

@testset "Qwen3 MoE sparse token-to-expert dispatch" begin
    layer = Qwen3SparseMoE(6, 5, 8, 2)
    parameters, _ = Lux.setup(Xoshiro(20260808), layer)
    x = randn(Xoshiro(20260809), Float32, 6, 7, 2)
    result = qwen3_moe_forward_with_stats(layer, x, parameters)
    dense = qwen3_dense_expert_reference(
        reshape(x, layer.d_model, :),
        result.routing,
        parameters.experts,
    )

    @test result.output ≈ reshape(dense, size(x)) atol = 2.0f-7 rtol = 2.0f-6
    @test result.stats.token_count == 14
    @test result.stats.expert_count == 8
    @test result.stats.routed_token_expert_pairs == 14 * 2
    @test result.stats.dense_token_expert_pairs == 14 * 8
    @test sum(result.stats.expert_token_counts) == 14 * 2
    @test count(!iszero, result.stats.expert_token_counts) ==
        result.stats.active_expert_count
    @test result.stats.routed_token_expert_pairs * 4 ==
        result.stats.dense_token_expert_pairs

    @test_throws DimensionMismatch qwen3_sparse_expert_dispatch(
        reshape(x, 6, :),
        result.routing[:, 1:end-1],
        parameters.experts,
    )
    @test_throws ArgumentError qwen3_sparse_expert_dispatch(
        reshape(x, 6, :),
        view(result.routing, :, :),
        parameters.experts,
    )
end

@testset "Qwen3 MoE inactive experts are not evaluated" begin
    layer = Qwen3SparseMoE(4, 3, 4, 1)
    initialized, _ = Lux.setup(Xoshiro(20260810), layer)
    router_weight = Float32[
         2  2  2  2;
         1  1  1  1;
         0  0  0  0;
        -1 -1 -1 -1
    ]
    gate_proj = copy(initialized.experts.gate_proj)
    up_proj = copy(initialized.experts.up_proj)
    down_proj = copy(initialized.experts.down_proj)
    gate_proj[:, :, 2:4] .= Float32(NaN)
    up_proj[:, :, 2:4] .= Float32(NaN)
    down_proj[:, :, 2:4] .= Float32(NaN)
    parameters = (;
        gate=(; weight=router_weight),
        experts=(; gate_proj, up_proj, down_proj),
    )
    x = ones(Float32, 4, 5, 1)

    result = qwen3_moe_forward_with_stats(layer, x, parameters)
    dense = qwen3_dense_expert_reference(
        reshape(x, 4, :),
        result.routing,
        parameters.experts,
    )
    @test all(isfinite, result.output)
    @test any(isnan, dense)
    @test result.stats.active_expert_count == 1
    @test result.stats.expert_token_counts == [5, 0, 0, 0]
    @test result.stats.routed_token_expert_pairs == 5
    @test result.stats.dense_token_expert_pairs == 20
end
