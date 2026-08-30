using Test
using Lux
using Random: Xoshiro
using LifeAI:
    Qwen3MoEDispatchStats,
    Qwen3SparseMoE,
    qwen3_dense_expert_reference,
    qwen3_moe_forward_with_stats,
    qwen3_sparse_expert_dispatch

@testset "Qwen3 MoE dispatch statistics enforce count invariants" begin
    source_counts = Int32[2, 0, 1]
    stats = Qwen3MoEDispatchStats(
        Int32(2),
        UInt8(3),
        big(2),
        Int128(3),
        Int64(6),
        source_counts,
    )
    @test stats.token_count === 2
    @test stats.expert_count === 3
    @test stats.active_expert_count === 2
    @test stats.routed_token_expert_pairs === 3
    @test stats.dense_token_expert_pairs === 6
    @test stats.expert_token_counts == [2, 0, 1]
    @test stats.expert_token_counts isa Vector{Int}
    source_counts[1] = 0
    @test stats.expert_token_counts == [2, 0, 1]

    empty_stats = Qwen3MoEDispatchStats(0, 0, 0, 0, 0, Int[])
    @test isempty(empty_stats.expert_token_counts)

    valid_fields = (2, 3, 2, 3, 6, [2, 0, 1])
    oversized = big(typemax(Int)) + 1
    for field in 1:5, invalid_value in (true, 1.0, -1, oversized)
        invalid_fields = Base.setindex(valid_fields, invalid_value, field)
        @test_throws ArgumentError Qwen3MoEDispatchStats(invalid_fields...)
    end
    for invalid_counts in ((2, 0, 1), reshape([2, 0, 1], 1, :), nothing)
        @test_throws ArgumentError Qwen3MoEDispatchStats(
            valid_fields[1:5]...,
            invalid_counts,
        )
    end
    for invalid_count in (true, 1.0, -1, oversized)
        @test_throws ArgumentError Qwen3MoEDispatchStats(
            valid_fields[1:5]...,
            Any[2, 0, invalid_count],
        )
    end
    @test_throws ArgumentError Qwen3MoEDispatchStats(2, 2, 1, 2, 4, [2])
    @test_throws ArgumentError Qwen3MoEDispatchStats(2, 3, 2, 4, 6, [3, 0, 1])
    @test_throws ArgumentError Qwen3MoEDispatchStats(2, 3, 1, 3, 6, [2, 0, 1])
    @test_throws ArgumentError Qwen3MoEDispatchStats(2, 3, 2, 4, 6, [2, 0, 1])
    @test_throws ArgumentError Qwen3MoEDispatchStats(2, 3, 2, 3, 7, [2, 0, 1])
    @test_throws ArgumentError Qwen3MoEDispatchStats(
        typemax(Int),
        2,
        0,
        0,
        0,
        [0, 0],
    )
end

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
    @test all(isfinite, dense)
    @test reshape(result.output, 4, :) ≈ dense
    @test result.stats.active_expert_count == 1
    @test result.stats.expert_token_counts == [5, 0, 0, 0]
    @test result.stats.routed_token_expert_pairs == 5
    @test result.stats.dense_token_expert_pairs == 20
end
