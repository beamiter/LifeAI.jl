using Test
using Lux
using Random: Xoshiro
using LifeAI:
    Qwen3SparseMoE,
    qwen3_dense_expert_reference,
    qwen3_device_sparse_expert_dispatch,
    qwen3_device_topk_routing,
    qwen3_moe_device_forward,
    qwen3_moe_forward_with_stats,
    qwen3_topk_routing

struct _OversizedQwen3Router <: AbstractMatrix{Float32} end

Base.size(::_OversizedQwen3Router) = (Int(typemax(Int32)) + 1, 0)
Base.getindex(::_OversizedQwen3Router, ::Int, ::Int) = error(
    "oversized router elements must not be accessed",
)

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

    for normalize in (true, false)
        all_tied_logits = zeros(Float32, 4, 2)
        compact_tied = qwen3_device_topk_routing(
            all_tied_logits,
            3;
            normalize,
        )
        dense_tied = qwen3_topk_routing(all_tied_logits, 3; normalize)
        reconstructed_tied = zeros(Float32, size(dense_tied))
        for token in axes(all_tied_logits, 2), slot in 1:3
            expert = compact_tied.expert_indices[slot, token]
            reconstructed_tied[expert, token] =
                compact_tied.routing_weights[slot, token]
        end
        @test compact_tied.expert_indices == repeat(Int32[4, 3, 2], 1, 2)
        @test reconstructed_tied == dense_tied
        @test all(iszero, dense_tied[1, :])

        boundary_logits = reshape(Float32[3, 2, 2, 2], :, 1)
        compact_boundary = qwen3_device_topk_routing(
            boundary_logits,
            2;
            normalize,
        )
        dense_boundary = qwen3_topk_routing(boundary_logits, 2; normalize)
        reconstructed_boundary = zeros(Float32, size(dense_boundary))
        for slot in 1:2
            expert = compact_boundary.expert_indices[slot, 1]
            reconstructed_boundary[expert, 1] =
                compact_boundary.routing_weights[slot, 1]
        end
        @test vec(compact_boundary.expert_indices) == Int32[1, 4]
        @test reconstructed_boundary == dense_boundary
        @test findall(!iszero, vec(dense_boundary)) == [1, 4]
    end

    poisoned_columns = (
        reshape(Float32[NaN, 0, 0, 0], :, 1),
        fill(-Inf32, 4, 1),
        reshape(Float32[Inf, 0, 0, 0], :, 1),
    )
    for poisoned in poisoned_columns, normalize in (true, false)
        compact_poisoned = qwen3_device_topk_routing(
            poisoned,
            2;
            normalize,
        )
        indices = vec(compact_poisoned.expert_indices)
        @test all(1 .<= indices .<= size(poisoned, 1))
        @test length(unique(indices)) == 2
        @test all(isnan, compact_poisoned.routing_weights)
    end
end

@testset "Qwen3 device router controls and index range are strict" begin
    logits = reshape(Float32[4, 3, 2, 1], :, 1)
    expected = qwen3_device_topk_routing(logits, 2)
    for count in (Int32(2), UInt8(2), big(2))
        actual = qwen3_device_topk_routing(logits, count)
        @test actual.expert_indices == expected.expert_indices
        @test actual.routing_weights == expected.routing_weights
    end

    oversized = big(typemax(Int)) + 1
    for invalid_count in (true, 2.0, oversized)
        @test_throws ArgumentError qwen3_device_topk_routing(
            logits,
            invalid_count,
        )
    end
    for invalid_normalize in (0, 1, :yes, nothing)
        @test_throws ArgumentError qwen3_device_topk_routing(
            logits,
            2;
            normalize=invalid_normalize,
        )
    end
    @test_throws ArgumentError qwen3_device_topk_routing(
        _OversizedQwen3Router(),
        1,
    )
end

@testset "Qwen3 extreme routing logits have backend-stable candidates" begin
    underflow_logits = reshape(Float32[0, -90, -91, -92], :, 1)
    for normalize in (true, false)
        compact = qwen3_device_topk_routing(
            underflow_logits,
            2;
            normalize,
        )
        dense = qwen3_topk_routing(underflow_logits, 2; normalize)
        @test vec(compact.expert_indices) == Int32[1, 2]
        @test vec(compact.routing_weights) == Float32[1, 0]
        @test vec(dense) == Float32[1, 0, 0, 0]
    end

    normal_logits = reshape(Float32[0, -80, -81, -82], :, 1)
    compact = qwen3_device_topk_routing(normal_logits, 2; normalize=false)
    @test vec(compact.expert_indices) == Int32[1, 2]
    @test compact.routing_weights[2, 1] >= floatmin(Float32)
end

@testset "Qwen3 compact dispatch masks underflowed zero-weight routes" begin
    layer = Qwen3SparseMoE(1, 1, 4, 2)
    poisoned_experts = reshape(Float32[1, NaN, NaN, NaN], 1, 1, 4)
    parameters = (;
        gate=(; weight=reshape(Float32[0, -200, -201, -202], 4, 1)),
        experts=(;
            gate_proj=copy(poisoned_experts),
            up_proj=copy(poisoned_experts),
            down_proj=copy(poisoned_experts),
        ),
    )
    input = ones(Float32, 1, 1, 1)

    host = qwen3_moe_forward_with_stats(layer, input, parameters)
    device = qwen3_moe_device_forward(layer, input, parameters)
    dense = qwen3_dense_expert_reference(
        reshape(input, 1, :),
        host.routing,
        parameters.experts,
    )
    @test vec(device.expert_indices) == Int32[1, 2]
    @test vec(device.routing_weights) == Float32[1, 0]
    @test host.stats.routed_token_expert_pairs == 1
    @test all(isfinite, host.output)
    @test all(isfinite, dense)
    @test all(isfinite, device.output)
    @test dense ≈ reshape(host.output, 1, :)
    @test device.output ≈ host.output
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
