using Test
using BFloat16s: BFloat16
using CUDA
using Lux
using Random: Xoshiro
import LifeAI
using LifeAI:
    Qwen3SparseMoE,
    qwen3_cuda_indexed_workspace_bytes,
    qwen3_device_sparse_expert_dispatch,
    qwen3_device_topk_routing,
    qwen3_moe_device_forward,
    qwen3_route_major_expert_dispatch

CUDA.functional() || error(
    "Qwen3 MoE CUDA dispatch test requested but CUDA is not functional",
)
CUDA.allowscalar(false)
const _QWEN3_MOE_CUDA_EXT = Base.get_extension(LifeAI, :LifeAICUDAExt)

@testset "Qwen3 MoE CUDA route bucketing is stable and device resident" begin
    expert_indices = CUDA.cu(Int32[
        3 2 3 1;
        1 3 2 2
    ])
    buckets = _QWEN3_MOE_CUDA_EXT.qwen3_cuda_bucket_routes(
        expert_indices,
        4,
    )
    @test Array(buckets.sorted_experts) == Int32[1, 1, 2, 2, 2, 3, 3, 3]
    @test Array(buckets.route_permutation) == Int32[2, 7, 3, 6, 8, 1, 4, 5]
    @test Array(buckets.expert_counts) == Int32[2, 3, 3, 0]
    @test Array(buckets.expert_offsets) == Int32[1, 3, 6, 9, 9]
    @test all(value -> value isa CUDA.CuArray, values(buckets))
    @test !_QWEN3_MOE_CUDA_EXT._qwen3_cuda_use_bucketed_dispatch(2_048, 768, 16)
    @test _QWEN3_MOE_CUDA_EXT._qwen3_cuda_use_bucketed_dispatch(2_048, 768, 32)
    @test !_QWEN3_MOE_CUDA_EXT._qwen3_cuda_use_bucketed_dispatch(128, 64, 64)
end

@testset "Qwen3 MoE CUDA wide prefill selects bucketed production dispatch" begin
    d_model = 1_024
    hidden_dim = 1_024
    num_tokens = 32
    tokens = CUDA.cu(randn(
        Xoshiro(20260824),
        Float32,
        d_model,
        num_tokens,
    ))
    expert_indices = CUDA.fill(Int32(1), 1, num_tokens)
    routing_weights = CUDA.fill(1.0f0, 1, num_tokens)
    parameter_value = BFloat16(0.0001)
    expert_parameters = (;
        gate_proj=CUDA.fill(parameter_value, hidden_dim, d_model, 1),
        up_proj=CUDA.fill(parameter_value, hidden_dim, d_model, 1),
        down_proj=CUDA.fill(parameter_value, d_model, hidden_dim, 1),
    )
    production = qwen3_device_sparse_expert_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters,
    )
    bucketed =
        _QWEN3_MOE_CUDA_EXT.qwen3_cuda_bucketed_sparse_expert_dispatch(
            tokens,
            expert_indices,
            routing_weights,
            expert_parameters,
        )
    @test Array(production) == Array(bucketed)
    @test size(production) == (d_model, num_tokens)
    @test all(isfinite, Array(production))
end

@testset "Qwen3 MoE compact sparse dispatch runs on CUDA" begin
    layer = Qwen3SparseMoE(16, 12, 8, 2)
    parameters, state = Lux.setup(Xoshiro(20260817), layer)
    x = randn(Xoshiro(20260818), Float32, 16, 9, 2)
    reference = qwen3_moe_device_forward(layer, x, parameters)
    parameters_gpu = CUDA.cu(parameters)
    x_gpu = CUDA.cu(x)

    actual_gpu, _ = layer(x_gpu, parameters_gpu, state)
    actual = Array(actual_gpu)
    tokens_gpu = reshape(x_gpu, layer.d_model, :)
    routes_gpu = qwen3_device_topk_routing(
        parameters_gpu.gate.weight * tokens_gpu,
        layer.experts_per_token,
    )
    route_major_gpu = qwen3_route_major_expert_dispatch(
        tokens_gpu,
        routes_gpu.expert_indices,
        routes_gpu.routing_weights,
        parameters_gpu.experts,
    )
    @test actual ≈ reference.output atol = 3.0f-6 rtol = 3.0f-5
    @test actual ≈ reshape(Array(route_major_gpu), size(actual)) atol = 3.0f-6 rtol = 3.0f-5
    @test size(actual) == size(x)
    @test all(isfinite, actual)
end

@testset "Qwen3 MoE CUDA bucketed dispatch preserves original route order" begin
    layer = Qwen3SparseMoE(32, 24, 8, 3)
    parameters, state = Lux.setup(Xoshiro(20260822), layer)
    x = randn(Xoshiro(20260823), Float32, 32, 11, 1)
    parameters_gpu = CUDA.cu(parameters)
    x_gpu = CUDA.cu(x)
    baseline_gpu, _ = layer(x_gpu, parameters_gpu, state)
    tokens_gpu = reshape(x_gpu, layer.d_model, :)
    routes_gpu = qwen3_device_topk_routing(
        parameters_gpu.gate.weight * tokens_gpu,
        layer.experts_per_token,
    )
    bucketed_gpu =
        _QWEN3_MOE_CUDA_EXT.qwen3_cuda_bucketed_sparse_expert_dispatch(
            tokens_gpu,
            routes_gpu.expert_indices,
            routes_gpu.routing_weights,
            parameters_gpu.experts,
        )
    baseline = Array(baseline_gpu)
    bucketed = reshape(Array(bucketed_gpu), size(baseline))
    @test bucketed == baseline
    @test size(bucketed) == size(x)
    @test all(isfinite, bucketed)
end

@testset "Qwen3 MoE CUDA indexed dispatch accepts BF16 expert weights" begin
    layer = Qwen3SparseMoE(16, 12, 8, 2)
    parameters, state = Lux.setup(Xoshiro(20260820), layer)
    x = randn(Xoshiro(20260821), Float32, 16, 7, 1)
    experts_bf16 = map(
        array -> BFloat16.(array),
        parameters.experts,
    )
    reference_parameters = (;
        gate=parameters.gate,
        experts=map(array -> Float32.(array), experts_bf16),
    )
    gpu_parameters = (;
        gate=CUDA.cu(parameters.gate),
        experts=CUDA.cu(experts_bf16),
    )
    reference = qwen3_moe_device_forward(layer, x, reference_parameters)
    actual_gpu, _ = layer(CUDA.cu(x), gpu_parameters, state)
    actual = Array(actual_gpu)
    @test eltype(gpu_parameters.experts.gate_proj) == BFloat16
    @test actual ≈ reference.output atol = 3.0f-5 rtol = 3.0f-4
    @test size(actual) == size(x)
    @test all(isfinite, actual)
end

@testset "Qwen3 MoE CUDA dispatch skips unselected NaN experts" begin
    layer = Qwen3SparseMoE(4, 3, 4, 1)
    initialized, state = Lux.setup(Xoshiro(20260819), layer)
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

    output_gpu, _ = layer(
        CUDA.cu(ones(Float32, 4, 5, 1)),
        CUDA.cu(parameters),
        state,
    )
    output = Array(output_gpu)
    @test all(isfinite, output)
    @test size(output) == (4, 5, 1)
end

@testset "Qwen3 MoE CUDA indexed kernels remove route weight materialization" begin
    workspace = qwen3_cuda_indexed_workspace_bytes(128, 64, 64, 8)
    route_major_weights = 64 * 8 * 3 * 128 * 64 * sizeof(Float32)
    @test workspace == 425_984
    @test route_major_weights == 50_331_648
    @test route_major_weights / workspace > 118
end
