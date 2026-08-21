using BFloat16s: BFloat16
using CUDA
using LifeAI
using Random: Xoshiro
using Test

CUDA.functional() || error(
    "Qwen3 MoE grouped scattered CUDA test requested but CUDA is not functional",
)
CUDA.allowscalar(false)

const QWEN3_MOE_GROUPED_SCATTERED_CUDA_EXT =
    Base.get_extension(LifeAI, :LifeAICUDAExt)

function grouped_scattered_case(hidden_dim::Int; seed::Int)
    d_model = 32
    num_experts = 4
    experts_per_token = 2
    num_tokens = 5
    rng = Xoshiro(seed)
    tokens = randn(rng, Float32, d_model, num_tokens)
    expert_indices = Int32[1 4 2 3 1; 3 2 4 1 2]
    routing_weights = Float32[
        0.7 0.6 0.8 0.55 0.9
        0.3 0.4 0.2 0.45 0.1
    ]
    parameters = (;
        gate_proj=BFloat16.(0.02f0 .* randn(
            rng,
            Float32,
            hidden_dim,
            d_model,
            num_experts,
        )),
        up_proj=BFloat16.(0.02f0 .* randn(
            rng,
            Float32,
            hidden_dim,
            d_model,
            num_experts,
        )),
        down_proj=BFloat16.(0.02f0 .* randn(
            rng,
            Float32,
            d_model,
            hidden_dim,
            num_experts,
        )),
    )
    parameters_gpu = map(CUDA.cu, parameters)
    entry_bytes = sizeof(BFloat16) * 3 * d_model * hidden_dim
    entries = LifeAI._Qwen3MoEExpertCacheEntry[
        LifeAI._Qwen3MoEExpertCacheEntry(
            copy(parameters_gpu.gate_proj[:, :, expert]),
            copy(parameters_gpu.up_proj[:, :, expert]),
            copy(parameters_gpu.down_proj[:, :, expert]),
            entry_bytes,
            expert,
        )
        for expert in 1:num_experts
    ]
    scattered = LifeAI._Qwen3MoEScatteredExperts(
        [3, 7, 11, 19],
        collect(1:num_experts),
        entries,
        hidden_dim,
        d_model,
    )
    return (;
        tokens=CUDA.cu(tokens),
        expert_indices=CUDA.cu(expert_indices),
        routing_weights=CUDA.cu(routing_weights),
        parameters_host=parameters,
        parameters=parameters_gpu,
        scattered,
    )
end

@testset "Qwen3 MoE grouped WMMA consumes scattered expert pointers" begin
    for (hidden_dim, seed) in ((16, 20260841), (32, 20260842))
        case = grouped_scattered_case(hidden_dim; seed)
        expected =
            QWEN3_MOE_GROUPED_SCATTERED_CUDA_EXT.qwen3_cuda_grouped_bf16_sparse_expert_dispatch(
                case.tokens,
                case.expert_indices,
                case.routing_weights,
                case.parameters,
            )
        first = LifeAI._qwen3_scattered_expert_dispatch(
            case.tokens,
            case.expert_indices,
            case.routing_weights,
            case.scattered,
            nothing,
            1,
            true,
        )
        second = LifeAI._qwen3_scattered_expert_dispatch(
            case.tokens,
            case.expert_indices,
            case.routing_weights,
            case.scattered,
            first.state,
            1,
            true,
        )
        CUDA.synchronize()

        @test Array(first.output) == Array(expected)
        @test Array(second.output) == Array(expected)
        @test first.pointer_table_built
        @test !first.pointer_table_reused
        @test first.pointer_bytes_uploaded == 3 * 4 * sizeof(CUDA.CuPtr{BFloat16})
        @test first.workspace_allocated
        @test !first.workspace_reused
        @test !second.pointer_table_built
        @test second.pointer_table_reused
        @test second.pointer_bytes_uploaded == 0
        @test !second.workspace_allocated
        @test second.workspace_reused
        @test second.pointer_table_bytes == first.pointer_table_bytes
        @test second.workspace_bytes == first.workspace_bytes > 0

        reloaded_parameters_host = map(case.parameters_host) do parameter
            BFloat16.(1.5f0 .* Float32.(parameter) .+ 0.01f0)
        end
        reloaded_parameters = map(CUDA.cu, reloaded_parameters_host)
        entry_bytes = sizeof(BFloat16) * 3 * 32 * hidden_dim
        reloaded_entries = LifeAI._Qwen3MoEExpertCacheEntry[
            LifeAI._Qwen3MoEExpertCacheEntry(
                copy(reloaded_parameters.gate_proj[:, :, expert]),
                copy(reloaded_parameters.up_proj[:, :, expert]),
                copy(reloaded_parameters.down_proj[:, :, expert]),
                entry_bytes,
                expert + 100,
            )
            for expert in 1:4
        ]
        reloaded = LifeAI._Qwen3MoEScatteredExperts(
            case.scattered.expert_ids,
            case.scattered.generations .+ 100,
            reloaded_entries,
            hidden_dim,
            32,
        )
        expected_reloaded =
            QWEN3_MOE_GROUPED_SCATTERED_CUDA_EXT.qwen3_cuda_grouped_bf16_sparse_expert_dispatch(
                case.tokens,
                case.expert_indices,
                case.routing_weights,
                reloaded_parameters,
            )
        rebuilt = LifeAI._qwen3_scattered_expert_dispatch(
            case.tokens,
            case.expert_indices,
            case.routing_weights,
            reloaded,
            second.state,
            1,
            true,
        )
        CUDA.synchronize()
        @test Array(rebuilt.output) == Array(expected_reloaded)
        @test Array(rebuilt.output) != Array(expected)
        @test rebuilt.pointer_table_built
        @test !rebuilt.pointer_table_reused
        @test rebuilt.workspace_reused
    end
end

@testset "Qwen3 MoE grouped scattered retained state is shape bounded" begin
    case = grouped_scattered_case(32; seed=20260843)
    state = nothing
    for num_tokens in 1:5
        result = LifeAI._qwen3_scattered_expert_dispatch(
            case.tokens[:, 1:num_tokens],
            case.expert_indices[:, 1:num_tokens],
            case.routing_weights[:, 1:num_tokens],
            case.scattered,
            state,
            1,
            true,
        )
        state = result.state
    end
    CUDA.synchronize()
    @test length(state.grouped_workspaces) == 4
    @test length(state.grouped_workspace_last_used) == 4
    @test state.workspace_bytes == sum(
        workspace.bytes for workspace in values(state.grouped_workspaces)
    )
    @test isempty(state.workspaces)

    plan_bytes = 3 * 4 * sizeof(CUDA.CuPtr{BFloat16})
    for generation in 2:6
        regenerated = LifeAI._Qwen3MoEScatteredExperts(
            case.scattered.expert_ids,
            fill(generation, 4),
            case.scattered.entries,
            32,
            32,
        )
        result = LifeAI._qwen3_scattered_expert_dispatch(
            case.tokens,
            case.expert_indices,
            case.routing_weights,
            regenerated,
            state,
            1,
            true,
        )
        state = result.state
    end
    CUDA.synchronize()
    @test length(state.pointer_plans[1]) == 4
    @test state.pointer_table_bytes == 4 * plan_bytes

    original_after_eviction = LifeAI._qwen3_scattered_expert_dispatch(
        case.tokens,
        case.expert_indices,
        case.routing_weights,
        case.scattered,
        state,
        1,
        true,
    )
    @test original_after_eviction.pointer_table_built
    @test !original_after_eviction.pointer_table_reused
    @test original_after_eviction.pointer_table_bytes == 4 * plan_bytes
end
