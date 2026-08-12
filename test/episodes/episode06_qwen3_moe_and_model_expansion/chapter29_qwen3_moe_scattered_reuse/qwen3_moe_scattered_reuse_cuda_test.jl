using CUDA
using LifeAI
using Test

CUDA.functional() || error(
    "Qwen3 MoE scattered reuse CUDA test requested but CUDA is not functional",
)
CUDA.allowscalar(false)

const QWEN3_MOE_REUSE_CUDA_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_REUSE_CUDA_FIXTURE = joinpath(
    QWEN3_MOE_REUSE_CUDA_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE CUDA scattered pointer/workspace reuse" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_REUSE_CUDA_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
        to_device=CUDA.cu,
    )

    first = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    repeated = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    @test repeated.logits == first.logits
    @test first.expert_cache.pointer_table_builds == 2
    @test first.expert_cache.pointer_table_reuses == 0
    @test first.expert_cache.pointer_bytes_uploaded == 192
    @test first.expert_cache.workspace_allocations == 1
    @test first.expert_cache.workspace_reuses == 1
    @test first.expert_cache.pointer_table_bytes == 192
    @test first.expert_cache.workspace_bytes > 0
    @test repeated.expert_cache.pointer_table_builds == 0
    @test repeated.expert_cache.pointer_table_reuses == 2
    @test repeated.expert_cache.pointer_bytes_uploaded == 0
    @test repeated.expert_cache.workspace_allocations == 0
    @test repeated.expert_cache.workspace_reuses == 2
    @test repeated.expert_cache.pointer_table_bytes ==
        first.expert_cache.pointer_table_bytes
    @test repeated.expert_cache.workspace_bytes ==
        first.expert_cache.workspace_bytes

    for _ in 1:20
        current = prefill_hf_qwen3_moe_offload!(session, [2, 3])
        @test current.logits == first.logits
        @test current.expert_cache.pointer_table_builds == 0
        @test current.expert_cache.workspace_allocations == 0
    end
    CUDA.synchronize()

    clear_hf_qwen3_moe_expert_cache!(session)
    cleared = qwen3_moe_expert_cache_stats(session)
    @test cleared.entries == 0
    @test cleared.pointer_table_bytes == 0
    @test cleared.workspace_bytes == 0

    rebuilt = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    @test rebuilt.logits == first.logits
    @test rebuilt.expert_cache.pointer_table_builds == 2
    @test rebuilt.expert_cache.pointer_bytes_uploaded == 192
    @test rebuilt.expert_cache.workspace_allocations == 1
end

@testset "Qwen3 MoE CUDA scattered plans reject reloaded generations" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_REUSE_CUDA_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=4 * 144,
        expert_cache_policy=:global_lru,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
        to_device=CUDA.cu,
    )
    first = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    reloaded = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    @test reloaded.logits == first.logits
    @test reloaded.expert_cache.hits == 0
    @test reloaded.expert_cache.misses == 8
    @test reloaded.expert_cache.pointer_table_builds == 2
    @test reloaded.expert_cache.pointer_table_reuses == 0
    @test reloaded.expert_cache.pointer_bytes_uploaded == 192
    @test reloaded.expert_cache.workspace_allocations == 0
    @test reloaded.expert_cache.workspace_reuses == 2
end
