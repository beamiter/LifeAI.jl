using CUDA
using LifeAI
using Test

CUDA.functional() || error(
    "Qwen3 MoE scattered cache CUDA test requested but CUDA is not functional",
)
CUDA.allowscalar(false)

const QWEN3_MOE_SCATTERED_CUDA_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_SCATTERED_CUDA_FIXTURE = joinpath(
    QWEN3_MOE_SCATTERED_CUDA_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE CUDA scattered cache dispatch" begin
    function tiny_session(dispatch; gc_interval=1)
        return load_hf_qwen3_moe_offload_session(
            QWEN3_MOE_SCATTERED_CUDA_FIXTURE;
            context_tokens=8,
            prefill_chunk_tokens=2,
            grouped_experts=false,
            expert_cache_budget_bytes=8 * 144,
            expert_cache_dispatch=dispatch,
            expert_gc_interval_layers=gc_interval,
            to_device=CUDA.cu,
        )
    end

    materialized = tiny_session(:materialized)
    materialized_first = prefill_hf_qwen3_moe_offload!(materialized, [2, 3])
    materialized_hit = prefill_hf_qwen3_moe_offload!(materialized, [2, 3])
    scattered = tiny_session(:scattered; gc_interval=0)
    scattered_first = prefill_hf_qwen3_moe_offload!(scattered, [2, 3])
    scattered_hit = prefill_hf_qwen3_moe_offload!(scattered, [2, 3])
    CUDA.synchronize()

    @test scattered_first.logits == materialized_first.logits
    @test scattered_hit.logits == materialized_hit.logits
    @test scattered_hit.expert_bytes_read == 0
    @test scattered_hit.expert_bytes_uploaded == 0
    @test scattered_hit.expert_cache.hits == 8
    @test scattered_hit.expert_cache.active_materializations == 0
    @test scattered_hit.expert_cache.active_materialization_bytes == 0
    @test scattered_hit.expert_cache.scattered_dispatches == 2
    @test scattered_hit.expert_cache.pointer_bytes_uploaded == 192
    @test scattered_hit.expert_cache.forced_gc_calls == 0
    @test materialized_hit.expert_cache.active_materializations == 2
    @test materialized_hit.expert_cache.active_materialization_bytes == 8 * 144
    @test materialized_hit.expert_cache.forced_gc_calls == 2
end
