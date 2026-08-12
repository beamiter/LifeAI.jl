using CUDA
using LifeAI
using Test

CUDA.functional() || error(
    "Qwen3 MoE async miss CUDA test requested but CUDA is not functional",
)
CUDA.allowscalar(false)

const QWEN3_MOE_ASYNC_CUDA_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_ASYNC_CUDA_FIXTURE = joinpath(
    QWEN3_MOE_ASYNC_CUDA_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE CUDA pinned asynchronous expert upload" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_ASYNC_CUDA_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
        to_device=CUDA.cu,
    )
    sequential = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    configure_hf_qwen3_moe_expert_cache!(
        session;
        miss_pipeline=:overlapped,
        read_workers=2,
        pinned_upload=true,
    )
    overlapped = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    @test overlapped.logits == sequential.logits
    @test overlapped.expert_cache.pinned_upload
    @test overlapped.expert_cache.read_tasks == 8
    @test overlapped.expert_cache.pinned_bytes_uploaded ==
        overlapped.expert_cache.bytes_uploaded == 8 * 144
    @test overlapped.expert_cache.upload_wait_seconds >= 0
    @test overlapped.expert_cache.pointer_table_builds == 2

    hit = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    @test hit.logits == sequential.logits
    @test hit.expert_cache.bytes_read == 0
    @test hit.expert_cache.bytes_uploaded == 0
    @test hit.expert_cache.pinned_bytes_uploaded == 0
    @test hit.expert_cache.read_tasks == 0
end
