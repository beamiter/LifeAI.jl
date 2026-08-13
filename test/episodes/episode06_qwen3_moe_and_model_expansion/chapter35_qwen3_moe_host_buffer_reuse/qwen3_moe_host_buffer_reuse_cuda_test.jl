using CUDA
using LifeAI
using Test

CUDA.functional() || error(
    "Qwen3 MoE host-buffer CUDA test requested but CUDA is not functional",
)
CUDA.allowscalar(false)

const QWEN3_MOE_HOST_BUFFER_CUDA_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_HOST_BUFFER_CUDA_FIXTURE = joinpath(
    QWEN3_MOE_HOST_BUFFER_CUDA_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE CUDA pageable host-buffer ownership" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_HOST_BUFFER_CUDA_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
        expert_read_buffer_reuse=true,
        expert_host_buffer_reuse=true,
        expert_miss_pipeline=:overlapped,
        expert_read_workers=2,
        to_device=CUDA.cu,
    )
    pooled = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    pooled_stats = pooled.expert_cache
    @test pooled_stats.host_buffer_reuse
    @test pooled_stats.host_buffer_count == 2
    @test pooled_stats.host_buffer_bytes == 2 * 144
    @test pooled_stats.host_buffer_borrows == 8
    @test pooled_stats.host_buffer_returns == 8
    @test pooled_stats.host_buffer_borrows == pooled_stats.misses

    configure_hf_qwen3_moe_expert_cache!(
        session;
        host_buffer_reuse=false,
    )
    unpooled = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    unpooled_stats = unpooled.expert_cache
    @test Array(unpooled.logits) == Array(pooled.logits)
    @test !unpooled_stats.host_buffer_reuse
    @test unpooled_stats.host_buffer_count == 0
    @test unpooled_stats.host_buffer_bytes == 0
    @test unpooled_stats.host_buffer_borrows == 0
    @test unpooled_stats.host_buffer_returns == 0

    configure_hf_qwen3_moe_expert_cache!(
        session;
        host_buffer_reuse=true,
        miss_pipeline=:sequential,
    )
    sequential = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    sequential_stats = sequential.expert_cache
    @test Array(sequential.logits) == Array(pooled.logits)
    @test sequential_stats.host_buffer_count == 1
    @test sequential_stats.host_buffer_bytes == 144
    @test sequential_stats.host_buffer_borrows == 8
    @test sequential_stats.host_buffer_returns == 8

    configure_hf_qwen3_moe_expert_cache!(
        session;
        miss_pipeline=:overlapped,
        read_workers=2,
        pinned_upload=true,
    )
    pinned = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    pinned_stats = pinned.expert_cache
    @test Array(pinned.logits) == Array(pooled.logits)
    @test pinned_stats.host_buffer_reuse
    @test pinned_stats.host_buffer_count == 0
    @test pinned_stats.host_buffer_bytes == 0
    @test pinned_stats.host_buffer_borrows == 0
    @test pinned_stats.host_buffer_returns == 0
    @test pinned_stats.pinned_bytes_uploaded == pinned_stats.bytes_uploaded

    configure_hf_qwen3_moe_expert_cache!(
        session;
        pinned_upload=false,
        host_buffer_reuse=true,
    )
    original_to_device = session.to_device
    session.to_device = value ->
        value isa NamedTuple && haskey(value, :gate_proj) ?
            error("injected pageable upload failure") :
            original_to_device(value)
    @test_throws ErrorException prefill_hf_qwen3_moe_offload!(
        session,
        [2, 3],
    )
    failed_stats = qwen3_moe_expert_cache_stats(session)
    @test failed_stats.host_buffer_borrows > 0
    @test failed_stats.host_buffer_borrows == failed_stats.host_buffer_returns
    session.to_device = original_to_device
    configure_hf_qwen3_moe_expert_cache!(session)
    recovered = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    CUDA.synchronize()
    @test Array(recovered.logits) == Array(pooled.logits)
    @test recovered.expert_cache.host_buffer_borrows ==
        recovered.expert_cache.host_buffer_returns == 8
end
