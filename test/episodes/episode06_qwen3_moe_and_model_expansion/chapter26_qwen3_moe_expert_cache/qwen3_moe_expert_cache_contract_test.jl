using JSON3
using LifeAI
using SHA: sha256
using Test

const QWEN3_MOE_EXPERT_CACHE_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_EXPERT_CACHE_TINY_FIXTURE = joinpath(
    QWEN3_MOE_EXPERT_CACHE_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE device expert LRU cache" begin
    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_EXPERT_CACHE_TINY_FIXTURE;
        context_tokens=8,
        expert_cache_budget_bytes=-1,
    )

    # Each tiny expert-layer entry owns 3 * 8 * 3 BF16 values = 144 bytes.
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_EXPERT_CACHE_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=1,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
    )
    first_prefill = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test first_prefill.expert_bytes_read == 8 * 144
    @test first_prefill.expert_bytes_uploaded == 8 * 144
    @test sum(chunk.expert_bytes_uploaded for chunk in first_prefill.chunks) ==
        first_prefill.expert_bytes_uploaded
    @test first_prefill.expert_cache.misses == 8
    @test first_prefill.expert_cache.hits == 0
    @test first_prefill.expert_cache.entries == 8
    @test first_prefill.expert_cache.current_bytes == 8 * 144

    second_prefill = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test second_prefill.logits == first_prefill.logits
    @test second_prefill.expert_bytes_read == 0
    @test second_prefill.expert_bytes_uploaded == 0
    @test second_prefill.expert_cache.hits == 8
    @test second_prefill.expert_cache.misses == 0
    @test reset_hf_qwen3_moe_offload_session!(session).expert_cache_bytes ==
        8 * 144

    prefill_hf_qwen3_moe_offload!(session, [2, 3])
    decode = decode_hf_qwen3_moe_offload!(session, 4)
    @test decode.expert_bytes_read == 0
    @test decode.expert_bytes_uploaded == 0
    @test decode.expert_cache_hits == 4
    @test decode.expert_cache_misses == 0

    clear_hf_qwen3_moe_expert_cache!(session)
    cleared = qwen3_moe_expert_cache_stats(session)
    @test cleared.entries == 0
    @test cleared.current_bytes == 0
    @test cleared.peak_bytes == 0

    limited = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_EXPERT_CACHE_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=2 * 144,
    )
    limited_prefill = prefill_hf_qwen3_moe_offload!(limited, [2, 3])
    @test limited_prefill.expert_cache.current_bytes <= 2 * 144
    @test limited_prefill.expert_cache.peak_bytes <= 2 * 144
    @test limited_prefill.expert_cache.evictions > 0
end

@testset "Qwen3 MoE real expert cache result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_expert_cache",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test Int(summary["session"]["context_tokens"]) == 40_960
    @test Int(summary["cache"]["budget_bytes"]) == 8 * 2^30
    @test Int(summary["cache"]["entry_bytes"]) == 9_437_184
    @test Int(summary["cache"]["entries"]) == 892
    @test Int(summary["cache"]["current_bytes"]) == 8_417_968_128
    @test Int(summary["cache"]["evictions"]) == 0
    @test Int(summary["gpu"]["free_final_bytes"]) > 4 * 2^30

    fill = summary["warm_fill"]
    hit = summary["warm_hit"]
    @test Int(fill["prefill_cache_misses"]) == 734
    @test Int(fill["decode_cache_misses"]) == 158
    @test Int(fill["decode_cache_hits"]) == 226
    @test Int(fill["prefill_expert_bytes_read"]) +
        Int(fill["decode_expert_bytes_read"]) ==
        Int(summary["cache"]["current_bytes"])
    @test Int(hit["prefill_expert_bytes_read"]) == 0
    @test Int(hit["prefill_expert_bytes_uploaded"]) == 0
    @test Int(hit["decode_expert_bytes_read"]) == 0
    @test Int(hit["decode_expert_bytes_uploaded"]) == 0
    @test Int(hit["prefill_cache_hits"]) == 734
    @test Int(hit["decode_cache_hits"]) == 384

    speedup = summary["warm_hit_over_fill_speedup"]
    @test Float64(speedup["prefill"]) > 1.7
    @test Float64(speedup["decode"]) > 1.5
    @test Float64(speedup["request"]) > 1.7
    parity = summary["parity"]
    @test Bool(parity["prefill_argmax_match"])
    @test Bool(parity["decode_argmax_match"])
    @test Bool(parity["cold_vs_first_hit_exact"])
    @test Bool(parity["warm_fill_vs_hit_exact"])
    @test !Bool(summary["decision"]["arbitrary_request_zero_eviction_guaranteed"])
    @test !Bool(summary["decision"]["pinned_memory_or_async_prefetch_implemented"])

    for (name, relative_path) in (
        "benchmark_script" => joinpath(
            "scripts",
            "benchmark_qwen3_moe_cuda_expert_cache.jl",
        ),
        "offload_implementation" => joinpath(
            "src",
            "generation",
            "qwen3_moe_offload.jl",
        ),
    )
        expected = String(summary["source_sha256"][name])
        actual = bytes2hex(sha256(read(joinpath(repo_root, relative_path))))
        @test actual == expected
    end
end
