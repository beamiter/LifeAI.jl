using JSON3
using LifeAI
using SHA: sha256
using Test

const QWEN3_MOE_LAYER_CACHE_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_LAYER_CACHE_TINY_FIXTURE = joinpath(
    QWEN3_MOE_LAYER_CACHE_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE layer-balanced expert cache" begin
    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_LAYER_CACHE_TINY_FIXTURE;
        context_tokens=8,
        expert_cache_budget_bytes=4 * 144,
        expert_cache_policy=:not_a_policy,
    )

    # The fixture has two layers and four routed experts per layer for this
    # two-token prompt. A four-entry global LRU is thrashed by each layer scan.
    global_session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_LAYER_CACHE_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=4 * 144,
        expert_cache_policy=:global_lru,
    )
    global_fill = prefill_hf_qwen3_moe_offload!(global_session, [2, 3])
    @test global_fill.expert_cache.policy == :global_lru
    @test global_fill.expert_cache.hits == 0
    @test global_fill.expert_cache.misses == 8
    @test global_fill.expert_cache.entries == 4
    @test global_fill.expert_cache.evictions == 4

    global_repeat = prefill_hf_qwen3_moe_offload!(global_session, [2, 3])
    @test global_repeat.logits == global_fill.logits
    @test global_repeat.expert_cache.hits == 0
    @test global_repeat.expert_cache.misses == 8
    @test global_repeat.expert_cache.evictions == 8

    configured = configure_hf_qwen3_moe_expert_cache!(
        global_session;
        budget_bytes=4 * 144,
        policy=:layer_balanced_lru,
    )
    configured_stats = qwen3_moe_expert_cache_stats(configured)
    @test configured_stats.policy == :layer_balanced_lru
    @test configured_stats.entries == 0
    @test configured_stats.current_bytes == 0
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        configured;
        policy=:not_a_policy,
    )
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        configured;
        budget_bytes=-1,
    )

    balanced_fill = prefill_hf_qwen3_moe_offload!(configured, [2, 3])
    @test balanced_fill.logits == global_fill.logits
    @test balanced_fill.expert_cache.policy == :layer_balanced_lru
    @test balanced_fill.expert_cache.hits == 0
    @test balanced_fill.expert_cache.misses == 8
    @test balanced_fill.expert_cache.entries == 4
    @test balanced_fill.expert_cache.current_bytes == 4 * 144
    @test balanced_fill.expert_cache.evictions == 0

    balanced_repeat = prefill_hf_qwen3_moe_offload!(configured, [2, 3])
    @test balanced_repeat.logits == balanced_fill.logits
    @test balanced_repeat.expert_cache.hits == 4
    @test balanced_repeat.expert_cache.misses == 4
    @test balanced_repeat.expert_bytes_read == 4 * 144
    @test balanced_repeat.expert_bytes_uploaded == 4 * 144
    @test balanced_repeat.expert_cache.evictions == 0
    @test balanced_repeat.expert_cache.current_bytes <=
        balanced_repeat.expert_cache.budget_bytes
end

@testset "Qwen3 MoE real layer-balanced cache result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary = JSON3.read(read(joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_layer_balanced_cache",
        "summary.json",
    ), String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test Int(summary["session"]["context_tokens"]) == 40_960
    @test String(summary["workload"]["trace"]) ==
        "english_32 -> chinese_32 -> english_32"
    @test Int(summary["workload"]["tokens_per_prompt"]) == 32

    configurations = summary["configurations"]
    global8 = configurations["global_8gib"]
    balanced4 = configurations["balanced_4gib"]
    balanced6 = configurations["balanced_6gib"]
    balanced8 = configurations["balanced_8gib"]
    @test String(global8["policy"]) == "global_lru"
    @test all(
        String(configuration["policy"]) == "layer_balanced_lru"
        for configuration in (balanced4, balanced6, balanced8)
    )
    @test Int(global8["budget_bytes"]) == 8 * 2^30
    @test Int(balanced4["budget_bytes"]) == 4 * 2^30
    @test Int(balanced6["budget_bytes"]) == 6 * 2^30
    @test Int(balanced8["budget_bytes"]) == 8 * 2^30
    @test (
        Int(balanced4["layer_slot_capacity"]),
        Int(balanced6["layer_slot_capacity"]),
        Int(balanced8["layer_slot_capacity"]),
    ) == (9, 14, 18)

    @test Float64(global8["trace_hit_rate"]) < 0.05
    @test Float64(balanced4["trace_hit_rate"]) > 0.12
    @test Float64(balanced6["trace_hit_rate"]) >
        Float64(balanced4["trace_hit_rate"])
    @test Float64(balanced8["trace_hit_rate"]) >
        Float64(balanced6["trace_hit_rate"])
    @test Float64(balanced8["english_revisit_hit_rate"]) > 0.34
    @test Int(balanced4["trace_expert_bytes_read"]) <
        Int(global8["trace_expert_bytes_read"])
    @test Int(balanced6["trace_expert_bytes_read"]) <
        Int(balanced4["trace_expert_bytes_read"])
    @test Int(balanced8["trace_expert_bytes_read"]) <
        Int(balanced6["trace_expert_bytes_read"])
    @test all(
        Bool(configuration["english_revisit_exact"])
        for configuration in (global8, balanced4, balanced6, balanced8)
    )
    @test Int(balanced4["gpu_free_final_bytes"]) > 6 * 2^30
    @test Int(balanced6["gpu_free_final_bytes"]) > 3 * 2^30
    @test Int(balanced8["gpu_free_final_bytes"]) < 2 * 2^30

    comparisons = summary["comparisons"]
    balanced_over_global = comparisons["balanced_8gib_over_global_8gib"]
    @test Float64(balanced_over_global["hit_rate_delta"]) > 0.17
    @test Float64(balanced_over_global["bytes_read_reduction"]) > 0.18
    @test Float64(balanced_over_global["speedup"]) > 1.08
    @test Float64(comparisons[
        "balanced_8gib_over_balanced_4gib"
    ]["speedup"]) < 1.02

    decision = summary["decision"]
    @test String(decision["recommended_frozen_trace_policy"]) ==
        "layer_balanced_lru"
    @test Int(decision["recommended_frozen_trace_budget_bytes"]) == 4 * 2^30
    @test !Bool(decision["arbitrary_workload_budget_guaranteed"])
    @test !Bool(decision["device_concat_or_gc_optimized"])
    @test !Bool(decision["pinned_memory_or_async_prefetch_implemented"])

    for (name, relative_path) in (
        "benchmark_script" => joinpath(
            "scripts",
            "benchmark_qwen3_moe_cuda_layer_balanced_cache.jl",
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
