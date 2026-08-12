using JSON3
using LifeAI
using SHA: sha256
using Test

const QWEN3_MOE_REUSE_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_REUSE_TINY_FIXTURE = joinpath(
    QWEN3_MOE_REUSE_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE scattered reuse portable contract" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_REUSE_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
    )
    first = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    repeated = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test repeated.logits == first.logits
    @test repeated.expert_cache.pointer_table_builds == 0
    @test repeated.expert_cache.pointer_table_reuses == 0
    @test repeated.expert_cache.workspace_allocations == 0
    @test repeated.expert_cache.workspace_reuses == 0
    @test repeated.expert_cache.pointer_table_bytes == 0
    @test repeated.expert_cache.workspace_bytes == 0

    clear_hf_qwen3_moe_expert_cache!(session)
    cleared = qwen3_moe_expert_cache_stats(session)
    @test cleared.entries == 0
    @test cleared.pointer_table_bytes == 0
    @test cleared.workspace_bytes == 0
end

@testset "Qwen3 MoE real scattered reuse result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary = JSON3.read(read(joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_scattered_reuse",
        "summary.json",
    ), String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test String(summary["session"]["dispatch"]) == "scattered"
    @test Int(summary["session"]["context_tokens"]) == 40_960
    @test Int(summary["session"]["cache_budget_bytes"]) == 8 * 2^30

    reuse = summary["reuse_state"]
    @test Int(reuse["pointer_plans_per_layer_limit"]) == 4
    @test Int(reuse["workspace_shapes_limit"]) == 4
    @test String(reuse["pointer_plan_signature"]) ==
        "ordered expert ids plus monotonic entry generations"
    @test Bool(reuse["cache_clear_invalidates_state"])
    @test Int(reuse["pointer_table_bytes"]) < 30_000
    @test Int(reuse["workspace_bytes"]) < 300_000
    @test !Bool(reuse["retains_expert_tensor_owners"])

    configurations = summary["configurations"]
    gc8 = configurations["scattered_reuse_gc8"]
    gc0 = configurations["scattered_reuse_gc0"]
    @test Int(gc8["gc_interval_layers"]) == 8
    @test Int(gc8["fill_pointer_table_builds"]) == 96
    @test Int(gc8["fill_workspace_allocations"]) == 2
    @test Int(gc8["fill_workspace_reuses"]) == 94
    @test Int(gc8["hit_pointer_bytes_uploaded"]) == 0
    @test Int(gc8["hit_pointer_table_builds"]) == 0
    @test Int(gc8["hit_pointer_table_reuses"]) == 96
    @test Int(gc8["hit_workspace_allocations"]) == 0
    @test Int(gc8["hit_workspace_reuses"]) == 96
    @test Int(gc8["hit_forced_gc_calls"]) == 12
    @test Int(gc8["repeat_count"]) == 100
    @test Bool(gc8["repeat_exact"])
    @test Bool(gc8["repeat_zero_io"])
    @test Bool(gc8["repeat_pointer_reuse_only"])
    @test Bool(gc8["repeat_workspace_reuse_only"])
    @test Int(gc8["allocator_free_drift_bytes"]) > 0
    @test Float64(gc8["repeat_request_seconds_p95"]) < 0.21
    @test Bool(gc8["fill_vs_hit_exact"])

    @test Int(gc0["gc_interval_layers"]) == 0
    @test Int(gc0["repeat_count"]) == 500
    @test Bool(gc0["repeat_exact"])
    @test Bool(gc0["repeat_zero_io"])
    @test Bool(gc0["repeat_pointer_reuse_only"])
    @test Bool(gc0["repeat_workspace_reuse_only"])
    @test Int(gc0["allocator_free_drift_bytes"]) < -800_000_000
    @test Int(gc0["live_free_drift_after_reclaim_bytes"]) > 0
    @test Float64(gc0["repeat_request_seconds_maximum"]) > 0.75

    comparisons = summary["comparisons"]
    previous = comparisons["chapter29_hit_over_chapter28_hit"]
    @test Int(previous["pointer_bytes_uploaded_reduction"]) == 26_832
    @test Int(previous["pointer_table_build_reduction"]) == 96
    @test Int(previous["workspace_allocation_reduction"]) == 96
    @test !Bool(previous["latency_speedup_claimed"])
    @test Float64(comparisons["gc0_over_gc8"][
        "repeat_median_speedup"
    ]) > 1.2

    correctness = summary["correctness"]
    @test Bool(correctness["all_fill_vs_hit_exact"])
    @test Bool(correctness["cross_configuration_exact"])
    @test Bool(correctness["prefill_argmax_match"])
    @test Bool(correctness["decode_argmax_match"])

    decision = summary["decision"]
    @test String(decision["recommended_frozen_workload_dispatch"]) ==
        "scattered"
    @test Int(decision[
        "recommended_frozen_workload_gc_interval_layers"
    ]) == 8
    @test !Bool(decision["gc0_live_leak_observed"])
    @test !Bool(decision["gc0_long_running_recommended"])
    @test Bool(decision["pointer_and_workspace_reuse_enabled_implicitly"])
    @test !Bool(decision["arbitrary_long_workload_stability_guaranteed"])
    @test !Bool(decision["route_attention_workspace_reuse_implemented"])
    @test !Bool(decision["pinned_memory_or_async_prefetch_implemented"])

    for (name, relative_path) in (
        "benchmark_script" => joinpath(
            "scripts",
            "benchmark_qwen3_moe_cuda_scattered_reuse.jl",
        ),
        "offload_implementation" => joinpath(
            "src",
            "generation",
            "qwen3_moe_offload.jl",
        ),
        "cuda_extension" => joinpath("ext", "LifeAICUDAExt.jl"),
    )
        expected = String(summary["source_sha256"][name])
        actual = bytes2hex(sha256(read(joinpath(repo_root, relative_path))))
        @test actual == expected
    end
end
