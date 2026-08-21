using JSON3
using LifeAI
using Test

isdefined(@__MODULE__, :qwen3_moe_historical_benchmark_source_status) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

const QWEN3_MOE_SCATTERED_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_SCATTERED_TINY_FIXTURE = joinpath(
    QWEN3_MOE_SCATTERED_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE scattered cache contract" begin
    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_SCATTERED_TINY_FIXTURE;
        expert_cache_dispatch=:invalid,
    )
    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_SCATTERED_TINY_FIXTURE;
        expert_gc_interval_layers=-1,
    )
    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_SCATTERED_TINY_FIXTURE;
        expert_cache_dispatch=:scattered,
        expert_cache_budget_bytes=0,
        grouped_experts=false,
    )
    grouped_scattered_fallback = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_SCATTERED_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        expert_cache_dispatch=:scattered,
        expert_cache_budget_bytes=8 * 144,
        grouped_experts=true,
    )
    @test grouped_scattered_fallback.grouped_experts
    @test grouped_scattered_fallback.expert_cache_dispatch === :scattered

    materialized = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_SCATTERED_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:materialized,
        expert_gc_interval_layers=0,
    )
    materialized_first = prefill_hf_qwen3_moe_offload!(materialized, [2, 3])
    materialized_hit = prefill_hf_qwen3_moe_offload!(materialized, [2, 3])
    @test materialized_hit.logits == materialized_first.logits
    @test materialized_hit.expert_cache.dispatch == :materialized
    @test materialized_hit.expert_cache.active_materializations == 2
    @test materialized_hit.expert_cache.active_materialization_bytes == 8 * 144
    @test materialized_hit.expert_cache.scattered_dispatches == 0
    @test materialized_hit.expert_cache.pointer_bytes_uploaded == 0
    @test materialized_hit.expert_cache.forced_gc_calls == 0

    # The portable fallback accepts :scattered but materializes before the
    # generic dispatch. CUDA supplies the direct pointer-table specialization.
    scattered_fallback = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_SCATTERED_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
    )
    fallback = prefill_hf_qwen3_moe_offload!(scattered_fallback, [2, 3])
    @test fallback.logits == materialized_first.logits
    @test fallback.expert_cache.dispatch == :scattered
    @test fallback.expert_cache.active_materializations == 2
    @test fallback.expert_cache.scattered_dispatches == 0
    @test fallback.expert_cache.pointer_bytes_uploaded == 0
    @test fallback.expert_cache.forced_gc_calls == 0

    # Non-CUDA devices keep the materialized portable fallback even when the
    # session requests grouped scattered dispatch. CUDA supplies the direct
    # pointer-backed WMMA specialization.
    grouped_fallback = prefill_hf_qwen3_moe_offload!(
        grouped_scattered_fallback,
        [2, 3],
    )
    @test grouped_fallback.logits == materialized_first.logits
    @test grouped_fallback.expert_cache.active_materializations == 2
    @test grouped_fallback.expert_cache.scattered_dispatches == 0

    configured = configure_hf_qwen3_moe_expert_cache!(
        scattered_fallback;
        budget_bytes=8 * 144,
        dispatch=:materialized,
        grouped_experts=true,
        gc_interval_layers=1,
    )
    @test qwen3_moe_expert_cache_stats(configured).entries == 0
    @test configured.grouped_experts
    @test qwen3_moe_expert_cache_stats(configured).grouped_experts
    configured_prefill = prefill_hf_qwen3_moe_offload!(configured, [2, 3])
    @test configured_prefill.logits == materialized_first.logits
    @test configured_prefill.expert_cache.dispatch == :materialized
    @test configured_prefill.expert_cache.gc_interval_layers == 1
    @test configured_prefill.expert_cache.forced_gc_calls == 2
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        configured;
        dispatch=:invalid,
    )
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        configured;
        gc_interval_layers=-1,
    )
end

@testset "Qwen3 MoE real scattered cache result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_scattered_cache",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test Int(summary["session"]["context_tokens"]) == 40_960
    @test Int(summary["session"]["prompt_tokens"]) == 2
    @test Int(summary["session"]["decode_tokens"]) == 1
    @test Int(summary["session"]["cache_budget_bytes"]) == 8 * 2^30

    configurations = summary["configurations"]
    materialized = configurations["materialized_gc1"]
    scattered1 = configurations["scattered_gc1"]
    scattered8 = configurations["scattered_gc8"]
    scattered0 = configurations["scattered_gc0"]
    @test String(materialized["dispatch"]) == "materialized"
    @test String(scattered8["dispatch"]) == "scattered"
    @test Int(materialized["gc_interval_layers"]) == 1
    @test Int(scattered8["gc_interval_layers"]) == 8
    @test Int(scattered0["gc_interval_layers"]) == 0
    @test Int(materialized["hit_active_materializations"]) == 96
    @test Int(materialized["hit_active_materialization_bytes"]) > 10_000_000_000
    @test Int(scattered8["hit_active_materializations"]) == 0
    @test Int(scattered8["hit_active_materialization_bytes"]) == 0
    @test Int(scattered8["hit_scattered_dispatches"]) == 96
    @test Int(scattered8["hit_pointer_bytes_uploaded"]) < 30_000
    @test Int(scattered8["hit_expert_bytes_read"]) == 0
    @test Int(scattered8["hit_expert_bytes_uploaded"]) == 0
    @test Int(scattered8["hit_cache_hits"]) == 1_118
    @test Int(scattered8["hit_cache_misses"]) == 0
    @test Int(scattered8["hit_forced_gc_calls"]) == 12
    @test Int(scattered1["hit_forced_gc_calls"]) == 96
    @test Float64(scattered8["hit_request_seconds"]) < 0.2
    @test Bool(scattered8["fill_vs_hit_exact"])

    comparisons = summary["comparisons"]
    @test Float64(comparisons[
        "scattered_gc8_over_materialized"
    ]["hit_request_speedup"]) > 80
    @test Int(comparisons[
        "scattered_gc1_over_materialized"
    ]["materialization_bytes_eliminated"]) > 10_000_000_000
    @test Float64(comparisons[
        "scattered_gc8_over_scattered_gc1"
    ]["hit_request_speedup"]) > 1.5

    @test Int(scattered0["repeat_count"]) == 100
    @test Bool(scattered0["repeat_exact"])
    @test Bool(scattered0["repeat_zero_io"])
    @test Int(scattered0["repeat_allocator_free_drift_bytes"]) < -900_000_000
    @test Float64(scattered0["repeat_request_seconds_maximum"]) > 0.7

    correctness = summary["correctness"]
    @test Bool(correctness["all_fill_vs_hit_exact"])
    @test Bool(correctness["all_cross_configuration_exact"])
    @test Bool(correctness["prefill_argmax_match"])
    @test Bool(correctness["decode_argmax_match"])

    decision = summary["decision"]
    @test String(decision["default_dispatch"]) == "materialized"
    @test String(decision["recommended_frozen_workload_dispatch"]) ==
        "scattered"
    @test Int(decision[
        "recommended_frozen_workload_gc_interval_layers"
    ]) == 8
    @test !Bool(decision["grouped_scattered_supported"])
    @test Bool(decision["positive_device_cache_required"])
    @test !Bool(decision["arbitrary_long_workload_stability_guaranteed"])
    @test !Bool(decision["pointer_table_reuse_implemented"])
    @test !Bool(decision["pinned_memory_or_async_prefetch_implemented"])

    for (name, relative_path) in (
        "benchmark_script" => joinpath(
            "scripts",
            "benchmark_qwen3_moe_cuda_scattered_cache.jl",
        ),
        "offload_implementation" => joinpath(
            "src",
            "generation",
            "qwen3_moe_offload.jl",
        ),
        "cuda_extension" => joinpath("ext", "LifeAICUDAExt.jl"),
    )
        status = qwen3_moe_historical_benchmark_source_status(
            summary_path,
            summary,
            name,
            relative_path,
        )
        @test status.source_path_matches
        @test status.digest_is_sha256
        @test status.current_source_exists
        @test status.report_registered
        @test status.snapshot_registered
        @test status.digest_matches_snapshot
        if status.git_snapshot_matches !== nothing
            @test status.git_snapshot_matches
        end
    end
end
