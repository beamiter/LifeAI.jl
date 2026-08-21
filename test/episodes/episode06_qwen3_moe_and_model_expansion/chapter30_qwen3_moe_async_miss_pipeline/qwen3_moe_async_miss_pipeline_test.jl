using JSON3
using LifeAI
using Test

isdefined(@__MODULE__, :qwen3_moe_historical_benchmark_source_status) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

const QWEN3_MOE_ASYNC_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_ASYNC_TINY_FIXTURE = joinpath(
    QWEN3_MOE_ASYNC_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE bounded overlapped miss pipeline" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_ASYNC_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
    )
    sequential = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    sequential_keys = sort!(collect(keys(session.expert_cache)))
    @test sequential.expert_cache.miss_pipeline == :sequential
    @test sequential.expert_cache.read_tasks == 8
    @test sequential.expert_cache.pinned_bytes_uploaded == 0

    configured = configure_hf_qwen3_moe_expert_cache!(
        session;
        miss_pipeline=:overlapped,
        read_workers=2,
    )
    overlapped = prefill_hf_qwen3_moe_offload!(configured, [2, 3])
    @test overlapped.logits == sequential.logits
    @test sort!(collect(keys(session.expert_cache))) == sequential_keys
    @test overlapped.expert_cache.miss_pipeline == :overlapped
    @test overlapped.expert_cache.read_workers == 2
    @test !overlapped.expert_cache.pinned_upload
    @test overlapped.expert_cache.misses == 8
    @test overlapped.expert_cache.read_tasks == 8
    @test overlapped.expert_cache.host_read_seconds >= 0
    @test overlapped.expert_cache.miss_stage_seconds >= 0
    @test overlapped.expert_cache.upload_wait_seconds == 0
    @test overlapped.expert_cache.pinned_bytes_uploaded == 0
    if Threads.nthreads() > 1
        @test overlapped.expert_cache.parallel_read_layers == 2
    else
        @test overlapped.expert_cache.parallel_read_layers == 0
    end

    hit = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test hit.logits == sequential.logits
    @test hit.expert_cache.hits == 8
    @test hit.expert_cache.misses == 0
    @test hit.expert_cache.read_tasks == 0
    @test hit.expert_cache.miss_stage_seconds == 0
end

@testset "Qwen3 MoE real async miss result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_async_miss_pipeline",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test Int(summary["session"]["julia_threads"]) == 4
    @test Int(summary["session"]["read_workers"]) == 4
    @test Int(summary["session"]["cache_budget_bytes"]) == 8 * 2^30
    @test String(summary["session"]["dispatch"]) == "scattered"

    configurations = summary["configurations"]
    sequential = configurations[1]
    pageable = configurations[2]
    pinned = configurations[3]
    @test String(sequential["name"]) == "sequential_pageable"
    @test String(pageable["name"]) == "overlapped_pageable"
    @test String(pinned["name"]) == "overlapped_pinned"
    @test Float64(sequential["request_seconds_median"]) > 10
    @test Float64(pageable["request_seconds_median"]) < 5.5
    @test Float64(pinned["request_seconds_median"]) < 5.6
    @test all(configuration -> Bool(configuration["all_runs_exact"]), configurations)
    @test all(configuration -> Bool(configuration["warmup_exact"]), configurations)

    for run in sequential["runs"]
        @test Int(run["bytes_read"]) == 8_417_968_128
        @test Int(run["bytes_uploaded"]) == 8_417_968_128
        @test Int(run["read_tasks"]) == 892
        @test Int(run["parallel_read_layers"]) == 0
        @test Int(run["pinned_bytes_uploaded"]) == 0
    end
    for run in pageable["runs"]
        @test Int(run["parallel_read_layers"]) == 95
        @test Int(run["pinned_bytes_uploaded"]) == 0
    end
    for run in pinned["runs"]
        @test Int(run["parallel_read_layers"]) == 95
        @test Int(run["pinned_bytes_uploaded"]) ==
            Int(run["bytes_uploaded"])
        @test Float64(run["upload_wait_seconds"]) > 0
    end

    comparisons = summary["comparisons"]
    @test Float64(comparisons[
        "overlapped_pageable_over_sequential"
    ]) > 1.9
    @test Float64(comparisons[
        "overlapped_pinned_over_sequential"
    ]) > 1.9
    @test Float64(comparisons[
        "pinned_over_pageable_overlap"
    ]) < 1

    correctness = summary["correctness"]
    @test Bool(correctness["all_configurations_exact"])
    @test Bool(correctness["prefill_argmax_match"])
    @test Bool(correctness["decode_argmax_match"])
    decision = summary["decision"]
    @test String(decision["fastest_configuration"]) ==
        "overlapped_pageable"
    @test !Bool(decision["enable_by_default"])
    @test Bool(decision["bounded_current_layer_staging"])
    @test !Bool(decision["layer_ahead_prefetch_implemented"])
    @test !Bool(decision["cold_disk_speedup_claimed"])

    for (name, relative_path) in pairs(summary["source_paths"])
        status = qwen3_moe_historical_benchmark_source_status(
            summary_path,
            summary,
            name,
            String(relative_path),
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

@testset "Qwen3 MoE miss pipeline validation is fail closed" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_ASYNC_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
    )
    prefill_hf_qwen3_moe_offload!(session, [2, 3])
    entries = qwen3_moe_expert_cache_stats(session).entries
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        session;
        miss_pipeline=:speculative,
    )
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        session;
        read_workers=0,
    )
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        session;
        pinned_upload=true,
    )
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes=0,
        dispatch=:materialized,
        miss_pipeline=:overlapped,
    )
    @test qwen3_moe_expert_cache_stats(session).entries == entries

    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_ASYNC_TINY_FIXTURE;
        expert_cache_budget_bytes=8 * 144,
        expert_miss_pipeline=:unknown,
    )
    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_ASYNC_TINY_FIXTURE;
        expert_cache_budget_bytes=8 * 144,
        expert_read_workers=0,
    )
end
