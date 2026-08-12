using JSON3
using LifeAI
using SHA: sha256
using Test

const QWEN3_MOE_WORKER_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_WORKER_TINY_FIXTURE = joinpath(
    QWEN3_MOE_WORKER_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE read workers follow bounded Julia concurrency" begin
    expected_workers = min(8, Threads.nthreads())
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_WORKER_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_miss_pipeline=:overlapped,
        expert_gc_interval_layers=0,
    )
    result = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test result.expert_cache.read_workers == expected_workers
    @test 1 <= result.expert_cache.read_workers <= 8
    @test result.expert_cache.read_tasks == 8
    @test result.expert_cache.misses == 8
    @test result.expert_cache.pinned_bytes_uploaded == 0
    if Threads.nthreads() > 1
        @test result.expert_cache.parallel_read_layers == 2
    else
        @test result.expert_cache.parallel_read_layers == 0
    end
end

@testset "Qwen3 MoE real read-worker sweep contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary = JSON3.read(read(joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_read_worker_sweep",
        "summary.json",
    ), String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test Int(summary["session"]["julia_threads"]) == 8
    @test Int(summary["session"]["cache_budget_bytes"]) == 4 * 2^30
    @test String(summary["session"]["cache_policy"]) ==
        "layer_balanced_lru"
    @test String(summary["session"]["dispatch"]) == "scattered"

    workload = summary["workload"]
    @test String(workload["trace"]) ==
        "english_32 plus one greedy decode"
    @test Int(workload["tokens_per_prompt"]) == 32
    @test String(workload["tokens_sha256"]) ==
        "0f06ce74166c11acb516970ec217b6ef52c9592f2ecd88067189d26959f74927"

    page_cache = summary["page_cache_control"]
    @test occursin("POSIX_FADV_DONTNEED", String(page_cache["mechanism"]))
    @test Int(page_cache["checkpoint_shards"]) == 16
    @test Int(page_cache["checkpoint_bytes"]) == 61_066_575_648
    @test Int(page_cache["advised_bytes"]) == 61_066_575_648
    @test !Bool(page_cache["cold_guaranteed"])

    configurations = summary["configurations"]
    @test Int.(getindex.(configurations, "workers")) == [1, 2, 4, 8]
    cold_seconds = Float64[
        configuration["cold"]["request_seconds_median"]
        for configuration in configurations
    ]
    revisit_seconds = Float64[
        configuration["revisit"]["request_seconds_median"]
        for configuration in configurations
    ]
    @test issorted(cold_seconds; rev=true)
    @test issorted(revisit_seconds; rev=true)
    @test cold_seconds[1] > 48
    @test cold_seconds[4] < 17
    @test revisit_seconds[1] > 37
    @test revisit_seconds[4] < 15

    for configuration in configurations
        @test Int(configuration["cold"]["repetitions"]) == 2
        @test Int(configuration["revisit"]["repetitions"]) == 2
        for run in configuration["cold"]["runs"]
            @test Int(run["cache_hits"]) == 71
            @test Int(run["cache_misses"]) == 3_039
            @test Int(run["expert_bytes_read"]) == 28_679_602_176
            @test Int(run["expert_bytes_uploaded"]) == 28_679_602_176
            @test Int(run["process_storage_read_bytes"]) in
                (26_235_793_408, 26_235_797_504)
        end
        for run in configuration["revisit"]["runs"]
            @test Int(run["cache_hits"]) == 71
            @test Int(run["cache_misses"]) == 3_039
            @test Int(run["process_storage_read_bytes"]) == 0
        end
    end

    comparisons = summary["comparisons"]
    @test Float64(comparisons[2]["cold_speedup_over_first"]) > 1.5
    @test Float64(comparisons[3]["cold_speedup_over_first"]) > 2.3
    @test Float64(comparisons[4]["cold_speedup_over_first"]) > 2.8
    @test Float64(comparisons[4]["revisit_speedup_over_first"]) > 2.5
    @test Bool(summary["correctness"]["all_outputs_exact"])

    decision = summary["decision"]
    @test Int(decision["fastest_cold_workers"]) == 8
    @test Int(decision["fastest_revisit_workers"]) == 8
    @test String(decision["default_read_workers"]) ==
        "min(8, Threads.nthreads())"
    @test Bool(decision["default_read_workers_changed"])
    @test !Bool(decision["arbitrary_storage_or_thread_count_generalized"])

    for (name, relative_path) in pairs(summary["source_paths"])
        expected = String(summary["source_sha256"][name])
        actual = bytes2hex(sha256(read(joinpath(
            repo_root,
            String(relative_path),
        ))))
        @test actual == expected
    end
end
