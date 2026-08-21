using BFloat16s: BFloat16
using JSON3
using LifeAI
using SHA: sha256
using Test

isdefined(@__MODULE__, :qwen3_moe_historical_benchmark_source_status) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

function _chapter33_bf16_bytes(values)
    bits = reinterpret(UInt16, BFloat16.(values))
    return collect(reinterpret(UInt8, bits))
end

@testset "Multidimensional safetensors decode elides only the intermediate copy" begin
    raw = _chapter33_bf16_bytes(1:6)
    matrix = LifeAI._decode_safetensors_values(
        raw,
        "BF16",
        [2, 3];
        target_dtype=BFloat16,
    )
    @test matrix == BFloat16[1 2 3; 4 5 6]
    @test matrix isa Matrix{BFloat16}
    fill!(raw, 0x00)
    @test matrix == BFloat16[1 2 3; 4 5 6]

    vector_raw = _chapter33_bf16_bytes(1:6)
    vector = LifeAI._decode_safetensors_values(
        vector_raw,
        "BF16",
        [6];
        target_dtype=BFloat16,
    )
    fill!(vector_raw, 0x00)
    @test vector == BFloat16.(1:6)
    @test vector isa Vector{BFloat16}

    scalar_raw = _chapter33_bf16_bytes([7])
    scalar = LifeAI._decode_safetensors_values(
        scalar_raw,
        "BF16",
        Int[];
        target_dtype=BFloat16,
    )
    fill!(scalar_raw, 0x00)
    @test scalar[] == BFloat16(7)
    @test size(scalar) == ()

    float_values = Float32.(1:6)
    float_raw = collect(reinterpret(UInt8, float_values))
    float_matrix = LifeAI._decode_safetensors_values(
        float_raw,
        "F32",
        [2, 3];
        target_dtype=Float32,
    )
    fill!(float_raw, 0x00)
    @test float_matrix == Float32[1 2 3; 4 5 6]
    @test float_matrix isa Matrix{Float32}

    allocation_raw = _chapter33_bf16_bytes(1:(256 * 256))
    LifeAI._decode_safetensors_values(
        allocation_raw,
        "BF16",
        [256, 256];
        target_dtype=BFloat16,
    )
    allocated = @allocated LifeAI._decode_safetensors_values(
        allocation_raw,
        "BF16",
        [256, 256];
        target_dtype=BFloat16,
    )
    @test allocated >= length(allocation_raw)
    @test allocated < 3 * length(allocation_raw) ÷ 2
end

@testset "Qwen3 MoE real decode-copy-elision result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_decode_copy_elision",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["revision"]) ==
        "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test String(summary["gpu"]["name"]) == "NVIDIA GeForce RTX 4090 D"

    session = summary["session"]
    @test Int(session["context_tokens"]) == 40_960
    @test Int(session["cache_budget_bytes"]) == 4 * 2^30
    @test String(session["cache_policy"]) == "layer_balanced_lru"
    @test String(session["dispatch"]) == "scattered"
    @test Int(session["gc_interval_layers"]) == 8
    @test String(session["read_mode"]) == "tensor"
    @test Int(session["read_workers"]) == 8
    @test Int(session["julia_threads"]) == 8

    workload = summary["workload"]
    @test String(workload["trace"]) ==
        "english_32 plus one greedy decode"
    @test Int(workload["tokens_per_prompt"]) == 32
    @test Int(workload["decode_tokens"]) == 1
    @test String(workload["tokens_sha256"]) ==
        "0f06ce74166c11acb516970ec217b6ef52c9592f2ecd88067189d26959f74927"

    methodology = summary["methodology"]
    @test Int(methodology["repetitions"]) == 3
    @test occursin(
        "POSIX_FADV_DONTNEED",
        String(methodology["cold_control"]),
    )
    @test occursin("/proc/self/io", String(methodology["cold_verification"]))
    @test String(methodology["comparison"]) ==
        "adjacent frozen Chapter 32 tensor baseline"
    @test !Bool(methodology["same_process_ab"])

    baseline = summary["baseline"]
    @test String(baseline["source_report"]) ==
        "benchmark_results/qwen3_moe_cuda_coalesced_reads/summary.json"
    baseline_path = joinpath(repo_root, String(baseline["source_report"]))
    @test bytes2hex(sha256(read(baseline_path))) ==
        String(baseline["source_report_sha256"])
    @test String(baseline["implementation"]) ==
        "copied_multidimensional_decode"
    @test Int(baseline["cold"]["allocated_bytes_median"]) ==
        86_325_851_872
    @test Int(baseline["revisit"]["allocated_bytes_median"]) ==
        86_325_817_760

    direct = summary["direct"]
    @test Int(direct["cold"]["repetitions"]) == 3
    @test Int(direct["revisit"]["repetitions"]) == 3
    @test Int(direct["cold"]["allocated_bytes_median"]) ==
        57_646_099_688
    @test Int(direct["revisit"]["allocated_bytes_median"]) ==
        57_646_086_296
    @test Float64(direct["cold"]["request_seconds_median"]) < 17
    @test Float64(direct["revisit"]["request_seconds_median"]) < 15
    for phase in ("cold", "revisit")
        for run in direct[phase]["runs"]
            @test String(run["implementation"]) ==
                "direct_multidimensional_decode"
            @test Int(run["cache_hits"]) == 71
            @test Int(run["cache_misses"]) == 3_039
            @test Int(run["cache_evictions"]) == 313
            @test Int(run["expert_bytes_read"]) == 28_679_602_176
            @test Int(run["expert_bytes_uploaded"]) == 28_679_602_176
            @test Int(run["read_jobs"]) == 3_039
            if phase == "cold"
                @test 26_235_000_000 <=
                    Int(run["process_storage_read_bytes"]) <=
                    26_236_000_000
            else
                @test Int(run["process_storage_read_bytes"]) == 0
            end
        end
    end

    comparisons = summary["comparisons"]
    @test 0.332 < Float64(comparisons[
        "cold_allocation_reduction"
    ]) < 0.333
    @test 0.332 < Float64(comparisons[
        "revisit_allocation_reduction"
    ]) < 0.333
    @test Int(comparisons["logical_payload_bytes"]) == 28_679_602_176
    @test abs(
        Int(comparisons["cold_eliminated_allocation_bytes"]) -
        Int(comparisons["logical_payload_bytes"]),
    ) < 200_000
    @test abs(
        Int(comparisons["revisit_eliminated_allocation_bytes"]) -
        Int(comparisons["logical_payload_bytes"]),
    ) < 200_000
    @test Float64(comparisons["cold_speedup"]) > 1
    @test 0.95 < Float64(comparisons["revisit_speedup"]) < 1.05

    @test Bool(summary["correctness"]["all_outputs_exact"])
    decision = summary["decision"]
    @test Bool(decision["remove_multidimensional_intermediate_copy"])
    @test Bool(decision["vector_and_scalar_ownership_preserved"])
    @test !Bool(decision["final_semantic_matrix_reused"])
    @test !Bool(decision["arbitrary_workload_speedup_generalized"])

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
