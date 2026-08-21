using JSON3
using SHA: sha256
using Test

@testset "Qwen3 MoE grouped scattered real-result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_grouped_scattered",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))

    @test Int(summary["schema_version"]) == 1
    @test String(summary["benchmark"]) ==
        "qwen3_moe_cuda_grouped_scattered_same_process"
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["revision"]) ==
        "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test String(summary["environment"]["gpu_name"]) ==
        "NVIDIA GeForce RTX 4090 D"
    @test Int(summary["environment"]["julia_threads"]) == 8
    @test Int(summary["session"]["context_tokens"]) == 40_960
    @test Int(summary["session"]["cache_budget_bytes"]) == 8 * 2^30
    @test String(summary["session"]["cache_policy"]) ==
        "layer_balanced_lru"
    @test Int(summary["session"]["gc_interval_layers"]) == 8

    verification = summary["verification"]
    @test Bool(verification[
        "materialized_grouped_vs_grouped_scattered_bitwise_exact"
    ])
    @test Bool(verification[
        "materialized_grouped_vs_grouped_scattered_measured_traffic_exact"
    ])
    @test Bool(verification["grouped_measured_repeats_bitwise_exact"])
    @test Bool(verification["scalar_scattered_is_diagnostic_only"])
    @test Bool(verification["passed"])

    cases = summary["cases"]
    reference = cases["reference_2_token"]
    wide = cases["wide_prefill"]
    @test Int(reference["prompt_tokens"]) == 2
    @test Int(wide["prompt_tokens"]) == 32
    for workload in (reference, wide)
        @test Bool(workload["grouped_output_and_routes_bitwise_exact"])
        @test Bool(workload["grouped_measured_traffic_exact"])
        @test Bool(workload["grouped_repeat_bitwise_exact"])
        @test Bool(workload["comparisons"][
            "grouped_measured_traffic"
        ]["exact"])
    end

    function configuration(workload, name)
        return workload["configurations"][name]
    end
    reference_materialized = configuration(reference, "materialized_grouped")
    reference_grouped = configuration(reference, "grouped_scattered")
    reference_scalar = configuration(reference, "scalar_scattered")
    wide_materialized = configuration(wide, "materialized_grouped")
    wide_grouped = configuration(wide, "grouped_scattered")
    wide_scalar = configuration(wide, "scalar_scattered")

    @test Bool(reference_materialized["grouped_experts"])
    @test String(reference_materialized["dispatch"]) == "materialized"
    @test Bool(reference_grouped["grouped_experts"])
    @test String(reference_grouped["dispatch"]) == "scattered"
    @test !Bool(reference_scalar["grouped_experts"])
    @test String(reference_scalar["dispatch"]) == "scattered"

    reference_speedup =
        Float64(reference_materialized["aggregate"][
            "request_seconds_median"
        ]) /
        Float64(reference_grouped["aggregate"][
            "request_seconds_median"
        ])
    wide_speedup =
        Float64(wide_materialized["aggregate"][
            "request_seconds_median"
        ]) /
        Float64(wide_grouped["aggregate"][
            "request_seconds_median"
        ])
    @test reference_speedup > 10
    @test wide_speedup > 6

    function first_traffic(configuration)
        return configuration["measurements"][1]["traffic"]
    end
    reference_materialized_traffic = first_traffic(reference_materialized)
    reference_grouped_traffic = first_traffic(reference_grouped)
    wide_materialized_traffic = first_traffic(wide_materialized)
    wide_grouped_traffic = first_traffic(wide_grouped)
    @test Int(reference_materialized_traffic[
        "materialization"
    ]["active_materialization_bytes"]) > 10_000_000_000
    @test Int(wide_materialized_traffic[
        "materialization"
    ]["active_materialization_bytes"]) > 16_000_000_000
    for traffic in (reference_grouped_traffic, wide_grouped_traffic)
        @test Int(traffic["materialization"]["active_materializations"]) == 0
        @test Int(traffic[
            "materialization"
        ]["active_materialization_bytes"]) == 0
        @test Int(traffic["scattered_state"]["dispatches"]) == 96
        @test Int(traffic[
            "scattered_state"
        ]["pointer_table_bytes"]) < 100_000
        @test Int(traffic["scattered_state"]["workspace_bytes"]) < 20_000_000
        @test Int(traffic[
            "scattered_state"
        ]["workspace_allocations"]) == 0
        @test Int(traffic["scattered_state"]["workspace_reuses"]) == 96
    end

    grouped_parity = reference["external_reference"]["grouped_scattered"]
    @test Bool(grouped_parity["output"]["prefill"]["argmax_match"])
    @test Bool(grouped_parity["output"]["decode"]["argmax_match"])
    scalar_diagnostic = wide["comparisons"][
        "scalar_scattered_vs_materialized_grouped"
    ]
    @test Float64(scalar_diagnostic["prefill"]["max_abs"]) > 0
    @test Float64(scalar_diagnostic["decode"]["max_abs"]) > 0
    @test Bool(scalar_diagnostic["prefill"]["argmax_match"])
    @test Bool(scalar_diagnostic["decode"]["argmax_match"])
    @test Float64(wide_scalar["aggregate"]["request_seconds_median"]) > 0

    source_paths = Dict(
        "benchmark_script" => joinpath(
            "scripts",
            "benchmark_qwen3_moe_cuda_grouped_scattered.jl",
        ),
        "offload_implementation" => joinpath(
            "src",
            "generation",
            "qwen3_moe_offload.jl",
        ),
        "cuda_extension" => joinpath("ext", "LifeAICUDAExt.jl"),
    )
    for (name, relative_path) in source_paths
        expected = String(summary["source_sha256"][name])
        actual = bytes2hex(sha256(read(joinpath(repo_root, relative_path))))
        @test actual == expected
    end
end
