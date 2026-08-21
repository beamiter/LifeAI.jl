using JSON3
using LifeAI
using BFloat16s: BFloat16
using Test

isdefined(@__MODULE__, :qwen3_moe_historical_benchmark_source_status) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

const QWEN3_MOE_COALESCED_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_COALESCED_TINY_FIXTURE = joinpath(
    QWEN3_MOE_COALESCED_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Safetensors batch reads preserve tensor semantics" begin
    reader = open_safetensors_reader(QWEN3_MOE_COALESCED_TINY_FIXTURE)
    prefix = "model.layers.0.mlp.experts.0"
    names = [
        "$prefix.down_proj.weight",
        "$prefix.gate_proj.weight",
        "$prefix.up_proj.weight",
    ]

    for dtype in (Float32, BFloat16)
        expected = Dict(
            name => read_safetensors_tensor(
                reader,
                name;
                target_dtype=dtype,
            ) for name in names
        )
        coalesced = read_safetensors_tensors(
            reader,
            reverse(names);
            target_dtype=dtype,
        )
        shared_open = read_safetensors_tensors(
            reader,
            names;
            target_dtype=dtype,
            coalesce_adjacent=false,
        )
        @test Set(keys(coalesced)) == Set(names)
        @test Set(keys(shared_open)) == Set(names)
        for name in names
            @test eltype(coalesced[name]) == dtype
            @test coalesced[name] == expected[name]
            @test shared_open[name] == expected[name]
        end
    end

    @test isempty(read_safetensors_tensors(reader, String[]))
    @test_throws ArgumentError read_safetensors_tensors(
        reader,
        [names[1], names[1]],
    )
    @test_throws ArgumentError read_safetensors_tensors(
        reader,
        ["missing.weight"],
    )
    @test_throws ArgumentError read_safetensors_tensors(
        reader,
        names;
        target_dtype=Float16,
    )
end

@testset "Qwen3 MoE expert read modes are exact and fail closed" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_COALESCED_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
    )
    tensor = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    tensor_keys = sort!(collect(keys(session.expert_cache)))
    @test tensor.expert_cache.read_mode == :tensor

    configure_hf_qwen3_moe_expert_cache!(session; read_mode=:shared_open)
    shared_open = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test shared_open.logits == tensor.logits
    @test sort!(collect(keys(session.expert_cache))) == tensor_keys
    @test shared_open.expert_cache.read_mode == :shared_open

    configure_hf_qwen3_moe_expert_cache!(session; read_mode=:coalesced)
    coalesced = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test coalesced.logits == tensor.logits
    @test sort!(collect(keys(session.expert_cache))) == tensor_keys
    @test coalesced.expert_cache.read_mode == :coalesced

    entries = qwen3_moe_expert_cache_stats(session).entries
    @test_throws ArgumentError configure_hf_qwen3_moe_expert_cache!(
        session;
        read_mode=:read_ahead,
    )
    @test qwen3_moe_expert_cache_stats(session).entries == entries
    @test qwen3_moe_expert_cache_stats(session).read_mode == :coalesced

    @test_throws ArgumentError load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_COALESCED_TINY_FIXTURE;
        expert_cache_budget_bytes=8 * 144,
        expert_read_mode=:read_ahead,
    )
end

@testset "Qwen3 MoE real coalesced-read result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_coalesced_reads",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test String(summary["revision"]) ==
        "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    @test String(summary["compute_dtype"]) == "bfloat16"
    @test Int(summary["session"]["julia_threads"]) == 8
    @test Int(summary["session"]["read_workers"]) == 8
    @test Int(summary["session"]["cache_budget_bytes"]) == 4 * 2^30
    @test String(summary["session"]["cache_policy"]) ==
        "layer_balanced_lru"
    @test String(summary["session"]["dispatch"]) == "scattered"
    @test Int(summary["session"]["gc_interval_layers"]) == 8

    workload = summary["workload"]
    @test String(workload["trace"]) ==
        "english_32 plus one greedy decode"
    @test Int(workload["tokens_per_prompt"]) == 32
    @test Int(workload["decode_tokens"]) == 1
    @test String(workload["tokens_sha256"]) ==
        "0f06ce74166c11acb516970ec217b6ef52c9592f2ecd88067189d26959f74927"

    layout = summary["read_layout"]
    @test String(layout["projection_order"]) == "down, gate, up"
    @test Int(layout["projection_bytes"]) == 3_145_728
    @test Int(layout["expert_bytes"]) == 9_437_184
    @test Bool(layout["exactly_adjacent"])
    @test !Bool(layout["bridge_unrelated_bytes"])

    methodology = summary["methodology"]
    @test Int(methodology["repetitions"]) == 3
    @test length(methodology["orders"]) == 3
    @test occursin(
        "POSIX_FADV_DONTNEED",
        String(methodology["cold_control"]),
    )
    @test occursin("/proc/self/io", String(methodology["cold_verification"]))
    @test !Bool(methodology["pinned_upload"])

    configurations = summary["configurations"]
    @test String.(getindex.(configurations, "mode")) ==
        ["tensor", "shared_open", "coalesced"]
    for configuration in configurations
        @test Int(configuration["cold"]["repetitions"]) == 3
        @test Int(configuration["revisit"]["repetitions"]) == 3
        for run in configuration["cold"]["runs"]
            @test Int(run["cache_hits"]) == 71
            @test Int(run["cache_misses"]) == 3_039
            @test Int(run["cache_evictions"]) == 313
            @test Int(run["expert_bytes_read"]) == 28_679_602_176
            @test Int(run["expert_bytes_uploaded"]) == 28_679_602_176
            @test Int(run["read_jobs"]) == 3_039
            @test 26_235_000_000 <=
                Int(run["process_storage_read_bytes"]) <=
                26_236_000_000
        end
        for run in configuration["revisit"]["runs"]
            @test Int(run["cache_hits"]) == 71
            @test Int(run["cache_misses"]) == 3_039
            @test Int(run["expert_bytes_read"]) == 28_679_602_176
            @test Int(run["process_storage_read_bytes"]) == 0
        end
    end

    tensor, shared_open, coalesced = configurations
    @test Float64(tensor["cold"]["request_seconds_median"]) < 18
    @test Float64(tensor["revisit"]["request_seconds_median"]) < 15
    @test Float64(coalesced["cold"]["request_seconds_median"]) > 20
    @test Float64(coalesced["revisit"]["request_seconds_median"]) > 19
    @test Int(coalesced["cold"]["read_syscalls_median"]) < 3_400
    @test Int(tensor["cold"]["read_syscalls_median"]) > 9_300
    @test abs(
        Float64(shared_open["cold"]["request_seconds_median"]) /
        Float64(tensor["cold"]["request_seconds_median"]) - 1,
    ) < 0.02

    comparisons = summary["comparisons"]
    @test 0.98 < Float64(comparisons["shared_open_cold_speedup"]) < 1.02
    @test 0.98 < Float64(comparisons["shared_open_revisit_speedup"]) < 1.02
    @test Float64(comparisons["coalesced_cold_speedup"]) < 0.85
    @test Float64(comparisons["coalesced_revisit_speedup"]) < 0.80
    @test Float64(comparisons[
        "coalesced_cold_read_syscall_reduction"
    ]) > 0.64
    @test Float64(comparisons[
        "coalesced_revisit_read_syscall_reduction"
    ]) > 0.64
    @test Float64(comparisons[
        "coalesced_cold_allocation_reduction"
    ]) < 0
    @test Float64(comparisons[
        "coalesced_revisit_allocation_reduction"
    ]) < 0

    @test Bool(summary["correctness"]["all_outputs_exact"])
    decision = summary["decision"]
    @test String(decision["fastest_cold_mode"]) == "tensor"
    @test String(decision["fastest_revisit_mode"]) == "shared_open"
    @test String(decision["selected_default_read_mode"]) == "tensor"
    @test !Bool(decision["arbitrary_checkpoint_layout_generalized"])

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
