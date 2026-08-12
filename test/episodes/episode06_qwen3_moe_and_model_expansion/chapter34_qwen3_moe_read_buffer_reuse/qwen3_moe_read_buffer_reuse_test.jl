using BFloat16s: BFloat16
using JSON3
using LifeAI
using SHA: sha256
using Test

const QWEN3_MOE_READ_BUFFER_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_READ_BUFFER_TINY_FIXTURE = joinpath(
    QWEN3_MOE_READ_BUFFER_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

@testset "Safetensors raw buffers preserve decoded ownership" begin
    reader = open_safetensors_reader(QWEN3_MOE_READ_BUFFER_TINY_FIXTURE)
    prefix = "model.layers.0.mlp.experts.0"
    names = [
        "$prefix.gate_proj.weight",
        "$prefix.up_proj.weight",
        "$prefix.down_proj.weight",
    ]
    byte_count = maximum(
        reader.locations[name].data_stop - reader.locations[name].data_start
        for name in names
    )
    raw = Vector{UInt8}(undef, byte_count)
    expected = Dict(
        name => read_safetensors_tensor(
            reader,
            name;
            target_dtype=BFloat16,
        ) for name in names
    )
    buffered = Dict{String,Any}()
    for name in names
        buffered[name] = read_safetensors_tensor(
            reader,
            name;
            target_dtype=BFloat16,
            raw_buffer=raw,
        )
        @test buffered[name] == expected[name]
    end
    fill!(raw, 0x00)
    for name in names
        @test buffered[name] == expected[name]
    end

    pool = LifeAI._SafetensorsReadBufferPool(1, byte_count)
    @test pool.buffer_count == 1
    @test pool.buffer_bytes == byte_count
    @test_throws ErrorException LifeAI._with_safetensors_read_buffer(
        _ -> error("return buffer on failure"),
        pool,
    )
    @test pool.borrows[] == 1
    borrowed_length = LifeAI._with_safetensors_read_buffer(pool) do buffer
        length(buffer)
    end
    @test borrowed_length == byte_count
    @test pool.borrows[] == 2
    @test LifeAI._reset_safetensors_read_buffer_pool!(pool) === pool
    @test pool.borrows[] == 0
    @test_throws ArgumentError LifeAI._SafetensorsReadBufferPool(0, 1)
    @test_throws ArgumentError LifeAI._SafetensorsReadBufferPool(1, -1)
    @test_throws ArgumentError LifeAI._expect_tensor(
        Dict("not_array" => 1),
        "not_array",
        (),
    )
end

@testset "Qwen3 MoE read-buffer modes are exact and bounded" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_READ_BUFFER_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
        expert_read_buffer_reuse=true,
        expert_read_mode=:tensor,
        expert_miss_pipeline=:overlapped,
        expert_read_workers=2,
    )
    pooled = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    pooled_stats = pooled.expert_cache
    pooled_keys = sort!(collect(keys(session.expert_cache)))
    @test pooled_stats.read_buffer_reuse
    @test pooled_stats.read_buffer_count == 2
    @test pooled_stats.read_buffer_bytes == 2 * 96
    @test pooled_stats.read_buffer_borrows == pooled_stats.misses == 8

    configure_hf_qwen3_moe_expert_cache!(
        session;
        read_buffer_reuse=false,
    )
    unpooled = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    unpooled_stats = unpooled.expert_cache
    @test unpooled.logits == pooled.logits
    @test sort!(collect(keys(session.expert_cache))) == pooled_keys
    @test !unpooled_stats.read_buffer_reuse
    @test unpooled_stats.read_buffer_count == 0
    @test unpooled_stats.read_buffer_bytes == 0
    @test unpooled_stats.read_buffer_borrows == 0

    configure_hf_qwen3_moe_expert_cache!(
        session;
        read_buffer_reuse=true,
        miss_pipeline=:sequential,
    )
    sequential = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    sequential_stats = sequential.expert_cache
    @test sequential.logits == pooled.logits
    @test sequential_stats.read_buffer_count == 1
    @test sequential_stats.read_buffer_bytes == 96
    @test sequential_stats.read_buffer_borrows == sequential_stats.misses == 8

    configure_hf_qwen3_moe_expert_cache!(session; read_mode=:shared_open)
    shared_open = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    shared_open_stats = shared_open.expert_cache
    @test shared_open.logits == pooled.logits
    @test shared_open_stats.read_buffer_reuse
    @test shared_open_stats.read_buffer_count == 0
    @test shared_open_stats.read_buffer_bytes == 0
    @test shared_open_stats.read_buffer_borrows == 0

    configure_hf_qwen3_moe_expert_cache!(session; read_mode=:coalesced)
    coalesced = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test coalesced.logits == pooled.logits
    @test coalesced.expert_cache.read_buffer_count == 0

    zero_budget = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_READ_BUFFER_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=0,
        expert_read_buffer_reuse=true,
    )
    zero_stats = qwen3_moe_expert_cache_stats(zero_budget)
    @test zero_stats.read_buffer_reuse
    @test zero_stats.read_buffer_count == 0
    @test zero_stats.read_buffer_bytes == 0
end

@testset "Qwen3 MoE real read-buffer-reuse result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary = JSON3.read(read(joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_read_buffer_reuse",
        "summary.json",
    ), String))
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
    @test String(session["miss_pipeline"]) == "overlapped"
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
    @test String.(methodology["orders"]) == [
        "unpooled then pooled",
        "pooled then unpooled",
        "unpooled then pooled",
    ]
    @test Bool(methodology["same_process_ab"])
    @test occursin(
        "POSIX_FADV_DONTNEED",
        String(methodology["cold_control"]),
    )
    @test occursin("/proc/self/io", String(methodology["cold_verification"]))

    unpooled, pooled = summary["configurations"]
    @test String(unpooled["mode"]) == "unpooled"
    @test String(pooled["mode"]) == "pooled"
    @test Int(unpooled["cold"]["allocated_bytes_median"]) ==
        57_646_593_592
    @test Int(unpooled["revisit"]["allocated_bytes_median"]) ==
        57_646_612_376
    @test Int(pooled["cold"]["allocated_bytes_median"]) ==
        28_966_979_608
    @test Int(pooled["revisit"]["allocated_bytes_median"]) ==
        28_966_615_984
    @test Float64(pooled["cold"]["request_seconds_median"]) < 15
    @test Float64(pooled["revisit"]["request_seconds_median"]) < 13

    for configuration in (unpooled, pooled)
        mode = String(configuration["mode"])
        for phase in ("cold", "revisit")
            @test Int(configuration[phase]["repetitions"]) == 3
            for run in configuration[phase]["runs"]
                @test Int(run["cache_hits"]) == 71
                @test Int(run["cache_misses"]) == 3_039
                @test Int(run["cache_evictions"]) == 313
                @test Int(run["expert_bytes_read"]) == 28_679_602_176
                @test Int(run["expert_bytes_uploaded"]) == 28_679_602_176
                @test Int(run["read_jobs"]) == 3_039
                if mode == "pooled"
                    @test Bool(run["read_buffer_reuse"])
                    @test Int(run["read_buffer_count"]) == 8
                    @test Int(run["read_buffer_bytes"]) == 25_165_824
                    @test Int(run["read_buffer_borrows"]) == 3_039
                else
                    @test !Bool(run["read_buffer_reuse"])
                    @test Int(run["read_buffer_count"]) == 0
                    @test Int(run["read_buffer_bytes"]) == 0
                    @test Int(run["read_buffer_borrows"]) == 0
                end
                if phase == "cold"
                    @test 26_235_000_000 <=
                        Int(run["process_storage_read_bytes"]) <=
                        26_236_000_000
                else
                    @test Int(run["process_storage_read_bytes"]) == 0
                end
            end
        end
    end

    comparisons = summary["comparisons"]
    @test 0.497 < Float64(comparisons[
        "cold_allocation_reduction"
    ]) < 0.498
    @test 0.497 < Float64(comparisons[
        "revisit_allocation_reduction"
    ]) < 0.498
    @test Int(comparisons["logical_payload_bytes"]) == 28_679_602_176
    @test Int(comparisons["resident_pool_bytes"]) == 25_165_824
    @test abs(
        Int(comparisons["cold_eliminated_allocation_bytes"]) -
        Int(comparisons["logical_payload_bytes"]),
    ) < 1_000_000
    @test abs(
        Int(comparisons["revisit_eliminated_allocation_bytes"]) -
        Int(comparisons["logical_payload_bytes"]),
    ) < 1_000_000
    @test Float64(comparisons["cold_speedup"]) > 1.20
    @test Float64(comparisons["revisit_speedup"]) > 1.30

    @test Bool(summary["correctness"]["all_outputs_exact"])
    decision = summary["decision"]
    @test Bool(decision["enable_read_buffer_reuse_by_default"])
    @test Bool(decision["pool_bounded_by_reader_workers"])
    @test !Bool(decision["final_semantic_matrix_reused"])
    @test !Bool(decision["arbitrary_workload_speedup_generalized"])

    for (name, relative_path) in pairs(summary["source_paths"])
        expected = String(summary["source_sha256"][name])
        actual = bytes2hex(sha256(read(joinpath(
            repo_root,
            String(relative_path),
        ))))
        @test actual == expected
    end
end
