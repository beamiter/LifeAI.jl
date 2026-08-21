using BFloat16s: BFloat16
using JSON3
using LifeAI
using Test

isdefined(@__MODULE__, :qwen3_moe_historical_benchmark_source_status) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

const QWEN3_MOE_HOST_BUFFER_EPISODE_DIR = dirname(@__DIR__)
const QWEN3_MOE_HOST_BUFFER_TINY_FIXTURE = joinpath(
    QWEN3_MOE_HOST_BUFFER_EPISODE_DIR,
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_tiny_parity",
)

function _chapter35_raw(values, dtype)
    if dtype == "BF16"
        bits = reinterpret(UInt16, BFloat16.(values))
        return collect(reinterpret(UInt8, bits))
    end
    return collect(reinterpret(UInt8, Float32.(values)))
end

@testset "Safetensors in-place decode preserves semantics" begin
    shape = [2, 3]
    for dtype in ("BF16", "F32"), target_dtype in (BFloat16, Float32)
        raw = _chapter35_raw(1:6, dtype)
        expected = LifeAI._decode_safetensors_values(
            raw,
            dtype,
            shape;
            target_dtype,
        )
        destination = similar(expected)
        result = LifeAI._decode_safetensors_values!(
            destination,
            raw,
            dtype,
            shape,
        )
        @test result === destination
        @test destination == expected
        fill!(raw, 0x00)
        @test destination == expected
    end

    allocation_raw = _chapter35_raw(1:(256 * 256), "BF16")
    allocation_destination = Matrix{BFloat16}(undef, 256, 256)
    LifeAI._decode_safetensors_values!(
        allocation_destination,
        allocation_raw,
        "BF16",
        [256, 256],
    )
    allocated = @allocated LifeAI._decode_safetensors_values!(
        allocation_destination,
        allocation_raw,
        "BF16",
        [256, 256],
    )
    @test allocated < 2_048

    @test_throws DimensionMismatch LifeAI._decode_safetensors_values!(
        Matrix{BFloat16}(undef, 3, 2),
        _chapter35_raw(1:6, "BF16"),
        "BF16",
        shape,
    )
    @test_throws ArgumentError LifeAI._decode_safetensors_values!(
        Matrix{Int}(undef, 2, 3),
        _chapter35_raw(1:6, "BF16"),
        "BF16",
        shape,
    )
    @test_throws ArgumentError LifeAI._decode_safetensors_values!(
        Matrix{BFloat16}(undef, 2, 3),
        UInt8[0x00],
        "BF16",
        shape,
    )
    @test_throws ArgumentError LifeAI._decode_safetensors_values!(
        Matrix{BFloat16}(undef, 2, 3),
        _chapter35_raw(1:6, "BF16"),
        "F16",
        shape,
    )

    reader = open_safetensors_reader(QWEN3_MOE_HOST_BUFFER_TINY_FIXTURE)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    expected = read_safetensors_tensor(
        reader,
        name;
        target_dtype=BFloat16,
    )
    destination = similar(expected)
    raw_buffer = UInt8[]
    @test LifeAI._read_safetensors_tensor!(
        destination,
        reader,
        name;
        raw_buffer,
    ) === destination
    @test destination == expected
    fill!(raw_buffer, 0x00)
    @test destination == expected
    @test_throws DimensionMismatch LifeAI._read_safetensors_tensor!(
        Matrix{BFloat16}(undef, 1, 1),
        reader,
        name;
        raw_buffer,
    )
end

@testset "Qwen3 MoE host staging leases are bounded and fail closed" begin
    pool = LifeAI._Qwen3MoEHostStagingPool(2, 3, 8)
    @test pool.buffer_count == 2
    @test pool.buffer_bytes == 144
    first_lease = LifeAI._borrow_qwen3_moe_host_staging!(pool)
    @test size(first_lease.slot.gate_proj) == (3, 8)
    @test size(first_lease.slot.up_proj) == (3, 8)
    @test size(first_lease.slot.down_proj) == (8, 3)
    @test pool.borrows[] == 1
    @test pool.returns[] == 0
    @test_throws ArgumentError LifeAI._reset_qwen3_moe_host_staging_pool!(
        pool,
    )
    @test LifeAI._return_qwen3_moe_host_staging!(first_lease) === nothing
    @test pool.returns[] == 1
    @test_throws ArgumentError LifeAI._return_qwen3_moe_host_staging!(
        first_lease,
    )
    @test LifeAI._reset_qwen3_moe_host_staging_pool!(pool) === pool
    @test pool.borrows[] == 0
    @test pool.returns[] == 0
    @test_throws ArgumentError LifeAI._Qwen3MoEHostStagingPool(0, 3, 8)
    @test_throws ArgumentError LifeAI._Qwen3MoEHostStagingPool(1, 0, 8)
    @test_throws ArgumentError LifeAI._Qwen3MoEHostStagingPool(1, 3, 0)

    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_HOST_BUFFER_TINY_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=2,
        grouped_experts=false,
        expert_cache_budget_bytes=8 * 144,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=0,
        expert_host_buffer_reuse=true,
        expert_miss_pipeline=:overlapped,
        expert_read_workers=2,
    )
    output = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test output.expert_cache.host_buffer_reuse
    @test output.expert_cache.host_buffer_count == 0
    @test output.expert_cache.host_buffer_bytes == 0
    @test output.expert_cache.host_buffer_borrows == 0
    @test output.expert_cache.host_buffer_returns == 0
end

@testset "Qwen3 MoE real host-buffer-reuse result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_host_buffer_reuse",
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
    @test String(session["miss_pipeline"]) == "overlapped"
    @test Int(session["read_workers"]) == 8
    @test !Bool(session["pinned_upload"])
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
    @test occursin("driver-owned", String(methodology["pageable_ownership"]))
    @test occursin("disabled", String(methodology["pinned_scope"]))

    unpooled, pooled = summary["configurations"]
    @test String(unpooled["mode"]) == "unpooled"
    @test String(pooled["mode"]) == "pooled"
    @test Int(unpooled["cold"]["allocated_bytes_median"]) ==
        28_967_112_320
    @test Int(unpooled["revisit"]["allocated_bytes_median"]) ==
        28_967_436_520
    @test Int(pooled["cold"]["allocated_bytes_median"]) == 288_093_712
    @test Int(pooled["revisit"]["allocated_bytes_median"]) == 288_111_120
    @test Float64(pooled["cold"]["request_seconds_median"]) < 11
    @test Float64(pooled["revisit"]["request_seconds_median"]) < 9

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
                @test Bool(run["read_buffer_reuse"])
                @test Int(run["read_buffer_count"]) == 8
                @test Int(run["read_buffer_bytes"]) == 25_165_824
                @test Int(run["read_buffer_borrows"]) == 3_039
                if mode == "pooled"
                    @test Bool(run["host_buffer_reuse"])
                    @test Int(run["host_buffer_count"]) == 8
                    @test Int(run["host_buffer_bytes"]) == 75_497_472
                    @test Int(run["host_buffer_borrows"]) == 3_039
                    @test Int(run["host_buffer_returns"]) == 3_039
                else
                    @test !Bool(run["host_buffer_reuse"])
                    @test Int(run["host_buffer_count"]) == 0
                    @test Int(run["host_buffer_bytes"]) == 0
                    @test Int(run["host_buffer_borrows"]) == 0
                    @test Int(run["host_buffer_returns"]) == 0
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
    @test 0.990 < Float64(comparisons[
        "cold_allocation_reduction"
    ]) < 0.991
    @test 0.990 < Float64(comparisons[
        "revisit_allocation_reduction"
    ]) < 0.991
    @test Int(comparisons["logical_payload_bytes"]) == 28_679_602_176
    @test Int(comparisons["resident_read_pool_bytes"]) == 25_165_824
    @test Int(comparisons["resident_host_pool_bytes"]) == 75_497_472
    @test abs(
        Int(comparisons["cold_eliminated_allocation_bytes"]) -
        Int(comparisons["logical_payload_bytes"]),
    ) < 1_000_000
    @test abs(
        Int(comparisons["revisit_eliminated_allocation_bytes"]) -
        Int(comparisons["logical_payload_bytes"]),
    ) < 1_000_000
    @test Float64(comparisons["cold_speedup"]) > 1.30
    @test Float64(comparisons["revisit_speedup"]) > 1.29

    @test Bool(summary["correctness"]["all_outputs_exact"])
    decision = summary["decision"]
    @test Bool(decision["enable_host_buffer_reuse_by_default"])
    @test Bool(decision["pool_bounded_by_reader_workers"])
    @test !Bool(decision["cpu_identity_pooling"])
    @test !Bool(decision["pinned_async_pooling"])
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
