using BFloat16s: BFloat16
using LifeAI
using JSON3
using Test

isdefined(@__MODULE__, :qwen3_moe_historical_benchmark_source_status) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

const QWEN3_MOE_CHAPTER24_FIXTURE = joinpath(
    dirname(@__DIR__),
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_real_checkpoint",
    "config.json",
)
const QWEN3_MOE_TINY_OFFLOAD_FIXTURE = joinpath(
    dirname(dirname(QWEN3_MOE_CHAPTER24_FIXTURE)),
    "qwen3_moe_tiny_parity",
)

function _qwen3_offload_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected Qwen3 MoE offload call to fail")
end

function _qwen3_offload_session_state(session)
    return (;
        position=session.position,
        kv_cache=Tuple(
            (keys=copy(layer.keys), values=copy(layer.values))
            for layer in session.caches
        ),
        expert_cache=qwen3_moe_expert_cache_stats(session),
    )
end

function _qwen3_offload_atomic_failure(thunk, session)
    preserved = _qwen3_offload_session_state(session)
    failure = _qwen3_offload_captured_error(thunk)
    return (;
        failure,
        unchanged=isequal(_qwen3_offload_session_state(session), preserved),
    )
end

@testset "Qwen3 MoE session options fail before checkpoint I/O" begin
    mktempdir() do directory
        too_large = big(typemax(Int)) + 1
        cases = (
            ("context_tokens must be an integer", (; context_tokens=true)),
            (
                "context_tokens is outside the host integer range",
                (; context_tokens=too_large),
            ),
            ("context_tokens must be positive", (; context_tokens=0)),
            (
                "prefill_chunk_tokens must be an integer",
                (; context_tokens=8, prefill_chunk_tokens=true),
            ),
            (
                "prefill_chunk_tokens is outside the host integer range",
                (; context_tokens=8, prefill_chunk_tokens=too_large),
            ),
            (
                "prefill_chunk_tokens must be in 1:context_tokens",
                (; context_tokens=8, prefill_chunk_tokens=0),
            ),
            (
                "prefill_chunk_tokens must be in 1:context_tokens",
                (; context_tokens=8, prefill_chunk_tokens=9),
            ),
            (
                "expert_cache_budget_bytes must be an integer",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_cache_budget_bytes=true),
            ),
            (
                "expert_cache_budget_bytes is outside the host integer range",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_cache_budget_bytes=too_large),
            ),
            (
                "expert_cache_budget_bytes must be non-negative",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_cache_budget_bytes=-1),
            ),
            (
                "expert_gc_interval_layers must be an integer",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_gc_interval_layers=true),
            ),
            (
                "expert_gc_interval_layers is outside the host integer range",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_gc_interval_layers=too_large),
            ),
            (
                "expert_gc_interval_layers must be non-negative",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_gc_interval_layers=-1),
            ),
            (
                "expert_read_workers must be an integer",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_read_workers=true),
            ),
            (
                "expert_read_workers is outside the host integer range",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_read_workers=too_large),
            ),
            (
                "expert_read_workers must be positive",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_read_workers=0),
            ),
            (
                "expert_cache_policy must be",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_cache_policy=:bad),
            ),
            (
                "expert_cache_dispatch must be",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_cache_dispatch=:bad),
            ),
            (
                "expert_read_mode must be",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_read_mode=:bad),
            ),
            (
                "expert_miss_pipeline must be",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_miss_pipeline=:bad),
            ),
            (
                "scattered expert cache dispatch requires a positive cache budget",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_cache_dispatch=:scattered),
            ),
            (
                "overlapped expert misses require a positive cache budget",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_miss_pipeline=:overlapped),
            ),
            (
                "pinned expert upload requires expert_miss_pipeline=:overlapped",
                (; context_tokens=8, prefill_chunk_tokens=1,
                    expert_pinned_upload=true),
            ),
        )
        for (needle, options) in cases
            failure = _qwen3_offload_captured_error() do
                load_hf_qwen3_moe_offload_session(
                    directory;
                    options...,
                )
            end
            @test failure isa ArgumentError
            message = sprint(showerror, failure)
            @test occursin(needle, message)
            @test !occursin("config.json", message)
        end

        io_failure = _qwen3_offload_captured_error() do
            load_hf_qwen3_moe_offload_session(
                directory;
                context_tokens=Int32(8),
                prefill_chunk_tokens=big(1),
                expert_cache_budget_bytes=Int128(0),
                expert_gc_interval_layers=Int16(0),
                expert_read_workers=big(1),
            )
        end
        @test io_failure isa ArgumentError
        @test occursin("JSON file does not exist", sprint(showerror, io_failure))
    end
end

@testset "Qwen3 MoE GPU offload contract" begin
    config = load_hf_qwen3_moe_config(
        QWEN3_MOE_CHAPTER24_FIXTURE;
        max_seq_len=40_960,
    )
    model = GPTModel(config)
    plan = qwen3_moe_offload_plan(model, 40_960)

    @test plan.context_tokens == 40_960
    @test plan.batch_size == 1
    @test plan.max_active_experts == 128
    @test plan.dtype_bytes == sizeof(BFloat16)
    @test plan.resident_parameter_bytes == 2_459_856_896
    @test plan.active_expert_layer_bytes == 1_207_959_552
    @test plan.kv_cache_bytes == 4_026_531_840
    @test plan.working_set_floor_bytes == 7_694_348_288
    @test plan.working_set_floor_bytes < 8 * 2^30

    @test_throws ArgumentError qwen3_moe_offload_plan(
        model,
        40_960;
        batch_size=2,
    )

    routed_floor = qwen3_moe_offload_plan(
        model,
        40_960;
        max_active_experts=model.experts_per_token,
    )
    @test model.experts_per_token == 8
    @test routed_floor.max_active_experts == 8
    @test routed_floor.active_expert_layer_bytes == 75_497_472
    @test routed_floor.working_set_floor_bytes == 6_561_886_208

    normalized_plan = qwen3_moe_offload_plan(
        model,
        Int32(1);
        batch_size=big(1),
        max_active_experts=Int128(model.experts_per_token),
        dtype_bytes=big(2),
    )
    @test normalized_plan.context_tokens isa Int
    @test normalized_plan.batch_size isa Int
    @test normalized_plan.max_active_experts isa Int
    @test normalized_plan.dtype_bytes isa Int

    too_large = big(typemax(Int)) + 1
    for invalid_context in (true, too_large)
        @test_throws ArgumentError qwen3_moe_offload_plan(
            model,
            invalid_context,
        )
    end
    for invalid_batch in (true, too_large)
        @test_throws ArgumentError qwen3_moe_offload_plan(
            model,
            1;
            batch_size=invalid_batch,
        )
    end
    for invalid_active in (true, too_large)
        failure = _qwen3_offload_captured_error() do
            qwen3_moe_offload_plan(
                model,
                1;
                max_active_experts=invalid_active,
            )
        end
        @test failure isa ArgumentError
        @test occursin(
            invalid_active isa Bool ?
                "max_active_experts must be an integer" :
                "max_active_experts is outside the host integer range",
            sprint(showerror, failure),
        )
    end
    for invalid_dtype in (true, too_large)
        @test_throws ArgumentError qwen3_moe_offload_plan(
            model,
            1;
            dtype_bytes=invalid_dtype,
        )
    end

    for routes in (
        Int8[5 2 8; 2 5 2],
        Int128[5 2 8; 2 5 2],
        BigInt[5 2 8; 2 5 2],
    )
        remapped = LifeAI._qwen3_local_expert_routes(routes, big(8))
        @test remapped.active_experts == [2, 5, 8]
        @test remapped.local_indices == Int32[2 1 3; 1 2 1]
        @test size(remapped.local_indices) == size(routes)
        @test eltype(remapped.local_indices) === Int32
    end

    invalid_routes = (
        Bool[true false],
        Matrix{Int}(undef, 0, 1),
        Int32[0 1],
        Int32[1 9],
        BigInt[big(typemax(Int)) + 1 1],
        BigInt[big(typemin(Int)) - 1 1],
    )
    for routes in invalid_routes
        @test_throws ArgumentError LifeAI._qwen3_local_expert_routes(routes, 8)
    end
    for invalid_expert_count in (
        true,
        0,
        -1,
        big(typemax(Int)) + 1,
    )
        @test_throws ArgumentError LifeAI._qwen3_local_expert_routes(
            Int8[1 2],
            invalid_expert_count,
        )
    end
    if typemax(Int) > typemax(Int32)
        sparse_large = LifeAI._qwen3_local_expert_routes(
            Int8[1 1],
            big(typemax(Int32)) + 1,
        )
        @test sparse_large.active_experts == [1]
        @test sparse_large.local_indices == Int32[1 1]
    end

    @test_throws ArgumentError qwen3_moe_offload_plan(model, 0)
    @test_throws ArgumentError qwen3_moe_offload_plan(
        model,
        40_960;
        max_active_experts=129,
    )
    @test_throws ArgumentError qwen3_moe_offload_plan(
        model,
        40_960;
        max_active_experts=model.experts_per_token - 1,
    )

    overflow_width = isqrt(typemax(Int)) + 1
    isodd(overflow_width) && (overflow_width += 1)
    @test_throws ArgumentError GPTModel(
        1,
        overflow_width,
        overflow_width ÷ 2,
        1;
        num_kv_heads=1,
        head_dim=2,
        mlp_hidden_dim=2,
        use_bias=false,
        lm_head_bias=false,
        use_rope=true,
        use_qk_norm=true,
        max_seq_len=1,
        rope_style=:rotate_half,
        norm_type=:rmsnorm,
        mlp_type=:qwen3_moe,
        num_experts=2,
        experts_per_token=1,
    )
    @test_throws ArgumentError qwen3_moe_offload_plan(
        model,
        1;
        dtype_bytes=typemax(Int),
    )
end

@testset "Qwen3 MoE offload chunked prefill and decode" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_TINY_OFFLOAD_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=1,
        grouped_experts=false,
    )
    raw_fields = ntuple(
        index -> getfield(session, index),
        fieldcount(typeof(session)),
    )
    @test_throws MethodError HFQwen3MoEOffloadSession(raw_fields...)

    prefill = prefill_hf_qwen3_moe_offload!(session, Int128[2, 3])
    @test prefill.position == 2
    @test length(prefill.chunks) == 2
    @test size(prefill.logits) == (17, 1, 1)
    @test sum(chunk.expert_bytes_read for chunk in prefill.chunks) ==
        prefill.expert_bytes_read
    @test all(length(chunk.active_experts) == 2 for chunk in prefill.chunks)

    decode = decode_hf_qwen3_moe_offload!(session, big(4))
    @test decode.position == 3
    @test size(decode.logits) == (17, 1, 1)
    @test decode.expert_bytes_read > 0

    # Unused preallocated cache slots may legitimately contain NaNs.  Seed that
    # state explicitly so invalid-input atomicity cannot regress to `==`, which
    # reports identical NaN-bearing snapshots as unequal.
    for layer in session.caches
        tail = (session.position + 1):session.context_tokens
        fill!(@view(layer.keys[:, :, tail, :]), BFloat16(NaN))
        fill!(@view(layer.values[:, :, tail, :]), BFloat16(NaN))
    end
    preserved = _qwen3_offload_session_state(session)
    too_large = big(typemax(Int)) + 1
    for invalid_prompt in (
        Bool[true],
        Float64[2.0],
        Char[Char(2)],
        BigInt[too_large],
    )
        failure = _qwen3_offload_captured_error() do
            prefill_hf_qwen3_moe_offload!(session, invalid_prompt)
        end
        @test failure isa ArgumentError
        @test occursin(
            "Qwen3 MoE offload prompt token",
            sprint(showerror, failure),
        )
        @test isequal(_qwen3_offload_session_state(session), preserved)
    end
    for invalid_token in (true, 4.0, Char(4), too_large)
        failure = _qwen3_offload_captured_error() do
            decode_hf_qwen3_moe_offload!(session, invalid_token)
        end
        @test failure isa ArgumentError
        @test occursin(
            "Qwen3 MoE offload decode token",
            sprint(showerror, failure),
        )
        @test isequal(_qwen3_offload_session_state(session), preserved)
    end


    valid_position = session.position
    session.position = session.context_tokens + 1
    for operation in (
        () -> reset_hf_qwen3_moe_offload_session!(session),
        () -> prefill_hf_qwen3_moe_offload!(session, [2]),
        () -> decode_hf_qwen3_moe_offload!(session, 4),
    )
        rejected = _qwen3_offload_atomic_failure(operation, session)
        @test rejected.failure isa ArgumentError
        @test occursin(
            "position must be in 0:context_tokens",
            sprint(showerror, rejected.failure),
        )
        @test rejected.unchanged
    end
    session.position = valid_position

    valid_chunk = session.prefill_chunk_tokens
    session.prefill_chunk_tokens = 0
    rejected_chunk = _qwen3_offload_atomic_failure(session) do
        reset_hf_qwen3_moe_offload_session!(session)
    end
    @test rejected_chunk.failure isa ArgumentError
    @test occursin(
        "prefill_chunk_tokens must be in 1:context_tokens",
        sprint(showerror, rejected_chunk.failure),
    )
    @test rejected_chunk.unchanged
    session.prefill_chunk_tokens = valid_chunk

    valid_context = session.context_tokens
    session.context_tokens -= 1
    rejected_context = _qwen3_offload_atomic_failure(session) do
        prefill_hf_qwen3_moe_offload!(session, [2])
    end
    @test rejected_context.failure isa DimensionMismatch
    @test occursin(
        "context does not match model.max_seq_len",
        sprint(showerror, rejected_context.failure),
    )
    @test rejected_context.unchanged
    session.context_tokens = valid_context

    removed_cache = pop!(session.caches)
    rejected_layer_count = _qwen3_offload_atomic_failure(session) do
        decode_hf_qwen3_moe_offload!(session, 4)
    end
    @test rejected_layer_count.failure isa DimensionMismatch
    @test occursin(
        "cache layer count does not match model.num_layers",
        sprint(showerror, rejected_layer_count.failure),
    )
    @test rejected_layer_count.unchanged
    push!(session.caches, removed_cache)

    valid_first_cache = session.caches[1]
    invalid_shape = (
        session.model.head_dim,
        session.model.num_kv_heads,
        session.context_tokens - 1,
        1,
    )
    session.caches[1] = LifeAI.BF16AStaticLayerCache(
        zeros(BFloat16, invalid_shape),
        zeros(BFloat16, invalid_shape),
    )
    rejected_geometry = _qwen3_offload_atomic_failure(session) do
        reset_hf_qwen3_moe_offload_session!(session)
    end
    @test rejected_geometry.failure isa DimensionMismatch
    @test occursin(
        "does not match the model/session geometry",
        sprint(showerror, rejected_geometry.failure),
    )
    @test rejected_geometry.unchanged
    session.caches[1] = valid_first_cache

    rejected_alias = _qwen3_offload_atomic_failure(session) do
        session.caches[1] = LifeAI.BF16AStaticLayerCache(
            valid_first_cache.keys,
            valid_first_cache.keys,
        )
    end
    @test rejected_alias.failure isa ArgumentError
    @test occursin(
        "keys and values must use distinct storage",
        sprint(showerror, rejected_alias.failure),
    )
    @test rejected_alias.unchanged

    valid_second_cache = session.caches[2]
    session.caches[2] = valid_first_cache
    rejected_cross_layer_alias = _qwen3_offload_atomic_failure(session) do
        prefill_hf_qwen3_moe_offload!(session, [2])
    end
    @test rejected_cross_layer_alias.failure isa ArgumentError
    @test occursin(
        "cache layers must use distinct storage",
        sprint(showerror, rejected_cross_layer_alias.failure),
    )
    @test rejected_cross_layer_alias.unchanged
    session.caches[2] = valid_second_cache

    rejected_dtype = _qwen3_offload_atomic_failure(session) do
        session.caches[1] = LifeAI.BF16AStaticLayerCache(
            zeros(Float32, size(valid_first_cache.keys)),
            zeros(Float32, size(valid_first_cache.values)),
        )
    end
    @test rejected_dtype.failure isa ArgumentError
    @test occursin(
        "cache storage must use BFloat16",
        sprint(showerror, rejected_dtype.failure),
    )
    @test rejected_dtype.unchanged

    @test reset_hf_qwen3_moe_offload_session!(session).position == 0
end

@testset "Qwen3 MoE real GPU offload result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_offload",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test Int(summary["session"]["context_tokens"]) == 40_960
    @test Bool(summary["session"]["full_context_cache_allocated"])
    @test Int(summary["session"]["kv_cache_bytes"]) == 4_026_531_840
    @test Int(summary["session"]["working_set_floor_bytes"]) == 7_694_348_288

    two_token = summary["two_token_reference"]
    @test Bool(two_token["grouped"]["prefill_argmax_match"])
    @test Bool(two_token["grouped"]["decode_argmax_match"])
    @test Float64(two_token["grouped_over_scalar_speedup"]["prefill"]) < 1
    @test Float64(two_token["grouped_over_scalar_speedup"]["decode"]) < 1

    wide = summary["thirty_two_token_reference"]
    @test Bool(wide["path_argmax_match"])
    @test Float64(wide["grouped_over_scalar_speedup"]["prefill"]) > 1
    @test Float64(wide["grouped_over_scalar_speedup"]["decode"]) > 1
    @test Int(wide["grouped"]["prompt_active_experts_maximum"]) < 128
    @test !Bool(summary["decision"]["full_window_prefill_executed"])

    for (name, relative_path) in (
        "benchmark_script" => joinpath(
            "scripts",
            "benchmark_qwen3_moe_cuda_offload.jl",
        ),
        "offload_implementation" => joinpath(
            "src",
            "generation",
            "qwen3_moe_offload.jl",
        ),
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
