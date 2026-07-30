#!/usr/bin/env julia

using BFloat16s: BFloat16
using LuxCUDA
using CUDA
using JSON3
using LifeAI
using Random: Xoshiro
using SHA: sha256

length(ARGS) in (2, 3) || error(
    "usage: julia --project=. scripts/benchmark_qwen3_cuda_deployment.jl " *
    "MODEL_DIR PROFILE_JSON [OUTPUT_JSON]",
)
model_dir, profile_path = ARGS[1:2]
output_path = length(ARGS) == 3 ? ARGS[3] : nothing

profile = load_qwen3_deployment_profile(profile_path)
asset_manifest_path = isabspath(profile.asset_manifest) ?
    profile.asset_manifest :
    joinpath(dirname(abspath(profile_path)), profile.asset_manifest)
CUDA.functional() || error("CUDA.jl is not functional")
Int(CUDA.total_memory()) >= profile.minimum_gpu_bytes || error(
    "GPU does not satisfy profile minimum_gpu_bytes",
)
spec = qwen3_dense_spec(profile.variant)
minimum_free_bytes = try
    Base.checked_add(
        Base.checked_add(
            Base.checked_mul(qwen3_dense_parameter_count(spec), 2),
            qwen3_kv_cache_bytes(spec, profile.context_tokens),
        ),
        profile.workspace_reserve_bytes,
    )
catch err
    err isa OverflowError || rethrow()
    Base.error("profile free-memory preflight budget overflows Int")
end
Int(CUDA.free_memory()) >= minimum_free_bytes || error(
    "GPU does not have the profile's required free-memory preflight budget",
)
println(stderr, "verifying frozen Qwen3 assets…")
asset_started = time_ns()
asset_report = verify_qwen3_deployment_assets(
    model_dir,
    asset_manifest_path;
    model_id=profile.model_id,
    revision=profile.revision,
)
asset_check_seconds = (time_ns() - asset_started) / 1.0e9
Int(CUDA.free_memory()) >= minimum_free_bytes || error(
    "GPU free memory fell below the profile preflight budget during asset verification",
)

function used_bytes()
    CUDA.synchronize()
    return Int(CUDA.total_memory() - CUDA.free_memory())
end

function timed_gpu(f)
    started = time_ns()
    value = f()
    CUDA.synchronize()
    return value, (time_ns() - started) / 1.0e9
end

function reclaim_cuda_pool!(; full_gc::Bool=true)
    CUDA.synchronize()
    GC.gc(full_gc)
    CUDA.reclaim()
    return nothing
end

prefill_reclaimer = function (position)
    chunk_index = cld(position, profile.prefill_chunk_tokens)
    mod(chunk_index, profile.prefill_reclaim_interval_chunks) == 0 &&
        reclaim_cuda_pool!(; full_gc=true)
    return nothing
end

println(stderr, "loading $(profile.model_id) BF16 parameters on host…")
host_bundle, host_load_seconds = timed_gpu() do
    load_hf_qwen3_bundle(
        model_dir;
        max_seq_len=profile.context_tokens,
        weight_dtype=BFloat16,
        revision=profile.revision,
        variant=profile.variant,
    )
end
free_before_upload = Int(CUDA.free_memory())
println(stderr, "uploading BF16 parameter tree to $(CUDA.name(CUDA.device()))…")
parameters, upload_seconds = timed_gpu() do
    CUDA.cu(host_bundle.parameters)
end
gpu_parameter_bytes = free_before_upload - Int(CUDA.free_memory())
device_bundle = merge(host_bundle, (; parameters))
free_before_cache = Int(CUDA.free_memory())
session, session_init_seconds = timed_gpu() do
    init_hf_qwen3_bf16_session(
        device_bundle;
        context_tokens=profile.context_tokens,
        prefill_chunk_tokens=profile.prefill_chunk_tokens,
    )
end
gpu_cache_and_rope_bytes = free_before_cache - Int(CUDA.free_memory())
host_bundle = nothing
device_bundle = nothing
GC.gc()
CUDA.reclaim()
vram_used_ready_bytes = used_bytes()
gpu_free_ready_bytes = Int(CUDA.free_memory())
gpu_free_ready_bytes >= profile.workspace_reserve_bytes || error(
    "actual ready GPU free memory is below workspace_reserve_bytes",
)
cuda_pool_live_ready_bytes = Int(CUDA.used_memory())
cuda_pool_reserved_ready_bytes = Int(CUDA.cached_memory())
println(
    stderr,
    "session ready: free=$(round(gpu_free_ready_bytes / 2.0^30; digits=2))GiB",
)

prompt = "用三点解释 KV Cache 为什么能加速自回归解码。"
function run_request(request_index, phase; strategy, rng=Xoshiro(19))
    requested_tokens = min(profile.max_new_tokens, 32)
    generated_count = Ref(0)
    reclaim_decode = function (_)
        generated_count[] += 1
        mod(generated_count[], profile.decode_reclaim_interval_tokens) == 0 &&
            reclaim_cuda_pool!(; full_gc=false)
        return nothing
    end
    result, wall_seconds = timed_gpu() do
        generate_hf_text!(
            session,
            prompt;
            chat=true,
            enable_thinking=profile.enable_thinking,
            max_new_tokens=requested_tokens,
            max_prompt_tokens=profile.max_prompt_tokens,
            strategy,
            rng,
            stop_token_ids=Int[],
            on_token=reclaim_decode,
            on_prefill_chunk=prefill_reclaimer,
        )
    end
    cleanup_started = time_ns()
    reclaim_cuda_pool!()
    cleanup_seconds = (time_ns() - cleanup_started) / 1.0e9
    length(result.generated_ids) == requested_tokens || error(
        "fixed-length request did not generate the requested token count",
    )
    return result, Dict(
        "request" => request_index,
        "phase" => phase,
        "strategy" => String(result.strategy),
        "prompt_tokens" => length(result.prompt_ids),
        "requested_tokens" => requested_tokens,
        "generated_tokens" => length(result.generated_ids),
        "stop_reason" => String(result.stop_reason),
        "prefill_seconds" => result.prefill_seconds,
        "decode_seconds" => result.decode_seconds,
        "reported_tokens_per_second" => result.tokens_per_second,
        "wall_seconds" => wall_seconds,
        "post_request_cleanup_seconds" => cleanup_seconds,
        "vram_used_after_bytes" => used_bytes(),
        "gpu_free_after_bytes" => Int(CUDA.free_memory()),
        "generated_ids_0_based" => result.generated_ids .- 1,
    )
end

request_runs = Any[]
println(stderr, "running cold 32-token request…")
cold_result, cold_metrics = run_request(1, "cold"; strategy=:greedy)
push!(request_runs, cold_metrics)
completion = cold_result.completion

probe_piece = encode(
    session.tokenizer,
    " LifeAI context probe.";
    add_special_tokens=false,
)
isempty(probe_piece) && error("probe text encoded to no tokens")
context_runs = Any[]
for token_count in (1024, 2048, profile.max_prompt_tokens)
    println(stderr, "auditing steady $token_count-token prefill…")
    tokens = [probe_piece[mod1(index, length(probe_piece))] for index in 1:token_count]
    performance_samples = Float64[]
    for _ in 1:2
        reclaim_cuda_pool!()
        _, seconds = timed_gpu() do
            prefill_hf_qwen3_bf16!(
                session,
                tokens;
                on_chunk=prefill_reclaimer,
            )
        end
        push!(performance_samples, seconds)
    end
    _, decode_seconds = timed_gpu() do
        decode_hf_qwen3_bf16!(session, first(probe_piece))
    end

    reclaim_cuda_pool!()
    used_before = used_bytes()
    sampled_peak_used = Ref(used_before)
    sampled_peak_pool_live = Ref(Int(CUDA.used_memory()))
    sampled_peak_pool_reserved = Ref(Int(CUDA.cached_memory()))
    record_memory = function ()
        current_used = used_bytes()
        sampled_peak_used[] = max(sampled_peak_used[], current_used)
        sampled_peak_pool_live[] = max(
            sampled_peak_pool_live[],
            Int(CUDA.used_memory()),
        )
        sampled_peak_pool_reserved[] = max(
            sampled_peak_pool_reserved[],
            Int(CUDA.cached_memory()),
        )
        return nothing
    end
    sample_memory = function (position)
        record_memory()
        prefill_reclaimer(position)
        return nothing
    end
    _, audit_seconds = timed_gpu() do
        prefill_hf_qwen3_bf16!(session, tokens; on_chunk=sample_memory)
    end
    _, audit_decode_seconds = timed_gpu() do
        decode_hf_qwen3_bf16!(session, first(probe_piece))
    end
    record_memory()
    reclaim_cuda_pool!()
    push!(context_runs, Dict(
        "phase" => "steady",
        "prompt_tokens" => token_count,
        "performance_prefill_seconds_samples" => performance_samples,
        "best_prefill_seconds" => minimum(performance_samples),
        "best_prefill_tokens_per_second" =>
            token_count / minimum(performance_samples),
        "one_token_decode_seconds" => decode_seconds,
        "memory_audit_prefill_seconds" => audit_seconds,
        "memory_audit_one_token_decode_seconds" => audit_decode_seconds,
        "memory_audit_chunk_synchronization" => true,
        "cache_position_after_decode" => session.position,
        "logical_kv_bytes" => qwen3_kv_cache_bytes(session.model, token_count),
        "vram_used_before_bytes" => used_before,
        "vram_used_after_bytes" => used_bytes(),
        "gpu_free_after_bytes" => Int(CUDA.free_memory()),
        "sampled_peak_vram_used_bytes" => sampled_peak_used[],
        "sampled_minimum_gpu_free_bytes" =>
            Int(CUDA.total_memory()) - sampled_peak_used[],
        "sampled_peak_pool_live_bytes" => sampled_peak_pool_live[],
        "sampled_peak_pool_reserved_bytes" => sampled_peak_pool_reserved[],
    ))
end

capacity_prompt = [
    probe_piece[mod1(index, length(probe_piece))]
    for index in 1:profile.max_prompt_tokens
]
capacity_minimum_free = Ref(Int(CUDA.free_memory()))
capacity_generated_count = Ref(0)
capacity_on_token = function (_)
    capacity_minimum_free[] = min(
        capacity_minimum_free[],
        Int(CUDA.free_memory()),
    )
    capacity_generated_count[] += 1
    mod(
        capacity_generated_count[],
        profile.decode_reclaim_interval_tokens,
    ) == 0 && reclaim_cuda_pool!(; full_gc=false)
    return nothing
end
println(
    stderr,
    "running full budget $(profile.max_prompt_tokens)+$(profile.max_new_tokens)…",
)
capacity_result, capacity_wall_seconds = timed_gpu() do
    generate_hf_qwen3_bf16!(
        session,
        capacity_prompt;
        max_new_tokens=profile.max_new_tokens,
        strategy=:greedy,
        stop_token_ids=Int[],
        on_token=capacity_on_token,
        on_prefill_chunk=prefill_reclaimer,
    )
end
reclaim_cuda_pool!()
length(capacity_result.generated_ids) == profile.max_new_tokens || error(
    "full-context capacity run did not generate max_new_tokens",
)
expected_cached_tokens =
    profile.max_prompt_tokens + profile.max_new_tokens - 1
session.position == expected_cached_tokens || error(
    "full-context capacity run ended at an unexpected cache position",
)
capacity_run = Dict(
    "phase" => "full_budget_capacity",
    "prompt_tokens" => length(capacity_result.prompt_ids),
    "generated_tokens" => length(capacity_result.generated_ids),
    "sequence_tokens" =>
        length(capacity_result.prompt_ids) + length(capacity_result.generated_ids),
    "cached_tokens" => session.position,
    "last_selected_token_is_not_cached" => true,
    "stop_reason" => String(capacity_result.stop_reason),
    "prefill_seconds" => capacity_result.prefill_seconds,
    "decode_seconds" => capacity_result.decode_seconds,
    "tokens_per_second" => capacity_result.tokens_per_second,
    "per_token_memory_sampling" => true,
    "periodic_pool_reclaim" => true,
    "tokens_per_second_is_instrumented" => true,
    "wall_seconds" => capacity_wall_seconds,
    "minimum_gpu_free_bytes_during_decode" => capacity_minimum_free[],
    "gpu_free_after_bytes" => Int(CUDA.free_memory()),
)

println(stderr, "running steady greedy and default-strategy requests…")
steady_result, steady_metrics = run_request(2, "steady"; strategy=:greedy)
completion = steady_result.completion
push!(request_runs, steady_metrics)
sample_result, sample_metrics = run_request(
    3,
    "steady_default_strategy";
    strategy=profile.strategy,
    rng=Xoshiro(19),
)
completion = sample_result.completion
push!(request_runs, sample_metrics)

minimum_audited_gpu_free_bytes = minimum(vcat(
    [gpu_free_ready_bytes, capacity_minimum_free[]],
    [
        Int(run["sampled_minimum_gpu_free_bytes"])
        for run in context_runs
    ],
))
steady_ttft_limit_seconds = 5.0
steady_decode_minimum_tps = 3.0
long_prefill_limit_seconds = 120.0
minimum_gpu_free_limit_bytes = 2 * 1024^3
acceptance_checks = Dict(
    "steady_short_ttft_seconds" => steady_metrics["prefill_seconds"],
    "steady_short_ttft_limit_seconds" => steady_ttft_limit_seconds,
    "steady_short_ttft_passed" =>
        steady_metrics["prefill_seconds"] <= steady_ttft_limit_seconds,
    "steady_decode_tokens_per_second" =>
        steady_metrics["reported_tokens_per_second"],
    "steady_decode_minimum_tokens_per_second" => steady_decode_minimum_tps,
    "steady_decode_passed" =>
        steady_metrics["reported_tokens_per_second"] >= steady_decode_minimum_tps,
    "long_prefill_seconds" => context_runs[end]["best_prefill_seconds"],
    "long_prefill_limit_seconds" => long_prefill_limit_seconds,
    "long_prefill_passed" =>
        context_runs[end]["best_prefill_seconds"] <= long_prefill_limit_seconds,
    "minimum_audited_gpu_free_bytes" => minimum_audited_gpu_free_bytes,
    "minimum_gpu_free_limit_bytes" => minimum_gpu_free_limit_bytes,
    "minimum_gpu_free_passed" =>
        minimum_audited_gpu_free_bytes >= minimum_gpu_free_limit_bytes,
)
acceptance_checks["daily_profile_passed"] = all((
    acceptance_checks["steady_short_ttft_passed"],
    acceptance_checks["steady_decode_passed"],
    acceptance_checks["long_prefill_passed"],
    acceptance_checks["minimum_gpu_free_passed"],
))

result = Dict(
    "source" => "LifeAI Week 19 Qwen3 BF16 local deployment",
    "profile" => profile.name,
    "profile_sha256" => bytes2hex(sha256(read(profile_path))),
    "model_id" => profile.model_id,
    "revision" => profile.revision,
    "model_dir" => abspath(model_dir),
    "implementation" => Dict(
        "julia_version" => string(VERSION),
        "project_sha256" => bytes2hex(sha256(read(joinpath(
            @__DIR__,
            "..",
            "Project.toml",
        )))),
        "manifest_sha256" => bytes2hex(sha256(read(joinpath(
            @__DIR__,
            "..",
            "Manifest.toml",
        )))),
        "bf16_accel_sha256" => bytes2hex(sha256(read(joinpath(
            @__DIR__,
            "..",
            "src",
            "models",
            "bf16_accel.jl",
        )))),
        "qwen3_deployment_sha256" => bytes2hex(sha256(read(joinpath(
            @__DIR__,
            "..",
            "src",
            "generation",
            "qwen3_deployment.jl",
        )))),
        "benchmark_script_sha256" => bytes2hex(sha256(read(@__FILE__))),
    ),
    "assets" => Dict(
        "manifest" => abspath(asset_manifest_path),
        "manifest_sha256" => bytes2hex(sha256(read(asset_manifest_path))),
        "verified_files" => length(asset_report.files),
        "verified_bytes" => asset_report.total_bytes,
        "check_seconds" => asset_check_seconds,
    ),
    "gpu" => Dict(
        "name" => CUDA.name(CUDA.device()),
        "driver_version" => string(CUDA.driver_version()),
        "cuda_runtime_version" => string(CUDA.runtime_version()),
        "total_bytes" => Int(CUDA.total_memory()),
    ),
    "deployment" => Dict(
        "context_tokens" => profile.context_tokens,
        "max_prompt_tokens" => profile.max_prompt_tokens,
        "max_new_tokens" => profile.max_new_tokens,
        "prefill_chunk_tokens" => profile.prefill_chunk_tokens,
        "prefill_reclaim_interval_chunks" =>
            profile.prefill_reclaim_interval_chunks,
        "decode_reclaim_interval_tokens" =>
            profile.decode_reclaim_interval_tokens,
        "preflight_minimum_free_bytes" => minimum_free_bytes,
        "workspace_reserve_bytes" => profile.workspace_reserve_bytes,
        "logical_full_kv_bytes" => qwen3_kv_cache_bytes(
            session.model,
            profile.context_tokens,
        ),
        "host_load_seconds" => host_load_seconds,
        "gpu_upload_seconds" => upload_seconds,
        "session_init_seconds" => session_init_seconds,
        "gpu_parameter_bytes" => gpu_parameter_bytes,
        "gpu_cache_and_rope_bytes" => gpu_cache_and_rope_bytes,
        "vram_used_ready_bytes" => vram_used_ready_bytes,
        "gpu_free_ready_bytes" => gpu_free_ready_bytes,
        "cuda_pool_live_ready_bytes" => cuda_pool_live_ready_bytes,
        "cuda_pool_reserved_ready_bytes" => cuda_pool_reserved_ready_bytes,
    ),
    "context_runs" => context_runs,
    "capacity_run" => capacity_run,
    "request_runs" => request_runs,
    "acceptance_checks" => acceptance_checks,
    "sample_completion" => completion,
)

JSON3.pretty(stdout, result)
println()
if output_path !== nothing
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
end
acceptance_checks["daily_profile_passed"] || error(
    "candidate daily profile failed one or more hardware acceptance checks",
)
