#!/usr/bin/env julia

# Reactant reads these settings during module initialization. Keep them before
# every `using LifeAI` / `using Reactant`; callers may override either value.
ENV["XLA_REACTANT_GPU_MEM_FRACTION"] =
    get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.87")
ENV["XLA_REACTANT_GPU_PREALLOCATE"] =
    get(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using JSON3
using LifeAI
using Reactant
using Dates
using SHA: sha256
using Statistics: median

# This script closes one frozen Week20 experiment. A caller-supplied oracle is
# not sufficient: its exact digest is pinned after the independent CUDA export.
const WEEK20_CUDA_REFERENCE_SHA256 =
    "83f62afbbb470b695b6990a3b86a8860407a37874354d6b039e1ce19917e2747"

length(ARGS) in (3, 4) || error(
    "usage: julia --project=. scripts/benchmark_qwen3_xla_deployment.jl " *
    "MODEL_DIR PROFILE_JSON CUDA_REFERENCE_JSON [OUTPUT_JSON]",
)
model_dir, profile_path, reference_path = ARGS[1:3]
output_path = length(ARGS) == 4 ?
    ARGS[4] :
    joinpath(
        @__DIR__,
        "..",
        "benchmark_results",
        "week20",
        "qwen3_8b_4090d_bf16_xla_daily.json",
    )
profile = load_qwen3_deployment_profile(profile_path)
profile.variant === :qwen3_8b || error("XLA deployment requires qwen3_8b")
profile.strategy === :greedy || error(
    "the Week20 XLA benchmark requires an explicit greedy profile",
)
profile_sha256 = bytes2hex(sha256(read(profile_path)))
asset_manifest_path = isabspath(profile.asset_manifest) ?
    profile.asset_manifest :
    joinpath(dirname(abspath(profile_path)), profile.asset_manifest)
asset_manifest_sha256 = bytes2hex(sha256(read(asset_manifest_path)))
reference_sha256 = bytes2hex(sha256(read(reference_path)))
reference_sha256 == WEEK20_CUDA_REFERENCE_SHA256 || error(
    "CUDA reference SHA256 is not the frozen Week20 oracle",
)
reference = JSON3.read(read(reference_path, String))
Int(reference["schema_version"]) == 2 ||
    error("CUDA reference schema_version must be 2")
String(reference["source"]) ==
    "LifeAI eager BF16 CUDA parity reference" ||
    error("CUDA reference source is not the independent eager path")
String(reference["model_id"]) == profile.model_id ||
    error("CUDA reference model_id does not match profile")
String(reference["revision"]) == profile.revision ||
    error("CUDA reference revision does not match profile")
String(reference["profile_sha256"]) == profile_sha256 ||
    error("CUDA reference profile SHA256 does not match")
String(reference["asset_manifest_sha256"]) == asset_manifest_sha256 ||
    error("CUDA reference asset manifest SHA256 does not match")
Int(reference["verified_asset_files"]) == 10 ||
    error("CUDA reference must cover all 10 frozen asset files")

reference_cases = Dict(
    String(case["name"]) => case
    for case in reference["cases"]
)
Set(keys(reference_cases)) == Set((
    "single_chunk_64",
    "left_padded_65",
    "full_prompt_3584",
)) || error("CUDA reference does not contain the exact Week20 parity cases")
for (name, prompt_tokens) in (
    "single_chunk_64" => 64,
    "left_padded_65" => 65,
    "full_prompt_3584" => profile.max_prompt_tokens,
)
    case = reference_cases[name]
    Int(case["prompt_tokens"]) == prompt_tokens ||
        error("$name prompt length is not frozen")
    Int(case["generated_tokens"]) == 32 ||
        error("$name must contain exactly 32 generated tokens")
    length(case["prompt_ids_0_based"]) == prompt_tokens ||
        error("$name prompt payload length is invalid")
    length(case["generated_ids_0_based"]) == 32 ||
        error("$name generated payload length is invalid")
end

println(stderr, "verifying frozen Qwen3-8B assets…")
asset_started = time_ns()
assets = verify_qwen3_deployment_assets(
    model_dir,
    asset_manifest_path;
    model_id=profile.model_id,
    revision=profile.revision,
)
asset_check_seconds = (time_ns() - asset_started) / 1.0e9

function physical_gpu_memory()
    output = strip(read(
        `nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits`,
        String,
    ))
    fields = strip.(split(output, ','))
    length(fields) == 4 || error("unexpected nvidia-smi memory output")
    return (;
        device_index=0,
        name=fields[1],
        total_bytes=parse(Int, fields[2]) * 1024^2,
        used_bytes=parse(Int, fields[3]) * 1024^2,
        free_bytes=parse(Int, fields[4]) * 1024^2,
    )
end

physical_before = physical_gpu_memory()
physical_before.total_bytes >= profile.minimum_gpu_bytes || error(
    "GPU 0 does not meet profile.minimum_gpu_bytes",
)
physical_trace_path = endswith(lowercase(output_path), ".json") ?
    output_path[1:(end - 5)] * "_nvidia_smi.csv" :
    output_path * "_nvidia_smi.csv"
mkpath(dirname(abspath(physical_trace_path)))
physical_monitor = run(
    `nvidia-smi --id=0 --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader,nounits --loop-ms=200 --filename=$(abspath(physical_trace_path))`;
    wait=false,
)
physical_monitor_running = Ref(true)
function stop_physical_monitor()
    physical_monitor_running[] || return
    try
        kill(physical_monitor, Base.SIGINT)
    catch
    end
    try
        wait(physical_monitor)
    catch
    end
    physical_monitor_running[] = false
    return
end
atexit(stop_physical_monitor)
sleep(0.25)

Reactant.set_default_backend("gpu")
println(
    stderr,
    "loading one compact XLA parameter tree and compiling 4K kernels…",
)
ready_started = time_ns()
session = load_hf_qwen3_bf16_xla_session(
    model_dir;
    context_tokens=profile.context_tokens,
    prefill_chunk_tokens=profile.prefill_chunk_tokens,
    strategy=:greedy,
    revision=profile.revision,
    variant=profile.variant,
)
ready_seconds = (time_ns() - ready_started) / 1.0e9
session.load_metrics.parameter_tensor_count == 291 ||
    error("Qwen3-8B compact tree must contain exactly 291 tensor leaves")
session.load_metrics.device_parameter_tree_count == 1 ||
    error("XLA session does not report one device parameter tree")
session.load_metrics.device_parameter_tree_transfer_count == 1 ||
    error("XLA session did not use exactly one parameter-tree transfer")

probe_piece = encode(
    session.tokenizer,
    " LifeAI XLA reference probe.";
    add_special_tokens=false,
)
isempty(probe_piece) && error("XLA probe encoded to no tokens")

reference_prompt(count) = [
    probe_piece[mod1(index, length(probe_piece))]
    for index in 1:count
]

function frozen_case(name)
    case = reference_cases[name]
    prompt = Int.(case["prompt_ids_0_based"]) .+ 1
    generated = Int.(case["generated_ids_0_based"]) .+ 1
    expected_prompt = reference_prompt(Int(case["prompt_tokens"]))
    prompt == expected_prompt ||
        error("$name prompt differs from the frozen tokenizer probe")
    all(1 <= id <= session.model.vocab_size for id in prompt) ||
        error("$name contains a prompt token outside the vocabulary")
    all(1 <= id <= session.model.vocab_size for id in generated) ||
        error("$name contains a generated token outside the vocabulary")
    return (; prompt, generated)
end

physical_ready = physical_gpu_memory()

function run_request(request_index, prompt_ids, output_tokens; phase)
    started = time_ns()
    result = generate_hf_qwen3_bf16_xla!(
        session,
        prompt_ids;
        max_new_tokens=output_tokens,
        stop_token_ids=Int[],
    )
    wall_seconds = (time_ns() - started) / 1.0e9
    length(result.generated_ids) == output_tokens ||
        error("$phase did not generate the requested token count")
    physical_after = physical_gpu_memory()
    return result, (;
        request=request_index,
        phase,
        prompt_tokens=length(prompt_ids),
        prompt_bucket_tokens=result.window_plan.prompt_bucket_tokens,
        requested_tokens=output_tokens,
        generated_tokens=length(result.generated_ids),
        cache_tokens=session.position,
        prefill_seconds=result.prefill_seconds,
        decode_seconds=result.decode_seconds,
        decode_tokens_per_second=result.tokens_per_second,
        wall_seconds,
        allocator_before=result.allocator_before,
        allocator_after=result.allocator_after,
        physical_after,
        generated_ids_0_based=result.generated_ids .- 1,
    )
end

single_chunk_reference = frozen_case("single_chunk_64")
short_prompt = single_chunk_reference.prompt
println(stderr, "running eight reusable-session requests…")
short_runs = Any[]
short_results = Any[]
for index in 1:8
    result, metrics = run_request(
        index,
        short_prompt,
        32;
        phase=index == 1 ? "cold" : "steady",
    )
    push!(short_results, result)
    push!(short_runs, metrics)
end
all(
    result -> result.generated_ids == first(short_results).generated_ids,
    short_results,
) || error("compiled XLA session is not deterministic across requests")
single_chunk_match_tokens = count(
    identity,
    first(short_results).generated_ids .==
        single_chunk_reference.generated,
)
single_chunk_match_tokens == 32 ||
    error("single-chunk XLA output does not match CUDA BF16 for 32 tokens")

left_padded_reference = frozen_case("left_padded_65")
println(stderr, "running 65-token left-padding parity request…")
padding_result, padding_run = run_request(
    9,
    left_padded_reference.prompt,
    32;
    phase="left_padded_parity",
)
padding_result.window_plan.left_padding_tokens == 63 ||
    error("65-token parity request did not exercise left padding")
left_padded_match_tokens = count(
    identity,
    padding_result.generated_ids .== left_padded_reference.generated,
)
left_padded_match_tokens == 32 ||
    error("left-padded XLA output does not match CUDA BF16 for 32 tokens")

full_prompt_reference = frozen_case("full_prompt_3584")
capacity_prompt = full_prompt_reference.prompt
println(
    stderr,
    "running full $(profile.max_prompt_tokens)+$(profile.max_new_tokens) window…",
)
capacity_result, capacity_run = run_request(
    10,
    capacity_prompt,
    profile.max_new_tokens;
    phase="full_budget_capacity",
)
session.position == profile.context_tokens - 1 ||
    error("full-window XLA cache position must end at context_tokens - 1")
capacity_result.window_plan.sequence_tokens == profile.context_tokens ||
    error("full-window logical sequence does not equal context_tokens")
full_prompt_match_tokens = count(
    identity,
    capacity_result.generated_ids[1:32] .==
        full_prompt_reference.generated,
)
full_prompt_match_tokens == 32 ||
    error("full-prompt XLA output does not match CUDA BF16 for 32 tokens")
parity_cases = [
    (;
        name="single_chunk_64",
        matched_tokens=single_chunk_match_tokens,
        expected_tokens=32,
        passed=single_chunk_match_tokens == 32,
    ),
    (;
        name="left_padded_65",
        matched_tokens=left_padded_match_tokens,
        expected_tokens=32,
        passed=left_padded_match_tokens == 32,
    ),
    (;
        name="full_prompt_3584",
        matched_tokens=full_prompt_match_tokens,
        expected_tokens=32,
        passed=full_prompt_match_tokens == 32,
    ),
]
parity_match_tokens = sum(case.matched_tokens for case in parity_cases)
parity_expected_tokens = sum(case.expected_tokens for case in parity_cases)

short_end_bytes = [
    run.allocator_after === nothing ?
        nothing : run.allocator_after.bytes_in_use
    for run in short_runs
]
memory_drift_bytes = any(isnothing, short_end_bytes) ?
    nothing : maximum(short_end_bytes) - minimum(short_end_bytes)
capacity_allocator = capacity_run.allocator_after
minimum_free_estimate = capacity_allocator === nothing ||
        capacity_allocator.bytes_limit === nothing ?
    nothing :
    capacity_allocator.bytes_limit - capacity_allocator.peak_bytes_in_use
snapshot_minimum_physical_free = minimum(vcat(
    [physical_before.free_bytes],
    [physical_ready.free_bytes],
    [run.physical_after.free_bytes for run in short_runs],
    [padding_run.physical_after.free_bytes],
    [capacity_run.physical_after.free_bytes],
))

stop_physical_monitor()
function physical_trace_summary(path)
    samples = Tuple{Int,Int}[]
    for line in eachline(path)
        fields = strip.(split(line, ','))
        length(fields) == 3 || error(
            "unexpected nvidia-smi trace row: $line",
        )
        push!(samples, (
            parse(Int, fields[2]) * 1024^2,
            parse(Int, fields[3]) * 1024^2,
        ))
    end
    isempty(samples) && error("nvidia-smi physical trace is empty")
    return (;
        device_index=0,
        sample_interval_ms=200,
        samples=length(samples),
        maximum_used_bytes=maximum(first, samples),
        minimum_free_bytes=minimum(last, samples),
        path=abspath(path),
        sha256=bytes2hex(sha256(read(path))),
    )
end
physical_trace = physical_trace_summary(physical_trace_path)
minimum_physical_free = physical_trace.minimum_free_bytes

steady_runs = short_runs[2:end]
steady_prefill_seconds =
    median([run.prefill_seconds for run in steady_runs])
steady_decode_tps =
    median([run.decode_tokens_per_second for run in steady_runs])
long_prefill_seconds = capacity_run.prefill_seconds
long_decode_tps = capacity_run.decode_tokens_per_second
parameter_allocator = session.load_metrics.allocator_after_parameter_transfer
parameter_residency_bytes = parameter_allocator === nothing ?
    nothing : parameter_allocator.bytes_in_use
parameter_residency_limit =
    session.load_metrics.parameter_logical_bytes + 1024^3
acceptance = (;
    gpu_minimum_bytes=profile.minimum_gpu_bytes,
    gpu_total_bytes=physical_before.total_bytes,
    gpu_capacity_passed=
        physical_before.total_bytes >= profile.minimum_gpu_bytes,
    compact_parameter_tensor_count_passed=
        session.load_metrics.parameter_tensor_count == 291,
    single_device_parameter_tree_passed=
        session.load_metrics.device_parameter_tree_count == 1 &&
        session.load_metrics.device_parameter_tree_transfer_count == 1 &&
        !session.load_metrics.original_parameter_tree_transferred &&
        !session.load_metrics.separate_packed_projection_tree_transferred,
    parameter_residency_bytes,
    parameter_residency_limit_bytes=parameter_residency_limit,
    parameter_residency_passed=
        parameter_residency_bytes !== nothing &&
        parameter_residency_bytes <= parameter_residency_limit,
    cuda_bf16_parity_cases=parity_cases,
    cuda_bf16_parity_match_tokens=parity_match_tokens,
    cuda_bf16_parity_expected_tokens=parity_expected_tokens,
    cuda_bf16_parity_passed=
        parity_match_tokens == parity_expected_tokens,
    full_window_sequence_tokens=
        capacity_result.window_plan.sequence_tokens,
    full_window_cache_tokens=session.position,
    full_window_passed=
        capacity_result.window_plan.sequence_tokens == 4096 &&
        session.position == 4095,
    long_prefill_seconds,
    long_prefill_limit_seconds=20.0,
    long_prefill_passed=long_prefill_seconds <= 20.0,
    long_decode_tokens_per_second=long_decode_tps,
    long_decode_minimum_tokens_per_second=25.0,
    long_decode_passed=long_decode_tps >= 25.0,
    reusable_request_allocator_drift_bytes=memory_drift_bytes,
    reusable_request_allocator_drift_limit_bytes=256 * 1024^2,
    reusable_request_memory_passed=
        memory_drift_bytes !== nothing &&
        memory_drift_bytes <= 256 * 1024^2,
    allocator_minimum_free_estimate_bytes=minimum_free_estimate,
    allocator_minimum_free_limit_bytes=profile.workspace_reserve_bytes,
    allocator_minimum_free_passed=
        minimum_free_estimate !== nothing &&
        minimum_free_estimate >= profile.workspace_reserve_bytes,
    physical_snapshot_minimum_free_bytes=
        snapshot_minimum_physical_free,
    physical_minimum_free_bytes=minimum_physical_free,
    physical_minimum_free_limit_bytes=profile.workspace_reserve_bytes,
    physical_minimum_free_passed=
        minimum_physical_free >= profile.workspace_reserve_bytes,
)
closed = all((
    acceptance.gpu_capacity_passed,
    acceptance.compact_parameter_tensor_count_passed,
    acceptance.single_device_parameter_tree_passed,
    acceptance.parameter_residency_passed,
    acceptance.cuda_bf16_parity_passed,
    acceptance.full_window_passed,
    acceptance.long_prefill_passed,
    acceptance.long_decode_passed,
    acceptance.reusable_request_memory_passed,
    acceptance.allocator_minimum_free_passed,
    acceptance.physical_minimum_free_passed,
))

report = (;
    schema_version=1,
    source="LifeAI Week 20 Qwen3 BF16 XLA single-residency deployment",
    recorded_at=string(Dates.now()),
    model_id=profile.model_id,
    revision=profile.revision,
    model_dir=abspath(model_dir),
    profile=profile.name,
    profile_sha256,
    reference_path=abspath(reference_path),
    reference_sha256,
    assets=(;
        verified_files=length(assets.files),
        verified_bytes=assets.total_bytes,
        check_seconds=asset_check_seconds,
        manifest_path=abspath(asset_manifest_path),
        manifest_sha256=asset_manifest_sha256,
    ),
    xla=(;
        reactant_version=string(pkgversion(Reactant)),
        gpu_memory_fraction=ENV["XLA_REACTANT_GPU_MEM_FRACTION"],
        gpu_preallocate=ENV["XLA_REACTANT_GPU_PREALLOCATE"],
        persistent_cache_directory=
            Reactant.PersistentCompileCache.CACHE_DIR[],
        persistent_kernel_cache_enabled=
            Reactant.PersistentCompileCache.kernel_cache_enabled(),
        persistent_autotune_cache_enabled=
            Reactant.PersistentCompileCache.autotune_cache_enabled(),
    ),
    deployment=(;
        context_tokens=profile.context_tokens,
        max_prompt_tokens=profile.max_prompt_tokens,
        max_new_tokens=profile.max_new_tokens,
        prefill_chunk_tokens=profile.prefill_chunk_tokens,
        logical_kv_bytes=qwen3_kv_cache_bytes(
            session.model,
            profile.context_tokens,
        ),
        ready_seconds,
        load_metrics=session.load_metrics,
        physical_before,
        physical_ready,
    ),
    short_runs,
    padding_run,
    capacity_run,
    physical_trace,
    steady=(;
        sample_count=length(steady_runs),
        statistic="median",
        prefill_seconds=steady_prefill_seconds,
        decode_tokens_per_second=steady_decode_tps,
    ),
    acceptance,
    closed,
)
mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(stderr, "wrote $(abspath(output_path)); closed=$closed")
closed || error(
    "Week20 acceptance failed; inspect $(abspath(output_path))",
)
