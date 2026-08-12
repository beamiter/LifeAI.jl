#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using SHA: sha256
using Statistics: mean, median

length(ARGS) == 3 || error(
    "usage: julia --threads=4 --project=. " *
    "scripts/benchmark_qwen3_moe_cuda_async_miss_pipeline.jl " *
    "MODEL_DIR REFERENCE_DIR OUTPUT_JSON",
)
Threads.nthreads() > 1 || error(
    "async miss benchmark requires multiple Julia threads (use --threads=4)",
)
model_dir, reference_dir, output_path = ARGS
context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))
cache_budget_bytes = parse(
    Int,
    get(ENV, "LIFEAI_MOE_CACHE_BUDGET_BYTES", string(8 * 2^30)),
)
read_workers = parse(Int, get(ENV, "LIFEAI_MOE_READ_WORKERS", "4"))
repetitions = parse(Int, get(ENV, "LIFEAI_MOE_MISS_REPETITIONS", "3"))
read_workers > 0 || error("LIFEAI_MOE_READ_WORKERS must be positive")
repetitions > 0 || error("LIFEAI_MOE_MISS_REPETITIONS must be positive")

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)
metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
String(metadata["compute_dtype"]) == "bfloat16" || error(
    "reference must use bfloat16 compute",
)
reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))
tokens = Int.(collect(metadata["token_ids_0_based"])) .+ 1
decode_token = Int(metadata["decode_token_id_0_based"]) + 1

elapsed_seconds(started) = (time_ns() - started) / 1.0e9
function timed_cuda(f)
    started = time_ns()
    value = f()
    CUDA.synchronize()
    return value, elapsed_seconds(started)
end

function run_request(session)
    prefill, prefill_seconds = timed_cuda(() ->
        prefill_hf_qwen3_moe_offload!(session, tokens)
    )
    decode, decode_seconds = timed_cuda(() ->
        decode_hf_qwen3_moe_offload!(session, decode_token)
    )
    stats = decode.expert_cache
    report = (;
        prefill_seconds,
        decode_seconds,
        request_seconds=prefill_seconds + decode_seconds,
        bytes_read=stats.bytes_read,
        bytes_uploaded=stats.bytes_uploaded,
        cache_hits=stats.hits,
        cache_misses=stats.misses,
        cache_evictions=stats.evictions,
        miss_stage_seconds=stats.miss_stage_seconds,
        summed_host_read_seconds=stats.host_read_seconds,
        upload_wait_seconds=stats.upload_wait_seconds,
        read_tasks=stats.read_tasks,
        parallel_read_layers=stats.parallel_read_layers,
        pinned_bytes_uploaded=stats.pinned_bytes_uploaded,
        pointer_table_builds=stats.pointer_table_builds,
        pointer_table_reuses=stats.pointer_table_reuses,
        forced_gc_calls=stats.forced_gc_calls,
        gpu_free_bytes=Int(CUDA.free_memory()),
    )
    return (; prefill, decode, report)
end

configurations = (
    (;
        name="sequential_pageable",
        miss_pipeline=:sequential,
        pinned_upload=false,
    ),
    (;
        name="overlapped_pageable",
        miss_pipeline=:overlapped,
        pinned_upload=false,
    ),
    (;
        name="overlapped_pinned",
        miss_pipeline=:overlapped,
        pinned_upload=true,
    ),
)

function prepare_configuration!(session, configuration)
    configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes=cache_budget_bytes,
        policy=:global_lru,
        dispatch=:scattered,
        gc_interval_layers=8,
        miss_pipeline=configuration.miss_pipeline,
        read_workers,
        pinned_upload=configuration.pinned_upload,
    )
    GC.gc(true)
    CUDA.reclaim()
    return session
end

gpu_free_before = Int(CUDA.free_memory())
session, load_seconds = timed_cuda(() ->
    load_hf_qwen3_moe_offload_session(
        model_dir;
        context_tokens,
        prefill_chunk_tokens=128,
        grouped_experts=false,
        expert_cache_budget_bytes=cache_budget_bytes,
        expert_cache_policy=:global_lru,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=8,
        to_device=CUDA.cu,
        on_resident_layer=(layer, total) ->
            (layer == 1 || layer % 8 == 0 || layer == total) &&
                println(stderr, "resident layer $layer/$total"),
    )
)
gpu_free_ready = Int(CUDA.free_memory())

# Warm every implementation path before measurement. This also makes the
# safetensors comparison explicitly a warm OS page-cache miss-path benchmark;
# the script does not claim cold-disk throughput or drop kernel caches.
warm_outputs = Dict{String,Any}()
for configuration in configurations
    println(stderr, "warmup $(configuration.name)")
    prepare_configuration!(session, configuration)
    warm = run_request(session)
    warm_outputs[configuration.name] = (;
        prefill=warm.prefill.logits,
        decode=warm.decode.logits,
    )
end

run_reports = Dict(configuration.name => NamedTuple[] for configuration in configurations)
run_outputs = Dict(configuration.name => Any[] for configuration in configurations)
for repetition in 1:repetitions
    for configuration in configurations
        println(stderr, "measured $(configuration.name) $repetition/$repetitions")
        prepare_configuration!(session, configuration)
        current = run_request(session)
        push!(run_reports[configuration.name], current.report)
        push!(run_outputs[configuration.name], (;
            prefill=current.prefill.logits,
            decode=current.decode.logits,
        ))
    end
end

baseline = first(run_outputs["sequential_pageable"])
function configuration_report(configuration)
    reports = run_reports[configuration.name]
    outputs = run_outputs[configuration.name]
    request_seconds = [report.request_seconds for report in reports]
    miss_seconds = [report.miss_stage_seconds for report in reports]
    return (;
        name=configuration.name,
        miss_pipeline=String(configuration.miss_pipeline),
        pinned_upload=configuration.pinned_upload,
        repetitions=length(reports),
        request_seconds_median=median(request_seconds),
        request_seconds_mean=mean(request_seconds),
        request_seconds_minimum=minimum(request_seconds),
        request_seconds_maximum=maximum(request_seconds),
        miss_stage_seconds_median=median(miss_seconds),
        all_runs_exact=all(outputs) do output
            output.prefill == baseline.prefill &&
                output.decode == baseline.decode
        end,
        warmup_exact=(
            warm_outputs[configuration.name].prefill == baseline.prefill &&
            warm_outputs[configuration.name].decode == baseline.decode
        ),
        runs=Tuple(reports),
    )
end

sequential = configuration_report(configurations[1])
overlapped_pageable = configuration_report(configurations[2])
overlapped_pinned = configuration_report(configurations[3])
configuration_reports = (
    sequential,
    overlapped_pageable,
    overlapped_pinned,
)
best = configuration_reports[argmin([
    configuration.request_seconds_median
    for configuration in configuration_reports
])]

hf_layout(array) = permutedims(array, (3, 2, 1))
expected_prefill = Float32.(hf_layout(reference["logits"])[:, end, 1])
expected_decode = Float32.(vec(hf_layout(reference["decode_logits"])))
actual_prefill = Float32.(vec(baseline.prefill))
actual_decode = Float32.(vec(baseline.decode))
prefill_abs = abs.(actual_prefill .- expected_prefill)
decode_abs = abs.(actual_decode .- expected_decode)

repo_root = normpath(joinpath(@__DIR__, ".."))
source_paths = (;
    benchmark_script=relpath(@__FILE__, repo_root),
    offload_implementation=joinpath("src", "generation", "qwen3_moe_offload.jl"),
    cuda_extension=joinpath("ext", "LifeAICUDAExt.jl"),
)
source_sha256 = (; (
    name => bytes2hex(sha256(read(joinpath(repo_root, path))))
    for (name, path) in pairs(source_paths)
)...)

report = (;
    schema_version=1,
    model_id="Qwen/Qwen3-30B-A3B",
    revision=String(metadata["revision"]),
    compute_dtype="bfloat16",
    gpu=(;
        name=CUDA.name(CUDA.device()),
        capability=string(CUDA.capability(CUDA.device())),
        free_before_bytes=gpu_free_before,
        free_ready_bytes=gpu_free_ready,
    ),
    session=(;
        context_tokens,
        prompt_tokens=length(tokens),
        decode_tokens=1,
        cache_budget_bytes,
        cache_policy="global_lru",
        dispatch="scattered",
        gc_interval_layers=8,
        read_workers,
        julia_threads=Threads.nthreads(),
        load_seconds,
    ),
    methodology=(;
        page_cache="warmed by one complete request per configuration",
        kernel_cache_dropped=false,
        configuration_order="sequential, overlapped-pageable, overlapped-pinned per repetition",
        router_dependency="only current-layer post-router misses are staged",
        cache_state="cleared before every request",
    ),
    configurations=configuration_reports,
    comparisons=(;
        overlapped_pageable_over_sequential=
            sequential.request_seconds_median /
                overlapped_pageable.request_seconds_median,
        overlapped_pinned_over_sequential=
            sequential.request_seconds_median /
                overlapped_pinned.request_seconds_median,
        pinned_over_pageable_overlap=
            overlapped_pageable.request_seconds_median /
                overlapped_pinned.request_seconds_median,
    ),
    correctness=(;
        all_configurations_exact=all(
            configuration -> configuration.all_runs_exact &&
                configuration.warmup_exact,
            configuration_reports,
        ),
        prefill_logits_max_abs=maximum(prefill_abs),
        prefill_logits_mean_abs=mean(prefill_abs),
        prefill_argmax_match=argmax(actual_prefill) == argmax(expected_prefill),
        decode_logits_max_abs=maximum(decode_abs),
        decode_logits_mean_abs=mean(decode_abs),
        decode_argmax_match=argmax(actual_decode) == argmax(expected_decode),
    ),
    decision=(;
        fastest_configuration=best.name,
        enable_by_default=false,
        layer_ahead_prefetch_implemented=false,
        bounded_current_layer_staging=true,
        cold_disk_speedup_claimed=false,
    ),
    source_paths,
    source_sha256,
)

mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(JSON3.write(report))
