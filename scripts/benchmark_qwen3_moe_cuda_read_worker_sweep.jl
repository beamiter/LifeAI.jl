#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using SHA: sha256
using Statistics: mean, median

length(ARGS) == 2 || error(
    "usage: julia --threads=8 --project=. " *
    "scripts/benchmark_qwen3_moe_cuda_read_worker_sweep.jl " *
    "MODEL_DIR OUTPUT_JSON",
)
Threads.nthreads() > 1 || error(
    "read-worker sweep requires multiple Julia threads",
)
model_dir, output_path = ARGS
context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))
workload_tokens = parse(Int, get(ENV, "LIFEAI_MOE_WORKLOAD_TOKENS", "32"))
cache_budget_bytes = parse(
    Int,
    get(ENV, "LIFEAI_MOE_CACHE_BUDGET_BYTES", string(4 * 2^30)),
)
repetitions = parse(Int, get(ENV, "LIFEAI_MOE_WORKER_REPETITIONS", "2"))
worker_counts = parse.(
    Int,
    split(get(ENV, "LIFEAI_MOE_WORKER_COUNTS", "1,2,4,8"), ','),
)
workload_tokens > 1 || error("LIFEAI_MOE_WORKLOAD_TOKENS must exceed one")
cache_budget_bytes > 0 || error("LIFEAI_MOE_CACHE_BUDGET_BYTES must be positive")
repetitions > 0 || error("LIFEAI_MOE_WORKER_REPETITIONS must be positive")
isempty(worker_counts) && error("LIFEAI_MOE_WORKER_COUNTS must not be empty")
all(>(0), worker_counts) || error("worker counts must be positive")
length(unique(worker_counts)) == length(worker_counts) || error(
    "worker counts must be unique",
)

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)

spec = qwen3_moe_checkpoint_spec()
tokenizer = load_hf_qwen3_tokenizer(model_dir; revision=spec.revision)
workload_text =
    "Explain how sparse mixture-of-experts models increase parameter " *
    "capacity while keeping per-token computation manageable, and " *
    "describe why routing locality matters for GPU inference. Focus on " *
    "memory traffic and cache behavior."
workload_ids = encode(tokenizer, workload_text; add_special_tokens=false)
length(workload_ids) >= workload_tokens || error(
    "natural-text workload has only $(length(workload_ids)) tokens; " *
    "need $workload_tokens",
)
tokens = workload_ids[1:workload_tokens]

sha256_string(value) = bytes2hex(sha256(codeunits(value)))
token_sha256(values) = sha256_string(join(values, ','))
elapsed_seconds(started) = (time_ns() - started) / 1.0e9
function timed_cuda(f)
    started = time_ns()
    value = f()
    CUDA.synchronize()
    return value, elapsed_seconds(started)
end

function process_io()
    values = Dict{String,Int}()
    for line in eachline("/proc/self/io")
        key, value = split(line, ':'; limit=2)
        values[key] = parse(Int, strip(value))
    end
    return (;
        rchar=values["rchar"],
        syscr=values["syscr"],
        read_bytes=values["read_bytes"],
    )
end

function checkpoint_shards(path)
    return sort!(filter(
        file -> endswith(file, ".safetensors"),
        readdir(path; join=true),
    ))
end

function evict_checkpoint_page_cache!(shards)
    total_bytes = 0
    for path in shards
        total_bytes = Base.checked_add(total_bytes, filesize(path))
        result = open(path, "r") do io
            ccall(
                :posix_fadvise,
                Cint,
                (Cint, Int64, Int64, Cint),
                Base.fd(io),
                0,
                0,
                4, # POSIX_FADV_DONTNEED on Linux.
            )
        end
        result == 0 || error("posix_fadvise failed for $path: $result")
    end
    return total_bytes
end

function configure_worker!(session, workers)
    configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes=cache_budget_bytes,
        policy=:layer_balanced_lru,
        dispatch=:scattered,
        gc_interval_layers=8,
        miss_pipeline=:overlapped,
        read_workers=workers,
        pinned_upload=false,
    )
    GC.gc(true)
    CUDA.reclaim()
    return session
end

function run_request(session, workers, page_cache_phase, repetition)
    io_before = process_io()
    prefill, prefill_seconds = timed_cuda(() ->
        prefill_hf_qwen3_moe_offload!(session, tokens)
    )
    prefill_logits = Float32.(vec(prefill.logits))
    decode_token = argmax(prefill_logits)
    decode, decode_seconds = timed_cuda(() ->
        decode_hf_qwen3_moe_offload!(session, decode_token)
    )
    decode_logits = Float32.(vec(decode.logits))
    io_after = process_io()
    stats = decode.expert_cache
    return (;
        output=(; prefill_logits, decode_logits, decode_token),
        report=(;
            workers,
            page_cache_phase,
            repetition,
            prefill_seconds,
            decode_seconds,
            request_seconds=prefill_seconds + decode_seconds,
            cache_hits=stats.hits,
            cache_misses=stats.misses,
            cache_evictions=stats.evictions,
            expert_bytes_read=stats.bytes_read,
            expert_bytes_uploaded=stats.bytes_uploaded,
            miss_stage_seconds=stats.miss_stage_seconds,
            summed_host_read_seconds=stats.host_read_seconds,
            upload_wait_seconds=stats.upload_wait_seconds,
            read_jobs=stats.read_tasks,
            parallel_read_layers=stats.parallel_read_layers,
            process_rchar=io_after.rchar - io_before.rchar,
            process_read_syscalls=io_after.syscr - io_before.syscr,
            process_storage_read_bytes=
                io_after.read_bytes - io_before.read_bytes,
            gpu_free_bytes=Int(CUDA.free_memory()),
        ),
    )
end

shards = checkpoint_shards(model_dir)
length(shards) == 16 || error(
    "expected 16 Qwen3-30B-A3B safetensors shards, found $(length(shards))",
)
checkpoint_bytes = sum(filesize, shards)
gpu_free_before = Int(CUDA.free_memory())
session, load_seconds = timed_cuda(() ->
    load_hf_qwen3_moe_offload_session(
        model_dir;
        context_tokens,
        prefill_chunk_tokens=128,
        grouped_experts=false,
        expert_cache_budget_bytes=cache_budget_bytes,
        expert_cache_policy=:layer_balanced_lru,
        expert_cache_dispatch=:scattered,
        expert_gc_interval_layers=8,
        expert_miss_pipeline=:overlapped,
        expert_read_workers=first(worker_counts),
        to_device=CUDA.cu,
        on_resident_layer=(layer, total) ->
            (layer == 1 || layer % 8 == 0 || layer == total) &&
                println(stderr, "resident layer $layer/$total"),
    )
)
gpu_free_ready = Int(CUDA.free_memory())

# Compile with two tokens, then discard both device cache and checkpoint pages.
println(stderr, "compile warmup")
prefill_hf_qwen3_moe_offload!(session, tokens[1:2])
CUDA.synchronize()
configure_worker!(session, first(worker_counts))
evicted_checkpoint_bytes = evict_checkpoint_page_cache!(shards)

runs = Dict(worker => NamedTuple[] for worker in worker_counts)
baseline = Ref{Any}(nothing)
all_outputs_exact = Ref(true)
orders = [
    isodd(repetition) ? worker_counts : reverse(worker_counts)
    for repetition in 1:repetitions
]
for repetition in 1:repetitions
    for workers in orders[repetition]
        println(stderr, "cold workers=$workers repetition=$repetition")
        configure_worker!(session, workers)
        evict_checkpoint_page_cache!(shards)
        cold = run_request(session, workers, "fadvise_cold", repetition)

        println(stderr, "revisit workers=$workers repetition=$repetition")
        configure_worker!(session, workers)
        revisit = run_request(
            session,
            workers,
            "post_cold_revisit",
            repetition,
        )
        if baseline[] === nothing
            baseline[] = cold.output
        end
        all_outputs_exact[] &=
            cold.output.prefill_logits == baseline[].prefill_logits &&
            cold.output.decode_logits == baseline[].decode_logits &&
            cold.output.decode_token == baseline[].decode_token &&
            revisit.output.prefill_logits == baseline[].prefill_logits &&
            revisit.output.decode_logits == baseline[].decode_logits &&
            revisit.output.decode_token == baseline[].decode_token
        push!(runs[workers], cold.report)
        push!(runs[workers], revisit.report)
    end
end

function phase_summary(worker_runs, phase)
    selected = filter(run -> run.page_cache_phase == phase, worker_runs)
    request_seconds = [run.request_seconds for run in selected]
    storage_bytes = [run.process_storage_read_bytes for run in selected]
    return (;
        repetitions=length(selected),
        request_seconds_median=median(request_seconds),
        request_seconds_mean=mean(request_seconds),
        request_seconds_minimum=minimum(request_seconds),
        request_seconds_maximum=maximum(request_seconds),
        storage_read_bytes_median=median(storage_bytes),
        storage_read_bytes_minimum=minimum(storage_bytes),
        storage_read_bytes_maximum=maximum(storage_bytes),
        runs=Tuple(selected),
    )
end

configuration_reports = map(worker_counts) do workers
    worker_runs = runs[workers]
    return (;
        workers,
        cold=phase_summary(worker_runs, "fadvise_cold"),
        revisit=phase_summary(worker_runs, "post_cold_revisit"),
    )
end
baseline_configuration = only(filter(
    configuration -> configuration.workers == first(worker_counts),
    configuration_reports,
))
fastest_cold = configuration_reports[argmin([
    configuration.cold.request_seconds_median
    for configuration in configuration_reports
])]
fastest_revisit = configuration_reports[argmin([
    configuration.revisit.request_seconds_median
    for configuration in configuration_reports
])]

repo_root = normpath(joinpath(@__DIR__, ".."))
source_paths = (;
    benchmark_script=relpath(@__FILE__, repo_root),
    offload_implementation=joinpath("src", "generation", "qwen3_moe_offload.jl"),
)
source_sha256 = (; (
    name => bytes2hex(sha256(read(joinpath(repo_root, path))))
    for (name, path) in pairs(source_paths)
)...)

report = (;
    schema_version=1,
    model_id=spec.model_id,
    revision=spec.revision,
    compute_dtype="bfloat16",
    gpu=(;
        name=CUDA.name(CUDA.device()),
        capability=string(CUDA.capability(CUDA.device())),
        free_before_bytes=gpu_free_before,
        free_ready_bytes=gpu_free_ready,
    ),
    session=(;
        context_tokens,
        prefill_chunk_tokens=128,
        cache_budget_bytes,
        cache_policy="layer_balanced_lru",
        dispatch="scattered",
        gc_interval_layers=8,
        julia_threads=Threads.nthreads(),
        load_seconds,
    ),
    workload=(;
        trace="english_32 plus one greedy decode",
        tokens_per_prompt=workload_tokens,
        decode_tokens=1,
        text_sha256=sha256_string(workload_text),
        tokens_sha256=token_sha256(tokens),
        tokenizer_sha256=tokenizer.tokenizer_sha256,
    ),
    page_cache_control=(;
        mechanism="Linux posix_fadvise(POSIX_FADV_DONTNEED) on all checkpoint shards",
        checkpoint_shards=length(shards),
        checkpoint_bytes,
        advised_bytes=evicted_checkpoint_bytes,
        verification="per-request /proc/self/io rchar and read_bytes deltas",
        cold_guaranteed=false,
    ),
    methodology=(;
        repetitions,
        worker_counts=Tuple(worker_counts),
        orders=Tuple(Tuple(order) for order in orders),
        device_cache="cleared before every request",
        allocator="GC.gc(true) plus CUDA.reclaim before every request",
        pinned_upload=false,
    ),
    configurations=Tuple(configuration_reports),
    comparisons=Tuple((;
        workers=configuration.workers,
        cold_speedup_over_first=
            baseline_configuration.cold.request_seconds_median /
                configuration.cold.request_seconds_median,
        revisit_speedup_over_first=
            baseline_configuration.revisit.request_seconds_median /
                configuration.revisit.request_seconds_median,
    ) for configuration in configuration_reports),
    correctness=(; all_outputs_exact=all_outputs_exact[]),
    decision=(;
        fastest_cold_workers=fastest_cold.workers,
        fastest_revisit_workers=fastest_revisit.workers,
        default_read_workers="min(8, Threads.nthreads())",
        default_read_workers_changed=true,
        arbitrary_storage_or_thread_count_generalized=false,
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
