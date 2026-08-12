#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using SHA: sha256
using Statistics: mean, median

length(ARGS) == 2 || error(
    "usage: julia --threads=8 --project=. " *
    "scripts/benchmark_qwen3_moe_cuda_coalesced_reads.jl " *
    "MODEL_DIR OUTPUT_JSON",
)
Threads.nthreads() >= 8 || error(
    "coalesced-read benchmark requires at least eight Julia threads",
)
model_dir, output_path = ARGS
context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))
workload_tokens = parse(Int, get(ENV, "LIFEAI_MOE_WORKLOAD_TOKENS", "32"))
cache_budget_bytes = parse(
    Int,
    get(ENV, "LIFEAI_MOE_CACHE_BUDGET_BYTES", string(4 * 2^30)),
)
repetitions = parse(Int, get(ENV, "LIFEAI_MOE_COALESCED_REPETITIONS", "3"))
repetitions > 0 || error("LIFEAI_MOE_COALESCED_REPETITIONS must be positive")

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

checkpoint_shards(path) = sort!(filter(
    file -> endswith(file, ".safetensors"),
    readdir(path; join=true),
))
function evict_checkpoint_page_cache!(shards)
    for path in shards
        result = open(path, "r") do io
            ccall(
                :posix_fadvise,
                Cint,
                (Cint, Int64, Int64, Cint),
                Base.fd(io),
                0,
                0,
                4,
            )
        end
        result == 0 || error("posix_fadvise failed for $path: $result")
    end
    return nothing
end

function configure_mode!(session, mode)
    configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes=cache_budget_bytes,
        policy=:layer_balanced_lru,
        dispatch=:scattered,
        gc_interval_layers=8,
        read_mode=mode,
        miss_pipeline=:overlapped,
        read_workers=8,
        pinned_upload=false,
    )
    GC.gc(true)
    CUDA.reclaim()
    return session
end

function run_request(session, mode, page_cache_phase, repetition)
    io_before = process_io()
    gc_before = Base.gc_num()
    prefill, prefill_seconds = timed_cuda(() ->
        prefill_hf_qwen3_moe_offload!(session, tokens)
    )
    prefill_logits = Float32.(vec(prefill.logits))
    decode_token = argmax(prefill_logits)
    decode, decode_seconds = timed_cuda(() ->
        decode_hf_qwen3_moe_offload!(session, decode_token)
    )
    decode_logits = Float32.(vec(decode.logits))
    gc_after = Base.gc_num()
    io_after = process_io()
    stats = decode.expert_cache
    return (;
        output=(; prefill_logits, decode_logits, decode_token),
        report=(;
            mode=String(mode),
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
            read_jobs=stats.read_tasks,
            process_rchar=io_after.rchar - io_before.rchar,
            process_read_syscalls=io_after.syscr - io_before.syscr,
            process_storage_read_bytes=
                io_after.read_bytes - io_before.read_bytes,
            julia_allocated_bytes=
                Int(gc_after.total_allocd - gc_before.total_allocd),
            julia_gc_time_ns=Int(gc_after.total_time - gc_before.total_time),
            julia_gc_collections=Int(gc_after.collect - gc_before.collect),
            gpu_free_bytes=Int(CUDA.free_memory()),
        ),
    )
end

shards = checkpoint_shards(model_dir)
length(shards) == 16 || error(
    "expected 16 Qwen3-30B-A3B safetensors shards, found $(length(shards))",
)
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
        expert_read_mode=:tensor,
        expert_miss_pipeline=:overlapped,
        expert_read_workers=8,
        to_device=CUDA.cu,
        on_resident_layer=(layer, total) ->
            (layer == 1 || layer % 8 == 0 || layer == total) &&
                println(stderr, "resident layer $layer/$total"),
    )
)
gpu_free_ready = Int(CUDA.free_memory())

println(stderr, "compile warmup")
prefill_hf_qwen3_moe_offload!(session, tokens[1:2])
CUDA.synchronize()
configure_mode!(session, :coalesced)
prefill_hf_qwen3_moe_offload!(session, tokens[1:2])
CUDA.synchronize()
configure_mode!(session, :shared_open)
prefill_hf_qwen3_moe_offload!(session, tokens[1:2])
CUDA.synchronize()
configure_mode!(session, :tensor)
evict_checkpoint_page_cache!(shards)

modes = (:tensor, :shared_open, :coalesced)
runs = Dict(mode => NamedTuple[] for mode in modes)
baseline = Ref{Any}(nothing)
all_outputs_exact = Ref(true)
orders = [isodd(index) ? modes : reverse(modes) for index in 1:repetitions]
for repetition in 1:repetitions
    for mode in orders[repetition]
        println(stderr, "cold mode=$mode repetition=$repetition")
        configure_mode!(session, mode)
        evict_checkpoint_page_cache!(shards)
        cold = run_request(session, mode, "fadvise_cold", repetition)

        println(stderr, "revisit mode=$mode repetition=$repetition")
        configure_mode!(session, mode)
        revisit = run_request(session, mode, "post_cold_revisit", repetition)
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
        push!(runs[mode], cold.report)
        push!(runs[mode], revisit.report)
    end
end

function phase_summary(mode_runs, phase)
    selected = filter(run -> run.page_cache_phase == phase, mode_runs)
    request_seconds = [run.request_seconds for run in selected]
    read_syscalls = [run.process_read_syscalls for run in selected]
    allocated_bytes = [run.julia_allocated_bytes for run in selected]
    return (;
        repetitions=length(selected),
        request_seconds_median=median(request_seconds),
        request_seconds_mean=mean(request_seconds),
        request_seconds_minimum=minimum(request_seconds),
        request_seconds_maximum=maximum(request_seconds),
        read_syscalls_median=median(read_syscalls),
        allocated_bytes_median=median(allocated_bytes),
        runs=Tuple(selected),
    )
end

configurations = map(modes) do mode
    return (;
        mode=String(mode),
        cold=phase_summary(runs[mode], "fadvise_cold"),
        revisit=phase_summary(runs[mode], "post_cold_revisit"),
    )
end
tensor, shared_open, coalesced = configurations
fastest_cold = configurations[argmin([
    configuration.cold.request_seconds_median
    for configuration in configurations
])]
fastest_revisit = configurations[argmin([
    configuration.revisit.request_seconds_median
    for configuration in configurations
])]

repo_root = normpath(joinpath(@__DIR__, ".."))
source_paths = (;
    benchmark_script=relpath(@__FILE__, repo_root),
    streaming_implementation=joinpath("src", "io", "hf_streaming.jl"),
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
        cache_budget_bytes,
        cache_policy="layer_balanced_lru",
        dispatch="scattered",
        gc_interval_layers=8,
        read_workers=8,
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
    read_layout=(;
        projection_order="down, gate, up",
        projection_bytes=3_145_728,
        expert_bytes=9_437_184,
        exactly_adjacent=true,
        bridge_unrelated_bytes=false,
    ),
    methodology=(;
        repetitions,
        orders=Tuple(Tuple(String.(order)) for order in orders),
        cold_control="POSIX_FADV_DONTNEED on all 16 checkpoint shards",
        cold_verification="per-request /proc/self/io read_bytes",
        device_cache="cleared before every request",
        pinned_upload=false,
    ),
    configurations,
    comparisons=(;
        shared_open_cold_speedup=tensor.cold.request_seconds_median /
            shared_open.cold.request_seconds_median,
        shared_open_revisit_speedup=tensor.revisit.request_seconds_median /
            shared_open.revisit.request_seconds_median,
        coalesced_cold_speedup=tensor.cold.request_seconds_median /
            coalesced.cold.request_seconds_median,
        coalesced_revisit_speedup=tensor.revisit.request_seconds_median /
            coalesced.revisit.request_seconds_median,
        coalesced_cold_read_syscall_reduction=1 -
            coalesced.cold.read_syscalls_median /
                tensor.cold.read_syscalls_median,
        coalesced_revisit_read_syscall_reduction=1 -
            coalesced.revisit.read_syscalls_median /
                tensor.revisit.read_syscalls_median,
        coalesced_cold_allocation_reduction=1 -
            coalesced.cold.allocated_bytes_median /
                tensor.cold.allocated_bytes_median,
        coalesced_revisit_allocation_reduction=1 -
            coalesced.revisit.allocated_bytes_median /
                tensor.revisit.allocated_bytes_median,
    ),
    correctness=(; all_outputs_exact=all_outputs_exact[]),
    decision=(;
        fastest_cold_mode=fastest_cold.mode,
        fastest_revisit_mode=fastest_revisit.mode,
        selected_default_read_mode=fastest_cold.mode,
        arbitrary_checkpoint_layout_generalized=false,
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
