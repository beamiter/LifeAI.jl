#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using Statistics: mean, median, quantile

length(ARGS) == 3 || error(
    "usage: julia --project=. " *
    "scripts/benchmark_qwen3_moe_cuda_scattered_reuse.jl " *
    "MODEL_DIR REFERENCE_DIR OUTPUT_JSON",
)
model_dir, reference_dir, output_path = ARGS
context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))
cache_budget_bytes = parse(
    Int,
    get(ENV, "LIFEAI_MOE_CACHE_BUDGET_BYTES", string(8 * 2^30)),
)
gc8_repeats = parse(Int, get(ENV, "LIFEAI_MOE_REUSE_GC8_REPEATS", "100"))
gc0_repeats = parse(Int, get(ENV, "LIFEAI_MOE_REUSE_GC0_REPEATS", "500"))
gc8_repeats > 0 || error("LIFEAI_MOE_REUSE_GC8_REPEATS must be positive")
gc0_repeats > 0 || error("LIFEAI_MOE_REUSE_GC0_REPEATS must be positive")

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
    return (; prefill, decode, prefill_seconds, decode_seconds)
end

function request_report(request)
    stats = request.decode.expert_cache
    return (;
        prefill_seconds=request.prefill_seconds,
        decode_seconds=request.decode_seconds,
        request_seconds=request.prefill_seconds + request.decode_seconds,
        expert_bytes_read=request.prefill.expert_bytes_read +
            request.decode.expert_bytes_read,
        expert_bytes_uploaded=request.prefill.expert_bytes_uploaded +
            request.decode.expert_bytes_uploaded,
        cache_hits=stats.hits,
        cache_misses=stats.misses,
        pointer_bytes_uploaded=stats.pointer_bytes_uploaded,
        pointer_table_builds=stats.pointer_table_builds,
        pointer_table_reuses=stats.pointer_table_reuses,
        workspace_allocations=stats.workspace_allocations,
        workspace_reuses=stats.workspace_reuses,
        pointer_table_bytes=stats.pointer_table_bytes,
        workspace_bytes=stats.workspace_bytes,
        forced_gc_calls=stats.forced_gc_calls,
    )
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

function run_configuration(session, name, gc_interval, repeat_count)
    println(stderr, "configuration $name repeats=$repeat_count")
    configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes=cache_budget_bytes,
        policy=:global_lru,
        dispatch=:scattered,
        gc_interval_layers=gc_interval,
    )
    GC.gc(true)
    CUDA.reclaim()
    free_start = Int(CUDA.free_memory())
    fill = run_request(session)
    hit = run_request(session)
    CUDA.synchronize()
    GC.gc(true)
    CUDA.reclaim()
    free_before_repeats = Int(CUDA.free_memory())

    request_seconds = Float64[]
    prefill_seconds = Float64[]
    decode_seconds = Float64[]
    exact = true
    zero_io = true
    pointer_reuse_only = true
    workspace_reuse_only = true
    checkpoints = NamedTuple[]
    checkpoint_interval = max(1, repeat_count ÷ 10)
    for index in 1:repeat_count
        repeated = run_request(session)
        push!(request_seconds, repeated.prefill_seconds + repeated.decode_seconds)
        push!(prefill_seconds, repeated.prefill_seconds)
        push!(decode_seconds, repeated.decode_seconds)
        exact &= repeated.prefill.logits == hit.prefill.logits &&
            repeated.decode.logits == hit.decode.logits
        zero_io &= repeated.prefill.expert_bytes_read == 0 &&
            repeated.decode.expert_bytes_read == 0 &&
            repeated.prefill.expert_bytes_uploaded == 0 &&
            repeated.decode.expert_bytes_uploaded == 0
        stats = repeated.decode.expert_cache
        pointer_reuse_only &= stats.pointer_table_builds == 0 &&
            stats.pointer_table_reuses == 96 &&
            stats.pointer_bytes_uploaded == 0
        workspace_reuse_only &= stats.workspace_allocations == 0 &&
            stats.workspace_reuses == 96
        if index % checkpoint_interval == 0 || index == repeat_count
            push!(checkpoints, (;
                request=index,
                gpu_free_bytes=Int(CUDA.free_memory()),
            ))
        end
    end
    CUDA.synchronize()
    free_after_repeats = Int(CUDA.free_memory())
    GC.gc(true)
    CUDA.reclaim()
    free_after_reclaim = Int(CUDA.free_memory())

    raw = (; fill, hit)
    report = (;
        name,
        gc_interval_layers=gc_interval,
        gpu_free_start_bytes=free_start,
        gpu_free_before_repeats_bytes=free_before_repeats,
        gpu_free_after_repeats_bytes=free_after_repeats,
        gpu_free_after_reclaim_bytes=free_after_reclaim,
        allocator_free_drift_bytes=free_after_repeats - free_before_repeats,
        live_free_drift_after_reclaim_bytes=
            free_after_reclaim - free_before_repeats,
        fill=request_report(fill),
        hit=request_report(hit),
        repeats=(;
            count=repeat_count,
            request_seconds_median=median(request_seconds),
            request_seconds_mean=mean(request_seconds),
            request_seconds_p95=quantile(request_seconds, 0.95),
            request_seconds_minimum=minimum(request_seconds),
            request_seconds_maximum=maximum(request_seconds),
            prefill_seconds_median=median(prefill_seconds),
            decode_seconds_median=median(decode_seconds),
            exact,
            zero_io,
            pointer_reuse_only,
            workspace_reuse_only,
            checkpoints=Tuple(checkpoints),
        ),
        fill_vs_hit_exact=(;
            prefill=fill.prefill.logits == hit.prefill.logits,
            decode=fill.decode.logits == hit.decode.logits,
        ),
    )
    return raw, report
end

gc8_raw, gc8 = run_configuration(session, "scattered_reuse_gc8", 8, gc8_repeats)
gc0_raw, gc0 = run_configuration(session, "scattered_reuse_gc0", 0, gc0_repeats)

hf_layout(array) = permutedims(array, (3, 2, 1))
expected_prefill = Float32.(hf_layout(reference["logits"])[:, end, 1])
expected_decode = Float32.(vec(hf_layout(reference["decode_logits"])))
actual_prefill = Float32.(vec(gc0_raw.hit.prefill.logits))
actual_decode = Float32.(vec(gc0_raw.hit.decode.logits))
prefill_abs = abs.(actual_prefill .- expected_prefill)
decode_abs = abs.(actual_decode .- expected_decode)

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
        load_seconds,
    ),
    configurations=(gc8, gc0),
    comparison=(;
        gc0_over_gc8_repeat_median_speedup=
            gc8.repeats.request_seconds_median /
                gc0.repeats.request_seconds_median,
        chapter28_gc0_allocator_free_drift_bytes=-1_006_632_960,
        chapter28_gc0_repeat_count=100,
        chapter28_pointer_bytes_uploaded_per_hit=26_832,
        chapter28_workspace_reuse_implemented=false,
    ),
    cross_configuration_exact=(;
        prefill=gc8_raw.hit.prefill.logits == gc0_raw.hit.prefill.logits,
        decode=gc8_raw.hit.decode.logits == gc0_raw.hit.decode.logits,
    ),
    parity=(;
        prefill_logits_max_abs=maximum(prefill_abs),
        prefill_logits_mean_abs=mean(prefill_abs),
        prefill_argmax_match=argmax(actual_prefill) == argmax(expected_prefill),
        decode_logits_max_abs=maximum(decode_abs),
        decode_logits_mean_abs=mean(decode_abs),
        decode_argmax_match=argmax(actual_decode) == argmax(expected_decode),
    ),
)

mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(JSON3.write(report))
