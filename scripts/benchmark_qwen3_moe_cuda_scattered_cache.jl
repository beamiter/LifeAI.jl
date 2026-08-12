#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using Statistics: mean, median

length(ARGS) == 3 || error(
    "usage: julia --project=. " *
    "scripts/benchmark_qwen3_moe_cuda_scattered_cache.jl " *
    "MODEL_DIR REFERENCE_DIR OUTPUT_JSON",
)
model_dir, reference_dir, output_path = ARGS
context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))
cache_budget_bytes = parse(
    Int,
    get(ENV, "LIFEAI_MOE_CACHE_BUDGET_BYTES", string(8 * 2^30)),
)

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

gpu_free_before = Int(CUDA.free_memory())
session, load_seconds = timed_cuda(() ->
    load_hf_qwen3_moe_offload_session(
        model_dir;
        context_tokens,
        prefill_chunk_tokens=128,
        grouped_experts=false,
        expert_cache_budget_bytes=cache_budget_bytes,
        expert_cache_policy=:global_lru,
        expert_cache_dispatch=:materialized,
        expert_gc_interval_layers=1,
        to_device=CUDA.cu,
        on_resident_layer=(layer, total) ->
            (layer == 1 || layer % 8 == 0 || layer == total) &&
                println(stderr, "resident layer $layer/$total"),
    )
)
gpu_free_ready = Int(CUDA.free_memory())

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
        cache_entries=stats.entries,
        cache_current_bytes=stats.current_bytes,
        active_materializations=stats.active_materializations,
        active_materialization_bytes=stats.active_materialization_bytes,
        scattered_dispatches=stats.scattered_dispatches,
        pointer_bytes_uploaded=stats.pointer_bytes_uploaded,
        forced_gc_calls=stats.forced_gc_calls,
    )
end

configurations = (
    (name="materialized_gc1", dispatch=:materialized, gc_interval=1),
    (name="scattered_gc1", dispatch=:scattered, gc_interval=1),
    (name="scattered_gc8", dispatch=:scattered, gc_interval=8),
    (name="scattered_gc0", dispatch=:scattered, gc_interval=0),
)
reports = Any[]
raw_results = Any[]
for configuration in configurations
    println(stderr, "configuration $(configuration.name)")
    configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes=cache_budget_bytes,
        policy=:global_lru,
        dispatch=configuration.dispatch,
        gc_interval_layers=configuration.gc_interval,
    )
    GC.gc(true)
    CUDA.reclaim()
    free_start = Int(CUDA.free_memory())
    fill = run_request(session)
    hit = run_request(session)
    repeat_hits = if configuration.gc_interval == 0
        repeat_count = parse(
            Int,
            get(ENV, "LIFEAI_MOE_SCATTERED_REPEAT_HITS", "100"),
        )
        repeat_count > 0 || error(
            "LIFEAI_MOE_SCATTERED_REPEAT_HITS must be positive",
        )
        free_before_repeats = Int(CUDA.free_memory())
        seconds = Float64[]
        prefill_seconds = Float64[]
        decode_seconds = Float64[]
        exact = true
        zero_io = true
        for _ in 1:repeat_count
            repeated = run_request(session)
            push!(seconds, repeated.prefill_seconds + repeated.decode_seconds)
            push!(prefill_seconds, repeated.prefill_seconds)
            push!(decode_seconds, repeated.decode_seconds)
            exact &= repeated.prefill.logits == hit.prefill.logits &&
                repeated.decode.logits == hit.decode.logits
            zero_io &= repeated.prefill.expert_bytes_read == 0 &&
                repeated.decode.expert_bytes_read == 0 &&
                repeated.prefill.expert_bytes_uploaded == 0 &&
                repeated.decode.expert_bytes_uploaded == 0
        end
        free_after_repeats = Int(CUDA.free_memory())
        (;
            count=repeat_count,
            request_seconds_median=median(seconds),
            request_seconds_minimum=minimum(seconds),
            request_seconds_maximum=maximum(seconds),
            prefill_seconds_median=median(prefill_seconds),
            decode_seconds_median=median(decode_seconds),
            exact,
            zero_io,
            free_before_bytes=free_before_repeats,
            free_after_bytes=free_after_repeats,
            allocator_free_drift_bytes=
                free_after_repeats - free_before_repeats,
        )
    else
        nothing
    end
    free_final = Int(CUDA.free_memory())
    push!(raw_results, (; fill, hit))
    push!(reports, (;
        name=configuration.name,
        dispatch=String(configuration.dispatch),
        gc_interval_layers=configuration.gc_interval,
        gpu_free_start_bytes=free_start,
        gpu_free_final_bytes=free_final,
        fill=request_report(fill),
        hit=request_report(hit),
        repeat_hits,
        fill_vs_hit_exact=(;
            prefill=fill.prefill.logits == hit.prefill.logits,
            decode=fill.decode.logits == hit.decode.logits,
        ),
    ))
end

materialized = reports[1]
scattered_gc1 = reports[2]
scattered_gc8 = reports[3]
scattered_gc0 = reports[4]
materialized_raw = raw_results[1]
scattered_gc1_raw = raw_results[2]
scattered_gc8_raw = raw_results[3]
scattered_gc0_raw = raw_results[4]

hf_layout(array) = permutedims(array, (3, 2, 1))
expected_prefill = Float32.(hf_layout(reference["logits"])[:, end, 1])
expected_decode = Float32.(vec(hf_layout(reference["decode_logits"])))
actual_prefill = Float32.(vec(scattered_gc8_raw.hit.prefill.logits))
actual_decode = Float32.(vec(scattered_gc8_raw.hit.decode.logits))
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
        load_seconds,
    ),
    configurations=Tuple(reports),
    comparisons=(;
        scattered_gc1_over_materialized=(;
            prefill_speedup=materialized.hit.prefill_seconds /
                scattered_gc1.hit.prefill_seconds,
            decode_speedup=materialized.hit.decode_seconds /
                scattered_gc1.hit.decode_seconds,
            request_speedup=materialized.hit.request_seconds /
                scattered_gc1.hit.request_seconds,
            materialization_bytes_eliminated=
                materialized.hit.active_materialization_bytes -
                scattered_gc1.hit.active_materialization_bytes,
        ),
        scattered_gc8_over_scattered_gc1=(;
            prefill_speedup=scattered_gc1.hit.prefill_seconds /
                scattered_gc8.hit.prefill_seconds,
            decode_speedup=scattered_gc1.hit.decode_seconds /
                scattered_gc8.hit.decode_seconds,
            request_speedup=scattered_gc1.hit.request_seconds /
                scattered_gc8.hit.request_seconds,
            forced_gc_call_reduction=scattered_gc1.hit.forced_gc_calls -
                scattered_gc8.hit.forced_gc_calls,
        ),
        scattered_gc8_over_materialized=(;
            prefill_speedup=materialized.hit.prefill_seconds /
                scattered_gc8.hit.prefill_seconds,
            decode_speedup=materialized.hit.decode_seconds /
                scattered_gc8.hit.decode_seconds,
            request_speedup=materialized.hit.request_seconds /
                scattered_gc8.hit.request_seconds,
        ),
        scattered_gc0_over_scattered_gc8=(;
            prefill_speedup=scattered_gc8.hit.prefill_seconds /
                scattered_gc0.hit.prefill_seconds,
            decode_speedup=scattered_gc8.hit.decode_seconds /
                scattered_gc0.hit.decode_seconds,
            request_speedup=scattered_gc8.hit.request_seconds /
                scattered_gc0.hit.request_seconds,
            forced_gc_call_reduction=scattered_gc8.hit.forced_gc_calls -
                scattered_gc0.hit.forced_gc_calls,
        ),
        scattered_gc0_over_materialized=(;
            prefill_speedup=materialized.hit.prefill_seconds /
                scattered_gc0.hit.prefill_seconds,
            decode_speedup=materialized.hit.decode_seconds /
                scattered_gc0.hit.decode_seconds,
            request_speedup=materialized.hit.request_seconds /
                scattered_gc0.hit.request_seconds,
        ),
    ),
    cross_configuration_exact=(;
        materialized_vs_scattered_gc1_prefill=
            materialized_raw.hit.prefill.logits ==
                scattered_gc1_raw.hit.prefill.logits,
        materialized_vs_scattered_gc1_decode=
            materialized_raw.hit.decode.logits ==
                scattered_gc1_raw.hit.decode.logits,
        materialized_vs_scattered_gc8_prefill=
            materialized_raw.hit.prefill.logits ==
                scattered_gc8_raw.hit.prefill.logits,
        materialized_vs_scattered_gc8_decode=
            materialized_raw.hit.decode.logits ==
                scattered_gc8_raw.hit.decode.logits,
        materialized_vs_scattered_gc0_prefill=
            materialized_raw.hit.prefill.logits ==
                scattered_gc0_raw.hit.prefill.logits,
        materialized_vs_scattered_gc0_decode=
            materialized_raw.hit.decode.logits ==
                scattered_gc0_raw.hit.decode.logits,
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
