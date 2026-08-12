#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using Statistics: mean

length(ARGS) in (3, 4) || error(
    "usage: julia --project=. scripts/benchmark_qwen3_moe_cuda_expert_cache.jl " *
    "MODEL_DIR REFERENCE_DIR OUTPUT_JSON [CACHE_GIB]",
)
model_dir, reference_dir, output_path = ARGS
cache_gib = length(ARGS) == 4 ? parse(Float64, ARGS[4]) : 9.0
cache_gib > 0 || error("CACHE_GIB must be positive")
cache_budget_bytes = floor(Int, cache_gib * 2.0^30)
context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))

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
        to_device=CUDA.cu,
        on_resident_layer=(layer, total) ->
            (layer == 1 || layer % 8 == 0 || layer == total) &&
                println(stderr, "resident layer $layer/$total"),
    )
)
gpu_free_ready = Int(CUDA.free_memory())

function run_request(session, tokens, decode_token)
    prefill, prefill_seconds = timed_cuda(() ->
        prefill_hf_qwen3_moe_offload!(session, tokens)
    )
    decode, decode_seconds = timed_cuda(() ->
        decode_hf_qwen3_moe_offload!(session, decode_token)
    )
    return (; prefill, decode, prefill_seconds, decode_seconds)
end

# Compile kernels and populate the cache once. The second request is the first
# all-hit observation, but it is not used as the sole timing evidence.
cold_fill = run_request(session, tokens, decode_token)
first_hit = run_request(session, tokens, decode_token)

# Clear only expert ownership, retaining the resident model, 40K KV buffers,
# allocator pool and compiled kernels. This produces a warm miss baseline.
clear_hf_qwen3_moe_expert_cache!(session)
warm_fill = run_request(session, tokens, decode_token)
warm_hit = run_request(session, tokens, decode_token)
gpu_free_final = Int(CUDA.free_memory())

hf_layout(array) = permutedims(array, (3, 2, 1))
expected_prefill = Float32.(hf_layout(reference["logits"])[:, end, 1])
expected_decode = Float32.(vec(hf_layout(reference["decode_logits"])))
actual_prefill = Float32.(vec(warm_hit.prefill.logits))
actual_decode = Float32.(vec(warm_hit.decode.logits))
prefill_abs = abs.(actual_prefill .- expected_prefill)
decode_abs = abs.(actual_decode .- expected_decode)

function request_report(request)
    return (;
        prefill_seconds=request.prefill_seconds,
        decode_seconds=request.decode_seconds,
        prefill_expert_bytes_read=request.prefill.expert_bytes_read,
        prefill_expert_bytes_uploaded=request.prefill.expert_bytes_uploaded,
        prefill_cache_hits=request.prefill.expert_cache.hits,
        prefill_cache_misses=request.prefill.expert_cache.misses,
        decode_expert_bytes_read=request.decode.expert_bytes_read,
        decode_expert_bytes_uploaded=request.decode.expert_bytes_uploaded,
        decode_cache_hits=request.decode.expert_cache_hits,
        decode_cache_misses=request.decode.expert_cache_misses,
        cache_entries=request.decode.expert_cache.entries,
        cache_current_bytes=request.decode.expert_cache.current_bytes,
        cache_peak_bytes=request.decode.expert_cache.peak_bytes,
        cache_evictions=request.decode.expert_cache.evictions,
    )
end

report = (;
    schema_version=1,
    model_id="Qwen/Qwen3-30B-A3B",
    revision=String(metadata["revision"]),
    compute_dtype="bfloat16",
    dispatch="active-expert-local-remap-production-scalar-f32-accumulation",
    gpu=(;
        name=CUDA.name(CUDA.device()),
        capability=string(CUDA.capability(CUDA.device())),
        free_before_bytes=gpu_free_before,
        free_ready_bytes=gpu_free_ready,
        free_final_bytes=gpu_free_final,
    ),
    context_tokens,
    prompt_tokens=length(tokens),
    decode_tokens=1,
    cache_budget_bytes,
    load_seconds,
    cold_fill=request_report(cold_fill),
    first_hit=request_report(first_hit),
    warm_fill=request_report(warm_fill),
    warm_hit=request_report(warm_hit),
    warm_hit_over_fill_speedup=(;
        prefill=warm_fill.prefill_seconds / warm_hit.prefill_seconds,
        decode=warm_fill.decode_seconds / warm_hit.decode_seconds,
        request=(warm_fill.prefill_seconds + warm_fill.decode_seconds) /
            (warm_hit.prefill_seconds + warm_hit.decode_seconds),
    ),
    parity=(;
        prefill_logits_max_abs=maximum(prefill_abs),
        prefill_logits_mean_abs=mean(prefill_abs),
        prefill_argmax_match=argmax(actual_prefill) == argmax(expected_prefill),
        decode_logits_max_abs=maximum(decode_abs),
        decode_logits_mean_abs=mean(decode_abs),
        decode_argmax_match=argmax(actual_decode) == argmax(expected_decode),
        cold_vs_first_hit_exact=
            cold_fill.prefill.logits == first_hit.prefill.logits &&
            cold_fill.decode.logits == first_hit.decode.logits,
        warm_fill_vs_hit_exact=
            warm_fill.prefill.logits == warm_hit.prefill.logits &&
            warm_fill.decode.logits == warm_hit.decode.logits,
    ),
)

mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(JSON3.write(report))
