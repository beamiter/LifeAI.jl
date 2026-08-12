#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using SHA: sha256

length(ARGS) == 2 || error(
    "usage: julia --project=. " *
    "scripts/benchmark_qwen3_moe_cuda_layer_balanced_cache.jl " *
    "MODEL_DIR OUTPUT_JSON",
)
model_dir, output_path = ARGS
context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))
workload_tokens = parse(Int, get(ENV, "LIFEAI_MOE_WORKLOAD_TOKENS", "32"))
workload_tokens > 1 || error("LIFEAI_MOE_WORKLOAD_TOKENS must exceed one")

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)

spec = qwen3_moe_checkpoint_spec()
tokenizer = load_hf_qwen3_tokenizer(model_dir; revision=spec.revision)
workload_text = (;
    english=
        "Explain how sparse mixture-of-experts models increase parameter " *
        "capacity while keeping per-token computation manageable, and " *
        "describe why routing locality matters for GPU inference. Focus on " *
        "memory traffic and cache behavior.",
    chinese=
        "请解释稀疏混合专家模型如何在控制每个 token 计算量的同时扩大参数容量，" *
        "并说明路由局部性为什么会影响单卡 GPU 推理效率。",
)
function fixed_tokens(tokenizer, text, count)
    tokens = encode(tokenizer, text; add_special_tokens=false)
    length(tokens) >= count || error(
        "natural-text workload has only $(length(tokens)) tokens; need $count",
    )
    return tokens[1:count]
end
english_tokens = fixed_tokens(tokenizer, workload_text.english, workload_tokens)
chinese_tokens = fixed_tokens(tokenizer, workload_text.chinese, workload_tokens)

sha256_string(value) = bytes2hex(sha256(codeunits(value)))
token_sha256(tokens) = sha256_string(join(tokens, ','))
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
        to_device=CUDA.cu,
        on_resident_layer=(layer, total) ->
            (layer == 1 || layer % 8 == 0 || layer == total) &&
                println(stderr, "resident layer $layer/$total"),
    )
)
gpu_free_ready = Int(CUDA.free_memory())

function run_request(session, name, tokens)
    println(stderr, "request $name")
    prefill, prefill_seconds = timed_cuda(() ->
        prefill_hf_qwen3_moe_offload!(session, tokens)
    )
    prefill_logits = Float32.(vec(Array(prefill.logits)))
    decode_token = argmax(prefill_logits)
    decode, decode_seconds = timed_cuda(() ->
        decode_hf_qwen3_moe_offload!(session, decode_token)
    )
    decode_logits = Float32.(vec(Array(decode.logits)))
    return (;
        name,
        prefill,
        decode,
        prefill_logits,
        decode_logits,
        decode_token,
        prefill_seconds,
        decode_seconds,
    )
end

function request_report(request)
    prefill_hits = request.prefill.expert_cache.hits
    prefill_misses = request.prefill.expert_cache.misses
    decode_hits = request.decode.expert_cache_hits
    decode_misses = request.decode.expert_cache_misses
    hits = prefill_hits + decode_hits
    misses = prefill_misses + decode_misses
    return (;
        name=request.name,
        prefill_seconds=request.prefill_seconds,
        decode_seconds=request.decode_seconds,
        request_seconds=request.prefill_seconds + request.decode_seconds,
        prefill_expert_entries=sum(
            length,
            request.prefill.chunks[end].active_experts,
        ),
        decode_expert_entries=sum(length, request.decode.active_experts),
        prefill_cache_hits=prefill_hits,
        prefill_cache_misses=prefill_misses,
        decode_cache_hits=decode_hits,
        decode_cache_misses=decode_misses,
        cache_hits=hits,
        cache_misses=misses,
        cache_hit_rate=hits / (hits + misses),
        expert_bytes_read=request.prefill.expert_bytes_read +
            request.decode.expert_bytes_read,
        expert_bytes_uploaded=request.prefill.expert_bytes_uploaded +
            request.decode.expert_bytes_uploaded,
        cache_entries=request.decode.expert_cache.entries,
        cache_current_bytes=request.decode.expert_cache.current_bytes,
        cache_peak_bytes=request.decode.expert_cache.peak_bytes,
        cache_evictions=request.decode.expert_cache.evictions,
        decode_token_1_based=request.decode_token,
    )
end

# Compile the full resident path before measuring any policy. The cache and
# allocator pool are cleared afterward, so every configuration starts with an
# empty logical cache while retaining compiled kernels and resident tensors.
println(stderr, "compile warmup")
warmup = run_request(session, "warmup", english_tokens[1:2])
clear_hf_qwen3_moe_expert_cache!(session)
warmup = nothing
GC.gc(true)
CUDA.reclaim()

configurations = (
    (name="global_8gib", policy=:global_lru, gib=8),
    (name="balanced_4gib", policy=:layer_balanced_lru, gib=4),
    (name="balanced_6gib", policy=:layer_balanced_lru, gib=6),
    (name="balanced_8gib", policy=:layer_balanced_lru, gib=8),
)
configuration_reports = Any[]
for configuration in configurations
    budget_bytes = configuration.gib * 2^30
    println(stderr, "configuration $(configuration.name)")
    configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes,
        policy=configuration.policy,
    )
    GC.gc(true)
    CUDA.reclaim()
    free_start = Int(CUDA.free_memory())

    english_first = run_request(
        session,
        "english_first",
        english_tokens,
    )
    chinese_transition = run_request(
        session,
        "chinese_transition",
        chinese_tokens,
    )
    english_revisit = run_request(
        session,
        "english_revisit",
        english_tokens,
    )
    requests = (
        request_report(english_first),
        request_report(chinese_transition),
        request_report(english_revisit),
    )
    total_hits = sum(request.cache_hits for request in requests)
    total_misses = sum(request.cache_misses for request in requests)
    total_seconds = sum(request.request_seconds for request in requests)
    total_bytes_read = sum(request.expert_bytes_read for request in requests)
    push!(configuration_reports, (;
        name=configuration.name,
        policy=String(configuration.policy),
        budget_bytes,
        layer_slot_capacity=(budget_bytes ÷ 9_437_184) ÷
            session.model.num_layers,
        total_slot_capacity=budget_bytes ÷ 9_437_184,
        gpu_free_start_bytes=free_start,
        gpu_free_final_bytes=Int(CUDA.free_memory()),
        requests,
        trace=(;
            cache_hits=total_hits,
            cache_misses=total_misses,
            cache_hit_rate=total_hits / (total_hits + total_misses),
            expert_bytes_read=total_bytes_read,
            expert_bytes_uploaded=sum(
                request.expert_bytes_uploaded
                for request in requests
            ),
            seconds=total_seconds,
        ),
        english_revisit_exact=(;
            prefill=english_revisit.prefill_logits ==
                english_first.prefill_logits,
            decode=english_revisit.decode_logits ==
                english_first.decode_logits,
            decode_token=english_revisit.decode_token ==
                english_first.decode_token,
        ),
    ))
    english_first = nothing
    chinese_transition = nothing
    english_revisit = nothing
end

global_8gib = configuration_reports[1]
balanced_8gib = configuration_reports[4]
report = (;
    schema_version=1,
    model_id=spec.model_id,
    revision=spec.revision,
    compute_dtype="bfloat16",
    dispatch="active-expert-local-remap-production-scalar-f32-accumulation",
    gpu=(;
        name=CUDA.name(CUDA.device()),
        capability=string(CUDA.capability(CUDA.device())),
        free_before_bytes=gpu_free_before,
        free_ready_bytes=gpu_free_ready,
    ),
    session=(;
        context_tokens,
        prefill_chunk_tokens=128,
        load_seconds,
        resident_parameter_bytes=session.resident_parameter_bytes,
    ),
    workload=(;
        trace="english_32 -> chinese_32 -> english_32",
        tokens_per_prompt=workload_tokens,
        decode_tokens_per_request=1,
        english_text_sha256=sha256_string(workload_text.english),
        chinese_text_sha256=sha256_string(workload_text.chinese),
        english_tokens_sha256=token_sha256(english_tokens),
        chinese_tokens_sha256=token_sha256(chinese_tokens),
        tokenizer_sha256=tokenizer.tokenizer_sha256,
    ),
    configurations=Tuple(configuration_reports),
    balanced_8gib_over_global_8gib=(;
        hit_rate_delta=balanced_8gib.trace.cache_hit_rate -
            global_8gib.trace.cache_hit_rate,
        bytes_read_reduction=1 - balanced_8gib.trace.expert_bytes_read /
            global_8gib.trace.expert_bytes_read,
        speedup=global_8gib.trace.seconds / balanced_8gib.trace.seconds,
    ),
)

mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(JSON3.write(report))
