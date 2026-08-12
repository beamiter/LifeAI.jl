#!/usr/bin/env julia

using CUDA
using JSON3
using LifeAI
using Statistics: mean

length(ARGS) in (3, 4, 5) || error(
    "usage: julia --project=. scripts/benchmark_qwen3_moe_cuda_offload.jl " *
    "MODEL_DIR REFERENCE_DIR OUTPUT_JSON [GROUPED_EXPERTS] [PROMPT_REPEAT]",
)
model_dir, reference_dir, output_path = ARGS
CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)

context_tokens = parse(Int, get(ENV, "LIFEAI_MOE_CONTEXT_TOKENS", "40960"))
prefill_chunk_tokens = parse(
    Int,
    get(ENV, "LIFEAI_MOE_PREFILL_CHUNK_TOKENS", "128"),
)
grouped_experts = lowercase(
    length(ARGS) >= 4 ? ARGS[4] : get(
        ENV,
        "LIFEAI_MOE_GROUPED_EXPERTS",
        "true",
    ),
) in ("1", "true", "yes")
metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
String(metadata["compute_dtype"]) == "bfloat16" || error(
    "reference must use bfloat16 compute",
)
reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))
base_tokens = Int.(collect(metadata["token_ids_0_based"])) .+ 1
prompt_repeat = length(ARGS) == 5 ? parse(Int, ARGS[5]) : 1
prompt_repeat > 0 || error("PROMPT_REPEAT must be positive")
tokens = repeat(base_tokens, prompt_repeat)
decode_token = Int(metadata["decode_token_id_0_based"]) + 1

hf_layout(array) = permutedims(array, (3, 2, 1))
elapsed_seconds(started) = (time_ns() - started) / 1.0e9
function synchronize_value(f)
    value = f()
    CUDA.synchronize()
    return value
end

gpu_free_before = Int(CUDA.free_memory())
load_started = time_ns()
session = load_hf_qwen3_moe_offload_session(
    model_dir;
    context_tokens,
    prefill_chunk_tokens,
    grouped_experts,
    to_device=CUDA.cu,
    on_resident_layer=(layer, total) ->
        (layer == 1 || layer % 8 == 0 || layer == total) &&
            println(stderr, "resident layer $layer/$total"),
)
CUDA.synchronize()
load_seconds = elapsed_seconds(load_started)
gpu_free_ready = Int(CUDA.free_memory())

plan = qwen3_moe_offload_plan(session.model, context_tokens)
session.resident_parameter_bytes == plan.resident_parameter_bytes || error(
    "resident parameter bytes differ from the static plan",
)

prefill_started = time_ns()
prefill = synchronize_value(() ->
    prefill_hf_qwen3_moe_offload!(session, tokens)
)
prefill_seconds = elapsed_seconds(prefill_started)
gpu_free_after_prefill = Int(CUDA.free_memory())

decode_started = time_ns()
decode = synchronize_value(() ->
    decode_hf_qwen3_moe_offload!(session, decode_token)
)
decode_seconds = elapsed_seconds(decode_started)
gpu_free_after_decode = Int(CUDA.free_memory())

steady_prefill_started = time_ns()
steady_prefill = synchronize_value(() ->
    prefill_hf_qwen3_moe_offload!(session, tokens)
)
steady_prefill_seconds = elapsed_seconds(steady_prefill_started)
steady_decode_started = time_ns()
steady_decode = synchronize_value(() ->
    decode_hf_qwen3_moe_offload!(session, decode_token)
)
steady_decode_seconds = elapsed_seconds(steady_decode_started)
gpu_free_steady = Int(CUDA.free_memory())

actual_prefill = Float32.(vec(prefill.logits))
actual_decode = Float32.(vec(decode.logits))
reference_comparable = prompt_repeat == 1
parity = if reference_comparable
    expected_prefill = Float32.(
        hf_layout(reference["logits"])[:, end, 1],
    )
    expected_decode = Float32.(vec(hf_layout(reference["decode_logits"])))
    prefill_abs = abs.(actual_prefill .- expected_prefill)
    decode_abs = abs.(actual_decode .- expected_decode)
    (;
        prefill_logits_max_abs=maximum(prefill_abs),
        prefill_logits_mean_abs=mean(prefill_abs),
        prefill_argmax_match=argmax(actual_prefill) == argmax(expected_prefill),
        decode_logits_max_abs=maximum(decode_abs),
        decode_logits_mean_abs=mean(decode_abs),
        decode_argmax_match=argmax(actual_decode) == argmax(expected_decode),
        prefill_repeat_exact=steady_prefill.logits == prefill.logits,
        decode_repeat_exact=steady_decode.logits == decode.logits,
    )
else
    nothing
end

prompt_active_counts = [
    length(active)
    for active in prefill.chunks[end].active_experts
]
decode_active_counts = [length(active) for active in decode.active_experts]
prompt_active_set_matches, decode_active_set_matches = if reference_comparable
    expected_prompt_active = [
        sort!(unique!(Int.(vec(reference["selected_experts.$layer"]))) .+ 1)
        for layer in 0:(session.model.num_layers - 1)
    ]
    expected_decode_active = [
        sort!(unique!(Int.(vec(reference["decode_selected_experts.$layer"]))) .+ 1)
        for layer in 0:(session.model.num_layers - 1)
    ]
    (
        count(identity, [
            prefill.chunks[end].active_experts[layer] == expected_prompt_active[layer]
            for layer in 1:session.model.num_layers
        ]),
        count(identity, [
            decode.active_experts[layer] == expected_decode_active[layer]
            for layer in 1:session.model.num_layers
        ]),
    )
else
    nothing, nothing
end

report = (;
    schema_version=1,
    model_id="Qwen/Qwen3-30B-A3B",
    revision=String(metadata["revision"]),
    compute_dtype="bfloat16",
    dispatch=grouped_experts ?
        "active-expert-local-remap-grouped-bf16-wmma" :
        "active-expert-local-remap-production-scalar-f32-accumulation",
    gpu=(;
        name=CUDA.name(CUDA.device()),
        capability=string(CUDA.capability(CUDA.device())),
        free_before_bytes=gpu_free_before,
        free_ready_bytes=gpu_free_ready,
        free_after_prefill_bytes=gpu_free_after_prefill,
        free_after_decode_bytes=gpu_free_after_decode,
        free_steady_bytes=gpu_free_steady,
    ),
    plan,
    resident_parameter_bytes=session.resident_parameter_bytes,
    context_cache_allocated=true,
    prompt_tokens=length(tokens),
    prompt_repeat,
    decode_tokens=1,
    load_seconds,
    prefill_seconds,
    decode_seconds,
    steady_prefill_seconds,
    steady_decode_seconds,
    prefill_expert_bytes_read=prefill.expert_bytes_read,
    decode_expert_bytes_read=decode.expert_bytes_read,
    steady_prefill_expert_bytes_read=steady_prefill.expert_bytes_read,
    steady_decode_expert_bytes_read=steady_decode.expert_bytes_read,
    prefill_argmax_1_based=argmax(actual_prefill),
    decode_argmax_1_based=argmax(actual_decode),
    prompt_active_experts=(;
        minimum=minimum(prompt_active_counts),
        maximum=maximum(prompt_active_counts),
        mean=mean(prompt_active_counts),
        reference_set_matches=prompt_active_set_matches,
        layers=session.model.num_layers,
    ),
    decode_active_experts=(;
        minimum=minimum(decode_active_counts),
        maximum=maximum(decode_active_counts),
        mean=mean(decode_active_counts),
        reference_set_matches=decode_active_set_matches,
        layers=session.model.num_layers,
    ),
    parity,
)

mkpath(dirname(abspath(output_path)))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(JSON3.write(report))
