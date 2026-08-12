#!/usr/bin/env julia

using JSON3
using SHA
using Statistics: mean
using BFloat16s: BFloat16
using LifeAI
using LifeAI: qwen3_topk_routing

2 <= length(ARGS) <= 3 || error(
    "usage: julia --project=. scripts/verify_qwen3_moe_real_parity.jl " *
    "MODEL_DIR REFERENCE_DIR [OUTPUT_JSON]",
)
model_dir = abspath(ARGS[1])
reference_dir = abspath(ARGS[2])
output_path = length(ARGS) == 3 ? abspath(ARGS[3]) : nothing
metadata_path = joinpath(reference_dir, "reference.json")
reference_path = joinpath(reference_dir, "reference.safetensors")
metadata = JSON3.read(read(metadata_path, String))
compute_dtype_name = String(metadata.compute_dtype)
compute_dtype = if compute_dtype_name == "bfloat16"
    BFloat16
elseif compute_dtype_name == "float32"
    Float32
else
    error("unsupported reference compute dtype: $compute_dtype_name")
end
is_bf16 = compute_dtype === BFloat16

file_sha256(path) = bytes2hex(SHA.sha256(read(path)))
file_sha256(reference_path) == String(metadata.checksums.reference_sha256) ||
    error("reference.safetensors checksum mismatch")
file_sha256(joinpath(model_dir, "config.json")) ==
    String(metadata.checksums.config_sha256) || error("config checksum mismatch")
file_sha256(joinpath(model_dir, "model.safetensors.index.json")) ==
    String(metadata.checksums.index_sha256) || error("index checksum mismatch")
file_sha256(joinpath(@__DIR__, "export_qwen3_moe_real_reference.py")) ==
    String(metadata.checksums.exporter_sha256) || error("exporter checksum mismatch")

reference = load_safetensors(reference_path)
tokens = reshape(hf_token_ids(
    Int.(collect(metadata.token_ids_0_based));
    vocab_size=151936,
), :, 1)
decode_token = hf_token_ids(
    [Int(metadata.decode_token_id_0_based)];
    vocab_size=151936,
)

GC.gc()
timing = @timed stream_hf_qwen3_moe_forward(
    model_dir,
    tokens;
    decode_token,
    max_seq_len=length(tokens) + 1,
    compute_dtype,
)
streamed = timing.value

hf_layout(array) = permutedims(array, (3, 2, 1))
hidden_atol = is_bf16 ? 5.0f-2 : 2.0f-3
hidden_rtol = is_bf16 ? 5.0f-2 : 2.0f-3
router_atol = is_bf16 ? 5.0f-2 : 2.0f-3
router_rtol = is_bf16 ? 5.0f-2 : 2.0f-3
logits_atol = is_bf16 ? 2.0f0 : 3.0f-3
logits_rtol = is_bf16 ? 5.0f-2 : 3.0f-3
routing_weight_atol = is_bf16 ? 1.0f-1 : 2.0f-5

function comparison(name, actual, expected; atol, rtol, check_argmax=false)
    size(actual) == size(expected) || error(
        "$name shape mismatch: $(size(actual)) != $(size(expected))",
    )
    actual_f32 = Float32.(actual)
    expected_f32 = Float32.(expected)
    differences = abs.(actual_f32 .- expected_f32)
    scale = max(1.0f0, maximum(abs.(expected_f32)))
    return (;
        name,
        max_abs=Float64(maximum(differences)),
        mean_abs=Float64(mean(differences)),
        reference_scale=Float64(scale),
        scaled_max_abs=Float64(maximum(differences) / scale),
        passed=maximum(differences) <= atol + rtol * scale,
        argmax_equal=check_argmax ?
            argmax(vec(actual_f32)) == argmax(vec(expected_f32)) : nothing,
    )
end

comparisons = Any[]
push!(comparisons, comparison(
    "embedding",
    streamed.embedding,
    hf_layout(reference["embedding"]);
    atol=0.0f0,
    rtol=0.0f0,
))
routing_checks = Any[]
for layer in 0:(streamed.model.num_layers - 1)
    push!(comparisons, comparison(
        "block.$layer",
        streamed.blocks[layer + 1],
        hf_layout(reference["block.$layer"]);
        atol=hidden_atol,
        rtol=hidden_rtol,
    ))
    push!(comparisons, comparison(
        "router_logits.$layer",
        streamed.router_logits[layer + 1],
        permutedims(reference["router_logits.$layer"], (2, 1));
        atol=router_atol,
        rtol=router_rtol,
    ))
    push!(comparisons, comparison(
        "decode_block.$layer",
        streamed.decode_blocks[layer + 1],
        hf_layout(reference["decode_block.$layer"]);
        atol=hidden_atol,
        rtol=hidden_rtol,
    ))
    push!(comparisons, comparison(
        "decode_router_logits.$layer",
        streamed.decode_router_logits[layer + 1],
        permutedims(reference["decode_router_logits.$layer"], (2, 1));
        atol=router_atol,
        rtol=router_rtol,
    ))

    for (prefix, actual_logits, actual_active) in (
        (
            "",
            streamed.router_logits[layer + 1],
            streamed.active_experts[layer + 1],
        ),
        (
            "decode_",
            streamed.decode_router_logits[layer + 1],
            streamed.decode_active_experts[layer + 1],
        ),
    )
        selected_reference = Int.(reference["$(prefix)selected_experts.$layer"]) .+ 1
        weights_reference = reference["$(prefix)routing_weights.$layer"]
        routing = qwen3_topk_routing(
            actual_logits,
            streamed.model.experts_per_token;
            normalize=streamed.model.normalize_routing,
        )
        selected_equal_tokens = 0
        selected_set_equal_tokens = 0
        selected_overlap_slots = 0
        selected_slots = 0
        weight_max_abs = 0.0
        common_weight_max_abs = 0.0
        missing_weight_max = 0.0
        for token in axes(actual_logits, 2)
            selected = partialsortperm(
                view(actual_logits, :, token),
                1:streamed.model.experts_per_token;
                rev=true,
            )
            expected = vec(selected_reference[token, :])
            selected_equal_tokens += selected == expected
            actual_set = Set(selected)
            expected_set = Set(expected)
            selected_set_equal_tokens += actual_set == expected_set
            selected_overlap_slots += length(intersect(actual_set, expected_set))
            selected_slots += length(expected)
            actual_weights = routing[expected, token]
            weight_max_abs = max(
                weight_max_abs,
                maximum(abs.(actual_weights .- weights_reference[token, :])),
            )
            for (slot, expert) in enumerate(expected)
                if expert in actual_set
                    common_weight_max_abs = max(
                        common_weight_max_abs,
                        abs(actual_weights[slot] - weights_reference[token, slot]),
                    )
                else
                    missing_weight_max = max(
                        missing_weight_max,
                        weights_reference[token, slot],
                    )
                end
            end
        end
        expected_active = sort!(unique!(vec(selected_reference)))
        token_count = size(selected_reference, 1)
        overlap_fraction = selected_overlap_slots / selected_slots
        push!(routing_checks, (;
            name="$(prefix)routing.$layer",
            token_count,
            selected_equal_tokens,
            selected_set_equal_tokens,
            selected_overlap_slots,
            selected_slots,
            overlap_fraction,
            active_experts_equal=actual_active == expected_active,
            weight_max_abs=Float64(weight_max_abs),
            common_weight_max_abs=Float64(common_weight_max_abs),
            missing_weight_max=Float64(missing_weight_max),
        ))
    end
end

for (name, actual, expected, atol, rtol, check_argmax) in (
    (
        "final_hidden",
        streamed.final_hidden,
        hf_layout(reference["final_hidden"]),
        hidden_atol,
        hidden_rtol,
        false,
    ),
    (
        "decode_final_hidden",
        streamed.decode_final_hidden,
        hf_layout(reference["decode_final_hidden"]),
        hidden_atol,
        hidden_rtol,
        false,
    ),
    (
        "logits",
        streamed.logits,
        hf_layout(reference["logits"]),
        logits_atol,
        logits_rtol,
        true,
    ),
    (
        "decode_logits",
        streamed.decode_logits,
        hf_layout(reference["decode_logits"]),
        logits_atol,
        logits_rtol,
        true,
    ),
)
    push!(comparisons, comparison(
        name,
        actual,
        expected;
        atol,
        rtol,
        check_argmax,
    ))
end

comparison_passed = all(item.passed for item in comparisons)
selected_overlap_slots = sum(item.selected_overlap_slots for item in routing_checks)
selected_slots = sum(item.selected_slots for item in routing_checks)
routing_overlap_fraction = selected_overlap_slots / selected_slots
routing_exact = all(
    item.selected_set_equal_tokens == item.token_count &&
        item.active_experts_equal &&
        item.common_weight_max_abs <= routing_weight_atol
    for item in routing_checks
)
routing_passed = if is_bf16
    routing_overlap_fraction >= 0.95 &&
        all(item.overlap_fraction >= 0.75 for item in routing_checks) &&
        maximum(item.common_weight_max_abs for item in routing_checks) <=
            routing_weight_atol
else
    routing_exact
end
argmax_passed = all(
    item.argmax_equal !== false for item in comparisons
)
report = (;
    schema_version=1,
    model_id=String(metadata.model_id),
    revision=String(metadata.revision),
    checkpoint_source=model_dir,
    reference_source=reference_dir,
    reference_sha256=String(metadata.checksums.reference_sha256),
    prompt_tokens=length(tokens),
    decode_tokens=1,
    compute_dtype=compute_dtype_name,
    tolerances=(;
        hidden_atol,
        hidden_rtol,
        router_atol,
        router_rtol,
        logits_atol,
        logits_rtol,
        routing_weight_atol,
    ),
    runtime=(;
        seconds=timing.time,
        allocated_bytes=timing.bytes,
        maxrss_bytes=Sys.maxrss(),
    ),
    expert_reads=(;
        prompt=sum(length, streamed.active_experts),
        decode=sum(length, streamed.decode_active_experts),
        dense_per_pass=streamed.model.num_layers * streamed.model.num_experts,
    ),
    comparisons,
    routing_summary=(;
        selected_overlap_slots,
        selected_slots,
        overlap_fraction=routing_overlap_fraction,
        exact=routing_exact,
        passed=routing_passed,
    ),
    routing_checks,
    passed=comparison_passed && routing_passed && argmax_passed,
)

buffer = IOBuffer()
JSON3.pretty(buffer, report)
encoded = String(take!(buffer))
println(encoded)
if output_path !== nothing
    mkpath(dirname(output_path))
    write(output_path, encoded * "\n")
end
report.passed || error("Qwen3-30B-A3B real streamed parity failed")
