#!/usr/bin/env julia

using JSON3
using LifeAI

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_sampling_parity.jl MODEL_DIR REFERENCE_DIR",
)

model_dir = abspath(ARGS[1])
reference_dir = abspath(ARGS[2])
reference = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
reference_tensors = load_safetensors(joinpath(reference_dir, "reference.safetensors"))
uniforms = Float32.(reference["uniforms"])
prompt = String(reference["prompt"])
bundle = load_hf_qwen3_bundle(
    model_dir;
    max_seq_len=length(reference["prompt_ids_0_based"]) + length(uniforms),
    revision=String(reference["revision"]),
)
bundle.tokenizer.generation_config_sha256 == String(reference["generation_config_sha256"]) ||
    error("generation_config.json checksum differs from the reference")

result = generate_hf_text(
    bundle,
    prompt;
    cache=:dynamic,
    max_new_tokens=length(uniforms),
    strategy=:config,
    sample_uniforms=uniforms,
    capture_logits=true,
    capture_distribution=true,
)

actual_ids = result.generated_ids .- 1
expected_ids = Int.(reference["generated_ids_0_based"])
actual_ids == expected_ids || error(
    "sampled token mismatch: actual=$actual_ids expected=$expected_ids",
)
String(result.stop_reason) == String(reference["stop_reason"]) || error(
    "stop reason mismatch: actual=$(result.stop_reason) expected=$(reference["stop_reason"])",
)
result.completion == String(reference["completion"]) || error("completion text mismatch")

raw_max_abs = 0.0f0
filtered_max_abs = 0.0f0
probability_max_abs = 0.0f0
for (actual, expected) in zip(result.trace, reference["steps"])
    expected_candidates = Int.(expected["candidate_ids_0_based"])
    actual.distribution.hf_token_ids == expected_candidates || error(
        "candidate mask mismatch at step $(actual.step)",
    )
    expected_logits = reference_tensors[String(expected["logits_key"])]
    expected_filtered = reference_tensors[String(expected["filtered_logits_key"])]
    expected_probabilities = reference_tensors[String(expected["probabilities_key"])]
    raw_error = maximum(abs.(actual.logits .- expected_logits))
    filtered_error = maximum(abs.(actual.distribution.logits .- expected_filtered))
    probability_error = maximum(abs.(
        actual.distribution.probabilities .- expected_probabilities,
    ))
    global raw_max_abs = max(raw_max_abs, raw_error)
    global filtered_max_abs = max(filtered_max_abs, filtered_error)
    global probability_max_abs = max(probability_max_abs, probability_error)
    isapprox(actual.logits, expected_logits; atol=5.0f-3, rtol=5.0f-4) || error(
        "raw logits mismatch at step $(actual.step)",
    )
    isapprox(actual.distribution.logits, expected_filtered; atol=5.0f-3, rtol=5.0f-4) ||
        error("filtered logits mismatch at step $(actual.step)")
    isapprox(
        actual.distribution.probabilities,
        expected_probabilities;
        atol=1.0f-5,
        rtol=1.0f-4,
    ) || error("sampling probability mismatch at step $(actual.step)")
    println(
        "step=$(actual.step) token=$(actual.hf_token_id) uniform=$(actual.sample_uniform) " *
        "candidates=$(actual.candidate_count) raw_max_abs=$(raw_error) " *
        "probability_max_abs=$(probability_error)",
    )
end

println("sampled_ids_0_based=$(actual_ids)")
println("completion=$(repr(result.completion))")
println("raw_logits_global_max_abs=$raw_max_abs")
println("filtered_logits_global_max_abs=$filtered_max_abs")
println("probability_global_max_abs=$probability_max_abs")
