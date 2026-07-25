#!/usr/bin/env julia

using Reactant
using BFloat16s: BFloat16
using JSON3
using Statistics: mean
using LifeAI
using LifeAI: hf_token_ids

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_bf16_xla.jl MODEL_DIR REFERENCE_DIR",
)
model_dir, reference_dir = ARGS
Reactant.set_default_backend("gpu")

metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
String(metadata["compute_dtype"]) == "bfloat16" || error(
    "reference was not exported with --compute-dtype bfloat16",
)
reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))

tokens = reshape(
    hf_token_ids(Int.(collect(metadata["token_ids_0_based"])); vocab_size=151936),
    :,
    1,
)
expected_greedy_first = Int(first(metadata["greedy_token_ids_0_based"])) + 1

loaded = load_hf_qwen3_model(model_dir; max_seq_len=64, weight_dtype=BFloat16)
model = loaded.model
rope = first(values(model.blocks.layers)).attn.rope
cos_table = BFloat16.(rope.cos_cache)
sin_table = BFloat16.(rope.sin_cache)
mask = LifeAI._bf16a_causal_mask(size(tokens, 1), size(tokens, 1))

function prefill_logits(ps, cos_r, sin_r, mask_r)
    caches = Vector{Any}(undef, model.num_layers)
    fill!(caches, (nothing, nothing))
    result = LifeAI._bf16a_forward_pass(
        model, ps, tokens, caches, cos_r, sin_r, mask_r;
        start_pos=1,
    )
    return result.logits
end

ps_r = Reactant.to_rarray(loaded.parameters)
cos_r = Reactant.to_rarray(cos_table)
sin_r = Reactant.to_rarray(sin_table)
mask_r = Reactant.to_rarray(mask)

compile_timing = @timed @compile prefill_logits(ps_r, cos_r, sin_r, mask_r)
compiled = compile_timing.value
first_timing = @timed compiled(ps_r, cos_r, sin_r, mask_r)
logits_first = first_timing.value
steady_timing = @timed compiled(ps_r, cos_r, sin_r, mask_r)
logits = Float32.(Array(steady_timing.value))
logits_first_host = Float32.(Array(logits_first))
logits == logits_first_host || error("XLA executions are not deterministic")

hf_layout(array) = permutedims(array, (3, 2, 1))
expected = Float32.(hf_layout(reference["logits"]))
differences = abs.(logits .- expected)
next_token = argmax(vec(logits[:, end, 1]))

println("model_revision\t", metadata["revision"])
println(
    "dense_variant\t",
    loaded.variant === nothing ? "custom" : String(loaded.variant.variant),
)
println("compile_seconds\t", compile_timing.time)
println("first_exec_seconds\t", first_timing.time)
println("steady_exec_seconds\t", steady_timing.time)
println("logits_max_abs\t", maximum(differences))
println("logits_mean_abs\t", mean(differences))
println("logits_argmax_equal\t", argmax(vec(logits)) == argmax(vec(expected)))
println(
    "greedy_first_token_equal\t",
    next_token == expected_greedy_first,
    "\t",
    next_token - 1,
)
