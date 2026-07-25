#!/usr/bin/env julia

using BFloat16s: BFloat16
using JSON3
using Lux
using Statistics: mean
using LifeAI
using LifeAI: qwen3_dense_parameter_count

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_bf16_parity.jl MODEL_DIR REFERENCE_DIR",
)
model_dir, reference_dir = ARGS

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
decode_token = hf_token_ids(
    [Int(metadata["decode_token_id_0_based"])];
    vocab_size=151936,
)
expected_greedy = Int.(collect(metadata["greedy_token_ids_0_based"])) .+ 1

GC.gc()
load_timing = @timed load_hf_qwen3_model(
    model_dir;
    max_seq_len=64,
    weight_dtype=BFloat16,
)
loaded = load_timing.value
forward_timing = @timed hf_qwen3_bf16_forward(
    loaded.model,
    loaded.parameters,
    tokens;
    decode_token,
    greedy_steps=length(expected_greedy),
)
result = forward_timing.value

hf_layout(array) = permutedims(array, (3, 2, 1))

function report(name, actual, expected)
    differences = abs.(Float32.(actual) .- Float32.(expected))
    println(join((
        name,
        string(maximum(differences)),
        string(mean(differences)),
        string(argmax(vec(Float32.(actual))) == argmax(vec(Float32.(expected)))),
    ), '\t'))
end

println("model_revision\t", metadata["revision"])
println(
    "dense_variant\t",
    loaded.variant === nothing ? "custom" : String(loaded.variant.variant),
)
println(
    "parameter_count\t",
    loaded.variant === nothing ?
        Lux.parameterlength(loaded.model) :
        qwen3_dense_parameter_count(loaded.variant),
)
println("transformers_version\t", metadata["transformers_version"])
println("torch_version\t", metadata["torch_version"])
println("load_seconds\t", load_timing.time)
println("forward_seconds\t", forward_timing.time)
println("parameter_tree_bytes\t", Base.summarysize(loaded.parameters))
println("maxrss_bytes\t", Sys.maxrss())
println(
    "greedy_match\t",
    result.greedy_tokens == expected_greedy,
    "\t",
    join(result.greedy_tokens .- 1, ','),
)
println("stage\tmax_abs\tmean_abs\targmax_equal")
report("embedding", result.embedding, hf_layout(reference["embedding"]))
for layer in 0:(loaded.model.num_layers - 1)
    report(
        "block.$layer",
        result.blocks[layer + 1],
        hf_layout(reference["block.$layer"]),
    )
end
report("final_hidden", result.final_hidden, hf_layout(reference["final_hidden"]))
report("logits", result.logits, hf_layout(reference["logits"]))
report("dynamic_decode", result.decode_logits, hf_layout(reference["decode_logits"]))
