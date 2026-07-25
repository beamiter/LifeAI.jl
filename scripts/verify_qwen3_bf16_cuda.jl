#!/usr/bin/env julia

using LuxCUDA
using CUDA
using MLDataDevices
using BFloat16s: BFloat16
using JSON3
using Statistics: mean
using LifeAI
using LifeAI: hf_token_ids, qwen3_dense_parameter_count

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_bf16_cuda.jl MODEL_DIR REFERENCE_DIR",
)
model_dir, reference_dir = ARGS
CUDA.functional() || error("CUDA.jl is not functional on this machine")

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

loaded = load_hf_qwen3_model(model_dir; max_seq_len=64, weight_dtype=BFloat16)
device = gpu_device()
free_before = CUDA.free_memory()
upload_timing = @timed device(loaded.parameters)
ps_gpu = upload_timing.value
CUDA.synchronize()
tree_bytes = free_before - CUDA.free_memory()

cold_timing = @timed begin
    result = hf_qwen3_bf16_accel_forward(
        loaded.model,
        ps_gpu,
        tokens;
        decode_token,
        greedy_steps=length(expected_greedy),
    )
    CUDA.synchronize()
    result
end
result = cold_timing.value

warm_timing = @timed begin
    warm = hf_qwen3_bf16_accel_forward(
        loaded.model,
        ps_gpu,
        tokens;
        greedy_steps=length(expected_greedy),
    )
    CUDA.synchronize()
    warm
end
warm = warm_timing.value
warm.greedy_tokens == result.greedy_tokens || error(
    "warm greedy run diverged from cold run",
)

hf_layout(array) = permutedims(array, (3, 2, 1))

function report(name, actual, expected)
    values = Float32.(Array(actual))
    differences = abs.(values .- Float32.(expected))
    println(join((
        name,
        string(maximum(differences)),
        string(mean(differences)),
        string(argmax(vec(values)) == argmax(vec(Float32.(expected)))),
    ), '\t'))
end

println("model_revision\t", metadata["revision"])
println(
    "dense_variant\t",
    loaded.variant === nothing ? "custom" : String(loaded.variant.variant),
)
println("gpu_name\t", CUDA.name(CUDA.device()))
println("transformers_version\t", metadata["transformers_version"])
println("torch_version\t", metadata["torch_version"])
println("gpu_tree_bytes\t", tree_bytes)
println("upload_seconds\t", upload_timing.time)
println("cold_seconds\t", cold_timing.time)
println("warm_greedy_seconds\t", warm_timing.time)
println(
    "warm_tokens_per_second\t",
    length(expected_greedy) / warm_timing.time,
)
println("vram_used_bytes\t", CUDA.total_memory() - CUDA.free_memory())
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
