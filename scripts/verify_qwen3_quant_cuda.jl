#!/usr/bin/env julia

using LuxCUDA
using CUDA
using BFloat16s: BFloat16
using JSON3
using Statistics: mean
using LifeAI
using LifeAI: hf_token_ids

length(ARGS) == 3 || error(
    "usage: julia --project=. scripts/verify_qwen3_quant_cuda.jl MODEL_DIR REFERENCE_DIR SCHEME",
)
model_dir, reference_dir, scheme_name = ARGS
scheme = Symbol(scheme_name)
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

GC.gc()
load_timing = @timed load_hf_qwen3_quantized(model_dir; max_seq_len=64, scheme)
loaded = load_timing.value
free_before = CUDA.free_memory()
ps_gpu = CUDA.cu(loaded.parameters)
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
warm_timing.value.greedy_tokens == result.greedy_tokens ||
    error("warm greedy diverged from cold run")

hf_layout(array) = permutedims(array, (3, 2, 1))
to_host(x) = Float32.(Array(x))
reference_logits = Float32.(hf_layout(reference["logits"]))
reference_decode = Float32.(hf_layout(reference["decode_logits"]))
logits = to_host(result.logits)
decode_logits = to_host(result.decode_logits)

matches = sum(result.greedy_tokens .== expected_greedy)
first_divergence = findfirst(result.greedy_tokens .!= expected_greedy)

println("model_revision\t", metadata["revision"])
println(
    "dense_variant\t",
    loaded.variant === nothing ? "custom" : String(loaded.variant.variant),
)
println("scheme\t", scheme)
println("gpu_name\t", CUDA.name(CUDA.device()))
println("load_seconds\t", load_timing.time)
println("host_maxrss_bytes\t", Sys.maxrss())
println("gpu_tree_bytes\t", tree_bytes)
println("vram_used_bytes\t", CUDA.total_memory() - CUDA.free_memory())
println("cold_seconds\t", cold_timing.time)
println("warm_greedy_seconds\t", warm_timing.time)
println(
    "warm_tokens_per_second\t",
    length(expected_greedy) / warm_timing.time,
)
println("logits_max_abs\t", maximum(abs.(logits .- reference_logits)))
println("logits_mean_abs\t", mean(abs.(logits .- reference_logits)))
println(
    "logits_argmax_equal\t",
    argmax(vec(logits)) == argmax(vec(reference_logits)),
)
println("decode_max_abs\t", maximum(abs.(decode_logits .- reference_decode)))
println(
    "decode_argmax_equal\t",
    argmax(vec(decode_logits)) == argmax(vec(reference_decode)),
)
println(
    "greedy_match\t",
    result.greedy_tokens == expected_greedy,
    "\t",
    join(result.greedy_tokens .- 1, ','),
)
println("greedy_agreement\t", matches, "/", length(expected_greedy))
println(
    "greedy_first_divergence\t",
    first_divergence === nothing ? "none" : string(first_divergence),
)
