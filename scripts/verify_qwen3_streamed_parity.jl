#!/usr/bin/env julia

using JSON3
using Lux
using Statistics: mean
using LifeAI
using LifeAI: qwen3_dense_parameter_count

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_streamed_parity.jl MODEL_DIR REFERENCE_DIR",
)
model_dir, reference_dir = ARGS

metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))

token_ids = hf_token_ids(Int.(collect(metadata["token_ids_0_based"])); vocab_size=151936)
tokens = reshape(token_ids, :, 1)
decode_token = hf_token_ids([Int(metadata["decode_token_id_0_based"])]; vocab_size=151936)

GC.gc()
streamed_timing = @timed stream_hf_qwen3_forward(
    model_dir,
    tokens;
    decode_token,
    max_seq_len=64,
)
streamed = streamed_timing.value

hf_layout(array) = permutedims(array, (3, 2, 1))

function report(name, actual, expected)
    differences = abs.(actual .- expected)
    println(join((
        name,
        string(maximum(differences)),
        string(mean(differences)),
        string(argmax(vec(actual)) == argmax(vec(expected))),
    ), '\t'))
end

println("model_revision\t", metadata["revision"])
println(
    "dense_variant\t",
    streamed.variant === nothing ? "custom" : String(streamed.variant.variant),
)
println(
    "parameter_count\t",
    streamed.variant === nothing ?
        Lux.parameterlength(streamed.model) :
        qwen3_dense_parameter_count(streamed.variant),
)
println("transformers_version\t", metadata["transformers_version"])
println("torch_version\t", metadata["torch_version"])
println("streamed_seconds\t", streamed_timing.time)
println("streamed_allocated_bytes\t", streamed_timing.bytes)
println("maxrss_bytes\t", Sys.maxrss())
println("stage\tmax_abs\tmean_abs\targmax_equal")
report("embedding", streamed.embedding, hf_layout(reference["embedding"]))
for layer in 0:(streamed.model.num_layers - 1)
    report(
        "block.$layer",
        streamed.blocks[layer + 1],
        hf_layout(reference["block.$layer"]),
    )
end
report("final_hidden", streamed.final_hidden, hf_layout(reference["final_hidden"]))
report("logits", streamed.logits, hf_layout(reference["logits"]))
report("dynamic_decode", streamed.decode_logits, hf_layout(reference["decode_logits"]))
