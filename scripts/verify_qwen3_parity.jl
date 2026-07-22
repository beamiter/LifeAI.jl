#!/usr/bin/env julia

using JSON3
using Statistics: mean
using LifeAI

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_parity.jl MODEL_DIR REFERENCE_DIR",
)
model_dir, reference_dir = ARGS

metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))

GC.gc()
loaded_timing = @timed load_hf_qwen3_model(model_dir; max_seq_len=64)
loaded = loaded_timing.value
tokens = reshape(hf_token_ids(
    Int.(collect(metadata["token_ids_0_based"]));
    vocab_size=loaded.model.vocab_size,
), :, 1)
trace = hf_qwen3_forward_trace(
    loaded.model,
    tokens,
    loaded.parameters,
    loaded.states,
)

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
println("transformers_version\t", metadata["transformers_version"])
println("torch_version\t", metadata["torch_version"])
println("load_seconds\t", loaded_timing.time)
println("load_allocated_bytes\t", loaded_timing.bytes)
println("stage\tmax_abs\tmean_abs\targmax_equal")
report("embedding", trace.embedding, hf_layout(reference["embedding"]))
for layer in 0:(loaded.model.num_layers - 1)
    report(
        "block.$layer",
        trace.blocks[layer + 1],
        hf_layout(reference["block.$layer"]),
    )
end
report("final_hidden", trace.final_hidden, hf_layout(reference["final_hidden"]))
report("logits", trace.logits, hf_layout(reference["logits"]))

decode_token = hf_token_ids(
    [Int(metadata["decode_token_id_0_based"])];
    vocab_size=loaded.model.vocab_size,
)
expected_decode = hf_layout(reference["decode_logits"])

dynamic_cache = init_kv_cache(loaded.model; batch_size=1)
_, dynamic_cache, dynamic_state = prefill(
    loaded.model,
    loaded.parameters,
    loaded.states,
    tokens,
    dynamic_cache,
)
dynamic_logits, _, _ = decode_step(
    loaded.model,
    loaded.parameters,
    dynamic_state,
    decode_token,
    dynamic_cache,
)
report("dynamic_decode", dynamic_logits, expected_decode)

static_cache = init_static_kv_cache(loaded.model; batch_size=1)
_, static_cache, static_state = prefill(
    loaded.model,
    loaded.parameters,
    loaded.states,
    tokens,
    static_cache,
)
static_logits, _, _ = decode_step(
    loaded.model,
    loaded.parameters,
    static_state,
    decode_token,
    static_cache,
)
report("static_decode", static_logits, expected_decode)
