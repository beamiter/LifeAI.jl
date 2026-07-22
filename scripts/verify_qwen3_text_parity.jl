using LifeAI
using JSON3
using Statistics

function main()
length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_text_parity.jl MODEL_DIR REFERENCE_DIR",
)
model_dir = abspath(ARGS[1])
reference_dir = abspath(ARGS[2])
reference = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
revision = String(reference["revision"])

tokenizer = load_hf_qwen3_tokenizer(model_dir; revision)
println("revision\t", revision)
println("tokenizer_sha256\t", tokenizer.tokenizer_sha256)
println("tokenizer_cases\t", length(reference["tokenizer_cases"]))
for case in reference["tokenizer_cases"]
    expected = Int.(case["ids_0_based"])
    actual = encode(tokenizer, String(case["text"])) .- 1
    actual == expected || error("tokenizer mismatch in $(case["name"])")
    println("tokenizer\t", case["name"], "\tids=", length(actual), "\texact=true")
end

println("chat_cases\t", length(reference["chat_cases"]))
for case in reference["chat_cases"]
    messages = [
        (role=String(message["role"]), content=String(message["content"]))
        for message in case["messages"]
    ]
    prompt = apply_qwen3_chat_template(
        tokenizer,
        messages;
        add_generation_prompt=Bool(case["add_generation_prompt"]),
        enable_thinking=Bool(case["enable_thinking"]),
    )
    prompt == String(case["prompt"]) || error("chat rendering mismatch in $(case["name"])")
    encode(tokenizer, prompt) .- 1 == Int.(case["ids_0_based"]) ||
        error("chat token mismatch in $(case["name"])")
    println("chat\t", case["name"], "\texact=true")
end

load_stats = @timed load_hf_qwen3_bundle(
    model_dir;
    max_seq_len=64,
    revision,
)
bundle = load_stats.value
println("load_seconds\t", load_stats.time)
println("load_allocated_bytes\t", load_stats.bytes)
reference_logits = load_safetensors(joinpath(reference_dir, "reference.safetensors"))

global_max_abs = 0.0f0
global_mean_abs = Float32[]
for case in reference["generation_cases"]
    name = String(case["name"])
    prompt = String(case["prompt"])
    expected_new = Int.(case["generated_ids_0_based"])
    dynamic = generate_hf_text(
        bundle,
        prompt;
        cache=:dynamic,
        max_new_tokens=length(expected_new),
        capture_logits=true,
    )
    dynamic.generated_ids .- 1 == expected_new || error("generated ids mismatch in $name")
    dynamic.completion == String(case["completion"]) || error("completion mismatch in $name")
    for (actual_step, expected_step) in zip(dynamic.trace, case["steps"])
        expected = reference_logits[String(expected_step["logits_key"])]
        difference = abs.(actual_step.logits .- expected)
        max_abs = maximum(difference)
        mean_abs = mean(difference)
        global_max_abs = max(global_max_abs, max_abs)
        push!(global_mean_abs, mean_abs)
        argmax_equal = argmax(actual_step.logits) == argmax(expected)
        argmax_equal || error("argmax mismatch in $name step $(actual_step.step)")
        println(
            "generation\t", name,
            "\tstep=", actual_step.step,
            "\tmax_abs=", max_abs,
            "\tmean_abs=", mean_abs,
            "\targmax=true",
            "\tmargin=", actual_step.margin,
        )
    end
    if name == "chat_no_thinking"
        full = generate_hf_text(bundle, prompt; cache=:full, max_new_tokens=length(expected_new))
        static = generate_hf_text(bundle, prompt; cache=:static, max_new_tokens=length(expected_new))
        full.generated_ids == dynamic.generated_ids == static.generated_ids ||
            error("full/dynamic/static token mismatch")
        println("cache_matrix\tchat_no_thinking\texact=true\tcompletion=", repr(dynamic.completion))
    end
end
println("global_max_abs\t", global_max_abs)
println("mean_of_step_mean_abs\t", mean(global_mean_abs))
end

main()
