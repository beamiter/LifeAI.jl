using LifeAI, JSON3, SHA

isempty(ARGS) && error("Usage: audit_lifeai.jl MODEL_DIR [CORPUS_DIR]")
model_dir = ARGS[1]
root = length(ARGS) >= 2 ? ARGS[2] : @__DIR__
tokenizer = LifeAI.load_hf_flint_tokenizer(model_dir)
manifest = JSON3.read(read(joinpath(root, "corpus-manifest.json"), String))
results = Dict{String, Any}[]
for case in manifest.cases
    prompt = read(joinpath(root, case.file), String)
    messages = [(role="user", content=prompt)]
    rendered = LifeAI.apply_flint_chat_template(tokenizer, messages)
    @assert rendered == "<|user|>" * prompt * "<|assistant|>"
    ids = LifeAI.encode(tokenizer, rendered)
    expected = Int.(reinterpret(UInt32, read(joinpath(root, case.ids_file)))) .+ 1
    @assert ids == expected "LifeAI and released-tokenizer IDs disagree"
    @assert length(ids) == Int(case.prompt_tokens_official_tokenizers)
    @assert length(ids) + Int(case.requested_output_tokens) == 262144
    result = Dict("file" => String(case.file), "prompt_tokens_lifeai" => length(ids),
        "requested_output_tokens" => Int(case.requested_output_tokens),
        "total_context_budget" => length(ids) + Int(case.requested_output_tokens),
        "prompt_utf8_bytes" => ncodeunits(prompt),
        "prompt_sha256" => bytes2hex(sha256(codeunits(prompt))),
        "all_ids_equal_official_zero_based_plus_one" => true,
        "template_equal_official_renderer" => true,
        "tokenizer_sha256" => bytes2hex(sha256(read(joinpath(model_dir, "tokenizer.json")))))
    push!(results, result)
    println(JSON3.write(result))
end
write(joinpath(root, "lifeai-tokenization-audit.json"), JSON3.write(results) * "\n")
