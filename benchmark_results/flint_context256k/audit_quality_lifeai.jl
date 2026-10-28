using LifeAI, JSON3, SHA
isempty(ARGS) && error("Usage: audit_quality_lifeai.jl MODEL_DIR [CORPUS_DIR] [MANIFEST_NAME] [AUDIT_NAME]")
model_dir = ARGS[1]
root = length(ARGS) >= 2 ? ARGS[2] : @__DIR__
tokenizer = LifeAI.load_hf_flint_tokenizer(model_dir)
manifest_name = length(ARGS) >= 3 ? ARGS[3] : "quality-manifest.json"
audit_name = length(ARGS) >= 4 ? ARGS[4] : "quality-lifeai-audit.json"
manifest = JSON3.read(read(joinpath(root, manifest_name), String))
results = Dict{String,Any}[]
for case in manifest.cases
    prompt = read(joinpath(root, case.file), String)
    rendered = LifeAI.apply_flint_chat_template(tokenizer, [(role="user", content=prompt)])
    @assert rendered == "<|user|>" * prompt * "<|assistant|>"
    ids = LifeAI.encode(tokenizer, rendered)
    expected = Int.(reinterpret(UInt32, read(joinpath(root, case.ids_file)))) .+ 1
    @assert ids == expected
    @assert length(ids) == Int(case.prompt_tokens_official_tokenizers)
    @assert length(ids) + Int(case.reserved_output_tokens) == Int(case.total_context_budget)
    result = Dict("file"=>String(case.file), "style"=>String(case.style),
        "prompt_tokens_lifeai"=>length(ids), "reserved_output_tokens"=>Int(case.reserved_output_tokens),
        "total_context_budget"=>Int(case.total_context_budget), "all_ids_match_official_plus_one"=>true,
        "template_matches_released_renderer"=>true, "sha256"=>bytes2hex(sha256(codeunits(prompt))))
    push!(results,result)
    println(JSON3.write(result)); flush(stdout)
    write(joinpath(root,audit_name),JSON3.write(results)*"\n")
end
