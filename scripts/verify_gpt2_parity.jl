#!/usr/bin/env julia

using JSON3
using LifeAI

length(ARGS) == 3 || error(
    "usage: julia --project=. scripts/verify_gpt2_parity.jl " *
    "MODEL_DIR REFERENCE_DIR OUTPUT_JSON",
)

model_dir = abspath(ARGS[1])
reference_dir = abspath(ARGS[2])
output_path = abspath(ARGS[3])
metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))
bundle = load_hf_gpt2_bundle(
    model_dir;
    revision=String(metadata.revision),
    max_seq_len=64,
)

tokenizer_cases = []
for item in metadata.corpus
    ids = encode(bundle.tokenizer, String(item.text); add_special_tokens=false)
    ids_passed = ids .- 1 == Int.(collect(item.ids))
    decoded = decode(bundle.tokenizer, ids; skip_special_tokens=false)
    decoded_skip = decode(bundle.tokenizer, ids; skip_special_tokens=true)
    bytes_hex = [bytes2hex(bundle.tokenizer.token_bytes[id]) for id in ids]
    pieces = hf_gpt2_pretokenize(bundle.tokenizer, String(item.text))
    spans_passed = [piece.symbols for piece in pieces] ==
        [String(piece.text) for piece in item.pretokenized] && [
            [piece.character_start, piece.character_stop] for piece in pieces
        ] == [Int.(collect(piece.offset)) for piece in item.pretokenized]
    push!(tokenizer_cases, (;
        text=String(item.text),
        ids_passed,
        decoded_passed=decoded == String(item.decoded),
        decoded_skip_special_passed=
            decoded_skip == String(item.decoded_skip_special),
        bytes_passed=bytes_hex == String.(collect(item.token_bytes_hex)),
        spans_passed,
    ))
end
tokenizer_passed = all(case ->
    case.ids_passed &&
    case.decoded_passed &&
    case.decoded_skip_special_passed &&
    case.bytes_passed &&
    case.spans_passed,
tokenizer_cases)

prompt_ids = hf_token_ids(
    Int.(collect(metadata.prompt_ids));
    vocab_size=bundle.model.vocab_size,
)
trace = hf_gpt2_forward_trace(
    bundle.model,
    reshape(prompt_ids, :, 1),
    bundle.parameters,
    bundle.states,
)
to_hf(array) = permutedims(array, (3, 2, 1))
embedding_error = maximum(abs.(to_hf(trace.embedding) .- reference["embedding"]))
block_errors = [
    maximum(abs.(
        to_hf(trace.blocks[index]) .-
        reference["block_" * lpad(string(index - 1), 2, "0")],
    )) for index in 1:bundle.model.num_layers
]
final_hidden_error = maximum(abs.(
    to_hf(trace.final_hidden) .- reference["final_hidden"],
))
logits_error = maximum(abs.(to_hf(trace.logits) .- reference["logits"]))
layer_passed = embedding_error == 0 &&
    maximum(block_errors) <= 5.0f-4 &&
    final_hidden_error <= 1.0f-4 &&
    logits_error <= 1.5f-4

expected_ids = Int.(collect(metadata.generated_ids))
expected_logits = reference["greedy_step_logits"]
generation = []
for mode in (:full, :dynamic, :static)
    result = generate_hf_text(
        bundle,
        String(metadata.prompt);
        cache=mode,
        strategy=:greedy,
        max_new_tokens=Int(metadata.greedy_steps),
        capture_logits=true,
    )
    step_errors = [
        maximum(abs.(
            result.trace[index].logits .-
            vec(@view(expected_logits[index, :])),
        )) for index in 1:Int(metadata.greedy_steps)
    ]
    push!(generation, (;
        mode=String(mode),
        ids_passed=result.generated_ids .- 1 == expected_ids,
        completion_passed=result.completion == String(metadata.completion),
        text_passed=result.text == String(metadata.text),
        step_max_abs=maximum(step_errors),
        step_errors,
    ))
end
generation_passed = all(item ->
    item.ids_passed &&
    item.completion_passed &&
    item.text_passed &&
    item.step_max_abs <= 1.5f-4,
generation)
passed = tokenizer_passed && layer_passed && generation_passed

report = (;
    schema_version=1,
    model_id=String(metadata.model_id),
    revision=String(metadata.revision),
    model_source=bundle.source,
    reference_source=reference_dir,
    transformers_versions=metadata.versions,
    tokenizer=(;
        passed=tokenizer_passed,
        case_count=length(tokenizer_cases),
        cases=tokenizer_cases,
    ),
    forward=(;
        passed=layer_passed,
        embedding_max_abs=embedding_error,
        block_max_abs=block_errors,
        final_hidden_max_abs=final_hidden_error,
        logits_max_abs=logits_error,
        tolerance=(;
            embedding=0.0,
            block=5.0e-4,
            final_hidden=1.0e-4,
            logits=1.5e-4,
        ),
    ),
    generation=(;
        passed=generation_passed,
        steps=Int(metadata.greedy_steps),
        modes=generation,
    ),
    passed,
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println("wrote $output_path; passed=$passed")
passed || error("GPT-2 parity verification failed")
