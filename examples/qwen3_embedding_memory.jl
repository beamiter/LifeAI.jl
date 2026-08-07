using BFloat16s: BFloat16
using LifeAI

length(ARGS) in (1, 2) || error(
    "usage: julia --project=. examples/qwen3_embedding_memory.jl " *
    "MODEL_DIR [QUERY]",
)

model_dir = abspath(ARGS[1])
query = length(ARGS) == 2 ?
    ARGS[2] :
    "项目里的 embedding 和 semantic memory 是在哪一章实现的？"

note_paths = [
    joinpath(
        @__DIR__, "..", "notes", "episodes",
        "episode05_deployment_memory_and_sampling",
        "chapter22_qwen3_embedding_memory.md",
    ),
    joinpath(
        @__DIR__, "..", "notes", "episodes",
        "episode05_deployment_memory_and_sampling",
        "chapter21_qwen3_xla_resident_service.md",
    ),
    joinpath(
        @__DIR__, "..", "notes", "episodes",
        "episode04_efficient_inference_and_quantization",
        "chapter17_qwen3_calibrated_int4.md",
    ),
    joinpath(
        @__DIR__, "..", "notes", "episodes",
        "episode01_transformer_and_training_foundations",
        "chapter05_tokenizer_data_pipeline.md",
    ),
]
function note_excerpt(path)
    text = read(path, String)
    return first(text, min(800, length(text)))
end
documents = note_excerpt.(note_paths)
metadata = [
    (;
        path=relpath(path, joinpath(@__DIR__, "..")),
        title=first(split(document, '\n')),
    )
    for (path, document) in zip(note_paths, documents)
]

bundle = load_hf_qwen3_embedding_bundle(
    model_dir;
    max_seq_len=256,
    weight_dtype=BFloat16,
)
memory = build_qwen3_semantic_memory(
    bundle,
    documents;
    metadata,
    dimension=512,
    max_length=256,
)
results = search_qwen3_semantic_memory(
    bundle,
    memory,
    query;
    top_k=3,
    max_length=256,
)

println("query: ", query)
for result in results
    println(
        result.rank,
        ". score=",
        round(result.score; digits=4),
        " ",
        result.metadata.path,
        " — ",
        result.metadata.title,
    )
end
