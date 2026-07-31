#!/usr/bin/env julia

using BFloat16s: BFloat16
using JSON3
using LinearAlgebra: BLAS, norm
using SHA

const WEEK22_VERIFICATION_DEVICE = Symbol(lowercase(get(
    ENV,
    "LIFEAI_WEEK22_EMBEDDING_DEVICE",
    "cpu",
)))
WEEK22_VERIFICATION_DEVICE in (:cpu, :cuda) || error(
    "LIFEAI_WEEK22_EMBEDDING_DEVICE must be cpu or cuda",
)
if WEEK22_VERIFICATION_DEVICE === :cuda
    using LuxCUDA
    using CUDA
end
using LifeAI

length(ARGS) in (2, 3) || error(
    "usage: julia --project=. scripts/verify_qwen3_embedding_parity.jl " *
    "MODEL_DIR REFERENCE_JSON [OUTPUT_JSON]",
)

model_dir = abspath(ARGS[1])
reference_path = abspath(ARGS[2])
output_path = length(ARGS) == 3 ?
    abspath(ARGS[3]) :
    joinpath(
        dirname(@__DIR__),
        "benchmark_results",
        "week22",
        "qwen3_embedding_0_6b_$(WEEK22_VERIFICATION_DEVICE).json",
    )

reference = JSON3.read(read(reference_path, String))
spec = qwen3_embedding_spec()
BLAS.set_num_threads(min(8, Sys.CPU_THREADS))
String(reference["revision"]) == spec.revision || error(
    "reference revision does not match the frozen Week 22 spec",
)
String(reference["compute_dtype"]) == "bfloat16" || error(
    "Week 22 verification requires the BF16 reference",
)

function columns(raw, ::Type{T}) where {T}
    return reduce(hcat, [T.(collect(row)) for row in raw])
end

function normalized_prefix(embeddings, dimension)
    output = copy(embeddings[1:dimension, :])
    for batch in axes(output, 2)
        output[:, batch] ./= norm(view(output, :, batch))
    end
    return output
end

function stable_topk(scores)
    return [
        sortperm(
            eachindex(view(scores, query, :));
            by=document -> (-scores[query, document], document),
        )
        for query in axes(scores, 1)
    ]
end

asset_timing = @timed verify_qwen3_embedding_assets(model_dir)
load_timing = @timed load_hf_qwen3_embedding_bundle(
    model_dir;
    max_seq_len=Int(reference["max_length"]),
    weight_dtype=BFloat16,
)
bundle = load_timing.value
upload_seconds = 0.0
gpu_parameter_bytes = 0
if WEEK22_VERIFICATION_DEVICE === :cuda
    CUDA.functional() || error(
        "Week 22 CUDA verification requested but CUDA.jl is not functional",
    )
    free_before_upload = Int(CUDA.free_memory())
    upload_timing = @timed begin
        parameters = CUDA.cu(bundle.parameters)
        CUDA.synchronize()
        parameters
    end
    bundle = merge(bundle, (; parameters=upload_timing.value))
    upload_seconds = upload_timing.time
    gpu_parameter_bytes = free_before_upload - Int(CUDA.free_memory())
end
texts = String.(collect(reference["texts"]))
forward_timing = @timed begin
    embedded = embed_texts(
        bundle,
        texts;
        dimension=bundle.model.d_model,
        max_length=Int(reference["max_length"]),
        padding_side=:left,
    )
    WEEK22_VERIFICATION_DEVICE === :cuda && CUDA.synchronize()
    embedded
end
result = forward_timing.value
warm_forward_seconds = 0.0
cold_warm_embedding_max_abs = 0.0f0
if WEEK22_VERIFICATION_DEVICE === :cuda
    warm_forward_timing = @timed begin
        embedded = embed_texts(
            bundle,
            texts;
            dimension=bundle.model.d_model,
            max_length=Int(reference["max_length"]),
            padding_side=:left,
        )
        CUDA.synchronize()
        embedded
    end
    warm_forward_seconds = warm_forward_timing.time
    cold_warm_embedding_max_abs = maximum(abs.(
        result.embeddings .- warm_forward_timing.value.embeddings,
    ))
    result = warm_forward_timing.value
end

expected_tokens = columns(reference["input_ids"], Int)
expected_mask = Bool.(columns(reference["attention_mask"], Int))
token_ids_equal = result.tokens .- 1 == expected_tokens
attention_mask_equal = result.attention_mask == expected_mask

query_count = length(reference["queries"])
dimension_results = Dict{String,Any}()
for dimension in (1024, 512, 256, 128, 64)
    key = string(dimension)
    actual_embeddings = normalized_prefix(result.embeddings, dimension)
    expected_embeddings = columns(reference["embeddings"][key], Float32)
    embedding_max_abs = maximum(abs.(actual_embeddings .- expected_embeddings))
    actual_scores = qwen3_embedding_similarity(
        actual_embeddings[:, 1:query_count],
        actual_embeddings[:, (query_count + 1):end],
    )
    expected_scores = Float32.(
        reduce(
            hcat,
            [
                Float32.(collect(row)) for
                row in reference["similarities"][key]
            ],
        )',
    )
    similarity_max_abs = maximum(abs.(actual_scores .- expected_scores))
    actual_topk = stable_topk(actual_scores)
    expected_topk = [
        Int.(collect(row)) .+ 1 for
        row in reference["top_k_document_indices_0_based"][key]
    ]
    topk_equal = actual_topk == expected_topk
    embedding_passed = embedding_max_abs <=
        Float64(reference["tolerances"]["embedding_max_abs"])
    similarity_passed = similarity_max_abs <=
        Float64(reference["tolerances"]["similarity_max_abs"])
    dimension_results[key] = Dict(
        "embedding_max_abs" => embedding_max_abs,
        "embedding_passed" => embedding_passed,
        "similarity_max_abs" => similarity_max_abs,
        "similarity_passed" => similarity_passed,
        "topk_equal" => topk_equal,
        "topk_document_indices_1_based" => actual_topk,
    )
end
all_embeddings_passed = all(
    result["embedding_passed"] for result in values(dimension_results)
)
all_similarities_passed = all(
    result["similarity_passed"] for result in values(dimension_results)
)
all_topk_passed = all(
    result["topk_equal"] for result in values(dimension_results)
)

documents = String.(collect(reference["documents"]))
memory = Qwen3SemanticMemory(
    documents,
    result.embeddings[:, (query_count + 1):end],
    Any[(; source="week22-reference", index) for index in eachindex(documents)],
)
memory_topk = [
    [
        hit.index for hit in retrieve_qwen3_semantic_memory(
            memory,
            result.embeddings[:, query];
            top_k=length(documents),
        )
    ]
    for query in 1:query_count
]
expected_full_topk = [
    Int.(collect(row)) .+ 1 for
    row in reference["top_k_document_indices_0_based"]["1024"]
]
memory_topk_equal = memory_topk == expected_full_topk
expected_first = Int.(
    collect(reference["expected_first_document_indices_0_based"]),
) .+ 1
memory_first_equal = first.(memory_topk) == expected_first

closed = token_ids_equal &&
    attention_mask_equal &&
    all_embeddings_passed &&
    all_similarities_passed &&
    all_topk_passed &&
    memory_topk_equal &&
    memory_first_equal &&
    cold_warm_embedding_max_abs == 0

hardware = Dict{String,Any}(
    "cpu_model" => first(Sys.cpu_info()).model,
    "logical_threads" => Sys.CPU_THREADS,
    "julia_threads" => Threads.nthreads(),
    "blas_threads" => BLAS.get_num_threads(),
    "machine" => Sys.MACHINE,
    "julia_version" => string(VERSION),
)
if WEEK22_VERIFICATION_DEVICE === :cuda
    device = CUDA.device()
    hardware["gpu_name"] = CUDA.name(device)
    hardware["compute_capability"] = string(CUDA.capability(device))
    hardware["cuda_driver_api_version"] = string(CUDA.driver_version())
    hardware["cuda_runtime_version"] = string(CUDA.runtime_version())
    hardware["gpu_total_bytes"] = Int(CUDA.total_memory())
    hardware["gpu_free_bytes_after_forward"] = Int(CUDA.free_memory())
    hardware["gpu_parameter_bytes"] = gpu_parameter_bytes
    hardware["nvidia_driver_version"] = strip(read(
        `nvidia-smi --query-gpu=driver_version --format=csv,noheader`,
        String,
    ))
end

report = Dict{String,Any}(
    "schema_version" => 1,
    "closed" => closed,
    "model_id" => spec.model_id,
    "revision" => spec.revision,
    "model_dir" => model_dir,
    "compute_dtype" => "bfloat16",
    "device" => string(WEEK22_VERIFICATION_DEVICE),
    "parameter_count" => qwen3_embedding_parameter_count(),
    "hardware" => hardware,
    "token_ids_equal" => token_ids_equal,
    "attention_mask_equal" => attention_mask_equal,
    "dimensions" => dimension_results,
    "semantic_memory" => Dict(
        "topk_equal" => memory_topk_equal,
        "first_document_equal" => memory_first_equal,
        "topk_document_indices_1_based" => memory_topk,
    ),
    "timing" => Dict(
        "asset_verification_seconds" => asset_timing.time,
        "load_seconds" => load_timing.time,
        "upload_seconds" => upload_seconds,
        "forward_seconds" => forward_timing.time,
        "warm_forward_seconds" => warm_forward_seconds,
        "cold_warm_embedding_max_abs" => cold_warm_embedding_max_abs,
        "batch_size" => length(texts),
        "padded_tokens" => length(result.tokens),
        "valid_tokens" => count(result.attention_mask),
        "max_rss_mib" => Sys.maxrss() / 1024^2,
    ),
    "reference" => Dict(
        "path" => reference_path,
        "sha256" => bytes2hex(sha256(read(reference_path))),
        "torch_version" => String(reference["versions"]["torch"]),
        "transformers_version" =>
            String(reference["versions"]["transformers"]),
    ),
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    println(io)
end
println(JSON3.write(report))
closed || error("Week 22 Qwen3 embedding parity acceptance failed")
