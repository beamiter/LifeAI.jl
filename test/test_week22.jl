using Test
using BFloat16s: BFloat16
using JSON3
using LinearAlgebra: norm
using LifeAI:
    QWEN3_EMBEDDING_RETRIEVAL_INSTRUCTION,
    Qwen3SemanticMemory,
    embed_texts,
    hf_qwen3_embedding_forward,
    load_hf_qwen3_embedding_bundle,
    load_hf_qwen3_embedding_config,
    load_hf_qwen3_embedding_tokenizer,
    load_hf_qwen3_model,
    load_tokenizer,
    prepare_qwen3_embedding_inputs,
    qwen3_embedding_parameter_count,
    qwen3_embedding_query,
    qwen3_embedding_similarity,
    qwen3_embedding_spec,
    qwen3_last_token_pool,
    retrieve_qwen3_semantic_memory,
    save_tokenizer,
    tokenizer_fingerprint,
    verify_qwen3_embedding_assets

isdefined(@__MODULE__, :week08_tokenizer_payloads) ||
    include("week08_fixture.jl")
isdefined(@__MODULE__, :_week16_fixture_dir) ||
    include("qwen3_bf16_fixture.jl")

const _WEEK22_FIXTURE_DIR = joinpath(
    @__DIR__,
    "fixtures",
    "week22_qwen3_embedding",
)
const _WEEK22_CONFIG_PATH = joinpath(_WEEK22_FIXTURE_DIR, "config.json")
const _WEEK22_ASSETS_PATH = joinpath(_WEEK22_FIXTURE_DIR, "assets.json")
const _WEEK22_REFERENCE_PATH = joinpath(_WEEK22_FIXTURE_DIR, "reference.json")
const _WEEK22_CUDA_REPORT_PATH = joinpath(
    dirname(@__DIR__),
    "benchmark_results",
    "week22",
    "qwen3_embedding_0_6b_cuda.json",
)

function _week22_embedding_tokenizer_payloads()
    payloads = week08_tokenizer_payloads()
    payloads.tokenizer["post_processor"] = Dict(
        "type" => "Sequence",
        "processors" => Any[
            Dict(
                "type" => "ByteLevel",
                "add_prefix_space" => false,
                "trim_offsets" => false,
                "use_regex" => false,
            ),
            Dict(
                "type" => "TemplateProcessing",
                "single" => Any[
                    Dict("Sequence" => Dict("id" => "A", "type_id" => 0)),
                    Dict("SpecialToken" => Dict(
                        "id" => "<|endoftext|>",
                        "type_id" => 0,
                    )),
                ],
                "pair" => Any[
                    Dict("Sequence" => Dict("id" => "A", "type_id" => 0)),
                    Dict("Sequence" => Dict("id" => "B", "type_id" => 0)),
                    Dict("SpecialToken" => Dict(
                        "id" => "<|endoftext|>",
                        "type_id" => 0,
                    )),
                ],
                "special_tokens" => Dict(
                    "<|endoftext|>" => Dict(
                        "id" => "<|endoftext|>",
                        "ids" => [258],
                        "tokens" => ["<|endoftext|>"],
                    ),
                ),
            ),
        ],
    )
    generation = Dict{String,Any}(
        "bos_token_id" => 258,
        "eos_token_id" => 258,
        "max_new_tokens" => 2048,
        "transformers_version" => "4.51.3",
    )
    return merge(payloads, (; generation_config=generation))
end

function _week22_embedding_tokenizer_fixture(directory)
    return write_week08_tokenizer_fixture(
        directory;
        payloads=_week22_embedding_tokenizer_payloads(),
    )
end

@testset "Qwen3 embedding frozen config contract" begin
    spec = qwen3_embedding_spec()
    @test spec.variant === :qwen3_embedding_0_6b
    @test spec.model_id == "Qwen/Qwen3-Embedding-0.6B"
    @test spec.revision ==
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    @test spec.vocab_size == 151_669
    @test spec.max_position_embeddings == 32_768
    @test spec.minimum_dimension == 32
    @test qwen3_embedding_parameter_count() == 595_776_512

    manifest = JSON3.read(read(_WEEK22_ASSETS_PATH, String))
    @test manifest["model_id"] == spec.model_id
    @test manifest["revision"] == spec.revision
    @test length(manifest["assets"]) == 8
    expected_hashes = Dict(
        String(asset["path"]) => String(asset["sha256"])
        for asset in manifest["assets"]
    )
    @test expected_hashes["config.json"] == spec.config_sha256
    @test expected_hashes["tokenizer.json"] == spec.tokenizer_sha256
    @test expected_hashes["model.safetensors"] == spec.model_sha256

    mapped = LifeAI._qwen3_embedding_base_state_dict(Dict(
        "embed_tokens.weight" => ones(Float32, 2, 2),
        "norm.weight" => ones(Float32, 2),
    ))
    @test Set(keys(mapped)) ==
        Set(["model.embed_tokens.weight", "model.norm.weight"])
    @test_throws ArgumentError LifeAI._qwen3_embedding_base_state_dict(Dict(
        "model.embed_tokens.weight" => ones(Float32, 2, 2),
    ))

    config = load_hf_qwen3_embedding_config(
        _WEEK22_CONFIG_PATH;
        max_seq_len=512,
    )
    @test config.vocab_size == spec.vocab_size
    @test config.max_seq_len == 512
    @test config.source_max_seq_len == spec.max_position_embeddings
    @test config.qwen3_variant === nothing
    @test config.qwen3_embedding_variant === spec.variant

    @test_throws ArgumentError load_hf_qwen3_embedding_config(
        _WEEK22_CONFIG_PATH;
        max_seq_len=32_769,
    )
    mktempdir() do directory
        changed = replace(read(_WEEK22_CONFIG_PATH, String), "151669" => "151936")
        path = joinpath(directory, "config.json")
        write(path, changed)
        @test_throws ArgumentError load_hf_qwen3_embedding_config(path)
    end
end

@testset "frozen RTX 4090 D CUDA acceptance" begin
    @test isfile(_WEEK22_CUDA_REPORT_PATH)
    report = JSON3.read(read(_WEEK22_CUDA_REPORT_PATH, String))
    reference = JSON3.read(read(_WEEK22_REFERENCE_PATH, String))
    spec = qwen3_embedding_spec()
    @test Int(report["schema_version"]) == 1
    @test Bool(report["closed"])
    @test String(report["model_id"]) == spec.model_id
    @test String(report["revision"]) == spec.revision
    @test String(report["device"]) == "cuda"
    @test String(report["compute_dtype"]) == "bfloat16"
    @test Int(report["parameter_count"]) == qwen3_embedding_parameter_count()
    hardware = report["hardware"]
    @test String(hardware["gpu_name"]) == "NVIDIA GeForce RTX 4090 D"
    @test String(hardware["compute_capability"]) == "8.9.0"
    @test String(hardware["nvidia_driver_version"]) == "570.153.02"
    @test String(hardware["cuda_runtime_version"]) == "12.9.0"
    @test Bool(report["token_ids_equal"])
    @test Bool(report["attention_mask_equal"])
    timing = report["timing"]
    @test Float64(timing["cold_warm_embedding_max_abs"]) == 0
    @test Float64(timing["warm_forward_seconds"]) > 0
    dimensions = report["dimensions"]
    @test all(Bool(result["embedding_passed"]) for result in values(dimensions))
    @test all(Bool(result["similarity_passed"]) for result in values(dimensions))
    @test all(Bool(result["topk_equal"]) for result in values(dimensions))
    @test all(
        Float64(result["embedding_max_abs"]) <=
            Float64(reference["tolerances"]["embedding_max_abs"])
        for result in values(dimensions)
    )
    @test all(
        Float64(result["similarity_max_abs"]) <=
            Float64(reference["tolerances"]["similarity_max_abs"])
        for result in values(dimensions)
    )
    @test Bool(report["semantic_memory"]["topk_equal"])
    @test Bool(report["semantic_memory"]["first_document_equal"])
end

@testset "embedding tokenizer profile and padding" begin
    mktempdir() do directory
        _week22_embedding_tokenizer_fixture(directory)
        tokenizer = load_hf_qwen3_embedding_tokenizer(
            directory;
            revision="week22-test",
        )
        @test tokenizer.profile === :embedding
        @test_throws ArgumentError LifeAI.hf_generation_config(tokenizer)
        @test qwen3_embedding_query("Julia") ==
            "Instruct: $QWEN3_EMBEDDING_RETRIEVAL_INSTRUCTION\nQuery:Julia"
        @test qwen3_embedding_query(
            "向量";
            instruction="Retrieve Chinese passages",
        ) == "Instruct: Retrieve Chinese passages\nQuery:向量"
        @test_throws ArgumentError qwen3_embedding_query("")
        @test_throws ArgumentError qwen3_embedding_query("x"; instruction="")

        raw_ids = LifeAI.encode(tokenizer, "hi!")
        special_ids = LifeAI.encode(
            tokenizer,
            "hi!";
            add_special_tokens=true,
        )
        @test special_ids == [raw_ids; LifeAI.special_token_id(tokenizer, :pad)]

        left = prepare_qwen3_embedding_inputs(
            tokenizer,
            ["hi!", "hi! hi!"];
            max_length=16,
            padding_side=:left,
        )
        @test size(left.tokens, 2) == 2
        @test left.lengths[1] < left.lengths[2]
        @test !left.attention_mask[1, 1]
        @test left.attention_mask[end, 1]
        @test all(left.attention_mask[:, 2])

        right = prepare_qwen3_embedding_inputs(
            tokenizer,
            ["hi!", "hi! hi!"];
            max_length=16,
            padding_side=:right,
        )
        @test right.tokens[end, 1] ==
            LifeAI.special_token_id(tokenizer, :pad)
        @test right.attention_mask[1, 1]
        @test !right.attention_mask[end, 1]

        truncated = prepare_qwen3_embedding_inputs(
            tokenizer,
            ["hi! hi!"];
            max_length=1,
        )
        @test truncated.truncated == [true]
        @test truncated.original_lengths[1] > truncated.lengths[1] == 1
        @test truncated.tokens[1, 1] ==
            LifeAI.special_token_id(tokenizer, :pad)
        @test_throws ArgumentError prepare_qwen3_embedding_inputs(
            tokenizer,
            [""],
        )
        @test_throws ArgumentError prepare_qwen3_embedding_inputs(
            tokenizer,
            ["hi!"];
            padding_side=:middle,
        )

        path = joinpath(directory, "embedding-tokenizer.toml")
        save_tokenizer(path, tokenizer)
        restored = load_tokenizer(path)
        @test restored.profile === :embedding
        @test tokenizer_fingerprint(restored) ==
            tokenizer_fingerprint(tokenizer)
    end

    mktempdir() do directory
        payloads = _week22_embedding_tokenizer_payloads()
        payloads.tokenizer["post_processor"]["processors"][2]["single"][1][
            "Sequence"
        ]["id"] = "X"
        write_week08_tokenizer_fixture(directory; payloads)
        @test_throws ArgumentError load_hf_qwen3_embedding_tokenizer(directory)
    end
end

@testset "last-token pooling, MRL, and cosine retrieval" begin
    hidden = reshape(Float32.(1:(64 * 4 * 3)), 64, 4, 3)
    mask = Bool[
        0 1 0
        0 1 1
        1 0 1
        1 0 1
    ]
    pooled = qwen3_last_token_pool(
        hidden,
        mask;
        dimension=32,
        minimum_dimension=32,
    )
    @test size(pooled) == (32, 3)
    @test all(isapprox(norm(view(pooled, :, index)), 1.0f0; atol=1.0f-6)
              for index in 1:3)
    expected_first = hidden[1:32, 4, 1]
    expected_first ./= norm(expected_first)
    @test pooled[:, 1] ≈ expected_first atol=1.0f-6

    full = qwen3_last_token_pool(hidden, mask; dimension=64)
    truncated_after_normalization = full[1:32, :]
    @test !isapprox(norm(view(truncated_after_normalization, :, 1)), 1.0f0)
    @test_throws ArgumentError qwen3_last_token_pool(
        hidden,
        mask;
        dimension=31,
        minimum_dimension=32,
    )
    bad_mask = copy(mask)
    bad_mask[:, 1] .= [true, false, true, false]
    @test_throws ArgumentError qwen3_last_token_pool(hidden, bad_mask)
    @test_throws ArgumentError qwen3_last_token_pool(
        zeros(Float32, 64, 1, 1),
        trues(1, 1),
    )

    queries = Float32[1 0; 0 1]
    documents = Float32[2 1 0; 0 1 2]
    scores = qwen3_embedding_similarity(queries, documents)
    @test size(scores) == (2, 3)
    @test scores[1, 1] ≈ 1.0f0
    @test scores[2, 3] ≈ 1.0f0
    @test_throws DimensionMismatch qwen3_embedding_similarity(
        zeros(Float32, 3, 1),
        zeros(Float32, 2, 1),
    )

    memory = Qwen3SemanticMemory(
        ["x-axis", "diagonal", "y-axis"],
        documents,
        Any[:x, :diagonal, :y],
    )
    @test all(isapprox(norm(view(memory.embeddings, :, index)), 1.0f0)
              for index in axes(memory.embeddings, 2))
    results = retrieve_qwen3_semantic_memory(memory, Float32[1, 0]; top_k=2)
    @test [result.index for result in results] == [1, 2]
    @test results[1].metadata === :x
    @test_throws DimensionMismatch Qwen3SemanticMemory(
        ["only one"],
        documents,
    )
    @test_throws DimensionMismatch Qwen3SemanticMemory(
        ["x-axis", "diagonal", "y-axis"],
        documents,
        [:missing],
    )
    @test_throws ArgumentError Qwen3SemanticMemory(
        [""],
        ones(Float32, 2, 1),
    )
    @test_throws ArgumentError Qwen3SemanticMemory(
        ["zero"],
        zeros(Float32, 2, 1),
    )
    @test_throws ArgumentError retrieve_qwen3_semantic_memory(
        memory,
        Float32[1, 0];
        top_k=4,
    )
end

@testset "BF16 embedding forward honors per-batch padding masks" begin
    mktempdir() do directory
        _week16_fixture_dir(directory; tie=true)
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        first_ids = [2, 3, 4]
        second_ids = [5, 6]

        first = hf_qwen3_embedding_forward(
            loaded.model,
            loaded.parameters,
            reshape(first_ids, :, 1),
            trues(3, 1);
            dimension=8,
        )
        second = hf_qwen3_embedding_forward(
            loaded.model,
            loaded.parameters,
            reshape(second_ids, :, 1),
            trues(2, 1);
            dimension=8,
        )

        right_tokens = [2 5; 3 6; 4 1]
        right_mask = Bool[1 1; 1 1; 1 0]
        right = hf_qwen3_embedding_forward(
            loaded.model,
            loaded.parameters,
            right_tokens,
            right_mask;
            dimension=8,
        )
        @test right.embeddings[:, 1] == first.embeddings[:, 1]
        @test right.embeddings[:, 2] == second.embeddings[:, 1]

        left_tokens = [2 1; 3 5; 4 6]
        left_mask = Bool[1 0; 1 1; 1 1]
        left = hf_qwen3_embedding_forward(
            loaded.model,
            loaded.parameters,
            left_tokens,
            left_mask;
            dimension=8,
        )
        @test left.embeddings[:, 1] == first.embeddings[:, 1]
        @test left.embeddings[:, 2] ≈ second.embeddings[:, 1] atol=2.0f-3

        @test_throws ArgumentError hf_qwen3_embedding_forward(
            loaded.model,
            loaded.parameters,
            left_tokens,
            Bool[1 0; 0 1; 1 1];
            dimension=8,
        )
        @test_throws ArgumentError hf_qwen3_embedding_forward(
            loaded.model,
            loaded.parameters,
            left_tokens,
            left_mask;
            dimension=7,
        )
    end
end

const _WEEK22_MODEL_DIR = get(
    ENV,
    "LIFEAI_QWEN3_EMBEDDING_0_6B_MODEL_DIR",
    "",
)

if !isempty(_WEEK22_MODEL_DIR)
    @testset "real Qwen3-Embedding-0.6B parity" begin
        @test isdir(_WEEK22_MODEL_DIR)
        @test isfile(_WEEK22_REFERENCE_PATH)
        assets = verify_qwen3_embedding_assets(_WEEK22_MODEL_DIR)
        @test length(assets) == 8
        reference = JSON3.read(read(_WEEK22_REFERENCE_PATH, String))
        bundle = load_hf_qwen3_embedding_bundle(
            _WEEK22_MODEL_DIR;
            max_seq_len=Int(reference["max_length"]),
            weight_dtype=BFloat16,
        )
        texts = String.(collect(reference["texts"]))
        result = embed_texts(
            bundle,
            texts;
            dimension=bundle.model.d_model,
            max_length=Int(reference["max_length"]),
        )
        @test result.tokens .- 1 ==
            reduce(
                hcat,
                [Int.(collect(row)) for row in reference["input_ids"]],
            )
        @test result.attention_mask ==
            Bool.(reduce(
                hcat,
                [Int.(collect(row)) for row in reference["attention_mask"]],
            ))

        expected = reduce(
            hcat,
            [
                Float32.(collect(row)) for
                row in reference["embeddings"]["1024"]
            ],
        )
        max_abs = maximum(abs.(result.embeddings .- expected))
        @test max_abs <= Float64(reference["tolerances"]["embedding_max_abs"])
        for raw_dimension in ("512", "256", "128", "64")
            dimension = parse(Int, raw_dimension)
            actual = copy(result.embeddings[1:dimension, :])
            for batch in axes(actual, 2)
                actual[:, batch] ./= norm(view(actual, :, batch))
            end
            expected_dimension = reduce(
                hcat,
                [
                    Float32.(collect(row)) for
                    row in reference["embeddings"][raw_dimension]
                ],
            )
            @test maximum(abs.(actual .- expected_dimension)) <=
                Float64(reference["tolerances"]["embedding_max_abs"])
        end
    end
end
