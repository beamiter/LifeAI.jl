using Test
using BFloat16s: BFloat16
using JSON3
using LinearAlgebra: norm
import LifeAI
using LifeAI:
    QWEN3_EMBEDDING_RETRIEVAL_INSTRUCTION,
    Qwen3EmbeddingSpec,
    Qwen3SemanticMemory,
    embed_texts,
    hf_qwen3_embedding_forward,
    load_hf_qwen3_embedding_bundle,
    load_hf_qwen3_embedding_config,
    load_hf_qwen3_embedding_model,
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

isdefined(@__MODULE__, :repository_test_asset) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

isdefined(@__MODULE__, :qwen3_tokenizer_fixture_payloads) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tokenizer_fixture.jl"))
isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tiny_model_fixture.jl"))

const _QWEN3_EMBEDDING_FIXTURE_DIR = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_embedding_memory",
)
const _QWEN3_EMBEDDING_CONFIG_PATH = joinpath(_QWEN3_EMBEDDING_FIXTURE_DIR, "config.json")
const _QWEN3_EMBEDDING_ASSETS_PATH = joinpath(_QWEN3_EMBEDDING_FIXTURE_DIR, "assets.json")
const _QWEN3_EMBEDDING_REFERENCE_PATH = joinpath(_QWEN3_EMBEDDING_FIXTURE_DIR, "reference.json")
const _QWEN3_EMBEDDING_CUDA_REPORT_PATH =
    repository_test_asset("qwen3_embedding_0_6b_cuda.json")

function _embedding_argument_error_message(f)
    try
        f()
    catch exception
        exception isa ArgumentError || rethrow()
        return exception.msg
    end
    return nothing
end

function _embedding_tokenizer_payloads()
    payloads = qwen3_tokenizer_fixture_payloads()
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

function _embedding_tokenizer_fixture(directory)
    return write_qwen3_tokenizer_fixture(
        directory;
        payloads=_embedding_tokenizer_payloads(),
    )
end

@testset "Qwen3 embedding text inputs are strict" begin
    text = SubString("xhello", 2)
    @test LifeAI._qwen3_embedding_text_list(text) == ["hello"]
    @test LifeAI._qwen3_embedding_text_list((text, "world")) ==
        ["hello", "world"]
    @test LifeAI._qwen3_embedding_text_list(
        (value for value in (text, "world")),
    ) == ["hello", "world"]

    for invalid in (
        Any["ok", 1],
        (nothing,),
        Char['a', 'b'],
        Dict("key" => "value"),
    )
        @test _embedding_argument_error_message() do
            LifeAI._qwen3_embedding_text_list(invalid)
        end == "embedding texts must contain only strings"
    end
    @test _embedding_argument_error_message() do
        LifeAI._qwen3_embedding_text_list(1)
    end == "embedding texts must contain only strings"
    @test _embedding_argument_error_message() do
        LifeAI._qwen3_embedding_text_list(String[])
    end == "embedding input must contain at least one text"
    @test _embedding_argument_error_message() do
        LifeAI._qwen3_embedding_text_list([""])
    end == "embedding texts must not be empty"
end

@testset "Qwen3 embedding specifications are strict" begin
    strings = ntuple(_ -> SubString("xvalue", 2), 10)
    valid = (
        :fixture,
        strings...,
        ntuple(_ -> big(1), 9)...,
    )
    spec = Qwen3EmbeddingSpec(valid...)
    @test spec.variant === :fixture
    @test spec.model_id === "value"
    @test spec.model_sha256 === "value"
    @test spec.vocab_size === 1
    @test spec.minimum_dimension === 1

    names = fieldnames(Qwen3EmbeddingSpec)
    for index in 12:20
        label = names[index]
        for (value, message) in (
            (true, "must be an integer"),
            (0, "must be positive"),
        )
            failure = _embedding_argument_error_message() do
                Qwen3EmbeddingSpec(Base.setindex(valid, value, index)...)
            end
            @test failure == "Qwen3 embedding $label $message"
        end
    end

    too_large = big(typemax(Int)) + 1
    for (index, value, message) in (
        (12, 1.0, "vocab_size must be an integer"),
        (12, too_large, "vocab_size is outside the host integer range"),
        (1, "fixture", "variant must be a Symbol"),
    )
        failure = _embedding_argument_error_message() do
            Qwen3EmbeddingSpec(Base.setindex(valid, value, index)...)
        end
        @test failure == "Qwen3 embedding $message"
    end

    for index in 2:11
        label = names[index]
        failure = _embedding_argument_error_message() do
            Qwen3EmbeddingSpec(Base.setindex(valid, label, index)...)
        end
        @test failure == "Qwen3 embedding $label must be a string"
    end
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
    overflow_fields = map(fieldnames(Qwen3EmbeddingSpec)) do name
        name === :vocab_size && return typemax(Int)
        name === :d_model && return 2
        return getfield(spec, name)
    end
    overflow_spec = Qwen3EmbeddingSpec(overflow_fields...)
    @test_throws ArgumentError qwen3_embedding_parameter_count(overflow_spec)

    manifest = JSON3.read(read(_QWEN3_EMBEDDING_ASSETS_PATH, String))
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
        _QWEN3_EMBEDDING_CONFIG_PATH;
        max_seq_len=big(512),
    )
    @test config.vocab_size == spec.vocab_size
    @test config.max_seq_len == 512
    @test config.source_max_seq_len == spec.max_position_embeddings
    @test config.qwen3_variant === nothing
    @test config.qwen3_embedding_variant === spec.variant

    @test_throws ArgumentError load_hf_qwen3_embedding_config(
        _QWEN3_EMBEDDING_CONFIG_PATH;
        max_seq_len=32_769,
    )
    mktempdir() do directory
        changed = replace(read(_QWEN3_EMBEDDING_CONFIG_PATH, String), "151669" => "151936")
        path = joinpath(directory, "config.json")
        write(path, changed)
        @test_throws ArgumentError load_hf_qwen3_embedding_config(path)
    end
end

@testset "Qwen3 embedding load preflight" begin
    mktempdir() do directory
        missing_config = joinpath(directory, "missing-config.json")
        cases = (
            (true, "max_seq_len must be an integer"),
            (0, "max_seq_len must be positive"),
            (big(typemax(Int)) + 1, "max_seq_len is outside the host integer range"),
            (32_769, "max_seq_len must be in 1:32768; got 32769"),
        )
        for (value, expected) in cases
            @test _embedding_argument_error_message() do
                load_hf_qwen3_embedding_config(
                    missing_config;
                    max_seq_len=value,
                )
            end == expected
        end

        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_model(directory; max_seq_len=true)
        end == "max_seq_len must be an integer"
        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_model(directory; max_seq_len=32_769)
        end == "max_seq_len must be in 1:32768; got 32769"
        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_model(directory; weight_dtype=Float64)
        end == "weight_dtype must be Float32 or BFloat16"

        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_bundle(directory; max_seq_len=true)
        end == "max_seq_len must be an integer"
        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_bundle(directory; max_seq_len=32_769)
        end == "max_seq_len must be in 1:32768; got 32769"
        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_bundle(directory; weight_dtype=Float64)
        end == "weight_dtype must be Float32 or BFloat16"

        revision_message = _embedding_argument_error_message() do
            load_hf_qwen3_embedding_bundle(
                directory;
                revision="moving",
                max_seq_len=true,
            )
        end
        @test startswith(
            revision_message,
            "unsupported Qwen3 embedding revision \"moving\"; expected ",
        )

        tokenizer_message = _embedding_argument_error_message() do
            load_hf_qwen3_embedding_bundle(
                directory;
                max_seq_len=big(512),
                weight_dtype=BFloat16,
            )
        end
        @test startswith(
            tokenizer_message,
            "required Qwen3 embedding tokenizer file does not exist: ",
        )
    end
end

@testset "frozen RTX 4090 D CUDA acceptance" begin
    @test isfile(_QWEN3_EMBEDDING_CUDA_REPORT_PATH)
    report = JSON3.read(read(_QWEN3_EMBEDDING_CUDA_REPORT_PATH, String))
    reference = JSON3.read(read(_QWEN3_EMBEDDING_REFERENCE_PATH, String))
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
        _embedding_tokenizer_fixture(directory)
        tokenizer = load_hf_qwen3_embedding_tokenizer(
            directory;
            revision="qwen3_embedding-test",
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
        for (max_length, message) in (
            (true, "max_length must be an integer"),
            (
                big(typemax(Int)) + 1,
                "max_length is outside the host integer range",
            ),
        )
            @test _embedding_argument_error_message() do
                prepare_qwen3_embedding_inputs(
                    tokenizer,
                    ["hi!"];
                    max_length,
                )
            end == message
        end

        path = joinpath(directory, "embedding-tokenizer.toml")
        save_tokenizer(path, tokenizer)
        restored = load_tokenizer(path)
        @test restored.profile === :embedding
        @test tokenizer_fingerprint(restored) ==
            tokenizer_fingerprint(tokenizer)
    end

    mktempdir() do directory
        payloads = _embedding_tokenizer_payloads()
        payloads.tokenizer["post_processor"]["processors"][2]["single"][1][
            "Sequence"
        ]["id"] = "X"
        write_qwen3_tokenizer_fixture(directory; payloads)
        @test_throws ArgumentError load_hf_qwen3_embedding_tokenizer(directory)
    end

    mktempdir() do directory
        payloads = _embedding_tokenizer_payloads()
        payloads.tokenizer["post_processor"]["processors"][1][
            "future_behavior"
        ] = true
        write_qwen3_tokenizer_fixture(directory; payloads)
        failure = try
            load_hf_qwen3_embedding_tokenizer(directory)
            nothing
        catch caught
            caught
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: unsupported tokenizer.json embedding " *
            "post_processor ByteLevel fields: future_behavior"
    end

    post_processor_field_mutations = (
        (
            payloads -> begin
                post_processor = payloads.tokenizer["post_processor"]
                delete!(post_processor, "processors")
                post_processor["z_future"] = true
                post_processor["a_future"] = true
            end,
            "unsupported tokenizer.json embedding post_processor fields: " *
            "a_future, z_future",
        ),
        (
            payloads -> delete!(
                payloads.tokenizer["post_processor"],
                "processors",
            ),
            "missing `processors` in tokenizer.json embedding post_processor",
        ),
        (
            payloads -> begin
                template = payloads.tokenizer["post_processor"]["processors"][2]
                delete!(template, "single")
                template["future_behavior"] = true
            end,
            "unsupported tokenizer.json embedding post_processor " *
            "TemplateProcessing fields: future_behavior",
        ),
        (
            payloads -> delete!(
                payloads.tokenizer["post_processor"]["processors"][2],
                "single",
            ),
            "missing `single` in tokenizer.json embedding post_processor",
        ),
        (
            payloads -> begin
                sequence = payloads.tokenizer["post_processor"]["processors"][2][
                    "single"
                ][1]["Sequence"]
                delete!(sequence, "id")
                sequence["future_behavior"] = true
            end,
            "unsupported tokenizer.json embedding post_processor single[1] " *
            "Sequence fields: future_behavior",
        ),
        (
            payloads -> delete!(
                payloads.tokenizer["post_processor"]["processors"][2][
                    "single"
                ][1]["Sequence"],
                "id",
            ),
            "missing `id` in tokenizer.json embedding post_processor single[1]",
        ),
        (
            payloads -> (
                payloads.tokenizer["post_processor"]["processors"][2]["pair"][2][
                    "Sequence"
                ]["future_behavior"] = true
            ),
            "unsupported tokenizer.json embedding post_processor pair[2] " *
            "Sequence fields: future_behavior",
        ),
        (
            payloads -> delete!(
                payloads.tokenizer["post_processor"]["processors"][2]["pair"][2][
                    "Sequence"
                ],
                "type_id",
            ),
            "missing `type_id` in tokenizer.json embedding post_processor pair[2]",
        ),
        (
            payloads -> begin
                special = payloads.tokenizer["post_processor"]["processors"][2][
                    "single"
                ][2]["SpecialToken"]
                delete!(special, "id")
                special["future_behavior"] = true
            end,
            "unsupported tokenizer.json embedding post_processor single[2] " *
            "SpecialToken fields: future_behavior",
        ),
        (
            payloads -> delete!(
                payloads.tokenizer["post_processor"]["processors"][2][
                    "single"
                ][2]["SpecialToken"],
                "id",
            ),
            "missing `id` in tokenizer.json embedding post_processor single[2]",
        ),
        (
            payloads -> (
                payloads.tokenizer["post_processor"]["processors"][2]["pair"][3][
                    "SpecialToken"
                ]["future_behavior"] = true
            ),
            "unsupported tokenizer.json embedding post_processor pair[3] " *
            "SpecialToken fields: future_behavior",
        ),
        (
            payloads -> delete!(
                payloads.tokenizer["post_processor"]["processors"][2]["pair"][3][
                    "SpecialToken"
                ],
                "type_id",
            ),
            "missing `type_id` in tokenizer.json embedding post_processor pair[3]",
        ),
        (
            payloads -> begin
                metadata = payloads.tokenizer["post_processor"]["processors"][2][
                    "special_tokens"
                ]["<|endoftext|>"]
                delete!(metadata, "ids")
                metadata["future_behavior"] = true
            end,
            "unsupported tokenizer.json embedding post_processor " *
            "<|endoftext|> metadata fields: future_behavior",
        ),
        (
            payloads -> delete!(
                payloads.tokenizer["post_processor"]["processors"][2][
                    "special_tokens"
                ]["<|endoftext|>"],
                "ids",
            ),
            "missing `ids` in tokenizer.json embedding post_processor",
        ),
    )
    for (mutate!, message) in post_processor_field_mutations
        mktempdir() do directory
            payloads = _embedding_tokenizer_payloads()
            mutate!(payloads)
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test _embedding_argument_error_message() do
                load_hf_qwen3_embedding_tokenizer(directory)
            end == message
        end
    end

    post_processor_shape_mutations = (
        (
            payloads -> (
                payloads.tokenizer["post_processor"]["processors"][2][
                    "single"
                ][1]["Future"] = Dict("enabled" => true)
            ),
            "tokenizer.json embedding post_processor single[1] must contain " *
            "exactly one Sequence entry",
        ),
        (
            payloads -> (
                payloads.tokenizer["post_processor"]["processors"][2]["pair"][3][
                    "Future"
                ] = Dict("enabled" => true)
            ),
            "tokenizer.json embedding post_processor pair[3] must contain " *
            "exactly one SpecialToken entry",
        ),
        (
            payloads -> begin
                special_tokens = payloads.tokenizer["post_processor"]["processors"][2][
                    "special_tokens"
                ]
                special_tokens["future_token"] = Dict{String,Any}()
                special_tokens["<|endoftext|>"]["future_behavior"] = true
            end,
            "embedding template must define only <|endoftext|>",
        ),
    )
    for (mutate!, message) in post_processor_shape_mutations
        mktempdir() do directory
            payloads = _embedding_tokenizer_payloads()
            mutate!(payloads)
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test _embedding_argument_error_message() do
                load_hf_qwen3_embedding_tokenizer(directory)
            end == message
        end
    end

    mktempdir() do directory
        payloads = _embedding_tokenizer_payloads()
        payloads.tokenizer_config["future_behavior"] = true
        write_qwen3_tokenizer_fixture(directory; payloads)
        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_tokenizer(directory)
        end == "unsupported tokenizer_config.json fields: future_behavior"
    end

    type_id_mutations = (
        payloads -> (
            payloads.tokenizer["post_processor"]["processors"][2]["single"][1][
                "Sequence"
            ]["type_id"] = false
        ),
        payloads -> (
            payloads.tokenizer["post_processor"]["processors"][2]["single"][2][
                "SpecialToken"
            ]["type_id"] = false
        ),
        payloads -> (
            payloads.tokenizer["post_processor"]["processors"][2]["pair"][1][
                "Sequence"
            ]["type_id"] = false
        ),
        payloads -> (
            payloads.tokenizer["post_processor"]["processors"][2]["pair"][2][
                "Sequence"
            ]["type_id"] = false
        ),
        payloads -> (
            payloads.tokenizer["post_processor"]["processors"][2]["pair"][3][
                "SpecialToken"
            ]["type_id"] = false
        ),
    )
    for mutate! in type_id_mutations
        mktempdir() do directory
            payloads = _embedding_tokenizer_payloads()
            mutate!(payloads)
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test _embedding_argument_error_message() do
                load_hf_qwen3_embedding_tokenizer(directory)
            end == "`type_id` must be an integer"
        end
    end

    mktempdir() do directory
        payloads = _embedding_tokenizer_payloads()
        payloads.tokenizer["post_processor"]["processors"][2]["single"][1][
            "Sequence"
        ]["type_id"] = 1
        write_qwen3_tokenizer_fixture(directory; payloads)
        @test _embedding_argument_error_message() do
            load_hf_qwen3_embedding_tokenizer(directory)
        end ==
              "unsupported `type_id=1` in tokenizer.json embedding post_processor single[1]; expected 0"
    end

    for (value, message) in (
        (true, "max_new_tokens must be an integer"),
        (1.5, "max_new_tokens must be an integer"),
        (0, "max_new_tokens must be a positive integer"),
        (-1, "max_new_tokens must be a positive integer"),
    )
        mktempdir() do directory
            payloads = _embedding_tokenizer_payloads()
            payloads.generation_config["max_new_tokens"] = value
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test _embedding_argument_error_message() do
                load_hf_qwen3_embedding_tokenizer(directory)
            end == message
        end
    end

    for (value, message) in (
        (true, "embedding <|endoftext|> id must be an integer"),
        (
            big(typemax(Int)) + 1,
            "embedding <|endoftext|> id must be an integer",
        ),
        (
            typemax(Int),
            "embedding <|endoftext|> id is outside the one-based token id range",
        ),
        (-1, "embedding <|endoftext|> id must be non-negative"),
    )
        mktempdir() do directory
            payloads = _embedding_tokenizer_payloads()
            payloads.tokenizer["post_processor"]["processors"][2][
                "special_tokens"
            ]["<|endoftext|>"]["ids"] = Any[value]
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test _embedding_argument_error_message() do
                load_hf_qwen3_embedding_tokenizer(directory)
            end == message
        end
    end

    for value in (true, 258, nothing)
        mktempdir() do directory
            payloads = _embedding_tokenizer_payloads()
            payloads.tokenizer["post_processor"]["processors"][2][
                "special_tokens"
            ]["<|endoftext|>"]["tokens"] = Any[value]
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test _embedding_argument_error_message() do
                load_hf_qwen3_embedding_tokenizer(directory)
            end == "embedding template token payload must be <|endoftext|>"
        end
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
    @test qwen3_last_token_pool(
        hidden,
        mask;
        dimension=Int128(32),
        minimum_dimension=big(32),
    ) == pooled
    @test qwen3_last_token_pool(
        hidden,
        Int8.(mask);
        dimension=32,
        minimum_dimension=32,
    ) == pooled

    missing_mask = Matrix{Any}(mask)
    missing_mask[1, 1] = missing
    for invalid_mask in (
        Float64.(mask),
        ComplexF64.(mask),
        missing_mask,
        fill(2, size(mask)),
    )
        @test _embedding_argument_error_message() do
            qwen3_last_token_pool(hidden, invalid_mask; dimension=32)
        end == "attention_mask values must be Bool or integer zero/one"
    end

    too_large = big(typemax(Int)) + 1
    for (options, message) in (
        ((; dimension=true), "dimension must be an integer"),
        (
            (; dimension=too_large),
            "dimension is outside the host integer range",
        ),
        ((; minimum_dimension=true), "minimum_dimension must be an integer"),
        (
            (; minimum_dimension=too_large),
            "minimum_dimension is outside the host integer range",
        ),
    )
        @test _embedding_argument_error_message() do
            qwen3_last_token_pool(hidden, mask; options...)
        end == message
    end

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
    @test [result.index for result in retrieve_qwen3_semantic_memory(
        memory,
        Float32[1, 0];
        top_k=big(2),
    )] == [1, 2]
    for (top_k, message) in (
        (true, "top_k must be an integer"),
        (too_large, "top_k is outside the host integer range"),
    )
        @test _embedding_argument_error_message() do
            retrieve_qwen3_semantic_memory(
                memory,
                Float32[1, 0];
                top_k,
            )
        end == message
    end
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
        _qwen3_tiny_model_fixture_dir(directory; tie=true)
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
        left_int32 = hf_qwen3_embedding_forward(
            loaded.model,
            loaded.parameters,
            Int32.(left_tokens),
            left_mask;
            dimension=8,
        )
        @test left_int32.embeddings == left.embeddings

        missing_tokens = Matrix{Any}(left_tokens)
        missing_tokens[1, 1] = missing
        for invalid_tokens in (
            Bool.(left_tokens .> 0),
            Float64.(left_tokens),
            ComplexF64.(left_tokens),
            missing_tokens,
        )
            @test _embedding_argument_error_message() do
                hf_qwen3_embedding_forward(
                    loaded.model,
                    loaded.parameters,
                    invalid_tokens,
                    left_mask;
                    dimension=8,
                )
            end == "embedding token ids must be an integer"
        end
        overflow_tokens = BigInt.(left_tokens)
        overflow_tokens[1, 1] = big(typemax(Int)) + 1
        @test _embedding_argument_error_message() do
            hf_qwen3_embedding_forward(
                loaded.model,
                loaded.parameters,
                overflow_tokens,
                left_mask;
                dimension=8,
            )
        end == "embedding token ids is outside the host integer range"

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
        @test _embedding_argument_error_message() do
            hf_qwen3_embedding_forward(
                loaded.model,
                loaded.parameters,
                left_tokens,
                left_mask;
                dimension=true,
            )
        end == "dimension must be an integer"
        @test _embedding_argument_error_message() do
            hf_qwen3_embedding_forward(
                loaded.model,
                loaded.parameters,
                left_tokens,
                left_mask;
                dimension=big(typemax(Int)) + 1,
            )
        end == "dimension is outside the host integer range"
    end
end

const _QWEN3_EMBEDDING_MODEL_DIR = get(
    ENV,
    "LIFEAI_QWEN3_EMBEDDING_0_6B_MODEL_DIR",
    "",
)

if !isempty(_QWEN3_EMBEDDING_MODEL_DIR)
    @testset "real Qwen3-Embedding-0.6B parity" begin
        @test isdir(_QWEN3_EMBEDDING_MODEL_DIR)
        @test isfile(_QWEN3_EMBEDDING_REFERENCE_PATH)
        assets = verify_qwen3_embedding_assets(_QWEN3_EMBEDDING_MODEL_DIR)
        @test length(assets) == 8
        reference = JSON3.read(read(_QWEN3_EMBEDDING_REFERENCE_PATH, String))
        bundle = load_hf_qwen3_embedding_bundle(
            _QWEN3_EMBEDDING_MODEL_DIR;
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
