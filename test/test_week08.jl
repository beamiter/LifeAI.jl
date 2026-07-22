using Test
using Random: Xoshiro
using Lux
using JSON3
using LifeAI:
    GPTModel,
    HFQwen3Tokenizer,
    TrainerGPT,
    apply_qwen3_chat_template,
    decode,
    decode_bytes,
    encode,
    generate_hf_text,
    hf_byte_unicode_alphabet,
    hf_qwen3_pretokenize,
    init_train_state,
    load_checkpoint,
    load_hf_qwen3_bundle,
    load_hf_qwen3_tokenizer,
    load_safetensors,
    load_tokenizer,
    save_checkpoint,
    save_tokenizer,
    special_token_id,
    tokenizer_config,
    tokenizer_fingerprint,
    vocab_size

include("week08_fixture.jl")

@testset "Qwen3 byte alphabet and imported BPE" begin
    alphabet = hf_byte_unicode_alphabet()
    @test length(alphabet.byte_to_char) == 256
    @test length(alphabet.char_to_byte) == 256
    @test all(alphabet.char_to_byte[alphabet.byte_to_char[UInt8(byte)]] == byte for byte in 0:255)

    mktempdir() do directory
        write_week08_tokenizer_fixture(directory)
        tokenizer = load_hf_qwen3_tokenizer(directory; revision="week08-test")
        @test tokenizer isa HFQwen3Tokenizer
        @test vocab_size(tokenizer) == 263
        @test tokenizer.model_vocabulary_size == 258
        @test tokenizer_config(tokenizer).merge_count == 2
        @test tokenizer_config(tokenizer).revision == "week08-test"
        @test special_token_id(tokenizer, :bos) == 259
        @test special_token_id(tokenizer, :eos) == 261
        @test special_token_id(tokenizer, :pad) == 259
        @test special_token_id(tokenizer, :unk) === nothing
        @test tokenizer.eos_ids == [261, 259]

        @test encode(tokenizer, "hi!") == [257, 34]
        @test encode(tokenizer, "hi"; add_special_tokens=true) == [257]
        @test encode(tokenizer, "hi"; add_special_tokens=false) == [257]
        @test decode(tokenizer, [258]) == "hi!"
        @test encode(tokenizer, "e\u0301") == [196, 170]
        @test decode(tokenizer, encode(tokenizer, "e\u0301")) == "é"
        pretokenized = hf_qwen3_pretokenize(tokenizer, "e\u0301 hi!")
        @test pretokenized.normalized == "é hi!"
        @test [piece.text for piece in pretokenized.pieces] == ["é", " hi", "!"]
        @test [(piece.character_start, piece.character_stop) for piece in pretokenized.pieces] ==
              [(0, 1), (1, 4), (4, 5)]
        @test [(piece.byte_start, piece.byte_stop) for piece in pretokenized.pieces] ==
              [(0, 2), (2, 5), (5, 6)]

        added = encode(tokenizer, "<|im_start|>hi<|im_end|><think>x</think>")
        @test added == [260, 257, 261, 262, 121, 263]
        @test decode(tokenizer, added) == "<|im_start|>hi<|im_end|><think>x</think>"
        @test decode(tokenizer, added; skip_special_tokens=true) == "hi<think>x</think>"
        @test decode_bytes(tokenizer, added; skip_special_tokens=true) ==
              Vector{UInt8}(codeunits("hi<think>x</think>"))
        @test_throws ArgumentError decode(tokenizer, [0])
        @test_throws ArgumentError decode(tokenizer, [264])
        @test_throws ArgumentError decode(tokenizer, [1.5])
    end
end

@testset "Qwen3 basic chat template" begin
    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_week08_tokenizer_fixture(directory))
        user_prompt = apply_qwen3_chat_template(
            tokenizer,
            [(role="user", content="Hi")];
            enable_thinking=false,
        )
        @test user_prompt ==
              "<|im_start|>user\nHi<|im_end|>\n" *
              "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        @test apply_qwen3_chat_template(
            tokenizer,
            [(role="system", content="S"), (role="user", content="U")];
            add_generation_prompt=false,
        ) == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
        history = [
            (role="user", content="U"),
            (role="assistant", content="A"),
            (role="user", content="V"),
        ]
        @test apply_qwen3_chat_template(tokenizer, history; enable_thinking=true) ==
              "<|im_start|>user\nU<|im_end|>\n" *
              "<|im_start|>assistant\nA<|im_end|>\n" *
              "<|im_start|>user\nV<|im_end|>\n" *
              "<|im_start|>assistant\n"
        reasoning = [
            (role="user", content="U"),
            (role="assistant", content="<think>\nR\n</think>\nA"),
        ]
        @test occursin("<think>\nR\n</think>\n\nA<|im_end|>",
            apply_qwen3_chat_template(tokenizer, reasoning; enable_thinking=false))
        @test_throws ArgumentError apply_qwen3_chat_template(tokenizer, [])
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="tool", content="result")],
        )
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="assistant", content="A", tool_calls=[(; name="x")])],
        )
    end
end

@testset "Qwen3 tokenizer fail-closed config parsing" begin
    mktempdir() do directory
        @test_throws ArgumentError load_hf_qwen3_tokenizer(directory)
    end

    mutations = [
        payloads -> (payloads.tokenizer["normalizer"]["type"] = "NFKC"),
        payloads -> (payloads.tokenizer["pre_tokenizer"]["pretokenizers"][1]["pattern"]["Regex"] = "\\w+"),
        payloads -> (payloads.tokenizer["added_tokens"][1]["lstrip"] = true),
        payloads -> (payloads.tokenizer["model"]["vocab"]["!"] = 1),
        payloads -> (payloads.tokenizer_config["added_tokens_decoder"]["258"]["content"] = "bad"),
        payloads -> push!(payloads.tokenizer_config["additional_special_tokens"], "bad"),
        payloads -> (payloads.tokenizer_config["chat_template"] = "changed-template"),
        payloads -> (payloads.generation_config["pad_token_id"] = 999),
    ]
    for mutate! in mutations
        mktempdir() do directory
            payloads = week08_tokenizer_payloads()
            mutate!(payloads)
            write_week08_tokenizer_fixture(directory; payloads)
            @test_throws ArgumentError load_hf_qwen3_tokenizer(directory)
        end
    end
end

@testset "Qwen3 tokenizer artifact and checkpoint round-trip" begin
    mktempdir() do directory
        tokenizer_dir = joinpath(directory, "tokenizer")
        tokenizer = load_hf_qwen3_tokenizer(
            write_week08_tokenizer_fixture(tokenizer_dir);
            revision="artifact-revision",
        )
        artifact_path = joinpath(directory, "tokenizer.toml")
        save_tokenizer(artifact_path, tokenizer)
        restored = load_tokenizer(artifact_path)
        @test restored isa HFQwen3Tokenizer
        @test tokenizer_fingerprint(restored) == tokenizer_fingerprint(tokenizer)
        @test encode(restored, "hi! <think>x</think>") ==
              encode(tokenizer, "hi! <think>x</think>")

        model = GPTModel(vocab_size(tokenizer) + 3, 8, 2, 1; max_seq_len=8)
        trainer = TrainerGPT(learning_rate=1.0f-3, return_gradients=false)
        state = init_train_state(Xoshiro(808), model, trainer)
        checkpoint_path = joinpath(directory, "week08.checkpoint")
        save_checkpoint(checkpoint_path, model, tokenizer, trainer, state)
        checkpoint = load_checkpoint(checkpoint_path; backend=:zygote)
        @test checkpoint.tokenizer isa HFQwen3Tokenizer
        @test checkpoint.model.vocab_size == vocab_size(tokenizer) + 3
        @test tokenizer_fingerprint(checkpoint.tokenizer) == tokenizer_fingerprint(tokenizer)
    end
end

@testset "Qwen3 host text generation cache matrix" begin
    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_week08_tokenizer_fixture(directory))
        model = GPTModel(
            vocab_size(tokenizer),
            8,
            2,
            1;
            max_seq_len=8,
            use_rope=true,
            norm_type=:rmsnorm,
            mlp_type=:swiglu,
            tie_embeddings=true,
        )
        parameters, states = Lux.setup(Xoshiro(809), model)
        bundle = (; model, parameters, states, tokenizer)
        results = Dict(
            mode => generate_hf_text(
                bundle,
                "hi";
                cache=mode,
                max_new_tokens=2,
                stop_token_ids=Int[],
            ) for mode in (:full, :dynamic, :static)
        )
        @test results[:full].prompt_ids == [257]
        @test results[:full].token_ids == results[:dynamic].token_ids == results[:static].token_ids
        @test results[:full].generated_ids == results[:dynamic].generated_ids == results[:static].generated_ids
        @test length(results[:dynamic].trace) == 2
        @test results[:dynamic].stop_reason == :length
        @test results[:dynamic].completion == decode(tokenizer, results[:dynamic].generated_ids; errors=:replace)
        @test_throws ArgumentError generate_hf_text(bundle, ""; max_new_tokens=1)
        @test_throws ArgumentError generate_hf_text(bundle, "hi"; strategy=:unknown)
        @test_throws ArgumentError generate_hf_text(bundle, "hi"; cache=:xla)
    end
end

const _WEEK08_MODEL_DIR = get(ENV, "LIFEAI_QWEN3_MODEL_DIR", "")
const _WEEK08_REFERENCE_DIR = get(ENV, "LIFEAI_QWEN3_TEXT_REFERENCE_DIR", "")

if !isempty(_WEEK08_MODEL_DIR) && !isempty(_WEEK08_REFERENCE_DIR)
    @testset "Qwen3-0.6B tokenizer and text generation integration" begin
        metadata_path = joinpath(_WEEK08_REFERENCE_DIR, "reference.json")
        @test isfile(metadata_path)
        reference = JSON3.read(read(metadata_path, String))
        revision = String(reference["revision"])
        tokenizer = load_hf_qwen3_tokenizer(_WEEK08_MODEL_DIR; revision)
        @test tokenizer.tokenizer_sha256 == String(reference["tokenizer_sha256"])
        @test tokenizer.tokenizer_config_sha256 == String(reference["tokenizer_config_sha256"])
        @test tokenizer.generation_config_sha256 == String(reference["generation_config_sha256"])

        for case in reference["tokenizer_cases"]
            text = String(case["text"])
            expected = Int.(case["ids_0_based"])
            actual = encode(tokenizer, text) .- 1
            @test actual == expected
            @test decode(tokenizer, actual .+ 1; errors=:replace) == String(case["decoded"])
            pretokenized = hf_qwen3_pretokenize(tokenizer, text)
            @test pretokenized.normalized == String(case["normalized"])
            @test [piece.symbols for piece in pretokenized.pieces] ==
                  String.(getindex.(case["pretokenized"], "symbols"))
            @test [[piece.character_start, piece.character_stop] for piece in pretokenized.pieces] ==
                  [Int.(entry["character_offsets"]) for entry in case["pretokenized"]]
        end
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
            @test prompt == String(case["prompt"])
            @test encode(tokenizer, prompt) .- 1 == Int.(case["ids_0_based"])
        end

        bundle = load_hf_qwen3_bundle(
            _WEEK08_MODEL_DIR;
            max_seq_len=64,
            revision,
        )
        reference_logits = load_safetensors(joinpath(_WEEK08_REFERENCE_DIR, "reference.safetensors"))
        generation_cases = reference["generation_cases"]
        for case in generation_cases
            prompt = String(case["prompt"])
            expected_new = Int.(case["generated_ids_0_based"])
            dynamic = generate_hf_text(
                bundle,
                prompt;
                cache=:dynamic,
                max_new_tokens=length(expected_new),
                capture_logits=true,
            )
            @test dynamic.prompt_ids .- 1 == Int.(case["prompt_ids_0_based"])
            @test dynamic.generated_ids .- 1 == expected_new
            @test dynamic.completion == String(case["completion"])
            @test String(dynamic.stop_reason) == String(case["stop_reason"])
            for (actual_step, expected_step) in zip(dynamic.trace, case["steps"])
                expected_logits = reference_logits[String(expected_step["logits_key"])]
                @test actual_step.hf_token_id == Int(expected_step["token_id_0_based"])
                @test argmax(actual_step.logits) == argmax(expected_logits)
                @test isapprox(actual_step.logits, expected_logits; atol=5.0f-3, rtol=5.0f-4)
            end

            if String(case["name"]) == "chat_no_thinking"
                full = generate_hf_text(bundle, prompt; cache=:full, max_new_tokens=length(expected_new))
                static = generate_hf_text(bundle, prompt; cache=:static, max_new_tokens=length(expected_new))
                @test full.generated_ids == dynamic.generated_ids == static.generated_ids
                @test full.completion == dynamic.completion == static.completion
            end
        end
    end
else
    @info "Skipping Qwen3 Week 08 integration; set LIFEAI_QWEN3_MODEL_DIR and LIFEAI_QWEN3_TEXT_REFERENCE_DIR"
end
