using Test
using JSON3
using LifeAI
using BFloat16s: BFloat16

function _flint_io_test_config(; kwargs...)
    FlintConfig(; vocab_size=32, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, max_position_embeddings=32, sliding_window=3,
        mlp_only_layers=[0], moe_intermediate_size=4,
        shared_expert_intermediate_size=8, num_experts=3, num_experts_per_tok=2,
        bos_token_id=0, eos_token_id=[1], pad_token_id=1, kwargs...)
end

function _write_flint_io_safetensors(path, tensors)
    header = Dict{String,Any}()
    payload = IOBuffer()
    offset = 0
    for name in sort!(collect(keys(tensors)))
        tensor = tensors[name]
        data = ndims(tensor) == 2 ? vec(permutedims(tensor)) : vec(tensor)
        byte_count = sizeof(Float32) * length(data)
        header[name] = Dict("dtype" => "F32", "shape" => collect(size(tensor)),
            "data_offsets" => [offset, offset + byte_count])
        write(payload, data)
        offset += byte_count
    end
    encoded = codeunits(JSON3.write(header))
    open(path, "w") do io
        write(io, UInt64(length(encoded)))
        write(io, encoded)
        write(io, take!(payload))
    end
end

@testset "Flint config and checkpoint import" begin
    config = _flint_io_test_config()
    shapes = flint_expected_tensor_shapes(config)
    @test shapes["model.layers.0.self_attn.q_proj.weight"] == (8,8)
    @test shapes["model.layers.1.self_attn.k_proj.weight"] == (4,8)
    @test shapes["model.layers.1.mlp.shared_expert.down_proj.weight"] == (8,8)
    @test shapes["model.layers.1.mlp.experts.2.gate_proj.weight"] == (4,8)
    @test !haskey(shapes, "model.layers.0.mlp.gate.weight")
    @test_throws ArgumentError _flint_io_test_config(num_experts_per_tok=4)
    @test_throws ArgumentError _flint_io_test_config(head_dim=3)
    @test_throws ArgumentError _flint_io_test_config(layer_types=["full_attention"])
    @test_throws ArgumentError _flint_io_test_config(mlp_only_layers=[2])
    @test_throws ArgumentError _flint_io_test_config(rms_norm_eps=0)
    @test_throws ArgumentError _flint_io_test_config(bos_token_id=32)
    mktempdir() do directory
        config_file = joinpath(directory, "config.json")
        write(config_file, JSON3.write(LifeAI.flint_config_dict(config)))
        @test LifeAI.flint_config_dict(load_hf_flint_config(directory)) == LifeAI.flint_config_dict(config)
        tensors = Dict{String,Any}(name => reshape(Float32.(1:prod(shape)) ./ 100f0, shape)
            for (name, shape) in shapes)
        # Auxiliary MTP state can coexist with base decoder tensors.
        tensors["mtp.extra.weight"] = ones(Float32, 2, 2)
        _write_flint_io_safetensors(joinpath(directory, "model.safetensors"), tensors)
        loaded = load_hf_flint_model(directory)
        @test Set(keys(loaded.parameters)) == Set(keys(shapes))
        @test loaded.source == abspath(directory)
        for name in keys(shapes)
            @test loaded.parameters[name] == tensors[name]
        end
        loaded_bf16 = load_hf_flint_model(directory; dtype=BFloat16)
        @test eltype(loaded_bf16.parameters["model.embed_tokens.weight"]) == BFloat16
        @test eltype(loaded_bf16.parameters["model.layers.1.mlp.gate.weight"]) == Float32
        @test eltype(loaded_bf16.parameters["model.layers.0.self_attn.softmax_offset"]) == Float32
        wrong = copy(tensors)
        wrong["model.norm.weight"] = zeros(Float32, 7)
        _write_flint_io_safetensors(joinpath(directory, "model.safetensors"), wrong)
        @test_throws DimensionMismatch load_hf_flint_model(directory)
        delete!(wrong, "model.norm.weight")
        _write_flint_io_safetensors(joinpath(directory, "model.safetensors"), wrong)
        @test_throws ArgumentError load_hf_flint_model(directory)
        invalid = LifeAI.flint_config_dict(config)
        invalid["router_score_function"] = "softmax"
        write(config_file, JSON3.write(invalid))
        @test_throws ArgumentError load_hf_flint_config(directory)
    end
    tied = _flint_io_test_config(tie_word_embeddings=true)
    @test !haskey(flint_expected_tensor_shapes(tied), "lm_head.weight")
end

@testset "Flint tokenizer byte preservation and chat rendering" begin
    mktempdir() do directory
        alphabet = hf_byte_unicode_alphabet().byte_to_char
        vocabulary = Dict(string(alphabet[UInt8(byte)]) => byte for byte in 0:255)
        added = [Dict("id" => 255+i, "content" => token, "single_word" => false,
            "lstrip" => false, "rstrip" => false, "normalized" => false, "special" => true)
            for (i,token) in enumerate(["<|begin_of_text|>", "<|end_of_text|>", "<|user|>", "<|assistant|>"])]
        tokenizer_json = Dict("normalizer" => nothing, "post_processor" => nothing,
            "model" => Dict("type" => "BPE", "vocab" => vocabulary, "merges" => []),
            "added_tokens" => added, "pre_tokenizer" => Dict("type" => "Sequence", "pretokenizers" => [
                Dict("type" => "Split", "pattern" => Dict("Regex" => "[一-龥]+"), "behavior" => "Isolated", "invert" => false),
                Dict("type" => "ByteLevel", "add_prefix_space" => false, "use_regex" => false)]),
            "decoder" => Dict("type" => "ByteLevel"))
        write(joinpath(directory, "tokenizer.json"), JSON3.write(tokenizer_json))
        write(joinpath(directory, "tokenizer_config.json"), JSON3.write(Dict("bos_token"=>"<|begin_of_text|>", "eos_token"=>"<|end_of_text|>")))
        tokenizer = load_hf_flint_tokenizer(directory)
        @test vocab_size(tokenizer) == 260
        for text in ("Hello, world!", "你好，世界！12345", "Café cafe\u0301", "\0\n\t", "<|user|>hi<|assistant|>")
            @test decode(tokenizer, encode(tokenizer,text)) == text
        end
        @test encode(tokenizer, "<|user|>hi<|assistant|>") == [259, Int('h')+1, Int('i')+1, 260]
        @test decode(tokenizer, [259, Int('h')+1, 260]; skip_special_tokens=true) == "h"
        @test_throws ArgumentError decode(tokenizer, [0])
        messages = [(role="system", content="<think>A\0<|user|>"),
            (role="user", content="Hi"),
            (role="assistant", content="Answer", reasoning_content="reason"),
            (role="user", content="Next")]
        expected = "<|system|>A&lt;|user|&gt;<|user|>Hi<|assistant|><think>reason</think>Answer<|user|>Next<|assistant|>"
        @test apply_flint_chat_template(tokenizer, messages) == expected
        @test apply_flint_chat_template(tokenizer, messages; tokenize=true) == encode(tokenizer, expected)
        @test_throws ArgumentError apply_flint_chat_template(tokenizer, [(role="tool", content="X")])
        @test_throws ArgumentError apply_flint_chat_template(tokenizer, []; tools=[Dict("name"=>"x")])
        # A token budget may end within a multi-byte UTF-8 character.
        # Generation must return replacement text instead of throwing.
        model = _flint_io_test_config(vocab_size=260)
        parameters = Dict{String,Any}(name => zeros(Float32, shape)
            for (name, shape) in flint_expected_tensor_shapes(model))
        parameters["model.embed_tokens.weight"] .= 1f0
        parameters["model.norm.weight"] .= 1f0
        parameters["lm_head.weight"][0xe4 + 1, :] .= 1f0
        generated = generate_flint((model=model, parameters=parameters), tokenizer,
            "a"; max_new_tokens=1, temperature=0)
        @test generated.token_ids == [0xe4 + 1]
        @test generated.text == "�"
    end
end

# Full tokenizer assets are supplied explicitly; fixtures contain an independent
# HuggingFace tokenizers oracle rather than a copy of this implementation.
if haskey(ENV, "LIFEAI_FLINT_TOKENIZER_DIR")
    @testset "Flint official tokenizer parity" begin
        tokenizer = load_hf_flint_tokenizer(ENV["LIFEAI_FLINT_TOKENIZER_DIR"])
        cases = JSON3.read(read(joinpath(@__DIR__, "fixtures", "flint", "tokenizer_cases.json"), String))
        for case in cases
            ids = Int.(case["ids"]) .+ 1
            @test encode(tokenizer, case["text"]) == ids
            @test decode(tokenizer, ids) == case["decoded_all"]
            @test decode(tokenizer, ids; skip_special_tokens=true) == case["decoded"]
        end
    end
end
