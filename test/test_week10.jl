using Test
using JSON3
using Lux
using Random: Xoshiro
using Statistics: mean
import LifeAI
using LifeAI:
    GPTModel,
    decode,
    decode_bytes,
    decode_step,
    encode,
    gelu_new,
    generate_hf_text,
    gpt_config,
    hf_gpt2_forward_trace,
    hf_gpt2_pretokenize,
    hf_token_ids,
    init_kv_cache,
    init_static_kv_cache,
    load_hf_gpt2_bundle,
    load_hf_gpt2_config,
    load_hf_gpt2_parameters,
    load_tokenizer,
    load_safetensors,
    prefill,
    save_tokenizer,
    tokenizer_fingerprint

function _week10_config(; kwargs...)
    return merge(
        Dict{String,Any}(
            "activation_function" => "gelu_new",
            "architectures" => ["GPT2LMHeadModel"],
            "attn_pdrop" => 0.1,
            "bos_token_id" => 6,
            "embd_pdrop" => 0.1,
            "eos_token_id" => 6,
            "initializer_range" => 0.02,
            "layer_norm_epsilon" => 1.0e-5,
            "model_type" => "gpt2",
            "n_ctx" => 4,
            "n_embd" => 4,
            "n_head" => 2,
            "n_layer" => 1,
            "n_positions" => 4,
            "resid_pdrop" => 0.1,
            "summary_activation" => nothing,
            "summary_first_dropout" => 0.1,
            "summary_proj_to_labels" => true,
            "summary_type" => "cls_index",
            "summary_use_proj" => true,
            "task_specific_params" => Dict(
                "text-generation" => Dict("do_sample" => true, "max_length" => 50),
            ),
            "vocab_size" => 7,
        ),
        Dict{String,Any}(String(key) => value for (key, value) in pairs(kwargs)),
    )
end

function _week10_causal_buffer(length)
    values = zeros(Float32, 1, 1, length, length)
    for query in 1:length, key in 1:query
        values[1, 1, query, key] = 1.0f0
    end
    return values
end

function _week10_values(shape, offset=0)
    return reshape(
        Float32[(index + offset) / 100 for index in 1:prod(shape)],
        shape,
    )
end

function _week10_tensors(model::GPTModel; source_max_seq_len=model.max_seq_len)
    d = model.d_model
    hidden = model.mlp_hidden_dim
    tensors = Dict{String,Any}(
        "wte.weight" => _week10_values((model.vocab_size, d), 1),
        "wpe.weight" => _week10_values((source_max_seq_len, d), 2),
        "ln_f.weight" => ones(Float32, d),
        "ln_f.bias" => zeros(Float32, d),
    )
    for layer in 0:(model.num_layers - 1)
        prefix = "h.$layer"
        tensors["$prefix.attn.bias"] = _week10_causal_buffer(source_max_seq_len)
        tensors["$prefix.attn.c_attn.weight"] = _week10_values((d, 3d), 10)
        tensors["$prefix.attn.c_attn.bias"] = _week10_values((3d,), 20)
        tensors["$prefix.attn.c_proj.weight"] = _week10_values((d, d), 30)
        tensors["$prefix.attn.c_proj.bias"] = _week10_values((d,), 40)
        tensors["$prefix.ln_1.weight"] = ones(Float32, d)
        tensors["$prefix.ln_1.bias"] = zeros(Float32, d)
        tensors["$prefix.ln_2.weight"] = ones(Float32, d)
        tensors["$prefix.ln_2.bias"] = zeros(Float32, d)
        tensors["$prefix.mlp.c_fc.weight"] = _week10_values((d, hidden), 50)
        tensors["$prefix.mlp.c_fc.bias"] = _week10_values((hidden,), 60)
        tensors["$prefix.mlp.c_proj.weight"] = _week10_values((hidden, d), 70)
        tensors["$prefix.mlp.c_proj.bias"] = _week10_values((d,), 80)
    end
    return tensors
end

@testset "GPT-2 GELU-New and shared architecture contract" begin
    inputs = Float32[-3, -1, 0, 1, 3]
    pytorch_reference = Float32[
        -0.0036374330520629883,
        -0.15880802273750305,
        0.0,
        0.8411920070648193,
        2.9963626861572266,
    ]
    @test gelu_new(inputs) ≈ pytorch_reference atol = 2.0f-7 rtol = 2.0f-7

    legacy = GPTModel(11, 4, 2, 1; max_seq_len=4)
    legacy_ps = Lux.initialparameters(Xoshiro(1), legacy)
    @test keys(legacy_ps) == (:token_embedding, :blocks, :final_norm, :lm_head)
    @test legacy.position_embedding_type === :rope
    @test GPTModel(gpt_config(legacy)).position_embedding_type === :rope

    model = GPTModel(
        7,
        4,
        2,
        1;
        max_seq_len=4,
        use_rope=false,
        position_embedding_type=:learned_absolute,
        use_bias=true,
        lm_head_bias=false,
        mlp_type=:gelu_new,
        tie_embeddings=true,
    )
    ps = Lux.initialparameters(Xoshiro(2), model)
    st = Lux.initialstates(Xoshiro(2), model)
    @test haskey(ps, :position_embedding)
    @test isempty(ps.lm_head)
    @test haskey(ps.blocks.layer_1.attn.q_proj, :bias)
    @test haskey(ps.blocks.layer_1.mlp.layer_1, :bias)
    tokens = reshape([1, 2, 3], :, 1)
    embedded, _ = model.token_embedding(tokens, ps.token_embedding, st.token_embedding)
    actual = LifeAI._add_position_embedding(model, embedded, ps, 1)
    expected = embedded .+ reshape(ps.position_embedding.weight[:, 1:3], 4, 3, 1)
    @test actual == expected
    @test_throws AssertionError model(reshape([1, 2, 3, 4, 5], :, 1), ps, st)
    @test_throws ArgumentError GPTModel(
        7,
        4,
        2,
        1;
        use_rope=true,
        position_embedding_type=:learned_absolute,
    )
end

@testset "GPT-2 strict config and Conv1D mapping" begin
    mktempdir() do directory
        path = joinpath(directory, "config.json")
        write(path, JSON3.write(_week10_config()))
        config = load_hf_gpt2_config(path)
        @test config.position_embedding_type === :learned_absolute
        @test config.mlp_type === :gelu_new
        @test config.use_bias
        @test !config.lm_head_bias
        @test config.tie_embeddings
        @test config.source_max_seq_len == 4
        @test load_hf_gpt2_config(path; max_seq_len=3).max_seq_len == 3
        @test_throws ArgumentError load_hf_gpt2_config(path; max_seq_len=5)
        @test_throws ArgumentError load_hf_gpt2_bundle(
            directory;
            revision="mutable-or-unknown",
        )

        bad_path = joinpath(directory, "bad.json")
        write(bad_path, JSON3.write(_week10_config(activation_function="gelu")))
        @test_throws ArgumentError load_hf_gpt2_config(bad_path)
        write(bad_path, JSON3.write(_week10_config(extra_field=true)))
        @test_throws ArgumentError load_hf_gpt2_config(bad_path)

        model = GPTModel(config)
        tensors = _week10_tensors(model)
        parameters = load_hf_gpt2_parameters(model, tensors)
        fused = tensors["h.0.attn.c_attn.weight"]
        fused_bias = tensors["h.0.attn.c_attn.bias"]
        @test parameters.blocks.layer_1.attn.q_proj.weight ==
            permutedims(fused[:, 1:4], (2, 1))
        @test parameters.blocks.layer_1.attn.k_proj.weight ==
            permutedims(fused[:, 5:8], (2, 1))
        @test parameters.blocks.layer_1.attn.v_proj.weight ==
            permutedims(fused[:, 9:12], (2, 1))
        @test parameters.blocks.layer_1.attn.k_proj.bias == fused_bias[5:8]
        @test parameters.position_embedding.weight ==
            permutedims(tensors["wpe.weight"], (2, 1))
        @test isempty(parameters.lm_head)

        tied = copy(tensors)
        tied["lm_head.weight"] = copy(tied["wte.weight"])
        @test load_hf_gpt2_parameters(model, tied).token_embedding ==
            parameters.token_embedding
        tied["lm_head.weight"][1] += 1
        @test_throws ArgumentError load_hf_gpt2_parameters(model, tied)

        missing = copy(tensors)
        delete!(missing, "h.0.mlp.c_fc.weight")
        @test_throws ArgumentError load_hf_gpt2_parameters(model, missing)
        unexpected = copy(tensors)
        unexpected["unknown"] = zeros(Float32, 1)
        @test_throws ArgumentError load_hf_gpt2_parameters(model, unexpected)
        bad_mask = copy(tensors)
        bad_mask["h.0.attn.bias"] = copy(bad_mask["h.0.attn.bias"])
        bad_mask["h.0.attn.bias"][1, 1, 1, 4] = 1
        @test_throws ArgumentError load_hf_gpt2_parameters(model, bad_mask)
    end
end

@testset "learned position full/dynamic/static cache agreement" begin
    model = GPTModel(
        7,
        4,
        2,
        1;
        max_seq_len=4,
        use_rope=false,
        position_embedding_type=:learned_absolute,
        use_bias=true,
        lm_head_bias=false,
        mlp_type=:gelu_new,
        tie_embeddings=true,
    )
    ps = load_hf_gpt2_parameters(model, _week10_tensors(model))
    st = Lux.initialstates(Xoshiro(3), model)
    prompt = [1, 2]
    next_id = 3
    full, _ = model(reshape([prompt; next_id], :, 1), ps, st)

    dynamic = init_kv_cache(model)
    _, dynamic, dynamic_state = prefill(model, ps, st, prompt, dynamic)
    dynamic_logits, dynamic, _ = decode_step(
        model,
        ps,
        dynamic_state,
        next_id,
        dynamic,
    )
    @test dynamic.position == 3
    @test dynamic_logits[:, 1, 1] ≈ full[:, 3, 1] atol = 1.0f-5 rtol = 1.0f-5

    static = init_static_kv_cache(model)
    _, static, static_state = prefill(model, ps, st, prompt, static)
    static_logits, static, _ = decode_step(
        model,
        ps,
        static_state,
        next_id,
        static,
    )
    @test static.position == 3
    @test static_logits[:, 1, 1] ≈ full[:, 3, 1] atol = 1.0f-5 rtol = 1.0f-5
end

model_dir = get(ENV, "LIFEAI_GPT2_MODEL_DIR", "")
reference_dir = get(ENV, "LIFEAI_GPT2_REFERENCE_DIR", "")
if !isempty(model_dir) && !isempty(reference_dir)
    @testset "frozen GPT-2 124M tokenizer, layer, cache and text parity" begin
        metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
        reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))
        bundle = load_hf_gpt2_bundle(
            model_dir;
            revision=String(metadata.revision),
            max_seq_len=64,
        )
        @test Lux.parameterlength(bundle.model) == 123_702_528
        @test bundle.checksums.model_safetensors ==
            String(metadata.files["model.safetensors"])
        mktempdir() do directory
            artifact = save_tokenizer(
                joinpath(directory, "gpt2-tokenizer.toml"),
                bundle.tokenizer,
            )
            restored = load_tokenizer(artifact)
            @test restored isa LifeAI.HFGPT2Tokenizer
            @test tokenizer_fingerprint(restored) ==
                tokenizer_fingerprint(bundle.tokenizer)
        end
        checkpoint_payload = LifeAI._tokenizer_payload(bundle.tokenizer)
        checkpoint_tokenizer = LifeAI._tokenizer_from_payload(checkpoint_payload)
        @test checkpoint_tokenizer isa LifeAI.HFGPT2Tokenizer
        @test tokenizer_fingerprint(checkpoint_tokenizer) ==
            tokenizer_fingerprint(bundle.tokenizer)

        for item in metadata.corpus
            ids = encode(
                bundle.tokenizer,
                String(item.text);
                add_special_tokens=false,
            )
            @test ids .- 1 == Int.(collect(item.ids))
            @test decode(bundle.tokenizer, ids; skip_special_tokens=false) ==
                String(item.decoded)
            @test decode(bundle.tokenizer, ids; skip_special_tokens=true) ==
                String(item.decoded_skip_special)
            @test [
                bytes2hex(bundle.tokenizer.token_bytes[id]) for id in ids
            ] == String.(collect(item.token_bytes_hex))
            pieces = hf_gpt2_pretokenize(bundle.tokenizer, String(item.text))
            @test [piece.symbols for piece in pieces] ==
                [String(piece.text) for piece in item.pretokenized]
            @test [
                [piece.character_start, piece.character_stop] for piece in pieces
            ] == [Int.(collect(piece.offset)) for piece in item.pretokenized]
        end

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
        @test maximum(abs.(to_hf(trace.embedding) .- reference["embedding"])) == 0
        block_errors = Float32[]
        for index in 1:12
            expected = reference["block_" * lpad(string(index - 1), 2, "0")]
            push!(block_errors, maximum(abs.(to_hf(trace.blocks[index]) .- expected)))
        end
        @test maximum(block_errors) <= 5.0f-4
        @test maximum(abs.(
            to_hf(trace.final_hidden) .- reference["final_hidden"],
        )) <= 1.0f-4
        @test maximum(abs.(to_hf(trace.logits) .- reference["logits"])) <= 1.5f-4

        expected_ids = Int.(collect(metadata.generated_ids))
        expected_logits = reference["greedy_step_logits"]
        for mode in (:full, :dynamic, :static)
            result = generate_hf_text(
                bundle,
                String(metadata.prompt);
                cache=mode,
                strategy=:greedy,
                max_new_tokens=8,
                capture_logits=true,
            )
            @test result.generated_ids .- 1 == expected_ids
            @test result.completion == String(metadata.completion)
            @test result.text == String(metadata.text)
            errors = [
                maximum(abs.(
                    result.trace[index].logits .-
                    vec(@view(expected_logits[index, :])),
                )) for index in 1:8
            ]
            @test maximum(errors) <= 1.5f-4
        end
    end
else
    @info "Skipping GPT-2 Week 10 integration; set LIFEAI_GPT2_MODEL_DIR and LIFEAI_GPT2_REFERENCE_DIR"
end
