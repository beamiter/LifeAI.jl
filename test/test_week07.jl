using Test
using JSON3
using Lux
using Optimisers
using Random: Xoshiro
import LifeAI
using LifeAI:
    GPTModel,
    Tokenizer,
    TrainerGPT,
    decode_step,
    fit_tokenizer,
    gpt_config,
    hf_qwen3_forward_trace,
    hf_token_ids,
    init_kv_cache,
    init_static_kv_cache,
    load_checkpoint,
    load_hf_qwen3_config,
    load_hf_qwen3_model,
    load_hf_qwen3_parameters,
    load_safetensors,
    prefill,
    save_checkpoint

function _week07_row_major_values(array)
    values = Float32.(array)
    ndims(values) <= 1 && return vec(values)
    return vec(permutedims(values, Tuple(reverse(1:ndims(values)))))
end

function _week07_tensor_bytes(array, dtype::String)
    values = _week07_row_major_values(array)
    if dtype == "F32"
        return collect(reinterpret(UInt8, values))
    elseif dtype == "BF16"
        bits = UInt16[UInt16(reinterpret(UInt32, value) >> 16) for value in values]
        return collect(reinterpret(UInt8, bits))
    end
    error("unsupported test dtype $dtype")
end

function _week07_write_safetensors(path, specs)
    header = Dict{String,Any}()
    data = UInt8[]
    offset = 0
    for spec in specs
        bytes = _week07_tensor_bytes(spec.array, spec.dtype)
        next_offset = offset + length(bytes)
        header[spec.name] = Dict(
            "dtype" => spec.dtype,
            "shape" => collect(size(spec.array)),
            "data_offsets" => [offset, next_offset],
        )
        append!(data, bytes)
        offset = next_offset
    end
    header_text = JSON3.write(header)
    padding = mod(-ncodeunits(header_text), 8)
    padded_header = header_text * repeat(" ", padding)
    open(path, "w") do io
        write(io, UInt64(ncodeunits(padded_header)))
        write(io, codeunits(padded_header))
        write(io, data)
    end
    return path
end

function _week07_write_raw_safetensors(path, header, data::Vector{UInt8})
    header_text = JSON3.write(header)
    padding = mod(-ncodeunits(header_text), 8)
    padded_header = header_text * repeat(" ", padding)
    open(path, "w") do io
        write(io, UInt64(ncodeunits(padded_header)))
        write(io, codeunits(padded_header))
        write(io, data)
    end
    return path
end

function _week07_config(; kwargs...)
    return merge(
        Dict{String,Any}(
            "architectures" => ["Qwen3ForCausalLM"],
            "attention_bias" => false,
            "attention_dropout" => 0.0,
            "head_dim" => 4,
            "hidden_act" => "silu",
            "hidden_size" => 8,
            "intermediate_size" => 12,
            "max_position_embeddings" => 32,
            "model_type" => "qwen3",
            "num_attention_heads" => 2,
            "num_hidden_layers" => 2,
            "num_key_value_heads" => 1,
            "rms_norm_eps" => 1.0e-6,
            "rope_scaling" => nothing,
            "rope_theta" => 1_000_000,
            "sliding_window" => nothing,
            "tie_word_embeddings" => true,
            "torch_dtype" => "bfloat16",
            "use_sliding_window" => false,
            "vocab_size" => 13,
        ),
        Dict{String,Any}(String(key) => value for (key, value) in pairs(kwargs)),
    )
end

function _week07_write_config(path; kwargs...)
    write(path, JSON3.write(_week07_config(; kwargs...)))
    return path
end

function _week07_values(shape; norm=false, seed=0)
    count = prod(shape)
    values = Float32[
        (norm ? 1.0f0 : 0.0f0) + Float32(mod(index + seed, 7) - 3) / 64.0f0
        for index in 1:count
    ]
    return reshape(values, shape)
end

function _week07_qwen_tensors(model::GPTModel)
    tensors = Dict{String,Any}()
    tensors["model.embed_tokens.weight"] = _week07_values(
        (model.vocab_size, model.d_model);
        seed=1,
    )
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        q_dim = model.num_heads * model.head_dim
        kv_dim = model.num_kv_heads * model.head_dim
        tensors["$prefix.input_layernorm.weight"] = _week07_values(
            (model.d_model,);
            norm=true,
            seed=10 + layer,
        )
        tensors["$prefix.self_attn.q_proj.weight"] = _week07_values(
            (q_dim, model.d_model);
            seed=20 + layer,
        )
        tensors["$prefix.self_attn.k_proj.weight"] = _week07_values(
            (kv_dim, model.d_model);
            seed=30 + layer,
        )
        tensors["$prefix.self_attn.v_proj.weight"] = _week07_values(
            (kv_dim, model.d_model);
            seed=40 + layer,
        )
        tensors["$prefix.self_attn.o_proj.weight"] = _week07_values(
            (model.d_model, q_dim);
            seed=50 + layer,
        )
        tensors["$prefix.self_attn.q_norm.weight"] = _week07_values(
            (model.head_dim,);
            norm=true,
            seed=60 + layer,
        )
        tensors["$prefix.self_attn.k_norm.weight"] = _week07_values(
            (model.head_dim,);
            norm=true,
            seed=70 + layer,
        )
        tensors["$prefix.post_attention_layernorm.weight"] = _week07_values(
            (model.d_model,);
            norm=true,
            seed=80 + layer,
        )
        tensors["$prefix.mlp.gate_proj.weight"] = _week07_values(
            (model.mlp_hidden_dim, model.d_model);
            seed=90 + layer,
        )
        tensors["$prefix.mlp.up_proj.weight"] = _week07_values(
            (model.mlp_hidden_dim, model.d_model);
            seed=100 + layer,
        )
        tensors["$prefix.mlp.down_proj.weight"] = _week07_values(
            (model.d_model, model.mlp_hidden_dim);
            seed=110 + layer,
        )
    end
    tensors["model.norm.weight"] = _week07_values(
        (model.d_model,);
        norm=true,
        seed=120,
    )
    model.tie_embeddings || (tensors["lm_head.weight"] = _week07_values(
        (model.vocab_size, model.d_model);
        seed=130,
    ))
    return tensors
end

function _week07_tensor_specs(tensors; dtype="BF16")
    return [
        (; name, dtype, array=tensors[name]) for name in sort!(collect(keys(tensors)))
    ]
end

@testset "rotate_half RoPE and config compatibility" begin
    x = reshape(Float32.(1:16), 8, 1, 2, 1)
    rope = LifeAI.RoPE(8; max_seq_len=8, theta=100.0, style=:rotate_half)
    actual = LifeAI.apply_rope(x, rope; start_pos=3)
    expected = similar(x)
    for token in 1:2, pair in 1:4
        angle = Float32(token + 1) * rope.inv_freq[pair]
        first = x[pair, 1, token, 1]
        second = x[pair + 4, 1, token, 1]
        expected[pair, 1, token, 1] = first * cos(angle) - second * sin(angle)
        expected[pair + 4, 1, token, 1] = first * sin(angle) + second * cos(angle)
    end
    @test actual ≈ expected atol = 1.0f-6 rtol = 1.0f-6
    @test actual != LifeAI.apply_rope(
        x,
        LifeAI.RoPE(8; max_seq_len=8, theta=100.0, style=:interleaved);
        start_pos=3,
    )
    @test_throws ArgumentError LifeAI.RoPE(8; style=:unknown)

    legacy = GPTModel(17, 8, 2, 1; head_dim=4)
    @test legacy.rope_style === :interleaved
    @test gpt_config(legacy).rope_style === :interleaved
    legacy_config = Base.structdiff(gpt_config(legacy), (; rope_style=:interleaved))
    @test GPTModel(legacy_config).rope_style === :interleaved
end

@testset "Qwen3 config parsing and validation" begin
    mktempdir() do directory
        path = _week07_write_config(joinpath(directory, "config.json"))
        config = load_hf_qwen3_config(path; max_seq_len=16)
        @test config.vocab_size == 13
        @test config.d_model == 8
        @test config.num_heads == 2
        @test config.num_kv_heads == 1
        @test config.head_dim == 4
        @test config.rope_style === :rotate_half
        @test config.norm_type === :rmsnorm
        @test config.mlp_type === :swiglu
        @test config.tie_embeddings
        @test config.max_seq_len == 16

        @test_throws ArgumentError load_hf_qwen3_config(path; max_seq_len=33)
        bad_model = _week07_write_config(
            joinpath(directory, "bad-model.json");
            model_type="llama",
        )
        @test_throws ArgumentError load_hf_qwen3_config(bad_model)
        bad_dropout = _week07_write_config(
            joinpath(directory, "bad-dropout.json");
            attention_dropout=0.1,
        )
        @test_throws ArgumentError load_hf_qwen3_config(bad_dropout)
        bad_sliding = _week07_write_config(
            joinpath(directory, "bad-sliding.json");
            use_sliding_window=true,
            sliding_window=8,
        )
        @test_throws ArgumentError load_hf_qwen3_config(bad_sliding)
        write(joinpath(directory, "invalid.json"), "[")
        @test_throws ArgumentError load_hf_qwen3_config(joinpath(directory, "invalid.json"))
    end
end

@testset "strict safetensors BF16/F32 and sharded index" begin
    mktempdir() do directory
        matrix = Float32[1 2 3; 4 5 6]
        vector = Float32[-2.5, 0.0, 3.25]
        single = _week07_write_safetensors(
            joinpath(directory, "single.safetensors"),
            [
                (; name="matrix", dtype="F32", array=matrix),
                (; name="vector", dtype="BF16", array=vector),
            ],
        )
        loaded = load_safetensors(single)
        @test loaded["matrix"] == matrix
        @test loaded["vector"] == vector
        @test size(loaded["matrix"]) == (2, 3)
        @test_throws ArgumentError load_safetensors(single; target_dtype=Float64)

        shard1 = _week07_write_safetensors(
            joinpath(directory, "part-1.safetensors"),
            [(; name="a", dtype="F32", array=Float32[1, 2])],
        )
        shard2 = _week07_write_safetensors(
            joinpath(directory, "part-2.safetensors"),
            [(; name="b", dtype="BF16", array=Float32[3, 4])],
        )
        @test isfile(shard1) && isfile(shard2)
        index_path = joinpath(directory, "model.safetensors.index.json")
        write(index_path, JSON3.write(Dict(
            "metadata" => Dict("total_size" => 16),
            "weight_map" => Dict(
                "a" => "part-1.safetensors",
                "b" => "part-2.safetensors",
            ),
        )))
        sharded = load_safetensors(index_path)
        @test sharded == Dict("a" => Float32[1, 2], "b" => Float32[3, 4])

        bad_dtype = Dict(
            "x" => Dict(
                "dtype" => "F16",
                "shape" => [1],
                "data_offsets" => [0, 2],
            ),
        )
        bad_dtype_path = _week07_write_raw_safetensors(
            joinpath(directory, "bad-dtype.safetensors"),
            bad_dtype,
            zeros(UInt8, 2),
        )
        @test_throws ArgumentError load_safetensors(bad_dtype_path)

        bad_offsets = Dict(
            "x" => Dict(
                "dtype" => "F32",
                "shape" => [1],
                "data_offsets" => [1, 5],
            ),
        )
        bad_offsets_path = _week07_write_raw_safetensors(
            joinpath(directory, "bad-offsets.safetensors"),
            bad_offsets,
            zeros(UInt8, 5),
        )
        @test_throws ArgumentError load_safetensors(bad_offsets_path)
        write(joinpath(directory, "short.safetensors"), UInt8[1, 2, 3])
        @test_throws ArgumentError load_safetensors(joinpath(directory, "short.safetensors"))
    end
end

@testset "Qwen3 parameter mapping, trace, cache, and checkpoint" begin
    mktempdir() do directory
        _week07_write_config(joinpath(directory, "config.json"))
        config = load_hf_qwen3_config(joinpath(directory, "config.json"); max_seq_len=16)
        model = GPTModel(config)
        tensors = _week07_qwen_tensors(model)
        _week07_write_safetensors(
            joinpath(directory, "model.safetensors"),
            _week07_tensor_specs(tensors),
        )

        loaded = load_hf_qwen3_model(directory; max_seq_len=16)
        @test loaded.config == config
        @test loaded.model.rope_style === :rotate_half
        @test loaded.parameters.token_embedding.weight ==
            permutedims(tensors["model.embed_tokens.weight"], (2, 1))
        @test loaded.parameters.blocks.layer_1.attn.q_proj.weight ==
            tensors["model.layers.0.self_attn.q_proj.weight"]
        @test loaded.parameters.blocks.layer_2.mlp.down_proj.weight ==
            tensors["model.layers.1.mlp.down_proj.weight"]
        @test isempty(loaded.parameters.lm_head)

        tied_duplicate = copy(tensors)
        tied_duplicate["lm_head.weight"] = copy(tensors["model.embed_tokens.weight"])
        @test isempty(load_hf_qwen3_parameters(model, tied_duplicate).lm_head)
        tied_mismatch = copy(tied_duplicate)
        tied_mismatch["lm_head.weight"] = copy(tied_mismatch["lm_head.weight"])
        tied_mismatch["lm_head.weight"][1] += 1.0f0
        @test_throws ArgumentError load_hf_qwen3_parameters(model, tied_mismatch)

        bad_tensors = copy(tensors)
        delete!(bad_tensors, "model.norm.weight")
        @test_throws ArgumentError load_hf_qwen3_parameters(model, bad_tensors)
        unexpected = copy(tensors)
        unexpected["unexpected.weight"] = Float32[1]
        @test_throws ArgumentError load_hf_qwen3_parameters(model, unexpected)
        wrong_shape = copy(tensors)
        wrong_shape["model.norm.weight"] = zeros(Float32, model.d_model + 1)
        @test_throws DimensionMismatch load_hf_qwen3_parameters(model, wrong_shape)

        @test hf_token_ids([0, 4, 12]; vocab_size=13) == [1, 5, 13]
        @test_throws ArgumentError hf_token_ids([-1])
        @test_throws ArgumentError hf_token_ids([13]; vocab_size=13)
        tokens = reshape(hf_token_ids([0, 4, 12]; vocab_size=13), :, 1)
        trace = hf_qwen3_forward_trace(
            loaded.model,
            tokens,
            loaded.parameters,
            loaded.states,
        )
        full_logits, _ = loaded.model(
            tokens,
            loaded.parameters,
            loaded.states,
        )
        @test trace.logits ≈ full_logits atol = 1.0f-6 rtol = 1.0f-6
        @test size(trace.embedding) == (8, 3, 1)
        @test length(trace.blocks) == 2
        @test all(isfinite, trace.logits)

        prompt = tokens[1:2, :]
        next_token = tokens[3, :]
        dynamic_cache = init_kv_cache(loaded.model; batch_size=1)
        prompt_logits, dynamic_cache, dynamic_state = prefill(
            loaded.model,
            loaded.parameters,
            loaded.states,
            prompt,
            dynamic_cache,
        )
        dynamic_logits, dynamic_cache, _ = decode_step(
            loaded.model,
            loaded.parameters,
            dynamic_state,
            next_token,
            dynamic_cache,
        )
        @test prompt_logits ≈ full_logits[:, 1:2, :] atol = 1.0f-5 rtol = 1.0f-5
        @test dynamic_logits ≈ full_logits[:, 3:3, :] atol = 1.0f-5 rtol = 1.0f-5

        static_cache = init_static_kv_cache(loaded.model; batch_size=1)
        static_prompt, static_cache, static_state = prefill(
            loaded.model,
            loaded.parameters,
            loaded.states,
            prompt,
            static_cache,
        )
        static_logits, static_cache, _ = decode_step(
            loaded.model,
            loaded.parameters,
            static_state,
            next_token,
            static_cache,
        )
        @test static_prompt ≈ full_logits[:, 1:2, :] atol = 1.0f-5 rtol = 1.0f-5
        @test static_logits ≈ full_logits[:, 3:3, :] atol = 1.0f-5 rtol = 1.0f-5

        tokenizer = fit_tokenizer("abcdefghijklm")
        trainer = TrainerGPT(optimizer=Optimisers.Adam(1.0f-3))
        train_state = Lux.Training.TrainState(
            loaded.model,
            loaded.parameters,
            loaded.states,
            trainer.optimizer,
        )
        checkpoint_path = joinpath(directory, "qwen3-tiny.checkpoint")
        save_checkpoint(
            checkpoint_path,
            loaded.model,
            tokenizer,
            trainer,
            train_state;
            metadata=(; week=7),
        )
        checkpoint = load_checkpoint(checkpoint_path; backend=:zygote)
        checkpoint_logits, _ = checkpoint.model(
            tokens,
            checkpoint.train_state.parameters,
            checkpoint.train_state.states,
        )
        @test checkpoint.model.rope_style === :rotate_half
        @test checkpoint_logits ≈ full_logits atol = 1.0f-6 rtol = 1.0f-6
    end
end

if haskey(ENV, "LIFEAI_QWEN3_MODEL_DIR")
    @testset "Qwen3-0.6B HuggingFace integration" begin
        model_dir = ENV["LIFEAI_QWEN3_MODEL_DIR"]
        reference_dir = get(
            ENV,
            "LIFEAI_QWEN3_REFERENCE_DIR",
            joinpath(model_dir, "lifeai_reference"),
        )
        isfile(joinpath(reference_dir, "reference.json")) || error(
            "missing reference.json; run scripts/export_qwen3_reference.py first",
        )
        metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
        reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))
        loaded = load_hf_qwen3_model(model_dir; max_seq_len=64)
        tokens = reshape(hf_token_ids(
            Int.(collect(metadata["token_ids_0_based"]));
            vocab_size=loaded.model.vocab_size,
        ), :, 1)
        trace = hf_qwen3_forward_trace(
            loaded.model,
            tokens,
            loaded.parameters,
            loaded.states,
        )
        hf_layout(array) = permutedims(array, (3, 2, 1))
        @test trace.embedding ≈ hf_layout(reference["embedding"]) atol = 2.0f-4 rtol = 2.0f-4
        for layer in 0:(loaded.model.num_layers - 1)
            @test trace.blocks[layer + 1] ≈ hf_layout(reference["block.$layer"]) atol = 2.0f-3 rtol = 2.0f-4
        end
        @test trace.final_hidden ≈ hf_layout(reference["final_hidden"]) atol = 2.0f-3 rtol = 2.0f-4
        @test trace.logits ≈ hf_layout(reference["logits"]) atol = 5.0f-3 rtol = 5.0f-4

        cache = init_kv_cache(loaded.model; batch_size=1)
        _, cache, state = prefill(
            loaded.model,
            loaded.parameters,
            loaded.states,
            tokens,
            cache,
        )
        decode_token = hf_token_ids(
            [Int(metadata["decode_token_id_0_based"])];
            vocab_size=loaded.model.vocab_size,
        )
        decode_logits, _, _ = decode_step(
            loaded.model,
            loaded.parameters,
            state,
            decode_token,
            cache,
        )
        expected_decode = hf_layout(reference["decode_logits"])
        @test decode_logits ≈ expected_decode atol = 5.0f-3 rtol = 5.0f-4
        @test argmax(vec(decode_logits)) == argmax(vec(expected_decode))

        static_cache = init_static_kv_cache(loaded.model; batch_size=1)
        _, static_cache, static_state = prefill(
            loaded.model,
            loaded.parameters,
            loaded.states,
            tokens,
            static_cache,
        )
        static_logits, _, _ = decode_step(
            loaded.model,
            loaded.parameters,
            static_state,
            decode_token,
            static_cache,
        )
        @test static_logits ≈ expected_decode atol = 5.0f-3 rtol = 5.0f-4
        @test argmax(vec(static_logits)) == argmax(vec(expected_decode))
    end
end
