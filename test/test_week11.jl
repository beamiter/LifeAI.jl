using Test
using JSON3
using Lux
using Random: Xoshiro
using LifeAI:
    GPTModel,
    decode_step,
    init_kv_cache,
    init_static_kv_cache,
    load_hf_qwen3_config,
    load_hf_qwen3_parameters,
    prefill,
    qwen3_dense_parameter_count,
    qwen3_dense_spec,
    qwen3_dense_specs

const _WEEK11_SPECS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "week11_qwen3_dense_family",
    "specs.json",
)

function _week11_values(shape, seed)
    values = Float32[
        Float32(mod(index + seed, 19) - 9) / 64.0f0
        for index in 1:prod(shape)
    ]
    return reshape(values, shape)
end

function _week11_untied_wide_tensors(model::GPTModel)
    tensors = Dict{String,Any}(
        "model.embed_tokens.weight" => _week11_values(
            (model.vocab_size, model.d_model),
            1,
        ),
        "model.norm.weight" => ones(Float32, model.d_model),
        "lm_head.weight" => _week11_values(
            (model.vocab_size, model.d_model),
            2,
        ),
    )
    query_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        tensors["$prefix.input_layernorm.weight"] = ones(Float32, model.d_model)
        tensors["$prefix.self_attn.q_proj.weight"] =
            _week11_values((query_dim, model.d_model), 10 + layer)
        tensors["$prefix.self_attn.k_proj.weight"] =
            _week11_values((kv_dim, model.d_model), 20 + layer)
        tensors["$prefix.self_attn.v_proj.weight"] =
            _week11_values((kv_dim, model.d_model), 30 + layer)
        tensors["$prefix.self_attn.o_proj.weight"] =
            _week11_values((model.d_model, query_dim), 40 + layer)
        tensors["$prefix.self_attn.q_norm.weight"] =
            ones(Float32, model.head_dim)
        tensors["$prefix.self_attn.k_norm.weight"] =
            ones(Float32, model.head_dim)
        tensors["$prefix.post_attention_layernorm.weight"] =
            ones(Float32, model.d_model)
        tensors["$prefix.mlp.gate_proj.weight"] = _week11_values(
            (model.mlp_hidden_dim, model.d_model),
            50 + layer,
        )
        tensors["$prefix.mlp.up_proj.weight"] = _week11_values(
            (model.mlp_hidden_dim, model.d_model),
            60 + layer,
        )
        tensors["$prefix.mlp.down_proj.weight"] = _week11_values(
            (model.d_model, model.mlp_hidden_dim),
            70 + layer,
        )
    end
    return tensors
end

@testset "Qwen3 official dense family contract" begin
    fixture = JSON3.read(read(_WEEK11_SPECS_PATH, String))
    entries = collect(fixture.variants)
    specs = qwen3_dense_specs()
    @test length(entries) == length(specs) == 6
    @test count(spec -> spec.tie_embeddings, specs) == 3
    @test count(spec -> !spec.tie_embeddings, specs) == 3

    mktempdir() do directory
        for (index, entry) in enumerate(entries)
            variant = Symbol(String(entry.variant))
            spec = qwen3_dense_spec(variant)
            @test spec === specs[index]
            @test qwen3_dense_spec(String(entry.model_id)) === spec
            @test qwen3_dense_spec(replace(String(entry.model_id), "Qwen/Qwen3-" => "")) ===
                spec
            @test spec.revision == String(entry.revision)
            @test spec.config_sha256 == String(entry.config_sha256)
            @test qwen3_dense_parameter_count(spec) == Int(entry.parameter_count)

            path = joinpath(directory, "$(entry.variant).json")
            write(path, JSON3.write(entry.config))
            config = load_hf_qwen3_config(
                path;
                max_seq_len=4,
                variant,
            )
            @test config.qwen3_variant === variant
            @test config.source_max_seq_len == spec.max_position_embeddings
            @test config.d_model == spec.d_model
            @test config.num_heads * config.head_dim ==
                spec.num_heads * spec.head_dim
            @test config.tie_embeddings == spec.tie_embeddings

            # Building the full layer topology is cheap at a four-token RoPE
            # limit and proves the generic GPTModel carries every official
            # width/depth/tied-head combination without allocating weights.
            model = GPTModel(config)
            @test Lux.parameterlength(model) == Int(entry.parameter_count)
        end

        first_path = joinpath(directory, "$(entries[1].variant).json")
        @test_throws ArgumentError load_hf_qwen3_config(
            first_path;
            max_seq_len=4,
            variant=:qwen3_4b,
        )
        altered_rope = Dict{String,Any}(
            String(key) => value for (key, value) in pairs(entries[1].config)
        )
        altered_rope["rope_theta"] = 10_000
        altered_rope_path = joinpath(directory, "altered-rope.json")
        write(altered_rope_path, JSON3.write(altered_rope))
        @test_throws ArgumentError load_hf_qwen3_config(
            altered_rope_path;
            max_seq_len=4,
            variant=:qwen3_0_6b,
        )

        custom = Dict{String,Any}(
            String(key) => value for (key, value) in pairs(entries[1].config)
        )
        custom["hidden_size"] = 1_536
        custom["intermediate_size"] = 4_608
        custom_path = joinpath(directory, "custom.json")
        write(custom_path, JSON3.write(custom))
        @test load_hf_qwen3_config(
            custom_path;
            max_seq_len=4,
        ).qwen3_variant === nothing

        moe = copy(custom)
        moe["model_type"] = "qwen3_moe"
        moe["num_experts"] = 128
        moe_path = joinpath(directory, "moe.json")
        write(moe_path, JSON3.write(moe))
        @test_throws ArgumentError load_hf_qwen3_config(
            moe_path;
            max_seq_len=4,
        )
    end

    @test_throws ArgumentError qwen3_dense_spec(:qwen3_moe)
end

@testset "Qwen3 untied wide-attention weight and cache path" begin
    # Scaled analogue of 32B: query projection width exceeds hidden size,
    # embeddings are untied, and GQA stores fewer K/V heads than Q heads.
    model = GPTModel(
        19,
        8,
        4,
        2;
        num_kv_heads=2,
        head_dim=4,
        mlp_hidden_dim=24,
        use_bias=false,
        use_rope=true,
        use_qk_norm=true,
        qk_norm_epsilon=1.0f-6,
        max_seq_len=8,
        rope_theta=1.0f6,
        rope_style=:rotate_half,
        norm_epsilon=1.0f-6,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=false,
    )
    @test model.num_heads * model.head_dim == 2 * model.d_model
    tensors = _week11_untied_wide_tensors(model)
    parameters = load_hf_qwen3_parameters(model, tensors)
    states = Lux.initialstates(Xoshiro(20260723), model)
    @test Lux.parameterlength(model) == Lux.parameterlength(parameters)
    @test size(parameters.lm_head.weight) == (model.vocab_size, model.d_model)
    @test parameters.lm_head.weight == tensors["lm_head.weight"]
    @test parameters.token_embedding.weight != permutedims(
        parameters.lm_head.weight,
        (2, 1),
    )

    tokens = reshape([1, 4, 7, 10], :, 1)
    full_logits, _ = model(tokens, parameters, states)
    prompt = tokens[1:3, :]
    next_token = tokens[4, :]

    dynamic = init_kv_cache(model; batch_size=1)
    prompt_logits, dynamic, dynamic_state = prefill(
        model,
        parameters,
        states,
        prompt,
        dynamic,
    )
    dynamic_logits, dynamic, _ = decode_step(
        model,
        parameters,
        dynamic_state,
        next_token,
        dynamic,
    )
    @test size(dynamic.layers[1].keys, 2) == model.num_kv_heads
    @test prompt_logits ≈ full_logits[:, 1:3, :] atol = 1.0f-5 rtol = 1.0f-5
    @test dynamic_logits ≈ full_logits[:, 4:4, :] atol = 1.0f-5 rtol = 1.0f-5

    static = init_static_kv_cache(model; batch_size=1)
    static_prompt, static, static_state = prefill(
        model,
        parameters,
        states,
        prompt,
        static,
    )
    static_logits, static, _ = decode_step(
        model,
        parameters,
        static_state,
        next_token,
        static,
    )
    @test size(static.layers[1].keys, 2) == model.num_kv_heads
    @test static_prompt ≈ full_logits[:, 1:3, :] atol = 1.0f-5 rtol = 1.0f-5
    @test static_logits ≈ full_logits[:, 4:4, :] atol = 1.0f-5 rtol = 1.0f-5
end
