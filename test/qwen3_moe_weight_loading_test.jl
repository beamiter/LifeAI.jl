using Test
using JSON3
using Lux
using LifeAI:
    GPTModel,
    Qwen3SparseMoE,
    load_hf_qwen3_moe_config,
    load_hf_qwen3_moe_parameters

function _qwen3_moe_test_config(; kwargs...)
    return merge(
        Dict{String,Any}(
            "architectures" => ["Qwen3MoeForCausalLM"],
            "attention_bias" => false,
            "attention_dropout" => 0.0,
            "decoder_sparse_step" => 1,
            "head_dim" => 4,
            "hidden_act" => "silu",
            "hidden_size" => 8,
            "intermediate_size" => 24,
            "max_position_embeddings" => 32,
            "mlp_only_layers" => Any[],
            "model_type" => "qwen3_moe",
            "moe_intermediate_size" => 3,
            "norm_topk_prob" => true,
            "num_attention_heads" => 2,
            "num_experts" => 4,
            "num_experts_per_tok" => 2,
            "num_hidden_layers" => 1,
            "num_key_value_heads" => 1,
            "rms_norm_eps" => 1.0e-6,
            "rope_scaling" => nothing,
            "rope_theta" => 1_000_000,
            "sliding_window" => nothing,
            "tie_word_embeddings" => false,
            "use_sliding_window" => false,
            "vocab_size" => 13,
        ),
        Dict{String,Any}(String(key) => value for (key, value) in pairs(kwargs)),
    )
end

function _qwen3_moe_test_values(shape, seed)
    return reshape(Float32[
        Float32(mod(index + seed, 17) - 8) / 32.0f0
        for index in 1:prod(shape)
    ], shape)
end

function _qwen3_moe_test_tensors(model::GPTModel)
    tensors = Dict{String,Any}(
        "model.embed_tokens.weight" => _qwen3_moe_test_values(
            (model.vocab_size, model.d_model),
            1,
        ),
        "model.norm.weight" => ones(Float32, model.d_model),
        "lm_head.weight" => _qwen3_moe_test_values(
            (model.vocab_size, model.d_model),
            2,
        ),
    )
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        tensors["$prefix.input_layernorm.weight"] = ones(Float32, model.d_model)
        tensors["$prefix.self_attn.q_proj.weight"] =
            _qwen3_moe_test_values((q_dim, model.d_model), 10)
        tensors["$prefix.self_attn.k_proj.weight"] =
            _qwen3_moe_test_values((kv_dim, model.d_model), 11)
        tensors["$prefix.self_attn.v_proj.weight"] =
            _qwen3_moe_test_values((kv_dim, model.d_model), 12)
        tensors["$prefix.self_attn.o_proj.weight"] =
            _qwen3_moe_test_values((model.d_model, q_dim), 13)
        tensors["$prefix.self_attn.q_norm.weight"] = ones(Float32, model.head_dim)
        tensors["$prefix.self_attn.k_norm.weight"] = ones(Float32, model.head_dim)
        tensors["$prefix.post_attention_layernorm.weight"] = ones(Float32, model.d_model)
        tensors["$prefix.mlp.gate.weight"] =
            _qwen3_moe_test_values((model.num_experts, model.d_model), 14)
        for expert in 0:(model.num_experts - 1)
            expert_prefix = "$prefix.mlp.experts.$expert"
            tensors["$expert_prefix.gate_proj.weight"] = _qwen3_moe_test_values(
                (model.mlp_hidden_dim, model.d_model),
                20 + expert,
            )
            tensors["$expert_prefix.up_proj.weight"] = _qwen3_moe_test_values(
                (model.mlp_hidden_dim, model.d_model),
                30 + expert,
            )
            tensors["$expert_prefix.down_proj.weight"] = _qwen3_moe_test_values(
                (model.d_model, model.mlp_hidden_dim),
                40 + expert,
            )
        end
    end
    return tensors
end

@testset "Qwen3 MoE config and HuggingFace expert weight mapping" begin
    mktempdir() do directory
        path = joinpath(directory, "config.json")
        write(path, JSON3.write(_qwen3_moe_test_config()))
        config = load_hf_qwen3_moe_config(path; max_seq_len=16)
        @test config.mlp_type === :qwen3_moe
        @test config.mlp_hidden_dim == 3
        @test config.dense_mlp_hidden_dim == 24
        @test config.num_experts == 4
        @test config.experts_per_token == 2
        @test config.normalize_routing
        @test config.max_seq_len == 16

        model = GPTModel(config)
        @test model.blocks.layers.layer_1.mlp isa Qwen3SparseMoE
        @test model.num_experts == 4
        tensors = _qwen3_moe_test_tensors(model)
        parameters = load_hf_qwen3_moe_parameters(model, tensors)
        @test size(parameters.blocks.layer_1.mlp.experts.gate_proj) == (3, 8, 4)
        @test parameters.blocks.layer_1.mlp.experts.gate_proj[:, :, 3] ==
            tensors["model.layers.0.mlp.experts.2.gate_proj.weight"]
        @test parameters.blocks.layer_1.mlp.gate.weight ==
            tensors["model.layers.0.mlp.gate.weight"]
        @test Lux.parameterlength(parameters) == Lux.parameterlength(model)

        missing = copy(tensors)
        delete!(missing, "model.layers.0.mlp.experts.3.down_proj.weight")
        @test_throws ArgumentError load_hf_qwen3_moe_parameters(model, missing)
        unexpected = copy(tensors)
        unexpected["model.layers.0.mlp.shared_expert.weight"] = ones(Float32, 1)
        @test_throws ArgumentError load_hf_qwen3_moe_parameters(model, unexpected)

        write(path, JSON3.write(_qwen3_moe_test_config(decoder_sparse_step=2)))
        @test_throws ArgumentError load_hf_qwen3_moe_config(path)
        write(path, JSON3.write(_qwen3_moe_test_config(mlp_only_layers=[0])))
        @test_throws ArgumentError load_hf_qwen3_moe_config(path)
        write(path, JSON3.write(_qwen3_moe_test_config(num_experts_per_tok=5)))
        @test_throws ArgumentError load_hf_qwen3_moe_config(path)
    end
end
