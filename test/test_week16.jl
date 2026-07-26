using Test
using BFloat16s: BFloat16
using JSON3
using Lux
using LifeAI:
    GPTModel,
    Int4GroupWeight,
    Int8ChannelWeight,
    hf_qwen3_bf16_accel_forward,
    hf_token_ids,
    load_hf_qwen3_model,
    load_hf_qwen3_quantized,
    load_safetensors,
    quantize_bf16_parameters

const _WEEK16_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "week16_qwen3_xla_decode_quant",
    "assets.json",
)

@testset "RTN quantization round-trip" begin
    weight = BFloat16.(randn(Float32, 12, 256))
    w_f32 = Float32.(weight)

    q8 = LifeAI._quantize_int8_channel(weight)
    @test eltype(q8.q) === Int8
    @test size(q8.q) == size(weight)
    d8 = LifeAI._dequantize_bf16(q8)
    @test eltype(d8) === BFloat16
    row_scales = vec(maximum(abs, w_f32; dims=2)) ./ 127.0f0
    # 每行误差 ≤ 半个量化步长 + BF16 舍入余量
    for row in 1:size(weight, 1)
        row_error = maximum(abs.(Float32.(d8[row, :]) .- w_f32[row, :]))
        @test row_error <= 0.5f0 * row_scales[row] + 0.01f0 * row_scales[row] * 127
    end
    @test (sizeof(q8.q) + sizeof(q8.scale)) < 0.6 * sizeof(weight)

    q4 = LifeAI._quantize_int4_group(weight; group=128)
    @test eltype(q4.packed) === UInt8
    @test size(q4.packed) == (12, 128)
    @test size(q4.scale) == (12, 2)
    d4 = LifeAI._dequantize_bf16(q4)
    @test size(d4) == size(weight)
    for row in 1:size(weight, 1), grp in 1:2
        columns = ((grp - 1) * 128 + 1):(grp * 128)
        group_error = maximum(abs.(
            Float32.(d4[row, columns]) .- w_f32[row, columns],
        ))
        @test group_error <= 0.5f0 * q4.scale[row, grp] * 1.1f0
    end
    @test (sizeof(q4.packed) + sizeof(q4.scale)) < 0.35 * sizeof(weight)

    # 打包可逆：dequant(quantize(dequant)) 不再变化（定点集合）
    q4b = LifeAI._quantize_int4_group(d4; group=128)
    @test LifeAI._dequantize_bf16(q4b) == d4

    @test_throws ArgumentError LifeAI._quantize_int4_group(weight; group=100)
    @test_throws ArgumentError LifeAI._quantize_linear(weight, :fp8, 128)
end

function _week16_fixture_dir(directory; tie=false)
    config_json = Dict{String,Any}(
        "architectures" => ["Qwen3ForCausalLM"],
        "attention_bias" => false,
        "attention_dropout" => 0.0,
        "head_dim" => 4,
        "hidden_act" => "silu",
        "hidden_size" => 8,
        "intermediate_size" => 12,
        "max_position_embeddings" => 32,
        "model_type" => "qwen3",
        "num_attention_heads" => 4,
        "num_hidden_layers" => 2,
        "num_key_value_heads" => 2,
        "rms_norm_eps" => 1.0e-6,
        "rope_scaling" => nothing,
        "rope_theta" => 1_000_000,
        "sliding_window" => nothing,
        "tie_word_embeddings" => tie,
        "torch_dtype" => "bfloat16",
        "use_sliding_window" => false,
        "vocab_size" => 19,
    )
    write(joinpath(directory, "config.json"), JSON3.write(config_json))
    config = LifeAI.load_hf_qwen3_config(
        joinpath(directory, "config.json");
        max_seq_len=16,
    )
    model = GPTModel(config)
    tensors = Dict{String,Any}()
    values_for(shape, seed; norm=false) = reshape(
        Float32[
            (norm ? 1.0f0 : 0.0f0) + Float32(mod(i + seed, 7) - 3) / 64.0f0
            for i in 1:prod(shape)
        ],
        shape,
    )
    tensors["model.embed_tokens.weight"] = values_for((model.vocab_size, model.d_model), 1)
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    for layer in 0:(model.num_layers - 1)
        prefix = "model.layers.$layer"
        tensors["$prefix.input_layernorm.weight"] = values_for((model.d_model,), 10 + layer; norm=true)
        tensors["$prefix.self_attn.q_proj.weight"] = values_for((q_dim, model.d_model), 20 + layer)
        tensors["$prefix.self_attn.k_proj.weight"] = values_for((kv_dim, model.d_model), 30 + layer)
        tensors["$prefix.self_attn.v_proj.weight"] = values_for((kv_dim, model.d_model), 40 + layer)
        tensors["$prefix.self_attn.o_proj.weight"] = values_for((model.d_model, q_dim), 50 + layer)
        tensors["$prefix.self_attn.q_norm.weight"] = values_for((model.head_dim,), 60 + layer; norm=true)
        tensors["$prefix.self_attn.k_norm.weight"] = values_for((model.head_dim,), 70 + layer; norm=true)
        tensors["$prefix.post_attention_layernorm.weight"] = values_for((model.d_model,), 80 + layer; norm=true)
        tensors["$prefix.mlp.gate_proj.weight"] = values_for((model.mlp_hidden_dim, model.d_model), 90 + layer)
        tensors["$prefix.mlp.up_proj.weight"] = values_for((model.mlp_hidden_dim, model.d_model), 100 + layer)
        tensors["$prefix.mlp.down_proj.weight"] = values_for((model.d_model, model.mlp_hidden_dim), 110 + layer)
    end
    tensors["model.norm.weight"] = values_for((model.d_model,), 120; norm=true)
    tie || (tensors["lm_head.weight"] = values_for((model.vocab_size, model.d_model), 130))

    header = Dict{String,Any}()
    data = UInt8[]
    offset = 0
    for name in sort!(collect(keys(tensors)))
        values = Float32.(tensors[name])
        flat = ndims(values) <= 1 ? vec(values) :
            vec(permutedims(values, Tuple(reverse(1:ndims(values)))))
        bits = UInt16[UInt16(reinterpret(UInt32, v) >> 16) for v in flat]
        bytes = collect(reinterpret(UInt8, bits))
        header[name] = Dict(
            "dtype" => "BF16",
            "shape" => collect(size(values)),
            "data_offsets" => [offset, offset + length(bytes)],
        )
        append!(data, bytes)
        offset += length(bytes)
    end
    header_text = JSON3.write(header)
    padding = mod(-ncodeunits(header_text), 8)
    padded = header_text * repeat(" ", padding)
    open(joinpath(directory, "model.safetensors"), "w") do io
        write(io, UInt64(ncodeunits(padded)))
        write(io, codeunits(padded))
        write(io, data)
    end
    return model
end

@testset "quantized parameter trees drive the accel forward" begin
    mktempdir() do directory
        model = _week16_fixture_dir(directory; tie=false)
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        tokens = reshape(
            hf_token_ids([0, 4, 7, 2, 11]; vocab_size=model.vocab_size),
            :,
            1,
        )
        baseline = hf_qwen3_bf16_accel_forward(
            loaded.model, loaded.parameters, tokens; greedy_steps=3,
        )
        for scheme in (:int8, :int4)
            quantized = quantize_bf16_parameters(
                loaded.parameters;
                scheme,
                group=4,
            )
            @test quantized.blocks.layer_1.attn.q_proj.weight isa
                (scheme === :int8 ? Int8ChannelWeight : Int4GroupWeight)
            @test eltype(quantized.token_embedding.weight) === BFloat16
            result = hf_qwen3_bf16_accel_forward(
                loaded.model, quantized, tokens; greedy_steps=3,
            )
            difference = maximum(abs.(
                Float32.(result.logits) .- Float32.(baseline.logits),
            ))
            @test difference > 0
            @test difference <= (scheme === :int8 ? 0.05f0 : 0.3f0)
            repeat_run = hf_qwen3_bf16_accel_forward(
                loaded.model, quantized, tokens; greedy_steps=3,
            )
            @test repeat_run.logits == result.logits

            streamed = load_hf_qwen3_quantized(
                directory;
                max_seq_len=16,
                scheme,
                group=4,
            )
            streamed_result = hf_qwen3_bf16_accel_forward(
                streamed.model, streamed.parameters, tokens; greedy_steps=3,
            )
            @test streamed_result.logits == result.logits
        end

        @test_throws ArgumentError quantize_bf16_parameters(
            load_hf_qwen3_model(directory; max_seq_len=16).parameters,
        )
    end
end

@testset "Qwen3 XLA decode and quantization asset contract" begin
    fixture = JSON3.read(read(_WEEK16_ASSETS_PATH, String))
    @test Int(fixture["greedy_steps"]) == 16

    xla = fixture["xla_decode"]
    @test String(xla["variant"]) == "qwen3_0_6b"
    @test Bool(xla["greedy_match"])
    @test Bool(xla["fast_greedy_match"])
    @test Float64(xla["fast_steady_tokens_per_second"]) >=
        10 * Float64(fixture["eager_reference_tokens_per_second"])

    for entry in fixture["quantized_models"]
        parity = entry["parity"]
        @test Bool(parity["logits_argmax_equal"])
        @test Float64(parity["gpu_tree_gib"]) < 16.0
        agreement = split(String(parity["greedy_agreement"]), "/")
        @test parse(Int, agreement[2]) == 16
        # 量化是有损近似：一致率按实测冻结（8B INT8 高、14B INT4 低），
        # 契约校验的是"漂移被如实记录"而不是假装无损。
        @test parse(Int, agreement[1]) == Int(entry["frozen_agreement"])
        @test Bool(parity["greedy_match"]) ==
            (parse(Int, agreement[1]) == 16)
        if !Bool(parity["greedy_match"])
            @test String(parity["greedy_first_divergence"]) != "none"
        end
    end
end
