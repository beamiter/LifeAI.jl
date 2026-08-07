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

isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
    include("qwen3_tiny_model_fixture.jl")

const _QWEN3_QUANTIZATION_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_xla_decode_quantization",
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

@testset "packed compact decode matches unpacked BF16 decode" begin
    for tie in (false, true)
        mktempdir() do directory
            _qwen3_tiny_model_fixture_dir(directory; tie)
            loaded = load_hf_qwen3_model(
                directory;
                max_seq_len=16,
                weight_dtype=BFloat16,
            )
            model = loaded.model
            parameters = loaded.parameters
            projections = LifeAI._bf16a_pack_decode_projections(parameters)
            compact = LifeAI._bf16a_compact_decode_parameters(
                parameters,
                projections,
            )
            q_dim = model.num_heads * model.head_dim
            kv_dim = model.num_kv_heads * model.head_dim
            for (original, packed, decode) in zip(
                values(parameters.blocks),
                values(projections.blocks),
                values(compact.blocks),
            )
                @test packed.qkv_weight[1:q_dim, :] ==
                    original.attn.q_proj.weight
                @test packed.qkv_weight[
                    (q_dim + 1):(q_dim + kv_dim),
                    :,
                ] == original.attn.k_proj.weight
                @test packed.qkv_weight[
                    (q_dim + kv_dim + 1):end,
                    :,
                ] == original.attn.v_proj.weight
                @test packed.gate_up_weight[
                    1:model.mlp_hidden_dim,
                    :,
                ] == original.mlp.gate_proj.weight
                @test packed.gate_up_weight[
                    (model.mlp_hidden_dim + 1):end,
                    :,
                ] == original.mlp.up_proj.weight
                @test !hasproperty(decode.attn, :q_proj)
                @test !hasproperty(decode.attn, :k_proj)
                @test !hasproperty(decode.attn, :v_proj)
                @test !hasproperty(decode.mlp, :gate_proj)
                @test !hasproperty(decode.mlp, :up_proj)
                @test decode.norm1 === original.norm1
                @test decode.attn.o_proj === original.attn.o_proj
                @test decode.attn.q_norm === original.attn.q_norm
                @test decode.attn.k_norm === original.attn.k_norm
                @test decode.norm2 === original.norm2
                @test decode.mlp.down_proj === original.mlp.down_proj
            end
            @test compact.token_embedding === parameters.token_embedding
            @test compact.final_norm === parameters.final_norm
            @test compact.logits_weight === projections.logits_weight
            if tie
                @test compact.logits_weight ==
                    permutedims(parameters.token_embedding.weight, (2, 1))
            else
                @test compact.logits_weight === parameters.lm_head.weight
            end

            tokens = reshape(
                hf_token_ids(
                    [0, 4, 7, 2, 11];
                    vocab_size=model.vocab_size,
                ),
                :,
                1,
            )
            rope = first(values(model.blocks.layers)).attn.rope
            cos_table = BFloat16.(rope.cos_cache)
            sin_table = BFloat16.(rope.sin_cache)
            mask = LifeAI._bf16a_causal_mask(
                size(tokens, 1),
                size(tokens, 1),
            )
            prefix_keys = [
                zeros(
                    BFloat16,
                    model.head_dim,
                    model.num_kv_heads,
                    model.max_seq_len,
                    1,
                )
                for _ in 1:model.num_layers
            ]
            prefix_values = deepcopy(prefix_keys)
            packed_prefix_keys = deepcopy(prefix_keys)
            packed_prefix_values = deepcopy(prefix_values)
            prefill_logits = LifeAI._bf16a_static_prefill(
                model,
                parameters,
                tokens,
                prefix_keys,
                prefix_values,
                cos_table,
                sin_table,
                mask,
            )
            packed_prefill_logits = LifeAI._bf16a_static_prefill(
                model,
                compact,
                tokens,
                packed_prefix_keys,
                packed_prefix_values,
                cos_table,
                sin_table,
                mask,
            )
            @test packed_prefill_logits == prefill_logits
            @test packed_prefix_keys == prefix_keys
            @test packed_prefix_values == prefix_values
            token = [argmax(vec(Float32.(prefill_logits)))]
            position = Int32[size(tokens, 1)]
            key_positions = Int32.(collect(1:model.max_seq_len))

            unpacked_keys = deepcopy(prefix_keys)
            unpacked_values = deepcopy(prefix_values)
            packed_keys = deepcopy(prefix_keys)
            packed_values = deepcopy(prefix_values)
            unpacked_logits = LifeAI._bf16a_static_decode_step(
                model,
                parameters,
                token,
                unpacked_keys,
                unpacked_values,
                position,
                cos_table,
                sin_table,
                key_positions,
            )
            packed_logits = LifeAI._bf16a_static_decode_step_packed(
                model,
                compact,
                token,
                packed_keys,
                packed_values,
                position,
                cos_table,
                sin_table,
                key_positions,
            )
            @test packed_logits == unpacked_logits
            @test packed_keys == unpacked_keys
            @test packed_values == unpacked_values

            unpacked_keys = deepcopy(prefix_keys)
            unpacked_values = deepcopy(prefix_values)
            packed_keys = deepcopy(prefix_keys)
            packed_values = deepcopy(prefix_values)
            unpacked_generated = zeros(Int, 4)
            packed_generated = zeros(Int, 4)
            LifeAI._bf16a_static_generate_greedy!(
                model,
                parameters,
                token,
                unpacked_keys,
                unpacked_values,
                copy(position),
                cos_table,
                sin_table,
                key_positions,
                unpacked_generated,
            )
            LifeAI._bf16a_static_generate_greedy_packed!(
                model,
                compact,
                token,
                packed_keys,
                packed_values,
                copy(position),
                cos_table,
                sin_table,
                key_positions,
                packed_generated,
            )
            @test packed_generated == unpacked_generated
            @test packed_keys == unpacked_keys
            @test packed_values == unpacked_values
        end
    end
end

@testset "quantized parameter trees drive the accel forward" begin
    mktempdir() do directory
        model = _qwen3_tiny_model_fixture_dir(directory; tie=false)
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
    fixture = JSON3.read(read(_QWEN3_QUANTIZATION_ASSETS_PATH, String))
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
