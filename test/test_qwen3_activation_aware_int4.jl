using Test
using BFloat16s: BFloat16
using JSON3
using SHA: sha256
using LifeAI:
    ActivationCalibration,
    estimate_qwen3_quantized_bytes,
    Int4GroupWeight,
    Int8ChannelWeight,
    LinearQuantizationSpec,
    QuantizationPlan,
    activation_second_moment,
    calibrate_hf_qwen3_activations,
    load_hf_qwen3_model,
    load_hf_qwen3_quantized,
    quantize_bf16_parameters,
    qwen3_dense_spec

isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
    include("qwen3_tiny_model_fixture.jl")
isdefined(@__MODULE__, :quantized_parameters_equal) ||
    include("qwen3_quantization_test_utils.jl")

const _QWEN3_ACTIVATION_QUANTIZATION_FIXTURE_DIR = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_activation_aware_int4",
)
const _QWEN3_ACTIVATION_QUANTIZATION_ASSETS_PATH = joinpath(_QWEN3_ACTIVATION_QUANTIZATION_FIXTURE_DIR, "assets.json")

function _qwen3_activation_quantization_weighted_error(weight, quantized, moment)
    source = Float32.(weight)
    reconstructed = Float32.(LifeAI._dequantize_bf16(quantized))
    return sum(
        abs2(reconstructed[row, column] - source[row, column]) *
            moment[column]
        for row in axes(source, 1), column in axes(source, 2)
    )
end

@testset "Qwen3-14B activation calibration asset contract" begin
    assets = JSON3.read(read(_QWEN3_ACTIVATION_QUANTIZATION_ASSETS_PATH, String))
    for (filename, expected) in pairs(assets["plan_sha256"])
        path = joinpath(_QWEN3_ACTIVATION_QUANTIZATION_FIXTURE_DIR, String(filename))
        @test bytes2hex(sha256(read(path))) == String(expected)
    end
    calibration_asset = assets["calibration"]
    calibration_path = joinpath(
        _QWEN3_ACTIVATION_QUANTIZATION_FIXTURE_DIR,
        String(calibration_asset["fixture"]),
    )
    @test bytes2hex(sha256(read(calibration_path))) ==
        String(calibration_asset["fixture_sha256"])
    generator_path = normpath(joinpath(
        @__DIR__,
        "..",
        String(calibration_asset["generator"]),
    ))
    @test bytes2hex(sha256(read(generator_path))) ==
        String(calibration_asset["generator_sha256"])
    calibration = JSON3.read(read(calibration_path, String))
    @test String(calibration["revision"]) ==
        String(assets["qwen3_14b"]["revision"])
    @test String(calibration["corpus_sha256"]) ==
        String(calibration_asset["corpus_sha256"])
    @test Int(calibration["sequence_length"]) ==
        Int(calibration_asset["sequence_length"])
    @test Int(calibration["batch_size"]) ==
        Int(calibration_asset["batch_size"])
    @test Int(calibration["sequence_length"]) *
        Int(calibration["batch_size"]) ==
        Int(calibration_asset["token_count"])
    @test all(
        length(sequence) == Int(calibration["sequence_length"])
        for sequence in calibration["token_ids_0_based"]
    )
    @test !Bool(calibration["evaluation_sequence_present"])
    evaluation = Int.(collect(calibration["evaluation_token_ids_0_based"]))
    selected = reduce(vcat, [
        Int.(collect(sequence))
        for sequence in calibration["token_ids_0_based"]
    ])
    @test !any(
        selected[index:(index + length(evaluation) - 1)] == evaluation
        for index in 1:(length(selected) - length(evaluation) + 1)
    )

    plan_fixture = JSON3.read(read(
        joinpath(_QWEN3_ACTIVATION_QUANTIZATION_FIXTURE_DIR, "plan_mixed_24g_activation.json"),
        String,
    ))
    @test String(plan_fixture["default"]["calibration"]) == "activation_mse"
    activation = LinearQuantizationSpec(
        :int4;
        group=128,
        calibration=:activation_mse,
    )
    overrides = Dict(
        projection => LinearQuantizationSpec(:int8)
        for projection in (:q_proj, :k_proj, :v_proj, :o_proj, :down_proj)
    )
    overrides[:lm_head] = LinearQuantizationSpec(:bf16)
    plan = QuantizationPlan(
        default=activation,
        projection_overrides=overrides,
    )
    @test estimate_qwen3_quantized_bytes(
        qwen3_dense_spec(:qwen3_14b),
        plan,
    ) == Int(assets["qwen3_14b"]["mixed_24g_tensor_bytes"])

    hardware = assets["hardware_validation"]
    @test String(hardware["status"]) == "complete"
    @test Bool(hardware["model_matches_frozen_huggingface_revision"])
    @test String(hardware["gpu"]) == "NVIDIA GeForce RTX 4090 D"
    run = hardware["activation_aware"]
    @test Int(run["calibration_token_count"]) ==
        Int(calibration_asset["token_count"])
    @test String(run["calibration_tokens_sha256"]) ==
        String(calibration_asset["fixture_sha256"])
    @test Float64(run["calibration_seconds"]) > 0
    @test Int(run["estimated_tree_bytes"]) == Int(run["host_tree_bytes"])
    @test abs(Int(run["gpu_tree_bytes"]) - Int(run["host_tree_bytes"])) <
        32 * 1024 * 1024
    @test Int(run["vram_used_bytes"]) <= Int(hardware["gpu_total_bytes"])
    @test Bool(run["logits_argmax_equal"])
    @test Bool(run["decode_argmax_equal"])
    @test Int(run["greedy_agreement"]) == 4
    @test Int(run["greedy_first_divergence"]) == 5
    @test length(run["greedy_token_ids_0_based"]) == 16

    baselines = hardware["calibrated_int4_baselines"]
    rtn = baselines["mixed_rtn"]
    weight_mse = baselines["mixed_weight_mse"]
    @test Int(rtn["greedy_agreement"]) == 16
    @test Int(weight_mse["greedy_agreement"]) == 4
    @test Float64(run["logits_max_abs"]) < Float64(rtn["logits_max_abs"])
    @test Float64(run["logits_max_abs"]) <
        Float64(weight_mse["logits_max_abs"])
    @test Float64(run["logits_mean_abs"]) <
        Float64(rtn["logits_mean_abs"])
    @test Float64(run["logits_mean_abs"]) >
        Float64(weight_mse["logits_mean_abs"])
end

@testset "activation-weighted INT4 scale calibration" begin
    weight = BFloat16.(reshape(Float32[
        -1.453125,
        4.15625,
        0.17382812,
        0.06347656,
        -0.37304688,
        2.578125,
        -1.1640625,
        -1.3046875,
        -2.421875,
        -0.18457031,
        2.171875,
        0.0059509277,
        0.55078125,
        -0.19921875,
        1.78125,
        3.03125,
    ], 1, :))
    moment = Float32[
        0.18041377,
        0.80778927,
        0.015428526,
        0.45216435,
        96.68152,
        0.73266363,
        0.062713064,
        31.540024,
        84.27287,
        42.711483,
        2.7510345,
        0.019945787,
        0.34113428,
        23.521303,
        0.17844394,
        12.704935,
    ]
    ratios = (1.0f0, 0.95f0, 0.9f0, 0.85f0, 0.8f0, 0.7f0)
    maxabs = LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:maxabs,
    )
    weight_mse = LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:mse,
        clip_ratios=ratios,
    )
    activation_mse = LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:activation_mse,
        clip_ratios=ratios,
        activation_second_moment=moment,
    )
    @test _qwen3_activation_quantization_weighted_error(weight, activation_mse, moment) <
        _qwen3_activation_quantization_weighted_error(weight, maxabs, moment)
    @test _qwen3_activation_quantization_weighted_error(weight, activation_mse, moment) <
        _qwen3_activation_quantization_weighted_error(weight, weight_mse, moment)
    @test activation_mse.scale != weight_mse.scale

    matrix = BFloat16.(reshape(
        Float32[
            sin(i / 9) * (iszero(i % 31) ? 9 : 1)
            for i in 1:(3 * 64)
        ],
        3,
        64,
    ))
    channel_moment = Float32[
        iszero(column % 17) ? 40.0 : 0.1 + column / 64
        for column in 1:64
    ]
    baseline = LifeAI._quantize_int4_group(matrix; group=16)
    calibrated = LifeAI._quantize_int4_group(
        matrix;
        group=16,
        calibration=:activation_mse,
        activation_second_moment=channel_moment,
    )
    baseline_dequant = Float32.(LifeAI._dequantize_bf16(baseline))
    calibrated_dequant = Float32.(LifeAI._dequantize_bf16(calibrated))
    for row in axes(matrix, 1), group_index in 1:4
        columns = ((group_index - 1) * 16 + 1):(group_index * 16)
        source = Float32.(matrix[row, columns])
        weights = channel_moment[columns]
        @test sum(weights .* abs2.(calibrated_dequant[row, columns] .- source)) <=
            sum(weights .* abs2.(baseline_dequant[row, columns] .- source))
    end

    @test_throws ArgumentError LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:activation_mse,
    )
    @test_throws DimensionMismatch LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:activation_mse,
        activation_second_moment=ones(Float32, 15),
    )
    @test_throws ArgumentError LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:activation_mse,
        activation_second_moment=vcat(-1.0f0, ones(Float32, 15)),
    )
    @test_throws ArgumentError LifeAI._quantize_int4_group(
        BFloat16.(reshape(1:32, 1, :));
        group=16,
        calibration=:activation_mse,
        activation_second_moment=vcat(zeros(Float32, 16), ones(Float32, 16)),
    )
    @test_throws ArgumentError LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:maxabs,
        activation_second_moment=moment,
    )
end

@testset "activation calibration contract fails closed" begin
    moments = Dict(
        (1, :q_proj) => ones(Float32, 8),
        (1, :gate_proj) => Float32[1, 2, 3, 4, 5, 6, 7, 8],
    )
    calibration = ActivationCalibration(
        moments;
        lm_head_moment=fill(2.0f0, 8),
        token_count=12,
        num_layers=1,
        source="synthetic",
    )
    @test calibration.token_count == 12
    @test calibration.num_layers == 1
    @test calibration.source == "synthetic"
    @test activation_second_moment(calibration, :q_proj; layer=1) ==
        ones(Float32, 8)
    @test activation_second_moment(calibration, :lm_head) ==
        fill(2.0f0, 8)
    @test_throws ArgumentError activation_second_moment(
        calibration,
        :up_proj;
        layer=1,
    )
    @test_throws ArgumentError activation_second_moment(
        calibration,
        :q_proj,
    )
    @test_throws ArgumentError activation_second_moment(
        calibration,
        :lm_head;
        layer=1,
    )
    @test_throws ArgumentError ActivationCalibration(
        moments;
        token_count=0,
        num_layers=1,
    )
    @test_throws ArgumentError ActivationCalibration(
        Dict((2, :q_proj) => ones(Float32, 8));
        token_count=1,
        num_layers=1,
    )
    @test_throws ArgumentError ActivationCalibration(
        Dict((1, :unknown) => ones(Float32, 8));
        token_count=1,
        num_layers=1,
    )
    @test_throws ArgumentError ActivationCalibration(
        Dict((1, :q_proj) => Float32[1, -1]);
        token_count=1,
        num_layers=1,
    )
    @test_throws ArgumentError ActivationCalibration(
        Dict((1, :q_proj) => Float32[1, NaN]);
        token_count=1,
        num_layers=1,
    )
    @test_throws ArgumentError ActivationCalibration(
        Dict((1, :q_proj) => zeros(Float32, 8));
        token_count=1,
        num_layers=1,
    )
end

@testset "streamed activation collection drives both quantizers" begin
    mktempdir() do directory
        model = _qwen3_tiny_model_fixture_dir(directory; tie=false)
        tokens = Int[
            1 5
            3 7
            9 11
            13 15
        ]
        first = calibrate_hf_qwen3_activations(
            directory,
            tokens;
            max_seq_len=16,
            source="qwen3_activation_quantization-test",
        )
        second = calibrate_hf_qwen3_activations(
            directory,
            tokens;
            max_seq_len=16,
            source="qwen3_activation_quantization-test",
            accelerated=true,
        )
        calibration = first.calibration
        @test calibration.token_count == length(tokens)
        @test calibration.num_layers == model.num_layers
        @test calibration.source == "qwen3_activation_quantization-test"
        @test calibration.layer_moments == second.calibration.layer_moments
        @test calibration.lm_head_moment ==
            second.calibration.lm_head_moment
        @test length(calibration.layer_moments) == 7 * model.num_layers
        for layer in 1:model.num_layers
            for projection in (:q_proj, :k_proj, :v_proj, :gate_proj, :up_proj)
                values = activation_second_moment(
                    calibration,
                    projection;
                    layer,
                )
                @test length(values) == model.d_model
                @test all(isfinite, values)
                @test all(>=(0.0f0), values)
                @test any(>(0.0f0), values)
            end
            @test length(activation_second_moment(
                calibration,
                :o_proj;
                layer,
            )) == model.num_heads * model.head_dim
            @test length(activation_second_moment(
                calibration,
                :down_proj;
                layer,
            )) == model.mlp_hidden_dim
        end
        @test length(activation_second_moment(calibration, :lm_head)) ==
            model.d_model

        activation = LinearQuantizationSpec(
            :int4;
            group=4,
            calibration=:activation_mse,
            clip_ratios=(1.0f0, 0.9f0, 0.8f0),
        )
        int8 = LinearQuantizationSpec(:int8; group=4)
        bf16 = LinearQuantizationSpec(:bf16; group=4)
        plan = QuantizationPlan(
            default=activation,
            projection_overrides=Dict(
                :q_proj => int8,
                :k_proj => int8,
                :v_proj => int8,
                :o_proj => int8,
                :down_proj => int8,
                :lm_head => bf16,
            ),
        )
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        in_memory = quantize_bf16_parameters(
            loaded.parameters;
            plan,
            activation_calibration=calibration,
        )
        streamed = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan,
            activation_calibration=calibration,
        )
        @test quantized_parameters_equal(in_memory, streamed.parameters)
        @test in_memory.blocks.layer_1.mlp.gate_proj.weight isa
            Int4GroupWeight
        @test in_memory.blocks.layer_1.mlp.up_proj.weight isa
            Int4GroupWeight
        @test in_memory.blocks.layer_1.attn.q_proj.weight isa
            Int8ChannelWeight
        @test in_memory.blocks.layer_1.mlp.down_proj.weight isa
            Int8ChannelWeight
        @test in_memory.lm_head.weight isa Matrix{BFloat16}

        @test_throws ArgumentError quantize_bf16_parameters(
            loaded.parameters;
            plan,
        )
        wrong_depth = ActivationCalibration(
            Dict((1, :gate_proj) => ones(Float32, model.d_model));
            token_count=1,
            num_layers=1,
        )
        @test_throws ArgumentError quantize_bf16_parameters(
            loaded.parameters;
            plan,
            activation_calibration=wrong_depth,
        )
        missing = copy(calibration.layer_moments)
        delete!(missing, (1, :gate_proj))
        missing_calibration = ActivationCalibration(
            missing;
            lm_head_moment=calibration.lm_head_moment,
            token_count=calibration.token_count,
            num_layers=calibration.num_layers,
        )
        @test_throws ArgumentError quantize_bf16_parameters(
            loaded.parameters;
            plan,
            activation_calibration=missing_calibration,
        )
        bad_dimension = copy(calibration.layer_moments)
        bad_dimension[(1, :gate_proj)] = ones(Float32, model.d_model - 1)
        dimension_calibration = ActivationCalibration(
            bad_dimension;
            lm_head_moment=calibration.lm_head_moment,
            token_count=calibration.token_count,
            num_layers=calibration.num_layers,
        )
        @test_throws DimensionMismatch quantize_bf16_parameters(
            loaded.parameters;
            plan,
            activation_calibration=dimension_calibration,
        )
        @test_throws ArgumentError quantize_bf16_parameters(
            loaded.parameters;
            plan=QuantizationPlan(
                default=LinearQuantizationSpec(:int4; group=4),
            ),
            activation_calibration=calibration,
        )
    end

    mktempdir() do directory
        model = _qwen3_tiny_model_fixture_dir(directory; tie=true)
        tokens = reshape(Int[1, 3, 5, 7], :, 1)
        result = calibrate_hf_qwen3_activations(
            directory,
            tokens;
            max_seq_len=16,
        )
        @test result.calibration.lm_head_moment === nothing
        plan = QuantizationPlan(
            default=LinearQuantizationSpec(
                :int4;
                group=4,
                calibration=:activation_mse,
            ),
        )
        quantized = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan,
            activation_calibration=result.calibration,
        )
        @test isempty(quantized.parameters.lm_head)
        @test quantized.model.tie_embeddings
        @test quantized.model.num_layers == model.num_layers
    end
end
