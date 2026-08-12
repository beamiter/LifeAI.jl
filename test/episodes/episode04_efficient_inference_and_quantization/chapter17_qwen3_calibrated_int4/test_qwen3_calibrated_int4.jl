using Test
using BFloat16s: BFloat16
using JSON3
using SHA: sha256
using LifeAI:
    Int4GroupWeight,
    Int8ChannelWeight,
    LinearQuantizationSpec,
    QuantizationPlan,
    estimate_qwen3_quantized_bytes,
    load_hf_qwen3_model,
    load_hf_qwen3_quantized,
    quantization_spec,
    quantize_bf16_parameters,
    quantized_parameter_bytes,
    qwen3_dense_spec

isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tiny_model_fixture.jl"))
isdefined(@__MODULE__, :quantized_parameters_equal) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_quantization_test_utils.jl"))

const _QWEN3_CALIBRATED_QUANTIZATION_ASSETS_PATH = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_calibrated_int4",
    "assets.json",
)

@testset "INT4 reconstruction-MSE calibration" begin
    # Each group has one outlier and many unit-scale values. Candidate 0.9
    # clips the outlier slightly but lowers total reconstruction error.
    row = Float32[10; fill(1, 15)]
    weight = BFloat16.(vcat(
        reshape(row, 1, :),
        reshape(-row, 1, :),
        reshape(reverse(row), 1, :),
    ))
    maxabs = LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:maxabs,
    )
    calibrated = LifeAI._quantize_int4_group(
        weight;
        group=16,
        calibration=:mse,
    )
    maxabs_reconstructed = Float32.(LifeAI._dequantize_bf16(maxabs))
    calibrated_reconstructed = Float32.(LifeAI._dequantize_bf16(calibrated))
    source = Float32.(weight)
    maxabs_error = sum(abs2, maxabs_reconstructed .- source)
    calibrated_error = sum(abs2, calibrated_reconstructed .- source)
    @test calibrated_error < maxabs_error
    @test any(calibrated.scale .< maxabs.scale)

    # Including 1.0 in the candidates makes MSE calibration no worse than the
    # max-abs baseline independently for every row/group.
    matrix = BFloat16.(reshape(
        Float32[sin(i / 7) * (iszero(i % 29) ? 8 : 1) for i in 1:(4 * 64)],
        4,
        64,
    ))
    baseline = LifeAI._quantize_int4_group(matrix; group=16)
    mse = LifeAI._quantize_int4_group(
        matrix;
        group=16,
        calibration=:mse,
    )
    baseline_dequant = Float32.(LifeAI._dequantize_bf16(baseline))
    mse_dequant = Float32.(LifeAI._dequantize_bf16(mse))
    for row_index in axes(matrix, 1), group_index in 1:4
        columns = ((group_index - 1) * 16 + 1):(group_index * 16)
        source_group = Float32.(matrix[row_index, columns])
        @test sum(abs2, mse_dequant[row_index, columns] .- source_group) <=
            sum(abs2, baseline_dequant[row_index, columns] .- source_group)
    end

    # Qwen3 XLA decode and quantization's default call remains exactly max-abs RTN.
    legacy = LifeAI._quantize_int4_group(matrix; group=16)
    explicit = LifeAI._quantize_int4_group(
        matrix;
        group=16,
        calibration=:maxabs,
    )
    @test legacy.packed == explicit.packed
    @test legacy.scale == explicit.scale
end

@testset "quantization plan validation and precedence" begin
    int4_mse = LinearQuantizationSpec(:int4; group=128, calibration=:mse)
    int8 = LinearQuantizationSpec(:int8)
    bf16 = LinearQuantizationSpec(:bf16)
    plan = QuantizationPlan(
        default=int4_mse,
        projection_overrides=Dict(:q_proj => int8, :lm_head => bf16),
        layer_overrides=Dict((2, :q_proj) => bf16),
    )
    @test quantization_spec(plan, :k_proj; layer=2) === int4_mse
    @test quantization_spec(plan, :q_proj; layer=1) === int8
    @test quantization_spec(plan, :q_proj; layer=2) === bf16
    @test quantization_spec(plan, :lm_head) === bf16

    @test_throws ArgumentError LinearQuantizationSpec(:int3)
    @test_throws ArgumentError LinearQuantizationSpec(:int4; group=3)
    @test_throws ArgumentError LinearQuantizationSpec(:int8; calibration=:mse)
    @test_throws ArgumentError LinearQuantizationSpec(
        :int4;
        calibration=:mse,
        clip_ratios=(0.9f0, 0.8f0),
    )
    @test_throws ArgumentError LinearQuantizationSpec(
        :int4;
        calibration=:mse,
        clip_ratios=(1.0f0, 0.0f0),
    )
    @test_throws ArgumentError QuantizationPlan(
        projection_overrides=Dict(:unknown => int8),
    )
    @test_throws ArgumentError QuantizationPlan(
        layer_overrides=Dict((0, :q_proj) => int8),
    )
    @test_throws ArgumentError QuantizationPlan(
        layer_overrides=Dict((1, :lm_head) => int8),
    )
    @test_throws ArgumentError quantization_spec(plan, :q_proj; layer=0)
    @test_throws ArgumentError quantization_spec(plan, :lm_head; layer=1)
end

@testset "one plan drives in-memory and streamed quantization" begin
    mktempdir() do directory
        model = _qwen3_tiny_model_fixture_dir(directory; tie=false)
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        int4_mse = LinearQuantizationSpec(
            :int4;
            group=4,
            calibration=:mse,
            clip_ratios=(1.0f0, 0.9f0, 0.8f0),
        )
        int8 = LinearQuantizationSpec(:int8; group=4)
        bf16 = LinearQuantizationSpec(:bf16; group=4)
        plan = QuantizationPlan(
            default=int4_mse,
            projection_overrides=Dict(
                :q_proj => int8,
                :down_proj => bf16,
                :lm_head => int8,
            ),
            layer_overrides=Dict(
                (1, :q_proj) => bf16,
                (2, :down_proj) => int8,
            ),
        )

        in_memory = quantize_bf16_parameters(loaded.parameters; plan)
        streamed = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan,
        )
        @test quantized_parameters_equal(in_memory, streamed.parameters)
        @test in_memory.blocks.layer_1.attn.q_proj.weight isa Matrix{BFloat16}
        @test in_memory.blocks.layer_2.attn.q_proj.weight isa Int8ChannelWeight
        @test in_memory.blocks.layer_1.mlp.down_proj.weight isa Matrix{BFloat16}
        @test in_memory.blocks.layer_2.mlp.down_proj.weight isa Int8ChannelWeight
        @test in_memory.blocks.layer_1.mlp.gate_proj.weight isa Int4GroupWeight
        @test in_memory.lm_head.weight isa Int8ChannelWeight

        actual_bytes = quantized_parameter_bytes(in_memory)
        @test actual_bytes == estimate_qwen3_quantized_bytes(model, plan)
        out_of_range = QuantizationPlan(
            layer_overrides=Dict((model.num_layers + 1, :q_proj) => int8),
        )
        @test_throws ArgumentError estimate_qwen3_quantized_bytes(
            model,
            out_of_range,
        )
        @test_throws ArgumentError quantize_bf16_parameters(
            loaded.parameters;
            plan=out_of_range,
        )

        # Legacy Qwen3 XLA decode and quantization arguments resolve to the same plan and tensor values.
        legacy = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            scheme=:int4,
            group=4,
            int8_projections=(:q_proj, :lm_head),
        )
        equivalent_plan = QuantizationPlan(
            default=LinearQuantizationSpec(:int4; group=4),
            projection_overrides=Dict(
                :q_proj => LinearQuantizationSpec(:int8; group=4),
                :lm_head => LinearQuantizationSpec(:int8; group=4),
            ),
        )
        planned = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan=equivalent_plan,
        )
        @test quantized_parameters_equal(legacy.parameters, planned.parameters)
    end

    mktempdir() do directory
        model = _qwen3_tiny_model_fixture_dir(directory; tie=true)
        loaded = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        plan = QuantizationPlan(
            default=LinearQuantizationSpec(:int4; group=4),
            projection_overrides=Dict(
                :lm_head => LinearQuantizationSpec(:bf16; group=4),
            ),
        )
        in_memory = quantize_bf16_parameters(loaded.parameters; plan)
        streamed = load_hf_qwen3_quantized(
            directory;
            max_seq_len=16,
            plan,
        )
        @test isempty(in_memory.lm_head)
        @test quantized_parameters_equal(in_memory, streamed.parameters)
        @test quantized_parameter_bytes(in_memory) ==
            estimate_qwen3_quantized_bytes(model, plan)
    end
end

@testset "Qwen3-14B frozen tensor-byte budgets" begin
    fixture = JSON3.read(read(_QWEN3_CALIBRATED_QUANTIZATION_ASSETS_PATH, String))
    for (filename, expected) in pairs(fixture["plan_sha256"])
        path = joinpath(dirname(_QWEN3_CALIBRATED_QUANTIZATION_ASSETS_PATH), String(filename))
        @test bytes2hex(sha256(read(path))) == String(expected)
    end
    qwen = fixture["qwen3_14b"]
    spec = qwen3_dense_spec(:qwen3_14b)
    pure_int4 = QuantizationPlan(
        default=LinearQuantizationSpec(
            :int4;
            group=128,
            calibration=:mse,
        ),
    )
    pure_int8 = QuantizationPlan(
        default=LinearQuantizationSpec(:int8),
    )
    mixed_head = QuantizationPlan(
        default=pure_int4.default,
        projection_overrides=Dict(
            :lm_head => LinearQuantizationSpec(:int8),
        ),
    )
    mixed_24g_overrides = Dict(
        projection => LinearQuantizationSpec(:int8)
        for projection in (:q_proj, :k_proj, :v_proj, :o_proj, :down_proj)
    )
    mixed_24g_overrides[:lm_head] = LinearQuantizationSpec(:bf16)
    mixed_24g = QuantizationPlan(
        default=pure_int4.default,
        projection_overrides=mixed_24g_overrides,
    )
    @test estimate_qwen3_quantized_bytes(spec, pure_int8) ==
        Int(qwen["int8_tensor_bytes"])
    @test estimate_qwen3_quantized_bytes(spec, pure_int4) ==
        Int(qwen["pure_int4_g128_tensor_bytes"])
    @test estimate_qwen3_quantized_bytes(spec, mixed_head) ==
        Int(qwen["int4_g128_int8_lm_head_tensor_bytes"])
    @test estimate_qwen3_quantized_bytes(spec, mixed_head) >
        estimate_qwen3_quantized_bytes(spec, pure_int4)
    @test estimate_qwen3_quantized_bytes(spec, mixed_24g) ==
        Int(qwen["mixed_24g_tensor_bytes"])

    hardware = fixture["hardware_validation"]
    @test String(hardware["status"]) == "complete"
    @test Bool(hardware["model_matches_frozen_huggingface_revision"])
    @test String(hardware["gpu"]) == "NVIDIA GeForce RTX 4090 D"
    runs = hardware["runs"]
    int8_run = runs["int8"]
    mse_run = runs["mixed_mse"]
    rtn_run = runs["mixed_rtn"]
    for run in (int8_run, mse_run, rtn_run)
        @test Int(run["estimated_tree_bytes"]) == Int(run["host_tree_bytes"])
        @test Int(run["gpu_tree_bytes"]) >= Int(run["host_tree_bytes"])
        @test Int(run["vram_used_bytes"]) <= Int(hardware["gpu_total_bytes"])
        @test Bool(run["logits_argmax_equal"])
        @test Bool(run["decode_argmax_equal"])
    end
    @test Int(int8_run["estimated_tree_bytes"]) ==
        Int(qwen["int8_tensor_bytes"])
    @test Int(mse_run["estimated_tree_bytes"]) ==
        Int(qwen["mixed_24g_tensor_bytes"])
    @test Int(rtn_run["estimated_tree_bytes"]) ==
        Int(qwen["mixed_24g_tensor_bytes"])
    @test Int(int8_run["greedy_agreement"]) == 16
    @test Int(rtn_run["greedy_agreement"]) == 16
    @test Int(mse_run["greedy_agreement"]) == 4
    @test Int(mse_run["greedy_first_divergence"]) == 5
    @test Float64(mse_run["logits_max_abs"]) <
        Float64(rtn_run["logits_max_abs"])
    @test Float64(mse_run["logits_mean_abs"]) <
        Float64(rtn_run["logits_mean_abs"])
end
