using BFloat16s: BFloat16
using Test
using LifeAI

@testset "Qwen3-VL gate/up scratch" begin
    prototype = zeros(BFloat16, 1)
    for (hidden_size, intermediate_size) in ((16, 32), (8, 12))
        scratch = LifeAI.init_qwen3_vl_gate_up_scratch(
            prototype;
            hidden_size,
            intermediate_size,
        )
        @test LifeAI.qwen3_vl_gate_up_scratch_bytes(scratch) ==
            LifeAI.qwen3_vl_gate_up_scratch_bytes(; intermediate_size)
        x = randn(BFloat16, hidden_size, 1, 1)
        gate_weight = randn(BFloat16, intermediate_size, hidden_size)
        up_weight = randn(BFloat16, intermediate_size, hidden_size)
        got_gate, got_up = LifeAI._qwen3_vl_decode_gate_up!(
            scratch,
            gate_weight,
            up_weight,
            x,
        )
        @test got_gate === scratch.gate
        @test got_up === scratch.up
        @test got_gate == LifeAI._bf16a_linear(gate_weight, x)
        @test got_up == LifeAI._bf16a_linear(up_weight, x)
        @test !Base.mightalias(got_gate, x)
        @test !Base.mightalias(got_gate, got_up)
    end
    @test_throws ArgumentError LifeAI.qwen3_vl_gate_up_scratch_bytes(;
        intermediate_size=0,
    )
    scratch = LifeAI.init_qwen3_vl_gate_up_scratch(
        prototype;
        hidden_size=16,
        intermediate_size=32,
    )
    @test_throws DimensionMismatch LifeAI._qwen3_vl_decode_gate_up!(
        scratch,
        randn(BFloat16, 32, 16),
        randn(BFloat16, 32, 16),
        randn(BFloat16, 16, 2, 1),
    )
end

@testset "Qwen3-VL static decode gate/up scratch ownership" begin
    isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || return
    to_bf16(value) = value isa AbstractArray{<:AbstractFloat} ? BFloat16.(value) :
        value isa NamedTuple ? NamedTuple{keys(value)}(map(to_bf16, values(value))) :
        value isa Tuple ? map(to_bf16, value) : value
    inputs = _ch46_tiny_prefill_inputs()
    float_parameters = _ch46_tiny_text_parameters()
    float_cache = init_qwen3_vl_static_kv_cache(float_parameters; capacity=12)
    hf_qwen3_vl_text_prefill_static(
        float_parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=float_cache,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(float_parameters, 2, float_cache)
    @test float_cache.gate_up_scratch === nothing

    parameters = to_bf16(float_parameters)
    features = to_bf16(inputs.vision_features)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    other = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    @test cache.gate_up_scratch === nothing
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    @test cache.gate_up_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.gate_up_scratch
    @test scratch isa LifeAI.Qwen3VLGateUpScratch
    @test scratch.hidden_size == parameters.spec.hidden_size
    @test scratch.intermediate_size == parameters.spec.intermediate_size
    gate = scratch.gate
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.gate_up_scratch === scratch
    @test scratch.gate === gate
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.gate_up_scratch === scratch
    @test scratch.gate === gate
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache=other,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(parameters, 2, other)
    @test other.gate_up_scratch !== scratch
    @test other.gate_up_scratch.gate !== gate
end
