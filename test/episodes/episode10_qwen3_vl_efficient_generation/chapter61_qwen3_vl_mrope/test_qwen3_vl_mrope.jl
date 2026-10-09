using BFloat16s: BFloat16
using Test
using LifeAI

function _ch61_text_spec(; head_dim::Int, mrope_section)
    return LifeAI.Qwen3VLTextSpec(
        32,
        16,
        32,
        1,
        2,
        1,
        head_dim,
        1.0e-6,
        5_000_000.0,
        4096,
        true,
        mrope_section,
        true,
        "silu",
    )
end

@testset "Qwen3-VL mRoPE scratch" begin
    prototype = zeros(BFloat16, 1)
    scratch = LifeAI.init_qwen3_vl_mrope_scratch(prototype; head_dim=128)
    @test LifeAI.qwen3_vl_mrope_scratch_bytes(scratch) ==
        LifeAI.qwen3_vl_mrope_scratch_bytes(; head_dim=128) == 256
    spec = _ch61_text_spec(; head_dim=128, mrope_section=(24, 20, 20))
    for coordinate in (0, 1, 97, 331)
        position_ids = fill(coordinate, 3, 1, 1)
        expected_cos, expected_sin = LifeAI._qwen3_vl_text_mrope(
            spec,
            prototype,
            position_ids,
        )
        got_cos, got_sin = LifeAI._qwen3_vl_decode_mrope!(
            scratch,
            spec,
            coordinate,
        )
        @test got_cos === scratch.cos_values
        @test got_sin === scratch.sin_values
        @test got_cos == expected_cos
        @test got_sin == expected_sin
    end
    @test_throws ArgumentError LifeAI.qwen3_vl_mrope_scratch_bytes(; head_dim=0)
    @test_throws ArgumentError LifeAI.qwen3_vl_mrope_scratch_bytes(; head_dim=7)
end

@testset "Qwen3-VL static decode mRoPE scratch ownership" begin
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
    @test float_cache.mrope_scratch === nothing

    parameters = to_bf16(float_parameters)
    features = to_bf16(inputs.vision_features)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    other = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    @test cache.mrope_scratch === nothing
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    @test cache.mrope_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.mrope_scratch
    @test scratch isa LifeAI.Qwen3VLMropeScratch
    @test scratch.half == parameters.spec.head_dim ÷ 2
    cos_values = scratch.cos_values
    sin_values = scratch.sin_values
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.mrope_scratch === scratch
    @test scratch.cos_values === cos_values
    @test scratch.sin_values === sin_values
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.mrope_scratch === scratch
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache=other,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(parameters, 2, other)
    @test other.mrope_scratch !== scratch
    @test other.mrope_scratch.cos_values !== cos_values
    @test other.mrope_scratch.sin_values !== sin_values
end
