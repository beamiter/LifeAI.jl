using BFloat16s: BFloat16
using Test
using LifeAI

@testset "Qwen3-VL hidden norm scratch" begin
    prototype = zeros(BFloat16, 1)
    epsilon = 1.0f-6
    for hidden_size in (16, 2048)
        scratch = LifeAI.init_qwen3_vl_hidden_norm_scratch(
            prototype;
            hidden_size,
        )
        @test LifeAI.qwen3_vl_hidden_norm_scratch_bytes(scratch) ==
            LifeAI.qwen3_vl_hidden_norm_scratch_bytes(; hidden_size)
        x = randn(BFloat16, hidden_size, 1, 1)
        scale = randn(BFloat16, hidden_size)
        got = LifeAI._qwen3_vl_decode_hidden_norm!(scratch, x, scale, epsilon)
        @test got === scratch.output
        @test got == LifeAI._bf16a_rmsnorm(x, scale, epsilon)
        @test !Base.mightalias(got, x)
        got_again = LifeAI._qwen3_vl_decode_hidden_norm!(
            scratch,
            x,
            scale,
            epsilon,
        )
        @test got_again === got
    end
    @test_throws ArgumentError LifeAI.qwen3_vl_hidden_norm_scratch_bytes(;
        hidden_size=0,
    )
    scratch = LifeAI.init_qwen3_vl_hidden_norm_scratch(
        prototype;
        hidden_size=16,
    )
    @test_throws DimensionMismatch LifeAI._qwen3_vl_decode_hidden_norm!(
        scratch,
        randn(BFloat16, 16, 2, 1),
        randn(BFloat16, 16),
        epsilon,
    )
    @test_throws DimensionMismatch LifeAI._qwen3_vl_decode_hidden_norm!(
        scratch,
        randn(BFloat16, 16, 1, 1),
        randn(BFloat16, 8),
        epsilon,
    )
end

@testset "Qwen3-VL static decode hidden norm scratch ownership" begin
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
    @test float_cache.hidden_norm_scratch === nothing

    parameters = to_bf16(float_parameters)
    features = to_bf16(inputs.vision_features)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    other = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    @test cache.hidden_norm_scratch === nothing
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    @test cache.hidden_norm_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.hidden_norm_scratch
    @test scratch isa LifeAI.Qwen3VLHiddenNormScratch
    @test scratch.hidden_size == parameters.spec.hidden_size
    output = scratch.output
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.hidden_norm_scratch === scratch
    @test scratch.output === output
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.hidden_norm_scratch === scratch
    @test scratch.output === output
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache=other,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(parameters, 2, other)
    @test other.hidden_norm_scratch !== scratch
    @test other.hidden_norm_scratch.output !== output
end
