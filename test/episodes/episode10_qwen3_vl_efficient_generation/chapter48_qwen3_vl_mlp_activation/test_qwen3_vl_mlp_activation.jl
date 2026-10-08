using BFloat16s: BFloat16
using Test
using LifeAI

@testset "Qwen3-VL MLP activation scratch" begin
    for intermediate_size in (8, 32, 6144)
        prototype = zeros(BFloat16, 1)
        scratch = LifeAI.init_qwen3_vl_mlp_activation_scratch(
            prototype;
            intermediate_size,
        )
        @test LifeAI.qwen3_vl_mlp_activation_scratch_bytes(scratch) ==
            LifeAI.qwen3_vl_mlp_activation_scratch_bytes(; intermediate_size)
        gate = randn(BFloat16, intermediate_size, 1, 1)
        up = randn(BFloat16, intermediate_size, 1, 1)
        reused = LifeAI._qwen3_vl_mlp_activation!(scratch, gate, up)
        reference = LifeAI._qwen3_vl_reference_mlp_activation(gate, up)
        @test reused == reference
        @test !Base.mightalias(scratch.product, gate)
        @test !Base.mightalias(scratch.product, up)
    end
    @test_throws ArgumentError LifeAI.qwen3_vl_mlp_activation_scratch_bytes(;
        intermediate_size=0,
    )
    prototype = zeros(BFloat16, 1)
    scratch = LifeAI.init_qwen3_vl_mlp_activation_scratch(
        prototype;
        intermediate_size=4,
    )
    @test_throws DimensionMismatch LifeAI._qwen3_vl_mlp_activation!(
        scratch,
        randn(BFloat16, 5, 1, 1),
        randn(BFloat16, 5, 1, 1),
    )
end

@testset "Qwen3-VL static decode MLP scratch ownership" begin
    isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || return
    to_bf16(value) = value isa AbstractArray{<:AbstractFloat} ? BFloat16.(value) :
        value isa NamedTuple ? NamedTuple{keys(value)}(map(to_bf16, values(value))) :
        value isa Tuple ? map(to_bf16, value) : value
    inputs = _ch46_tiny_prefill_inputs()
    parameters = to_bf16(_ch46_tiny_text_parameters())
    features = to_bf16(inputs.vision_features)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    other = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    @test cache.mlp_activation_scratch === nothing
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    @test cache.mlp_activation_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.mlp_activation_scratch
    @test scratch isa LifeAI.Qwen3VLMlpActivationScratch
    @test scratch.intermediate_size == parameters.spec.intermediate_size
    product = scratch.product
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.mlp_activation_scratch === scratch
    @test scratch.product === product
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.mlp_activation_scratch === scratch
    @test scratch.product === product
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache=other,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(parameters, 2, other)
    @test other.mlp_activation_scratch !== scratch
    @test other.mlp_activation_scratch.product !== product
end
