using BFloat16s: BFloat16
using Test
using LifeAI

@testset "Qwen3-VL QK RoPE scratch" begin
    prototype = zeros(BFloat16, 1)
    for (head_dim, num_heads, num_kv_heads) in ((8, 4, 2), (128, 16, 8))
        scratch = LifeAI.init_qwen3_vl_qk_rope_scratch(
            prototype;
            head_dim,
            num_heads,
            num_kv_heads,
        )
        @test LifeAI.qwen3_vl_qk_rope_scratch_bytes(scratch) ==
            LifeAI.qwen3_vl_qk_rope_scratch_bytes(;
                head_dim,
                num_heads,
                num_kv_heads,
            )
        half = head_dim ÷ 2
        queries = randn(BFloat16, head_dim, num_heads, 1, 1)
        keys = randn(BFloat16, head_dim, num_kv_heads, 1, 1)
        cos_values = randn(BFloat16, half, 1, 1)
        sin_values = randn(BFloat16, half, 1, 1)
        got_q, got_k = LifeAI._qwen3_vl_decode_qk_rope!(
            scratch,
            queries,
            keys,
            cos_values,
            sin_values,
        )
        @test got_q === scratch.query_out
        @test got_k === scratch.key_out
        @test got_q == LifeAI._qwen3_vl_text_apply_rope(
            queries,
            cos_values,
            sin_values,
        )
        @test got_k == LifeAI._qwen3_vl_text_apply_rope(
            keys,
            cos_values,
            sin_values,
        )
        @test !Base.mightalias(got_q, queries)
        @test !Base.mightalias(got_k, keys)
        @test !Base.mightalias(got_q, got_k)
    end
    @test_throws ArgumentError LifeAI.qwen3_vl_qk_rope_scratch_bytes(;
        head_dim=0,
        num_heads=4,
        num_kv_heads=2,
    )
    @test_throws ArgumentError LifeAI.qwen3_vl_qk_rope_scratch_bytes(;
        head_dim=7,
        num_heads=4,
        num_kv_heads=2,
    )
    @test_throws ArgumentError LifeAI.qwen3_vl_qk_rope_scratch_bytes(;
        head_dim=8,
        num_heads=2,
        num_kv_heads=4,
    )
    scratch = LifeAI.init_qwen3_vl_qk_rope_scratch(
        prototype;
        head_dim=8,
        num_heads=4,
        num_kv_heads=2,
    )
    cos_values = randn(BFloat16, 4, 1, 1)
    sin_values = randn(BFloat16, 4, 1, 1)
    @test_throws DimensionMismatch LifeAI._qwen3_vl_decode_qk_rope!(
        scratch,
        randn(BFloat16, 8, 3, 1, 1),
        randn(BFloat16, 8, 2, 1, 1),
        cos_values,
        sin_values,
    )
    @test_throws DimensionMismatch LifeAI._qwen3_vl_decode_qk_rope!(
        scratch,
        randn(BFloat16, 8, 4, 2, 1),
        randn(BFloat16, 8, 2, 1, 1),
        cos_values,
        sin_values,
    )
end

@testset "Qwen3-VL static decode QK RoPE scratch ownership" begin
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
    @test float_cache.qk_rope_scratch === nothing

    parameters = to_bf16(float_parameters)
    features = to_bf16(inputs.vision_features)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    other = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    @test cache.qk_rope_scratch === nothing
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    @test cache.qk_rope_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.qk_rope_scratch
    @test scratch isa LifeAI.Qwen3VLQKRopeScratch
    @test scratch.head_dim == parameters.spec.head_dim
    @test scratch.num_heads == parameters.spec.num_attention_heads
    @test scratch.num_kv_heads == parameters.spec.num_key_value_heads
    query_out = scratch.query_out
    key_out = scratch.key_out
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.qk_rope_scratch === scratch
    @test scratch.query_out === query_out
    @test scratch.key_out === key_out
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.qk_rope_scratch === scratch
    @test scratch.query_out === query_out
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache=other,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(parameters, 2, other)
    @test other.qk_rope_scratch !== scratch
    @test other.qk_rope_scratch.query_out !== query_out
end
