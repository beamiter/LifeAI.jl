using BFloat16s: BFloat16
using Test
using LifeAI

function _ch47_attention_case(head_dim, num_heads, num_kv_heads, seq, capacity)
    prototype = zeros(BFloat16, 1)
    scratch = LifeAI.init_qwen3_vl_decode_attention_scratch(
        prototype;
        head_dim,
        num_attention_heads=num_heads,
        num_kv_heads,
        capacity,
    )
    @test LifeAI.qwen3_vl_decode_attention_scratch_bytes(scratch) ==
        LifeAI.qwen3_vl_decode_attention_scratch_bytes(;
            head_dim,
            num_attention_heads=num_heads,
            num_kv_heads,
            capacity,
        )
    queries = randn(BFloat16, head_dim, num_heads, 1, 1)
    keys = randn(BFloat16, head_dim, num_kv_heads, capacity, 1)
    values = randn(BFloat16, head_dim, num_kv_heads, capacity, 1)
    key_prefix = view(keys, :, :, 1:seq, :)
    value_prefix = view(values, :, :, 1:seq, :)
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    reused = LifeAI._qwen3_vl_decode_attention!(
        scratch,
        queries,
        key_prefix,
        value_prefix;
        scaling,
    )
    reference = LifeAI._bf16a_attention(
        queries,
        key_prefix,
        value_prefix;
        scaling,
        mask=nothing,
    )
    @test reused == reference
    @test !any(
        Base.mightalias(scratch.keys, array) for array in (keys, values, queries)
    )
    return scratch
end

@testset "Qwen3-VL decode attention scratch" begin
    first = _ch47_attention_case(8, 2, 1, 3, 5)
    second = _ch47_attention_case(8, 2, 1, 5, 5)
    _ch47_attention_case(128, 16, 8, 4, 9)
    @test first.keys !== second.keys
    @test_throws ArgumentError LifeAI.qwen3_vl_decode_attention_scratch_bytes(;
        head_dim=8,
        num_attention_heads=2,
        num_kv_heads=1,
        capacity=0,
    )
    @test_throws ArgumentError LifeAI.qwen3_vl_decode_attention_scratch_bytes(;
        head_dim=8,
        num_attention_heads=3,
        num_kv_heads=2,
        capacity=4,
    )

    prototype = zeros(BFloat16, 1)
    scratch = LifeAI.init_qwen3_vl_decode_attention_scratch(
        prototype;
        head_dim=8,
        num_attention_heads=2,
        num_kv_heads=1,
        capacity=2,
    )
    queries = randn(BFloat16, 8, 2, 1, 1)
    keys = randn(BFloat16, 8, 1, 4, 1)
    values = randn(BFloat16, 8, 1, 4, 1)
    @test_throws ArgumentError LifeAI._qwen3_vl_decode_attention!(
        scratch,
        queries,
        view(keys, :, :, 1:4, :),
        view(values, :, :, 1:4, :);
        scaling=0.1f0,
    )
end

function _ch47_bf16_parameters(parameters)
    convert_value(value) = value isa AbstractArray{<:AbstractFloat} ?
        BFloat16.(value) : value
    convert_tree(value) = value isa NamedTuple ?
        NamedTuple{keys(value)}(map(convert_tree, values(value))) :
        value isa Tuple ? map(convert_tree, value) : convert_value(value)
    return convert_tree(parameters)
end

@testset "Qwen3-VL static decode scratch ownership" begin
    isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || return
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
    hf_qwen3_vl_text_decode_step_static(
        float_parameters,
        2,
        float_cache,
    )
    @test float_cache.attention_scratch === nothing

    parameters = _ch47_bf16_parameters(float_parameters)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    other = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    @test cache.attention_scratch === nothing
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=_ch47_bf16_parameters(inputs.vision_features),
        cache=cache,
        logits_to_keep=1,
    )
    @test cache.attention_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.attention_scratch
    @test scratch isa LifeAI.Qwen3VLDecodeAttentionScratch
    @test scratch.capacity == 12
    @test !any(
        Base.mightalias(scratch.keys, layer.keys) for layer in cache.layers
    )
    keys_before = scratch.keys
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.attention_scratch === scratch
    @test scratch.keys === keys_before
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.position == 0
    @test cache.attention_scratch === scratch
    @test scratch.keys === keys_before
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=_ch47_bf16_parameters(inputs.vision_features),
        cache=other,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(parameters, 2, other)
    @test other.attention_scratch !== scratch
    @test other.attention_scratch.keys !== scratch.keys
end
