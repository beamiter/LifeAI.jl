using BFloat16s: BFloat16
using Test
using LifeAI

@testset "Qwen3-VL QKV scratch" begin
    prototype = zeros(BFloat16, 1)
    for (hidden_size, head_dim, num_heads, num_kv_heads) in (
        (16, 8, 2, 1),
        (8, 4, 2, 2),
    )
        scratch = LifeAI.init_qwen3_vl_qkv_scratch(
            prototype;
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
        )
        @test LifeAI.qwen3_vl_qkv_scratch_bytes(scratch) ==
            LifeAI.qwen3_vl_qkv_scratch_bytes(;
                head_dim,
                num_heads,
                num_kv_heads,
            )
        x = randn(BFloat16, hidden_size, 1, 1)
        q_weight = randn(BFloat16, head_dim * num_heads, hidden_size)
        k_weight = randn(BFloat16, head_dim * num_kv_heads, hidden_size)
        v_weight = randn(BFloat16, head_dim * num_kv_heads, hidden_size)
        got_q, got_k, got_v = LifeAI._qwen3_vl_decode_qkv!(
            scratch,
            q_weight,
            k_weight,
            v_weight,
            x,
        )
        @test size(got_q) == (head_dim, num_heads, 1, 1)
        @test size(got_k) == (head_dim, num_kv_heads, 1, 1)
        @test got_q == reshape(
            LifeAI._bf16a_linear(q_weight, x),
            head_dim, num_heads, 1, 1,
        )
        @test got_k == reshape(
            LifeAI._bf16a_linear(k_weight, x),
            head_dim, num_kv_heads, 1, 1,
        )
        @test got_v == reshape(
            LifeAI._bf16a_linear(v_weight, x),
            head_dim, num_kv_heads, 1, 1,
        )
    end
    @test_throws ArgumentError LifeAI.qwen3_vl_qkv_scratch_bytes(;
        head_dim=0,
        num_heads=2,
        num_kv_heads=1,
    )
end

@testset "Qwen3-VL static decode QKV scratch ownership" begin
    isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || return
    to_bf16(value) = value isa AbstractArray{<:AbstractFloat} ? BFloat16.(value) :
        value isa NamedTuple ? NamedTuple{keys(value)}(map(to_bf16, values(value))) :
        value isa Tuple ? map(to_bf16, value) : value
    inputs = _ch46_tiny_prefill_inputs()
    parameters = to_bf16(_ch46_tiny_text_parameters())
    features = to_bf16(inputs.vision_features)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    @test cache.qkv_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.qkv_scratch
    @test scratch isa LifeAI.Qwen3VLQKVScratch
    query = scratch.query
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.qkv_scratch === scratch
    @test scratch.query === query
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.qkv_scratch === scratch
end
