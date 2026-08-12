using Test
using Lux
using Random: Xoshiro
using LifeAI:
    GPTModel,
    decode_step,
    gpt_config,
    init_kv_cache,
    init_static_kv_cache,
    prefill

@testset "Qwen3 MoE full forward and cached decode equivalence" begin
    model = GPTModel(
        17,
        8,
        2,
        2;
        num_kv_heads=1,
        head_dim=4,
        mlp_hidden_dim=5,
        use_bias=false,
        use_rope=true,
        use_qk_norm=true,
        qk_norm_epsilon=1.0f-6,
        max_seq_len=8,
        rope_theta=1.0f6,
        rope_style=:rotate_half,
        norm_epsilon=1.0f-6,
        norm_type=:rmsnorm,
        mlp_type=:qwen3_moe,
        tie_embeddings=false,
        num_experts=4,
        experts_per_token=2,
        normalize_routing=true,
    )
    parameters, states = Lux.setup(Xoshiro(20260807), model)
    rebuilt = GPTModel(gpt_config(model))
    @test gpt_config(rebuilt) == gpt_config(model)
    @test Lux.parameterlength(parameters) == Lux.parameterlength(model)

    tokens = reshape([1, 4, 7, 10], :, 1)
    full_logits, _ = model(tokens, parameters, states)
    prompt = tokens[1:3, :]
    next_token = tokens[4, :]

    dynamic = init_kv_cache(model; batch_size=1)
    prompt_logits, dynamic, dynamic_states = prefill(
        model,
        parameters,
        states,
        prompt,
        dynamic,
    )
    decode_logits, dynamic, _ = decode_step(
        model,
        parameters,
        dynamic_states,
        next_token,
        dynamic,
    )
    @test prompt_logits ≈ full_logits[:, 1:3, :] atol = 2.0f-5 rtol = 2.0f-5
    @test decode_logits ≈ full_logits[:, 4:4, :] atol = 2.0f-5 rtol = 2.0f-5

    static = init_static_kv_cache(model; batch_size=1)
    static_prompt, static, static_states = prefill(
        model,
        parameters,
        states,
        prompt,
        static,
    )
    static_decode, static, _ = decode_step(
        model,
        parameters,
        static_states,
        next_token,
        static,
    )
    @test static_prompt ≈ full_logits[:, 1:3, :] atol = 2.0f-5 rtol = 2.0f-5
    @test static_decode ≈ full_logits[:, 4:4, :] atol = 2.0f-5 rtol = 2.0f-5
end
