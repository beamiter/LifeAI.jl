using Test
using Lux
using Random: Xoshiro
using LifeAI: GPTModel, XLAKVDecoder, xla_decode_step!, xla_prefill!

@testset "Week 11 untied wide-attention XLA prefill/decode" begin
    model = GPTModel(
        31,
        8,
        4,
        1;
        num_kv_heads=2,
        head_dim=4,
        mlp_hidden_dim=24,
        use_bias=false,
        use_rope=true,
        use_qk_norm=true,
        qk_norm_epsilon=1.0f-6,
        max_seq_len=8,
        rope_theta=1.0f6,
        rope_style=:rotate_half,
        norm_epsilon=1.0f-6,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=false,
    )
    ps, st = Lux.setup(Xoshiro(20260723), model)
    prompt = reshape([1, 3, 5], :, 1)
    complete = reshape([1, 3, 5, 7], :, 1)
    reference, _ = model(complete, ps, st)

    decoder = XLAKVDecoder(model, ps, st; xla_backend="cpu")
    prefill_logits, _, _ = xla_prefill!(decoder, prompt)
    decode_logits, _, _ = xla_decode_step!(decoder, 7)
    host = Lux.cpu_device()

    @test model.num_heads * model.head_dim == 2 * model.d_model
    @test isapprox(
        host(prefill_logits),
        reference[:, 1:3, :];
        atol=1.0f-3,
        rtol=1.0f-3,
    )
    @test isapprox(
        host(decode_logits),
        reference[:, 4:4, :];
        atol=1.0f-3,
        rtol=1.0f-3,
    )
    @test decoder.host_position == 4
end
