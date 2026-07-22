using Test
using Lux
using Random: Xoshiro
using LifeAI: GPTModel, XLAKVDecoder, xla_decode_step!, xla_prefill!

@testset "Week 07 rotate_half XLA prefill/decode" begin
    model = GPTModel(
        31,
        16,
        4,
        1;
        num_kv_heads=2,
        head_dim=4,
        mlp_hidden_dim=24,
        use_rope=true,
        rope_style=:rotate_half,
        use_qk_norm=true,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
        max_seq_len=12,
    )
    ps, st = Lux.setup(Xoshiro(20260722), model)
    prompt = reshape([1, 3, 5, 7], :, 1)
    complete = reshape([1, 3, 5, 7, 9], :, 1)
    reference, _ = model(complete, ps, st)

    decoder = XLAKVDecoder(model, ps, st; xla_backend="cpu")
    prefill_logits, _, _ = xla_prefill!(decoder, prompt)
    decode_logits, _, _ = xla_decode_step!(decoder, 9)
    host = Lux.cpu_device()

    @test isapprox(
        host(prefill_logits),
        reference[:, 1:4, :];
        atol=1.0f-3,
        rtol=1.0f-3,
    )
    @test isapprox(
        host(decode_logits),
        reference[:, 5:5, :];
        atol=1.0f-3,
        rtol=1.0f-3,
    )
    @test decoder.host_position == 5
end
