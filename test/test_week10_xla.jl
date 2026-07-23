using Test
using Lux
using Random: Xoshiro
using LifeAI: GPTModel, XLAKVDecoder, xla_decode_step!, xla_prefill!

@testset "Week 10 GPT-2 learned-position XLA prefill/decode" begin
    model = GPTModel(
        31,
        16,
        4,
        1;
        mlp_hidden_dim=64,
        use_bias=true,
        lm_head_bias=false,
        use_rope=false,
        position_embedding_type=:learned_absolute,
        norm_type=:layernorm,
        mlp_type=:gelu_new,
        tie_embeddings=true,
        max_seq_len=8,
    )
    ps, st = Lux.setup(Xoshiro(20260723), model)
    prompt = reshape([1, 3, 5], :, 1)
    complete = reshape([1, 3, 5, 7], :, 1)
    reference, _ = model(complete, ps, st)

    decoder = XLAKVDecoder(model, ps, st; xla_backend="cpu")
    prefill_logits, _, _ = xla_prefill!(decoder, prompt)
    decode_logits, _, _ = xla_decode_step!(decoder, 7)
    host = Lux.cpu_device()

    @test isapprox(
        host(prefill_logits),
        reference[:, 1:3, :];
        atol=1.0f-4,
        rtol=1.0f-4,
    )
    @test isapprox(
        host(decode_logits),
        reference[:, 4:4, :];
        atol=1.0f-4,
        rtol=1.0f-4,
    )
    @test decoder.host_position == 4
    @test length(decoder.prefill_thunks) == 1
end
