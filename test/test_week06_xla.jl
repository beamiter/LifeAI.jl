using Test
using Random
using Lux
using Optimisers
using LifeAI:
    GPTModel,
    TrainerGPT,
    XLAKVDecoder,
    generate_xla_cached!,
    init_train_state,
    train_step!,
    xla_decode_step!,
    xla_prefill!

# Week 06 XLA smoke: a GQA + QK-Norm model must compile and match the CPU
# eager path for training, prefill, and single-token decode.
@testset "Week 06 GQA + QK-Norm XLA smoke" begin
    model = GPTModel(
        61,
        32,
        4,
        2;
        num_kv_heads=2,
        use_qk_norm=true,
        use_rope=true,
        rope_theta=1.0f6,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
        max_seq_len=64,
    )

    rng = MersenneTwister(66_000)
    ps, st = Lux.setup(rng, model)

    prompt = rand(MersenneTwister(66_001), 1:model.vocab_size, 7)
    decode_tokens = rand(MersenneTwister(66_002), 1:model.vocab_size, 4)

    # CPU eager reference over the full sequence.
    full_sequence = vcat(prompt, decode_tokens)
    reference_logits, _ = model(reshape(full_sequence, :, 1), ps, st)

    decoder = XLAKVDecoder(model, ps, st; xla_backend="cpu")
    prefill_logits, _, _ = xla_prefill!(decoder, prompt)
    host = Lux.cpu_device()

    @test isapprox(
        host(prefill_logits)[:, end, 1],
        reference_logits[:, length(prompt), 1];
        atol=1.0f-3,
        rtol=1.0f-3,
    )

    for (offset, token) in enumerate(decode_tokens)
        step_logits, _, _ = xla_decode_step!(decoder, token)
        @test isapprox(
            host(step_logits)[:, 1, 1],
            reference_logits[:, length(prompt) + offset, 1];
            atol=1.0f-3,
            rtol=1.0f-3,
        )
    end

    # Compiled generation smoke through the same decoder.
    generated, _ = generate_xla_cached!(
        decoder,
        prompt;
        max_new_tokens=6,
        temperature=1.0f0,
        rng=MersenneTwister(66_003),
    )
    @test length(generated) == length(prompt) + 6
    @test all(id -> 1 <= id <= model.vocab_size, generated)

    # Compiled XLA training step on the same architecture.
    trainer = TrainerGPT(;
        optimizer=Adam(1.0f-3),
        backend=:xla,
        xla_backend="cpu",
    )
    state = init_train_state(MersenneTwister(66_004), model, trainer)
    x = rand(MersenneTwister(66_005), 1:model.vocab_size, 8, 2)
    y = rand(MersenneTwister(66_006), 1:model.vocab_size, 8, 2)

    losses = Float32[]
    for _ in 1:3
        state, loss, _ = train_step!(trainer, state, (x, y))
        push!(losses, Float32(loss))
    end
    @test all(isfinite, losses)
    @test losses[end] < losses[1]
end
