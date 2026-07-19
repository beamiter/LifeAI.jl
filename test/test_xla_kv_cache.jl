using Test
using Random
using Lux
using LifeAI:
    GPTModel,
    TrainerGPT,
    XLAKVDecoder,
    benchmark_xla_cache_modes,
    init_train_state,
    train_step!,
    xla_decode_step!,
    xla_prefill!

@testset "Reactant/XLA fixed-shape KV decoding" begin
    rng = Xoshiro(20260717)
    model = GPTModel(17, 16, 2, 2; max_seq_len=8, use_rope=true)
    ps, st = Lux.setup(rng, model)
    decoder = XLAKVDecoder(
        model,
        ps,
        st;
        batch_size=1,
        xla_backend="cpu",
    )

    prompt = reshape([1, 3, 5], 3, 1)
    reference_logits, _ = model(prompt, ps, st)
    logits, _, _ = xla_prefill!(decoder, prompt)

    @test isapprox(
        Array(logits),
        reference_logits;
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test decoder.host_position == 3
    @test length(decoder.prefill_thunks) == 1

    context = vec(prompt)
    cache_shapes = map(layer -> size(layer.keys), decoder.cache.layers)
    compiled_decode = nothing

    for token in (7, 9, 11)
        logits, _, _ = xla_decode_step!(decoder, token)
        if compiled_decode === nothing
            compiled_decode = decoder.decode_thunk
        else
            @test decoder.decode_thunk === compiled_decode
        end
        push!(context, token)
        reference_logits, _ = model(reshape(context, :, 1), ps, st)

        @test isapprox(
            vec(Array(logits)[:, 1, 1]),
            vec(reference_logits[:, end, 1]);
            atol=1.0f-5,
            rtol=1.0f-4,
        )
        @test map(layer -> size(layer.keys), decoder.cache.layers) == cache_shapes
    end

    @test decoder.decode_thunk === compiled_decode
    @test decoder.host_position == 6

    # Matching prompt shapes reuse the existing prefill executable.
    xla_prefill!(decoder, reshape([2, 4, 6], 3, 1))
    @test length(decoder.prefill_thunks) == 1

    modes = benchmark_xla_cache_modes(
        model,
        ps,
        st,
        [1, 3],
        [5];
        xla_backend="cpu",
        samples=1,
    )
    @test modes.no_cache.correctness.passed
    @test modes.dynamic_cache.correctness.passed
    @test modes.static_cache.correctness.passed
    @test length(modes.no_cache.steady.prefill_samples_seconds) == 1
    @test length(modes.dynamic_cache.steady.decode_samples_seconds) == 1
    @test length(modes.static_cache.steady.decode_samples_seconds) == 1
    @test modes.no_cache.executable_count == 2
    @test modes.dynamic_cache.executable_count == 2
    @test modes.static_cache.executable_count == 2
    @test modes.no_cache.theoretical_cache_bytes == 0
    @test modes.runtime_warmup_seconds >= 0
    @test modes.dynamic_cache.theoretical_cache_bytes <
        modes.static_cache.theoretical_cache_bytes
end

@testset "Modern GPT components compile for XLA training and static decoding" begin
    rng = Xoshiro(20260718)
    model = GPTModel(
        17,
        16,
        2,
        1;
        max_seq_len=8,
        use_rope=true,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
    )
    ps, st = Lux.setup(rng, model)
    decoder = XLAKVDecoder(
        model,
        ps,
        st;
        batch_size=1,
        xla_backend="cpu",
    )

    prompt = reshape([1, 3, 5], 3, 1)
    reference_prefill, _ = model(prompt, ps, st)
    prefill_logits, _, _ = xla_prefill!(decoder, prompt)
    @test isapprox(
        Array(prefill_logits),
        reference_prefill;
        atol=1.0f-5,
        rtol=1.0f-4,
    )

    decode_logits, _, _ = xla_decode_step!(decoder, 7)
    reference_decode, _ = model(reshape([1, 3, 5, 7], 4, 1), ps, st)
    @test isapprox(
        vec(Array(decode_logits)[:, 1, 1]),
        vec(reference_decode[:, end, 1]);
        atol=1.0f-5,
        rtol=1.0f-4,
    )

    trainer = TrainerGPT(
        backend=:xla,
        xla_backend="cpu",
        learning_rate=1.0f-3,
        max_grad_norm=1.0f0,
    )
    train_state = init_train_state(Xoshiro(20260719), model, trainer)
    inputs = [1 2; 3 4; 5 6; 7 8]
    targets = [3 4; 5 6; 7 8; 9 10]
    train_state, loss, _ = train_step!(
        trainer,
        train_state,
        inputs,
        targets,
    )

    @test isfinite(loss)
    @test train_state.step == 1
end
