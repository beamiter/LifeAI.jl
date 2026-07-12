using Test
using Random
using LifeAI:
    DatasetLoader,
    GPTModel,
    TrainerGPT,
    decode,
    encode,
    fit_tokenizer,
    generate,
    init_train_state,
    next_token_loss,
    train_step!,
    vocab_size

_all_finite(::Nothing) = true
_all_finite(x::Number) = isfinite(x)
_all_finite(x::AbstractArray) = all(isfinite, x)
_all_finite(x::NamedTuple) = all(_all_finite, values(x))
_all_finite(x::Tuple) = all(_all_finite, x)
_all_finite(_) = true

@testset "Sparse next-token loss" begin
    logits = zeros(Float32, 3, 2, 1)
    targets = reshape([1, 2], 2, 1)

    @test next_token_loss(logits, targets) ≈ log(3.0f0) atol=1.0f-6
    @test_throws DimensionMismatch next_token_loss(logits, reshape([1, 2], 1, 2))
    @test_throws ArgumentError next_token_loss(logits, reshape([1, 4], 2, 1))
end

@testset "Tiny GPT can learn one batch" begin
    rng = Xoshiro(20260712)

    text = repeat("abc", 40)
    tokenizer = fit_tokenizer(text)
    loader = DatasetLoader(
        tokenizer,
        text;
        seq_len=6,
        batch_size=4,
        stride=1,
        drop_last=true,
    )

    model = GPTModel(
        vocab_size(tokenizer),
        16,
        2,
        1;
        max_seq_len=6,
        use_rope=true,
    )

    trainer = TrainerGPT(learning_rate=2.0f-2)
    train_state = init_train_state(rng, model, trainer)

    x, targets = loader[1]
    initial_loss, _ = next_token_loss(
        model,
        train_state.parameters,
        train_state.states,
        x,
        targets,
    )
    initial_loss = Float32(initial_loss)

    train_state, _, gradients = train_step!(
        trainer,
        train_state,
        x,
        targets,
    )

    @test _all_finite(gradients)

    for _ in 2:80
        train_state, _, _ = train_step!(
            trainer,
            train_state,
            x,
            targets,
        )
    end

    final_loss, _ = next_token_loss(
        model,
        train_state.parameters,
        train_state.states,
        x,
        targets,
    )
    final_loss = Float32(final_loss)

    @test isfinite(final_loss)
    @test final_loss < initial_loss
    @test final_loss < initial_loss * 0.75f0

    generated_ids, _ = generate(
        model,
        train_state.parameters,
        train_state.states,
        encode(tokenizer, "a");
        max_new_tokens=6,
        temperature=0,
        rng=rng,
    )

    @test length(generated_ids) == 7
    @test all(id -> 1 <= id <= vocab_size(tokenizer), generated_ids)
    @test startswith(decode(tokenizer, generated_ids), "a")
end
