using Test
using Random
using Lux
using LifeAI:
    ByteBPETokenizer,
    GPTModel,
    TrainerGPT,
    XLAKVDecoder,
    encode,
    fit_byte_bpe,
    generate_xla_cached!,
    init_train_state,
    train_step!,
    vocab_size,
    xla_decode_step!,
    xla_prefill!

@testset "Week 05 byte-BPE XLA train and cached generation smoke" begin
    tokenizer = fit_byte_bpe(
        repeat("生命感来自观察、记忆、反馈与行动。", 40);
        vocab_size=272,
        min_frequency=2,
    )
    @test tokenizer isa ByteBPETokenizer
    model = GPTModel(
        vocab_size(tokenizer),
        16,
        2,
        1;
        max_seq_len=12,
        use_rope=true,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
    )

    trainer = TrainerGPT(
        backend=:xla,
        xla_backend="cpu",
        learning_rate=1.0f-3,
        max_grad_norm=1.0f0,
    )
    train_state = init_train_state(Xoshiro(20260720), model, trainer)
    sequence = encode(tokenizer, repeat("生命感", 64))
    @test length(sequence) >= 9
    inputs = reshape(sequence[1:8], 4, 2)
    targets = reshape(sequence[2:9], 4, 2)
    train_state, loss, _ = train_step!(trainer, train_state, inputs, targets)
    @test isfinite(loss)
    @test train_state.step == 1

    ps, st = Lux.setup(Xoshiro(20260721), model)
    decoder = XLAKVDecoder(model, ps, st; batch_size=1, xla_backend="cpu")
    prompt = encode(tokenizer, "生命")
    prompt = prompt[1:min(length(prompt), 4)]
    logits, _, _ = xla_prefill!(decoder, prompt)
    @test size(logits, 1) == vocab_size(tokenizer)
    next_logits, _, _ = xla_decode_step!(decoder, first(sequence))
    @test size(next_logits) == (vocab_size(tokenizer), 1, 1)

    generated, _ = generate_xla_cached!(
        decoder,
        tokenizer,
        "生命";
        max_new_tokens=1,
        temperature=0,
        rng=Xoshiro(20260722),
    )
    @test !isempty(generated)
end
