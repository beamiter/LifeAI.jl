using LifeAI
using Enzyme
using Lux
using Random
using Reactant

rng = Xoshiro(42)

text = repeat(
    """
    生命是一场不断学习的旅程。
    小机器人睁开眼睛，第一次看见这个世界。
    它学会观察，也学会记住温柔的声音。
    """,
    64,
)

tokenizer = fit_tokenizer(text; add_unk=true)
loader = DatasetLoader(
    tokenizer,
    text;
    seq_len=32,
    batch_size=16,
    stride=16,
    drop_last=true,
)

model = GPTModel(
    vocab_size(tokenizer),
    64,
    4,
    2;
    max_seq_len=160,
    use_rope=true,
)

# Reactant compiles the complete training step through XLA. The first batch
# includes compilation; later fixed-shape batches reuse the cached executable.
# Override with LIFEAI_XLA_BACKEND=cpu when testing without an NVIDIA GPU.
xla_backend = get(ENV, "LIFEAI_XLA_BACKEND", "gpu")
println("Training with Reactant/XLA backend: $xla_backend")

trainer = TrainerGPT(
    learning_rate=3.0f-3,
    backend=:xla,
    xla_backend=xla_backend,
    return_gradients=false,
    static_shapes=true,
)
train_state = init_train_state(rng, model, trainer)

callback = info -> begin
    if info.step == 1 || info.step % 50 == 0
        compile_note = info.xla_compilation ? " (includes XLA compilation)" : ""
        println(
            "step=$(info.step) epoch=$(info.epoch) " *
            "loss=$(round(info.loss; digits=4)) " *
            "time=$(round(info.step_seconds; digits=3))s" * compile_note,
        )
    end
end

train_state, losses = train_gpt!(
    trainer,
    train_state,
    loader;
    epochs=25,
    callback=callback,
)

println("initial loss = $(first(losses))")
println("final loss   = $(last(losses))")

# Keep parameters and states on the Reactant device. Prefill compiles once for
# this prompt shape; the fixed-shape one-token decode executable is then reused
# for all later token positions.
decoder = XLAKVDecoder(
    model,
    train_state.parameters,
    train_state.states;
    xla_backend=xla_backend,
)

generated, _ = generate_xla_cached!(
    decoder,
    tokenizer,
    "小机器人";
    max_new_tokens=120,
    temperature=0.8f0,
    top_k=8,
    rng=rng,
)

println()
println("Generated text:")
println(generated)
