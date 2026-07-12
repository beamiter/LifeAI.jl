using LifeAI
using Lux
using Random

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
    max_seq_len=32,
    use_rope=true,
)

# CPU works without optional packages. For NVIDIA:
#
#   using LuxCUDA
#   device = Lux.gpu_device()
#
device = Lux.cpu_device()
trainer = TrainerGPT(
    learning_rate=3.0f-3,
    device=device,
)
train_state = init_train_state(rng, model, trainer)

callback = info -> begin
    if info.step == 1 || info.step % 50 == 0
        println(
            "step=$(info.step) epoch=$(info.epoch) " *
            "loss=$(round(info.loss; digits=4))",
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

generated, _ = generate(
    model,
    train_state.parameters,
    train_state.states,
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
