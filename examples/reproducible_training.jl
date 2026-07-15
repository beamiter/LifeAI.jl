using LifeAI
using Random

rng = Xoshiro(20260715)

text = repeat(
    """
    小机器人每天学习一个新词。
    它会记住温柔的声音，也会认真观察周围的变化。
    生命感来自连续的记忆、反馈和成长。
    """,
    48,
)

data = train_validation_loaders(
    text;
    validation_fraction=0.15,
    seq_len=24,
    batch_size=8,
    stride=12,
    drop_last=true,
    add_unk=true,
)

model = GPTModel(
    vocab_size(data.tokenizer),
    48,
    4,
    2;
    max_seq_len=96,
    use_rope=true,
)

trainer = TrainerGPT(
    learning_rate=3.0f-3,
    backend=:zygote,
    max_grad_norm=1.0f0,
)

train_state = init_train_state(rng, model, trainer)
last_progress = Ref((; epoch=0, batch=0, step=0))
latest_metrics = Ref((; train_loss=Inf32, validation_loss=Inf32, perplexity=Inf32))

callback = info -> begin
    last_progress[] = info.progress

    if info.validation_loss !== nothing
        latest_metrics[] = (;
            train_loss=info.loss,
            validation_loss=info.validation_loss,
            perplexity=info.perplexity,
        )
        println(
            "step=$(info.step) train=$(round(info.loss; digits=4)) " *
            "validation=$(round(info.validation_loss; digits=4)) " *
            "ppl=$(round(info.perplexity; digits=3)) " *
            "grad=$(round(info.grad_norm_before; digits=3))→" *
            "$(round(info.grad_norm_after; digits=3))",
        )
    end
end

train_state, _ = train_gpt!(
    trainer,
    train_state,
    data.train;
    epochs=2,
    validation_loader=data.validation,
    callback,
)

checkpoint_path = joinpath(
    @__DIR__,
    "..",
    "artifacts",
    "week03_reproducible_training.checkpoint",
)
save_checkpoint(
    checkpoint_path,
    model,
    data.tokenizer,
    trainer,
    train_state;
    rng,
    progress=last_progress[],
    train_config=(;
        seq_len=data.train.seq_len,
        batch_size=data.train.batch_size,
        stride=data.train.stride,
    ),
    metrics=latest_metrics[],
)

println("saved checkpoint: $checkpoint_path")

checkpoint = load_checkpoint(
    checkpoint_path;
    backend=:zygote,
)

resumed_state, _ = resume_gpt!(
    checkpoint,
    data.train;
    epochs=1,
    validation_loader=data.validation,
    callback,
)

validation_metrics, _ = evaluate_gpt(
    checkpoint.model,
    resumed_state.parameters,
    resumed_state.states,
    data.validation,
)

prompt = "小机器人"
generated_ids, _ = generate(
    checkpoint.model,
    resumed_state.parameters,
    resumed_state.states,
    encode(checkpoint.tokenizer, prompt);
    max_new_tokens=48,
    temperature=0.8f0,
    top_k=8,
    rng=checkpoint.rng === nothing ? rng : checkpoint.rng,
)

println()
println("resumed step: $(resumed_state.step)")
println("validation loss: $(round(validation_metrics.loss; digits=4))")
println("validation perplexity: $(round(validation_metrics.perplexity; digits=3))")
println("generated:")
println(decode(checkpoint.tokenizer, generated_ids))
