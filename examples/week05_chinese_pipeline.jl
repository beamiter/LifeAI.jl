using LifeAI
using Lux
using Printf
using Random

fixture_path = normpath(joinpath(@__DIR__, "..", "data", "fixtures", "week05_chinese.toml"))
documents = load_text_documents(fixture_path)

data = build_document_dataset(
    documents;
    tokenizer_type=:byte_bpe,
    normalization=:none,
    vocab_size=320,
    min_frequency=2,
    validation_size=1,
    split_seed=20260720,
    seq_len=16,
    batch_size=2,
    stride=8,
    drop_last=true,
)

model = GPTModel(
    vocab_size(data.tokenizer),
    32,
    4,
    2;
    max_seq_len=32,
    use_rope=true,
    norm_type=:rmsnorm,
    mlp_type=:swiglu,
    tie_embeddings=true,
)
trainer = TrainerGPT(
    backend=:zygote,
    device=Lux.cpu_device(),
    learning_rate=2.0f-3,
    return_gradients=false,
    max_grad_norm=1.0f0,
)
train_state = init_train_state(Xoshiro(20260720), model, trainer)
train_state, train_loss, _ = train_step!(trainer, train_state, data.train[1])
validation, _ = evaluate_gpt(
    model,
    train_state.parameters,
    train_state.states,
    data.validation,
)

println("Week 05 Chinese pipeline")
println("  tokenizer fingerprint: ", tokenizer_fingerprint(data.tokenizer))
println("  split fingerprint:     ", data.split.fingerprint)
println("  vocabulary size:       ", vocab_size(data.tokenizer))
println("  model parameters:      ", Lux.parameterlength(train_state.parameters))
@printf("  train loss:            %.4f\n", Float64(train_loss))
@printf("  validation loss:       %.4f\n", Float64(validation.loss))
@printf("  validation bits/byte:  %.4f\n", Float64(validation.bits_per_byte))

mktempdir() do directory
    dataset_directory = joinpath(directory, "dataset")
    checkpoint_path = joinpath(directory, "week05.checkpoint")

    artifact = save_dataset_artifact(
        dataset_directory,
        data;
        name="lifeai-week05-chinese-fixture",
        version="1",
    )
    reloaded_data = load_dataset_artifact(
        dataset_directory;
        seq_len=16,
        batch_size=2,
        stride=8,
        drop_last=true,
    )
    reloaded_data.fingerprint == artifact.fingerprint || error(
        "dataset artifact fingerprint changed after reload",
    )

    save_checkpoint(
        checkpoint_path,
        model,
        data.tokenizer,
        trainer,
        train_state;
        rng=Xoshiro(20260721),
        progress=(; epoch=1, batch=1),
        train_config=(;
            tokenizer_fingerprint=tokenizer_fingerprint(data.tokenizer),
            dataset_fingerprint=artifact.fingerprint,
            seq_len=16,
            batch_size=2,
        ),
        metrics=(;
            train_loss=Float32(train_loss),
            validation_loss=validation.loss,
            validation_bits_per_byte=validation.bits_per_byte,
        ),
        metadata=(; week=5, fixture=basename(fixture_path)),
    )

    checkpoint = load_checkpoint(checkpoint_path; backend=:zygote)
    resumed_state, resumed_losses = resume_gpt!(
        checkpoint,
        reloaded_data.train;
        epochs=1,
        max_steps=1,
    )
    generated, _ = generate_cached(
        checkpoint.model,
        resumed_state.parameters,
        resumed_state.states,
        checkpoint.tokenizer,
        "生命感";
        max_new_tokens=8,
        temperature=0,
        rng=Xoshiro(20260722),
    )

    println("  restored tokenizer:    ", typeof(checkpoint.tokenizer))
    println("  resumed step:          ", resumed_state.step)
    println("  resumed losses:        ", resumed_losses)
    println("  generated:             ", repr(generated))
end
