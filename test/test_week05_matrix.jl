using Test
using Random
using LifeAI:
    GPTModel,
    TextDocument,
    TrainerGPT,
    build_document_dataset,
    evaluate_gpt,
    generate_cached,
    init_train_state,
    load_checkpoint,
    resume_gpt!,
    save_checkpoint,
    tokenizer_fingerprint,
    train_step!,
    vocab_size

function _week05_matrix_document(id, suffix)
    return TextDocument(
        id,
        repeat("生命感来自观察、记忆、反馈和行动。$suffix", 24);
        source_id="week05-matrix",
        source_location="repository test fixture",
        license="CC0-1.0",
    )
end

@testset "Week 05 three-tokenizer end-to-end matrix" begin
    documents = [
        _week05_matrix_document("matrix-a", "甲"),
        _week05_matrix_document("matrix-b", "乙"),
        _week05_matrix_document("matrix-c", "丙"),
    ]

    for (index, profile) in enumerate((:character, :byte, :byte_bpe))
        @testset "$profile" begin
            data = build_document_dataset(
                documents;
                tokenizer_type=profile,
                add_unk=true,
                vocab_size=268,
                min_frequency=2,
                validation_size=1,
                split_seed=20260720,
                seq_len=4,
                batch_size=1,
                stride=4,
                drop_last=true,
            )
            model = GPTModel(
                vocab_size(data.tokenizer),
                8,
                2,
                1;
                max_seq_len=8,
                use_rope=true,
                norm_type=:rmsnorm,
                mlp_type=:swiglu,
                tie_embeddings=true,
            )
            trainer = TrainerGPT(
                learning_rate=1.0f-3,
                return_gradients=false,
                max_grad_norm=1.0f0,
            )
            state = init_train_state(Xoshiro(700 + index), model, trainer)
            state, loss, _ = train_step!(trainer, state, data.train[1])
            @test isfinite(loss)
            @test state.step == 1

            metrics, _ = evaluate_gpt(
                model,
                state.parameters,
                state.states,
                data.validation,
            )
            @test isfinite(metrics.loss)
            @test metrics.bytes !== nothing
            @test isfinite(metrics.bits_per_byte)

            mktempdir() do directory
                path = joinpath(directory, "$(profile).checkpoint")
                save_checkpoint(
                    path,
                    model,
                    data.tokenizer,
                    trainer,
                    state;
                    progress=(; epoch=1, batch=1),
                )
                checkpoint = load_checkpoint(path; backend=:zygote)
                @test typeof(checkpoint.tokenizer) == typeof(data.tokenizer)
                @test tokenizer_fingerprint(checkpoint.tokenizer) ==
                      tokenizer_fingerprint(data.tokenizer)

                resumed, resumed_losses = resume_gpt!(
                    checkpoint,
                    data.train;
                    epochs=1,
                    max_steps=1,
                )
                @test resumed.step == state.step + 1
                @test length(resumed_losses) == 1
                @test isfinite(only(resumed_losses))

                generated, _ = generate_cached(
                    checkpoint.model,
                    resumed.parameters,
                    resumed.states,
                    checkpoint.tokenizer,
                    "生命";
                    max_new_tokens=1,
                    temperature=0,
                    rng=Xoshiro(800 + index),
                )
                @test !isempty(generated)
            end
        end
    end
end
