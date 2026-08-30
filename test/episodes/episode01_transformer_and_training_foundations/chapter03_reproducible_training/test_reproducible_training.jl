using Test
using Random
using Lux
using Serialization
import LifeAI
using LifeAI:
    CHECKPOINT_FORMAT_VERSION,
    DatasetLoader,
    DocumentDatasetLoader,
    GPTModel,
    TrainerGPT,
    benchmark_kv_cache,
    benchmark_xla_cache_modes,
    bits_per_byte,
    clip_global_gradient_norm,
    evaluate_gpt,
    fit_tokenizer,
    global_gradient_norm,
    gpt_config,
    init_train_state,
    kv_cache_correctness,
    load_checkpoint,
    next_token_loss,
    next_token_nll_sum,
    resume_gpt!,
    save_checkpoint,
    split_token_stream,
    target_byte_count,
    train_gpt!,
    train_step!,
    train_validation_loaders,
    vocab_size

function _tree_isapprox(left, right; atol=1.0f-6, rtol=1.0f-5)
    typeof(left) == typeof(right) || return false

    if left === nothing
        return true
    elseif left isa Number
        return isapprox(left, right; atol, rtol)
    elseif left isa AbstractArray
        return size(left) == size(right) && isapprox(left, right; atol, rtol)
    elseif left isa NamedTuple
        return keys(left) == keys(right) && all(
            _tree_isapprox(a, b; atol, rtol)
            for (a, b) in zip(values(left), values(right))
        )
    elseif left isa Tuple
        return length(left) == length(right) && all(
            _tree_isapprox(a, b; atol, rtol)
            for (a, b) in zip(left, right)
        )
    elseif isstructtype(typeof(left)) && fieldcount(typeof(left)) > 0
        return all(
            _tree_isapprox(
                getfield(left, index),
                getfield(right, index);
                atol,
                rtol,
            )
            for index in 1:fieldcount(typeof(left))
        )
    end

    return isequal(left, right)
end

struct ExtremeEvaluationModel
    vocab_size::Int
end

function (model::ExtremeEvaluationModel)(x, ps, st)
    logits = fill(-1.0f3, model.vocab_size, size(x)...)
    logits[1, :, :] .= 0.0f0
    return logits, st
end

struct MaximumLengthLoader end
Base.length(::MaximumLengthLoader) = typemax(Int)

struct InvalidTrainingModel <: Lux.AbstractLuxLayer
    vocab_size::Int
    failure::Symbol
end

function Lux.initialparameters(::AbstractRNG, model::InvalidTrainingModel)
    initial_scale = model.failure in (:gradient, :large_gradient) ? 0.0 : 1.0
    return (; scale=Float64[initial_scale])
end

Lux.initialstates(::AbstractRNG, ::InvalidTrainingModel) = NamedTuple()

function (model::InvalidTrainingModel)(x, ps, st)
    offset = if model.failure === :loss
        1.0e40 + only(ps.scale)
    elseif model.failure === :gradient
        sqrt(only(ps.scale))
    elseif model.failure === :large_gradient
        only(ps.scale) * 1.0e100
    elseif model.failure === :valid
        only(ps.scale)
    else
        error("unsupported invalid-training fixture")
    end
    logits = repeat(
        reshape(Float64[0, -offset], 2, 1, 1),
        1,
        size(x, 1),
        size(x, 2),
    )
    return logits, st
end

@testset "Leakage-safe train/validation split" begin
    split = split_token_stream(collect(1:20); validation_size=6)
    @test split.train == collect(1:14)
    @test split.validation == collect(15:20)
    @test split.split_index == 14

    text = repeat("abcd", 20) * repeat("z", 20)
    data = train_validation_loaders(
        text;
        validation_size=20,
        seq_len=4,
        batch_size=3,
        stride=2,
        drop_last=false,
        add_unk=true,
    )

    @test data.text_split.train == repeat("abcd", 20)
    @test data.text_split.validation == repeat("z", 20)
    @test !('z' in data.tokenizer)
    @test data.tokenizer.unk_id !== nothing
    @test all(==(data.tokenizer.unk_id), data.token_split.validation)
    @test all(
        start -> start + data.train.seq_len <= length(data.train.token_ids),
        data.train.starts,
    )
    @test all(
        start -> start + data.validation.seq_len <= length(data.validation.token_ids),
        data.validation.starts,
    )
end

@testset "Token-weighted evaluation and perplexity" begin
    logits = zeros(Float32, 3, 2, 1)
    targets = reshape([1, 2], 2, 1)
    expected_nll = 2 * log(3.0f0)

    @test next_token_nll_sum(logits, targets) ≈ expected_nll atol=1.0f-6
    @test next_token_loss(logits, targets) ≈ log(3.0f0) atol=1.0f-6

    rng = Xoshiro(123)
    text = repeat("abc", 12)
    tokenizer = fit_tokenizer(text)
    loader = DatasetLoader(
        tokenizer,
        text;
        seq_len=4,
        batch_size=3,
        stride=3,
        drop_last=false,
    )
    model = GPTModel(
        vocab_size(tokenizer),
        12,
        3,
        1;
        max_seq_len=4,
        use_rope=true,
    )
    ps, st = Lux.setup(rng, model)
    ps_before = deepcopy(ps)

    metrics, _ = evaluate_gpt(model, ps, st, loader)
    document_loader_without_bytes = DocumentDatasetLoader(
        [Int[1, 2, 1, 2, 1]];
        seq_len=2,
        batch_size=1,
        drop_last=false,
    )
    metrics_without_bytes, _ = evaluate_gpt(
        model,
        ps,
        st,
        document_loader_without_bytes,
    )
    @test metrics_without_bytes.bytes === nothing
    @test metrics_without_bytes.nll_per_byte === nothing
    @test metrics_without_bytes.bits_per_byte === nothing
    @test_throws ArgumentError evaluate_gpt(
        model, ps, st, loader; byte_count=true,
    )
    @test_throws ArgumentError evaluate_gpt(
        model, ps, st, loader; byte_count=big(typemax(Int)) + 1,
    )
    overflowing_byte_loader = DocumentDatasetLoader(
        [Int[1, 2, 3, 4]];
        byte_lengths=[[0, typemax(Int), typemax(Int), typemax(Int)]],
        seq_len=3,
        batch_size=1,
        drop_last=false,
    )
    @test_throws OverflowError target_byte_count(overflowing_byte_loader)
    extreme_loader = [(
        reshape(Int[1], 1, 1),
        reshape(Int[2], 1, 1),
    )]
    @test_throws OverflowError evaluate_gpt(
        ExtremeEvaluationModel(2),
        NamedTuple(),
        NamedTuple(),
        extreme_loader;
        device=identity,
    )

    manual_nll = 0.0
    manual_tokens = 0
    for (x, y) in loader
        batch_logits, _ = model(x, ps, st)
        manual_nll += Float64(next_token_nll_sum(batch_logits, y))
        manual_tokens += length(y)
    end
    manual_mean = manual_nll / manual_tokens

    @test metrics.tokens == manual_tokens
    @test metrics.total_nll ≈ manual_nll atol=1.0e-6
    @test metrics.loss ≈ manual_mean atol=1.0f-6
    @test metrics.perplexity ≈ exp(manual_mean) atol=1.0f-5
    @test _tree_isapprox(ps, ps_before)
end

@testset "Training and evaluation numeric contracts" begin
    @test bits_per_byte(2.0, 2) ≈ 1 / log(2.0)
    @test_throws ArgumentError bits_per_byte(true, 1)
    @test_throws ArgumentError bits_per_byte(-1.0, 1)
    @test_throws ArgumentError bits_per_byte(-big"1e-1000", 1)
    @test_throws OverflowError bits_per_byte(big"1e-1000", 1)
    @test_throws OverflowError bits_per_byte(nextfloat(0.0), typemax(Int))
    @test_throws ArgumentError bits_per_byte(NaN, 1)
    @test_throws ArgumentError bits_per_byte(Inf, 1)
    @test_throws ArgumentError bits_per_byte(big"1e1000", 1)
    @test_throws ArgumentError bits_per_byte(1.0, true)
    @test_throws ArgumentError bits_per_byte(1.0, big(typemax(Int)) + 1)

    @test_throws ArgumentError TrainerGPT(learning_rate=true)
    @test_throws ArgumentError TrainerGPT(learning_rate=NaN)
    @test_throws ArgumentError TrainerGPT(learning_rate=Inf)
    @test_throws ArgumentError TrainerGPT(learning_rate=big"1e1000")
    @test_throws OverflowError TrainerGPT(learning_rate=1.0e-100)
    @test_throws ArgumentError TrainerGPT(max_grad_norm=true)
    @test_throws ArgumentError TrainerGPT(max_grad_norm=Inf)
    @test_throws ArgumentError TrainerGPT(max_grad_norm=big"1e1000")
    @test_throws OverflowError TrainerGPT(max_grad_norm=1.0e-100)
    @test_throws ArgumentError TrainerGPT(return_gradients=1)

    @test LifeAI._training_metric_float32(1.25, "test metric") == 1.25f0
    @test_throws ArgumentError LifeAI._training_metric_float32(
        NaN, "test metric",
    )
    @test_throws ArgumentError LifeAI._training_metric_float32(
        Inf, "test metric",
    )
    @test_throws OverflowError LifeAI._training_metric_float32(
        floatmax(Float64), "test metric",
    )
    @test_throws OverflowError LifeAI._training_metric_float32(
        big"1e-1000", "test metric",
    )
    @test_throws OverflowError LifeAI._evaluation_metric_float32(
        big"1e-1000", "test metric",
    )
    @test_throws ArgumentError LifeAI._training_metric_float32(
        -1.0, "test metric"; nonnegative=true,
    )
    @test_throws ArgumentError LifeAI._training_metric_float32(
        -1.0e-100, "test metric"; nonnegative=true,
    )
    @test_throws ArgumentError LifeAI._training_metric_float32(
        true, "test metric",
    )
    @test_throws ArgumentError LifeAI._training_metric_float32(
        Complex(1, 0), "test metric",
    )
    @test_throws ArgumentError LifeAI._checked_training_gradient_norm((;
        value=Float64[Inf],
    ))
    @test_throws OverflowError LifeAI._checked_training_gradient_norm((;
        value=Float64[1.0e100],
    ))
    @test global_gradient_norm((; value=Float32[1.0f-30])) == 1.0f-30
    @test global_gradient_norm((; value=Float32[1.9f19])) == 1.9f19
    @test global_gradient_norm((; value=Int[typemax(Int)])) ==
        Float64(typemax(Int))
    @test global_gradient_norm((; value=Int[typemin(Int)])) ==
        -Float64(typemin(Int))

    trainer = TrainerGPT()
    @test_throws ArgumentError train_gpt!(trainer, nothing, (); epochs=true)
    @test_throws ArgumentError train_gpt!(trainer, nothing, (); start_epoch=true)
    @test_throws ArgumentError train_gpt!(trainer, nothing, (); start_batch=true)
    @test_throws ArgumentError train_gpt!(trainer, nothing, (); max_steps=true)
    @test_throws ArgumentError train_gpt!(
        trainer, nothing, (); epochs=big(typemax(Int)) + 1,
    )
    @test_throws ArgumentError train_gpt!(
        trainer, nothing, (); max_steps=big(typemax(Int)) + 1,
    )
    @test_throws ArgumentError train_gpt!(
        trainer,
        nothing,
        ();
        epochs=2,
        start_epoch=typemax(Int),
    )
    @test_throws ArgumentError train_gpt!(
        trainer,
        nothing,
        ();
        validation_loader=(),
        evaluate_every=true,
    )
    max_length_state, max_length_losses = train_gpt!(
        trainer,
        nothing,
        MaximumLengthLoader();
        start_batch=typemax(Int),
        max_steps=0,
    )
    @test max_length_state === nothing
    @test isempty(max_length_losses)
    @test_throws OverflowError train_gpt!(
        trainer,
        (; step=typemax(Int)),
        [(reshape(Int[1], 1, 1), reshape(Int[1], 1, 1))];
        max_steps=1,
    )

    invalid_batch = (
        reshape(Int[1], 1, 1),
        reshape(Int[2], 1, 1),
    )
    for (failure, exception_type) in [
        (:loss, OverflowError),
        (:gradient, ArgumentError),
    ]
        invalid_model = InvalidTrainingModel(2, failure)
        invalid_state = init_train_state(
            Xoshiro(20260829),
            invalid_model,
            trainer,
        )
        parameters_before = deepcopy(invalid_state.parameters)
        optimizer_state_before = deepcopy(invalid_state.optimizer_state)
        step_before = invalid_state.step

        @test_throws exception_type train_step!(
            trainer,
            invalid_state,
            invalid_batch,
        )
        @test _tree_isapprox(
            invalid_state.parameters,
            parameters_before;
            atol=0,
            rtol=0,
        )
        @test _tree_isapprox(
            invalid_state.optimizer_state,
            optimizer_state_before;
            atol=0,
            rtol=0,
        )
        @test invalid_state.step == step_before
    end

    clipping_trainer = TrainerGPT(max_grad_norm=1.0f0)
    for (failure, exception_type) in [
        (:gradient, ArgumentError),
        (:large_gradient, OverflowError),
    ]
        invalid_model = InvalidTrainingModel(2, failure)
        invalid_state = init_train_state(
            Xoshiro(20260829),
            invalid_model,
            clipping_trainer,
        )
        parameters_before = deepcopy(invalid_state.parameters)
        optimizer_state_before = deepcopy(invalid_state.optimizer_state)
        step_before = invalid_state.step

        @test_throws exception_type train_step!(
            clipping_trainer,
            invalid_state,
            invalid_batch,
        )
        @test _tree_isapprox(
            invalid_state.parameters,
            parameters_before;
            atol=0,
            rtol=0,
        )
        @test _tree_isapprox(
            invalid_state.optimizer_state,
            optimizer_state_before;
            atol=0,
            rtol=0,
        )
        @test invalid_state.step == step_before
    end

    no_return_trainer = TrainerGPT(return_gradients=false)
    no_return_state = init_train_state(
        Xoshiro(20260829),
        InvalidTrainingModel(2, :valid),
        no_return_trainer,
    )
    no_return_state, no_return_loss, returned_gradients = train_step!(
        no_return_trainer,
        no_return_state,
        invalid_batch,
    )
    @test isfinite(no_return_loss)
    @test returned_gradients === nothing
    @test no_return_state.step == 1

    gradients = (; value=Float32[3, 4])
    @test_throws ArgumentError clip_global_gradient_norm(gradients, true)
    @test_throws ArgumentError clip_global_gradient_norm(gradients, Inf)
    @test_throws ArgumentError clip_global_gradient_norm(gradients, big"1e1000")
    @test_throws OverflowError clip_global_gradient_norm(gradients, 1.0e-100)
    @test_throws ArgumentError clip_global_gradient_norm(
        gradients, 1.0f0; epsilon=true,
    )
    @test_throws OverflowError clip_global_gradient_norm(
        gradients, 1.0f0; epsilon=1.0e-100,
    )
    @test_throws ArgumentError clip_global_gradient_norm(
        (; value=Float32[Inf]), 1.0f0,
    )
    @test_throws ArgumentError clip_global_gradient_norm(
        (; value=Float32[NaN]), 1.0f0,
    )
    @test_throws OverflowError clip_global_gradient_norm(
        (; value=Float32[1.0f19]), nextfloat(0.0f0),
    )

    checkpoint_with_boolean_progress = (;
        trainer,
        train_state=(; step=0),
        progress=(; epoch=true, batch=0, step=0),
    )
    @test_throws ArgumentError resume_gpt!(
        checkpoint_with_boolean_progress,
        (),
    )
    checkpoint_with_overflowing_batch = (;
        trainer,
        train_state=(; step=0),
        progress=(; epoch=1, batch=typemax(Int), step=0),
    )
    @test_throws ArgumentError resume_gpt!(
        checkpoint_with_overflowing_batch,
        (),
    )
    checkpoint_with_mismatched_step = (;
        trainer,
        train_state=(; step=1),
        progress=(; epoch=1, batch=0, step=2),
    )
    @test_throws ArgumentError resume_gpt!(
        checkpoint_with_mismatched_step,
        (),
    )
    checkpoint_with_missing_step = (;
        trainer,
        train_state=(; step=1),
        progress=(; epoch=1, batch=0),
    )
    @test_throws ArgumentError resume_gpt!(
        checkpoint_with_missing_step,
        (),
    )
    checkpoint_past_loader = (;
        trainer,
        train_state=(; step=0),
        progress=(; epoch=1, batch=2, step=0),
    )
    @test_throws ArgumentError resume_gpt!(
        checkpoint_past_loader,
        [invalid_batch],
    )
    checkpoint_with_impossible_position = (;
        trainer,
        train_state=(; step=0),
        progress=(; epoch=0, batch=1, step=0),
    )
    @test_throws ArgumentError resume_gpt!(
        checkpoint_with_impossible_position,
        [invalid_batch],
    )
end

@testset "Global gradient norm clipping" begin
    gradients = (;
        first=Float32[3, 4],
        nested=(; second=Float32[0, 12]),
    )
    @test global_gradient_norm(gradients) ≈ 13.0f0 atol=1.0f-6

    clipped, metrics = clip_global_gradient_norm(gradients, 6.5f0)
    @test metrics.before ≈ 13.0f0 atol=1.0f-5
    @test metrics.after ≈ 6.5f0 atol=1.0f-5
    @test metrics.scale ≈ 0.5f0 atol=1.0f-5
    @test clipped.first ≈ gradients.first .* 0.5f0 atol=1.0f-5
    @test clipped.nested.second ≈ gradients.nested.second .* 0.5f0 atol=1.0f-5

    unchanged, unchanged_metrics = clip_global_gradient_norm(gradients, 26.0f0)
    @test unchanged_metrics.scale ≈ 1.0f0 atol=1.0f-6
    @test unchanged_metrics.after ≈ unchanged_metrics.before atol=1.0f-5
    @test unchanged.first ≈ gradients.first atol=1.0f-6
    @test unchanged.nested.second ≈ gradients.nested.second atol=1.0f-6
end

@testset "Checkpoint round-trip and deterministic resume" begin
    text = repeat("abc", 30)
    tokenizer = fit_tokenizer(text; add_unk=true)
    loader = DatasetLoader(
        tokenizer,
        text;
        seq_len=6,
        batch_size=4,
        stride=2,
        drop_last=true,
    )
    model = GPTModel(
        vocab_size(tokenizer),
        16,
        2,
        1;
        head_dim=8,
        mlp_hidden_dim=40,
        use_bias=true,
        max_seq_len=6,
        use_rope=true,
        rope_theta=5000.0,
        norm_epsilon=2.0f-5,
    )
    trainer = TrainerGPT(
        learning_rate=5.0f-3,
        max_grad_norm=1.0f0,
    )

    continuous_state = init_train_state(Xoshiro(20260715), model, trainer)
    interrupted_state = init_train_state(Xoshiro(20260715), model, trainer)

    continuous_state, _, _ = train_step!(trainer, continuous_state, loader[1])
    continuous_state, continuous_loss, _ = train_step!(
        trainer,
        continuous_state,
        loader[2],
    )
    interrupted_state, _, _ = train_step!(
        trainer,
        interrupted_state,
        loader[1],
    )

    x, _ = loader[1]
    logits_before, _ = model(
        x,
        interrupted_state.parameters,
        interrupted_state.states,
    )

    mktempdir() do directory
        path = joinpath(directory, "reproducible_training.checkpoint")
        saved_path = save_checkpoint(
            path,
            model,
            tokenizer,
            trainer,
            interrupted_state;
            rng=Xoshiro(99),
            progress=(; epoch=1, batch=1),
            train_config=(; seq_len=6, batch_size=4),
            metrics=(; train_loss=1.25f0),
            metadata=(; purpose="test"),
        )

        @test saved_path == abspath(path)
        @test isfile(path)
        @test_throws ArgumentError save_checkpoint(
            joinpath(directory, "invalid-progress.checkpoint"),
            model,
            tokenizer,
            trainer,
            interrupted_state;
            progress=(; epoch=true, batch=0),
        )
        @test_throws ArgumentError save_checkpoint(
            joinpath(directory, "mismatched-progress-step.checkpoint"),
            model,
            tokenizer,
            trainer,
            interrupted_state;
            progress=(; epoch=1, batch=1, step=interrupted_state.step + 1),
        )
        @test_throws ArgumentError save_checkpoint(
            joinpath(directory, "impossible-progress-position.checkpoint"),
            model,
            tokenizer,
            trainer,
            interrupted_state;
            progress=(; epoch=0, batch=1),
        )

        raw_payload = open(path, "r") do io
            deserialize(io)
        end
        tampered_progresses = (
            (; epoch=1, batch=1, step=raw_payload.step + 1),
            (; epoch=true, batch=1, step=raw_payload.step),
            (; epoch=1, batch=-1, step=raw_payload.step),
            (; epoch=1, batch=1, step=true),
            (; epoch=0, batch=1, step=raw_payload.step),
            (; epoch=1, batch=1),
        )
        for (index, tampered_progress) in enumerate(tampered_progresses)
            tampered_path = joinpath(directory, "tampered-progress-$index.checkpoint")
            open(tampered_path, "w") do io
                serialize(io, merge(raw_payload, (; progress=tampered_progress)))
            end
            @test_throws ArgumentError load_checkpoint(
                tampered_path;
                backend=:zygote,
            )
        end
        for (label, invalid_version) in (
            ("boolean", true),
            ("overflow", big(typemax(Int)) + 1),
        )
            invalid_version_path = joinpath(
                directory,
                "invalid-$label-version.checkpoint",
            )
            open(invalid_version_path, "w") do io
                serialize(
                    io,
                    merge(raw_payload, (; format_version=invalid_version)),
                )
            end
            @test_throws ArgumentError load_checkpoint(invalid_version_path)
        end

        checkpoint = load_checkpoint(path; backend=:zygote)
        @test checkpoint.format_version == CHECKPOINT_FORMAT_VERSION
        @test gpt_config(checkpoint.model) == gpt_config(model)
        @test checkpoint.tokenizer.id_to_token == tokenizer.id_to_token
        @test checkpoint.tokenizer.unk_id == tokenizer.unk_id
        @test checkpoint.train_state.step == interrupted_state.step
        @test checkpoint.progress == (;
            epoch=1,
            batch=1,
            step=interrupted_state.step,
        )
        @test checkpoint.train_config == (; seq_len=6, batch_size=4)
        @test checkpoint.metadata == (; purpose="test")
        @test _tree_isapprox(
            checkpoint.train_state.optimizer_state,
            interrupted_state.optimizer_state,
        )

        logits_after, _ = checkpoint.model(
            x,
            checkpoint.train_state.parameters,
            checkpoint.train_state.states,
        )
        @test isapprox(logits_after, logits_before; atol=1.0f-6, rtol=1.0f-5)

        resumed_state, resumed_losses = resume_gpt!(
            checkpoint,
            loader;
            epochs=1,
            max_steps=1,
        )

        @test resumed_state.step == continuous_state.step
        @test length(resumed_losses) == 1
        @test first(resumed_losses) ≈ Float32(continuous_loss) atol=1.0f-6
        @test _tree_isapprox(
            resumed_state.parameters,
            continuous_state.parameters;
            atol=2.0f-6,
            rtol=2.0f-5,
        )
        @test _tree_isapprox(
            resumed_state.optimizer_state,
            continuous_state.optimizer_state;
            atol=2.0f-6,
            rtol=2.0f-5,
        )

        completed_events = NamedTuple[]
        completed_checkpoint = merge(
            checkpoint,
            (;
                progress=(;
                    epoch=1,
                    batch=length(loader),
                    step=checkpoint.train_state.step,
                ),
            ),
        )
        completed_state, completed_losses = resume_gpt!(
            completed_checkpoint,
            loader;
            epochs=1,
            max_steps=1,
            callback=event -> push!(completed_events, event),
        )
        @test completed_state.step == checkpoint.train_state.step + 1
        @test length(completed_losses) == 1
        @test length(completed_events) == 1
        @test only(completed_events).epoch == 2
        @test only(completed_events).batch == 1

        progress_past_loader = merge(
            checkpoint,
            (;
                progress=(;
                    epoch=1,
                    batch=length(loader) + 1,
                    step=checkpoint.train_state.step,
                ),
            ),
        )
        @test_throws ArgumentError resume_gpt!(progress_past_loader, loader)
    end
end

@testset "KV cache correctness matrix and benchmark schema" begin
    rng = Xoshiro(77)
    model = GPTModel(17, 16, 2, 2; max_seq_len=8, use_rope=true)
    ps, st = Lux.setup(rng, model)

    correctness = kv_cache_correctness(
        model,
        ps,
        st,
        [1, 3, 5],
        [7, 9],
    )
    @test correctness.passed
    @test correctness.prefill.dynamic_passed
    @test correctness.prefill.static_passed
    @test correctness.decode.dynamic_passed
    @test correctness.decode.static_passed

    report = benchmark_kv_cache(
        model,
        ps,
        st,
        [1, 3, 5],
        [7, 9];
        samples=1,
    )
    @test report.configuration.prompt_tokens == 3
    @test report.configuration.decode_tokens == 2
    @test report.dynamic.theoretical_cache_bytes > 0
    @test report.static.theoretical_cache_bytes > report.dynamic.theoretical_cache_bytes
    @test report.eager.steady.prefill_seconds >= 0
    @test report.dynamic.steady.decode_seconds >= 0
    @test report.static.steady.decode_seconds >= 0
    @test length(report.eager.samples) == 1
    @test length(report.dynamic.samples) == 1
    @test length(report.static.samples) == 1

    @test_throws ArgumentError benchmark_xla_cache_modes(
        model,
        ps,
        st,
        [1, 3, 5],
        Int[];
        xla_backend="cpu",
        samples=1,
    )
end
