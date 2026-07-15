using Test
using Random
using Lux
using LifeAI:
    CHECKPOINT_FORMAT_VERSION,
    DatasetLoader,
    GPTModel,
    TrainerGPT,
    benchmark_kv_cache,
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
        path = joinpath(directory, "week03.checkpoint")
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
end
