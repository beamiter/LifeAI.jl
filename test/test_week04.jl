using Test
using Random
using Lux
using NNlib: swish
using Serialization
using Zygote
using LifeAI:
    CHECKPOINT_FORMAT_VERSION,
    DatasetLoader,
    GPTModel,
    RMSNormLayer,
    SwiGLU,
    TiedOutputProjection,
    TrainerGPT,
    decode,
    encode,
    fit_tokenizer,
    generate_cached,
    gpt_config,
    init_train_state,
    kv_cache_correctness,
    load_checkpoint,
    resume_gpt!,
    save_checkpoint,
    train_step!,
    vocab_size

_week04_all_finite(::Nothing) = true
_week04_all_finite(x::Number) = isfinite(x)
_week04_all_finite(x::AbstractArray) = all(isfinite, x)
_week04_all_finite(x::NamedTuple) = all(_week04_all_finite, values(x))
_week04_all_finite(x::Tuple) = all(_week04_all_finite, x)
_week04_all_finite(_) = true

function _week04_tree_isapprox(left::NamedTuple, right::NamedTuple; kwargs...)
    keys(left) == keys(right) || return false
    return all(
        _week04_tree_isapprox(l, r; kwargs...) for
        (l, r) in zip(values(left), values(right))
    )
end

function _week04_tree_isapprox(left::Tuple, right::Tuple; kwargs...)
    length(left) == length(right) || return false
    return all(
        _week04_tree_isapprox(l, r; kwargs...) for
        (l, r) in zip(left, right)
    )
end

function _week04_tree_isapprox(left::AbstractArray, right::AbstractArray; kwargs...)
    return isapprox(left, right; kwargs...)
end

_week04_tree_isapprox(left, right; kwargs...) = left == right

function _week04_dense_reference(ps, x)
    output = ps.weight * reshape(x, size(x, 1), :)
    if hasproperty(ps, :bias)
        output = output .+ ps.bias
    end
    return reshape(output, size(ps.weight, 1), size(x, 2), size(x, 3))
end

function _week04_legacy_config(config)
    return (;
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        head_dim=config.head_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        use_bias=config.use_bias,
        is_causal=config.is_causal,
        use_rope=config.use_rope,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        norm_epsilon=config.norm_epsilon,
    )
end

@testset "RMSNorm reference, dtype, and gradients" begin
    rng = Xoshiro(20260718)
    norm = RMSNormLayer(4; epsilon=2.0f-5)
    ps, st = Lux.setup(rng, norm)
    x = reshape(
        Float32[
            1, -2, 3, -4,
            2, 4, -6, 8,
            -1, 3, 5, -7,
            4, -3, 2, -1,
        ],
        4,
        2,
        2,
    )

    y, st_new = norm(x, ps, st)
    mean_square = sum(abs2, x; dims=1) ./ Float32(size(x, 1))
    reference = x .* (1.0f0 ./ sqrt.(mean_square .+ norm.epsilon)) .* ps.scale

    @test size(y) == size(x)
    @test eltype(y) == Float32
    @test y ≈ reference atol=1.0f-6
    @test st_new == st
    @test size(ps.scale) == (4, 1, 1)
    @test Lux.parameterlength(norm) == 4

    grad_x, grad_ps = Zygote.gradient(
        (input, parameters) -> sum(abs2, first(norm(input, parameters, st))),
        x,
        ps,
    )
    @test _week04_all_finite(grad_x)
    @test _week04_all_finite(grad_ps)
    @test sum(abs, grad_ps.scale) > 0

    @test_throws AssertionError RMSNormLayer(0)
    @test_throws AssertionError RMSNormLayer(4; epsilon=0)
    @test_throws AssertionError norm(randn(rng, Float32, 4, 2), ps, st)
    @test_throws AssertionError norm(randn(rng, Float32, 5, 2, 1), ps, st)
end

@testset "SwiGLU reference, width, parameters, and gradients" begin
    rng = Xoshiro(20260719)
    mlp = SwiGLU(4, 7; use_bias=true)
    ps, st = Lux.setup(rng, mlp)
    x = randn(rng, Float32, 4, 3, 2)

    y, st_new = mlp(x, ps, st)
    gate = _week04_dense_reference(ps.gate_proj, x)
    up = _week04_dense_reference(ps.up_proj, x)
    hidden = swish.(gate) .* up
    reference = _week04_dense_reference(ps.down_proj, hidden)

    @test size(y) == size(x)
    @test y ≈ reference atol=1.0f-6
    @test keys(ps) == (:gate_proj, :up_proj, :down_proj)
    @test keys(st_new) == (:gate_proj, :up_proj, :down_proj)
    @test Lux.parameterlength(mlp) == 3 * 4 * 7 + 2 * 7 + 4

    grad_x, grad_ps = Zygote.gradient(
        (input, parameters) -> sum(abs2, first(mlp(input, parameters, st))),
        x,
        ps,
    )
    @test _week04_all_finite(grad_x)
    @test _week04_all_finite(grad_ps)
    @test sum(abs, grad_ps.gate_proj.weight) > 0
    @test sum(abs, grad_ps.up_proj.weight) > 0
    @test sum(abs, grad_ps.down_proj.weight) > 0

    @test_throws AssertionError SwiGLU(0, 7)
    @test_throws AssertionError SwiGLU(4, 0)

    default_swiglu = GPTModel(13, 16, 2, 1; mlp_type=:swiglu)
    explicit_swiglu = GPTModel(
        13,
        16,
        2,
        1;
        mlp_type=:swiglu,
        mlp_hidden_dim=31,
    )
    @test default_swiglu.mlp_hidden_dim == round(Int, 8 * 16 / 3)
    @test explicit_swiglu.mlp_hidden_dim == 31
end

@testset "Tied output projection and shared gradient" begin
    rng = Xoshiro(20260720)
    head = TiedOutputProjection(7, 5; use_bias=true)
    ps, st = Lux.setup(rng, head)
    ps = (; bias=randn(rng, Float32, 7))
    hidden = randn(rng, Float32, 5, 3, 2)
    embedding = randn(rng, Float32, 5, 7)

    logits, st_new = head((hidden, embedding), ps, st)
    reference = reshape(
        transpose(embedding) * reshape(hidden, 5, :),
        7,
        3,
        2,
    ) .+ reshape(ps.bias, :, 1, 1)

    @test logits ≈ reference atol=1.0f-6
    @test st_new == st
    @test Lux.parameterlength(head) == 7
    @test_throws AssertionError head(
        (randn(rng, Float32, 6, 3, 2), embedding),
        ps,
        st,
    )
    @test_throws AssertionError head(
        (hidden, randn(rng, Float32, 6, 7)),
        ps,
        st,
    )

    vocab_size, d_model = 11, 8
    untied = GPTModel(
        vocab_size,
        d_model,
        2,
        1;
        max_seq_len=6,
        use_bias=true,
        tie_embeddings=false,
    )
    tied = GPTModel(
        vocab_size,
        d_model,
        2,
        1;
        max_seq_len=6,
        use_bias=true,
        tie_embeddings=true,
    )
    untied_ps, _ = Lux.setup(Xoshiro(1), untied)
    tied_ps, _ = Lux.setup(Xoshiro(1), tied)

    @test Lux.parameterlength(untied_ps) - Lux.parameterlength(tied_ps) ==
        vocab_size * d_model
    @test !haskey(tied_ps.lm_head, :weight)
    @test haskey(tied_ps.lm_head, :bias)

    shared_head = TiedOutputProjection(vocab_size, d_model)
    shared_ps, shared_st = Lux.setup(Xoshiro(2), shared_head)
    shared_embedding = randn(rng, Float32, d_model, vocab_size)
    tokens = [1, 3, 1, 5]
    fixed_hidden = reshape(shared_embedding[:, tokens], d_model, :, 1)
    fixed_output_weight = copy(shared_embedding)

    shared_objective = weight -> begin
        input_hidden = reshape(weight[:, tokens], d_model, :, 1)
        output, _ = shared_head((input_hidden, weight), shared_ps, shared_st)
        return sum(abs2, output)
    end
    output_objective = weight -> begin
        output, _ = shared_head((fixed_hidden, weight), shared_ps, shared_st)
        return sum(abs2, output)
    end
    input_objective = weight -> begin
        input_hidden = reshape(weight[:, tokens], d_model, :, 1)
        output, _ = shared_head(
            (input_hidden, fixed_output_weight),
            shared_ps,
            shared_st,
        )
        return sum(abs2, output)
    end

    shared_gradient = only(Zygote.gradient(shared_objective, shared_embedding))
    output_gradient = only(Zygote.gradient(output_objective, shared_embedding))
    input_gradient = only(Zygote.gradient(input_objective, shared_embedding))

    @test shared_gradient ≈ output_gradient .+ input_gradient atol=2.0f-5 rtol=2.0f-4
    @test sum(abs, output_gradient) > 0
    @test sum(abs, input_gradient) > 0
end

@testset "Legacy defaults and version-stable GPT config" begin
    default_model = GPTModel(
        19,
        16,
        2,
        2;
        max_seq_len=8,
        use_bias=true,
        use_rope=true,
    )
    explicit_legacy = GPTModel(
        19,
        16,
        2,
        2;
        max_seq_len=8,
        use_bias=true,
        use_rope=true,
        norm_type=:layernorm,
        mlp_type=:gelu,
        tie_embeddings=false,
        mlp_hidden_dim=64,
    )

    default_ps, default_st = Lux.setup(Xoshiro(77), default_model)
    explicit_ps, explicit_st = Lux.setup(Xoshiro(77), explicit_legacy)
    tokens = reshape([1, 3, 5, 7], 4, 1)
    default_logits, _ = default_model(tokens, default_ps, default_st)
    explicit_logits, _ = explicit_legacy(tokens, explicit_ps, explicit_st)

    @test default_model.norm_type == :layernorm
    @test default_model.mlp_type == :gelu
    @test !default_model.tie_embeddings
    @test _week04_tree_isapprox(default_ps, explicit_ps; atol=0, rtol=0)
    @test _week04_tree_isapprox(default_st, explicit_st; atol=0, rtol=0)
    @test default_logits == explicit_logits

    modern = GPTModel(
        19,
        16,
        2,
        2;
        max_seq_len=8,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
    )
    rebuilt = GPTModel(gpt_config(modern))
    @test gpt_config(rebuilt) == gpt_config(modern)

    rebuilt_legacy = GPTModel(_week04_legacy_config(gpt_config(default_model)))
    @test rebuilt_legacy.norm_type == :layernorm
    @test rebuilt_legacy.mlp_type == :gelu
    @test !rebuilt_legacy.tie_embeddings
    @test gpt_config(rebuilt_legacy) == gpt_config(default_model)

    @test_throws ArgumentError GPTModel(19, 16, 2, 1; norm_type=:unknown)
    @test_throws ArgumentError GPTModel(19, 16, 2, 1; mlp_type=:unknown)
end

@testset "Five-configuration train and KV-cache correctness matrix" begin
    configurations = (
        baseline=(;),
        rmsnorm_only=(; norm_type=:rmsnorm),
        swiglu_only=(; mlp_type=:swiglu),
        tied_only=(; tie_embeddings=true),
        modern=(;
            norm_type=:rmsnorm,
            mlp_type=:swiglu,
            tie_embeddings=true,
        ),
    )

    for (index, (name, options)) in enumerate(pairs(configurations))
        @testset "$(name)" begin
            model = GPTModel(
                17,
                16,
                2,
                2;
                max_seq_len=8,
                use_rope=true,
                options...,
            )
            ps, st = Lux.setup(Xoshiro(100 + index), model)
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

            trainer = TrainerGPT(
                learning_rate=1.0f-3,
                max_grad_norm=1.0f0,
            )
            train_state = init_train_state(Xoshiro(200 + index), model, trainer)
            x = [1 2; 3 4; 5 6; 7 8]
            targets = [3 4; 5 6; 7 8; 9 10]
            train_state, loss, gradients = train_step!(
                trainer,
                train_state,
                x,
                targets,
            )

            @test isfinite(loss)
            @test train_state.step == 1
            @test _week04_all_finite(gradients)
        end
    end
end

@testset "Modern checkpoint round-trip, resume, generation, and v1 migration" begin
    text = repeat("abcde", 24)
    tokenizer = fit_tokenizer(text; add_unk=true)
    loader = DatasetLoader(
        tokenizer,
        text;
        seq_len=6,
        batch_size=4,
        stride=2,
        drop_last=true,
    )
    modern = GPTModel(
        vocab_size(tokenizer),
        12,
        3,
        2;
        max_seq_len=12,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
        use_bias=true,
    )
    trainer = TrainerGPT(learning_rate=2.0f-3, max_grad_norm=1.0f0)
    train_state = init_train_state(Xoshiro(300), modern, trainer)
    train_state, _, _ = train_step!(trainer, train_state, loader[1])
    input_tokens, _ = loader[1]
    logits_before, _ = modern(
        input_tokens,
        train_state.parameters,
        train_state.states,
    )

    mktempdir() do directory
        modern_path = joinpath(directory, "modern-v2.checkpoint")
        save_checkpoint(
            modern_path,
            modern,
            tokenizer,
            trainer,
            train_state;
            rng=Xoshiro(301),
            progress=(; epoch=1, batch=1),
            metadata=(; week=4),
        )

        checkpoint = load_checkpoint(modern_path; backend=:zygote)
        @test checkpoint.format_version == CHECKPOINT_FORMAT_VERSION == 2
        @test checkpoint.source_format_version == 2
        @test checkpoint.model.norm_type == :rmsnorm
        @test checkpoint.model.mlp_type == :swiglu
        @test checkpoint.model.tie_embeddings
        @test checkpoint.metadata == (; week=4)

        logits_after, _ = checkpoint.model(
            input_tokens,
            checkpoint.train_state.parameters,
            checkpoint.train_state.states,
        )
        @test logits_after ≈ logits_before atol=1.0f-6 rtol=1.0f-5

        resumed_state, losses = resume_gpt!(
            checkpoint,
            loader;
            epochs=1,
            max_steps=1,
        )
        @test resumed_state.step == train_state.step + 1
        @test length(losses) == 1
        @test all(isfinite, losses)

        generated, _ = generate_cached(
            checkpoint.model,
            resumed_state.parameters,
            resumed_state.states,
            encode(checkpoint.tokenizer, "a");
            max_new_tokens=4,
            temperature=0,
            rng=Xoshiro(302),
        )
        @test length(generated) == 5
        @test !isempty(decode(checkpoint.tokenizer, generated))

        legacy_model = GPTModel(
            vocab_size(tokenizer),
            12,
            3,
            2;
            max_seq_len=12,
            use_bias=true,
        )
        legacy_state = init_train_state(Xoshiro(303), legacy_model, trainer)
        legacy_v2_path = joinpath(directory, "legacy-v2.checkpoint")
        save_checkpoint(
            legacy_v2_path,
            legacy_model,
            tokenizer,
            trainer,
            legacy_state,
        )
        raw_payload = open(legacy_v2_path, "r") do io
            deserialize(io)
        end
        legacy_payload = merge(
            raw_payload,
            (;
                format_version=1,
                model_config=_week04_legacy_config(raw_payload.model_config),
            ),
        )
        legacy_v1_path = joinpath(directory, "legacy-v1.checkpoint")
        open(legacy_v1_path, "w") do io
            serialize(io, legacy_payload)
        end

        migrated = load_checkpoint(legacy_v1_path; backend=:zygote)
        @test migrated.format_version == 2
        @test migrated.source_format_version == 1
        @test migrated.model.norm_type == :layernorm
        @test migrated.model.mlp_type == :gelu
        @test !migrated.model.tie_embeddings
        @test gpt_config(migrated.model) == gpt_config(legacy_model)

        legacy_tokens = reshape([1, 2, 3], 3, 1)
        original_logits, _ = legacy_model(
            legacy_tokens,
            legacy_state.parameters,
            legacy_state.states,
        )
        migrated_logits, _ = migrated.model(
            legacy_tokens,
            migrated.train_state.parameters,
            migrated.train_state.states,
        )
        @test migrated_logits ≈ original_logits atol=1.0f-6 rtol=1.0f-5

        unsupported_path = joinpath(directory, "unsupported.checkpoint")
        open(unsupported_path, "w") do io
            serialize(io, merge(raw_payload, (; format_version=99)))
        end
        @test_throws ArgumentError load_checkpoint(unsupported_path)
    end
end
