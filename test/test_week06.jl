using Test
using Random
using Lux
using Optimisers
using Zygote
import LifeAI
using LifeAI:
    GPTModel,
    TrainerGPT,
    batched_scaled_dot_product_attention,
    decode,
    encode,
    fit_tokenizer,
    generate_cached,
    gpt_config,
    init_kv_cache,
    init_static_kv_cache,
    init_train_state,
    kv_cache_correctness,
    load_checkpoint,
    manual_scaled_dot_product_attention,
    prefill,
    repeat_kv,
    resume_gpt!,
    save_checkpoint,
    train_step!,
    vocab_size
const _W06MHA = LifeAI.MultiHeadAttention

function _week06_gqa_model(;
    vocab_size=61,
    num_kv_heads=2,
    use_qk_norm=true,
    kwargs...,
)
    return GPTModel(
        vocab_size,
        32,
        4,
        2;
        num_kv_heads,
        use_qk_norm,
        use_rope=true,
        rope_theta=1.0f6,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
        max_seq_len=64,
        kwargs...,
    )
end

@testset "GQA attention equivalence" begin
    D, T, B = 8, 6, 2

    for (H, Hk) in ((4, 4), (4, 2), (4, 1), (8, 2), (6, 3), (8, 1))
        rng = MersenneTwister(60_000 + 10 * H + Hk)
        q = randn(rng, Float32, D, H, T, B)
        k = randn(rng, Float32, D, Hk, T, B)
        v = randn(rng, Float32, D, Hk, T, B)
        groups = H ÷ Hk

        for is_causal in (true, false)
            manual_context, manual_attn = manual_scaled_dot_product_attention(
                q, k, v; is_causal,
            )
            batched_context, batched_attn = batched_scaled_dot_product_attention(
                q, k, v; is_causal,
            )
            # Reference semantics: contiguous repeat_kv expansion followed by
            # the unmodified full-head kernel.
            repeat_context, repeat_attn = batched_scaled_dot_product_attention(
                q, repeat_kv(k, groups), repeat_kv(v, groups); is_causal,
            )

            @test isapprox(manual_context, batched_context; atol=1.0f-5)
            @test isapprox(manual_attn, batched_attn; atol=1.0f-5)
            @test isapprox(batched_context, repeat_context; atol=1.0f-5)
            @test isapprox(batched_attn, repeat_attn; atol=1.0f-5)
        end
    end

    # Query head h must read KV head (h - 1) ÷ groups + 1: give each KV head a
    # distinct constant signature and check which one every query head sees.
    let H = 4, Hk = 2, groups = H ÷ Hk
        q = zeros(Float32, D, H, 1, 1)
        k = zeros(Float32, D, Hk, 1, 1)
        v = zeros(Float32, D, Hk, 1, 1)
        for kv in 1:Hk
            v[:, kv, 1, 1] .= Float32(kv)
        end
        context, _ = batched_scaled_dot_product_attention(q, k, v)
        for h in 1:H
            expected = Float32((h - 1) ÷ groups + 1)
            @test all(context[:, h, 1, 1] .== expected)
        end
    end

    @test repeat_kv(randn(Float32, D, 2, T, B), 1) isa Array{Float32,4}
    @test size(repeat_kv(randn(Float32, D, 2, T, B), 3), 2) == 6
    @test_throws ArgumentError repeat_kv(randn(Float32, D, 2, T, B), 0)
end

@testset "GQA layer construction and parameters" begin
    attn = _W06MHA(16, 4; num_kv_heads=2, head_dim=8)
    @test attn.num_kv_heads == 2
    @test attn.kv_dim == 16
    @test attn.d_out == 32

    ps, _ = Lux.setup(MersenneTwister(1), attn)
    @test size(ps.q_proj.weight) == (32, 16)
    @test size(ps.k_proj.weight) == (16, 16)
    @test size(ps.v_proj.weight) == (16, 16)
    @test !haskey(ps, :q_norm)

    # Invalid grouping fails immediately.
    @test_throws AssertionError _W06MHA(16, 4; num_kv_heads=3)
    @test_throws AssertionError _W06MHA(16, 4; num_kv_heads=0)
    @test_throws AssertionError GPTModel(61, 32, 4, 2; num_kv_heads=3)
    @test_throws AssertionError _W06MHA(16, 4; use_qk_norm=true, qk_norm_epsilon=0)

    # Default construction reproduces the legacy layout and parameter tree.
    legacy = _W06MHA(16, 4)
    @test legacy.num_kv_heads == 4
    @test legacy.use_qk_norm == false
    legacy_ps, _ = Lux.setup(MersenneTwister(1), legacy)
    @test keys(legacy_ps) == (:q_proj, :k_proj, :v_proj, :o_proj)

    # The legacy RNG stream is untouched. These values were captured from the
    # pre-Week-06 implementation (verified elementwise identical against the
    # old `MultiHeadAttention` under the same seed) and pin the stream forward.
    @test legacy_ps.q_proj.weight[1:3] ≈
        Float32[0.34169587, 0.39485088, 0.2599711] atol = 1.0f-6
    @test sum(legacy_ps.q_proj.weight) ≈ 3.7512083f0 atol = 1.0f-4
    @test sum(legacy_ps.o_proj.weight) ≈ 2.6697168f0 atol = 1.0f-4
end

@testset "QK-Norm semantics" begin
    D, H, T, B = 8, 4, 5, 2
    rng = MersenneTwister(61_000)
    x = randn(rng, Float32, D, H, T, B)
    scale = randn(rng, Float32, D)
    epsilon = 1.0f-6

    normed = LifeAI._apply_qk_norm(x, scale, epsilon)

    # Hand-written per-head RMSNorm reference over the head dimension.
    expected = similar(x)
    for b in 1:B, t in 1:T, h in 1:H
        column = x[:, h, t, b]
        inv_rms = 1.0f0 / sqrt(sum(abs2, column) / D + epsilon)
        expected[:, h, t, b] = column .* inv_rms .* scale
    end
    @test isapprox(normed, expected; atol=1.0f-6)

    # Enabled QK-norm inserts unit-scale normalization ahead of RoPE, so it
    # must change the outputs of an otherwise identical layer.
    base = _W06MHA(16, 4; num_kv_heads=2, use_rope=true, max_seq_len=32)
    with_norm = _W06MHA(
        16, 4;
        num_kv_heads=2, use_rope=true, use_qk_norm=true, max_seq_len=32,
    )
    base_ps, base_st = Lux.setup(MersenneTwister(2), base)
    norm_ps, norm_st = Lux.setup(MersenneTwister(2), with_norm)
    @test haskey(norm_ps, :q_norm) && haskey(norm_ps, :k_norm)
    @test size(norm_ps.q_norm.scale) == (4,)
    @test base_ps.q_proj.weight == norm_ps.q_proj.weight

    input = randn(MersenneTwister(3), Float32, 16, 6, 2)
    base_out, _ = base(input, base_ps, base_st)
    norm_out, _ = with_norm(input, norm_ps, norm_st)
    @test !isapprox(base_out, norm_out; atol=1.0f-3)
    @test all(isfinite, norm_out)
end

@testset "GQA + QK-Norm cache correctness matrix" begin
    configurations = (
        (; label="gqa", num_kv_heads=2, use_qk_norm=false),
        (; label="gqa_qk_norm", num_kv_heads=2, use_qk_norm=true),
        (; label="mqa_qk_norm", num_kv_heads=1, use_qk_norm=true),
    )

    for config in configurations
        model = _week06_gqa_model(;
            num_kv_heads=config.num_kv_heads,
            use_qk_norm=config.use_qk_norm,
        )
        ps, st = Lux.setup(MersenneTwister(62_000 + config.num_kv_heads), model)

        rng = MersenneTwister(62_100)
        prompt = rand(rng, 1:model.vocab_size, 7)
        decode_tokens = rand(rng, 1:model.vocab_size, 5)

        report = kv_cache_correctness(model, ps, st, prompt, decode_tokens)
        @test report.passed

        # Both cache flavors must store exactly num_kv_heads heads.
        cache = init_kv_cache(model; batch_size=1)
        _, cache, _ = prefill(model, ps, st, prompt, cache)
        @test size(cache.layers[1].keys, 2) == config.num_kv_heads

        static_cache = init_static_kv_cache(model; batch_size=1)
        @test size(static_cache.layers[1].keys, 2) == config.num_kv_heads
        @test size(static_cache.layers[1].keys, 3) == model.max_seq_len
    end
end

@testset "GQA reduces parameters and cache memory" begin
    full = _week06_gqa_model(; num_kv_heads=4, use_qk_norm=false)
    grouped = _week06_gqa_model(; num_kv_heads=2, use_qk_norm=false)

    full_ps, _ = Lux.setup(MersenneTwister(1), full)
    grouped_ps, _ = Lux.setup(MersenneTwister(1), grouped)

    @test LuxCore.parameterlength(grouped_ps) < LuxCore.parameterlength(full_ps)

    full_cache = init_static_kv_cache(full; batch_size=1)
    grouped_cache = init_static_kv_cache(grouped; batch_size=1)
    @test length(grouped_cache.layers[1].keys) * 2 ==
        length(full_cache.layers[1].keys)
end

@testset "Config round-trip and legacy compatibility" begin
    model = _week06_gqa_model()
    config = gpt_config(model)
    @test config.num_kv_heads == 2
    @test config.use_qk_norm == true
    @test config.qk_norm_epsilon == 1.0f-6

    rebuilt = GPTModel(config)
    @test gpt_config(rebuilt) == config

    # A rebuilt model reproduces logits exactly with the same parameters.
    ps, st = Lux.setup(MersenneTwister(63_000), model)
    tokens = rand(MersenneTwister(63_001), 1:model.vocab_size, 9, 2)
    logits, _ = model(tokens, ps, st)
    rebuilt_logits, _ = rebuilt(tokens, ps, st)
    @test logits == rebuilt_logits

    # Pre-Week-06 configs carry none of the new fields and must default to the
    # exact legacy architecture.
    legacy_model = GPTModel(61, 32, 4, 2; use_rope=true)
    legacy_config = Base.structdiff(
        gpt_config(legacy_model),
        NamedTuple{(:num_kv_heads, :use_qk_norm, :qk_norm_epsilon)},
    )
    @test !hasproperty(legacy_config, :num_kv_heads)
    from_legacy = GPTModel(legacy_config)
    @test from_legacy.num_kv_heads == 4
    @test from_legacy.use_qk_norm == false

    legacy_ps, legacy_st = Lux.setup(MersenneTwister(63_002), legacy_model)
    legacy_tokens = rand(MersenneTwister(63_003), 1:61, 9, 2)
    expected_logits, _ = legacy_model(legacy_tokens, legacy_ps, legacy_st)
    migrated_logits, _ = from_legacy(legacy_tokens, legacy_ps, legacy_st)
    @test expected_logits == migrated_logits
end

@testset "GQA training, checkpoint, resume, generation" begin
    corpus = "青山不老，绿水长流。合抱之木生于毫末，千里之行始于足下。"
    tokenizer = fit_tokenizer(corpus)
    model = _week06_gqa_model(; vocab_size=vocab_size(tokenizer))

    trainer = TrainerGPT(; optimizer=Adam(1.0f-3))
    state = init_train_state(MersenneTwister(64_000), model, trainer)

    token_stream = encode(tokenizer, corpus)
    seq_len, batch_size = 8, 2
    windows = [
        token_stream[i:(i + seq_len)] for
        i in 1:(length(token_stream) - seq_len)
    ]
    x = reduce(hcat, [w[1:seq_len] for w in windows[1:batch_size]])
    y = reduce(hcat, [w[2:(seq_len + 1)] for w in windows[1:batch_size]])

    losses = Float32[]
    for _ in 1:6
        state, loss, _ = train_step!(trainer, state, (x, y))
        push!(losses, Float32(loss))
    end
    @test all(isfinite, losses)
    @test losses[end] < losses[1]

    checkpoint_dir = mktempdir()
    checkpoint_path = joinpath(checkpoint_dir, "week06_gqa.checkpoint")
    save_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        trainer,
        state;
        progress=(; epoch=1, batch=6),
    )

    restored = load_checkpoint(checkpoint_path)
    @test restored.model.num_kv_heads == 2
    @test restored.model.use_qk_norm == true
    @test gpt_config(restored.model) == gpt_config(model)

    probe = rand(MersenneTwister(64_001), 1:model.vocab_size, 9, 1)
    original_logits, _ = model(probe, state.parameters, state.states)
    restored_logits, _ = restored.model(
        probe,
        restored.train_state.parameters,
        restored.train_state.states,
    )
    @test isapprox(original_logits, restored_logits; atol=1.0f-6)

    # Resumed training continues from the recorded step.
    @test restored.train_state.step == 6
    resumed_state, _ = train_step!(
        restored.trainer, restored.train_state, (x, y),
    )
    @test resumed_state.step == 7

    # Cached generation runs through the GQA decode path end to end.
    prompt_tokens = encode(tokenizer, "青山")
    generated, _ = generate_cached(
        model,
        state.parameters,
        state.states,
        prompt_tokens;
        max_new_tokens=8,
        temperature=1.0f0,
        rng=MersenneTwister(64_002),
    )
    @test length(generated) == length(prompt_tokens) + 8
    @test all(id -> 1 <= id <= model.vocab_size, generated)
    @test decode(tokenizer, generated) isa String
end
