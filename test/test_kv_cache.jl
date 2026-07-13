using Test
using Random
using Lux
using LifeAI:
    GPTModel,
    decode_step,
    generate,
    generate_cached,
    init_kv_cache,
    init_static_kv_cache,
    prefill

@testset "KV cache prefill and incremental decode" begin
    rng = Xoshiro(20260713)
    model = GPTModel(
        13,
        16,
        2,
        2;
        max_seq_len=8,
        use_rope=true,
    )
    ps, st = Lux.setup(rng, model)

    prompt = reshape([1, 3, 5, 7], 4, 1)
    full_logits, _ = model(prompt, ps, st)

    cache = init_kv_cache(model; batch_size=1)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)

    @test isapprox(cached_logits, full_logits; atol=1.0f-5, rtol=1.0f-4)
    @test length(cache) == size(prompt, 1)
    @test all(layer_cache -> length(layer_cache) == length(cache), cache.layers)

    next_token = 9
    step_logits, cache, cached_state = decode_step(
        model,
        ps,
        cached_state,
        next_token,
        cache,
    )

    extended_prompt = vcat(prompt, reshape([next_token], 1, 1))
    extended_logits, _ = model(extended_prompt, ps, st)

    @test size(step_logits) == (model.vocab_size, 1, 1)
    @test isapprox(
        vec(step_logits[:, 1, 1]),
        vec(extended_logits[:, end, 1]);
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test length(cache) == size(extended_prompt, 1)
end


@testset "Batched KV cache matches full forward" begin
    rng = Xoshiro(23)
    model = GPTModel(19, 16, 2, 2; max_seq_len=6, use_rope=true)
    ps, st = Lux.setup(rng, model)
    prompt = [1 2; 3 4; 5 6]

    full_logits, _ = model(prompt, ps, st)
    cache = init_kv_cache(model; batch_size=2)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)

    @test isapprox(cached_logits, full_logits; atol=1.0f-5, rtol=1.0f-4)

    next_tokens = [7, 8]
    step_logits, cache, _ = decode_step(
        model,
        ps,
        cached_state,
        next_tokens,
        cache,
    )
    extended_prompt = vcat(prompt, reshape(next_tokens, 1, 2))
    extended_logits, _ = model(extended_prompt, ps, st)

    @test size(step_logits) == (model.vocab_size, 1, 2)
    @test isapprox(
        step_logits[:, 1, :],
        extended_logits[:, end, :];
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test length(cache) == 4
end

@testset "KV-cached greedy generation matches eager generation" begin
    rng = Xoshiro(7)
    model = GPTModel(
        17,
        24,
        3,
        2;
        max_seq_len=12,
        use_rope=true,
    )
    ps, st = Lux.setup(rng, model)
    prompt = [2, 4, 6]

    eager_tokens, _ = generate(
        model,
        ps,
        st,
        prompt;
        max_new_tokens=5,
        temperature=0,
    )
    cached_tokens, _ = generate_cached(
        model,
        ps,
        st,
        prompt;
        max_new_tokens=5,
        temperature=0,
    )

    @test cached_tokens == eager_tokens
end

@testset "KV cache validation" begin
    rng = Xoshiro(11)
    model = GPTModel(9, 8, 2, 1; max_seq_len=3, use_rope=true)
    ps, st = Lux.setup(rng, model)
    empty_cache = init_kv_cache(model)

    @test_throws ArgumentError decode_step(model, ps, st, 1, empty_cache)

    _, full_cache, cached_state = prefill(model, ps, st, [1, 2, 3], empty_cache)
    @test_throws ArgumentError prefill(model, ps, cached_state, [1], full_cache)
    @test_throws ArgumentError decode_step(model, ps, cached_state, 4, full_cache)
    @test_throws ArgumentError generate_cached(
        model,
        ps,
        st,
        [1, 2];
        max_new_tokens=3,
        temperature=0,
    )
end


@testset "Static KV cache keeps fixed storage and matches full forward" begin
    rng = Xoshiro(20260714)
    model = GPTModel(23, 24, 3, 2; max_seq_len=10, use_rope=true)
    ps, st = Lux.setup(rng, model)
    prompt = reshape([1, 4, 7, 10], 4, 1)

    cache = init_static_kv_cache(model; batch_size=1)
    key_buffers = map(layer -> layer.keys, cache.layers)
    value_buffers = map(layer -> layer.values, cache.layers)

    full_logits, _ = model(prompt, ps, st)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)

    @test isapprox(cached_logits, full_logits; atol=1.0f-5, rtol=1.0f-4)
    @test length(cache) == size(prompt, 1)
    @test all(
        layer -> size(layer.keys, 3) == model.max_seq_len,
        cache.layers,
    )
    @test all(
        index -> cache.layers[index].keys === key_buffers[index],
        eachindex(cache.layers),
    )
    @test all(
        index -> cache.layers[index].values === value_buffers[index],
        eachindex(cache.layers),
    )

    generated_context = vec(prompt)
    for next_token in (13, 16, 19)
        step_logits, cache, cached_state = decode_step(
            model,
            ps,
            cached_state,
            next_token,
            cache,
        )
        push!(generated_context, next_token)
        reference_logits, _ = model(
            reshape(generated_context, :, 1),
            ps,
            st,
        )

        @test isapprox(
            vec(step_logits[:, 1, 1]),
            vec(reference_logits[:, end, 1]);
            atol=1.0f-5,
            rtol=1.0f-4,
        )
        @test length(cache) == length(generated_context)
        @test all(
            index -> cache.layers[index].keys === key_buffers[index],
            eachindex(cache.layers),
        )
    end
end

@testset "Batched static KV cache matches full forward" begin
    rng = Xoshiro(20260715)
    model = GPTModel(29, 16, 2, 2; max_seq_len=8, use_rope=true)
    ps, st = Lux.setup(rng, model)
    prompt = [1 2; 3 4; 5 6]

    cache = init_static_kv_cache(model; batch_size=2)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)
    reference_logits, _ = model(prompt, ps, st)

    @test isapprox(cached_logits, reference_logits; atol=1.0f-5, rtol=1.0f-4)

    next_tokens = [7, 8]
    step_logits, cache, _ = decode_step(
        model,
        ps,
        cached_state,
        next_tokens,
        cache,
    )
    extended_prompt = vcat(prompt, reshape(next_tokens, 1, 2))
    extended_logits, _ = model(extended_prompt, ps, st)

    @test isapprox(
        step_logits[:, 1, :],
        extended_logits[:, end, :];
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test length(cache) == 4
    @test all(layer -> size(layer.keys, 3) == 8, cache.layers)
end

@testset "Static KV cache validation" begin
    rng = Xoshiro(20260716)
    model = GPTModel(11, 8, 2, 1; max_seq_len=3, use_rope=true)
    ps, st = Lux.setup(rng, model)
    cache = init_static_kv_cache(model)

    @test_throws ArgumentError decode_step(model, ps, st, 1, cache)
    _, cache, cached_state = prefill(model, ps, st, [1, 2, 3], cache)
    @test_throws ArgumentError prefill(model, ps, cached_state, [1], cache)
    @test_throws ArgumentError decode_step(model, ps, cached_state, 4, cache)
end
