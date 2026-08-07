using Test
using Random
using LifeAI: device_sample_token,
    _device_sample_next_index,
    _validate_device_sampling_options,
    _validate_device_sampling_uniform,
    _sample_token,
    _sampling_distribution

@testset "device sampling option gate" begin
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=0, top_k=20, top_p=0.95)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=-1, top_k=20, top_p=0.95)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=Inf, top_k=20, top_p=0.95)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=NaN, top_k=20, top_p=0.95)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=0.6, top_k=0, top_p=0.95)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=0.6, top_k=2.5, top_p=0.95)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=0.6, top_k=20, top_p=0)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=0.6, top_k=20, top_p=1.5)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=0.6, top_k=20, top_p=NaN)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=0.6, top_k=20, top_p=0.95, vocab_size=10)

    options = _validate_device_sampling_options(;
        temperature=0.6, top_k=20, top_p=0.95, vocab_size=151936)
    @test options.temperature === 0.6f0
    @test options.top_k === 20
    @test options.top_p === 0.95f0

    @test _validate_device_sampling_uniform(0.0) === 0.0f0
    @test _validate_device_sampling_uniform(0.5) === 0.5f0
    @test_throws ArgumentError _validate_device_sampling_uniform(1.0)
    @test_throws ArgumentError _validate_device_sampling_uniform(-0.1)
    @test_throws ArgumentError _validate_device_sampling_uniform(NaN)

    @test_throws ArgumentError device_sample_token(
        Float32[], 0.5f0; temperature=1.0, top_k=1, top_p=1.0)
    @test_throws ArgumentError device_sample_token(
        Float32[1, Inf], 0.5f0; temperature=1.0, top_k=1, top_p=1.0)
    @test_throws ArgumentError device_sample_token(
        Float32[1, 2], 1.0f0; temperature=1.0, top_k=1, top_p=1.0)
end

@testset "device sampling degenerate policies" begin
    logits = Float32[0.5, 3.0, -1.0, 2.0]

    # top_k = 1 is greedy for every uniform.
    for uniform in (0.0f0, 0.25f0, 0.9999f0)
        @test device_sample_token(
            logits, uniform; temperature=0.7, top_k=1, top_p=1.0) == 2
    end

    # A very small nucleus also collapses onto the most likely token.
    @test device_sample_token(
        logits, 0.99f0; temperature=1.0, top_k=4, top_p=0.05) == 2

    # uniform = 0 always selects the lowest surviving vocabulary index.
    @test device_sample_token(
        logits, 0.0f0; temperature=1.0, top_k=4, top_p=1.0) == 1

    # top_k larger than the vocabulary is clamped, matching the host policy.
    @test device_sample_token(
        logits, 0.6f0; temperature=1.0, top_k=99, top_p=1.0) ==
        _sample_token(logits, Random.default_rng();
            temperature=1.0, top_k=4, top_p=1.0, sample_uniform=0.6f0)
end

@testset "device sampling matches the host policy without exact ties" begin
    rng = MersenneTwister(20260801)
    mismatches = 0
    trials = 0
    for _ in 1:1200
        vocab = rand(rng, 8:256)
        logits = randn(rng, Float32, vocab) .* Float32(rand(rng, 0.5:0.5:6.0))
        length(unique(logits)) == vocab || continue
        temperature = Float32(rand(rng, [0.1, 0.6, 1.0, 2.0]))
        top_k = min(rand(rng, [1, 2, 5, 20, 50, vocab]), vocab)
        top_p = Float32(rand(rng, [0.1, 0.5, 0.95, 1.0]))
        uniform = rand(rng, Float32)
        host = _sample_token(logits, rng;
            temperature, top_k, top_p, sample_uniform=uniform)
        device = device_sample_token(logits, uniform; temperature, top_k, top_p)
        trials += 1
        host == device || (mismatches += 1)
    end
    @test trials > 1000
    @test mismatches == 0
end

@testset "device sampling reproduces the host empirical distribution" begin
    rng = MersenneTwister(24)
    logits = Float32[2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0]
    _, probabilities = _sampling_distribution(logits;
        temperature=0.9, top_k=5, top_p=0.9)
    counts = zeros(Int, length(logits))
    draws = 4000
    mismatches = 0
    for _ in 1:draws
        uniform = rand(rng, Float32)
        host = _sample_token(logits, rng;
            temperature=0.9, top_k=5, top_p=0.9, sample_uniform=uniform)
        device = device_sample_token(logits, uniform;
            temperature=0.9, top_k=5, top_p=0.9)
        host == device || (mismatches += 1)
        counts[device] += 1
    end
    @test mismatches == 0
    empirical = counts ./ draws
    @test maximum(abs.(empirical .- Float32.(probabilities))) < 0.02
    # top-p removed the tail, so it must never be sampled.
    @test count(>(0), counts) == count(>(0), probabilities)
end

@testset "device sampling ranks exact ties like HuggingFace" begin
    # Two bit-identical maxima. The host sorts ascending and stably, then
    # protects the last entry, so index 3 outranks index 1; a nucleus tight
    # enough to keep a single token must therefore keep index 3. `findmax`
    # would have kept index 1 and silently sampled a different token.
    tied = Float32[4.0, 1.0, 4.0, 0.0]
    @test device_sample_token(tied, 0.5f0;
        temperature=1.0, top_k=4, top_p=0.4) == 3
    @test _sample_token(tied, Random.default_rng();
        temperature=1.0, top_k=4, top_p=0.4, sample_uniform=0.5f0) == 3

    # The documented contract difference: the host keeps every score tied with
    # the k-th largest, the device keeps exactly k candidates. With three tied
    # scores and top_k = 2 the host can still reach index 1, the device cannot.
    tie_at_k = Float32[3.0, 3.0, 3.0, -20.0]
    host_choices = Set(
        _sample_token(tie_at_k, Random.default_rng();
            temperature=1.0, top_k=2, top_p=1.0, sample_uniform=u)
        for u in 0.0f0:0.05f0:0.95f0
    )
    device_choices = Set(
        device_sample_token(tie_at_k, u;
            temperature=1.0, top_k=2, top_p=1.0)
        for u in 0.0f0:0.05f0:0.95f0
    )
    @test host_choices == Set([1, 2, 3])
    @test device_choices == Set([2, 3])
    @test issubset(device_choices, host_choices)
end

@testset "device sampling is deterministic and RNG compatible" begin
    logits = Float32[0.1, 2.0, 1.0, -3.0, 0.7, 1.9]
    for uniform in 0.0f0:0.1f0:0.9f0
        first_call = device_sample_token(logits, uniform;
            temperature=0.6, top_k=3, top_p=0.95)
        second_call = device_sample_token(logits, uniform;
            temperature=0.6, top_k=3, top_p=0.95)
        @test first_call == second_call
    end

    # Drawing from an RNG consumes exactly one Float32 per token, so the host
    # and device paths stay on the same random stream.
    left = MersenneTwister(7)
    right = MersenneTwister(7)
    for _ in 1:32
        uniform = rand(right, Float32)
        @test device_sample_token(logits, left;
            temperature=0.6, top_k=3, top_p=0.95) ==
            device_sample_token(logits, uniform;
                temperature=0.6, top_k=3, top_p=0.95)
    end
end

@testset "device sampling accepts one-element device-style inputs" begin
    logits = reshape(Float32[0.2, 1.7, -0.4, 2.2, 0.9], :, 1, 1)
    scalar_choice = _device_sample_next_index(
        vec(logits), 0.4f0, 0.8f0, 0.9f0, 3)
    array_choice = _device_sample_next_index(
        logits, Float32[0.4], Float32[0.8], Float32[0.9], 3)
    @test scalar_choice == array_choice
    @test scalar_choice == _sample_token(vec(logits), Random.default_rng();
        temperature=0.8, top_k=3, top_p=0.9, sample_uniform=0.4f0)
end
