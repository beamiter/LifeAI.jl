using Test
using Random: Xoshiro
using NNlib: softmax

if isdefined(@__MODULE__, :LifeAI)
    using .LifeAI:
        MAX_SAMPLING_VOCAB_SIZE,
        _validate_device_sampling_options,
        _validate_device_sampling_uniform,
        device_sample_token,
        _sample_categorical,
        _sample_token,
        _sampling_distribution
else
    include(joinpath(@__DIR__, "..", "src", "generation", "sampling.jl"))
    include(joinpath(@__DIR__, "..", "src", "generation", "xla_sampling.jl"))
end

@testset "sampling temperature overflow fallback" begin
    for logits in (Float32[10, 9, 8], Float32[-8, -9, -10],
                   Float32[floatmax(Float32), 0, -floatmax(Float32)])
        scores, probabilities = _sampling_distribution(logits;
            temperature=nextfloat(0.0f0))
        @test all(isfinite, scores)
        @test probabilities == [1.0, 0.0, 0.0]
        @test _sample_categorical(probabilities, prevfloat(1.0)) == 1
    end
    for logits in (Float32[10, 10, 9], Float32[-9, -9, -10],
                   fill(floatmax(Float32), 3))
        scores, probabilities = _sampling_distribution(logits; temperature=1f-38)
        @test all(isfinite, scores)
        expected = logits[1] == logits[3] ? fill(1 / 3, 3) : [0.5, 0.5, 0.0]
        @test probabilities ≈ expected
        filtered, kept = _sampling_distribution(logits;
            temperature=1f-38, top_k=1, top_p=0.5)
        @test findall(isfinite, filtered) == [1]
        @test kept == [1, 0, 0]
    end
    # Ordinary inputs still return the original, uncentered Float32 scores and
    # use exactly the existing Float32 softmax path.
    logits = Float32[4, 3, 2, 1]
    scores, probabilities = _sampling_distribution(logits; temperature=0.7f0)
    @test eltype(scores) === Float32
    @test scores == logits ./ 0.7f0
    @test probabilities == softmax(logits ./ 0.7f0)
    for logits in (Float32[Inf, 0], Float32[-Inf, 0], Float32[NaN, 0],
                   Float64[floatmax(Float64), 0])
        @test_throws ArgumentError _sampling_distribution(logits; temperature=1f-38)
    end
end

if isdefined(@__MODULE__, :LifeAI)
    @testset "sampling trace preserves overflow fallback precision" begin
        for (logits, temperature) in ((Float32[10, 10, 9], 1f-38),
                                      (Float32[4, 3, 2], 0.7f0))
            for uniform in (prevfloat(0.5), 0.5, prevfloat(1.0))
                token, trace = LifeAI._hf_sample_choice(
                    reshape(logits, :, 1, 1), identity, Xoshiro(148);
                    temperature, top_k=length(logits), top_p=1.0f0,
                    sample_uniform=uniform, capture_distribution=true,
                )
                scores, probabilities = _sampling_distribution(logits; temperature)
                expected_type = temperature == 0.7f0 ? Float32 : Float64
                @test eltype(trace.distribution.logits) === expected_type
                @test eltype(trace.distribution.probabilities) === expected_type
                @test all(isfinite, trace.distribution.logits)
                @test trace.distribution.logits == scores
                @test trace.distribution.probabilities == probabilities
                @test token == _sample_categorical(probabilities, uniform)
                @test trace.sampled_probability === probabilities[token]
                @test trace.candidate_count == length(scores)
            end
        end
    end
end

struct OversizedSamplingVector <: AbstractVector{Float32} end
Base.size(::OversizedSamplingVector) = (MAX_SAMPLING_VOCAB_SIZE + 1,)
Base.getindex(::OversizedSamplingVector, ::Int) = 0.0f0

@testset "sampling validation and deterministic filters" begin
    logits = Float32[4, 3, 2, 1]
    @test_throws ArgumentError _sampling_distribution(logits; top_k=true)
    @test_throws ArgumentError _sampling_distribution(logits; top_p=true)
    @test_throws ArgumentError _sampling_distribution(logits; top_p=big"1e-1000")
    @test_throws ArgumentError _sampling_distribution(logits; temperature=true)
    @test_throws ArgumentError _sampling_distribution(logits; top_k=big(typemax(Int)) + 1)
    @test_throws ArgumentError _sampling_distribution(logits; temperature=1.0e-100)
    @test_throws ArgumentError _sampling_distribution(BigFloat[big"1e10000", 0])
    @test_throws ArgumentError _sampling_distribution(OversizedSamplingVector())
    @test_throws ArgumentError _sample_categorical(OversizedSamplingVector(), 0.5)
    @test_throws ArgumentError _sample_categorical(Float32[0.5, 0.5], false)
    @test_throws ArgumentError _sample_categorical(Float32[0.5, 0.5], true)
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=true, top_k=2, top_p=1,
    )
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=1, top_k=true, top_p=1,
    )
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=1, top_k=big(typemax(Int)) + 1, top_p=1,
    )
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=1, top_k=2, top_p=true,
    )
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=big"1e-1000", top_k=2, top_p=1,
    )
    @test_throws ArgumentError _validate_device_sampling_options(;
        temperature=1, top_k=2, top_p=big"1e-1000",
    )
    @test_throws ArgumentError _validate_device_sampling_uniform(false)
    @test_throws ArgumentError _validate_device_sampling_uniform(prevfloat(1.0))
    @test_throws ArgumentError device_sample_token(
        BigFloat[big"1e1000", 0], 0.5; temperature=1, top_k=2, top_p=1,
    )
    @test_throws ArgumentError device_sample_token(
        OversizedSamplingVector(), 0.5; temperature=1, top_k=2, top_p=1,
    )

    tied, _ = _sampling_distribution(Float32[2, 1, 1, 0]; top_k=2, top_p=1)
    @test findall(isfinite, tied) == [1, 2]
    top_k_only, top_k_probabilities = _sampling_distribution(logits; top_k=2, top_p=1)
    @test findall(isfinite, top_k_only) == [1, 2]
    @test top_k_probabilities[3:4] == zeros(Float32, 2)

    for p in (1, 1.0f0, 1.0)
        tail_scores, tail_probabilities = _sampling_distribution(Float32[0, -20]; top_p=p)
        @test all(isfinite, tail_scores)
        @test all(>(0), tail_probabilities)
        @test tail_scores == Float32[0, -20]
        limited, limited_probabilities = _sampling_distribution(Float32[0, -20]; top_k=1, top_p=p)
        @test findall(isfinite, limited) == [1]
        @test limited_probabilities == Float32[1, 0]
    end
    for invalid in (true, 0, -1, Inf, NaN, 1.01)
        @test_throws ArgumentError _sampling_distribution(Float32[0, -20]; top_p=invalid)
    end

    nucleus_logits = log.(Float32[0.4, 0.3, 0.2, 0.1])
    filtered, probabilities = _sampling_distribution(nucleus_logits; top_p=0.6)
    @test findall(isfinite, filtered) == [1, 2]
    @test probabilities ≈ Float32[4 / 7, 3 / 7, 0, 0] atol=1.0f-6

    smallest, _ = _sampling_distribution(nucleus_logits; top_p=0.4)
    @test findall(isfinite, smallest) == [1]

    stable_tie, _ = _sampling_distribution(log.(Float32[0.4, 0.3, 0.3]); top_p=0.6)
    @test findall(isfinite, stable_tie) == [1, 2]

    tied_logits = Float32[4, 1, 4, 0]
    @test _sample_token(
        tied_logits, Xoshiro(1); top_k=4, top_p=0.4, sample_uniform=0.5,
    ) == 1
    @test device_sample_token(
        tied_logits, 0.5; temperature=1, top_k=4, top_p=0.4,
    ) == 1
    tie_at_k = Float32[3, 3, 3, -20]
    for uniform in 0.0f0:0.05f0:0.95f0
        @test device_sample_token(
            tie_at_k, uniform; temperature=1, top_k=2, top_p=1,
        ) == _sample_token(
            tie_at_k, Xoshiro(2); top_k=2, top_p=1, sample_uniform=uniform,
        )
    end
    @test device_sample_token(
        Float32[0, 0], 0.5; temperature=1, top_k=2, top_p=1,
    ) == _sample_token(
        Float32[0, 0], Xoshiro(3); top_k=2, top_p=1, sample_uniform=0.5,
    ) == 2

    rng = Xoshiro(12)
    @test _sample_token(Float32[3, 3, 1], rng; top_k=1, sample_uniform=0.5) == 1
end
