using Test
using Random: Xoshiro

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
