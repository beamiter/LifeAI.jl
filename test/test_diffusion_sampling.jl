using Test

if isdefined(@__MODULE__, :LifeAI)
    using .LifeAI:
        MAX_DIFFUSION_STEPS,
        DiffusionSchedule,
        cosine_diffusion_schedule,
        diffuse_sample,
        diffusion_alpha,
        diffusion_alpha_cumprod,
        diffusion_beta,
        diffusion_coefficients,
        diffusion_ddim_step,
        diffusion_epsilon_from_sample,
        diffusion_epsilon_from_velocity,
        diffusion_logsnr,
        diffusion_min_snr_weight,
        diffusion_p2_weight,
        diffusion_posterior_mean,
        diffusion_posterior_mean_coefficients,
        diffusion_posterior_variance,
        diffusion_previous_alpha_cumprod,
        diffusion_sample_from_epsilon,
        diffusion_sample_from_velocity,
        diffusion_snr,
        diffusion_velocity,
        linear_diffusion_schedule,
        scaled_linear_diffusion_schedule,
        validate_diffusion_schedule
else
    include(joinpath(@__DIR__, "..", "src", "core", "sampling.jl"))
end

if isdefined(@__MODULE__, :LifeAI)
    @test isdefined(LifeAI, :DiffusionSchedule)
    @test isdefined(LifeAI, :diffusion_epsilon_from_velocity)
end

struct OversizedDiffusionBetas <: AbstractVector{Float32} end
Base.size(::OversizedDiffusionBetas) = (MAX_DIFFUSION_STEPS + 1,)
Base.getindex(::OversizedDiffusionBetas, ::Int) = 0.1f0

@testset "diffusion schedules" begin
    custom = DiffusionSchedule(Float64[0.1, 0.2, 0.3])
    @test isnothing(validate_diffusion_schedule(custom))
    @test custom.alphas == [0.9, 0.8, 0.7]
    @test custom.alpha_cumprod ≈ [0.9, 0.72, 0.504]
    @test_throws ArgumentError DiffusionSchedule(Float32[])
    @test_throws ArgumentError DiffusionSchedule([0.0, 0.1])
    @test_throws ArgumentError DiffusionSchedule([0.1, 1.0])
    @test_throws ArgumentError DiffusionSchedule([0.1, NaN])
    @test_throws ArgumentError DiffusionSchedule(Float32[nextfloat(0.0f0)])
    @test_throws ArgumentError DiffusionSchedule(AbstractFloat[0.1f0, 0.2])
    @test_throws ArgumentError DiffusionSchedule(OversizedDiffusionBetas())

    linear = linear_diffusion_schedule(5; beta_start=0.01, beta_end=0.05)
    @test eltype(linear.betas) === Float32
    @test linear.betas ≈ Float32[0.01, 0.02, 0.03, 0.04, 0.05]
    @test_throws ArgumentError linear_diffusion_schedule(true)
    @test_throws ArgumentError linear_diffusion_schedule(0)
    @test_throws ArgumentError linear_diffusion_schedule(MAX_DIFFUSION_STEPS + 1)
    @test_throws ArgumentError linear_diffusion_schedule(2; beta_start=0.2, beta_end=0.1)
    @test_throws ArgumentError linear_diffusion_schedule(2; T=Int)

    scaled = scaled_linear_diffusion_schedule(
        3;
        beta_start=0.01,
        beta_end=0.09,
        T=Float64,
    )
    @test scaled.betas ≈ [0.01, 0.04, 0.09]

    cosine = cosine_diffusion_schedule(100; T=Float64)
    @test length(cosine.betas) == 100
    @test all(0 .< cosine.betas .< 1)
    @test issorted(cosine.betas)
    @test all(diff(cosine.alpha_cumprod) .< 0)
    @test maximum(cosine.betas) <= 0.999
    @test_throws ArgumentError cosine_diffusion_schedule(4; offset=-0.1)
    @test_throws ArgumentError cosine_diffusion_schedule(4; max_beta=1)
end

@testset "diffusion parameterizations" begin
    schedule = DiffusionSchedule([0.1, 0.2])
    coefficients = diffusion_coefficients(schedule, 2)
    @test coefficients.signal ≈ sqrt(0.72)
    @test coefficients.noise ≈ sqrt(0.28)
    @test diffusion_snr(schedule, 2) ≈ 0.72 / 0.28
    @test isfinite(diffusion_snr(schedule, 1))
    @test diffusion_logsnr(schedule, 2) ≈ log(0.72 / 0.28)
    @test_throws BoundsError diffusion_coefficients(schedule, 0)
    @test_throws BoundsError diffusion_coefficients(schedule, 3)
    @test_throws ArgumentError diffusion_coefficients(schedule, true)

    mutated = DiffusionSchedule([0.1])
    mutated.alpha_cumprod[1] = 1.0
    @test_throws ArgumentError diffusion_coefficients(mutated, 1)
    @test_throws ArgumentError diffusion_snr(mutated, 1)

    accessors = DiffusionSchedule([0.1, 0.2])
    @test diffusion_beta(accessors, 2) == 0.2
    @test diffusion_alpha(accessors, 2) == 0.8
    @test diffusion_alpha_cumprod(accessors, 2) ≈ 0.72
    @test diffusion_previous_alpha_cumprod(accessors, 1) == 1.0
    @test diffusion_previous_alpha_cumprod(accessors, 2) == 0.9
    broken = DiffusionSchedule([0.1, 0.2])
    broken.alphas[2] = 0.7
    @test_throws ArgumentError diffusion_alpha(broken, 2)
    @test_throws ArgumentError validate_diffusion_schedule(broken)
    shortened = DiffusionSchedule([0.1, 0.2])
    pop!(shortened.alphas)
    @test_throws ArgumentError diffusion_beta(shortened, 1)

    sample = [1.0, -2.0, 0.5]
    epsilon = [-0.25, 0.75, 2.0]
    noisy = diffuse_sample(schedule, sample, epsilon, 2)
    @test noisy ≈ coefficients.signal .* sample .+ coefficients.noise .* epsilon
    @test diffusion_sample_from_epsilon(schedule, noisy, epsilon, 2) ≈ sample
    @test diffusion_epsilon_from_sample(schedule, noisy, sample, 2) ≈ epsilon

    velocity = diffusion_velocity(schedule, sample, epsilon, 2)
    @test diffusion_sample_from_velocity(schedule, noisy, velocity, 2) ≈ sample
    @test diffusion_epsilon_from_velocity(schedule, noisy, velocity, 2) ≈ epsilon
    @test_throws DimensionMismatch diffuse_sample(schedule, sample, epsilon[1:2], 1)

    @test diffusion_posterior_variance(schedule, 1) == 0
    @test diffusion_posterior_variance(schedule, 2) ≈ 0.2 * 0.1 / 0.28
    first_posterior = diffusion_posterior_mean_coefficients(schedule, 1)
    @test first_posterior.sample ≈ 1
    @test first_posterior.noisy ≈ 0
    posterior = diffusion_posterior_mean_coefficients(schedule, 2)
    @test posterior.sample ≈ 0.2 * sqrt(0.9) / 0.28
    @test posterior.noisy ≈ sqrt(0.8) * 0.1 / 0.28
    @test diffusion_posterior_mean(schedule, sample, noisy, 2) ≈
          posterior.sample .* sample .+ posterior.noisy .* noisy
    @test_throws DimensionMismatch diffusion_posterior_mean(
        schedule, sample, noisy[1:2], 2,
    )

    @test diffusion_ddim_step(schedule, sample, epsilon, 2; previous_timestep=0) ≈ sample
    @test diffusion_ddim_step(schedule, sample, epsilon, 2) ≈
          sqrt(0.9) .* sample .+ sqrt(0.1) .* epsilon
    @test_throws ArgumentError diffusion_ddim_step(
        schedule, sample, epsilon, 2; previous_timestep=2,
    )
    @test_throws ArgumentError diffusion_ddim_step(
        schedule, sample, epsilon, 2; previous_timestep=true,
    )
    @test_throws DimensionMismatch diffusion_ddim_step(
        schedule, sample, epsilon[1:2], 2,
    )

    snr = diffusion_snr(schedule, 2)
    @test diffusion_min_snr_weight(schedule, 2; gamma=0.5, prediction=:epsilon) ≈
          min(snr, 0.5) / snr
    @test diffusion_min_snr_weight(schedule, 2; gamma=0.5, prediction=:sample) ≈
          min(snr, 0.5)
    @test diffusion_min_snr_weight(schedule, 2; gamma=0.5, prediction=:velocity) ≈
          min(snr, 0.5) / (snr + 1)
    @test_throws ArgumentError diffusion_min_snr_weight(schedule, 2; gamma=true)
    @test_throws ArgumentError diffusion_min_snr_weight(schedule, 2; prediction=:unknown)
    @test diffusion_p2_weight(schedule, 2; gamma=0) == 1
    @test diffusion_p2_weight(schedule, 2; k=1, gamma=1) ≈ 1 / (1 + snr)
    @test_throws ArgumentError diffusion_p2_weight(schedule, 2; k=0)
    @test_throws ArgumentError diffusion_p2_weight(schedule, 2; gamma=-1)

    float32_schedule = linear_diffusion_schedule(4)
    @test @inferred(diffusion_coefficients(float32_schedule, 1)) isa
          NamedTuple{(:signal, :noise),Tuple{Float32,Float32}}
end
