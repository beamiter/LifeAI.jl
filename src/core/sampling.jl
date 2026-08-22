const MAX_DIFFUSION_STEPS = 1_000_000

function _diffusion_schedule_length(values)
    isempty(values) && throw(ArgumentError("`betas` must not be empty"))
    length(values) <= MAX_DIFFUSION_STEPS || throw(ArgumentError(
        "diffusion schedule exceeds the supported resource limit of $MAX_DIFFUSION_STEPS",
    ))
    return length(values)
end

"""Precomputed coefficients for a discrete variance-preserving diffusion process."""
struct DiffusionSchedule{T<:AbstractFloat}
    betas::Vector{T}
    alphas::Vector{T}
    alpha_cumprod::Vector{T}

    function DiffusionSchedule(betas::AbstractVector{T}) where {T<:AbstractFloat}
        _diffusion_schedule_length(betas)
        isconcretetype(T) || throw(ArgumentError(
            "diffusion schedule element type must be a concrete floating-point type",
        ))
        values = collect(betas)
        all(beta -> isfinite(beta) && zero(T) < beta < one(T), values) ||
            throw(ArgumentError("every diffusion beta must be finite and in (0, 1)"))
        alphas = one(T) .- values
        all(alpha -> zero(T) < alpha < one(T), alphas) || throw(ArgumentError(
            "diffusion betas must remain distinguishable from zero and one at their storage precision",
        ))
        alpha_cumprod = accumulate(*, alphas)
        all(value -> isfinite(value) && zero(T) < value < one(T), alpha_cumprod) ||
            throw(ArgumentError("diffusion cumulative alphas underflowed or became invalid"))
        new{T}(values, alphas, alpha_cumprod)
    end
end

function DiffusionSchedule(betas::AbstractVector{<:Real})
    _diffusion_schedule_length(betas)
    return DiffusionSchedule(Float64.(betas))
end

function _diffusion_steps(steps::Integer)
    steps isa Bool && throw(ArgumentError("`steps` must be an integer, not Bool"))
    steps > 0 || throw(ArgumentError("`steps` must be positive"))
    steps <= typemax(Int) || throw(ArgumentError("`steps` is too large"))
    steps <= MAX_DIFFUSION_STEPS || throw(ArgumentError(
        "`steps` exceeds the supported resource limit of $MAX_DIFFUSION_STEPS",
    ))
    return Int(steps)
end

function _diffusion_float_type(::Type{T}) where {T}
    T <: AbstractFloat || throw(ArgumentError("diffusion schedule type must be floating-point"))
    isconcretetype(T) || throw(ArgumentError("diffusion schedule type must be concrete"))
    return T
end

"""Construct a linearly spaced beta schedule."""
function linear_diffusion_schedule(
    steps::Integer;
    beta_start::Real=1e-4,
    beta_end::Real=2e-2,
    T::Type=Float32,
)
    count = _diffusion_steps(steps)
    F = _diffusion_float_type(T)
    first_beta = F(beta_start)
    last_beta = F(beta_end)
    isfinite(first_beta) && isfinite(last_beta) &&
        zero(F) < first_beta <= last_beta < one(F) || throw(ArgumentError(
            "linear beta endpoints must be finite and satisfy 0 < beta_start <= beta_end < 1",
        ))
    return DiffusionSchedule(collect(range(first_beta, last_beta; length=count)))
end

"""Construct the squared-linear schedule commonly used by latent diffusion models."""
function scaled_linear_diffusion_schedule(
    steps::Integer;
    beta_start::Real=8.5e-4,
    beta_end::Real=1.2e-2,
    T::Type=Float32,
)
    count = _diffusion_steps(steps)
    F = _diffusion_float_type(T)
    first_beta = F(beta_start)
    last_beta = F(beta_end)
    isfinite(first_beta) && isfinite(last_beta) &&
        zero(F) < first_beta <= last_beta < one(F) || throw(ArgumentError(
            "scaled-linear beta endpoints must be finite and satisfy " *
            "0 < beta_start <= beta_end < 1",
        ))
    roots = range(sqrt(first_beta), sqrt(last_beta); length=count)
    return DiffusionSchedule(F[value * value for value in roots])
end

"""
Construct the cosine schedule from a normalized cumulative-alpha curve.

`offset` avoids a singularly small first beta and `max_beta` prevents a final
step of exactly one, which would make inverse parameterizations undefined.
"""
function cosine_diffusion_schedule(
    steps::Integer;
    offset::Real=0.008,
    max_beta::Real=0.999,
    T::Type=Float32,
)
    count = _diffusion_steps(steps)
    F = _diffusion_float_type(T)
    shift = F(offset)
    cap = F(max_beta)
    isfinite(shift) && shift >= zero(F) || throw(ArgumentError(
        "cosine schedule offset must be finite and non-negative",
    ))
    isfinite(cap) && zero(F) < cap < one(F) || throw(ArgumentError(
        "cosine schedule max_beta must be finite and in (0, 1)",
    ))
    denominator = one(F) + shift
    alpha_bar(t) = cospi(((F(t) / F(count)) + shift) / denominator / F(2))^2
    normalization = alpha_bar(0)
    betas = Vector{F}(undef, count)
    previous = one(F)
    for index in 1:count
        current = alpha_bar(index) / normalization
        beta = one(F) - current / previous
        betas[index] = clamp(beta, eps(F), cap)
        previous = current
    end
    return DiffusionSchedule(betas)
end

function _diffusion_timestep(schedule::DiffusionSchedule, timestep::Integer)
    timestep isa Bool && throw(ArgumentError("`timestep` must be an integer, not Bool"))
    1 <= timestep <= length(schedule.betas) || throw(BoundsError(schedule.betas, timestep))
    return Int(timestep)
end

function _diffusion_local_values(schedule::DiffusionSchedule, timestep::Integer)
    steps = _diffusion_schedule_length(schedule.betas)
    length(schedule.alphas) == steps && length(schedule.alpha_cumprod) == steps ||
        throw(ArgumentError("diffusion schedule vectors have inconsistent lengths"))
    index = _diffusion_timestep(schedule, timestep)
    beta = schedule.betas[index]
    alpha = schedule.alphas[index]
    cumulative = schedule.alpha_cumprod[index]
    previous = index == 1 ? one(alpha) : schedule.alpha_cumprod[index - 1]
    isfinite(beta) && zero(beta) < beta < one(beta) || throw(ArgumentError(
        "diffusion schedule beta at timestep $index is invalid",
    ))
    isfinite(alpha) && zero(alpha) < alpha < one(alpha) && alpha == one(alpha) - beta ||
        throw(ArgumentError("diffusion schedule alpha at timestep $index is inconsistent"))
    isfinite(previous) && zero(previous) < previous <= one(previous) || throw(ArgumentError(
        "diffusion previous cumulative alpha at timestep $index is invalid",
    ))
    isfinite(cumulative) && zero(cumulative) < cumulative < one(cumulative) &&
        cumulative == previous * alpha || throw(ArgumentError(
            "diffusion cumulative alpha at timestep $index is inconsistent",
        ))
    return (; index, beta, alpha, previous, cumulative)
end

"""Validate every stored schedule coefficient and recurrence in O(steps)."""
function validate_diffusion_schedule(schedule::DiffusionSchedule)
    steps = _diffusion_schedule_length(schedule.betas)
    length(schedule.alphas) == steps && length(schedule.alpha_cumprod) == steps ||
        throw(ArgumentError("diffusion schedule vectors have inconsistent lengths"))
    for timestep in 1:steps
        _diffusion_local_values(schedule, timestep)
    end
    return nothing
end

"""Return beta at one validated one-based timestep."""
diffusion_beta(schedule::DiffusionSchedule, timestep::Integer) =
    _diffusion_local_values(schedule, timestep).beta

"""Return alpha (`1 - beta`) at one validated one-based timestep."""
diffusion_alpha(schedule::DiffusionSchedule, timestep::Integer) =
    _diffusion_local_values(schedule, timestep).alpha

"""Return cumulative alpha-bar at one validated one-based timestep."""
diffusion_alpha_cumprod(schedule::DiffusionSchedule, timestep::Integer) =
    _diffusion_local_values(schedule, timestep).cumulative

"""Return alpha-bar from the preceding step, using one at timestep one."""
diffusion_previous_alpha_cumprod(schedule::DiffusionSchedule, timestep::Integer) =
    _diffusion_local_values(schedule, timestep).previous

"""Return square-root signal/noise coefficients for one one-based timestep."""
function diffusion_coefficients(schedule::DiffusionSchedule, timestep::Integer)
    signal_squared = _diffusion_local_values(schedule, timestep).cumulative
    return (
        signal=sqrt(signal_squared),
        noise=sqrt(one(signal_squared) - signal_squared),
    )
end

"""Signal-to-noise ratio at one one-based timestep."""
function diffusion_snr(schedule::DiffusionSchedule, timestep::Integer)
    alpha_bar = _diffusion_local_values(schedule, timestep).cumulative
    return alpha_bar / (one(alpha_bar) - alpha_bar)
end

"""Log signal-to-noise ratio at one one-based timestep."""
diffusion_logsnr(schedule::DiffusionSchedule, timestep::Integer) =
    log(diffusion_snr(schedule, timestep))

"""Variance of `q(x[t-1] | x[t], x[0])` for one timestep."""
function diffusion_posterior_variance(schedule::DiffusionSchedule, timestep::Integer)
    values = _diffusion_local_values(schedule, timestep)
    return values.beta * (one(values.previous) - values.previous) /
        (one(values.cumulative) - values.cumulative)
end

"""Coefficients multiplying clean and noisy samples in the DDPM posterior mean."""
function diffusion_posterior_mean_coefficients(
    schedule::DiffusionSchedule,
    timestep::Integer,
)
    values = _diffusion_local_values(schedule, timestep)
    denominator = one(values.cumulative) - values.cumulative
    return (
        sample=values.beta * sqrt(values.previous) / denominator,
        noisy=sqrt(values.alpha) * (one(values.previous) - values.previous) / denominator,
    )
end

"""Mean of `q(x[t-1] | x[t], x[0])` for matching clean/noisy arrays."""
function diffusion_posterior_mean(
    schedule::DiffusionSchedule,
    sample,
    noisy,
    timestep::Integer,
)
    _diffusion_matching_axes(sample, noisy)
    coefficients = diffusion_posterior_mean_coefficients(schedule, timestep)
    return coefficients.sample .* sample .+ coefficients.noisy .* noisy
end

"""
Advance a predicted clean sample/epsilon pair to an earlier DDIM timestep.

`previous_timestep=0` denotes the clean endpoint. This is the deterministic
(`eta = 0`) path and therefore requires no random input.
"""
function diffusion_ddim_step(
    schedule::DiffusionSchedule,
    sample,
    epsilon,
    timestep::Integer;
    previous_timestep::Integer=timestep - 1,
)
    _diffusion_matching_axes(sample, epsilon)
    current = _diffusion_timestep(schedule, timestep)
    previous_timestep isa Bool && throw(ArgumentError(
        "`previous_timestep` must be an integer, not Bool",
    ))
    0 <= previous_timestep < current || throw(ArgumentError(
        "`previous_timestep` must lie in 0:$(current - 1)",
    ))
    previous_timestep <= typemax(Int) || throw(ArgumentError(
        "`previous_timestep` is too large",
    ))
    # Validate the current schedule location even though the deterministic
    # reconstruction only needs the destination coefficient.
    _diffusion_local_values(schedule, current)
    previous_alpha = previous_timestep == 0 ? one(eltype(schedule.betas)) :
        _diffusion_local_values(schedule, Int(previous_timestep)).cumulative
    return sqrt(previous_alpha) .* sample .+
        sqrt(one(previous_alpha) - previous_alpha) .* epsilon
end

"""Min-SNR loss weight for epsilon, clean-sample, or velocity prediction."""
function diffusion_min_snr_weight(
    schedule::DiffusionSchedule,
    timestep::Integer;
    gamma::Real=5,
    prediction::Symbol=:epsilon,
)
    gamma isa Bool && throw(ArgumentError("`gamma` must not be Bool"))
    F = eltype(schedule.betas)
    cap = F(gamma)
    isfinite(cap) && cap > zero(F) || throw(ArgumentError(
        "min-SNR gamma must remain finite and positive at schedule precision",
    ))
    snr = diffusion_snr(schedule, timestep)
    clipped = min(snr, cap)
    if prediction === :epsilon
        return clipped / snr
    elseif prediction === :sample
        return clipped
    elseif prediction === :velocity
        return clipped / (snr + one(snr))
    end
    throw(ArgumentError("prediction must be :epsilon, :sample, or :velocity"))
end

"""P2 reweighting `(k + SNR)^(-gamma)` at one timestep."""
function diffusion_p2_weight(
    schedule::DiffusionSchedule,
    timestep::Integer;
    k::Real=1,
    gamma::Real=1,
)
    k isa Bool && throw(ArgumentError("P2 k must not be Bool"))
    gamma isa Bool && throw(ArgumentError("P2 gamma must not be Bool"))
    F = eltype(schedule.betas)
    offset = F(k)
    exponent = F(gamma)
    isfinite(offset) && offset > zero(F) || throw(ArgumentError(
        "P2 k must remain finite and positive at schedule precision",
    ))
    isfinite(exponent) && exponent >= zero(F) || throw(ArgumentError(
        "P2 gamma must remain finite and non-negative at schedule precision",
    ))
    return (offset + diffusion_snr(schedule, timestep))^(-exponent)
end

function _diffusion_matching_axes(sample, noise)
    axes(sample) == axes(noise) || throw(DimensionMismatch(
        "diffusion sample and noise must have identical axes",
    ))
    return nothing
end

"""Apply the closed-form forward diffusion process at one timestep."""
function diffuse_sample(schedule::DiffusionSchedule, sample, noise, timestep::Integer)
    _diffusion_matching_axes(sample, noise)
    coefficients = diffusion_coefficients(schedule, timestep)
    return coefficients.signal .* sample .+ coefficients.noise .* noise
end

"""Recover the clean sample from a noisy sample and an epsilon prediction."""
function diffusion_sample_from_epsilon(
    schedule::DiffusionSchedule,
    noisy,
    epsilon,
    timestep::Integer,
)
    _diffusion_matching_axes(noisy, epsilon)
    coefficients = diffusion_coefficients(schedule, timestep)
    return (noisy .- coefficients.noise .* epsilon) ./ coefficients.signal
end

"""Recover epsilon from a noisy sample and a clean-sample prediction."""
function diffusion_epsilon_from_sample(
    schedule::DiffusionSchedule,
    noisy,
    sample,
    timestep::Integer,
)
    _diffusion_matching_axes(noisy, sample)
    coefficients = diffusion_coefficients(schedule, timestep)
    return (noisy .- coefficients.signal .* sample) ./ coefficients.noise
end

"""Convert clean-sample/epsilon targets to the velocity parameterization."""
function diffusion_velocity(
    schedule::DiffusionSchedule,
    sample,
    epsilon,
    timestep::Integer,
)
    _diffusion_matching_axes(sample, epsilon)
    coefficients = diffusion_coefficients(schedule, timestep)
    return coefficients.signal .* epsilon .- coefficients.noise .* sample
end

"""Recover the clean sample from a noisy sample and a velocity prediction."""
function diffusion_sample_from_velocity(
    schedule::DiffusionSchedule,
    noisy,
    velocity,
    timestep::Integer,
)
    _diffusion_matching_axes(noisy, velocity)
    coefficients = diffusion_coefficients(schedule, timestep)
    return coefficients.signal .* noisy .- coefficients.noise .* velocity
end

"""Recover epsilon from a noisy sample and a velocity prediction."""
function diffusion_epsilon_from_velocity(
    schedule::DiffusionSchedule,
    noisy,
    velocity,
    timestep::Integer,
)
    _diffusion_matching_axes(noisy, velocity)
    coefficients = diffusion_coefficients(schedule, timestep)
    return coefficients.noise .* noisy .+ coefficients.signal .* velocity
end
