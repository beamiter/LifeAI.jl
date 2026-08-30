using Lux
using NNlib: gather, logsoftmax
using Optimisers
using Random: AbstractRNG

# Loading these packages activates Lux's Zygote and Reactant/Enzyme extensions.
import Enzyme
import Reactant
import Zygote

"""
    TrainerGPT(; learning_rate=3f-4, optimizer=nothing, backend=:zygote,
                 xla_backend="gpu", device=nothing, ad=nothing,
                 return_gradients=nothing, static_shapes=true,
                 max_grad_norm=nothing)

Small configuration object for GPT training.

Backends:

- `backend=:zygote`: ordinary Lux training, defaulting to the CPU.
- `backend=:xla`: move parameters, states, optimizer state, and batches to a
  Reactant device. Gradient computation and the optimizer update are compiled
  as separate stages so loss and global-gradient metrics can be materialized
  and validated before any parameter update.

`return_gradients=false` suppresses the gradient tree in the return value only;
all backends still compute and validate the loss and global gradient norm before
applying an optimizer update.

Set `max_grad_norm` to a positive value to enable true global L2 norm clipping
over the complete parameter tree. Clipping requires a separate gradient
computation and optimizer application, including on XLA.
"""
struct TrainerGPT{O, A, D}
    optimizer::O
    ad::A
    device::D
    backend::Symbol
    return_gradients::Bool
    static_shapes::Bool
    max_grad_norm::Union{Nothing,Float32}
end

function _training_positive_float32(value, name::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "`$name` must be a real number",
    ))
    is_positive = try
        value > 0
    catch error
        error isa MethodError || error isa DomainError ||
            error isa ArgumentError || rethrow()
        throw(ArgumentError("`$name` must support a finite sign comparison"))
    end
    is_positive isa Bool || throw(ArgumentError(
        "`$name` sign comparison must return Bool",
    ))
    is_positive || throw(ArgumentError("`$name` must be positive"))
    converted = try
        Float32(value)
    catch error
        error isa InexactError || error isa OverflowError ||
            error isa DomainError || error isa MethodError || rethrow()
        throw(ArgumentError("`$name` must be representable as Float32"))
    end
    iszero(converted) && throw(OverflowError(
        "`$name` underflows Float32",
    ))
    isfinite(converted) || throw(ArgumentError(
        "`$name` must be finite and positive at Float32 precision",
    ))
    return converted
end

function _training_loop_int(value, name::AbstractString; allow_zero::Bool=false)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "`$name` must be an integer",
    ))
    converted = try
        Int(value)
    catch error
        error isa InexactError || error isa OverflowError ||
            error isa DomainError || error isa MethodError || rethrow()
        throw(ArgumentError("`$name` is outside the supported integer range"))
    end
    lower_bound = allow_zero ? 0 : 1
    converted >= lower_bound || throw(ArgumentError(
        "`$name` must be $(allow_zero ? "non-negative" : "positive")",
    ))
    return converted
end

function _training_metric_float32(
    value,
    name::AbstractString;
    nonnegative::Bool=false,
)
    value isa Number && !(value isa Bool) && !(value isa Complex) || throw(ArgumentError(
        "`$name` must be a real scalar number",
    ))
    if nonnegative
        is_negative = try
            value < 0
        catch error
            error isa MethodError || error isa DomainError ||
                error isa ArgumentError || rethrow()
            throw(ArgumentError("`$name` must support a finite sign comparison"))
        end
        is_negative isa Bool || throw(ArgumentError(
            "`$name` sign comparison must return Bool",
        ))
        is_negative && throw(ArgumentError("`$name` must be non-negative"))
    end
    if value isa AbstractFloat && !isfinite(value)
        throw(ArgumentError("`$name` must be finite"))
    end
    converted = try
        Float32(value)
    catch error
        error isa InexactError || error isa OverflowError ||
            error isa DomainError || error isa MethodError || rethrow()
        throw(ArgumentError("`$name` must be representable as Float32"))
    end
    isfinite(converted) || throw(OverflowError(
        "`$name` is not representable as finite Float32",
    ))
    if iszero(converted)
        is_nonzero = try
            value != 0
        catch error
            error isa MethodError || error isa DomainError ||
                error isa ArgumentError || rethrow()
            throw(ArgumentError(
                "`$name` must support a finite zero comparison",
            ))
        end
        is_nonzero isa Bool || throw(ArgumentError(
            "`$name` zero comparison must return Bool",
        ))
        is_nonzero && throw(OverflowError("`$name` underflows Float32"))
    end
    if nonnegative && converted < 0
        throw(ArgumentError("`$name` must be non-negative"))
    end
    return converted
end

function TrainerGPT(;
    learning_rate::Real=3.0f-4,
    optimizer=nothing,
    backend::Symbol=:zygote,
    xla_backend::AbstractString="gpu",
    device=nothing,
    ad=nothing,
    return_gradients=nothing,
    static_shapes::Bool=true,
    max_grad_norm=nothing,
)
    learning_rate_value = _training_positive_float32(learning_rate, "learning_rate")

    optimizer_rule = if optimizer === nothing
        Optimisers.Adam(learning_rate_value)
    else
        optimizer
    end

    if backend === :xla
        xla_backend in ("cpu", "gpu", "tpu") ||
            throw(ArgumentError("`xla_backend` must be \"cpu\", \"gpu\", or \"tpu\""))

        Reactant.set_default_backend(String(xla_backend))
        device === nothing && (device = Lux.reactant_device(; force=true))
        ad === nothing && (ad = Lux.Training.AutoEnzyme())
    elseif backend === :zygote
        device === nothing && (device = Lux.cpu_device())
        ad === nothing && (ad = Lux.Training.AutoZygote())
    else
        throw(ArgumentError("`backend` must be `:zygote` or `:xla`"))
    end

    return_gradients_flag = if return_gradients === nothing
        backend !== :xla
    else
        return_gradients isa Bool || throw(ArgumentError(
            "`return_gradients` must be Bool or `nothing`",
        ))
        return_gradients
    end

    max_grad_norm_value = if max_grad_norm === nothing
        nothing
    else
        _training_positive_float32(max_grad_norm, "max_grad_norm")
    end

    return TrainerGPT(
        optimizer_rule,
        ad,
        device,
        backend,
        return_gradients_flag,
        static_shapes,
        max_grad_norm_value,
    )
end

function _validate_target_ids(targets::Array{T,N}, vocab_size::Int) where {T<:Integer,N}
    all(id -> 1 <= id <= vocab_size, targets) ||
        throw(ArgumentError("target token id is outside 1:$vocab_size"))
    return nothing
end

function _validate_target_ids(targets::Array, vocab_size::Int)
    throw(ArgumentError("`targets` must contain integer token ids"))
end

# Target ids are checked before transfer to Reactant. Keeping this method a
# no-op for device arrays avoids data-dependent control flow during XLA tracing.
_validate_target_ids(targets, vocab_size::Int) = nothing

"""
    next_token_nll_sum(logits, targets)

Sum token-level negative log likelihood without constructing one-hot targets.
"""
function next_token_nll_sum(logits, targets)
    ndims(logits) == 3 ||
        throw(DimensionMismatch("`logits` must have shape (vocab_size, seq_len, batch)"))
    ndims(targets) == 2 ||
        throw(DimensionMismatch("`targets` must have shape (seq_len, batch)"))

    vocab_size, seq_len, batch_size = size(logits)

    size(targets) == (seq_len, batch_size) ||
        throw(DimensionMismatch("target shape does not match logits"))
    _validate_target_ids(targets, vocab_size)

    log_probs = logsoftmax(logits; dims=1)
    target_ids = vec(targets)
    token_count = seq_len * batch_size
    linear_indices = target_ids .+ vocab_size .* (0:(token_count - 1))
    log_probs_flat = Reactant.materialize_traced_array(vec(log_probs))
    selected_log_probs = gather(log_probs_flat, linear_indices)

    return -sum(selected_log_probs)
end

"""
    next_token_loss(logits, targets)

Sparse mean next-token cross entropy.
"""
function next_token_loss(logits, targets)
    return next_token_nll_sum(logits, targets) / length(targets)
end

"""Run the model and return `(loss, updated_state)`."""
function next_token_loss(model, ps, st, x, targets)
    logits, st_new = model(x, ps, st)
    return next_token_loss(logits, targets), st_new
end

"""
    init_train_state(rng, model, trainer)

Initialize model parameters/states on `trainer.device` and create a Lux
`TrainState`.
"""
function init_train_state(rng::AbstractRNG, model, trainer::TrainerGPT)
    ps, st = trainer.device(Lux.setup(rng, model))
    return Lux.Training.TrainState(model, ps, st, trainer.optimizer)
end

function _gpt_objective(model, ps, st, data)
    x, targets = data
    loss, st_new = next_token_loss(model, ps, st, x, targets)
    return loss, st_new, NamedTuple()
end

_gradient_norm(::Nothing) = 0.0f0
_gradient_norm(x::Number) = abs(float(x))

function _gradient_array_norm(x::AbstractArray)
    isempty(x) && return 0.0f0
    max_abs = maximum(abs, x)
    safe_scale = ifelse(iszero(max_abs), one(max_abs), max_abs)
    return max_abs * sqrt(sum(abs2, x ./ safe_scale))
end

function _gradient_array_norm(x::Reactant.ConcreteRArray)
    isempty(x) && return 0.0f0
    max_abs_device = maximum(abs, x)
    max_abs = Float64(max_abs_device)
    (iszero(max_abs) || !isfinite(max_abs)) && return max_abs
    normalized_sq_sum = Float64(sum(abs2, x ./ max_abs_device))
    return max_abs * sqrt(normalized_sq_sum)
end

_gradient_norm(x::Reactant.ConcreteRNumber) = abs(Float64(x))
_gradient_norm(x::AbstractArray{<:AbstractFloat}) = _gradient_array_norm(x)
_gradient_norm(x::AbstractArray) = _gradient_array_norm(float.(x))
_gradient_norm(x::NamedTuple) = _gradient_norm_children(values(x))
_gradient_norm(x::Tuple) = _gradient_norm_children(x)
_gradient_norm(_) = 0.0f0

function _combine_gradient_norms(left::Real, right::Real)
    scale = max(abs(left), abs(right))
    safe_scale = ifelse(iszero(scale), one(scale), scale)
    return scale * sqrt(abs2(left / safe_scale) + abs2(right / safe_scale))
end

function _gradient_norm_children(children)
    accumulator = nothing
    for child in children
        value = _gradient_norm(child)
        accumulator = accumulator === nothing ? value :
            _combine_gradient_norms(accumulator, value)
    end
    return accumulator === nothing ? 0.0f0 : accumulator
end

"""Compute the L2 norm over every numeric leaf in a nested gradient tree."""
global_gradient_norm(gradients) = _gradient_norm(gradients)

function _checked_training_gradient_norm(gradients)
    norm = global_gradient_norm(gradients)
    return _training_metric_float32(
        norm,
        "gradient norm";
        nonnegative=true,
    )
end

_scale_gradient(::Nothing, _) = nothing
_scale_gradient(x::Number, scale) = x * scale
_scale_gradient(x::AbstractArray, scale) = x .* scale

function _scale_gradient(x::NamedTuple, scale)
    return NamedTuple{keys(x)}(map(value -> _scale_gradient(value, scale), values(x)))
end

_scale_gradient(x::Tuple, scale) = map(value -> _scale_gradient(value, scale), x)
_scale_gradient(x, _) = x

"""
    clip_global_gradient_norm(gradients, max_norm; epsilon=1f-6)

Scale the complete gradient tree by one common factor when its global L2 norm
exceeds `max_norm`. Returns `(clipped_gradients, metrics)`.
"""
function clip_global_gradient_norm(
    gradients,
    max_norm::Real;
    epsilon::Real=1.0f-6,
)
    max_norm_value = _training_positive_float32(max_norm, "max_norm")
    epsilon_value = _training_positive_float32(epsilon, "epsilon")

    norm_before = _checked_training_gradient_norm(gradients)
    scale = _training_metric_float32(
        min(
            1.0,
            Float64(max_norm_value) /
                (Float64(norm_before) + Float64(epsilon_value)),
        ),
        "gradient clipping scale";
        nonnegative=true,
    )
    norm_after = _training_metric_float32(
        Float64(norm_before) * Float64(scale),
        "gradient norm after clipping";
        nonnegative=true,
    )
    clipped_gradients = _scale_gradient(gradients, scale)
    metrics = (;
        before=norm_before,
        after=norm_after,
        scale,
    )

    return clipped_gradients, metrics
end

function _train_step_with_metrics!(trainer::TrainerGPT, train_state, batch)
    hasproperty(train_state, :step) || throw(ArgumentError(
        "`train_state` has no step counter",
    ))
    current_step = _training_loop_int(
        train_state.step,
        "train_state.step";
        allow_zero=true,
    )
    current_step < typemax(Int) || throw(OverflowError(
        "training step counter cannot be incremented beyond typemax(Int)",
    ))
    length(batch) == 2 || throw(ArgumentError("`batch` must be `(x, targets)`"))
    x, targets = batch

    _validate_token_ids(x, train_state.model.vocab_size)
    _validate_target_ids(targets, train_state.model.vocab_size)

    x_device, targets_device = trainer.device((x, targets))
    device_batch = (x_device, targets_device)

    # Lux's fused in-place step updates parameter arrays before returning its
    # metrics; the Reactant cache-hit path can do so even through the nominally
    # functional wrapper. Keep every backend on the explicit two-stage path so
    # invalid loss or gradient metrics fail before any optimiser update.
    gradients, loss, _, candidate_state = Lux.Training.compute_gradients(
        trainer.ad,
        _gpt_objective,
        device_batch,
        train_state,
    )
    _training_metric_float32(loss, "training loss"; nonnegative=true)

    if trainer.max_grad_norm === nothing
        norm = _checked_training_gradient_norm(gradients)
        metrics = (; before=norm, after=norm, scale=one(norm))
        candidate_state = Lux.Training.apply_gradients!(candidate_state, gradients)
        returned_gradients = trainer.return_gradients ? gradients : nothing

        return candidate_state, loss, returned_gradients, metrics
    end

    clipped_gradients, metrics = clip_global_gradient_norm(
        gradients,
        trainer.max_grad_norm,
    )
    candidate_state = Lux.Training.apply_gradients!(candidate_state, clipped_gradients)

    returned_gradients = trainer.return_gradients ? clipped_gradients : nothing
    return candidate_state, loss, returned_gradients, metrics
end

"""
    train_step!(trainer, train_state, batch)
    train_step!(trainer, train_state, x, targets)

Perform one optimizer step. When clipping is enabled, returned gradients are
the clipped gradients.
"""
function train_step!(trainer::TrainerGPT, train_state, batch)
    train_state, loss, gradients, _ = _train_step_with_metrics!(
        trainer,
        train_state,
        batch,
    )
    return train_state, loss, gradients
end

function train_step!(trainer::TrainerGPT, train_state, x, targets)
    return train_step!(trainer, train_state, (x, targets))
end

_metric_float(value, name::AbstractString) = value === nothing ? nothing :
    _training_metric_float32(value, name; nonnegative=true)

"""
    train_gpt!(trainer, train_state, loader; kwargs...)

Train over `loader`. `start_epoch`, `start_batch`, and `max_steps` support
resuming. With `validation_loader`, evaluation runs at each epoch end, or every
`evaluate_every` steps. The callback receives progress, gradient norms, and
optional validation loss/perplexity.
"""
function train_gpt!(
    trainer::TrainerGPT,
    train_state,
    loader;
    epochs::Integer=1,
    start_epoch::Integer=1,
    start_batch::Integer=1,
    max_steps=nothing,
    validation_loader=nothing,
    evaluate_every=nothing,
    callback=nothing,
)
    epochs = _training_loop_int(epochs, "epochs")
    start_epoch = _training_loop_int(start_epoch, "start_epoch")
    start_batch = _training_loop_int(start_batch, "start_batch")
    loader_length = _training_loop_int(
        length(loader),
        "loader length";
        allow_zero=true,
    )
    epochs - 1 <= typemax(Int) - start_epoch || throw(ArgumentError(
        "`start_epoch + epochs - 1` exceeds the supported integer range",
    ))
    start_batch_limit = loader_length == typemax(Int) ?
        typemax(Int) : loader_length + 1
    start_batch <= start_batch_limit || throw(ArgumentError(
        "`start_batch` must be in 1:$start_batch_limit",
    ))

    if max_steps !== nothing
        max_steps = _training_loop_int(
            max_steps,
            "max_steps";
            allow_zero=true,
        )
        max_steps == 0 && return train_state, Float32[]
    end

    if evaluate_every !== nothing
        validation_loader === nothing && throw(ArgumentError(
            "`validation_loader` is required when `evaluate_every` is set",
        ))
        evaluate_every = _training_loop_int(evaluate_every, "evaluate_every")
    end

    losses = Float32[]
    expected_batch_shape = nothing
    invocation_steps = 0

    for epoch_offset in 0:(epochs - 1)
        epoch = start_epoch + epoch_offset
        first_batch = epoch_offset == 0 ? start_batch : 1

        for batch_index in first_batch:loader_length
            batch = loader[batch_index]
            batch_shape = (size(batch[1]), size(batch[2]))

            if trainer.backend === :xla && trainer.static_shapes
                if expected_batch_shape === nothing
                    expected_batch_shape = batch_shape
                elseif batch_shape != expected_batch_shape
                    throw(ArgumentError(
                        "XLA training requires a stable batch shape; expected " *
                        "$(expected_batch_shape), got $(batch_shape). Use `drop_last=true` " *
                        "or set `static_shapes=false` to permit recompilation.",
                    ))
                end
            end

            step_start = time_ns()
            train_state, loss, _, gradient_metrics = _train_step_with_metrics!(
                trainer,
                train_state,
                batch,
            )
            step_seconds = Float64(time_ns() - step_start) / 1.0e9

            invocation_steps < typemax(Int) || throw(OverflowError(
                "training invocation-step counter overflow",
            ))
            invocation_steps += 1
            loss_value = _training_metric_float32(
                loss,
                "training loss";
                nonnegative=true,
            )
            push!(losses, loss_value)

            should_evaluate = validation_loader !== nothing && (
                evaluate_every === nothing ?
                    batch_index == loader_length :
                    train_state.step % evaluate_every == 0
            )
            validation_metrics = if should_evaluate
                metrics, _ = evaluate_gpt(
                    train_state.model,
                    train_state.parameters,
                    train_state.states,
                    validation_loader;
                    device=trainer.device,
                )
                metrics
            else
                nothing
            end

            if callback !== nothing
                callback((;
                    epoch,
                    batch=batch_index,
                    step=train_state.step,
                    progress=(; epoch, batch=batch_index, step=train_state.step),
                    loss=loss_value,
                    step_seconds,
                    xla_compilation=(trainer.backend === :xla && invocation_steps == 1),
                    grad_norm_before=gradient_metrics === nothing ?
                        nothing : _metric_float(
                            gradient_metrics.before,
                            "gradient norm before clipping",
                        ),
                    grad_norm_after=gradient_metrics === nothing ?
                        nothing : _metric_float(
                            gradient_metrics.after,
                            "gradient norm after clipping",
                        ),
                    grad_clip_scale=gradient_metrics === nothing ?
                        nothing : _metric_float(
                            gradient_metrics.scale,
                            "gradient clipping scale",
                        ),
                    validation_loss=validation_metrics === nothing ?
                        nothing : validation_metrics.loss,
                    perplexity=validation_metrics === nothing ?
                        nothing : validation_metrics.perplexity,
                ))
            end

            if max_steps !== nothing && invocation_steps >= max_steps
                return train_state, losses
            end
        end
    end

    return train_state, losses
end

"""
    train_gpt!(rng, model, loader; kwargs...)

Convenience overload that constructs `TrainerGPT`, initializes the train state,
and runs training. Returns `(trainer, train_state, losses)`.
"""
function train_gpt!(
    rng::AbstractRNG,
    model,
    loader;
    epochs::Int=1,
    learning_rate::Real=3.0f-4,
    optimizer=nothing,
    backend::Symbol=:zygote,
    xla_backend::AbstractString="gpu",
    ad=nothing,
    device=nothing,
    return_gradients=nothing,
    static_shapes::Bool=true,
    max_grad_norm=nothing,
    validation_loader=nothing,
    evaluate_every=nothing,
    callback=nothing,
)
    trainer = TrainerGPT(;
        learning_rate,
        optimizer,
        backend,
        xla_backend,
        ad,
        device,
        return_gradients,
        static_shapes,
        max_grad_norm,
    )
    train_state = init_train_state(rng, model, trainer)
    train_state, losses = train_gpt!(
        trainer,
        train_state,
        loader;
        epochs,
        validation_loader,
        evaluate_every,
        callback,
    )

    return trainer, train_state, losses
end
