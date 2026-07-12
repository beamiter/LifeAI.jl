using Lux
using NNlib: gather, logsoftmax
using Optimisers
using Random: AbstractRNG
using Statistics: mean

# Loading these packages activates Lux's Zygote and Reactant/Enzyme extensions.
import Enzyme
import Reactant
import Zygote

"""
    TrainerGPT(; learning_rate=3f-4, optimizer=nothing, backend=:zygote,
                 xla_backend="gpu", device=nothing, ad=nothing,
                 return_gradients=nothing, static_shapes=true)

Small configuration object for GPT training.

Backends:

- `backend=:zygote`: ordinary Lux training, defaulting to the CPU.
- `backend=:xla`: move parameters, states, optimizer state, and batches to a
  Reactant device. Lux automatically compiles the complete Enzyme gradient and
  optimizer update into one XLA train step on the first batch, then caches and
  reuses it for later batches with the same shape.

`xla_backend` may be `"gpu"`, `"cpu"`, or `"tpu"`. Returning gradients is
disabled by default for XLA because it adds an unnecessary output and memory
cost when the caller only needs the loss.
"""
struct TrainerGPT{O, A, D}
    optimizer::O
    ad::A
    device::D
    backend::Symbol
    return_gradients::Bool
    static_shapes::Bool
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
)
    learning_rate > 0 || throw(ArgumentError("`learning_rate` must be positive"))

    optimizer_rule = if optimizer === nothing
        Optimisers.Adam(Float32(learning_rate))
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
        Bool(return_gradients)
    end

    return TrainerGPT(
        optimizer_rule,
        ad,
        device,
        backend,
        return_gradients_flag,
        static_shapes,
    )
end

"""
    next_token_loss(logits, targets)

Sparse next-token cross entropy.

Shapes:

    logits:  (vocab_size, seq_len, batch)
    targets: (seq_len, batch)

Targets use the package's 1-based token ids. The implementation uses
`NNlib.gather`, so it does not materialize a dense one-hot target tensor.
"""
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

function next_token_loss(logits, targets)
    ndims(logits) == 3 ||
        throw(DimensionMismatch("`logits` must have shape (vocab_size, seq_len, batch)"))
    ndims(targets) == 2 ||
        throw(DimensionMismatch("`targets` must have shape (seq_len, batch)"))

    vocab_size, seq_len, batch_size = size(logits)

    size(targets) == (seq_len, batch_size) ||
        throw(DimensionMismatch("target shape does not match logits"))
    _validate_target_ids(targets, vocab_size)

    log_probs = logsoftmax(logits; dims=1)

    # Flatten to a single indexed lookup. This avoids NNlib's CartesianIndex
    # construction, which is not supported for Reactant traced integer arrays.
    # Julia's column-major layout keeps each (sequence, batch) position aligned.
    target_ids = vec(targets)
    token_count = seq_len * batch_size
    linear_indices = target_ids .+ vocab_size .* (0:(token_count - 1))
    log_probs_flat = Reactant.materialize_traced_array(vec(log_probs))
    selected_log_probs = gather(log_probs_flat, linear_indices)

    return -mean(selected_log_probs)
end

"""
    next_token_loss(model, ps, st, x, targets)

Run the model and return `(loss, updated_state)`.
"""
function next_token_loss(model, ps, st, x, targets)
    logits, st_new = model(x, ps, st)
    return next_token_loss(logits, targets), st_new
end

"""
    init_train_state(rng, model, trainer)

Initialize model parameters/states, move them to `trainer.device`, and create a
Lux `TrainState`.
"""
function init_train_state(rng::AbstractRNG, model, trainer::TrainerGPT)
    # Move ps/st before constructing TrainState. Reactant then creates the
    # optimizer state directly on the XLA device instead of adapting it later.
    ps, st = trainer.device(Lux.setup(rng, model))

    return Lux.Training.TrainState(
        model,
        ps,
        st,
        trainer.optimizer,
    )
end

function _gpt_objective(model, ps, st, data)
    x, targets = data
    loss, st_new = next_token_loss(model, ps, st, x, targets)
    return loss, st_new, NamedTuple()
end

"""
    train_step!(trainer, train_state, batch)
    train_step!(trainer, train_state, x, targets)

Perform one optimizer step.

Returns:

    updated_train_state, loss, gradients
"""
function train_step!(trainer::TrainerGPT, train_state, batch)
    length(batch) == 2 || throw(ArgumentError("`batch` must be `(x, targets)`"))
    x, targets = batch

    # Perform data-dependent validation while ids are still ordinary host
    # arrays. The compiled objective contains only tensor operations.
    _validate_token_ids(x, train_state.model.vocab_size)
    _validate_target_ids(targets, train_state.model.vocab_size)

    x_device, targets_device = trainer.device((x, targets))
    device_batch = (x_device, targets_device)

    gradients, loss, _, train_state = if trainer.return_gradients
        Lux.Training.single_train_step!(
            trainer.ad,
            _gpt_objective,
            device_batch,
            train_state,
        )
    else
        Lux.Training.single_train_step!(
            trainer.ad,
            _gpt_objective,
            device_batch,
            train_state;
            return_gradients=Val(false),
        )
    end

    return train_state, loss, gradients
end

function train_step!(trainer::TrainerGPT, train_state, x, targets)
    return train_step!(trainer, train_state, (x, targets))
end

"""
    train_gpt!(trainer, train_state, loader; epochs=1, callback=nothing)

Train over `loader` for the requested number of epochs.

The optional callback receives a named tuple with `epoch`, `batch`, `step`, and
`loss`.
"""
function train_gpt!(
    trainer::TrainerGPT,
    train_state,
    loader;
    epochs::Int=1,
    callback=nothing,
)
    epochs > 0 || throw(ArgumentError("`epochs` must be positive"))

    losses = Float32[]
    expected_batch_shape = nothing

    for epoch in 1:epochs
        for (batch_index, batch) in enumerate(loader)
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
            train_state, loss, _ = train_step!(trainer, train_state, batch)
            step_seconds = Float64(time_ns() - step_start) / 1.0e9

            loss_value = Float32(loss)
            push!(losses, loss_value)

            if callback !== nothing
                callback((;
                    epoch,
                    batch=batch_index,
                    step=train_state.step,
                    loss=loss_value,
                    step_seconds,
                    xla_compilation=(trainer.backend === :xla && train_state.step == 1),
                ))
            end
        end
    end

    return train_state, losses
end

"""
    train_gpt!(rng, model, loader; kwargs...)

Convenience overload that constructs `TrainerGPT`, initializes the train state,
and runs training.

Returns:

    trainer, train_state, losses
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
    )
    train_state = init_train_state(rng, model, trainer)
    train_state, losses = train_gpt!(
        trainer,
        train_state,
        loader;
        epochs,
        callback,
    )

    return trainer, train_state, losses
end
