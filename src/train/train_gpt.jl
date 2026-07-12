using Lux
using NNlib: gather, logsoftmax
using Optimisers
using Random: AbstractRNG
using Statistics: mean

# Loading Zygote activates Lux's AutoZygote training extension.
import Zygote

"""
    TrainerGPT(; learning_rate=3f-4, optimizer=nothing,
                 ad=Lux.Training.AutoZygote(), device=Lux.cpu_device())

Small configuration object for GPT training.

`device` is a callable device returned by `Lux.cpu_device()` or
`Lux.gpu_device()`. For NVIDIA GPU training, load `LuxCUDA` before requesting
the GPU device.
"""
struct TrainerGPT{O, A, D}
    optimizer::O
    ad::A
    device::D
end

function TrainerGPT(;
    learning_rate::Real=3.0f-4,
    optimizer=nothing,
    ad=Lux.Training.AutoZygote(),
    device=Lux.cpu_device(),
)
    learning_rate > 0 || throw(ArgumentError("`learning_rate` must be positive"))

    optimizer_rule = if optimizer === nothing
        Optimisers.Adam(Float32(learning_rate))
    else
        optimizer
    end

    return TrainerGPT(optimizer_rule, ad, device)
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
function next_token_loss(logits, targets)
    ndims(logits) == 3 ||
        throw(DimensionMismatch("`logits` must have shape (vocab_size, seq_len, batch)"))
    ndims(targets) == 2 ||
        throw(DimensionMismatch("`targets` must have shape (seq_len, batch)"))

    vocab_size, seq_len, batch_size = size(logits)

    size(targets) == (seq_len, batch_size) ||
        throw(DimensionMismatch("target shape does not match logits"))
    eltype(targets) <: Integer ||
        throw(ArgumentError("`targets` must contain integer token ids"))
    all(id -> 1 <= id <= vocab_size, targets) ||
        throw(ArgumentError("target token id is outside 1:$vocab_size"))

    log_probs = logsoftmax(logits; dims=1)

    # Julia arrays are column-major, so vec(targets) visits every sequence
    # position in batch 1, then batch 2, and so on.
    target_ids = vec(targets)
    token_positions = repeat(collect(1:seq_len), batch_size)
    batch_indices = repeat(collect(1:batch_size); inner=seq_len)

    selected_log_probs = gather(
        log_probs,
        target_ids,
        token_positions,
        batch_indices,
    )

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
    ps, st = Lux.setup(rng, model)
    ps, st = trainer.device((ps, st))

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

    # Inputs follow the parameters to the selected device. Targets deliberately
    # remain on the host; NNlib.gather moves its index array to the source
    # backend when logits live on a GPU.
    x_device = trainer.device(x)

    gradients, loss, _, train_state = Lux.Training.single_train_step!(
        trainer.ad,
        _gpt_objective,
        (x_device, targets),
        train_state,
    )

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

    for epoch in 1:epochs
        for (batch_index, batch) in enumerate(loader)
            train_state, loss, _ = train_step!(trainer, train_state, batch)
            loss_value = Float32(loss)
            push!(losses, loss_value)

            if callback !== nothing
                callback((;
                    epoch,
                    batch=batch_index,
                    step=train_state.step,
                    loss=loss_value,
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
    ad=Lux.Training.AutoZygote(),
    device=Lux.cpu_device(),
    callback=nothing,
)
    trainer = TrainerGPT(;
        learning_rate,
        optimizer,
        ad,
        device,
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
