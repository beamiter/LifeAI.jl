using Lux
using Serialization

const CHECKPOINT_FORMAT_VERSION = 2
const _SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = (1, CHECKPOINT_FORMAT_VERSION)

function _migrate_v1_model_config(config::NamedTuple)
    return merge(
        config,
        (;
            norm_type=:layernorm,
            mlp_type=:gelu,
            tie_embeddings=false,
        ),
    )
end

function _migrate_checkpoint_payload(payload::NamedTuple)
    source_version = Int(payload.format_version)

    if source_version == CHECKPOINT_FORMAT_VERSION
        return payload, source_version
    elseif source_version == 1
        migrated = merge(
            payload,
            (;
                format_version=CHECKPOINT_FORMAT_VERSION,
                model_config=_migrate_v1_model_config(payload.model_config),
            ),
        )
        return migrated, source_version
    end

    supported = join(_SUPPORTED_CHECKPOINT_FORMAT_VERSIONS, ", ")
    throw(ArgumentError(
        "unsupported checkpoint format version $source_version; supported: $supported",
    ))
end

function _tokenizer_payload(tokenizer::Tokenizer)
    return (;
        type=:character,
        id_to_token=copy(tokenizer.id_to_token),
        unk_id=tokenizer.unk_id,
    )
end

function _tokenizer_from_payload(payload)
    payload.type === :character ||
        throw(ArgumentError("unsupported tokenizer type $(repr(payload.type))"))

    id_to_token = Char.(collect(payload.id_to_token))
    token_to_id = Dict(token => id for (id, token) in enumerate(id_to_token))
    unk_id = payload.unk_id === nothing ? nothing : Int(payload.unk_id)
    return Tokenizer(token_to_id, id_to_token, unk_id)
end

function _normalize_progress(progress, step::Int)
    epoch = hasproperty(progress, :epoch) ? Int(progress.epoch) : 0
    batch = hasproperty(progress, :batch) ? Int(progress.batch) : 0

    epoch >= 0 || throw(ArgumentError("checkpoint progress epoch must be non-negative"))
    batch >= 0 || throw(ArgumentError("checkpoint progress batch must be non-negative"))

    return (; epoch, batch, step)
end

"""
    save_checkpoint(path, model, tokenizer, trainer, train_state; kwargs...)

Save a versioned, device-independent experiment checkpoint.

The payload includes architecture configuration, tokenizer vocabulary,
parameters, Lux states, optimizer rule/state, global step, data progress, RNG,
training configuration, metrics, and optional metadata. Device arrays are moved
to the CPU before serialization. The file is replaced atomically.
"""
function save_checkpoint(
    path::AbstractString,
    model::GPTModel,
    tokenizer::Tokenizer,
    trainer::TrainerGPT,
    train_state;
    rng=nothing,
    progress=(; epoch=0, batch=0),
    train_config=NamedTuple(),
    metrics=NamedTuple(),
    metadata=NamedTuple(),
)
    isempty(path) && throw(ArgumentError("checkpoint path must not be empty"))
    train_state.model === model ||
        throw(ArgumentError("`train_state.model` must be the supplied `model`"))

    host = Lux.cpu_device()
    payload = (;
        format_version=CHECKPOINT_FORMAT_VERSION,
        model_config=gpt_config(model),
        tokenizer=_tokenizer_payload(tokenizer),
        parameters=host(train_state.parameters),
        states=host(train_state.states),
        optimizer=trainer.optimizer,
        optimizer_state=host(train_state.optimizer_state),
        step=Int(train_state.step),
        trainer_config=(;
            backend=trainer.backend,
            return_gradients=trainer.return_gradients,
            static_shapes=trainer.static_shapes,
            max_grad_norm=trainer.max_grad_norm,
        ),
        progress=_normalize_progress(progress, Int(train_state.step)),
        rng=rng === nothing ? nothing : deepcopy(rng),
        train_config=deepcopy(train_config),
        metrics=deepcopy(metrics),
        metadata=deepcopy(metadata),
    )

    absolute_path = abspath(path)
    directory = dirname(absolute_path)
    mkpath(directory)
    temporary_path = tempname(directory)

    try
        open(temporary_path, "w") do io
            serialize(io, payload)
        end
        mv(temporary_path, absolute_path; force=true)
    finally
        isfile(temporary_path) && rm(temporary_path; force=true)
    end

    return absolute_path
end

function _restore_train_state(model, trainer, payload)
    parameters, states, optimizer_state = trainer.device((
        payload.parameters,
        payload.states,
        payload.optimizer_state,
    ))

    # Use Lux's public constructor first so the target backend can prepare its
    # optimizer rule and allocator cache. Then replace the initialized optimizer
    # state and step with the checkpoint values.
    base_state = Lux.Training.TrainState(
        model,
        parameters,
        states,
        trainer.optimizer,
    )

    return Lux.Training.TrainState(
        base_state.cache,
        base_state.objective_function,
        base_state.allocator_cache,
        base_state.model,
        base_state.parameters,
        base_state.states,
        base_state.optimizer,
        optimizer_state,
        Int(payload.step),
    )
end

"""
    load_checkpoint(path; kwargs...)

Load a checkpoint and rebuild the model, tokenizer, trainer, and Lux train state.

By default trainer settings are restored from the checkpoint. Pass `backend`,
`device`, `ad`, `return_gradients`, `static_shapes`, or `max_grad_norm` to
override them for the destination machine.
"""
function load_checkpoint(
    path::AbstractString;
    backend::Symbol=:checkpoint,
    xla_backend::AbstractString="gpu",
    device=nothing,
    ad=nothing,
    return_gradients=:checkpoint,
    static_shapes=:checkpoint,
    max_grad_norm=:checkpoint,
)
    isfile(path) || throw(ArgumentError("checkpoint does not exist: $path"))

    raw_payload = open(path, "r") do io
        deserialize(io)
    end
    raw_payload isa NamedTuple ||
        throw(ArgumentError("checkpoint payload must be a named tuple"))
    hasproperty(raw_payload, :format_version) ||
        throw(ArgumentError("checkpoint has no format version"))
    payload, source_format_version = _migrate_checkpoint_payload(raw_payload)

    model = GPTModel(payload.model_config)
    tokenizer = _tokenizer_from_payload(payload.tokenizer)
    saved = payload.trainer_config

    resolved_backend = backend === :checkpoint ? saved.backend : backend
    resolved_return_gradients = return_gradients === :checkpoint ?
        saved.return_gradients : return_gradients
    resolved_static_shapes = static_shapes === :checkpoint ?
        saved.static_shapes : static_shapes
    resolved_max_grad_norm = max_grad_norm === :checkpoint ?
        saved.max_grad_norm : max_grad_norm

    resolved_return_gradients isa Bool ||
        throw(ArgumentError("`return_gradients` must be Bool or `:checkpoint`"))
    resolved_static_shapes isa Bool ||
        throw(ArgumentError("`static_shapes` must be Bool or `:checkpoint`"))

    trainer = TrainerGPT(;
        optimizer=payload.optimizer,
        backend=resolved_backend,
        xla_backend,
        device,
        ad,
        return_gradients=resolved_return_gradients,
        static_shapes=resolved_static_shapes,
        max_grad_norm=resolved_max_grad_norm,
    )
    train_state = _restore_train_state(model, trainer, payload)

    return (;
        format_version=payload.format_version,
        source_format_version,
        model,
        tokenizer,
        trainer,
        train_state,
        rng=payload.rng,
        progress=payload.progress,
        train_config=payload.train_config,
        metrics=payload.metrics,
        metadata=payload.metadata,
    )
end

"""
    resume_gpt!(checkpoint, loader; epochs=1, kwargs...)

Resume from the next unprocessed batch recorded in `checkpoint.progress`.
`checkpoint` is the named tuple returned by [`load_checkpoint`](@ref).
"""
function resume_gpt!(
    checkpoint,
    loader;
    epochs::Int=1,
    max_steps=nothing,
    validation_loader=nothing,
    evaluate_every=nothing,
    callback=nothing,
)
    hasproperty(checkpoint, :trainer) ||
        throw(ArgumentError("checkpoint has no trainer"))
    hasproperty(checkpoint, :train_state) ||
        throw(ArgumentError("checkpoint has no train_state"))
    hasproperty(checkpoint, :progress) ||
        throw(ArgumentError("checkpoint has no progress"))

    progress = checkpoint.progress
    start_epoch = max(1, Int(progress.epoch))
    start_batch = Int(progress.batch) + 1

    if start_batch > length(loader)
        start_epoch += 1
        start_batch = 1
    end

    return train_gpt!(
        checkpoint.trainer,
        checkpoint.train_state,
        loader;
        epochs,
        start_epoch,
        start_batch,
        max_steps,
        validation_loader,
        evaluate_every,
        callback,
    )
end
