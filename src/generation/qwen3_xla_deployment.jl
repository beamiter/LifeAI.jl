using BFloat16s: BFloat16
using Random: AbstractRNG, default_rng
using Reactant

"""
    Qwen3XLAWindowPlan(
        context_tokens, prompt_tokens, max_new_tokens, chunk_tokens)

Host-side request plan for the fixed-shape XLA prefill path. Prompts are
left-padded to a multiple of `chunk_tokens`; padding cache slots are excluded
through a per-request key-position vector. Bucket, padding, sequence, and cache
sizes are derived by the constructor and cannot be supplied inconsistently.
"""
struct Qwen3XLAWindowPlan
    context_tokens::Int
    prompt_tokens::Int
    max_new_tokens::Int
    chunk_tokens::Int
    prompt_bucket_tokens::Int
    left_padding_tokens::Int
    sequence_tokens::Int
    cache_tokens::Int

    function Qwen3XLAWindowPlan(
        context_tokens,
        prompt_tokens,
        max_new_tokens,
        chunk_tokens,
    )
        context = _strict_host_int(context_tokens, "context_tokens")
        prompt = _strict_host_int(prompt_tokens, "prompt_tokens")
        output = _strict_host_int(max_new_tokens, "max_new_tokens")
        chunk = _strict_host_int(chunk_tokens, "chunk_tokens")
        context > 0 || throw(ArgumentError("context_tokens must be positive"))
        context <= typemax(Int32) || throw(ArgumentError(
            "context_tokens must fit in Int32 device positions",
        ))
        prompt > 0 || throw(ArgumentError("prompt_tokens must be positive"))
        output > 0 || throw(ArgumentError("max_new_tokens must be positive"))
        0 < chunk <= context || throw(ArgumentError(
            "chunk_tokens must be in 1:context_tokens",
        ))
        output <= context || throw(ArgumentError(
            "max_new_tokens exceeds context_tokens",
        ))
        prompt <= context - output || throw(ArgumentError(
            "prompt plus requested output exceeds context_tokens",
        ))

        bucket = try
            Base.Checked.checked_mul(cld(prompt, chunk), chunk)
        catch error
            error isa OverflowError || rethrow()
            throw(ArgumentError(
                "padded XLA prompt bucket exceeds the host integer range",
            ))
        end
        bucket <= context - output || throw(ArgumentError(
            "padded XLA prompt bucket plus requested output exceeds " *
            "context_tokens",
        ))
        return new(
            context,
            prompt,
            output,
            chunk,
            bucket,
            bucket - prompt,
            prompt + output,
            bucket + output - 1,
        )
    end
end

function Qwen3XLAWindowPlan(
    context_tokens,
    prompt_tokens,
    max_new_tokens,
    chunk_tokens,
    prompt_bucket_tokens,
    left_padding_tokens,
    sequence_tokens,
    cache_tokens,
)
    plan = Qwen3XLAWindowPlan(
        context_tokens,
        prompt_tokens,
        max_new_tokens,
        chunk_tokens,
    )
    bucket = _strict_host_int(prompt_bucket_tokens, "prompt_bucket_tokens")
    padding = _strict_host_int(left_padding_tokens, "left_padding_tokens")
    sequence = _strict_host_int(sequence_tokens, "sequence_tokens")
    cache = _strict_host_int(cache_tokens, "cache_tokens")
    bucket == plan.prompt_bucket_tokens || throw(ArgumentError(
        "prompt_bucket_tokens is inconsistent with prompt_tokens and chunk_tokens",
    ))
    padding == plan.left_padding_tokens || throw(ArgumentError(
        "left_padding_tokens is inconsistent with prompt_bucket_tokens",
    ))
    sequence == plan.sequence_tokens || throw(ArgumentError(
        "sequence_tokens is inconsistent with prompt_tokens and max_new_tokens",
    ))
    cache == plan.cache_tokens || throw(ArgumentError(
        "cache_tokens is inconsistent with the padded prompt and requested output",
    ))
    return plan
end

"""
    plan_qwen3_xla_window(
        prompt_tokens, max_new_tokens;
        context_tokens=4096, chunk_tokens=64,
    )

Validate the exact logical and physical context budget without overflow.
`cache_tokens` is one less than the padded prompt plus requested output because
the final selected token does not need to be written back to K/V.
"""
function plan_qwen3_xla_window(
    prompt_tokens,
    max_new_tokens;
    context_tokens=4096,
    chunk_tokens=64,
)
    return Qwen3XLAWindowPlan(
        context_tokens,
        prompt_tokens,
        max_new_tokens,
        chunk_tokens,
    )
end

function qwen3_xla_pad_prompt(
    prompt_tokens,
    plan::Qwen3XLAWindowPlan;
    pad_token_id::Integer=1,
)
    prompt = vec(_strict_host_int_array(
        prompt_tokens,
        "Qwen3 XLA prompt token",
    ))
    length(prompt) == plan.prompt_tokens || throw(ArgumentError(
        "prompt token count does not match the XLA window plan",
    ))
    pad = _strict_host_int(pad_token_id, "pad_token_id")
    pad > 0 || throw(ArgumentError("pad_token_id must be positive"))
    return vcat(fill(pad, plan.left_padding_tokens), prompt)
end

function qwen3_xla_key_positions(plan::Qwen3XLAWindowPlan)
    positions = Int32.(collect(1:plan.context_tokens))
    if plan.left_padding_tokens > 0
        positions[1:plan.left_padding_tokens] .= typemax(Int32)
    end
    return positions
end

function _qwen3_xla_generated_token(value, vocab_size::Int)
    token = _strict_host_int(value, "Qwen3 XLA generated token")
    _validate_generation_ids((token,), vocab_size)
    return token
end

function _qwen3_xla_host_token(output_state, vocab_size::Int)
    values = vec(Array(output_state))
    length(values) == 1 || throw(ArgumentError(
        "Qwen3 XLA generated-token output must contain exactly one value",
    ))
    return _qwen3_xla_generated_token(only(values), vocab_size)
end

_qwen3_xla_tensor_bytes(x::AbstractArray) =
    length(x) * sizeof(eltype(x))
_qwen3_xla_tensor_bytes(x::NamedTuple) =
    sum(_qwen3_xla_tensor_bytes, values(x); init=0)
_qwen3_xla_tensor_bytes(x::Tuple) =
    sum(_qwen3_xla_tensor_bytes, values(x); init=0)
_qwen3_xla_tensor_bytes(x) = 0

_qwen3_xla_tensor_leaves(x::AbstractArray) = 1
_qwen3_xla_tensor_leaves(x::NamedTuple) =
    sum(_qwen3_xla_tensor_leaves, values(x); init=0)
_qwen3_xla_tensor_leaves(x::Tuple) =
    sum(_qwen3_xla_tensor_leaves, values(x); init=0)
_qwen3_xla_tensor_leaves(x) = 0

function _qwen3_xla_allocator_snapshot()
    stats = try
        Reactant.XLA.allocatorstats()
    catch
        return nothing
    end
    return (;
        num_allocs=stats.num_allocs,
        bytes_in_use=stats.bytes_in_use,
        peak_bytes_in_use=stats.peak_bytes_in_use,
        largest_alloc_size=stats.largest_alloc_size,
        bytes_limit=stats.bytes_limit,
        bytes_reserved=stats.bytes_reserved,
        peak_bytes_reserved=stats.peak_bytes_reserved,
        bytes_reservable_limit=stats.bytes_reservable_limit,
        largest_free_block_bytes=stats.largest_free_block_bytes,
        pool_bytes=stats.pool_bytes,
        peak_pool_bytes=stats.peak_pool_bytes,
    )
end

"""
    HFQwen3BF16XLASession

Reusable batch-1 XLA runtime with one packed parameter tree, one compiled
64-token prefill executable, one compiled single-token decode executable and a
fixed-capacity BF16 K/V cache.
"""
struct _HFQwen3BF16XLASessionValidated end
const _HF_QWEN3_BF16_XLA_SESSION_VALIDATED =
    _HFQwen3BF16XLASessionValidated()

struct _HFQwen3BF16XLASourceSignature
    eos_ids::Tuple
    temperature::Float32
    top_k::Int
    top_p::Float32
end

struct _HFQwen3BF16XLASessionContract
    model::Any
    parameters::Any
    tokenizer::Any
    generation_config::Any
    cos_table::Any
    sin_table::Any
    key_caches::Tuple
    value_caches::Tuple
    compiled_prefill::Any
    compiled_decode::Any
    strategy::Symbol
    context_tokens::Int
    prefill_chunk_tokens::Int
    sample_top_k::Int
    source_signature::_HFQwen3BF16XLASourceSignature
end

mutable struct HFQwen3BF16XLASession
    model::Any
    parameters::Any
    tokenizer::Any
    generation_config::Any
    cos_table::Any
    sin_table::Any
    key_caches::Any
    value_caches::Any
    compiled_prefill::Any
    compiled_decode::Any
    strategy::Symbol
    context_tokens::Int
    prefill_chunk_tokens::Int
    sample_top_k::Int
    position::Int
    load_metrics::Any
    contract::_HFQwen3BF16XLASessionContract

    function HFQwen3BF16XLASession(
        ::_HFQwen3BF16XLASessionValidated,
        fields...,
    )
        return new(fields...)
    end
end

function _qwen3_xla_session_model_integer(model, name::Symbol)
    hasproperty(model, name) || throw(ArgumentError(
        "Qwen3 XLA session model must expose a $(repr(name)) field",
    ))
    value = _strict_host_int(
        getproperty(model, name),
        "Qwen3 XLA session model $(String(name))",
    )
    value > 0 || throw(ArgumentError(
        "Qwen3 XLA session model $(String(name)) must be positive",
    ))
    return value
end

function _qwen3_xla_session_metadata(
    model,
    strategy,
    context_tokens,
    prefill_chunk_tokens,
    sample_top_k,
    position,
)
    strategy isa Symbol || throw(ArgumentError(
        "Qwen3 XLA session strategy must be a Symbol",
    ))
    strategy in (:greedy, :sample, :device_sample) || throw(ArgumentError(
        "Qwen3 XLA session strategy must be :greedy, :sample, or :device_sample",
    ))
    context = _strict_host_int(context_tokens, "Qwen3 XLA session context_tokens")
    chunk = _strict_host_int(
        prefill_chunk_tokens,
        "Qwen3 XLA session prefill_chunk_tokens",
    )
    compiled_top_k = _strict_host_int(
        sample_top_k,
        "Qwen3 XLA session sample_top_k",
    )
    cache_position = _strict_host_int(
        position,
        "Qwen3 XLA session position",
    )
    context > 0 || throw(ArgumentError(
        "Qwen3 XLA session context_tokens must be positive",
    ))
    context <= typemax(Int32) || throw(ArgumentError(
        "Qwen3 XLA session context_tokens must fit in Int32 device positions",
    ))
    0 < chunk <= context || throw(ArgumentError(
        "Qwen3 XLA session prefill_chunk_tokens must be in 1:context_tokens",
    ))
    context % chunk == 0 || throw(ArgumentError(
        "Qwen3 XLA session context_tokens must be divisible by prefill_chunk_tokens",
    ))
    0 <= cache_position <= context || throw(ArgumentError(
        "Qwen3 XLA session position must be in 0:context_tokens",
    ))

    vocab_size = _qwen3_xla_session_model_integer(model, :vocab_size)
    model_context = _qwen3_xla_session_model_integer(model, :max_seq_len)
    _qwen3_xla_session_model_integer(model, :num_layers)
    head_dim = _qwen3_xla_session_model_integer(model, :head_dim)
    _qwen3_xla_session_model_integer(model, :num_kv_heads)
    iseven(head_dim) || throw(ArgumentError(
        "Qwen3 XLA session model head_dim must be even",
    ))
    if strategy === :device_sample
        0 < compiled_top_k <= vocab_size || throw(ArgumentError(
            "Qwen3 XLA session sample_top_k must be in 1:model.vocab_size " *
            "for :device_sample",
        ))
    else
        compiled_top_k == 0 || throw(ArgumentError(
            "Qwen3 XLA session sample_top_k must be zero unless strategy " *
            "is :device_sample",
        ))
    end
    context == model_context || throw(DimensionMismatch(
        "Qwen3 XLA session context_tokens must match model.max_seq_len",
    ))
    return (;
        strategy,
        context_tokens=context,
        prefill_chunk_tokens=chunk,
        sample_top_k=compiled_top_k,
        position=cache_position,
        vocab_size,
    )
end

function _qwen3_xla_session_source_signature(
    tokenizer,
    generation_config,
    strategy::Symbol,
    vocab_size::Int,
)
    hasproperty(tokenizer, :eos_ids) || throw(ArgumentError(
        "Qwen3 XLA session tokenizer must expose an :eos_ids field",
    ))
    eos_ids = Tuple(vec(_strict_host_int_array(
        tokenizer.eos_ids,
        "Qwen3 XLA session tokenizer EOS token",
    )))
    _qwen3_stop_token_set(eos_ids, vocab_size)
    for name in (:temperature, :top_k, :top_p)
        hasproperty(generation_config, name) || throw(ArgumentError(
            "Qwen3 XLA session generation_config must expose a " *
            "$(repr(name)) field",
        ))
    end
    options = _qwen3_session_sampling_options(
        generation_config.temperature,
        generation_config.top_k,
        generation_config.top_p,
    )
    if strategy === :device_sample
        _validate_device_sampling_options(;
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            vocab_size,
        )
    end
    return _HFQwen3BF16XLASourceSignature(
        eos_ids,
        options.temperature,
        options.top_k,
        options.top_p,
    )
end

function _qwen3_xla_session_array(
    value,
    shape::Tuple,
    label::AbstractString,
    device,
)
    value isa AbstractArray || throw(ArgumentError("$label must be an array"))
    ndims(value) == length(shape) || throw(DimensionMismatch(
        "$label must be $(length(shape))-dimensional",
    ))
    all(
        dimension -> axes(value, dimension) == Base.OneTo(size(value, dimension)),
        eachindex(shape),
    ) || throw(ArgumentError("$label must use one-based axes"))
    size(value) == shape || throw(DimensionMismatch(
        "$label must have shape $shape; got $(size(value))",
    ))
    eltype(value) === BFloat16 || throw(ArgumentError(
        "$label must contain BFloat16 values",
    ))
    get_device(value) == device || throw(ArgumentError(
        "$label must use the Qwen3 XLA session device",
    ))
    return nothing
end

function _qwen3_xla_record_session_storage!(
    storages::Vector{Tuple{String,Any}},
    label::AbstractString,
    storage::AbstractArray,
)
    resolved_label = String(label)
    for (existing_label, existing) in storages
        Base.mightalias(storage, existing) && throw(ArgumentError(
            "Qwen3 XLA session $resolved_label and $existing_label " *
            "must use non-overlapping storage",
        ))
    end
    push!(storages, (resolved_label, storage))
    return nothing
end

function _qwen3_xla_validate_session_storage(
    model,
    cos_table,
    sin_table,
    key_caches,
    value_caches,
    context_tokens::Int,
)
    key_caches isa Tuple || throw(ArgumentError(
        "Qwen3 XLA session key_caches must be a tuple",
    ))
    value_caches isa Tuple || throw(ArgumentError(
        "Qwen3 XLA session value_caches must be a tuple",
    ))
    isempty(key_caches) && throw(ArgumentError(
        "Qwen3 XLA session cache tuples must not be empty",
    ))
    length(key_caches) == length(value_caches) || throw(DimensionMismatch(
        "Qwen3 XLA session key/value cache tuple lengths must match",
    ))
    num_layers = _qwen3_xla_session_model_integer(model, :num_layers)
    length(key_caches) == num_layers || throw(DimensionMismatch(
        "Qwen3 XLA session cache layer count must match model.num_layers",
    ))

    first_keys = first(key_caches)
    first_keys isa AbstractArray || throw(ArgumentError(
        "Qwen3 XLA session cache storage must be arrays",
    ))
    ndims(first_keys) == 4 || throw(DimensionMismatch(
        "Qwen3 XLA session cache storage must be four-dimensional",
    ))
    head_dim = _qwen3_xla_session_model_integer(model, :head_dim)
    num_kv_heads = _qwen3_xla_session_model_integer(model, :num_kv_heads)
    cache_shape = (head_dim, num_kv_heads, context_tokens, 1)
    device = get_device(first_keys)
    # Pairwise alias checks are intentionally confined to construction and
    # full request/reset preflight. Per-chunk contract validation remains O(1).
    storages = Tuple{String,Any}[]
    for index in eachindex(key_caches)
        for (kind, storage) in (
            ("key", key_caches[index]),
            ("value", value_caches[index]),
        )
            _qwen3_xla_session_array(
                storage,
                cache_shape,
                "Qwen3 XLA session layer $index $kind cache",
                device,
            )
            _qwen3_xla_record_session_storage!(
                storages,
                "layer $index $kind cache",
                storage,
            )
        end
    end

    rope_shape = (head_dim ÷ 2, context_tokens)
    _qwen3_xla_session_array(
        cos_table,
        rope_shape,
        "Qwen3 XLA session cosine table",
        device,
    )
    _qwen3_xla_session_array(
        sin_table,
        rope_shape,
        "Qwen3 XLA session sine table",
        device,
    )
    _qwen3_xla_record_session_storage!(
        storages,
        "cosine table",
        cos_table,
    )
    _qwen3_xla_record_session_storage!(
        storages,
        "sine table",
        sin_table,
    )
    return nothing
end

function _qwen3_xla_session_contract(
    model,
    parameters,
    tokenizer,
    generation_config,
    cos_table,
    sin_table,
    key_caches::Tuple,
    value_caches::Tuple,
    compiled_prefill,
    compiled_decode,
    metadata,
    source_signature,
)
    return _HFQwen3BF16XLASessionContract(
        model,
        parameters,
        tokenizer,
        generation_config,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        compiled_prefill,
        compiled_decode,
        metadata.strategy,
        metadata.context_tokens,
        metadata.prefill_chunk_tokens,
        metadata.sample_top_k,
        source_signature,
    )
end

function _qwen3_xla_validate_session_contract(
    session::HFQwen3BF16XLASession,
)
    contract = session.contract
    for (name, current, original) in (
        ("model", session.model, contract.model),
        ("parameters", session.parameters, contract.parameters),
        ("tokenizer", session.tokenizer, contract.tokenizer),
        ("generation_config", session.generation_config, contract.generation_config),
        ("cosine table", session.cos_table, contract.cos_table),
        ("sine table", session.sin_table, contract.sin_table),
        ("key cache tuple", session.key_caches, contract.key_caches),
        ("value cache tuple", session.value_caches, contract.value_caches),
        ("compiled prefill", session.compiled_prefill, contract.compiled_prefill),
        ("compiled decode", session.compiled_decode, contract.compiled_decode),
    )
        current === original || throw(ArgumentError(
            "Qwen3 XLA session $name changed after compilation",
        ))
    end
    metadata = _qwen3_xla_session_metadata(
        session.model,
        session.strategy,
        session.context_tokens,
        session.prefill_chunk_tokens,
        session.sample_top_k,
        session.position,
    )
    for name in (
        :strategy,
        :context_tokens,
        :prefill_chunk_tokens,
        :sample_top_k,
    )
        getproperty(metadata, name) == getproperty(contract, name) ||
            throw(ArgumentError(
                "Qwen3 XLA session $(String(name)) changed after compilation",
            ))
    end
    source_signature = _qwen3_xla_session_source_signature(
        session.tokenizer,
        session.generation_config,
        metadata.strategy,
        metadata.vocab_size,
    )
    expected_signature = contract.source_signature
    source_signature.eos_ids == expected_signature.eos_ids &&
        isequal(source_signature.temperature, expected_signature.temperature) &&
        source_signature.top_k == expected_signature.top_k &&
        isequal(source_signature.top_p, expected_signature.top_p) ||
        throw(ArgumentError(
            "Qwen3 XLA session tokenizer or generation metadata changed " *
            "after compilation",
        ))
    return metadata
end

function _qwen3_xla_validate_session(session::HFQwen3BF16XLASession)
    _qwen3_xla_validate_session_contract(session)
    _qwen3_xla_validate_session_storage(
        session.model,
        session.cos_table,
        session.sin_table,
        session.key_caches,
        session.value_caches,
        session.context_tokens,
    )
    return nothing
end

"""
    HFQwen3BF16XLASession(; ...)

Construct a validated XLA session from an already compiled runtime. The keyword
form keeps test doubles injectable without exposing the mutable struct's raw
field constructor.
"""
function HFQwen3BF16XLASession(;
    model,
    parameters,
    tokenizer,
    generation_config,
    cos_table,
    sin_table,
    key_caches,
    value_caches,
    compiled_prefill,
    compiled_decode,
    strategy=:greedy,
    context_tokens,
    prefill_chunk_tokens,
    sample_top_k=0,
    position=0,
    load_metrics=nothing,
)
    metadata = _qwen3_xla_session_metadata(
        model,
        strategy,
        context_tokens,
        prefill_chunk_tokens,
        sample_top_k,
        position,
    )
    compiled_prefill === nothing && throw(ArgumentError(
        "Qwen3 XLA session must provide compiled_prefill",
    ))
    compiled_decode === nothing && throw(ArgumentError(
        "Qwen3 XLA session must provide compiled_decode",
    ))
    source_signature = _qwen3_xla_session_source_signature(
        tokenizer,
        generation_config,
        metadata.strategy,
        metadata.vocab_size,
    )
    _qwen3_xla_validate_session_storage(
        model,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        metadata.context_tokens,
    )
    contract = _qwen3_xla_session_contract(
        model,
        parameters,
        tokenizer,
        generation_config,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        compiled_prefill,
        compiled_decode,
        metadata,
        source_signature,
    )
    return HFQwen3BF16XLASession(
        _HF_QWEN3_BF16_XLA_SESSION_VALIDATED,
        model,
        parameters,
        tokenizer,
        generation_config,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        compiled_prefill,
        compiled_decode,
        metadata.strategy,
        metadata.context_tokens,
        metadata.prefill_chunk_tokens,
        metadata.sample_top_k,
        metadata.position,
        load_metrics,
        contract,
    )
end

"""
    load_hf_qwen3_bf16_xla_session(
        model_dir;
        context_tokens=4096,
        prefill_chunk_tokens=64,
        strategy=:greedy,
        revision="",
        variant=nothing,
    )

Stream a compact host parameter tree, recursively transfer that tree exactly
once, allocate static K/V, and compile fixed-shape prefill/decode executables.
The ordinary unpacked parameter topology is never built or transferred.
Callers must select the desired Reactant backend before invoking this function.
"""
function load_hf_qwen3_bf16_xla_session(
    model_dir::AbstractString;
    context_tokens::Integer=4096,
    prefill_chunk_tokens::Integer=64,
    strategy::Symbol=:greedy,
    sample_top_k=nothing,
    revision::AbstractString="",
    variant=nothing,
)
    strategy in (:greedy, :sample, :device_sample) || throw(ArgumentError(
        "XLA strategy must be :greedy, :sample, or :device_sample",
    ))
    requested_sample_top_k = if sample_top_k === nothing
        nothing
    else
        strategy === :device_sample || throw(ArgumentError(
            "sample_top_k only applies to the :device_sample strategy",
        ))
        requested = _strict_host_int(sample_top_k, "sample_top_k")
        requested > 0 || throw(ArgumentError("sample_top_k must be positive"))
        requested
    end
    context = _strict_host_int(context_tokens, "context_tokens")
    chunk = _strict_host_int(prefill_chunk_tokens, "prefill_chunk_tokens")
    context > 0 || throw(ArgumentError("context_tokens must be positive"))
    context <= typemax(Int32) || throw(ArgumentError(
        "context_tokens must fit in Int32 device positions",
    ))
    0 < chunk <= context || throw(ArgumentError(
        "prefill_chunk_tokens must be in 1:context_tokens",
    ))
    context % chunk == 0 || throw(ArgumentError(
        "context_tokens must be divisible by prefill_chunk_tokens",
    ))

    allocator_before_load = _qwen3_xla_allocator_snapshot()
    load_started = time_ns()
    loaded = load_hf_qwen3_compact_bundle(
        model_dir;
        max_seq_len=context,
        weight_dtype=BFloat16,
        revision,
        variant,
    )
    host_load_seconds = (time_ns() - load_started) / 1.0e9
    model = loaded.model
    tokenizer = loaded.tokenizer
    generation_config = loaded.generation_config
    host_parameters = loaded.parameters
    parameter_logical_bytes = _qwen3_xla_tensor_bytes(host_parameters)
    parameter_tensor_count = _qwen3_xla_tensor_leaves(host_parameters)
    rope = first(values(model.blocks.layers)).attn.rope

    transfer_started = time_ns()
    parameters = Reactant.to_rarray(host_parameters)
    parameter_transfer_seconds = (time_ns() - transfer_started) / 1.0e9
    allocator_after_parameter_transfer = _qwen3_xla_allocator_snapshot()

    # Remove every host reference to the large tree before allocating K/V or
    # compiling. `parameters` is the only device parameter topology.
    loaded = nothing
    host_parameters = nothing
    GC.gc(true)

    runtime_started = time_ns()
    cos_table = Reactant.to_rarray(BFloat16.(rope.cos_cache[:, 1:context]))
    sin_table = Reactant.to_rarray(BFloat16.(rope.sin_cache[:, 1:context]))
    cache_shape = (
        model.head_dim,
        model.num_kv_heads,
        context,
        1,
    )
    key_caches = Tuple(
        Reactant.to_rarray(zeros(BFloat16, cache_shape))
        for _ in 1:model.num_layers
    )
    value_caches = Tuple(
        Reactant.to_rarray(zeros(BFloat16, cache_shape))
        for _ in 1:model.num_layers
    )
    runtime_allocation_seconds = (time_ns() - runtime_started) / 1.0e9
    allocator_after_runtime_allocation = _qwen3_xla_allocator_snapshot()

    compile_tokens = Reactant.to_rarray(ones(Int, chunk, 1))
    compile_token = Reactant.to_rarray(ones(Int, 1))
    compile_position = Reactant.to_rarray(zeros(Int32, 1))
    compile_key_positions =
        Reactant.to_rarray(Int32.(collect(1:context)))

    # `top_k` fixes the number of extraction passes, so it is the one sampling
    # option baked into the executable; temperature and top-p stay as device
    # scalars and can change per request without recompiling.
    top_k_static = 0
    if strategy === :device_sample
        configured = requested_sample_top_k === nothing ?
            generation_config.top_k : requested_sample_top_k
        configured isa Integer && configured > 0 || throw(ArgumentError(
            "device sampling requires a positive top_k; the generation " *
            "config did not supply one, pass sample_top_k explicitly",
        ))
        top_k_static = min(Int(configured), model.vocab_size)
    end
    compile_uniform = Reactant.to_rarray(zeros(Float32, 1))
    compile_temperature = Reactant.to_rarray(ones(Float32, 1))
    compile_top_p = Reactant.to_rarray(ones(Float32, 1))

    prefill_kernel = if strategy === :device_sample
        (ps, tokens, kc, vc, position, cos_t, sin_t, kp, u, t, p) ->
            _bf16a_static_prefill_chunk_sample(
                model,
                ps,
                tokens,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
                u,
                t,
                p,
                top_k_static,
            )
    elseif strategy === :greedy
        (ps, tokens, kc, vc, position, cos_t, sin_t, kp) ->
            _bf16a_static_prefill_chunk_greedy(
                model,
                ps,
                tokens,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
    else
        (ps, tokens, kc, vc, position, cos_t, sin_t, kp) ->
            _bf16a_static_prefill_chunk_core(
                model,
                ps,
                tokens,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
    end
    decode_kernel = if strategy === :device_sample
        (ps, token, kc, vc, position, cos_t, sin_t, kp, u, t, p) ->
            _bf16a_static_decode_sample_step_packed(
                model,
                ps,
                token,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
                u,
                t,
                p,
                top_k_static,
            )
    elseif strategy === :greedy
        (ps, token, kc, vc, position, cos_t, sin_t, kp) ->
            _bf16a_static_decode_greedy_step_packed(
                model,
                ps,
                token,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
    else
        function (ps, token, kc, vc, position, cos_t, sin_t, kp)
            logits = _bf16a_static_decode_step_packed(
                model,
                ps,
                token,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
            return logits, position .+ one(Int32)
        end
    end

    prefill_compile_started = time_ns()
    compiled_prefill = if strategy === :device_sample
        Reactant.@compile prefill_kernel(
            parameters,
            compile_tokens,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
            compile_uniform,
            compile_temperature,
            compile_top_p,
        )
    else
        Reactant.@compile prefill_kernel(
            parameters,
            compile_tokens,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
        )
    end
    prefill_compile_seconds =
        (time_ns() - prefill_compile_started) / 1.0e9

    decode_compile_started = time_ns()
    compiled_decode = if strategy === :device_sample
        Reactant.@compile decode_kernel(
            parameters,
            compile_token,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
            compile_uniform,
            compile_temperature,
            compile_top_p,
        )
    else
        Reactant.@compile decode_kernel(
            parameters,
            compile_token,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
        )
    end
    decode_compile_seconds =
        (time_ns() - decode_compile_started) / 1.0e9
    allocator_ready = _qwen3_xla_allocator_snapshot()
    load_metrics = (;
        host_load_seconds,
        parameter_transfer_seconds,
        runtime_allocation_seconds,
        prefill_compile_seconds,
        decode_compile_seconds,
        parameter_logical_bytes,
        parameter_tensor_count,
        device_parameter_tree_count=1,
        device_parameter_tree_transfer_count=1,
        original_parameter_tree_constructed=false,
        original_parameter_tree_transferred=false,
        separate_packed_projection_tree_transferred=false,
        allocator_before_load,
        allocator_after_parameter_transfer,
        allocator_after_runtime_allocation,
        allocator_ready,
    )
    metadata = _qwen3_xla_session_metadata(
        model,
        strategy,
        context,
        chunk,
        top_k_static,
        0,
    )
    source_signature = _qwen3_xla_session_source_signature(
        tokenizer,
        generation_config,
        metadata.strategy,
        metadata.vocab_size,
    )
    _qwen3_xla_validate_session_storage(
        model,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        context,
    )
    contract = _qwen3_xla_session_contract(
        model,
        parameters,
        tokenizer,
        generation_config,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        compiled_prefill,
        compiled_decode,
        metadata,
        source_signature,
    )
    session = HFQwen3BF16XLASession(
        _HF_QWEN3_BF16_XLA_SESSION_VALIDATED,
        model,
        parameters,
        tokenizer,
        generation_config,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        compiled_prefill,
        compiled_decode,
        strategy,
        context,
        chunk,
        top_k_static,
        0,
        load_metrics,
        contract,
    )
    _qwen3_xla_validate_session(session)
    return session
end

function reset_hf_qwen3_bf16_xla_session!(
    session::HFQwen3BF16XLASession,
)
    _qwen3_xla_validate_session(session)
    session.position = 0
    return session
end

"""
    generate_hf_qwen3_bf16_xla!(
        session, prompt_tokens; max_new_tokens=512, kwargs...
    )

Generate from a reusable single-tree XLA session.

  * `:greedy` keeps argmax on device and transfers one token per step.
  * `:device_sample` keeps the whole temperature/top-k/top-p policy on device;
    the host sends one uniform and receives one token per step.
  * `:sample` transfers one logits vector per step and applies the host
    temperature/top-k/top-p policy of the eager deployment path.

`sample_uniforms` replaces the RNG draws with a fixed sequence, which is how
the two sampling strategies are compared token by token.
`on_token`, when supplied, is called as `on_token(token_id)` after the cache
mutation for that generation stage has been committed to `session.position`.
"""
function generate_hf_qwen3_bf16_xla!(
    session::HFQwen3BF16XLASession,
    prompt_tokens;
    max_new_tokens::Integer=512,
    temperature=nothing,
    top_k=nothing,
    top_p=nothing,
    rng::AbstractRNG=default_rng(),
    stop_token_ids=nothing,
    pad_token_id::Integer=1,
    on_token=nothing,
    sample_uniforms=nothing,
)
    _qwen3_xla_validate_session(session)
    prompt_ids = vec(_strict_host_int_array(
        prompt_tokens,
        "Qwen3 XLA prompt token",
    ))
    _validate_generation_ids(prompt_ids, session.model.vocab_size)
    plan = plan_qwen3_xla_window(
        length(prompt_ids),
        max_new_tokens;
        context_tokens=session.context_tokens,
        chunk_tokens=session.prefill_chunk_tokens,
    )
    padded = qwen3_xla_pad_prompt(
        prompt_ids,
        plan;
        pad_token_id,
    )
    1 <= pad_token_id <= session.model.vocab_size || throw(ArgumentError(
        "pad_token_id is outside the model vocabulary",
    ))
    raw_stop_token_ids = stop_token_ids === nothing ?
        session.tokenizer.eos_ids : stop_token_ids
    stops = _qwen3_stop_token_set(
        raw_stop_token_ids,
        session.model.vocab_size,
    )

    resolved_temperature = temperature === nothing ?
        session.generation_config.temperature : temperature
    resolved_top_k = top_k === nothing ?
        session.generation_config.top_k : top_k
    resolved_top_p = top_p === nothing ?
        session.generation_config.top_p : top_p
    if session.strategy === :sample
        options = _qwen3_session_sampling_options(
            resolved_temperature,
            resolved_top_k,
            resolved_top_p,
        )
        resolved_temperature = options.temperature
        resolved_top_k = options.top_k
        resolved_top_p = options.top_p
    end

    uniforms = nothing
    if sample_uniforms !== nothing
        session.strategy === :greedy && throw(ArgumentError(
            "sample_uniforms requires a sampling strategy",
        ))
        uniforms = Float32[
            _validate_device_sampling_uniform(value)
            for value in vec(collect(sample_uniforms))
        ]
        length(uniforms) >= plan.max_new_tokens || throw(ArgumentError(
            "sample_uniforms must supply one value per requested token",
        ))
    end
    draw_uniform(step) = uniforms === nothing ?
        rand(rng, Float32) : uniforms[step]

    temperature_state = nothing
    top_p_state = nothing
    if session.strategy === :device_sample
        options = _validate_device_sampling_options(;
            temperature=resolved_temperature,
            top_k=resolved_top_k,
            top_p=resolved_top_p,
            vocab_size=session.model.vocab_size,
        )
        min(options.top_k, session.model.vocab_size) == session.sample_top_k ||
            throw(ArgumentError(
                "top_k must match the compiled device sampling constant " *
                "$(session.sample_top_k); recompile the session to change it",
            ))
        temperature_state = Reactant.to_rarray(Float32[options.temperature])
        top_p_state = Reactant.to_rarray(Float32[options.top_p])
    end

    key_positions = Reactant.to_rarray(qwen3_xla_key_positions(plan))
    position_state = Reactant.to_rarray(zeros(Int32, 1))
    reset_hf_qwen3_bf16_xla_session!(session)
    allocator_before = _qwen3_xla_allocator_snapshot()

    prefill_started = time_ns()
    output_state = nothing
    # Every prefill chunk runs the same executable but only the final chunk
    # selects a token, so one uniform is consumed for the first output token
    # regardless of how many chunks the prompt needs.
    prefill_uniform_state = session.strategy === :device_sample ?
        Reactant.to_rarray(Float32[draw_uniform(1)]) : nothing
    for first_index in 1:session.prefill_chunk_tokens:length(padded)
        _qwen3_xla_validate_session_contract(session)
        last_index = first_index + session.prefill_chunk_tokens - 1
        token_state = Reactant.to_rarray(reshape(
            padded[first_index:last_index],
            :,
            1,
        ))
        output_state, position_state = if session.strategy === :device_sample
            session.compiled_prefill(
                session.parameters,
                token_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
                prefill_uniform_state,
                temperature_state,
                top_p_state,
            )
        else
            session.compiled_prefill(
                session.parameters,
                token_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
            )
        end
        session.position = last_index
    end
    first_output = if session.strategy === :sample
        logits = Array(output_state)
        choice, _ = _qwen3_session_choice(
            logits,
            identity,
            :sample,
            rng;
            temperature=resolved_temperature,
            top_k=resolved_top_k,
            top_p=resolved_top_p,
            sample_uniform=uniforms === nothing ? nothing : uniforms[1],
        )
        _qwen3_xla_generated_token(choice, session.model.vocab_size)
    else
        _qwen3_xla_host_token(output_state, session.model.vocab_size)
    end
    prefill_seconds = (time_ns() - prefill_started) / 1.0e9

    generated_ids = Int[first_output]
    on_token === nothing || on_token(first_output)
    stop_reason = first_output in stops ? :eos : :length

    decode_started = time_ns()
    while length(generated_ids) < plan.max_new_tokens &&
            !(last(generated_ids) in stops)
        _qwen3_xla_validate_session_contract(session)
        step = length(generated_ids) + 1
        input_state = session.strategy === :sample ?
            Reactant.to_rarray([last(generated_ids)]) :
            output_state
        output_state, position_state = if session.strategy === :device_sample
            session.compiled_decode(
                session.parameters,
                input_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
                Reactant.to_rarray(Float32[draw_uniform(step)]),
                temperature_state,
                top_p_state,
            )
        else
            session.compiled_decode(
                session.parameters,
                input_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
            )
        end
        session.position += 1
        next_token = if session.strategy === :sample
            logits = Array(output_state)
            choice, _ = _qwen3_session_choice(
                logits,
                identity,
                :sample,
                rng;
                temperature=resolved_temperature,
                top_k=resolved_top_k,
                top_p=resolved_top_p,
                sample_uniform=uniforms === nothing ? nothing : uniforms[step],
            )
            _qwen3_xla_generated_token(choice, session.model.vocab_size)
        else
            _qwen3_xla_host_token(output_state, session.model.vocab_size)
        end
        push!(generated_ids, next_token)
        on_token === nothing || on_token(next_token)
    end
    decode_seconds = (time_ns() - decode_started) / 1.0e9
    last(generated_ids) in stops && (stop_reason = :eos)
    expected_position =
        plan.prompt_bucket_tokens + length(generated_ids) - 1
    session.position == expected_position || error(
        "Qwen3 XLA session position disagrees with committed cache progress",
    )
    allocator_after = _qwen3_xla_allocator_snapshot()
    completion = decode(
        session.tokenizer,
        generated_ids;
        errors=:replace,
        skip_special_tokens=true,
    )
    return (;
        prompt_ids,
        generated_ids,
        token_ids=vcat(prompt_ids, generated_ids),
        completion,
        stop_reason,
        strategy=session.strategy,
        window_plan=plan,
        prefill_seconds,
        decode_seconds,
        tokens_per_second=length(generated_ids) <= 1 ?
            0.0 : (length(generated_ids) - 1) / decode_seconds,
        allocator_before,
        allocator_after,
    )
end
