using BFloat16s: BFloat16

# Qwen3 MoE deployment keeps the comparatively small attention/router tree,
# final norm and LM head resident on the accelerator. Expert matrices remain
# in safetensors and are uploaded one active layer at a time. The implementation
# is device-generic; the CUDA extension supplies the grouped BF16 WMMA method.

function _qwen3_moe_tree_bytes(x::AbstractArray)
    return Base.checked_mul(length(x), sizeof(eltype(x)))
end
_qwen3_moe_tree_bytes(x::NamedTuple) =
    sum(_qwen3_moe_tree_bytes, values(x); init=0)
_qwen3_moe_tree_bytes(x::Tuple) =
    sum(_qwen3_moe_tree_bytes, values(x); init=0)
_qwen3_moe_tree_bytes(x) = 0

"""
    qwen3_moe_offload_plan(model, context_tokens; ...)

Estimate the irreducible BF16 device payload for a Qwen3 MoE offload session.
The estimate includes resident attention/router/norm/LM-head parameters, a
static KV cache and at most `max_active_experts` experts for one layer. It does
not include allocator slack, attention scores or grouped-dispatch workspace.
"""
function qwen3_moe_offload_plan(
    model::GPTModel,
    context_tokens::Integer;
    batch_size::Integer=1,
    max_active_experts::Integer=model.num_experts,
    dtype_bytes::Integer=sizeof(BFloat16),
)
    _qwen3_validate_moe_semantics(model)
    context = Int(context_tokens)
    batch = Int(batch_size)
    active = Int(max_active_experts)
    bytes = Int(dtype_bytes)
    0 < context <= model.max_seq_len || throw(ArgumentError(
        "context_tokens must be in 1:model.max_seq_len",
    ))
    batch > 0 || throw(ArgumentError("batch_size must be positive"))
    1 <= active <= model.num_experts || throw(ArgumentError(
        "max_active_experts must be in 1:model.num_experts",
    ))
    bytes > 0 || throw(ArgumentError("dtype_bytes must be positive"))

    d_model = model.d_model
    query_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    attention_elements =
        query_dim * d_model +
        2 * kv_dim * d_model +
        d_model * query_dim
    layer_resident_elements =
        attention_elements +
        model.num_experts * d_model +
        2 * d_model +
        2 * model.head_dim
    resident_parameter_elements =
        model.num_layers * layer_resident_elements +
        d_model +
        model.vocab_size * d_model
    active_expert_elements =
        active * 3 * d_model * model.mlp_hidden_dim
    resident_parameter_bytes = Base.checked_mul(
        resident_parameter_elements,
        bytes,
    )
    active_expert_layer_bytes = Base.checked_mul(active_expert_elements, bytes)
    kv_cache_bytes = qwen3_kv_cache_bytes(
        model,
        context;
        batch_size=batch,
        dtype_bytes=bytes,
    )
    working_set_floor_bytes = Base.checked_add(
        Base.checked_add(resident_parameter_bytes, active_expert_layer_bytes),
        kv_cache_bytes,
    )
    return (;
        context_tokens=context,
        batch_size=batch,
        max_active_experts=active,
        dtype_bytes=bytes,
        resident_parameter_bytes,
        active_expert_layer_bytes,
        kv_cache_bytes,
        working_set_floor_bytes,
    )
end

function _qwen3_local_expert_routes(
    expert_indices::AbstractMatrix{<:Integer},
    num_experts::Int,
)
    isempty(expert_indices) && throw(ArgumentError(
        "expert route table must not be empty",
    ))
    active_experts = sort!(unique!(Int.(vec(collect(expert_indices)))))
    all(expert -> 1 <= expert <= num_experts, active_experts) ||
        throw(ArgumentError("expert route index is outside 1:num_experts"))
    global_to_local = zeros(Int32, num_experts)
    for (local_index, global_expert) in enumerate(active_experts)
        global_to_local[global_expert] = Int32(local_index)
    end
    local_indices = reshape(
        global_to_local[Int.(vec(collect(expert_indices)))],
        size(expert_indices),
    )
    return (; active_experts, local_indices)
end

# Portable fallback used by non-CUDA devices. LifeAICUDAExt specializes this
# hook to the grouped BF16 WMMA implementation.
function _qwen3_grouped_bf16_expert_dispatch(
    tokens,
    expert_indices,
    routing_weights,
    expert_parameters,
)
    return qwen3_device_sparse_expert_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters,
    )
end

mutable struct HFQwen3MoEOffloadSession
    model
    config
    reader::HFSafetensorsReader
    tensors
    resident_blocks::Vector{Any}
    final_scale
    logits_weight
    cos_table
    sin_table
    caches::Vector{Any}
    position::Int
    context_tokens::Int
    prefill_chunk_tokens::Int
    grouped_experts::Bool
    to_device
    source::String
    resident_parameter_bytes::Int
    expert_bytes_read::Int
    expert_bytes_uploaded::Int
    expert_cache_budget_bytes::Int
    expert_cache_policy::Symbol
    expert_cache_dispatch::Symbol
    expert_gc_interval_layers::Int
    expert_cache_bytes::Int
    expert_cache_peak_bytes::Int
    expert_cache::Dict{Tuple{Int,Int},Any}
    expert_cache_last_used::Dict{Tuple{Int,Int},Int}
    expert_cache_clock::Int
    expert_cache_entry_generation::Int
    expert_cache_hits::Int
    expert_cache_misses::Int
    expert_cache_evictions::Int
    expert_active_materializations::Int
    expert_active_materialization_bytes::Int
    expert_scattered_dispatches::Int
    expert_pointer_bytes_uploaded::Int
    expert_forced_gc_calls::Int
    expert_scattered_state
    expert_pointer_table_builds::Int
    expert_pointer_table_reuses::Int
    expert_workspace_allocations::Int
    expert_workspace_reuses::Int
    expert_pointer_table_bytes::Int
    expert_workspace_bytes::Int
    expert_miss_pipeline::Symbol
    expert_read_workers::Int
    expert_pinned_upload::Bool
    expert_miss_stage_seconds::Float64
    expert_host_read_seconds::Float64
    expert_upload_wait_seconds::Float64
    expert_read_tasks::Int
    expert_parallel_read_layers::Int
    expert_pinned_bytes_uploaded::Int
end

struct _Qwen3MoEExpertCacheEntry
    gate_proj
    up_proj
    down_proj
    bytes::Int
    generation::Int
end

struct _Qwen3MoEScatteredExperts
    expert_ids::Vector{Int}
    generations::Vector{Int}
    entries::Vector{_Qwen3MoEExpertCacheEntry}
    hidden_dim::Int
    d_model::Int
end

function qwen3_moe_expert_cache_stats(session::HFQwen3MoEOffloadSession)
    return (;
        budget_bytes=session.expert_cache_budget_bytes,
        policy=session.expert_cache_policy,
        dispatch=session.expert_cache_dispatch,
        gc_interval_layers=session.expert_gc_interval_layers,
        current_bytes=session.expert_cache_bytes,
        peak_bytes=session.expert_cache_peak_bytes,
        entries=length(session.expert_cache),
        hits=session.expert_cache_hits,
        misses=session.expert_cache_misses,
        evictions=session.expert_cache_evictions,
        bytes_read=session.expert_bytes_read,
        bytes_uploaded=session.expert_bytes_uploaded,
        active_materializations=session.expert_active_materializations,
        active_materialization_bytes=
            session.expert_active_materialization_bytes,
        scattered_dispatches=session.expert_scattered_dispatches,
        pointer_bytes_uploaded=session.expert_pointer_bytes_uploaded,
        forced_gc_calls=session.expert_forced_gc_calls,
        pointer_table_builds=session.expert_pointer_table_builds,
        pointer_table_reuses=session.expert_pointer_table_reuses,
        workspace_allocations=session.expert_workspace_allocations,
        workspace_reuses=session.expert_workspace_reuses,
        pointer_table_bytes=session.expert_pointer_table_bytes,
        workspace_bytes=session.expert_workspace_bytes,
        miss_pipeline=session.expert_miss_pipeline,
        read_workers=session.expert_read_workers,
        pinned_upload=session.expert_pinned_upload,
        miss_stage_seconds=session.expert_miss_stage_seconds,
        host_read_seconds=session.expert_host_read_seconds,
        upload_wait_seconds=session.expert_upload_wait_seconds,
        read_tasks=session.expert_read_tasks,
        parallel_read_layers=session.expert_parallel_read_layers,
        pinned_bytes_uploaded=session.expert_pinned_bytes_uploaded,
    )
end

function _qwen3_moe_validate_expert_cache_dispatch(dispatch::Symbol)
    dispatch in (:materialized, :scattered) || throw(ArgumentError(
        "expert_cache_dispatch must be :materialized or :scattered",
    ))
    return dispatch
end

function _qwen3_moe_validate_expert_cache_policy(policy::Symbol)
    policy in (:global_lru, :layer_balanced_lru) || throw(ArgumentError(
        "expert_cache_policy must be :global_lru or :layer_balanced_lru",
    ))
    return policy
end

function _qwen3_moe_validate_expert_miss_pipeline(pipeline::Symbol)
    pipeline in (:sequential, :overlapped) || throw(ArgumentError(
        "expert_miss_pipeline must be :sequential or :overlapped",
    ))
    return pipeline
end

# Portable devices can overlap independent host reads, but pinned asynchronous
# uploads are accelerator-specific. LifeAICUDAExt supplies the CUDA methods.
_qwen3_moe_pinned_upload_supported(prototype) = false

function _qwen3_moe_begin_expert_upload(
    prototype,
    pinned::Bool,
    max_inflight::Int,
)
    pinned && throw(ArgumentError(
        "pinned expert upload is not supported by the selected device",
    ))
    return nothing
end

function _qwen3_moe_upload_host_expert(
    prototype,
    host_parameters,
    to_device,
    upload_state,
)
    return to_device(host_parameters)
end

_qwen3_moe_finish_expert_upload(prototype, upload_state) = (;
    wait_seconds=0.0,
    pinned_bytes=0,
)

function _qwen3_moe_reset_expert_traffic!(
    session::HFQwen3MoEOffloadSession,
)
    session.expert_bytes_read = 0
    session.expert_bytes_uploaded = 0
    session.expert_cache_hits = 0
    session.expert_cache_misses = 0
    session.expert_cache_evictions = 0
    session.expert_active_materializations = 0
    session.expert_active_materialization_bytes = 0
    session.expert_scattered_dispatches = 0
    session.expert_pointer_bytes_uploaded = 0
    session.expert_forced_gc_calls = 0
    session.expert_pointer_table_builds = 0
    session.expert_pointer_table_reuses = 0
    session.expert_workspace_allocations = 0
    session.expert_workspace_reuses = 0
    session.expert_miss_stage_seconds = 0.0
    session.expert_host_read_seconds = 0.0
    session.expert_upload_wait_seconds = 0.0
    session.expert_read_tasks = 0
    session.expert_parallel_read_layers = 0
    session.expert_pinned_bytes_uploaded = 0
    return session
end

function _qwen3_moe_materialize_active_experts(
    session::HFQwen3MoEOffloadSession,
    scattered::_Qwen3MoEScatteredExperts,
)
    parameters = (;
        gate_proj=cat((
            reshape(
                entry.gate_proj,
                scattered.hidden_dim,
                scattered.d_model,
                1,
            )
            for entry in scattered.entries
        )...; dims=3),
        up_proj=cat((
            reshape(
                entry.up_proj,
                scattered.hidden_dim,
                scattered.d_model,
                1,
            )
            for entry in scattered.entries
        )...; dims=3),
        down_proj=cat((
            reshape(
                entry.down_proj,
                scattered.d_model,
                scattered.hidden_dim,
                1,
            )
            for entry in scattered.entries
        )...; dims=3),
    )
    bytes = sum(entry.bytes for entry in scattered.entries)
    session.expert_active_materializations = Base.checked_add(
        session.expert_active_materializations,
        1,
    )
    session.expert_active_materialization_bytes = Base.checked_add(
        session.expert_active_materialization_bytes,
        bytes,
    )
    return parameters
end

# Portable devices retain the materialized dispatch. LifeAICUDAExt specializes
# this hook to index cached expert matrices through a compact device pointer
# table. The opaque state owns bounded pointer plans and shape-keyed workspaces
# across calls; cache clear/reconfigure invalidates it.
function _qwen3_scattered_expert_dispatch(
    tokens,
    expert_indices,
    routing_weights,
    scattered::_Qwen3MoEScatteredExperts,
    state,
    layer_index::Int,
)
    return nothing
end

"""
    clear_hf_qwen3_moe_expert_cache!(session)

Drop logical ownership of all cached expert-layer tensors while retaining the
resident model, static KV buffers and request position. Request traffic counters
are unchanged; the next prefill resets them as usual.
"""
function clear_hf_qwen3_moe_expert_cache!(
    session::HFQwen3MoEOffloadSession,
)
    empty!(session.expert_cache)
    empty!(session.expert_cache_last_used)
    session.expert_cache_bytes = 0
    session.expert_cache_peak_bytes = 0
    session.expert_scattered_state = nothing
    session.expert_pointer_table_bytes = 0
    session.expert_workspace_bytes = 0
    GC.gc(false)
    return session
end

"""
    configure_hf_qwen3_moe_expert_cache!(
        session; budget_bytes, policy, dispatch, gc_interval_layers,
        miss_pipeline, read_workers, pinned_upload)

Clear the existing expert cache and reconfigure its byte budget and eviction
policy while retaining the resident model, static KV buffers and compiled
kernels. `:global_lru` preserves the original policy;
`:layer_balanced_lru` reserves an approximately equal number of expert slots
per decoder layer to resist sequential layer-scan thrashing. Dispatch and
forced-GC cadence can be changed in the same cold-cache transition.
"""
function configure_hf_qwen3_moe_expert_cache!(
    session::HFQwen3MoEOffloadSession;
    budget_bytes::Integer=session.expert_cache_budget_bytes,
    policy::Symbol=session.expert_cache_policy,
    dispatch::Symbol=session.expert_cache_dispatch,
    gc_interval_layers::Integer=session.expert_gc_interval_layers,
    miss_pipeline::Symbol=session.expert_miss_pipeline,
    read_workers::Integer=session.expert_read_workers,
    pinned_upload::Bool=session.expert_pinned_upload,
)
    budget = Int(budget_bytes)
    budget >= 0 || throw(ArgumentError(
        "expert cache budget_bytes must be non-negative",
    ))
    validated_policy = _qwen3_moe_validate_expert_cache_policy(policy)
    validated_dispatch = _qwen3_moe_validate_expert_cache_dispatch(dispatch)
    validated_pipeline = _qwen3_moe_validate_expert_miss_pipeline(
        miss_pipeline,
    )
    gc_interval = Int(gc_interval_layers)
    workers = Int(read_workers)
    gc_interval >= 0 || throw(ArgumentError(
        "expert cache gc_interval_layers must be non-negative",
    ))
    workers > 0 || throw(ArgumentError(
        "expert read_workers must be positive",
    ))
    validated_dispatch === :scattered && session.grouped_experts &&
        throw(ArgumentError(
            "scattered expert cache dispatch does not support grouped experts",
        ))
    validated_dispatch === :scattered && budget == 0 && throw(ArgumentError(
        "scattered expert cache dispatch requires a positive cache budget",
    ))
    validated_pipeline === :overlapped && budget == 0 && throw(ArgumentError(
        "overlapped expert misses require a positive cache budget",
    ))
    pinned_upload && validated_pipeline !== :overlapped && throw(ArgumentError(
        "pinned expert upload requires miss_pipeline=:overlapped",
    ))
    pinned_upload &&
        !_qwen3_moe_pinned_upload_supported(session.final_scale) &&
        throw(ArgumentError(
            "pinned expert upload is not supported by the selected device",
        ))
    clear_hf_qwen3_moe_expert_cache!(session)
    session.expert_cache_budget_bytes = budget
    session.expert_cache_policy = validated_policy
    session.expert_cache_dispatch = validated_dispatch
    session.expert_gc_interval_layers = gc_interval
    session.expert_miss_pipeline = validated_pipeline
    session.expert_read_workers = workers
    session.expert_pinned_upload = pinned_upload
    return session
end

function _qwen3_moe_touch_expert_cache!(
    session::HFQwen3MoEOffloadSession,
    key::Tuple{Int,Int},
)
    session.expert_cache_clock = Base.checked_add(
        session.expert_cache_clock,
        1,
    )
    session.expert_cache_last_used[key] = session.expert_cache_clock
    return nothing
end

function _qwen3_moe_make_expert_cache_room!(
    session::HFQwen3MoEOffloadSession,
    entry_bytes::Int,
    protected::Set{Tuple{Int,Int}},
)
    while session.expert_cache_bytes + entry_bytes >
            session.expert_cache_budget_bytes
        candidates = filter(
            key -> !(key in protected),
            collect(keys(session.expert_cache)),
        )
        isempty(candidates) && return false
        victim = candidates[argmin([
            session.expert_cache_last_used[key]
            for key in candidates
        ])]
        _qwen3_moe_evict_expert_cache!(session, victim)
    end
    return true
end

function _qwen3_moe_evict_expert_cache!(
    session::HFQwen3MoEOffloadSession,
    key::Tuple{Int,Int},
)
    entry = pop!(session.expert_cache, key)
    delete!(session.expert_cache_last_used, key)
    session.expert_cache_bytes -= entry.bytes
    session.expert_cache_evictions = Base.checked_add(
        session.expert_cache_evictions,
        1,
    )
    return nothing
end

function _qwen3_moe_make_layer_cache_room!(
    session::HFQwen3MoEOffloadSession,
    layer::Int,
    entry_bytes::Int,
    protected::Set{Tuple{Int,Int}},
)
    total_slots = session.expert_cache_budget_bytes ÷ entry_bytes
    base_slots, extra_layers = divrem(total_slots, session.model.num_layers)
    layer_slots = base_slots + (layer <= extra_layers ? 1 : 0)
    layer_slots == 0 && return false
    layer_keys = filter(
        key -> key[1] == layer,
        collect(keys(session.expert_cache)),
    )
    while length(layer_keys) >= layer_slots
        candidates = filter(key -> !(key in protected), layer_keys)
        isempty(candidates) && return false
        victim = candidates[argmin([
            session.expert_cache_last_used[key]
            for key in candidates
        ])]
        _qwen3_moe_evict_expert_cache!(session, victim)
        filter!(key -> key != victim, layer_keys)
    end
    return true
end

function _qwen3_moe_read_expert_host(
    session::HFQwen3MoEOffloadSession,
    expert::Int,
    layer::Int,
)
    started = time_ns()
    model = session.model
    hidden_dim = model.mlp_hidden_dim
    d_model = model.d_model
    prefix = "model.layers.$layer.mlp.experts.$(expert - 1)"
    host_parameters = (;
        gate_proj=_expect_tensor(
            session.tensors,
            "$prefix.gate_proj.weight",
            (hidden_dim, d_model),
        ),
        up_proj=_expect_tensor(
            session.tensors,
            "$prefix.up_proj.weight",
            (hidden_dim, d_model),
        ),
        down_proj=_expect_tensor(
            session.tensors,
            "$prefix.down_proj.weight",
            (d_model, hidden_dim),
        ),
    )
    entry_bytes = _qwen3_moe_tree_bytes(host_parameters)
    return (;
        parameters=host_parameters,
        bytes=entry_bytes,
        seconds=(time_ns() - started) / 1.0e9,
    )
end

function _qwen3_moe_make_expert_cache_entry!(
    session::HFQwen3MoEOffloadSession,
    device_parameters,
    entry_bytes::Int,
)
    session.expert_cache_entry_generation = Base.checked_add(
        session.expert_cache_entry_generation,
        1,
    )
    return _Qwen3MoEExpertCacheEntry(
        device_parameters.gate_proj,
        device_parameters.up_proj,
        device_parameters.down_proj,
        entry_bytes,
        session.expert_cache_entry_generation,
    )
end

function _qwen3_moe_store_expert_cache_entry!(
    session::HFQwen3MoEOffloadSession,
    key::Tuple{Int,Int},
    entry::_Qwen3MoEExpertCacheEntry,
    protected::Set{Tuple{Int,Int}},
)
    policy_allows_store = session.expert_cache_policy === :global_lru ||
        _qwen3_moe_make_layer_cache_room!(
            session,
            key[1],
            entry.bytes,
            protected,
        )
    can_store = entry.bytes <= session.expert_cache_budget_bytes &&
        policy_allows_store &&
        _qwen3_moe_make_expert_cache_room!(
            session,
            entry.bytes,
            protected,
        )
    if can_store
        session.expert_cache[key] = entry
        session.expert_cache_bytes = Base.checked_add(
            session.expert_cache_bytes,
            entry.bytes,
        )
        session.expert_cache_peak_bytes = max(
            session.expert_cache_peak_bytes,
            session.expert_cache_bytes,
        )
        _qwen3_moe_touch_expert_cache!(session, key)
    end
    return entry
end

function _qwen3_moe_read_expert(
    session::HFQwen3MoEOffloadSession,
    expert::Int,
    layer::Int,
)
    host = _qwen3_moe_read_expert_host(session, expert, layer)
    session.expert_bytes_read = Base.checked_add(
        session.expert_bytes_read,
        host.bytes,
    )
    session.expert_host_read_seconds += host.seconds
    session.expert_read_tasks = Base.checked_add(
        session.expert_read_tasks,
        1,
    )
    device_parameters = session.to_device(host.parameters)
    session.expert_bytes_uploaded = Base.checked_add(
        session.expert_bytes_uploaded,
        host.bytes,
    )
    return _qwen3_moe_make_expert_cache_entry!(
        session,
        device_parameters,
        host.bytes,
    )
end

function _qwen3_moe_cached_expert(
    session::HFQwen3MoEOffloadSession,
    expert::Int,
    layer::Int,
    protected::Set{Tuple{Int,Int}},
)
    key = (layer + 1, expert)
    cached = get(session.expert_cache, key, nothing)
    if cached !== nothing
        session.expert_cache_hits = Base.checked_add(
            session.expert_cache_hits,
            1,
        )
        _qwen3_moe_touch_expert_cache!(session, key)
        return cached
    end

    session.expert_cache_misses = Base.checked_add(
        session.expert_cache_misses,
        1,
    )
    started = time_ns()
    entry = _qwen3_moe_read_expert(session, expert, layer)
    stored = _qwen3_moe_store_expert_cache_entry!(
        session,
        key,
        entry,
        protected,
    )
    session.expert_miss_stage_seconds += (time_ns() - started) / 1.0e9
    return stored
end

function _qwen3_moe_load_uncached_active_experts(
    session::HFQwen3MoEOffloadSession,
    active_experts::Vector{Int},
    layer::Int,
)
    model = session.model
    hidden_dim = model.mlp_hidden_dim
    d_model = model.d_model
    prefix = "model.layers.$layer.mlp.experts"
    gate_values = map(active_experts) do expert
        reshape(_expect_tensor(
            session.tensors,
            "$prefix.$(expert - 1).gate_proj.weight",
            (hidden_dim, d_model),
        ), hidden_dim, d_model, 1)
    end
    up_values = map(active_experts) do expert
        reshape(_expect_tensor(
            session.tensors,
            "$prefix.$(expert - 1).up_proj.weight",
            (hidden_dim, d_model),
        ), hidden_dim, d_model, 1)
    end
    down_values = map(active_experts) do expert
        reshape(_expect_tensor(
            session.tensors,
            "$prefix.$(expert - 1).down_proj.weight",
            (d_model, hidden_dim),
        ), d_model, hidden_dim, 1)
    end
    host_parameters = (;
        gate_proj=cat(gate_values...; dims=3),
        up_proj=cat(up_values...; dims=3),
        down_proj=cat(down_values...; dims=3),
    )
    payload_bytes = _qwen3_moe_tree_bytes(host_parameters)
    session.expert_bytes_read = Base.checked_add(
        session.expert_bytes_read,
        payload_bytes,
    )
    parameters = session.to_device(host_parameters)
    session.expert_bytes_uploaded = Base.checked_add(
        session.expert_bytes_uploaded,
        payload_bytes,
    )
    host_parameters = nothing
    gate_values = nothing
    up_values = nothing
    down_values = nothing
    GC.gc(false)
    return parameters
end

function _qwen3_moe_load_active_experts_overlapped(
    session::HFQwen3MoEOffloadSession,
    active_experts::Vector{Int},
    layer::Int,
    protected::Set{Tuple{Int,Int}},
)
    started = time_ns()
    entries = Vector{_Qwen3MoEExpertCacheEntry}(undef, length(active_experts))
    misses = NamedTuple[]
    for (entry_index, expert) in enumerate(active_experts)
        key = (layer + 1, expert)
        cached = get(session.expert_cache, key, nothing)
        if cached === nothing
            session.expert_cache_misses = Base.checked_add(
                session.expert_cache_misses,
                1,
            )
            push!(misses, (; entry_index, expert, key))
        else
            session.expert_cache_hits = Base.checked_add(
                session.expert_cache_hits,
                1,
            )
            _qwen3_moe_touch_expert_cache!(session, key)
            entries[entry_index] = cached
        end
    end

    isempty(misses) && return _Qwen3MoEScatteredExperts(
        copy(active_experts),
        [entry.generation for entry in entries],
        entries,
        session.model.mlp_hidden_dim,
        session.model.d_model,
    )

    window = min(session.expert_read_workers, length(misses))
    tasks = Vector{Union{Nothing,Task}}(undef, length(misses))
    fill!(tasks, nothing)
    launch!(miss_index) = tasks[miss_index] = Threads.@spawn(
        _qwen3_moe_read_expert_host(
            session,
            misses[miss_index].expert,
            layer,
        )
    )
    for miss_index in 1:window
        launch!(miss_index)
    end
    session.expert_read_tasks = Base.checked_add(
        session.expert_read_tasks,
        length(misses),
    )
    if window > 1 && Threads.nthreads() > 1
        session.expert_parallel_read_layers = Base.checked_add(
            session.expert_parallel_read_layers,
            1,
        )
    end

    upload_state = _qwen3_moe_begin_expert_upload(
        session.final_scale,
        session.expert_pinned_upload,
        window,
    )
    try
        for miss_index in eachindex(misses)
            read_task = tasks[miss_index]::Task
            host = fetch(read_task)
            tasks[miss_index] = nothing
            next_index = miss_index + window
            next_index <= length(misses) && launch!(next_index)

            session.expert_host_read_seconds += host.seconds
            session.expert_bytes_read = Base.checked_add(
                session.expert_bytes_read,
                host.bytes,
            )
            device_parameters = _qwen3_moe_upload_host_expert(
                session.final_scale,
                host.parameters,
                session.to_device,
                upload_state,
            )
            session.expert_bytes_uploaded = Base.checked_add(
                session.expert_bytes_uploaded,
                host.bytes,
            )
            entry = _qwen3_moe_make_expert_cache_entry!(
                session,
                device_parameters,
                host.bytes,
            )
            miss = misses[miss_index]
            entries[miss.entry_index] = _qwen3_moe_store_expert_cache_entry!(
                session,
                miss.key,
                entry,
                protected,
            )
        end
    finally
        # Do not leave reader tasks running if a sibling read/upload failed.
        for task in tasks
            if task !== nothing && !istaskdone(task)
                try
                    wait(task)
                catch
                end
            end
        end
        upload = _qwen3_moe_finish_expert_upload(
            session.final_scale,
            upload_state,
        )
        session.expert_upload_wait_seconds += upload.wait_seconds
        session.expert_pinned_bytes_uploaded = Base.checked_add(
            session.expert_pinned_bytes_uploaded,
            upload.pinned_bytes,
        )
        session.expert_miss_stage_seconds +=
            (time_ns() - started) / 1.0e9
    end
    return _Qwen3MoEScatteredExperts(
        copy(active_experts),
        [entry.generation for entry in entries],
        entries,
        session.model.mlp_hidden_dim,
        session.model.d_model,
    )
end

function _qwen3_moe_load_active_experts(
    session::HFQwen3MoEOffloadSession,
    active_experts::Vector{Int},
    layer::Int,
)
    session.expert_cache_budget_bytes == 0 &&
        return _qwen3_moe_load_uncached_active_experts(
            session,
            active_experts,
            layer,
        )
    model = session.model
    hidden_dim = model.mlp_hidden_dim
    d_model = model.d_model
    protected = Set((layer + 1, expert) for expert in active_experts)
    session.expert_miss_pipeline === :overlapped &&
        return _qwen3_moe_load_active_experts_overlapped(
            session,
            active_experts,
            layer,
            protected,
        )
    entries = [
        _qwen3_moe_cached_expert(
            session,
            expert,
            layer,
            protected,
        )
        for expert in active_experts
    ]
    return _Qwen3MoEScatteredExperts(
        copy(active_experts),
        [entry.generation for entry in entries],
        entries,
        hidden_dim,
        d_model,
    )
end

function _qwen3_moe_maybe_collect!(
    session::HFQwen3MoEOffloadSession,
    layer_index::Int,
)
    interval = session.expert_gc_interval_layers
    if interval > 0 && layer_index % interval == 0
        GC.gc(false)
        session.expert_forced_gc_calls = Base.checked_add(
            session.expert_forced_gc_calls,
            1,
        )
    end
    return nothing
end

function _qwen3_moe_offload_block(
    session::HFQwen3MoEOffloadSession,
    x,
    layer_index::Int,
    mask,
)
    model = session.model
    ps = session.resident_blocks[layer_index]
    num_tokens, batch_size = size(x, 2), size(x, 3)
    head_dim = model.head_dim
    query_dim = model.num_heads * head_dim
    cache_position = session.position
    positions = (cache_position + 1):(cache_position + num_tokens)

    normed = _bf16a_rmsnorm(x, ps.norm1.scale, model.norm_epsilon)
    queries = reshape(
        _bf16a_linear(ps.attn.q_proj.weight, normed),
        head_dim, model.num_heads, num_tokens, batch_size,
    )
    keys = reshape(
        _bf16a_linear(ps.attn.k_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    values = reshape(
        _bf16a_linear(ps.attn.v_proj.weight, normed),
        head_dim, model.num_kv_heads, num_tokens, batch_size,
    )
    queries = _bf16a_rmsnorm(
        queries,
        ps.attn.q_norm.scale,
        model.qk_norm_epsilon,
    )
    keys = _bf16a_rmsnorm(
        keys,
        ps.attn.k_norm.scale,
        model.qk_norm_epsilon,
    )
    cos_slice = session.cos_table[:, positions]
    sin_slice = session.sin_table[:, positions]
    queries = _bf16a_apply_rope(queries, cos_slice, sin_slice)
    keys = _bf16a_apply_rope(keys, cos_slice, sin_slice)
    all_keys, all_values, updated_cache = _bf16a_append_cache(
        session.caches[layer_index],
        keys,
        values,
        cache_position,
    )
    context = _bf16a_attention(
        queries,
        all_keys,
        all_values;
        scaling=1.0f0 / sqrt(Float32(head_dim)),
        mask,
    )
    attention_output = _bf16a_linear(
        ps.attn.o_proj.weight,
        reshape(context, query_dim, num_tokens, batch_size),
    )
    residual = BFloat16.(_bf16a_f32(x) .+ _bf16a_f32(attention_output))
    normed2 = _bf16a_rmsnorm(residual, ps.norm2.scale, model.norm_epsilon)
    router_logits = reshape(
        _bf16a_linear(ps.gate.weight, normed2),
        model.num_experts,
        :,
    )
    routed = qwen3_device_topk_routing(
        router_logits,
        model.experts_per_token;
        normalize=model.normalize_routing,
    )

    host_routes = Array(routed.expert_indices)
    remapped = _qwen3_local_expert_routes(host_routes, model.num_experts)
    local_indices = session.to_device(remapped.local_indices)
    loaded_experts = _qwen3_moe_load_active_experts(
        session,
        remapped.active_experts,
        layer_index - 1,
    )
    expert_tokens = reshape(_bf16a_f32(normed2), model.d_model, :)
    scattered_result = if loaded_experts isa _Qwen3MoEScatteredExperts &&
            session.expert_cache_dispatch === :scattered
        _qwen3_scattered_expert_dispatch(
            expert_tokens,
            local_indices,
            routed.routing_weights,
            loaded_experts,
            session.expert_scattered_state,
            layer_index,
        )
    else
        nothing
    end
    expert_output = if scattered_result !== nothing
        session.expert_scattered_dispatches = Base.checked_add(
            session.expert_scattered_dispatches,
            1,
        )
        session.expert_pointer_bytes_uploaded = Base.checked_add(
            session.expert_pointer_bytes_uploaded,
            scattered_result.pointer_bytes_uploaded,
        )
        session.expert_scattered_state = scattered_result.state
        session.expert_pointer_table_builds = Base.checked_add(
            session.expert_pointer_table_builds,
            scattered_result.pointer_table_built ? 1 : 0,
        )
        session.expert_pointer_table_reuses = Base.checked_add(
            session.expert_pointer_table_reuses,
            scattered_result.pointer_table_reused ? 1 : 0,
        )
        session.expert_workspace_allocations = Base.checked_add(
            session.expert_workspace_allocations,
            scattered_result.workspace_allocated ? 1 : 0,
        )
        session.expert_workspace_reuses = Base.checked_add(
            session.expert_workspace_reuses,
            scattered_result.workspace_reused ? 1 : 0,
        )
        session.expert_pointer_table_bytes =
            scattered_result.pointer_table_bytes
        session.expert_workspace_bytes = scattered_result.workspace_bytes
        scattered_result.output
    else
        expert_parameters = loaded_experts isa _Qwen3MoEScatteredExperts ?
            _qwen3_moe_materialize_active_experts(
                session,
                loaded_experts,
            ) : loaded_experts
        if session.grouped_experts
            _qwen3_grouped_bf16_expert_dispatch(
                expert_tokens,
                local_indices,
                routed.routing_weights,
                expert_parameters,
            )
        else
            qwen3_device_sparse_expert_dispatch(
                expert_tokens,
                local_indices,
                routed.routing_weights,
                expert_parameters,
            )
        end
    end
    output = BFloat16.(
        _bf16a_f32(residual) .+
        reshape(expert_output, model.d_model, num_tokens, batch_size)
    )
    loaded_experts = nothing
    _qwen3_moe_maybe_collect!(session, layer_index)
    return output, updated_cache, remapped.active_experts
end

function _qwen3_moe_offload_chunk!(
    session::HFQwen3MoEOffloadSession,
    token_matrix::Matrix{Int},
    ;
    project_logits::Bool=true,
)
    seq_len, batch_size = size(token_matrix)
    batch_size == 1 || throw(ArgumentError(
        "Qwen3 MoE offload sessions currently require batch_size == 1",
    ))
    session.position + seq_len <= session.context_tokens || throw(ArgumentError(
        "tokens exceed the offload session context capacity",
    ))
    host_embedding = _read_embedding_rows(
        session.reader,
        "model.embed_tokens.weight",
        token_matrix,
        session.model.d_model,
        session.model.vocab_size;
        target_dtype=BFloat16,
    )
    x = session.to_device(host_embedding)
    host_embedding = nothing
    key_tokens = session.position + seq_len
    mask = session.to_device(_bf16a_causal_mask(seq_len, key_tokens))
    active_experts = Vector{Vector{Int}}(undef, session.model.num_layers)
    for layer_index in 1:session.model.num_layers
        x, session.caches[layer_index], active_experts[layer_index] =
            _qwen3_moe_offload_block(
                session,
                x,
                layer_index,
                mask,
            )
    end
    logits = if project_logits
        last_hidden = _bf16a_rmsnorm(
            x[:, end:end, :],
            session.final_scale,
            session.model.norm_epsilon,
        )
        Array(_bf16a_linear(session.logits_weight, last_hidden))
    else
        nothing
    end
    session.position = key_tokens
    return (; logits, active_experts=Tuple(active_experts))
end

"""
    load_hf_qwen3_moe_offload_session(model_dir; to_device=identity, ...)

Construct a BF16 Qwen3 MoE session whose attention/router/norm/LM-head tensors
and static KV cache stay on `to_device`, while active expert tensors stream one
layer at a time. Passing `to_device=CUDA.cu` activates CUDA without coupling
the core package API to CUDA. `grouped_experts=true` selects the CUDA grouped
BF16 tensor-core extension when available. A positive
`expert_cache_budget_bytes` enables a device-side LRU keyed by
`(one_based_layer, one_based_expert)`; the default zero preserves pure
streaming. `expert_cache_policy=:global_lru` preserves the original policy;
`:layer_balanced_lru` partitions entry capacity across layers to resist
sequential scan thrashing. `expert_cache_dispatch=:scattered` lets supported
accelerators index cached expert matrices through device pointers instead of
materializing an active 3D tensor. CUDA sessions retain bounded pointer plans
and dispatch workspaces across layers/requests; expert-entry generations guard
against stale pointer reuse, and cache clear/reconfigure drops the state.
`expert_gc_interval_layers` controls forced GC frequency (`1` preserves the
original per-layer behavior; `0` disables forced GC). Request reset retains
cached experts. `expert_miss_pipeline=:overlapped` uses a bounded number of
host reader tasks for independent active-expert misses; `expert_pinned_upload`
additionally enables accelerator-specific pinned asynchronous transfers.
Neither mode speculates beyond the current layer's completed router. Meanwhile,
`clear_hf_qwen3_moe_expert_cache!` releases their logical ownership.
"""
function load_hf_qwen3_moe_offload_session(
    model_dir::AbstractString;
    context_tokens::Integer=4096,
    prefill_chunk_tokens::Integer=128,
    grouped_experts::Bool=true,
    expert_cache_budget_bytes::Integer=0,
    expert_cache_policy::Symbol=:global_lru,
    expert_cache_dispatch::Symbol=:materialized,
    expert_gc_interval_layers::Integer=1,
    expert_miss_pipeline::Symbol=:sequential,
    expert_read_workers::Integer=4,
    expert_pinned_upload::Bool=false,
    to_device=identity,
    on_resident_layer=nothing,
)
    isdir(model_dir) || throw(ArgumentError(
        "model directory does not exist: $model_dir",
    ))
    context = Int(context_tokens)
    chunk = Int(prefill_chunk_tokens)
    cache_budget = Int(expert_cache_budget_bytes)
    cache_policy = _qwen3_moe_validate_expert_cache_policy(
        expert_cache_policy,
    )
    cache_dispatch = _qwen3_moe_validate_expert_cache_dispatch(
        expert_cache_dispatch,
    )
    gc_interval = Int(expert_gc_interval_layers)
    miss_pipeline = _qwen3_moe_validate_expert_miss_pipeline(
        expert_miss_pipeline,
    )
    read_workers = Int(expert_read_workers)
    config = load_hf_qwen3_moe_config(
        joinpath(model_dir, "config.json");
        max_seq_len=context,
    )
    model = GPTModel(config)
    _qwen3_validate_moe_semantics(model)
    0 < chunk <= context || throw(ArgumentError(
        "prefill_chunk_tokens must be in 1:context_tokens",
    ))
    cache_budget >= 0 || throw(ArgumentError(
        "expert_cache_budget_bytes must be non-negative",
    ))
    gc_interval >= 0 || throw(ArgumentError(
        "expert_gc_interval_layers must be non-negative",
    ))
    read_workers > 0 || throw(ArgumentError(
        "expert_read_workers must be positive",
    ))
    cache_dispatch === :scattered && grouped_experts && throw(ArgumentError(
        "scattered expert cache dispatch does not support grouped_experts=true",
    ))
    cache_dispatch === :scattered && cache_budget == 0 && throw(ArgumentError(
        "scattered expert cache dispatch requires a positive cache budget",
    ))
    miss_pipeline === :overlapped && cache_budget == 0 && throw(ArgumentError(
        "overlapped expert misses require a positive cache budget",
    ))
    expert_pinned_upload && miss_pipeline !== :overlapped && throw(ArgumentError(
        "pinned expert upload requires expert_miss_pipeline=:overlapped",
    ))
    reader = open_safetensors_reader(model_dir)
    _qwen3_validate_moe_tensor_names(
        model,
        Set(String.(collect(keys(reader)))),
    )
    tensors = _StreamedTensors(reader, BFloat16)

    resident_blocks = Vector{Any}(undef, model.num_layers)
    resident_parameter_bytes = 0
    for layer_index in 1:model.num_layers
        host_parameters = _streamed_qwen3_moe_block_parameters(
            model,
            tensors,
            layer_index - 1,
        )
        resident_parameter_bytes = Base.checked_add(
            resident_parameter_bytes,
            _qwen3_moe_tree_bytes(host_parameters),
        )
        resident_blocks[layer_index] = to_device(host_parameters)
        host_parameters = nothing
        GC.gc(false)
        on_resident_layer === nothing || on_resident_layer(
            layer_index,
            model.num_layers,
        )
    end
    final_scale_host = reshape(_expect_tensor(
        tensors,
        "model.norm.weight",
        (model.d_model,),
    ), model.d_model, 1, 1)
    logits_weight_host = _expect_tensor(
        tensors,
        "lm_head.weight",
        (model.vocab_size, model.d_model),
    )
    resident_parameter_bytes = Base.checked_add(
        resident_parameter_bytes,
        Base.checked_add(
            _qwen3_moe_tree_bytes(final_scale_host),
            _qwen3_moe_tree_bytes(logits_weight_host),
        ),
    )
    final_scale = to_device(final_scale_host)
    logits_weight = to_device(logits_weight_host)
    final_scale_host = nothing
    logits_weight_host = nothing
    rope = first(values(model.blocks.layers)).attn.rope
    cos_table = to_device(BFloat16.(rope.cos_cache[:, 1:context]))
    sin_table = to_device(BFloat16.(rope.sin_cache[:, 1:context]))
    caches = _qwen3_session_cache(final_scale, model, context)
    GC.gc()
    expert_pinned_upload &&
        !_qwen3_moe_pinned_upload_supported(final_scale) &&
        throw(ArgumentError(
            "pinned expert upload is not supported by the selected device",
        ))

    return HFQwen3MoEOffloadSession(
        model,
        config,
        reader,
        tensors,
        resident_blocks,
        final_scale,
        logits_weight,
        cos_table,
        sin_table,
        caches,
        0,
        context,
        chunk,
        grouped_experts,
        to_device,
        abspath(model_dir),
        resident_parameter_bytes,
        0,
        0,
        cache_budget,
        cache_policy,
        cache_dispatch,
        gc_interval,
        0,
        0,
        Dict{Tuple{Int,Int},Any}(),
        Dict{Tuple{Int,Int},Int}(),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        nothing,
        0,
        0,
        0,
        0,
        0,
        0,
        miss_pipeline,
        read_workers,
        expert_pinned_upload,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
    )
end

"""Reset the logical prefix while retaining resident tensors and static KV buffers."""
function reset_hf_qwen3_moe_offload_session!(
    session::HFQwen3MoEOffloadSession,
)
    session.position = 0
    _qwen3_moe_reset_expert_traffic!(session)
    return session
end

"""
    prefill_hf_qwen3_moe_offload!(session, prompt_tokens)

Reset and prefill a batch-one prompt in bounded chunks. Expert cache tensors are
retained across this request reset. The result contains host logits for the final
prompt token, per-chunk active expert ids, exact read/upload byte counts and
request-scoped cache traffic.
"""
function prefill_hf_qwen3_moe_offload!(
    session::HFQwen3MoEOffloadSession,
    prompt_tokens,
)
    tokens = Int.(vec(collect(prompt_tokens)))
    isempty(tokens) && throw(ArgumentError("prompt must contain at least one token"))
    _validate_generation_ids(tokens, session.model.vocab_size)
    length(tokens) <= session.context_tokens || throw(ArgumentError(
        "prompt exceeds the offload session context capacity",
    ))
    reset_hf_qwen3_moe_offload_session!(session)
    chunks = Any[]
    result = nothing
    for first_index in 1:session.prefill_chunk_tokens:length(tokens)
        last_index = min(
            first_index + session.prefill_chunk_tokens - 1,
            length(tokens),
        )
        before = session.expert_bytes_read
        before_uploaded = session.expert_bytes_uploaded
        before_hits = session.expert_cache_hits
        before_misses = session.expert_cache_misses
        before_evictions = session.expert_cache_evictions
        result = _qwen3_moe_offload_chunk!(
            session,
            reshape(tokens[first_index:last_index], :, 1),
            ;
            project_logits=last_index == length(tokens),
        )
        push!(chunks, (;
            first_token=first_index,
            last_token=last_index,
            active_experts=result.active_experts,
            expert_bytes_read=session.expert_bytes_read - before,
            expert_bytes_uploaded=
                session.expert_bytes_uploaded - before_uploaded,
            expert_cache_hits=session.expert_cache_hits - before_hits,
            expert_cache_misses=session.expert_cache_misses - before_misses,
            expert_cache_evictions=
                session.expert_cache_evictions - before_evictions,
        ))
    end
    return (;
        logits=result.logits,
        position=session.position,
        chunks=Tuple(chunks),
        expert_bytes_read=session.expert_bytes_read,
        expert_bytes_uploaded=session.expert_bytes_uploaded,
        expert_cache=qwen3_moe_expert_cache_stats(session),
    )
end

"""
Decode one batch-one token using the session's resident static KV cache and
optional expert cache. Returned read/upload and cache counters are for this
decode step; the nested cache stats cover the current prefill/decode request.
"""
function decode_hf_qwen3_moe_offload!(
    session::HFQwen3MoEOffloadSession,
    token,
)
    session.position > 0 || throw(ArgumentError(
        "prefill must run before Qwen3 MoE offload decode",
    ))
    values = Int.(vec(collect(token isa Integer ? [token] : token)))
    length(values) == 1 || throw(ArgumentError(
        "Qwen3 MoE offload decode requires exactly one token",
    ))
    _validate_generation_ids(values, session.model.vocab_size)
    before = session.expert_bytes_read
    before_uploaded = session.expert_bytes_uploaded
    before_hits = session.expert_cache_hits
    before_misses = session.expert_cache_misses
    before_evictions = session.expert_cache_evictions
    result = _qwen3_moe_offload_chunk!(session, reshape(values, 1, 1))
    return (;
        logits=result.logits,
        position=session.position,
        active_experts=result.active_experts,
        expert_bytes_read=session.expert_bytes_read - before,
        expert_bytes_uploaded=session.expert_bytes_uploaded - before_uploaded,
        expert_cache_hits=session.expert_cache_hits - before_hits,
        expert_cache_misses=session.expert_cache_misses - before_misses,
        expert_cache_evictions=
            session.expert_cache_evictions - before_evictions,
        expert_cache=qwen3_moe_expert_cache_stats(session),
    )
end
