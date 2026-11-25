using JSON3

"""
Adapter-agnostic action safety contracts for embodied and external environments.

This layer wraps any `AbstractAgentEnvironment` (not tied to GridWorld) with four
fail-closed runtime invariants:
1. Bounded action execution timeouts with explicit uncertainty accounting.
2. Structured execution-failure classification (`:not_executed` vs `:uncertain`)
   and a consecutive-failure circuit breaker.
3. Latchable emergency stop (`e-stop`), driven manually, by safety policy, or by
   an external interlock predicate, blocking all action dispatch until cleared
   with explicit operator provenance.
4. Bounded per-episode idempotent action execution keyed by `idempotency_key`,
   rejecting payload conflicts and uncertain-side-effect replays fail closed.

Every policy, request payload, e-stop state, and outcome carries a canonical
SHA-256 digest so audit logs can be verified offline without the live adapter.
"""

const AGENT_SAFETY_FORMAT_VERSION = 1

const AGENT_ACTION_OUTCOME_STATUSES = (
    :executed,
    :replayed,
    :timed_out,
    :failed,
    :estopped,
    :rejected,
)

const AGENT_FAILURE_MODES = (
    :none,
    :precondition,
    :timeout,
    :execution_error,
    :uncertain_side_effect,
    :estop_active,
    :idempotency_conflict,
)

const AGENT_SIDE_EFFECT_CERTAINTIES = (
    :committed,
    :not_executed,
    :uncertain,
)

function _safety_nonempty_string(value, label::AbstractString; max_bytes::Int=512)
    value isa AbstractString || throw(ArgumentError("$label must be a string"))
    text = String(strip(String(value)))
    !isempty(text) || throw(ArgumentError("$label must not be empty"))
    ncodeunits(text) <= max_bytes || throw(ArgumentError(
        "$label must not exceed $max_bytes bytes",
    ))
    return text
end

function _safety_positive_float64(value, label::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "$label must be a finite positive real number",
    ))
    parsed = try
        Float64(value)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("$label must be representable in Float64"))
    end
    isfinite(parsed) && parsed > 0.0 || throw(ArgumentError(
        "$label must be a finite positive real number",
    ))
    return parsed
end

function _safety_idempotency_key(value)
    value isa AbstractString || throw(ArgumentError("idempotency_key must be a string"))
    key = String(value)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$", key) || throw(ArgumentError(
        "idempotency_key must start with an alphanumeric character and contain only " *
        "letters, digits, '.', '_', ':', '/', or '-' (maximum 128 bytes)",
    ))
    return key
end

"""Structured exception an environment adapter may throw to declare failure certainty."""
struct AgentActionExecutionError <: Exception
    code::String
    message::String
    side_effect_certainty::Symbol

    function AgentActionExecutionError(
        code::AbstractString,
        message::AbstractString;
        side_effect_certainty::Symbol=:uncertain,
    )
        code_value = _safety_nonempty_string(code, "execution error code"; max_bytes=128)
        msg_value = _safety_nonempty_string(message, "execution error message"; max_bytes=2048)
        side_effect_certainty in (:not_executed, :uncertain) || throw(ArgumentError(
            "AgentActionExecutionError side_effect_certainty must be :not_executed or :uncertain",
        ))
        return new(code_value, msg_value, side_effect_certainty)
    end
end

function Base.showerror(io::IO, err::AgentActionExecutionError)
    print(
        io,
        "AgentActionExecutionError(",
        err.code,
        ", ",
        err.side_effect_certainty,
        "): ",
        err.message,
    )
end

function _action_safety_policy_payload(
    default_timeout_seconds::Float64,
    max_timeout_seconds::Float64,
    max_consecutive_failures::Int,
    estop_on_timeout::Bool,
    estop_on_uncertain_failure::Bool,
    require_idempotency_key::Bool,
    max_idempotency_entries::Int,
)
    return (;
        schema_version=AGENT_SAFETY_FORMAT_VERSION,
        kind="action_safety_policy",
        default_timeout_seconds,
        max_timeout_seconds,
        max_consecutive_failures,
        estop_on_timeout,
        estop_on_uncertain_failure,
        require_idempotency_key,
        max_idempotency_entries,
    )
end

"""Immutable, hashed safety policy for an adapter-agnostic action boundary."""
struct ActionSafetyPolicy
    default_timeout_seconds::Float64
    max_timeout_seconds::Float64
    max_consecutive_failures::Int
    estop_on_timeout::Bool
    estop_on_uncertain_failure::Bool
    require_idempotency_key::Bool
    max_idempotency_entries::Int
    sha256::String
end

function ActionSafetyPolicy(;
    default_timeout_seconds::Real=5.0,
    max_timeout_seconds::Real=30.0,
    max_consecutive_failures::Integer=3,
    estop_on_timeout::Bool=true,
    estop_on_uncertain_failure::Bool=true,
    require_idempotency_key::Bool=true,
    max_idempotency_entries::Integer=4096,
)
    default_t = _safety_positive_float64(default_timeout_seconds, "default_timeout_seconds")
    max_t = _safety_positive_float64(max_timeout_seconds, "max_timeout_seconds")
    default_t <= max_t || throw(ArgumentError(
        "default_timeout_seconds must not exceed max_timeout_seconds",
    ))
    max_failures = _strict_host_int(max_consecutive_failures, "max_consecutive_failures")
    max_failures >= 1 || throw(ArgumentError("max_consecutive_failures must be >= 1"))
    max_entries = _strict_host_int(max_idempotency_entries, "max_idempotency_entries")
    max_entries >= 1 || throw(ArgumentError("max_idempotency_entries must be >= 1"))
    payload = _action_safety_policy_payload(
        default_t,
        max_t,
        max_failures,
        estop_on_timeout,
        estop_on_uncertain_failure,
        require_idempotency_key,
        max_entries,
    )
    return ActionSafetyPolicy(
        default_t,
        max_t,
        max_failures,
        estop_on_timeout,
        estop_on_uncertain_failure,
        require_idempotency_key,
        max_entries,
        _sha256_hex(String(JSON3.write(payload))),
    )
end

function _safe_action_payload_sha256(action::AgentEnvironmentAction)
    return _sha256_hex(String(JSON3.write((;
        schema_version=AGENT_SAFETY_FORMAT_VERSION,
        name=action.name,
        value=action.value,
    ))))
end

"""Validated action dispatch request with optional idempotency key and timeout override."""
struct SafeActionRequest
    action::AgentEnvironmentAction
    idempotency_key::Union{Nothing,String}
    timeout_seconds::Union{Nothing,Float64}
    payload_sha256::String
end

function SafeActionRequest(
    action::AgentEnvironmentAction;
    idempotency_key=nothing,
    timeout_seconds=nothing,
)
    _safety_nonempty_string(action.name, "action.name"; max_bytes=128)
    key_value = idempotency_key === nothing ? nothing : _safety_idempotency_key(idempotency_key)
    timeout_value = timeout_seconds === nothing ? nothing :
        _safety_positive_float64(timeout_seconds, "timeout_seconds")
    return SafeActionRequest(
        action,
        key_value,
        timeout_value,
        _safe_action_payload_sha256(action),
    )
end

function _estop_state_payload(
    active::Bool,
    reason::Union{Nothing,String},
    source::Union{Nothing,String},
    latched_at_sequence::Int,
    clear_count::Int,
)
    return (;
        schema_version=AGENT_SAFETY_FORMAT_VERSION,
        kind="emergency_stop_state",
        active,
        reason,
        source,
        latched_at_sequence,
        clear_count,
    )
end

"""Immutable snapshot of the adapter emergency-stop latch."""
struct EmergencyStopState
    active::Bool
    reason::Union{Nothing,String}
    source::Union{Nothing,String}
    latched_at_sequence::Int
    clear_count::Int
    sha256::String
end

function _make_estop_state(
    active::Bool,
    reason::Union{Nothing,String},
    source::Union{Nothing,String},
    latched_at_sequence::Int,
    clear_count::Int,
)
    payload = _estop_state_payload(active, reason, source, latched_at_sequence, clear_count)
    return EmergencyStopState(
        active,
        reason,
        source,
        latched_at_sequence,
        clear_count,
        _sha256_hex(String(JSON3.write(payload))),
    )
end

"""Audit event for explicit e-stop latch/clear or adapter reset transitions."""
struct SafeAdapterSafetyEvent
    event_index::Int
    kind::String
    sequence::Int
    source::String
    operator::Union{Nothing,String}
    reason::String
    estop_active_after::Bool
    sha256::String
end

function _make_safety_event(
    event_index::Int,
    kind::String,
    sequence::Int,
    source::String,
    operator::Union{Nothing,String},
    reason::String,
    estop_active_after::Bool,
)
    payload = (;
        schema_version=AGENT_SAFETY_FORMAT_VERSION,
        kind,
        event_index,
        sequence,
        source,
        operator,
        reason,
        estop_active_after,
    )
    return SafeAdapterSafetyEvent(
        event_index,
        kind,
        sequence,
        source,
        operator,
        reason,
        estop_active_after,
        _sha256_hex(String(JSON3.write(payload))),
    )
end

function _safe_outcome_deterministic_payload(
    sequence::Int,
    request::SafeActionRequest,
    status::Symbol,
    failure_mode::Symbol,
    side_effect_certainty::Symbol,
    replayed_from_sequence::Union{Nothing,Int},
    transition_sha256::Union{Nothing,String},
    error::Union{Nothing,String},
    estop_triggered::Bool,
    estop_active_after::Bool,
)
    return (;
        schema_version=AGENT_SAFETY_FORMAT_VERSION,
        kind="safe_action_outcome",
        sequence,
        action=(; name=request.action.name, value=request.action.value),
        idempotency_key=request.idempotency_key,
        payload_sha256=request.payload_sha256,
        status=String(status),
        failure_mode=String(failure_mode),
        side_effect_certainty=String(side_effect_certainty),
        replayed_from_sequence,
        transition_sha256,
        error,
        estop_triggered,
        estop_active_after,
    )
end

"""Canonical result of one action dispatch through [`SafeEnvironmentAdapter`](@ref)."""
struct SafeActionOutcome
    sequence::Int
    request::SafeActionRequest
    status::Symbol
    failure_mode::Symbol
    side_effect_certainty::Symbol
    replayed_from_sequence::Union{Nothing,Int}
    transition::Union{Nothing,AgentEnvironmentTransition}
    error::Union{Nothing,String}
    estop_triggered::Bool
    estop_active_after::Bool
    elapsed_seconds::Float64
    sha256::String
end

function _make_safe_outcome(
    sequence::Int,
    request::SafeActionRequest,
    status::Symbol,
    failure_mode::Symbol,
    side_effect_certainty::Symbol,
    replayed_from_sequence::Union{Nothing,Int},
    transition::Union{Nothing,AgentEnvironmentTransition},
    error::Union{Nothing,String},
    estop_triggered::Bool,
    estop_active_after::Bool,
    elapsed_seconds::Float64,
)
    status in AGENT_ACTION_OUTCOME_STATUSES || throw(ArgumentError(
        "unsupported outcome status $(repr(status))",
    ))
    failure_mode in AGENT_FAILURE_MODES || throw(ArgumentError(
        "unsupported failure mode $(repr(failure_mode))",
    ))
    side_effect_certainty in AGENT_SIDE_EFFECT_CERTAINTIES || throw(ArgumentError(
        "unsupported side_effect_certainty $(repr(side_effect_certainty))",
    ))
    transition_sha256 = transition === nothing ? nothing : transition.sha256
    payload = _safe_outcome_deterministic_payload(
        sequence,
        request,
        status,
        failure_mode,
        side_effect_certainty,
        replayed_from_sequence,
        transition_sha256,
        error,
        estop_triggered,
        estop_active_after,
    )
    return SafeActionOutcome(
        sequence,
        request,
        status,
        failure_mode,
        side_effect_certainty,
        replayed_from_sequence,
        transition,
        error,
        estop_triggered,
        estop_active_after,
        elapsed_seconds,
        _sha256_hex(String(JSON3.write(payload))),
    )
end

"""
    SafeEnvironmentAdapter(environment; adapter_id, policy=ActionSafetyPolicy(),
                           external_interlock=nothing)

Wrap any `AbstractAgentEnvironment` with timeout, execution-failure, e-stop, and
idempotent-action safety enforcement.
"""
mutable struct SafeEnvironmentAdapter{E<:AbstractAgentEnvironment} <: AbstractAgentEnvironment
    adapter_id::String
    environment::E
    policy::ActionSafetyPolicy
    estop::EmergencyStopState
    external_interlock::Union{Nothing,Function}
    sequence::Int
    consecutive_failures::Int
    in_flight_tasks::Vector{Task}
    idempotency_cache::Dict{String,SafeActionOutcome}
    idempotency_order::Vector{String}
    audit_log::Vector{SafeActionOutcome}
    safety_events::Vector{SafeAdapterSafetyEvent}
    dispatch_lock::ReentrantLock
end

function SafeEnvironmentAdapter(
    environment::E;
    adapter_id::AbstractString="safe-adapter",
    policy::ActionSafetyPolicy=ActionSafetyPolicy(),
    external_interlock::Union{Nothing,Function}=nothing,
) where {E<:AbstractAgentEnvironment}
    id_value = _agent_environment_id(String(adapter_id))
    return SafeEnvironmentAdapter{E}(
        id_value,
        environment,
        policy,
        _make_estop_state(false, nothing, nothing, 0, 0),
        external_interlock,
        0,
        0,
        Task[],
        Dict{String,SafeActionOutcome}(),
        String[],
        SafeActionOutcome[],
        SafeAdapterSafetyEvent[],
        ReentrantLock(),
    )
end

observe_agent_environment(adapter::SafeEnvironmentAdapter) =
    lock(adapter.dispatch_lock) do
        observe_agent_environment(adapter.environment)
    end

function _has_unfinished_timed_out_task(adapter::SafeEnvironmentAdapter)
    filter!(task -> !istaskdone(task), adapter.in_flight_tasks)
    return !isempty(adapter.in_flight_tasks)
end

function _check_external_interlock(adapter::SafeEnvironmentAdapter)
    adapter.external_interlock === nothing && return (false, nothing)
    result = try
        adapter.external_interlock()
    catch error
        error isa InterruptException && rethrow()
        return (true, "external interlock check failed: " * sprint(showerror, error))
    end
    if result isa Bool
        return (result, result ? "external interlock asserted" : nothing)
    elseif result isa Tuple && length(result) == 2 && result[1] isa Bool
        tripped = result[1]
        detail = result[2] === nothing ? "external interlock asserted" : String(result[2])
        return (tripped, tripped ? detail : nothing)
    end
    return (true, "external interlock returned invalid value $(repr(result))")
end

function _latch_estop_locked!(
    adapter::SafeEnvironmentAdapter,
    reason::AbstractString,
    source::AbstractString;
    sequence::Int=adapter.sequence,
)
    reason_text = _safety_nonempty_string(reason, "estop reason"; max_bytes=512)
    source_text = _safety_nonempty_string(source, "estop source"; max_bytes=128)
    adapter.estop.active && return false
    adapter.estop = _make_estop_state(
        true,
        reason_text,
        source_text,
        sequence,
        adapter.estop.clear_count,
    )
    push!(
        adapter.safety_events,
        _make_safety_event(
            length(adapter.safety_events) + 1,
            "estop_latched",
            sequence,
            source_text,
            nothing,
            reason_text,
            true,
        ),
    )
    return true
end

"""Latch the adapter emergency stop. Idempotent if already latched."""
function trigger_agent_estop!(
    adapter::SafeEnvironmentAdapter,
    reason::AbstractString;
    source::AbstractString="manual",
)
    lock(adapter.dispatch_lock) do
        _latch_estop_locked!(adapter, reason, source; sequence=adapter.sequence)
        return adapter.estop
    end
end

"""
    clear_agent_estop!(adapter; operator, reason, force=false)

Explicitly clear a latched e-stop with operator provenance. Fails closed if an
external hardware interlock is still active, or if a timed-out underlying task is
still running unless `force=true` is explicitly specified.
"""
function clear_agent_estop!(
    adapter::SafeEnvironmentAdapter;
    operator::AbstractString,
    reason::AbstractString,
    force::Bool=false,
)
    operator_text = _safety_nonempty_string(operator, "operator"; max_bytes=128)
    reason_text = _safety_nonempty_string(reason, "clear reason"; max_bytes=512)
    return lock(adapter.dispatch_lock) do
        tripped, interlock_reason = _check_external_interlock(adapter)
        tripped && throw(ArgumentError(
            "cannot clear e-stop while external interlock is active: $(something(interlock_reason, "interlock active"))",
        ))
        if !force && _has_unfinished_timed_out_task(adapter)
            throw(ArgumentError(
                "cannot clear e-stop while a timed-out adapter task is still running; wait for completion or pass force=true",
            ))
        end
        adapter.estop.active || return adapter.estop
        adapter.estop = _make_estop_state(
            false,
            nothing,
            nothing,
            0,
            adapter.estop.clear_count + 1,
        )
        adapter.consecutive_failures = 0
        push!(
            adapter.safety_events,
            _make_safety_event(
                length(adapter.safety_events) + 1,
                "estop_cleared",
                adapter.sequence,
                "operator",
                operator_text,
                reason_text,
                false,
            ),
        )
        return adapter.estop
    end
end

"""
    reset_agent_environment!(adapter::SafeEnvironmentAdapter)

Reset the wrapped environment and clear per-episode idempotency entries. Fails
closed while e-stop is active or an unacknowledged timed-out task is running.
"""
function reset_agent_environment!(adapter::SafeEnvironmentAdapter)
    return lock(adapter.dispatch_lock) do
        adapter.estop.active && throw(ArgumentError(
            "cannot reset environment while emergency stop is active; clear e-stop explicitly first",
        ))
        _has_unfinished_timed_out_task(adapter) && throw(ArgumentError(
            "cannot reset environment while a timed-out adapter task is still running",
        ))
        reset_agent_environment!(adapter.environment)
        empty!(adapter.idempotency_cache)
        empty!(adapter.idempotency_order)
        adapter.consecutive_failures = 0
        push!(
            adapter.safety_events,
            _make_safety_event(
                length(adapter.safety_events) + 1,
                "environment_reset",
                adapter.sequence,
                "adapter",
                nothing,
                "episode reset",
                false,
            ),
        )
        return adapter
    end
end

function _record_outcome_locked!(
    adapter::SafeEnvironmentAdapter,
    outcome::SafeActionOutcome;
    cache_idempotency::Bool=false,
)
    push!(adapter.audit_log, outcome)
    if cache_idempotency && outcome.request.idempotency_key !== nothing
        key = outcome.request.idempotency_key
        if !haskey(adapter.idempotency_cache, key)
            push!(adapter.idempotency_order, key)
        end
        adapter.idempotency_cache[key] = outcome
    end
    return outcome
end

function _run_with_timeout(thunk::Function, timeout_seconds::Float64)
    result_channel = Channel{Tuple{Symbol,Any}}(1)
    worker = @async begin
        try
            value = thunk()
            put!(result_channel, (:ok, value))
        catch error
            put!(result_channel, (:error, error))
        end
    end
    wait_status = timedwait(() -> isready(result_channel), timeout_seconds; pollint=0.002)
    if wait_status === :timed_out
        return (:timed_out, nothing, worker)
    end
    kind, payload = take!(result_channel)
    return (kind, payload, worker)
end

"""
    execute_safe_action!(adapter::SafeEnvironmentAdapter, request::SafeActionRequest)
    execute_safe_action!(adapter::SafeEnvironmentAdapter, action::AgentEnvironmentAction;
                         idempotency_key=nothing, timeout_seconds=nothing)

Dispatch one action through the adapter safety boundary. Never throws for normal
policy/timeout/execution/e-stop outcomes; returns a canonical [`SafeActionOutcome`](@ref).
"""
function execute_safe_action!(
    adapter::SafeEnvironmentAdapter,
    request::SafeActionRequest,
)
    return lock(adapter.dispatch_lock) do
        started = time()
        adapter.sequence += 1
        seq = adapter.sequence
        policy = adapter.policy

        # 1. Check external interlock first (hardware/watchdog e-stop).
        tripped, interlock_reason = _check_external_interlock(adapter)
        estop_triggered = false
        if tripped
            estop_triggered = _latch_estop_locked!(
                adapter,
                something(interlock_reason, "external interlock asserted"),
                "external_interlock";
                sequence=seq,
            )
        end

        # 2. E-stop overrides everything fail-closed.
        if adapter.estop.active
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :estopped,
                    :estop_active,
                    :not_executed,
                    nothing,
                    nothing,
                    "emergency stop active (" *
                    something(adapter.estop.source, "unknown") * "): " *
                    something(adapter.estop.reason, "unspecified"),
                    estop_triggered,
                    true,
                    time() - started,
                ),
            )
        end

        # 3. Validate timeout override against policy ceiling.
        effective_timeout = request.timeout_seconds === nothing ?
            policy.default_timeout_seconds : request.timeout_seconds
        if effective_timeout > policy.max_timeout_seconds
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :rejected,
                    :precondition,
                    :not_executed,
                    nothing,
                    nothing,
                    "requested timeout_seconds ($effective_timeout) exceeds policy max_timeout_seconds ($(policy.max_timeout_seconds))",
                    false,
                    false,
                    time() - started,
                ),
            )
        end

        # 4. Require idempotency key when mandated by policy.
        key = request.idempotency_key
        if policy.require_idempotency_key && key === nothing
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :rejected,
                    :precondition,
                    :not_executed,
                    nothing,
                    nothing,
                    "policy requires a non-empty idempotency_key",
                    false,
                    false,
                    time() - started,
                ),
            )
        end

        # 5. Idempotency lookup (deduplicate retries, fail closed on conflicts or uncertain state).
        if key !== nothing && haskey(adapter.idempotency_cache, key)
            cached = adapter.idempotency_cache[key]
            if cached.request.payload_sha256 != request.payload_sha256
                return _record_outcome_locked!(
                    adapter,
                    _make_safe_outcome(
                        seq,
                        request,
                        :rejected,
                        :idempotency_conflict,
                        :not_executed,
                        cached.sequence,
                        nothing,
                        "idempotency_key $(repr(key)) was already used for a different action payload",
                        false,
                        adapter.estop.active,
                        time() - started,
                    ),
                )
            end
            if cached.side_effect_certainty === :uncertain || cached.status === :timed_out
                return _record_outcome_locked!(
                    adapter,
                    _make_safe_outcome(
                        seq,
                        request,
                        :rejected,
                        :uncertain_side_effect,
                        :uncertain,
                        cached.sequence,
                        nothing,
                        "idempotency_key $(repr(key)) has uncertain side effects from sequence $(cached.sequence); manual verification and a new key are required",
                        false,
                        adapter.estop.active,
                        time() - started,
                    ),
                )
            end
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :replayed,
                    cached.failure_mode,
                    cached.side_effect_certainty,
                    cached.sequence,
                    cached.transition,
                    cached.error,
                    false,
                    adapter.estop.active,
                    time() - started,
                ),
            )
        end

        # 6. Bound idempotency cache size without silently evicting old keys.
        if key !== nothing && length(adapter.idempotency_cache) >= policy.max_idempotency_entries
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :rejected,
                    :precondition,
                    :not_executed,
                    nothing,
                    nothing,
                    "idempotency cache reached policy limit ($(policy.max_idempotency_entries))",
                    false,
                    false,
                    time() - started,
                ),
            )
        end

        # 7. Refuse dispatch if an earlier timed-out task is still executing on the adapter.
        if _has_unfinished_timed_out_task(adapter)
            estop_triggered = _latch_estop_locked!(
                adapter,
                "previous timed-out adapter action is still running",
                "timeout";
                sequence=seq,
            )
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :estopped,
                    :estop_active,
                    :not_executed,
                    nothing,
                    nothing,
                    "previous timed-out adapter action is still running; emergency stop latched",
                    estop_triggered,
                    adapter.estop.active,
                    time() - started,
                ),
            )
        end

        # 8. Dispatch underlying environment action under bounded timeout.
        kind, payload, worker = _run_with_timeout(
            () -> step_agent_environment!(adapter.environment, request.action),
            effective_timeout,
        )
        elapsed = time() - started

        if kind === :ok
            if !(payload isa AgentEnvironmentTransition)
                adapter.consecutive_failures += 1
                msg = "underlying adapter returned $(typeof(payload)) instead of AgentEnvironmentTransition"
                if policy.estop_on_uncertain_failure
                    estop_triggered = _latch_estop_locked!(
                        adapter,
                        msg,
                        "uncertain_failure";
                        sequence=seq,
                    )
                elseif adapter.consecutive_failures >= policy.max_consecutive_failures
                    estop_triggered = _latch_estop_locked!(
                        adapter,
                        "consecutive action failures reached $(policy.max_consecutive_failures)",
                        "consecutive_failures";
                        sequence=seq,
                    )
                end
                return _record_outcome_locked!(
                    adapter,
                    _make_safe_outcome(
                        seq,
                        request,
                        :failed,
                        :uncertain_side_effect,
                        :uncertain,
                        nothing,
                        nothing,
                        msg,
                        estop_triggered,
                        adapter.estop.active,
                        elapsed,
                    );
                    cache_idempotency=true,
                )
            end
            adapter.consecutive_failures = 0
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :executed,
                    :none,
                    :committed,
                    nothing,
                    payload,
                    nothing,
                    false,
                    adapter.estop.active,
                    elapsed,
                );
                cache_idempotency=true,
            )
        elseif kind === :timed_out
            push!(adapter.in_flight_tasks, worker)
            adapter.consecutive_failures += 1
            msg = "action $(repr(request.action.name)) timed out after $(effective_timeout)s"
            if policy.estop_on_timeout
                estop_triggered = _latch_estop_locked!(
                    adapter,
                    msg,
                    "timeout";
                    sequence=seq,
                )
            elseif adapter.consecutive_failures >= policy.max_consecutive_failures
                estop_triggered = _latch_estop_locked!(
                    adapter,
                    "consecutive action failures reached $(policy.max_consecutive_failures)",
                    "consecutive_failures";
                    sequence=seq,
                )
            end
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :timed_out,
                    :timeout,
                    :uncertain,
                    nothing,
                    nothing,
                    msg,
                    estop_triggered,
                    adapter.estop.active,
                    elapsed,
                );
                cache_idempotency=true,
            )
        else
            # Execution raised an exception.
            err = payload
            err isa InterruptException && rethrow(err)
            adapter.consecutive_failures += 1
            failure_mode, certainty, msg = if err isa AgentActionExecutionError
                mode = err.side_effect_certainty === :uncertain ?
                    :uncertain_side_effect : :execution_error
                (mode, err.side_effect_certainty, sprint(showerror, err))
            elseif err isa ArgumentError
                (:precondition, :not_executed, sprint(showerror, err))
            else
                (:uncertain_side_effect, :uncertain, sprint(showerror, err))
            end

            if certainty === :uncertain && policy.estop_on_uncertain_failure
                estop_triggered = _latch_estop_locked!(
                    adapter,
                    msg,
                    "uncertain_failure";
                    sequence=seq,
                )
            elseif adapter.consecutive_failures >= policy.max_consecutive_failures
                estop_triggered = _latch_estop_locked!(
                    adapter,
                    "consecutive action failures reached $(policy.max_consecutive_failures)",
                    "consecutive_failures";
                    sequence=seq,
                )
            end
            return _record_outcome_locked!(
                adapter,
                _make_safe_outcome(
                    seq,
                    request,
                    :failed,
                    failure_mode,
                    certainty,
                    nothing,
                    nothing,
                    msg,
                    estop_triggered,
                    adapter.estop.active,
                    elapsed,
                );
                cache_idempotency=true,
            )
        end
    end
end

function execute_safe_action!(
    adapter::SafeEnvironmentAdapter,
    action::AgentEnvironmentAction;
    idempotency_key=nothing,
    timeout_seconds=nothing,
)
    request = SafeActionRequest(action; idempotency_key, timeout_seconds)
    return execute_safe_action!(adapter, request)
end

function step_agent_environment!(
    adapter::SafeEnvironmentAdapter,
    action::AgentEnvironmentAction;
    idempotency_key=nothing,
    timeout_seconds=nothing,
)
    outcome = execute_safe_action!(adapter, action; idempotency_key, timeout_seconds)
    outcome.transition === nothing && throw(AgentActionExecutionError(
        String(outcome.failure_mode),
        something(outcome.error, "safe action did not produce a transition");
        side_effect_certainty=outcome.side_effect_certainty === :committed ?
            :uncertain : outcome.side_effect_certainty,
    ))
    return outcome.transition
end

"""Render a canonical JSON tool response from one [`SafeActionOutcome`](@ref)."""
function render_safe_action_outcome(
    outcome::SafeActionOutcome;
    feedback::Symbol=:full,
)
    feedback in (:full, :none) || throw(ArgumentError("feedback must be :full or :none"))
    obs = outcome.transition === nothing ? nothing :
        JSON3.read(render_agent_environment_feedback(outcome.transition; feedback))
    payload = (;
        schema_version=AGENT_SAFETY_FORMAT_VERSION,
        sequence=outcome.sequence,
        idempotency_key=outcome.request.idempotency_key,
        status=String(outcome.status),
        failure_mode=String(outcome.failure_mode),
        side_effect_certainty=String(outcome.side_effect_certainty),
        replayed_from_sequence=outcome.replayed_from_sequence,
        estop_triggered=outcome.estop_triggered,
        estop_active=outcome.estop_active_after,
        error=outcome.error,
        observation=obs,
        outcome_sha256=outcome.sha256,
    )
    return String(JSON3.write(payload))
end

"""
    safe_environment_tool(adapter; name, description, properties, required,
                          action_from_arguments, feedback=:full)

Create an adapter-agnostic `AgentTool` bound to a [`SafeEnvironmentAdapter`](@ref).
Exposes `idempotency_key` (required when `adapter.policy.require_idempotency_key`)
and optional `timeout_seconds` alongside the adapter-specific arguments.
"""
function safe_environment_tool(
    adapter::SafeEnvironmentAdapter;
    name::AbstractString,
    description::AbstractString,
    properties::NamedTuple,
    required::AbstractVector{<:AbstractString},
    action_from_arguments::Function,
    feedback::Symbol=:full,
)
    feedback in (:full, :none) || throw(ArgumentError("feedback must be :full or :none"))
    merged_properties = merge(
        properties,
        (;
            idempotency_key=(;
                type="string",
                description="Unique idempotency key identifying this action intent.",
            ),
            timeout_seconds=(;
                type="number",
                description="Optional execution timeout override in seconds.",
            ),
        ),
    )
    merged_required = String[String(item) for item in required]
    if adapter.policy.require_idempotency_key && !("idempotency_key" in merged_required)
        push!(merged_required, "idempotency_key")
    end
    allowed_keys = Set(String(key) for key in keys(merged_properties))
    return AgentTool(;
        name=String(name),
        description=String(description),
        properties=merged_properties,
        required=merged_required,
        handler=function (arguments, _coerced)
            actual_keys = Set(String(key) for key in keys(arguments))
            issubset(actual_keys, allowed_keys) || throw(ArgumentError(
                "unexpected arguments for safe tool $(repr(String(name)))",
            ))
            raw_key = get(arguments, "idempotency_key", nothing)
            raw_timeout = get(arguments, "timeout_seconds", nothing)
            action = action_from_arguments(arguments)
            action isa AgentEnvironmentAction || throw(ArgumentError(
                "action_from_arguments must return AgentEnvironmentAction",
            ))
            outcome = execute_safe_action!(
                adapter,
                action;
                idempotency_key=raw_key,
                timeout_seconds=raw_timeout,
            )
            return render_safe_action_outcome(outcome; feedback)
        end,
    )
end

"""GridWorld move tool wrapped with [`SafeEnvironmentAdapter`](@ref)."""
function safe_gridworld_move_tool(
    adapter::SafeEnvironmentAdapter{GridWorldEnvironment};
    feedback::Symbol=:full,
)
    return safe_environment_tool(
        adapter;
        name="move",
        description="Move one cell in the deterministic GridWorld through the safety adapter.",
        properties=(;
            direction=(;
                type="string",
                enum=collect(GRIDWORLD_DIRECTIONS),
                description="One of north, east, south, or west.",
            ),
        ),
        required=["direction"],
        action_from_arguments=function (arguments)
            direction = _tool_string(arguments, "direction")
            return AgentEnvironmentAction("move", direction)
        end,
        feedback,
    )
end

safe_gridworld_tool_registry(
    adapter::SafeEnvironmentAdapter{GridWorldEnvironment};
    feedback::Symbol=:full,
) = ToolRegistry([safe_gridworld_move_tool(adapter; feedback)])

"""Canonical JSON-ready audit record for offline cross-process safety verification."""
function safe_adapter_audit_payload(adapter::SafeEnvironmentAdapter)
    return lock(adapter.dispatch_lock) do
        policy = adapter.policy
        estop = adapter.estop
        outcomes = [
            (;
                sequence=outcome.sequence,
                action=(; name=outcome.request.action.name, value=outcome.request.action.value),
                idempotency_key=outcome.request.idempotency_key,
                payload_sha256=outcome.request.payload_sha256,
                status=String(outcome.status),
                failure_mode=String(outcome.failure_mode),
                side_effect_certainty=String(outcome.side_effect_certainty),
                replayed_from_sequence=outcome.replayed_from_sequence,
                transition_sha256=outcome.transition === nothing ? nothing : outcome.transition.sha256,
                error=outcome.error,
                estop_triggered=outcome.estop_triggered,
                estop_active_after=outcome.estop_active_after,
                sha256=outcome.sha256,
            )
            for outcome in adapter.audit_log
        ]
        safety_events = [
            (;
                event_index=ev.event_index,
                kind=ev.kind,
                sequence=ev.sequence,
                source=ev.source,
                operator=ev.operator,
                reason=ev.reason,
                estop_active_after=ev.estop_active_after,
                sha256=ev.sha256,
            )
            for ev in adapter.safety_events
        ]
        summary = (;
            total_requests=length(adapter.audit_log),
            executed=count(o -> o.status === :executed, adapter.audit_log),
            replayed=count(o -> o.status === :replayed, adapter.audit_log),
            timed_out=count(o -> o.status === :timed_out, adapter.audit_log),
            failed=count(o -> o.status === :failed, adapter.audit_log),
            estopped=count(o -> o.status === :estopped, adapter.audit_log),
            rejected=count(o -> o.status === :rejected, adapter.audit_log),
            estop_triggers=count(o -> o.estop_triggered, adapter.audit_log) +
                count(ev -> ev.kind == "estop_latched" && ev.source == "manual", adapter.safety_events),
            estop_clears=estop.clear_count,
        )
        body = (;
            schema_version=AGENT_SAFETY_FORMAT_VERSION,
            adapter_id=adapter.adapter_id,
            policy=(;
                default_timeout_seconds=policy.default_timeout_seconds,
                max_timeout_seconds=policy.max_timeout_seconds,
                max_consecutive_failures=policy.max_consecutive_failures,
                estop_on_timeout=policy.estop_on_timeout,
                estop_on_uncertain_failure=policy.estop_on_uncertain_failure,
                require_idempotency_key=policy.require_idempotency_key,
                max_idempotency_entries=policy.max_idempotency_entries,
                sha256=policy.sha256,
            ),
            estop=(;
                active=estop.active,
                reason=estop.reason,
                source=estop.source,
                latched_at_sequence=estop.latched_at_sequence,
                clear_count=estop.clear_count,
                sha256=estop.sha256,
            ),
            outcomes,
            safety_events,
            summary,
        )
        return merge(body, (; audit_sha256=_sha256_hex(String(JSON3.write(body)))))
    end
end

"""
    verify_safe_adapter_audit(payload)

Independently verify a serialized safety audit payload (from `JSON3.read` or
`safe_adapter_audit_payload`) without instantiating the live environment adapter.
Checks policy digest, e-stop digest, every outcome/event digest, contiguous
sequence numbering, e-stop blocking invariant, and idempotency consistency.
"""
function verify_safe_adapter_audit(payload)
    version = get(payload, :schema_version, nothing)
    version isa Integer && !(version isa Bool) && Int(version) == AGENT_SAFETY_FORMAT_VERSION ||
        throw(ArgumentError("unsupported safety audit schema_version $(repr(version))"))
    adapter_id = _agent_environment_id(String(payload.adapter_id))

    # Recompute policy digest.
    p = payload.policy
    rebuilt_policy = ActionSafetyPolicy(;
        default_timeout_seconds=p.default_timeout_seconds,
        max_timeout_seconds=p.max_timeout_seconds,
        max_consecutive_failures=p.max_consecutive_failures,
        estop_on_timeout=Bool(p.estop_on_timeout),
        estop_on_uncertain_failure=Bool(p.estop_on_uncertain_failure),
        require_idempotency_key=Bool(p.require_idempotency_key),
        max_idempotency_entries=p.max_idempotency_entries,
    )
    String(p.sha256) == rebuilt_policy.sha256 || throw(ArgumentError(
        "safety audit policy sha256 mismatch",
    ))

    # Recompute e-stop digest.
    e = payload.estop
    rebuilt_estop = _make_estop_state(
        Bool(e.active),
        e.reason === nothing ? nothing : String(e.reason),
        e.source === nothing ? nothing : String(e.source),
        Int(e.latched_at_sequence),
        Int(e.clear_count),
    )
    String(e.sha256) == rebuilt_estop.sha256 || throw(ArgumentError(
        "safety audit estop sha256 mismatch",
    ))

    # Verify every safety event digest and monotonicity.
    verified_events = 0
    for (idx, ev) in enumerate(payload.safety_events)
        Int(ev.event_index) == idx || throw(ArgumentError(
            "safety event index mismatch at $idx",
        ))
        rebuilt_ev = _make_safety_event(
            Int(ev.event_index),
            String(ev.kind),
            Int(ev.sequence),
            String(ev.source),
            ev.operator === nothing ? nothing : String(ev.operator),
            String(ev.reason),
            Bool(ev.estop_active_after),
        )
        String(ev.sha256) == rebuilt_ev.sha256 || throw(ArgumentError(
            "safety event digest mismatch at index $idx",
        ))
        verified_events += 1
    end

    # Verify every outcome digest, sequence contiguity, and safety invariants.
    by_sequence = Dict{Int,Any}()
    verified_outcomes = 0
    for (idx, row) in enumerate(payload.outcomes)
        seq = Int(row.sequence)
        seq == idx || throw(ArgumentError(
            "safety outcome sequence must be contiguous 1-based; expected $idx, got $seq",
        ))
        action = AgentEnvironmentAction(String(row.action.name), String(row.action.value))
        expected_payload_sha = _safe_action_payload_sha256(action)
        String(row.payload_sha256) == expected_payload_sha || throw(ArgumentError(
            "action payload sha256 mismatch at sequence $seq",
        ))
        status = Symbol(String(row.status))
        failure_mode = Symbol(String(row.failure_mode))
        certainty = Symbol(String(row.side_effect_certainty))
        status in AGENT_ACTION_OUTCOME_STATUSES || throw(ArgumentError(
            "unknown outcome status $(repr(status)) at sequence $seq",
        ))
        failure_mode in AGENT_FAILURE_MODES || throw(ArgumentError(
            "unknown failure mode $(repr(failure_mode)) at sequence $seq",
        ))
        certainty in AGENT_SIDE_EFFECT_CERTAINTIES || throw(ArgumentError(
            "unknown certainty $(repr(certainty)) at sequence $seq",
        ))
        idempotency_key = row.idempotency_key === nothing ? nothing : String(row.idempotency_key)
        replayed_from = row.replayed_from_sequence === nothing ? nothing : Int(row.replayed_from_sequence)
        transition_sha = row.transition_sha256 === nothing ? nothing : String(row.transition_sha256)
        err_str = row.error === nothing ? nothing : String(row.error)
        estop_triggered = Bool(row.estop_triggered)
        estop_active_after = Bool(row.estop_active_after)

        # Structural safety invariants:
        if status === :executed
            certainty === :committed || throw(ArgumentError(
                "executed outcome must have :committed certainty at sequence $seq",
            ))
            !estop_active_after || throw(ArgumentError(
                "executed outcome cannot leave e-stop active at sequence $seq",
            ))
            transition_sha !== nothing || throw(ArgumentError(
                "executed outcome must reference a transition sha256 at sequence $seq",
            ))
        elseif status === :estopped
            failure_mode === :estop_active && certainty === :not_executed && estop_active_after ||
                throw(ArgumentError("invalid estopped outcome invariants at sequence $seq"))
        elseif status === :replayed
            replayed_from !== nothing && haskey(by_sequence, replayed_from) ||
                throw(ArgumentError("replayed outcome references missing origin at sequence $seq"))
            origin = by_sequence[replayed_from]
            String(origin.payload_sha256) == expected_payload_sha || throw(ArgumentError(
                "replayed outcome payload disagrees with origin at sequence $seq",
            ))
            String(origin.transition_sha256 === nothing ? "" : origin.transition_sha256) ==
                String(transition_sha === nothing ? "" : transition_sha) || throw(ArgumentError(
                    "replayed outcome transition digest disagrees with origin at sequence $seq",
                ))
        elseif failure_mode === :idempotency_conflict
            status === :rejected && certainty === :not_executed && replayed_from !== nothing &&
                haskey(by_sequence, replayed_from) || throw(ArgumentError(
                    "invalid idempotency conflict record at sequence $seq",
                ))
            String(by_sequence[replayed_from].payload_sha256) != expected_payload_sha ||
                throw(ArgumentError(
                    "idempotency conflict record shares identical payload at sequence $seq",
                ))
        end

        req = SafeActionRequest(action; idempotency_key)
        det = _safe_outcome_deterministic_payload(
            seq,
            req,
            status,
            failure_mode,
            certainty,
            replayed_from,
            transition_sha,
            err_str,
            estop_triggered,
            estop_active_after,
        )
        _sha256_hex(String(JSON3.write(det))) == String(row.sha256) || throw(ArgumentError(
            "safety outcome sha256 mismatch at sequence $seq",
        ))
        by_sequence[seq] = row
        verified_outcomes += 1
    end

    # Verify outer audit_sha256 by reconstructing the body object in canonical order.
    body = (;
        schema_version=AGENT_SAFETY_FORMAT_VERSION,
        adapter_id,
        policy=(;
            default_timeout_seconds=rebuilt_policy.default_timeout_seconds,
            max_timeout_seconds=rebuilt_policy.max_timeout_seconds,
            max_consecutive_failures=rebuilt_policy.max_consecutive_failures,
            estop_on_timeout=rebuilt_policy.estop_on_timeout,
            estop_on_uncertain_failure=rebuilt_policy.estop_on_uncertain_failure,
            require_idempotency_key=rebuilt_policy.require_idempotency_key,
            max_idempotency_entries=rebuilt_policy.max_idempotency_entries,
            sha256=rebuilt_policy.sha256,
        ),
        estop=(;
            active=rebuilt_estop.active,
            reason=rebuilt_estop.reason,
            source=rebuilt_estop.source,
            latched_at_sequence=rebuilt_estop.latched_at_sequence,
            clear_count=rebuilt_estop.clear_count,
            sha256=rebuilt_estop.sha256,
        ),
        outcomes=[
            (;
                sequence=Int(row.sequence),
                action=(; name=String(row.action.name), value=String(row.action.value)),
                idempotency_key=row.idempotency_key === nothing ? nothing : String(row.idempotency_key),
                payload_sha256=String(row.payload_sha256),
                status=String(row.status),
                failure_mode=String(row.failure_mode),
                side_effect_certainty=String(row.side_effect_certainty),
                replayed_from_sequence=row.replayed_from_sequence === nothing ? nothing : Int(row.replayed_from_sequence),
                transition_sha256=row.transition_sha256 === nothing ? nothing : String(row.transition_sha256),
                error=row.error === nothing ? nothing : String(row.error),
                estop_triggered=Bool(row.estop_triggered),
                estop_active_after=Bool(row.estop_active_after),
                sha256=String(row.sha256),
            )
            for row in payload.outcomes
        ],
        safety_events=[
            (;
                event_index=Int(ev.event_index),
                kind=String(ev.kind),
                sequence=Int(ev.sequence),
                source=String(ev.source),
                operator=ev.operator === nothing ? nothing : String(ev.operator),
                reason=String(ev.reason),
                estop_active_after=Bool(ev.estop_active_after),
                sha256=String(ev.sha256),
            )
            for ev in payload.safety_events
        ],
        summary=(;
            total_requests=Int(payload.summary.total_requests),
            executed=Int(payload.summary.executed),
            replayed=Int(payload.summary.replayed),
            timed_out=Int(payload.summary.timed_out),
            failed=Int(payload.summary.failed),
            estopped=Int(payload.summary.estopped),
            rejected=Int(payload.summary.rejected),
            estop_triggers=Int(payload.summary.estop_triggers),
            estop_clears=Int(payload.summary.estop_clears),
        ),
    )
    _sha256_hex(String(JSON3.write(body))) == String(payload.audit_sha256) ||
        throw(ArgumentError("safety audit outer sha256 mismatch"))

    return (;
        adapter_id,
        verified_outcomes,
        verified_events,
        estop_active=rebuilt_estop.active,
        audit_sha256=String(payload.audit_sha256),
    )
end
