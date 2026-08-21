using JSON3

"""
Deterministic observation → action → transition contracts for the minimal agent.

The first concrete environment is intentionally small: a hidden-wall GridWorld
whose only side effect is moving an in-memory coordinate. Model-authored actions
are restricted to one enum-valued tool, every attempt is bounded, and every state
and transition has a stable digest so a run can be replayed without a model.
"""

const AGENT_ENVIRONMENT_FORMAT_VERSION = 1
const AGENT_ENVIRONMENT_TRACE_FORMAT_VERSION = 2
const GRIDWORLD_DIRECTIONS = ("north", "east", "south", "west")

abstract type AbstractAgentEnvironment end

function _agent_environment_id(value::AbstractString)
    id = String(value)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$", id) || throw(ArgumentError(
        "environment id must start with an alphanumeric character and contain only " *
        "letters, digits, '.', '_', ':', '/', or '-' (maximum 128 bytes)",
    ))
    return id
end

"""Immutable geometry and action budget for one deterministic GridWorld."""
struct GridWorldSpec
    id::String
    width::Int
    height::Int
    start::NTuple{2,Int}
    goal::NTuple{2,Int}
    walls::Tuple{Vararg{NTuple{2,Int}}}
    max_actions::Int
    sha256::String
end

function _gridworld_coordinate(value, name::AbstractString)
    value isa Union{Tuple,AbstractVector} || throw(ArgumentError(
        "$name must contain exactly [x, y]",
    ))
    length(value) == 2 || throw(ArgumentError("$name must contain exactly [x, y]"))
    all(entry -> entry isa Integer && !isa(entry, Bool), value) || throw(ArgumentError(
        "$name coordinates must be integers",
    ))
    return (Int(value[1]), Int(value[2]))
end

_gridworld_in_bounds(width::Int, height::Int, position::NTuple{2,Int}) =
    1 <= position[1] <= width && 1 <= position[2] <= height

function _gridworld_spec_payload(
    id,
    width,
    height,
    start,
    goal,
    walls,
    max_actions,
)
    return (;
        schema_version=AGENT_ENVIRONMENT_FORMAT_VERSION,
        environment="deterministic_gridworld",
        id,
        width,
        height,
        start=(; x=start[1], y=start[2]),
        goal=(; x=goal[1], y=goal[2]),
        walls=[(; x=wall[1], y=wall[2]) for wall in walls],
        max_actions,
    )
end

function GridWorldSpec(;
    id::AbstractString,
    width::Integer,
    height::Integer,
    start,
    goal,
    walls=NTuple{2,Int}[],
    max_actions::Integer,
)
    id_value = _agent_environment_id(String(id))
    width_value = Int(width)
    height_value = Int(height)
    width_value > 0 && height_value > 0 || throw(ArgumentError(
        "grid dimensions must be positive",
    ))
    start_value = _gridworld_coordinate(start, "start")
    goal_value = _gridworld_coordinate(goal, "goal")
    start_value != goal_value || throw(ArgumentError("grid start and goal must differ"))
    wall_values = sort!(NTuple{2,Int}[
        _gridworld_coordinate(wall, "wall") for wall in walls
    ]; by=position -> (position[2], position[1]))
    length(unique(wall_values)) == length(wall_values) || throw(ArgumentError(
        "grid walls must be unique",
    ))
    for (name, position) in (("start", start_value), ("goal", goal_value))
        _gridworld_in_bounds(width_value, height_value, position) || throw(ArgumentError(
            "$name lies outside the grid",
        ))
    end
    all(_gridworld_in_bounds(width_value, height_value, wall) for wall in wall_values) ||
        throw(ArgumentError("a wall lies outside the grid"))
    start_value in wall_values && throw(ArgumentError("grid start cannot be a wall"))
    goal_value in wall_values && throw(ArgumentError("grid goal cannot be a wall"))
    max_actions_value = Int(max_actions)
    max_actions_value > 0 || throw(ArgumentError("max_actions must be positive"))
    wall_tuple = Tuple(wall_values)
    payload = _gridworld_spec_payload(
        id_value,
        width_value,
        height_value,
        start_value,
        goal_value,
        wall_tuple,
        max_actions_value,
    )
    return GridWorldSpec(
        id_value,
        width_value,
        height_value,
        start_value,
        goal_value,
        wall_tuple,
        max_actions_value,
        _sha256_hex(String(JSON3.write(payload))),
    )
end

gridworld_spec_fingerprint(spec::GridWorldSpec) = spec.sha256

"""One exact observation returned to a policy."""
struct AgentEnvironmentObservation
    task_id::String
    step::Int
    terminal::Bool
    success::Bool
    state_sha256::String
    rendered::String
    rendered_sha256::String
end

"""One allowlisted environment action."""
struct AgentEnvironmentAction
    name::String
    value::String
end

"""One deterministic state transition caused by an action attempt."""
struct AgentEnvironmentTransition
    step::Int
    action::AgentEnvironmentAction
    accepted::Bool
    reason::String
    before_state_sha256::String
    after_state_sha256::String
    observation::AgentEnvironmentObservation
    sha256::String
end

"""Mutable state for one episode of a [`GridWorldSpec`](@ref)."""
mutable struct GridWorldEnvironment <: AbstractAgentEnvironment
    spec::GridWorldSpec
    position::NTuple{2,Int}
    actions::Int
    terminal::Bool
    success::Bool
    transitions::Vector{AgentEnvironmentTransition}
end

GridWorldEnvironment(spec::GridWorldSpec) = GridWorldEnvironment(
    spec,
    spec.start,
    0,
    false,
    false,
    AgentEnvironmentTransition[],
)

function reset_agent_environment!(environment::GridWorldEnvironment)
    environment.position = environment.spec.start
    environment.actions = 0
    environment.terminal = false
    environment.success = false
    empty!(environment.transitions)
    return environment
end

function _gridworld_delta(direction::AbstractString)
    direction == "north" && return (0, -1)
    direction == "east" && return (1, 0)
    direction == "south" && return (0, 1)
    direction == "west" && return (-1, 0)
    throw(ArgumentError(
        "direction must be one of " * join(GRIDWORLD_DIRECTIONS, ", "),
    ))
end

function _gridworld_target(position::NTuple{2,Int}, direction::AbstractString)
    delta = _gridworld_delta(direction)
    return (position[1] + delta[1], position[2] + delta[2])
end

function _gridworld_passable(spec::GridWorldSpec, position::NTuple{2,Int})
    return _gridworld_in_bounds(spec.width, spec.height, position) &&
           !(position in spec.walls)
end

function gridworld_legal_actions(environment::GridWorldEnvironment)
    environment.terminal && return String[]
    return String[
        direction
        for direction in GRIDWORLD_DIRECTIONS
        if _gridworld_passable(
            environment.spec,
            _gridworld_target(environment.position, direction),
        )
    ]
end

function _gridworld_state_payload(environment::GridWorldEnvironment)
    return (;
        schema_version=AGENT_ENVIRONMENT_FORMAT_VERSION,
        environment="deterministic_gridworld",
        task_id=environment.spec.id,
        spec_sha256=environment.spec.sha256,
        step=environment.actions,
        position=(; x=environment.position[1], y=environment.position[2]),
        terminal=environment.terminal,
        success=environment.success,
    )
end

function _gridworld_observation_payload(
    environment::GridWorldEnvironment,
    state_sha256::AbstractString,
)
    return (;
        schema_version=AGENT_ENVIRONMENT_FORMAT_VERSION,
        environment="deterministic_gridworld",
        task_id=environment.spec.id,
        step=environment.actions,
        position=(; x=environment.position[1], y=environment.position[2]),
        goal=(; x=environment.spec.goal[1], y=environment.spec.goal[2]),
        legal_actions=gridworld_legal_actions(environment),
        remaining_actions=max(0, environment.spec.max_actions - environment.actions),
        terminal=environment.terminal,
        success=environment.success,
        state_sha256=String(state_sha256),
    )
end

function observe_agent_environment(environment::GridWorldEnvironment)
    state_json = String(JSON3.write(_gridworld_state_payload(environment)))
    state_sha256 = _sha256_hex(state_json)
    rendered = String(JSON3.write(
        _gridworld_observation_payload(environment, state_sha256),
    ))
    return AgentEnvironmentObservation(
        environment.spec.id,
        environment.actions,
        environment.terminal,
        environment.success,
        state_sha256,
        rendered,
        _sha256_hex(rendered),
    )
end

function _gridworld_transition_payload(
    step,
    action,
    accepted,
    reason,
    before_state_sha256,
    after_state_sha256,
    observation,
)
    return (;
        schema_version=AGENT_ENVIRONMENT_FORMAT_VERSION,
        environment="deterministic_gridworld",
        step,
        action=(; name=action.name, value=action.value),
        accepted,
        reason,
        before_state_sha256,
        after_state_sha256,
        observation_sha256=observation.rendered_sha256,
    )
end

function step_agent_environment!(
    environment::GridWorldEnvironment,
    action::AgentEnvironmentAction,
)
    environment.terminal && throw(ArgumentError("environment is already terminal"))
    action.name == "move" || throw(ArgumentError("unknown environment action $(repr(action.name))"))
    direction = String(action.value)
    _gridworld_delta(direction) # validate before mutating the state

    before = observe_agent_environment(environment)
    environment.actions += 1
    target = _gridworld_target(environment.position, direction)
    accepted = _gridworld_passable(environment.spec, target)
    accepted && (environment.position = target)
    environment.success = environment.position == environment.spec.goal
    environment.terminal = environment.success ||
        environment.actions >= environment.spec.max_actions
    reason = environment.success ? "goal_reached" : accepted ? "moved" : "blocked"
    observation = observe_agent_environment(environment)
    payload = _gridworld_transition_payload(
        environment.actions,
        action,
        accepted,
        reason,
        before.state_sha256,
        observation.state_sha256,
        observation,
    )
    transition = AgentEnvironmentTransition(
        environment.actions,
        action,
        accepted,
        reason,
        before.state_sha256,
        observation.state_sha256,
        observation,
        _sha256_hex(String(JSON3.write(payload))),
    )
    push!(environment.transitions, transition)
    return transition
end

function _gridworld_observation_json(observation::AgentEnvironmentObservation)
    return JSON3.read(observation.rendered)
end

"""
    render_agent_environment_feedback(transition; feedback=:full)

Render the exact tool result. `:none` is the Chapter 40 diagnostic control: the
state still changes, but position, legal actions, acceptance, and failure reason
are withheld. Terminal success remains visible so a policy can stop cleanly.
"""
function render_agent_environment_feedback(
    transition::AgentEnvironmentTransition;
    feedback::Symbol=:full,
)
    feedback in (:full, :none) || throw(ArgumentError(
        "feedback must be :full or :none",
    ))
    feedback === :full && return transition.observation.rendered
    payload = (;
        schema_version=AGENT_ENVIRONMENT_FORMAT_VERSION,
        feedback="withheld",
        terminal=transition.observation.terminal,
        success=transition.observation.success,
    )
    return String(JSON3.write(payload))
end

function gridworld_move_tool(
    environment::GridWorldEnvironment;
    feedback::Symbol=:full,
)
    feedback in (:full, :none) || throw(ArgumentError(
        "feedback must be :full or :none",
    ))
    return AgentTool(;
        name="move",
        description="Move one cell in the deterministic GridWorld. Choose one direction from the current observation's legal_actions and inspect the returned observation before moving again.",
        properties=(;
            direction=(;
                type="string",
                enum=collect(GRIDWORLD_DIRECTIONS),
                description="One of north, east, south, or west.",
            ),
        ),
        required=["direction"],
        handler=function (arguments, _coerced)
            Set(String(key) for key in keys(arguments)) == Set(("direction",)) ||
                throw(ArgumentError("move accepts exactly one `direction` argument"))
            direction = _tool_string(arguments, "direction")
            transition = step_agent_environment!(
                environment,
                AgentEnvironmentAction("move", direction),
            )
            return render_agent_environment_feedback(transition; feedback)
        end,
    )
end

gridworld_tool_registry(environment::GridWorldEnvironment; feedback::Symbol=:full) =
    ToolRegistry([gridworld_move_tool(environment; feedback)])

const GRIDWORLD_SYSTEM_PROMPT =
    "You control a deterministic hidden-wall GridWorld. Coordinates use a top-left origin: " *
    "x increases east and y increases south. Reach the goal using only the move tool. " *
    "Choose a direction listed in the latest observation's legal_actions, issue exactly one " *
    "move call, then inspect the returned observation before deciding again. Never invent a " *
    "position or claim success before an observation has terminal=true and success=true."

gridworld_system_prompt() = GRIDWORLD_SYSTEM_PROMPT

function gridworld_user_prompt(observation::AgentEnvironmentObservation)
    return "Reach the goal in the deterministic GridWorld. The walls are hidden; use only " *
        "the observations returned by the environment.\nInitial observation:\n" *
        observation.rendered
end

"""A Qwen3 run tied to its exact initial observation and environment transitions."""
struct AgentEnvironmentTrace
    environment::String
    task_id::String
    spec_sha256::String
    feedback::Symbol
    system_prompt::String
    initial_observation::AgentEnvironmentObservation
    transitions::Vector{AgentEnvironmentTransition}
    agent::AgentLoopTrace
    terminal::Bool
    success::Bool
    final_state_sha256::String
end

# Preserve the Chapter 40 constructor for callers that materialize a trace with
# the original system prompt and no explicit protocol override.
AgentEnvironmentTrace(
    environment,
    task_id,
    spec_sha256,
    feedback,
    initial_observation,
    transitions,
    agent,
    terminal,
    success,
    final_state_sha256,
) = AgentEnvironmentTrace(
    environment,
    task_id,
    spec_sha256,
    feedback,
    GRIDWORLD_SYSTEM_PROMPT,
    initial_observation,
    transitions,
    agent,
    terminal,
    success,
    final_state_sha256,
)

function run_qwen3_environment_loop(
    session::HFQwen3BF16Session,
    environment::GridWorldEnvironment;
    feedback::Symbol=:full,
    system::AbstractString=GRIDWORLD_SYSTEM_PROMPT,
    memory_context::Union{Nothing,AgentMemoryContext}=nothing,
    allow_cross_spec_memory::Bool=false,
    max_steps=nothing,
    max_new_tokens::Integer=128,
    enable_thinking::Bool=false,
    strategy::Symbol=:greedy,
    stop_token_ids=nothing,
)
    turns = max_steps === nothing ? environment.spec.max_actions + 1 : Int(max_steps)
    turns > 0 || throw(ArgumentError("max_steps must be positive"))
    feedback in (:full, :none) || throw(ArgumentError(
        "feedback must be :full or :none",
    ))
    _validate_environment_memory_spec(environment.spec)
    if memory_context !== nothing
        validate_agent_memory_context(memory_context)
        if allow_cross_spec_memory
            for hit in memory_context.hits
                validate_agent_environment_memory_record(
                    AgentMemoryRecord(
                        hit.id,
                        hit.sequence,
                        hit.text,
                        copy(hit.metadata),
                        _sha256_hex(hit.text),
                    ),
                )
            end
        else
            _validated_gridworld_memory_context(memory_context, environment.spec)
        end
    end
    reset_agent_environment!(environment)
    initial = observe_agent_environment(environment)
    registry = gridworld_tool_registry(environment; feedback)
    agent = run_qwen3_tool_loop(
        session,
        registry,
        gridworld_user_prompt(initial);
        system,
        memory_context,
        max_steps=turns,
        max_new_tokens,
        enable_thinking,
        strategy,
        stop_token_ids,
    )
    final = observe_agent_environment(environment)
    return AgentEnvironmentTrace(
        "deterministic_gridworld",
        environment.spec.id,
        environment.spec.sha256,
        feedback,
        String(system),
        initial,
        copy(environment.transitions),
        agent,
        environment.terminal,
        environment.success,
        final.state_sha256,
    )
end

function agent_environment_summary(trace::AgentEnvironmentTrace)
    tool_calls = sum(length(step.tool_calls) for step in trace.agent.steps; init=0)
    tool_failures = sum(
        count(call -> !call.ok, step.tool_calls) for step in trace.agent.steps;
        init=0,
    )
    rejected = count(transition -> !transition.accepted, trace.transitions)
    return (;
        task_id=trace.task_id,
        feedback=String(trace.feedback),
        terminal=trace.terminal,
        success=trace.success,
        model_turns=length(trace.agent.steps),
        actions=length(trace.transitions),
        accepted_actions=length(trace.transitions) - rejected,
        rejected_actions=rejected,
        tool_calls,
        tool_failures,
        invalid_actions=rejected + tool_failures,
        stop_reason=String(trace.agent.stop_reason),
        final_state_sha256=trace.final_state_sha256,
    )
end

function _agent_environment_transition_payload(transition::AgentEnvironmentTransition)
    return (;
        step=transition.step,
        action=(; name=transition.action.name, value=transition.action.value),
        accepted=transition.accepted,
        reason=transition.reason,
        before_state_sha256=transition.before_state_sha256,
        after_state_sha256=transition.after_state_sha256,
        observation=_gridworld_observation_json(transition.observation),
        observation_sha256=transition.observation.rendered_sha256,
        transition_sha256=transition.sha256,
    )
end

"""Stable JSON-ready representation written by the Chapter 40 runner."""
function agent_environment_trace_payload(trace::AgentEnvironmentTrace)
    memory = trace.agent.memory_context
    return (;
        schema_version=AGENT_ENVIRONMENT_TRACE_FORMAT_VERSION,
        environment=trace.environment,
        task=trace.task_id,
        spec_sha256=trace.spec_sha256,
        feedback=String(trace.feedback),
        system_prompt=trace.system_prompt,
        system_prompt_sha256=_sha256_hex(trace.system_prompt),
        memory=memory === nothing ? nothing : (;
            query=memory.query,
            query_sha256=memory.query_sha256,
            store_sha256=memory.store_sha256,
            context_sha256=memory.rendered_sha256,
            ids=[hit.id for hit in memory.hits],
            scores=[hit.score for hit in memory.hits],
        ),
        initial_observation=_gridworld_observation_json(trace.initial_observation),
        initial_observation_sha256=trace.initial_observation.rendered_sha256,
        transitions=[
            _agent_environment_transition_payload(transition)
            for transition in trace.transitions
        ],
        agent_steps=[(;
            turn=step.turn,
            prompt_sha256=step.prompt_sha256,
            prompt_token_count=step.prompt_token_count,
            completion=step.completion,
            generated_ids=step.generated_ids,
            stop_reason=String(step.stop_reason),
            validity=String(step.validity),
            tool_calls=[(;
                name=call.name,
                arguments=call.arguments_json,
                ok=call.ok,
                output=call.output,
                error=call.error,
                coerced_arguments=call.coerced_arguments,
            ) for call in step.tool_calls],
        ) for step in trace.agent.steps],
        answer=trace.agent.answer,
        agent_stop_reason=String(trace.agent.stop_reason),
        terminal=trace.terminal,
        success=trace.success,
        final_state_sha256=trace.final_state_sha256,
        summary=agent_environment_summary(trace),
    )
end

"""Shortest deterministic action sequence, used as the environment oracle."""
function gridworld_shortest_actions(spec::GridWorldSpec)
    queue = NTuple{2,Int}[spec.start]
    head = 1
    parents = Dict{NTuple{2,Int},Tuple{NTuple{2,Int},String}}()
    visited = Set{NTuple{2,Int}}([spec.start])
    while head <= length(queue)
        position = queue[head]
        head += 1
        for direction in GRIDWORLD_DIRECTIONS
            target = _gridworld_target(position, direction)
            _gridworld_passable(spec, target) || continue
            target in visited && continue
            push!(visited, target)
            parents[target] = (position, direction)
            if target == spec.goal
                actions = String[]
                cursor = target
                while cursor != spec.start
                    parent, action = parents[cursor]
                    push!(actions, action)
                    cursor = parent
                end
                reverse!(actions)
                return actions
            end
            push!(queue, target)
        end
    end
    throw(ArgumentError("grid task $(repr(spec.id)) has no path from start to goal"))
end

function replay_gridworld_actions(
    spec::GridWorldSpec,
    actions;
    feedback::Symbol=:full,
)
    environment = GridWorldEnvironment(spec)
    outputs = String[]
    for direction in actions
        transition = step_agent_environment!(
            environment,
            AgentEnvironmentAction("move", String(direction)),
        )
        push!(outputs, render_agent_environment_feedback(transition; feedback))
        environment.terminal && break
    end
    return (; environment, outputs)
end

function _recorded_environment_error(value)
    return value === nothing ? nothing : String(value)
end

"""
    replay_qwen3_environment_trace(tokenizer, spec, row)

Rebuild a recorded run without loading an embedding or generation model. Recorded
completions drive a fresh tool registry, so prompts, tool outcomes, state changes,
and transition digests are independently recomputed rather than trusted.
"""
function replay_qwen3_environment_trace(
    tokenizer,
    spec::GridWorldSpec,
    row;
    memory_context::Union{Nothing,AgentMemoryContext}=nothing,
    memory_store::Union{Nothing,AgentMemoryStore}=nothing,
    allow_cross_spec_memory::Bool=false,
)
    _validate_environment_memory_spec(spec)
    version = get(row, :schema_version, nothing)
    version isa Integer && !isa(version, Bool) || throw(ArgumentError(
        "environment trace schema_version must be an integer",
    ))
    expected_v1 = Set((
        :schema_version,
        :environment,
        :task,
        :spec_sha256,
        :feedback,
        :initial_observation,
        :initial_observation_sha256,
        :transitions,
        :agent_steps,
        :answer,
        :agent_stop_reason,
        :terminal,
        :success,
        :final_state_sha256,
        :summary,
    ))
    expected_v2 = union(expected_v1, Set((
        :system_prompt,
        :system_prompt_sha256,
        :memory,
    )))
    actual_fields = Set(Symbol(key) for key in keys(row))
    if Int(version) == 1
        actual_fields == expected_v1 || throw(ArgumentError(
            "environment trace fields do not match schema version 1",
        ))
    elseif Int(version) == AGENT_ENVIRONMENT_TRACE_FORMAT_VERSION
        actual_fields == expected_v2 || throw(ArgumentError(
            "environment trace fields do not match schema version 2",
        ))
    else
        throw(ArgumentError("unsupported environment trace schema version $(repr(version))"))
    end
    String(row.environment) == "deterministic_gridworld" || throw(ArgumentError(
        "environment trace has an unsupported adapter",
    ))
    String(row.task) == spec.id || throw(ArgumentError("trace task/spec mismatch"))
    String(row.spec_sha256) == spec.sha256 || throw(ArgumentError(
        "trace GridWorld spec digest mismatch for $(spec.id)",
    ))
    feedback = Symbol(String(row.feedback))
    feedback in (:full, :none) || throw(ArgumentError("trace has unknown feedback arm"))
    environment = GridWorldEnvironment(spec)
    system = Int(version) == AGENT_ENVIRONMENT_TRACE_FORMAT_VERSION ?
        String(row.system_prompt) : GRIDWORLD_SYSTEM_PROMPT
    if Int(version) == AGENT_ENVIRONMENT_TRACE_FORMAT_VERSION
        _sha256_hex(system) == String(row.system_prompt_sha256) || error(
            "system prompt digest mismatch for $(spec.id)",
        )
    end
    recorded_memory = Int(version) == AGENT_ENVIRONMENT_TRACE_FORMAT_VERSION ?
        row.memory : nothing
    if recorded_memory === nothing
        memory_context === nothing || error(
            "trace omits memory evidence for $(spec.id)",
        )
    else
        query = String(recorded_memory.query)
        _sha256_hex(query) == String(recorded_memory.query_sha256) || error(
            "recorded memory query digest mismatch for $(spec.id)",
        )
        ids = String[String(id) for id in recorded_memory.ids]
        isempty(ids) && error("recorded memory IDs are empty for $(spec.id)")
        length(unique(ids)) == length(ids) || error(
            "recorded memory IDs repeat for $(spec.id)",
        )
        scores = Float32[]
        for score in recorded_memory.scores
            score isa Real || error("recorded memory score is not numeric for $(spec.id)")
            value = Float32(score)
            isfinite(value) || error("recorded memory score is not finite for $(spec.id)")
            push!(scores, value)
        end
        length(scores) == length(ids) || error(
            "recorded memory IDs/scores differ in length for $(spec.id)",
        )
        if memory_store !== nothing
            agent_memory_fingerprint(memory_store) == String(recorded_memory.store_sha256) ||
                error("loaded memory store digest mismatch for $(spec.id)")
            selected = allow_cross_spec_memory ?
                select_agent_memory_context(memory_store, query, ids) :
                select_gridworld_memory_context(memory_store, spec, query, ids)
            if allow_cross_spec_memory
                for hit in selected.hits
                    validate_agent_environment_memory_record(
                        AgentMemoryRecord(
                            hit.id,
                            hit.sequence,
                            hit.text,
                            copy(hit.metadata),
                            _sha256_hex(hit.text),
                        ),
                    )
                end
            end
            scored_hits = AgentMemoryHit[
                AgentMemoryHit(
                    hit.rank,
                    hit.id,
                    hit.sequence,
                    hit.text,
                    scores[hit.rank],
                    copy(hit.metadata),
                )
                for hit in selected.hits
            ]
            memory_context = agent_memory_context(
                query,
                scored_hits;
                store_sha256=selected.store_sha256,
            )
        else
            memory_context === nothing && error(
                "trace requires a memory store or context for $(spec.id)",
            )
            memory_context.query_sha256 == _sha256_hex(memory_context.query) || error(
                "memory context has an invalid query digest for $(spec.id)",
            )
            memory_context.rendered == render_agent_memory_context(memory_context.hits) ||
                error("memory context bytes are not canonical for $(spec.id)")
            memory_context.rendered_sha256 == _sha256_hex(memory_context.rendered) ||
                error("memory context rendered digest is invalid for $(spec.id)")
            if allow_cross_spec_memory
                for hit in memory_context.hits
                    validate_agent_environment_memory_record(
                        AgentMemoryRecord(
                            hit.id,
                            hit.sequence,
                            hit.text,
                            copy(hit.metadata),
                            _sha256_hex(hit.text),
                        ),
                    )
                end
            else
                _validated_gridworld_memory_context(memory_context, spec)
            end
        end
        memory_context.query == query || error(
            "memory query mismatch for $(spec.id)",
        )
        memory_context.query_sha256 == String(recorded_memory.query_sha256) || error(
            "memory query digest mismatch for $(spec.id)",
        )
        memory_context.store_sha256 == String(recorded_memory.store_sha256) || error(
            "memory store digest mismatch for $(spec.id)",
        )
        memory_context.rendered_sha256 == String(recorded_memory.context_sha256) || error(
            "memory context digest mismatch for $(spec.id)",
        )
        [hit.id for hit in memory_context.hits] == ids || error(
                "memory IDs mismatch for $(spec.id)",
            )
        [hit.score for hit in memory_context.hits] == scores || error(
                "memory scores mismatch for $(spec.id)",
            )
    end
    initial = observe_agent_environment(environment)
    initial.rendered_sha256 == String(row.initial_observation_sha256) || error(
        "initial observation digest mismatch for $(spec.id)",
    )
    initial.rendered == String(JSON3.write(row.initial_observation)) || error(
        "initial observation bytes mismatch for $(spec.id)",
    )
    memory_context === nothing || memory_context.query == gridworld_memory_query(initial) ||
        error("memory query does not match the initial observation for $(spec.id)")
    registry = gridworld_tool_registry(environment; feedback)
    tools = qwen3_tool_specs(registry)
    messages = _agent_initial_messages(
        gridworld_user_prompt(initial);
        system,
        memory_context,
    )

    prompt_hashes_equal = 0
    prompt_tokens_equal = 0
    tool_outcomes_equal = 0
    generated_ids_present = 0
    for (position, recorded_step) in enumerate(row.agent_steps)
        Int(recorded_step.turn) == position || error(
            "non-contiguous model turn for $(spec.id)",
        )
        prompt = apply_qwen3_chat_template(
            tokenizer,
            messages;
            tools,
            add_generation_prompt=true,
            enable_thinking=false,
        )
        _sha256_hex(prompt) == String(recorded_step.prompt_sha256) || error(
            "prompt digest mismatch for $(spec.id), turn $position",
        )
        prompt_hashes_equal += 1
        prompt_ids = encode(tokenizer, prompt; add_special_tokens=false)
        length(prompt_ids) == Int(recorded_step.prompt_token_count) || error(
            "prompt token count mismatch for $(spec.id), turn $position",
        )
        prompt_tokens_equal += 1
        !isempty(recorded_step.generated_ids) || error(
            "recorded generation is empty for $(spec.id), turn $position",
        )
        generated_ids_present += 1

        completion = String(recorded_step.completion)
        parsed = parse_qwen3_tool_calls(completion)
        String(agent_tool_call_validity(registry, parsed)) == String(recorded_step.validity) ||
            error("tool-call validity mismatch for $(spec.id), turn $position")
        length(parsed.calls) == length(recorded_step.tool_calls) || error(
            "tool-call count mismatch for $(spec.id), turn $position",
        )
        push!(messages, _agent_assistant_message(
            _qwen3_visible_assistant_content(completion),
            parsed.calls,
        ))
        for (call, recorded_call) in zip(parsed.calls, recorded_step.tool_calls)
            outcome = invoke_agent_tool(registry, call)
            arguments_json = try
                _python_json_text(call.arguments)
            catch replay_error
                "<unrenderable: " * sprint(showerror, replay_error) * ">"
            end
            call.name == String(recorded_call.name) || error("tool name mismatch")
            arguments_json == String(recorded_call.arguments) || error(
                "tool argument mismatch for $(spec.id), turn $position",
            )
            outcome.ok == Bool(recorded_call.ok) || error("tool status mismatch")
            outcome.output == String(recorded_call.output) || error(
                "tool output mismatch for $(spec.id), turn $position",
            )
            outcome.error == _recorded_environment_error(recorded_call.error) || error(
                "tool error mismatch for $(spec.id), turn $position",
            )
            outcome.coerced_arguments == String[
                String(value) for value in recorded_call.coerced_arguments
            ] || error("tool coercion mismatch")
            push!(messages, _agent_message(
                "tool",
                outcome.ok ? outcome.output :
                    "error: " * something(outcome.error, "tool failed"),
            ))
            tool_outcomes_equal += 1
        end
    end

    length(environment.transitions) == length(row.transitions) || error(
        "environment transition count mismatch for $(spec.id)",
    )
    transition_hashes_equal = 0
    for (transition, recorded) in zip(environment.transitions, row.transitions)
        transition.step == Int(recorded.step) || error("transition step mismatch")
        transition.action.name == String(recorded.action.name) || error(
            "transition action name mismatch",
        )
        transition.action.value == String(recorded.action.value) || error(
            "transition action value mismatch",
        )
        transition.accepted == Bool(recorded.accepted) || error(
            "transition acceptance mismatch",
        )
        transition.reason == String(recorded.reason) || error("transition reason mismatch")
        transition.before_state_sha256 == String(recorded.before_state_sha256) || error(
            "transition before-state mismatch",
        )
        transition.after_state_sha256 == String(recorded.after_state_sha256) || error(
            "transition after-state mismatch",
        )
        transition.observation.rendered == String(JSON3.write(recorded.observation)) || error(
            "transition observation bytes mismatch",
        )
        transition.sha256 == String(recorded.transition_sha256) || error(
            "transition digest mismatch for $(spec.id), step $(transition.step)",
        )
        transition.observation.rendered_sha256 == String(recorded.observation_sha256) ||
            error("observation digest mismatch for $(spec.id), step $(transition.step)")
        transition_hashes_equal += 1
    end
    final = observe_agent_environment(environment)
    final.state_sha256 == String(row.final_state_sha256) || error(
        "final state digest mismatch for $(spec.id)",
    )
    environment.success == Bool(row.success) || error(
        "replayed success mismatch for $(spec.id)",
    )
    environment.terminal == Bool(row.terminal) || error(
        "replayed terminal mismatch for $(spec.id)",
    )
    return (;
        task=spec.id,
        feedback=String(feedback),
        model_steps=length(row.agent_steps),
        prompt_hashes_equal,
        prompt_tokens_equal,
        generated_ids_present,
        tool_outcomes_equal,
        transition_hashes_equal,
        final_state_equal=true,
        success=environment.success,
    )
end
