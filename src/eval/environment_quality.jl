using JSON3

"""One frozen GridWorld task plus its independently checked shortest path length."""
struct GridWorldTask
    id::String
    spec::GridWorldSpec
    shortest_steps::Int
end

"""Versioned Chapter 40 task fixture and its exact source digest."""
struct GridWorldTaskSet
    tasks::Vector{GridWorldTask}
    source::Any
    coordinate_system::String
    sha256::String
end

function _gridworld_task_integer(entry, field::Symbol, id::AbstractString)
    value = get(entry, field, nothing)
    value isa Integer && !isa(value, Bool) || throw(ArgumentError(
        "GridWorld task $(repr(String(id))) field $(repr(String(field))) must be an integer",
    ))
    return Int(value)
end

function _gridworld_task_coordinate(entry, field::Symbol, id::AbstractString)
    value = get(entry, field, nothing)
    value isa AbstractVector || throw(ArgumentError(
        "GridWorld task $(repr(String(id))) field $(repr(String(field))) must be [x, y]",
    ))
    return _gridworld_coordinate(value, String(field))
end

"""
    load_gridworld_tasks(path)

Load the strict Chapter 40 fixture. Every declared shortest path is recomputed by
BFS, making task solvability an executable oracle rather than fixture metadata.
"""
function load_gridworld_tasks(path::AbstractString)
    raw = read(path, String)
    payload = JSON3.read(raw)
    Set(Symbol(key) for key in keys(payload)) == Set((
        :schema_version,
        :source,
        :coordinate_system,
        :tasks,
    )) || throw(ArgumentError("GridWorld task fixture fields do not match schema version 1"))
    version = get(payload, :schema_version, nothing)
    version isa Integer && !isa(version, Bool) || throw(ArgumentError(
        "GridWorld task schema_version must be an integer",
    ))
    Int(version) == AGENT_ENVIRONMENT_FORMAT_VERSION || throw(ArgumentError(
        "unsupported GridWorld task schema version $(repr(version))",
    ))
    coordinate_system = String(get(payload, :coordinate_system, ""))
    coordinate_system == "origin_top_left_x_east_y_south" || throw(ArgumentError(
        "unsupported GridWorld coordinate system $(repr(coordinate_system))",
    ))

    tasks = GridWorldTask[]
    for entry in get(payload, :tasks, ())
        Set(Symbol(key) for key in keys(entry)) == Set((
            :id,
            :width,
            :height,
            :start,
            :goal,
            :walls,
            :max_actions,
            :shortest_steps,
        )) || throw(ArgumentError("GridWorld task entry fields do not match schema version 1"))
        id_value = get(entry, :id, nothing)
        id_value isa AbstractString || throw(ArgumentError("GridWorld task id must be a string"))
        id = String(id_value)
        walls_value = get(entry, :walls, nothing)
        walls_value isa AbstractVector || throw(ArgumentError(
            "GridWorld task $(repr(id)) walls must be a list",
        ))
        spec = GridWorldSpec(;
            id,
            width=_gridworld_task_integer(entry, :width, id),
            height=_gridworld_task_integer(entry, :height, id),
            start=_gridworld_task_coordinate(entry, :start, id),
            goal=_gridworld_task_coordinate(entry, :goal, id),
            walls=[_gridworld_coordinate(wall, "wall") for wall in walls_value],
            max_actions=_gridworld_task_integer(entry, :max_actions, id),
        )
        shortest_steps = _gridworld_task_integer(entry, :shortest_steps, id)
        shortest_steps > 0 || throw(ArgumentError(
            "GridWorld task $(repr(id)) shortest_steps must be positive",
        ))
        oracle = gridworld_shortest_actions(spec)
        length(oracle) == shortest_steps || throw(ArgumentError(
            "GridWorld task $(repr(id)) declares shortest_steps=$shortest_steps, " *
            "but BFS found $(length(oracle))",
        ))
        shortest_steps <= spec.max_actions || throw(ArgumentError(
            "GridWorld task $(repr(id)) cannot fit its shortest path in max_actions",
        ))
        push!(tasks, GridWorldTask(id, spec, shortest_steps))
    end
    isempty(tasks) && throw(ArgumentError("GridWorld task set is empty: $path"))
    ids = [task.id for task in tasks]
    length(unique(ids)) == length(ids) || throw(ArgumentError(
        "GridWorld task ids are not unique",
    ))
    return GridWorldTaskSet(
        tasks,
        get(payload, :source, nothing),
        coordinate_system,
        _sha256_hex(raw),
    )
end

"""One task-level result derived only from terminal environment state."""
function gridworld_task_result(task::GridWorldTask, trace::AgentEnvironmentTrace)
    trace.task_id == task.id || throw(ArgumentError("task and trace IDs differ"))
    trace.spec_sha256 == task.spec.sha256 || throw(ArgumentError(
        "task and trace GridWorld specs differ",
    ))
    summary = agent_environment_summary(trace)
    return (;
        id=task.id,
        success=trace.success,
        terminal=trace.terminal,
        actions=summary.actions,
        invalid_actions=summary.invalid_actions,
        model_turns=summary.model_turns,
        tool_calls=summary.tool_calls,
        tool_failures=summary.tool_failures,
        shortest_steps=task.shortest_steps,
        excess_actions=trace.success ? summary.actions - task.shortest_steps : nothing,
        stop_reason=summary.stop_reason,
        final_state_sha256=trace.final_state_sha256,
    )
end

function _gridworld_result_field(row, field::Symbol)
    hasproperty(row, field) || throw(ArgumentError(
        "GridWorld result row is missing $(repr(String(field)))",
    ))
    return getproperty(row, field)
end

"""Task success, invalid action, and path-efficiency report over frozen IDs."""
function gridworld_task_report(task_set::GridWorldTaskSet, rows)
    by_id = Dict{String,Any}()
    for row in rows
        id = String(_gridworld_result_field(row, :id))
        haskey(by_id, id) && throw(ArgumentError(
            "duplicate GridWorld result for $(repr(id))",
        ))
        by_id[id] = row
    end
    Set(keys(by_id)) == Set(task.id for task in task_set.tasks) || throw(ArgumentError(
        "GridWorld results do not match the frozen task IDs",
    ))
    successes = 0
    terminal = 0
    invalid_actions = 0
    actions = 0
    excess = Int[]
    for task in task_set.tasks
        row = by_id[task.id]
        success = _gridworld_result_field(row, :success)
        terminal_value = _gridworld_result_field(row, :terminal)
        success isa Bool && terminal_value isa Bool || throw(ArgumentError(
            "GridWorld success/terminal fields must be booleans",
        ))
        success && !terminal_value && throw(ArgumentError(
            "GridWorld success cannot be non-terminal",
        ))
        action_count = _gridworld_result_field(row, :actions)
        invalid_count = _gridworld_result_field(row, :invalid_actions)
        action_count isa Integer && invalid_count isa Integer || throw(ArgumentError(
            "GridWorld action counts must be integers",
        ))
        action_count >= 0 && invalid_count >= 0 || throw(ArgumentError(
            "GridWorld action counts must be non-negative",
        ))
        successes += success
        terminal += terminal_value
        actions += Int(action_count)
        invalid_actions += Int(invalid_count)
        success && push!(excess, Int(action_count) - task.shortest_steps)
    end
    total = length(task_set.tasks)
    interval = wilson_interval(successes, total)
    return (;
        total,
        successes,
        success_rate=interval.point,
        wilson_lower=interval.lower,
        wilson_upper=interval.upper,
        terminal,
        actions,
        invalid_actions,
        mean_actions=actions / total,
        mean_excess_actions=isempty(excess) ? nothing : sum(excess) / length(excess),
    )
end
