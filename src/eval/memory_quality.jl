using JSON3

"""One frozen cross-request fact-recall task for the persistent-memory A/B/C."""
struct AgentMemoryTask
    id::String
    memory_id::String
    memory::String
    distractor_id::String
    distractor::String
    question::String
    expected::String
end

"""Versioned task fixture plus a checksum of its exact source bytes."""
struct AgentMemoryTaskSet
    tasks::Vector{AgentMemoryTask}
    source::Any
    sha256::String
end

const _AGENT_MEMORY_ANSWER_PATTERN = r"(?i)(?<![a-z0-9])([a-z]{5}-[0-9]{3})(?![a-z0-9])"

function _required_memory_task_string(entry, field::Symbol, task_id::AbstractString)
    value = get(entry, field, nothing)
    value isa AbstractString || throw(ArgumentError(
        "memory task $(repr(String(task_id))) field $(repr(String(field))) must be a string",
    ))
    output = String(value)
    isempty(strip(output)) && throw(ArgumentError(
        "memory task $(repr(String(task_id))) field $(repr(String(field))) must not be empty",
    ))
    return output
end

"""
    load_agent_memory_tasks(path)

Load the frozen Chapter 39 task set. Facts and matched distractors must have equal
UTF-8 byte length, while the expected answer must occur only in the relevant fact
and never leak into the question or distractor.
"""
function load_agent_memory_tasks(path::AbstractString)
    raw = read(path, String)
    payload = JSON3.read(raw)
    Set(Symbol(key) for key in keys(payload)) == Set((:schema_version, :source, :tasks)) ||
        throw(ArgumentError("memory task fixture fields do not match schema version 1"))
    version = get(payload, :schema_version, nothing)
    version isa Integer && !isa(version, Bool) || throw(ArgumentError(
        "memory task schema_version must be an integer",
    ))
    Int(version) == 1 || throw(ArgumentError(
        "unsupported memory task schema version $(repr(version))",
    ))
    tasks = AgentMemoryTask[]
    for entry in get(payload, :tasks, ())
        Set(Symbol(key) for key in keys(entry)) == Set((
            :id,
            :memory_id,
            :memory,
            :distractor_id,
            :distractor,
            :question,
            :expected,
        )) || throw(ArgumentError("memory task entry fields do not match schema version 1"))
        id = _required_memory_task_string(entry, :id, "<unknown>")
        _agent_memory_id(id)
        memory_id = _required_memory_task_string(entry, :memory_id, id)
        memory = _required_memory_task_string(entry, :memory, id)
        distractor_id = _required_memory_task_string(entry, :distractor_id, id)
        distractor = _required_memory_task_string(entry, :distractor, id)
        question = _required_memory_task_string(entry, :question, id)
        expected = lowercase(_required_memory_task_string(entry, :expected, id))
        _agent_memory_id(memory_id)
        _agent_memory_id(distractor_id)
        memory_id != distractor_id || throw(ArgumentError(
            "memory task $(repr(id)) reuses one ID for fact and distractor",
        ))
        occursin(r"^[a-z]{5}-[0-9]{3}$", expected) || throw(ArgumentError(
            "memory task $(repr(id)) expected answer must match word5-digits3",
        ))
        occursin(expected, lowercase(memory)) || throw(ArgumentError(
            "memory task $(repr(id)) relevant fact does not contain its answer",
        ))
        !occursin(expected, lowercase(question)) || throw(ArgumentError(
            "memory task $(repr(id)) leaks its answer in the question",
        ))
        !occursin(expected, lowercase(distractor)) || throw(ArgumentError(
            "memory task $(repr(id)) leaks its answer in the distractor",
        ))
        ncodeunits(memory) == ncodeunits(distractor) || throw(ArgumentError(
            "memory task $(repr(id)) fact and distractor byte lengths differ",
        ))
        push!(tasks, AgentMemoryTask(
            id,
            memory_id,
            memory,
            distractor_id,
            distractor,
            question,
            expected,
        ))
    end
    isempty(tasks) && throw(ArgumentError("memory task set is empty: $path"))
    task_ids = [task.id for task in tasks]
    length(unique(task_ids)) == length(task_ids) || throw(ArgumentError(
        "memory task ids are not unique",
    ))
    record_ids = vcat(
        [task.memory_id for task in tasks],
        [task.distractor_id for task in tasks],
    )
    length(unique(record_ids)) == length(record_ids) || throw(ArgumentError(
        "memory fact/distractor ids are not globally unique",
    ))
    return AgentMemoryTaskSet(
        tasks,
        get(payload, :source, nothing),
        _sha256_hex(raw),
    )
end

"""Seed a journal idempotently, rejecting an existing ID with changed bytes."""
function seed_agent_memory_tasks!(
    store::AgentMemoryStore,
    task_set::AgentMemoryTaskSet,
)
    for task in task_set.tasks
        for (id, text, role) in (
            (task.memory_id, task.memory, "relevant"),
            (task.distractor_id, task.distractor, "distractor"),
        )
            if haskey(store.by_id, id)
                existing = store.records[store.by_id[id]]
                existing.text == text || throw(ArgumentError(
                    "existing memory $(repr(id)) does not match the frozen task fixture",
                ))
                existing.metadata == Dict("task" => task.id, "role" => role) ||
                    throw(ArgumentError(
                        "existing memory $(repr(id)) metadata does not match the fixture",
                    ))
            else
                append_agent_memory!(
                    store,
                    text;
                    id,
                    metadata=(; task=task.id, role),
                )
            end
        end
    end
    return store
end

"""
Extract exactly one distinct `word5-digits3` code. Multiple different candidates
are ambiguous and therefore unparsed; repeating the same answer is accepted.
"""
function extract_agent_memory_answer(text::AbstractString)
    candidates = unique(
        lowercase(String(only(match.captures)))
        for match in eachmatch(_AGENT_MEMORY_ANSWER_PATTERN, text)
    )
    return length(candidates) == 1 ? only(candidates) : nothing
end

agent_memory_answer_matches(expected::AbstractString, extracted) =
    extracted !== nothing && lowercase(strip(String(expected))) == lowercase(strip(String(extracted)))

"""Top-1 and recall-at-k report from `(id=..., retrieved_ids=...)` rows."""
function agent_memory_retrieval_report(task_set::AgentMemoryTaskSet, rows)
    by_id = Dict{String,Vector{String}}()
    for row in rows
        id = String(row.id)
        haskey(by_id, id) && throw(ArgumentError(
            "duplicate retrieval row for task $(repr(id))",
        ))
        by_id[id] = String[String(value) for value in row.retrieved_ids]
    end
    expected_ids = Set(task.id for task in task_set.tasks)
    Set(keys(by_id)) == expected_ids || throw(ArgumentError(
        "retrieval rows do not match the frozen memory task IDs",
    ))
    top1 = 0
    recalled = 0
    for task in task_set.tasks
        ids = by_id[task.id]
        !isempty(ids) && first(ids) == task.memory_id && (top1 += 1)
        task.memory_id in ids && (recalled += 1)
    end
    total = length(task_set.tasks)
    return (;
        total,
        top1_correct=top1,
        top1_accuracy=top1 / total,
        recalled,
        recall_at_k=recalled / total,
    )
end
