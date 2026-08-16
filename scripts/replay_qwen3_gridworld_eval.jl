#!/usr/bin/env julia

# Chapter 40 — rebuild every prompt, tool result, and environment transition from
# a trace without loading generation weights.

using JSON3
using LifeAI

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/replay_qwen3_gridworld_eval.jl TASKS TRACE TOKENIZER_DIR [--out PATH]
""")
end

function main(args)
    !isempty(args) && args[1] in ("-h", "--help") && (usage(); return 0)
    length(args) >= 3 || (usage(); return 2)
    out_path = nothing
    if length(args) == 5 && args[4] == "--out"
        out_path = args[5]
    elseif length(args) != 3
        usage()
        return 2
    end

    task_set = load_gridworld_tasks(args[1])
    rows = [JSON3.read(line) for line in eachline(args[2]) if !isempty(strip(line))]
    tokenizer = load_hf_qwen3_tokenizer(args[3])
    !isempty(rows) && iseven(length(rows)) || error(
        "trace must contain a non-empty pair of arms for every evaluated task",
    )
    tasks = Dict(task.id => task for task in task_set.tasks)
    seen = Set{Tuple{String,String}}()
    reports = Any[]
    for row in rows
        id = String(row.task)
        arm = String(row.feedback)
        key = (id, arm)
        key in seen && error("duplicate trace row for $(repr(key))")
        push!(seen, key)
        task = get(tasks, id, nothing)
        task === nothing && error("trace has unknown task $(repr(id))")
        push!(reports, replay_qwen3_environment_trace(tokenizer, task.spec, row))
    end
    evaluated_ids = Set(first(key) for key in seen)
    expected = Set((id, arm) for id in evaluated_ids for arm in ("full", "none"))
    seen == expected || error("trace arms do not match the frozen task set")

    by_task = Dict{String,Vector{Any}}()
    for row in rows
        push!(get!(by_task, String(row.task), Any[]), row)
    end
    first_prompt_pairs_equal = 0
    for id in sort!(collect(evaluated_ids))
        pair = by_task[id]
        length(pair) == 2 || error("task $id does not have two arms")
        all(!isempty(row.agent_steps) for row in pair) || error(
            "task $id has an empty model trace",
        )
        first(pair).agent_steps[1].prompt_sha256 ==
            last(pair).agent_steps[1].prompt_sha256 || error(
                "first prompt digest differs between arms for $id",
            )
        first(pair).agent_steps[1].prompt_token_count ==
            last(pair).agent_steps[1].prompt_token_count || error(
                "first prompt token count differs between arms for $id",
            )
        first_prompt_pairs_equal += 1
    end

    report = (;
        schema_version=AGENT_ENVIRONMENT_FORMAT_VERSION,
        tasks_sha256=task_set.sha256,
        evaluated_tasks=length(evaluated_ids),
        rows=length(rows),
        first_prompt_pairs_equal,
        model_steps=sum(row.model_steps for row in reports),
        prompt_hashes_equal=sum(row.prompt_hashes_equal for row in reports),
        prompt_tokens_equal=sum(row.prompt_tokens_equal for row in reports),
        generated_ids_present=sum(row.generated_ids_present for row in reports),
        tool_outcomes_equal=sum(row.tool_outcomes_equal for row in reports),
        transition_hashes_equal=sum(row.transition_hashes_equal for row in reports),
        final_states_equal=count(row -> row.final_state_equal, reports),
        replayed=true,
    )
    if out_path !== nothing
        open(out_path, "w") do io
            JSON3.pretty(io, JSON3.write(report))
            println(io)
        end
    end
    println(JSON3.write(report))
    return 0
end

exit(main(ARGS))
