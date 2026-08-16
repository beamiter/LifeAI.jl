#!/usr/bin/env julia

# Chapter 40 — real Qwen3 environment/action evaluation.
#
# Every frozen task is run twice from an identical initial prompt:
#   1. full environment feedback after each action;
#   2. feedback withheld after each action (terminal success remains visible).
# A BFS oracle independently proves that every task is solvable within its budget.

using LuxCUDA
using CUDA
using JSON3
using LifeAI

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_TASKS = joinpath(
    REPO_ROOT,
    "test",
    "episodes",
    "episode08_environment_action_loop",
    "chapter40_deterministic_gridworld",
    "fixtures",
    "gridworld_tasks.json",
)

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/run_qwen3_gridworld_eval.jl MODEL_DIR --out DIR [options]

options:
  --tasks PATH          frozen Chapter 40 task fixture
  --out DIR             output directory (required)
  --label NAME          output filename prefix (default: qwen3-4b)
  --limit N             evaluate only the first N tasks
  --max-new-tokens N    output allowance per model turn (default: 96)
  --context N           generation session context (default: 4096)
  --variant NAME        generation model variant (default: qwen3_4b)
  --revision SHA        expected generation checkpoint revision
  --cpu                 run generation on CPU
""")
end

function parse_args(args)
    !isempty(args) && args[1] in ("-h", "--help") && (usage(); exit())
    isempty(args) && (usage(); exit(2))
    options = Dict{Symbol,Any}(
        :tasks => DEFAULT_TASKS,
        :out => nothing,
        :label => "qwen3-4b",
        :limit => 0,
        :max_new_tokens => 96,
        :context => 4096,
        :variant => "qwen3_4b",
        :revision => "",
        :cpu => false,
    )
    integers = (:limit, :max_new_tokens, :context)
    index = 2
    while index <= length(args)
        option = args[index]
        if option == "--cpu"
            options[:cpu] = true
            index += 1
        elseif startswith(option, "--")
            index < length(args) || (usage(); exit(2))
            key = Symbol(replace(option[3:end], "-" => "_"))
            haskey(options, key) || (usage(); exit(2))
            value = args[index + 1]
            if key in integers
                parsed = tryparse(Int, value)
                parsed === nothing && (usage(); exit(2))
                options[key] = parsed
            else
                options[key] = value
            end
            index += 2
        else
            usage()
            exit(2)
        end
    end
    options[:out] === nothing && (usage(); exit(2))
    return args[1], options
end

function write_jsonl(path, rows)
    open(path, "w") do io
        for row in rows
            JSON3.write(io, row)
            println(io)
        end
    end
end

function paired_eval_result(task, trace, protocol)
    first_step = isempty(trace.agent.steps) ? nothing : first(trace.agent.steps)
    summary = agent_environment_summary(trace)
    return EvalItemResult(
        task.id,
        "",
        protocol,
        first_step === nothing ? "" : first_step.prompt_sha256,
        first_step === nothing ? 0 : first_step.prompt_token_count,
        trace.agent.answer,
        trace.success ? "goal" : "",
        "goal",
        trace.success,
        true,
        summary,
    )
end

function selected_task_set(task_set, tasks)
    return GridWorldTaskSet(
        GridWorldTask[task for task in tasks],
        task_set.source,
        task_set.coordinate_system,
        task_set.sha256,
    )
end

model_dir, options = parse_args(ARGS)
output_dir = abspath(String(options[:out]))
mkpath(output_dir)
label = String(options[:label])
task_set = load_gridworld_tasks(String(options[:tasks]))
tasks = options[:limit] > 0 ?
    task_set.tasks[1:min(Int(options[:limit]), length(task_set.tasks))] : task_set.tasks
evaluated_task_set = selected_task_set(task_set, tasks)

oracle_rows = Any[]
for task in tasks
    actions = gridworld_shortest_actions(task.spec)
    replay = replay_gridworld_actions(task.spec, actions)
    replay.environment.success || error("BFS oracle failed task $(task.id)")
    push!(oracle_rows, (;
        id=task.id,
        success=true,
        terminal=true,
        actions=length(actions),
        invalid_actions=0,
    ))
end
oracle_report = gridworld_task_report(evaluated_task_set, oracle_rows)

use_cuda = !Bool(options[:cpu])
use_cuda && !CUDA.functional() && error("CUDA.jl is not functional; pass --cpu")
device = use_cuda ? String(CUDA.name(CUDA.device())) : "cpu"
println(stderr, "loading $(abspath(model_dir)) on $device")
load_started = time_ns()
session = load_hf_qwen3_bf16_session(
    model_dir;
    context_tokens=Int(options[:context]),
    revision=String(options[:revision]),
    variant=Symbol(options[:variant]),
    to_device=use_cuda ? CUDA.cu : identity,
)
use_cuda && (CUDA.synchronize(); GC.gc(); CUDA.reclaim())
load_seconds = (time_ns() - load_started) / 1.0e9

traces = Dict(:full => AgentEnvironmentTrace[], :none => AgentEnvironmentTrace[])
task_rows = Dict(:full => Any[], :none => Any[])
trace_payloads = Any[]
started = time_ns()
for (position, task) in enumerate(tasks)
    for feedback in (:full, :none)
        environment = GridWorldEnvironment(task.spec)
        trace = run_qwen3_environment_loop(
            session,
            environment;
            feedback,
            max_steps=task.spec.max_actions + 1,
            max_new_tokens=Int(options[:max_new_tokens]),
            enable_thinking=false,
            strategy=:greedy,
        )
        push!(traces[feedback], trace)
        push!(task_rows[feedback], gridworld_task_result(task, trace))
        push!(trace_payloads, agent_environment_trace_payload(trace))
        println(stderr, "$(position)/$(length(tasks)) $(task.id) $feedback " *
            "success=$(trace.success) actions=$(length(trace.transitions)) " *
            "turns=$(length(trace.agent.steps))")
    end
end
seconds = (time_ns() - started) / 1.0e9

trace_path = joinpath(output_dir, "$(label)_gridworld_trace.jsonl")
write_jsonl(trace_path, trace_payloads)
write_jsonl(
    joinpath(output_dir, "$(label)_gridworld_full_feedback_items.jsonl"),
    task_rows[:full],
)
write_jsonl(
    joinpath(output_dir, "$(label)_gridworld_no_feedback_items.jsonl"),
    task_rows[:none],
)

full_report = gridworld_task_report(evaluated_task_set, task_rows[:full])
none_report = gridworld_task_report(evaluated_task_set, task_rows[:none])
full_eval = [
    paired_eval_result(task, trace, :gridworld_full_feedback)
    for (task, trace) in zip(tasks, traces[:full])
]
none_eval = [
    paired_eval_result(task, trace, :gridworld_no_feedback)
    for (task, trace) in zip(tasks, traces[:none])
]
paired = paired_comparison(none_eval, full_eval)
first_prompt_pairs_equal = count(
    pair -> !isempty(first(pair).agent.steps) && !isempty(last(pair).agent.steps) &&
        first(first(pair).agent.steps).prompt_sha256 ==
        first(last(pair).agent.steps).prompt_sha256 &&
        first(first(pair).agent.steps).prompt_token_count ==
        first(last(pair).agent.steps).prompt_token_count,
    zip(traces[:full], traces[:none]),
)
first_prompt_pairs_equal == length(tasks) || error(
    "full/withheld first prompts differ for " *
    "$(length(tasks) - first_prompt_pairs_equal) task(s)",
)

summary = (;
    schema_version=AGENT_ENVIRONMENT_FORMAT_VERSION,
    label,
    model=abspath(model_dir),
    device,
    revision=String(options[:revision]),
    tasks_sha256=task_set.sha256,
    evaluated_tasks=length(tasks),
    coordinate_system=task_set.coordinate_system,
    system_prompt=gridworld_system_prompt(),
    system_prompt_sha256=LifeAI._sha256_hex(gridworld_system_prompt()),
    tool="move(direction enum)",
    context_tokens=Int(options[:context]),
    max_new_tokens=Int(options[:max_new_tokens]),
    strategy="greedy",
    enable_thinking=false,
    load_seconds,
    seconds,
    first_prompt_pairs_equal,
    oracle=oracle_report,
    full_feedback=full_report,
    no_feedback=none_report,
    no_feedback_vs_full_feedback=paired,
    trace=trace_path,
)
summary_path = joinpath(output_dir, "$(label)_gridworld_summary.json")
open(summary_path, "w") do io
    JSON3.pretty(io, JSON3.write(summary))
    println(io)
end
println(stderr, "full $(full_report.successes)/$(full_report.total), " *
    "no-feedback $(none_report.successes)/$(none_report.total), " *
    "first prompts equal $first_prompt_pairs_equal/$(length(tasks)), " *
    "completed in $(round(seconds; digits=1))s")
