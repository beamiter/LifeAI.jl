#!/usr/bin/env julia

# Chapter 36 — run the frozen tool-loop task set against a resident Qwen3 BF16
# session and record one replayable JSONL trace per model turn.
#
# The reported valid-tool-call rate is a descriptive measurement of exactly the
# frozen task set, not a capability claim: it is printed with its Wilson 95%
# interval and the task-set digest.

using LuxCUDA
using CUDA
using JSON3
using SHA: sha256
using LifeAI

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_TASKS = joinpath(
    REPO_ROOT,
    "test",
    "episodes",
    "episode07_agent_closed_loop",
    "chapter36_qwen3_tools_chat_template",
    "fixtures",
    "tool_loop_tasks.json",
)

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/run_qwen3_tool_loop.jl MODEL_DIR --out TRACE.jsonl [options]

options:
  --tasks PATH         frozen task set (default: the Chapter 36 fixture)
  --sandbox PATH       root the filesystem tools may read (default: repository root)
  --out PATH           JSONL trace, one line per model turn (required)
  --summary PATH       JSON summary with the pre-registered rate and its interval
  --context N          session context tokens (default 4096)
  --max-new-tokens N   per-turn output allowance (default 256)
  --max-steps N        model turns per task (default 4)
  --variant NAME       Qwen3 dense variant to enforce, e.g. qwen3_4b
  --revision SHA       expected checkpoint revision
  --thinking           enable Qwen3 thinking mode
  --cpu                run on CPU instead of CUDA
  -h, --help           show this help
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    model_dir = args[1]
    options = Dict{Symbol,Any}(
        :tasks => DEFAULT_TASKS,
        :sandbox => REPO_ROOT,
        :out => nothing,
        :summary => nothing,
        :context => 4096,
        :max_new_tokens => 256,
        :max_steps => 4,
        :variant => nothing,
        :revision => "",
        :thinking => false,
        :cpu => false,
    )
    index = 2
    while index <= length(args)
        option = args[index]
        if option == "--thinking"
            options[:thinking] = true
            index += 1
        elseif option == "--cpu"
            options[:cpu] = true
            index += 1
        elseif option in ("-h", "--help")
            usage()
            exit()
        elseif startswith(option, "--")
            index < length(args) || (usage(); exit(2))
            value = args[index + 1]
            key = Symbol(replace(option[3:end], "-" => "_"))
            haskey(options, key) || (usage(); exit(2))
            if key in (:context, :max_new_tokens, :max_steps)
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
    return model_dir, options
end

file_sha256(path) = bytes2hex(sha256(read(path)))

json_line(io, payload) = (JSON3.write(io, payload); println(io))

model_dir, options = parse_args(ARGS)
tasks_path = abspath(String(options[:tasks]))
task_set = JSON3.read(read(tasks_path, String))
tasks = task_set.tasks
isempty(tasks) && error("task set contains no tasks: $tasks_path")
system_prompt = String(task_set.system)
sandbox_root = abspath(String(options[:sandbox]))
registry = default_agent_tools(sandbox_root)

to_device = identity
device_label = "cpu"
if !options[:cpu]
    CUDA.functional() || error("CUDA.jl is not functional; pass --cpu to run on the CPU")
    to_device = CUDA.cu
    device_label = CUDA.name(CUDA.device())
end

println(stderr, "device=$device_label tasks=$(length(tasks)) sandbox=$sandbox_root")
println(stderr, "task_set_sha256=$(file_sha256(tasks_path))")

load_started = time_ns()
session = load_hf_qwen3_bf16_session(
    model_dir;
    context_tokens=options[:context],
    revision=String(options[:revision]),
    variant=options[:variant] === nothing ? nothing : Symbol(options[:variant]),
    to_device,
)
options[:cpu] || (CUDA.synchronize(); GC.gc(); CUDA.reclaim())
load_seconds = (time_ns() - load_started) / 1.0e9
free_after_load = options[:cpu] ? 0 : Int(CUDA.free_memory())
println(stderr, "loaded in $(round(load_seconds; digits=2))s free=$(free_after_load)")

records = Any[]
verdicts = Symbol[]
coerced_tasks = String[]
minimum_free = Ref(free_after_load)
open(String(options[:out]), "w") do trace_io
    for task in tasks
        name = String(task.name)
        trace = run_qwen3_tool_loop(
            session,
            registry,
            String(task.user);
            system=system_prompt,
            max_steps=options[:max_steps],
            max_new_tokens=options[:max_new_tokens],
            enable_thinking=options[:thinking],
            strategy=:greedy,
        )
        summary = agent_loop_summary(trace)
        push!(verdicts, summary.first_turn_validity)
        coerced = any(
            step -> any(call -> !isempty(call.coerced_arguments), step.tool_calls),
            trace.steps,
        )
        coerced && push!(coerced_tasks, name)
        for step in trace.steps
            json_line(trace_io, (;
                task=name,
                turn=step.turn,
                prompt_sha256=step.prompt_sha256,
                prompt_token_count=step.prompt_token_count,
                generated_ids=step.generated_ids,
                completion=step.completion,
                stop_reason=String(step.stop_reason),
                validity=String(step.validity),
                invalid_blocks=[(; raw=entry.raw, reason=entry.reason) for entry in step.invalid_blocks],
                tool_calls=[(;
                    name=call.name,
                    arguments=call.arguments_json,
                    ok=call.ok,
                    output=call.output,
                    error=call.error,
                    coerced_arguments=call.coerced_arguments,
                ) for call in step.tool_calls],
                prefill_seconds=step.prefill_seconds,
                decode_seconds=step.decode_seconds,
            ))
        end
        flush(trace_io)
        options[:cpu] || (minimum_free[] = min(minimum_free[], Int(CUDA.free_memory())))
        push!(records, (;
            task=name,
            turns=summary.turns,
            first_turn_validity=String(summary.first_turn_validity),
            coerced_arguments=coerced,
            stop_reason=String(summary.stop_reason),
            tool_calls_executed=summary.tool_calls_executed,
            tool_calls_succeeded=summary.tool_calls_succeeded,
            answer=trace.answer,
        ))
        println(stderr, "$name: $(summary.first_turn_validity) turns=$(summary.turns) " *
                        "calls=$(summary.tool_calls_succeeded)/$(summary.tool_calls_executed)")
    end
end

valid_count = count(==(:valid), verdicts)
# Strict count: a valid first turn that also needed no string-to-integer coercion.
strict_valid = count(
    record -> record.first_turn_validity == "valid" && !record.coerced_arguments,
    records,
)
interval = wilson_interval(valid_count, length(tasks))
summary_payload = (;
    model_dir=abspath(model_dir),
    device=device_label,
    task_set=tasks_path,
    task_set_sha256=file_sha256(tasks_path),
    task_count=length(tasks),
    system_prompt_sha256=bytes2hex(sha256(codeunits(system_prompt))),
    context_tokens=options[:context],
    max_new_tokens=options[:max_new_tokens],
    max_steps=options[:max_steps],
    enable_thinking=options[:thinking],
    strategy="greedy",
    load_seconds=load_seconds,
    free_bytes_after_load=free_after_load,
    minimum_free_bytes=minimum_free[],
    valid_tool_call_count=valid_count,
    valid_tool_call_rate=interval.point,
    wilson_lower=interval.lower,
    wilson_upper=interval.upper,
    strict_valid_without_coercion=strict_valid,
    coerced_tasks=coerced_tasks,
    invalid_count=count(==(:invalid), verdicts),
    no_call_count=count(==(:none), verdicts),
    tasks=records,
)
options[:summary] === nothing || open(String(options[:summary]), "w") do io
    JSON3.pretty(io, JSON3.write(summary_payload))
    println(io)
end

println(stderr, "valid $valid_count/$(length(tasks)) " *
                "(Wilson 95% $(round(interval.lower; digits=3))–$(round(interval.upper; digits=3))) " *
                "invalid=$(count(==(:invalid), verdicts)) none=$(count(==(:none), verdicts))")
println(stderr, "minimum free bytes during the run: $(minimum_free[])")
