#!/usr/bin/env julia

# Chapter 38 — run the frozen GSM8K items through the tool loop with a calculator,
# so the result pairs item-for-item with the Chapter 37 no-tool baseline.
#
# The manipulation is kept as narrow as it can be: the user message is byte-identical
# to the baseline's (`gsm8k_chat_messages`), no system message is added, and decoding
# is the same greedy. The only difference is the `# Tools` header the chat template
# injects plus the loop's follow-up turns. That header is itself part of the
# manipulation and cannot be removed — declaring a tool *is* changing the prompt.

using LuxCUDA
using CUDA
using JSON3
using SHA: sha256
using LifeAI

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
# Frozen verbatim: the nudge is part of the manipulation and must be reported, not
# paraphrased. Reproducing an arm requires the exact bytes.
const CHAPTER38_NUDGE = "You have a calculator tool. Use it for every arithmetic " *
    "computation instead of doing the arithmetic yourself."

const DEFAULT_TASKS = joinpath(
    REPO_ROOT, "test", "episodes", "episode07_agent_closed_loop",
    "chapter37_qwen3_task_quality", "fixtures", "eval_tasks.json",
)

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/run_qwen3_gsm8k_tool_eval.jl MODEL_DIR --out DIR [options]

options:
  --tasks PATH        frozen task fixture (default: the Chapter 37 fixture)
  --out DIR           output directory (required)
  --label NAME        name used in the output filenames (default: qwen3-tool)
  --limit N           score only the first N GSM8K items
  --max-steps N       model turns per item (default 4)
  --max-new-tokens N  per-turn output allowance (default 512)
  --system TEXT       system message to prepend (default: none)
  --nudge             use the frozen system message that asks for the calculator
  --context N         session context tokens (default 4096)
  --variant NAME      Qwen3 dense variant to enforce
  --revision SHA      expected checkpoint revision
  --cpu               run on CPU instead of CUDA
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    options = Dict{Symbol,Any}(
        :tasks => DEFAULT_TASKS, :out => nothing, :label => "qwen3-tool",
        :limit => 0, :max_steps => 4, :max_new_tokens => 512, :context => 4096,
        :variant => nothing, :revision => "", :cpu => false,
        :system => nothing, :nudge => false,
    )
    integers = (:limit, :max_steps, :max_new_tokens, :context)
    index = 2
    while index <= length(args)
        option = args[index]
        if option == "--cpu"
            options[:cpu] = true
            index += 1
        elseif option == "--nudge"
            options[:nudge] = true
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

sha256_hex(text::AbstractString) = bytes2hex(sha256(codeunits(text)))

model_dir, options = parse_args(ARGS)
tasks = load_eval_tasks(String(options[:tasks]))
items = options[:limit] > 0 ? tasks.gsm8k[1:min(options[:limit], length(tasks.gsm8k))] :
    tasks.gsm8k
output_dir = abspath(String(options[:out]))
mkpath(output_dir)
label = String(options[:label])
registry = ToolRegistry([calculator_tool()])
options[:nudge] && options[:system] !== nothing && error(
    "pass either --nudge or --system, not both",
)
system_message = options[:nudge] ? CHAPTER38_NUDGE :
    (options[:system] === nothing ? nothing : String(options[:system]))

to_device = identity
device_label = "cpu"
if !options[:cpu]
    CUDA.functional() || error("CUDA.jl is not functional; pass --cpu to run on the CPU")
    to_device = CUDA.cu
    device_label = CUDA.name(CUDA.device())
end

println(stderr, "label=$label device=$device_label items=$(length(items)) " *
                "tasks_sha256=$(tasks.sha256)")

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
println(stderr, "loaded in $(round(load_seconds; digits=2))s")

results = EvalItemResult[]
records = Any[]
started = time_ns()
open(joinpath(output_dir, "$(label)_gsm8k_tool_trace.jsonl"), "w") do trace_io
    for (position, item) in enumerate(items)
        # Byte-identical to the Chapter 37 baseline's user message.
        user_content = only(gsm8k_chat_messages(item))[:content]
        trace = run_qwen3_tool_loop(
            session, registry, user_content;
            system=system_message,
            max_steps=options[:max_steps],
            max_new_tokens=options[:max_new_tokens],
            enable_thinking=false,
            strategy=:greedy,
        )
        predicted = extract_gsm8k_answer(trace.answer)
        calls = sum(length(step.tool_calls) for step in trace.steps; init=0)
        failures = sum(
            count(call -> !call.ok, step.tool_calls) for step in trace.steps; init=0
        )
        push!(results, EvalItemResult(
            item.id, "", :gsm8k_tool_loop,
            isempty(trace.steps) ? "" : trace.steps[1].prompt_sha256,
            isempty(trace.steps) ? 0 : trace.steps[1].prompt_token_count,
            trace.answer,
            predicted === nothing ? "" : string(predicted),
            item.answer,
            gsm8k_answer_matches(item.answer, predicted),
            predicted !== nothing,
            (; stop_reason=String(trace.stop_reason), turns=length(trace.steps),
               tool_calls=calls, tool_failures=failures),
        ))
        for step in trace.steps
            JSON3.write(trace_io, (;
                task=item.id, turn=step.turn, prompt_sha256=step.prompt_sha256,
                prompt_token_count=step.prompt_token_count,
                completion=step.completion, stop_reason=String(step.stop_reason),
                validity=String(step.validity),
                tool_calls=[(; name=call.name, arguments=call.arguments_json,
                              ok=call.ok, output=call.output, error=call.error)
                            for call in step.tool_calls],
                prefill_seconds=step.prefill_seconds, decode_seconds=step.decode_seconds,
            ))
            println(trace_io)
        end
        flush(trace_io)
        push!(records, (;
            id=item.id, subject="", protocol="gsm8k_tool_loop",
            prompt_sha256=results[end].prompt_sha256,
            prompt_token_count=results[end].prompt_token_count,
            completion=trace.answer, extracted=results[end].extracted,
            expected=item.answer, correct=results[end].correct,
            parsed=results[end].parsed, detail=results[end].detail,
        ))
        position % 25 == 0 && println(stderr, "  $position/$(length(items))")
    end
end
seconds = (time_ns() - started) / 1.0e9

open(joinpath(output_dir, "$(label)_gsm8k_tool_items.jsonl"), "w") do io
    for record in records
        JSON3.write(io, record)
        println(io)
    end
end

report = accuracy_report(results)
total_calls = sum(result.detail.tool_calls for result in results)
used_tool = count(result -> result.detail.tool_calls > 0, results)
failed_calls = sum(result.detail.tool_failures for result in results)
exhausted = count(result -> result.detail.stop_reason == "max_steps", results)
with_tool = filter(result -> result.detail.tool_calls > 0, results)
without_tool = filter(result -> result.detail.tool_calls == 0, results)

open(joinpath(output_dir, "$(label)_gsm8k_tool_summary.json"), "w") do io
    JSON3.pretty(io, JSON3.write((;
        label, model_dir=abspath(model_dir), device=device_label,
        tasks_sha256=tasks.sha256, tools=["calculator"],
        system_message=system_message === nothing ? "" : system_message,
        system_sha256=system_message === nothing ? "" : sha256_hex(system_message),
        max_steps=options[:max_steps], max_new_tokens=options[:max_new_tokens],
        context_tokens=options[:context], strategy="greedy", enable_thinking=false,
        load_seconds, seconds, report...,
        items_using_tool=used_tool, tool_calls=total_calls,
        failed_tool_calls=failed_calls, max_steps_exhausted=exhausted,
        accuracy_with_tool_use=isempty(with_tool) ? nothing :
            count(r -> r.correct, with_tool) / length(with_tool),
        accuracy_without_tool_use=isempty(without_tool) ? nothing :
            count(r -> r.correct, without_tool) / length(without_tool),
    )))
    println(io)
end

println(stderr, "$(report.correct)/$(report.total) = $(round(report.accuracy; digits=4)) " *
                "(Wilson 95% $(round(report.wilson_lower; digits=3))–" *
                "$(round(report.wilson_upper; digits=3))) " *
                "unparsed=$(report.unparsed) in $(round(seconds; digits=1))s")
println(stderr, "used tool on $used_tool/$(length(results)) items, " *
                "$total_calls calls ($failed_calls failed), " *
                "$exhausted hit max_steps")
