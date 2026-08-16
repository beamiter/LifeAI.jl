#!/usr/bin/env julia

# Chapter 37 — score a reproduced Qwen3 dense checkpoint on the frozen task set.
#
# Three protocols over the same items:
#   mmlu_loglikelihood  base prompt, no chat template, argmax over ` A`..` D`
#   mmlu_generative     official chat template, greedy, letter parsed from the text
#   gsm8k_generative    official chat template, greedy, final number parsed
#
# Every accuracy is printed with its Wilson 95% interval, and unparseable
# completions are counted separately from wrong ones.

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
    "chapter37_qwen3_task_quality",
    "fixtures",
    "eval_tasks.json",
)
const ALL_PROTOCOLS = ("mmlu_loglikelihood", "mmlu_generative", "gsm8k_generative")

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/run_qwen3_eval.jl MODEL_DIR --out DIR [options]

options:
  --tasks PATH               frozen task fixture (default: the Chapter 37 fixture)
  --out DIR                  output directory for per-item JSONL and summaries (required)
  --label NAME               name used in the output filenames (default: the model directory
                             basename, or its parent when that basename is a revision sha)
  --protocols LIST           comma separated subset of $(join(ALL_PROTOCOLS, ","))
  --mmlu-limit N             score only the first N MMLU items
  --gsm8k-limit N            score only the first N GSM8K items
  --max-new-tokens N         generative MMLU output allowance (default 256)
  --gsm8k-max-new-tokens N   generative GSM8K output allowance (default 512)
  --context N                session context tokens (default 2048)
  --variant NAME             Qwen3 dense variant to enforce, e.g. qwen3_4b
  --revision SHA             expected checkpoint revision
  --cpu                      run on CPU instead of CUDA
  -h, --help                 show this help
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    model_dir = args[1]
    options = Dict{Symbol,Any}(
        :tasks => DEFAULT_TASKS,
        :out => nothing,
        :label => nothing,
        :protocols => join(ALL_PROTOCOLS, ","),
        :mmlu_limit => 0,
        :gsm8k_limit => 0,
        :max_new_tokens => 256,
        :gsm8k_max_new_tokens => 512,
        :context => 2048,
        :variant => nothing,
        :revision => "",
        :cpu => false,
    )
    integers = (:mmlu_limit, :gsm8k_limit, :max_new_tokens, :gsm8k_max_new_tokens, :context)
    index = 2
    while index <= length(args)
        option = args[index]
        if option == "--cpu"
            options[:cpu] = true
            index += 1
        elseif option in ("-h", "--help")
            usage()
            exit()
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
    return model_dir, options
end

sha256_hex(text::AbstractString) = bytes2hex(sha256(codeunits(text)))

function write_jsonl(path, records)
    open(path, "w") do io
        for record in records
            JSON3.write(io, record)
            println(io)
        end
    end
    return path
end

function write_json(path, payload)
    open(path, "w") do io
        JSON3.pretty(io, JSON3.write(payload))
        println(io)
    end
    return path
end

limited(items, limit) = limit > 0 ? items[1:min(limit, length(items))] : items

model_dir, options = parse_args(ARGS)
protocols = split(String(options[:protocols]), ",")
all(protocol -> protocol in ALL_PROTOCOLS, protocols) || error(
    "unknown protocol in $(options[:protocols]); allowed: $(join(ALL_PROTOCOLS, ", "))",
)
tasks_path = abspath(String(options[:tasks]))
tasks = load_eval_tasks(tasks_path)
output_dir = abspath(String(options[:out]))
mkpath(output_dir)
# Assets live under <model>/<revision-sha>, so the directory's own basename is the
# revision. Fall back to the parent only when the basename looks like one.
function default_label(directory)
    name = basename(rstrip(abspath(directory), '/'))
    occursin(r"^[0-9a-f]{40}$", name) || return name
    return basename(rstrip(dirname(rstrip(abspath(directory), '/')), '/'))
end
label = options[:label] === nothing ? default_label(model_dir) : String(options[:label])

to_device = identity
device_label = "cpu"
if !options[:cpu]
    CUDA.functional() || error("CUDA.jl is not functional; pass --cpu to run on the CPU")
    to_device = CUDA.cu
    device_label = CUDA.name(CUDA.device())
end

println(stderr, "label=$label device=$device_label tasks_sha256=$(tasks.sha256)")
println(stderr, "mmlu=$(length(tasks.mmlu)) gsm8k=$(length(tasks.gsm8k)) protocols=$(join(protocols, ","))")

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

summaries = Dict{String,Any}()

# Truncation is counted separately so "the model was cut off" is never reported as
# "the model was wrong".
truncated_count(results) = count(
    result -> result.detail isa NamedTuple && hasproperty(result.detail, :stop_reason) &&
        result.detail.stop_reason == "length",
    results,
)

function report_protocol(name, results, seconds)
    report = accuracy_report(results)
    truncated = truncated_count(results)
    println(stderr, "$name: $(report.correct)/$(report.total) = " *
                    "$(round(report.accuracy; digits=4)) " *
                    "(Wilson 95% $(round(report.wilson_lower; digits=3))–" *
                    "$(round(report.wilson_upper; digits=3))) " *
                    "unparsed=$(report.unparsed) truncated=$truncated " *
                    "in $(round(seconds; digits=1))s")
    return (; report..., truncated)
end

item_record(result) = (;
    id=result.id,
    subject=result.subject,
    protocol=String(result.protocol),
    prompt_sha256=result.prompt_sha256,
    prompt_token_count=result.prompt_token_count,
    completion=result.completion,
    extracted=result.extracted,
    expected=result.expected,
    correct=result.correct,
    parsed=result.parsed,
    detail=result.detail,
)

if "mmlu_loglikelihood" in protocols
    items = limited(tasks.mmlu, options[:mmlu_limit])
    started = time_ns()
    results = EvalItemResult[]
    for (position, item) in enumerate(items)
        scores, prompt_tokens = mmlu_loglikelihood_scores(
            session.model,
            session.parameters,
            session.tokenizer,
            item,
        )
        prediction = argmax(scores)
        push!(results, EvalItemResult(
            item.id,
            item.subject,
            :mmlu_loglikelihood,
            sha256_hex(mmlu_loglikelihood_prompt(item)),
            prompt_tokens,
            "",
            LifeAI.MMLU_LETTERS[prediction],
            LifeAI.MMLU_LETTERS[item.answer_index + 1],
            prediction == item.answer_index + 1,
            true,
            (; logprobs=scores),
        ))
        position % 50 == 0 && println(stderr, "  loglikelihood $position/$(length(items))")
    end
    seconds = (time_ns() - started) / 1.0e9
    report = report_protocol("mmlu_loglikelihood", results, seconds)
    write_jsonl(joinpath(output_dir, "$(label)_mmlu_loglikelihood_items.jsonl"),
        [item_record(result) for result in results])
    summaries["mmlu_loglikelihood"] = (;
        report..., seconds, subjects=subject_report(results))
end

if "mmlu_generative" in protocols
    items = limited(tasks.mmlu, options[:mmlu_limit])
    started = time_ns()
    results = EvalItemResult[]
    for (position, item) in enumerate(items)
        messages = mmlu_chat_messages(item)
        prompt = apply_qwen3_chat_template(
            session.tokenizer, messages; add_generation_prompt=true, enable_thinking=false)
        prompt_ids = encode(session.tokenizer, prompt; add_special_tokens=false)
        generated = generate_hf_qwen3_bf16!(
            session, prompt_ids;
            max_new_tokens=options[:max_new_tokens], strategy=:greedy)
        letter = extract_mmlu_letter(generated.completion)
        expected = LifeAI.MMLU_LETTERS[item.answer_index + 1]
        # The prompt asks for a single letter and nothing else. Whether the model
        # obeyed is a separate property from whether it was right, so it is recorded
        # rather than folded into the accuracy.
        trimmed = strip(generated.completion)
        compliant = letter !== nothing && startswith(trimmed, letter)
        push!(results, EvalItemResult(
            item.id,
            item.subject,
            :mmlu_generative,
            sha256_hex(prompt),
            length(prompt_ids),
            generated.completion,
            letter === nothing ? "" : letter,
            expected,
            letter == expected,
            letter !== nothing,
            (; stop_reason=String(generated.stop_reason),
               generated_tokens=length(generated.generated_ids),
               format_compliant=compliant),
        ))
        position % 50 == 0 && println(stderr, "  generative $position/$(length(items))")
    end
    seconds = (time_ns() - started) / 1.0e9
    report = report_protocol("mmlu_generative", results, seconds)
    compliant = count(result -> result.detail.format_compliant, results)
    println(stderr, "  format_compliant=$compliant/$(length(results))")
    write_jsonl(joinpath(output_dir, "$(label)_mmlu_generative_items.jsonl"),
        [item_record(result) for result in results])
    summaries["mmlu_generative"] = (;
        report..., seconds, format_compliant=compliant, subjects=subject_report(results))
end

if "gsm8k_generative" in protocols
    items = limited(tasks.gsm8k, options[:gsm8k_limit])
    started = time_ns()
    results = EvalItemResult[]
    for (position, item) in enumerate(items)
        messages = gsm8k_chat_messages(item)
        prompt = apply_qwen3_chat_template(
            session.tokenizer, messages; add_generation_prompt=true, enable_thinking=false)
        prompt_ids = encode(session.tokenizer, prompt; add_special_tokens=false)
        generated = generate_hf_qwen3_bf16!(
            session, prompt_ids;
            max_new_tokens=options[:gsm8k_max_new_tokens], strategy=:greedy)
        predicted = extract_gsm8k_answer(generated.completion)
        push!(results, EvalItemResult(
            item.id,
            "",
            :gsm8k_generative,
            sha256_hex(prompt),
            length(prompt_ids),
            generated.completion,
            predicted === nothing ? "" : string(predicted),
            item.answer,
            gsm8k_answer_matches(item.answer, predicted),
            predicted !== nothing,
            (; stop_reason=String(generated.stop_reason),
               generated_tokens=length(generated.generated_ids)),
        ))
        position % 25 == 0 && println(stderr, "  gsm8k $position/$(length(items))")
    end
    seconds = (time_ns() - started) / 1.0e9
    report = report_protocol("gsm8k_generative", results, seconds)
    write_jsonl(joinpath(output_dir, "$(label)_gsm8k_generative_items.jsonl"),
        [item_record(result) for result in results])
    summaries["gsm8k_generative"] = (; report..., seconds)
end

write_json(joinpath(output_dir, "$(label)_summary.json"), (;
    label,
    model_dir=abspath(model_dir),
    revision=String(options[:revision]),
    device=device_label,
    tasks=tasks_path,
    tasks_sha256=tasks.sha256,
    context_tokens=options[:context],
    max_new_tokens=options[:max_new_tokens],
    gsm8k_max_new_tokens=options[:gsm8k_max_new_tokens],
    strategy="greedy",
    enable_thinking=false,
    load_seconds=load_seconds,
    protocols=summaries,
))
println(stderr, "wrote $(joinpath(output_dir, "$(label)_summary.json"))")
