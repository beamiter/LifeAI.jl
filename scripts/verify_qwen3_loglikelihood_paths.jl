#!/usr/bin/env julia

# Chapter 37 — quantify the gap between the two mathematically equivalent ways of
# scoring an MMLU continuation.
#
#   fast     one forward over the prompt; read the four single-token log
#            probabilities off the last position.
#   general  one forward per continuation over prompt+continuation; read the log
#            probability at the position that predicts it.
#
# Causal attention makes these identical in exact arithmetic. Under BF16 they are
# not: changing the sequence length changes GEMM tiling and reduction order. This
# script measures how far apart they land and how often the decision flips, so the
# cheaper path is used with evidence rather than assumption.

using LuxCUDA
using CUDA
using JSON3
using BFloat16s: BFloat16
using LifeAI

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_TASKS = joinpath(
    REPO_ROOT,
    "test", "episodes", "episode07_agent_closed_loop",
    "chapter37_qwen3_task_quality", "fixtures", "eval_tasks.json",
)

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/verify_qwen3_loglikelihood_paths.jl MODEL_DIR --out PATH [options]

options:
  --tasks PATH     frozen task fixture (default: the Chapter 37 fixture)
  --out PATH       JSON report (required)
  --limit N        compare only the first N MMLU items (default: all)
  --context N      max_seq_len for the loaded bundle (default 1024)
  --revision SHA   expected checkpoint revision
  --variant NAME   Qwen3 dense variant to enforce
  --cpu            run on CPU instead of CUDA
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    options = Dict{Symbol,Any}(
        :tasks => DEFAULT_TASKS, :out => nothing, :limit => 0,
        :context => 1024, :revision => "", :variant => nothing, :cpu => false,
    )
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
            if key in (:limit, :context)
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

"""General path: one forward per continuation, scored at the predicting position."""
function general_scores(model, parameters, tokenizer, item)
    prompt_ids = encode(tokenizer, mmlu_loglikelihood_prompt(item); add_special_tokens=false)
    scores = Float64[]
    for continuation in LifeAI.MMLU_CONTINUATIONS
        ids = encode(tokenizer, continuation; add_special_tokens=false)
        tokens = vcat(prompt_ids, ids)
        result = hf_qwen3_bf16_accel_forward(model, parameters, reshape(tokens, :, 1))
        logits = Array(result.logits)
        total = 0.0
        for (offset, token) in enumerate(ids)
            column = Float64.(logits[:, length(prompt_ids) + offset - 1, 1])
            ceiling = maximum(column)
            total += column[token] - (ceiling + log(sum(exp.(column .- ceiling))))
        end
        push!(scores, total)
    end
    return scores
end

model_dir, options = parse_args(ARGS)
tasks = load_eval_tasks(String(options[:tasks]))
items = options[:limit] > 0 ? tasks.mmlu[1:min(options[:limit], length(tasks.mmlu))] : tasks.mmlu

to_device = identity
device_label = "cpu"
if !options[:cpu]
    CUDA.functional() || error("CUDA.jl is not functional; pass --cpu to run on the CPU")
    to_device = CUDA.cu
    device_label = CUDA.name(CUDA.device())
end

bundle = load_hf_qwen3_bundle(
    model_dir;
    weight_dtype=BFloat16,
    max_seq_len=options[:context],
    revision=String(options[:revision]),
    variant=options[:variant] === nothing ? nothing : Symbol(options[:variant]),
)
parameters = to_device(bundle.parameters)
tokenizer = load_hf_qwen3_tokenizer(model_dir)

records = Any[]
gaps = Float64[]
disagreements = String[]
for (position, item) in enumerate(items)
    fast, _ = mmlu_loglikelihood_scores(bundle.model, parameters, tokenizer, item)
    general = general_scores(bundle.model, parameters, tokenizer, item)
    gap = maximum(abs.(fast .- general))
    push!(gaps, gap)
    agree = argmax(fast) == argmax(general)
    agree || push!(disagreements, item.id)
    push!(records, (;
        id=item.id, subject=item.subject, fast, general,
        max_abs_gap=gap,
        fast_prediction=LifeAI.MMLU_LETTERS[argmax(fast)],
        general_prediction=LifeAI.MMLU_LETTERS[argmax(general)],
        agree,
    ))
    position % 25 == 0 && println(stderr, "  $position/$(length(items))")
end

sorted = sort(gaps)
median_gap = isodd(length(sorted)) ? sorted[(length(sorted) + 1) ÷ 2] :
    (sorted[length(sorted) ÷ 2] + sorted[length(sorted) ÷ 2 + 1]) / 2
payload = (;
    model_dir=abspath(model_dir),
    revision=String(options[:revision]),
    device=device_label,
    tasks_sha256=tasks.sha256,
    compared=length(items),
    max_abs_gap=maximum(gaps),
    median_abs_gap=median_gap,
    decision_agreements=length(items) - length(disagreements),
    disagreement_ids=disagreements,
    items=records,
)
open(String(options[:out]), "w") do io
    JSON3.pretty(io, JSON3.write(payload))
    println(io)
end
println(stderr, "compared=$(length(items)) max_gap=$(round(maximum(gaps); digits=4)) " *
                "median_gap=$(round(median_gap; digits=4)) " *
                "decisions_agree=$(length(items) - length(disagreements))/$(length(items))")
