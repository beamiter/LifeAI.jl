#!/usr/bin/env julia

# Chapter 37 — re-score a frozen generative run with the current extraction rules.
#
# The per-item JSONL keeps every raw completion, so changing an extraction rule does
# not require re-running the model. It also means an extraction change is auditable:
# this prints the before/after counts and every item whose verdict moved.

using JSON3
using LifeAI

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/rescore_qwen3_eval.jl ITEMS.jsonl [--out PATH] [--show-changes]
""")
end

isempty(ARGS) && (usage(); exit(2))
ARGS[1] in ("-h", "--help") && (usage(); exit())
items_path = ARGS[1]
out_path = nothing
show_changes = false
index = 2
while index <= length(ARGS)
    if ARGS[index] == "--show-changes"
        global show_changes = true
        global index += 1
    elseif ARGS[index] == "--out" && index < length(ARGS)
        global out_path = ARGS[index + 1]
        global index += 2
    else
        usage()
        exit(2)
    end
end

rows = [JSON3.read(line) for line in eachline(items_path) if !isempty(strip(line))]
isempty(rows) && error("no items in $items_path")
protocol = String(rows[1].protocol)

records = Any[]
changed = Any[]
for row in rows
    completion = String(row.completion)
    expected = String(row.expected)
    if protocol == "mmlu_generative"
        letter = extract_mmlu_letter(completion)
        extracted = letter === nothing ? "" : letter
        parsed = letter !== nothing
        correct = letter == expected
    elseif protocol == "gsm8k_generative"
        value = extract_gsm8k_answer(completion)
        extracted = value === nothing ? "" : string(value)
        parsed = value !== nothing
        correct = gsm8k_answer_matches(expected, value)
    else
        error("rescoring only applies to generative protocols, got $protocol")
    end
    (correct != row.correct || parsed != row.parsed) && push!(changed, (;
        id=String(row.id),
        was=(; correct=row.correct, parsed=row.parsed, extracted=String(row.extracted)),
        now=(; correct, parsed, extracted),
    ))
    push!(records, (; id=String(row.id), subject=String(row.subject), extracted, expected,
        correct, parsed, detail=row.detail))
end

previous_correct = count(row -> row.correct, rows)
previous_parsed = count(row -> row.parsed, rows)
now_correct = count(record -> record.correct, records)
now_parsed = count(record -> record.parsed, records)
interval = wilson_interval(now_correct, length(records))
truncated = count(
    row -> haskey(row.detail, :stop_reason) && String(row.detail.stop_reason) == "length",
    rows,
)

println(stderr, "$protocol  $(basename(items_path))")
println(stderr, "  correct  $previous_correct -> $now_correct / $(length(records))" *
                " (Wilson 95% $(round(interval.lower; digits=3))–$(round(interval.upper; digits=3)))")
println(stderr, "  parsed   $previous_parsed -> $now_parsed")
println(stderr, "  unparsed $(length(records) - now_parsed), truncated $truncated," *
                " verdicts changed $(length(changed))")
if show_changes
    for entry in changed
        println(stderr, "    $(entry.id): $(entry.was) -> $(entry.now)")
    end
end

out_path === nothing || open(out_path, "w") do io
    for record in records
        JSON3.write(io, record)
        println(io)
    end
end
