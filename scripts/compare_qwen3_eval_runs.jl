#!/usr/bin/env julia

# Chapter 38 — paired comparison of two eval runs over the same items.
#
# Two independent Wilson intervals cannot answer "did the change help": they throw
# away the pairing. This prints the 2x2 discordance table and an exact two-sided
# McNemar p-value, which is what a same-items A/B actually supports.

using JSON3
using LifeAI

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/compare_qwen3_eval_runs.jl BASELINE.jsonl VARIANT.jsonl [--out PATH]
""")
end

length(ARGS) >= 2 || (usage(); exit(2))
ARGS[1] in ("-h", "--help") && (usage(); exit())
out_path = nothing
if length(ARGS) >= 4 && ARGS[3] == "--out"
    out_path = ARGS[4]
end

function load_results(path)
    results = EvalItemResult[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        row = JSON3.read(line)
        push!(results, EvalItemResult(
            String(row.id), String(row.subject), Symbol(String(row.protocol)),
            String(row.prompt_sha256), Int(row.prompt_token_count),
            String(row.completion), String(row.extracted), String(row.expected),
            Bool(row.correct), Bool(row.parsed), row.detail,
        ))
    end
    isempty(results) && error("no items in $path")
    return results
end

baseline = load_results(ARGS[1])
variant = load_results(ARGS[2])
comparison = paired_comparison(baseline, variant)

baseline_report = accuracy_report(baseline)
variant_report = accuracy_report(variant)

println(stderr, "baseline $(basename(ARGS[1])): $(baseline_report.correct)/$(baseline_report.total)" *
                " = $(round(baseline_report.accuracy; digits=4))" *
                " (Wilson $(round(baseline_report.wilson_lower; digits=3))–$(round(baseline_report.wilson_upper; digits=3)))")
println(stderr, "variant  $(basename(ARGS[2])): $(variant_report.correct)/$(variant_report.total)" *
                " = $(round(variant_report.accuracy; digits=4))" *
                " (Wilson $(round(variant_report.wilson_lower; digits=3))–$(round(variant_report.wilson_upper; digits=3)))")
println(stderr, "paired $(comparison.paired) items")
println(stderr, "  both correct        $(comparison.both)")
println(stderr, "  both wrong          $(comparison.neither)")
println(stderr, "  baseline only right $(comparison.baseline_only)   <- lost by the variant")
println(stderr, "  variant only right  $(comparison.variant_only)   <- gained by the variant")
println(stderr, "  difference $(round(100 * comparison.difference; digits=2)) points, " *
                "exact McNemar p = $(round(comparison.p_value; digits=4))")

out_path === nothing || open(out_path, "w") do io
    JSON3.pretty(io, JSON3.write((;
        baseline=abspath(ARGS[1]), variant=abspath(ARGS[2]),
        baseline_accuracy=baseline_report.accuracy,
        variant_accuracy=variant_report.accuracy,
        comparison...,
    )))
    println(io)
end
