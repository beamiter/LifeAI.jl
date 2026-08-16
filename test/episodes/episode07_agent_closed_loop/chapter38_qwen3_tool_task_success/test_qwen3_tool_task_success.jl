using Test
using JSON3
using LifeAI
using LifeAI:
    AgentTool,
    EvalItemResult,
    ToolRegistry,
    calculator_tool,
    evaluate_arithmetic,
    invoke_agent_tool,
    mcnemar_exact,
    paired_comparison,
    parse_qwen3_tool_calls,
    qwen3_tool_specs

@testset "Chapter 38 — arithmetic evaluator" begin
    for (expression, expected) in [
        ("1+2", 3.0),
        ("(16 - 3 - 4) * 2", 18.0),
        ("2*3+4*5", 26.0),
        ("10/4", 2.5),
        ("-5 + 3", -2.0),
        ("+7", 7.0),
        ("2.5*4", 10.0),
        ("((1+2)*3)", 9.0),
        ("100 - 10 - 10", 80.0),
        ("2 * -3", -6.0),
        # Models write grouped digits; accepted only between digits of an integer part.
        ("1,000 + 1", 1001.0),
        ("1_000 * 2", 2000.0),
        (" 8 ", 8.0),
        (".5 + .5", 1.0),
    ]
        @test evaluate_arithmetic(expression) ≈ expected
    end

    # Everything outside the grammar is refused. The expression comes from model
    # output, so the tool must never be a way to run model-authored code.
    for rejected in [
        "1+", "2**3", "eval(1)", "rm -rf /", "", "1 2", "(1+2", "1+2)",
        "3 % 2", "2^8", "sqrt(4)", "0x10", "1e5", "1,,000", "1.2.3", "--", "1,",
    ]
        @test_throws ArgumentError evaluate_arithmetic(rejected)
    end
    @test_throws ArgumentError evaluate_arithmetic("1/0")
    @test_throws ArgumentError evaluate_arithmetic("1/(2-2)")
end

@testset "Chapter 38 — calculator tool" begin
    registry = ToolRegistry([calculator_tool()])
    call(text) = only(parse_qwen3_tool_calls(text).calls)
    invoke(expression) = invoke_agent_tool(registry, call(
        "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"$expression\"}}\n</tool_call>",
    ))

    result = invoke("(16-3-4)*2")
    @test result.ok
    # Whole results are written the way a grade-school answer is, not as `18.0`.
    @test result.output == "18"
    @test invoke("10/4").output == "2.5"
    @test invoke("7").output == "7"

    failed = invoke("rm -rf /")
    @test !failed.ok
    @test occursin("unexpected character", something(failed.error, ""))
    @test !invoke("1/0").ok

    # A tool failure is reported, never thrown, so the loop can feed it back.
    @test invoke("nonsense").ok == false
    @test LifeAI._python_json_text(qwen3_tool_specs(registry)[1]) ==
          "{\"type\": \"function\", \"function\": {\"name\": \"calculator\", " *
          "\"description\": \"Evaluate an arithmetic expression over + - * / and parentheses.\", " *
          "\"parameters\": {\"type\": \"object\", \"properties\": {\"expression\": " *
          "{\"type\": \"string\", \"description\": \"An arithmetic expression, for example " *
          "\\\"(16 - 3 - 4) * 2\\\".\"}}, \"required\": [\"expression\"]}}}"
end

@testset "Chapter 38 — exact McNemar" begin
    @test mcnemar_exact(0, 0) == 1.0
    @test mcnemar_exact(3, 3) == 1.0
    @test mcnemar_exact(0, 1) == 1.0
    @test mcnemar_exact(0, 5) ≈ 2 / 32
    @test mcnemar_exact(1, 9) ≈ 2 * (1 + 10) / 1024
    @test mcnemar_exact(5, 0) == mcnemar_exact(0, 5)
    # A p-value is a probability: the doubling must never push it past one.
    for lost in 0:6, gained in 0:6
        @test 0 < mcnemar_exact(lost, gained) <= 1
    end
    @test_throws ArgumentError mcnemar_exact(-1, 2)
end

@testset "Chapter 38 — paired comparison" begin
    make(id, correct) = EvalItemResult(
        id, "", :gsm8k_generative, "sha", 1, "", "", "7", correct, true, nothing)
    baseline = [make("a", true), make("b", true), make("c", false), make("d", false)]
    variant = [make("a", true), make("b", false), make("c", true), make("d", false)]
    comparison = paired_comparison(baseline, variant)
    @test comparison.paired == 4
    @test comparison.both == 1
    @test comparison.neither == 1
    @test comparison.baseline_only == 1
    @test comparison.variant_only == 1
    @test comparison.baseline_correct == 2
    @test comparison.variant_correct == 2
    @test comparison.difference == 0.0
    @test comparison.p_value == 1.0

    # Accuracy alone hides the churn: both runs score 2/4 while half the items moved.
    @test comparison.both + comparison.neither == 2

    gained = [make("a", true), make("b", true), make("c", true), make("d", true)]
    improved = paired_comparison(baseline, gained)
    @test improved.variant_only == 2
    @test improved.baseline_only == 0
    @test improved.difference == 0.5
    @test improved.p_value ≈ 0.5

    @test_throws ArgumentError paired_comparison(baseline, [make("zz", true)])
    @test_throws ArgumentError paired_comparison(
        [make("a", true), make("a", false)], [make("a", true)])
end
