using Test
using JSON3
using LifeAI
using LifeAI:
    AgentTool,
    ToolRegistry,
    agent_tool_call_validity,
    default_agent_tools,
    invoke_agent_tool,
    parse_qwen3_tool_calls,
    qwen3_tool_specs,
    wilson_interval

if !isdefined(@__MODULE__, :LIFEAI_REPO_ROOT)
    @eval const LIFEAI_REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
end

@testset "Chapter 36 — tool declarations" begin
    registry = default_agent_tools(LIFEAI_REPO_ROOT)
    @test length(registry) == 3
    @test haskey(registry, "add_integers")
    specs = qwen3_tool_specs(registry)
    @test length(specs) == 3
    @test LifeAI._python_json_text(specs[1]) ==
          "{\"type\": \"function\", \"function\": {\"name\": \"add_integers\", " *
          "\"description\": \"Add two integers and return their sum.\", " *
          "\"parameters\": {\"type\": \"object\", \"properties\": " *
          "{\"a\": {\"type\": \"integer\", \"description\": \"Left addend.\"}, " *
          "\"b\": {\"type\": \"integer\", \"description\": \"Right addend.\"}}, " *
          "\"required\": [\"a\", \"b\"]}}}"
    @test_throws ArgumentError ToolRegistry([
        AgentTool(; name="dup", description="", handler=(a, c) -> ""),
        AgentTool(; name="dup", description="", handler=(a, c) -> ""),
    ])
end

@testset "Chapter 36 — tool call parsing" begin
    registry = default_agent_tools(LIFEAI_REPO_ROOT)

    parsed = parse_qwen3_tool_calls(
        "Let me check.\n<tool_call>\n{\"name\": \"add_integers\", \"arguments\": {\"a\": 1, \"b\": 2}}\n</tool_call>",
    )
    @test length(parsed.calls) == 1
    @test isempty(parsed.invalid)
    @test parsed.calls[1].name == "add_integers"
    @test agent_tool_call_validity(registry, parsed) === :valid

    two = parse_qwen3_tool_calls(
        "<tool_call>\n{\"name\": \"add_integers\", \"arguments\": {\"a\": 1, \"b\": 2}}\n</tool_call>\n" *
        "<tool_call>\n{\"name\": \"list_directory\", \"arguments\": {\"path\": \".\"}}\n</tool_call>",
    )
    @test length(two.calls) == 2
    @test agent_tool_call_validity(registry, two) === :valid

    # Some checkpoints emit `arguments` as a JSON string rather than an object.
    stringified = parse_qwen3_tool_calls(
        "<tool_call>\n{\"name\": \"add_integers\", \"arguments\": \"{\\\"a\\\": 1, \\\"b\\\": 2}\"}\n</tool_call>",
    )
    @test length(stringified.calls) == 1
    @test invoke_agent_tool(registry, stringified.calls[1]).output == "3"

    duplicate_fields = parse_qwen3_tool_calls(
        "<tool_call>\n{\"name\": \"nope\", \"name\": \"add_integers\", " *
        "\"arguments\": {\"a\": 0}, \"arguments\": {\"a\": 2, \"b\": 3}}\n" *
        "</tool_call>",
    )
    @test isempty(duplicate_fields.invalid)
    @test only(duplicate_fields.calls).name == "add_integers"
    @test invoke_agent_tool(registry, only(duplicate_fields.calls)).output == "5"

    duplicate_stringified = parse_qwen3_tool_calls(
        "<tool_call>\n{\"name\": \"add_integers\", " *
        "\"arguments\": \"{\\\"a\\\": 1, \\\"a\\\": 4, \\\"b\\\": 2}\"}\n" *
        "</tool_call>",
    )
    @test isempty(duplicate_stringified.invalid)
    @test invoke_agent_tool(
        registry,
        only(duplicate_stringified.calls),
    ).output == "6"

    @test agent_tool_call_validity(registry, parse_qwen3_tool_calls("no tools here")) === :none

    for (text, reason_fragment) in [
        ("<tool_call>\n{not json}\n</tool_call>", "invalid JSON"),
        ("<tool_call>\n[1, 2]\n</tool_call>", "must be a JSON object"),
        ("<tool_call>\n{\"arguments\": {}}\n</tool_call>", "string `name`"),
        ("<tool_call>\n{\"name\": \"add_integers\"}\n</tool_call>", "`arguments` must be a JSON object"),
        ("<tool_call>\n{\"name\": \"x\", \"arguments\": {}}", "unterminated"),
    ]
        malformed = parse_qwen3_tool_calls(text)
        @test length(malformed.invalid) == 1
        @test occursin(reason_fragment, malformed.invalid[1].reason)
        @test agent_tool_call_validity(registry, malformed) === :invalid
    end

    unknown = parse_qwen3_tool_calls("<tool_call>\n{\"name\": \"nope\", \"arguments\": {}}\n</tool_call>")
    @test agent_tool_call_validity(registry, unknown) === :invalid
    @test !invoke_agent_tool(registry, unknown.calls[1]).ok

    incomplete = parse_qwen3_tool_calls(
        "<tool_call>\n{\"name\": \"add_integers\", \"arguments\": {\"a\": 1}}\n</tool_call>",
    )
    @test agent_tool_call_validity(registry, incomplete) === :invalid
    outcome = invoke_agent_tool(registry, incomplete.calls[1])
    @test !outcome.ok
    @test occursin("missing required argument", something(outcome.error, ""))
end

@testset "Chapter 36 — builtin tool handlers" begin
    registry = default_agent_tools(LIFEAI_REPO_ROOT)
    call(text) = only(parse_qwen3_tool_calls(text).calls)

    @test invoke_agent_tool(
        registry,
        call("<tool_call>\n{\"name\": \"add_integers\", \"arguments\": {\"a\": 40, \"b\": 2}}\n</tool_call>"),
    ).output == "42"

    # Models frequently emit integers as strings; the coercion is recorded rather
    # than hidden, so a run can report the strict and the lenient count.
    coerced = invoke_agent_tool(
        registry,
        call("<tool_call>\n{\"name\": \"add_integers\", \"arguments\": {\"a\": \"40\", \"b\": 2}}\n</tool_call>"),
    )
    @test coerced.ok
    @test coerced.output == "42"
    @test coerced.coerced_arguments == ["a"]

    @test !invoke_agent_tool(
        registry,
        call("<tool_call>\n{\"name\": \"add_integers\", \"arguments\": {\"a\": true, \"b\": 2}}\n</tool_call>"),
    ).ok

    for value in (big(typemax(Int)) + 1, big(typemin(Int)) - 1)
        overflow = invoke_agent_tool(
            registry,
            call(
                "<tool_call>\n{\"name\": \"add_integers\", \"arguments\": " *
                "{\"a\": $value, \"b\": 0}}\n</tool_call>",
            ),
        )
        @test !overflow.ok
        @test overflow.error ==
            "ArgumentError: argument \"a\" is outside the host integer range"
        @test isempty(overflow.coerced_arguments)
    end
    for value in (typemin(Int), typemax(Int))
        boundary = invoke_agent_tool(
            registry,
            call(
                "<tool_call>\n{\"name\": \"add_integers\", \"arguments\": " *
                "{\"a\": $value, \"b\": 0}}\n</tool_call>",
            ),
        )
        @test boundary.ok
        @test boundary.output == string(value)
    end

    for (left, right) in ((typemax(Int), 1), (typemin(Int), -1))
        overflow = invoke_agent_tool(
            registry,
            call(
                "<tool_call>\n{\"name\": \"add_integers\", \"arguments\": " *
                "{\"a\": $left, \"b\": $right}}\n</tool_call>",
            ),
        )
        @test !overflow.ok
        @test overflow.error ==
            "ArgumentError: integer addition result is outside the host integer range"
        @test isempty(overflow.coerced_arguments)
    end

    unsigned_arguments = LifeAI.OrderedJSONObject([
        "a" => UInt(typemax(Int)) + UInt(1),
    ])
    unsigned_failure = try
        LifeAI._tool_integer(unsigned_arguments, "a", String[])
        nothing
    catch error
        error
    end
    @test unsigned_failure isa ArgumentError
    @test sprint(showerror, unsigned_failure) ==
        "ArgumentError: argument \"a\" is outside the host integer range"

    listing = invoke_agent_tool(
        registry,
        call("<tool_call>\n{\"name\": \"list_directory\", \"arguments\": {\"path\": \"src\"}}\n</tool_call>"),
    )
    @test listing.ok
    @test "api.jl" in split(listing.output, "\n")

    reading = invoke_agent_tool(
        registry,
        call("<tool_call>\n{\"name\": \"read_text_file\", \"arguments\": {\"path\": \"Project.toml\", \"max_bytes\": 16}}\n</tool_call>"),
    )
    @test reading.ok
    @test startswith(reading.output, "name = \"LifeAI\"")
    @test ncodeunits(reading.output) <= 16

    for escape in ("../etc/passwd", "/etc/passwd", "src/../../..")
        blocked = invoke_agent_tool(
            registry,
            call("<tool_call>\n{\"name\": \"read_text_file\", \"arguments\": {\"path\": \"$escape\"}}\n</tool_call>"),
        )
        @test !blocked.ok
    end

    # A root with a trailing separator must behave identically: the first measured
    # run passed `normpath(joinpath(@__DIR__, ".."))`, whose trailing slash made the
    # prefix test reject every legitimate path.
    for root in (LIFEAI_REPO_ROOT, LIFEAI_REPO_ROOT * "/", LIFEAI_REPO_ROOT * "//")
        trailing = default_agent_tools(root)
        listing = invoke_agent_tool(
            trailing,
            call("<tool_call>\n{\"name\": \"list_directory\", \"arguments\": {\"path\": \"src\"}}\n</tool_call>"),
        )
        @test listing.ok
        @test "api.jl" in split(listing.output, "\n")
        @test !invoke_agent_tool(
            trailing,
            call("<tool_call>\n{\"name\": \"list_directory\", \"arguments\": {\"path\": \"../\"}}\n</tool_call>"),
        ).ok
    end

    for (value, message) in (
        (true, "default_max_bytes must be an integer"),
        (1.0, "default_max_bytes must be an integer"),
        (
            big(typemax(Int)) + 1,
            "default_max_bytes is outside the host integer range",
        ),
        (0, "default_max_bytes must be in 1:1048576"),
        (-1, "default_max_bytes must be in 1:1048576"),
        (1_048_577, "default_max_bytes must be in 1:1048576"),
    )
        failure = try
            LifeAI.read_text_file_tool(
                LIFEAI_REPO_ROOT;
                default_max_bytes=value,
            )
            nothing
        catch error
            error
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end
    @test LifeAI.read_text_file_tool(
        LIFEAI_REPO_ROOT;
        default_max_bytes=Int32(1),
    ) isa AgentTool
    @test LifeAI.read_text_file_tool(
        LIFEAI_REPO_ROOT;
        default_max_bytes=1_048_576,
    ) isa AgentTool
end

@testset "Chapter 36 — assistant content recovered from a generation" begin
    visible = LifeAI._qwen3_visible_assistant_content
    @test visible("Let me compute that.\n<tool_call>\n{\"a\": 1}\n</tool_call>") == "Let me compute that."
    @test visible("<tool_call>\n{\"a\": 1}\n</tool_call>") == ""
    @test visible("<think>\nplan\n</think>\n\nsure\n<tool_call>\n{}\n</tool_call>") ==
          "<think>\nplan\n</think>\n\nsure"
    @test visible("plain answer") == "plain answer"
    @test visible("answer\n\n") == "answer"
    # The template has no slot for text between or after tool calls, so only the
    # prefix is retained; the raw completion keeps the rest.
    @test visible("a\n<tool_call>\n{}\n</tool_call>\nb\n<tool_call>\n{}\n</tool_call>") == "a"
end

@testset "Chapter 36 — Wilson interval for small-sample rates" begin
    interval = wilson_interval(12, 20)
    @test interval.point == 0.6
    @test interval.lower < 0.6 < interval.upper
    # A 20-sample rate carries roughly a ±0.2 band; the chapter reports it so the
    # count is never read as a precise capability number.
    @test interval.upper - interval.lower > 0.35
    @test wilson_interval(20, 20).upper == 1.0
    @test wilson_interval(0, 20).lower == 0.0
    @test_throws ArgumentError wilson_interval(21, 20)
    @test_throws ArgumentError wilson_interval(0, 0)
end
