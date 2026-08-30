using Test
using JSON3
using LifeAI
using LifeAI:
    AgentTool,
    AgentToolResult,
    OrderedJSONObject,
    Qwen3ToolCall,
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

    declared = AgentTool(;
        name="dangerous",
        description="Require confirmation before execution.",
        properties=(;
            confirm=(; type="boolean", description="Authorize execution."),
        ),
        required=["confirm"],
        handler=(_arguments, _coerced) -> "ran",
    )
    @test declared.parameters.required === declared.required
    declared_fields = ntuple(
        index -> getfield(declared, index),
        fieldcount(AgentTool),
    )
    @test_throws MethodError AgentTool(declared_fields...)
    @test_throws MethodError AgentTool(
        "dangerous",
        "Advertises confirmation but does not enforce it.",
        declared.parameters,
        String[],
        declared.handler,
    )
    @test_throws MethodError AgentTool(
        "",
        "Empty raw name",
        (; type="object", properties=(;), required=String[]),
        String[],
        declared.handler,
    )
    @test_throws ArgumentError ToolRegistry([
        AgentTool(; name="dup", description="", handler=(a, c) -> ""),
        AgentTool(; name="dup", description="", handler=(a, c) -> ""),
    ])

    advertised = AgentTool(;
        name="advertised",
        description="Visible tool",
        handler=(_arguments, _coerced) -> "advertised ran",
    )
    hidden = AgentTool(;
        name="hidden",
        description="Undeclared tool",
        handler=(_arguments, _coerced) -> "hidden ran",
    )
    @test_throws MethodError ToolRegistry(
        AgentTool[advertised],
        Dict{String,AgentTool}("hidden" => hidden),
    )
    canonical = ToolRegistry(advertised)
    @test haskey(canonical, "advertised")
    @test !haskey(canonical, "hidden")
    @test_throws MethodError ToolRegistry(canonical.tools, canonical.by_name)
    @test isempty(ToolRegistry())

    canonical.by_name["hidden"] = hidden
    @test_throws ArgumentError haskey(canonical, "hidden")
    @test_throws ArgumentError qwen3_tool_specs(canonical)
    hidden_parse = parse_qwen3_tool_calls(
        "<tool_call>{\"name\":\"hidden\",\"arguments\":{}}</tool_call>",
    )
    hidden_call = only(hidden_parse.calls)
    @test_throws ArgumentError agent_tool_call_validity(
        canonical,
        hidden_parse,
    )
    @test_throws ArgumentError invoke_agent_tool(canonical, hidden_call)
end

@testset "Chapter 36 — tool call parsing" begin
    registry = default_agent_tools(LIFEAI_REPO_ROOT)

    name_source = "direct"
    raw_source = "{\"name\":\"direct\",\"arguments\":{\"a\":1}}"
    arguments = OrderedJSONObject(["a" => 1])
    direct = Qwen3ToolCall(
        SubString(name_source, 1, 6),
        arguments,
        SubString(raw_source, 1, lastindex(raw_source)),
    )
    @test direct.name == "direct"
    @test direct.name isa String
    @test direct.raw == raw_source
    @test direct.raw isa String
    @test direct.arguments !== arguments
    @test direct.arguments.entries !== arguments.entries
    arguments.entries[1] = "a" => 99
    @test direct.arguments["a"] == 1

    invalid_direct_calls = (
        (
            ("", OrderedJSONObject(), "{}"),
            "Qwen3 tool call name must not be empty",
        ),
        (
            (42, OrderedJSONObject(), "{}"),
            "Qwen3 tool call name must be a string",
        ),
        (
            ("add_integers", nothing, "forged"),
            "Qwen3 tool call arguments must be an OrderedJSONObject",
        ),
        (
            ("ping", 42, "forged"),
            "Qwen3 tool call arguments must be an OrderedJSONObject",
        ),
        (
            ("ping", Any[], "forged"),
            "Qwen3 tool call arguments must be an OrderedJSONObject",
        ),
        (
            ("ping", "{}", "forged"),
            "Qwen3 tool call arguments must be an OrderedJSONObject",
        ),
        (
            ("ping", (;), "forged"),
            "Qwen3 tool call arguments must be an OrderedJSONObject",
        ),
        (
            ("ping", OrderedJSONObject(), 42),
            "Qwen3 tool call raw payload must be a string",
        ),
    )
    for (arguments, message) in invalid_direct_calls
        failure = try
            Qwen3ToolCall(arguments...)
            nothing
        catch caught
            caught
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end

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

    output_source = "success"
    coercion_source = ["a"]
    direct_success = AgentToolResult(
        true,
        SubString(output_source, 1, 7),
        nothing,
        coercion_source,
    )
    @test direct_success.output == "success"
    @test direct_success.output isa String
    @test direct_success.coerced_arguments == ["a"]
    @test direct_success.coerced_arguments !== coercion_source
    push!(coercion_source, "late")
    @test direct_success.coerced_arguments == ["a"]

    error_source = "failed"
    direct_failure = AgentToolResult(
        false,
        "partial output",
        SubString(error_source, 1, 6),
        (),
    )
    @test !direct_failure.ok
    @test direct_failure.error == "failed"
    @test direct_failure.error isa String

    invalid_results = (
        (
            (true, "ran", "failure", String[]),
            "agent tool result success and error state are inconsistent",
        ),
        (
            (false, "", nothing, String[]),
            "agent tool result success and error state are inconsistent",
        ),
        (
            (1, "", nothing, String[]),
            "agent tool result ok must be Bool",
        ),
        (
            (true, 1, nothing, String[]),
            "agent tool result output must be a string",
        ),
        (
            (false, "", 42, String[]),
            "agent tool result error must be a string or nothing",
        ),
        (
            (true, "", nothing, nothing),
            "agent tool result coerced_arguments must be iterable",
        ),
        (
            (true, "", nothing, Any["a", 1]),
            "agent tool result coerced arguments must be strings",
        ),
    )
    for (arguments, message) in invalid_results
        failure = try
            AgentToolResult(arguments...)
            nothing
        catch caught
            caught
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end

    retained_coercions = Ref{Vector{String}}()
    alias_tool = AgentTool(;
        name="alias_probe",
        description="Retain the handler coercion buffer.",
        handler=(_arguments, coerced) -> begin
            retained_coercions[] = coerced
            push!(coerced, "during_handler")
            return "ran"
        end,
    )
    alias_outcome = invoke_agent_tool(
        ToolRegistry(alias_tool),
        Qwen3ToolCall("alias_probe", OrderedJSONObject(), "{}"),
    )
    @test alias_outcome.ok
    @test alias_outcome.coerced_arguments == ["during_handler"]
    @test alias_outcome.coerced_arguments !== retained_coercions[]
    push!(retained_coercions[], "after_return")
    @test alias_outcome.coerced_arguments == ["during_handler"]

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
