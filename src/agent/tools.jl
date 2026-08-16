"""
Tool declarations, `<tool_call>` parsing and sandboxed builtin handlers for the
Qwen3 tool-calling protocol.

The declaration side is order preserving on purpose: the official chat template
serializes each tool with CPython `json.dumps(..., ensure_ascii=false)`, so the
schema must keep insertion order to stay byte identical to the reference prompt.
"""

"""One registered tool: an ordered JSON schema plus a pure Julia handler."""
struct AgentTool
    name::String
    description::String
    parameters::Any
    required::Vector{String}
    handler::Any
end

function AgentTool(;
    name::AbstractString,
    description::AbstractString,
    properties=NamedTuple(),
    required=String[],
    handler,
)
    isempty(name) && throw(ArgumentError("tool name must not be empty"))
    required_names = String[String(entry) for entry in required]
    parameters = (; type="object", properties, required=required_names)
    return AgentTool(String(name), String(description), parameters, required_names, handler)
end

"""Ordered collection of tools addressable by name."""
struct ToolRegistry
    tools::Vector{AgentTool}
    by_name::Dict{String,AgentTool}
end

function ToolRegistry(tools)
    ordered = AgentTool[tool for tool in tools]
    by_name = Dict{String,AgentTool}()
    for tool in ordered
        haskey(by_name, tool.name) && throw(ArgumentError(
            "duplicate tool name $(repr(tool.name))",
        ))
        by_name[tool.name] = tool
    end
    return ToolRegistry(ordered, by_name)
end

ToolRegistry(tools::AgentTool...) = ToolRegistry(collect(tools))

Base.length(registry::ToolRegistry) = length(registry.tools)
Base.isempty(registry::ToolRegistry) = isempty(registry.tools)
Base.haskey(registry::ToolRegistry, name::AbstractString) = haskey(registry.by_name, String(name))

"""
    qwen3_tool_specs(registry)

Tool declarations shaped for `apply_qwen3_chat_template(...; tools=...)`.
"""
qwen3_tool_specs(registry::ToolRegistry) = Any[
    (; type="function", var"function"=(;
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    ))
    for tool in registry.tools
]

"""One `<tool_call>` block recovered from generated text."""
struct Qwen3ToolCall
    name::String
    arguments::Any
    raw::String
end

"""Result of scanning generated text for `<tool_call>` blocks."""
struct Qwen3ToolCallParse
    calls::Vector{Qwen3ToolCall}
    invalid::Vector{NamedTuple{(:raw, :reason),Tuple{String,String}}}
end

const _QWEN3_TOOL_CALL_OPEN = "<tool_call>"
const _QWEN3_TOOL_CALL_CLOSE = "</tool_call>"

"""
    parse_qwen3_tool_calls(text)

Recover every `<tool_call>` block from `text`. Blocks whose payload is not a JSON
object with a string `name` and an object `arguments` are reported as invalid
rather than silently dropped, so a caller can measure how often the model emits a
well-formed call.
"""
function parse_qwen3_tool_calls(text::AbstractString)
    calls = Qwen3ToolCall[]
    invalid = NamedTuple{(:raw, :reason),Tuple{String,String}}[]
    cursor = firstindex(text)
    while true
        open_range = findnext(_QWEN3_TOOL_CALL_OPEN, text, cursor)
        open_range === nothing && break
        close_range = findnext(_QWEN3_TOOL_CALL_CLOSE, text, last(open_range) + 1)
        if close_range === nothing
            push!(invalid, (
                raw=String(SubString(text, first(open_range), lastindex(text))),
                reason="unterminated <tool_call> block",
            ))
            break
        end
        payload = strip(String(SubString(text, last(open_range) + 1, prevind(text, first(close_range)))))
        cursor = last(close_range) + 1
        # parse_qwen3_json rather than JSON3.read: the arguments are re-serialized into
        # the next turn's prompt, and JSON3 would collapse `1.0` to `1` on the way.
        parsed = try
            parse_qwen3_json(payload)
        catch error
            push!(invalid, (raw=payload, reason="invalid JSON: $(sprint(showerror, error))"))
            continue
        end
        if !(parsed isa OrderedJSONObject)
            push!(invalid, (raw=payload, reason="tool call payload must be a JSON object"))
            continue
        end
        name = get(parsed, "name", nothing)
        if !(name isa AbstractString) || isempty(name)
            push!(invalid, (raw=payload, reason="tool call is missing a string `name`"))
            continue
        end
        arguments = get(parsed, "arguments", nothing)
        if arguments isa AbstractString
            arguments = try
                parse_qwen3_json(arguments)
            catch error
                push!(invalid, (raw=payload, reason="invalid stringified arguments: $(sprint(showerror, error))"))
                continue
            end
        end
        if !(arguments isa OrderedJSONObject)
            push!(invalid, (raw=payload, reason="tool call `arguments` must be a JSON object"))
            continue
        end
        push!(calls, Qwen3ToolCall(String(name), arguments, payload))
    end
    return Qwen3ToolCallParse(calls, invalid)
end

"""Outcome of running one tool call against a registry."""
struct AgentToolResult
    ok::Bool
    output::String
    error::Union{Nothing,String}
    coerced_arguments::Vector{String}
end

_tool_argument(arguments, name::AbstractString) = get(arguments, Symbol(name), nothing)

"""Read an integer argument, recording when a model emitted it as a string."""
function _tool_integer(arguments, name::AbstractString, coerced::Vector{String})
    value = _tool_argument(arguments, name)
    value isa Bool && throw(ArgumentError("argument $(repr(name)) must be an integer"))
    value isa Integer && return Int(value)
    if value isa AbstractString
        parsed = tryparse(Int, strip(value))
        parsed === nothing && throw(ArgumentError(
            "argument $(repr(name)) is not an integer: $(repr(String(value)))",
        ))
        push!(coerced, String(name))
        return parsed
    end
    throw(ArgumentError("argument $(repr(name)) must be an integer"))
end

function _tool_string(arguments, name::AbstractString)
    value = _tool_argument(arguments, name)
    value isa AbstractString || throw(ArgumentError("argument $(repr(name)) must be a string"))
    return String(value)
end

"""
    invoke_agent_tool(registry, call)

Run one parsed call. Unknown tools and missing required arguments fail closed with
an explanatory message instead of throwing, so a loop can feed the failure back to
the model and still record the step.
"""
function invoke_agent_tool(registry::ToolRegistry, call::Qwen3ToolCall)
    tool = get(registry.by_name, call.name, nothing)
    tool === nothing && return AgentToolResult(
        false,
        "",
        "unknown tool $(repr(call.name)); registered: " * join(sort(collect(keys(registry.by_name))), ", "),
        String[],
    )
    missing_arguments = [
        name for name in tool.required if _tool_argument(call.arguments, name) === nothing
    ]
    isempty(missing_arguments) || return AgentToolResult(
        false,
        "",
        "missing required argument(s): " * join(missing_arguments, ", "),
        String[],
    )
    coerced = String[]
    output = try
        tool.handler(call.arguments, coerced)
    catch error
        return AgentToolResult(false, "", sprint(showerror, error), coerced)
    end
    return AgentToolResult(true, String(output), nothing, coerced)
end

"""
    agent_tool_call_validity(registry, parse)

Mechanical pre-registered verdict for one generation: `:valid` when at least one
block parsed and every parsed block names a registered tool with its required
arguments present, `:invalid` when any block failed, `:none` when no block was
emitted.
"""
function agent_tool_call_validity(registry::ToolRegistry, parse::Qwen3ToolCallParse)
    isempty(parse.invalid) || return :invalid
    isempty(parse.calls) && return :none
    for call in parse.calls
        haskey(registry.by_name, call.name) || return :invalid
        tool = registry.by_name[call.name]
        for name in tool.required
            _tool_argument(call.arguments, name) === nothing && return :invalid
        end
    end
    return :valid
end

# ---------------------------------------------------------------------------
# Builtin tools. Every handler is pure with respect to the process: it either
# computes, or reads inside one frozen sandbox root that the caller supplies.
# ---------------------------------------------------------------------------

"""Integer addition; the cheapest possible end-to-end check of the protocol."""
add_integers_tool() = AgentTool(;
    name="add_integers",
    description="Add two integers and return their sum.",
    properties=(;
        a=(; type="integer", description="Left addend."),
        b=(; type="integer", description="Right addend."),
    ),
    required=["a", "b"],
    handler=(arguments, coerced) -> string(
        _tool_integer(arguments, "a", coerced) + _tool_integer(arguments, "b", coerced),
    ),
)

"""List the immediate subdirectories of a frozen root, newest name order aside."""
function list_directory_tool(root::AbstractString)
    resolved_root = _normalize_sandbox_root(root)
    isdir(resolved_root) || throw(ArgumentError("tool root is not a directory: $resolved_root"))
    return AgentTool(;
        name="list_directory",
        description="List the entries of a directory inside the frozen sandbox root.",
        properties=(;
            path=(; type="string", description="Relative path inside the sandbox root; use \".\" for the root itself."),
        ),
        required=["path"],
        handler=function (arguments, _coerced)
            target = _sandboxed_path(resolved_root, _tool_string(arguments, "path"))
            isdir(target) || throw(ArgumentError("not a directory: $(relpath(target, resolved_root))"))
            entries = sort(readdir(target))
            return isempty(entries) ? "(empty)" : join(entries, "\n")
        end,
    )
end

"""Upper bound on a model-controlled `max_bytes`, so one tool call cannot ask for a
gigabyte of prompt."""
const _AGENT_READ_BYTE_CEILING = 1 << 20

"""Read a bounded UTF-8 prefix of a file inside a frozen root."""
function read_text_file_tool(root::AbstractString; default_max_bytes::Integer=512)
    resolved_root = _normalize_sandbox_root(root)
    isdir(resolved_root) || throw(ArgumentError("tool root is not a directory: $resolved_root"))
    limit_default = Int(default_max_bytes)
    return AgentTool(;
        name="read_text_file",
        description="Read the beginning of a UTF-8 text file inside the frozen sandbox root.",
        properties=(;
            path=(; type="string", description="Relative path inside the sandbox root."),
            max_bytes=(; type="integer", description="Maximum number of bytes to return."),
        ),
        required=["path"],
        handler=function (arguments, coerced)
            target = _sandboxed_path(resolved_root, _tool_string(arguments, "path"))
            isfile(target) || throw(ArgumentError("not a file: $(relpath(target, resolved_root))"))
            limit = _tool_argument(arguments, "max_bytes") === nothing ? limit_default :
                _tool_integer(arguments, "max_bytes", coerced)
            0 < limit <= _AGENT_READ_BYTE_CEILING || throw(ArgumentError(
                "max_bytes must be in 1:$(_AGENT_READ_BYTE_CEILING)",
            ))
            bytes = open(target, "r") do io
                read(io, limit)
            end
            text = String(_truncate_to_valid_utf8(bytes))
            isvalid(text) || throw(ArgumentError("file is not valid UTF-8 text"))
            return text
        end,
    )
end

"""Absolute sandbox root without a trailing separator, so prefix tests are exact."""
function _normalize_sandbox_root(root::AbstractString)
    resolved = normpath(abspath(String(root)))
    while length(resolved) > 1 && endswith(resolved, '/')
        resolved = chop(resolved)
    end
    return resolved
end

"""
Resolve `relative` inside `root`, rejecting anything that leaves the sandbox.

The lexical check alone is not enough: `normpath` does not resolve symbolic links, and
every subsequent `isfile` / `open` / `readdir` follows them, so a link inside the root
pointing outside would be a full read primitive. The target is therefore resolved with
`realpath` and re-checked against the resolved root.
"""
function _sandboxed_path(root::AbstractString, relative::AbstractString)
    isabspath(relative) && throw(ArgumentError("path must be relative to the sandbox root"))
    candidate = normpath(joinpath(root, relative))
    _sandbox_contains(root, candidate) || throw(ArgumentError("path escapes the sandbox root"))
    ispath(candidate) || throw(ArgumentError("no such path: $relative"))
    resolved = realpath(candidate)
    _sandbox_contains(realpath(root), resolved) || throw(ArgumentError(
        "path escapes the sandbox root through a symbolic link",
    ))
    return resolved
end

function _sandbox_contains(root::AbstractString, candidate::AbstractString)
    candidate == root && return true
    return startswith(candidate, endswith(root, '/') ? root : root * "/")
end

"""
Drop a trailing partial UTF-8 sequence left by a byte-count truncation.

Only the last (at most three) continuation bytes are examined, so this stays linear.
Invalid bytes anywhere else are a signal that the file is not UTF-8 text, and the
caller fails closed rather than returning silently mangled content.
"""
function _truncate_to_valid_utf8(bytes::Vector{UInt8})
    isempty(bytes) && return bytes
    start = length(bytes)
    dropped = 0
    while start > 1 && dropped < 3 && (bytes[start] & 0xC0) == 0x80
        start -= 1
        dropped += 1
    end
    leader = bytes[start]
    width = leader < 0x80 ? 1 :
        (leader & 0xE0) == 0xC0 ? 2 :
        (leader & 0xF0) == 0xE0 ? 3 :
        (leader & 0xF8) == 0xF0 ? 4 : 0
    width == 0 && return bytes[1:start - 1]
    return start + width - 1 <= length(bytes) ? bytes : bytes[1:start - 1]
end

"""
    default_agent_tools(sandbox_root)

The builtin registry used by the Chapter 36 loop: integer addition plus two
read-only filesystem tools confined to `sandbox_root`.
"""
default_agent_tools(sandbox_root::AbstractString) = ToolRegistry([
    add_integers_tool(),
    list_directory_tool(sandbox_root),
    read_text_file_tool(sandbox_root),
])
