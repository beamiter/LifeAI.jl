"""
Minimal observe → decide → call → feed-back loop over a resident Qwen3 BF16 session.

The loop owns no model state of its own: every turn re-renders the whole message
list through the official chat template, so the prompt at turn `n` is exactly the
prompt HuggingFace would build from the same history. Each step is recorded with
the prompt digest and the raw generated ids, which is what makes a run replayable.
"""

"""One tool invocation inside a loop step."""
struct AgentLoopToolCall
    name::String
    arguments_json::String
    ok::Bool
    output::String
    error::Union{Nothing,String}
    coerced_arguments::Vector{String}
end

"""One model turn: the prompt it saw, what it produced and what that triggered."""
struct AgentLoopStep
    turn::Int
    prompt::String
    prompt_sha256::String
    prompt_token_count::Int
    generated_ids::Vector{Int}
    completion::String
    stop_reason::Symbol
    validity::Symbol
    tool_calls::Vector{AgentLoopToolCall}
    invalid_blocks::Vector{NamedTuple{(:raw, :reason),Tuple{String,String}}}
    prefill_seconds::Float64
    decode_seconds::Float64
end

"""Full trace of one loop run."""
struct AgentLoopTrace
    steps::Vector{AgentLoopStep}
    messages::Vector{Any}
    answer::String
    stop_reason::Symbol
    memory_context::Union{Nothing,AgentMemoryContext}
end

# Preserve the Chapter 36 constructor for callers that materialize a trace without
# retrieval evidence.
AgentLoopTrace(steps, messages, answer, stop_reason) =
    AgentLoopTrace(steps, messages, answer, stop_reason, nothing)

"""
Assistant content to retain in history, given one raw generation.

The official template emits all visible content first and only then the
`<tool_call>` blocks, so text interleaved between or after calls has no
representation in a re-rendered prompt. Keeping the prefix before the first call
is therefore the only choice that round-trips exactly for the shape Qwen3 actually
produces; anything after the first call is dropped here and survives only in the
recorded raw completion.
"""
function _qwen3_visible_assistant_content(text::AbstractString)
    open_range = findfirst(_QWEN3_TOOL_CALL_OPEN, text)
    prefix = open_range === nothing ? text :
        SubString(text, firstindex(text), prevind(text, first(open_range)))
    return _strip_newlines(String(prefix); left=false)
end

_agent_message(role::AbstractString, content::AbstractString) =
    Dict{Symbol,Any}(:role => String(role), :content => String(content))

function _agent_initial_messages(
    user_content::AbstractString;
    system::Union{Nothing,AbstractString}=nothing,
    memory_context::Union{Nothing,AgentMemoryContext}=nothing,
)
    messages = Any[]
    system_content = system === nothing ? nothing : String(system)
    if memory_context !== nothing
        system_content = system_content === nothing ? memory_context.rendered :
            system_content * "\n\n" * memory_context.rendered
    end
    system_content === nothing || push!(messages, _agent_message("system", system_content))
    push!(messages, _agent_message("user", user_content))
    return messages
end

function _agent_assistant_message(content::AbstractString, calls::Vector{Qwen3ToolCall})
    message = _agent_message("assistant", content)
    isempty(calls) && return message
    message[:tool_calls] = Any[
        (; type="function", var"function"=(; name=call.name, arguments=call.arguments))
        for call in calls
    ]
    return message
end

"""
    _run_qwen3_agent_loop(session, registry, user_content; kwargs...)

Run the loop until the model answers without calling a tool, `max_steps` model
turns have been spent, or the prompt no longer fits. Returns an [`AgentLoopTrace`](@ref).

The loop is deterministic under `strategy=:greedy`: the same session, registry and
`user_content` produce the same prompt digests and the same generated ids.
"""
function _run_qwen3_agent_loop(
    session::HFQwen3BF16Session,
    registry::ToolRegistry,
    user_content::AbstractString;
    system::Union{Nothing,AbstractString}=nothing,
    memory_context::Union{Nothing,AgentMemoryContext}=nothing,
    max_steps::Integer=4,
    max_new_tokens::Integer=256,
    enable_thinking::Bool=false,
    strategy::Symbol=:greedy,
    stop_token_ids=nothing,
    on_step=nothing,
)
    steps = Int(max_steps)
    steps > 0 || throw(ArgumentError("max_steps must be positive"))
    tools = isempty(registry) ? nothing : qwen3_tool_specs(registry)
    messages = _agent_initial_messages(user_content; system, memory_context)

    trace = AgentLoopStep[]
    answer = ""
    stop_reason = :max_steps
    for turn in 1:steps
        prompt = apply_qwen3_chat_template(
            session.tokenizer,
            messages;
            tools,
            add_generation_prompt=true,
            enable_thinking,
        )
        prompt_ids = encode(session.tokenizer, prompt; add_special_tokens=false)
        result = generate_hf_qwen3_bf16!(
            session,
            prompt_ids;
            max_new_tokens,
            strategy,
            stop_token_ids,
        )
        parse = parse_qwen3_tool_calls(result.completion)
        validity = agent_tool_call_validity(registry, parse)
        visible = _qwen3_visible_assistant_content(result.completion)

        executed = AgentLoopToolCall[]
        for call in parse.calls
            outcome = invoke_agent_tool(registry, call)
            # Model output must never abort the run: record what cannot be rendered
            # instead of letting the serializer throw out of the loop.
            arguments_json = try
                _python_json_text(call.arguments)
            catch error
                "<unrenderable: " * sprint(showerror, error) * ">"
            end
            push!(executed, AgentLoopToolCall(
                call.name,
                arguments_json,
                outcome.ok,
                outcome.output,
                outcome.error,
                outcome.coerced_arguments,
            ))
        end

        step = AgentLoopStep(
            turn,
            prompt,
            _sha256_hex(prompt),
            length(prompt_ids),
            collect(Int, result.generated_ids),
            result.completion,
            result.stop_reason,
            validity,
            executed,
            parse.invalid,
            result.prefill_seconds,
            result.decode_seconds,
        )
        push!(trace, step)
        on_step === nothing || on_step(step)

        push!(messages, _agent_assistant_message(visible, parse.calls))
        if isempty(parse.calls)
            answer = visible
            stop_reason = isempty(parse.invalid) ? :answered : :invalid_tool_call
            break
        end
        for call in executed
            push!(messages, _agent_message(
                "tool",
                call.ok ? call.output : "error: " * something(call.error, "tool failed"),
            ))
        end
    end
    return AgentLoopTrace(trace, messages, answer, stop_reason, memory_context)
end

"""
    run_qwen3_tool_loop(session, registry, user_content; kwargs...)

Run the Chapter 36 tool loop. The registry must remain non-empty so an accidental
empty declaration cannot silently turn a tool experiment into a plain chat run.
"""
function run_qwen3_tool_loop(
    session::HFQwen3BF16Session,
    registry::ToolRegistry,
    user_content::AbstractString;
    kwargs...,
)
    isempty(registry) && throw(ArgumentError("tool registry must not be empty"))
    return _run_qwen3_agent_loop(session, registry, user_content; kwargs...)
end

"""
    run_qwen3_memory_loop(session, user_content, memory_context;
                          registry=ToolRegistry(), kwargs...)

Run a request with a frozen retrieval context. Tools are optional; an empty
registry renders an ordinary Qwen3 chat prompt without a `# Tools` header. The
context query, hit IDs/scores, rendered bytes, and digests are retained in the
returned trace.
"""
function run_qwen3_memory_loop(
    session::HFQwen3BF16Session,
    user_content::AbstractString,
    memory_context::AgentMemoryContext;
    registry::ToolRegistry=ToolRegistry(),
    kwargs...,
)
    return _run_qwen3_agent_loop(
        session,
        registry,
        user_content;
        memory_context,
        kwargs...,
    )
end

"""
    agent_loop_summary(trace)

Machine-readable summary of one run, including the pre-registered validity verdict
of the first model turn.
"""
function agent_loop_summary(trace::AgentLoopTrace)
    first_validity = isempty(trace.steps) ? :none : trace.steps[1].validity
    executed = sum(length(step.tool_calls) for step in trace.steps; init=0)
    succeeded = sum(
        count(call -> call.ok, step.tool_calls) for step in trace.steps; init=0
    )
    memory = trace.memory_context
    return (;
        turns=length(trace.steps),
        first_turn_validity=first_validity,
        stop_reason=trace.stop_reason,
        tool_calls_executed=executed,
        tool_calls_succeeded=succeeded,
        prompt_sha256=[step.prompt_sha256 for step in trace.steps],
        generated_token_counts=[length(step.generated_ids) for step in trace.steps],
        memory_query_sha256=memory === nothing ? nothing : memory.query_sha256,
        memory_store_sha256=memory === nothing ? nothing : memory.store_sha256,
        memory_context_sha256=memory === nothing ? nothing : memory.rendered_sha256,
        memory_ids=memory === nothing ? String[] : [hit.id for hit in memory.hits],
        memory_scores=memory === nothing ? Float32[] : [hit.score for hit in memory.hits],
    )
end

"""
    wilson_interval(successes, trials; z=1.96)

Wilson score interval. Reported next to every small-sample rate in Chapter 36 so a
`X/20` count is never read as a precise capability number.
"""
function wilson_interval(successes::Integer, trials::Integer; z::Real=1.96)
    trials > 0 || throw(ArgumentError("trials must be positive"))
    0 <= successes <= trials || throw(ArgumentError("successes must be in 0:trials"))
    n = float(trials)
    phat = successes / n
    denominator = 1 + z^2 / n
    center = (phat + z^2 / (2n)) / denominator
    margin = z * sqrt(phat * (1 - phat) / n + z^2 / (4n^2)) / denominator
    return (; point=phat, lower=max(0.0, center - margin), upper=min(1.0, center + margin))
end
