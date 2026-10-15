# CPU-only helpers shared by the CLI and its regression tests.
struct Qwen3ChatInputError <: Exception
    message::String
end

Base.showerror(io::IO, error::Qwen3ChatInputError) = print(io, error.message)

function fit_chat_context(
    session,
    messages;
    max_prompt_tokens::Integer,
    enable_thinking::Bool,
)
    limit = Int(max_prompt_tokens)
    0 < limit <= session.context_tokens || error(
        "max prompt must be in 1:session context",
    )
    retained = collect(messages)
    isempty(retained) && error("chat messages must not be empty")
    dropped = 0
    while true
        prompt = apply_qwen3_chat_template(
            session.tokenizer,
            retained;
            add_generation_prompt=true,
            enable_thinking,
        )
        prompt_ids = encode(
            session.tokenizer,
            prompt;
            add_special_tokens=false,
        )
        length(prompt_ids) <= limit && return (;
            messages=retained,
            prompt,
            prompt_ids,
            dropped_messages=dropped,
        )

        protected_prefix =
            String(getproperty(first(retained), :role)) == "system" ? 1 : 0
        first_droppable = protected_prefix + 1
        first_droppable < length(retained) || throw(Qwen3ChatInputError(
            "newest chat message cannot fit within max prompt",
        ))
        remove_count = 1
        role = String(getproperty(retained[first_droppable], :role))
        if role == "user" && first_droppable + 1 < length(retained)
            next_role = String(
                getproperty(retained[first_droppable + 1], :role),
            )
            next_role == "assistant" && (remove_count = 2)
        end
        deleteat!(
            retained,
            first_droppable:(first_droppable + remove_count - 1),
        )
        dropped += remove_count
    end
end

# Only rejected input is recoverable here. Runtime failures and interrupts
# propagate to the caller rather than pretending a partially run request succeeded.
function run_qwen3_chat_turn!(history, line, runner; error_io=stderr)
    candidate = copy(history)
    push!(candidate, (role="user", content=String(line)))
    result = try
        runner(candidate)
    catch error
        error isa Qwen3ChatInputError || rethrow()
        println(error_io, "request rejected: ", sprint(showerror, error))
        return nothing
    end
    empty!(history)
    append!(history, result.messages)
    push!(history, (role="assistant", content=result.completion))
    return result
end
