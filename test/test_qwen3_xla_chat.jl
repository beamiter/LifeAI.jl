using Test

# Import only CPU helpers: no LifeAI/Reactant import, model load, or GPU work.
module Qwen3XLAChatRegression
include(joinpath(@__DIR__, "..", "scripts", "qwen3_xla_chat_helpers.jl"))

apply_qwen3_chat_template(tokenizer, messages; kwargs...) =
    join(message.content for message in messages)
encode(tokenizer, prompt; kwargs...) = collect(codeunits(prompt))
end

@testset "Qwen3 XLA chat rejects oversized input without losing history" begin
    chat = Qwen3XLAChatRegression
    session = (context_tokens=16, tokenizer=nothing)
    history = [(role="system", content="s"),
               (role="user", content="old"),
               (role="assistant", content="ok")]
    original = copy(history)
    generation_calls = Ref(0)
    runner = function (messages)
        fitted = chat.fit_chat_context(session, messages;
            max_prompt_tokens=16, enable_thinking=false)
        generation_calls[] += 1
        return (messages=fitted.messages, completion="reply")
    end
    errors = IOBuffer()

    # Fitting first discards candidate turns, then discovers the newest input
    # cannot fit. Neither action may mutate the conversation being retained.
    @test chat.run_qwen3_chat_turn!(history, repeat("x", 17), runner;
        error_io=errors) === nothing
    @test history == original
    @test generation_calls[] == 0
    @test occursin("newest chat message cannot fit", String(take!(errors)))

    result = chat.run_qwen3_chat_turn!(history, "hi", runner; error_io=errors)
    @test result.completion == "reply"
    @test generation_calls[] == 1
    @test history == vcat(original,
        [(role="user", content="hi"), (role="assistant", content="reply")])
    @test isempty(take!(errors))
end

@testset "Qwen3 XLA chat commits successful compaction only" begin
    chat = Qwen3XLAChatRegression
    history = [(role="system", content="s"),
               (role="user", content="old"),
               (role="assistant", content="ok")]
    runner = function (messages)
        fitted = chat.fit_chat_context((context_tokens=8, tokenizer=nothing), messages;
            max_prompt_tokens=8, enable_thinking=false)
        return (messages=fitted.messages, completion="new")
    end
    chat.run_qwen3_chat_turn!(history, "hello", runner)
    @test history == [(role="system", content="s"),
                      (role="user", content="hello"),
                      (role="assistant", content="new")]
end

@testset "Qwen3 XLA chat does not swallow runtime failures or interrupts" begin
    chat = Qwen3XLAChatRegression
    history = [(role="system", content="s")]
    original = copy(history)
    for failure in (ErrorException("backend failure"), ArgumentError("internal contract"),
                    InterruptException())
        observed = try
            chat.run_qwen3_chat_turn!(history, "hi", messages -> throw(failure))
            nothing
        catch error
            error
        end
        @test observed === failure
        @test history == original
    end
end
