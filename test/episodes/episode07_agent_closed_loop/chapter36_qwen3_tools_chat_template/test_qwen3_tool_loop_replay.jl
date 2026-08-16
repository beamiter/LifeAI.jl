using Test
using JSON3
using SHA: sha256
using LifeAI
using LifeAI:
    apply_qwen3_chat_template,
    default_agent_tools,
    load_hf_qwen3_tokenizer,
    parse_qwen3_tool_calls,
    qwen3_tool_specs

if !isdefined(@__MODULE__, :LIFEAI_REPO_ROOT)
    @eval const LIFEAI_REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
end

if !isdefined(@__MODULE__, :write_qwen3_tokenizer_fixture)
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tokenizer_fixture.jl"))
end

# Replays the frozen Qwen3-4B run without loading a model. Rendering depends on the
# template text alone, so every recorded prompt digest must be reproducible from the
# task set plus the recorded completions and tool outputs. A change that alters any
# prompt byte — in the template, the tool schema or the history rebuild — breaks this.
@testset "Chapter 36 — frozen tool-loop trace replays" begin
    fixtures = joinpath(@__DIR__, "fixtures")
    template = read(joinpath(fixtures, "official_chat_template.jinja"), String)
    task_set = JSON3.read(read(joinpath(fixtures, "tool_loop_tasks.json"), String))
    trace_path = joinpath(fixtures, "tool_loop_trace.jsonl")
    steps = [JSON3.read(line) for line in eachline(trace_path) if !isempty(strip(line))]
    @test !isempty(steps)

    tools = qwen3_tool_specs(default_agent_tools(LIFEAI_REPO_ROOT))
    system_prompt = String(task_set.system)

    mktempdir() do directory
        payloads = qwen3_tokenizer_fixture_payloads()
        payloads.tokenizer_config["chat_template"] = template
        tokenizer = load_hf_qwen3_tokenizer(write_qwen3_tokenizer_fixture(directory; payloads))

        by_task = Dict{String,Vector{Any}}()
        for step in steps
            push!(get!(by_task, String(step.task), Any[]), step)
        end
        @test length(by_task) == length(task_set.tasks)

        for task in task_set.tasks
            name = String(task.name)
            recorded = sort(by_task[name]; by=step -> step.turn)
            messages = Any[
                Dict{Symbol,Any}(:role => "system", :content => system_prompt),
                Dict{Symbol,Any}(:role => "user", :content => String(task.user)),
            ]
            for (position, step) in enumerate(recorded)
                @test step.turn == position
                prompt = apply_qwen3_chat_template(
                    tokenizer,
                    messages;
                    tools,
                    add_generation_prompt=true,
                    enable_thinking=false,
                )
                @test bytes2hex(sha256(codeunits(prompt))) == String(step.prompt_sha256)

                completion = String(step.completion)
                parsed = parse_qwen3_tool_calls(completion)
                @test length(parsed.calls) == length(step.tool_calls)
                assistant = Dict{Symbol,Any}(
                    :role => "assistant",
                    :content => LifeAI._qwen3_visible_assistant_content(completion),
                )
                isempty(parsed.calls) || (assistant[:tool_calls] = Any[
                    (; type="function", var"function"=(; name=call.name, arguments=call.arguments))
                    for call in parsed.calls
                ])
                push!(messages, assistant)
                for call in step.tool_calls
                    push!(messages, Dict{Symbol,Any}(
                        :role => "tool",
                        :content => call.ok ? String(call.output) :
                                    "error: " * String(something(call.error, "tool failed")),
                    ))
                end
            end
        end
    end
end
