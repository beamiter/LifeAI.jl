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

@testset "Chapter 36 — loop-step evidence contracts" begin
    function make_step(;
        turn=1,
        prompt="prompt",
        prompt_sha256=nothing,
        prompt_token_count=1,
        generated_ids=[1],
        completion="done",
        stop_reason=:eos,
        validity=:none,
        tool_calls=LifeAI.AgentLoopToolCall[],
        invalid_blocks=NamedTuple{(:raw, :reason),Tuple{String,String}}[],
        prefill_seconds=0.0,
        decode_seconds=0.0,
    )
        digest = prompt_sha256 === nothing ? LifeAI._sha256_hex(prompt) : prompt_sha256
        return AgentLoopStep(
            turn,
            prompt,
            digest,
            prompt_token_count,
            generated_ids,
            completion,
            stop_reason,
            validity,
            tool_calls,
            invalid_blocks,
            prefill_seconds,
            decode_seconds,
        )
    end

    coercion_source = ["a"]
    source_call = LifeAI.AgentLoopToolCall(
        "add_integers",
        "{\"a\":\"1\",\"b\":2}",
        true,
        "3",
        nothing,
        coercion_source,
    )
    id_source = Int32[1, 2]
    call_source = [source_call]
    prompt_source = "prompt!"
    digest_source = LifeAI._sha256_hex("prompt") * "!"
    completion_source = "done!"
    normalized = make_step(
        turn=Int32(1),
        prompt=SubString(prompt_source, 1, 6),
        prompt_sha256=SubString(digest_source, 1, 64),
        prompt_token_count=UInt8(2),
        generated_ids=id_source,
        completion=SubString(completion_source, 1, 4),
        validity=:valid,
        tool_calls=call_source,
        prefill_seconds=Float32(0.25),
        decode_seconds=1 // 2,
    )
    @test normalized.turn === 1
    @test normalized.prompt == "prompt"
    @test normalized.prompt isa String
    @test normalized.prompt_sha256 == LifeAI._sha256_hex("prompt")
    @test normalized.prompt_sha256 isa String
    @test normalized.prompt_token_count === 2
    @test normalized.generated_ids == [1, 2]
    @test normalized.generated_ids isa Vector{Int}
    @test normalized.generated_ids !== id_source
    @test normalized.completion == "done"
    @test normalized.completion isa String
    @test normalized.prefill_seconds === 0.25
    @test normalized.decode_seconds === 0.5
    @test normalized.tool_calls !== call_source
    @test only(normalized.tool_calls).coerced_arguments !== coercion_source
    id_source[1] = 99
    empty!(call_source)
    push!(coercion_source, "late")
    @test normalized.generated_ids == [1, 2]
    @test length(normalized.tool_calls) == 1
    @test only(normalized.tool_calls).coerced_arguments == ["a"]

    invalid_source = [(
        raw=SubString("bad!", 1, 3),
        reason=SubString("broken!", 1, 6),
    )]
    invalid = make_step(
        generated_ids=(),
        completion="",
        stop_reason=:length,
        validity=:invalid,
        invalid_blocks=invalid_source,
    )
    @test only(invalid.invalid_blocks) == (raw="bad", reason="broken")
    @test only(invalid.invalid_blocks).raw isa String
    @test only(invalid.invalid_blocks).reason isa String
    @test invalid.invalid_blocks !== invalid_source
    empty!(invalid_source)
    @test length(invalid.invalid_blocks) == 1

    redacted = make_step(
        prompt="",
        prompt_sha256="a"^64,
        prompt_token_count=42,
    )
    @test redacted.prompt_sha256 == "a"^64
    empty_prompt = make_step(
        prompt="",
        prompt_token_count=0,
        generated_ids=(),
        completion="",
        stop_reason=:length,
    )
    @test empty_prompt.prompt_sha256 == LifeAI._sha256_hex("")
    @test make_step(stop_reason=:stop_token).stop_reason === :stop_token
    @test make_step(
        validity=:invalid,
        tool_calls=[source_call],
    ).validity === :invalid
    failed_call = LifeAI.AgentLoopToolCall(
        "add_integers",
        "{}",
        false,
        "partial",
        "missing arguments",
        (),
    )
    @test make_step(
        validity=:valid,
        tool_calls=[failed_call],
    ).tool_calls[1].ok === false
    @test length(methods(AgentLoopStep)) == 1

    invalid_cases = (
        (
            () -> make_step(turn=true),
            "agent loop step turn must be an integer",
        ),
        (
            () -> make_step(turn=big(typemax(Int)) + 1),
            "agent loop step turn is outside the host integer range",
        ),
        (
            () -> make_step(turn=0),
            "agent loop step turn must be positive",
        ),
        (
            () -> make_step(prompt=42, prompt_sha256="a"^64),
            "agent loop step prompt must be a string",
        ),
        (
            () -> make_step(prompt_sha256=42),
            "agent loop step prompt_sha256 must be a string",
        ),
        (
            () -> make_step(prompt_sha256="A"^64),
            "agent loop step prompt_sha256 must be a lowercase SHA-256 digest",
        ),
        (
            () -> make_step(prompt_sha256="a"^64),
            "agent loop step prompt digest does not match prompt",
        ),
        (
            () -> make_step(
                prompt="",
                prompt_sha256="a"^64,
                prompt_token_count=0,
                generated_ids=(),
                stop_reason=:length,
            ),
            "agent loop step prompt digest does not match prompt",
        ),
        (
            () -> make_step(prompt_token_count=true),
            "agent loop step prompt_token_count must be an integer",
        ),
        (
            () -> make_step(prompt_token_count=-1),
            "agent loop step prompt_token_count must be nonnegative",
        ),
        (
            () -> make_step(prompt_token_count=0),
            "agent loop step nonempty prompt must have a positive token count",
        ),
        (
            () -> make_step(generated_ids=nothing),
            "agent loop step generated_ids must be iterable",
        ),
        (
            () -> make_step(generated_ids=[true]),
            "agent loop step generated id must be an integer",
        ),
        (
            () -> make_step(generated_ids=[big(typemax(Int)) + 1]),
            "agent loop step generated id is outside the host integer range",
        ),
        (
            () -> make_step(generated_ids=[0]),
            "agent loop step generated ids must be positive",
        ),
        (
            () -> make_step(completion=42),
            "agent loop step completion must be a string",
        ),
        (
            () -> make_step(stop_reason="eos"),
            "agent loop step stop_reason must be a Symbol",
        ),
        (
            () -> make_step(stop_reason=:forged),
            "unsupported agent loop step stop_reason: :forged",
        ),
        (
            () -> make_step(generated_ids=(), stop_reason=:eos),
            "agent loop step without generated ids must use :length stop_reason",
        ),
        (
            () -> make_step(validity="none"),
            "agent loop step validity must be a Symbol",
        ),
        (
            () -> make_step(validity=:forged),
            "unsupported agent loop step validity: :forged",
        ),
        (
            () -> make_step(tool_calls=nothing),
            "agent loop step tool_calls must be iterable",
        ),
        (
            () -> make_step(tool_calls=[1]),
            "agent loop step tool_calls must contain AgentLoopToolCall values",
        ),
        (
            () -> make_step(invalid_blocks=nothing),
            "Qwen3 tool-call parse invalid blocks must be iterable",
        ),
        (
            () -> make_step(tool_calls=[source_call]),
            "agent loop step :none validity requires no tool-call evidence",
        ),
        (
            () -> make_step(validity=:valid),
            "agent loop step :valid validity requires calls and no invalid blocks",
        ),
        (
            () -> make_step(
                validity=:valid,
                tool_calls=[source_call],
                invalid_blocks=[(raw="bad", reason="broken")],
            ),
            "agent loop step :valid validity requires calls and no invalid blocks",
        ),
        (
            () -> make_step(validity=:invalid),
            "agent loop step :invalid validity requires tool-call evidence",
        ),
        (
            () -> make_step(prefill_seconds=true),
            "agent loop step prefill_seconds must be a real number",
        ),
        (
            () -> make_step(prefill_seconds=NaN),
            "agent loop step prefill_seconds must be finite and nonnegative",
        ),
        (
            () -> make_step(decode_seconds=Inf),
            "agent loop step decode_seconds must be finite and nonnegative",
        ),
        (
            () -> make_step(decode_seconds=-1),
            "agent loop step decode_seconds must be finite and nonnegative",
        ),
    )
    for (build, message) in invalid_cases
        failure = try
            build()
            nothing
        catch caught
            caught
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end
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

    archived_steps = AgentLoopStep[]
    for step in steps
        calls = LifeAI.AgentLoopToolCall[
            LifeAI.AgentLoopToolCall(
                call.name,
                call.arguments,
                call.ok,
                call.output,
                call.error,
                call.coerced_arguments,
            ) for call in step.tool_calls
        ]
        invalid_blocks = [
            (raw=String(block.raw), reason=String(block.reason))
            for block in step.invalid_blocks
        ]
        push!(archived_steps, AgentLoopStep(
            step.turn,
            "",
            step.prompt_sha256,
            step.prompt_token_count,
            step.generated_ids,
            step.completion,
            Symbol(String(step.stop_reason)),
            Symbol(String(step.validity)),
            calls,
            invalid_blocks,
            step.prefill_seconds,
            step.decode_seconds,
        ))
    end
    archived_calls = [call for step in archived_steps for call in step.tool_calls]
    @test length(archived_steps) == length(steps)
    @test length(archived_calls) == 18
    @test all(call -> call.ok == (call.error === nothing), archived_calls)
    @test any(call -> call.ok, archived_calls)
    @test any(call -> !call.ok, archived_calls)

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
