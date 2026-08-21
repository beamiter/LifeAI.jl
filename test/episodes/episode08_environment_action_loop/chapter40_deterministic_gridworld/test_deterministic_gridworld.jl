using Test
using JSON3
using LifeAI
using LifeAI:
    AgentEnvironmentAction,
    AgentEnvironmentTrace,
    AgentLoopStep,
    AgentLoopToolCall,
    AgentLoopTrace,
    ToolRegistry,
    _agent_assistant_message,
    _agent_initial_messages,
    _agent_message,
    _python_json_text,
    _qwen3_visible_assistant_content,
    apply_qwen3_chat_template,
    encode,
    invoke_agent_tool,
    load_hf_qwen3_tokenizer,
    parse_qwen3_tool_calls,
    qwen3_tool_specs

if !isdefined(@__MODULE__, :write_qwen3_tokenizer_fixture)
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tokenizer_fixture.jl"))
end

const CHAPTER40_TASKS = joinpath(@__DIR__, "fixtures", "gridworld_tasks.json")

@testset "Chapter 40 — frozen GridWorld task and BFS oracle contract" begin
    task_set = load_gridworld_tasks(CHAPTER40_TASKS)
    @test length(task_set.tasks) == 8
    @test task_set.coordinate_system == "origin_top_left_x_east_y_south"
    @test task_set.sha256 ==
          "037ec76c95aa24d1429de11bf6d9cd988c0b056cd177f9c5354bc37d37a5377f"
    @test [task.shortest_steps for task in task_set.tasks] == [4, 4, 4, 4, 6, 6, 8, 8]
    @test length(unique(task.spec.sha256 for task in task_set.tasks)) == 8

    oracle_rows = Any[]
    for task in task_set.tasks
        actions = gridworld_shortest_actions(task.spec)
        @test length(actions) == task.shortest_steps
        replay = replay_gridworld_actions(task.spec, actions)
        @test replay.environment.success
        @test replay.environment.terminal
        @test length(replay.environment.transitions) == task.shortest_steps
        @test all(transition.accepted for transition in replay.environment.transitions)
        push!(oracle_rows, (;
            id=task.id,
            success=true,
            terminal=true,
            actions=length(actions),
            invalid_actions=0,
        ))
    end
    report = gridworld_task_report(task_set, oracle_rows)
    @test report.successes == report.total == 8
    @test report.success_rate == 1.0
    @test report.invalid_actions == 0
    @test report.mean_excess_actions == 0.0

    mktempdir() do directory
        payload = JSON3.read(read(CHAPTER40_TASKS, String), Dict{String,Any})
        function rejected(name, mutate)
            changed = deepcopy(payload)
            mutate(changed)
            path = joinpath(directory, name)
            write(path, JSON3.write(changed))
            @test_throws ArgumentError load_gridworld_tasks(path)
        end
        rejected("version.json", data -> data["schema_version"] = 2)
        rejected("coordinates.json", data -> data["coordinate_system"] = "bottom-left")
        rejected("shortest.json", data -> data["tasks"][1]["shortest_steps"] = 3)
        rejected("outside.json", data -> data["tasks"][1]["walls"] = [[9, 9]])
        rejected("blocked-goal.json", data -> data["tasks"][1]["walls"] = [[3, 1]])
        rejected("duplicate.json", data -> data["tasks"][2]["id"] = data["tasks"][1]["id"])
        rejected("field.json", data -> data["tasks"][1]["extra"] = true)
    end
end

@testset "Chapter 40 — deterministic observation/action/transition safety" begin
    task = first(load_gridworld_tasks(CHAPTER40_TASKS).tasks)
    environment = GridWorldEnvironment(task.spec)
    initial = observe_agent_environment(environment)
    @test initial.step == 0
    @test !initial.terminal
    @test gridworld_legal_actions(environment) == ["south"]
    @test initial == observe_agent_environment(GridWorldEnvironment(task.spec))

    blocked = step_agent_environment!(
        environment,
        AgentEnvironmentAction("move", "east"),
    )
    @test !blocked.accepted
    @test blocked.reason == "blocked"
    @test environment.position == task.spec.start
    @test environment.actions == 1
    @test blocked.before_state_sha256 != blocked.after_state_sha256
    @test blocked.sha256 == only(environment.transitions).sha256
    full = render_agent_environment_feedback(blocked; feedback=:full)
    withheld = render_agent_environment_feedback(blocked; feedback=:none)
    @test occursin("\"position\"", full)
    @test occursin("\"legal_actions\"", full)
    @test !occursin("position", withheld)
    @test !occursin("legal_actions", withheld)
    @test occursin("\"feedback\":\"withheld\"", withheld)
    @test_throws ArgumentError render_agent_environment_feedback(blocked; feedback=:bad)

    reset_agent_environment!(environment)
    @test observe_agent_environment(environment).rendered_sha256 == initial.rendered_sha256
    for direction in gridworld_shortest_actions(task.spec)
        step_agent_environment!(environment, AgentEnvironmentAction("move", direction))
    end
    @test environment.success
    @test environment.terminal
    @test isempty(gridworld_legal_actions(environment))
    @test_throws ArgumentError step_agent_environment!(
        environment,
        AgentEnvironmentAction("move", "north"),
    )

    invalid = GridWorldEnvironment(task.spec)
    before = observe_agent_environment(invalid).state_sha256
    @test_throws ArgumentError step_agent_environment!(
        invalid,
        AgentEnvironmentAction("move", "teleport"),
    )
    @test invalid.actions == 0
    @test isempty(invalid.transitions)
    @test observe_agent_environment(invalid).state_sha256 == before

    limited_spec = GridWorldSpec(;
        id="grid/limited",
        width=3,
        height=2,
        start=(1, 1),
        goal=(3, 1),
        walls=[(2, 1)],
        max_actions=2,
    )
    limited = GridWorldEnvironment(limited_spec)
    step_agent_environment!(limited, AgentEnvironmentAction("move", "east"))
    final = step_agent_environment!(limited, AgentEnvironmentAction("move", "east"))
    @test final.observation.terminal
    @test !final.observation.success
end

@testset "Chapter 40 — move tool is the only state mutation surface" begin
    task = first(load_gridworld_tasks(CHAPTER40_TASKS).tasks)
    environment = GridWorldEnvironment(task.spec)
    registry = gridworld_tool_registry(environment)
    @test length(registry) == 1
    @test haskey(registry, "move")
    spec = only(qwen3_tool_specs(registry))
    @test spec.function.parameters.properties.direction.enum ==
          ["north", "east", "south", "west"]

    function call(payload)
        parsed = parse_qwen3_tool_calls(
            "<tool_call>\n" * JSON3.write(payload) * "\n</tool_call>",
        )
        @test isempty(parsed.invalid)
        return only(parsed.calls)
    end
    moved = invoke_agent_tool(registry, call((; name="move", arguments=(; direction="south"))))
    @test moved.ok
    @test environment.position == (1, 2)
    extra = invoke_agent_tool(
        registry,
        call((; name="move", arguments=(; direction="east", extra="ignored?"))),
    )
    @test !extra.ok
    @test occursin("exactly one", something(extra.error, ""))
    @test environment.position == (1, 2)
    unknown = invoke_agent_tool(
        registry,
        call((; name="move", arguments=(; direction="teleport"))),
    )
    @test !unknown.ok
    @test environment.actions == 1
end

function _scripted_agent_environment_trace(task, tokenizer; feedback=:full)
    environment = GridWorldEnvironment(task.spec)
    initial = observe_agent_environment(environment)
    registry = gridworld_tool_registry(environment; feedback)
    tools = qwen3_tool_specs(registry)
    messages = _agent_initial_messages(
        gridworld_user_prompt(initial);
        system=gridworld_system_prompt(),
    )
    steps = AgentLoopStep[]
    turn = 0
    for direction in gridworld_shortest_actions(task.spec)
        turn += 1
        prompt = apply_qwen3_chat_template(
            tokenizer,
            messages;
            tools,
            add_generation_prompt=true,
            enable_thinking=false,
        )
        prompt_ids = encode(tokenizer, prompt; add_special_tokens=false)
        completion = "<tool_call>\n{\"name\": \"move\", \"arguments\": {\"direction\": \"$direction\"}}\n</tool_call>"
        parsed = parse_qwen3_tool_calls(completion)
        call = only(parsed.calls)
        outcome = invoke_agent_tool(registry, call)
        executed = AgentLoopToolCall[
            AgentLoopToolCall(
                call.name,
                _python_json_text(call.arguments),
                outcome.ok,
                outcome.output,
                outcome.error,
                outcome.coerced_arguments,
            ),
        ]
        push!(steps, AgentLoopStep(
            turn,
            prompt,
            LifeAI._sha256_hex(prompt),
            length(prompt_ids),
            [turn],
            completion,
            :stop_token,
            :valid,
            executed,
            NamedTuple{(:raw, :reason),Tuple{String,String}}[],
            0.0,
            0.0,
        ))
        push!(messages, _agent_assistant_message("", parsed.calls))
        push!(messages, _agent_message("tool", outcome.output))
    end

    turn += 1
    prompt = apply_qwen3_chat_template(
        tokenizer,
        messages;
        tools,
        add_generation_prompt=true,
        enable_thinking=false,
    )
    prompt_ids = encode(tokenizer, prompt; add_special_tokens=false)
    completion = "Goal reached."
    push!(steps, AgentLoopStep(
        turn,
        prompt,
        LifeAI._sha256_hex(prompt),
        length(prompt_ids),
        [turn],
        completion,
        :stop_token,
        :none,
        AgentLoopToolCall[],
        NamedTuple{(:raw, :reason),Tuple{String,String}}[],
        0.0,
        0.0,
    ))
    push!(messages, _agent_assistant_message(completion, Qwen3ToolCall[]))
    agent = AgentLoopTrace(steps, messages, completion, :answered)
    final = observe_agent_environment(environment)
    return AgentEnvironmentTrace(
        "deterministic_gridworld",
        task.id,
        task.spec.sha256,
        feedback,
        initial,
        copy(environment.transitions),
        agent,
        environment.terminal,
        environment.success,
        final.state_sha256,
    )
end

@testset "Chapter 40 — model-free prompt and environment replay" begin
    task = first(load_gridworld_tasks(CHAPTER40_TASKS).tasks)
    template_path = joinpath(
        @__DIR__,
        "..",
        "..",
        "episode07_agent_closed_loop",
        "chapter36_qwen3_tools_chat_template",
        "fixtures",
        "official_chat_template.jinja",
    )
    mktempdir() do directory
        payloads = qwen3_tokenizer_fixture_payloads()
        payloads.tokenizer_config["chat_template"] = read(template_path, String)
        tokenizer = load_hf_qwen3_tokenizer(
            write_qwen3_tokenizer_fixture(directory; payloads),
        )
        for feedback in (:full, :none)
            trace = _scripted_agent_environment_trace(task, tokenizer; feedback)
            @test trace.success
            payload = agent_environment_trace_payload(trace)
            row = JSON3.read(JSON3.write(payload))
            report = replay_qwen3_environment_trace(tokenizer, task.spec, row)
            @test report.prompt_hashes_equal == 5
            @test report.prompt_tokens_equal == 5
            @test report.generated_ids_present == 5
            @test report.tool_outcomes_equal == 4
            @test report.transition_hashes_equal == 4
            @test report.final_state_equal
            @test report.success

            legacy_fields = Base.structdiff(payload, (;
                schema_version=nothing,
                system_prompt=nothing,
                system_prompt_sha256=nothing,
                memory=nothing,
            ))
            legacy = (; schema_version=1, legacy_fields...)
            legacy_report = replay_qwen3_environment_trace(
                tokenizer,
                task.spec,
                JSON3.read(JSON3.write(legacy)),
            )
            @test legacy_report.final_state_equal
            @test legacy_report.success

            changed = JSON3.read(JSON3.write(payload), Dict{String,Any})
            changed["agent_steps"][1]["prompt_sha256"] = "0"^64
            @test_throws ErrorException replay_qwen3_environment_trace(
                tokenizer,
                task.spec,
                JSON3.read(JSON3.write(changed)),
            )
            changed_transition = JSON3.read(JSON3.write(payload), Dict{String,Any})
            changed_transition["transitions"][1]["action"]["value"] = "north"
            @test_throws ErrorException replay_qwen3_environment_trace(
                tokenizer,
                task.spec,
                JSON3.read(JSON3.write(changed_transition)),
            )
        end
    end
end

@testset "Chapter 40 — task report rejects denominator drift" begin
    task_set = load_gridworld_tasks(CHAPTER40_TASKS)
    rows = [
        (; id=task.id, success=false, terminal=false, actions=0, invalid_actions=0)
        for task in task_set.tasks
    ]
    report = gridworld_task_report(task_set, rows)
    @test report.successes == 0
    @test report.success_rate == 0.0
    @test report.terminal == 0
    @test_throws ArgumentError gridworld_task_report(task_set, rows[1:end-1])
    @test_throws ArgumentError gridworld_task_report(task_set, [rows; rows[1]])
    bad = copy(rows)
    bad[1] = (; id=bad[1].id, success=true, terminal=false, actions=1, invalid_actions=0)
    @test_throws ArgumentError gridworld_task_report(task_set, bad)
end
