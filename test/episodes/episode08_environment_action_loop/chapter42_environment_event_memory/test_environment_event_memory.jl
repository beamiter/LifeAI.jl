using Test
using JSON3
using LifeAI
using LifeAI:
    AgentLoopStep,
    AgentLoopToolCall,
    AgentLoopTrace,
    Qwen3ToolCall,
    _agent_assistant_message,
    _agent_initial_messages,
    _agent_message,
    _python_json_text,
    apply_qwen3_chat_template,
    load_hf_qwen3_tokenizer

if !isdefined(@__MODULE__, :write_qwen3_tokenizer_fixture)
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tokenizer_fixture.jl"))
end
if !isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir)
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tiny_model_fixture.jl"))
end

const CHAPTER42_TASKS = joinpath(
    @__DIR__,
    "..",
    "chapter40_deterministic_gridworld",
    "fixtures",
    "gridworld_tasks.json",
)

function _chapter42_trace(
    task,
    actions;
    feedback=:full,
    system=gridworld_system_prompt(),
)
    environment = GridWorldEnvironment(task.spec)
    initial = observe_agent_environment(environment)
    steps = AgentLoopStep[]
    for (turn, direction) in enumerate(actions)
        transition = step_agent_environment!(
            environment,
            AgentEnvironmentAction("move", String(direction)),
        )
        call = AgentLoopToolCall(
            "move",
            String(JSON3.write((; direction=String(direction)))),
            true,
            render_agent_environment_feedback(transition; feedback),
            nothing,
            String[],
        )
        push!(steps, AgentLoopStep(
            turn,
            "",
            LifeAI._sha256_hex(""),
            0,
            [turn],
            "",
            :stop_token,
            :valid,
            AgentLoopToolCall[call],
            NamedTuple{(:raw, :reason),Tuple{String,String}}[],
            0.0,
            0.0,
        ))
        environment.terminal && break
    end
    final = observe_agent_environment(environment)
    return AgentEnvironmentTrace(
        "deterministic_gridworld",
        task.id,
        task.spec.sha256,
        feedback,
        String(system),
        initial,
        copy(environment.transitions),
        AgentLoopTrace(steps, Any[], "", :answered),
        environment.terminal,
        environment.success,
        final.state_sha256,
    )
end

function _chapter42_replayable_trace(
    task,
    tokenizer,
    actions;
    feedback=:none,
    memory_context=nothing,
    system=gridworld_memory_system_prompt(),
)
    environment = GridWorldEnvironment(task.spec)
    initial = observe_agent_environment(environment)
    registry = gridworld_tool_registry(environment; feedback)
    tools = qwen3_tool_specs(registry)
    messages = _agent_initial_messages(
        gridworld_user_prompt(initial);
        system,
        memory_context,
    )
    steps = AgentLoopStep[]
    for (turn, direction) in enumerate(actions)
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
        environment.terminal && break
    end
    turn = length(steps) + 1
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
    final = observe_agent_environment(environment)
    return AgentEnvironmentTrace(
        "deterministic_gridworld",
        task.id,
        task.spec.sha256,
        feedback,
        String(system),
        initial,
        copy(environment.transitions),
        AgentLoopTrace(steps, messages, completion, :answered, memory_context),
        environment.terminal,
        environment.success,
        final.state_sha256,
    )
end

@testset "Chapter 42 — explicit successful-episode writeback policy" begin
    writer_system = gridworld_memory_writer_system_prompt()
    @test startswith(writer_system, gridworld_system_prompt())
    @test writer_system != gridworld_memory_system_prompt()
    @test occursin("legal_actions list is authoritative", writer_system)
    @test occursin("one blocked move makes the whole episode ineligible", writer_system)
    @test !occursin(r"grid/0[1-8]", writer_system)

    task = first(load_gridworld_tasks(CHAPTER42_TASKS).tasks)
    policy = gridworld_successful_episode_memory_policy()
    trace = _chapter42_trace(
        task,
        gridworld_shortest_actions(task.spec);
        system=writer_system,
    )
    trace_payload = agent_environment_trace_payload(trace)
    @test trace_payload.system_prompt == writer_system
    @test trace_payload.system_prompt_sha256 == LifeAI._sha256_hex(writer_system)
    @test_throws UndefKeywordError agent_environment_memory_events(
        trace,
        task.spec;
        run_id="chapter42/write/grid-01",
    )
    events = agent_environment_memory_events(
        trace,
        task.spec;
        run_id="chapter42/write/grid-01",
        policy,
    )
    @test length(events) == 1
    event = only(events)
    @test event.id ==
        "environment-event/chapter42/write/grid-01/verified_successful_episode"
    @test event.event_sha256 == LifeAI._sha256_hex(event.text)
    @test event.metadata["spec_sha256"] == task.spec.sha256
    @test event.metadata["source_trace_sha256"] == event.source_trace_sha256
    @test occursin("verified_successful_episode", event.text)
    @test occursin("south, east, east, north", event.text)
    @test !occursin("walls", event.text)
    @test !occursin(task.spec.sha256, event.text)

    forged_spec = GridWorldSpec(
        task.spec.id,
        task.spec.width,
        task.spec.height,
        task.spec.start,
        task.spec.goal,
        ((2, 2),),
        task.spec.max_actions,
        task.spec.sha256,
    )
    forged_trace = _chapter42_trace(
        (; id=task.id, spec=forged_spec),
        ["east", "east"],
    )
    @test forged_trace.success
    @test_throws ArgumentError agent_environment_memory_events(
        forged_trace,
        forged_spec;
        run_id="chapter42/forged/grid-01",
        policy,
    )

    mktempdir() do directory
        journal = joinpath(directory, "environment-memory.jsonl")
        store = load_agent_memory_store(journal; create=true)
        first_write = append_agent_environment_events!(
            store,
            trace,
            task.spec;
            run_id="chapter42/write/grid-01",
            policy,
        )
        @test first_write.admitted
        @test first_write.reason == "admitted"
        @test first_write.appended_ids == [event.id]
        @test isempty(first_write.existing_ids)
        @test first_write.store_before_sha256 != first_write.store_after_sha256

        restored = load_agent_memory_store(journal)
        @test agent_memory_fingerprint(restored) == first_write.store_after_sha256
        @test length(agent_environment_memory_records(restored)) == 1
        validated = validate_agent_environment_memory_record(only(restored.records))
        @test validated.text == event.text
        @test validated.metadata == event.metadata

        retry = append_agent_environment_events!(
            restored,
            trace,
            task.spec;
            run_id="chapter42/write/grid-01",
            policy,
        )
        @test isempty(retry.appended_ids)
        @test retry.existing_ids == [event.id]
        @test retry.store_before_sha256 == retry.store_after_sha256
        @test length(readlines(journal)) == 1

        alternate = _chapter42_trace(
            task,
            ["south", "east", "east", "east", "north", "west"],
        )
        @test alternate.success
        unchanged = agent_memory_fingerprint(restored)
        @test_throws ArgumentError append_agent_environment_events!(
            restored,
            alternate,
            task.spec;
            run_id="chapter42/write/grid-01",
            policy,
        )
        @test agent_memory_fingerprint(restored) == unchanged
        @test length(readlines(journal)) == 1

        second_run = append_agent_environment_events!(
            restored,
            alternate,
            task.spec;
            run_id="chapter42/write/grid-01-alternate",
            policy,
        )
        @test length(second_run.appended_ids) == 1
        @test length(readlines(journal)) == 2
    end
end

@testset "Chapter 42 — failed, partial, blocked and diagnostic runs do not write" begin
    tasks = load_gridworld_tasks(CHAPTER42_TASKS).tasks
    task = first(tasks)
    policy = gridworld_successful_episode_memory_policy()
    candidates = (
        _chapter42_trace(task, ["south"]; feedback=:full),
        _chapter42_trace(task, fill("east", task.spec.max_actions); feedback=:full),
        _chapter42_trace(task, repeat(["south", "north"], 4); feedback=:full),
        _chapter42_trace(
            task,
            gridworld_shortest_actions(task.spec);
            feedback=:none,
        ),
        _chapter42_trace(
            task,
            ["east"; gridworld_shortest_actions(task.spec)];
            feedback=:full,
        ),
    )
    expected_reasons = (
        "policy_requires_terminal_episode",
        "policy_requires_successful_episode",
        "policy_requires_successful_episode",
        "policy_requires_full_feedback",
        "policy_rejects_blocked_actions",
    )
    mktempdir() do directory
        store = load_agent_memory_store(joinpath(directory, "memory.jsonl"); create=true)
        empty_sha = agent_memory_fingerprint(store)
        for (index, (trace, reason)) in enumerate(zip(candidates, expected_reasons))
            @test isempty(agent_environment_memory_events(
                trace,
                task.spec;
                run_id="chapter42/rejected/$index",
                policy,
            ))
            report = append_agent_environment_events!(
                store,
                trace,
                task.spec;
                run_id="chapter42/rejected/$index",
                policy,
            )
            @test !report.admitted
            @test report.reason == reason
            @test isempty(report.event_ids)
            @test report.store_before_sha256 == report.store_after_sha256 == empty_sha
        end
        @test isempty(store.records)
        @test !isfile(store.path)
    end

    pilot_task = tasks[5]
    blocked_then_success = _chapter42_trace(
        pilot_task,
        ["east"; gridworld_shortest_actions(pilot_task.spec)];
        feedback=:full,
        system=gridworld_memory_writer_system_prompt(),
    )
    @test blocked_then_success.success
    @test count(transition -> !transition.accepted, blocked_then_success.transitions) == 1
    pilot_report = append_agent_environment_events!(
        load_agent_memory_store(tempname(); create=true),
        blocked_then_success,
        pilot_task.spec;
        run_id="chapter42/rejected/pilot-grid-05",
        policy,
    )
    @test !pilot_report.admitted
    @test pilot_report.reason == "policy_rejects_blocked_actions"

    good = _chapter42_trace(task, gridworld_shortest_actions(task.spec))
    first_transition = first(good.transitions)
    corrupted = AgentEnvironmentTransition(
        first_transition.step,
        first_transition.action,
        first_transition.accepted,
        first_transition.reason,
        first_transition.before_state_sha256,
        first_transition.after_state_sha256,
        first_transition.observation,
        "0"^64,
    )
    bad = AgentEnvironmentTrace(
        good.environment,
        good.task_id,
        good.spec_sha256,
        good.feedback,
        good.system_prompt,
        good.initial_observation,
        [corrupted; good.transitions[2:end]],
        good.agent,
        good.terminal,
        good.success,
        good.final_state_sha256,
    )
    @test_throws ArgumentError agent_environment_memory_events(
        bad,
        task.spec;
        run_id="chapter42/corrupt/grid-01",
        policy,
    )
end

@testset "Chapter 42 — delayed context injection and tokenizer-only replay" begin
    tasks = load_gridworld_tasks(CHAPTER42_TASKS).tasks
    policy = gridworld_successful_episode_memory_policy()
    mktempdir() do directory
        store = load_agent_memory_store(joinpath(directory, "memory.jsonl"); create=true)
        event_ids = String[]
        for task in tasks
            trace = _chapter42_trace(task, gridworld_shortest_actions(task.spec))
            report = append_agent_environment_events!(
                store,
                trace,
                task.spec;
                run_id="chapter42/write/$(replace(task.id, '/' => '-'))",
                policy,
            )
            push!(event_ids, only(report.event_ids))
        end
        restored = load_agent_memory_store(store.path)
        @test length(agent_environment_memory_records(restored)) == length(tasks)

        embeddings = zeros(Float32, length(tasks), length(tasks))
        for index in eachindex(tasks)
            embeddings[index, index] = 1
        end
        exact_index = build_agent_memory_index(restored, embeddings)
        for (index, task) in enumerate(tasks)
            initial = observe_agent_environment(GridWorldEnvironment(task.spec))
            safe = retrieve_gridworld_memory_context(
                exact_index,
                task.spec,
                gridworld_memory_query(initial),
                view(embeddings, :, index);
                top_k=1,
            )
            @test [hit.id for hit in safe.hits] == [event_ids[index]]
        end

        first_task = first(tasks)
        stale_spec = GridWorldSpec(;
            id=first_task.spec.id,
            width=first_task.spec.width,
            height=first_task.spec.height,
            start=first_task.spec.start,
            goal=first_task.spec.goal,
            walls=[(2, 2)],
            max_actions=first_task.spec.max_actions,
        )
        stale_query = gridworld_memory_query(
            observe_agent_environment(GridWorldEnvironment(stale_spec)),
        )
        @test_throws ArgumentError select_gridworld_memory_context(
            restored,
            stale_spec,
            stale_query,
            [first(event_ids)],
        )
        @test select_agent_memory_context(
            restored,
            stale_query,
            [first(event_ids)],
        ).hits[1].id == first(event_ids)

        payloads = qwen3_tokenizer_fixture_payloads()
        template_path = joinpath(
            @__DIR__,
            "..",
            "..",
            "episode07_agent_closed_loop",
            "chapter36_qwen3_tools_chat_template",
            "fixtures",
            "official_chat_template.jinja",
        )
        payloads.tokenizer_config["chat_template"] = read(template_path, String)
        tokenizer = load_hf_qwen3_tokenizer(write_qwen3_tokenizer_fixture(
            joinpath(directory, "tokenizer");
            payloads,
        ))
        mate = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3, 5 => 6, 6 => 5, 7 => 8, 8 => 7)
        for (index, task) in enumerate(tasks)
            initial = observe_agent_environment(GridWorldEnvironment(task.spec))
            query = gridworld_memory_query(initial)
            relevant = select_gridworld_memory_context(
                restored,
                task.spec,
                query,
                [event_ids[index]],
            )
            distractor = select_agent_memory_context(restored, query, [event_ids[mate[index]]])
            system = gridworld_memory_system_prompt()
            relevant_prompt = apply_qwen3_chat_template(
                tokenizer,
                _agent_initial_messages(
                    gridworld_user_prompt(initial);
                    system,
                    memory_context=relevant,
                );
                tools=qwen3_tool_specs(gridworld_tool_registry(
                    GridWorldEnvironment(task.spec);
                    feedback=:none,
                )),
                add_generation_prompt=true,
                enable_thinking=false,
            )
            distractor_prompt = apply_qwen3_chat_template(
                tokenizer,
                _agent_initial_messages(
                    gridworld_user_prompt(initial);
                    system,
                    memory_context=distractor,
                );
                tools=qwen3_tool_specs(gridworld_tool_registry(
                    GridWorldEnvironment(task.spec);
                    feedback=:none,
                )),
                add_generation_prompt=true,
                enable_thinking=false,
            )
            @test length(encode(tokenizer, relevant_prompt; add_special_tokens=false)) ==
                  length(encode(tokenizer, distractor_prompt; add_special_tokens=false))

            own = replay_gridworld_actions(task.spec, gridworld_shortest_actions(task.spec))
            other_actions = gridworld_shortest_actions(tasks[mate[index]].spec)
            other = replay_gridworld_actions(task.spec, other_actions)
            @test own.environment.success
            @test !other.environment.success
        end

        task = first(tasks)
        initial = observe_agent_environment(GridWorldEnvironment(task.spec))
        query = gridworld_memory_query(initial)
        context = select_gridworld_memory_context(
            restored,
            task.spec,
            query,
            [first(event_ids)],
        )
        trace = _chapter42_replayable_trace(
            task,
            tokenizer,
            gridworld_shortest_actions(task.spec);
            feedback=:none,
            memory_context=context,
        )
        @test trace.success
        payload = agent_environment_trace_payload(trace)
        @test payload.memory.ids == [first(event_ids)]
        row = JSON3.read(JSON3.write(payload))
        report = replay_qwen3_environment_trace(
            tokenizer,
            task.spec,
            row;
            memory_store=restored,
        )
        @test report.success
        @test report.final_state_equal
        @test_throws ErrorException replay_qwen3_environment_trace(
            tokenizer,
            task.spec,
            row,
        )

        changed = JSON3.read(JSON3.write(payload), Dict{String,Any})
        changed["memory"]["context_sha256"] = "0"^64
        @test_throws ErrorException replay_qwen3_environment_trace(
            tokenizer,
            task.spec,
            JSON3.read(JSON3.write(changed));
            memory_store=restored,
        )
    end
end

@testset "Chapter 42 — environment loop retains its frozen retrieval evidence" begin
    task = first(load_gridworld_tasks(CHAPTER42_TASKS).tasks)
    mktempdir() do directory
        model_dir = joinpath(directory, "model")
        mkpath(model_dir)
        _qwen3_tiny_model_fixture_dir(
            model_dir;
            tie=false,
            vocab_size=263,
            max_seq_len=4096,
        )
        payloads = qwen3_tokenizer_fixture_payloads()
        template_path = joinpath(
            @__DIR__,
            "..",
            "..",
            "episode07_agent_closed_loop",
            "chapter36_qwen3_tools_chat_template",
            "fixtures",
            "official_chat_template.jinja",
        )
        payloads.tokenizer_config["chat_template"] = read(template_path, String)
        write_qwen3_tokenizer_fixture(model_dir; payloads)
        session = LifeAI.load_hf_qwen3_bf16_session(
            model_dir;
            context_tokens=4096,
            prefill_chunk_tokens=64,
        )
        store = load_agent_memory_store(joinpath(directory, "memory.jsonl"); create=true)
        source = _chapter42_trace(task, gridworld_shortest_actions(task.spec))
        event_id = only(append_agent_environment_events!(
            store,
            source,
            task.spec;
            run_id="chapter42/write/grid-01",
            policy=gridworld_successful_episode_memory_policy(),
        ).event_ids)
        initial = observe_agent_environment(GridWorldEnvironment(task.spec))
        context = select_gridworld_memory_context(
            store,
            task.spec,
            gridworld_memory_query(initial),
            [event_id],
        )
        trace = run_qwen3_environment_loop(
            session,
            GridWorldEnvironment(task.spec);
            feedback=:none,
            system=gridworld_memory_system_prompt(),
            memory_context=context,
            max_steps=1,
            max_new_tokens=1,
            enable_thinking=false,
            strategy=:greedy,
        )
        @test trace.agent.memory_context === context
        @test trace.system_prompt == gridworld_memory_system_prompt()
        @test occursin(context.rendered, only(trace.agent.steps).prompt)
        @test agent_environment_trace_payload(trace).memory.ids == [event_id]

        forged_context = AgentMemoryContext(
            context.query,
            "0"^64,
            context.store_sha256,
            context.hits,
            context.rendered,
            context.rendered_sha256,
        )
        @test_throws ArgumentError validate_agent_memory_context(forged_context)
        untouched = GridWorldEnvironment(task.spec)
        @test_throws ArgumentError run_qwen3_environment_loop(
            session,
            untouched;
            feedback=:none,
            system=gridworld_memory_system_prompt(),
            memory_context=forged_context,
            max_steps=1,
            max_new_tokens=1,
        )
        @test untouched.actions == 0
    end
end
