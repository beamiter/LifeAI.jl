#!/usr/bin/env julia

# Chapter 42 — tokenizer-only, read-only replay of the delayed environment-memory
# experiment. The journal is never opened for append. Reader contexts are rebuilt
# from persisted IDs, while writer completions are re-executed in authoritative
# environments so policy event bytes and source bindings can be derived again.

using JSON3
using SHA: sha256
using LifeAI

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const FROZEN_TASKS = joinpath(
    REPO_ROOT,
    "test",
    "episodes",
    "episode08_environment_action_loop",
    "chapter40_deterministic_gridworld",
    "fixtures",
    "gridworld_tasks.json",
)
const EXPECTED_TASK_IDS = ["grid/$(lpad(index, 2, '0'))" for index in 1:8]
const MIRRORED_TASK = Dict(
    "grid/01" => "grid/02",
    "grid/02" => "grid/01",
    "grid/03" => "grid/04",
    "grid/04" => "grid/03",
    "grid/05" => "grid/06",
    "grid/06" => "grid/05",
    "grid/07" => "grid/08",
    "grid/08" => "grid/07",
)
const READER_ARMS = ("memory_none", "memory_retrieved", "memory_distractor")
const WRAPPER_FIELDS = Set((
    :experiment_schema_version,
    :phase,
    :arm,
    :run_id,
    :writeback,
    :expected_memory_id,
    :mirrored_task,
    :journal_store_sha256,
    :generation_tokenizer_sha256,
    :generation_tokenizer_config_sha256,
    :generation_config_sha256,
    :trace,
))

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/replay_qwen3_environment_memory_eval.jl TASKS JOURNAL TRACE TOKENIZER_DIR [--out PATH]

Loads no embedding or generation weights and never writes the journal.
""")
end

sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

stable_run_id(id::AbstractString) =
    "chapter42/write/" * replace(String(id), '/' => '-')

function parse_args(args)
    !isempty(args) && args[1] in ("-h", "--help") && (usage(); return nothing)
    length(args) >= 4 || (usage(); return :usage_error)
    out_path = nothing
    if length(args) == 6 && args[5] == "--out"
        out_path = abspath(args[6])
    elseif length(args) != 4
        usage()
        return :usage_error
    end
    return (;
        tasks=abspath(args[1]),
        journal=abspath(args[2]),
        trace=abspath(args[3]),
        tokenizer=abspath(args[4]),
        out_path,
    )
end

function prefix_fingerprints(path::AbstractString)
    lines = readlines(path)
    fingerprints = String[LifeAI._sha256_hex("")]
    payload = ""
    for line in lines
        payload *= line * "\n"
        push!(fingerprints, LifeAI._sha256_hex(payload))
    end
    return fingerprints
end

function expected_trace_order(tasks)
    keys = Tuple{String,String,String}[]
    for task in tasks
        push!(keys, ("writer", task.id, "writer_full_feedback"))
    end
    for task in tasks, arm in READER_ARMS
        push!(keys, ("reader", task.id, arm))
    end
    return keys
end

function validate_wrapper(row)
    Set(Symbol(key) for key in keys(row)) == WRAPPER_FIELDS || throw(ArgumentError(
        "Chapter 42 trace wrapper fields do not match schema version 1",
    ))
    version = row.experiment_schema_version
    version isa Integer && !isa(version, Bool) || throw(ArgumentError(
        "Chapter 42 trace wrapper schema version must be an integer",
    ))
    Int(version) == 1 || throw(ArgumentError(
        "unsupported Chapter 42 trace wrapper schema",
    ))
    return row
end

function reconstructed_writer_trace(spec::GridWorldSpec, row)
    String(row.feedback) == "full" || error("writer trace must use full feedback")
    row.memory === nothing || error("writer trace unexpectedly contains memory")
    environment = GridWorldEnvironment(spec)
    initial = observe_agent_environment(environment)
    registry = gridworld_tool_registry(environment; feedback=:full)
    steps = AgentLoopStep[]
    for (position, recorded_step) in enumerate(row.agent_steps)
        completion = String(recorded_step.completion)
        parsed = parse_qwen3_tool_calls(completion)
        validity = agent_tool_call_validity(registry, parsed)
        validity == Symbol(String(recorded_step.validity)) || error(
            "writer validity drift for $(spec.id), turn $position",
        )
        executed = LifeAI.AgentLoopToolCall[]
        for call in parsed.calls
            outcome = invoke_agent_tool(registry, call)
            arguments_json = try
                LifeAI._python_json_text(call.arguments)
            catch replay_error
                "<unrenderable: " * sprint(showerror, replay_error) * ">"
            end
            push!(executed, LifeAI.AgentLoopToolCall(
                call.name,
                arguments_json,
                outcome.ok,
                outcome.output,
                outcome.error,
                outcome.coerced_arguments,
            ))
        end
        push!(steps, AgentLoopStep(
            position,
            "",
            String(recorded_step.prompt_sha256),
            Int(recorded_step.prompt_token_count),
            Int[Int(value) for value in recorded_step.generated_ids],
            completion,
            Symbol(String(recorded_step.stop_reason)),
            validity,
            executed,
            parsed.invalid,
            0.0,
            0.0,
        ))
    end
    final = observe_agent_environment(environment)
    agent = AgentLoopTrace(
        steps,
        Any[],
        String(row.answer),
        Symbol(String(row.agent_stop_reason)),
    )
    return AgentEnvironmentTrace(
        "deterministic_gridworld",
        spec.id,
        spec.sha256,
        :full,
        String(row.system_prompt),
        initial,
        copy(environment.transitions),
        agent,
        environment.terminal,
        environment.success,
        final.state_sha256,
    )
end

function record_matches_event(record::AgentMemoryRecord, event::AgentEnvironmentMemoryEvent)
    return record.id == event.id &&
        record.text == event.text &&
        record.text_sha256 == event.event_sha256 &&
        record.metadata == event.metadata
end

function recomputed_result(task::GridWorldTask, row)
    transitions = row.transitions
    steps = row.agent_steps
    actions = length(transitions)
    rejected = count(transition -> !Bool(transition.accepted), transitions)
    tool_calls = sum(length(step.tool_calls) for step in steps; init=0)
    tool_failures = sum(
        count(call -> !Bool(call.ok), step.tool_calls) for step in steps;
        init=0,
    )
    summary = row.summary
    expected_summary = (;
        task_id=task.id,
        feedback=String(row.feedback),
        terminal=Bool(row.terminal),
        success=Bool(row.success),
        model_turns=length(steps),
        actions,
        accepted_actions=actions - rejected,
        rejected_actions=rejected,
        tool_calls,
        tool_failures,
        invalid_actions=rejected + tool_failures,
        stop_reason=String(row.agent_stop_reason),
        final_state_sha256=String(row.final_state_sha256),
    )
    String(JSON3.write(summary)) == String(JSON3.write(expected_summary)) || error(
        "recorded environment summary drift for $(task.id)",
    )
    return (;
        id=task.id,
        success=Bool(row.success),
        terminal=Bool(row.terminal),
        actions,
        invalid_actions=rejected + tool_failures,
        model_turns=length(steps),
        tool_calls,
        tool_failures,
        shortest_steps=task.shortest_steps,
        excess_actions=Bool(row.success) ? actions - task.shortest_steps : nothing,
        stop_reason=String(row.agent_stop_reason),
        final_state_sha256=String(row.final_state_sha256),
    )
end

function paired_eval_result(task::GridWorldTask, row, protocol::Symbol)
    first_step = isempty(row.agent_steps) ? nothing : first(row.agent_steps)
    return EvalItemResult(
        task.id,
        "",
        protocol,
        first_step === nothing ? "" : String(first_step.prompt_sha256),
        first_step === nothing ? 0 : Int(first_step.prompt_token_count),
        String(row.answer),
        Bool(row.success) ? "goal" : "",
        "goal",
        Bool(row.success),
        true,
        (; feedback=String(row.feedback)),
    )
end

function main(args)
    parsed = parse_args(args)
    parsed === nothing && return 0
    parsed === :usage_error && return 2

    task_set = load_gridworld_tasks(parsed.tasks)
    task_set.sha256 == load_gridworld_tasks(FROZEN_TASKS).sha256 || error(
        "Chapter 42 replay requires the exact frozen Chapter 40 task fixture bytes",
    )
    [task.id for task in task_set.tasks] == EXPECTED_TASK_IDS || error(
        "Chapter 42 replay requires the ordered frozen eight-task fixture",
    )
    tasks = task_set.tasks
    task_by_id = Dict(task.id => task for task in tasks)
    store = load_agent_memory_store(parsed.journal)
    environment_records = agent_environment_memory_records(store)
    length(store.records) == length(tasks) == length(environment_records) || error(
        "journal must contain exactly one environment event per frozen task",
    )
    for task in tasks
        record = only(filter(
            record -> get(record.metadata, "task_id", "") == task.id,
            environment_records,
        ))
        validate_agent_environment_memory_record(record, task.spec)
    end
    store_sha256 = agent_memory_fingerprint(store)
    sha256_file(parsed.journal) == store_sha256 || error(
        "journal file SHA256 differs from its canonical store fingerprint",
    )

    wrappers = [
        validate_wrapper(JSON3.read(line))
        for line in eachline(parsed.trace)
        if !isempty(strip(line))
    ]
    length(wrappers) == 32 || error(
        "trace has $(length(wrappers)) rows; expected 8 writers and 24 readers",
    )
    observed_order = [
        (String(row.phase), String(row.trace.task), String(row.arm))
        for row in wrappers
    ]
    observed_order == expected_trace_order(tasks) || error(
        "trace rows do not follow writer-then-interleaved-reader order",
    )
    all(String(row.journal_store_sha256) == store_sha256 for row in wrappers) || error(
        "trace journal fingerprint differs from the loaded journal",
    )

    tokenizer = load_hf_qwen3_tokenizer(parsed.tokenizer)
    for row in wrappers
        String(row.generation_tokenizer_sha256) == tokenizer.tokenizer_sha256 || error(
            "generation tokenizer checksum mismatch",
        )
        String(row.generation_tokenizer_config_sha256) ==
            tokenizer.tokenizer_config_sha256 || error(
            "generation tokenizer-config checksum mismatch",
        )
        String(row.generation_config_sha256) ==
            tokenizer.generation_config_sha256 || error(
            "generation config checksum mismatch",
        )
    end

    replay_rows = Any[]
    result_rows = Dict(
        "writer_full_feedback" => Any[],
        (arm => Any[] for arm in READER_ARMS)...,
    )
    eval_rows = Dict(arm => EvalItemResult[] for arm in READER_ARMS)
    by_key = Dict{Tuple{String,String,String},Any}()
    memory_contexts_rebuilt = 0
    journal_fingerprints_equal = 0
    for wrapper in wrappers
        phase = String(wrapper.phase)
        arm = String(wrapper.arm)
        trace = wrapper.trace
        task = task_by_id[String(trace.task)]
        allow_cross = phase == "reader" && arm == "memory_distractor"

        report = replay_qwen3_environment_trace(
            tokenizer,
            task.spec,
            trace;
            memory_store=store,
            allow_cross_spec_memory=allow_cross,
        )
        push!(replay_rows, merge((; phase, arm), report))
        wrapper.journal_store_sha256 == store_sha256 || error(
            "journal fingerprint mismatch for $(task.id) / $arm",
        )
        journal_fingerprints_equal += 1

        memory = trace.memory
        if phase == "writer"
            arm == "writer_full_feedback" || error("unknown writer arm")
            wrapper.mirrored_task === nothing || error("writer unexpectedly names a mirror")
            String(trace.feedback) == "full" || error("writer feedback drift")
            trace.memory === nothing || error("writer contains delayed memory")
            String(trace.system_prompt) == gridworld_memory_writer_system_prompt() || error(
                "writer system prompt drift",
            )
        else
            arm in READER_ARMS || error("unknown reader arm $arm")
            wrapper.run_id === nothing || error("reader unexpectedly has a write run ID")
            wrapper.writeback === nothing || error("reader unexpectedly has writeback evidence")
            String(trace.feedback) == "none" || error("reader feedback drift")
            String(trace.system_prompt) == gridworld_memory_system_prompt() || error(
                "reader system prompt drift",
            )
            if arm == "memory_none"
                memory === nothing || error("no-memory reader contains memory")
                wrapper.expected_memory_id === nothing || error(
                    "no-memory reader has an expected memory ID",
                )
                wrapper.mirrored_task === nothing || error(
                    "no-memory reader unexpectedly names a mirror",
                )
            else
                memory === nothing && error("memory reader omits retrieval evidence")
                ids = String[String(id) for id in memory.ids]
                length(ids) == 1 || error("reader must contain one memory ID")
                expected_id = arm == "memory_retrieved" ?
                    "environment-event/$(stable_run_id(task.id))/verified_successful_episode" :
                    "environment-event/$(stable_run_id(MIRRORED_TASK[task.id]))/verified_successful_episode"
                ids == [expected_id] || error("reader memory ID/arm mismatch for $(task.id)")
                String(wrapper.expected_memory_id) == expected_id || error(
                    "wrapper expected memory ID mismatch for $(task.id)",
                )
                if arm == "memory_distractor"
                    String(wrapper.mirrored_task) == MIRRORED_TASK[task.id] || error(
                        "mirrored distractor task mismatch for $(task.id)",
                    )
                else
                    wrapper.mirrored_task === nothing || error(
                        "relevant arm unexpectedly names a mirror",
                    )
                end
                memory_contexts_rebuilt += 1
            end
        end

        push!(result_rows[arm], recomputed_result(task, trace))
        phase == "reader" && push!(
            eval_rows[arm],
            paired_eval_result(task, trace, Symbol(arm)),
        )
        key = (phase, task.id, arm)
        haskey(by_key, key) && error("duplicate trace key $(repr(key))")
        by_key[key] = wrapper
    end

    policy = gridworld_successful_episode_memory_policy()
    prefixes = prefix_fingerprints(parsed.journal)
    length(prefixes) == length(tasks) + 1 || error("journal prefix count drift")
    writer_events_equal = 0
    writer_source_bindings_equal = 0
    writeback_prefix_fingerprints_equal = 0
    for task in tasks
        wrapper = by_key[("writer", task.id, "writer_full_feedback")]
        run_id = stable_run_id(task.id)
        String(wrapper.run_id) == run_id || error("unstable writer run ID for $(task.id)")
        trace = reconstructed_writer_trace(task.spec, wrapper.trace)
        events = agent_environment_memory_events(
            trace,
            task.spec;
            run_id,
            policy,
        )
        length(events) == 1 || error("writer replay did not derive one event for $(task.id)")
        event = only(events)
        position = get(store.by_id, event.id, 0)
        position > 0 || error("derived writer event is absent from journal")
        record = store.records[position]
        record_matches_event(record, event) || error(
            "recomputed writer event bytes differ from journal for $(task.id)",
        )
        validate_agent_environment_memory_record(record, task.spec)
        String(wrapper.expected_memory_id) == event.id || error(
            "writer expected event ID mismatch",
        )
        writer_events_equal += 1

        writeback = wrapper.writeback
        String(writeback.policy) == String(policy.name) || error("writer policy drift")
        String(writeback.run_id) == run_id || error("writeback run ID drift")
        Bool(writeback.admitted) || error("writer writeback was not admitted")
        String(writeback.reason) == "admitted" || error("writer admission reason drift")
        String[String(id) for id in writeback.event_ids] == [event.id] || error(
            "writeback event IDs drift",
        )
        String[String(id) for id in writeback.appended_ids] == [event.id] || error(
            "writeback append IDs drift",
        )
        isempty(writeback.existing_ids) || error("fresh writer reused an existing ID")
        String(writeback.source_trace_sha256) == event.source_trace_sha256 || error(
            "writer source trace binding drift",
        )
        String(writeback.transition_chain_sha256) == event.transition_chain_sha256 || error(
            "writer transition-chain binding drift",
        )
        record.metadata["source_trace_sha256"] == event.source_trace_sha256 || error(
            "journal source-trace metadata drift",
        )
        writer_source_bindings_equal += 1

        String(writeback.store_before_sha256) == prefixes[record.sequence] || error(
            "writer before-prefix fingerprint drift for $(task.id)",
        )
        String(writeback.store_after_sha256) == prefixes[record.sequence + 1] || error(
            "writer after-prefix fingerprint drift for $(task.id)",
        )
        writeback_prefix_fingerprints_equal += 1
    end

    token_matched_pairs = 0
    for task in tasks
        relevant = by_key[("reader", task.id, "memory_retrieved")].trace
        distractor = by_key[("reader", task.id, "memory_distractor")].trace
        isempty(relevant.agent_steps) && error("relevant reader has no model step")
        isempty(distractor.agent_steps) && error("distractor reader has no model step")
        Int(first(relevant.agent_steps).prompt_token_count) ==
            Int(first(distractor.agent_steps).prompt_token_count) || error(
            "relevant/distractor first prompt lengths differ for $(task.id)",
        )
        token_matched_pairs += 1
    end

    environment_reports = (;
        writer=gridworld_task_report(task_set, result_rows["writer_full_feedback"]),
        no_memory=gridworld_task_report(task_set, result_rows["memory_none"]),
        retrieved_memory=gridworld_task_report(task_set, result_rows["memory_retrieved"]),
        matched_distractor=gridworld_task_report(
            task_set,
            result_rows["memory_distractor"],
        ),
    )
    paired = (;
        no_memory_vs_retrieved=paired_comparison(
            eval_rows["memory_none"],
            eval_rows["memory_retrieved"],
        ),
        distractor_vs_retrieved=paired_comparison(
            eval_rows["memory_distractor"],
            eval_rows["memory_retrieved"],
        ),
        no_memory_vs_distractor=paired_comparison(
            eval_rows["memory_none"],
            eval_rows["memory_distractor"],
        ),
    )

    report = (;
        schema_version=1,
        tasks_sha256=task_set.sha256,
        journal_store_sha256=store_sha256,
        journal_file_sha256=sha256_file(parsed.journal),
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        tokenizer_config_sha256=tokenizer.tokenizer_config_sha256,
        generation_config_sha256=tokenizer.generation_config_sha256,
        rows=length(wrappers),
        writer_rows=length(tasks),
        reader_rows=length(tasks) * length(READER_ARMS),
        model_steps=sum(row.model_steps for row in replay_rows),
        prompt_hashes_equal=sum(row.prompt_hashes_equal for row in replay_rows),
        prompt_tokens_equal=sum(row.prompt_tokens_equal for row in replay_rows),
        generated_ids_present=sum(row.generated_ids_present for row in replay_rows),
        tool_outcomes_equal=sum(row.tool_outcomes_equal for row in replay_rows),
        transition_hashes_equal=sum(row.transition_hashes_equal for row in replay_rows),
        final_states_equal=count(row -> row.final_state_equal, replay_rows),
        journal_fingerprints_equal,
        memory_contexts_rebuilt,
        token_matched_pairs,
        writer_events_equal,
        writer_source_bindings_equal,
        writeback_prefix_fingerprints_equal,
        environment=environment_reports,
        paired,
        embedding_loaded=false,
        generation_loaded=false,
        journal_appended=false,
        replayed=true,
    )
    if parsed.out_path !== nothing
        mkpath(dirname(parsed.out_path))
        open(parsed.out_path, "w") do io
            JSON3.pretty(io, JSON3.write(report))
            println(io)
        end
    end
    println(JSON3.write(report))
    return 0
end

exit(main(ARGS))
