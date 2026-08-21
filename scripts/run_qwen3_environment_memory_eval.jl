#!/usr/bin/env julia

# Chapter 42 — real, delayed environment-event memory evaluation.
#
# Request A runs one full-feedback episode and explicitly commits a verified
# successful route. Only after every writer has committed do we fresh-load the
# journal, rebuild an exact CPU embedding index, and run three interleaved
# feedback-withheld readers:
#   1. no memory;
#   2. actual top-1 semantic retrieval;
#   3. a mirrored irrelevant route with the same complete prompt-token count.

using LuxCUDA
using CUDA
using JSON3
using SHA: sha256
using LifeAI

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_TASKS = joinpath(
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
const READER_ARMS = (:memory_none, :memory_retrieved, :memory_distractor)

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/run_qwen3_environment_memory_eval.jl GENERATION_MODEL EMBEDDING_MODEL --out DIR [options]

options:
  --tasks PATH          frozen eight-task Chapter 40 fixture
  --out DIR             output directory (required)
  --journal PATH        new append-only journal (default: OUT/environment-memory.jsonl)
  --label NAME          output filename prefix (default: qwen3-4b)
  --max-new-tokens N    output allowance per model turn (default: 96)
  --context N           resident generation context (default: 4096)
  --variant NAME        generation model variant (default: qwen3_4b)
  --revision SHA        immutable generation revision (default: frozen variant revision)
  --supersedes PATH     optional prior failed-attempt manifest
  --cpu                 run generation on CPU; embedding is always CPU

The journal must be absent or empty. This keeps the writer-prefix fingerprints,
fresh-load boundary, and eight-task causal comparison unambiguous.
""")
end

function parse_args(args)
    !isempty(args) && args[1] in ("-h", "--help") && (usage(); exit())
    length(args) >= 2 || (usage(); exit(2))
    options = Dict{Symbol,Any}(
        :tasks => DEFAULT_TASKS,
        :out => nothing,
        :journal => nothing,
        :label => "qwen3-4b",
        :max_new_tokens => 96,
        :context => 4096,
        :variant => "qwen3_4b",
        :revision => "",
        :supersedes => nothing,
        :cpu => false,
    )
    integer_options = (:max_new_tokens, :context)
    index = 3
    while index <= length(args)
        option = args[index]
        if option == "--cpu"
            options[:cpu] = true
            index += 1
        elseif startswith(option, "--")
            index < length(args) || (usage(); exit(2))
            key = Symbol(replace(option[3:end], "-" => "_"))
            haskey(options, key) || (usage(); exit(2))
            value = args[index + 1]
            if key in integer_options
                parsed = tryparse(Int, value)
                parsed === nothing && (usage(); exit(2))
                options[key] = parsed
            else
                options[key] = value
            end
            index += 2
        else
            usage()
            exit(2)
        end
    end
    options[:out] === nothing && (usage(); exit(2))
    options[:max_new_tokens] > 0 || error("--max-new-tokens must be positive")
    options[:context] > options[:max_new_tokens] || error(
        "--context must exceed --max-new-tokens",
    )
    spec = qwen3_dense_spec(Symbol(options[:variant]))
    isempty(String(options[:revision])) && (options[:revision] = spec.revision)
    String(options[:revision]) == spec.revision || error(
        "generation revision must equal frozen $(spec.variant) revision $(spec.revision)",
    )
    return abspath(args[1]), abspath(args[2]), options, spec
end

sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function checkpoint_asset_report(model_dir::AbstractString)
    isdir(model_dir) || error("checkpoint directory does not exist: $model_dir")
    relative_paths = String[]
    for relative in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "modules.json",
        "config_sentence_transformers.json",
        joinpath("1_Pooling", "config.json"),
    )
        isfile(joinpath(model_dir, relative)) && push!(relative_paths, relative)
    end
    for name in readdir(model_dir)
        (name == "model.safetensors" ||
         occursin(r"^model-\d+-of-\d+\.safetensors$", name)) || continue
        push!(relative_paths, name)
    end
    sort!(unique!(relative_paths))
    any(endswith(path, ".safetensors") for path in relative_paths) || error(
        "checkpoint contains no model safetensors: $model_dir",
    )
    rows = NamedTuple[]
    for relative in relative_paths
        path = joinpath(model_dir, relative)
        push!(rows, (;
            path=replace(relative, '\\' => '/'),
            bytes=filesize(path),
            sha256=sha256_file(path),
        ))
    end
    canonical = String(JSON3.write(rows))
    return (;
        directory=abspath(model_dir),
        files=rows,
        total_bytes=sum(row.bytes for row in rows),
        manifest_sha256=LifeAI._sha256_hex(canonical),
    )
end

function verified_embedding_asset_report(model_dir::AbstractString)
    rows = collect(verify_qwen3_embedding_assets(model_dir))
    canonical = String(JSON3.write(rows))
    return (;
        directory=abspath(model_dir),
        files=rows,
        total_bytes=sum(row.bytes for row in rows),
        manifest_sha256=LifeAI._sha256_hex(canonical),
    )
end

function stable_run_id(task::GridWorldTask)
    return "chapter42/write/" * replace(task.id, '/' => '-')
end

function write_jsonl(path::AbstractString, rows)
    open(path, "w") do io
        for row in rows
            JSON3.write(io, row)
            println(io)
        end
    end
end

function write_json(path::AbstractString, row)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, JSON3.write(row))
        println(io)
    end
    return path
end

function writeback_payload(report::AgentEnvironmentMemoryWriteback)
    return (;
        policy=String(report.policy),
        run_id=report.run_id,
        admitted=report.admitted,
        reason=report.reason,
        event_ids=report.event_ids,
        appended_ids=report.appended_ids,
        existing_ids=report.existing_ids,
        store_before_sha256=report.store_before_sha256,
        store_after_sha256=report.store_after_sha256,
        source_trace_sha256=report.source_trace_sha256,
        transition_chain_sha256=report.transition_chain_sha256,
    )
end

function paired_eval_result(task, trace, protocol)
    first_step = isempty(trace.agent.steps) ? nothing : first(trace.agent.steps)
    summary = agent_environment_summary(trace)
    return EvalItemResult(
        task.id,
        "",
        protocol,
        first_step === nothing ? "" : first_step.prompt_sha256,
        first_step === nothing ? 0 : first_step.prompt_token_count,
        trace.agent.answer,
        trace.success ? "goal" : "",
        "goal",
        trace.success,
        true,
        summary,
    )
end

function result_item(task, trace; phase, arm, memory_context=nothing, writeback=nothing)
    result = gridworld_task_result(task, trace)
    first_step = isempty(trace.agent.steps) ? nothing : first(trace.agent.steps)
    memory = memory_context === nothing ? nothing : (;
        query=memory_context.query,
        query_sha256=memory_context.query_sha256,
        store_sha256=memory_context.store_sha256,
        context_sha256=memory_context.rendered_sha256,
        ids=[hit.id for hit in memory_context.hits],
        scores=[hit.score for hit in memory_context.hits],
    )
    return merge((;
        phase=String(phase),
        arm=String(arm),
        feedback=String(trace.feedback),
        system_prompt_sha256=LifeAI._sha256_hex(trace.system_prompt),
        first_prompt_sha256=first_step === nothing ? "" : first_step.prompt_sha256,
        first_prompt_tokens=first_step === nothing ? 0 : first_step.prompt_token_count,
        memory,
        writeback,
    ), result)
end

function initial_reader_prompt(tokenizer, task, memory_context)
    environment = GridWorldEnvironment(task.spec)
    initial = observe_agent_environment(environment)
    messages = LifeAI._agent_initial_messages(
        gridworld_user_prompt(initial);
        system=gridworld_memory_system_prompt(),
        memory_context,
    )
    prompt = apply_qwen3_chat_template(
        tokenizer,
        messages;
        tools=qwen3_tool_specs(gridworld_tool_registry(environment; feedback=:none)),
        add_generation_prompt=true,
        enable_thinking=false,
    )
    ids = encode(tokenizer, prompt; add_special_tokens=false)
    return (; prompt, sha256=LifeAI._sha256_hex(prompt), tokens=length(ids))
end

function recorded_route(store::AgentMemoryStore, id::AbstractString)
    position = get(store.by_id, String(id), 0)
    position > 0 || error("journal does not contain $(repr(String(id)))")
    event = validate_agent_environment_memory_record(store.records[position])
    row = JSON3.read(event.text)
    return String[String(action) for action in row.actions]
end

function trace_row(
    trace;
    phase,
    arm,
    journal_sha256,
    tokenizer,
    run_id=nothing,
    writeback=nothing,
    expected_memory_id=nothing,
    mirrored_task=nothing,
)
    return (;
        experiment_schema_version=1,
        phase=String(phase),
        arm=String(arm),
        run_id,
        writeback,
        expected_memory_id,
        mirrored_task,
        journal_store_sha256=journal_sha256,
        generation_tokenizer_sha256=tokenizer.tokenizer_sha256,
        generation_tokenizer_config_sha256=tokenizer.tokenizer_config_sha256,
        generation_config_sha256=tokenizer.generation_config_sha256,
        trace=agent_environment_trace_payload(trace),
    )
end

generation_model_dir, embedding_model_dir, options, generation_spec = parse_args(ARGS)
output_dir = abspath(String(options[:out]))
mkpath(output_dir)
label = String(options[:label])
journal_path = options[:journal] === nothing ?
    joinpath(output_dir, "environment-memory.jsonl") :
    abspath(String(options[:journal]))

task_set = load_gridworld_tasks(String(options[:tasks]))
frozen_task_sha256 = load_gridworld_tasks(DEFAULT_TASKS).sha256
task_set.sha256 == frozen_task_sha256 || error(
    "Chapter 42 requires the exact frozen Chapter 40 task fixture bytes",
)
[task.id for task in task_set.tasks] == EXPECTED_TASK_IDS || error(
    "Chapter 42 requires the ordered frozen task IDs " * join(EXPECTED_TASK_IDS, ", "),
)
tasks = task_set.tasks
task_by_id = Dict(task.id => task for task in tasks)

use_cuda = !Bool(options[:cpu])
use_cuda && !CUDA.functional() && error("CUDA.jl is not functional; pass --cpu")

println(stderr, "hashing frozen generation checkpoint assets")
generation_assets = checkpoint_asset_report(generation_model_dir)
generation_config = only(filter(
    row -> row.path == "config.json",
    generation_assets.files,
))
generation_config.sha256 == generation_spec.config_sha256 || error(
    "generation config checksum does not match frozen $(generation_spec.variant)",
)
println(stderr, "verifying frozen CPU embedding checkpoint assets")
embedding_assets = verified_embedding_asset_report(embedding_model_dir)
embedding_spec = qwen3_embedding_spec()

source_paths = (
    environment=joinpath(REPO_ROOT, "src", "agent", "environment.jl"),
    environment_memory=joinpath(REPO_ROOT, "src", "agent", "environment_memory.jl"),
    memory=joinpath(REPO_ROOT, "src", "agent", "memory.jl"),
)
source_sha256 = (;
    runner=sha256_file(abspath(@__FILE__)),
    environment=sha256_file(source_paths.environment),
    environment_memory=sha256_file(source_paths.environment_memory),
    memory=sha256_file(source_paths.memory),
)
supersedes = if options[:supersedes] === nothing
    nothing
else
    path = abspath(String(options[:supersedes]))
    isfile(path) || error("--supersedes manifest does not exist: $path")
    (; path, sha256=sha256_file(path))
end
run_start_path = joinpath(output_dir, "run_start.json")
run_start = (;
    schema_version=1,
    status="started",
    argv=copy(ARGS),
    tasks=abspath(String(options[:tasks])),
    tasks_sha256=task_set.sha256,
    expected_task_ids=EXPECTED_TASK_IDS,
    generation=(;
        model=abspath(generation_model_dir),
        model_id=generation_spec.model_id,
        variant=String(generation_spec.variant),
        revision=String(options[:revision]),
        config_sha256=generation_spec.config_sha256,
        assets=generation_assets,
    ),
    embedding=(;
        model=abspath(embedding_model_dir),
        model_id=embedding_spec.model_id,
        revision=embedding_spec.revision,
        config_sha256=embedding_spec.config_sha256,
        model_sha256=embedding_spec.model_sha256,
        assets=embedding_assets,
    ),
    writer_system_prompt=gridworld_memory_writer_system_prompt(),
    writer_system_prompt_sha256=LifeAI._sha256_hex(
        gridworld_memory_writer_system_prompt(),
    ),
    reader_system_prompt=gridworld_memory_system_prompt(),
    reader_system_prompt_sha256=LifeAI._sha256_hex(gridworld_memory_system_prompt()),
    max_new_tokens=Int(options[:max_new_tokens]),
    context_tokens=Int(options[:context]),
    strategy="greedy",
    enable_thinking=false,
    source_sha256,
    supersedes,
)
write_json(run_start_path, run_start)
run_start_sha256 = sha256_file(run_start_path)

writer_store = load_agent_memory_store(journal_path; create=true)
isempty(writer_store.records) || error(
    "Chapter 42 real evaluation requires an absent or empty journal: $journal_path",
)

generation_device = use_cuda ? String(CUDA.name(CUDA.device())) : "cpu"
println(stderr, "loading generation model on $generation_device")
load_started = time_ns()
session = load_hf_qwen3_bf16_session(
    generation_model_dir;
    context_tokens=Int(options[:context]),
    revision=String(options[:revision]),
    variant=Symbol(options[:variant]),
    to_device=use_cuda ? CUDA.cu : identity,
)
use_cuda && (CUDA.synchronize(); GC.gc(); CUDA.reclaim())
generation_load_seconds = (time_ns() - load_started) / 1.0e9

policy = gridworld_successful_episode_memory_policy()
writer_traces = Dict{String,AgentEnvironmentTrace}()
writebacks = Dict{String,AgentEnvironmentMemoryWriteback}()
event_id_by_task = Dict{String,String}()
writer_attempt_dir = joinpath(output_dir, "writer_attempts")
mkpath(writer_attempt_dir)
writer_started = time_ns()
for (position, task) in enumerate(tasks)
    trace = run_qwen3_environment_loop(
        session,
        GridWorldEnvironment(task.spec);
        feedback=:full,
        system=gridworld_memory_writer_system_prompt(),
        max_steps=task.spec.max_actions + 1,
        max_new_tokens=Int(options[:max_new_tokens]),
        enable_thinking=false,
        strategy=:greedy,
    )
    run_id = stable_run_id(task)
    report = append_agent_environment_events!(
        writer_store,
        trace,
        task.spec;
        run_id,
        policy,
    )
    journal_after_sha256 = agent_memory_fingerprint(writer_store)
    sha256_file(journal_path) == journal_after_sha256 || error(
        "writer $(task.id) journal bytes diverged from the in-memory store",
    )
    attempt_path = joinpath(
        writer_attempt_dir,
        "$(lpad(position, 2, '0'))-$(replace(task.id, '/' => '-')).json",
    )
    attempt = (;
        schema_version=1,
        task_index=position,
        task_id=task.id,
        run_id,
        admitted=report.admitted,
        reason=report.reason,
        transitions=length(trace.transitions),
        accepted_transitions=count(transition -> transition.accepted, trace.transitions),
        blocked_transitions=count(transition -> !transition.accepted, trace.transitions),
        terminal=trace.terminal,
        success=trace.success,
        journal_before_sha256=report.store_before_sha256,
        journal_after_sha256,
        writeback=writeback_payload(report),
        trace=agent_environment_trace_payload(trace),
        run_start_sha256,
    )
    write_json(attempt_path, attempt)
    if !report.admitted
        failure_path = joinpath(output_dir, "failure.json")
        write_json(failure_path, (;
            schema_version=1,
            status="failed",
            phase="writer",
            task_index=position,
            task_id=task.id,
            reason=report.reason,
            attempt=attempt_path,
            attempt_sha256=sha256_file(attempt_path),
            journal=journal_path,
            journal_sha256=journal_after_sha256,
            run_start=run_start_path,
            run_start_sha256,
        ))
    end
    report.admitted || error(
        "writer $(task.id) was not admitted: $(report.reason)",
    )
    length(report.appended_ids) == 1 || error(
        "writer $(task.id) did not append exactly one new event",
    )
    isempty(report.existing_ids) || error("writer $(task.id) reused an existing event")
    writer_traces[task.id] = trace
    writebacks[task.id] = report
    event_id_by_task[task.id] = only(report.event_ids)
    println(stderr, "writer $position/$(length(tasks)) $(task.id) " *
        "success=$(trace.success) actions=$(length(trace.transitions))")
end
writer_seconds = (time_ns() - writer_started) / 1.0e9

# The causal boundary: all reader evidence comes from a strict load of persisted
# bytes, never from the mutable writer store or its in-process objects.
writer_store = nothing
GC.gc()
store = load_agent_memory_store(journal_path)
environment_records = agent_environment_memory_records(store)
length(store.records) == length(tasks) == length(environment_records) || error(
    "fresh-loaded journal must contain exactly one environment event per task",
)
for task in tasks
    record = only(filter(
        record -> get(record.metadata, "task_id", "") == task.id,
        environment_records,
    ))
    validate_agent_environment_memory_record(record, task.spec)
end
journal_sha256 = agent_memory_fingerprint(store)
sha256_file(journal_path) == journal_sha256 || error(
    "journal file bytes and canonical store fingerprint differ",
)

println(stderr, "loading frozen Qwen3 embedding model on CPU")
embedding_started = time_ns()
embedding_bundle = load_hf_qwen3_embedding_bundle(
    embedding_model_dir;
    max_seq_len=256,
)
index = build_agent_memory_index(embedding_bundle, store; max_length=256)
queries = String[]
for task in tasks
    initial = observe_agent_environment(GridWorldEnvironment(task.spec))
    push!(queries, gridworld_memory_query(initial))
end
query_embeddings = embed_texts(
    embedding_bundle,
    [qwen3_embedding_query(query) for query in queries];
    dimension=size(index.semantic.embeddings, 1),
    max_length=256,
    padding_side=:left,
).embeddings

retrieved_context = Dict{String,AgentMemoryContext}()
retrieval_rows = Any[]
for (position, task) in enumerate(tasks)
    raw_hits = retrieve_agent_memory(
        index,
        view(query_embeddings, :, position);
        top_k=1,
    )
    raw_hit = only(raw_hits)
    context = retrieve_gridworld_memory_context(
        index,
        task.spec,
        queries[position],
        view(query_embeddings, :, position);
        top_k=1,
    )
    hit = only(context.hits)
    expected = event_id_by_task[task.id]
    raw_hit.id == expected || error(
        "actual unfiltered embedding top-1 for $(task.id) was $(raw_hit.id), expected $expected",
    )
    hit.id == expected || error("safe exact-spec retrieval drift for $(task.id)")
    hit.score == raw_hit.score || error("safe retrieval score drift for $(task.id)")
    retrieved_context[task.id] = context
    push!(retrieval_rows, (;
        id=task.id,
        query=queries[position],
        query_sha256=context.query_sha256,
        expected_id=expected,
        raw_top1_id=raw_hit.id,
        safe_context_id=hit.id,
        score=hit.score,
        correct=true,
    ))
end
embedding_seconds = (time_ns() - embedding_started) / 1.0e9
embedding_bundle = nothing
query_embeddings = nothing
GC.gc()

distractor_context = Dict{String,AgentMemoryContext}()
token_match_rows = Any[]
for task in tasks
    mate_id = MIRRORED_TASK[task.id]
    distractor_id = event_id_by_task[mate_id]
    context = select_agent_memory_context(
        store,
        gridworld_memory_query(observe_agent_environment(GridWorldEnvironment(task.spec))),
        [distractor_id],
    )
    distractor_context[task.id] = context
    relevant_prompt = initial_reader_prompt(
        session.tokenizer,
        task,
        retrieved_context[task.id],
    )
    distractor_prompt = initial_reader_prompt(session.tokenizer, task, context)
    relevant_prompt.tokens == distractor_prompt.tokens || error(
        "relevant/distractor prompt-token mismatch for $(task.id): " *
        "$(relevant_prompt.tokens) != $(distractor_prompt.tokens)",
    )
    route = recorded_route(store, distractor_id)
    mirrored_replay = replay_gridworld_actions(task.spec, route)
    !mirrored_replay.environment.success || error(
        "mirrored distractor route unexpectedly solves $(task.id)",
    )
    push!(token_match_rows, (;
        id=task.id,
        mirrored_task=mate_id,
        relevant_id=event_id_by_task[task.id],
        distractor_id,
        relevant_prompt_sha256=relevant_prompt.sha256,
        distractor_prompt_sha256=distractor_prompt.sha256,
        relevant_prompt_tokens=relevant_prompt.tokens,
        distractor_prompt_tokens=distractor_prompt.tokens,
        token_delta=distractor_prompt.tokens - relevant_prompt.tokens,
        distractor_route_succeeds=false,
    ))
end

reader_traces = Dict(arm => Dict{String,AgentEnvironmentTrace}() for arm in READER_ARMS)
reader_started = time_ns()
for (position, task) in enumerate(tasks)
    contexts = (
        memory_none=nothing,
        memory_retrieved=retrieved_context[task.id],
        memory_distractor=distractor_context[task.id],
    )
    for arm in READER_ARMS
        trace = run_qwen3_environment_loop(
            session,
            GridWorldEnvironment(task.spec);
            feedback=:none,
            system=gridworld_memory_system_prompt(),
            memory_context=getproperty(contexts, arm),
            allow_cross_spec_memory=arm === :memory_distractor,
            max_steps=task.spec.max_actions + 1,
            max_new_tokens=Int(options[:max_new_tokens]),
            enable_thinking=false,
            strategy=:greedy,
        )
        reader_traces[arm][task.id] = trace
        println(stderr, "reader $position/$(length(tasks)) $(task.id) $arm " *
            "success=$(trace.success) actions=$(length(trace.transitions))")
    end
end
reader_seconds = (time_ns() - reader_started) / 1.0e9

for task in tasks
    relevant = reader_traces[:memory_retrieved][task.id]
    distractor = reader_traces[:memory_distractor][task.id]
    relevant.system_prompt == distractor.system_prompt ==
        reader_traces[:memory_none][task.id].system_prompt ==
        gridworld_memory_system_prompt() || error("reader system prompt drift")
    first(relevant.agent.steps).prompt_token_count ==
        first(distractor.agent.steps).prompt_token_count || error(
        "executed relevant/distractor first prompts differ in length for $(task.id)",
    )
end

writer_rows = [gridworld_task_result(task, writer_traces[task.id]) for task in tasks]
reader_rows = Dict(
    arm => [gridworld_task_result(task, reader_traces[arm][task.id]) for task in tasks]
    for arm in READER_ARMS
)
writer_report = gridworld_task_report(task_set, writer_rows)
reader_reports = Dict(arm => gridworld_task_report(task_set, reader_rows[arm]) for arm in READER_ARMS)

paired = Dict{Symbol,Vector{EvalItemResult}}(
    arm => [paired_eval_result(task, reader_traces[arm][task.id], arm) for task in tasks]
    for arm in READER_ARMS
)
no_memory_vs_retrieved = paired_comparison(
    paired[:memory_none],
    paired[:memory_retrieved],
)
distractor_vs_retrieved = paired_comparison(
    paired[:memory_distractor],
    paired[:memory_retrieved],
)
no_memory_vs_distractor = paired_comparison(
    paired[:memory_none],
    paired[:memory_distractor],
)

trace_rows = Any[]
item_rows = Any[]
for task in tasks
    writeback = writeback_payload(writebacks[task.id])
    push!(trace_rows, trace_row(
        writer_traces[task.id];
        phase=:writer,
        arm=:writer_full_feedback,
        journal_sha256,
        tokenizer=session.tokenizer,
        run_id=stable_run_id(task),
        writeback,
        expected_memory_id=event_id_by_task[task.id],
    ))
    push!(item_rows, result_item(
        task,
        writer_traces[task.id];
        phase=:writer,
        arm=:writer_full_feedback,
        writeback,
    ))
end
for task in tasks, arm in READER_ARMS
    context = arm === :memory_none ? nothing :
        arm === :memory_retrieved ? retrieved_context[task.id] :
        distractor_context[task.id]
    expected = arm === :memory_none ? nothing :
        arm === :memory_retrieved ? event_id_by_task[task.id] :
        event_id_by_task[MIRRORED_TASK[task.id]]
    push!(trace_rows, trace_row(
        reader_traces[arm][task.id];
        phase=:reader,
        arm,
        journal_sha256,
        tokenizer=session.tokenizer,
        expected_memory_id=expected,
        mirrored_task=arm === :memory_distractor ? MIRRORED_TASK[task.id] : nothing,
    ))
    push!(item_rows, result_item(
        task,
        reader_traces[arm][task.id];
        phase=:reader,
        arm,
        memory_context=context,
    ))
end

trace_path = joinpath(output_dir, "$(label)_environment_memory_trace.jsonl")
items_path = joinpath(output_dir, "$(label)_environment_memory_items.jsonl")
summary_path = joinpath(output_dir, "$(label)_environment_memory_summary.json")
write_jsonl(trace_path, trace_rows)
write_jsonl(items_path, item_rows)

writeback_rows = [writeback_payload(writebacks[task.id]) for task in tasks]
summary = (;
    schema_version=1,
    label,
    tasks=abspath(String(options[:tasks])),
    tasks_sha256=task_set.sha256,
    evaluated_tasks=length(tasks),
    coordinate_system=task_set.coordinate_system,
    generation=(;
        model=abspath(generation_model_dir),
        model_id=generation_spec.model_id,
        variant=String(generation_spec.variant),
        revision=String(options[:revision]),
        device=generation_device,
        context_tokens=Int(options[:context]),
        max_new_tokens=Int(options[:max_new_tokens]),
        strategy="greedy",
        enable_thinking=false,
        load_seconds=generation_load_seconds,
        config_sha256=generation_spec.config_sha256,
        tokenizer_sha256=session.tokenizer.tokenizer_sha256,
        tokenizer_config_sha256=session.tokenizer.tokenizer_config_sha256,
        generation_config_sha256=session.tokenizer.generation_config_sha256,
        assets=generation_assets,
    ),
    embedding=(;
        model=abspath(embedding_model_dir),
        model_id=embedding_spec.model_id,
        variant=String(embedding_spec.variant),
        revision=embedding_spec.revision,
        device="cpu",
        dimension=size(index.semantic.embeddings, 1),
        max_length=256,
        seconds=embedding_seconds,
        config_sha256=embedding_spec.config_sha256,
        model_sha256=embedding_spec.model_sha256,
        assets=embedding_assets,
    ),
    protocol=(;
        writer_feedback="full",
        writer_system_prompt=gridworld_memory_writer_system_prompt(),
        writer_system_prompt_sha256=LifeAI._sha256_hex(
            gridworld_memory_writer_system_prompt(),
        ),
        reader_feedback="none",
        reader_system_prompt=gridworld_memory_system_prompt(),
        reader_system_prompt_sha256=LifeAI._sha256_hex(gridworld_memory_system_prompt()),
        reader_order=String[String(arm) for arm in READER_ARMS],
        interleaved_by_task=true,
        policy=String(policy.name),
        stable_run_id_prefix="chapter42/write/",
    ),
    journal=(;
        path=journal_path,
        records=length(store.records),
        environment_records=length(environment_records),
        store_sha256=journal_sha256,
        file_sha256=sha256_file(journal_path),
        fresh_loaded=true,
        initially_empty=true,
    ),
    writeback=(;
        attempted=length(writeback_rows),
        admitted=count(row -> row.admitted, writeback_rows),
        appended=sum(length(row.appended_ids) for row in writeback_rows),
        existing=sum(length(row.existing_ids) for row in writeback_rows),
        rows=writeback_rows,
    ),
    retrieval=(;
        total=length(retrieval_rows),
        top1_correct=count(row -> row.correct, retrieval_rows),
        recall_at_1=count(row -> row.correct, retrieval_rows) / length(retrieval_rows),
        exact_spec_matches=count(row -> row.correct, retrieval_rows),
        rows=retrieval_rows,
    ),
    token_match=(;
        total=length(token_match_rows),
        equal=count(row -> row.token_delta == 0, token_match_rows),
        mirrored_routes_non_solving=count(
            row -> !row.distractor_route_succeeds,
            token_match_rows,
        ),
        rows=token_match_rows,
    ),
    environment=(;
        writer=writer_report,
        no_memory=reader_reports[:memory_none],
        retrieved_memory=reader_reports[:memory_retrieved],
        matched_distractor=reader_reports[:memory_distractor],
    ),
    paired=(;
        no_memory_vs_retrieved,
        distractor_vs_retrieved,
        no_memory_vs_distractor,
    ),
    timing=(; writer_seconds, reader_seconds),
    provenance=(;
        run_start=run_start_path,
        run_start_sha256,
        writer_attempts=writer_attempt_dir,
        supersedes,
    ),
    outputs=(; trace=trace_path, items=items_path, summary=summary_path),
    source_sha256,
)
write_json(summary_path, summary)
run_complete_path = joinpath(output_dir, "run_complete.json")
write_json(run_complete_path, (;
    schema_version=1,
    status="completed",
    run_start=run_start_path,
    run_start_sha256,
    summary=summary_path,
    summary_sha256=sha256_file(summary_path),
    trace=trace_path,
    trace_sha256=sha256_file(trace_path),
    items=items_path,
    items_sha256=sha256_file(items_path),
    journal=journal_path,
    journal_sha256=sha256_file(journal_path),
))

println(stderr, "writers $(writer_report.successes)/$(writer_report.total); " *
    "readers none=$(reader_reports[:memory_none].successes), " *
    "retrieved=$(reader_reports[:memory_retrieved].successes), " *
    "distractor=$(reader_reports[:memory_distractor].successes); " *
    "retrieval=$(count(row -> row.correct, retrieval_rows))/$(length(retrieval_rows)); " *
    "journal=$journal_sha256")
