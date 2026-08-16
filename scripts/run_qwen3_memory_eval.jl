#!/usr/bin/env julia

# Chapter 39 — three-arm, same-item cross-request memory evaluation.
#
# The journal is seeded and then loaded again before either model is constructed,
# making the write/read boundary explicit. Each question runs as:
#   1. no memory;
#   2. exact top-1 semantic retrieval;
#   3. an irrelevant record with exactly the same Qwen3-4B prompt-token count.
# The third arm isolates useful content from the prompt-length perturbation observed
# in Chapter 38.

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
    "episode07_agent_closed_loop",
    "chapter39_persistent_semantic_memory",
    "fixtures",
    "memory_tasks.json",
)

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/run_qwen3_memory_eval.jl GENERATION_MODEL EMBEDDING_MODEL --out DIR [options]

options:
  --tasks PATH          frozen Chapter 39 task fixture
  --out DIR             output directory (required)
  --journal PATH        append-only memory journal (default: OUT/memory.jsonl)
  --label NAME          output filename prefix (default: qwen3-4b)
  --limit N             evaluate only the first N tasks
  --max-new-tokens N    output allowance per arm (default: 64)
  --context N           generation session context (default: 2048)
  --variant NAME        generation model variant (default: qwen3_4b)
  --revision SHA        expected generation checkpoint revision
  --cpu                 run generation on CPU
  --embedding-cuda      move the embedding model to CUDA (default: CPU)
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
        :limit => 0,
        :max_new_tokens => 64,
        :context => 2048,
        :variant => "qwen3_4b",
        :revision => "",
        :cpu => false,
        :embedding_cuda => false,
    )
    integers = (:limit, :max_new_tokens, :context)
    index = 3
    while index <= length(args)
        option = args[index]
        if option == "--cpu"
            options[:cpu] = true
            index += 1
        elseif option == "--embedding-cuda"
            options[:embedding_cuda] = true
            index += 1
        elseif startswith(option, "--")
            index < length(args) || (usage(); exit(2))
            key = Symbol(replace(option[3:end], "-" => "_"))
            haskey(options, key) || (usage(); exit(2))
            value = args[index + 1]
            if key in integers
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
    return args[1], args[2], options
end

sha256_hex(text::AbstractString) = bytes2hex(sha256(codeunits(text)))

function prompt_for_context(tokenizer, task, context)
    messages = LifeAI._agent_initial_messages(task.question; memory_context=context)
    prompt = apply_qwen3_chat_template(
        tokenizer,
        messages;
        add_generation_prompt=true,
        enable_thinking=false,
    )
    return prompt, length(encode(tokenizer, prompt; add_special_tokens=false))
end

function matched_distractor_context(index, tokenizer, task, retrieved_context)
    _, target_tokens = prompt_for_context(tokenizer, task, retrieved_context)
    excluded = Set([task.memory_id; [hit.id for hit in retrieved_context.hits]])
    candidates = String[task.distractor_id]
    append!(candidates, sort!(String[
        record.id
        for record in index.records
        if !(record.id in excluded) &&
           !occursin(task.expected, lowercase(record.text))
    ]))
    unique!(candidates)
    for id in candidates
        id in excluded && continue
        context = select_agent_memory_context(index, task.question, [id])
        _, tokens = prompt_for_context(tokenizer, task, context)
        tokens == target_tokens && return context, tokens
    end
    error(
        "no irrelevant memory matches the retrieved prompt length for $(task.id) " *
        "($target_tokens tokens)",
    )
end

function eval_result(task, arm, prompt, completion, stop_reason, detail)
    extracted = extract_agent_memory_answer(completion)
    return EvalItemResult(
        task.id,
        "",
        arm,
        sha256_hex(prompt),
        length(detail.prompt_ids),
        completion,
        extracted === nothing ? "" : extracted,
        task.expected,
        agent_memory_answer_matches(task.expected, extracted),
        extracted !== nothing,
        merge((; stop_reason=String(stop_reason)), detail.extra),
    )
end

function item_payload(result::EvalItemResult)
    return (;
        id=result.id,
        subject=result.subject,
        protocol=String(result.protocol),
        prompt_sha256=result.prompt_sha256,
        prompt_token_count=result.prompt_token_count,
        completion=result.completion,
        extracted=result.extracted,
        expected=result.expected,
        correct=result.correct,
        parsed=result.parsed,
        detail=result.detail,
    )
end

function write_items(path, results)
    open(path, "w") do io
        for result in results
            JSON3.write(io, item_payload(result))
            println(io)
        end
    end
end

generation_model_dir, embedding_model_dir, options = parse_args(ARGS)
output_dir = abspath(String(options[:out]))
mkpath(output_dir)
journal_path = options[:journal] === nothing ? joinpath(output_dir, "memory.jsonl") :
    abspath(String(options[:journal]))
label = String(options[:label])

task_set = load_agent_memory_tasks(String(options[:tasks]))
tasks = options[:limit] > 0 ?
    task_set.tasks[1:min(Int(options[:limit]), length(task_set.tasks))] : task_set.tasks

# Request A writes; request/process B can reconstruct solely from this journal.
writer = load_agent_memory_store(journal_path; create=true)
seed_agent_memory_tasks!(writer, task_set)
store = load_agent_memory_store(journal_path)
store_sha256 = agent_memory_fingerprint(store)

use_generation_cuda = !Bool(options[:cpu])
use_embedding_cuda = Bool(options[:embedding_cuda])
(use_generation_cuda || use_embedding_cuda) && !CUDA.functional() && error(
    "CUDA.jl is not functional; use --cpu and omit --embedding-cuda",
)

println(stderr, "loading embedding model (device=$(use_embedding_cuda ? "cuda" : "cpu"))")
embedding_loaded = load_hf_qwen3_embedding_bundle(
    embedding_model_dir;
    max_seq_len=256,
)
embedding_bundle = use_embedding_cuda ?
    merge(embedding_loaded, (; parameters=CUDA.cu(embedding_loaded.parameters))) :
    embedding_loaded
index = build_agent_memory_index(
    embedding_bundle,
    store;
    max_length=256,
)

# Batch all queries so retrieval cost is paid once and the retrieval rows are frozen
# before generation begins.
query_inputs = [qwen3_embedding_query(task.question) for task in tasks]
query_embeddings = embed_texts(
    embedding_bundle,
    query_inputs;
    dimension=size(index.semantic.embeddings, 1),
    max_length=256,
    padding_side=:left,
).embeddings
retrieved_contexts = AgentMemoryContext[]
retrieval_rows = Any[]
for (position, task) in enumerate(tasks)
    context = retrieve_agent_memory_context(
        index,
        task.question,
        view(query_embeddings, :, position);
        top_k=1,
    )
    push!(retrieved_contexts, context)
    push!(retrieval_rows, (; id=task.id, retrieved_ids=[hit.id for hit in context.hits]))
end
retrieval_report = agent_memory_retrieval_report(
    AgentMemoryTaskSet(tasks, task_set.source, task_set.sha256),
    retrieval_rows,
)

# Retrieval evidence is now plain CPU memory. Release the embedding parameter tree
# before constructing the resident generation session; reclaim CUDA explicitly when
# that tree was placed there.
embedding_bundle = nothing
embedding_loaded = nothing
query_embeddings = nothing
GC.gc()
if use_embedding_cuda
    CUDA.reclaim()
end

generation_device = use_generation_cuda ? String(CUDA.name(CUDA.device())) : "cpu"
println(stderr, "loading generation model (device=$generation_device)")
session = load_hf_qwen3_bf16_session(
    generation_model_dir;
    context_tokens=Int(options[:context]),
    revision=String(options[:revision]),
    variant=Symbol(options[:variant]),
    to_device=use_generation_cuda ? CUDA.cu : identity,
)
use_generation_cuda && (CUDA.synchronize(); GC.gc(); CUDA.reclaim())

results = Dict(
    :memory_none => EvalItemResult[],
    :memory_retrieved => EvalItemResult[],
    :memory_distractor => EvalItemResult[],
)
trace_path = joinpath(output_dir, "$(label)_memory_trace.jsonl")
started = time_ns()
open(trace_path, "w") do trace_io
    for (position, task) in enumerate(tasks)
        # Arm 1: byte-for-byte ordinary chat, no system message and no tools.
        plain = generate_hf_text!(
            session,
            task.question;
            chat=true,
            enable_thinking=false,
            max_new_tokens=Int(options[:max_new_tokens]),
            max_prompt_tokens=Int(options[:context]) - Int(options[:max_new_tokens]),
            strategy=:greedy,
        )
        plain_result = eval_result(
            task,
            :memory_none,
            plain.prompt,
            plain.completion,
            plain.stop_reason,
            (;
                prompt_ids=plain.prompt_ids,
                extra=(; memory_ids=String[], memory_context_sha256=""),
            ),
        )
        push!(results[:memory_none], plain_result)
        JSON3.write(trace_io, (;
            task=task.id,
            arm="memory_none",
            prompt_sha256=plain_result.prompt_sha256,
            prompt_token_count=plain_result.prompt_token_count,
            completion=plain.completion,
            generated_ids=plain.generated_ids,
            stop_reason=String(plain.stop_reason),
            memory_ids=String[],
            memory_context_sha256="",
        ))
        println(trace_io)

        # Arm 2: actual top-1 embedding retrieval.
        retrieved_context = retrieved_contexts[position]
        retrieved = run_qwen3_memory_loop(
            session,
            task.question,
            retrieved_context;
            max_steps=1,
            max_new_tokens=Int(options[:max_new_tokens]),
            enable_thinking=false,
            strategy=:greedy,
        )
        retrieved_step = only(retrieved.steps)
        retrieved_result = eval_result(
            task,
            :memory_retrieved,
            retrieved_step.prompt,
            retrieved.answer,
            retrieved.stop_reason,
            (;
                prompt_ids=encode(
                    session.tokenizer,
                    retrieved_step.prompt;
                    add_special_tokens=false,
                ),
                extra=(;
                    memory_ids=[hit.id for hit in retrieved_context.hits],
                    memory_scores=[hit.score for hit in retrieved_context.hits],
                    memory_context_sha256=retrieved_context.rendered_sha256,
                ),
            ),
        )
        push!(results[:memory_retrieved], retrieved_result)
        JSON3.write(trace_io, (;
            task=task.id,
            arm="memory_retrieved",
            prompt_sha256=retrieved_result.prompt_sha256,
            prompt_token_count=retrieved_result.prompt_token_count,
            completion=retrieved.answer,
            generated_ids=retrieved_step.generated_ids,
            stop_reason=String(retrieved.stop_reason),
            memory_ids=[hit.id for hit in retrieved_context.hits],
            memory_scores=[hit.score for hit in retrieved_context.hits],
            memory_context_sha256=retrieved_context.rendered_sha256,
        ))
        println(trace_io)

        # Arm 3: irrelevant content with the same total prompt-token count as arm 2.
        distractor_context, matched_tokens = matched_distractor_context(
            index,
            session.tokenizer,
            task,
            retrieved_context,
        )
        matched_tokens == retrieved_result.prompt_token_count || error(
            "matched-control token count drift for $(task.id)",
        )
        distractor = run_qwen3_memory_loop(
            session,
            task.question,
            distractor_context;
            max_steps=1,
            max_new_tokens=Int(options[:max_new_tokens]),
            enable_thinking=false,
            strategy=:greedy,
        )
        distractor_step = only(distractor.steps)
        distractor_result = eval_result(
            task,
            :memory_distractor,
            distractor_step.prompt,
            distractor.answer,
            distractor.stop_reason,
            (;
                prompt_ids=encode(
                    session.tokenizer,
                    distractor_step.prompt;
                    add_special_tokens=false,
                ),
                extra=(;
                    memory_ids=[hit.id for hit in distractor_context.hits],
                    memory_context_sha256=distractor_context.rendered_sha256,
                    matched_prompt_tokens=matched_tokens,
                ),
            ),
        )
        distractor_result.prompt_token_count == retrieved_result.prompt_token_count ||
            error("retrieved/distractor prompt lengths differ for $(task.id)")
        push!(results[:memory_distractor], distractor_result)
        JSON3.write(trace_io, (;
            task=task.id,
            arm="memory_distractor",
            prompt_sha256=distractor_result.prompt_sha256,
            prompt_token_count=distractor_result.prompt_token_count,
            completion=distractor.answer,
            generated_ids=distractor_step.generated_ids,
            stop_reason=String(distractor.stop_reason),
            memory_ids=[hit.id for hit in distractor_context.hits],
            memory_context_sha256=distractor_context.rendered_sha256,
        ))
        println(trace_io)
        flush(trace_io)
        println(stderr, "$(position)/$(length(tasks)) $(task.id)")
    end
end
seconds = (time_ns() - started) / 1.0e9

for (arm, arm_results) in results
    write_items(joinpath(output_dir, "$(label)_$(arm)_items.jsonl"), arm_results)
end

reports = Dict(arm => accuracy_report(arm_results) for (arm, arm_results) in results)
none_vs_retrieved = paired_comparison(results[:memory_none], results[:memory_retrieved])
distractor_vs_retrieved = paired_comparison(
    results[:memory_distractor],
    results[:memory_retrieved],
)
summary_path = joinpath(output_dir, "$(label)_memory_summary.json")
open(summary_path, "w") do io
    JSON3.pretty(io, JSON3.write((;
        schema_version=1,
        label,
        generation_model=abspath(generation_model_dir),
        embedding_model=abspath(embedding_model_dir),
        generation_device,
        generation_revision=String(options[:revision]),
        embedding_revision=qwen3_embedding_spec().revision,
        embedding_device=use_embedding_cuda ? "cuda" : "cpu",
        tasks_sha256=task_set.sha256,
        journal=journal_path,
        store_sha256,
        records=length(store.records),
        evaluated_tasks=length(tasks),
        context_tokens=Int(options[:context]),
        max_new_tokens=Int(options[:max_new_tokens]),
        seconds,
        retrieval=retrieval_report,
        no_memory=reports[:memory_none],
        retrieved_memory=reports[:memory_retrieved],
        matched_distractor=reports[:memory_distractor],
        no_memory_vs_retrieved=none_vs_retrieved,
        distractor_vs_retrieved,
    )))
    println(io)
end

for arm in (:memory_none, :memory_retrieved, :memory_distractor)
    report = reports[arm]
    println(stderr, "$arm $(report.correct)/$(report.total) = " *
        string(round(report.accuracy; digits=4)))
end
println(stderr, "retrieval top1 $(retrieval_report.top1_correct)/$(retrieval_report.total); " *
                "completed in $(round(seconds; digits=1))s")
