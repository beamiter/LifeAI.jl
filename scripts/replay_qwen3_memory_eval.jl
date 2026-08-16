#!/usr/bin/env julia

# Chapter 39 — replay all prompt/context bytes from persisted source records and
# the recorded generated trajectory. No embedding or generation model is loaded.

using JSON3
using LifeAI

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/replay_qwen3_memory_eval.jl TASKS JOURNAL TRACE TOKENIZER_DIR [--out PATH]
""")
end

function main(args)
    !isempty(args) && args[1] in ("-h", "--help") && (usage(); return 0)
    length(args) >= 4 || (usage(); return 2)
    out_path = nothing
    if length(args) == 6 && args[5] == "--out"
        out_path = args[6]
    elseif length(args) != 4
        usage()
        return 2
    end

    task_set = load_agent_memory_tasks(args[1])
    store = load_agent_memory_store(args[2])
    rows = [JSON3.read(line) for line in eachline(args[3]) if !isempty(strip(line))]
    tokenizer = load_hf_qwen3_tokenizer(args[4])

    expected_arms = ("memory_none", "memory_retrieved", "memory_distractor")
    length(rows) == length(task_set.tasks) * length(expected_arms) || error(
        "trace has $(length(rows)) rows; expected $(length(task_set.tasks) * length(expected_arms))",
    )
    by_key = Dict{Tuple{String,String},Any}()
    for row in rows
        key = (String(row.task), String(row.arm))
        haskey(by_key, key) && error("duplicate trace row for $(repr(key))")
        by_key[key] = row
    end

    prompt_hashes_equal = 0
    prompt_tokens_equal = 0
    context_hashes_equal = 0
    generated_ids_present = 0
    for task in task_set.tasks, arm in expected_arms
        row = get(by_key, (task.id, arm), nothing)
        row === nothing && error("missing trace row for $(task.id) / $arm")
        ids = String[String(id) for id in row.memory_ids]
        context = isempty(ids) ? nothing : select_agent_memory_context(
            store,
            task.question,
            ids,
        )
        if arm == "memory_none"
            context === nothing || error("no-memory arm unexpectedly has memory IDs")
            isempty(String(row.memory_context_sha256)) || error(
                "no-memory arm unexpectedly has a context digest",
            )
            context_hashes_equal += 1
        else
            context === nothing && error("memory arm has no persisted memory IDs")
            context.rendered_sha256 == String(row.memory_context_sha256) || error(
                "memory context digest mismatch for $(task.id) / $arm",
            )
            context_hashes_equal += 1
        end
        prompt = apply_qwen3_chat_template(
            tokenizer,
            LifeAI._agent_initial_messages(task.question; memory_context=context);
            add_generation_prompt=true,
            enable_thinking=false,
        )
        LifeAI._sha256_hex(prompt) == String(row.prompt_sha256) || error(
            "prompt digest mismatch for $(task.id) / $arm",
        )
        prompt_hashes_equal += 1
        prompt_ids = encode(tokenizer, prompt; add_special_tokens=false)
        length(prompt_ids) == Int(row.prompt_token_count) || error(
            "prompt token count mismatch for $(task.id) / $arm",
        )
        prompt_tokens_equal += 1
        !isempty(row.generated_ids) || error(
            "recorded generation is empty for $(task.id) / $arm",
        )
        generated_ids_present += 1
    end

    report = (;
        schema_version=1,
        tasks_sha256=task_set.sha256,
        store_sha256=agent_memory_fingerprint(store),
        rows=length(rows),
        prompt_hashes_equal,
        prompt_tokens_equal,
        context_hashes_equal,
        generated_ids_present,
        replayed=true,
    )

    if out_path !== nothing
        open(out_path, "w") do io
            JSON3.pretty(io, JSON3.write(report))
            println(io)
        end
    end
    println(JSON3.write(report))
    return 0
end

exit(main(ARGS))
