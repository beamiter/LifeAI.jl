#!/usr/bin/env julia

# Reactant reads these settings while it initializes. Keep the 4090D daily
# defaults conservative and allow callers to override either value.
ENV["XLA_REACTANT_GPU_MEM_FRACTION"] =
    get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.87")
ENV["XLA_REACTANT_GPU_PREALLOCATE"] =
    get(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using LifeAI
using Random: Xoshiro, default_rng
using Reactant

const DEFAULT_PROFILE = joinpath(
    @__DIR__,
    "..",
    "configs",
    "deployment",
    "qwen3_8b_4090d_bf16_xla_daily.json",
)

function usage()
    println(stderr, """
usage:
  julia --project=. --startup-file=no scripts/run_qwen3_xla_chat.jl MODEL_DIR [options]

options:
  --prompt TEXT              run one request instead of the interactive chat
  --system TEXT              retain one system message
  --thinking                 enable Qwen3 thinking mode
  --greedy                   force deterministic greedy decode
  --seed N                   seed generation (greedy remains deterministic)
  --max-new-tokens N         override the per-request output allowance
  --profile PATH             use a deployment profile other than the XLA daily default
  --skip-asset-verification  trust local model files without hashing the frozen manifest
  -h, --help                 show this help

Interactive commands: /clear resets history, /exit ends the process.

XLA defaults may be overridden before launch with
XLA_REACTANT_GPU_MEM_FRACTION and XLA_REACTANT_GPU_PREALLOCATE.
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    startswith(args[1], "--") && error("MODEL_DIR must be the first argument")
    model_dir = args[1]
    options = Dict{Symbol,Any}(
        :prompt => nothing,
        :system => nothing,
        :thinking => nothing,
        :greedy => false,
        :max_new_tokens => nothing,
        :seed => nothing,
        :profile => DEFAULT_PROFILE,
        :skip_asset_verification => false,
    )
    index = 2
    while index <= length(args)
        option = args[index]
        if option == "--thinking"
            options[:thinking] = true
        elseif option == "--greedy"
            options[:greedy] = true
        elseif option == "--skip-asset-verification"
            options[:skip_asset_verification] = true
        elseif option in (
            "--prompt",
            "--system",
            "--seed",
            "--max-new-tokens",
            "--profile",
        )
            index < length(args) || error("$option requires a value")
            value = args[index + 1]
            key = Symbol(replace(option[3:end], "-" => "_"))
            options[key] = option in ("--seed", "--max-new-tokens") ?
                parse(Int, value) : value
            index += 1
        elseif option in ("-h", "--help")
            usage()
            exit()
        else
            error("unknown option: $option")
        end
        index += 1
    end
    return abspath(model_dir), abspath(String(options[:profile])), options
end

function format_bytes(bytes)
    bytes === nothing && return "unavailable"
    return "$(round(Int(bytes) / 2.0^30; digits=2))GiB"
end

function print_allocator(label, snapshot)
    if snapshot === nothing
        println(stderr, "allocator $label unavailable")
        return
    end
    println(
        stderr,
        "allocator $label " *
        "in-use=$(format_bytes(snapshot.bytes_in_use)) " *
        "peak=$(format_bytes(snapshot.peak_bytes_in_use)) " *
        "reserved=$(format_bytes(snapshot.bytes_reserved)) " *
        "limit=$(format_bytes(snapshot.bytes_limit)) " *
        "largest-free=$(format_bytes(snapshot.largest_free_block_bytes))",
    )
end

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
        first_droppable < length(retained) || error(
            "newest chat message cannot fit within max prompt",
        )
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

model_dir, profile_path, options = parse_args(ARGS)
profile = load_qwen3_deployment_profile(profile_path)
strategy = options[:greedy] ? :greedy : profile.strategy
strategy === :greedy || error(
    "the Qwen3 XLA daily runtime currently supports greedy decode only; " *
    "use a greedy profile or pass --greedy",
)
max_new_tokens = something(options[:max_new_tokens], profile.max_new_tokens)
max_prompt_tokens = profile.max_prompt_tokens
max_new_tokens > 0 || error("max output must be positive for XLA generation")
max_new_tokens <= profile.context_tokens || error(
    "max output exceeds profile context",
)
max_prompt_tokens <= profile.context_tokens - max_new_tokens || error(
    "max prompt + max output exceeds profile context",
)
profile.context_tokens % profile.prefill_chunk_tokens == 0 || error(
    "XLA context must be divisible by the fixed prefill chunk size",
)
cld(max_prompt_tokens, profile.prefill_chunk_tokens) *
    profile.prefill_chunk_tokens <=
    profile.context_tokens - max_new_tokens || error(
        "padded XLA max-prompt bucket + max output exceeds profile context",
    )
enable_thinking = something(options[:thinking], profile.enable_thinking)
rng = options[:seed] === nothing ? default_rng() : Xoshiro(options[:seed])

asset_manifest_path = isabspath(profile.asset_manifest) ?
    profile.asset_manifest :
    joinpath(dirname(profile_path), profile.asset_manifest)
if options[:skip_asset_verification]
    isdir(model_dir) || error("model directory does not exist: $model_dir")
    println(stderr, "WARNING: frozen asset verification skipped by request")
else
    println(stderr, "verifying frozen model assets…")
    asset_check_started = time_ns()
    asset_report = verify_qwen3_deployment_assets(
        model_dir,
        asset_manifest_path;
        model_id=profile.model_id,
        revision=profile.revision,
    )
    asset_check_seconds = (time_ns() - asset_check_started) / 1.0e9
    println(
        stderr,
        "verified $(length(asset_report.files)) files, " *
        "$(round(asset_report.total_bytes / 2.0^30; digits=2))GiB in " *
        "$(round(asset_check_seconds; digits=2))s",
    )
end

Reactant.set_default_backend("gpu")
println(stderr, "LifeAI XLA loading $(profile.model_id) $(profile.revision)")
println(
    stderr,
    "backend=gpu context=$(profile.context_tokens) " *
    "prompt≤$max_prompt_tokens output≤$max_new_tokens " *
    "chunk=$(profile.prefill_chunk_tokens) strategy=greedy",
)
println(
    stderr,
    "XLA allocator fraction=$(ENV["XLA_REACTANT_GPU_MEM_FRACTION"]) " *
    "preallocate=$(ENV["XLA_REACTANT_GPU_PREALLOCATE"])",
)
println(
    stderr,
    "single-tree required=true device-trees=1 transfers=1 " *
    "duplicate-packed-tree=false",
)

load_started = time_ns()
session = load_hf_qwen3_bf16_xla_session(
    model_dir;
    context_tokens=profile.context_tokens,
    prefill_chunk_tokens=profile.prefill_chunk_tokens,
    strategy,
    revision=profile.revision,
    variant=profile.variant,
)
load_compile_seconds = (time_ns() - load_started) / 1.0e9
load_metrics = session.load_metrics
load_metrics.device_parameter_tree_count == 1 || error(
    "XLA loader violated the single-tree device residency contract",
)
load_metrics.device_parameter_tree_transfer_count == 1 || error(
    "XLA loader transferred more than one parameter tree",
)
!load_metrics.original_parameter_tree_constructed || error(
    "XLA loader constructed the ordinary unpacked parameter tree",
)
!load_metrics.original_parameter_tree_transferred || error(
    "XLA loader transferred the ordinary unpacked parameter tree",
)
!load_metrics.separate_packed_projection_tree_transferred || error(
    "XLA loader transferred a duplicate packed projection tree",
)
println(
    stderr,
    "ready single-tree=true parameters=" *
    "$(format_bytes(load_metrics.parameter_logical_bytes)) " *
    "tensors=$(load_metrics.parameter_tensor_count) " *
    "load+compile=$(round(load_compile_seconds; digits=2))s",
)
println(
    stderr,
    "load host=$(round(load_metrics.host_load_seconds; digits=2))s " *
    "transfer=$(round(load_metrics.parameter_transfer_seconds; digits=2))s " *
    "runtime=$(round(load_metrics.runtime_allocation_seconds; digits=2))s",
)
println(
    stderr,
    "compile prefill=$(round(load_metrics.prefill_compile_seconds; digits=2))s " *
    "decode=$(round(load_metrics.decode_compile_seconds; digits=2))s",
)
print_allocator("before-load", load_metrics.allocator_before_load)
print_allocator(
    "after-parameter-transfer",
    load_metrics.allocator_after_parameter_transfer,
)
print_allocator(
    "after-runtime-allocation",
    load_metrics.allocator_after_runtime_allocation,
)
print_allocator("ready", load_metrics.allocator_ready)

function run_request(messages)
    fitted = fit_chat_context(
        session,
        messages;
        max_prompt_tokens,
        enable_thinking,
    )
    function on_token(token_id, _...)
        write(
            stdout,
            decode_bytes(
                session.tokenizer,
                [token_id];
                skip_special_tokens=true,
            ),
        )
        flush(stdout)
        return nothing
    end
    result = generate_hf_qwen3_bf16_xla!(
        session,
        fitted.prompt_ids;
        max_new_tokens,
        temperature=profile.temperature,
        top_k=profile.top_k,
        top_p=profile.top_p,
        rng,
        on_token,
    )
    println()
    fitted.dropped_messages > 0 && println(
        stderr,
        "history compacted: dropped $(fitted.dropped_messages) old message(s)",
    )
    println(
        stderr,
        "prompt=$(length(result.prompt_ids)) " *
        "bucket=$(result.window_plan.prompt_bucket_tokens) " *
        "generated=$(length(result.generated_ids)) " *
        "prefill=$(round(result.prefill_seconds; digits=3))s " *
        "decode=$(round(result.tokens_per_second; digits=2)) tok/s " *
        "cached=$(session.position)",
    )
    print_allocator("request-before", result.allocator_before)
    print_allocator("request-after", result.allocator_after)
    return merge(result, (;
        prompt=fitted.prompt,
        messages=fitted.messages,
        dropped_messages=fitted.dropped_messages,
    ))
end

if options[:prompt] !== nothing
    input = options[:system] === nothing ?
        [(role="user", content=String(options[:prompt]))] :
        [
            (role="system", content=String(options[:system])),
            (role="user", content=String(options[:prompt])),
        ]
    run_request(input)
    exit()
end

history = NamedTuple{(:role, :content),Tuple{String,String}}[]
options[:system] === nothing || push!(
    history,
    (role="system", content=String(options[:system])),
)
println(stderr, "interactive XLA chat ready (/clear, /exit)")
while true
    print("you> ")
    flush(stdout)
    line = try
        readline()
    catch error
        error isa EOFError && break
        rethrow()
    end
    isempty(line) && eof(stdin) && break
    command = strip(line)
    command == "/exit" && break
    if command == "/clear"
        empty!(history)
        options[:system] === nothing || push!(
            history,
            (role="system", content=String(options[:system])),
        )
        reset_hf_qwen3_bf16_xla_session!(session)
        println(stderr, "history cleared")
        continue
    end
    isempty(command) && continue
    push!(history, (role="user", content=line))
    print("lifeai> ")
    result = run_request(history)
    empty!(history)
    append!(history, result.messages)
    push!(history, (role="assistant", content=result.completion))
end
