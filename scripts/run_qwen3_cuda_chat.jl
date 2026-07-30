#!/usr/bin/env julia

using LuxCUDA
using CUDA
using JSON3
using LifeAI
using Random: Xoshiro, default_rng

const DEFAULT_PROFILE = joinpath(
    @__DIR__,
    "..",
    "configs",
    "deployment",
    "qwen3_8b_4090d_bf16_daily.json",
)

function usage()
    println(stderr, """
usage:
  julia --project=. scripts/run_qwen3_cuda_chat.jl MODEL_DIR [PROFILE_JSON] [options]

options:
  --prompt TEXT          run one request instead of the interactive chat
  --system TEXT          retain one system message in chat mode
  --raw                  do not apply the Qwen3 chat template
  --thinking             enable Qwen3 thinking mode
  --greedy               override the profile with deterministic greedy decode
  --seed N               seed sampled generation for repeatable tests
  --max-new-tokens N     override the per-request output allowance
  --max-prompt-tokens N  override the retained prompt/history allowance
  --metrics PATH         write the last request metrics as JSON
  -h, --help             show this help

Interactive commands: /clear resets history, /exit ends the process.
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    model_dir = args[1]
    index = 2
    profile_path = DEFAULT_PROFILE
    if index <= length(args) && !startswith(args[index], "--")
        profile_path = args[index]
        index += 1
    end
    options = Dict{Symbol,Any}(
        :prompt => nothing,
        :system => nothing,
        :chat => true,
        :thinking => nothing,
        :greedy => false,
        :max_new_tokens => nothing,
        :max_prompt_tokens => nothing,
        :metrics => nothing,
        :seed => nothing,
    )
    while index <= length(args)
        option = args[index]
        if option == "--raw"
            options[:chat] = false
        elseif option == "--thinking"
            options[:thinking] = true
        elseif option == "--greedy"
            options[:greedy] = true
        elseif option in (
            "--prompt",
            "--system",
            "--seed",
            "--max-new-tokens",
            "--max-prompt-tokens",
            "--metrics",
        )
            index < length(args) || error("$option requires a value")
            value = args[index + 1]
            key = Symbol(replace(option[3:end], "-" => "_"))
            options[key] = option in ("--seed", "--max-new-tokens", "--max-prompt-tokens") ?
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
    return model_dir, profile_path, options
end

function request_metrics(result, session, gpu_total, gpu_free)
    return Dict(
        "prompt_tokens" => length(result.prompt_ids),
        "generated_tokens" => length(result.generated_ids),
        "dropped_messages" => result.dropped_messages,
        "stop_reason" => String(result.stop_reason),
        "strategy" => String(result.strategy),
        "prefill_seconds" => result.prefill_seconds,
        "decode_seconds" => result.decode_seconds,
        "tokens_per_second" => result.tokens_per_second,
        "session_context_tokens" => session.context_tokens,
        "session_cached_tokens" => session.position,
        "sequence_tokens" =>
            length(result.prompt_ids) + length(result.generated_ids),
        "last_selected_token_is_not_cached" => !isempty(result.generated_ids),
        "gpu_total_bytes" => gpu_total,
        "gpu_free_bytes_after_request" => gpu_free,
    )
end

function write_metrics(path, metrics)
    path === nothing && return
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        JSON3.pretty(io, metrics)
        println(io)
    end
end

function reclaim_cuda_pool!(; full_gc::Bool=true)
    CUDA.synchronize()
    GC.gc(full_gc)
    CUDA.reclaim()
    return nothing
end

model_dir, profile_path, options = parse_args(ARGS)
profile = load_qwen3_deployment_profile(profile_path)
max_new_tokens = something(options[:max_new_tokens], profile.max_new_tokens)
max_prompt_tokens = something(options[:max_prompt_tokens], profile.max_prompt_tokens)
max_new_tokens >= 0 || error("max output must be non-negative")
0 < max_prompt_tokens <= profile.context_tokens || error(
    "max prompt must be in 1:profile context",
)
max_new_tokens <= profile.context_tokens || error(
    "max output exceeds profile context",
)
max_prompt_tokens <= profile.context_tokens - max_new_tokens || error(
    "max prompt + max output exceeds profile context",
)
options[:chat] || options[:system] === nothing || error(
    "--system requires chat mode",
)
options[:prompt] === nothing && !options[:chat] && error(
    "interactive mode requires chat mode; use --prompt with --raw",
)
enable_thinking = something(options[:thinking], profile.enable_thinking)
strategy = options[:greedy] ? :greedy : profile.strategy
rng = options[:seed] === nothing ? default_rng() : Xoshiro(options[:seed])
prefill_reclaimer = function (position)
    chunk_index = cld(position, profile.prefill_chunk_tokens)
    mod(chunk_index, profile.prefill_reclaim_interval_chunks) == 0 &&
        reclaim_cuda_pool!(; full_gc=true)
    return nothing
end

asset_manifest_path = isabspath(profile.asset_manifest) ?
    profile.asset_manifest :
    joinpath(dirname(abspath(profile_path)), profile.asset_manifest)
CUDA.functional() || error("CUDA.jl is not functional")
gpu_total = Int(CUDA.total_memory())
gpu_total >= profile.minimum_gpu_bytes || error(
    "GPU has $gpu_total bytes, profile requires at least $(profile.minimum_gpu_bytes)",
)
spec = qwen3_dense_spec(profile.variant)
expected_parameter_bytes = qwen3_dense_parameter_count(spec) * 2
expected_kv_bytes = qwen3_kv_cache_bytes(spec, profile.context_tokens)
minimum_free_bytes = try
    Base.checked_add(
        Base.checked_add(expected_parameter_bytes, expected_kv_bytes),
        profile.workspace_reserve_bytes,
    )
catch err
    err isa OverflowError || rethrow()
    Base.error("profile free-memory preflight budget overflows Int")
end
gpu_free_before = Int(CUDA.free_memory())
gpu_free_before >= minimum_free_bytes || error(
    "GPU has $gpu_free_before free bytes; this profile requires at least " *
    "$minimum_free_bytes (BF16 parameters + KV + workspace reserve)",
)

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
gpu_free_before = Int(CUDA.free_memory())
gpu_free_before >= minimum_free_bytes || error(
    "GPU free memory fell below the profile preflight budget during asset verification",
)

println(stderr, "LifeAI loading $(profile.model_id) $(profile.revision)")
println(
    stderr,
    "GPU=$(CUDA.name(CUDA.device())) context=$(profile.context_tokens) " *
    "prompt≤$max_prompt_tokens output≤$max_new_tokens chunk=$(profile.prefill_chunk_tokens)",
)
println(
    stderr,
    "preflight free=$(round(gpu_free_before / 2.0^30; digits=2))GiB " *
    "required=$(round(minimum_free_bytes / 2.0^30; digits=2))GiB",
)
load_started = time_ns()
session = load_hf_qwen3_bf16_session(
    model_dir;
    context_tokens=profile.context_tokens,
    prefill_chunk_tokens=profile.prefill_chunk_tokens,
    revision=profile.revision,
    variant=profile.variant,
    to_device=CUDA.cu,
)
CUDA.synchronize()
GC.gc()
CUDA.reclaim()
load_seconds = (time_ns() - load_started) / 1.0e9
gpu_free_ready = Int(CUDA.free_memory())
gpu_free_ready >= profile.workspace_reserve_bytes || error(
    "ready GPU free memory $gpu_free_ready is below the configured " *
    "workspace reserve $(profile.workspace_reserve_bytes)",
)
println(
    stderr,
    "ready load=$(round(load_seconds; digits=2))s " *
    "KV=$(round(qwen3_kv_cache_bytes(session.model, session.context_tokens) / 2.0^20; digits=1))MiB " *
    "free=$(round(gpu_free_ready / 2.0^30; digits=2))GiB",
)

function run_request(input)
    generated_count = Ref(0)
    on_token = function (token_id)
        write(stdout, decode_bytes(
            session.tokenizer,
            [token_id];
            skip_special_tokens=true,
        ))
        flush(stdout)
        generated_count[] += 1
        mod(generated_count[], profile.decode_reclaim_interval_tokens) == 0 &&
            reclaim_cuda_pool!(; full_gc=false)
    end
    result = generate_hf_text!(
        session,
        input;
        chat=options[:chat],
        enable_thinking,
        max_new_tokens,
        max_prompt_tokens,
        strategy,
        rng,
        temperature=profile.temperature,
        top_k=profile.top_k,
        top_p=profile.top_p,
        on_token,
        on_prefill_chunk=prefill_reclaimer,
    )
    CUDA.synchronize()
    cleanup_started = time_ns()
    reclaim_cuda_pool!()
    cleanup_seconds = (time_ns() - cleanup_started) / 1.0e9
    println()
    metrics = request_metrics(
        result,
        session,
        gpu_total,
        Int(CUDA.free_memory()),
    )
    metrics["dropped_messages"] > 0 && println(
        stderr,
        "history compacted: dropped $(metrics["dropped_messages"]) old message(s)",
    )
    println(
        stderr,
        "prompt=$(metrics["prompt_tokens"]) generated=$(metrics["generated_tokens"]) " *
        "prefill=$(round(result.prefill_seconds; digits=3))s " *
        "decode=$(round(result.tokens_per_second; digits=2)) tok/s " *
        "free=$(round(metrics["gpu_free_bytes_after_request"] / 2.0^30; digits=2))GiB",
    )
    write_metrics(options[:metrics], merge(metrics, Dict(
        "asset_check_seconds" => asset_check_seconds,
        "load_seconds" => load_seconds,
        "post_request_cleanup_seconds" => cleanup_seconds,
    )))
    return result
end

if options[:prompt] !== nothing
    input = options[:system] === nothing || !options[:chat] ?
        options[:prompt] :
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
println(stderr, "interactive chat ready (/clear, /exit)")
while true
    print("you> ")
    flush(stdout)
    line = try
        readline()
    catch error
        error isa EOFError && break
        rethrow()
    end
    command = strip(line)
    command == "/exit" && break
    if command == "/clear"
        empty!(history)
        options[:system] === nothing || push!(
            history,
            (role="system", content=String(options[:system])),
        )
        reset_hf_qwen3_bf16_session!(session)
        println(stderr, "history cleared")
        continue
    end
    isempty(command) && continue
    push!(history, (role="user", content=line))
    print("lifeai> ")
    result = run_request(history)
    history = convert(
        Vector{NamedTuple{(:role, :content),Tuple{String,String}}},
        collect(result.messages),
    )
    push!(history, (role="assistant", content=result.completion))
end
