#!/usr/bin/env julia

# Reactant reads these settings during module initialization. They must remain
# before every `using LifeAI` / `using Reactant`.
ENV["XLA_REACTANT_GPU_MEM_FRACTION"] =
    get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.87")
ENV["XLA_REACTANT_GPU_PREALLOCATE"] =
    get(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using LifeAI
using Reactant

const DEFAULT_PROFILE = joinpath(
    @__DIR__,
    "..",
    "configs",
    "deployment",
    "qwen3_8b_4090d_bf16_xla_daily.json",
)
const LOOPBACK_HOSTS = Set(("127.0.0.1", "localhost", "::1"))

function usage()
    println(stderr, """
usage:
  julia --project=. --startup-file=no scripts/run_qwen3_xla_server.jl MODEL_DIR [options]

options:
  --host HOST                bind address (default: 127.0.0.1)
  --port PORT                listen port (default: 11435)
  --profile PATH             deployment profile
  --max-body-bytes N         maximum JSON request body (default: 1048576)
  --allow-non-loopback       explicitly permit a non-loopback bind
  --skip-asset-verification  trust local files without the frozen hash check
  -h, --help                 show this help

The server loads and compiles one Qwen3-8B XLA session, then reuses it for
every request. It implements GET /healthz and a strict POST /api/generate
subset. The default port intentionally differs from Ollama's 11434.
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    startswith(args[1], "--") &&
        error("MODEL_DIR must be the first argument")
    model_dir = abspath(args[1])
    options = Dict{Symbol,Any}(
        :host => "127.0.0.1",
        :port => 11435,
        :profile => DEFAULT_PROFILE,
        :max_body_bytes => 1024^2,
        :allow_non_loopback => false,
        :skip_asset_verification => false,
    )
    index = 2
    while index <= length(args)
        option = args[index]
        if option == "--allow-non-loopback"
            options[:allow_non_loopback] = true
        elseif option == "--skip-asset-verification"
            options[:skip_asset_verification] = true
        elseif option in ("--host", "--port", "--profile", "--max-body-bytes")
            index < length(args) || error("$option requires a value")
            value = args[index + 1]
            key = Symbol(replace(option[3:end], "-" => "_"))
            options[key] = option in ("--port", "--max-body-bytes") ?
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
    options[:profile] = abspath(String(options[:profile]))
    return model_dir, options
end

function format_gib(bytes)
    bytes === nothing && return "unavailable"
    return "$(round(Int(bytes) / 2.0^30; digits=2))GiB"
end

model_dir, options = parse_args(ARGS)
host = String(options[:host])
port = Int(options[:port])
1 <= port <= 65535 || error("--port must be in 1:65535")
Int(options[:max_body_bytes]) > 0 ||
    error("--max-body-bytes must be positive")
if !(host in LOOPBACK_HOSTS) && !options[:allow_non_loopback]
    error(
        "refusing non-loopback bind '$host'; pass --allow-non-loopback " *
        "only after applying your own network access controls",
    )
end

profile_path = String(options[:profile])
profile = load_qwen3_deployment_profile(profile_path)
profile.model_id == "Qwen/Qwen3-8B" ||
    error("the resident XLA server requires Qwen/Qwen3-8B")
profile.variant === :qwen3_8b ||
    error("the resident XLA server requires the qwen3_8b variant")
profile.strategy === :greedy ||
    error("the resident XLA server currently supports greedy decode only")
profile.context_tokens % profile.prefill_chunk_tokens == 0 ||
    error("profile context must be divisible by its prefill chunk")

asset_manifest_path = isabspath(profile.asset_manifest) ?
    profile.asset_manifest :
    joinpath(dirname(profile_path), profile.asset_manifest)
if options[:skip_asset_verification]
    isdir(model_dir) || error("model directory does not exist: $model_dir")
    println(stderr, "WARNING: frozen asset verification skipped by request")
else
    println(stderr, "verifying frozen Qwen3-8B assets…")
    started = time_ns()
    report = verify_qwen3_deployment_assets(
        model_dir,
        asset_manifest_path;
        model_id=profile.model_id,
        revision=profile.revision,
    )
    seconds = (time_ns() - started) / 1.0e9
    println(
        stderr,
        "verified $(length(report.files)) files, " *
        "$(round(report.total_bytes / 2.0^30; digits=2))GiB in " *
        "$(round(seconds; digits=2))s",
    )
end

Reactant.set_default_backend("gpu")
println(
    stderr,
    "loading one resident $(profile.model_id) XLA session " *
    "(context=$(profile.context_tokens), chunk=$(profile.prefill_chunk_tokens))…",
)
service = load_qwen3_xla_http_service(
    model_dir;
    model_id=profile.model_id,
    context_tokens=profile.context_tokens,
    prefill_chunk_tokens=profile.prefill_chunk_tokens,
    max_new_tokens=profile.max_new_tokens,
    max_body_bytes=Int(options[:max_body_bytes]),
    revision=profile.revision,
    variant=profile.variant,
)
metrics = service.session.load_metrics
println(
    stderr,
    "loaded once in $(round(service.load_seconds; digits=2))s: " *
    "host=$(round(metrics.host_load_seconds; digits=2))s " *
    "transfer=$(round(metrics.parameter_transfer_seconds; digits=2))s " *
    "prefill-compile=$(round(metrics.prefill_compile_seconds; digits=2))s " *
    "decode-compile=$(round(metrics.decode_compile_seconds; digits=2))s",
)
println(
    stderr,
    "resident parameters=$(format_gib(metrics.parameter_logical_bytes)) " *
    "trees=$(metrics.device_parameter_tree_count) " *
    "transfers=$(metrics.device_parameter_tree_transfer_count)",
)

server = serve_qwen3_xla_http!(service; host, port)
println(stderr, "ready http://$host:$port load_count=$(service.load_count)")
println(stderr, "health: http://$host:$port/healthz")
try
    wait(server)
catch error
    error isa InterruptException || rethrow()
finally
    close(server)
end
