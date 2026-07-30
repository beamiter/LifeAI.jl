#!/usr/bin/env julia

using HTTP
using JSON3
using LifeAI

function usage()
    println(stderr, """
usage:
  julia --project=. --startup-file=no scripts/qwen3_xla_client.jl MODEL_DIR --prompt TEXT [options]

options:
  --endpoint URL             default: http://127.0.0.1:11435/api/generate
  --system TEXT              add one system message before the user message
  --raw                      send TEXT without the Qwen3 chat template
  --thinking                 enable thinking in the client-side chat template
  --max-new-tokens N         default: 128 (server maximum: 512)
  --context N                compiled server context (default: 4096)
  --timeout SECONDS          read timeout (default: 600)
  -h, --help                 show this help

Start run_qwen3_xla_server.jl once, then invoke this lightweight client from
any terminal. Only tokenizer assets are read by the client; model weights and
compiled executables remain in the resident server process.
""")
end

function parse_args(args)
    isempty(args) && (usage(); exit(2))
    args[1] in ("-h", "--help") && (usage(); exit())
    startswith(args[1], "--") &&
        error("MODEL_DIR must be the first argument")
    options = Dict{Symbol,Any}(
        :prompt => nothing,
        :endpoint => "http://127.0.0.1:11435/api/generate",
        :system => nothing,
        :raw => false,
        :thinking => false,
        :max_new_tokens => 128,
        :context => 4096,
        :timeout => 600,
    )
    index = 2
    while index <= length(args)
        option = args[index]
        if option == "--raw"
            options[:raw] = true
        elseif option == "--thinking"
            options[:thinking] = true
        elseif option in (
            "--prompt",
            "--endpoint",
            "--system",
            "--max-new-tokens",
            "--context",
            "--timeout",
        )
            index < length(args) || error("$option requires a value")
            value = args[index + 1]
            key = Symbol(replace(option[3:end], "-" => "_"))
            options[key] = if option in ("--max-new-tokens", "--context")
                parse(Int, value)
            elseif option == "--timeout"
                parse(Int, value)
            else
                value
            end
            index += 1
        elseif option in ("-h", "--help")
            usage()
            exit()
        else
            error("unknown option: $option")
        end
        index += 1
    end
    options[:prompt] === nothing && error("--prompt is required")
    Int(options[:max_new_tokens]) > 0 ||
        error("--max-new-tokens must be positive")
    Int(options[:context]) > 0 || error("--context must be positive")
    Int(options[:timeout]) > 0 || error("--timeout must be positive")
    options[:raw] && options[:system] !== nothing &&
        error("--system cannot be combined with --raw")
    options[:raw] && options[:thinking] &&
        error("--thinking cannot be combined with --raw")
    return abspath(args[1]), options
end

function message_string(message, key, default=nothing)
    haskey(message, key) || return default
    value = message[key]
    value isa AbstractString || error("response field '$key' must be a string")
    return String(value)
end

find_newline(bytes) =
    something(findfirst(==(UInt8('\n')), bytes), 0)

model_dir, options = parse_args(ARGS)
prompt = String(options[:prompt])
raw_prompt = if options[:raw]
    prompt
else
    tokenizer = load_hf_qwen3_tokenizer(model_dir)
    messages = options[:system] === nothing ?
        [(role="user", content=prompt)] :
        [
            (role="system", content=String(options[:system])),
            (role="user", content=prompt),
        ]
    apply_qwen3_chat_template(
        tokenizer,
        messages;
        add_generation_prompt=true,
        enable_thinking=Bool(options[:thinking]),
    )
end

payload = (;
    model="Qwen/Qwen3-8B",
    prompt=raw_prompt,
    raw=true,
    stream=true,
    think=false,
    keep_alive="resident",
    options=(;
        temperature=0.0,
        repeat_penalty=1.0,
        repeat_last_n=0,
        num_predict=Int(options[:max_new_tokens]),
        num_ctx=Int(options[:context]),
    ),
)
body = JSON3.write(payload)
headers = [
    "Accept" => "application/x-ndjson",
    "Content-Type" => "application/json; charset=utf-8",
    "User-Agent" => "LifeAI-Qwen3-XLA-Client/1",
]
final_message = Ref{Any}(nothing)
HTTP.open(
    "POST",
    String(options[:endpoint]),
    headers;
    status_exception=false,
    readtimeout=Int(options[:timeout]),
) do stream
    write(stream, body)
    closewrite(stream)
    response = HTTP.startread(stream)
    if response.status != 200
        detail = String(read(stream))
        error("server returned HTTP $(response.status): $detail")
    end
    while !eof(stream)
        bytes = HTTP.IOExtras.readuntil(stream, find_newline)
        isempty(bytes) && break
        line = strip(String(bytes))
        isempty(line) && continue
        message = JSON3.read(line)
        if haskey(message, "error")
            error("server generation failed: $(message["error"])")
        end
        piece = message_string(message, "response", "")
        isempty(piece) || (write(stdout, piece); flush(stdout))
        if get(message, "done", false) === true
            final_message[] = message
            break
        end
    end
end
final_message[] === nothing &&
    error("server closed the stream without done=true")
println()
prompt_count = Int(get(final_message[], "prompt_eval_count", 0))
eval_count = Int(get(final_message[], "eval_count", 0))
prompt_ns = Int(get(final_message[], "prompt_eval_duration", 0))
eval_ns = Int(get(final_message[], "eval_duration", 0))
decode_rate = eval_ns > 0 ? eval_count * 1.0e9 / eval_ns : 0.0
println(
    stderr,
    "prompt=$prompt_count generated=$eval_count " *
    "prefill=$(round(prompt_ns / 1.0e9; digits=3))s " *
    "decode=$(round(decode_rate; digits=2)) tok/s",
)
