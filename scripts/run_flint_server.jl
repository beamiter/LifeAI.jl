#!/usr/bin/env julia
"""Loopback-only HTTP worker for native NVFP4 Flint generation."""
module FlintHTTPWorker
using LifeAI, HTTP, JSON3, Sockets, Random, Dates
using CUDA
using Profile

# Linux SIGUSR1 collects a one-second diagnostic sample on demand; it adds
# no periodic profiling to normal inference and writes only to the worker log.
Profile.peek_report[] = () -> Profile.print(stderr; format=:flat, C=true,
    sortedby=:count, mincount=5, groupby=:thread)

const MODEL_ID = "RecursiveSI/flint"
const MAX_BODY_BYTES = 1024 * 1024
struct RequestError <: Exception
    status::Int
    code::String
    message::String
end
Base.showerror(io::IO, error::RequestError) = print(io, error.message)

mutable struct Worker
    bundle::Any
    tokenizer::Any
    ready::Bool
    phase::String
    load_error::Union{Nothing,String}
    context_tokens::Int
    max_new_tokens::Int
    generation_lock::ReentrantLock
    load_ns::Int
    generator::Any
end
Worker(; context_tokens=2048, max_new_tokens=512, generator=generate_flint_tokens) =
    Worker(nothing, nothing, false, "starting", nothing, context_tokens,
        max_new_tokens, ReentrantLock(), 0, generator)

function status(worker::Worker)
    backend = worker.bundle !== nothing && hasproperty(worker.bundle,:nvfp4_backend) &&
        worker.bundle.nvfp4_backend === :native ? "cublaslt_nvfp4" : "reference"
    return (ready=worker.ready, model=MODEL_ID, context_tokens=worker.context_tokens,
        max_new_tokens=worker.max_new_tokens, phase=worker.phase, error=worker.load_error,
        native_fp4_backend=backend, inference_engine="lifeai_flint_decode_v3")
end

json(value) = JSON3.write(value)
function write_json(stream, code, value; ndjson=false)
    HTTP.setstatus(stream, code)
    HTTP.setheader(stream, "Content-Type" => (ndjson ? "application/x-ndjson; charset=utf-8" : "application/json; charset=utf-8"))
    HTTP.setheader(stream, "Cache-Control" => "no-store")
    HTTP.startwrite(stream)
    write(stream, json(value))
    ndjson && write(stream, '\n')
    flush(stream.stream)
end

function read_body(stream)
    length_header = HTTP.header(stream.message, "Content-Length", "")
    declared_large = false
    if !isempty(length_header)
        amount = tryparse(Int, length_header)
        amount === nothing && throw(RequestError(400, "invalid_content_length", "Content-Length must be an integer"))
        amount >= 0 || throw(RequestError(400, "invalid_content_length", "Content-Length must be nonnegative"))
        declared_large = amount > MAX_BODY_BYTES
    end
    body = UInt8[]
    too_large = declared_large
    # Drain a rejected body in bounded chunks so HTTP.jl can finish framing the
    # 413 response instead of resetting a connection with unread request bytes.
    while !eof(stream)
        chunk = read(stream, 8192)
        if !too_large && length(body) + length(chunk) <= MAX_BODY_BYTES
            append!(body, chunk)
        else
            too_large = true
        end
    end
    HTTP.closeread(stream)
    too_large && throw(RequestError(413, "body_too_large", "Request body exceeds 1 MiB"))
    return body
end

function prepare_request(worker::Worker, body)
    object = try
        JSON3.read(String(body), Dict{String,Any})
    catch
        throw(RequestError(400, "invalid_json", "Body must be a JSON object"))
    end
    messages = get(object, "messages", nothing)
    messages isa AbstractVector && 1 <= length(messages) <= 64 ||
        throw(RequestError(400, "invalid_messages", "messages must contain 1 to 64 text messages"))
    total_chars = 0
    normalized = NamedTuple{(:role,:content),Tuple{String,String}}[]
    for message in messages
        message isa AbstractDict || throw(RequestError(400, "invalid_message", "Each message must be an object"))
        role, content = get(message,"role",nothing), get(message,"content",nothing)
        role isa AbstractString && role in ("system","developer","user","assistant") ||
            throw(RequestError(400, "invalid_role", "Supported roles: system, developer, user, assistant"))
        content isa AbstractString || throw(RequestError(400, "invalid_content", "Message content must be text"))
        length(content) <= 8192 || throw(RequestError(413, "message_too_large", "Each message is limited to 8192 characters"))
        total_chars += length(content)
        total_chars <= 32768 || throw(RequestError(413, "messages_too_large", "Messages are limited to 32768 total characters"))
        calls = get(message,"tool_calls",nothing)
        (calls === nothing || (calls isa AbstractVector && isempty(calls))) ||
            throw(RequestError(400,"unsupported_tools","Tool calls are not supported by this worker"))
        push!(normalized, (role=String(role), content=String(content)))
    end
    streaming = get(object, "stream", true)
    streaming isa Bool || throw(RequestError(400,"invalid_stream","stream must be true or false"))
    options = get(object, "options", Dict{String,Any}())
    options isa AbstractDict || throw(RequestError(400,"invalid_options","options must be an object"))
    count = get(options,"num_predict",min(128,worker.max_new_tokens))
    count isa Integer && !(count isa Bool) && 1 <= count <= worker.max_new_tokens ||
        throw(RequestError(400,"invalid_num_predict","num_predict must be in 1:$(worker.max_new_tokens)"))
    temperature = get(options,"temperature",1.0)
    temperature isa Real && !(temperature isa Bool) && isfinite(temperature) && 0 <= temperature <= 5 ||
        throw(RequestError(400,"invalid_temperature","temperature must be finite and in [0,5]"))
    worker.ready || throw(RequestError(503,"model_not_ready",something(worker.load_error,"Flint is loading")))
    prompt = apply_flint_chat_template(worker.tokenizer, normalized)
    ids = encode(worker.tokenizer,prompt)
    length(ids) + count <= worker.context_tokens || throw(RequestError(400,"context_length_exceeded",
        "Prompt ($(length(ids)) tokens) plus output ($count) exceeds $(worker.context_tokens) tokens"))
    return (ids=ids, count=Int(count), temperature=Float32(temperature), stream=streaming)
end

# Keep incomplete UTF-8 suffixes until a later token supplies the missing bytes.
# Known-invalid complete bytes may be replaced, but an incomplete valid prefix
# must never be replaced merely because it crossed a token boundary.
mutable struct UTF8Buffer
    pending::Vector{UInt8}
end
UTF8Buffer() = UTF8Buffer(UInt8[])
function take_text!(buffer::UTF8Buffer, bytes; final=false)
    append!(buffer.pending, bytes)
    data = buffer.pending
    i = 1
    while i <= length(data)
        lead = data[i]
        width = lead <= 0x7f ? 1 : 0xc2 <= lead <= 0xdf ? 2 : 0xe0 <= lead <= 0xef ? 3 : 0xf0 <= lead <= 0xf4 ? 4 : 1
        valid_prefix = true
        for offset in 1:min(width-1,length(data)-i)
            byte = data[i+offset]
            valid_prefix &= 0x80 <= byte <= 0xbf
            if offset == 1
                valid_prefix &= lead != 0xe0 || byte >= 0xa0
                valid_prefix &= lead != 0xed || byte <= 0x9f
                valid_prefix &= lead != 0xf0 || byte >= 0x90
                valid_prefix &= lead != 0xf4 || byte <= 0x8f
            end
        end
        if valid_prefix && i+width-1 > length(data) && !final
            break
        end
        i += valid_prefix ? min(width,length(data)-i+1) : 1
    end
    consumed = i-1
    consumed == 0 && return ""
    text = LifeAI._decode_utf8(data[1:consumed]; errors=:replace)
    deleteat!(data,1:consumed)
    return text
end

function generate(worker::Worker, request; write_piece=nothing)
    started = time_ns()
    native_before = hasproperty(worker.bundle,:parameters) ?
        flint_native_fp4_stats(worker.bundle).native_calls : 0
    first_token_ns = Ref{Union{Nothing,Int}}(nothing)
    buffer = UTF8Buffer()
    callback = function(token)
        first_token_ns[] === nothing && (first_token_ns[] = time_ns())
        if write_piece !== nothing
            piece = take_text!(buffer, decode_bytes(worker.tokenizer,[token];skip_special_tokens=true))
            isempty(piece) || write_piece(piece)
        end
        return nothing
    end
    output = worker.generator(worker.bundle, request.ids; max_new_tokens=request.count,
        temperature=request.temperature, on_token=callback)
    if write_piece !== nothing
        suffix = take_text!(buffer,UInt8[];final=true)
        isempty(suffix) || write_piece(suffix)
    end
    duration = time_ns()-started
    prefill_duration = first_token_ns[] === nothing ? duration : first_token_ns[]-started
    decode_duration = max(0,duration-prefill_duration)
    decode_count = max(0,length(output)-1)
    decode_rate = decode_count > 0 && decode_duration > 0 ? decode_count/(decode_duration/1e9) : nothing
    stopped = !isempty(output) && last(output) in worker.bundle.model.eos_token_id .+ 1
    response = write_piece === nothing ? decode(worker.tokenizer,output;skip_special_tokens=true,errors=:replace) : ""
    native_stats = hasproperty(worker.bundle,:parameters) ? flint_native_fp4_stats(worker.bundle) :
        (backend=:reference,native_calls=0)
    return (model=MODEL_ID, created_at=string(now(UTC)), response=response, done=true,
        done_reason=stopped ? "stop" : "length", total_duration=duration,
        load_duration=worker.load_ns, prompt_eval_count=length(request.ids), eval_count=length(output),
        eval_duration=duration, time_to_first_token=first_token_ns[] === nothing ? nothing : prefill_duration,
        prompt_eval_duration=prefill_duration,decode_duration=decode_duration,
        lifeai=(decode_tokens_per_second=decode_rate,decode_eval_count=decode_count),
        native_fp4_backend=String(native_stats.backend),native_fp4_calls=native_stats.native_calls,
        native_fp4_calls_delta=native_stats.native_calls-native_before)
end

function handler(worker::Worker, stream::HTTP.Stream)
    started = false
    owns_lock = false
    try
        path = first(split(stream.message.target,'?';limit=2))
        if stream.message.method == "GET" && path == "/healthz"
            HTTP.closeread(stream)
            write_json(stream,200,status(worker))
            return
        end
        path == "/api/chat" || throw(RequestError(404,"not_found","Unknown worker endpoint"))
        stream.message.method == "POST" || throw(RequestError(405,"method_not_allowed","Use POST /api/chat"))
        request = prepare_request(worker,read_body(stream))
        owns_lock = trylock(worker.generation_lock)
        owns_lock || throw(RequestError(409,"worker_busy","Flint is generating another response"))
        if request.stream
            HTTP.setstatus(stream,200)
            HTTP.setheader(stream,"Content-Type"=>"application/x-ndjson; charset=utf-8")
            HTTP.setheader(stream,"Cache-Control"=>"no-store")
            HTTP.startwrite(stream)
            started = true
            piece_writer = piece -> begin
                write(stream,json((model=MODEL_ID,response=piece,done=false))*"\n")
                flush(stream.stream)
            end
            result = generate(worker,request;write_piece=piece_writer)
            write(stream,json(result)*"\n")
            flush(stream.stream)
        else
            result = generate(worker,request)
            write_json(stream,200,result)
        end
    catch error
        code = error isa RequestError ? error.status : 500
        message = error isa RequestError ? error.message : sprint(showerror,error)
        payload = (error=message,code=error isa RequestError ? error.code : "generation_failed",model=MODEL_ID,response="",done=true)
        try
            if started
                write(stream,json(payload)*"\n")
                flush(stream.stream)
            else
                HTTP.setheader(stream,"Connection"=>"close")
                write_json(stream,code,payload)
            end
        catch
            # The client may already be gone; lock cleanup still runs below.
        end
    finally
        owns_lock && unlock(worker.generation_lock)
    end
end

function load_worker!(worker::Worker; model_dir, quantized_dir=nothing, device="cuda", backend="native", save_quantized=false, warmup=true)
    started = time_ns()
    try
        mapper = if device == "cuda"
            CUDA.functional() || error("CUDA is not functional")
            CUDA.allowscalar(false)
            CUDA.cu
        else
            identity
        end
        source = quantized_dir !== nothing && isfile(joinpath(quantized_dir,"manifest.json")) ? quantized_dir : model_dir
        source === nothing && error("A model directory or existing quantized artifact is required")
        artifact = isfile(joinpath(source,"manifest.json"))
        progress = (name,index,total) -> begin
            if index == 1 || index % 250 == 0 || index == total
                println(stderr,"[Flint ",worker.phase,"] ",index,"/",total," ",name)
                flush(stderr)
                yield()
            end
        end
        worker.phase = artifact ? "loading_artifact" : "quantizing"
        println(stderr,"[Flint] ",worker.phase," ",source); flush(stderr)
        raw = artifact ? load_flint_quantized(source;device=mapper) :
            load_hf_flint_quantized(source;scheme=:nvfp4,device=mapper,
                quantize_on_device=(device == "cuda"),on_progress=progress)
        if save_quantized
            quantized_dir === nothing && error("--save-quantized requires --quantized-dir")
            artifact || save_flint_quantized(quantized_dir,raw)
        end
        worker.phase = "reclaiming_quantization_buffers"
        GC.gc()
        device == "cuda" && CUDA.reclaim()
        worker.phase = "preparing_native"
        bundle = backend == "native" ? prepare_flint_nvfp4(raw;on_progress=progress) : raw
        worker.tokenizer = load_hf_flint_tokenizer(source)
        worker.context_tokens = min(worker.context_tokens,bundle.model.max_position_embeddings)
        worker.max_new_tokens = min(worker.max_new_tokens,worker.context_tokens-1)
        worker.bundle = bundle
        if warmup && device == "cuda" && backend == "native"
            worker.phase = "warming"
            println(stderr,"[Flint] warming prefill/decode kernels"); flush(stderr)
            generate_flint_tokens(bundle,[bundle.model.bos_token_id+1];
                max_new_tokens=2,temperature=0,eos_ids=Int[])
            CUDA.synchronize()
        end
        GC.gc()
        device == "cuda" && CUDA.reclaim()
        worker.load_ns = time_ns()-started
        worker.phase = "ready"
        worker.ready = true
        println(stderr,"[Flint] ready in ",round(worker.load_ns/1e9;digits=2),"s; context=",worker.context_tokens)
        flush(stderr)
    catch error
        worker.load_error = sprint(showerror,error)
        worker.phase = "failed"
        println(stderr,"[Flint] load failed: ",worker.load_error)
        flush(stderr)
    end
    return worker
end

function main(args)
    options = Dict("port"=>"11437","model-dir"=>"","quantized-dir"=>"","device"=>"cuda","backend"=>"native")
    save = false
    warmup = true
    i = 1
    while i <= length(args)
        if args[i] == "--save-quantized"
            save = true; i += 1
        elseif args[i] == "--no-warmup"
            warmup = false; i += 1
        elseif !startswith(args[i],"--") && isempty(options["model-dir"])
            options["model-dir"] = args[i]; i += 1
        else
            startswith(args[i],"--") && i < length(args) || error("Invalid argument $(args[i])")
            key = args[i][3:end]
            haskey(options,key) || error("Unknown option --$key")
            options[key] = args[i+1]; i += 2
        end
    end
    port = parse(Int,options["port"])
    0 <= port <= 65535 || error("--port must be in 0:65535")
    options["device"] in ("cpu","cuda") || error("--device must be cpu or cuda")
    options["backend"] in ("native","reference") || error("--backend must be native or reference")
    options["backend"] == "native" && options["device"] != "cuda" && error("Native backend requires CUDA")
    model_dir = isempty(options["model-dir"]) ? nothing : abspath(options["model-dir"])
    quantized_dir = isempty(options["quantized-dir"]) ? nothing : abspath(options["quantized-dir"])
    model_dir === nothing && quantized_dir === nothing && error("Usage: run_flint_server.jl MODEL_DIR [--quantized-dir ARTIFACT] [--port 11437]")
    worker = Worker()
    server = HTTP.listen!(stream -> handler(worker,stream), ip"127.0.0.1",port;verbose=-1)
    println(stderr,"[Flint] listening on 127.0.0.1:",port,"; loading asynchronously")
    flush(stderr)
    @async load_worker!(worker;model_dir,quantized_dir,device=options["device"],backend=options["backend"],save_quantized=save,warmup)
    wait(server)
end
end # module

if abspath(PROGRAM_FILE) == @__FILE__
    FlintHTTPWorker.main(ARGS)
end
