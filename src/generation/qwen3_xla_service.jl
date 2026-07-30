using HTTP
using JSON3
using Sockets

struct Qwen3XLAServiceError <: Exception
    status::Int
    code::String
    message::String
end

Base.showerror(io::IO, error::Qwen3XLAServiceError) =
    print(io, error.message)

"""
    Qwen3XLAHTTPService

A process-local HTTP façade around one reusable Qwen3 XLA session. The session
loader is called by the constructor exactly once. All generation calls share a
single lock because the underlying static K/V cache is mutated in place.

The callable contracts are intentionally small so protocol tests can inject a
fake runtime without loading Reactant or allocating a GPU:

* `loader() -> session`
* `prompt_encoder(session, prompt) -> token_ids`
* `generator(session, token_ids, max_new_tokens, on_token) -> result`
* `token_decoder(session, token_ids) -> Vector{UInt8}`
"""
mutable struct Qwen3XLAHTTPService
    session::Any
    generator::Any
    prompt_encoder::Any
    token_decoder::Any
    model_id::String
    context_tokens::Int
    prefill_chunk_tokens::Int
    max_new_tokens::Int
    max_body_bytes::Int
    load_seconds::Float64
    load_count::Int
    generation_lock::ReentrantLock
    metrics_lock::ReentrantLock
    request_count::Int
    completed_request_count::Int
    failed_request_count::Int
    queued_request_count::Int
    active_request_count::Int
    max_active_request_count::Int
end

function _qwen3_xla_service_prompt_encoder(session, prompt)
    return encode(
        session.tokenizer,
        prompt;
        add_special_tokens=false,
    )
end

function _qwen3_xla_service_token_decoder(session, token_ids)
    return decode_bytes(
        session.tokenizer,
        token_ids;
        skip_special_tokens=true,
    )
end

function _qwen3_xla_service_generator(
    session,
    prompt_ids,
    max_new_tokens,
    on_token,
)
    return generate_hf_qwen3_bf16_xla!(
        session,
        prompt_ids;
        max_new_tokens,
        on_token,
    )
end

function Qwen3XLAHTTPService(;
    loader,
    generator=_qwen3_xla_service_generator,
    prompt_encoder=_qwen3_xla_service_prompt_encoder,
    token_decoder=_qwen3_xla_service_token_decoder,
    model_id::AbstractString="Qwen/Qwen3-8B",
    context_tokens::Integer=4096,
    prefill_chunk_tokens::Integer=64,
    max_new_tokens::Integer=512,
    max_body_bytes::Integer=1024^2,
)
    context = Int(context_tokens)
    chunk = Int(prefill_chunk_tokens)
    output = Int(max_new_tokens)
    body_limit = Int(max_body_bytes)
    !isempty(model_id) || throw(ArgumentError("model_id must not be empty"))
    context > 0 || throw(ArgumentError("context_tokens must be positive"))
    0 < chunk <= context || throw(ArgumentError(
        "prefill_chunk_tokens must be in 1:context_tokens",
    ))
    context % chunk == 0 || throw(ArgumentError(
        "context_tokens must be divisible by prefill_chunk_tokens",
    ))
    0 < output < context || throw(ArgumentError(
        "max_new_tokens must be in 1:(context_tokens - 1)",
    ))
    body_limit > 0 || throw(ArgumentError("max_body_bytes must be positive"))

    started = time_ns()
    session = loader()
    load_seconds = (time_ns() - started) / 1.0e9
    return Qwen3XLAHTTPService(
        session,
        generator,
        prompt_encoder,
        token_decoder,
        String(model_id),
        context,
        chunk,
        output,
        body_limit,
        load_seconds,
        1,
        ReentrantLock(),
        ReentrantLock(),
        0,
        0,
        0,
        0,
        0,
        0,
    )
end

"""
    load_qwen3_xla_http_service(model_dir; kwargs...)

Load and compile one production BF16 XLA session, then expose it through the
injectable HTTP service core. Callers must select the Reactant backend before
calling this function.
"""
function load_qwen3_xla_http_service(
    model_dir::AbstractString;
    model_id::AbstractString="Qwen/Qwen3-8B",
    context_tokens::Integer=4096,
    prefill_chunk_tokens::Integer=64,
    max_new_tokens::Integer=512,
    max_body_bytes::Integer=1024^2,
    revision::AbstractString="",
    variant=nothing,
)
    loader = () -> load_hf_qwen3_bf16_xla_session(
        model_dir;
        context_tokens,
        prefill_chunk_tokens,
        strategy=:greedy,
        revision,
        variant,
    )
    return Qwen3XLAHTTPService(;
        loader,
        model_id,
        context_tokens,
        prefill_chunk_tokens,
        max_new_tokens,
        max_body_bytes,
    )
end

function qwen3_xla_service_status(service::Qwen3XLAHTTPService)
    return lock(service.metrics_lock) do
        (;
            status="ok",
            ready=true,
            model=service.model_id,
            context_tokens=service.context_tokens,
            max_new_tokens=service.max_new_tokens,
            load_seconds=service.load_seconds,
            load_count=service.load_count,
            requests=(;
                total=service.request_count,
                completed=service.completed_request_count,
                failed=service.failed_request_count,
                queued=service.queued_request_count,
                active=service.active_request_count,
                max_active=service.max_active_request_count,
            ),
        )
    end
end

_qwen3_service_error(status, code, message) =
    Qwen3XLAServiceError(status, String(code), String(message))

function _qwen3_service_json_object(body)
    isempty(body) && throw(_qwen3_service_error(
        400,
        "empty_body",
        "request body must contain a JSON object",
    ))
    object = try
        JSON3.read(String(body), Dict{String,Any})
    catch
        throw(_qwen3_service_error(
            400,
            "invalid_json",
            "request body is not valid JSON",
        ))
    end
    return object
end

function _qwen3_service_reject_unknown(object, allowed, label)
    unknown = sort!([
        String(key) for key in keys(object)
        if !(String(key) in allowed)
    ])
    isempty(unknown) || throw(_qwen3_service_error(
        400,
        "unsupported_field",
        "$label contains unsupported field(s): $(join(unknown, ", "))",
    ))
    return nothing
end

function _qwen3_service_integer(value, label)
    value isa Integer && !(value isa Bool) || throw(
        _qwen3_service_error(400, "invalid_option", "$label must be an integer"),
    )
    return Int(value)
end

function _qwen3_service_real(value, label)
    value isa Real && !(value isa Bool) || throw(
        _qwen3_service_error(400, "invalid_option", "$label must be a number"),
    )
    return Float64(value)
end

function _qwen3_service_prepare_generate(
    service::Qwen3XLAHTTPService,
    request::HTTP.Request,
)
    length(request.body) <= service.max_body_bytes || throw(
        _qwen3_service_error(413, "body_too_large", "request body is too large"),
    )
    content_type = lowercase(strip(first(
        split(HTTP.header(request, "Content-Type"), ';'; limit=2),
    )))
    content_type == "application/json" || throw(_qwen3_service_error(
        415,
        "unsupported_media_type",
        "Content-Type must be application/json",
    ))

    object = _qwen3_service_json_object(request.body)
    _qwen3_service_reject_unknown(
        object,
        (
            "model",
            "prompt",
            "raw",
            "stream",
            "think",
            "keep_alive",
            "options",
        ),
        "request",
    )

    haskey(object, "model") || throw(_qwen3_service_error(
        400,
        "missing_model",
        "model is required",
    ))
    model = get(object, "model", nothing)
    model isa AbstractString || throw(_qwen3_service_error(
        400,
        "invalid_model",
        "model must be a string",
    ))
    model == service.model_id || throw(_qwen3_service_error(
        400,
        "model_mismatch",
        "this process serves only $(service.model_id)",
    ))

    haskey(object, "prompt") || throw(_qwen3_service_error(
        400,
        "missing_prompt",
        "prompt is required",
    ))
    prompt = get(object, "prompt", nothing)
    prompt isa AbstractString || throw(_qwen3_service_error(
        400,
        "invalid_prompt",
        "prompt must be a string",
    ))
    isempty(prompt) && throw(_qwen3_service_error(
        400,
        "invalid_prompt",
        "prompt must not be empty",
    ))

    get(object, "raw", nothing) === true || throw(_qwen3_service_error(
        400,
        "raw_required",
        "the LifeAI XLA endpoint requires raw=true",
    ))
    stream = get(object, "stream", true)
    stream isa Bool || throw(_qwen3_service_error(
        400,
        "invalid_stream",
        "stream must be a boolean",
    ))
    get(object, "think", false) === false || throw(_qwen3_service_error(
        400,
        "thinking_unsupported",
        "thinking mode is not supported by this greedy endpoint",
    ))
    if haskey(object, "keep_alive")
        keep_alive = object["keep_alive"]
        (
            keep_alive isa AbstractString ||
            (keep_alive isa Real && !(keep_alive isa Bool))
        ) || throw(_qwen3_service_error(
            400,
            "invalid_keep_alive",
            "keep_alive must be a string or number",
        ))
    end

    options = get(object, "options", Dict{String,Any}())
    options isa AbstractDict || throw(_qwen3_service_error(
        400,
        "invalid_options",
        "options must be a JSON object",
    ))
    _qwen3_service_reject_unknown(
        options,
        (
            "temperature",
            "repeat_penalty",
            "repeat_last_n",
            "num_predict",
            "num_ctx",
            "seed",
        ),
        "options",
    )
    temperature = _qwen3_service_real(
        get(options, "temperature", 0.0),
        "options.temperature",
    )
    temperature == 0.0 || throw(_qwen3_service_error(
        400,
        "sampling_unsupported",
        "options.temperature must be 0 for greedy XLA generation",
    ))
    repeat_penalty = _qwen3_service_real(
        get(options, "repeat_penalty", 1.0),
        "options.repeat_penalty",
    )
    repeat_penalty == 1.0 || throw(_qwen3_service_error(
        400,
        "repeat_penalty_unsupported",
        "options.repeat_penalty must be 1",
    ))
    repeat_last_n = _qwen3_service_integer(
        get(options, "repeat_last_n", 0),
        "options.repeat_last_n",
    )
    repeat_last_n == 0 || throw(_qwen3_service_error(
        400,
        "repeat_penalty_unsupported",
        "options.repeat_last_n must be 0",
    ))
    output = _qwen3_service_integer(
        get(options, "num_predict", service.max_new_tokens),
        "options.num_predict",
    )
    0 < output <= service.max_new_tokens || throw(_qwen3_service_error(
        400,
        "invalid_num_predict",
        "options.num_predict must be in 1:$(service.max_new_tokens)",
    ))
    context = _qwen3_service_integer(
        get(options, "num_ctx", service.context_tokens),
        "options.num_ctx",
    )
    context == service.context_tokens || throw(_qwen3_service_error(
        400,
        "context_mismatch",
        "options.num_ctx must equal the compiled context " *
        "$(service.context_tokens)",
    ))
    haskey(options, "seed") &&
        _qwen3_service_integer(options["seed"], "options.seed")

    prompt_ids = try
        Int.(vec(collect(service.prompt_encoder(service.session, String(prompt)))))
    catch error
        throw(_qwen3_service_error(
            400,
            "invalid_prompt",
            "prompt tokenization failed: $(sprint(showerror, error))",
        ))
    end
    isempty(prompt_ids) && throw(_qwen3_service_error(
        400,
        "invalid_prompt",
        "prompt produced no tokens",
    ))
    window_plan = try
        plan_qwen3_xla_window(
            length(prompt_ids),
            output;
            context_tokens=service.context_tokens,
            chunk_tokens=service.prefill_chunk_tokens,
        )
    catch error
        error isa ArgumentError || rethrow()
        throw(_qwen3_service_error(
            400,
            "context_length_exceeded",
            sprint(showerror, error),
        ))
    end
    return (;
        prompt=String(prompt),
        prompt_ids,
        max_new_tokens=output,
        stream,
        window_plan,
    )
end

function _qwen3_service_result_field(result, name::Symbol, default=nothing)
    if hasproperty(result, name)
        return getproperty(result, name)
    elseif result isa AbstractDict
        haskey(result, name) && return result[name]
        haskey(result, String(name)) && return result[String(name)]
    end
    return default
end

function _qwen3_service_bytes(value)
    value isa AbstractString && return Vector{UInt8}(codeunits(value))
    return UInt8.(vec(collect(value)))
end

function _qwen3_service_metric_update(update, service)
    return lock(service.metrics_lock) do
        update()
    end
end

function _qwen3_service_generate(
    service::Qwen3XLAHTTPService,
    prepared;
    piece_writer=nothing,
)
    request_started = time_ns()
    request_id = _qwen3_service_metric_update(service) do
        service.request_count += 1
        service.queued_request_count += 1
        service.request_count
    end

    return lock(service.generation_lock) do
        queue_duration_ns = time_ns() - request_started
        _qwen3_service_metric_update(service) do
            service.queued_request_count -= 1
            service.active_request_count += 1
            service.max_active_request_count = max(
                service.max_active_request_count,
                service.active_request_count,
            )
        end

        pending_bytes = UInt8[]
        emitted_bytes = UInt8[]
        on_token = if piece_writer === nothing
            nothing
        else
            function (token_id, _...)
                token_bytes = _qwen3_service_bytes(
                    service.token_decoder(service.session, [token_id]),
                )
                append!(pending_bytes, token_bytes)
                if !isempty(pending_bytes) && isvalid(String, pending_bytes)
                    piece = String(copy(pending_bytes))
                    piece_writer(piece)
                    append!(emitted_bytes, pending_bytes)
                    empty!(pending_bytes)
                end
                return nothing
            end
        end

        try
            result = service.generator(
                service.session,
                prepared.prompt_ids,
                prepared.max_new_tokens,
                on_token,
            )
            completion = String(_qwen3_service_result_field(
                result,
                :completion,
                "",
            ))
            if piece_writer !== nothing
                completion_bytes = Vector{UInt8}(codeunits(completion))
                emitted_is_prefix =
                    length(emitted_bytes) <= length(completion_bytes) &&
                    emitted_bytes == completion_bytes[1:length(emitted_bytes)]
                emitted_is_prefix || error(
                    "streamed token bytes do not match the final completion",
                )
                if length(emitted_bytes) < length(completion_bytes)
                    remainder = String(copy(
                        completion_bytes[(length(emitted_bytes) + 1):end],
                    ))
                    piece_writer(remainder)
                end
            end

            generated_ids = Int.(vec(collect(_qwen3_service_result_field(
                result,
                :generated_ids,
                Int[],
            ))))
            prefill_seconds = Float64(_qwen3_service_result_field(
                result,
                :prefill_seconds,
                0.0,
            ))
            decode_seconds = Float64(_qwen3_service_result_field(
                result,
                :decode_seconds,
                0.0,
            ))
            stop_reason = _qwen3_service_result_field(
                result,
                :stop_reason,
                :length,
            )
            decode_token_count = max(length(generated_ids) - 1, 0)
            decode_tokens_per_second = decode_seconds > 0 ?
                decode_token_count / decode_seconds : 0.0
            actual_cache_tokens =
                prepared.window_plan.prompt_bucket_tokens +
                length(generated_ids) - 1
            done_reason = stop_reason in (:eos, "eos", :stop, "stop") ?
                "stop" : "length"
            total_duration_ns = time_ns() - request_started
            final = (;
                model=service.model_id,
                response=prepared.stream ? "" : completion,
                done=true,
                done_reason,
                total_duration=total_duration_ns,
                load_duration=0,
                prompt_eval_count=length(prepared.prompt_ids),
                prompt_eval_duration=round(Int, prefill_seconds * 1.0e9),
                eval_count=length(generated_ids),
                eval_duration=round(Int, decode_seconds * 1.0e9),
                lifeai=(;
                    request_id,
                    queue_duration_ns,
                    prefill_seconds,
                    decode_seconds,
                    decode_token_count,
                    decode_tokens_per_second,
                    prompt_bucket_tokens=
                        prepared.window_plan.prompt_bucket_tokens,
                    cache_tokens=actual_cache_tokens,
                ),
            )
            _qwen3_service_metric_update(service) do
                service.completed_request_count += 1
            end
            return final
        catch
            _qwen3_service_metric_update(service) do
                service.failed_request_count += 1
            end
            rethrow()
        finally
            _qwen3_service_metric_update(service) do
                service.active_request_count -= 1
            end
        end
    end
end

function _qwen3_service_path(target)
    return first(split(String(target), '?'; limit=2))
end

function _qwen3_service_route(service, request)
    method = uppercase(String(request.method))
    path = _qwen3_service_path(request.target)
    if path == "/healthz"
        method == "GET" || throw(_qwen3_service_error(
            405,
            "method_not_allowed",
            "use GET for /healthz",
        ))
        return (:health, nothing)
    elseif path == "/api/generate"
        method == "POST" || throw(_qwen3_service_error(
            405,
            "method_not_allowed",
            "use POST for /api/generate",
        ))
        return (:generate, _qwen3_service_prepare_generate(service, request))
    end
    throw(_qwen3_service_error(404, "not_found", "endpoint not found"))
end

_qwen3_service_json(value) = JSON3.write(value)

function _qwen3_service_json_line(value)
    return _qwen3_service_json(value) * "\n"
end

function _qwen3_service_error_payload(error::Qwen3XLAServiceError)
    return (error=error.message, code=error.code)
end

function _qwen3_service_internal_error_payload()
    return (error="generation failed", code="generation_failed")
end

function _qwen3_service_response(status, content_type, body)
    return HTTP.Response(
        status,
        [
            "Content-Type" => content_type,
            "Cache-Control" => "no-store",
        ],
        body,
    )
end

"""
    qwen3_xla_http_handler(service, request) -> HTTP.Response

Handle one request as a regular buffered `HTTP.Handler`. A `stream=true`
request still uses Ollama NDJSON framing, while `serve_qwen3_xla_http!` uses
the stream handler below to flush each token as it is produced.
"""
function qwen3_xla_http_handler(
    service::Qwen3XLAHTTPService,
    request::HTTP.Request,
)
    try
        route, prepared = _qwen3_service_route(service, request)
        if route === :health
            return _qwen3_service_response(
                200,
                "application/json; charset=utf-8",
                _qwen3_service_json(qwen3_xla_service_status(service)),
            )
        elseif prepared.stream
            output = IOBuffer()
            piece_writer = piece -> write(
                output,
                _qwen3_service_json_line((;
                    model=service.model_id,
                    response=piece,
                    done=false,
                )),
            )
            final = _qwen3_service_generate(
                service,
                prepared;
                piece_writer,
            )
            write(output, _qwen3_service_json_line(final))
            return _qwen3_service_response(
                200,
                "application/x-ndjson; charset=utf-8",
                take!(output),
            )
        else
            final = _qwen3_service_generate(service, prepared)
            return _qwen3_service_response(
                200,
                "application/json; charset=utf-8",
                _qwen3_service_json(final),
            )
        end
    catch error
        if error isa Qwen3XLAServiceError
            return _qwen3_service_response(
                error.status,
                "application/json; charset=utf-8",
                _qwen3_service_json(_qwen3_service_error_payload(error)),
            )
        end
        return _qwen3_service_response(
            500,
            "application/json; charset=utf-8",
            _qwen3_service_json(_qwen3_service_internal_error_payload()),
        )
    end
end

function _qwen3_service_read_body(stream, maximum_bytes)
    body = UInt8[]
    too_large = false
    while !eof(stream)
        chunk = readavailable(stream)
        if !too_large && length(body) + length(chunk) <= maximum_bytes
            append!(body, chunk)
        else
            too_large = true
        end
    end
    HTTP.closeread(stream)
    too_large && throw(_qwen3_service_error(
        413,
        "body_too_large",
        "request body is too large",
    ))
    return body
end

function _qwen3_service_start_stream(
    stream,
    status,
    content_type,
)
    HTTP.setstatus(stream, status)
    HTTP.setheader(stream, "Content-Type" => content_type)
    HTTP.setheader(stream, "Cache-Control" => "no-store")
    HTTP.startwrite(stream)
    return nothing
end

function _qwen3_service_write_response(stream, response::HTTP.Response)
    _qwen3_service_start_stream(
        stream,
        response.status,
        HTTP.header(response, "Content-Type"),
    )
    write(stream, response.body)
    return nothing
end

"""
    qwen3_xla_http_stream_handler(service, stream)

Low-level HTTP.jl stream handler. Validation errors are returned as JSON before
the response starts. Once NDJSON generation starts, a runtime or client-write
failure is represented by one final JSON error object when the connection is
still writable; the generation lock is always released.
"""
function qwen3_xla_http_stream_handler(
    service::Qwen3XLAHTTPService,
    stream::HTTP.Stream,
)
    started = false
    try
        request = stream.message
        request.body = _qwen3_service_read_body(
            stream,
            service.max_body_bytes,
        )
        route, prepared = _qwen3_service_route(service, request)
        if route === :health
            response = _qwen3_service_response(
                200,
                "application/json; charset=utf-8",
                _qwen3_service_json(qwen3_xla_service_status(service)),
            )
            _qwen3_service_write_response(stream, response)
            return nothing
        elseif !prepared.stream
            final = _qwen3_service_generate(service, prepared)
            response = _qwen3_service_response(
                200,
                "application/json; charset=utf-8",
                _qwen3_service_json(final),
            )
            _qwen3_service_write_response(stream, response)
            return nothing
        end

        _qwen3_service_start_stream(
            stream,
            200,
            "application/x-ndjson; charset=utf-8",
        )
        started = true
        piece_writer = function (piece)
            write(
                stream,
                _qwen3_service_json_line((;
                    model=service.model_id,
                    response=piece,
                    done=false,
                )),
            )
            flush(stream)
            return nothing
        end
        final = _qwen3_service_generate(
            service,
            prepared;
            piece_writer,
        )
        write(stream, _qwen3_service_json_line(final))
        flush(stream)
        return nothing
    catch error
        if started
            payload = error isa Qwen3XLAServiceError ?
                _qwen3_service_error_payload(error) :
                _qwen3_service_internal_error_payload()
            try
                write(
                    stream,
                    _qwen3_service_json_line(merge(payload, (; done=true))),
                )
                flush(stream)
            catch
            end
            return nothing
        end
        response = if error isa Qwen3XLAServiceError
            _qwen3_service_response(
                error.status,
                "application/json; charset=utf-8",
                _qwen3_service_json(_qwen3_service_error_payload(error)),
            )
        else
            _qwen3_service_response(
                500,
                "application/json; charset=utf-8",
                _qwen3_service_json(_qwen3_service_internal_error_payload()),
            )
        end
        _qwen3_service_write_response(stream, response)
        return nothing
    end
end

"""
    serve_qwen3_xla_http!(service; host="127.0.0.1", port=11435, kwargs...)

Start a non-blocking HTTP server. The loopback-only default is deliberate;
binding a non-loopback address must be an explicit caller choice. The returned
`HTTP.Server` can be waited on or closed by the CLI.
"""
function serve_qwen3_xla_http!(
    service::Qwen3XLAHTTPService;
    host::AbstractString="127.0.0.1",
    port::Integer=11435,
    server=nothing,
    verbose=-1,
    kwargs...,
)
    isempty(host) && throw(ArgumentError("host must not be empty"))
    0 <= port <= 65535 || throw(ArgumentError("port must be in 0:65535"))
    handler = stream -> qwen3_xla_http_stream_handler(service, stream)
    if server === nothing
        return HTTP.listen!(
            handler,
            String(host),
            Int(port);
            verbose,
            kwargs...,
        )
    end
    return HTTP.listen!(
        handler;
        server,
        verbose,
        kwargs...,
    )
end
