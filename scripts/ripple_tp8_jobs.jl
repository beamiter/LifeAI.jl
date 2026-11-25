# Bounded asynchronous HTTP jobs keep long prefill independent of TCP lifetimes.
# Supports both incremental polling (`GET /v1/chat/completions/jobs/{id}?after=N`)
# and live OpenAI-compatible Server-Sent Events (`POST /v1/chat/completions` with `stream: true`).
mutable struct RippleHTTPJob
    id::String
    status::String
    phase::String
    processed::Int
    total::Int
    delta_chunks::Vector{String}
    result::Any
    error::Union{Nothing,String}
    error_status::Int
    cancelled::Threads.Atomic{Bool}
    done::Base.Event
    updated::Threads.Condition
    created::Float64
    finished::Float64
    payload::Any
end

mutable struct RippleHTTPJobs
    lock::ReentrantLock
    jobs::Dict{String,RippleHTTPJob}
    queue::Vector{RippleHTTPJob}
    condition::Threads.Condition
    worker::Task
    max_pending::Int
    max_results::Int
    ttl::Float64
    timeout::Float64
    closed::Bool
end

function _ripple_http_job_worker!(manager, execute)
    while true
        job = lock(manager.lock) do
            while isempty(manager.queue) && !manager.closed
                wait(manager.condition)
            end
            isempty(manager.queue) && return nothing
            next = popfirst!(manager.queue)
            next.status = "running"
            next.phase = "tokenizing"
            notify(next.updated; all=true)
            next
        end
        job === nothing && return
        cancelled = () -> begin
            time() - job.created > manager.timeout && (job.cancelled[] = true)
            job.cancelled[]
        end
        progress = (phase, processed, total) -> lock(manager.lock) do
            job.phase = String(phase)
            job.processed = Int(processed)
            job.total = Int(total)
            notify(job.updated; all=true)
        end
        emit_delta = chunk -> begin
            text = String(chunk)
            isempty(text) && return
            lock(manager.lock) do
                push!(job.delta_chunks, text)
                notify(job.updated; all=true)
            end
        end
        try
            result = if cancelled()
                nothing
            elseif applicable(execute, job.payload, progress, cancelled, emit_delta)
                execute(job.payload, progress, cancelled, emit_delta)
            else
                execute(job.payload, progress, cancelled)
            end
            lock(manager.lock) do
                job.status = cancelled() ? "cancelled" : "completed"
                job.phase = job.status
                job.result = job.cancelled[] ? nothing : result
            end
        catch error
            lock(manager.lock) do
                job.status = cancelled() ? "cancelled" : "failed"
                job.phase = job.status
                job.error = sprint(showerror, error)
                job.error_status = error isa ArgumentError ? 400 : 500
            end
        finally
            lock(manager.lock) do
                job.finished = time()
                job.payload = nothing
                notify(job.updated; all=true)
            end
            notify(job.done)
        end
    end
end

function RippleHTTPJobs(execute; max_pending=4, max_results=8, ttl=3600.0, timeout=21600.0)
    max_pending isa Integer && !(max_pending isa Bool) && max_pending > 0 || throw(ArgumentError("max_pending must be positive"))
    max_results isa Integer && !(max_results isa Bool) && max_results > 0 || throw(ArgumentError("max_results must be positive"))
    isfinite(ttl) && ttl > 0 || throw(ArgumentError("job TTL must be finite and positive"))
    isfinite(timeout) && timeout > 0 || throw(ArgumentError("job timeout must be finite and positive"))
    guard = ReentrantLock()
    manager = RippleHTTPJobs(
        guard, Dict{String,RippleHTTPJob}(), RippleHTTPJob[], Threads.Condition(guard),
        Task(() -> nothing), Int(max_pending), Int(max_results), Float64(ttl), Float64(timeout), false,
    )
    manager.worker = Threads.@spawn _ripple_http_job_worker!(manager, execute)
    return manager
end

function ripple_job_cleanup!(manager)
    finished = sort!([job for job in values(manager.jobs) if job.finished > 0]; by=j -> j.finished)
    keep_from = max(1, length(finished) - manager.max_results + 1)
    for (index, job) in enumerate(finished)
        if index < keep_from || time() - job.finished > manager.ttl
            delete!(manager.jobs, job.id)
        end
    end
end

function ripple_job_submit!(manager, payload)
    lock(manager.lock) do
        manager.closed && throw(InvalidStateException("generation service is shutting down", :closed))
        ripple_job_cleanup!(manager)
        active = count(j -> j.status in ("queued", "running"), values(manager.jobs))
        active < manager.max_pending || throw(ArgumentError("generation queue is full; retry after the current request"))
        job = RippleHTTPJob(
            "job-" * string(uuid4()), "queued", "queued", 0, 0, String[],
            nothing, nothing, 500, Threads.Atomic{Bool}(false), Base.Event(),
            Threads.Condition(manager.lock), time(), 0.0, payload,
        )
        manager.jobs[job.id] = job
        push!(manager.queue, job)
        notify(manager.condition)
        return job
    end
end

function ripple_job_status(manager, id; cancel::Bool=false, after::Int=0)
    after >= 0 || throw(ArgumentError("after cursor must be non-negative"))
    lock(manager.lock) do
        ripple_job_cleanup!(manager)
        haskey(manager.jobs, id) || return nothing
        job = manager.jobs[id]
        if cancel && job.status in ("queued", "running")
            job.cancelled[] = true
            if job.status == "queued"
                index = findfirst(candidate -> candidate === job, manager.queue)
                index === nothing || deleteat!(manager.queue, index)
                job.status = "cancelled"
                job.phase = "cancelled"
                job.payload = nothing
                job.result = nothing
                job.error = "Generation cancelled before execution"
                job.error_status = 409
                job.finished = time()
                notify(job.updated; all=true)
                notify(job.done)
            end
        end
        total_deltas = length(job.delta_chunks)
        deltas = after < total_deltas ? copy(job.delta_chunks[(after + 1):total_deltas]) : String[]
        return (
            id=job.id, status=job.status, phase=job.phase,
            prefill_processed=job.processed, prompt_eval_count=job.total,
            deltas=deltas, next_cursor=total_deltas,
            result=job.result, error=job.error, error_status=job.error_status,
        )
    end
end

function close_ripple_http_jobs!(manager)
    lock(manager.lock) do
        manager.closed = true
        for job in values(manager.jobs)
            if job.status == "queued"
                job.cancelled[] = true
                job.status = "cancelled"
                job.phase = "cancelled"
                job.payload = nothing
                job.result = nothing
                job.error = "Generation service is shutting down"
                job.error_status = 409
                job.finished = time()
                notify(job.updated; all=true)
                notify(job.done)
            elseif job.status == "running"
                job.cancelled[] = true
            end
        end
        empty!(manager.queue)
        notify(manager.condition; all=true)
    end
    wait(manager.worker)
end

function _ripple_query_after(target::AbstractString)
    parts = split(String(target), '?'; limit=2)
    length(parts) == 2 || return 0
    for pair in split(parts[2], '&')
        kv = split(pair, '='; limit=2)
        length(kv) == 2 && kv[1] == "after" || continue
        parsed = tryparse(Int, kv[2])
        parsed !== nothing && parsed >= 0 || throw(ArgumentError("invalid after query parameter"))
        return parsed
    end
    return 0
end

function _ripple_sse_chunk(id::String, created::Int, model_id::String, delta, finish_reason; usage=nothing, lifeai=nothing)
    base = (
        id=id,
        object="chat.completion.chunk",
        created=created,
        model=model_id,
        choices=[(index=0, delta=delta, finish_reason=finish_reason)],
    )
    payload = usage === nothing ? base : merge(base, (usage=usage,))
    payload = lifeai === nothing ? payload : merge(payload, (lifeai=lifeai,))
    return "data: " * JSON3.write(payload) * "\n\n"
end

function _ripple_format_sse_final(stream_id::String, created::Int, model_id::String, result, streamed_any::Bool)
    io = IOBuffer()
    choice = hasproperty(result, :choices) && !isempty(result.choices) ? result.choices[1] : nothing
    finish = choice === nothing ? "stop" : get(choice, :finish_reason, "stop")
    msg = choice === nothing ? nothing : get(choice, :message, nothing)
    tool_calls = msg === nothing ? nothing : (hasproperty(msg, :tool_calls) ? msg.tool_calls : (msg isa AbstractDict ? get(msg, "tool_calls", nothing) : nothing))
    content = msg === nothing ? "" : String(something(hasproperty(msg, :content) ? msg.content : (msg isa AbstractDict ? get(msg, "content", "") : ""), ""))
    if !streamed_any && !isempty(content)
        write(io, _ripple_sse_chunk(stream_id, created, model_id, (role="assistant", content=content), nothing))
    end
    if tool_calls !== nothing && !isempty(tool_calls)
        write(io, _ripple_sse_chunk(stream_id, created, model_id, (tool_calls=tool_calls,), nothing))
    end
    usage = hasproperty(result, :usage) ? result.usage : nothing
    lifeai = hasproperty(result, :lifeai) ? result.lifeai : nothing
    write(io, _ripple_sse_chunk(stream_id, created, model_id, NamedTuple(), finish; usage, lifeai))
    write(io, "data: [DONE]\n\n")
    return String(take!(io))
end

struct RippleTPHTTPHandler{E,V,H}
    manager::RippleHTTPJobs
    execute::E
    validate::V
    key::String
    model_id::String
    health::H
    max_body::Int
end

function _ripple_parse_json(body)
    text = body isa AbstractString ? String(body) : String(copy(body))
    return isdefined(@__MODULE__, :parse_qwen3_json) ? parse_qwen3_json(text) : JSON3.read(text, Dict{String,Any})
end

function (self::RippleTPHTTPHandler)(request::HTTP.Request)
    respond = (status, value) -> HTTP.Response(status, ["Content-Type" => "application/json"], JSON3.write(value))
    path = split(String(request.target), '?'; limit=2)[1]
    if path == "/healthz" && request.method in ("GET", "HEAD")
        return respond(200, merge(
            (async_jobs=true, streaming=true, tool_calls=true, max_request_bytes=self.max_body),
            self.health(),
        ))
    end
    if !isempty(self.key) && HTTP.header(request, "Authorization", "") != "Bearer " * self.key
        return respond(401, (error=(message="Unauthorized", type="authentication_error"),))
    end
    path == "/v1/models" && return respond(200, (object="list", data=[(id=self.model_id, object="model", owned_by="lifeai")]))
    prefix = "/v1/chat/completions/jobs/"
    if startswith(path, prefix)
        request.method in ("GET", "DELETE") || return respond(405, (error=(message="Method not allowed",),))
        after = try
            _ripple_query_after(request.target)
        catch error
            return respond(400, (error=(message=sprint(showerror, error), type="invalid_request_error"),))
        end
        id = path[length(prefix)+1:end]
        result = ripple_job_status(self.manager, id; cancel=request.method == "DELETE", after)
        result === nothing && return respond(404, (error=(message="Generation job not found or expired",),))
        return respond(200, result)
    end
    if request.method != "POST" || !(path in ("/v1/chat/completions", "/v1/chat/completions/jobs"))
        return respond(404, (error=(message="Not found",),))
    end
    try
        length(request.body) <= self.max_body || return respond(413, (error=(message="request exceeds body limit",),))
        payload = try
            _ripple_parse_json(request.body)
        catch
            throw(ArgumentError("invalid JSON"))
        end
        payload isa AbstractDict || throw(ArgumentError("request must be a JSON object"))
        wants_stream = get(payload, "stream", false) === true && path == "/v1/chat/completions"
        prepared = self.validate(payload)
        job = ripple_job_submit!(self.manager, prepared)
        if path == "/v1/chat/completions/jobs"
            return respond(202, (id=job.id, status="queued"))
        end
        wait(job.done)
        if job.status == "completed"
            if wants_stream
                stream_id = "chatcmpl-" * string(uuid4())
                created = floor(Int, time())
                io = IOBuffer()
                streamed_any = false
                for chunk in job.delta_chunks
                    write(io, _ripple_sse_chunk(stream_id, created, self.model_id,
                        streamed_any ? (content=chunk,) : (role="assistant", content=chunk), nothing))
                    streamed_any = true
                end
                write(io, _ripple_format_sse_final(stream_id, created, self.model_id, job.result, streamed_any))
                return HTTP.Response(200, ["Content-Type" => "text/event-stream", "Cache-Control" => "no-cache"], String(take!(io)))
            end
            return respond(200, job.result)
        end
        return respond(
            job.status == "cancelled" ? 409 : job.error_status,
            (error=(message=something(job.error, "Generation cancelled"), type="generation_error"),),
        )
    catch error
        status = error isa ArgumentError ? 400 : error isa InvalidStateException ? 503 : 500
        return respond(status, (error=(message=sprint(showerror, error), type="invalid_request_error"),))
    end
end

function ripple_tp_http_handler(execute, validate; key, model_id, health, max_body=128 * 1024^2, timeout=21600.0)
    manager = RippleHTTPJobs(execute; timeout)
    handler = RippleTPHTTPHandler(manager, execute, validate, String(key), String(model_id), health, Int(max_body))
    return handler, manager
end

# Read authentication/framing headers before allowing HTTP.jl to buffer a body.
# HTTP.serve!(...;stream=false) would read(stream) before our Request handler.
function _ripple_http_body_framing(request, max_body)
    lengths = [strip(value) for (name, value) in request.headers if lowercase(name) == "content-length"]
    transfers = [strip(value) for (name, value) in request.headers if lowercase(name) == "transfer-encoding"]
    length(lengths) <= 1 && length(transfers) <= 1 || throw(ArgumentError("duplicate HTTP body framing headers"))
    isempty(lengths) || isempty(transfers) || throw(ArgumentError("Content-Length and Transfer-Encoding cannot be combined"))
    if !isempty(transfers)
        lowercase(transfers[1]) == "chunked" || throw(ArgumentError("unsupported Transfer-Encoding"))
    end
    if !isempty(lengths)
        value = lengths[1]
        !isempty(value) && all(c -> '0' <= c <= '9', value) || throw(ArgumentError("invalid Content-Length"))
        count = tryparse(UInt64, value)
        count === nothing && throw(ArgumentError("invalid Content-Length"))
        count > UInt64(max_body) && return (:too_large, nothing)
        return (:accepted, Int(count))
    end
    return (:accepted, nothing)
end

function _ripple_http_bounded_body(stream, max_body, declared)
    buffer = IOBuffer(; maxsize=max_body + 1, sizehint=min(max_body + 1, 64 * 1024))
    while !eof(stream)
        room = max_body + 1 - position(buffer)
        room > 0 || return nothing
        chunk = read(stream, min(room, 64 * 1024))
        isempty(chunk) && break
        write(buffer, chunk)
        position(buffer) > max_body && return nothing
    end
    body = take!(buffer)
    declared === nothing || length(body) == declared || throw(EOFError())
    HTTP.Streams.incomplete(stream) && throw(EOFError())
    HTTP.closeread(stream)
    return body
end

function _ripple_http_stream_response!(stream, response; close_connection=false)
    request = stream.message
    response.request = request
    request.response = response
    body = response.body isa AbstractString ? codeunits(response.body) : response.body
    HTTP.setheader(response, "Content-Length" => string(length(body)))
    close_connection && HTTP.setheader(response, "Connection" => "close")
    try
        HTTP.startwrite(stream)
        request.method == "HEAD" || write(stream, body)
        close_connection && HTTP.closewrite(stream)
    finally
        close_connection && close(stream)
    end
    return nothing
end

function _ripple_http_stream_sse!(stream::HTTP.Stream, handler::RippleTPHTTPHandler, prepared)
    job = ripple_job_submit!(handler.manager, prepared)
    request = stream.message
    response = HTTP.Response(200, [
        "Content-Type" => "text/event-stream",
        "Cache-Control" => "no-cache",
        "Connection" => "close",
    ])
    response.request = request
    request.response = response
    stream_id = "chatcmpl-" * string(uuid4())
    created = floor(Int, time())
    cursor = 0
    streamed_any = false
    completed = false
    try
        HTTP.startwrite(stream)
        while true
            chunks, status, result, err_msg = lock(handler.manager.lock) do
                while length(job.delta_chunks) == cursor && job.status in ("queued", "running")
                    wait(job.updated)
                end
                new_chunks = cursor < length(job.delta_chunks) ? copy(job.delta_chunks[(cursor + 1):end]) : String[]
                cursor += length(new_chunks)
                (new_chunks, job.status, job.result, job.error)
            end
            for chunk in chunks
                frame = _ripple_sse_chunk(
                    stream_id, created, handler.model_id,
                    streamed_any ? (content=chunk,) : (role="assistant", content=chunk),
                    nothing,
                )
                write(stream, frame)
                flush(stream)
                streamed_any = true
            end
            if status == "completed"
                write(stream, _ripple_format_sse_final(stream_id, created, handler.model_id, result, streamed_any))
                flush(stream)
                completed = true
                break
            elseif status in ("failed", "cancelled")
                err_frame = "data: " * JSON3.write((error=(message=something(err_msg, "Generation " * status), type="generation_error"),)) * "\n\n" * "data: [DONE]\n\n"
                write(stream, err_frame)
                flush(stream)
                completed = true
                break
            end
        end
        HTTP.closewrite(stream)
    catch
        completed || ripple_job_status(handler.manager, job.id; cancel=true)
    finally
        completed || ripple_job_status(handler.manager, job.id; cancel=true)
        try
            close(stream)
        catch
        end
    end
    return nothing
end

"""Wrap a Request handler for HTTP.serve!(...;stream=true), with bounded admission and live SSE."""
function ripple_tp_http_stream_handler(handler; key, max_body=128 * 1024^2)
    max_body isa Integer && !(max_body isa Bool) && 0 < max_body < typemax(Int) || throw(ArgumentError("invalid request body limit"))
    respond = (status, message) -> HTTP.Response(
        status, ["Content-Type" => "application/json"],
        JSON3.write((error=(message=message, type=status == 401 ? "authentication_error" : "invalid_request_error"),)),
    )
    return function (stream::HTTP.Stream)
        request = stream.message
        path = split(String(request.target), '?'; limit=2)[1]
        public_health = path == "/healthz" && request.method in ("GET", "HEAD")
        if !public_health && !isempty(key) && HTTP.header(request, "Authorization", "") != "Bearer " * key
            return _ripple_http_stream_response!(stream, respond(401, "Unauthorized"); close_connection=true)
        end
        if path == "/healthz" && !public_health
            return _ripple_http_stream_response!(stream, respond(405, "health requires GET or HEAD"); close_connection=true)
        end
        framing, declared = try
            _ripple_http_body_framing(request, max_body)
        catch error
            return _ripple_http_stream_response!(stream, respond(400, sprint(showerror, error)); close_connection=true)
        end
        framing === :too_large && return _ripple_http_stream_response!(stream, respond(413, "request exceeds body limit"); close_connection=true)
        if public_health
            if HTTP.hasheader(request, "Transfer-Encoding") || declared !== nothing && declared != 0
                return _ripple_http_stream_response!(stream, respond(400, "health does not accept a request body"); close_connection=true)
            end
            request.body = UInt8[]
            return _ripple_http_stream_response!(stream, handler(request))
        end
        body = try
            _ripple_http_bounded_body(stream, max_body, declared)
        catch
            return _ripple_http_stream_response!(stream, respond(400, "malformed or truncated request body"); close_connection=true)
        end
        body === nothing && return _ripple_http_stream_response!(stream, respond(413, "request exceeds body limit"); close_connection=true)
        request.body = body

        # Live Server-Sent Events when POST /v1/chat/completions requests stream=true.
        if handler isa RippleTPHTTPHandler && request.method == "POST" && path == "/v1/chat/completions"
            parsed = try
                _ripple_parse_json(body)
            catch
                nothing
            end
            if parsed isa AbstractDict && get(parsed, "stream", false) === true
                prepared = try
                    handler.validate(parsed)
                catch error
                    status = error isa ArgumentError ? 400 : error isa InvalidStateException ? 503 : 500
                    return _ripple_http_stream_response!(stream, respond(status, sprint(showerror, error)); close_connection=true)
                end
                return _ripple_http_stream_sse!(stream, handler, prepared)
            end
        end

        response = handler(request)
        return _ripple_http_stream_response!(stream, response)
    end
end
