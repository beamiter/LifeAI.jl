# Bounded asynchronous HTTP jobs keep long prefill independent of TCP lifetimes.
mutable struct RippleHTTPJob
    id::String
    status::String
    phase::String
    processed::Int
    total::Int
    result::Any
    error::Union{Nothing,String}
    error_status::Int
    cancelled::Threads.Atomic{Bool}
    done::Base.Event
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
function _ripple_http_job_worker!(manager,execute)
    while true
        job=lock(manager.lock) do
            while isempty(manager.queue) && !manager.closed
                wait(manager.condition)
            end
            isempty(manager.queue) && return nothing
            next=popfirst!(manager.queue)
            next.status="running";next.phase="tokenizing"
            next
        end
        job===nothing && return
        cancelled=()->begin
            time()-job.created>manager.timeout && (job.cancelled[]=true)
            job.cancelled[]
        end
        progress=(phase,processed,total)->lock(manager.lock) do
            job.phase=String(phase);job.processed=Int(processed);job.total=Int(total)
        end
        try
            result=cancelled() ? nothing : execute(job.payload,progress,cancelled)
            lock(manager.lock) do
                job.status=cancelled() ? "cancelled" : "completed"
                job.phase=job.status;job.result=job.cancelled[] ? nothing : result
            end
        catch error
            lock(manager.lock) do
                job.status=cancelled() ? "cancelled" : "failed"
                job.phase=job.status;job.error=sprint(showerror,error)
                job.error_status=error isa ArgumentError ? 400 : 500
            end
        finally
            lock(manager.lock) do
                job.finished=time();job.payload=nothing
            end
            notify(job.done)
        end
    end
end
function RippleHTTPJobs(execute;max_pending=4,max_results=8,ttl=3600.0,timeout=21600.0)
    max_pending isa Integer && !(max_pending isa Bool) && max_pending>0 || throw(ArgumentError("max_pending must be positive"))
    max_results isa Integer && !(max_results isa Bool) && max_results>0 || throw(ArgumentError("max_results must be positive"))
    isfinite(ttl) && ttl>0 || throw(ArgumentError("job TTL must be finite and positive"))
    isfinite(timeout) && timeout>0 || throw(ArgumentError("job timeout must be finite and positive"))
    guard=ReentrantLock()
    manager=RippleHTTPJobs(guard,Dict{String,RippleHTTPJob}(),RippleHTTPJob[],Threads.Condition(guard),
        Task(()->nothing),Int(max_pending),Int(max_results),Float64(ttl),Float64(timeout),false)
    manager.worker=Threads.@spawn _ripple_http_job_worker!(manager,execute)
    return manager
end

function ripple_job_cleanup!(manager)
    finished=sort!([job for job in values(manager.jobs) if job.finished>0];by=j->j.finished)
    keep_from=max(1,length(finished)-manager.max_results+1)
    for (index,job) in enumerate(finished)
        if index<keep_from || time()-job.finished>manager.ttl
            delete!(manager.jobs,job.id)
        end
    end
end
function ripple_job_submit!(manager,payload)
    lock(manager.lock) do
        manager.closed && throw(InvalidStateException("generation service is shutting down",:closed))
        ripple_job_cleanup!(manager)
        active=count(j->j.status in ("queued","running"),values(manager.jobs))
        active<manager.max_pending || throw(ArgumentError("generation queue is full; retry after the current request"))
        job=RippleHTTPJob("job-"*string(uuid4()),"queued","queued",0,0,nothing,nothing,500,
            Threads.Atomic{Bool}(false),Base.Event(),time(),0.0,payload)
        manager.jobs[job.id]=job;push!(manager.queue,job)
        notify(manager.condition)
        return job
    end
end
function ripple_job_status(manager,id;cancel=false)
    lock(manager.lock) do
        ripple_job_cleanup!(manager)
        haskey(manager.jobs,id) || return nothing
        job=manager.jobs[id]
        if cancel && job.status in ("queued","running")
            job.cancelled[]=true
            if job.status=="queued"
                index=findfirst(candidate->candidate===job,manager.queue)
                index===nothing || deleteat!(manager.queue,index)
                job.status="cancelled";job.phase="cancelled";job.payload=nothing
                job.result=nothing;job.error="Generation cancelled before execution";job.error_status=409
                job.finished=time();notify(job.done)
            end
        end
        return (id=job.id,status=job.status,phase=job.phase,prefill_processed=job.processed,
            prompt_eval_count=job.total,result=job.result,error=job.error,error_status=job.error_status)
    end
end
function close_ripple_http_jobs!(manager)
    lock(manager.lock) do
        manager.closed=true
        for job in values(manager.jobs)
            if job.status=="queued"
                job.cancelled[]=true;job.status="cancelled";job.phase="cancelled"
                job.payload=nothing;job.result=nothing;job.error="Generation service is shutting down"
                job.error_status=409;job.finished=time();notify(job.done)
            elseif job.status=="running"
                job.cancelled[]=true
            end
        end
        empty!(manager.queue)
        notify(manager.condition;all=true)
    end
    wait(manager.worker)
end

function ripple_tp_http_handler(execute,validate;key,model_id,health,max_body=128*1024^2,timeout=21600.0)
    manager=RippleHTTPJobs(execute;timeout)
    respond=(status,value)->HTTP.Response(status,["Content-Type"=>"application/json"],JSON3.write(value))
    function handler(request)
        path=split(String(request.target),'?';limit=2)[1]
        path=="/healthz" && request.method in ("GET","HEAD") && return respond(200,merge(health(),(async_jobs=true,max_request_bytes=max_body)))
        if !isempty(key) && HTTP.header(request,"Authorization","")!="Bearer "*key
            return respond(401,(error=(message="Unauthorized",type="authentication_error"),))
        end
        path=="/v1/models" && return respond(200,(object="list",data=[(id=model_id,object="model",owned_by="lifeai")]))
        prefix="/v1/chat/completions/jobs/"
        if startswith(path,prefix)
            request.method in ("GET","DELETE") || return respond(405,(error=(message="Method not allowed",),))
            id=path[length(prefix)+1:end];result=ripple_job_status(manager,id;cancel=request.method=="DELETE")
            result===nothing && return respond(404,(error=(message="Generation job not found or expired",),))
            return respond(200,result)
        end
        if request.method!="POST" || !(path in ("/v1/chat/completions","/v1/chat/completions/jobs"))
            return respond(404,(error=(message="Not found",),))
        end
        try
            length(request.body)<=max_body || return respond(413,(error=(message="request exceeds body limit",),))
            payload=try
                parse_qwen3_json(String(request.body))
            catch
                throw(ArgumentError("invalid JSON"))
            end
            payload isa AbstractDict || throw(ArgumentError("request must be a JSON object"))
            prepared=validate(payload)
            job=ripple_job_submit!(manager,prepared)
            if path=="/v1/chat/completions/jobs"
                return respond(202,(id=job.id,status="queued"))
            end
            wait(job.done)
            if job.status=="completed"
                return respond(200,job.result)
            end
            return respond(job.status=="cancelled" ? 409 : job.error_status,
                (error=(message=something(job.error,"Generation cancelled"),type="generation_error"),))
        catch error
            status=error isa ArgumentError ? 400 : error isa InvalidStateException ? 503 : 500
            return respond(status,(error=(message=sprint(showerror,error),type="invalid_request_error"),))
        end
    end
    return handler,manager
end


# Read authentication/framing headers before allowing HTTP.jl to buffer a body.
# HTTP.serve!(...;stream=false) would read(stream) before our Request handler.
function _ripple_http_body_framing(request,max_body)
    lengths=[strip(value) for (name,value) in request.headers if lowercase(name)=="content-length"]
    transfers=[strip(value) for (name,value) in request.headers if lowercase(name)=="transfer-encoding"]
    length(lengths)<=1 && length(transfers)<=1 || throw(ArgumentError("duplicate HTTP body framing headers"))
    isempty(lengths) || isempty(transfers) || throw(ArgumentError("Content-Length and Transfer-Encoding cannot be combined"))
    if !isempty(transfers)
        lowercase(transfers[1])=="chunked" || throw(ArgumentError("unsupported Transfer-Encoding"))
    end
    if !isempty(lengths)
        value=lengths[1]
        !isempty(value) && all(c->'0'<=c<='9',value) || throw(ArgumentError("invalid Content-Length"))
        count=tryparse(UInt64,value)
        count===nothing && throw(ArgumentError("invalid Content-Length"))
        count>UInt64(max_body) && return (:too_large,nothing)
        return (:accepted,Int(count))
    end
    return (:accepted,nothing)
end
function _ripple_http_bounded_body(stream,max_body,declared)
    buffer=IOBuffer(;maxsize=max_body+1,sizehint=min(max_body+1,64*1024))
    while !eof(stream)
        room=max_body+1-position(buffer)
        room>0 || return nothing
        chunk=read(stream,min(room,64*1024))
        isempty(chunk) && break
        write(buffer,chunk)
        position(buffer)>max_body && return nothing
    end
    body=take!(buffer)
    declared===nothing || length(body)==declared || throw(EOFError())
    HTTP.Streams.incomplete(stream) && throw(EOFError())
    HTTP.closeread(stream)
    return body
end
function _ripple_http_stream_response!(stream,response;close_connection=false)
    request=stream.message
    response.request=request;request.response=response
    body=response.body isa AbstractString ? codeunits(response.body) : response.body
    HTTP.setheader(response,"Content-Length"=>string(length(body)))
    close_connection && HTTP.setheader(response,"Connection"=>"close")
    try
        HTTP.startwrite(stream)
        request.method=="HEAD" || write(stream,body)
        # Normal server connections must remain writable until HTTP.jl performs
        # its own post-handler closewrite; early rejects flush before closing.
        close_connection && HTTP.closewrite(stream)
    finally
        # close(Stream) also sets ntoread=0, preventing the server's later
        # closeread from trying to drain an unread or truncated request body.
        close_connection && close(stream)
    end
    return nothing
end
"""Wrap a Request handler for HTTP.serve!(...;stream=true), with bounded admission.

No unauthenticated body is read. Accepted Content-Length/chunked entities are
read in at most64KiB pieces into a buffer capped at max_body+1 bytes. Rejected
or malformed entities close after their response, without draining the sender.
"""
function ripple_tp_http_stream_handler(handler;key,max_body=128*1024^2)
    max_body isa Integer && !(max_body isa Bool) && 0<max_body<typemax(Int) || throw(ArgumentError("invalid request body limit"))
    respond=(status,message)->HTTP.Response(status,["Content-Type"=>"application/json"],
        JSON3.write((error=(message=message,type=status==401 ? "authentication_error" : "invalid_request_error"),)))
    return function(stream::HTTP.Stream)
        request=stream.message;path=split(String(request.target),'?';limit=2)[1]
        public_health=path=="/healthz" && request.method in ("GET","HEAD")
        if !public_health && !isempty(key) && HTTP.header(request,"Authorization","")!="Bearer "*key
            return _ripple_http_stream_response!(stream,respond(401,"Unauthorized");close_connection=true)
        end
        if path=="/healthz" && !public_health
            return _ripple_http_stream_response!(stream,respond(405,"health requires GET or HEAD");close_connection=true)
        end
        framing,declared=try
            _ripple_http_body_framing(request,max_body)
        catch error
            return _ripple_http_stream_response!(stream,respond(400,sprint(showerror,error));close_connection=true)
        end
        framing===:too_large && return _ripple_http_stream_response!(stream,respond(413,"request exceeds body limit");close_connection=true)
        if public_health
            if HTTP.hasheader(request,"Transfer-Encoding") || declared!==nothing && declared!=0
                return _ripple_http_stream_response!(stream,respond(400,"health does not accept a request body");close_connection=true)
            end
            request.body=UInt8[]
            return _ripple_http_stream_response!(stream,handler(request))
        end
        body=try
            _ripple_http_bounded_body(stream,max_body,declared)
        catch
            return _ripple_http_stream_response!(stream,respond(400,"malformed or truncated request body");close_connection=true)
        end
        body===nothing && return _ripple_http_stream_response!(stream,respond(413,"request exceeds body limit");close_connection=true)
        request.body=body
        response=handler(request)
        return _ripple_http_stream_response!(stream,response)
    end
end
