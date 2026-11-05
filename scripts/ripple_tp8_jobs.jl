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
    queue::Channel{RippleHTTPJob}
    worker::Task
    max_pending::Int
    max_results::Int
    ttl::Float64
    timeout::Float64
end
function RippleHTTPJobs(execute;max_pending=4,max_results=8,ttl=3600.0,timeout=21600.0)
    guard=ReentrantLock();jobs=Dict{String,RippleHTTPJob}();queue=Channel{RippleHTTPJob}(max_pending)
    worker=Threads.@spawn begin
        for job in queue
            lock(guard) do
                job.status="running";job.phase="tokenizing"
            end
            cancelled=()->begin
                time()-job.created>timeout && (job.cancelled[]=true)
                job.cancelled[]
            end
            progress=(phase,processed,total)->lock(guard) do
                job.phase=String(phase);job.processed=Int(processed);job.total=Int(total)
            end
            try
                result=cancelled() ? nothing : execute(job.payload,progress,cancelled)
                lock(guard) do
                    job.status=cancelled() ? "cancelled" : "completed"
                    job.phase=job.status;job.result=job.cancelled[] ? nothing : result
                end
            catch error
                lock(guard) do
                    job.status=cancelled() ? "cancelled" : "failed"
                    job.phase=job.status;job.error=sprint(showerror,error)
                    job.error_status=error isa ArgumentError ? 400 : 500
                end
            finally
                lock(guard) do
                    job.finished=time();job.payload=nothing
                end
                notify(job.done)
            end
        end
    end
    RippleHTTPJobs(guard,jobs,queue,worker,max_pending,max_results,Float64(ttl),Float64(timeout))
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
        ripple_job_cleanup!(manager)
        active=count(j->j.status in ("queued","running"),values(manager.jobs))
        active<manager.max_pending || throw(ArgumentError("generation queue is full; retry after the current request"))
        job=RippleHTTPJob("job-"*string(uuid4()),"queued","queued",0,0,nothing,nothing,500,
            Threads.Atomic{Bool}(false),Base.Event(),time(),0.0,payload)
        manager.jobs[job.id]=job;put!(manager.queue,job)
        return job
    end
end
function ripple_job_status(manager,id;cancel=false)
    lock(manager.lock) do
        ripple_job_cleanup!(manager)
        haskey(manager.jobs,id) || return nothing
        job=manager.jobs[id]
        cancel && (job.cancelled[]=true)
        return (id=job.id,status=job.status,phase=job.phase,prefill_processed=job.processed,
            prompt_eval_count=job.total,result=job.result,error=job.error,error_status=job.error_status)
    end
end
function close_ripple_http_jobs!(manager)
    lock(manager.lock) do
        for job in values(manager.jobs);job.cancelled[]=true;end
    end
    close(manager.queue)
    wait(manager.worker)
end

function ripple_tp_http_handler(execute,validate;key,model_id,health,max_body=128*1024^2,timeout=21600.0)
    manager=RippleHTTPJobs(execute;timeout)
    respond=(status,value)->HTTP.Response(status,["Content-Type"=>"application/json"],JSON3.write(value))
    function handler(request)
        path=split(String(request.target),'?';limit=2)[1]
        path=="/healthz" && return respond(200,merge(health(),(async_jobs=true,max_request_bytes=max_body)))
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
            status=error isa ArgumentError ? 400 : 500
            return respond(status,(error=(message=sprint(showerror,error),type="invalid_request_error"),))
        end
    end
    return handler,manager
end
