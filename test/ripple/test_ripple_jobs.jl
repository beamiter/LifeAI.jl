using Test,LifeAI,HTTP,JSON3,UUIDs
include(joinpath(@__DIR__,"..","..","scripts","ripple_tp8_jobs.jl"))

@testset "Long-prefill asynchronous jobs and bounded HTTP" begin
    execute=(payload,progress,cancelled)->begin
        get(payload,"fail",false) && throw(ArgumentError("context limit exceeded"))
        progress("prefill",512,1024)
        while get(payload,"blocked",false) && !cancelled();sleep(.01);end
        cancelled() && throw(InterruptException())
        progress("decode",1024,1024)
        (model="test-ripple",choices=[(message=(role="assistant",content="OK"),finish_reason="stop")])
    end
    validate=p->p
    handler,manager=ripple_tp_http_handler(execute,validate;key="test-key",model_id="test-ripple",
        health=()->(status="ready",),max_body=1024)
    request=(method,path,payload=nothing;auth=true)->handler(HTTP.Request(method,path,
        auth ? ["Authorization"=>"Bearer test-key"] : Pair{String,String}[],
        payload===nothing ? UInt8[] : Vector{UInt8}(codeunits(JSON3.write(payload)))))
    readjson=response->JSON3.read(String(response.body))
    function waitstatus(id,state)
        deadline=time()+3
        while time()<deadline
            value=readjson(request("GET","/v1/chat/completions/jobs/"*id))
            value.status==state && return value
            sleep(.01)
        end
        error("job did not reach $state")
    end
    try
        @test readjson(request("GET","/healthz";auth=false)).async_jobs
        @test request("GET","/v1/models";auth=false).status==401
        @test request("GET","/missing").status==404
        @test request("GET","/v1/chat/completions/jobs/absent").status==404
        @test request("POST","/v1/chat/completions",Dict("blob"=>repeat("x",2000))).status==413
        @test handler(HTTP.Request("POST","/v1/chat/completions",["Authorization"=>"Bearer test-key"],"bad-json")).status==400
        @test request("POST","/v1/chat/completions",[]).status==400
        @test readjson(request("POST","/v1/chat/completions",Dict())).choices[1].message.content=="OK"
        created=request("POST","/v1/chat/completions/jobs",Dict("blocked"=>true));@test created.status==202
        id=String(readjson(created).id);running=waitstatus(id,"running")
        @test running.prefill_processed==512 && running.prompt_eval_count==1024
        @test request("DELETE","/v1/chat/completions/jobs/"*id).status==200
        @test waitstatus(id,"cancelled").result===nothing
        failed=String(readjson(request("POST","/v1/chat/completions/jobs",Dict("fail"=>true))).id)
        @test waitstatus(failed,"failed").error_status==400
        @test readjson(request("POST","/v1/chat/completions",Dict())).choices[1].message.content=="OK"
    finally
        close_ripple_http_jobs!(manager)
    end
end
