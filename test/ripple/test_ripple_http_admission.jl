using Test,LifeAI,HTTP,JSON3,UUIDs,Sockets
if !isdefined(@__MODULE__,:ripple_tp_http_stream_handler)
    include(joinpath(@__DIR__,"..","..","scripts","ripple_tp8_jobs.jl"))
end

@testset "TCP auth and bounded body admission before read" begin
    calls=Threads.Atomic{Int}(0)
    execute=(payload,progress,cancelled)->begin
        Threads.atomic_add!(calls,1)
        (model="test-ripple",choices=[(message=(role="assistant",content=get(payload,"unicode",false) ? "你好，世界🌟" : "OK"),finish_reason="stop")])
    end
    validate=payload->begin
        get(payload,"bad",false) && throw(ArgumentError("上下文超出限制，请缩短。"))
        payload
    end
    handler,manager=ripple_tp_http_handler(execute,validate;key="test-key",model_id="test-ripple",health=()->(status="ready",),max_body=1024)
    tcp=listen(ip"127.0.0.1",0);port=getsockname(tcp)[2]
    server=HTTP.serve!(ripple_tp_http_stream_handler(handler;key="test-key",max_body=1024),ip"127.0.0.1",port;
        server=tcp,stream=true,verbose=-1)
    url="http://127.0.0.1:$port"
    function wire_status(headers;body="",halfclose=false,method="POST",path="/v1/chat/completions")
        socket=connect(ip"127.0.0.1",port)
        timer=Timer(_->close(socket),5)
        try
            write(socket,method*" "*path*" HTTP/1.1\r\nHost: localhost\r\n"*headers*"\r\n"*body);flush(socket)
            halfclose && closewrite(socket)
            statusline=readline(socket)
            isempty(statusline) && return nothing
            return parse(Int,split(statusline)[2])
        finally
            close(timer);close(socket)
        end
    end
    auth="Authorization: Bearer test-key\r\n"
    try
        # Deliberately send headers only. Both responses must arrive without
        # reading/draining any of the declared body or waiting for an EOF.
        @test wire_status("Content-Length: 999999999\r\n")==401
        @test wire_status(auth*"Content-Length: 1025\r\n")==413
        @test calls[]==0
        @test wire_status("Content-Length: 1024\r\n";method="GET",path="/healthz")==400
        @test wire_status("Transfer-Encoding: chunked\r\n";method="HEAD",path="/healthz")==400
        @test wire_status("Content-Length: 1024\r\n";method="POST",path="/healthz")==401
        @test wire_status(auth*"Content-Length: 1024\r\n";method="POST",path="/healthz")==405
        @test wire_status(auth*"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n")==400
        # HTTP.jl can reject duplicate Content-Length while parsing headers,
        # before invoking a stream handler; immediate EOF is also fail-closed.
        @test wire_status(auth*"Content-Length: 2\r\nContent-Length: 3\r\n") in (400,nothing)
        @test wire_status(auth*"Transfer-Encoding: gzip\r\n")==400
        exact="{\"blob\":\""*repeat("x",1024-11)*"\"}"
        @test sizeof(exact)==1024
        response=HTTP.request("POST",url*"/v1/chat/completions",["Authorization"=>"Bearer test-key"],exact;status_exception=false)
        @test response.status==200
        @test JSON3.read(String(response.body)).choices[1].message.content=="OK"
        @test wire_status(auth*"Transfer-Encoding: chunked\r\n";body="1\r\n{\r\n1\r\n}\r\n0\r\n\r\n")==200
        # Oversized chunked entity omits its terminal zero chunk. The reader
        # must reject after max+1 bytes, not wait for or buffer the remainder.
        @test wire_status(auth*"Transfer-Encoding: chunked\r\n";body="400\r\n"*repeat("x",1024)*"\r\n1\r\nx\r\n")==413
        @test wire_status(auth*"Content-Length: 10\r\n";body="{}",halfclose=true)==400
        @test wire_status(auth*"Transfer-Encoding: chunked\r\n";body="10\r\n{}",halfclose=true)==400
        @test wire_status(auth*"Transfer-Encoding: chunked\r\n";body="not-hex\r\n",halfclose=true)==400
        # HEAD must send no entity bytes despite its would-be Content-Length.
        # Reusing the same TCP connection makes any forbidden body corrupt the
        # following GET status line, even when normal clients silently ignore it.
        for path in ("/healthz","/v1/models")
            socket=connect(ip"127.0.0.1",port);timer=Timer(_->close(socket),5)
            try
                write(socket,"HEAD "*path*" HTTP/1.1\r\nHost: localhost\r\n"*auth*"\r\n");flush(socket)
                @test occursin(" 200 ",readline(socket))
                headers=Dict{String,String}()
                while true
                    line=readline(socket);isempty(line) && break
                    parts=split(line,':';limit=2);headers[lowercase(parts[1])]=strip(parts[2])
                end
                @test parse(Int,headers["content-length"])>0
                write(socket,"GET "*path*" HTTP/1.1\r\nHost: localhost\r\n"*auth*"Connection: close\r\n\r\n");flush(socket)
                @test occursin(" 200 ",readline(socket))
                response_headers=Dict{String,String}()
                while true
                    line=readline(socket);isempty(line) && break
                    parts=split(line,':';limit=2);response_headers[lowercase(parts[1])]=strip(parts[2])
                end
                body=read(socket,parse(Int,response_headers["content-length"]))
                @test !isempty(body)
                @test JSON3.read(String(body)) isa AbstractDict
            finally
                close(timer);close(socket)
            end
        end
        socket=connect(ip"127.0.0.1",port);timer=Timer(_->close(socket),5)
        function read_wire_response()
            statusline=readline(socket);@test occursin("HTTP/1.1",statusline)
            headers=Dict{String,String}()
            while true
                line=readline(socket);isempty(line) && break
                parts=split(line,':';limit=2);headers[lowercase(parts[1])]=strip(parts[2])
            end
            body=read(socket,parse(Int,headers["content-length"]))
            return parse(Int,split(statusline)[2]),JSON3.read(String(body))
        end
        try
            # Two ordinary GETs plus async admission/status on the same socket.
            for _ in 1:2
                write(socket,"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n");flush(socket)
                code,value=read_wire_response();@test code==200 && value.status=="ready"
            end
            write(socket,"POST /v1/chat/completions/jobs HTTP/1.1\r\nHost: localhost\r\n"*auth*"Content-Length: 2\r\n\r\n{}");flush(socket)
            code,value=read_wire_response();@test code==202
            id=String(value.id)
            write(socket,"GET /v1/chat/completions/jobs/"*id*" HTTP/1.1\r\nHost: localhost\r\n"*auth*"Connection: close\r\n\r\n");flush(socket)
            code,value=read_wire_response();@test code==200 && value.id==id
            @test value.status in ("queued","running","completed")
        finally
            close(timer);close(socket)
        end
        response=HTTP.get(url*"/healthz";status_exception=false)
        @test response.status==200
        @test JSON3.read(String(response.body)).status=="ready"
        response=HTTP.request("POST",url*"/v1/chat/completions",["Authorization"=>"Bearer test-key"],"{\"unicode\":true}";status_exception=false)
        @test response.status==200
        @test parse(Int,HTTP.header(response,"Content-Length"))==length(response.body)
        @test JSON3.read(String(response.body)).choices[1].message.content=="你好，世界🌟"
        response=HTTP.request("POST",url*"/v1/chat/completions",["Authorization"=>"Bearer test-key"],"{\"bad\":true}";status_exception=false)
        @test response.status==400
        @test parse(Int,HTTP.header(response,"Content-Length"))==length(response.body)
        @test occursin("上下文超出限制",JSON3.read(String(response.body)).error.message)
        # Two ASCII, one Unicode and one async requests executed.
        @test timedwait(()->calls[]==4,2)==:ok
    finally
        close(server);close(tcp);close_ripple_http_jobs!(manager)
    end
end
