using Test, HTTP, JSON3, Sockets, LifeAI
include(joinpath(@__DIR__,"..","scripts","run_flint_server.jl"))
const FHW = FlintHTTPWorker

struct _FlintWorkerTestTokenizer <: AbstractTokenizer end
LifeAI.apply_flint_chat_template(::_FlintWorkerTestTokenizer,messages) = join((message.content for message in messages)," ")
LifeAI.encode(::_FlintWorkerTestTokenizer,text::AbstractString) = fill(10,max(1,length(text)))
function LifeAI.decode_bytes(::_FlintWorkerTestTokenizer,ids;skip_special_tokens=false)
    bytes = Dict(1=>UInt8[0xe4],2=>UInt8[0xbd],3=>UInt8[0xa0],4=>UInt8[0x21],100=>UInt8[])
    return reduce(vcat,(bytes[id] for id in ids);init=UInt8[])
end
LifeAI.decode(tokenizer::_FlintWorkerTestTokenizer,ids;skip_special_tokens=false,errors=:strict) =
    LifeAI._decode_utf8(decode_bytes(tokenizer,ids;skip_special_tokens);errors)

@testset "Flint UTF-8 streaming preserves incomplete characters" begin
    buffer=FHW.UTF8Buffer()
    @test FHW.take_text!(buffer,UInt8[0xe4]) == ""
    @test FHW.take_text!(buffer,UInt8[0xbd]) == ""
    @test FHW.take_text!(buffer,UInt8[0xa0]) == "你"
    @test FHW.take_text!(buffer,UInt8[0x21]) == "!"
    @test FHW.take_text!(buffer,UInt8[];final=true) == ""
    buffer=FHW.UTF8Buffer()
    @test FHW.take_text!(buffer,UInt8[0xf0,0x9f]) == ""
    @test FHW.take_text!(buffer,UInt8[0x98,0x80]) == "😀"
    @test FHW.take_text!(buffer,UInt8[0xe4]) == ""
    @test FHW.take_text!(buffer,UInt8[];final=true) == "�"
end

@testset "Flint loopback HTTP worker protocol" begin
    finished=Ref(false)
    fake_generator = function(bundle,ids;max_new_tokens,temperature,on_token)
        result=Int[]
        finished[]=false
        for token in [1,2,3,4][1:min(max_new_tokens,4)]
            sleep(0.01)
            push!(result,token)
            on_token(token)
        end
        finished[]=true
        return result
    end
    worker=FHW.Worker(context_tokens=32,max_new_tokens=8,generator=fake_generator)
    worker.bundle=(model=(eos_token_id=[99],),)
    worker.tokenizer=_FlintWorkerTestTokenizer()
    worker.ready=true
    worker.phase="ready"
    listener=listen(ip"127.0.0.1",0)
    port=getsockname(listener)[2]
    server=HTTP.listen!(stream->FHW.handler(worker,stream);server=listener,verbose=-1)
    base="http://127.0.0.1:$port"
    headers=["Content-Type"=>"application/json"]
    payload=(messages=[(role="user",content="hi")],stream=false,options=(num_predict=4,temperature=0))
    post(value)=HTTP.post(base*"/api/chat",headers,JSON3.write(value);status_exception=false)
    try
        health=JSON3.read(String(HTTP.get(base*"/healthz").body))
        @test health.ready && health.model == "RecursiveSI/flint"
        @test health.context_tokens == 32 && health.max_new_tokens == 8
        @test health.inference_engine == "lifeai_flint_decode_v3"
        response=post(payload)
        @test response.status == 200
        object=JSON3.read(String(response.body))
        @test object.response == "你!" && object.done
        @test object.eval_count == 4 && object.prompt_eval_count == 2
        @test object.native_fp4_calls_delta == 0
        @test object.prompt_eval_duration == object.time_to_first_token
        @test object.prompt_eval_duration + object.decode_duration == object.total_duration
        @test 0 < object.prompt_eval_duration < object.total_duration
        @test object.lifeai.decode_eval_count == 3
        @test object.lifeai.decode_tokens_per_second ≈ 3 / (object.decode_duration / 1e9)
        stream_payload=merge(payload,(stream=true,))
        body=JSON3.write(stream_payload)
        chunks=Any[]
        HTTP.open("POST",base*"/api/chat",[headers;"Content-Length"=>string(sizeof(body))]) do stream
            write(stream,body)
            HTTP.closewrite(stream)
            HTTP.startread(stream)
            pending=UInt8[]
            while !eof(stream)
                bytes = try
                    readavailable(stream)
                catch error
                    error isa EOFError || rethrow()
                    UInt8[]
                end
                isempty(bytes) && (eof(stream) ? break : (yield(); continue))
                append!(pending,bytes)
                while (newline=findfirst(==(0x0a),pending)) !== nothing
                    line=String(pending[1:newline-1])
                    deleteat!(pending,1:newline)
                    isempty(line) || push!(chunks,JSON3.read(line))
                end
            end
            @test isempty(pending)
        end
        @test chunks[end].done
        @test chunks[end].native_fp4_calls_delta == 0
        @test join(String(chunk.response) for chunk in chunks) == "你!"
        @test all(!occursin('�',String(chunk.response)) for chunk in chunks)
        @test length(chunks) >= 3
        # Close a real streaming connection after its first delivered chunk.
        try
            HTTP.open("POST",base*"/api/chat",[headers;"Content-Length"=>string(sizeof(body))]) do stream
                write(stream,body)
                HTTP.closewrite(stream)
                HTTP.startread(stream)
                readavailable(stream)
                close(stream)
            end
        catch error
            @test error isa Union{HTTP.RequestError,EOFError,Base.IOError}
        end
        @test timedwait(() -> !islocked(worker.generation_lock),2.0;pollint=0.01) == :ok
        @test post(payload).status == 200
        @test post(merge(payload,(options=(num_predict=9,temperature=0),))).status == 400
        @test post(merge(payload,(options=(num_predict=4,temperature=-1),))).status == 400
        @test post(merge(payload,(messages=[(role="tool",content="x")],))).status == 400
        @test post(merge(payload,(messages=[(role="user",content=repeat("a",40))],))).status == 400
        @test HTTP.post(base*"/api/chat",headers,"{";status_exception=false).status == 400
        @test HTTP.post(base*"/api/chat",headers,repeat("x",1024^2+1);status_exception=false).status == 413
        lock(worker.generation_lock)
        try
            @test post(payload).status == 409
        finally
            unlock(worker.generation_lock)
        end
        worker.ready=false
        @test post(payload).status == 503
        worker.ready=true
        # Runtime exceptions occur while the HTTP handler owns the lock.
        original_generator=worker.generator
        worker.generator=(args...;kwargs...)->error("injected generation failure")
        failure=post(payload)
        @test failure.status == 500
        @test occursin("injected generation failure",String(failure.body))
        @test trylock(worker.generation_lock)
        unlock(worker.generation_lock)
        worker.generator=original_generator
        # A disconnected sink aborts before another token is evaluated.
        request=FHW.prepare_request(worker,JSON3.write(merge(payload,(stream=true,))))
        @test_throws ErrorException FHW.generate(worker,request;write_piece=piece->error("closed client"))
        @test trylock(worker.generation_lock)
        unlock(worker.generation_lock)
        @test post(payload).status == 200
    finally
        close(server)
    end
end
