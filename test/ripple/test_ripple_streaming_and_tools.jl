using Test, LifeAI, HTTP, JSON3, UUIDs, Sockets

if !isdefined(@__MODULE__, :ripple_tp_http_stream_handler)
    include(joinpath(@__DIR__, "..", "..", "scripts", "ripple_tp8_jobs.jl"))
end

@testset "Ripple tool-call parser" begin
    single = "Let me check.\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>"
    parsed = parse_ripple_tool_calls(single)
    @test parsed.content == "Let me check."
    @test length(parsed.calls) == 1
    @test parsed.calls[1].name == "execute_bash"
    @test parsed.calls[1].arguments["command"] == "pwd"
    @test isempty(parsed.invalid_blocks)

    multi = "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"y\": \"你好\"}}\n</tool_call>"
    pm = parse_ripple_tool_calls(multi)
    @test pm.content == ""
    @test length(pm.calls) == 2
    @test pm.calls[2].arguments["y"] == "你好"

    # Malformed JSON or non-object arguments are captured in invalid_blocks, not calls.
    bad = "<tool_call>\n{\"name\": \"a\", \"arguments\": \"not-an-object\"}\n</tool_call>\n<tool_call>\n{broken"
    pb = parse_ripple_tool_calls(bad)
    @test isempty(pb.calls)
    @test length(pb.invalid_blocks) == 2
end

@testset "Ripple live TCP SSE streaming, job delta cursors, and tool_calls" begin
    execute = (payload, progress, cancelled, emit_delta) -> begin
        progress("prefill", 16, 16)
        if get(payload, "tools", nothing) !== nothing
            raw = "Thinking...\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}\n</tool_call>"
            parsed = parse_ripple_tool_calls(raw)
            calls = [(id="call_test01", type="function", var"function"=(name=c.name, arguments=c.arguments_json)) for c in parsed.calls]
            return (
                id="chatcmpl-tool", object="chat.completion", created=1, model="test-ripple",
                choices=[(index=0, message=(role="assistant", content=parsed.content, tool_calls=calls), finish_reason="tool_calls")],
                usage=(prompt_tokens=16, completion_tokens=12, total_tokens=28),
            )
        end
        for piece in ("你", "好", "，", "Ripple", "!")
            cancelled() && throw(InterruptException())
            emit_delta(piece)
            progress("decode", 16, 16)
            sleep(0.005)
        end
        return (
            id="chatcmpl-stream", object="chat.completion", created=1, model="test-ripple",
            choices=[(index=0, message=(role="assistant", content="你好，Ripple!"), finish_reason="stop")],
            usage=(prompt_tokens=16, completion_tokens=5, total_tokens=21),
        )
    end
    validate = p -> p
    handler, manager = ripple_tp_http_handler(
        execute, validate;
        key="test-key", model_id="test-ripple",
        health=() -> (status="ready", streaming=true, tool_calls=true),
        max_body=4096,
    )
    tcp = listen(ip"127.0.0.1", 0)
    port = getsockname(tcp)[2]
    server = HTTP.serve!(
        ripple_tp_http_stream_handler(handler; key="test-key", max_body=4096),
        ip"127.0.0.1", port; server=tcp, stream=true, verbose=-1,
    )
    url = "http://127.0.0.1:$port"
    headers = ["Authorization" => "Bearer test-key", "Content-Type" => "application/json"]
    try
        # 1. Health advertises streaming and tool_calls.
        h = JSON3.read(String(HTTP.get(url * "/healthz").body))
        @test h.streaming == true
        @test h.tool_calls == true

        # 2. Live SSE streaming over TCP emits per-token chunks ending with [DONE].
        resp = HTTP.post(url * "/v1/chat/completions", headers, JSON3.write(Dict("stream" => true, "messages" => [])))
        @test resp.status == 200
        @test occursin("text/event-stream", HTTP.header(resp, "Content-Type"))
        body_text = String(resp.body)
        data_lines = [strip(line[6:end]) for line in split(body_text, '\n') if startswith(line, "data:")]
        @test !isempty(data_lines)
        @test last(data_lines) == "[DONE]"
        chunks = [JSON3.read(line) for line in data_lines[1:end-1]]
        reassembled = join([get(c.choices[1].delta, :content, "") for c in chunks])
        @test reassembled == "你好，Ripple!"
        @test chunks[end].choices[1].finish_reason == "stop"

        # 3. Async job polling with `?after=N` returns non-overlapping incremental deltas.
        created = JSON3.read(String(HTTP.post(url * "/v1/chat/completions/jobs", headers, JSON3.write(Dict("messages" => []))).body))
        job_id = String(created.id)
        cursor = 0
        collected = String[]
        deadline = time() + 5.0
        final_status = nothing
        while time() < deadline
            st = JSON3.read(String(HTTP.get(url * "/v1/chat/completions/jobs/" * job_id * "?after=$cursor", headers).body))
            for d in st.deltas
                push!(collected, String(d))
            end
            @test Int(st.next_cursor) >= cursor
            cursor = Int(st.next_cursor)
            if String(st.status) == "completed"
                final_status = st
                break
            end
            sleep(0.005)
        end
        @test final_status !== nothing
        @test join(collected) == "你好，Ripple!"
        @test cursor == 5

        # Invalid negative cursor fails closed with 400.
        bad_cursor = HTTP.get(url * "/v1/chat/completions/jobs/" * job_id * "?after=-1", headers; status_exception=false)
        @test bad_cursor.status == 400

        # 4. Tool calling returns OpenAI-compatible tool_calls and finish_reason="tool_calls".
        tool_resp = HTTP.post(
            url * "/v1/chat/completions", headers,
            JSON3.write(Dict("stream" => false, "messages" => [], "tools" => [Dict("type" => "function", "function" => Dict("name" => "get_weather"))])),
        )
        @test tool_resp.status == 200
        tool_json = JSON3.read(String(tool_resp.body))
        @test tool_json.choices[1].finish_reason == "tool_calls"
        @test tool_json.choices[1].message.content == "Thinking..."
        @test length(tool_json.choices[1].message.tool_calls) == 1
        @test tool_json.choices[1].message.tool_calls[1].var"function".name == "get_weather"
    finally
        close(server)
        close(tcp)
        close_ripple_http_jobs!(manager)
    end
end
