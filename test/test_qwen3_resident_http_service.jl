using HTTP
using JSON3
using SHA: sha256
using Sockets
using LifeAI:
    Qwen3XLAHTTPService,
    qwen3_xla_http_handler,
    qwen3_xla_service_status,
    serve_qwen3_xla_http!

isdefined(@__MODULE__, :repository_test_asset) ||
    include("repository_test_assets.jl")

const _QWEN3_RESIDENT_SERVICE_MODEL = "Qwen/Qwen3-8B"

function _qwen3_resident_service_fake_service(;
    delay_seconds=0.0,
    max_body_bytes=1024^2,
)
    load_calls = Ref(0)
    active_calls = Ref(0)
    maximum_active_calls = Ref(0)
    counter_lock = ReentrantLock()
    loader = function ()
        load_calls[] += 1
        return (name=:fake_xla_session,)
    end
    prompt_encoder = function (_, prompt)
        prompt == "overflow" && return fill(7, 49)
        prompt == "boom" && return [999]
        return fill(7, max(1, ncodeunits(prompt)))
    end
    token_bytes = Dict(
        101 => UInt8[0xe4],
        102 => UInt8[0xbd],
        103 => UInt8[0xa0],
        104 => UInt8[0x21],
    )
    token_decoder = function (_, ids)
        output = UInt8[]
        for id in ids
            append!(output, get(token_bytes, Int(id), UInt8[]))
        end
        return output
    end
    generator = function (_, prompt_ids, _, on_token)
        lock(counter_lock) do
            active_calls[] += 1
            maximum_active_calls[] = max(
                maximum_active_calls[],
                active_calls[],
            )
        end
        try
            first(prompt_ids) == 999 && error("injected generation failure")
            delay_seconds > 0 && sleep(delay_seconds)
            generated_ids = [101, 102, 103, 104]
            if on_token !== nothing
                for (index, token_id) in enumerate(generated_ids)
                    on_token(token_id, index)
                end
            end
            return (;
                generated_ids,
                completion="你!",
                stop_reason=:length,
                prefill_seconds=0.012,
                decode_seconds=0.034,
            )
        finally
            lock(counter_lock) do
                active_calls[] -= 1
            end
        end
    end
    service = Qwen3XLAHTTPService(;
        loader,
        generator,
        prompt_encoder,
        token_decoder,
        model_id=_QWEN3_RESIDENT_SERVICE_MODEL,
        context_tokens=64,
        prefill_chunk_tokens=8,
        max_new_tokens=16,
        max_body_bytes,
    )
    return (; service, load_calls, active_calls, maximum_active_calls)
end

function _qwen3_resident_service_payload(;
    prompt="hi",
    stream=false,
    model=_QWEN3_RESIDENT_SERVICE_MODEL,
    raw=true,
    think=false,
    temperature=0.0,
    repeat_penalty=1.0,
    repeat_last_n=0,
    num_predict=4,
    num_ctx=64,
    extra=Dict{String,Any}(),
)
    object = Dict{String,Any}(
        "model" => model,
        "prompt" => prompt,
        "raw" => raw,
        "stream" => stream,
        "think" => think,
        "keep_alive" => "10m",
        "options" => Dict{String,Any}(
            "temperature" => temperature,
            "repeat_penalty" => repeat_penalty,
            "repeat_last_n" => repeat_last_n,
            "num_predict" => num_predict,
            "num_ctx" => num_ctx,
            "seed" => 0,
        ),
    )
    merge!(object, extra)
    return JSON3.write(object)
end

function _qwen3_resident_service_request(
    method,
    target;
    body=UInt8[],
    content_type=nothing,
)
    headers = Pair{String,String}[]
    content_type === nothing ||
        push!(headers, "Content-Type" => String(content_type))
    return HTTP.Request(
        String(method),
        String(target),
        headers,
        body,
    )
end

function _qwen3_resident_service_generate_request(; kwargs...)
    return _qwen3_resident_service_request(
        "POST",
        "/api/generate";
        body=_qwen3_resident_service_payload(; kwargs...),
        content_type="application/json; charset=utf-8",
    )
end

_qwen3_resident_service_json(response) =
    JSON3.read(String(response.body), Dict{String,Any})

function _qwen3_resident_service_error_code(response)
    return String(_qwen3_resident_service_json(response)["code"])
end

@testset "service loads once and serves Ollama-compatible JSON" begin
    fixture = _qwen3_resident_service_fake_service()
    service = fixture.service
    @test fixture.load_calls[] == 1

    health = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_request("GET", "/healthz?probe=1"),
    )
    @test health.status == 200
    @test HTTP.header(health, "Content-Type") ==
        "application/json; charset=utf-8"
    health_json = _qwen3_resident_service_json(health)
    @test health_json["status"] == "ok"
    @test health_json["ready"] === true
    @test health_json["model"] == _QWEN3_RESIDENT_SERVICE_MODEL
    @test health_json["load_count"] == 1

    response = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_generate_request(stream=false, num_predict=16),
    )
    @test response.status == 200
    @test HTTP.header(response, "Content-Type") ==
        "application/json; charset=utf-8"
    object = _qwen3_resident_service_json(response)
    @test object["model"] == _QWEN3_RESIDENT_SERVICE_MODEL
    @test object["response"] == "你!"
    @test object["done"] === true
    @test object["done_reason"] == "length"
    @test object["load_duration"] == 0
    @test object["prompt_eval_count"] == 2
    @test object["prompt_eval_duration"] == 12_000_000
    @test object["eval_count"] == 4
    @test object["eval_duration"] == 34_000_000
    @test object["lifeai"]["prompt_bucket_tokens"] == 8
    @test object["lifeai"]["decode_token_count"] == 3
    @test object["lifeai"]["decode_seconds"] == 0.034
    @test object["lifeai"]["decode_tokens_per_second"] ≈ 3 / 0.034
    @test object["lifeai"]["cache_tokens"] == 11
    @test fixture.load_calls[] == 1
end

@testset "NDJSON streams only valid UTF-8 pieces" begin
    fixture = _qwen3_resident_service_fake_service()
    response = qwen3_xla_http_handler(
        fixture.service,
        _qwen3_resident_service_generate_request(stream=true),
    )
    @test response.status == 200
    @test HTTP.header(response, "Content-Type") ==
        "application/x-ndjson; charset=utf-8"
    lines = filter(!isempty, split(String(response.body), '\n'))
    messages = [JSON3.read(line, Dict{String,Any}) for line in lines]
    @test length(messages) == 3
    @test messages[1]["response"] == "你"
    @test messages[2]["response"] == "!"
    @test all(message["done"] === false for message in messages[1:2])
    @test messages[3]["done"] === true
    @test messages[3]["response"] == ""
    @test join(String(message["response"]) for message in messages) == "你!"
end

@testset "request gates fail closed with JSON errors" begin
    fixture = _qwen3_resident_service_fake_service()
    service = fixture.service

    missing = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_request("GET", "/missing"),
    )
    @test missing.status == 404
    @test _qwen3_resident_service_error_code(missing) == "not_found"

    wrong_method = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_request("GET", "/api/generate"),
    )
    @test wrong_method.status == 405
    @test _qwen3_resident_service_error_code(wrong_method) == "method_not_allowed"

    wrong_type = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_request(
            "POST",
            "/api/generate";
            body=_qwen3_resident_service_payload(),
            content_type="text/plain",
        ),
    )
    @test wrong_type.status == 415
    @test _qwen3_resident_service_error_code(wrong_type) == "unsupported_media_type"

    malformed = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_request(
            "POST",
            "/api/generate";
            body="{",
            content_type="application/json",
        ),
    )
    @test malformed.status == 400
    @test _qwen3_resident_service_error_code(malformed) == "invalid_json"

    mismatch = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_generate_request(model="another-model"),
    )
    @test mismatch.status == 400
    @test _qwen3_resident_service_error_code(mismatch) == "model_mismatch"

    templated = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_generate_request(raw=false),
    )
    @test templated.status == 400
    @test _qwen3_resident_service_error_code(templated) == "raw_required"

    sampling = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_generate_request(temperature=0.8),
    )
    @test sampling.status == 400
    @test _qwen3_resident_service_error_code(sampling) == "sampling_unsupported"

    context_mismatch = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_generate_request(num_ctx=32),
    )
    @test context_mismatch.status == 400
    @test _qwen3_resident_service_error_code(context_mismatch) == "context_mismatch"

    overflow = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_generate_request(prompt="overflow", num_predict=16),
    )
    @test overflow.status == 400
    @test _qwen3_resident_service_error_code(overflow) == "context_length_exceeded"

    unknown = qwen3_xla_http_handler(
        service,
        _qwen3_resident_service_generate_request(extra=Dict("format" => "json")),
    )
    @test unknown.status == 400
    @test _qwen3_resident_service_error_code(unknown) == "unsupported_field"

    small = _qwen3_resident_service_fake_service(max_body_bytes=32).service
    oversized = qwen3_xla_http_handler(
        small,
        _qwen3_resident_service_generate_request(),
    )
    @test oversized.status == 413
    @test _qwen3_resident_service_error_code(oversized) == "body_too_large"

    status = qwen3_xla_service_status(service)
    @test status.load_count == 1
    @test status.requests.total == 0
    @test fixture.load_calls[] == 1
end

@testset "static K/V access is single-flight" begin
    fixture = _qwen3_resident_service_fake_service(delay_seconds=0.04)
    responses = Vector{HTTP.Response}(undef, 2)
    @sync for index in eachindex(responses)
        @async responses[index] = qwen3_xla_http_handler(
            fixture.service,
            _qwen3_resident_service_generate_request(prompt="p$index"),
        )
    end
    @test all(response.status == 200 for response in responses)
    @test fixture.maximum_active_calls[] == 1
    @test fixture.active_calls[] == 0
    status = qwen3_xla_service_status(fixture.service)
    @test status.requests.total == 2
    @test status.requests.completed == 2
    @test status.requests.failed == 0
    @test status.requests.active == 0
    @test status.requests.queued == 0
    @test status.requests.max_active == 1
    @test fixture.load_calls[] == 1
end

@testset "generation failures release the lock" begin
    fixture = _qwen3_resident_service_fake_service()
    failed = qwen3_xla_http_handler(
        fixture.service,
        _qwen3_resident_service_generate_request(prompt="boom"),
    )
    @test failed.status == 500
    @test _qwen3_resident_service_error_code(failed) == "generation_failed"

    recovered = qwen3_xla_http_handler(
        fixture.service,
        _qwen3_resident_service_generate_request(prompt="ok"),
    )
    @test recovered.status == 200
    @test _qwen3_resident_service_json(recovered)["response"] == "你!"
    status = qwen3_xla_service_status(fixture.service)
    @test status.requests.total == 2
    @test status.requests.completed == 1
    @test status.requests.failed == 1
    @test status.requests.active == 0
    @test status.requests.queued == 0
    @test fixture.load_calls[] == 1
end

@testset "server defaults remain loopback-only" begin
    source = read(
        joinpath(@__DIR__, "..", "src", "generation", "qwen3_xla_service.jl"),
        String,
    )
    project = read(joinpath(@__DIR__, "..", "Project.toml"), String)
    @test occursin(
        r"host::AbstractString=\"127\.0\.0\.1\"",
        source,
    )
    @test occursin(
        "HTTP = \"cd3eb016-35fb-5094-929b-558a96fad6f3\"",
        project,
    )
    @test occursin(
        "Sockets = \"6462fe0b-24de-5631-8697-dd941f90decc\"",
        project,
    )
end

@testset "frozen Qwen3 resident HTTP service hardware evidence is self-consistent" begin
    report_path = repository_test_asset("qwen3_8b_4090d_bf16_xla_service.json")
    trace_path = repository_test_asset(
        "qwen3_8b_4090d_bf16_xla_service_nvidia_smi.csv",
    )
    report_sha =
        "73c1bf7a0a7dbabbd2ee1b4ae246022e05d2296e11d8e1d665624fb4cc4b6152"
    trace_sha =
        "e006940214ecabb3802dda178faaad994491cfeae2fc2cfd3425a0d71c2d960b"

    @test bytes2hex(sha256(read(report_path))) == report_sha
    report = JSON3.read(read(report_path, String))
    @test Int(report["schema_version"]) == 1
    @test Int(report["acceptance_revision"]) == 2
    @test Bool(report["closed"])
    acceptance = report["acceptance"]
    passed_fields = [
        Bool(value)
        for (key, value) in pairs(acceptance)
        if endswith(String(key), "_passed")
    ]
    @test !isempty(passed_fields)
    @test all(passed_fields)
    @test Int(acceptance["load_count"]) == 1
    @test Int(acceptance["reuse_requests"]) == 10
    @test Int(acceptance["cuda_bf16_parity_match_tokens"]) == 96
    @test Int(acceptance["cuda_bf16_parity_expected_tokens"]) == 96
    @test Int(acceptance["full_window_prompt_tokens"]) == 3584
    @test Int(acceptance["full_window_generated_tokens"]) == 512
    @test Int(acceptance["full_window_cache_tokens"]) == 4095
    @test Float64(acceptance["steady_decode_tokens_per_second"]) >=
        Float64(acceptance["steady_decode_minimum_tokens_per_second"])
    @test Float64(acceptance["maximum_long_prefill_seconds"]) <=
        Float64(acceptance["maximum_long_prefill_limit_seconds"])
    @test Float64(
        acceptance["full_window_decode_tokens_per_second"],
    ) >= Float64(acceptance["full_window_decode_minimum_tokens_per_second"])
    requests = acceptance["service_requests"]
    @test Int(requests["total"]) == 15
    @test Int(requests["completed"]) == 15
    @test Int(requests["failed"]) == 0
    @test Int(requests["max_active"]) == 1

    concurrent = report["concurrent"]
    service_work = sum(
        Float64(run["service_total_seconds"])
        for run in concurrent["runs"]
    )
    @test Float64(concurrent["service_work_seconds"]) ≈ service_work
    @test Float64(concurrent["pair_to_service_work_ratio"]) ≈
        Float64(concurrent["pair_wall_seconds"]) / service_work
    @test Float64(concurrent["pair_to_service_work_ratio"]) >=
        Float64(acceptance["concurrent_minimum_serial_ratio"])

    @test bytes2hex(sha256(read(trace_path))) == trace_sha
    trace_lines = readlines(trace_path)
    trace = report["memory"]["physical_trace"]
    @test length(trace_lines) == Int(trace["samples"])
    trace_minimum_free = minimum(trace_lines) do line
        fields = strip.(split(line, ','))
        parse(Int, fields[3]) * 1024^2
    end
    @test trace_minimum_free == Int(trace["minimum_free_bytes"])
    @test String(trace["sha256"]) == trace_sha
    @test trace_minimum_free >=
        Int(acceptance["physical_minimum_free_limit_bytes"])
end

if lowercase(get(ENV, "LIFEAI_TEST_HTTP_SOCKET", "false")) in
        ("1", "true", "yes")
    @testset "loopback HTTP transport" begin
        fixture = _qwen3_resident_service_fake_service()
        port, socket = Sockets.listenany(Sockets.localhost, 0)
        server = serve_qwen3_xla_http!(
            fixture.service;
            server=socket,
        )
        try
            health = HTTP.get("http://127.0.0.1:$port/healthz")
            @test health.status == 200
            @test _qwen3_resident_service_json(health)["load_count"] == 1

            generated = HTTP.post(
                "http://127.0.0.1:$port/api/generate",
                ["Content-Type" => "application/json"],
                _qwen3_resident_service_payload(stream=true),
            )
            @test generated.status == 200
            @test HTTP.header(generated, "Content-Type") ==
                "application/x-ndjson; charset=utf-8"
            lines = filter(!isempty, split(String(generated.body), '\n'))
            messages = [
                JSON3.read(line, Dict{String,Any}) for line in lines
            ]
            @test join(
                String(message["response"]) for message in messages
            ) == "你!"
            @test last(messages)["done"] === true
            @test fixture.load_calls[] == 1
        finally
            close(server)
        end
    end
end
