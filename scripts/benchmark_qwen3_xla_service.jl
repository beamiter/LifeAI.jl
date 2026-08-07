#!/usr/bin/env julia

# Reactant reads these settings during module initialization. They must stay
# before every `using LifeAI` / `using Reactant`; callers may override them.
ENV["XLA_REACTANT_GPU_MEM_FRACTION"] =
    get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.87")
ENV["XLA_REACTANT_GPU_PREALLOCATE"] =
    get(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using Dates
using HTTP
using JSON3
using LifeAI
using Reactant
using SHA: sha256
using Sockets
using Statistics: median

const QWEN3_XLA_DEPLOYMENT_CUDA_REFERENCE_SHA256 =
    "83f62afbbb470b695b6990a3b86a8860407a37874354d6b039e1ce19917e2747"
const LOOPBACK_HOST = "127.0.0.1"
const REUSE_REQUESTS = 10
const PARITY_TOKENS = 32
const PHYSICAL_SAMPLE_INTERVAL_MS = 200
const GIB = 1024^3

elapsed_seconds(started) = (time_ns() - started) / 1.0e9
file_sha256(path) = bytes2hex(sha256(read(path)))
token_sha256(ids) =
    bytes2hex(sha256(codeunits(join(Int.(ids) .- 1, ','))))

function physical_gpu_memory()
    output = strip(read(
        `nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits`,
        String,
    ))
    fields = strip.(split(output, ','))
    length(fields) == 4 || error("unexpected nvidia-smi memory output")
    return (;
        device_index=0,
        name=fields[1],
        total_bytes=parse(Int, fields[2]) * 1024^2,
        used_bytes=parse(Int, fields[3]) * 1024^2,
        free_bytes=parse(Int, fields[4]) * 1024^2,
    )
end

function stop_process(process)
    process === nothing && return
    try
        kill(process, Base.SIGINT)
    catch
    end
    try
        wait(process)
    catch
    end
    return
end

function physical_trace_summary(path)
    samples = Tuple{Int,Int}[]
    for line in eachline(path)
        fields = strip.(split(line, ','))
        length(fields) == 3 ||
            error("unexpected nvidia-smi trace row: $line")
        push!(samples, (
            parse(Int, fields[2]) * 1024^2,
            parse(Int, fields[3]) * 1024^2,
        ))
    end
    isempty(samples) && error("nvidia-smi physical trace is empty")
    return (;
        device_index=0,
        sample_interval_ms=PHYSICAL_SAMPLE_INTERVAL_MS,
        samples=length(samples),
        maximum_used_bytes=maximum(first, samples),
        minimum_free_bytes=minimum(last, samples),
        path=abspath(path),
        sha256=file_sha256(path),
    )
end

function allocator_bytes(snapshot)
    snapshot === nothing && return nothing
    return snapshot.bytes_in_use
end

function health_summary(payload)
    requests = payload["requests"]
    return (;
        status=String(payload["status"]),
        ready=Bool(payload["ready"]),
        model=String(payload["model"]),
        context_tokens=Int(payload["context_tokens"]),
        max_new_tokens=Int(payload["max_new_tokens"]),
        load_seconds=Float64(payload["load_seconds"]),
        load_count=Int(payload["load_count"]),
        requests=(;
            total=Int(requests["total"]),
            completed=Int(requests["completed"]),
            failed=Int(requests["failed"]),
            queued=Int(requests["queued"]),
            active=Int(requests["active"]),
            max_active=Int(requests["max_active"]),
        ),
    )
end

function http_health(endpoint)
    started = time_ns()
    response = HTTP.request(
        "GET",
        endpoint * "/healthz",
        ["Accept" => "application/json", "Connection" => "close"];
        status_exception=false,
        retry=false,
        readtimeout=60,
    )
    wall_seconds = elapsed_seconds(started)
    response.status == 200 ||
        error("GET /healthz returned HTTP $(response.status)")
    payload = JSON3.read(String(response.body), Dict{String,Any})
    return health_summary(payload), wall_seconds
end

function http_generate(
    endpoint,
    service,
    profile,
    prompt,
    expected_prompt_tokens,
    output_tokens;
    label,
)
    body = JSON3.write((;
        model=profile.model_id,
        prompt,
        raw=true,
        stream=false,
        think=false,
        keep_alive=-1,
        options=(;
            temperature=0.0,
            repeat_penalty=1.0,
            repeat_last_n=0,
            num_predict=output_tokens,
            num_ctx=profile.context_tokens,
            seed=0,
        ),
    ))
    started = time_ns()
    response = HTTP.request(
        "POST",
        endpoint * "/api/generate",
        [
            "Accept" => "application/json",
            "Content-Type" => "application/json",
            "Connection" => "close",
        ],
        body;
        status_exception=false,
        retry=false,
        readtimeout=600,
    )
    client_wall_seconds = elapsed_seconds(started)
    response.status == 200 || error(
        "$label returned HTTP $(response.status): $(String(response.body))",
    )
    payload = JSON3.read(String(response.body), Dict{String,Any})
    Bool(payload["done"]) || error("$label response is not final")
    String(payload["model"]) == profile.model_id ||
        error("$label response model changed")
    completion = String(payload["response"])

    # HTTP is the system boundary. Recover IDs only from the returned text,
    # using the exact tokenizer owned by the one resident session.
    generated_ids = encode(
        service.session.tokenizer,
        completion;
        add_special_tokens=false,
    )
    eval_count = Int(payload["eval_count"])
    length(generated_ids) == eval_count || error(
        "$label completion re-encoded to $(length(generated_ids)) tokens, " *
        "but the service reported eval_count=$eval_count",
    )
    prompt_eval_count = Int(payload["prompt_eval_count"])
    prompt_eval_count == expected_prompt_tokens || error(
        "$label service prompt count $prompt_eval_count differs from " *
        "the frozen count $expected_prompt_tokens",
    )
    lifeai = payload["lifeai"]
    prefill_seconds = Int(payload["prompt_eval_duration"]) / 1.0e9
    decode_seconds = Int(payload["eval_duration"]) / 1.0e9
    decode_steps = max(eval_count - 1, 0)
    return (;
        generated_ids=Int.(generated_ids),
        metrics=(;
            label,
            http_status=Int(response.status),
            request_id=Int(lifeai["request_id"]),
            prompt_tokens=prompt_eval_count,
            prompt_bucket_tokens=Int(lifeai["prompt_bucket_tokens"]),
            cache_tokens=Int(lifeai["cache_tokens"]),
            generated_tokens=eval_count,
            done_reason=String(payload["done_reason"]),
            queue_seconds=Int(lifeai["queue_duration_ns"]) / 1.0e9,
            prefill_seconds,
            decode_seconds,
            decode_tokens_per_second=
                decode_steps == 0 ? 0.0 : decode_steps / decode_seconds,
            service_total_seconds=Int(payload["total_duration"]) / 1.0e9,
            client_wall_seconds,
            completion_reencoded=true,
            generated_token_sha256=token_sha256(generated_ids),
        ),
    )
end

function frozen_case(reference_cases, tokenizer, name, prompt_tokens)
    case = reference_cases[name]
    Int(case["prompt_tokens"]) == prompt_tokens ||
        error("$name prompt length is not frozen")
    Int(case["generated_tokens"]) == PARITY_TOKENS ||
        error("$name must contain exactly $PARITY_TOKENS generated tokens")
    prompt_ids = Int.(case["prompt_ids_0_based"]) .+ 1
    generated_ids = Int.(case["generated_ids_0_based"]) .+ 1
    length(prompt_ids) == prompt_tokens ||
        error("$name prompt payload length is invalid")
    length(generated_ids) == PARITY_TOKENS ||
        error("$name generated payload length is invalid")
    prompt = decode(
        tokenizer,
        prompt_ids;
        errors=:strict,
        skip_special_tokens=false,
    )
    roundtrip = encode(tokenizer, prompt; add_special_tokens=false)
    roundtrip == prompt_ids || error(
        "$name frozen prompt does not survive tokenizer text round-trip",
    )
    return (; name, prompt, prompt_ids, generated_ids)
end

function parity_summary(name, actual, expected)
    matched = length(actual) == length(expected) ?
        count(identity, actual .== expected) : 0
    return (;
        name,
        matched_tokens=matched,
        expected_tokens=length(expected),
        passed=matched == length(expected),
        actual_token_sha256=token_sha256(actual),
        expected_token_sha256=token_sha256(expected),
    )
end

function main(args)
    length(args) in (3, 4) || error(
        "usage: julia --threads=auto --project=. --startup-file=no " *
        "scripts/benchmark_qwen3_xla_service.jl " *
        "MODEL_DIR PROFILE_JSON CUDA_REFERENCE_JSON [OUTPUT_JSON]",
    )
    model_dir, profile_path, reference_path = args[1:3]
    output_path = length(args) == 4 ?
        args[4] :
        joinpath(
            @__DIR__,
            "..",
            "benchmark_results",
            "week21",
            "qwen3_8b_4090d_bf16_xla_service.json",
        )
    requested_port = parse(
        Int,
        get(ENV, "LIFEAI_XLA_SERVICE_BENCH_PORT", "0"),
    )
    0 <= requested_port <= 65535 ||
        error("LIFEAI_XLA_SERVICE_BENCH_PORT must be in 0:65535")

    profile = load_qwen3_deployment_profile(profile_path)
    profile.variant === :qwen3_8b ||
        error("XLA service benchmark requires qwen3_8b")
    profile.strategy === :greedy ||
        error("XLA service benchmark requires the frozen greedy profile")
    profile.context_tokens == 4096 ||
        error("XLA service benchmark requires the frozen 4K context")
    profile.max_prompt_tokens == 3584 ||
        error("XLA service benchmark requires the frozen 3584-token prompt")
    profile.max_new_tokens == 512 ||
        error("XLA service benchmark requires the frozen 512-token output")

    profile_sha256 = file_sha256(profile_path)
    asset_manifest_path = isabspath(profile.asset_manifest) ?
        profile.asset_manifest :
        joinpath(dirname(abspath(profile_path)), profile.asset_manifest)
    asset_manifest_sha256 = file_sha256(asset_manifest_path)
    reference_sha256 = file_sha256(reference_path)
    reference_sha256 == QWEN3_XLA_DEPLOYMENT_CUDA_REFERENCE_SHA256 || error(
        "CUDA reference SHA256 is not the frozen Week20 oracle",
    )
    reference = JSON3.read(read(reference_path, String))
    Int(reference["schema_version"]) == 2 ||
        error("CUDA reference schema_version must be 2")
    String(reference["source"]) ==
        "LifeAI eager BF16 CUDA parity reference" ||
        error("CUDA reference source is not the independent eager path")
    String(reference["model_id"]) == profile.model_id ||
        error("CUDA reference model_id does not match profile")
    String(reference["revision"]) == profile.revision ||
        error("CUDA reference revision does not match profile")
    String(reference["profile_sha256"]) == profile_sha256 ||
        error("CUDA reference profile SHA256 does not match")
    String(reference["asset_manifest_sha256"]) ==
        asset_manifest_sha256 ||
        error("CUDA reference asset manifest SHA256 does not match")
    Int(reference["verified_asset_files"]) == 10 ||
        error("CUDA reference must cover all 10 frozen assets")
    reference_cases = Dict(
        String(case["name"]) => case for case in reference["cases"]
    )
    Set(keys(reference_cases)) == Set((
        "single_chunk_64",
        "left_padded_65",
        "full_prompt_3584",
    )) || error("CUDA reference does not contain the frozen parity cases")

    ready_started = time_ns()
    println(stderr, "verifying frozen Qwen3-8B assets…")
    asset_started = time_ns()
    assets = verify_qwen3_deployment_assets(
        model_dir,
        asset_manifest_path;
        model_id=profile.model_id,
        revision=profile.revision,
    )
    asset_check_seconds = elapsed_seconds(asset_started)
    physical_before = physical_gpu_memory()
    occursin("RTX 4090 D", physical_before.name) ||
        error("Week21 acceptance requires NVIDIA GeForce RTX 4090 D")
    physical_before.total_bytes >= profile.minimum_gpu_bytes ||
        error("GPU 0 does not meet profile.minimum_gpu_bytes")

    physical_trace_path = endswith(lowercase(output_path), ".json") ?
        output_path[1:(end - 5)] * "_nvidia_smi.csv" :
        output_path * "_nvidia_smi.csv"
    mkpath(dirname(abspath(physical_trace_path)))
    open(physical_trace_path, "w") do _
    end

    monitor = nothing
    monitor_running = false
    server = nothing
    try
        monitor = run(
            `nvidia-smi --id=0 --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader,nounits --loop-ms=$(PHYSICAL_SAMPLE_INTERVAL_MS) --filename=$(abspath(physical_trace_path))`;
            wait=false,
        )
        monitor_running = true
        sleep(0.25)

        Reactant.set_default_backend("gpu")
        println(
            stderr,
            "loading the resident 4K XLA service exactly once…",
        )
        service_started = time_ns()
        service = load_qwen3_xla_http_service(
            model_dir;
            model_id=profile.model_id,
            context_tokens=profile.context_tokens,
            prefill_chunk_tokens=profile.prefill_chunk_tokens,
            max_new_tokens=profile.max_new_tokens,
            revision=profile.revision,
            variant=profile.variant,
        )
        service_constructor_seconds = elapsed_seconds(service_started)
        service.load_count == 1 ||
            error("resident service loader did not run exactly once")
        session = service.session
        load_metrics = session.load_metrics
        load_metrics.parameter_tensor_count == 291 ||
            error("Qwen3-8B compact tree must have 291 tensor leaves")
        load_metrics.device_parameter_tree_count == 1 ||
            error("resident session does not own exactly one parameter tree")
        load_metrics.device_parameter_tree_transfer_count == 1 ||
            error("resident session did not perform exactly one tree transfer")
        session_identity_before = string(objectid(session))
        allocator_ready = LifeAI._qwen3_xla_allocator_snapshot()

        case64 = frozen_case(
            reference_cases,
            session.tokenizer,
            "single_chunk_64",
            64,
        )
        case65 = frozen_case(
            reference_cases,
            session.tokenizer,
            "left_padded_65",
            65,
        )
        case3584 = frozen_case(
            reference_cases,
            session.tokenizer,
            "full_prompt_3584",
            profile.max_prompt_tokens,
        )

        server_started = time_ns()
        actual_port, server = if requested_port == 0
            port, socket = Sockets.listenany(Sockets.localhost, 0)
            running = serve_qwen3_xla_http!(
                service;
                server=socket,
                verbose=-1,
            )
            Int(port), running
        else
            running = serve_qwen3_xla_http!(
                service;
                host=LOOPBACK_HOST,
                port=requested_port,
                verbose=-1,
            )
            Int(HTTP.port(running)), running
        end
        server_start_seconds = elapsed_seconds(server_started)
        endpoint = "http://$LOOPBACK_HOST:$actual_port"
        initial_health, initial_health_seconds = http_health(endpoint)
        endpoint_ready_seconds = elapsed_seconds(ready_started)
        initial_health.ready ||
            error("resident endpoint is not ready after loading")
        initial_health.load_count == 1 ||
            error("initial health does not report load_count=1")
        initial_health.requests.total == 0 ||
            error("health request must not increment generation count")
        physical_ready = physical_gpu_memory()

        println(stderr, "running $REUSE_REQUESTS cross-connection reuses…")
        reuse_runs = Any[]
        reuse_results = Any[]
        reuse_allocator_bytes = Union{Nothing,Int}[]
        for index in 1:REUSE_REQUESTS
            result = http_generate(
                endpoint,
                service,
                profile,
                case64.prompt,
                length(case64.prompt_ids),
                PARITY_TOKENS;
                label="reuse_$index",
            )
            result.generated_ids == case64.generated_ids || error(
                "reuse_$index differs from the frozen 64-token CUDA oracle",
            )
            snapshot = LifeAI._qwen3_xla_allocator_snapshot()
            bytes = allocator_bytes(snapshot)
            push!(reuse_allocator_bytes, bytes)
            push!(reuse_results, result)
            push!(reuse_runs, merge(
                result.metrics,
                (;
                    phase=index == 1 ? "cold" : "steady",
                    oracle_match_tokens=PARITY_TOKENS,
                    allocator_after_bytes=bytes,
                ),
            ))
        end
        all(
            result.generated_ids ==
                first(reuse_results).generated_ids
            for result in reuse_results
        ) || error("resident session is not deterministic across reuses")

        println(stderr, "running frozen 65/3584 HTTP parity requests…")
        padding_result = http_generate(
            endpoint,
            service,
            profile,
            case65.prompt,
            length(case65.prompt_ids),
            PARITY_TOKENS;
            label="left_padded_65",
        )
        padding_result.generated_ids == case65.generated_ids ||
            error("65-token HTTP result differs from CUDA oracle")
        padding_result.metrics.prompt_bucket_tokens == 128 ||
            error("65-token request did not exercise left padding")
        long_parity_result = http_generate(
            endpoint,
            service,
            profile,
            case3584.prompt,
            length(case3584.prompt_ids),
            PARITY_TOKENS;
            label="full_prompt_3584_parity",
        )
        long_parity_result.generated_ids == case3584.generated_ids ||
            error("3584-token HTTP result differs from CUDA oracle")

        println(stderr, "running two concurrent TCP requests…")
        ready_gate = Channel{Nothing}(2)
        start_gate = Channel{Nothing}(2)
        concurrent_tasks = [
            Threads.@spawn begin
                put!(ready_gate, nothing)
                take!(start_gate)
                http_generate(
                    endpoint,
                    service,
                    profile,
                    case64.prompt,
                    length(case64.prompt_ids),
                    PARITY_TOKENS;
                    label="concurrent_$index",
                )
            end
            for index in 1:2
        ]
        take!(ready_gate)
        take!(ready_gate)
        concurrent_started = time_ns()
        put!(start_gate, nothing)
        put!(start_gate, nothing)
        concurrent_results = fetch.(concurrent_tasks)
        concurrent_wall_seconds = elapsed_seconds(concurrent_started)
        all(
            result.generated_ids == case64.generated_ids
            for result in concurrent_results
        ) || error("a concurrent response differs from the CUDA oracle")
        concurrent_runs = [result.metrics for result in concurrent_results]

        println(
            stderr,
            "running the full 3584+512 HTTP context window…",
        )
        capacity_result = http_generate(
            endpoint,
            service,
            profile,
            case3584.prompt,
            length(case3584.prompt_ids),
            profile.max_new_tokens;
            label="full_window_3584_plus_512",
        )
        length(capacity_result.generated_ids) ==
            profile.max_new_tokens ||
            error("full-window HTTP request did not return 512 tokens")
        capacity_result.metrics.cache_tokens ==
            profile.context_tokens - 1 ||
            error("full-window HTTP cache did not end at 4095")
        capacity_prefix_match = count(
            identity,
            capacity_result.generated_ids[1:PARITY_TOKENS] .==
                case3584.generated_ids,
        )
        capacity_prefix_match == PARITY_TOKENS || error(
            "full-window first 32 tokens differ from the CUDA oracle",
        )
        allocator_final = LifeAI._qwen3_xla_allocator_snapshot()
        physical_final = physical_gpu_memory()
        final_health, final_health_seconds = http_health(endpoint)
        session_identity_after = string(objectid(service.session))

        expected_generation_requests =
            REUSE_REQUESTS + 2 + length(concurrent_results) + 1
        final_health.ready ||
            error("resident service is not ready after acceptance workload")
        final_health.load_count == 1 ||
            error("resident service reloaded during request reuse")
        final_health.requests.total == expected_generation_requests ||
            error("final health request total is incorrect")
        final_health.requests.completed == expected_generation_requests ||
            error("not every service request completed")
        final_health.requests.failed == 0 ||
            error("resident service recorded a failed request")
        final_health.requests.active == 0 ||
            error("resident service still has an active request")
        final_health.requests.queued == 0 ||
            error("resident service still has a queued request")
        final_health.requests.max_active == 1 ||
            error("resident service violated single-flight generation")

        close(server)
        server = nothing
        stop_process(monitor)
        monitor_running = false
        physical_trace = physical_trace_summary(physical_trace_path)

        parity_cases = [
            parity_summary(
                "single_chunk_64",
                first(reuse_results).generated_ids,
                case64.generated_ids,
            ),
            parity_summary(
                "left_padded_65",
                padding_result.generated_ids,
                case65.generated_ids,
            ),
            parity_summary(
                "full_prompt_3584",
                long_parity_result.generated_ids,
                case3584.generated_ids,
            ),
        ]
        parity_match_tokens =
            sum(case.matched_tokens for case in parity_cases)
        parity_expected_tokens =
            sum(case.expected_tokens for case in parity_cases)
        steady_runs = reuse_runs[2:end]
        steady_prefill_seconds =
            median([run.prefill_seconds for run in steady_runs])
        steady_decode_tps = median([
            run.decode_tokens_per_second for run in steady_runs
        ])
        maximum_long_prefill_seconds = max(
            long_parity_result.metrics.prefill_seconds,
            capacity_result.metrics.prefill_seconds,
        )
        concurrent_max_queue_seconds = maximum(
            run.queue_seconds for run in concurrent_runs
        )
        concurrent_service_work_seconds = sum(
            run.service_total_seconds for run in concurrent_runs
        )
        concurrent_pair_to_service_work_ratio =
            concurrent_wall_seconds / concurrent_service_work_seconds
        valid_reuse_allocator_bytes =
            Int[value for value in reuse_allocator_bytes if value !== nothing]
        reuse_allocator_drift_bytes =
            length(valid_reuse_allocator_bytes) ==
                length(reuse_allocator_bytes) ?
            maximum(valid_reuse_allocator_bytes) -
                minimum(valid_reuse_allocator_bytes) :
            nothing
        parameter_allocator =
            load_metrics.allocator_after_parameter_transfer
        parameter_residency_bytes = parameter_allocator === nothing ?
            nothing : parameter_allocator.bytes_in_use
        parameter_residency_limit_bytes =
            load_metrics.parameter_logical_bytes + GIB
        load_component_seconds = (
            load_metrics.host_load_seconds +
            load_metrics.parameter_transfer_seconds +
            load_metrics.runtime_allocation_seconds +
            load_metrics.prefill_compile_seconds +
            load_metrics.decode_compile_seconds
        )
        service_load_unattributed_seconds =
            max(service.load_seconds - load_component_seconds, 0.0)

        acceptance = (;
            gpu_model_required="NVIDIA GeForce RTX 4090 D",
            gpu_model_passed=occursin(
                "RTX 4090 D",
                physical_before.name,
            ),
            gpu_minimum_bytes=profile.minimum_gpu_bytes,
            gpu_total_bytes=physical_before.total_bytes,
            gpu_capacity_passed=
                physical_before.total_bytes >= profile.minimum_gpu_bytes,
            load_count=final_health.load_count,
            load_count_required=1,
            load_once_passed=
                initial_health.load_count == 1 &&
                final_health.load_count == 1 &&
                session_identity_before == session_identity_after,
            compact_parameter_tensor_count_passed=
                load_metrics.parameter_tensor_count == 291,
            single_device_parameter_tree_passed=
                load_metrics.device_parameter_tree_count == 1 &&
                load_metrics.device_parameter_tree_transfer_count == 1 &&
                !load_metrics.original_parameter_tree_transferred &&
                !load_metrics.separate_packed_projection_tree_transferred,
            parameter_residency_bytes,
            parameter_residency_limit_bytes,
            parameter_residency_passed=
                parameter_residency_bytes !== nothing &&
                parameter_residency_bytes <=
                    parameter_residency_limit_bytes,
            reuse_requests=REUSE_REQUESTS,
            reuse_requests_required=10,
            reuse_passed=
                length(reuse_runs) >= 10 &&
                all(
                    result.generated_ids == case64.generated_ids
                    for result in reuse_results
                ),
            cuda_bf16_parity_cases=parity_cases,
            cuda_bf16_parity_match_tokens=parity_match_tokens,
            cuda_bf16_parity_expected_tokens=parity_expected_tokens,
            cuda_bf16_parity_passed=
                parity_match_tokens == parity_expected_tokens,
            concurrent_requests=length(concurrent_runs),
            concurrent_request_ids=
                sort([run.request_id for run in concurrent_runs]),
            concurrent_max_queue_seconds,
            concurrent_service_work_seconds,
            concurrent_pair_to_service_work_ratio,
            concurrent_minimum_serial_ratio=0.9,
            concurrent_single_flight_passed=
                length(unique(
                    run.request_id for run in concurrent_runs
                )) == 2 &&
                concurrent_pair_to_service_work_ratio >= 0.9 &&
                final_health.requests.max_active == 1,
            service_requests=final_health.requests,
            expected_generation_requests,
            health_passed=
                initial_health.ready &&
                final_health.ready &&
                final_health.requests.total ==
                    expected_generation_requests &&
                final_health.requests.completed ==
                    expected_generation_requests &&
                final_health.requests.failed == 0 &&
                final_health.requests.active == 0 &&
                final_health.requests.queued == 0,
            full_window_prompt_tokens=
                capacity_result.metrics.prompt_tokens,
            full_window_generated_tokens=
                capacity_result.metrics.generated_tokens,
            full_window_cache_tokens=
                capacity_result.metrics.cache_tokens,
            full_window_cuda_prefix_match_tokens=
                capacity_prefix_match,
            full_window_passed=
                capacity_result.metrics.prompt_tokens == 3584 &&
                capacity_result.metrics.generated_tokens == 512 &&
                capacity_result.metrics.cache_tokens == 4095 &&
                capacity_prefix_match == PARITY_TOKENS,
            steady_decode_tokens_per_second=steady_decode_tps,
            steady_decode_minimum_tokens_per_second=35.0,
            steady_decode_passed=steady_decode_tps >= 35.0,
            maximum_long_prefill_seconds,
            maximum_long_prefill_limit_seconds=2.0,
            maximum_long_prefill_passed=
                maximum_long_prefill_seconds <= 2.0,
            full_window_decode_tokens_per_second=
                capacity_result.metrics.decode_tokens_per_second,
            full_window_decode_minimum_tokens_per_second=35.0,
            full_window_decode_passed=
                capacity_result.metrics.decode_tokens_per_second >= 35.0,
            reusable_request_allocator_drift_bytes=
                reuse_allocator_drift_bytes,
            reusable_request_allocator_drift_limit_bytes=256 * 1024^2,
            reusable_request_memory_passed=
                reuse_allocator_drift_bytes !== nothing &&
                reuse_allocator_drift_bytes <= 256 * 1024^2,
            physical_minimum_free_bytes=
                physical_trace.minimum_free_bytes,
            physical_minimum_free_limit_bytes=
                profile.workspace_reserve_bytes,
            physical_minimum_free_passed=
                physical_trace.minimum_free_bytes >=
                    profile.workspace_reserve_bytes,
        )
        closed = all((
            acceptance.gpu_model_passed,
            acceptance.gpu_capacity_passed,
            acceptance.load_once_passed,
            acceptance.compact_parameter_tensor_count_passed,
            acceptance.single_device_parameter_tree_passed,
            acceptance.parameter_residency_passed,
            acceptance.reuse_passed,
            acceptance.cuda_bf16_parity_passed,
            acceptance.concurrent_single_flight_passed,
            acceptance.health_passed,
            acceptance.full_window_passed,
            acceptance.steady_decode_passed,
            acceptance.maximum_long_prefill_passed,
            acceptance.full_window_decode_passed,
            acceptance.reusable_request_memory_passed,
            acceptance.physical_minimum_free_passed,
        ))

        report = (;
            schema_version=1,
            acceptance_revision=2,
            source="LifeAI Week 21 Qwen3 BF16 XLA resident HTTP service",
            recorded_at=string(Dates.now()),
            model_id=profile.model_id,
            revision=profile.revision,
            model_dir=abspath(model_dir),
            profile=profile.name,
            profile_sha256,
            reference_path=abspath(reference_path),
            reference_sha256,
            assets=(;
                verified_files=length(assets.files),
                verified_bytes=assets.total_bytes,
                check_seconds=asset_check_seconds,
                manifest_path=abspath(asset_manifest_path),
                manifest_sha256=asset_manifest_sha256,
            ),
            xla=(;
                reactant_version=string(pkgversion(Reactant)),
                gpu_memory_fraction=
                    ENV["XLA_REACTANT_GPU_MEM_FRACTION"],
                gpu_preallocate=
                    ENV["XLA_REACTANT_GPU_PREALLOCATE"],
            ),
            service=(;
                host=LOOPBACK_HOST,
                requested_port,
                actual_port,
                cross_connection_header="Connection: close",
                session_identity_before,
                session_identity_after,
                context_tokens=profile.context_tokens,
                max_prompt_tokens=profile.max_prompt_tokens,
                max_new_tokens=profile.max_new_tokens,
                prefill_chunk_tokens=profile.prefill_chunk_tokens,
            ),
            ready=(;
                endpoint_ready_seconds,
                asset_check_seconds,
                service_constructor_seconds,
                service_loader_seconds=service.load_seconds,
                server_start_seconds,
                initial_health_seconds,
                load_breakdown=(;
                    host_load_seconds=load_metrics.host_load_seconds,
                    parameter_transfer_seconds=
                        load_metrics.parameter_transfer_seconds,
                    runtime_allocation_seconds=
                        load_metrics.runtime_allocation_seconds,
                    prefill_compile_seconds=
                        load_metrics.prefill_compile_seconds,
                    decode_compile_seconds=
                        load_metrics.decode_compile_seconds,
                    unattributed_seconds=
                        service_load_unattributed_seconds,
                ),
                parameter_logical_bytes=
                    load_metrics.parameter_logical_bytes,
                parameter_tensor_count=
                    load_metrics.parameter_tensor_count,
                physical_before,
                physical_ready,
            ),
            health=(;
                initial=initial_health,
                final=final_health,
                final_health_seconds,
            ),
            reuse_runs,
            parity=(;
                cases=parity_cases,
                matched_tokens=parity_match_tokens,
                expected_tokens=parity_expected_tokens,
            ),
            left_padded_run=padding_result.metrics,
            full_prompt_parity_run=long_parity_result.metrics,
            concurrent=(;
                pair_wall_seconds=concurrent_wall_seconds,
                runs=concurrent_runs,
                max_queue_seconds=concurrent_max_queue_seconds,
                service_work_seconds=concurrent_service_work_seconds,
                pair_to_service_work_ratio=
                    concurrent_pair_to_service_work_ratio,
            ),
            full_window=merge(
                capacity_result.metrics,
                (;
                    cuda_prefix_match_tokens=
                        capacity_prefix_match,
                ),
            ),
            steady=(;
                sample_count=length(steady_runs),
                statistic="median",
                prefill_seconds=steady_prefill_seconds,
                decode_tokens_per_second=steady_decode_tps,
            ),
            memory=(;
                allocator_ready_bytes=allocator_bytes(allocator_ready),
                allocator_final_bytes=allocator_bytes(allocator_final),
                reusable_request_allocator_drift_bytes=
                    reuse_allocator_drift_bytes,
                physical_before,
                physical_ready,
                physical_final,
                physical_trace,
            ),
            acceptance,
            closed,
        )
        mkpath(dirname(abspath(output_path)))
        open(output_path, "w") do io
            JSON3.pretty(io, report)
            write(io, '\n')
        end
        println(
            stderr,
            "wrote $(abspath(output_path)); closed=$closed",
        )
        closed || error(
            "Week21 service acceptance failed; inspect " *
            abspath(output_path),
        )
        return nothing
    finally
        if server !== nothing
            try
                close(server)
            catch
                try
                    HTTP.forceclose(server)
                catch
                end
            end
        end
        monitor_running && stop_process(monitor)
    end
end

main(ARGS)
