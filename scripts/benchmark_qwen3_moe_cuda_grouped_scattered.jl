#!/usr/bin/env julia

const MODEL_ENV = "LIFEAI_QWEN3_30B_A3B_MODEL_DIR"
const REFERENCE_ENV = "LIFEAI_QWEN3_MOE_REFERENCE_DIR"
const OUTPUT_ENV = "LIFEAI_MOE_GROUPED_SCATTERED_OUTPUT"
const EXPECTED_MODEL_ID = "Qwen/Qwen3-30B-A3B"

function usage(io::IO=stdout)
    println(io, "usage: julia --threads=8 --project=. --startup-file=no \\")
    println(io, "  scripts/benchmark_qwen3_moe_cuda_grouped_scattered.jl \\")
    println(io, "  [MODEL_DIR [REFERENCE_DIR [OUTPUT_JSON]]]")
    println(io)
    println(io, "Defaults:")
    println(io, "  MODEL_DIR:      \$$MODEL_ENV")
    println(io, "  REFERENCE_DIR:  \$$REFERENCE_ENV, or")
    println(io, "                  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16")
    println(io, "  OUTPUT_JSON:    \$$OUTPUT_ENV, or")
    println(io, "                  /tmp/qwen3_moe_cuda_grouped_scattered.json")
    println(io)
    println(io, "Workload controls:")
    println(io, "  LIFEAI_MOE_WIDE_PROMPT_TOKENS       default 32")
    println(io, "  LIFEAI_MOE_WARMUP_RUNS              default 1")
    println(io, "  LIFEAI_MOE_MEASURED_RUNS            default 3")
    println(io, "  LIFEAI_MOE_CONTEXT_TOKENS           default 40960")
    println(io, "  LIFEAI_MOE_PREFILL_CHUNK_TOKENS     default 128")
    println(io, "  LIFEAI_MOE_CACHE_BUDGET_BYTES       default 8 GiB")
    println(io, "  LIFEAI_MOE_GC_INTERVAL_LAYERS       default 8")
    println(io, "  LIFEAI_MOE_READ_WORKERS             default min(8, Threads.nthreads())")
end

if any(argument -> argument in ("-h", "--help"), ARGS)
    usage()
    exit()
end
length(ARGS) <= 3 || begin
    usage(stderr)
    error("expected at most three positional arguments")
end

# Keep `--help` usable as a dependency-free smoke check. Real benchmark imports
# remain below the argument-only fast path.
using CUDA
using JSON3
using LifeAI
using SHA: sha256
using Statistics: mean, median

function positive_env_int(name::AbstractString, default::Integer)
    value = parse(Int, get(ENV, name, string(default)))
    value > 0 || error("$name must be positive")
    return value
end

model_dir = if !isempty(ARGS)
    abspath(ARGS[1])
elseif haskey(ENV, MODEL_ENV) && !isempty(ENV[MODEL_ENV])
    abspath(ENV[MODEL_ENV])
else
    error("set $MODEL_ENV or pass MODEL_DIR as the first argument")
end
reference_dir = if length(ARGS) >= 2
    abspath(ARGS[2])
elseif haskey(ENV, REFERENCE_ENV) && !isempty(ENV[REFERENCE_ENV])
    abspath(ENV[REFERENCE_ENV])
else
    joinpath(
        model_dir,
        "lifeai-references",
        "chapter24-real-parity",
        "bfloat16",
    )
end
output_path = abspath(length(ARGS) >= 3 ? ARGS[3] : get(
    ENV,
    OUTPUT_ENV,
    "/tmp/qwen3_moe_cuda_grouped_scattered.json",
))

context_tokens = positive_env_int("LIFEAI_MOE_CONTEXT_TOKENS", 40_960)
prefill_chunk_tokens = positive_env_int(
    "LIFEAI_MOE_PREFILL_CHUNK_TOKENS",
    128,
)
wide_prompt_tokens = positive_env_int("LIFEAI_MOE_WIDE_PROMPT_TOKENS", 32)
wide_prompt_tokens > 2 || error(
    "LIFEAI_MOE_WIDE_PROMPT_TOKENS must be greater than two",
)
warmup_runs = positive_env_int("LIFEAI_MOE_WARMUP_RUNS", 1)
measured_runs = positive_env_int("LIFEAI_MOE_MEASURED_RUNS", 3)
cache_budget_bytes = positive_env_int(
    "LIFEAI_MOE_CACHE_BUDGET_BYTES",
    8 * 2^30,
)
gc_interval_layers = positive_env_int("LIFEAI_MOE_GC_INTERVAL_LAYERS", 8)
read_workers = positive_env_int(
    "LIFEAI_MOE_READ_WORKERS",
    min(8, Threads.nthreads()),
)

isdir(model_dir) || error("model directory does not exist: $model_dir")
isdir(reference_dir) || error(
    "reference directory does not exist: $reference_dir",
)
metadata_path = joinpath(reference_dir, "reference.json")
reference_path = joinpath(reference_dir, "reference.safetensors")
isfile(metadata_path) || error("missing reference metadata: $metadata_path")
isfile(reference_path) || error("missing reference tensors: $reference_path")
context_tokens > wide_prompt_tokens || error(
    "LIFEAI_MOE_CONTEXT_TOKENS must exceed the wide prompt length",
)
prefill_chunk_tokens <= context_tokens || error(
    "LIFEAI_MOE_PREFILL_CHUNK_TOKENS must not exceed context capacity",
)

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)
CUDA.capability(CUDA.device()) >= v"8.0" || error(
    "grouped BF16 WMMA requires an Ampere-or-newer CUDA device",
)

metadata = JSON3.read(read(metadata_path, String))
String(metadata["model_id"]) == EXPECTED_MODEL_ID || error(
    "reference model_id must be $EXPECTED_MODEL_ID",
)
String(metadata["compute_dtype"]) == "bfloat16" || error(
    "the grouped-scattered benchmark requires a bfloat16 reference",
)
Int(metadata["num_experts"]) == 128 || error(
    "the 30B-A3B reference must contain 128 routed experts",
)
Int(metadata["num_experts_per_tok"]) == 8 || error(
    "the 30B-A3B reference must route eight experts per token",
)
num_reference_layers = Int(metadata["num_hidden_layers"])
num_reference_layers == 48 || error(
    "the 30B-A3B reference must contain 48 decoder layers",
)
reference = load_safetensors(reference_path)
reference_tokens = Int.(collect(metadata["token_ids_0_based"])) .+ 1
length(reference_tokens) == 2 || error(
    "the frozen parity workload must contain exactly two prompt tokens",
)
decode_token = Int(metadata["decode_token_id_0_based"]) + 1
wide_tokens = [
    reference_tokens[mod1(index, length(reference_tokens))]
    for index in 1:wide_prompt_tokens
]

elapsed_seconds(started) = (time_ns() - started) / 1.0e9
sha256_string(value::AbstractString) = bytes2hex(sha256(codeunits(value)))
token_sha256(values) = sha256_string(join(values, ','))

function float32_sha256(values::Vector{Float32})
    return bytes2hex(sha256(collect(reinterpret(UInt8, values))))
end

function gpu_memory_snapshot()
    return (;
        free_bytes=Int(CUDA.free_memory()),
        total_bytes=Int(CUDA.total_memory()),
        used_bytes=Int(CUDA.used_memory()),
        cached_bytes=Int(CUDA.cached_memory()),
    )
end

function process_io()
    path = "/proc/self/io"
    isfile(path) || return (; rchar=0, syscr=0, read_bytes=0)
    values = Dict{String,Int}()
    for line in eachline(path)
        key, value = split(line, ':'; limit=2)
        values[key] = parse(Int, strip(value))
    end
    return (;
        rchar=get(values, "rchar", 0),
        syscr=get(values, "syscr", 0),
        read_bytes=get(values, "read_bytes", 0),
    )
end

function io_delta(before, after)
    return (;
        rchar=after.rchar - before.rchar,
        syscr=after.syscr - before.syscr,
        read_bytes=after.read_bytes - before.read_bytes,
    )
end

function captured_cuda(f; count_allocations::Bool)
    gpu_before = gpu_memory_snapshot()
    io_before = process_io()
    value = Ref{Any}()
    started = time_ns()
    allocated = if count_allocations
        @allocated begin
            value[] = f()
            CUDA.synchronize()
        end
    else
        value[] = f()
        CUDA.synchronize()
        nothing
    end
    seconds = elapsed_seconds(started)
    io_after = process_io()
    gpu_after = gpu_memory_snapshot()
    return (;
        value=value[],
        report=(;
            seconds,
            julia_allocated_bytes=allocated,
            process_io=io_delta(io_before, io_after),
            gpu_before,
            gpu_after,
        ),
    )
end

function output_record(values::Vector{Float32})
    return (;
        length=length(values),
        eltype="Float32",
        sha256=float32_sha256(values),
        argmax_1_based=argmax(values),
        minimum=minimum(values),
        maximum=maximum(values),
        mean=mean(values),
        all_finite=all(isfinite, values),
    )
end

active_layers(active_experts) = [
    Int.(collect(experts))
    for experts in active_experts
]

function route_record(prefill, decode)
    chunks = [
        (;
            first_token=chunk.first_token,
            last_token=chunk.last_token,
            active_experts=active_layers(chunk.active_experts),
        )
        for chunk in prefill.chunks
    ]
    decode_layers = active_layers(decode.active_experts)
    canonical = JSON3.write((; chunks, decode=decode_layers))
    counts = Int[]
    for chunk in chunks, experts in chunk.active_experts
        push!(counts, length(experts))
    end
    append!(counts, length.(decode_layers))
    return (;
        raw=(; chunks, decode=decode_layers),
        report=(;
            sha256=sha256_string(canonical),
            chunks,
            decode=decode_layers,
            active_expert_count_minimum=minimum(counts),
            active_expert_count_maximum=maximum(counts),
            active_expert_count_mean=mean(counts),
        ),
    )
end

function request_traffic(prefill, decode)
    stats = decode.expert_cache
    return (;
        expert_io=(;
            prefill_bytes_read=prefill.expert_bytes_read,
            decode_bytes_read=decode.expert_bytes_read,
            total_bytes_read=
                prefill.expert_bytes_read + decode.expert_bytes_read,
            prefill_bytes_uploaded=prefill.expert_bytes_uploaded,
            decode_bytes_uploaded=decode.expert_bytes_uploaded,
            total_bytes_uploaded=
                prefill.expert_bytes_uploaded + decode.expert_bytes_uploaded,
        ),
        cache=(;
            hits=stats.hits,
            misses=stats.misses,
            evictions=stats.evictions,
            entries=stats.entries,
            current_bytes=stats.current_bytes,
            peak_bytes=stats.peak_bytes,
            prefill_hits=sum(chunk.expert_cache_hits for chunk in prefill.chunks),
            prefill_misses=sum(
                chunk.expert_cache_misses for chunk in prefill.chunks
            ),
            prefill_evictions=sum(
                chunk.expert_cache_evictions for chunk in prefill.chunks
            ),
            decode_hits=decode.expert_cache_hits,
            decode_misses=decode.expert_cache_misses,
            decode_evictions=decode.expert_cache_evictions,
        ),
        materialization=(;
            active_materializations=stats.active_materializations,
            active_materialization_bytes=stats.active_materialization_bytes,
        ),
        scattered_state=(;
            dispatches=stats.scattered_dispatches,
            pointer_bytes_uploaded=stats.pointer_bytes_uploaded,
            pointer_table_builds=stats.pointer_table_builds,
            pointer_table_reuses=stats.pointer_table_reuses,
            pointer_table_bytes=stats.pointer_table_bytes,
            workspace_allocations=stats.workspace_allocations,
            workspace_reuses=stats.workspace_reuses,
            workspace_bytes=stats.workspace_bytes,
        ),
        forced_gc_calls=stats.forced_gc_calls,
        staging=(;
            read_tasks=stats.read_tasks,
            parallel_read_layers=stats.parallel_read_layers,
            read_buffer_bytes=stats.read_buffer_bytes,
            host_buffer_bytes=stats.host_buffer_bytes,
        ),
    )
end

function run_request(session, tokens, decode_token; measured::Bool)
    prefill_call = captured_cuda(
        () -> prefill_hf_qwen3_moe_offload!(session, tokens);
        count_allocations=measured,
    )
    decode_call = captured_cuda(
        () -> decode_hf_qwen3_moe_offload!(session, decode_token);
        count_allocations=measured,
    )
    prefill = prefill_call.value
    decode = decode_call.value
    prefill_logits = Float32.(vec(prefill.logits))
    decode_logits = Float32.(vec(decode.logits))
    routes = route_record(prefill, decode)
    allocated = measured ?
        prefill_call.report.julia_allocated_bytes +
            decode_call.report.julia_allocated_bytes : nothing
    return (;
        raw=(; prefill_logits, decode_logits, routes=routes.raw),
        report=(;
            timing=(;
                prefill_seconds=prefill_call.report.seconds,
                decode_seconds=decode_call.report.seconds,
                request_seconds=
                    prefill_call.report.seconds + decode_call.report.seconds,
            ),
            julia_allocated_bytes=(;
                prefill=prefill_call.report.julia_allocated_bytes,
                decode=decode_call.report.julia_allocated_bytes,
                total=allocated,
            ),
            process_io=(;
                prefill=prefill_call.report.process_io,
                decode=decode_call.report.process_io,
                total=(;
                    rchar=
                        prefill_call.report.process_io.rchar +
                        decode_call.report.process_io.rchar,
                    syscr=
                        prefill_call.report.process_io.syscr +
                        decode_call.report.process_io.syscr,
                    read_bytes=
                        prefill_call.report.process_io.read_bytes +
                        decode_call.report.process_io.read_bytes,
                ),
            ),
            gpu_memory=(;
                before_prefill=prefill_call.report.gpu_before,
                after_prefill=prefill_call.report.gpu_after,
                before_decode=decode_call.report.gpu_before,
                after_decode=decode_call.report.gpu_after,
                minimum_free_bytes=minimum((
                    prefill_call.report.gpu_before.free_bytes,
                    prefill_call.report.gpu_after.free_bytes,
                    decode_call.report.gpu_before.free_bytes,
                    decode_call.report.gpu_after.free_bytes,
                )),
                maximum_used_bytes=maximum((
                    prefill_call.report.gpu_before.used_bytes,
                    prefill_call.report.gpu_after.used_bytes,
                    decode_call.report.gpu_before.used_bytes,
                    decode_call.report.gpu_after.used_bytes,
                )),
            ),
            output=(;
                prefill=output_record(prefill_logits),
                decode=output_record(decode_logits),
            ),
            routes=routes.report,
            traffic=request_traffic(prefill, decode),
        ),
    )
end

const CONFIGURATIONS = (
    (;
        name="materialized_grouped",
        grouped_experts=true,
        dispatch=:materialized,
    ),
    (;
        name="grouped_scattered",
        grouped_experts=true,
        dispatch=:scattered,
    ),
    (;
        name="scalar_scattered",
        grouped_experts=false,
        dispatch=:scattered,
    ),
)

function configure!(session, configuration)
    configure_hf_qwen3_moe_expert_cache!(
        session;
        budget_bytes=cache_budget_bytes,
        policy=:layer_balanced_lru,
        dispatch=configuration.dispatch,
        grouped_experts=configuration.grouped_experts,
        gc_interval_layers,
        read_buffer_reuse=true,
        host_buffer_reuse=true,
        read_mode=:tensor,
        miss_pipeline=:overlapped,
        read_workers,
        pinned_upload=false,
    )
    GC.gc(true)
    CUDA.reclaim()
    return session
end

function raw_request_equal(left, right)
    return left.prefill_logits == right.prefill_logits &&
        left.decode_logits == right.decode_logits &&
        left.routes == right.routes
end

function run_configuration(session, configuration, workload_name, tokens)
    println(
        stderr,
        "workload=$workload_name configuration=$(configuration.name)",
    )
    configure!(session, configuration)
    gpu_start = gpu_memory_snapshot()

    warmups = Any[]
    for index in 1:warmup_runs
        println(stderr, "  warmup $index/$warmup_runs")
        request = run_request(session, tokens, decode_token; measured=false)
        push!(warmups, request.report)
    end

    measurements = Any[]
    baseline = nothing
    repeat_bitwise_exact = true
    for index in 1:measured_runs
        println(stderr, "  measured $index/$measured_runs")
        request = run_request(session, tokens, decode_token; measured=true)
        if baseline === nothing
            baseline = request.raw
        else
            repeat_bitwise_exact &= raw_request_equal(baseline, request.raw)
        end
        push!(measurements, request.report)
    end
    baseline === nothing && error("configuration produced no measured output")
    gpu_end = gpu_memory_snapshot()

    prefill_seconds = [entry.timing.prefill_seconds for entry in measurements]
    decode_seconds = [entry.timing.decode_seconds for entry in measurements]
    request_seconds = [entry.timing.request_seconds for entry in measurements]
    allocation_bytes = [
        entry.julia_allocated_bytes.total
        for entry in measurements
    ]
    return (;
        raw=baseline,
        report=(;
            name=configuration.name,
            grouped_experts=configuration.grouped_experts,
            dispatch=String(configuration.dispatch),
            gpu_start,
            gpu_end,
            warmup_runs,
            measured_runs,
            warmups=Tuple(warmups),
            measurements=Tuple(measurements),
            aggregate=(;
                prefill_seconds_median=median(prefill_seconds),
                decode_seconds_median=median(decode_seconds),
                request_seconds_median=median(request_seconds),
                request_seconds_mean=mean(request_seconds),
                request_seconds_minimum=minimum(request_seconds),
                request_seconds_maximum=maximum(request_seconds),
                julia_allocated_bytes_median=median(allocation_bytes),
                julia_allocated_bytes_mean=mean(allocation_bytes),
                repeat_bitwise_exact,
            ),
        ),
    )
end

function logits_comparison(reference_values, candidate_values)
    length(reference_values) == length(candidate_values) || error(
        "cannot compare logits with different lengths",
    )
    difference = abs.(reference_values .- candidate_values)
    return (;
        bitwise_exact=reference_values == candidate_values,
        max_abs=maximum(difference),
        mean_abs=mean(difference),
        argmax_match=argmax(reference_values) == argmax(candidate_values),
    )
end

function logits_diagnostic(reference_values, candidate_values)
    length(reference_values) == length(candidate_values) || error(
        "cannot compare logits with different lengths",
    )
    difference = abs.(reference_values .- candidate_values)
    return (;
        max_abs=maximum(difference),
        mean_abs=mean(difference),
        reference_argmax_1_based=argmax(reference_values),
        candidate_argmax_1_based=argmax(candidate_values),
        argmax_match=argmax(reference_values) == argmax(candidate_values),
    )
end

function request_comparison(reference_request, candidate_request)
    return (;
        output=(;
            prefill=logits_comparison(
                reference_request.prefill_logits,
                candidate_request.prefill_logits,
            ),
            decode=logits_comparison(
                reference_request.decode_logits,
                candidate_request.decode_logits,
            ),
        ),
        routes=(;
            exact=reference_request.routes == candidate_request.routes,
            reference_sha256=sha256_string(JSON3.write(
                reference_request.routes,
            )),
            candidate_sha256=sha256_string(JSON3.write(
                candidate_request.routes,
            )),
        ),
    )
end


function diagnostic_request_comparison(reference_request, candidate_request)
    return (;
        prefill=logits_diagnostic(
            reference_request.prefill_logits,
            candidate_request.prefill_logits,
        ),
        decode=logits_diagnostic(
            reference_request.decode_logits,
            candidate_request.decode_logits,
        ),
    )
end

function measured_traffic_signature(measurement)
    io = measurement.traffic.expert_io
    cache = measurement.traffic.cache
    return (;
        prefill_bytes_read=io.prefill_bytes_read,
        decode_bytes_read=io.decode_bytes_read,
        total_bytes_read=io.total_bytes_read,
        prefill_bytes_uploaded=io.prefill_bytes_uploaded,
        decode_bytes_uploaded=io.decode_bytes_uploaded,
        total_bytes_uploaded=io.total_bytes_uploaded,
        prefill_cache_hits=cache.prefill_hits,
        prefill_cache_misses=cache.prefill_misses,
        prefill_cache_evictions=cache.prefill_evictions,
        decode_cache_hits=cache.decode_hits,
        decode_cache_misses=cache.decode_misses,
        decode_cache_evictions=cache.decode_evictions,
        cache_hits=cache.hits,
        cache_misses=cache.misses,
        cache_evictions=cache.evictions,
    )
end

function measured_traffic_comparison(
    materialized_grouped,
    grouped_scattered,
)
    materialized = [
        measured_traffic_signature(measurement)
        for measurement in materialized_grouped.measurements
    ]
    scattered = [
        measured_traffic_signature(measurement)
        for measurement in grouped_scattered.measurements
    ]
    return (;
        exact=materialized == scattered,
        materialized_grouped=materialized,
        grouped_scattered=scattered,
    )
end

hf_layout(array) = permutedims(array, (3, 2, 1))
expected_prefill = Float32.(hf_layout(reference["logits"])[:, end, 1])
expected_decode = Float32.(vec(hf_layout(reference["decode_logits"])))
expected_prompt_active = [
    sort!(unique!(Int.(vec(reference["selected_experts.$layer"]))) .+ 1)
    for layer in 0:(num_reference_layers - 1)
]
expected_decode_active = [
    sort!(unique!(Int.(vec(reference["decode_selected_experts.$layer"]))) .+ 1)
    for layer in 0:(num_reference_layers - 1)
]

function reference_parity(request)
    prompt_layers = [
        sort!(unique!(reduce(vcat, (
            chunk.active_experts[layer]
            for chunk in request.routes.chunks
        ))))
        for layer in 1:num_reference_layers
    ]
    return (;
        output=(;
            prefill=logits_diagnostic(expected_prefill, request.prefill_logits),
            decode=logits_diagnostic(expected_decode, request.decode_logits),
        ),
        routes=(;
            prefill_active_set_layer_matches=count(identity, [
                prompt_layers[layer] == expected_prompt_active[layer]
                for layer in eachindex(expected_prompt_active)
            ]),
            decode_active_set_layer_matches=count(identity, [
                request.routes.decode[layer] == expected_decode_active[layer]
                for layer in eachindex(expected_decode_active)
            ]),
            layers=length(expected_prompt_active),
        ),
    )
end

gpu_before_load = gpu_memory_snapshot()
load_call = captured_cuda(
    () -> load_hf_qwen3_moe_offload_session(
        model_dir;
        context_tokens,
        prefill_chunk_tokens,
        grouped_experts=true,
        expert_cache_budget_bytes=cache_budget_bytes,
        expert_cache_policy=:layer_balanced_lru,
        expert_cache_dispatch=:materialized,
        expert_gc_interval_layers=gc_interval_layers,
        expert_read_buffer_reuse=true,
        expert_host_buffer_reuse=true,
        expert_read_mode=:tensor,
        expert_miss_pipeline=:overlapped,
        expert_read_workers=read_workers,
        expert_pinned_upload=false,
        to_device=CUDA.cu,
        on_resident_layer=(layer, total) ->
            (layer == 1 || layer % 8 == 0 || layer == total) &&
                println(stderr, "resident layer $layer/$total"),
    );
    count_allocations=false,
)
session = load_call.value
gpu_after_load = gpu_memory_snapshot()
plan = qwen3_moe_offload_plan(session.model, context_tokens)
session.resident_parameter_bytes == plan.resident_parameter_bytes || error(
    "resident parameter bytes differ from qwen3_moe_offload_plan",
)

workloads = (
    (;
        name="reference_2_token",
        tokens=reference_tokens,
        has_external_reference=true,
    ),
    (;
        name="wide_prefill",
        tokens=wide_tokens,
        has_external_reference=false,
    ),
)

case_reports = Dict{String,Any}()
grouped_output_routes_exact = true
grouped_traffic_exact = true
grouped_repeats_exact = true
for workload in workloads
    configuration_reports = Dict{String,Any}()
    raw = Dict{String,Any}()
    for configuration in CONFIGURATIONS
        result = run_configuration(
            session,
            configuration,
            workload.name,
            workload.tokens,
        )
        configuration_reports[configuration.name] = result.report
        raw[configuration.name] = result.raw
    end

    grouped_comparison = request_comparison(
        raw["materialized_grouped"],
        raw["grouped_scattered"],
    )
    scalar_comparison = diagnostic_request_comparison(
        raw["materialized_grouped"],
        raw["scalar_scattered"],
    )
    traffic_comparison = measured_traffic_comparison(
        configuration_reports["materialized_grouped"],
        configuration_reports["grouped_scattered"],
    )
    this_grouped_output_routes_exact =
        grouped_comparison.output.prefill.bitwise_exact &&
        grouped_comparison.output.decode.bitwise_exact &&
        grouped_comparison.routes.exact
    this_grouped_repeats_exact =
        configuration_reports["materialized_grouped"].aggregate.repeat_bitwise_exact &&
        configuration_reports["grouped_scattered"].aggregate.repeat_bitwise_exact
    global grouped_output_routes_exact =
        grouped_output_routes_exact && this_grouped_output_routes_exact
    global grouped_traffic_exact =
        grouped_traffic_exact && traffic_comparison.exact
    global grouped_repeats_exact =
        grouped_repeats_exact && this_grouped_repeats_exact
    external_reference = workload.has_external_reference ? Dict(
        name => reference_parity(output)
        for (name, output) in raw
    ) : nothing

    case_reports[workload.name] = (;
        prompt_tokens=length(workload.tokens),
        token_ids_1_based=workload.tokens,
        token_sha256=token_sha256(workload.tokens),
        decode_token_1_based=decode_token,
        configurations=configuration_reports,
        comparisons=(;
            grouped_scattered_vs_materialized_grouped=grouped_comparison,
            scalar_scattered_vs_materialized_grouped=scalar_comparison,
            grouped_measured_traffic=traffic_comparison,
        ),
        external_reference,
        grouped_output_and_routes_bitwise_exact=
            this_grouped_output_routes_exact,
        grouped_measured_traffic_exact=traffic_comparison.exact,
        grouped_repeat_bitwise_exact=this_grouped_repeats_exact,
    )
end

repo_root = normpath(joinpath(@__DIR__, ".."))
script_path = abspath(@__FILE__)
report = (;
    schema_version=1,
    benchmark="qwen3_moe_cuda_grouped_scattered_same_process",
    model_id=EXPECTED_MODEL_ID,
    revision=String(metadata["revision"]),
    compute_dtype="bfloat16",
    model_dir,
    reference_dir,
    environment=(;
        julia_version=string(VERSION),
        julia_threads=Threads.nthreads(),
        cuda_runtime=string(CUDA.runtime_version()),
        gpu_name=CUDA.name(CUDA.device()),
        gpu_capability=string(CUDA.capability(CUDA.device())),
    ),
    session=(;
        context_tokens,
        prefill_chunk_tokens,
        cache_budget_bytes,
        cache_policy="layer_balanced_lru",
        gc_interval_layers,
        read_workers,
        warmup_runs,
        measured_runs,
        resident_parameter_bytes=session.resident_parameter_bytes,
        offload_plan=plan,
        load_seconds=load_call.report.seconds,
        gpu_before_load,
        gpu_after_load,
    ),
    cases=case_reports,
    verification=(;
        materialized_grouped_vs_grouped_scattered_bitwise_exact=
            grouped_output_routes_exact,
        materialized_grouped_vs_grouped_scattered_measured_traffic_exact=
            grouped_traffic_exact,
        grouped_measured_repeats_bitwise_exact=grouped_repeats_exact,
        scalar_scattered_is_diagnostic_only=true,
        passed=grouped_output_routes_exact &&
            grouped_traffic_exact &&
            grouped_repeats_exact,
    ),
    source_sha256=(;
        benchmark_script=bytes2hex(sha256(read(script_path))),
        offload_implementation=bytes2hex(sha256(read(joinpath(
            repo_root,
            "src",
            "generation",
            "qwen3_moe_offload.jl",
        )))),
        cuda_extension=bytes2hex(sha256(read(joinpath(
            repo_root,
            "ext",
            "LifeAICUDAExt.jl",
        )))),
    ),
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println(JSON3.write((;
    output_path,
    verification=report.verification,
)))
report.verification.passed || error(
    "grouped scattered verification diverged from materialized grouped; " *
    "diagnostic report written to $output_path",
)
