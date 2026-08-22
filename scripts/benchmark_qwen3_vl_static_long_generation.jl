#!/usr/bin/env julia

const _CH47_MODEL_ENV = "LIFEAI_QWEN3_VL_MODEL_DIR"
const _CH47_REFERENCE_ENV = "LIFEAI_QWEN3_VL_REFERENCE_DIR"
const _CH47_OUTPUT_ENV = "LIFEAI_QWEN3_VL_LONG_PROFILE_OUTPUT"

function _ch47_usage(io::IO=stdout)
    println(io, "usage: julia --project=. --startup-file=no \\")
    println(io, "  scripts/benchmark_qwen3_vl_static_long_generation.jl \\")
    println(io, "  [MODEL_DIR [REFERENCE_DIR [OUTPUT_JSON]]]")
    println(io)
    println(io, "Defaults:")
    println(io, "  MODEL_DIR:     \$", _CH47_MODEL_ENV)
    println(io, "  REFERENCE_DIR: \$", _CH47_REFERENCE_ENV)
    println(io, "  OUTPUT_JSON:   \$", _CH47_OUTPUT_ENV, " or /tmp/qwen3_vl_long_profile.json")
    println(io)
    println(io, "Controls:")
    println(io, "  LIFEAI_QWEN3_VL_PROFILE_LENGTHS   default 32,128,256; each >= 4")
    println(io, "  LIFEAI_QWEN3_VL_PROFILE_REPEATS   default 3")
end

if any(argument -> argument in ("-h", "--help"), ARGS)
    _ch47_usage()
    exit()
end
length(ARGS) <= 3 || begin
    _ch47_usage(stderr)
    error("expected at most three positional arguments")
end

using CUDA
using JSON3
using LifeAI
using SHA: sha256
using Statistics: mean, median, quantile

# Chapter 46 now exposes its frozen single-image BF16 preparation as an
# import-safe script helper. Importing this file does not execute its verifier.
include(joinpath(@__DIR__, "verify_qwen3_vl_static_cache_cuda.jl"))

function _ch47_positive_int(name::AbstractString, default::Int)
    value = tryparse(Int, get(ENV, name, string(default)))
    value === nothing && error("$name must be an integer")
    value > 0 || error("$name must be positive")
    return value
end

function _ch47_lengths()
    raw = split(get(ENV, "LIFEAI_QWEN3_VL_PROFILE_LENGTHS", "32,128,256"), ',')
    isempty(raw) && error("profile length list is empty")
    values = Int[]
    for item in raw
        value = tryparse(Int, strip(item))
        value === nothing && error("profile lengths must be comma-separated integers")
        value >= 4 || error("each profile length must be at least four tokens")
        push!(values, value)
    end
    length(unique(values)) == length(values) || error(
        "profile lengths must not contain duplicates",
    )
    return sort(values)
end

function _ch47_required_path(args, index, environment, label)
    value = if length(args) >= index
        args[index]
    else
        get(ENV, environment, "")
    end
    isempty(value) && error("pass $label or set $environment")
    path = abspath(value)
    isdir(path) || error("$label does not exist: $path")
    return path
end

function _ch47_sha256_bytes(value)
    return bytes2hex(sha256(reinterpret(UInt8, vec(value))))
end

function _ch47_timing(timed)
    return (;
        seconds=timed.time,
        cpu_bytes=timed.cpu_bytes,
        cpu_gc_seconds=timed.cpu_gctime,
        gpu_bytes=timed.gpu_bytes,
        gpu_allocation_count=timed.gpu_memstats.alloc_count,
        gpu_memory_seconds=timed.gpu_memtime,
    )
end

function _ch47_host_timing(timed)
    return (;
        seconds=timed.time,
        cpu_bytes=timed.bytes,
        cpu_gc_seconds=timed.gctime,
    )
end

function _ch47_select_token(logits)
    timed = @timed begin
        values = vec(Array(@view(logits[:, end, 1])))
        count = min(2, length(values))
        top_ids = partialsortperm(values, 1:count; rev=true)
        token = first(top_ids)
        second_token = count == 2 ? top_ids[2] : token
        top_logit = Float32(values[token])
        second_logit = Float32(values[second_token])
        (;
            token,
            second_token,
            top_logit,
            second_logit,
            margin=top_logit - second_logit,
            values,
        )
    end
    value = timed.value
    return (;
        token=value.token,
        second_token=value.second_token,
        top_logit=value.top_logit,
        second_logit=value.second_logit,
        margin=value.margin,
        logits_sha256=_ch47_sha256_bytes(value.values),
        seconds=timed.time,
        cpu_bytes=timed.bytes,
        cpu_gc_seconds=timed.gctime,
    )
end

function _ch47_storage_snapshot(cache)
    return (;
        key_objects=map(layer -> layer.keys, cache.layers),
        value_objects=map(layer -> layer.values, cache.layers),
        key_pointers=map(layer -> UInt(pointer(layer.keys)), cache.layers),
        value_pointers=map(layer -> UInt(pointer(layer.values)), cache.layers),
    )
end

function _ch47_assert_storage_identity(cache, snapshot)
    for layer in eachindex(cache.layers)
        cache.layers[layer].keys === snapshot.key_objects[layer] || error(
            "static key object identity changed at layer $layer",
        )
        cache.layers[layer].values === snapshot.value_objects[layer] || error(
            "static value object identity changed at layer $layer",
        )
        UInt(pointer(cache.layers[layer].keys)) == snapshot.key_pointers[layer] ||
            error("static key device pointer changed at layer $layer")
        UInt(pointer(cache.layers[layer].values)) == snapshot.value_pointers[layer] ||
            error("static value device pointer changed at layer $layer")
    end
    return nothing
end

function _ch47_reset_pool_high_watermarks()
    pool = CUDA.memory_pool(CUDA.device())
    CUDA.attribute!(pool, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
    CUDA.attribute!(pool, CUDA.MEMPOOL_ATTR_RESERVED_MEM_HIGH, UInt64(0))
    return pool
end

function _ch47_pool_high_watermarks(pool)
    return (;
        used_bytes=Int(CUDA.attribute(
            UInt64,
            pool,
            CUDA.MEMPOOL_ATTR_USED_MEM_HIGH,
        )),
        reserved_bytes=Int(CUDA.attribute(
            UInt64,
            pool,
            CUDA.MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        )),
    )
end

function _ch47_memory_snapshot()
    CUDA.synchronize()
    return (;
        device_free_bytes=CUDA.free_memory(),
        pool_used_bytes=CUDA.used_memory(),
        pool_cached_bytes=CUDA.cached_memory(),
    )
end

function _ch47_sum(records, field::Symbol)
    return sum(record -> getproperty(record, field), records; init=0)
end

function _ch47_decode_summary(records)
    seconds = [record.seconds for record in records]
    return (;
        appends=length(records),
        seconds=sum(seconds),
        seconds_per_append=(;
            median=median(seconds),
            p95=quantile(seconds, 0.95),
            maximum=maximum(seconds),
        ),
        cpu_bytes=_ch47_sum(records, :cpu_bytes),
        cpu_gc_seconds=_ch47_sum(records, :cpu_gc_seconds),
        gpu_bytes=_ch47_sum(records, :gpu_bytes),
        gpu_bytes_per_append=mean(record.gpu_bytes for record in records),
        gpu_allocation_count=_ch47_sum(records, :gpu_allocation_count),
        gpu_allocations_per_append=mean(
            record.gpu_allocation_count for record in records
        ),
        raw=records,
    )
end

function _ch47_selection_summary(records)
    return (;
        calls=length(records),
        seconds=_ch47_sum(records, :seconds),
        cpu_bytes=_ch47_sum(records, :cpu_bytes),
        cpu_gc_seconds=_ch47_sum(records, :cpu_gc_seconds),
        raw=records,
    )
end

function _ch47_run_generation(prepared, generated_tokens::Int)
    prompt_tokens = length(prepared.input_ids)
    capacity = prompt_tokens + generated_tokens - 1
    capacity <= prepared.checkpoint.text.max_position_embeddings || error(
        "profile workload exceeds the checkpoint context limit",
    )

    GC.gc(false)
    pool = _ch47_reset_pool_high_watermarks()
    free_before = CUDA.free_memory()
    cache_timed = CUDA.@timed init_qwen3_vl_static_kv_cache(
        prepared.text_parameters;
        capacity,
        batch_size=1,
    )
    cache = cache_timed.value
    snapshot = _ch47_storage_snapshot(cache)
    prefill_timed = CUDA.@timed hf_qwen3_vl_text_prefill_static(
        prepared.text_parameters,
        prepared.input_ids,
        prepared.rope_layout;
        vision_features=prepared.features,
        cache,
        logits_to_keep=1,
    )
    prefill, returned = prefill_timed.value
    returned === cache || error("static prefill replaced the request cache")
    _ch47_assert_storage_identity(cache, snapshot)

    selection_records = NamedTuple[]
    first_selection = _ch47_select_token(prefill.logits)
    push!(selection_records, first_selection)
    generated_ids = Int[first_selection.token]
    final_logits_sha256 = first_selection.logits_sha256
    decode_records = NamedTuple[]
    token = first_selection.token
    for append_index in 1:(generated_tokens - 1)
        decode_timed = CUDA.@timed hf_qwen3_vl_text_decode_step_static(
            prepared.text_parameters,
            token,
            cache,
        )
        logits, returned = decode_timed.value
        returned === cache || error("static decode replaced the request cache")
        _ch47_assert_storage_identity(cache, snapshot)
        push!(decode_records, merge(
            (; append_index, cache_position=cache.position),
            _ch47_timing(decode_timed),
        ))
        selection = _ch47_select_token(logits)
        push!(selection_records, selection)
        token = selection.token
        push!(generated_ids, token)
        final_logits_sha256 = selection.logits_sha256
    end
    CUDA.synchronize()
    cache.position == capacity || error("final static cache position changed")
    cache.rope_delta == -56 || error("profile workload rope_delta changed")
    _ch47_assert_storage_identity(cache, snapshot)
    pool_high = _ch47_pool_high_watermarks(pool)
    free_after = CUDA.free_memory()

    return (;
        generated_tokens,
        prompt_tokens,
        decode_appends=generated_tokens - 1,
        capacity,
        final_position=cache.position,
        rope_delta=cache.rope_delta,
        generated_ids,
        generated_ids_sha256=bytes2hex(sha256(JSON3.write(generated_ids))),
        final_logits_sha256,
        cache_init=_ch47_timing(cache_timed),
        prefill=_ch47_timing(prefill_timed),
        decode=_ch47_decode_summary(decode_records),
        host_greedy_selection=_ch47_selection_summary(selection_records),
        pool_high_watermark=pool_high,
        device_free_bytes_before=free_before,
        device_free_bytes_after=free_after,
        storage_identity_passed=true,
    )
end

function _ch47_stage_aggregate(records)
    totals = Dict{Tuple{Symbol,Int},NamedTuple}()
    for record in records
        key = (record.stage, record.layer)
        previous = get(totals, key, (; gpu_bytes=0, gpu_allocation_count=0))
        totals[key] = (;
            gpu_bytes=previous.gpu_bytes + record.gpu_bytes,
            gpu_allocation_count=previous.gpu_allocation_count +
                record.gpu_allocation_count,
        )
    end
    entries = [
        (;
            stage=String(stage),
            layer,
            values.gpu_bytes,
            values.gpu_allocation_count,
        )
        for ((stage, layer), values) in totals
    ]
    sort!(entries; by=entry -> (-entry.gpu_bytes, entry.layer, entry.stage))
    return entries
end

function _ch47_stage_family_aggregate(records)
    totals = Dict{Symbol,NamedTuple}()
    for record in records
        previous = get(
            totals,
            record.stage,
            (; gpu_bytes=0, gpu_allocation_count=0, invocations=0),
        )
        totals[record.stage] = (;
            gpu_bytes=previous.gpu_bytes + record.gpu_bytes,
            gpu_allocation_count=previous.gpu_allocation_count +
                record.gpu_allocation_count,
            invocations=previous.invocations + 1,
        )
    end
    entries = [
        (;
            stage=String(stage),
            values.gpu_bytes,
            values.gpu_allocation_count,
            values.invocations,
        )
        for (stage, values) in totals
    ]
    sort!(entries; by=entry -> (
        -entry.gpu_bytes,
        -entry.gpu_allocation_count,
        entry.stage,
    ))
    return entries
end

function _ch47_stage_runner(stage_records)
    return function (stage, layer, thunk)
        timed = CUDA.@timed thunk()
        push!(stage_records, merge(
            (; stage, layer),
            _ch47_timing(timed),
        ))
        return timed.value
    end
end

function _ch47_warm_profiled_path(prepared)
    capacity = length(prepared.input_ids) + 1
    cache = init_qwen3_vl_static_kv_cache(
        prepared.text_parameters;
        capacity,
        batch_size=1,
    )
    prefill, _ = hf_qwen3_vl_text_prefill_static(
        prepared.text_parameters,
        prepared.input_ids,
        prepared.rope_layout;
        vision_features=prepared.features,
        cache,
        logits_to_keep=1,
    )
    token = _ch47_select_token(prefill.logits).token
    stage_records = NamedTuple[]
    LifeAI._profile_qwen3_vl_text_decode_step_static(
        prepared.text_parameters,
        token,
        cache,
        _ch47_stage_runner(stage_records),
    )
    CUDA.synchronize()
    isempty(stage_records) && error("profile runner warmup observed no stages")
    return nothing
end

function _ch47_run_attribution(prepared, baseline)
    generated_tokens = baseline.generated_tokens
    capacity = baseline.capacity
    cache = init_qwen3_vl_static_kv_cache(
        prepared.text_parameters;
        capacity,
        batch_size=1,
    )
    snapshot = _ch47_storage_snapshot(cache)
    prefill, _ = hf_qwen3_vl_text_prefill_static(
        prepared.text_parameters,
        prepared.input_ids,
        prepared.rope_layout;
        vision_features=prepared.features,
        cache,
        logits_to_keep=1,
    )
    selection = _ch47_select_token(prefill.logits)
    generated_ids = Int[selection.token]
    token = selection.token
    for _ in 1:max(0, generated_tokens - 2)
        logits, _ = hf_qwen3_vl_text_decode_step_static(
            prepared.text_parameters,
            token,
            cache,
        )
        selection = _ch47_select_token(logits)
        token = selection.token
        push!(generated_ids, token)
    end
    cache.position == capacity - 1 || error(
        "attribution cache did not stop before the final append",
    )

    stage_records = NamedTuple[]
    runner = _ch47_stage_runner(stage_records)
    outer = CUDA.@timed LifeAI._profile_qwen3_vl_text_decode_step_static(
        prepared.text_parameters,
        token,
        cache,
        runner,
    )
    logits, returned = outer.value
    returned === cache || error("profiled decode replaced the request cache")
    _ch47_assert_storage_identity(cache, snapshot)
    final_selection = _ch47_select_token(logits)
    push!(generated_ids, final_selection.token)

    generated_ids == baseline.generated_ids || error(
        "profiled decode changed the generated token sequence",
    )
    final_selection.logits_sha256 == baseline.final_logits_sha256 || error(
        "profiled decode changed the final logits",
    )
    stage_gpu_bytes = _ch47_sum(stage_records, :gpu_bytes)
    stage_gpu_allocations = _ch47_sum(stage_records, :gpu_allocation_count)
    stage_gpu_bytes == outer.gpu_bytes || error(
        "profile stages do not cover all GPU allocation traffic",
    )
    stage_gpu_allocations == outer.gpu_memstats.alloc_count || error(
        "profile stages do not cover all GPU allocations",
    )
    baseline_final = last(baseline.decode.raw)
    baseline_final.cache_position == capacity || error(
        "baseline final decode record has the wrong cache position",
    )
    outer.gpu_bytes == baseline_final.gpu_bytes || error(
        "profile instrumentation changed final-step GPU allocation traffic",
    )
    outer.gpu_memstats.alloc_count == baseline_final.gpu_allocation_count ||
        error("profile instrumentation changed final-step GPU allocation count")
    all(
        record -> record.stage !== :kv_write || record.gpu_bytes == 0,
        stage_records,
    ) || error("static K/V writes unexpectedly allocated GPU storage")
    stage_families = _ch47_stage_family_aggregate(stage_records)
    isempty(stage_families) && error("profile attribution observed no stages")

    return (;
        generated_tokens,
        final_position=cache.position,
        profiled_logits_match_baseline=true,
        profiled_tokens_match_baseline=true,
        storage_identity_passed=true,
        unprofiled_allocation_surface=(;
            gpu_bytes=baseline_final.gpu_bytes,
            gpu_allocation_count=baseline_final.gpu_allocation_count,
            exact=true,
        ),
        attribution_coverage=(;
            gpu_bytes=stage_gpu_bytes,
            outer_gpu_bytes=outer.gpu_bytes,
            gpu_allocation_count=stage_gpu_allocations,
            outer_gpu_allocation_count=outer.gpu_memstats.alloc_count,
            exact=true,
        ),
        kv_write_gpu_bytes=_ch47_sum(
            filter(record -> record.stage === :kv_write, stage_records),
            :gpu_bytes,
        ),
        dominant_stage=first(stage_families),
        stage_families,
        stages=_ch47_stage_aggregate(stage_records),
        raw=stage_records,
    )
end

model_dir = _ch47_required_path(ARGS, 1, _CH47_MODEL_ENV, "MODEL_DIR")
reference_dir = _ch47_required_path(
    ARGS,
    2,
    _CH47_REFERENCE_ENV,
    "REFERENCE_DIR",
)
output_path = abspath(length(ARGS) >= 3 ? ARGS[3] : get(
    ENV,
    _CH47_OUTPUT_ENV,
    "/tmp/qwen3_vl_long_profile.json",
))
lengths = _ch47_lengths()
repeats = _ch47_positive_int("LIFEAI_QWEN3_VL_PROFILE_REPEATS", 3)

CUDA.functional() || error("CUDA.jl is not functional on this machine")
CUDA.allowscalar(false)
preparation_timed = @timed _static_prepare_bfloat16(model_dir, reference_dir)
prepared = preparation_timed.value
preparation = merge(
    (; total=_ch47_host_timing(preparation_timed)),
    prepared.preparation,
)
length(prepared.input_ids) == 76 || error("frozen profile prompt length changed")

# One longest-shape warmup is excluded from every reported sample. Full GC
# releases the warmup request state while retaining CUDA library workspaces.
warmup = _ch47_run_generation(prepared, maximum(lengths))
warmup.generated_ids[1:4] == _STATIC_EXPECTED_GREEDY_IDS_1_BASED || error(
    "long-generation warmup changed the frozen four-token prefix",
)
_ch47_warm_profiled_path(prepared)
warmup_report = merge((; excluded_from_samples=true), warmup)
warmup = nothing
GC.gc(true)
steady_memory_baseline = _ch47_memory_snapshot()

samples = Dict(token_count => NamedTuple[] for token_count in lengths)
for repetition in 1:repeats
    order = isodd(repetition) ? lengths : reverse(lengths)
    for token_count in order
        sample = _ch47_run_generation(prepared, token_count)
        sample.generated_ids[1:4] == _STATIC_EXPECTED_GREEDY_IDS_1_BASED || error(
            "length-$token_count sample changed the frozen four-token prefix",
        )
        push!(samples[token_count], merge((; repetition), sample))
    end
end

for token_count in lengths
    hashes = unique(
        sample.generated_ids_sha256 for sample in samples[token_count]
    )
    length(hashes) == 1 || error(
        "length-$token_count samples are not deterministic",
    )
end
attribution = Dict(
    string(token_count) =>
        _ch47_run_attribution(prepared, first(samples[token_count]))
    for token_count in lengths
)

repo_root = normpath(joinpath(@__DIR__, ".."))
source_files = [
    "src/LifeAI.jl",
    "src/io/hf_qwen3_vl.jl",
    "src/data/hf_qwen3_vl_processor.jl",
    "src/models/bf16_accel.jl",
    "src/models/qwen3_vl_vision.jl",
    "src/models/qwen3_vl_decoder.jl",
    "src/models/qwen3_vl_cache.jl",
    "src/generation/hf_text_generation.jl",
    "src/generation/qwen3_vl_generation.jl",
    "scripts/verify_qwen3_vl_static_cache_cuda.jl",
    "scripts/benchmark_qwen3_vl_static_long_generation.jl",
]
source_sha256 = Dict(
    relative => _static_file_sha256(joinpath(repo_root, split(relative, '/')...))
    for relative in source_files
)
manifest_path = joinpath(repo_root, "Manifest.toml")
GC.gc(true)
post_run_memory = _ch47_memory_snapshot()
CUDA.reclaim()
post_reclaim_memory = _ch47_memory_snapshot()
report = (;
    schema_version=1,
    closed=false,
    chapter=47,
    title="Qwen3-VL long-generation allocation profile",
    environment=(;
        julia=string(VERSION),
        cuda_jl=string(pkgversion(CUDA)),
        cuda_driver=string(CUDA.driver_version()),
        cuda_runtime=string(CUDA.runtime_version()),
        device=CUDA.name(CUDA.device()),
        total_device_bytes=CUDA.total_memory(),
        repository_head=readchomp(`git -C $repo_root rev-parse HEAD`),
        repository_dirty=!isempty(readchomp(
            `git -C $repo_root status --porcelain=v1 --untracked-files=all`,
        )),
        project_sha256=_static_file_sha256(joinpath(repo_root, "Project.toml")),
        manifest_sha256=isfile(manifest_path) ?
            _static_file_sha256(manifest_path) : nothing,
        manifest_tracked=!isempty(readchomp(
            `git -C $repo_root ls-files -- Manifest.toml`,
        )),
        source_sha256,
    ),
    provenance=(;
        model_asset_sha256=_STATIC_EXPECTED_ASSET_SHA256,
        reference_safetensors_sha256=_STATIC_EXPECTED_REFERENCE_SHA256,
        reference_metadata_sha256=_STATIC_EXPECTED_METADATA_SHA256,
        deterministic_image_sha256=_STATIC_EXPECTED_IMAGE_SHA256,
        rendered_prompt_sha256=_STATIC_EXPECTED_RENDERED_PROMPT_SHA256,
    ),
    workload=(;
        model_id=_STATIC_EXPECTED_MODEL_ID,
        modelscope_revision=_STATIC_EXPECTED_MODELSCOPE_REVISION,
        huggingface_revision=_STATIC_EXPECTED_HF_REVISION,
        dtype="bfloat16",
        batch_size=1,
        images=1,
        prompt_tokens=length(prepared.input_ids),
        rope_delta=only(prepared.rope_layout.rope_deltas),
        generated_token_lengths=lengths,
        repeats,
        warmup_generated_tokens=maximum(lengths),
        stop_token_ids=Int[],
        greedy=true,
    ),
    semantics=(;
        allocation_bytes="CUDA.jl cumulative allocator traffic, not live workspace size",
        pool_high_watermark="CUDA memory-pool live/reserved high-water mark",
        device_free_snapshots="before/after snapshots, not a sampled physical peak",
        latency_is_acceptance_gate=false,
        hf_bf16_strict_parity_claimed=false,
        same_device_static_profile=true,
    ),
    preparation,
    warmup=warmup_report,
    samples=Dict(
        string(token_count) => samples[token_count] for token_count in lengths
    ),
    attribution,
    memory_drift=(;
        steady_baseline=steady_memory_baseline,
        after_samples_before_reclaim=post_run_memory,
        device_free_bytes_delta=post_run_memory.device_free_bytes -
            steady_memory_baseline.device_free_bytes,
        pool_used_bytes_delta=post_run_memory.pool_used_bytes -
            steady_memory_baseline.pool_used_bytes,
        pool_cached_bytes_delta=post_run_memory.pool_cached_bytes -
            steady_memory_baseline.pool_cached_bytes,
    ),
    reclaim=(;
        before=post_run_memory,
        after=post_reclaim_memory,
        recovered_device_free_bytes=post_reclaim_memory.device_free_bytes -
            post_run_memory.device_free_bytes,
        released_pool_used_bytes=post_run_memory.pool_used_bytes -
            post_reclaim_memory.pool_used_bytes,
        released_pool_cached_bytes=post_run_memory.pool_cached_bytes -
            post_reclaim_memory.pool_cached_bytes,
    ),
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
end
println(JSON3.write((;
    output_path,
    lengths,
    repeats,
    device=report.environment.device,
    closed=report.closed,
)))
