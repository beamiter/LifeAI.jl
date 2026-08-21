#!/usr/bin/env julia

using BFloat16s: BFloat16
using CUDA
using JSON3
using LifeAI
using LinearAlgebra: dot, norm
using SHA: sha256
using Statistics: mean, median

const _STATIC_CAPACITY = 128
const _STATIC_GENERATED_TOKENS = 32
const _STATIC_PERFORMANCE_SAMPLES = 3
const _STATIC_EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
const _STATIC_EXPECTED_MODELSCOPE_REVISION =
    "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53"
const _STATIC_EXPECTED_HF_REVISION =
    "78448d793a7eb2f7a987a1da76d464384aa1becd"
const _STATIC_EXPECTED_REFERENCE_SHA256 =
    "a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81"
const _STATIC_EXPECTED_METADATA_SHA256 =
    "569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc"
const _STATIC_EXPECTED_IMAGE_SHA256 =
    "ec143579b36852cf212bbb368798479d193a8c6d039942fca89a49fc820dff3f"
const _STATIC_EXPECTED_RENDERED_PROMPT_SHA256 =
    "7a50d10ccb53359de53e3e9b032c39b15fd3abbfed51ec844117c3b93da07271"
const _STATIC_EXPECTED_GREEDY_IDS_1_BASED = [1987, 2169, 375, 265]
const _STATIC_EXPECTED_GENERATED_TEXT = "This image is a"
const _STATIC_EXPECTED_ASSET_SHA256 = Dict(
    "model.safetensors" =>
        "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    "config.json" =>
        "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "preprocessor_config.json" =>
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer_config.json" =>
        "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "tokenizer.json" =>
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "generation_config.json" =>
        "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4",
    "chat_template.json" =>
        "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
)

_static_file_sha256(path) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function _static_parse_arguments(args)
    usage = "usage: julia --project=. " *
        "scripts/verify_qwen3_vl_static_cache_cuda.jl " *
        "[MODEL_DIR] REFERENCE_DIR [cuda] [float32|bfloat16]"
    values = String.(args)
    dtype_name = "float32"
    if !isempty(values) && last(values) in ("float32", "f32", "bfloat16", "bf16")
        raw = pop!(values)
        dtype_name = raw in ("float32", "f32") ? "float32" : "bfloat16"
    end
    length(values) in 1:3 || error(usage)
    backend = "cuda"
    if length(values) == 1
        model_dir = ""
        for name in ("LIFEAI_QWEN3_VL_MODEL_DIR", "QWEN3_VL_MODEL_DIR")
            value = get(ENV, name, "")
            isempty(value) || (model_dir = value; break)
        end
        isempty(model_dir) && error(
            "checkpoint directory is missing; pass MODEL_DIR or set " *
            "LIFEAI_QWEN3_VL_MODEL_DIR",
        )
        reference_dir = values[1]
    elseif length(values) == 2 && values[2] == "cuda"
        model_dir = ""
        for name in ("LIFEAI_QWEN3_VL_MODEL_DIR", "QWEN3_VL_MODEL_DIR")
            value = get(ENV, name, "")
            isempty(value) || (model_dir = value; break)
        end
        isempty(model_dir) && error(
            "checkpoint directory is missing; pass MODEL_DIR or set " *
            "LIFEAI_QWEN3_VL_MODEL_DIR",
        )
        reference_dir = values[1]
    else
        model_dir = values[1]
        reference_dir = values[2]
        length(values) == 3 && (backend = values[3])
    end
    backend == "cuda" || error("the static-cache verifier requires CUDA")
    chapter45_args = String[abspath(model_dir), abspath(reference_dir), "cuda"]
    return (;
        model_dir=abspath(model_dir),
        reference_dir=abspath(reference_dir),
        dtype_name,
        chapter45_args,
    )
end

function _static_stage_metrics(actual, expected)
    values = Float64.(Array(actual))
    reference = Float64.(expected)
    size(values) == size(reference) || error(
        "static-cache stage shape mismatch: $(size(values)) != $(size(reference))",
    )
    all(isfinite, values) || error("static-cache stage contains non-finite values")
    differences = abs.(values .- reference)
    denominator = norm(reference)
    relative_l2 = denominator == 0 ? norm(values) :
        norm(values .- reference) / denominator
    cosine_denominator = norm(values) * denominator
    cosine = cosine_denominator == 0 ? 1.0 :
        dot(vec(values), vec(reference)) / cosine_denominator
    return (;
        max_abs=maximum(differences),
        mean_abs=mean(differences),
        relative_l2,
        cosine,
    )
end

function _static_logits_reference(reader, name)
    value = read_safetensors_tensor(reader, name; target_dtype=Float32)
    (size(value, 1) == 1 && size(value, 2) == 1) || error(
        "$name is not a batch-1 token logit",
    )
    return permutedims(value, (3, 2, 1))
end

function _static_storage_bytes(cache)
    return sum(cache.layers; init=0) do layer
        length(layer.keys) * sizeof(eltype(layer.keys)) +
            length(layer.values) * sizeof(eltype(layer.values))
    end
end

function _static_storage_snapshot(cache)
    key_refs = map(layer -> layer.keys, cache.layers)
    value_refs = map(layer -> layer.values, cache.layers)
    key_pointers = map(value -> UInt(pointer(value)), key_refs)
    value_pointers = map(value -> UInt(pointer(value)), value_refs)
    return (; key_refs, value_refs, key_pointers, value_pointers)
end

function _static_assert_storage(cache, snapshot, spec, dtype, expected_bytes)
    cache.capacity == _STATIC_CAPACITY || error("static cache capacity changed")
    cache.batch_size == 1 || error("static cache batch size changed")
    length(cache.layers) == spec.num_hidden_layers || error(
        "static cache layer count changed",
    )
    expected_shape = (
        spec.head_dim,
        spec.num_key_value_heads,
        _STATIC_CAPACITY,
        1,
    )
    for index in eachindex(cache.layers)
        layer = cache.layers[index]
        layer.keys === snapshot.key_refs[index] || error(
            "layer $index key object identity changed",
        )
        layer.values === snapshot.value_refs[index] || error(
            "layer $index value object identity changed",
        )
        UInt(pointer(layer.keys)) == snapshot.key_pointers[index] || error(
            "layer $index key device pointer changed",
        )
        UInt(pointer(layer.values)) == snapshot.value_pointers[index] || error(
            "layer $index value device pointer changed",
        )
        layer.keys isa CUDA.CuArray || error("layer $index keys are not on CUDA")
        layer.values isa CUDA.CuArray || error("layer $index values are not on CUDA")
        eltype(layer.keys) === dtype || error("layer $index key dtype changed")
        eltype(layer.values) === dtype || error("layer $index value dtype changed")
        size(layer.keys) == expected_shape || error("layer $index key shape changed")
        size(layer.values) == expected_shape || error("layer $index value shape changed")
    end
    pointers = UInt[
        snapshot.key_pointers...,
        snapshot.value_pointers...,
    ]
    length(pointers) == 2 * spec.num_hidden_layers || error(
        "static cache does not contain the expected K/V buffers",
    )
    length(unique(pointers)) == length(pointers) || error(
        "static cache K/V buffers alias one another",
    )
    all(!iszero, pointers) || error("static cache contains a null device pointer")
    _static_storage_bytes(cache) == expected_bytes || error(
        "static cache logical allocated bytes changed",
    )
    return nothing
end

function _static_bitwise_equal(left, right)
    left_host = Array(left)
    right_host = Array(right)
    size(left_host) == size(right_host) || return false
    eltype(left_host) === eltype(right_host) || return false
    return reinterpret(UInt8, vec(left_host)) ==
        reinterpret(UInt8, vec(right_host))
end

function _static_assert_prefix_equal(static_cache, dynamic_cache)
    static_cache.position == dynamic_cache.position || error(
        "static and dynamic cache positions differ",
    )
    tokens = static_cache.position
    for index in eachindex(static_cache.layers)
        static_layer = static_cache.layers[index]
        dynamic_layer = dynamic_cache.layers[index]
        static_keys = @view static_layer.keys[:, :, 1:tokens, :]
        static_values = @view static_layer.values[:, :, 1:tokens, :]
        _static_bitwise_equal(static_keys, dynamic_layer.keys) || error(
            "layer $index static/dynamic key prefix is not bitwise equal",
        )
        _static_bitwise_equal(static_values, dynamic_layer.values) || error(
            "layer $index static/dynamic value prefix is not bitwise equal",
        )
    end
    return nothing
end

function _static_verify_four_phases(
    text_parameters,
    features,
    input_ids,
    rope_layout,
    checkpoint,
    metadata,
    reference,
    dtype,
)
    spec = checkpoint.text
    bytes_per_token = 2 * spec.num_hidden_layers * spec.head_dim *
        spec.num_key_value_heads * sizeof(dtype)
    expected_bytes = bytes_per_token * _STATIC_CAPACITY
    dynamic_cache = init_qwen3_vl_kv_cache(text_parameters; batch_size=1)
    static_cache = init_qwen3_vl_static_kv_cache(
        text_parameters;
        capacity=_STATIC_CAPACITY,
        batch_size=1,
    )
    snapshot = _static_storage_snapshot(static_cache)
    _static_assert_storage(static_cache, snapshot, spec, dtype, expected_bytes)

    dynamic_prefill, dynamic_cache = hf_qwen3_vl_text_prefill_cached(
        text_parameters,
        input_ids,
        rope_layout;
        vision_features=features,
        cache=dynamic_cache,
        logits_to_keep=1,
    )
    static_prefill, returned_cache = hf_qwen3_vl_text_prefill_static(
        text_parameters,
        input_ids,
        rope_layout;
        vision_features=features,
        cache=static_cache,
        logits_to_keep=1,
    )
    CUDA.synchronize()
    returned_cache === static_cache || error("static prefill replaced the cache object")

    phase_names = ("prefill", "decode.0", "decode.1", "decode.2")
    phases = metadata["phases"]
    static_logits = static_prefill.logits
    dynamic_logits = dynamic_prefill.logits
    generated_ids = Int[]
    println("static_stage\tmax_abs\tmean_abs\trelative_l2\tcosine\targmax_1_based")
    for phase_index in eachindex(phase_names)
        phase = phase_names[phase_index]
        String(phases[phase_index]["name"]) == phase || error(
            "reference phase order changed",
        )
        expected_position = Int(phases[phase_index]["cache_length"])
        static_cache.position == expected_position || error(
            "$phase static cache position changed",
        )
        static_cache.rope_delta == -56 || error("$phase static rope_delta changed")
        _static_assert_storage(static_cache, snapshot, spec, dtype, expected_bytes)
        _static_assert_prefix_equal(static_cache, dynamic_cache)
        _static_bitwise_equal(static_logits, dynamic_logits) || error(
            "$phase static and dynamic logits are not bitwise equal",
        )

        expected_logits = _static_logits_reference(reference, "$phase.logits")
        metrics = _static_stage_metrics(static_logits, expected_logits)
        token_id = argmax(vec(Array(static_logits)))
        reference_token = Int(phases[phase_index]["top_two"]["top1_token_id_1_based"])
        token_id == reference_token || error(
            "$phase greedy token $token_id != frozen reference $reference_token",
        )
        push!(generated_ids, token_id)
        println(join((
            phase,
            metrics.max_abs,
            metrics.mean_abs,
            metrics.relative_l2,
            metrics.cosine,
            token_id,
        ), '\t'))
        if dtype === Float32
            metrics.max_abs <= 1.0e-4 || error("$phase Float32 max error exceeded")
            metrics.mean_abs <= 2.0e-5 || error("$phase Float32 mean error exceeded")
            metrics.relative_l2 <= 5.0e-6 || error(
                "$phase Float32 relative L2 exceeded",
            )
            metrics.cosine >= 0.99999999 || error("$phase Float32 cosine fell")
            margin = Float64(phases[phase_index]["top_two"]["margin_f32"])
            2 * metrics.max_abs < margin || error(
                "$phase Float32 error does not preserve the frozen greedy margin",
            )
        end

        phase_index == length(phase_names) && continue
        dynamic_logits, dynamic_cache = hf_qwen3_vl_text_decode_step(
            text_parameters,
            token_id,
            dynamic_cache,
        )
        static_logits, returned_cache = hf_qwen3_vl_text_decode_step_static(
            text_parameters,
            token_id,
            static_cache,
        )
        CUDA.synchronize()
        returned_cache === static_cache || error(
            "static decode replaced the cache object",
        )
        coordinate = expected_position + static_cache.rope_delta
        reference_coordinate = Int(
            phases[phase_index + 1]["mrope_position_ids_thw_0_based"][1],
        )
        coordinate == reference_coordinate || error(
            "static decode mRoPE coordinate changed",
        )
    end

    generated_ids == _STATIC_EXPECTED_GREEDY_IDS_1_BASED || error(
        "static four-stage greedy ids changed",
    )
    fixed_bytes = _static_storage_bytes(static_cache)
    reset_qwen3_vl_static_kv_cache!(static_cache) === static_cache || error(
        "static cache reset replaced the cache object",
    )
    isempty(static_cache) || error("static cache reset did not clear position")
    static_cache.rope_delta == 0 || error("static cache reset retained rope_delta")
    _static_assert_storage(static_cache, snapshot, spec, dtype, expected_bytes)
    _static_storage_bytes(static_cache) == fixed_bytes || error(
        "static cache reset changed logical allocated bytes",
    )
    return (; bytes_per_token, expected_bytes, generated_ids)
end

function _static_generation_run(
    text_parameters,
    features,
    input_ids,
    rope_layout,
    mode::Symbol,
)
    timed = CUDA.@timed generate_hf_qwen3_vl_tokens(
        text_parameters,
        input_ids,
        rope_layout;
        vision_features=features,
        max_new_tokens=_STATIC_GENERATED_TOKENS,
        stop_token_ids=Int[],
        capture_logits=false,
        cache=mode,
        static_capacity=mode === :static ? _STATIC_CAPACITY : nothing,
    )
    value = timed.value
    record = (;
        generated_ids=copy(value.generated_ids),
        position=value.cache.position,
        rope_delta=value.cache.rope_delta,
        capacity=mode === :static ? value.cache.capacity : value.cache.position,
        seconds=timed.time,
        cpu_bytes=timed.cpu_bytes,
        cpu_gctime=timed.cpu_gctime,
        gpu_bytes=timed.gpu_bytes,
        gpu_allocations=timed.gpu_memstats.alloc_count,
    )
    return record
end

function _static_benchmark_32_tokens(
    text_parameters,
    features,
    input_ids,
    rope_layout,
    checkpoint,
    dtype,
)
    length(input_ids) == 76 || error("32-token benchmark prompt length changed")
    required_capacity = length(input_ids) + _STATIC_GENERATED_TOKENS - 1
    required_capacity == 107 || error("32-token benchmark cache length changed")
    _STATIC_CAPACITY >= required_capacity || error(
        "static benchmark capacity is too small",
    )

    # Compile and populate library workspaces before collecting allocation and
    # latency samples. These calls are deliberately excluded from the report.
    for mode in (:dynamic, :static)
        warm = _static_generation_run(
            text_parameters,
            features,
            input_ids,
            rope_layout,
            mode,
        )
        warm.generated_ids[1:4] == _STATIC_EXPECTED_GREEDY_IDS_1_BASED || error(
            "$mode warmup differs from the frozen four-token prefix",
        )
        warm.position == required_capacity || error("$mode warmup cache length changed")
        warm = nothing
        GC.gc(false)
    end

    dynamic_samples = NamedTuple[]
    static_samples = NamedTuple[]
    for sample_index in 1:_STATIC_PERFORMANCE_SAMPLES
        records = Dict{Symbol,NamedTuple}()
        order = isodd(sample_index) ? (:dynamic, :static) : (:static, :dynamic)
        for mode in order
            GC.gc(false)
            records[mode] = _static_generation_run(
                text_parameters,
                features,
                input_ids,
                rope_layout,
                mode,
            )
        end
        dynamic = records[:dynamic]
        static = records[:static]
        dynamic.generated_ids == static.generated_ids || error(
            "32-token static and dynamic greedy sequences differ",
        )
        static.generated_ids[1:4] == _STATIC_EXPECTED_GREEDY_IDS_1_BASED || error(
            "32-token run differs from the frozen four-token prefix",
        )
        dynamic.position == required_capacity || error(
            "32-token dynamic cache position changed",
        )
        static.position == required_capacity || error(
            "32-token static cache position changed",
        )
        dynamic.rope_delta == -56 || error("32-token dynamic rope_delta changed")
        static.rope_delta == -56 || error("32-token static rope_delta changed")
        static.capacity == _STATIC_CAPACITY || error(
            "32-token static capacity changed",
        )
        push!(dynamic_samples, dynamic)
        push!(static_samples, static)
    end

    dynamic_gpu = [sample.gpu_bytes for sample in dynamic_samples]
    static_gpu = [sample.gpu_bytes for sample in static_samples]
    dynamic_seconds = [sample.seconds for sample in dynamic_samples]
    static_seconds = [sample.seconds for sample in static_samples]
    bytes_per_token = 2 * checkpoint.text.num_hidden_layers *
        checkpoint.text.head_dim * checkpoint.text.num_key_value_heads * sizeof(dtype)
    append_count = _STATIC_GENERATED_TOKENS - 1
    dynamic_cat_output_bytes = bytes_per_token * sum(77:required_capacity)
    dynamic_old_prefix_copy_bytes = bytes_per_token * sum(76:(required_capacity - 1))
    static_slice_write_bytes = bytes_per_token * append_count
    static_capacity_bytes = bytes_per_token * _STATIC_CAPACITY
    expected_gpu_allocation_savings =
        dynamic_cat_output_bytes - static_capacity_bytes
    measured_gpu_allocation_savings = dynamic_gpu .- static_gpu
    all(==(expected_gpu_allocation_savings), measured_gpu_allocation_savings) ||
        error(
            "measured static GPU allocation savings differ from the exact " *
            "cat-minus-capacity contract",
        )
    dynamic_allocation_counts =
        [sample.gpu_allocations for sample in dynamic_samples]
    static_allocation_counts =
        [sample.gpu_allocations for sample in static_samples]
    measured_allocation_count_savings =
        dynamic_allocation_counts .- static_allocation_counts
    expected_allocation_count_savings =
        2 * checkpoint.text.num_hidden_layers * (append_count - 1)
    all(==(expected_allocation_count_savings), measured_allocation_count_savings) ||
        error("static GPU allocation-count savings changed")

    println("generation_tokens\t", _STATIC_GENERATED_TOKENS)
    println("decode_appends\t", append_count)
    println("final_cache_position\t", required_capacity)
    println("dynamic_seconds_samples\t", join(dynamic_seconds, ','))
    println("static_seconds_samples\t", join(static_seconds, ','))
    println("dynamic_seconds_median\t", median(dynamic_seconds))
    println("static_seconds_median\t", median(static_seconds))
    println("dynamic_gpu_allocated_bytes_samples\t", join(dynamic_gpu, ','))
    println("static_gpu_allocated_bytes_samples\t", join(static_gpu, ','))
    println("dynamic_gpu_allocated_bytes_median\t", median(dynamic_gpu))
    println("static_gpu_allocated_bytes_median\t", median(static_gpu))
    println(
        "measured_gpu_allocation_savings_samples\t",
        join(measured_gpu_allocation_savings, ','),
    )
    println(
        "expected_gpu_allocation_savings\t",
        expected_gpu_allocation_savings,
    )
    println(
        "dynamic_cpu_allocated_bytes_samples\t",
        join((sample.cpu_bytes for sample in dynamic_samples), ','),
    )
    println(
        "static_cpu_allocated_bytes_samples\t",
        join((sample.cpu_bytes for sample in static_samples), ','),
    )
    println(
        "dynamic_cpu_gc_seconds_samples\t",
        join((sample.cpu_gctime for sample in dynamic_samples), ','),
    )
    println(
        "static_cpu_gc_seconds_samples\t",
        join((sample.cpu_gctime for sample in static_samples), ','),
    )
    println(
        "dynamic_gpu_allocation_count_samples\t",
        join(dynamic_allocation_counts, ','),
    )
    println(
        "static_gpu_allocation_count_samples\t",
        join(static_allocation_counts, ','),
    )
    println(
        "measured_gpu_allocation_count_savings_samples\t",
        join(measured_allocation_count_savings, ','),
    )
    println(
        "expected_gpu_allocation_count_savings\t",
        expected_allocation_count_savings,
    )
    println("theoretical_dynamic_cat_output_bytes\t", dynamic_cat_output_bytes)
    println(
        "theoretical_dynamic_old_prefix_copy_bytes\t",
        dynamic_old_prefix_copy_bytes,
    )
    println("theoretical_static_slice_write_bytes\t", static_slice_write_bytes)
    println("latency_is_acceptance_gate\tfalse")
    return nothing
end

function _static_deterministic_image()
    image = Array{UInt8}(undef, 256, 256, 3)
    @inbounds for row in 1:256, column in 1:256
        x = column - 1
        y = row - 1
        image[row, column, 1] = UInt8(mod(3 * x + 5 * y + 17, 256))
        image[row, column, 2] = UInt8(mod(11 * x + 7 * y + 29, 256))
        image[row, column, 3] = UInt8(mod(13 * x + 19 * y + 43, 256))
    end
    return image
end

function _static_prepare_bfloat16(model_dir, reference_dir)
    metadata_path = joinpath(reference_dir, "reference.json")
    reference_path = joinpath(reference_dir, "reference.safetensors")
    _static_file_sha256(metadata_path) == _STATIC_EXPECTED_METADATA_SHA256 || error(
        "decode reference metadata SHA-256 is not frozen",
    )
    _static_file_sha256(reference_path) == _STATIC_EXPECTED_REFERENCE_SHA256 || error(
        "decode reference safetensors SHA-256 is not frozen",
    )
    metadata = JSON3.read(read(metadata_path, String))
    String(metadata["model_id"]) == _STATIC_EXPECTED_MODEL_ID || error(
        "decode reference model id changed",
    )
    String(metadata["modelscope_revision"]) ==
        _STATIC_EXPECTED_MODELSCOPE_REVISION || error(
        "decode reference ModelScope revision changed",
    )
    String(metadata["huggingface_revision"]) == _STATIC_EXPECTED_HF_REVISION || error(
        "decode reference Hugging Face revision changed",
    )
    String(metadata["compute_dtype"]) == "float32" || error(
        "BF16 smoke must use the frozen Float32 semantic decode oracle",
    )
    String(metadata["compute_device"]) == "cpu" || error(
        "BF16 smoke reference must be the frozen CPU oracle",
    )
    String(metadata["reference_sha256"]) == _STATIC_EXPECTED_REFERENCE_SHA256 || error(
        "decode reference payload hash changed",
    )
    Int.(collect(metadata["greedy_token_ids_0_based"])) .+ 1 ==
        _STATIC_EXPECTED_GREEDY_IDS_1_BASED || error(
        "decode reference greedy ids changed",
    )
    String(metadata["generated_text"]) == _STATIC_EXPECTED_GENERATED_TEXT || error(
        "decode reference generated text changed",
    )
    for (filename, expected) in _STATIC_EXPECTED_ASSET_SHA256
        String(metadata["asset_sha256"][filename]) == expected || error(
            "reference asset hash changed for $filename",
        )
    end

    actual_revision = readchomp(`git -C $model_dir rev-parse HEAD`)
    actual_revision == _STATIC_EXPECTED_MODELSCOPE_REVISION || error(
        "checkpoint is not the frozen ModelScope revision",
    )
    isempty(readchomp(
        `git -C $model_dir status --porcelain=v1 --untracked-files=all`,
    )) || error("checkpoint git tree is dirty")
    checkpoint = qwen3_vl_checkpoint_spec()
    report = verify_qwen3_vl_checkpoint(model_dir)
    report.config.text == checkpoint.text || error("checkpoint text config changed")
    report.config.vision == checkpoint.vision || error("checkpoint vision config changed")
    processor_spec = load_hf_qwen3_vl_processor_config(
        joinpath(model_dir, "preprocessor_config.json"),
        checkpoint.vision,
    )

    image = _static_deterministic_image()
    image_bytes = vec(permutedims(image, (3, 2, 1)))
    bytes2hex(sha256(image_bytes)) == _STATIC_EXPECTED_IMAGE_SHA256 || error(
        "deterministic image changed",
    )
    processed = qwen3_vl_process_image(image; layout=:hwc, spec=processor_spec)
    processed.grid_thw == reshape(Int[1, 16, 16], 3, 1) || error(
        "processed image grid changed",
    )
    tokenizer = load_hf_qwen3_vl_tokenizer(
        model_dir;
        revision=_STATIC_EXPECTED_MODELSCOPE_REVISION,
    )
    messages = [(
        role="user",
        content=Any[
            (type="image", image=image),
            (type="text", text="Describe."),
        ],
    )]
    rendered_prompt = apply_qwen3_vl_chat_template(
        tokenizer,
        messages;
        add_generation_prompt=true,
        add_vision_id=false,
    )
    bytes2hex(sha256(codeunits(rendered_prompt))) ==
        _STATIC_EXPECTED_RENDERED_PROMPT_SHA256 || error(
        "rendered prompt changed",
    )
    rendered_prompt == String(metadata["rendered_prompt"]) || error(
        "rendered prompt differs from the decode oracle",
    )
    expanded_prompt = qwen3_vl_expand_image_placeholders(
        rendered_prompt,
        processed.grid_thw;
        spec=processor_spec,
    )
    input_ids = encode(tokenizer, expanded_prompt; add_special_tokens=false)
    input_ids .- 1 == Int.(collect(metadata["input_ids_0_based"])) || error(
        "tokenization differs from the decode oracle",
    )
    rope_layout = qwen3_vl_rope_layout(
        input_ids,
        processed.grid_thw;
        checkpoint,
    )
    only(rope_layout.rope_deltas) == -56 || error("rope_delta changed")
    all(rope_layout.attention_mask) || error("prompt is unexpectedly padded")

    CUDA.reclaim()
    vision_load = @timed begin
        value = load_hf_qwen3_vl_vision_parameters(
            model_dir;
            target_dtype=BFloat16,
            to_device=CUDA.cu,
            spec=checkpoint.vision,
        )
        CUDA.synchronize()
        value
    end
    text_load = @timed begin
        value = load_hf_qwen3_vl_text_parameters(
            model_dir;
            target_dtype=BFloat16,
            to_device=CUDA.cu,
            checkpoint,
        )
        CUDA.synchronize()
        value
    end
    vision_input = Qwen3VLVisionInput(
        CUDA.cu(BFloat16.(processed.pixel_values)),
        processed.grid_thw;
        spec=checkpoint.vision,
    )
    vision_forward = @timed begin
        value = hf_qwen3_vl_vision_forward(vision_load.value, vision_input)
        CUDA.synchronize()
        value
    end
    reference = open_safetensors_reader(reference_path)
    println("bf16_vision_load_seconds\t", vision_load.time)
    println("bf16_text_load_seconds\t", text_load.time)
    println("bf16_vision_forward_seconds\t", vision_forward.time)
    return (;
        text_parameters=text_load.value,
        features=vision_forward.value,
        input_ids,
        rope_layout,
        checkpoint,
        metadata,
        reference,
    )
end

options = _static_parse_arguments(ARGS)
CUDA.functional() || error("CUDA.jl is not functional on this machine")
CUDA.allowscalar(false)

if options.dtype_name == "float32"
    empty!(ARGS)
    append!(ARGS, options.chapter45_args)
    include(joinpath(@__DIR__, "verify_qwen3_vl_decode_cuda.jl"))
    static_report = _static_verify_four_phases(
        text_parameters,
        features,
        input_ids,
        rope_layout,
        checkpoint,
        metadata,
        reference,
        Float32,
    )
    println("static_compute_dtype\tfloat32")
    println("static_capacity_tokens\t", _STATIC_CAPACITY)
    println("static_bytes_per_token\t", static_report.bytes_per_token)
    println("static_allocated_bytes\t", static_report.expected_bytes)
    println("static_final_valid_bytes\t", static_report.bytes_per_token * 79)
    println("static_storage_identity_passed\ttrue")
    println("static_dynamic_prefix_bitwise_passed\ttrue")
    println("static_float32_reference_parity_passed\ttrue")
    _static_benchmark_32_tokens(
        text_parameters,
        features,
        input_ids,
        rope_layout,
        checkpoint,
        Float32,
    )
else
    prepared = _static_prepare_bfloat16(options.model_dir, options.reference_dir)
    static_report = _static_verify_four_phases(
        prepared.text_parameters,
        prepared.features,
        prepared.input_ids,
        prepared.rope_layout,
        prepared.checkpoint,
        prepared.metadata,
        prepared.reference,
        BFloat16,
    )
    println("static_compute_dtype\tbfloat16")
    println("bf16_numerical_contract\tstatic_dynamic_same_device_smoke")
    println("bf16_hf_decode_tensor_parity_claimed\tfalse")
    println("static_capacity_tokens\t", _STATIC_CAPACITY)
    println("static_bytes_per_token\t", static_report.bytes_per_token)
    println("static_allocated_bytes\t", static_report.expected_bytes)
    println("static_final_valid_bytes\t", static_report.bytes_per_token * 79)
    println("static_storage_identity_passed\ttrue")
    println("static_dynamic_prefix_bitwise_passed\ttrue")
    println("static_bfloat16_smoke_passed\ttrue")
    _static_benchmark_32_tokens(
        prepared.text_parameters,
        prepared.features,
        prepared.input_ids,
        prepared.rope_layout,
        prepared.checkpoint,
        BFloat16,
    )
end
