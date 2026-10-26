#!/usr/bin/env julia

# Real checkpoint acceptance only; loading this file never starts GPU work.
include(joinpath(@__DIR__, "verify_qwen3_vl_sampling_cuda.jl"))

const _MULTI_ORACLE_METADATA_SHA256 =
    "d0ce46eb851c371620897d31e72621610852bed26757e3a12e02ba5ef97b83eb"
const _MULTI_ORACLE_TENSORS_SHA256 =
    "b9e7a8735c5c2ab36bb89f9dc7079715fe001027a9bc7552ac49ea468e2e226e"

# The production checkpoint reader intentionally only accepts floating weights.
# This local oracle reader also needs exact UInt8 images and Int64 token/layout
# inputs, without broadening the production checkpoint format contract.
function _multi_reference(path, metadata)
    bytes = read(path)
    length(bytes) >= 8 || error("truncated oracle header")
    header_size = Int(ltoh(reinterpret(UInt64, bytes[1:8])[1]))
    0 < header_size <= length(bytes) - 8 || error("invalid oracle header size")
    header = JSON3.read(String(bytes[9:8 + header_size]))
    base = 8 + header_size
    result = Dict{String,Any}()
    for (raw_name, entry) in pairs(header)
        name = String(raw_name)
        name == "__metadata__" && continue
        dtype = String(entry["dtype"])
        T = dtype == "U8" ? UInt8 : dtype == "I64" ? Int64 :
            dtype == "F32" ? Float32 : dtype == "BF16" ? BFloat16 :
            error("unsupported oracle dtype $dtype")
        shape = Int.(collect(entry["shape"]))
        offsets = Int.(collect(entry["data_offsets"]))
        length(offsets) == 2 || error("invalid oracle offsets for $name")
        first, last = offsets
        0 <= first <= last <= length(bytes) - base || error("oracle offset out of range")
        prod(big.(shape)) * sizeof(T) == last - first || error("oracle byte count mismatch")
        payload = bytes[base + first + 1:base + last]
        bytes2hex(sha256(payload)) == String(metadata["tensor_sha256"][name]) ||
            error("oracle tensor hash mismatch for $name")
        values = copy(reinterpret(T, payload))
        result[name] = LifeAI._semantic_array(values, shape)
    end
    Set(keys(result)) == Set(String.(keys(metadata["tensor_sha256"]))) ||
        error("oracle tensor inventory mismatch")
    return result
end

function _multi_prepare(model_dir, reference_dir, dtype)
    metadata_path = joinpath(reference_dir, "reference.json")
    reference_path = joinpath(reference_dir, "reference.safetensors")
    _static_file_sha256(metadata_path) == _MULTI_ORACLE_METADATA_SHA256 ||
        error("multi-image oracle metadata is not the frozen two-process reference")
    _static_file_sha256(reference_path) == _MULTI_ORACLE_TENSORS_SHA256 ||
        error("multi-image oracle tensors are not the frozen two-process reference")
    metadata = JSON3.read(read(metadata_path, String))
    String(metadata["oracle"]) == "qwen3_vl_2b_two_images_dynamic_cache_greedy" ||
        error("wrong multi-image oracle")
    Int(metadata["schema_version"]) == 1 || error("unsupported oracle schema")
    String(metadata["model_id"]) == _STATIC_EXPECTED_MODEL_ID || error("model id mismatch")
    String(metadata["modelscope_revision"]) == _STATIC_EXPECTED_MODELSCOPE_REVISION ||
        error("ModelScope revision mismatch")
    String(metadata["huggingface_revision"]) == _STATIC_EXPECTED_HF_REVISION ||
        error("Hugging Face revision mismatch")
    String(metadata["transformers"]) == "4.57.0" || error("Transformers version mismatch")
    String(metadata["compute_dtype"]) == "float32" || error("oracle must use Float32")
    String(metadata["attention_implementation"]) == "eager" || error("oracle must use eager attention")
    String(metadata["attention_mask_contract"]) == "explicit_all_ones_every_call" ||
        error("oracle attention mask contract changed")
    String(metadata["reference_sha256"]) == _static_file_sha256(reference_path) ||
        error("oracle payload SHA256 mismatch")
    for (name, expected) in _STATIC_EXPECTED_ASSET_SHA256
        String(metadata["asset_sha256"][name]) == expected || error("oracle asset mismatch: $name")
        _static_file_sha256(joinpath(model_dir, name)) == expected || error("model asset mismatch: $name")
    end
    reference = _multi_reference(reference_path, metadata)
    checkpoint = qwen3_vl_checkpoint_spec()
    verified = verify_qwen3_vl_checkpoint(model_dir)
    verified.config.text == checkpoint.text || error("text checkpoint contract mismatch")
    verified.config.vision == checkpoint.vision || error("vision checkpoint contract mismatch")
    processor = load_hf_qwen3_vl_processor_config(
        joinpath(model_dir, "preprocessor_config.json"), checkpoint.vision)
    tokenizer = load_hf_qwen3_vl_tokenizer(model_dir; revision=_STATIC_EXPECTED_MODELSCOPE_REVISION)
    images = [reference["raw_image.$i"] for i in 0:1]
    all(image -> image isa Array{UInt8,3}, images) || error("oracle images must be UInt8 HWC")
    messages = [(role="user", content=Any[
        (type="image", image=images[1]), (type="image", image=images[2]),
        (type="text", text=String(metadata["prompt"])),
    ])]
    processed = LifeAI._qwen3_vl_generation_process_images(images, processor)
    processed.pixel_values == permutedims(reference["pixel_values"]) ||
        error("raw processor pixel values differ from HF")
    processed.grid_thw == permutedims(reference["image_grid_thw"]) ||
        error("raw processor image grids differ from HF")
    rendered = apply_qwen3_vl_chat_template(tokenizer, messages; add_generation_prompt=true)
    rendered == String(metadata["rendered_prompt"]) || error("rendered prompt differs from HF")
    expanded = qwen3_vl_expand_image_placeholders(rendered, processed.grid_thw; spec=processor)
    expanded == String(metadata["expanded_prompt"]) || error("expanded prompt differs from HF")
    ids = encode(tokenizer, expanded; add_special_tokens=false)
    ids .- 1 == vec(reference["input_ids"]) || error("input token IDs differ from HF")
    layout = qwen3_vl_rope_layout(ids, processed.grid_thw; checkpoint)
    layout.position_ids == permutedims(reference["position_ids"], (1, 3, 2)) ||
        error("mRoPE positions differ from HF")
    layout.attention_mask == permutedims(reference["attention_mask"]) || error("attention mask mismatch")
    only(layout.rope_deltas) == Int(metadata["rope_delta"]) || error("mRoPE delta mismatch")
    println("Loading multi-image parameters: ", dtype)
    vision = load_hf_qwen3_vl_vision_parameters(model_dir;
        target_dtype=dtype, to_device=CUDA.cu, spec=checkpoint.vision)
    text = load_hf_qwen3_vl_text_parameters(model_dir;
        target_dtype=dtype, to_device=CUDA.cu, checkpoint)
    features = LifeAI._qwen3_vl_generation_vision_features(vision, processed, 1)
    CUDA.synchronize()
    return (; metadata, reference, checkpoint, processor, tokenizer, images,
        messages, processed, ids, layout, vision, text, features)
end

function _multi_report(result)
    return (;
        generated_ids=copy(result.generated_ids),
        token_sha256=_reuse_hash_arrays((result.generated_ids,)),
        logits_sha256=_reuse_hash_arrays(step.logits for step in result.trace),
        distribution_sha256=result.strategy === :sample ? _reuse_hash_arrays(
            array for step in result.trace for array in
                (step.distribution.token_ids, step.distribution.logits,
                 step.distribution.probabilities)) : nothing,
        kv_sha256=_reuse_kv_hash(result.cache),
        position=result.cache.position, rope_delta=result.cache.rope_delta,
    )
end

function _multi_storage(cache)
    arrays = Any[value for layer in cache.layers for value in (layer.keys, layer.values)]
    cache.decode_workspace === nothing || append!(arrays, cache.decode_workspace)
    return (; arrays, pointers=map(pointer, arrays), pack=cache.decode_pack,
        workspace=cache.decode_workspace)
end

function _multi_assert_storage(cache, saved)
    current = _multi_storage(cache)
    current.pack === saved.pack || error("reuse replaced decode pack")
    current.workspace === saved.workspace || error("reuse replaced decode workspace")
    length(current.arrays) == length(saved.arrays) || error("reuse changed storage count")
    all(a === b for (a, b) in zip(current.arrays, saved.arrays)) || error("reuse replaced arrays")
    current.pointers == saved.pointers || error("reuse changed device pointers")
    return nothing
end

function _multi_hf_gate(prepared, result, n)
    metadata, reference = prepared.metadata, prepared.reference
    expected_ids = Int.(collect(metadata["greedy_token_ids_0_based"]))[1:n]
    metrics = NamedTuple[]
    for step in 1:n
        name = step == 1 ? "prefill.logits" : "decode.$(step - 2).logits"
        expected = vec(reference[name])
        actual = result.trace[step].logits
        measured = _static_stage_metrics(actual, expected)
        margin = Float64(metadata["phases"][step]["top_two"]["margin_f32"])
        println("Float32 HF stage ", name, ": ", measured, "; greedy_margin=", margin)
        result.generated_ids[step] - 1 == expected_ids[step] || error(
            "$name greedy token $(result.generated_ids[step] - 1) differs from HF $(expected_ids[step]): $measured")
        # These are the unchanged Chapter 46 Float32 acceptance thresholds.
        measured.max_abs <= 1e-4 || error("$name exceeds strict Float32 max-abs tolerance 1e-4: $measured")
        measured.mean_abs <= 2e-5 || error("$name exceeds strict Float32 mean-abs tolerance 2e-5: $measured")
        measured.relative_l2 <= 5e-6 || error("$name exceeds strict Float32 relative-L2 tolerance 5e-6: $measured")
        measured.cosine >= 0.99999999 || error("$name exceeds strict Float32 cosine tolerance 0.99999999: $measured")
        2 * measured.max_abs < margin || error("$name does not preserve the HF greedy margin $margin: $measured")
        push!(metrics, merge((; phase=name, hf_greedy_margin=margin), measured))
    end
    return metrics
end

function _multi_run(prepared, n, strategy; raw, cache=:dynamic)
    options = strategy === :sample ? (;
        strategy=:sample, temperature=0.8f0, top_k=20, top_p=0.9f0,
        sample_uniforms=_sampling_uniforms(n), capture_distribution=true,
    ) : (;)
    capacity = length(prepared.ids) + n - 1
    static = cache === :static ? (; static_capacity=capacity) : (;)
    common = (; max_new_tokens=n, stop_token_ids=Int[], capture_logits=true, cache,
        options..., static...)
    result = if raw
        generate_hf_qwen3_vl(prepared.vision, prepared.text, prepared.tokenizer,
            prepared.messages; processor_spec=prepared.processor, common...)
    else
        generate_hf_qwen3_vl_tokens(prepared.text, prepared.ids, prepared.layout;
            vision_features=prepared.features, common...)
    end
    CUDA.synchronize()
    length(result.generated_ids) == n || error("generation stopped early")
    result.cache.position == capacity || error("wrong physical cache position")
    if strategy === :sample
        for (step, uniform) in zip(result.trace, _sampling_uniforms(n))
            _sampling_oracle(step, uniform)
        end
    end
    return result
end

function _multi_dtype(model_dir, reference_dir, dtype, n)
    prepared = _multi_prepare(model_dir, reference_dir, dtype)
    n <= Int(prepared.metadata["greedy_token_count"]) || error("oracle is shorter than requested run")
    reports = Dict{String,Any}()
    hf_metrics = nothing
    for strategy in (:greedy, :sample)
        println("Verifying ", dtype, " ", strategy, " raw/token dynamic/static/reuse")
        dynamic = _multi_run(prepared, n, strategy; raw=true)
        baseline = _multi_report(dynamic)
        if dtype === Float32 && strategy === :greedy
            hf_metrics = _multi_hf_gate(prepared, dynamic, n)
        end
        dynamic = nothing
        GC.gc(true)
        comparisons = Dict{String,Any}("raw_dynamic" => baseline)
        for raw in (true, false)
            prefix = raw ? "raw" : "tokens"
            if !raw
                result = _multi_run(prepared, n, strategy; raw, cache=:dynamic)
                report = _multi_report(result)
                report == baseline || error("$prefix dynamic differs from raw dynamic")
                comparisons[prefix * "_dynamic"] = report
                result = nothing
                GC.gc(true)
            end
            result = _multi_run(prepared, n, strategy; raw, cache=:static)
            report = _multi_report(result)
            report == baseline || error("$prefix static differs from raw dynamic")
            comparisons[prefix * "_static"] = report
            storage = _multi_storage(result.cache)
            reused = _multi_run(prepared, n, strategy; raw, cache=result.cache)
            reused.cache === result.cache || error("reuse replaced caller cache")
            _multi_assert_storage(reused.cache, storage)
            reused_report = _multi_report(reused)
            reused_report == baseline || error("$prefix reuse differs from raw dynamic")
            _reuse_hash_arrays(step.logits for step in result.trace) == report.logits_sha256 ||
                error("reuse modified a retained trace")
            comparisons[prefix * "_reused"] = reused_report
            result = reused = storage = nothing
            GC.gc(true)
        end
        reports[String(strategy)] = comparisons
    end
    return (; compute_dtype=string(dtype), generated_tokens=n, reports, hf_metrics,
        raw_preprocessing_exact=true, same_dtype_logits_distributions_and_kv_exact=true,
        storage_identity_preserved=true, independent_sampling_oracle_passed=true,
        hf_float32_strict_gate_passed=dtype === Float32,
        hf_cross_dtype_parity_claimed=false)
end

function _multi_main(args=ARGS)
    length(args) in (3, 4, 5) || error(
        "usage: verify_qwen3_vl_multi_image_cuda.jl MODEL_DIR REFERENCE_DIR OUTPUT_JSON [TOKENS=8] [float32|bfloat16|both]")
    model_dir, reference_dir, output_json = abspath.(args[1:3])
    n = length(args) >= 4 ? parse(Int, args[4]) : 8
    n > 0 || error("TOKENS must be positive")
    mode = length(args) == 5 ? args[5] : "both"
    mode in ("float32", "bfloat16", "both") || error("invalid dtype mode")
    CUDA.functional() || error("a functional CUDA device is required")
    CUDA.allowscalar(false)
    dtype_reports = Any[]
    for dtype in (mode == "both" ? (Float32, BFloat16) : mode == "float32" ? (Float32,) : (BFloat16,))
        push!(dtype_reports, _multi_dtype(model_dir, reference_dir, dtype, n))
        GC.gc(true)
        CUDA.reclaim()
    end
    root = dirname(@__DIR__)
    sources = (
        "scripts/verify_qwen3_vl_multi_image_cuda.jl",
        "scripts/verify_qwen3_vl_sampling_cuda.jl",
        "scripts/verify_qwen3_vl_generation_reuse_cuda.jl",
        "scripts/verify_qwen3_vl_static_cache_cuda.jl",
        "scripts/export_qwen3_vl_multi_image_reference.py",
        "scripts/export_qwen3_vl_decode_reference.py",
        "scripts/export_qwen3_vl_prefill_reference.py",
        "src/generation/qwen3_vl_generation.jl", "src/generation/sampling.jl",
        "src/generation/hf_text_generation.jl", "src/models/qwen3_vl_vision.jl",
        "src/data/hf_qwen3_vl_processor.jl", "src/models/qwen3_vl_decoder.jl",
        "src/models/qwen3_vl_cache.jl", "src/models/qwen3_vl_attention_scratch.jl",
        "ext/LifeAICUDAExt.jl",
    )
    metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
    report = (;
        schema_version=1, accepted=true, model_id=_STATIC_EXPECTED_MODEL_ID,
        model_dir, reference_dir, device=CUDA.name(CUDA.device()),
        modelscope_revision=_STATIC_EXPECTED_MODELSCOPE_REVISION,
        huggingface_revision=_STATIC_EXPECTED_HF_REVISION,
        model_assets_sha256=metadata["asset_sha256"],
        oracle_compute_device=metadata["compute_device"],
        oracle_metadata_sha256=_static_file_sha256(joinpath(reference_dir, "reference.json")),
        oracle_tensors_sha256=_static_file_sha256(joinpath(reference_dir, "reference.safetensors")),
        source_sha256=Dict(path => _static_file_sha256(joinpath(root, path)) for path in sources),
        uniform_formula="Float64(mod(104729*i+12345,1048576))/1048576; i=1:TOKENS",
        uniforms=_sampling_uniforms(n), dtype_reports,
        performance_improvement_claimed=false,
    )
    mkpath(dirname(output_json))
    open(output_json, "w") do io
        JSON3.pretty(io, report)
        println(io)
    end
    println("Multi-image CUDA gates passed: ", output_json)
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && _multi_main()
