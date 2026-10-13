#!/usr/bin/env julia

# Real-checkpoint sequential request reuse. The imported verifier is main-guarded.
include(joinpath(@__DIR__, "verify_qwen3_vl_static_cache_cuda.jl"))

function _reuse_hash_arrays(arrays)
    buffer = IOBuffer()
    for array in arrays
        write(buffer, reinterpret(UInt8, vec(Array(array))))
    end
    return bytes2hex(sha256(take!(buffer)))
end

function _reuse_kv_hash(cache)
    return _reuse_hash_arrays(
        view(storage, :, :, 1:cache.position, :)
        for layer in cache.layers for storage in (layer.keys, layer.values)
    )
end

function _reuse_storage(cache)
    arrays = Any[storage for layer in cache.layers for storage in (layer.keys, layer.values)]
    append!(arrays, cache.decode_workspace)
    return (;
        ready=cache.decode_pack, workspace=cache.decode_workspace,
        arrays, pointers=map(pointer, arrays), shapes=map(size, arrays),
    )
end

function _reuse_assert_storage(cache, saved)
    current = _reuse_storage(cache)
    current.ready === saved.ready || error("reuse replaced Ready")
    current.workspace === saved.workspace || error("reuse replaced workspace parents")
    all(a === b for (a, b) in zip(current.arrays, saved.arrays)) ||
        error("reuse replaced K/V or parent arrays")
    current.pointers == saved.pointers || error("reuse changed storage pointers")
    current.shapes == saved.shapes || error("reuse changed storage shapes")
    return true
end

function _reuse_assert_disjoint(left, right)
    for a in left.arrays, b in right.arrays
        a_start, b_start = UInt(pointer(a)), UInt(pointer(b))
        a_end = a_start + UInt(length(a) * sizeof(eltype(a)))
        b_end = b_start + UInt(length(b) * sizeof(eltype(b)))
        max(a_start, b_start) >= min(a_end, b_end) ||
            error("independent caches have overlapping device storage")
    end
    return true
end

function _reuse_run(parameters, ids, layout, count; features=nothing, cache=:static)
    options = cache === :static ? (; static_capacity=120) : NamedTuple()
    result = LifeAI.generate_hf_qwen3_vl_tokens(
        parameters, ids, layout;
        vision_features=features, max_new_tokens=count, stop_token_ids=Int[],
        capture_logits=true, cache, options...,
    )
    CUDA.synchronize()
    length(result.generated_ids) == count || error("generation stopped early")
    result.cache_mode === :static || error("wrong reported cache mode")
    report = (;
        generated_ids=copy(result.generated_ids),
        token_sha256=_reuse_hash_arrays((result.generated_ids,)),
        trace_sha256=_reuse_hash_arrays(step.logits for step in result.trace),
        kv_sha256=_reuse_kv_hash(result.cache),
        position=result.cache.position, rope_delta=result.cache.rope_delta,
    )
    return result, report
end

function _reuse_equal(actual, expected)
    actual == expected || error("reused request differs from fresh tokens, trace, or K/V")
    return true
end

function _reuse_main(args=ARGS)
    length(args) == 3 || error(
        "usage: verify_qwen3_vl_generation_reuse_cuda.jl MODEL_DIR REFERENCE_DIR OUTPUT_JSON",
    )
    CUDA.functional() || error("a functional CUDA device is required")
    CUDA.allowscalar(false)
    model_dir, reference_dir, output_json = abspath.(args)
    prepared = _static_prepare_bfloat16(model_dir, reference_dir)
    parameters = prepared.text_parameters
    first, first_report = _reuse_run(
        parameters, prepared.input_ids, prepared.rope_layout, 32;
        features=prepared.features,
    )
    original = first.cache
    storage = _reuse_storage(original)
    second, second_report = _reuse_run(
        parameters, prepared.input_ids, prepared.rope_layout, 32;
        features=prepared.features, cache=original,
    )
    second.cache === original || error("image reuse replaced cache")
    _reuse_equal(second_report, first_report)
    _reuse_assert_storage(original, storage)

    text_ids = [42, 43, 44, 45]
    text_layout = LifeAI.Qwen3VLRopeLayout(
        reshape(repeat(collect(0:3); inner=3), 3, 4, 1),
        zeros(Int, 1, 1), falses(4, 1), trues(4, 1),
    )
    text_reused, text_report = _reuse_run(
        parameters, text_ids, text_layout, 8; cache=original,
    )
    text_reused.cache === original || error("text reuse replaced cache")
    _reuse_assert_storage(original, storage)
    text_fresh, text_fresh_report = _reuse_run(parameters, text_ids, text_layout, 8)
    _reuse_equal(text_report, text_fresh_report)
    _reuse_assert_disjoint(storage, _reuse_storage(text_fresh.cache))
    # Keep the independent cache live while returning to the original request.
    final, final_report = _reuse_run(
        parameters, prepared.input_ids, prepared.rope_layout, 32;
        features=prepared.features, cache=original,
    )
    final.cache === original || error("final image reuse replaced cache")
    _reuse_equal(final_report, first_report)
    _reuse_assert_storage(original, storage)
    _reuse_assert_disjoint(storage, _reuse_storage(text_fresh.cache))
    source_files = (
        "src/generation/qwen3_vl_generation.jl", "src/models/qwen3_vl_cache.jl",
        "src/models/qwen3_vl_attention_scratch.jl", "ext/LifeAICUDAExt.jl",
        "scripts/verify_qwen3_vl_generation_reuse_cuda.jl",
        "scripts/verify_qwen3_vl_static_cache_cuda.jl",
    )
    root = dirname(@__DIR__)
    report = (;
        model_id=prepared.metadata["model_id"], model_dir, reference_dir,
        modelscope_revision=prepared.metadata["modelscope_revision"],
        hf_revision=_STATIC_EXPECTED_HF_REVISION,
        model_assets_sha256=prepared.metadata["asset_sha256"],
        reference_metadata_sha256=_static_file_sha256(joinpath(reference_dir, "reference.json")),
        reference_tensors_sha256=_static_file_sha256(joinpath(reference_dir, "reference.safetensors")),
        source_sha256=Dict(path => _static_file_sha256(joinpath(root, path)) for path in source_files),
        device=CUDA.name(CUDA.device()), compute_dtype="BFloat16", capacity=120,
        fresh_image=first_report, reused_image=second_report,
        reused_text=text_report, fresh_text=text_fresh_report, final_image=final_report,
        same_cache_passed=true, ready_identity_passed=true,
        workspace_identity_passed=true, kv_identity_passed=true,
        independent_storage_passed=true, exact_request_parity_passed=true,
        hf_tensor_parity_claimed=false,
    )
    mkpath(dirname(output_json))
    open(output_json, "w") do io
        JSON3.pretty(io, report)
        println(io)
    end
    println("Sequential image/text/image reuse passed: ", output_json)
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && _reuse_main()
