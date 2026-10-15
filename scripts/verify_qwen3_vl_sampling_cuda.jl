#!/usr/bin/env julia

# Frozen-checkpoint sampling and request-reuse gates; no performance claim.
include(joinpath(@__DIR__, "verify_qwen3_vl_generation_reuse_cuda.jl"))

# Exact integer arithmetic followed by a power-of-two division. Independent of
# Julia's RNG implementation and exactly representable in Float64.
_sampling_uniforms(n) = [Float64(mod(104729 * i + 12345, 1048576)) / 1048576 for i in 1:n]

function _sampling_oracle(step, uniform; top_k=20, top_p=0.9f0, temperature=0.8f0)
    logits = Float32.(step.logits)
    all(isfinite, logits) || error("nonfinite captured logits")
    scores = logits ./ Float32(temperature)
    # Sort by score and then token index, independently of production filtering.
    ranked = sort(collect(eachindex(scores)); by=i -> (-Float64(scores[i]), i))
    selected = ranked[1:min(top_k, length(ranked))]
    masses = exp.(Float64.(scores[selected]) .- maximum(Float64.(scores[selected])))
    masses ./= sum(masses)
    nucleus_cdf = cumsum(masses)
    # Endpoints cannot change a nonempty nucleus. Only interior boundaries
    # distinguish two possible retained candidate sets.
    top_p_margin = Float32(top_p) == 1.0f0 || length(masses) <= 1 ? nothing :
        minimum(abs.(nucleus_cdf[1:end-1] .- Float64(top_p)))
    # Conservative diagnostic guard for Float32 exp/normalization/summation;
    # this is an ambiguity detector, not a relaxed parity tolerance.
    top_p_guard = 8 * Float64(eps(Float32)) * length(masses)
    if top_p_margin !== nothing && top_p_margin <= top_p_guard
        error("independent oracle top-p boundary is Float32-sensitive: " *
              "margin=$top_p_margin guard=$top_p_guard top_p=$top_p; " *
              "cannot classify a candidate mismatch as an implementation defect")
    end
    cutoff = Float32(top_p) == 1.0f0 ? length(selected) :
        findfirst(>=(Float64(top_p)), nucleus_cdf)
    selected = selected[1:something(cutoff, length(selected))]
    sort!(selected)
    weights = exp.(Float64.(scores[selected]) .- maximum(Float64.(scores[selected])))
    probabilities = weights ./ sum(weights)
    cdf_margin = setprecision(BigFloat, 256) do
        exact_weights = BigFloat.(probabilities)
        boundaries = cumsum(exact_weights) ./ sum(exact_weights)
        length(boundaries) <= 1 ? nothing :
            Float64(minimum(abs.(boundaries[1:end-1] .- BigFloat(uniform))))
    end
    cdf_guard = 8 * Float64(eps(Float32)) * length(probabilities)
    if cdf_margin !== nothing && cdf_margin <= cdf_guard
        error("independent oracle categorical boundary is Float32-sensitive: " *
              "margin=$cdf_margin guard=$cdf_guard uniform=$uniform; " *
              "cannot classify a token mismatch as an implementation defect")
    end
    d = step.distribution
    d === nothing && error("sampling distribution missing")
    collect(d.token_ids) == selected || error("independent nucleus candidate mismatch")
    collect(d.hf_token_ids) == selected .- 1 || error("HF token index mismatch")
    collect(d.logits) == scores[selected] || error("filtered score mismatch")
    all(isapprox.(Float64.(d.probabilities), probabilities; atol=2e-6, rtol=2e-5)) ||
        error("independent softmax probability mismatch")
    step.sample_uniform == uniform || error("uniform stream mismatch")
    # BigFloat CDF avoids reusing production categorical accumulation logic.
    # Select from independently reconstructed probabilities, not the trace.
    token = setprecision(BigFloat, 256) do
        weights = BigFloat.(probabilities)
        threshold = BigFloat(uniform) * sum(weights)
        cumulative = BigFloat(0)
        for i in eachindex(weights)
            cumulative += weights[i]
            threshold < cumulative && return selected[i]
        end
        error("categorical CDF did not select a token")
    end
    step.token_id == token || error("independent categorical token mismatch")
    step.hf_token_id == token - 1 || error("sampled HF token index mismatch")
    return (; token, top_p_margin, cdf_margin, top_p_guard, cdf_guard)
end

function _sampling_run(prepared, n, capacity, uniforms; cache=:static, top_k=20)
    options = cache === :static ? (; static_capacity=capacity) : NamedTuple()
    result = LifeAI.generate_hf_qwen3_vl_tokens(
        prepared.text_parameters, prepared.input_ids, prepared.rope_layout;
        vision_features=prepared.features, max_new_tokens=n, stop_token_ids=Int[],
        strategy=:sample, temperature=0.8f0, top_k, top_p=0.9f0,
        sample_uniforms=uniforms, capture_logits=true, capture_distribution=true,
        cache, options...,
    )
    CUDA.synchronize()
    length(result.generated_ids) == n || error("sampling stopped early")
    length(result.trace) == n || error("incomplete sampling trace")
    result.cache_mode === :static || error("sampling did not use static cache")
    oracle_checks = map(enumerate(result.trace)) do (i, step)
        check = _sampling_oracle(step, uniforms[i]; top_k)
        check.token == result.generated_ids[i] || error("trace and generated token mismatch")
        check
    end
    minimum_margin(name) = let values = [getproperty(check, name) for check in oracle_checks
                                        if getproperty(check, name) !== nothing]
        isempty(values) ? nothing : minimum(values)
    end
    report = (;
        generated_tokens=n, generated_ids=copy(result.generated_ids),
        oracle_minimum_top_p_boundary_margin=minimum_margin(:top_p_margin),
        oracle_minimum_categorical_boundary_margin=minimum_margin(:cdf_margin),
        oracle_maximum_top_p_boundary_guard=maximum(check.top_p_guard for check in oracle_checks),
        oracle_maximum_categorical_boundary_guard=maximum(check.cdf_guard for check in oracle_checks),
        token_sha256=_reuse_hash_arrays((result.generated_ids,)),
        logits_sha256=_reuse_hash_arrays(step.logits for step in result.trace),
        distribution_sha256=_reuse_hash_arrays(
            a for step in result.trace for a in
                (step.distribution.token_ids, step.distribution.logits, step.distribution.probabilities)),
        kv_sha256=_reuse_kv_hash(result.cache),
        position=result.cache.position, rope_delta=result.cache.rope_delta,
    )
    return result, report
end

function _sampling_main(args=ARGS)
    length(args) == 3 || error(
        "usage: verify_qwen3_vl_sampling_cuda.jl MODEL_DIR REFERENCE_DIR OUTPUT_JSON")
    CUDA.functional() || error("a functional CUDA device is required")
    CUDA.allowscalar(false)
    model_dir, reference_dir, output_json = abspath.(args)
    prepared = _static_prepare_bfloat16(model_dir, reference_dir)
    uniforms = _sampling_uniforms(256)
    capacity = length(prepared.input_ids) + 256
    comparisons = NamedTuple[]
    # A fresh cache for each length; then reuse exactly that cache. Holding the
    # first result live also exercises retained trace ownership after reuse.
    for n in (32, 128, 256)
        fresh, expected = _sampling_run(prepared, n, capacity, uniforms)
        storage = _reuse_storage(fresh.cache)
        reused, actual = _sampling_run(prepared, n, capacity, uniforms; cache=fresh.cache)
        reused.cache === fresh.cache || error("sampling replaced explicit cache")
        _reuse_assert_storage(reused.cache, storage)
        actual == expected || error("fresh/reused tokens, logits, distribution or KV differ")
        _reuse_hash_arrays(step.logits for step in fresh.trace) == expected.logits_sha256 ||
            error("reuse modified retained logits")
        push!(comparisons, (; length=n, fresh=expected, reused=actual))
    end
    n = 8
    greedy = LifeAI.generate_hf_qwen3_vl_tokens(
        prepared.text_parameters, prepared.input_ids, prepared.rope_layout;
        vision_features=prepared.features, max_new_tokens=n, stop_token_ids=Int[],
        capture_logits=true, cache=:static, static_capacity=capacity,
    )
    sampled, _ = _sampling_run(prepared, n, capacity, uniforms; top_k=1)
    length(greedy.generated_ids) == n || error("greedy stopped early")
    length(greedy.trace) == n || error("incomplete greedy trace")
    greedy.generated_ids == sampled.generated_ids || error("top-k=1 differs from default greedy")
    all(a.logits == b.logits for (a, b) in zip(greedy.trace, sampled.trace)) ||
        error("top-k=1 logits differ from default greedy")
    source_files = (
        "src/generation/qwen3_vl_generation.jl", "src/generation/sampling.jl",
        "src/generation/hf_text_generation.jl", "src/models/qwen3_vl_cache.jl",
        "ext/LifeAICUDAExt.jl", "scripts/verify_qwen3_vl_sampling_cuda.jl",
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
        device=CUDA.name(CUDA.device()), compute_dtype="BFloat16", capacity,
        uniform_formula="Float64(mod(104729*i+12345,1048576))/1048576; i=1:256",
        uniform_sha256=_reuse_hash_arrays((uniforms,)),
        uniform_encoding="native-endian IEEE754 Float64", uniforms,
        temperature=0.8f0, top_k=20, top_p=0.9f0, comparisons,
        independent_sampling_oracle_passed=true, exact_request_parity_passed=true,
        same_cache_and_storage_passed=true, greedy_top_k_one_passed=true,
        hf_cross_dtype_parity_claimed=false, performance_improvement_claimed=false,
    )
    mkpath(dirname(output_json))
    open(output_json, "w") do io
        JSON3.pretty(io, report)
        println(io)
    end
    println("Sampling and request reuse gates passed: ", output_json)
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && _sampling_main()
