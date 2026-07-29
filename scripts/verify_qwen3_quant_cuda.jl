#!/usr/bin/env julia

using LuxCUDA
using CUDA
using BFloat16s: BFloat16
using JSON3
using SHA: sha256
using Statistics: mean
using LifeAI
using LifeAI: hf_token_ids

length(ARGS) in (3, 4, 5) || error(
    "usage: julia --project=. scripts/verify_qwen3_quant_cuda.jl " *
    "MODEL_DIR REFERENCE_DIR SCHEME [PLAN_JSON] [CALIBRATION_TOKENS_JSON]",
)
model_dir, reference_dir, scheme_name = ARGS
scheme = Symbol(scheme_name)
CUDA.functional() || error("CUDA.jl is not functional on this machine")

_json_get(object, key, default) = haskey(object, key) ? object[key] : default

function _quantization_spec_from_json(object)
    scheme = Symbol(String(object["scheme"]))
    group = Int(_json_get(object, "group", 128))
    calibration = Symbol(String(_json_get(object, "calibration", "maxabs")))
    if haskey(object, "clip_ratios")
        clip_ratios = Tuple(Float32.(collect(object["clip_ratios"])))
        return LinearQuantizationSpec(
            scheme;
            group,
            calibration,
            clip_ratios,
        )
    end
    return LinearQuantizationSpec(scheme; group, calibration)
end

function _quantization_plan_from_json(path)
    object = JSON3.read(read(path, String))
    projections = Dict{Symbol,LinearQuantizationSpec}()
    for (projection, spec) in pairs(_json_get(
        object,
        "projection_overrides",
        Dict{String,Any}(),
    ))
        projections[Symbol(projection)] = _quantization_spec_from_json(spec)
    end
    layers = Dict{Tuple{Int,Symbol},LinearQuantizationSpec}()
    for entry in _json_get(object, "layer_overrides", Any[])
        target = (Int(entry["layer"]), Symbol(String(entry["projection"])))
        layers[target] = _quantization_spec_from_json(entry["spec"])
    end
    return QuantizationPlan(
        default=_quantization_spec_from_json(object["default"]),
        projection_overrides=projections,
        layer_overrides=layers,
    )
end

plan = length(ARGS) == 4 ?
    _quantization_plan_from_json(ARGS[4]) :
    length(ARGS) == 5 ?
        _quantization_plan_from_json(ARGS[4]) :
        QuantizationPlan(default=LinearQuantizationSpec(scheme))

calibration_tokens_sha256 = "none"
calibration_timing = nothing
activation_calibration = nothing
if length(ARGS) == 5
    calibration_path = ARGS[5]
    calibration_bytes = read(calibration_path)
    calibration_tokens_sha256 = bytes2hex(sha256(calibration_bytes))
    calibration_object = JSON3.read(String(calibration_bytes))
    sequences = collect(calibration_object["token_ids_0_based"])
    isempty(sequences) && error("calibration token fixture has no sequences")
    sequence_length = length(first(sequences))
    all(length(sequence) == sequence_length for sequence in sequences) ||
        error("calibration token sequences must have equal length")
    calibration_tokens = hcat([
        hf_token_ids(
            Int.(collect(sequence));
            vocab_size=151936,
        )
        for sequence in sequences
    ]...)
    calibration_timing = @timed calibrate_hf_qwen3_activations(
        model_dir,
        calibration_tokens;
        max_seq_len=max(64, sequence_length),
        source="$(abspath(calibration_path))#$calibration_tokens_sha256",
        accelerated=true,
        to_device=CUDA.cu,
        to_host=Array,
    )
    activation_calibration = calibration_timing.value.calibration
    GC.gc()
    CUDA.reclaim()
end

metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
String(metadata["compute_dtype"]) == "bfloat16" || error(
    "reference was not exported with --compute-dtype bfloat16",
)
reference = load_safetensors(joinpath(reference_dir, "reference.safetensors"))

tokens = reshape(
    hf_token_ids(Int.(collect(metadata["token_ids_0_based"])); vocab_size=151936),
    :,
    1,
)
decode_token = hf_token_ids(
    [Int(metadata["decode_token_id_0_based"])];
    vocab_size=151936,
)
expected_greedy = Int.(collect(metadata["greedy_token_ids_0_based"])) .+ 1

GC.gc()
load_timing = @timed load_hf_qwen3_quantized(
    model_dir;
    max_seq_len=64,
    plan,
    activation_calibration,
)
loaded = load_timing.value
estimated_tree_bytes = estimate_qwen3_quantized_bytes(loaded.model, plan)
host_tree_bytes = quantized_parameter_bytes(loaded.parameters)
free_before = CUDA.free_memory()
ps_gpu = CUDA.cu(loaded.parameters)
CUDA.synchronize()
tree_bytes = free_before - CUDA.free_memory()

cold_timing = @timed begin
    result = hf_qwen3_bf16_accel_forward(
        loaded.model,
        ps_gpu,
        tokens;
        decode_token,
        greedy_steps=length(expected_greedy),
    )
    CUDA.synchronize()
    result
end
result = cold_timing.value
warm_timing = @timed begin
    warm = hf_qwen3_bf16_accel_forward(
        loaded.model,
        ps_gpu,
        tokens;
        greedy_steps=length(expected_greedy),
    )
    CUDA.synchronize()
    warm
end
warm_timing.value.greedy_tokens == result.greedy_tokens ||
    error("warm greedy diverged from cold run")

hf_layout(array) = permutedims(array, (3, 2, 1))
to_host(x) = Float32.(Array(x))
reference_logits = Float32.(hf_layout(reference["logits"]))
reference_decode = Float32.(hf_layout(reference["decode_logits"]))
logits = to_host(result.logits)
decode_logits = to_host(result.decode_logits)

matches = sum(result.greedy_tokens .== expected_greedy)
first_divergence = findfirst(result.greedy_tokens .!= expected_greedy)

println("model_revision\t", metadata["revision"])
println(
    "dense_variant\t",
    loaded.variant === nothing ? "custom" : String(loaded.variant.variant),
)
println("scheme\t", scheme)
println("default_scheme\t", plan.default.scheme)
println("default_calibration\t", plan.default.calibration)
println("projection_overrides\t", join(
    ["$(target):$(spec.scheme)" for (target, spec) in sort!(
        collect(plan.projection_overrides);
        by=first,
    )],
    ",",
))
println("layer_overrides\t", join(
    ["$(target[1]):$(target[2]):$(spec.scheme)" for (target, spec) in sort!(
        collect(plan.layer_overrides);
        by=first,
    )],
    ",",
))
println(
    "calibration_seconds\t",
    calibration_timing === nothing ? "none" : string(calibration_timing.time),
)
println(
    "calibration_token_count\t",
    activation_calibration === nothing ?
        0 : activation_calibration.token_count,
)
println("calibration_tokens_sha256\t", calibration_tokens_sha256)
println("gpu_name\t", CUDA.name(CUDA.device()))
println("load_seconds\t", load_timing.time)
println("host_maxrss_bytes\t", Sys.maxrss())
println("estimated_tree_bytes\t", estimated_tree_bytes)
println("host_tree_bytes\t", host_tree_bytes)
println("gpu_tree_bytes\t", tree_bytes)
println("vram_used_bytes\t", CUDA.total_memory() - CUDA.free_memory())
println("cold_seconds\t", cold_timing.time)
println("warm_greedy_seconds\t", warm_timing.time)
println(
    "warm_tokens_per_second\t",
    length(expected_greedy) / warm_timing.time,
)
println("logits_max_abs\t", maximum(abs.(logits .- reference_logits)))
println("logits_mean_abs\t", mean(abs.(logits .- reference_logits)))
println(
    "logits_argmax_equal\t",
    argmax(vec(logits)) == argmax(vec(reference_logits)),
)
println("decode_max_abs\t", maximum(abs.(decode_logits .- reference_decode)))
println(
    "decode_argmax_equal\t",
    argmax(vec(decode_logits)) == argmax(vec(reference_decode)),
)
println(
    "greedy_match\t",
    result.greedy_tokens == expected_greedy,
    "\t",
    join(result.greedy_tokens .- 1, ','),
)
println("greedy_agreement\t", matches, "/", length(expected_greedy))
println(
    "greedy_first_divergence\t",
    first_divergence === nothing ? "none" : string(first_divergence),
)
