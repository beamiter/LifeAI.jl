#!/usr/bin/env julia

using BFloat16s: BFloat16
using CUDA
using JSON3
using LifeAI
using LinearAlgebra: dot, norm
using SHA: sha256
using Statistics: mean

length(ARGS) in (2, 3) || error(
    "usage: julia --project=. scripts/verify_qwen3_vl_vision_cuda.jl " *
    "MODEL_DIR REFERENCE_DIR [cuda|cpu]",
)
model_dir, reference_dir = ARGS
backend = length(ARGS) == 3 ? ARGS[3] : "cuda"
backend in ("cuda", "cpu") || error("backend must be cuda or cpu")
use_cuda = backend == "cuda"
use_cuda && !CUDA.functional() && error("CUDA.jl is not functional on this machine")
to_device = use_cuda ? CUDA.cu : identity
synchronize() = use_cuda ? CUDA.synchronize() : nothing

const EXPECTED_MODELSCOPE_REVISION =
    "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53"
const EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
const EXPECTED_HF_REVISION =
    "78448d793a7eb2f7a987a1da76d464384aa1becd"
const EXPECTED_TRANSFORMERS_VERSION = "4.57.0"
const EXPECTED_TORCH_VERSION = "2.7.1+cpu"
const EXPECTED_CHECKPOINT_SHA256 =
    "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
const EXPECTED_CONFIG_SHA256 =
    "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
const EXPECTED_PREPROCESSOR_CONFIG_SHA256 =
    "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"

metadata_path = joinpath(reference_dir, "reference.json")
reference_path = joinpath(reference_dir, "reference.safetensors")
metadata = JSON3.read(read(metadata_path, String))
String(metadata["model_id"]) == EXPECTED_MODEL_ID ||
    error("reference model id is not frozen")
String(metadata["modelscope_revision"]) == EXPECTED_MODELSCOPE_REVISION ||
    error("reference ModelScope revision is not frozen")
String(metadata["huggingface_revision"]) == EXPECTED_HF_REVISION ||
    error("reference Hugging Face revision is not frozen")
String(metadata["transformers_version"]) == EXPECTED_TRANSFORMERS_VERSION ||
    error("reference Transformers version is not frozen")
String(metadata["torch_version"]) == EXPECTED_TORCH_VERSION ||
    error("reference Torch version is not frozen")
String(metadata["checkpoint_sha256"]) == EXPECTED_CHECKPOINT_SHA256 ||
    error("reference checkpoint SHA-256 is not frozen")
String(metadata["config_sha256"]) == EXPECTED_CONFIG_SHA256 ||
    error("reference config SHA-256 is not frozen")
String(metadata["preprocessor_config_sha256"]) ==
    EXPECTED_PREPROCESSOR_CONFIG_SHA256 ||
    error("reference preprocessor config SHA-256 is not frozen")
String(metadata["attention_implementation"]) == "eager" ||
    error("reference attention implementation must be eager")
compute_dtype_name = String(metadata["compute_dtype"])
compute_dtype = if compute_dtype_name == "float32"
    Float32
elseif compute_dtype_name == "bfloat16"
    BFloat16
else
    error("reference compute dtype is unsupported: $compute_dtype_name")
end
reference_sha = open(reference_path, "r") do io
    bytes2hex(sha256(io))
end
expected_reference_sha = LifeAI._qwen3_vl_vision_reference_sha256(
    compute_dtype_name,
)
reference_sha == expected_reference_sha ||
    error("reference safetensors is not the frozen oracle")
reference_sha == String(metadata["reference_sha256"]) ||
    error("reference safetensors SHA-256 mismatch")

actual_modelscope_revision = readchomp(
    `git -C $model_dir rev-parse HEAD`,
)
actual_modelscope_revision == EXPECTED_MODELSCOPE_REVISION || error(
    "checkpoint git revision is not the frozen ModelScope revision",
)
checkpoint_git_status = readchomp(
    `git -C $model_dir status --porcelain=v1 --untracked-files=all`,
)
isempty(checkpoint_git_status) || error(
    "checkpoint git tree must be clean:\n$checkpoint_git_status",
)
checkpoint = qwen3_vl_checkpoint_spec()
checkpoint_report = verify_qwen3_vl_checkpoint(model_dir)
config = checkpoint_report.config
config.vision == checkpoint.vision || error("checkpoint vision config mismatch")
processor = load_hf_qwen3_vl_processor_config(
    joinpath(model_dir, "preprocessor_config.json"),
    checkpoint.vision,
)
reference = open_safetensors_reader(reference_path)

normalized = read_safetensors_tensor(
    reference,
    "normalized_chw_f32";
    target_dtype=Float32,
)
expected_pixels_f32 = permutedims(read_safetensors_tensor(
    reference,
    "pixel_values_f32";
    target_dtype=Float32,
))
patchified = qwen3_vl_patchify(normalized; spec=processor)
size(patchified) == size(expected_pixels_f32) || error(
    "official processor and Julia patchify shapes differ",
)
patchify_difference = maximum(abs.(patchified .- expected_pixels_f32))
patchify_difference == 0.0f0 || error(
    "official processor and Julia patchify differ by $patchify_difference",
)

grid_rows = [Int.(collect(row)) for row in metadata["grid_thw"]]
grid_thw = reduce(hcat, grid_rows)
size(grid_thw, 1) == 3 || error("reference grid must contain t,h,w rows")
expected_pixels = permutedims(read_safetensors_tensor(
    reference,
    "pixel_values_compute";
    target_dtype=compute_dtype,
))

use_cuda && CUDA.reclaim()
load_timing = @timed begin
    loaded = load_hf_qwen3_vl_vision_parameters(
        model_dir;
        target_dtype=compute_dtype,
        to_device,
        spec=checkpoint.vision,
    )
    synchronize()
    loaded
end
parameters = load_timing.value
parameter_bytes = LifeAI._qwen3_vl_vision_parameter_count(checkpoint.vision) *
    sizeof(compute_dtype)

input = Qwen3VLVisionInput(
    to_device(expected_pixels),
    grid_thw;
    spec=checkpoint.vision,
)
forward_timing = @timed begin
    output = hf_qwen3_vl_vision_forward(
        parameters,
        input;
        capture_layers=(0, 5, 11, 17, 23),
    )
    synchronize()
    output
end
features = forward_timing.value

warm_forward_timing = @timed begin
    output = hf_qwen3_vl_vision_forward(
        parameters,
        input;
        capture_layers=(0, 5, 11, 17, 23),
    )
    synchronize()
    output
end
warm_features = warm_forward_timing.value
Array(warm_features.visual_embeddings) ==
    Array(features.visual_embeddings) || error(
    "warm Qwen3-VL vision output differs from the cold output",
)

patch_embedding = LifeAI._qwen3_vl_linear(
    parameters.patch_weight,
    parameters.patch_bias,
    input.pixel_values,
)
position_embedding = LifeAI._qwen3_vl_interpolated_positions(
    checkpoint.vision,
    parameters.pos_embedding,
    grid_thw,
)
position_added = LifeAI._qwen3_vl_cast_like(
    patch_embedding,
    LifeAI._qwen3_vl_f32(patch_embedding) .+
        LifeAI._qwen3_vl_f32(position_embedding),
)
synchronize()

function hf_reference(name)
    value = read_safetensors_tensor(reference, name; target_dtype=Float32)
    ndims(value) == 2 || error("reference tensor $name must be a matrix")
    return permutedims(value)
end

function stage_metrics(actual, expected)
    values = Float32.(Array(actual))
    size(values) == size(expected) || error(
        "stage shape mismatch: $(size(values)) != $(size(expected))",
    )
    differences = abs.(values .- expected)
    denominator = norm(expected)
    relative_l2 = denominator == 0 ? norm(values) : norm(values .- expected) / denominator
    cosine_denominator = norm(values) * denominator
    cosine = cosine_denominator == 0 ? 1.0 : dot(vec(values), vec(expected)) / cosine_denominator
    return (;
        max_abs=maximum(differences),
        mean_abs=mean(differences),
        relative_l2,
        cosine,
    )
end

stages = Pair{String,Any}[
    "patch_embed" => patch_embedding,
    "position_added" => position_added,
    "block.0" => features.checkpoints[0],
    "block.5" => features.checkpoints[5],
    "block.11" => features.checkpoints[11],
    "block.17" => features.checkpoints[17],
    "block.23" => features.checkpoints[23],
    "deepstack.0" => features.deepstack[1],
    "deepstack.1" => features.deepstack[2],
    "deepstack.2" => features.deepstack[3],
    "visual_embeddings" => features.visual_embeddings,
]

println("modelscope_revision\t", metadata["modelscope_revision"])
println("huggingface_revision\t", metadata["huggingface_revision"])
println("transformers_version\t", metadata["transformers_version"])
println("torch_version\t", metadata["torch_version"])
println("compute_dtype\t", compute_dtype_name)
println(
    "numerical_contract\t",
    compute_dtype === Float32 ?
        "strict_float32_parity" : "cross_backend_bfloat16_boundary",
)
println("reference_sha256\t", reference_sha)
println("backend\t", backend)
println("device_name\t", use_cuda ? CUDA.name(CUDA.device()) : "CPU")
println("patchify_max_abs\t", patchify_difference)
println("vision_parameter_bytes\t", parameter_bytes)
println("vision_load_seconds\t", load_timing.time)
println("vision_cold_forward_seconds\t", forward_timing.time)
println("vision_warm_forward_seconds\t", warm_forward_timing.time)
println("stage\tmax_abs\tmean_abs\trelative_l2\tcosine")

bf16_limits = Dict(
    "patch_embed" => (0.015625f0, 1.0f-5, 2.0e-4, 0.99999),
    "position_added" => (0.015625f0, 1.0f-5, 2.0e-4, 0.99999),
    "block.0" => (0.5f0, 0.005f0, 0.01, 0.9999),
    "block.5" => (1.0f0, 0.01f0, 0.025, 0.9997),
    "block.11" => (40.0f0, 0.015f0, 0.11, 0.994),
    "block.17" => (64.0f0, 0.05f0, 0.15, 0.99),
    "block.23" => (1024.0f0, 0.6f0, 0.12, 0.993),
    "deepstack.0" => (0.125f0, 0.005f0, 0.015, 0.9999),
    "deepstack.1" => (1.6f0, 0.02f0, 0.09, 0.996),
    "deepstack.2" => (4.0f0, 0.03f0, 0.08, 0.996),
    "visual_embeddings" => (2.0f0, 0.04f0, 0.09, 0.996),
)
float32_limits = Dict(
    "patch_embed" => (2.0f-5, 1.0f-6, 2.0e-6, 0.99999),
    "position_added" => (2.0f-5, 1.0f-6, 2.0e-6, 0.99999),
    "block.0" => (3.0f-5, 2.0f-6, 2.0e-6, 0.99999),
    "block.5" => (1.0f-4, 2.0f-6, 4.0e-6, 0.99999),
    "block.11" => (2.0f-3, 3.0f-6, 6.0e-6, 0.99999),
    "block.17" => (2.0f-2, 1.0f-5, 3.0e-5, 0.99998),
    "block.23" => (2.5f-1, 1.0f-4, 2.0e-5, 0.99998),
    "deepstack.0" => (2.0f-5, 2.0f-6, 3.0e-6, 0.99999),
    "deepstack.1" => (2.0f-4, 3.0f-6, 1.0e-5, 0.99999),
    "deepstack.2" => (1.0f-3, 1.0f-5, 2.0e-5, 0.99998),
    "visual_embeddings" => (1.0f-3, 1.0f-5, 2.0e-5, 0.99998),
)
passed = Ref(true)
for (name, actual) in stages
    metrics = stage_metrics(actual, hf_reference(name))
    println(join((
        name,
        metrics.max_abs,
        metrics.mean_abs,
        metrics.relative_l2,
        metrics.cosine,
    ), '\t'))
    limits = compute_dtype === Float32 ? float32_limits[name] : bf16_limits[name]
    max_abs, mean_abs, relative_l2, cosine = limits
    passed[] &= metrics.max_abs <= max_abs
    passed[] &= metrics.mean_abs <= mean_abs
    passed[] &= metrics.relative_l2 <= relative_l2
    passed[] &= metrics.cosine >= cosine
end
features.patch_hidden_state === features.checkpoints[23] || error(
    "patch_hidden_state must be the captured final block",
)
passed[] || error(
    "Qwen3-VL vision numerical contract exceeded the frozen tolerances",
)
println(
    compute_dtype === Float32 ?
        "parity_passed\ttrue" : "cross_backend_boundary_passed\ttrue",
)
