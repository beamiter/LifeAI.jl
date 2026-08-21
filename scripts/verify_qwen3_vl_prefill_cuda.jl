#!/usr/bin/env julia

using BFloat16s: BFloat16
using CUDA
using JSON3
using LifeAI
using LinearAlgebra: dot, norm
using SHA: sha256
using Statistics: mean

const EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
const EXPECTED_MODELSCOPE_REVISION =
    "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53"
const EXPECTED_HF_REVISION =
    "78448d793a7eb2f7a987a1da76d464384aa1becd"
const EXPECTED_TRANSFORMERS_VERSION = "4.57.0"
const EXPECTED_TORCH_VERSION = "2.7.1+cpu"
const EXPECTED_TORCHVISION_VERSION = "0.22.1+cu126"
const EXPECTED_CHAT_TEMPLATE_SHA256 =
    "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
const EXPECTED_IMAGE_SHA256 =
    "ec143579b36852cf212bbb368798479d193a8c6d039942fca89a49fc820dff3f"
const EXPECTED_RENDERED_PROMPT_SHA256 =
    "7a50d10ccb53359de53e3e9b032c39b15fd3abbfed51ec844117c3b93da07271"
const EXPECTED_EXPANDED_PROMPT_SHA256 =
    "55c60c75dd9158d487d87c8fd8670f72c6ff5bf78dbf20fd86989cea5f8bf828"
const EXPECTED_REFERENCE_SHA256 = Dict(
    "float32" =>
        "d7d3b58cea35cf90806bdd14ade7e453e1b486355b190d094ec95f852f6b60f5",
    "bfloat16" =>
        "711749d9cb0d2c33b34c6fc87a4f9dd06bbf7cc52b589daf01f0210bd58cb5ae",
)
const EXPECTED_METADATA_SHA256 = Dict(
    "float32" =>
        "1141ef4c503800607d60d8fa23eb795b00d94a7ff879f97ac230abf05681362c",
    "bfloat16" =>
        "7a941b837e7c6df13490386dc42160b9ac6f7d5befcc6bba148cc10e1e81afb5",
)
const EXPECTED_ASSET_SHA256 = Dict(
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
const EXPECTED_PROMPT = "Describe."
const VISION_CAPTURE_LAYERS = (0, 5, 11, 17, 23)

function model_dir_from_environment()
    for name in ("LIFEAI_QWEN3_VL_MODEL_DIR", "QWEN3_VL_MODEL_DIR")
        value = get(ENV, name, "")
        isempty(value) || return value
    end
    error(
        "checkpoint directory is missing; pass MODEL_DIR or set " *
        "LIFEAI_QWEN3_VL_MODEL_DIR",
    )
end

function parse_arguments(args)
    usage = "usage: julia --project=. scripts/verify_qwen3_vl_prefill_cuda.jl " *
        "[MODEL_DIR] REFERENCE_DIR [cuda|cpu]"
    length(args) in 1:3 || error(usage)
    backend = "cuda"
    if length(args) == 1
        model_dir = model_dir_from_environment()
        reference_dir = args[1]
    elseif length(args) == 2 && args[2] in ("cuda", "cpu")
        model_dir = model_dir_from_environment()
        reference_dir = args[1]
        backend = args[2]
    else
        model_dir = args[1]
        reference_dir = args[2]
        length(args) == 3 && (backend = args[3])
    end
    backend in ("cuda", "cpu") || error("backend must be cuda or cpu")
    return abspath(model_dir), abspath(reference_dir), backend
end

model_dir, reference_dir, backend = parse_arguments(ARGS)
use_cuda = backend == "cuda"
use_cuda && !CUDA.functional() && error("CUDA.jl is not functional on this machine")
use_cuda && CUDA.allowscalar(false)
to_device = use_cuda ? CUDA.cu : identity
synchronize() = use_cuda ? CUDA.synchronize() : nothing

metadata_path = joinpath(reference_dir, "reference.json")
reference_path = joinpath(reference_dir, "reference.safetensors")
isfile(metadata_path) || error("reference metadata does not exist: $metadata_path")
isfile(reference_path) || error("reference tensors do not exist: $reference_path")
metadata_sha = open(metadata_path, "r") do io
    bytes2hex(sha256(io))
end
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
String(metadata["torchvision_version"]) == EXPECTED_TORCHVISION_VERSION ||
    error("reference torchvision version is not frozen")
String(metadata["chat_template_sha256"]) == EXPECTED_CHAT_TEMPLATE_SHA256 ||
    error("reference chat-template hash is not frozen")
String(metadata["attention_implementation"]) == "eager" ||
    error("reference attention implementation must be eager")
String(metadata["attention_mask_contract"]) == "explicit_all_ones" ||
    error("reference must explicitly disable packed-sequence auto-detection")
String(metadata["capture_contract"]) ==
    "detach_clone_before_deepstack_inplace_mutation" ||
    error("reference decoder captures do not declare the safe clone contract")
String(metadata["processor_class"]) == "Qwen3VLProcessor" ||
    error("reference processor class is not frozen")
String(metadata["image_processor_class"]) == "Qwen2VLImageProcessorFast" ||
    error("reference did not use the official fast raw-image processor")
String(metadata["image_processor_device"]) == "cpu" ||
    error("the frozen torchvision raw-image oracle must execute on CPU")
String(metadata["compute_device"]) == "cpu" ||
    error("the frozen Transformers prefill oracle must execute on CPU")
String(metadata["prompt"]) == EXPECTED_PROMPT ||
    error("reference prompt is not frozen")
Bool(metadata["add_generation_prompt"]) ||
    error("reference must include the assistant generation prompt")
!Bool(metadata["add_vision_id"]) ||
    error("reference must not add a human-readable vision id")
String(metadata["image_sha256"]) == EXPECTED_IMAGE_SHA256 ||
    error("reference deterministic image is not frozen")

asset_hashes = metadata["asset_sha256"]
for (filename, expected_sha256) in EXPECTED_ASSET_SHA256
    String(asset_hashes[filename]) == expected_sha256 ||
        error("reference asset hash is not frozen for $filename")
end
String(metadata["checkpoint_sha256"]) == EXPECTED_ASSET_SHA256["model.safetensors"] ||
    error("reference checkpoint SHA-256 is not frozen")
String(metadata["config_sha256"]) == EXPECTED_ASSET_SHA256["config.json"] ||
    error("reference config SHA-256 is not frozen")
String(metadata["preprocessor_config_sha256"]) ==
    EXPECTED_ASSET_SHA256["preprocessor_config.json"] ||
    error("reference preprocessor SHA-256 is not frozen")

compute_dtype_name = String(metadata["compute_dtype"])
compute_dtype = if compute_dtype_name == "float32"
    Float32
elseif compute_dtype_name == "bfloat16"
    BFloat16
else
    error("reference compute dtype is unsupported: $compute_dtype_name")
end
expected_reference_sha = EXPECTED_REFERENCE_SHA256[compute_dtype_name]
metadata_sha == EXPECTED_METADATA_SHA256[compute_dtype_name] ||
    error("reference metadata SHA-256 is not frozen")
reference_sha = open(reference_path, "r") do io
    bytes2hex(sha256(io))
end
reference_sha == expected_reference_sha ||
    error("reference safetensors SHA-256 is not frozen")
String(metadata["reference_sha256"]) == expected_reference_sha ||
    error("reference metadata contains an unfrozen safetensors SHA-256")
external_reference_sha = get(
    ENV,
    "LIFEAI_QWEN3_VL_PREFILL_REFERENCE_SHA256",
    "",
)
isempty(external_reference_sha) || reference_sha == external_reference_sha ||
    error("reference safetensors disagrees with the externally frozen SHA-256")

actual_modelscope_revision = readchomp(`git -C $model_dir rev-parse HEAD`)
actual_modelscope_revision == EXPECTED_MODELSCOPE_REVISION ||
    error("checkpoint git revision is not the frozen ModelScope revision")
checkpoint_git_status = readchomp(
    `git -C $model_dir status --porcelain=v1 --untracked-files=all`,
)
isempty(checkpoint_git_status) ||
    error("checkpoint git tree must be clean:\n$checkpoint_git_status")

checkpoint = qwen3_vl_checkpoint_spec()
checkpoint.model_id == EXPECTED_MODEL_ID || error("compiled checkpoint id mismatch")
checkpoint.modelscope_revision == EXPECTED_MODELSCOPE_REVISION ||
    error("compiled ModelScope revision mismatch")
checkpoint.hf_revision == EXPECTED_HF_REVISION ||
    error("compiled Hugging Face revision mismatch")
checkpoint_report = verify_qwen3_vl_checkpoint(model_dir)
checkpoint_report.config.text == checkpoint.text ||
    error("checkpoint text config mismatch")
checkpoint_report.config.vision == checkpoint.vision ||
    error("checkpoint vision config mismatch")
processor_spec = load_hf_qwen3_vl_processor_config(
    joinpath(model_dir, "preprocessor_config.json"),
    checkpoint.vision,
)

reference = open_safetensors_reader(reference_path)
decoder_layers = Int(metadata["decoder_layers"])
decoder_layers == checkpoint.text.num_hidden_layers ||
    error("reference decoder layer count is not frozen")
expected_names = Set([
    "raw_image_hwc_f32",
    "pixel_values_f32",
    "pixel_values_compute",
    "image_grid_thw_f32",
    "input_ids_f32",
    "processor_attention_mask_f32",
    "rope_deltas_f32",
    "vision.patch_embed",
    "vision.position_added",
    "vision.visual_embeddings",
    "decoder.input_embeddings",
    "decoder.final_hidden",
    "logits",
    "position_ids",
    "attention_mask",
    "visual_mask",
])
union!(expected_names, ("vision.block.$layer" for layer in VISION_CAPTURE_LAYERS))
union!(expected_names, ("vision.deepstack.$index" for index in 0:2))
union!(expected_names, ("decoder.block.$layer" for layer in 0:(decoder_layers - 1)))
union!(expected_names, ("decoder.layer.$layer" for layer in 0:(decoder_layers - 1)))
Set(String.(keys(reference))) == expected_names ||
    error("reference safetensors names do not match the Chapter 44 contract")

Int.(collect(metadata["vision_capture_layers"])) == collect(VISION_CAPTURE_LAYERS) ||
    error("reference vision capture layers are not frozen")
Int.(collect(metadata["image_shape_hwc"])) == [256, 256, 3] ||
    error("reference image shape is not frozen")
Int(metadata["image_token_id_0_based"]) == checkpoint.image_token_id ||
    error("reference image token id is not frozen")
Int(metadata["image_token_count"]) == 64 ||
    error("reference image token count is not frozen")
bytes2hex(sha256(codeunits(String(metadata["rendered_prompt"])))) ==
    EXPECTED_RENDERED_PROMPT_SHA256 ||
    error("reference rendered prompt text is not frozen")
String(metadata["rendered_prompt_sha256"]) == EXPECTED_RENDERED_PROMPT_SHA256 ||
    error("reference rendered prompt SHA-256 is not frozen")
bytes2hex(sha256(codeunits(String(metadata["expanded_prompt"])))) ==
    EXPECTED_EXPANDED_PROMPT_SHA256 ||
    error("reference expanded prompt text is not frozen")
String(metadata["expanded_prompt_sha256"]) == EXPECTED_EXPANDED_PROMPT_SHA256 ||
    error("reference expanded prompt SHA-256 is not frozen")

tensor_shapes = metadata["tensor_shapes"]
tensor_shapes isa JSON3.Object || error("reference tensor_shapes must be an object")
Set(String.(keys(tensor_shapes))) == expected_names ||
    error("reference tensor_shapes names do not match the tensor file")
tensor_dtypes = metadata["tensor_dtypes"]
tensor_dtypes isa JSON3.Object || error("reference tensor_dtypes must be an object")
Set(String.(keys(tensor_dtypes))) == expected_names ||
    error("reference tensor_dtypes names do not match the tensor file")
for name in expected_names
    location = reference.locations[name]
    Int.(collect(tensor_shapes[name])) == location.shape ||
        error("reference metadata shape differs from safetensors for $name")
    expected_dtype = location.dtype == "F32" ? "torch.float32" :
        location.dtype == "BF16" ? "torch.bfloat16" :
        error("unsupported reference storage dtype $(location.dtype) for $name")
    String(tensor_dtypes[name]) == expected_dtype ||
        error("reference metadata dtype differs from safetensors for $name")
end
semantic_dtypes = Dict(
    String(name) => String(value) for
    (name, value) in pairs(metadata["semantic_dtypes"])
)
semantic_dtypes == Dict(
    "raw_image_hwc_f32" => "uint8",
    "image_grid_thw_f32" => "int64",
    "input_ids_f32" => "int64",
    "processor_attention_mask_f32" => "int64",
    "rope_deltas_f32" => "int64",
    "position_ids" => "int64",
    "attention_mask" => "int64",
    "visual_mask" => "bool",
) || error("reference semantic_dtypes are not frozen")

read_reference(name) = read_safetensors_tensor(
    reference,
    name;
    target_dtype=Float32,
)

function exact_integer_reference(name)
    values = read_reference(name)
    all(isfinite, values) || error("integer reference $name is not finite")
    integers = round.(Int, values)
    all(Float32.(integers) .== values) ||
        error("integer reference $name contains a fractional value")
    return integers
end

function deterministic_image()
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

image = deterministic_image()
image_bytes = vec(permutedims(image, (3, 2, 1)))
bytes2hex(sha256(image_bytes)) == EXPECTED_IMAGE_SHA256 ||
    error("Julia deterministic image formula is not frozen")
raw_reference = read_reference("raw_image_hwc_f32")
raw_reference == Float32.(image) ||
    error("Julia deterministic image differs from the exported raw image")

processed = qwen3_vl_process_image(image; layout=:hwc, spec=processor_spec)
processed.original_size == (256, 256) || error("unexpected original image size")
processed.resized_size == (256, 256) || error("unexpected resized image size")
expected_pixels_f32 = permutedims(read_reference("pixel_values_f32"))
size(processed.pixel_values) == size(expected_pixels_f32) ||
    error("official fast processor and Julia raw path shapes differ")
raw_processor_difference = maximum(abs.(processed.pixel_values .- expected_pixels_f32))
raw_processor_difference == 0.0f0 || error(
    "official fast processor and Julia raw path differ by $raw_processor_difference",
)

grid_reference = exact_integer_reference("image_grid_thw_f32")
size(grid_reference) == (1, 3) || error("reference image grid must have shape (1,3)")
grid_thw = permutedims(grid_reference)
processed.grid_thw == grid_thw || error("Julia and reference image grids differ")
metadata_grid_rows = [Int.(collect(row)) for row in metadata["grid_thw"]]
metadata_grid = hcat(metadata_grid_rows...)
grid_thw == metadata_grid || error("reference tensor and metadata grids differ")

tokenizer = load_hf_qwen3_vl_tokenizer(
    model_dir;
    revision=EXPECTED_MODELSCOPE_REVISION,
)
messages = [(
    role="user",
    content=Any[(type="image",), (type="text", text=EXPECTED_PROMPT)],
)]
rendered_prompt = apply_qwen3_vl_chat_template(
    tokenizer,
    messages;
    add_generation_prompt=true,
    add_vision_id=false,
)
rendered_prompt == String(metadata["rendered_prompt"]) ||
    error("Julia and Transformers content-list chat rendering differ")
expanded_prompt = qwen3_vl_expand_image_placeholders(
    rendered_prompt,
    grid_thw;
    spec=processor_spec,
)
expanded_prompt == String(metadata["expanded_prompt"]) ||
    error("Julia and Transformers image-placeholder expansion differ")
input_ids_reference = exact_integer_reference("input_ids_f32")
size(input_ids_reference, 1) == 1 || error("reference batch size must be one")
input_ids_0_based = vec(input_ids_reference)
input_ids = encode(tokenizer, expanded_prompt; add_special_tokens=false)
input_ids .- 1 == input_ids_0_based ||
    error("Julia and Transformers expanded-prompt tokenization differ")
Int.(collect(metadata["input_ids_0_based"])) == input_ids_0_based ||
    error("reference token tensor and metadata differ")
length(input_ids) == Int(metadata["sequence_length"]) ||
    error("reference sequence length is inconsistent")

processor_attention = exact_integer_reference("processor_attention_mask_f32")
all(==(1), processor_attention) ||
    error("processor attention mask is not all ones")
attention_reference = exact_integer_reference("attention_mask")
attention_reference == processor_attention ||
    error("model did not receive the explicit processor attention mask")
attention_mask = reshape(Bool.(vec(attention_reference)), :, 1)
rope_layout = qwen3_vl_rope_layout(
    input_ids,
    grid_thw;
    attention_mask,
    checkpoint,
)
expected_positions = permutedims(
    exact_integer_reference("position_ids"),
    (1, 3, 2),
)
rope_layout.position_ids == expected_positions ||
    error("Julia and Transformers multimodal position ids differ")
expected_rope_deltas = exact_integer_reference("rope_deltas_f32")
rope_layout.rope_deltas == expected_rope_deltas ||
    error("Julia and Transformers rope deltas differ")
expected_visual_mask = permutedims(
    Bool.(exact_integer_reference("visual_mask")),
)
rope_layout.visual_mask == expected_visual_mask ||
    error("Julia and Transformers visual masks differ")
rope_layout.attention_mask == attention_mask ||
    error("Julia rope layout did not retain the explicit attention mask")

function assert_compute_array(value, label)
    value isa AbstractArray || error("$label is not an array")
    eltype(value) == compute_dtype || error(
        "$label dtype $(eltype(value)) does not match $compute_dtype",
    )
    if use_cuda
        value isa CUDA.CuArray || error("$label is not CUDA resident")
    else
        !(value isa CUDA.CuArray) || error("$label unexpectedly resides on CUDA")
    end
    return value
end

function parameter_device_stats(value, label="parameters")
    if value isa AbstractArray
        assert_compute_array(value, label)
        return (arrays=1, bytes=length(value) * sizeof(eltype(value)))
    elseif value isa NamedTuple
        arrays = 0
        bytes = 0
        for name in keys(value)
            stats = parameter_device_stats(getproperty(value, name), "$label.$name")
            arrays += stats.arrays
            bytes += stats.bytes
        end
        return (; arrays, bytes)
    elseif value isa Tuple
        arrays = 0
        bytes = 0
        for (index, item) in enumerate(value)
            stats = parameter_device_stats(item, "$label[$index]")
            arrays += stats.arrays
            bytes += stats.bytes
        end
        return (; arrays, bytes)
    end
    return (arrays=0, bytes=0)
end

use_cuda && CUDA.reclaim()
vision_load_timing = @timed begin
    loaded = load_hf_qwen3_vl_vision_parameters(
        model_dir;
        target_dtype=compute_dtype,
        to_device,
        spec=checkpoint.vision,
    )
    synchronize()
    loaded
end
vision_parameters = vision_load_timing.value
vision_parameter_stats = parameter_device_stats(vision_parameters, "vision")

text_load_timing = @timed begin
    loaded = load_hf_qwen3_vl_text_parameters(
        model_dir;
        target_dtype=compute_dtype,
        to_device,
        checkpoint,
    )
    synchronize()
    loaded
end
text_parameters = text_load_timing.value
text_parameter_stats = parameter_device_stats(text_parameters, "text")
total_parameter_bytes = vision_parameter_stats.bytes + text_parameter_stats.bytes
expected_parameter_bytes = qwen3_vl_parameter_count(checkpoint) * sizeof(compute_dtype)
total_parameter_bytes == expected_parameter_bytes || error(
    "resident parameter bytes $total_parameter_bytes != $expected_parameter_bytes",
)

input_pixels = compute_dtype.(processed.pixel_values)
exported_compute_pixels = permutedims(read_safetensors_tensor(
    reference,
    "pixel_values_compute";
    target_dtype=compute_dtype,
))
input_pixels == exported_compute_pixels ||
    error("Julia raw-image pixels differ after compute-dtype conversion")
vision_input = Qwen3VLVisionInput(
    to_device(input_pixels),
    grid_thw;
    spec=checkpoint.vision,
)
assert_compute_array(vision_input.pixel_values, "vision_input.pixel_values")

vision_forward_timing = @timed begin
    value = hf_qwen3_vl_vision_forward(
        vision_parameters,
        vision_input;
        capture_layers=VISION_CAPTURE_LAYERS,
    )
    synchronize()
    value
end
features = vision_forward_timing.value
assert_compute_array(features.patch_hidden_state, "vision.patch_hidden_state")
assert_compute_array(features.visual_embeddings, "vision.visual_embeddings")
for (index, value) in enumerate(features.deepstack)
    assert_compute_array(value, "vision.deepstack.$(index - 1)")
end
for layer in VISION_CAPTURE_LAYERS
    assert_compute_array(features.checkpoints[layer], "vision.block.$layer")
end

text_forward_timing = @timed begin
    value = hf_qwen3_vl_text_prefill(
        text_parameters,
        input_ids,
        rope_layout;
        vision_features=features,
        logits_to_keep=0,
        capture_layers=0:(decoder_layers - 1),
        capture_input_embeddings=true,
    )
    synchronize()
    value
end
prefill = text_forward_timing.value
assert_compute_array(prefill.input_embeddings, "decoder.input_embeddings")
assert_compute_array(prefill.final_hidden, "decoder.final_hidden")
assert_compute_array(prefill.logits, "logits")
for layer in 0:(decoder_layers - 1)
    assert_compute_array(prefill.block_outputs[layer], "decoder.block.$layer")
    assert_compute_array(prefill.layer_outputs[layer], "decoder.layer.$layer")
end

warm_timing = @timed begin
    warm_features = hf_qwen3_vl_vision_forward(
        vision_parameters,
        vision_input;
        capture_layers=VISION_CAPTURE_LAYERS,
    )
    warm_prefill = hf_qwen3_vl_text_prefill(
        text_parameters,
        input_ids,
        rope_layout;
        vision_features=warm_features,
        logits_to_keep=0,
    )
    synchronize()
    (; warm_features, warm_prefill)
end
Array(warm_timing.value.warm_features.visual_embeddings) ==
    Array(features.visual_embeddings) ||
    error("warm vision output differs from cold output")
Array(warm_timing.value.warm_prefill.logits) == Array(prefill.logits) ||
    error("warm prefill logits differ from cold output")

patch_embedding = LifeAI._qwen3_vl_linear(
    vision_parameters.patch_weight,
    vision_parameters.patch_bias,
    vision_input.pixel_values,
)
position_embedding = LifeAI._qwen3_vl_interpolated_positions(
    checkpoint.vision,
    vision_parameters.pos_embedding,
    grid_thw,
)
position_added = LifeAI._qwen3_vl_cast_like(
    patch_embedding,
    LifeAI._qwen3_vl_f32(patch_embedding) .+
        LifeAI._qwen3_vl_f32(position_embedding),
)
synchronize()

function hf_matrix_reference(name)
    value = read_reference(name)
    ndims(value) == 2 || error("reference tensor $name must be a matrix")
    return permutedims(value)
end

function hf_decoder_reference(name)
    value = read_reference(name)
    ndims(value) == 3 || error("reference tensor $name must have rank three")
    size(value, 1) == 1 || error("reference tensor $name must have batch size one")
    return permutedims(value, (3, 2, 1))
end

function stage_metrics(actual, expected)
    # Float32 reductions over the 11.5M-element logits tensor can dominate the
    # actual model error and even report cosine > 1. Accumulate every metric in
    # Float64 while preserving the model/reference values exactly.
    values = Float64.(Array(actual))
    reference_values = Float64.(expected)
    size(values) == size(expected) || error(
        "stage shape mismatch: $(size(values)) != $(size(expected))",
    )
    all(isfinite, values) || error("Julia stage contains a non-finite value")
    all(isfinite, reference_values) ||
        error("reference stage contains a non-finite value")
    differences = abs.(values .- reference_values)
    denominator = norm(reference_values)
    relative_l2 = denominator == 0 ? norm(values) :
        norm(values .- reference_values) / denominator
    cosine_denominator = norm(values) * denominator
    cosine = cosine_denominator == 0 ? 1.0 :
        dot(vec(values), vec(reference_values)) / cosine_denominator
    return (;
        max_abs=maximum(differences),
        mean_abs=mean(differences),
        relative_l2,
        cosine,
    )
end

const BF16_VISION_LIMITS = Dict(
    "vision.patch_embed" => (0.0078125f0, 5.0f-7, 1.0e-4, 0.99999),
    "vision.position_added" => (0.0078125f0, 5.0f-7, 1.0e-4, 0.99999),
    "vision.block.0" => (0.0625f0, 0.001f0, 0.004, 0.99998),
    "vision.block.5" => (0.125f0, 0.006f0, 0.02, 0.9999),
    "vision.block.11" => (32.0f0, 0.01f0, 0.09, 0.996),
    "vision.block.17" => (48.0f0, 0.03f0, 0.10, 0.996),
    "vision.block.23" => (512.0f0, 0.45f0, 0.07, 0.998),
    "vision.deepstack.0" => (0.125f0, 0.004f0, 0.012, 0.9999),
    "vision.deepstack.1" => (1.75f0, 0.015f0, 0.08, 0.997),
    "vision.deepstack.2" => (2.5f0, 0.02f0, 0.06, 0.9985),
    "vision.visual_embeddings" => (1.25f0, 0.03f0, 0.07, 0.998),
)
const FLOAT32_VISION_LIMITS = Dict(
    "vision.patch_embed" => (2.0f-5, 1.0f-6, 2.0e-6, 0.99999),
    "vision.position_added" => (2.0f-5, 1.0f-6, 2.0e-6, 0.99999),
    "vision.block.0" => (3.0f-5, 2.0f-6, 2.0e-6, 0.99999),
    "vision.block.5" => (1.0f-4, 2.0f-6, 4.0e-6, 0.99999),
    "vision.block.11" => (2.0f-3, 3.0f-6, 6.0e-6, 0.99999),
    "vision.block.17" => (2.0f-2, 1.0f-5, 3.0e-5, 0.99998),
    "vision.block.23" => (2.5f-1, 1.0f-4, 2.0e-5, 0.99998),
    "vision.deepstack.0" => (2.0f-5, 2.0f-6, 3.0e-6, 0.99999),
    "vision.deepstack.1" => (2.0f-4, 3.0f-6, 1.0e-5, 0.99999),
    "vision.deepstack.2" => (1.0f-3, 1.0f-5, 2.0e-5, 0.99998),
    "vision.visual_embeddings" => (1.0f-3, 1.0f-5, 2.0e-5, 0.99998),
)

function text_limits(name, layer=nothing)
    if compute_dtype === BFloat16
        name == "decoder.input_embeddings" && return (1.25, 0.03, 0.07, 0.998)
        name == "decoder.final_hidden" && return (8.0, 0.18, 0.12, 0.994)
        name == "logits" && return (10.0, 0.28, 0.08, 0.997)
        layer == 0 && return (1.5, 0.05, 0.07, 0.998)
        layer == 1 && return (2.25, 0.055, 0.06, 0.9985)
        layer == 2 && return (3.5, 0.06, 0.005, 0.9999)
        layer <= 9 && return (4.0, 0.07, 0.004, 0.9999)
        layer <= 13 && return (12.0, 0.13, 0.006, 0.9999)
        layer <= 17 && return (8.0, 0.28, 0.012, 0.9999)
        layer <= 20 && return (16.0, 0.55, 0.025, 0.9997)
        layer <= 23 && return (56.0, 1.1, 0.045, 0.9993)
        layer <= 26 && return (80.0, 1.6, 0.055, 0.999)
        return (80.0, 1.75, 0.12, 0.994)
    end
    name == "decoder.input_embeddings" &&
        return (3.0e-4, 1.0e-5, 2.0e-5, 0.999999)
    name == "decoder.final_hidden" &&
        return (2.0e-3, 5.0e-5, 3.0e-5, 0.999999)
    name == "logits" && return (2.0e-3, 5.0e-5, 3.0e-5, 0.999999)
    layer <= 1 && return (5.0e-4, 1.5e-5, 2.0e-5, 0.999999)
    layer <= 24 && return (1.0e-2, 2.0e-4, 3.0e-5, 0.999999)
    return (2.0e-2, 3.0e-4, 3.0e-5, 0.999999)
end

function check_stage!(passed, name, actual, expected, limits)
    metrics = stage_metrics(actual, expected)
    println(join((
        name,
        metrics.max_abs,
        metrics.mean_abs,
        metrics.relative_l2,
        metrics.cosine,
    ), '\t'))
    max_abs, mean_abs, relative_l2, cosine = limits
    passed[] &= metrics.max_abs <= max_abs
    passed[] &= metrics.mean_abs <= mean_abs
    passed[] &= metrics.relative_l2 <= relative_l2
    passed[] &= metrics.cosine >= cosine
    return metrics
end

println("modelscope_revision\t", metadata["modelscope_revision"])
println("huggingface_revision\t", metadata["huggingface_revision"])
println("transformers_version\t", metadata["transformers_version"])
println("torch_version\t", metadata["torch_version"])
println("torchvision_version\t", metadata["torchvision_version"])
println("compute_dtype\t", compute_dtype_name)
println(
    "numerical_contract\t",
    compute_dtype === Float32 ?
        "strict_float32_prefill_parity" : "cross_backend_bfloat16_prefill_boundary",
)
println("reference_sha256\t", reference_sha)
println("backend\t", backend)
println("device_name\t", use_cuda ? CUDA.name(CUDA.device()) : "CPU")
println("sequence_length\t", length(input_ids))
println("image_tokens\t", count(rope_layout.visual_mask))
println("raw_processor_max_abs\t", raw_processor_difference)
println("vision_parameter_arrays\t", vision_parameter_stats.arrays)
println("text_parameter_arrays\t", text_parameter_stats.arrays)
println("total_parameter_bytes\t", total_parameter_bytes)
println("vision_load_seconds\t", vision_load_timing.time)
println("text_load_seconds\t", text_load_timing.time)
println("vision_cold_forward_seconds\t", vision_forward_timing.time)
println("text_cold_forward_seconds\t", text_forward_timing.time)
println("combined_warm_forward_seconds\t", warm_timing.time)
println("stage\tmax_abs\tmean_abs\trelative_l2\tcosine")

vision_stages = Pair{String,Any}[
    "vision.patch_embed" => patch_embedding,
    "vision.position_added" => position_added,
    "vision.block.0" => features.checkpoints[0],
    "vision.block.5" => features.checkpoints[5],
    "vision.block.11" => features.checkpoints[11],
    "vision.block.17" => features.checkpoints[17],
    "vision.block.23" => features.checkpoints[23],
    "vision.deepstack.0" => features.deepstack[1],
    "vision.deepstack.1" => features.deepstack[2],
    "vision.deepstack.2" => features.deepstack[3],
    "vision.visual_embeddings" => features.visual_embeddings,
]
passed = Ref(true)
vision_limits = compute_dtype === Float32 ?
    FLOAT32_VISION_LIMITS : BF16_VISION_LIMITS
for (name, actual) in vision_stages
    check_stage!(passed, name, actual, hf_matrix_reference(name), vision_limits[name])
end
features.patch_hidden_state === features.checkpoints[23] ||
    error("vision patch_hidden_state must be the captured final block")

check_stage!(
    passed,
    "decoder.input_embeddings",
    prefill.input_embeddings,
    hf_decoder_reference("decoder.input_embeddings"),
    text_limits("decoder.input_embeddings"),
)
for layer in 0:(decoder_layers - 1)
    for kind in ("block", "layer")
        name = "decoder.$kind.$layer"
        actual = kind == "block" ?
            prefill.block_outputs[layer] : prefill.layer_outputs[layer]
        check_stage!(
            passed,
            name,
            actual,
            hf_decoder_reference(name),
            text_limits(name, layer),
        )
    end
end
check_stage!(
    passed,
    "decoder.final_hidden",
    prefill.final_hidden,
    hf_decoder_reference("decoder.final_hidden"),
    text_limits("decoder.final_hidden"),
)
check_stage!(
    passed,
    "logits",
    prefill.logits,
    hf_decoder_reference("logits"),
    text_limits("logits"),
)

passed[] || error(
    "Qwen3-VL image-prefill numerical contract exceeded frozen tolerances",
)
println(
    compute_dtype === Float32 ?
        "prefill_parity_passed\ttrue" : "prefill_cross_backend_boundary_passed\ttrue",
)
