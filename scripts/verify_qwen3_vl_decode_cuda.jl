#!/usr/bin/env julia

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
const EXPECTED_IMAGE_SHA256 =
    "ec143579b36852cf212bbb368798479d193a8c6d039942fca89a49fc820dff3f"
const EXPECTED_RENDERED_PROMPT_SHA256 =
    "7a50d10ccb53359de53e3e9b032c39b15fd3abbfed51ec844117c3b93da07271"
const EXPECTED_REFERENCE_SHA256 =
    "a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81"
const EXPECTED_METADATA_SHA256 =
    "569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc"
const EXPECTED_GREEDY_IDS_0_BASED = [1986, 2168, 374, 264]
const EXPECTED_GENERATED_TEXT = "This image is a"
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

file_sha256(path) = open(path, "r") do io
    bytes2hex(sha256(io))
end

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
    usage = "usage: julia --project=. scripts/verify_qwen3_vl_decode_cuda.jl " *
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

function stage_metrics(actual, expected)
    values = Float64.(Array(actual))
    reference = Float64.(expected)
    size(values) == size(reference) || error(
        "decode stage shape mismatch: $(size(values)) != $(size(reference))",
    )
    all(isfinite, values) || error("Julia decode stage contains non-finite values")
    all(isfinite, reference) || error("reference decode stage is non-finite")
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

function cache_bytes(cache::Qwen3VLKVCache)
    return sum(cache.layers; init=0) do layer
        length(layer.keys) * sizeof(eltype(layer.keys)) +
            length(layer.values) * sizeof(eltype(layer.values))
    end
end

function assert_cache_geometry(cache, spec, expected_length)
    cache.position == expected_length || error(
        "cache position $(cache.position) != $expected_length",
    )
    expected_shape = (
        spec.head_dim,
        spec.num_key_value_heads,
        expected_length,
        1,
    )
    for (index, layer) in enumerate(cache.layers)
        size(layer.keys) == expected_shape || error(
            "layer $index key cache geometry changed",
        )
        size(layer.values) == expected_shape || error(
            "layer $index value cache geometry changed",
        )
        layer.keys isa CUDA.CuArray || error("layer $index keys are not CUDA resident")
        layer.values isa CUDA.CuArray || error("layer $index values are not CUDA resident")
    end
    return nothing
end

function assert_cache_prefix_immutable(old, new)
    new.position == old.position + 1 || error(
        "decode did not append exactly one physical cache position",
    )
    for (index, (old_layer, new_layer)) in enumerate(zip(old.layers, new.layers))
        Array(@view(new_layer.keys[:, :, 1:old.position, :])) ==
            Array(old_layer.keys) || error("decode mutated key prefix in layer $index")
        Array(@view(new_layer.values[:, :, 1:old.position, :])) ==
            Array(old_layer.values) || error("decode mutated value prefix in layer $index")
    end
    return nothing
end

model_dir, reference_dir, backend = parse_arguments(ARGS)
use_cuda = backend == "cuda"
use_cuda || error("Chapter 45 strict verifier currently requires the CUDA backend")
CUDA.functional() || error("CUDA.jl is not functional on this machine")
CUDA.allowscalar(false)
to_device = CUDA.cu
synchronize() = CUDA.synchronize()

metadata_path = joinpath(reference_dir, "reference.json")
reference_path = joinpath(reference_dir, "reference.safetensors")
isfile(metadata_path) || error("reference metadata does not exist: $metadata_path")
isfile(reference_path) || error("reference tensors do not exist: $reference_path")
file_sha256(metadata_path) == EXPECTED_METADATA_SHA256 ||
    error("decode reference metadata SHA-256 is not frozen")
file_sha256(reference_path) == EXPECTED_REFERENCE_SHA256 ||
    error("decode reference safetensors SHA-256 is not frozen")
metadata = JSON3.read(read(metadata_path, String))

String(metadata["oracle"]) ==
    "qwen3_vl_2b_256_describe_dynamic_cache_greedy_decode" ||
    error("decode oracle identity is not frozen")
String(metadata["model_id"]) == EXPECTED_MODEL_ID ||
    error("decode reference model id is not frozen")
String(metadata["modelscope_revision"]) == EXPECTED_MODELSCOPE_REVISION ||
    error("decode reference ModelScope revision is not frozen")
String(metadata["huggingface_revision"]) == EXPECTED_HF_REVISION ||
    error("decode reference Hugging Face revision is not frozen")
String(metadata["transformers"]) == EXPECTED_TRANSFORMERS_VERSION ||
    error("decode reference Transformers version is not frozen")
String(metadata["torch"]) == EXPECTED_TORCH_VERSION ||
    error("decode reference Torch version is not frozen")
String(metadata["compute_dtype"]) == "float32" ||
    error("strict decode reference must use Float32")
String(metadata["compute_device"]) == "cpu" ||
    error("strict decode reference must use the frozen CPU oracle")
String(metadata["attention_implementation"]) == "eager" ||
    error("decode reference attention implementation is not eager")
String(metadata["attention_mask_contract"]) == "explicit_all_ones_every_call" ||
    error("decode reference attention-mask contract changed")
String(metadata["cache_capture_contract"]) ==
    "detach_clone_cpu_before_next_dynamic_cache_update" ||
    error("decode cache snapshot contract changed")
Bool(metadata["cache_prefix_immutable"]) ||
    error("reference did not verify DynamicCache prefix immutability")
String(metadata["hf_cache_layout"]) == "batch,kv_heads,tokens,head_dim" ||
    error("HF cache layout changed")
String(metadata["julia_cache_layout"]) == "head_dim,kv_heads,tokens,batch" ||
    error("Julia cache layout changed")
Int.(collect(metadata["hf_to_julia_permutation_1_based"])) == [4, 2, 3, 1] ||
    error("HF-to-Julia cache permutation changed")
String(metadata["image_sha256"]) == EXPECTED_IMAGE_SHA256 ||
    error("decode reference image is not frozen")
String(metadata["rendered_prompt_sha256"]) == EXPECTED_RENDERED_PROMPT_SHA256 ||
    error("decode rendered prompt is not frozen")
String(metadata["reference_sha256"]) == EXPECTED_REFERENCE_SHA256 ||
    error("metadata reference SHA-256 changed")
Int.(collect(metadata["greedy_token_ids_0_based"])) ==
    EXPECTED_GREEDY_IDS_0_BASED || error("reference greedy token ids changed")
String(metadata["generated_text"]) == EXPECTED_GENERATED_TEXT ||
    error("reference generated text changed")
Int(metadata["prefill_length"]) == 76 || error("reference prompt length changed")
Int(metadata["image_token_count"]) == 64 ||
    error("reference image token count changed")
Int(metadata["rope_delta"]) == -56 || error("reference rope_delta changed")
Int(metadata["greedy_token_count"]) == 4 ||
    error("reference greedy token count changed")
for (filename, expected) in EXPECTED_ASSET_SHA256
    String(metadata["asset_sha256"][filename]) == expected ||
        error("reference asset hash changed for $filename")
end

actual_revision = readchomp(`git -C $model_dir rev-parse HEAD`)
actual_revision == EXPECTED_MODELSCOPE_REVISION ||
    error("checkpoint is not the frozen ModelScope revision")
isempty(readchomp(`git -C $model_dir status --porcelain=v1 --untracked-files=all`)) ||
    error("checkpoint git tree is dirty")
checkpoint = qwen3_vl_checkpoint_spec()
report = verify_qwen3_vl_checkpoint(model_dir)
report.config.text == checkpoint.text || error("checkpoint text config changed")
report.config.vision == checkpoint.vision || error("checkpoint vision config changed")
processor_spec = load_hf_qwen3_vl_processor_config(
    joinpath(model_dir, "preprocessor_config.json"),
    checkpoint.vision,
)

reference = open_safetensors_reader(reference_path)
expected_tensor_names = Set([
    "prefill.final_hidden_last",
    "prefill.logits",
    "decode.0.final_hidden_last",
    "decode.0.logits",
    "decode.1.final_hidden_last",
    "decode.1.logits",
    "decode.2.final_hidden_last",
    "decode.2.logits",
])
Set(String.(keys(reference))) == expected_tensor_names ||
    error("decode reference tensor names changed")
read_reference(name) = read_safetensors_tensor(
    reference,
    name;
    target_dtype=Float32,
)
function logits_reference(name)
    value = read_reference(name)
    (size(value, 1) == 1 && size(value, 2) == 1) ||
        error("$name is not a batch-1 token logit")
    return permutedims(value, (3, 2, 1))
end

image = deterministic_image()
image_bytes = vec(permutedims(image, (3, 2, 1)))
bytes2hex(sha256(image_bytes)) == EXPECTED_IMAGE_SHA256 ||
    error("Julia deterministic image changed")
processed = qwen3_vl_process_image(image; layout=:hwc, spec=processor_spec)
processed.original_size == (256, 256) || error("raw image size changed")
processed.resized_size == (256, 256) || error("processed image size changed")
processed.grid_thw == reshape(Int[1, 16, 16], 3, 1) ||
    error("processed image grid changed")

tokenizer = load_hf_qwen3_vl_tokenizer(
    model_dir;
    revision=EXPECTED_MODELSCOPE_REVISION,
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
rendered_prompt == String(metadata["rendered_prompt"]) ||
    error("Julia and reference rendered prompts differ")
expanded_prompt = qwen3_vl_expand_image_placeholders(
    rendered_prompt,
    processed.grid_thw;
    spec=processor_spec,
)
input_ids = encode(tokenizer, expanded_prompt; add_special_tokens=false)
input_ids .- 1 == Int.(collect(metadata["input_ids_0_based"])) ||
    error("Julia and reference tokenization differ")
rope_layout = qwen3_vl_rope_layout(
    input_ids,
    processed.grid_thw;
    checkpoint,
)
only(rope_layout.rope_deltas) == -56 || error("Julia rope_delta changed")
all(rope_layout.attention_mask) || error("generation prompt is unexpectedly padded")

CUDA.reclaim()
vision_load = @timed begin
    value = load_hf_qwen3_vl_vision_parameters(
        model_dir;
        target_dtype=Float32,
        to_device,
        spec=checkpoint.vision,
    )
    synchronize()
    value
end
vision_parameters = vision_load.value
text_load = @timed begin
    value = load_hf_qwen3_vl_text_parameters(
        model_dir;
        target_dtype=Float32,
        to_device,
        checkpoint,
    )
    synchronize()
    value
end
text_parameters = text_load.value

vision_input = Qwen3VLVisionInput(
    to_device(Float32.(processed.pixel_values)),
    processed.grid_thw;
    spec=checkpoint.vision,
)
vision_timing = @timed begin
    value = hf_qwen3_vl_vision_forward(vision_parameters, vision_input)
    synchronize()
    value
end
features = vision_timing.value
cache = init_qwen3_vl_kv_cache(text_parameters; batch_size=1)
prefill_timing = @timed begin
    value = hf_qwen3_vl_text_prefill_cached(
        text_parameters,
        input_ids,
        rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    synchronize()
    value
end
prefill, cache = prefill_timing.value
cache.rope_delta == -56 || error("cached request lost its rope_delta")
assert_cache_geometry(cache, checkpoint.text, 76)

phase_names = ("prefill", "decode.0", "decode.1", "decode.2")
phases = metadata["phases"]
length(phases) == length(phase_names) || error("reference phase count changed")
decode_result = let cache=cache, logits=prefill.logits
    generated_ids = Int[]
    decode_seconds = Float64[]
    passed = true
    println("stage\tmax_abs\tmean_abs\trelative_l2\tcosine\targmax_1_based")
    for phase_index in eachindex(phase_names)
        phase = phase_names[phase_index]
        String(phases[phase_index]["name"]) == phase ||
            error("reference phase order changed")
        expected_logits = logits_reference("$phase.logits")
        metrics = stage_metrics(logits, expected_logits)
        token_id = argmax(vec(Array(logits)))
        reference_token =
            Int(phases[phase_index]["top_two"]["top1_token_id_1_based"])
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
        passed &= metrics.max_abs <= 1.0e-4
        passed &= metrics.mean_abs <= 2.0e-5
        passed &= metrics.relative_l2 <= 5.0e-6
        passed &= metrics.cosine >= 0.99999999

        phase_index == length(phase_names) && continue
        old_cache = cache
        started = time_ns()
        logits, cache = hf_qwen3_vl_text_decode_step(
            text_parameters,
            token_id,
            cache,
        )
        synchronize()
        push!(decode_seconds, (time_ns() - started) / 1.0e9)
        assert_cache_prefix_immutable(old_cache, cache)
        expected_length = 76 + phase_index
        assert_cache_geometry(cache, checkpoint.text, expected_length)
        cache.rope_delta == -56 || error("decode changed request rope_delta")
        expected_coordinate = old_cache.position + old_cache.rope_delta
        reference_coordinate = Int(
            phases[phase_index + 1]["mrope_position_ids_thw_0_based"][1],
        )
        expected_coordinate == reference_coordinate ||
            error("decode mRoPE coordinate differs from the reference")
    end
    passed || error("Qwen3-VL Float32 cached-decode tolerance was exceeded")
    (; cache, generated_ids, decode_seconds)
end
cache = decode_result.cache
generated_ids = decode_result.generated_ids
decode_seconds = decode_result.decode_seconds
generated_ids .- 1 == EXPECTED_GREEDY_IDS_0_BASED ||
    error("manual cached decode greedy ids changed")
decode(tokenizer, generated_ids; skip_special_tokens=false) ==
    EXPECTED_GENERATED_TEXT || error("manual cached decode text changed")

high_level_timing = @timed begin
    value = generate_hf_qwen3_vl(
        vision_parameters,
        text_parameters,
        tokenizer,
        messages;
        max_new_tokens=4,
        capture_logits=false,
        skip_special_tokens=false,
    )
    synchronize()
    value
end
generated = high_level_timing.value
generated.generated_ids == generated_ids ||
    error("raw-image generation wrapper differs from manual cached decode")
generated.completion == EXPECTED_GENERATED_TEXT ||
    error("raw-image generation wrapper text changed")
generated.prompt == rendered_prompt ||
    error("raw-image generation wrapper prompt changed")
generated.expanded_prompt == expanded_prompt ||
    error("raw-image generation wrapper placeholder expansion changed")
generated.cache.position == 79 || error("generation wrapper cache length changed")
generated.cache.rope_delta == -56 || error("generation wrapper rope_delta changed")

expected_cache_bytes = 2 * checkpoint.text.num_hidden_layers *
    checkpoint.text.head_dim * checkpoint.text.num_key_value_heads *
    cache.position * sizeof(Float32)
cache_bytes(cache) == expected_cache_bytes ||
    error("dynamic cache byte count changed")
println("modelscope_revision\t", EXPECTED_MODELSCOPE_REVISION)
println("huggingface_revision\t", EXPECTED_HF_REVISION)
println("reference_sha256\t", EXPECTED_REFERENCE_SHA256)
println("metadata_sha256\t", EXPECTED_METADATA_SHA256)
println("device_name\t", CUDA.name(CUDA.device()))
println("sequence_length\t", length(input_ids))
println("rope_delta\t", cache.rope_delta)
println("decode_coordinates_0_based\t20,21,22")
println("greedy_ids_0_based\t", join(generated_ids .- 1, ','))
println("generated_text\t", generated.completion)
println("final_cache_tokens\t", cache.position)
println("final_cache_bytes\t", cache_bytes(cache))
println("vision_load_seconds\t", vision_load.time)
println("text_load_seconds\t", text_load.time)
println("vision_forward_seconds\t", vision_timing.time)
println("cached_prefill_seconds\t", prefill_timing.time)
println("decode_seconds\t", join(decode_seconds, ','))
println("high_level_generation_seconds\t", high_level_timing.time)
println("decode_parity_passed\ttrue")
