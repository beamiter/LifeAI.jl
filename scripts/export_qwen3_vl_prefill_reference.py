#!/usr/bin/env python3
"""Export a frozen Transformers 4.57 Qwen3-VL image-prefill oracle.

This is an offline provenance script, not a general inference entry point.  It
uses one deterministic 256 x 256 RGB image, the content-list chat prompt
``Describe.``, the official ``Qwen2VLImageProcessorFast`` raw-image path, and
an explicit all-ones attention mask.  References are written outside the
repository by the Chapter 44 verification workflow.

Decoder hooks clone their values immediately.  That detail is essential:
Transformers' Qwen3-VL implementation adds DeepStack features by mutating the
decoder-layer output tensor in place after the layer's forward hooks run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import safetensors
import torch
import jinja2
import tokenizers

# The frozen oracle uses CPU-only Torch with a torchvision wheel whose
# optional CUDA NMS extension cannot load. Pure tensor/PIL transforms remain
# available, but torchvision otherwise aborts while registering a fake kernel
# for the absent operator. Skip only that unavailable optional registration;
# every other import failure remains fatal.
_original_register_fake = torch.library.register_fake


def _register_fake_if_operator_exists(operator_name: str, *args, **kwargs):
    decorator = _original_register_fake(operator_name, *args, **kwargs)

    def register(function):
        try:
            return decorator(function)
        except RuntimeError as error:
            if (
                operator_name == "torchvision::nms"
                and "does not exist" in str(error)
            ):
                return function
            raise

    return register


torch.library.register_fake = _register_fake_if_operator_exists
try:
    import torchvision
finally:
    torch.library.register_fake = _original_register_fake

import transformers
from PIL import Image
from safetensors.torch import save_file
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_NUMPY_VERSION = "1.26.4"
EXPECTED_PILLOW_VERSION = "11.3.0"
EXPECTED_SAFETENSORS_VERSION = "0.5.3"
EXPECTED_TOKENIZERS_VERSION = "0.22.1"
EXPECTED_JINJA2_VERSION = "3.1.6"
EXPECTED_TRANSFORMERS_VERSION = "4.57.0"
EXPECTED_TORCH_VERSION = "2.7.1+cpu"
EXPECTED_TORCH_GIT_VERSION = "e2d141dbde55c2a4370fac5165b0561b6af4798b"
EXPECTED_TORCH_CONFIG_SHA256 = (
    "dcb7f1e248794c5c144992643f929d0dc623511210f6fe5615f78cb009d64745"
)
EXPECTED_TORCHVISION_VERSION = "0.22.1+cu126"
EXPECTED_TORCH_NUM_THREADS = 24
EXPECTED_TORCH_INTEROP_THREADS = 24
EXPECTED_CPU_CAPABILITY = "AVX2"
EXPECTED_MODELSCOPE_REVISION = "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53"
EXPECTED_HF_REVISION = "78448d793a7eb2f7a987a1da76d464384aa1becd"
EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
)
EXPECTED_ASSET_SHA256 = {
    "model.safetensors": (
        "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
    ),
    "config.json": (
        "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
    ),
    "preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "tokenizer_config.json": (
        "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399"
    ),
    "tokenizer.json": (
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
    ),
    "generation_config.json": (
        "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4"
    ),
    "chat_template.json": (
        "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4"
    ),
}
PROMPT = "Describe."
IMAGE_SIZE = 256
VISION_CAPTURE_LAYERS = (0, 5, 11, 17, 23)


def _git_revision(model_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(model_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _git_status(model_dir: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(model_dir),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _deterministic_image() -> np.ndarray:
    rows, columns = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint32)
    return np.stack(
        (
            (3 * columns + 5 * rows + 17) % 256,
            (11 * columns + 7 * rows + 29) % 256,
            (13 * columns + 19 * rows + 43) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    """Detach and clone before any caller can mutate ``value`` in place."""

    return value.detach().clone().to(device="cpu").contiguous()


def _tensor_output(output: Any, label: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"{label} hook returned unsupported value {type(output).__name__}")


def _expanded_image_prompt(
    rendered_prompt: str,
    processor: Qwen3VLProcessor,
    grid_thw: torch.Tensor,
) -> tuple[str, int]:
    image_token = processor.image_token
    if rendered_prompt.count(image_token) != 1:
        raise RuntimeError("the frozen content-list prompt must contain one image token")
    merge_length = processor.image_processor.merge_size**2
    image_token_count = int(grid_thw[0].prod().item()) // merge_length
    placeholder = "<|placeholder|>"
    expanded = rendered_prompt.replace(
        image_token,
        placeholder * image_token_count,
        1,
    ).replace(placeholder, image_token)
    if expanded.count(image_token) != image_token_count:
        raise RuntimeError("expanded image placeholder count is inconsistent")
    return expanded, image_token_count


def _parse_args() -> tuple[Path, Path, str, torch.dtype]:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "usage: export_qwen3_vl_prefill_reference.py MODEL_DIR OUTPUT_DIR "
            "[float32|bfloat16]"
        )
    model_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    dtype_name = sys.argv[3] if len(sys.argv) == 4 else "float32"
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported compute dtype: {dtype_name}")
    return model_dir, output_dir, dtype_name, dtypes[dtype_name]


def main() -> None:
    model_dir, output_dir, compute_dtype_name, compute_dtype = _parse_args()

    # CPU reduction order is part of the frozen oracle. Set it explicitly so
    # a machine-wide thread default cannot silently produce a new reference.
    torch.set_num_threads(EXPECTED_TORCH_NUM_THREADS)
    torch.set_num_interop_threads(EXPECTED_TORCH_INTEROP_THREADS)
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"expected Python {EXPECTED_PYTHON_VERSION}, "
            f"got {platform.python_version()}"
        )
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"expected NumPy {EXPECTED_NUMPY_VERSION}, got {np.__version__}"
        )
    if PIL.__version__ != EXPECTED_PILLOW_VERSION:
        raise RuntimeError(
            f"expected Pillow {EXPECTED_PILLOW_VERSION}, got {PIL.__version__}"
        )
    if safetensors.__version__ != EXPECTED_SAFETENSORS_VERSION:
        raise RuntimeError(
            f"expected safetensors {EXPECTED_SAFETENSORS_VERSION}, "
            f"got {safetensors.__version__}"
        )
    if tokenizers.__version__ != EXPECTED_TOKENIZERS_VERSION:
        raise RuntimeError(
            f"expected tokenizers {EXPECTED_TOKENIZERS_VERSION}, "
            f"got {tokenizers.__version__}"
        )
    if jinja2.__version__ != EXPECTED_JINJA2_VERSION:
        raise RuntimeError(
            f"expected Jinja2 {EXPECTED_JINJA2_VERSION}, got {jinja2.__version__}"
        )
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {EXPECTED_TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"expected Torch {EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )
    if torch.version.git_version != EXPECTED_TORCH_GIT_VERSION:
        raise RuntimeError("Torch build git revision is not frozen")
    torch_config_sha256 = hashlib.sha256(
        torch.__config__.show().encode("utf-8")
    ).hexdigest()
    if torch_config_sha256 != EXPECTED_TORCH_CONFIG_SHA256:
        raise RuntimeError("Torch build configuration is not frozen")
    if torchvision.__version__ != EXPECTED_TORCHVISION_VERSION:
        raise RuntimeError(
            f"expected torchvision {EXPECTED_TORCHVISION_VERSION}, "
            f"got {torchvision.__version__}"
        )
    if torch.get_num_threads() != EXPECTED_TORCH_NUM_THREADS:
        raise RuntimeError("Torch intra-op thread count is not frozen")
    if torch.get_num_interop_threads() != EXPECTED_TORCH_INTEROP_THREADS:
        raise RuntimeError("Torch inter-op thread count is not frozen")
    if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled:
        raise RuntimeError("the frozen oracle requires enabled oneDNN/MKLDNN")
    if torch.backends.mkldnn.deterministic:
        raise RuntimeError("the frozen oracle requires MKLDNN deterministic=false")
    if not torch.backends.mkl.is_available() or not torch.backends.openmp.is_available():
        raise RuntimeError("the frozen oracle requires MKL and OpenMP")
    if torch.backends.cpu.get_cpu_capability() != EXPECTED_CPU_CAPABILITY:
        raise RuntimeError("the frozen oracle CPU capability is not AVX2")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("the frozen oracle requires highest Float32 matmul precision")

    revision = _git_revision(model_dir)
    if revision != EXPECTED_MODELSCOPE_REVISION:
        raise RuntimeError(
            f"expected ModelScope revision {EXPECTED_MODELSCOPE_REVISION}, "
            f"got {revision}"
        )
    dirty = _git_status(model_dir)
    if dirty:
        raise RuntimeError(f"checkpoint git tree must be clean:\n{dirty}")

    actual_asset_hashes: dict[str, str] = {}
    for filename, expected_sha256 in EXPECTED_ASSET_SHA256.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"frozen asset SHA-256 mismatch for {filename}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        actual_asset_hashes[filename] = actual_sha256

    chat_template_payload = json.loads(
        (model_dir / "chat_template.json").read_text(encoding="utf-8")
    )
    if set(chat_template_payload) != {"chat_template"}:
        raise RuntimeError("frozen chat_template.json has unexpected fields")
    chat_template = chat_template_payload["chat_template"]
    if _sha256_text(chat_template) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise RuntimeError("frozen Qwen3-VL chat-template text SHA-256 mismatch")

    # Qwen3VLProcessor loads the checkpoint tokenizer, video processor, and
    # chat template.  The class assertion below pins the image leg to the
    # official fast raw-image implementation rather than its slow sibling.
    processor = Qwen3VLProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    if type(processor.image_processor) is not Qwen2VLImageProcessorFast:
        raise RuntimeError(
            "expected the official Qwen2VLImageProcessorFast, got "
            f"{type(processor.image_processor).__name__}"
        )
    if processor.chat_template != chat_template:
        raise RuntimeError("processor chat template disagrees with frozen asset")

    image = _deterministic_image()
    pil_image = Image.fromarray(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    rendered_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=False,
    )
    if not isinstance(rendered_prompt, str):
        raise RuntimeError("content-list chat rendering did not return text")

    processed = processor(
        images=[pil_image],
        text=[rendered_prompt],
        padding=False,
        return_tensors="pt",
    )
    required_inputs = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }
    missing_inputs = required_inputs.difference(processed)
    if missing_inputs:
        raise RuntimeError(f"processor omitted inputs: {sorted(missing_inputs)}")
    input_ids = processed["input_ids"].to(torch.int64).contiguous()
    processor_attention_mask = processed["attention_mask"].to(torch.int64).contiguous()
    # Construct a fresh mask instead of forwarding None.  Transformers 4.57
    # otherwise interprets repeated temporal mRoPE positions as packed-sequence
    # boundaries in create_causal_mask.
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    if not torch.equal(processor_attention_mask, attention_mask):
        raise RuntimeError("the unpadded frozen prompt must have an all-ones mask")
    grid_thw = processed["image_grid_thw"].to(torch.int64).contiguous()
    pixel_values_f32 = processed["pixel_values"].to(torch.float32).contiguous()
    if tuple(image.shape) != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise RuntimeError(f"unexpected deterministic image shape: {image.shape}")
    if tuple(pixel_values_f32.shape) != (256, 1536):
        raise RuntimeError(f"unexpected pixel_values shape: {pixel_values_f32.shape}")
    if grid_thw.tolist() != [[1, 16, 16]]:
        raise RuntimeError(f"unexpected image grid: {grid_thw.tolist()}")

    expanded_prompt, image_token_count = _expanded_image_prompt(
        rendered_prompt,
        processor,
        grid_thw,
    )
    independently_tokenized = processor.tokenizer(
        [expanded_prompt],
        padding=False,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    if not torch.equal(
        independently_tokenized["input_ids"].to(torch.int64),
        input_ids,
    ):
        raise RuntimeError("expanded prompt tokenization disagrees with processor output")
    if int((input_ids == processor.image_token_id).sum().item()) != image_token_count:
        raise RuntimeError("tokenized image placeholder count is inconsistent")

    config = Qwen3VLConfig.from_pretrained(str(model_dir), local_files_only=True)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    config.use_cache = False
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        config=config,
        dtype=compute_dtype,
        attn_implementation="eager",
        local_files_only=True,
        use_safetensors=True,
    )
    model.eval()
    model.config.use_cache = False
    if model.get_input_embeddings().weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("the frozen checkpoint requires tied input/output embeddings")
    if model.model.language_model.config._attn_implementation != "eager":
        raise RuntimeError("text attention implementation is not eager")
    if model.model.visual.config._attn_implementation != "eager":
        raise RuntimeError("vision attention implementation is not eager")

    captures: dict[str, torch.Tensor] = {}

    def save_capture(name: str, value: torch.Tensor) -> None:
        if name in captures:
            raise RuntimeError(f"capture {name} fired more than once")
        captures[name] = _clone_cpu(value)

    def capture_output(name: str):
        def hook(_module, _inputs, output):
            save_capture(name, _tensor_output(output, name))

        return hook

    def capture_input(name: str):
        def hook(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{name} hook did not receive a tensor input")
            save_capture(name, inputs[0])

        return hook

    def capture_language_inputs(_module, args, kwargs):
        if args:
            raise RuntimeError("language model was expected to receive keyword inputs")
        for key in ("inputs_embeds", "position_ids", "attention_mask", "visual_pos_masks"):
            value = kwargs.get(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"language-model {key} is not a tensor")
        save_capture("decoder.input_embeddings", kwargs["inputs_embeds"])
        save_capture("position_ids", kwargs["position_ids"])
        save_capture("attention_mask", kwargs["attention_mask"])
        save_capture("visual_mask", kwargs["visual_pos_masks"])
        deepstack = kwargs.get("deepstack_visual_embeds")
        if not isinstance(deepstack, (tuple, list)) or len(deepstack) != 3:
            raise RuntimeError("language model did not receive three DeepStack features")

    def capture_decoder_entry(previous_layer: int):
        def hook(_module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"decoder layer {previous_layer + 1} entry is not a tensor"
                )
            save_capture(f"decoder.layer.{previous_layer}", value)

        return hook

    hooks = [
        model.model.visual.patch_embed.register_forward_hook(
            capture_output("vision.patch_embed")
        ),
        model.model.visual.blocks[0].register_forward_pre_hook(
            capture_input("vision.position_added")
        ),
        model.model.language_model.register_forward_pre_hook(
            capture_language_inputs,
            with_kwargs=True,
        ),
    ]
    for layer in VISION_CAPTURE_LAYERS:
        hooks.append(
            model.model.visual.blocks[layer].register_forward_hook(
                capture_output(f"vision.block.{layer}")
            )
        )
    for index, merger in enumerate(model.model.visual.deepstack_merger_list):
        hooks.append(
            merger.register_forward_hook(capture_output(f"vision.deepstack.{index}"))
        )
    hooks.append(
        model.model.visual.merger.register_forward_hook(
            capture_output("vision.visual_embeddings")
        )
    )
    decoder_layers = model.model.language_model.layers
    for layer, decoder_layer in enumerate(decoder_layers):
        hooks.append(
            decoder_layer.register_forward_hook(
                capture_output(f"decoder.block.{layer}")
            )
        )
        if layer > 0:
            hooks.append(
                decoder_layer.register_forward_pre_hook(
                    capture_decoder_entry(layer - 1),
                    with_kwargs=True,
                )
            )
    hooks.append(
        model.model.language_model.norm.register_forward_pre_hook(
            capture_decoder_entry(len(decoder_layers) - 1),
            with_kwargs=True,
        )
    )
    hooks.append(
        model.model.language_model.norm.register_forward_hook(
            capture_output("decoder.final_hidden")
        )
    )
    hooks.append(model.lm_head.register_forward_hook(capture_output("logits")))

    pixel_values_compute = pixel_values_f32.to(compute_dtype).clone().contiguous()
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values_compute,
                image_grid_thw=grid_thw,
                use_cache=False,
                logits_to_keep=0,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    if outputs.past_key_values is not None:
        raise RuntimeError("cache-free prefill unexpectedly returned a KV cache")
    if not torch.equal(_clone_cpu(outputs.logits), captures["logits"]):
        raise RuntimeError("logits hook disagrees with model output")
    if outputs.rope_deltas is None:
        raise RuntimeError("multimodal prefill did not return rope_deltas")
    rope_deltas = _clone_cpu(outputs.rope_deltas.to(torch.int64))

    visual_mask = captures["visual_mask"].to(torch.bool)
    if int(visual_mask.sum().item()) != image_token_count:
        raise RuntimeError("model visual mask disagrees with expanded placeholders")
    visual_embeddings = captures["vision.visual_embeddings"]
    if not torch.equal(
        captures["decoder.input_embeddings"][visual_mask],
        visual_embeddings,
    ):
        raise RuntimeError("main visual embeddings did not replace image-token embeddings")
    for layer in range(3):
        raw = captures[f"decoder.block.{layer}"]
        post = captures[f"decoder.layer.{layer}"]
        deepstack = captures[f"vision.deepstack.{layer}"]
        if not torch.equal(raw[~visual_mask], post[~visual_mask]):
            raise RuntimeError(f"DeepStack layer {layer} changed non-visual rows")
        if not torch.equal(raw[visual_mask] + deepstack, post[visual_mask]):
            raise RuntimeError(f"DeepStack layer {layer} addition is inconsistent")

    # LifeAI's streamed safetensors reader intentionally accepts only F32 and
    # BF16 payloads.  Integer-valued oracle tensors are therefore stored as
    # exactly representable F32 values; their original semantic dtype is
    # recorded below and checked by the Julia verifier.
    integer_capture_names = ("position_ids", "attention_mask", "visual_mask")
    for name in integer_capture_names:
        captures[name] = captures[name].to(torch.float32).contiguous()
    tensors: dict[str, torch.Tensor] = {
        "raw_image_hwc_f32": torch.from_numpy(image.astype(np.float32)).contiguous(),
        "pixel_values_f32": pixel_values_f32,
        "pixel_values_compute": pixel_values_compute,
        "image_grid_thw_f32": grid_thw.to(torch.float32),
        "input_ids_f32": input_ids.to(torch.float32),
        "processor_attention_mask_f32": processor_attention_mask.to(torch.float32),
        "rope_deltas_f32": rope_deltas.to(torch.float32),
        **captures,
    }
    expected_names = {
        "raw_image_hwc_f32",
        "pixel_values_f32",
        "pixel_values_compute",
        "image_grid_thw_f32",
        "input_ids_f32",
        "processor_attention_mask_f32",
        "rope_deltas_f32",
        "vision.patch_embed",
        "vision.position_added",
        *(f"vision.block.{layer}" for layer in VISION_CAPTURE_LAYERS),
        *(f"vision.deepstack.{index}" for index in range(3)),
        "vision.visual_embeddings",
        "decoder.input_embeddings",
        *(f"decoder.block.{layer}" for layer in range(len(decoder_layers))),
        *(f"decoder.layer.{layer}" for layer in range(len(decoder_layers))),
        "decoder.final_hidden",
        "logits",
        "position_ids",
        "attention_mask",
        "visual_mask",
    }
    if set(tensors) != expected_names:
        missing = sorted(expected_names.difference(tensors))
        extra = sorted(set(tensors).difference(expected_names))
        raise RuntimeError(f"capture mismatch; missing={missing}, extra={extra}")
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"reference tensor {name} contains a non-finite value")

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "reference.safetensors"
    save_file(tensors, tensor_path)
    metadata = {
        "model_id": EXPECTED_MODEL_ID,
        "modelscope_revision": revision,
        "huggingface_revision": EXPECTED_HF_REVISION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "asset_sha256": actual_asset_hashes,
        "checkpoint_sha256": actual_asset_hashes["model.safetensors"],
        "config_sha256": actual_asset_hashes["config.json"],
        "preprocessor_config_sha256": actual_asset_hashes[
            "preprocessor_config.json"
        ],
        "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
        "attention_implementation": "eager",
        "attention_mask_contract": "explicit_all_ones",
        "capture_contract": "detach_clone_before_deepstack_inplace_mutation",
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_processor_device": str(pixel_values_f32.device),
        "compute_device": "cpu",
        "compute_dtype": compute_dtype_name,
        "prompt": PROMPT,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": _sha256_text(rendered_prompt),
        "expanded_prompt": expanded_prompt,
        "expanded_prompt_sha256": _sha256_text(expanded_prompt),
        "add_generation_prompt": True,
        "add_vision_id": False,
        "input_ids_0_based": input_ids[0].tolist(),
        "sequence_length": input_ids.shape[1],
        "image_token_id_0_based": processor.image_token_id,
        "image_token_count": image_token_count,
        "grid_thw": grid_thw.tolist(),
        "image_shape_hwc": list(image.shape),
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "decoder_layers": len(decoder_layers),
        "vision_capture_layers": list(VISION_CAPTURE_LAYERS),
        "reference_sha256": _sha256_file(tensor_path),
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "tensor_dtypes": {name: str(value.dtype) for name, value in tensors.items()},
        "semantic_dtypes": {
            "raw_image_hwc_f32": "uint8",
            "image_grid_thw_f32": "int64",
            "input_ids_f32": "int64",
            "processor_attention_mask_f32": "int64",
            "rope_deltas_f32": "int64",
            "position_ids": "int64",
            "attention_mask": "int64",
            "visual_mask": "bool",
        },
    }
    (output_dir / "reference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
