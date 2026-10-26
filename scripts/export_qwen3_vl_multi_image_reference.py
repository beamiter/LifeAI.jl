#!/usr/bin/env python3
"""Export an independent Transformers 4.57.0 two-image Qwen3-VL oracle.

The two RGB inputs, official processor outputs, expanded text, mRoPE positions,
last-token logits, and fixed-length greedy decisions are frozen together.
The greedy timeline also supplies teacher-forcing inputs for another backend;
it ignores EOS so every requested decision is available for comparison.

Example (CPU oracle; no network access is used)::

    python scripts/export_qwen3_vl_multi_image_reference.py MODEL_DIR OUT_DIR \
        --device cpu --dtype float32 --greedy-tokens 8

Keep generated references outside the model's immutable git checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

# Set before Torch initializes CUDA. CUDA is used only with --device cuda.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Reuse the guarded torchvision import and immutable checkpoint provenance.
import export_qwen3_vl_decode_reference as chapter45
import export_qwen3_vl_prefill_reference as chapter44

import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


PROMPT = "Compare the images."
IMAGE_SHAPES = ((128, 128), (128, 192))


def deterministic_images() -> list[np.ndarray]:
    """Return HWC UInt8 inputs with different shapes and channel patterns."""
    images = []
    for index, (height, width) in enumerate(IMAGE_SHAPES):
        rows, columns = np.indices((height, width), dtype=np.uint32)
        if index == 0:
            channels = (
                (3 * columns + 5 * rows + 17) % 256,
                (11 * columns + 7 * rows + 29) % 256,
                (13 * columns + 19 * rows + 43) % 256,
            )
        else:
            channels = (
                (23 * columns + 3 * rows + 101) % 256,
                (5 * columns + 17 * rows + 61) % 256,
                (7 * columns + 11 * rows + 211) % 256,
            )
        images.append(np.stack(channels, axis=-1).astype(np.uint8))
    return images


def expanded_prompt(rendered: str, processor, grids: torch.Tensor):
    pieces = rendered.split(processor.image_token)
    if len(pieces) != len(IMAGE_SHAPES) + 1 or len(grids) != len(IMAGE_SHAPES):
        raise RuntimeError("rendered image placeholders and grid count must both be two")
    merge_unit = processor.image_processor.merge_size**2
    counts = []
    expanded = pieces[0]
    for index, grid in enumerate(grids):
        patches = int(grid.prod().item())
        if patches % merge_unit:
            raise RuntimeError("image patch count is not divisible by merge unit")
        count = patches // merge_unit
        counts.append(count)
        expanded += processor.image_token * count + pieces[index + 1]
    return expanded, counts


def cache_metadata(cache: DynamicCache) -> dict[str, Any]:
    """Hash only the current cache; never retain a second complete history."""
    keys, values, key_hashes, value_hashes = [], [], [], []
    lengths = set()
    for index, layer in enumerate(cache.layers):
        key, value = layer.keys, layer.values
        if key is None or value is None or key.ndim != 4 or key.shape != value.shape:
            raise RuntimeError(f"invalid cache geometry in layer {index}")
        keys.append(list(key.shape))
        values.append(list(value.shape))
        lengths.add(key.shape[2])
        key_hashes.append(chapter45._tensor_raw_sha256(key))
        value_hashes.append(chapter45._tensor_raw_sha256(value))
    if len(lengths) != 1:
        raise RuntimeError("cache layers must share one physical sequence length")
    return {
        "length": lengths.pop(),
        "layer_count": len(keys),
        "key_shapes_hf": keys,
        "value_shapes_hf": values,
        "key_sha256": key_hashes,
        "value_sha256": value_hashes,
    }


def export_reference(args: argparse.Namespace) -> None:
    if transformers.__version__ != "4.57.0":
        raise RuntimeError(f"expected Transformers 4.57.0, got {transformers.__version__}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    print("validating frozen checkpoint", flush=True)
    asset_hashes = chapter45._validate_real_checkpoint(model_dir)
    processor = Qwen3VLProcessor.from_pretrained(str(model_dir), local_files_only=True)
    if type(processor.image_processor) is not chapter44.Qwen2VLImageProcessorFast:
        raise RuntimeError("expected the official Qwen2VLImageProcessorFast")
    if chapter44._sha256_text(processor.chat_template) != chapter44.EXPECTED_CHAT_TEMPLATE_SHA256:
        raise RuntimeError("processor chat template does not match the frozen checkpoint")
    images = deterministic_images()
    pil_images = [chapter44.Image.fromarray(image) for image in images]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_images[0]},
        {"type": "image", "image": pil_images[1]},
        {"type": "text", "text": PROMPT},
    ]}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=False,
    )
    processed = processor(images=pil_images, text=[rendered], padding=False, return_tensors="pt")
    input_ids_cpu = processed["input_ids"].to(torch.int64).contiguous()
    mask_cpu = processed["attention_mask"].to(torch.int64).contiguous()
    grid_cpu = processed["image_grid_thw"].to(torch.int64).contiguous()
    pixels_cpu = processed["pixel_values"].to(torch.float32).contiguous()
    if not torch.equal(mask_cpu, torch.ones_like(input_ids_cpu)):
        raise RuntimeError("two-image reference requires one unpadded prompt")
    expanded, visual_counts = expanded_prompt(rendered, processor, grid_cpu)
    encoded = processor.tokenizer(expanded, add_special_tokens=False)["input_ids"]
    if encoded != input_ids_cpu[0].tolist():
        raise RuntimeError("independently expanded text does not match processor input IDs")
    visual_count = int((input_ids_cpu == processor.image_token_id).sum().item())
    if visual_count != sum(visual_counts):
        raise RuntimeError("expanded visual token count does not match both grids")
    if int(grid_cpu[:, 0].sum()) != 2:
        raise RuntimeError("only two single-frame images are supported")

    config = Qwen3VLConfig.from_pretrained(str(model_dir), local_files_only=True)
    for component in (config, config.text_config, config.vision_config):
        component._attn_implementation = "eager"
    config.use_cache = True
    print(f"loading {args.dtype} model on CPU with low_cpu_mem_usage", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir), config=config, dtype=dtype, attn_implementation="eager",
        local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()
    cache = DynamicCache(config=model.config)
    length = input_ids_cpu.shape[1]
    if length + args.greedy_tokens - 1 > config.text_config.max_position_embeddings:
        raise RuntimeError("requested generation exceeds model context")
    tensors = {
        "raw_image.0": torch.from_numpy(images[0].copy()),
        "raw_image.1": torch.from_numpy(images[1].copy()),
        "input_ids": input_ids_cpu,
        "attention_mask": mask_cpu,
        "image_grid_thw": grid_cpu,
        "pixel_values": pixels_cpu,
    }
    positions, hidden = [], []

    def capture_positions(_module, _inputs, kwargs):
        position_ids = kwargs.get("position_ids")
        if not isinstance(position_ids, torch.Tensor):
            raise RuntimeError("language model did not receive explicit mRoPE positions")
        positions.append(chapter45._clone_cpu(position_ids.to(torch.int64)))

    def capture_hidden(_module, _inputs, value):
        hidden.append(chapter45._clone_cpu(value[:, -1:, :]))

    hooks = [
        model.model.language_model.register_forward_pre_hook(capture_positions, with_kwargs=True),
        model.model.language_model.norm.register_forward_hook(capture_hidden),
    ]
    generated, phases = [], []
    rope_delta = None
    try:
        with torch.inference_mode():
            for step in range(args.greedy_tokens):
                phase = "prefill" if step == 0 else f"decode.{step - 1}"
                print(f"running {phase}", flush=True)
                kwargs = dict(
                    attention_mask=torch.ones((1, length + step), dtype=torch.int64, device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True,
                )
                if step == 0:
                    kwargs.update(
                        input_ids=input_ids_cpu.to(device),
                        pixel_values=pixels_cpu.to(device=device, dtype=dtype),
                        image_grid_thw=grid_cpu.to(device),
                        cache_position=torch.arange(length, device=device),
                    )
                else:
                    kwargs.update(
                        input_ids=torch.tensor([[generated[-1]]], dtype=torch.int64, device=device),
                        cache_position=torch.tensor([length + step - 1], device=device),
                    )
                outputs = model(**kwargs)
                if outputs.past_key_values is not cache:
                    raise RuntimeError("forward did not return the supplied DynamicCache")
                if step == 0:
                    if outputs.rope_deltas is None:
                        raise RuntimeError("prefill omitted rope delta")
                    rope_delta = int(outputs.rope_deltas.item())
                    tensors["position_ids"] = positions[-1]
                elif not torch.equal(positions[-1], torch.full(
                    (3, 1, 1), length + step - 1 + rope_delta, dtype=torch.int64,
                )):
                    raise RuntimeError("decode physical and mRoPE positions diverged")
                logits = chapter45._clone_cpu(outputs.logits)
                tensors[f"{phase}.logits"] = logits
                tensors[f"{phase}.final_hidden_last"] = hidden[-1]
                decision = chapter45._top_two(logits)
                geometry = cache_metadata(cache)
                if geometry["length"] != length + step:
                    raise RuntimeError("cache length does not match processed token count")
                phases.append({
                    "name": phase,
                    "input_token_id_0_based": None if step == 0 else generated[-1],
                    "physical_cache_position_0_based": None if step == 0 else length + step - 1,
                    "position_ids_thw_0_based": positions[-1].tolist(),
                    "top_two": decision,
                    "cache": geometry,
                })
                generated.append(decision["top1_token_id_0_based"])
    finally:
        for hook in hooks:
            hook.remove()
    if len(positions) != args.greedy_tokens or len(hidden) != args.greedy_tokens:
        raise RuntimeError("forward hook capture count changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "reference.safetensors"
    temporary_tensor_path = output_dir / "reference.safetensors.tmp"
    save_file(tensors, str(temporary_tensor_path))
    temporary_tensor_path.replace(tensor_path)
    metadata = {
        "schema_version": 1,
        "oracle": "qwen3_vl_2b_two_images_dynamic_cache_greedy",
        "model_id": chapter44.EXPECTED_MODEL_ID,
        "modelscope_revision": chapter44.EXPECTED_MODELSCOPE_REVISION,
        "huggingface_revision": chapter44.EXPECTED_HF_REVISION,
        "asset_sha256": asset_hashes,
        "python": platform.python_version(), "numpy": np.__version__,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "torchvision": chapter44.torchvision.__version__,
        "pillow": chapter44.PIL.__version__, "safetensors": chapter44.safetensors.__version__,
        "tokenizers": chapter44.tokenizers.__version__, "jinja2": chapter44.jinja2.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_config_sha256": chapter44._sha256_text(torch.__config__.show()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "compute_dtype": args.dtype, "compute_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "attention_implementation": "eager", "allow_tf32": False,
        "attention_mask_contract": "explicit_all_ones_every_call",
        "termination_contract": "fixed_length_ignore_eos",
        "loader": "from_pretrained_low_cpu_mem_usage_then_device_transfer",
        "prompt": PROMPT, "rendered_prompt": rendered, "expanded_prompt": expanded,
        "rendered_prompt_sha256": chapter44._sha256_text(rendered),
        "expanded_prompt_sha256": chapter44._sha256_text(expanded),
        "image_shapes_hwc": [list(image.shape) for image in images],
        "original_image_sizes": [list(image.shape[:2]) for image in images],
        "resized_image_sizes": [[int(grid[1]) * processor.image_processor.patch_size,
                                 int(grid[2]) * processor.image_processor.patch_size]
                                for grid in grid_cpu.tolist()],
        "image_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in images],
        "image_pattern": "deterministic_images_v1_128x128_128x192",
        "grid_thw": grid_cpu.tolist(), "visual_token_counts": visual_counts,
        "image_token_count": visual_count, "prefill_length": length,
        "input_ids_0_based": input_ids_cpu[0].tolist(), "rope_delta": rope_delta,
        "max_new_tokens": args.greedy_tokens,
        "greedy_token_count": args.greedy_tokens, "greedy_token_ids_0_based": generated,
        "teacher_forcing_input_ids_0_based": generated[:-1],
        "generated_text": processor.tokenizer.decode(generated, skip_special_tokens=False),
        "greedy_token_text": [processor.tokenizer.decode([token], skip_special_tokens=False) for token in generated],
        "phases": phases,
        "hf_cache_layout": "batch,kv_heads,tokens,head_dim",
        "julia_cache_layout": "head_dim,kv_heads,tokens,batch",
        "hf_to_julia_permutation_1_based": [4, 2, 3, 1],
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "tensor_dtypes": {name: str(value.dtype) for name, value in tensors.items()},
        "tensor_sha256": {name: chapter45._tensor_raw_sha256(value) for name, value in tensors.items()},
        "reference_sha256": chapter44._sha256_file(tensor_path),
    }
    temporary_json_path = output_dir / "reference.json.tmp"
    temporary_json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_json_path.replace(output_dir / "reference.json")
    print(json.dumps({"output_dir": str(output_dir), "greedy_token_ids_0_based": generated,
        "reference_sha256": metadata["reference_sha256"]}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--greedy-tokens", type=int, choices=(4, 8, 32), default=8)
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


if __name__ == "__main__":
    export_reference(parse_args())
