#!/usr/bin/env python3
"""Export a frozen Transformers 4.57 Qwen3-VL vision-tower oracle.

The script deliberately stops at the official preprocessed ``pixel_values``
boundary.  It does not exercise chat templates, the text decoder, or
generation.  Reference tensors are written outside the repository by the
Chapter 43 verification workflow.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


EXPECTED_TRANSFORMERS_VERSION = "4.57.0"
EXPECTED_TORCH_VERSION = "2.7.1+cpu"
EXPECTED_MODELSCOPE_REVISION = "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53"
EXPECTED_HF_REVISION = "78448d793a7eb2f7a987a1da76d464384aa1becd"
EXPECTED_CHECKPOINT_SHA256 = (
    "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
)
EXPECTED_CONFIG_SHA256 = (
    "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
)
EXPECTED_PREPROCESSOR_CONFIG_SHA256 = (
    "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
)


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


def _deterministic_image() -> np.ndarray:
    rows, columns = np.indices((256, 256), dtype=np.uint32)
    return np.stack(
        (
            (3 * columns + 5 * rows + 17) % 256,
            (11 * columns + 7 * rows + 29) % 256,
            (13 * columns + 19 * rows + 43) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _load_vision_state(model: Qwen3VLVisionModel, checkpoint: Path) -> None:
    state = {}
    prefix = "model.visual."
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith(prefix):
                state[name.removeprefix(prefix)] = handle.get_tensor(name)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"vision state mismatch: {incompatible}")


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "usage: export_qwen3_vl_vision_reference.py MODEL_DIR OUTPUT_DIR "
            "[float32|bfloat16]"
        )
    model_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    compute_dtype_name = sys.argv[3] if len(sys.argv) == 4 else "float32"
    compute_dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if compute_dtype_name not in compute_dtypes:
        raise ValueError(f"unsupported compute dtype: {compute_dtype_name}")
    compute_dtype = compute_dtypes[compute_dtype_name]
    checkpoint = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    preprocessor_config_path = model_dir / "preprocessor_config.json"
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {EXPECTED_TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"expected Torch {EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )
    revision = _git_revision(model_dir)
    if revision != EXPECTED_MODELSCOPE_REVISION:
        raise RuntimeError(
            f"expected ModelScope revision {EXPECTED_MODELSCOPE_REVISION}, "
            f"got {revision}"
        )
    dirty = _git_status(model_dir)
    if dirty:
        raise RuntimeError(f"checkpoint git tree must be clean:\n{dirty}")
    frozen_files = (
        (checkpoint, EXPECTED_CHECKPOINT_SHA256),
        (config_path, EXPECTED_CONFIG_SHA256),
        (preprocessor_config_path, EXPECTED_PREPROCESSOR_CONFIG_SHA256),
    )
    actual_hashes: dict[Path, str] = {}
    for path, expected_sha256 in frozen_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"frozen asset SHA-256 mismatch for {path.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        actual_hashes[path] = actual_sha256

    image = _deterministic_image()
    processor = Qwen2VLImageProcessor.from_pretrained(str(model_dir))
    processed = processor(images=[Image.fromarray(image)], return_tensors="pt")
    pixel_values_f32 = processed["pixel_values"].contiguous()
    grid_thw = processed["image_grid_thw"].to(torch.int64).contiguous()
    if tuple(pixel_values_f32.shape) != (256, 1536):
        raise RuntimeError(f"unexpected pixel_values shape: {pixel_values_f32.shape}")
    if grid_thw.tolist() != [[1, 16, 16]]:
        raise RuntimeError(f"unexpected image grid: {grid_thw.tolist()}")

    image_f32 = image.astype(np.float32).transpose(2, 0, 1)
    normalized_chw = (image_f32 / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)

    config = Qwen3VLConfig.from_pretrained(str(model_dir), local_files_only=True)
    config.vision_config._attn_implementation = "eager"
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(compute_dtype)
    try:
        model = Qwen3VLVisionModel(config.vision_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    _load_vision_state(model, checkpoint)
    model.eval()

    captures: dict[str, torch.Tensor] = {}

    def capture_output(name: str):
        def hook(_module, _inputs, output):
            captures[name] = output.detach().cpu().contiguous()

        return hook

    def capture_input(name: str):
        def hook(_module, inputs):
            captures[name] = inputs[0].detach().cpu().contiguous()

        return hook

    hooks = [
        model.patch_embed.register_forward_hook(capture_output("patch_embed")),
        model.blocks[0].register_forward_pre_hook(capture_input("position_added")),
    ]
    for layer in (0, 5, 11, 17, 23):
        hooks.append(
            model.blocks[layer].register_forward_hook(capture_output(f"block.{layer}"))
        )
    for index, merger in enumerate(model.deepstack_merger_list):
        hooks.append(
            merger.register_forward_hook(capture_output(f"deepstack.{index}"))
        )
    hooks.append(
        model.merger.register_forward_hook(capture_output("visual_embeddings"))
    )

    pixel_values_compute = pixel_values_f32.to(compute_dtype).clone()
    with torch.inference_mode():
        visual_embeddings, deepstack = model(
            pixel_values_compute,
            grid_thw=grid_thw,
        )
    for hook in hooks:
        hook.remove()
    if not torch.equal(
        visual_embeddings.cpu(),
        captures["visual_embeddings"],
    ):
        raise RuntimeError("main merger hook disagrees with model output")
    if len(deepstack) != 3:
        raise RuntimeError(f"expected three DeepStack features, got {len(deepstack)}")
    for index, feature in enumerate(deepstack):
        if not torch.equal(feature.cpu(), captures[f"deepstack.{index}"]):
            raise RuntimeError(f"DeepStack hook {index} disagrees with model output")

    tensors = {
        "normalized_chw_f32": torch.from_numpy(normalized_chw).contiguous(),
        "pixel_values_f32": pixel_values_f32,
        "pixel_values_compute": pixel_values_compute,
        **captures,
    }
    expected_names = {
        "normalized_chw_f32",
        "pixel_values_f32",
        "pixel_values_compute",
        "patch_embed",
        "position_added",
        "block.0",
        "block.5",
        "block.11",
        "block.17",
        "block.23",
        "deepstack.0",
        "deepstack.1",
        "deepstack.2",
        "visual_embeddings",
    }
    if set(tensors) != expected_names:
        raise RuntimeError(f"capture mismatch: {sorted(tensors)}")
    if not all(torch.isfinite(tensor.float()).all() for tensor in tensors.values()):
        raise RuntimeError("reference contains a non-finite value")

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "reference.safetensors"
    save_file(tensors, tensor_path)
    metadata = {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "modelscope_revision": revision,
        "huggingface_revision": EXPECTED_HF_REVISION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "checkpoint_sha256": actual_hashes[checkpoint],
        "config_sha256": actual_hashes[config_path],
        "preprocessor_config_sha256": actual_hashes[preprocessor_config_path],
        "attention_implementation": "eager",
        "compute_dtype": compute_dtype_name,
        "grid_thw": grid_thw.tolist(),
        "image_shape_hwc": list(image.shape),
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "reference_sha256": _sha256_file(tensor_path),
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
    }
    (output_dir / "reference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
