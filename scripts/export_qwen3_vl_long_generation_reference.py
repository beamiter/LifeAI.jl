#!/usr/bin/env python3
"""Export a frozen Hugging Face Qwen3-VL long greedy token oracle.

This exporter reuses Chapter 44/45's immutable checkpoint, processor, and
Python environment contract.  Unlike the short Chapter 45 tensor reference,
it streams a DynamicCache for exactly 256 generated tokens without retaining a
full K/V clone after every step.  The output is a compact JSON token/top-2
timeline intended to gate Chapter 47's independent long-sequence correctness;
it is not a cross-backend tensor-parity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path
from typing import Any

import export_qwen3_vl_decode_reference as chapter45
import export_qwen3_vl_prefill_reference as chapter44

import numpy as np
import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


SCHEMA_VERSION = 1
ORACLE = "qwen3_vl_2b_256_describe_hf_dynamic_cache_long_greedy"
DEFAULT_GREEDY_TOKENS = 256
DEFAULT_CHECKPOINT_LENGTHS = (32, 128, 256)
CLAIM = "float32_cpu_greedy_token_timeline_only"
TERMINATION_CONTRACT = "fixed_length_ignore_eos"
CHAPTER45_REFERENCE_SAFETENSORS_SHA256 = (
    "a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81"
)
CHAPTER45_REFERENCE_METADATA_SHA256 = (
    "569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timeline_sha256(token_ids_0_based: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids_0_based:
        if token_id < 0 or token_id > 0xFFFFFFFF:
            raise RuntimeError("generated token id is outside UInt32")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def _parse_checkpoint_lengths(raw: str) -> tuple[int, ...]:
    try:
        parsed = [int(item.strip()) for item in raw.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "checkpoint lengths must be comma-separated integers"
        ) from error
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("checkpoint lengths must not contain duplicates")
    values = tuple(sorted(parsed))
    if not values or values[0] < 4:
        raise argparse.ArgumentTypeError(
            "checkpoint lengths must contain values of at least four"
        )
    return values


def _cache_geometry(cache: DynamicCache) -> dict[str, Any]:
    lengths: set[int] = set()
    key_shapes: list[list[int]] = []
    value_shapes: list[list[int]] = []
    for layer_index, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            raise RuntimeError(f"cache layer {layer_index} is uninitialized")
        key_shape = list(layer.keys.shape)
        value_shape = list(layer.values.shape)
        if len(key_shape) != 4 or key_shape != value_shape:
            raise RuntimeError(f"cache layer {layer_index} geometry changed")
        lengths.add(key_shape[2])
        key_shapes.append(key_shape)
        value_shapes.append(value_shape)
    if len(lengths) != 1:
        raise RuntimeError("cache layers do not share one sequence length")
    return {
        "length": lengths.pop(),
        "layer_count": len(cache.layers),
        "key_shapes_hf": key_shapes,
        "value_shapes_hf": value_shapes,
    }


def _validate_position_capture(
    position_ids: torch.Tensor,
    *,
    generated_step: int,
    prompt_tokens: int,
    rope_delta: int,
) -> list[int]:
    if generated_step == 1:
        if tuple(position_ids.shape) != (3, 1, prompt_tokens):
            raise RuntimeError("prefill mRoPE position shape changed")
        return []
    physical_position = prompt_tokens + generated_step - 2
    expected = physical_position + rope_delta
    expected_tensor = torch.full((3, 1, 1), expected, dtype=torch.int64)
    if position_ids.shape != expected_tensor.shape or not torch.equal(
        position_ids, expected_tensor
    ):
        raise RuntimeError(
            f"decode mRoPE coordinate changed at generated step {generated_step}"
        )
    return [expected, expected, expected]


def _decision(
    logits: torch.Tensor,
    *,
    generated_step: int,
    input_token_id_0_based: int | None,
    prompt_tokens: int,
    rope_delta: int,
    position_ids: torch.Tensor,
    cache_geometry: dict[str, Any],
) -> dict[str, Any]:
    expected_cache_length = prompt_tokens + generated_step - 1
    if cache_geometry["length"] != expected_cache_length:
        raise RuntimeError(
            f"cache length changed at generated step {generated_step}"
        )
    top_two = chapter45._top_two(logits)
    top1_id = top_two["top1_token_id_0_based"]
    top2_id = top_two["top2_token_id_0_based"]
    top1 = np.float32(top_two["top1_logit_f32"])
    top2 = np.float32(top_two["top2_logit_f32"])
    margin = np.float32(top_two["margin_f32"])
    if top1_id == top2_id:
        raise RuntimeError("long oracle top-1 and top-2 token ids coincide")
    if not np.isfinite([top1, top2, margin]).all():
        raise RuntimeError("long oracle top-2 metrics are non-finite")
    if not top1 > top2 or not margin > 0:
        raise RuntimeError("long oracle top-2 margin is not strictly positive")
    if margin != np.float32(top1 - top2):
        raise RuntimeError("long oracle top-2 margin is inconsistent")
    physical_position = (
        None if generated_step == 1 else prompt_tokens + generated_step - 2
    )
    positions = _validate_position_capture(
        position_ids,
        generated_step=generated_step,
        prompt_tokens=prompt_tokens,
        rope_delta=rope_delta,
    )
    return {
        "generated_step": generated_step,
        "phase": "prefill" if generated_step == 1 else f"decode.{generated_step - 2}",
        "input_token_id_0_based": input_token_id_0_based,
        "physical_cache_position_0_based": physical_position,
        "mrope_position_ids_thw_0_based": positions,
        "attention_mask_shape": [1, expected_cache_length],
        "cache_length": expected_cache_length,
        "top_two": top_two,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_raw_sha256": chapter45._tensor_raw_sha256(logits),
    }


def export_long_reference(
    model_dir: Path,
    output_json: Path,
    *,
    greedy_tokens: int,
    checkpoint_lengths: tuple[int, ...],
) -> None:
    if greedy_tokens != DEFAULT_GREEDY_TOKENS:
        raise ValueError(
            f"long oracle requires exactly {DEFAULT_GREEDY_TOKENS} greedy tokens"
        )
    if checkpoint_lengths != DEFAULT_CHECKPOINT_LENGTHS:
        raise ValueError(
            "long oracle checkpoint lengths must be exactly "
            f"{DEFAULT_CHECKPOINT_LENGTHS}"
        )

    source_dir = Path(__file__).resolve().parent
    source_paths = {
        "export_qwen3_vl_long_generation_reference.py": Path(__file__).resolve(),
        "export_qwen3_vl_decode_reference.py": Path(chapter45.__file__).resolve(),
        "export_qwen3_vl_prefill_reference.py": Path(chapter44.__file__).resolve(),
    }
    expected_source_paths = {
        name: source_dir / name for name in source_paths
    }
    if source_paths != expected_source_paths:
        raise RuntimeError("long oracle imported exporter helpers from another tree")
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }

    torch.set_num_threads(chapter44.EXPECTED_TORCH_NUM_THREADS)
    torch.set_num_interop_threads(chapter44.EXPECTED_TORCH_INTEROP_THREADS)
    chapter45._validate_real_environment()
    asset_hashes = chapter45._validate_real_checkpoint(model_dir)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if torch.initial_seed() != 0:
        raise RuntimeError("long oracle Torch seed was not frozen to zero")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("long oracle deterministic algorithms are not enabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("long oracle deterministic algorithms are warn-only")

    compute_dtype = torch.float32
    device = torch.device("cpu")

    processor = Qwen3VLProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    if type(processor.image_processor) is not chapter44.Qwen2VLImageProcessorFast:
        raise RuntimeError("long oracle requires Qwen2VLImageProcessorFast")
    image_array = chapter44._deterministic_image()
    image = chapter44.Image.fromarray(image_array)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": chapter44.PROMPT},
            ],
        }
    ]
    rendered_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=False,
    )
    processed = processor(
        images=[image],
        text=[rendered_prompt],
        padding=False,
        return_tensors="pt",
    )
    input_ids_cpu = processed["input_ids"].to(torch.int64).contiguous()
    attention_mask_cpu = processed["attention_mask"].to(torch.int64).contiguous()
    grid_cpu = processed["image_grid_thw"].to(torch.int64).contiguous()
    pixels_cpu = processed["pixel_values"].to(torch.float32).contiguous()
    if not torch.equal(attention_mask_cpu, torch.ones_like(attention_mask_cpu)):
        raise RuntimeError("the long oracle prompt must be unpadded and all ones")
    if tuple(image_array.shape) != (256, 256, 3):
        raise RuntimeError("long oracle image geometry changed")
    if tuple(pixels_cpu.shape) != (256, 1536):
        raise RuntimeError("long oracle processor pixels changed")
    if grid_cpu.tolist() != [[1, 16, 16]]:
        raise RuntimeError("long oracle image grid changed")

    config = Qwen3VLConfig.from_pretrained(str(model_dir), local_files_only=True)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    config.use_cache = True
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        config=config,
        dtype=compute_dtype,
        attn_implementation="eager",
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    model.eval()
    model.config.use_cache = True

    input_ids = input_ids_cpu.to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64, device=device)
    grid = grid_cpu.to(device)
    pixels = pixels_cpu.to(device=device, dtype=compute_dtype)
    prompt_tokens = int(input_ids.shape[1])
    image_token_count = int((input_ids_cpu == processor.image_token_id).sum().item())
    if prompt_tokens != 76 or image_token_count != 64:
        raise RuntimeError("long oracle prompt must contain 76 total and 64 image tokens")
    if prompt_tokens + greedy_tokens - 1 > config.text_config.max_position_embeddings:
        raise RuntimeError("long oracle exceeds the checkpoint context limit")

    cache = DynamicCache(config=model.config)
    position_captures: list[torch.Tensor] = []

    def capture_positions(_module, args, kwargs):
        if args:
            raise RuntimeError("language model unexpectedly received positional args")
        position_ids = kwargs.get("position_ids")
        if not isinstance(position_ids, torch.Tensor):
            raise RuntimeError("language model did not receive position_ids")
        position_captures.append(
            chapter45._clone_cpu(position_ids.to(torch.int64))
        )

    hook = model.model.language_model.register_forward_pre_hook(
        capture_positions,
        with_kwargs=True,
    )
    decisions: list[dict[str, Any]] = []
    generated: list[int] = []
    checkpoint_cache_geometry: dict[str, Any] = {}
    rope_delta: int | None = None
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixels,
                image_grid_thw=grid,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(prompt_tokens, device=device),
                logits_to_keep=1,
                return_dict=True,
            )
            if outputs.past_key_values is not cache:
                raise RuntimeError("long oracle prefill replaced DynamicCache")
            if outputs.rope_deltas is None:
                raise RuntimeError("long oracle prefill omitted rope delta")
            rope_delta = int(chapter45._clone_cpu(outputs.rope_deltas).item())
            if rope_delta != -56:
                raise RuntimeError("long oracle rope delta changed")
            if len(position_captures) != 1:
                raise RuntimeError("long oracle prefill position hook count changed")
            geometry = _cache_geometry(cache)
            decision = _decision(
                outputs.logits,
                generated_step=1,
                input_token_id_0_based=None,
                prompt_tokens=prompt_tokens,
                rope_delta=rope_delta,
                position_ids=position_captures[-1],
                cache_geometry=geometry,
            )
            decisions.append(decision)
            generated.append(decision["top_two"]["top1_token_id_0_based"])
            next_token = torch.tensor(
                [[generated[-1]]], dtype=torch.int64, device=device
            )

            for generated_step in range(2, greedy_tokens + 1):
                physical_position = prompt_tokens + generated_step - 2
                full_mask = torch.ones(
                    (1, physical_position + 1),
                    dtype=torch.int64,
                    device=device,
                )
                consumed = int(next_token.item())
                outputs = model(
                    input_ids=next_token,
                    attention_mask=full_mask,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([physical_position], device=device),
                    logits_to_keep=1,
                    return_dict=True,
                )
                if outputs.past_key_values is not cache:
                    raise RuntimeError("long oracle decode replaced DynamicCache")
                if len(position_captures) != generated_step:
                    raise RuntimeError("long oracle decode position hook count changed")
                geometry = _cache_geometry(cache)
                decision = _decision(
                    outputs.logits,
                    generated_step=generated_step,
                    input_token_id_0_based=consumed,
                    prompt_tokens=prompt_tokens,
                    rope_delta=rope_delta,
                    position_ids=position_captures[-1],
                    cache_geometry=geometry,
                )
                decisions.append(decision)
                generated.append(decision["top_two"]["top1_token_id_0_based"])
                if generated_step in checkpoint_lengths:
                    checkpoint_cache_geometry[str(generated_step)] = geometry
                next_token = torch.tensor(
                    [[generated[-1]]], dtype=torch.int64, device=device
                )
    finally:
        hook.remove()

    if rope_delta is None:
        raise RuntimeError("long oracle did not initialize rope delta")
    if len(generated) != greedy_tokens or len(decisions) != greedy_tokens:
        raise RuntimeError("long oracle token timeline is incomplete")
    if generated[:4] != [1986, 2168, 374, 264]:
        raise RuntimeError("long oracle changed the frozen Chapter 45 four-token prefix")

    prefix_hashes = {
        str(length): _timeline_sha256(generated[:length])
        for length in checkpoint_lengths
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "oracle": ORACLE,
        "claim": CLAIM,
        "model_id": chapter44.EXPECTED_MODEL_ID,
        "modelscope_revision": chapter44.EXPECTED_MODELSCOPE_REVISION,
        "huggingface_revision": chapter44.EXPECTED_HF_REVISION,
        "asset_sha256": asset_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": chapter44.PIL.__version__,
        "safetensors": chapter44.safetensors.__version__,
        "tokenizers": chapter44.tokenizers.__version__,
        "jinja2": chapter44.jinja2.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torchvision": chapter44.torchvision.__version__,
        "torch_git_revision": torch.version.git_version,
        "torch_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_seed": torch.initial_seed(),
        "deterministic_algorithms":
            torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only":
            torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
        "mkl_available": torch.backends.mkl.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
        "compute_dtype": "float32",
        "compute_device": str(device),
        "compute_device_type": device.type,
        "cuda_device": "",
        "attention_implementation": "eager",
        "attention_mask_contract": "explicit_all_ones_every_call",
        "cache_contract": "streaming_hf_dynamic_cache_geometry_no_kv_snapshots",
        "greedy": True,
        "termination_contract": TERMINATION_CONTRACT,
        "stop_token_ids_0_based": [],
        "prompt": chapter44.PROMPT,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": chapter44._sha256_text(rendered_prompt),
        "image_shape_hwc": list(image_array.shape),
        "image_sha256": hashlib.sha256(image_array.tobytes()).hexdigest(),
        "grid_thw": grid_cpu.tolist(),
        "input_ids_0_based": input_ids_cpu[0].tolist(),
        "input_ids_u32le_sha256": _timeline_sha256(input_ids_cpu[0].tolist()),
        "prompt_tokens": prompt_tokens,
        "image_token_count": image_token_count,
        "rope_delta": rope_delta,
        "generated_token_count": greedy_tokens,
        "generated_token_ids_0_based": generated,
        "token_timeline_u32le_sha256": _timeline_sha256(generated),
        "checkpoint_prefix_u32le_sha256": prefix_hashes,
        "checkpoint_cache_geometry": checkpoint_cache_geometry,
        "decisions": decisions,
        "generated_token_text": [
            processor.tokenizer.decode([token], skip_special_tokens=False)
            for token in generated
        ],
        "generated_text": processor.tokenizer.decode(
            generated,
            skip_special_tokens=False,
        ),
        "chapter45_reference_sha256": {
            "reference.safetensors": CHAPTER45_REFERENCE_SAFETENSORS_SHA256,
            "reference.json": CHAPTER45_REFERENCE_METADATA_SHA256,
        },
        "source_sha256": source_hashes,
    }
    final_source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    if final_source_hashes != source_hashes:
        raise RuntimeError("long oracle exporter sources changed during the run")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "reference_sha256": _sha256_file(output_json),
                "generated_token_count": greedy_tokens,
                "token_timeline_u32le_sha256": metadata[
                    "token_timeline_u32le_sha256"
                ],
                "checkpoint_prefix_u32le_sha256": prefix_hashes,
                "generated_text": metadata["generated_text"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--greedy-tokens",
        type=int,
        choices=(DEFAULT_GREEDY_TOKENS,),
        default=DEFAULT_GREEDY_TOKENS,
    )
    parser.add_argument(
        "--checkpoint-lengths",
        type=_parse_checkpoint_lengths,
        default=DEFAULT_CHECKPOINT_LENGTHS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_long_reference(
        args.model_dir.resolve(),
        args.output_json.resolve(),
        greedy_tokens=args.greedy_tokens,
        checkpoint_lengths=args.checkpoint_lengths,
    )


if __name__ == "__main__":
    main()
