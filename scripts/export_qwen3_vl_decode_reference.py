#!/usr/bin/env python3
"""Export Qwen3-VL cached-prefill and greedy-decode reference tensors.

Two deliberately separate oracle modes live in this file:

* ``tiny`` rebuilds Chapter 44's deterministic 4-layer Float32 language
  tower and writes a small, reviewable JSON fixture.  It freezes a cached
  multimodal prefill and two one-token decode calls, including every layer's
  DynamicCache key/value tensors.
* ``real`` runs the frozen official Qwen3-VL-2B-Instruct checkpoint on the
  Chapter 44 256 x 256 image and ``Describe.`` prompt.  It writes only the
  last hidden state and logits for each greedy decision plus cache geometry
  and hashes; the large real-checkpoint KV tensors are intentionally not
  serialized.  Keep this output outside the repository (normally in /tmp).

HF cache tensors use ``(batch, kv_heads, tokens, head_dim)``.  The JSON
metadata explicitly records the Julia conversion
``permutedims(hf, (4, 2, 3, 1))`` -> ``(head_dim, kv_heads, tokens, batch)``.

An explicit all-ones attention mask is mandatory.  Transformers 4.57 takes a
packed-sequence mask branch when the tiny prompt's repeated temporal mRoPE
coordinates are passed with ``attention_mask=None``; that is not the
processor-driven inference contract tested here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

# Reuse the Chapter 44 frozen environment/checkpoint contract and its guarded
# torchvision import.  The local scripts directory is on sys.path when this
# file is invoked directly.
import export_qwen3_vl_prefill_reference as chapter44

import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


TINY_SEQUENCE_LENGTH = 8
TINY_ROPE_DELTA = -2
TINY_DECODE_STEPS = 2
TINY_CHAPTER44_FINAL_HIDDEN_SHA256 = (
    "f2e5565afbb79353ff4f218b1c2d541b60533a33428bde116a9f2faf784cf154"
)
TINY_CHAPTER44_LOGITS_SHA256 = (
    "9f38d30c7a1860666b9045c29bca84f728b278472b07a121f4aae71258add0d6"
)


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    """Clone now: HF DynamicCache is mutated by every later decode call."""

    return value.detach().clone().to(device="cpu").contiguous()


def _f32_bytes(value: torch.Tensor) -> bytes:
    array = _clone_cpu(value).to(torch.float32).numpy().astype("<f4", copy=False)
    return array.tobytes(order="C")


def _f32_json_entry(value: torch.Tensor) -> dict[str, Any]:
    payload = _f32_bytes(value)
    return {
        "shape": list(value.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "f32_le_base64": base64.b64encode(payload).decode("ascii"),
    }


def _tensor_raw_sha256(value: torch.Tensor) -> str:
    tensor = _clone_cpu(value)
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _tiny_values(
    shape: tuple[int, ...],
    offset: int,
    scale: float = 0.02,
) -> torch.Tensor:
    count = math.prod(shape)
    indices = torch.arange(1, count + 1, dtype=torch.float32)
    values = torch.tensor(scale, dtype=torch.float32) * torch.sin(
        torch.tensor(0.173, dtype=torch.float32) * (offset + indices)
    )
    return values.reshape(shape)


def _tiny_config() -> Qwen3VLTextConfig:
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1.0e-6,
        rope_theta=10_000.0,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        hidden_act="silu",
        use_cache=True,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return config


def _initialize_tiny_model(config: Qwen3VLTextConfig) -> Qwen3VLTextModel:
    model = Qwen3VLTextModel(config).to(dtype=torch.float32).eval()
    with torch.no_grad():
        model.embed_tokens.weight.copy_(_tiny_values((32, 16), 10))
        for layer_index, layer in enumerate(model.layers):
            offset = 10_000 * (layer_index + 1)
            layer.input_layernorm.weight.copy_(
                1.0 + _tiny_values((16,), offset, 0.01)
            )
            layer.self_attn.q_proj.weight.copy_(
                _tiny_values((16, 16), offset + 100)
            )
            layer.self_attn.k_proj.weight.copy_(
                _tiny_values((8, 16), offset + 200)
            )
            layer.self_attn.v_proj.weight.copy_(
                _tiny_values((8, 16), offset + 300)
            )
            layer.self_attn.o_proj.weight.copy_(
                _tiny_values((16, 16), offset + 400)
            )
            layer.self_attn.q_norm.weight.copy_(
                1.0 + _tiny_values((8,), offset + 500, 0.01)
            )
            layer.self_attn.k_norm.weight.copy_(
                1.0 + _tiny_values((8,), offset + 600, 0.01)
            )
            layer.post_attention_layernorm.weight.copy_(
                1.0 + _tiny_values((16,), offset + 700, 0.01)
            )
            layer.mlp.gate_proj.weight.copy_(
                _tiny_values((32, 16), offset + 800)
            )
            layer.mlp.up_proj.weight.copy_(
                _tiny_values((32, 16), offset + 900)
            )
            layer.mlp.down_proj.weight.copy_(
                _tiny_values((16, 32), offset + 1_000)
            )
        model.norm.weight.copy_(1.0 + _tiny_values((16,), 90_000, 0.01))
    return model


def _tiny_inputs(model: Qwen3VLTextModel) -> dict[str, Any]:
    input_ids = torch.arange(TINY_SEQUENCE_LENGTH, dtype=torch.int64).reshape(1, -1)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4, 5]],
            [[0, 1, 2, 2, 3, 3, 4, 5]],
            [[0, 1, 2, 3, 2, 3, 4, 5]],
        ],
        dtype=torch.int64,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    visual_mask[:, 2:6] = True
    inputs_embeds = model.embed_tokens(input_ids).detach().clone()
    visual_embeddings = _tiny_values((4, 16), 100_000, 0.1)
    inputs_embeds[visual_mask] = visual_embeddings
    deepstack = [
        _tiny_values((4, 16), 110_000 + 1_000 * layer, 0.1)
        for layer in range(3)
    ]
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "visual_mask": visual_mask,
        "inputs_embeds": inputs_embeds,
        "visual_embeddings": visual_embeddings,
        "deepstack": deepstack,
    }


def _snapshot_cache(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snapshot: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            raise RuntimeError("DynamicCache contains an uninitialized layer")
        snapshot.append((_clone_cpu(layer.keys), _clone_cpu(layer.values)))
    return snapshot


def _cache_length(snapshot: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    if len(snapshot) == 0:
        raise RuntimeError("cache snapshot has no layers")
    lengths: set[int] = set()
    for layer_index, (key, value) in enumerate(snapshot):
        if key.shape != value.shape:
            raise RuntimeError(
                f"cache key/value shapes differ in layer {layer_index}"
            )
        if key.ndim != 4:
            raise RuntimeError(f"cache layer {layer_index} is not rank four")
        lengths.add(key.shape[2])
    if len(lengths) != 1:
        raise RuntimeError("cache layers do not have one common sequence length")
    return lengths.pop()


def _assert_cache_prefix_immutable(
    old: list[tuple[torch.Tensor, torch.Tensor]],
    new: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if len(old) != len(new):
        raise RuntimeError("decode changed the number of cache layers")
    old_length = _cache_length(old)
    if _cache_length(new) != old_length + 1:
        raise RuntimeError("decode did not append exactly one cache position")
    for layer_index, ((old_key, old_value), (new_key, new_value)) in enumerate(
        zip(old, new)
    ):
        if not torch.equal(new_key[:, :, :old_length, :], old_key):
            raise RuntimeError(f"decode mutated key prefix in layer {layer_index}")
        if not torch.equal(new_value[:, :, :old_length, :], old_value):
            raise RuntimeError(f"decode mutated value prefix in layer {layer_index}")


def _top_two(logits: torch.Tensor) -> dict[str, Any]:
    scores, indices = torch.topk(
        _clone_cpu(logits[:, -1, :]).to(torch.float32),
        k=2,
        dim=-1,
    )
    return {
        "top1_token_id_0_based": int(indices[0, 0]),
        "top1_token_id_1_based": int(indices[0, 0]) + 1,
        "top2_token_id_0_based": int(indices[0, 1]),
        "top2_token_id_1_based": int(indices[0, 1]) + 1,
        "top1_logit_f32": float(scores[0, 0]),
        "top2_logit_f32": float(scores[0, 1]),
        "margin_f32": float(scores[0, 0] - scores[0, 1]),
    }


def _add_tiny_cache_tensors(
    tensors: dict[str, dict[str, Any]],
    phase: str,
    snapshot: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for layer, (key, value) in enumerate(snapshot):
        tensors[f"cache.{phase}.layer.{layer}.key"] = _f32_json_entry(key)
        tensors[f"cache.{phase}.layer.{layer}.value"] = _f32_json_entry(value)


def export_tiny(output_path: Path) -> None:
    if platform.python_version() != chapter44.EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"expected Python {chapter44.EXPECTED_PYTHON_VERSION}, "
            f"got {platform.python_version()}"
        )
    if np.__version__ != chapter44.EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"expected NumPy {chapter44.EXPECTED_NUMPY_VERSION}, got {np.__version__}"
        )
    if transformers.__version__ != chapter44.EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {chapter44.EXPECTED_TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    if torch.__version__ != chapter44.EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"expected Torch {chapter44.EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    config = _tiny_config()
    model = _initialize_tiny_model(config)
    inputs = _tiny_inputs(model)
    cache = DynamicCache(config=config)
    tensors: dict[str, dict[str, Any]] = {}
    phases: list[dict[str, Any]] = []

    with torch.inference_mode():
        prefill = model(
            inputs_embeds=inputs["inputs_embeds"],
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(TINY_SEQUENCE_LENGTH, dtype=torch.int64),
            visual_pos_masks=inputs["visual_mask"],
            deepstack_visual_embeds=inputs["deepstack"],
            return_dict=True,
        )
        if prefill.past_key_values is not cache:
            raise RuntimeError("HF prefill did not return the supplied DynamicCache")
        prefill_hidden = _clone_cpu(prefill.last_hidden_state)
        prefill_logits = _clone_cpu(
            prefill.last_hidden_state @ model.embed_tokens.weight.T
        )
        cache_snapshots = [_snapshot_cache(cache)]

        tensors["prefill.final_hidden"] = _f32_json_entry(prefill_hidden)
        tensors["prefill.logits"] = _f32_json_entry(prefill_logits)
        _add_tiny_cache_tensors(tensors, "prefill", cache_snapshots[-1])
        phases.append(
            {
                "name": "prefill",
                "cache_length": _cache_length(cache_snapshots[-1]),
                "top_two": _top_two(prefill_logits),
            }
        )

        next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).reshape(1, 1)
        for step in range(TINY_DECODE_STEPS):
            physical_position = TINY_SEQUENCE_LENGTH + step
            rope_coordinate = physical_position + TINY_ROPE_DELTA
            position_ids = torch.full(
                (3, 1, 1),
                rope_coordinate,
                dtype=torch.int64,
            )
            attention_mask = torch.ones(
                (1, physical_position + 1),
                dtype=torch.int64,
            )
            consumed_token = int(next_token.item())
            decoded = model(
                input_ids=next_token,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([physical_position], dtype=torch.int64),
                return_dict=True,
            )
            if decoded.past_key_values is not cache:
                raise RuntimeError("HF decode did not mutate the supplied DynamicCache")
            hidden = _clone_cpu(decoded.last_hidden_state)
            logits = _clone_cpu(hidden @ model.embed_tokens.weight.T)
            snapshot = _snapshot_cache(cache)
            _assert_cache_prefix_immutable(cache_snapshots[-1], snapshot)
            cache_snapshots.append(snapshot)

            phase = f"decode.{step}"
            tensors[f"{phase}.final_hidden"] = _f32_json_entry(hidden)
            tensors[f"{phase}.logits"] = _f32_json_entry(logits)
            _add_tiny_cache_tensors(tensors, phase, snapshot)
            phases.append(
                {
                    "name": phase,
                    "input_token_id_0_based": consumed_token,
                    "input_token_id_1_based": consumed_token + 1,
                    "physical_cache_position_0_based": physical_position,
                    "mrope_position_ids_thw_0_based": [rope_coordinate] * 3,
                    "attention_mask_shape": [1, physical_position + 1],
                    "cache_length": _cache_length(snapshot),
                    "top_two": _top_two(logits),
                }
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1).reshape(1, 1)

    if tensors["prefill.final_hidden"]["sha256"] != TINY_CHAPTER44_FINAL_HIDDEN_SHA256:
        raise RuntimeError("cached prefill final hidden differs from Chapter 44")
    if tensors["prefill.logits"]["sha256"] != TINY_CHAPTER44_LOGITS_SHA256:
        raise RuntimeError("cached prefill logits differ from Chapter 44")
    expected_lengths = [TINY_SEQUENCE_LENGTH + index for index in range(3)]
    if [phase["cache_length"] for phase in phases] != expected_lengths:
        raise RuntimeError("tiny cache lengths do not follow L -> L+1 -> L+2")

    metadata = {
        "oracle": "qwen3_vl_tiny_dynamic_cache_greedy_decode",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "attention_implementation": "eager",
        "attention_mask_contract": "explicit_all_ones_every_call",
        "cache_capture_contract": "detach_clone_cpu_before_next_dynamic_cache_update",
        "hf_cache_layout": "batch,kv_heads,tokens,head_dim",
        "julia_cache_layout": "head_dim,kv_heads,tokens,batch",
        "hf_to_julia_permutation_1_based": [4, 2, 3, 1],
        "model": {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
            "tie_word_embeddings": True,
        },
        "input_ids_0_based": inputs["input_ids"][0].tolist(),
        "input_ids_1_based": (inputs["input_ids"][0] + 1).tolist(),
        "position_ids_thw_0_based": inputs["position_ids"][:, 0, :].tolist(),
        "visual_positions_1_based": [3, 4, 5, 6],
        "rope_delta": TINY_ROPE_DELTA,
        "prefill_length": TINY_SEQUENCE_LENGTH,
        "decode_forward_calls": TINY_DECODE_STEPS,
        "cache_lengths": [phase["cache_length"] for phase in phases],
        "phases": phases,
        "greedy_token_ids_0_based": [
            phase["top_two"]["top1_token_id_0_based"] for phase in phases
        ],
        "greedy_token_ids_1_based": [
            phase["top_two"]["top1_token_id_1_based"] for phase in phases
        ],
        "chapter44_prefill_sha256": {
            "final_hidden": TINY_CHAPTER44_FINAL_HIDDEN_SHA256,
            "logits": TINY_CHAPTER44_LOGITS_SHA256,
        },
    }
    payload = {"metadata": metadata, "tensors": tensors}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _validate_real_checkpoint(model_dir: Path) -> dict[str, str]:
    if transformers.__version__ != chapter44.EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("real decode requires the frozen Chapter 44 Transformers")
    if torch.__version__ != chapter44.EXPECTED_TORCH_VERSION:
        raise RuntimeError("real decode requires the frozen Chapter 44 Torch build")
    revision = chapter44._git_revision(model_dir)
    if revision != chapter44.EXPECTED_MODELSCOPE_REVISION:
        raise RuntimeError(
            f"expected checkpoint revision {chapter44.EXPECTED_MODELSCOPE_REVISION}, "
            f"got {revision}"
        )
    dirty = chapter44._git_status(model_dir)
    if dirty:
        raise RuntimeError(f"checkpoint git tree must be clean:\n{dirty}")
    hashes: dict[str, str] = {}
    for filename, expected in chapter44.EXPECTED_ASSET_SHA256.items():
        actual = chapter44._sha256_file(model_dir / filename)
        if actual != expected:
            raise RuntimeError(f"frozen checkpoint asset mismatch: {filename}")
        hashes[filename] = actual
    return hashes


def _validate_real_environment() -> None:
    """Apply Chapter 44's complete CPU-oracle environment contract."""

    checks = (
        ("Python", platform.python_version(), chapter44.EXPECTED_PYTHON_VERSION),
        ("NumPy", np.__version__, chapter44.EXPECTED_NUMPY_VERSION),
        ("Pillow", chapter44.PIL.__version__, chapter44.EXPECTED_PILLOW_VERSION),
        (
            "safetensors",
            chapter44.safetensors.__version__,
            chapter44.EXPECTED_SAFETENSORS_VERSION,
        ),
        (
            "tokenizers",
            chapter44.tokenizers.__version__,
            chapter44.EXPECTED_TOKENIZERS_VERSION,
        ),
        (
            "Jinja2",
            chapter44.jinja2.__version__,
            chapter44.EXPECTED_JINJA2_VERSION,
        ),
        (
            "Transformers",
            transformers.__version__,
            chapter44.EXPECTED_TRANSFORMERS_VERSION,
        ),
        ("Torch", torch.__version__, chapter44.EXPECTED_TORCH_VERSION),
        (
            "torchvision",
            chapter44.torchvision.__version__,
            chapter44.EXPECTED_TORCHVISION_VERSION,
        ),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise RuntimeError(f"expected {label} {expected}, got {actual}")
    if torch.version.git_version != chapter44.EXPECTED_TORCH_GIT_VERSION:
        raise RuntimeError("Torch build git revision is not frozen")
    torch_config_sha256 = hashlib.sha256(
        torch.__config__.show().encode("utf-8")
    ).hexdigest()
    if torch_config_sha256 != chapter44.EXPECTED_TORCH_CONFIG_SHA256:
        raise RuntimeError("Torch build configuration is not frozen")
    if torch.get_num_threads() != chapter44.EXPECTED_TORCH_NUM_THREADS:
        raise RuntimeError("Torch intra-op thread count is not frozen")
    if torch.get_num_interop_threads() != chapter44.EXPECTED_TORCH_INTEROP_THREADS:
        raise RuntimeError("Torch inter-op thread count is not frozen")
    if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled:
        raise RuntimeError("the frozen oracle requires enabled oneDNN/MKLDNN")
    if torch.backends.mkldnn.deterministic:
        raise RuntimeError("the frozen oracle requires MKLDNN deterministic=false")
    if not torch.backends.mkl.is_available() or not torch.backends.openmp.is_available():
        raise RuntimeError("the frozen oracle requires MKL and OpenMP")
    if torch.backends.cpu.get_cpu_capability() != chapter44.EXPECTED_CPU_CAPABILITY:
        raise RuntimeError("the frozen oracle CPU capability is not AVX2")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("the frozen oracle requires highest Float32 matmul precision")


def _real_cache_metadata(
    snapshot: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "length": _cache_length(snapshot),
        "layer_count": len(snapshot),
        "key_shapes_hf": [list(key.shape) for key, _ in snapshot],
        "value_shapes_hf": [list(value.shape) for _, value in snapshot],
        "key_sha256": [_tensor_raw_sha256(key) for key, _ in snapshot],
        "value_sha256": [_tensor_raw_sha256(value) for _, value in snapshot],
    }


def export_real(
    model_dir: Path,
    output_dir: Path,
    *,
    dtype_name: str,
    device_name: str,
    greedy_tokens: int,
) -> None:
    if greedy_tokens not in range(2, 5):
        raise ValueError("real oracle freezes between 2 and 4 greedy tokens")
    torch.set_num_threads(chapter44.EXPECTED_TORCH_NUM_THREADS)
    torch.set_num_interop_threads(chapter44.EXPECTED_TORCH_INTEROP_THREADS)
    _validate_real_environment()
    asset_hashes = _validate_real_checkpoint(model_dir)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    compute_dtype = dtypes[dtype_name]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA real-decode export requested but CUDA is unavailable")

    processor = Qwen3VLProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    if type(processor.image_processor) is not chapter44.Qwen2VLImageProcessorFast:
        raise RuntimeError("real decode requires Qwen2VLImageProcessorFast")
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
    if not torch.equal(attention_mask_cpu, torch.ones_like(attention_mask_cpu)):
        raise RuntimeError("the frozen real prompt must be unpadded and all ones")
    grid_cpu = processed["image_grid_thw"].to(torch.int64).contiguous()
    pixels_cpu = processed["pixel_values"].to(torch.float32).contiguous()
    if tuple(image_array.shape) != (256, 256, 3):
        raise RuntimeError("real oracle image shape changed")
    if tuple(pixels_cpu.shape) != (256, 1536):
        raise RuntimeError("real oracle processor geometry changed")
    if grid_cpu.tolist() != [[1, 16, 16]]:
        raise RuntimeError("real oracle image grid changed")

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
    cache = DynamicCache(config=model.config)
    input_ids = input_ids_cpu.to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64, device=device)
    grid = grid_cpu.to(device)
    pixels = pixels_cpu.to(device=device, dtype=compute_dtype)
    prefill_length = input_ids.shape[1]
    image_token_count = int((input_ids_cpu == processor.image_token_id).sum().item())
    if prefill_length != 76 or image_token_count != 64:
        raise RuntimeError(
            "the frozen 256 x 256 Describe prompt must have 76 total and "
            "64 visual tokens"
        )

    hidden_captures: list[torch.Tensor] = []
    position_captures: list[torch.Tensor] = []

    def capture_hidden(_module, _inputs, output):
        hidden_captures.append(_clone_cpu(output[:, -1:, :]))

    def capture_positions(_module, args, kwargs):
        if args:
            raise RuntimeError("language model unexpectedly received positional args")
        position_ids = kwargs.get("position_ids")
        if not isinstance(position_ids, torch.Tensor):
            raise RuntimeError("language model did not receive position_ids")
        position_captures.append(_clone_cpu(position_ids.to(torch.int64)))

    hooks = [
        model.model.language_model.norm.register_forward_hook(capture_hidden),
        model.model.language_model.register_forward_pre_hook(
            capture_positions,
            with_kwargs=True,
        ),
    ]
    phase_logits: list[torch.Tensor] = []
    cache_snapshots: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    phases: list[dict[str, Any]] = []
    generated: list[int] = []
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixels,
                image_grid_thw=grid,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(prefill_length, device=device),
                logits_to_keep=1,
                return_dict=True,
            )
            if outputs.past_key_values is not cache:
                raise RuntimeError("real prefill did not return supplied DynamicCache")
            if outputs.rope_deltas is None:
                raise RuntimeError("real multimodal prefill omitted rope delta")
            rope_delta = int(_clone_cpu(outputs.rope_deltas).item())
            if rope_delta != -56:
                raise RuntimeError("the frozen real prompt rope delta changed")
            logits = _clone_cpu(outputs.logits)
            phase_logits.append(logits)
            cache_snapshots.append(_snapshot_cache(cache))
            decision = _top_two(logits)
            generated.append(decision["top1_token_id_0_based"])
            phases.append(
                {
                    "name": "prefill",
                    "cache_length": _cache_length(cache_snapshots[-1]),
                    "top_two": decision,
                }
            )

            next_token = torch.tensor([[generated[-1]]], dtype=torch.int64, device=device)
            for step in range(greedy_tokens - 1):
                physical_position = prefill_length + step
                full_mask = torch.ones(
                    (1, physical_position + 1),
                    dtype=torch.int64,
                    device=device,
                )
                consumed = int(next_token.item())
                decoded = model(
                    input_ids=next_token,
                    attention_mask=full_mask,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([physical_position], device=device),
                    logits_to_keep=1,
                    return_dict=True,
                )
                if decoded.past_key_values is not cache:
                    raise RuntimeError("real decode did not mutate supplied DynamicCache")
                logits = _clone_cpu(decoded.logits)
                phase_logits.append(logits)
                snapshot = _snapshot_cache(cache)
                _assert_cache_prefix_immutable(cache_snapshots[-1], snapshot)
                cache_snapshots.append(snapshot)
                decision = _top_two(logits)
                generated.append(decision["top1_token_id_0_based"])
                phases.append(
                    {
                        "name": f"decode.{step}",
                        "input_token_id_0_based": consumed,
                        "physical_cache_position_0_based": physical_position,
                        "mrope_position_ids_thw_0_based": [
                            physical_position + rope_delta
                        ]
                        * 3,
                        "attention_mask_shape": [1, physical_position + 1],
                        "cache_length": _cache_length(snapshot),
                        "top_two": decision,
                    }
                )
                next_token = torch.tensor(
                    [[generated[-1]]],
                    dtype=torch.int64,
                    device=device,
                )
    finally:
        for hook in hooks:
            hook.remove()

    if len(hidden_captures) != greedy_tokens:
        raise RuntimeError("real final-hidden hook count is inconsistent")
    if len(position_captures) != greedy_tokens:
        raise RuntimeError("real position-id hook count is inconsistent")
    if position_captures[0].shape != (3, 1, prefill_length):
        raise RuntimeError("real prefill mRoPE position shape changed")
    for step, positions in enumerate(position_captures[1:]):
        expected = prefill_length + step + rope_delta
        if positions.shape != (3, 1, 1) or not torch.equal(
            positions,
            torch.full((3, 1, 1), expected, dtype=torch.int64),
        ):
            raise RuntimeError(f"real decode {step} mRoPE coordinate changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_payload: dict[str, torch.Tensor] = {}
    for index, (hidden, logits) in enumerate(zip(hidden_captures, phase_logits)):
        phase = "prefill" if index == 0 else f"decode.{index - 1}"
        tensor_payload[f"{phase}.final_hidden_last"] = hidden
        tensor_payload[f"{phase}.logits"] = logits
    tensor_path = output_dir / "reference.safetensors"
    save_file(tensor_payload, tensor_path)
    cache_metadata = [_real_cache_metadata(value) for value in cache_snapshots]
    metadata = {
        "oracle": "qwen3_vl_2b_256_describe_dynamic_cache_greedy_decode",
        "model_id": chapter44.EXPECTED_MODEL_ID,
        "modelscope_revision": chapter44.EXPECTED_MODELSCOPE_REVISION,
        "huggingface_revision": chapter44.EXPECTED_HF_REVISION,
        "asset_sha256": asset_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "compute_dtype": dtype_name,
        "compute_device": str(device),
        "attention_implementation": "eager",
        "attention_mask_contract": "explicit_all_ones_every_call",
        "cache_capture_contract": "detach_clone_cpu_before_next_dynamic_cache_update",
        "real_kv_storage_contract": "metadata_shapes_and_hashes_only",
        "hf_cache_layout": "batch,kv_heads,tokens,head_dim",
        "julia_cache_layout": "head_dim,kv_heads,tokens,batch",
        "hf_to_julia_permutation_1_based": [4, 2, 3, 1],
        "prompt": chapter44.PROMPT,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": chapter44._sha256_text(rendered_prompt),
        "image_shape_hwc": list(image_array.shape),
        "image_sha256": hashlib.sha256(image_array.tobytes()).hexdigest(),
        "grid_thw": grid_cpu.tolist(),
        "input_ids_0_based": input_ids_cpu[0].tolist(),
        "prefill_length": prefill_length,
        "image_token_count": image_token_count,
        "rope_delta": rope_delta,
        "greedy_token_count": greedy_tokens,
        "greedy_token_ids_0_based": generated,
        "greedy_token_text": [
            processor.tokenizer.decode([token], skip_special_tokens=False)
            for token in generated
        ],
        "generated_text": processor.tokenizer.decode(
            generated,
            skip_special_tokens=False,
        ),
        "phases": phases,
        "cache": cache_metadata,
        "cache_prefix_immutable": True,
        "tensor_shapes": {
            name: list(value.shape) for name, value in tensor_payload.items()
        },
        "tensor_dtypes": {
            name: str(value.dtype) for name, value in tensor_payload.items()
        },
        "tensor_sha256": {
            name: _tensor_raw_sha256(value) for name, value in tensor_payload.items()
        },
        "reference_sha256": chapter44._sha256_file(tensor_path),
    }
    (output_dir / "reference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    tiny = subparsers.add_parser("tiny", help="export the committed tiny JSON fixture")
    tiny.add_argument("output_json", type=Path)

    real = subparsers.add_parser("real", help="export the official 2B decode oracle")
    real.add_argument("model_dir", type=Path)
    real.add_argument("output_dir", type=Path)
    real.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    real.add_argument("--device", default="cpu")
    real.add_argument("--greedy-tokens", type=int, choices=range(2, 5), default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "tiny":
        export_tiny(args.output_json.resolve())
    else:
        export_real(
            args.model_dir.resolve(),
            args.output_dir.resolve(),
            dtype_name=args.dtype,
            device_name=args.device,
            greedy_tokens=args.greedy_tokens,
        )


if __name__ == "__main__":
    main()
