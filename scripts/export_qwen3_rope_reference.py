#!/usr/bin/env python3
"""Export an independent Transformers Qwen3 rotate-half RoPE reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)


DEFAULT_POSITIONS = (0, 2048, 32767, 40959)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_positions(value: str) -> list[int]:
    positions = [int(item) for item in value.split(",") if item.strip()]
    if not positions or any(position < 0 for position in positions):
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated non-negative integers"
        )
    if len(set(positions)) != len(positions):
        raise argparse.ArgumentTypeError("positions must not contain duplicates")
    return positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--positions",
        type=parse_positions,
        default=list(DEFAULT_POSITIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    max_position_embeddings = int(config.max_position_embeddings)
    if max(args.positions) >= max_position_embeddings:
        raise ValueError(
            f"position {max(args.positions)} exceeds max_position_embeddings "
            f"{max_position_embeddings}"
        )

    position_ids = torch.tensor([args.positions], dtype=torch.long)
    input_values = torch.linspace(-1.0, 1.0, head_dim, dtype=torch.float32)
    query = input_values.view(1, 1, 1, head_dim).expand(
        1, 1, len(args.positions), head_dim
    )
    rotary = Qwen3RotaryEmbedding(config)
    cos, sin = rotary(query, position_ids)
    rotated, _ = apply_rotary_pos_emb(query, query, cos, sin)

    tensor_path = output_dir / "reference.safetensors"
    save_file(
        {
            "input": query[0, 0].contiguous(),
            "cos": cos[0].contiguous(),
            "sin": sin[0].contiguous(),
            "rotated": rotated[0, 0].contiguous(),
        },
        tensor_path,
    )
    reference = {
        "revision": args.revision,
        "config_sha256": sha256_file(model_dir / "config.json"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "compute_dtype": "float32",
        "rope_type": getattr(rotary, "rope_type", "default"),
        "rope_style": "rotate_half",
        "rope_theta": float(config.rope_theta),
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
        "positions_0_based": args.positions,
        "tensor_sha256": sha256_file(tensor_path),
    }
    (output_dir / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
