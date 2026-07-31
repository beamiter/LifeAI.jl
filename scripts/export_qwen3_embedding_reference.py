#!/usr/bin/env python3
"""Export a deterministic Qwen3-Embedding-0.6B BF16 parity reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import safetensors
import tokenizers
import torch
import transformers
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
DIMENSIONS = (1024, 512, 256, 128, 64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query(text: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery:{text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output = Path(args.output).resolve()
    if args.revision != REVISION:
        raise ValueError(f"unsupported revision {args.revision!r}; expected {REVISION}")
    if not 1 <= args.max_length <= 32768:
        raise ValueError("max-length must be in 1:32768")

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    raw_queries = [
        "What is the capital of China?",
        "如何减少大语言模型增量解码时的重复计算？",
        "How do I load a HuggingFace safetensors checkpoint in Julia?",
    ]
    queries = [query(text) for text in raw_queries]
    documents = [
        "Beijing is the capital city of China.",
        "KV cache stores previous attention keys and values so incremental "
        "decoding does not recompute the full prefix.",
        "A Julia loader can parse the safetensors header, validate tensor "
        "shapes, and map HuggingFace parameter names into a Lux model.",
        "To bake bread, combine flour, water, yeast, and salt.",
        "Rain is likely tomorrow, with lower temperatures in the afternoon.",
    ]
    texts = queries + documents

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
        padding_side="left",
    )
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).eval()
    load_seconds = time.perf_counter() - load_start

    forward_start = time.perf_counter()
    outputs = model(**encoded, use_cache=False)
    forward_seconds = time.perf_counter() - forward_start
    last_hidden = outputs.last_hidden_state
    last_positions = encoded["attention_mask"].sum(dim=1) - 1
    # With left padding every sequence ends at the final batch position. Keep
    # the generic branch explicit so the fixture records the official pooling
    # rule rather than relying on that incidental batch property.
    if bool(torch.all(encoded["attention_mask"][:, -1])):
        pooled = last_hidden[:, -1]
    else:
        pooled = last_hidden[
            torch.arange(last_hidden.shape[0]),
            last_positions,
        ]

    embeddings: dict[str, list[list[float]]] = {}
    similarities: dict[str, list[list[float]]] = {}
    top_k: dict[str, list[list[int]]] = {}
    for dimension in DIMENSIONS:
        values = F.normalize(pooled[:, :dimension].float(), p=2, dim=1)
        scores = values[: len(queries)] @ values[len(queries) :].T
        embeddings[str(dimension)] = values.cpu().tolist()
        similarities[str(dimension)] = scores.cpu().tolist()
        top_k[str(dimension)] = torch.argsort(
            scores,
            dim=1,
            descending=True,
            stable=True,
        ).cpu().tolist()

    files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "modules.json",
        "config_sentence_transformers.json",
        "1_Pooling/config.json",
        "model.safetensors",
    )
    reference = {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": args.revision,
        "compute_dtype": args.dtype,
        "weight_storage_dtype": "bfloat16",
        "max_length": args.max_length,
        "instruction": INSTRUCTION,
        "raw_queries": raw_queries,
        "queries": queries,
        "documents": documents,
        "texts": texts,
        "input_ids": encoded["input_ids"].cpu().tolist(),
        "attention_mask": encoded["attention_mask"].cpu().tolist(),
        "pooled_hidden": pooled.float().cpu().tolist(),
        "embeddings": embeddings,
        "similarities": similarities,
        "top_k_document_indices_0_based": top_k,
        "expected_first_document_indices_0_based": [0, 1, 2],
        "versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
        },
        "timing": {
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "batch_size": len(texts),
            "padded_tokens": int(encoded["input_ids"].numel()),
            "valid_tokens": int(encoded["attention_mask"].sum()),
            "torch_threads": torch.get_num_threads(),
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "asset_sha256": {
            filename: sha256_file(model_dir / filename) for filename in files
        },
        "tolerances": {
            "embedding_max_abs": 0.02,
            "similarity_max_abs": 0.02,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
