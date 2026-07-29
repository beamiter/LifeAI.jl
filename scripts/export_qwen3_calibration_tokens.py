#!/usr/bin/env python3
"""Freeze a small, evaluation-disjoint Qwen3 activation-calibration corpus."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


EVALUATION_TOKEN_IDS = [1, 9707, 13, 151643, 100, 42, 151645, 2]
CALIBRATION_TEXTS = [
    "量化模型时，校准数据应覆盖真实输入分布，但必须与最终评测样本分离。",
    "A small robot observes a hallway, updates its memory, plans a safe route, "
    "and checks the result before taking another action.",
    "Julia code example: function stable_softmax(x); y = x .- maximum(x); "
    "exp.(y) ./ sum(exp.(y)); end",
    "设矩阵 W 的形状为输出维度乘输入维度。若输入第 j 个通道的二阶矩较大，"
    "该通道的权重量化误差通常更值得保留。",
    "The experiment records provenance, checksums, tensor bytes, allocator "
    "memory, logits error, and autoregressive token agreement separately.",
    "For a quadratic approximation, minimize sum_j h_j * (w_j - q_j*s)^2 "
    "subject to q_j being a signed four-bit integer.",
    "多语言校准样本包括中文、English、代码、数学表达和系统行为描述，"
    "但不包含用于 Week 17 parity 的冻结 token 序列。",
    "Failure is evidence: a lower reconstruction loss does not guarantee that "
    "the next-token argmax remains stable under autoregressive decoding.",
    "The embodied agent receives an observation, retrieves relevant memory, "
    "chooses an action, and waits for feedback from the environment.",
    "可靠的软件实验应当可以复现、可以证伪，并明确区分算法契约、"
    "硬件驻留、性能与任务质量。",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def contains_subsequence(values, needle):
    return any(
        values[index : index + len(needle)] == needle
        for index in range(len(values) - len(needle) + 1)
    )


def main():
    args = parse_args()
    if args.sequence_length <= 0 or args.batch_size <= 0:
        raise ValueError("sequence length and batch size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    separator = tokenizer.eos_token or "\n"
    corpus = separator.join(CALIBRATION_TEXTS)
    token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    required = args.sequence_length * args.batch_size
    if len(token_ids) < required:
        raise RuntimeError(
            f"calibration corpus produced {len(token_ids)} tokens; "
            f"{required} are required"
        )
    selected = token_ids[:required]
    if contains_subsequence(selected, EVALUATION_TOKEN_IDS):
        raise RuntimeError("Week 17 evaluation token sequence leaked into calibration")

    sequences = [
        selected[start : start + args.sequence_length]
        for start in range(0, required, args.sequence_length)
    ]
    payload = {
        "source": "LifeAI Week 18 fixed multilingual/code/math calibration corpus",
        "revision": args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": __import__("transformers").__version__,
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "evaluation_token_ids_0_based": EVALUATION_TOKEN_IDS,
        "evaluation_sequence_present": False,
        "token_ids_0_based": sequences,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
