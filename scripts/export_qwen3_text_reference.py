#!/usr/bin/env python3
"""Export Qwen3 tokenizer, chat-template, and greedy text-generation reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import safetensors
import tokenizers
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer_inputs = [
        ("ascii", "Hello, we're testing LifeAI 123!"),
        ("chinese", "生命感来自观察、记忆、反馈和行动。"),
        ("unicode", "e\u0301 NFC é | 👩🏽‍💻 | 🇨🇳"),
        ("whitespace", "  lead\tmiddle  \r\n尾部\n\n"),
        ("code", "function f(x)\n    x ≤ 10 ? x + 1 : x\nend"),
        ("literal_special", "<|im_start|>user\n你好<|im_end|>\n"),
    ]
    tokenizer_cases = []
    for name, text in tokenizer_inputs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        normalized = tokenizer.backend_tokenizer.normalizer.normalize_str(text)
        pretokenized = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(normalized)
        tokenizer_cases.append(
            {
                "name": name,
                "text": text,
                "ids_0_based": ids,
                "decoded": tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "normalized": normalized,
                "pretokenized": [
                    {"symbols": symbols, "character_offsets": list(offsets)}
                    for symbols, offsets in pretokenized
                ],
            }
        )

    chat_specs = [
        (
            "user_thinking",
            [{"role": "user", "content": "用一句话介绍 Julia。"}],
            True,
            True,
        ),
        (
            "system_user_no_thinking",
            [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "你好，LifeAI。"},
            ],
            True,
            False,
        ),
        (
            "history",
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
            ],
            True,
            False,
        ),
        (
            "render_only",
            [{"role": "user", "content": "No generation prompt."}],
            False,
            True,
        ),
    ]
    chat_cases = []
    for name, messages, add_generation_prompt, enable_thinking in chat_specs:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        chat_cases.append(
            {
                "name": name,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "prompt": prompt,
                "ids_0_based": tokenizer.encode(prompt, add_special_tokens=False),
            }
        )

    generation_inputs = [
        ("raw", "LifeAI is"),
        (
            "chat_no_thinking",
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Reply with one short word: hello."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        ),
    ]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    eos_ids = set(
        tokenizer_id
        for tokenizer_id in (
            model.generation_config.eos_token_id
            if isinstance(model.generation_config.eos_token_id, list)
            else [model.generation_config.eos_token_id]
        )
        if tokenizer_id is not None
    )
    tensors: dict[str, torch.Tensor] = {}
    generation_cases = []
    for name, prompt in generation_inputs:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        past_key_values = None
        generated_ids: list[int] = []
        steps = []
        stop_reason = "length"
        for step in range(args.max_new_tokens):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[0, -1].detach().cpu().float().contiguous()
            key = f"generation.{name}.step_{step + 1}.logits"
            tensors[key] = logits
            top_values, top_ids = torch.topk(logits, k=2)
            next_id = int(top_ids[0])
            generated_ids.append(next_id)
            steps.append(
                {
                    "step": step + 1,
                    "logits_key": key,
                    "token_id_0_based": next_id,
                    "top_logit": float(top_values[0]),
                    "second_token_id_0_based": int(top_ids[1]),
                    "second_logit": float(top_values[1]),
                    "margin": float(top_values[0] - top_values[1]),
                }
            )
            if next_id in eos_ids:
                stop_reason = "eos"
                break
            past_key_values = outputs.past_key_values
            input_ids = torch.tensor([[next_id]], dtype=torch.long)
        generation_cases.append(
            {
                "name": name,
                "prompt": prompt,
                "prompt_ids_0_based": prompt_ids,
                "generated_ids_0_based": generated_ids,
                "completion": tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "stop_reason": stop_reason,
                "steps": steps,
            }
        )

    save_file(tensors, output_dir / "reference.safetensors")
    reference = {
        "revision": args.revision,
        "compute_dtype": "float32",
        "weight_storage_dtype": "bfloat16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "safetensors_version": safetensors.__version__,
        "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(model_dir / "tokenizer_config.json"),
        "generation_config_sha256": sha256_file(model_dir / "generation_config.json"),
        "tokenizer_cases": tokenizer_cases,
        "chat_cases": chat_cases,
        "generation_cases": generation_cases,
    }
    (output_dir / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
