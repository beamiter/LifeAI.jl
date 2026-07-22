#!/usr/bin/env python3
"""Export deterministic Qwen3 temperature/top-k/top-p sampling reference data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


DEFAULT_UNIFORMS = (
    0.13,
    0.73,
    0.42,
    0.91,
    0.27,
    0.58,
    0.84,
    0.06,
    0.67,
    0.35,
    0.96,
    0.18,
    0.51,
    0.79,
    0.24,
    0.62,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_uniforms(value: str) -> list[float]:
    uniforms = [float(item) for item in value.split(",") if item.strip()]
    if not uniforms or any(not 0 <= item < 1 for item in uniforms):
        raise argparse.ArgumentTypeError("uniforms must be comma-separated values in [0, 1)")
    return uniforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--uniforms",
        type=parse_uniforms,
        default=list(DEFAULT_UNIFORMS),
        help="comma-separated categorical CDF thresholds",
    )
    parser.add_argument("--prompt", default="What is 2 + 2? Explain briefly.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
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
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    config = model.generation_config
    if not config.do_sample:
        raise ValueError("the target generation_config.json does not enable sampling")
    temperature = float(config.temperature)
    top_k = int(config.top_k)
    top_p = float(config.top_p)
    warpers = (
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k),
        TopPLogitsWarper(top_p),
    )
    eos_ids = {
        int(token_id)
        for token_id in (
            config.eos_token_id
            if isinstance(config.eos_token_id, list)
            else [config.eos_token_id]
        )
        if token_id is not None
    }

    tensors: dict[str, torch.Tensor] = {}
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    context_ids = input_ids.clone()
    past_key_values = None
    generated_ids: list[int] = []
    steps = []
    stop_reason = "length"
    for step_index, uniform in enumerate(args.uniforms, start=1):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :].detach().cpu().float().contiguous()
        filtered = logits.clone()
        for warper in warpers:
            filtered = warper(context_ids, filtered)
        probabilities = torch.softmax(filtered, dim=-1)
        candidate_ids = torch.nonzero(torch.isfinite(filtered[0]), as_tuple=False)[:, 0]
        candidate_logits = filtered[0, candidate_ids].contiguous()
        candidate_probabilities = probabilities[0, candidate_ids].contiguous()

        cumulative = torch.cumsum(probabilities[0], dim=0)
        selected = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(uniform, dtype=cumulative.dtype),
                right=False,
            ).clamp_max(cumulative.numel() - 1)
        )
        if probabilities[0, selected] == 0:
            raise RuntimeError("uniform threshold selected a filtered token")

        prefix = f"sampling.step_{step_index}"
        logits_key = f"{prefix}.logits"
        filtered_key = f"{prefix}.filtered_logits"
        probabilities_key = f"{prefix}.probabilities"
        tensors[logits_key] = logits[0]
        tensors[filtered_key] = candidate_logits
        tensors[probabilities_key] = candidate_probabilities
        generated_ids.append(selected)
        steps.append(
            {
                "step": step_index,
                "uniform": uniform,
                "token_id_0_based": selected,
                "sampled_probability": float(probabilities[0, selected]),
                "candidate_ids_0_based": candidate_ids.tolist(),
                "logits_key": logits_key,
                "filtered_logits_key": filtered_key,
                "probabilities_key": probabilities_key,
            }
        )
        if selected in eos_ids:
            stop_reason = "eos"
            break
        past_key_values = outputs.past_key_values
        input_ids = torch.tensor([[selected]], dtype=torch.long)
        context_ids = torch.cat((context_ids, input_ids), dim=1)

    save_file(tensors, output_dir / "reference.safetensors")
    reference = {
        "revision": args.revision,
        "compute_dtype": "float32",
        "weight_storage_dtype": "bfloat16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "generation_config_sha256": sha256_file(model_dir / "generation_config.json"),
        "prompt_input": args.prompt,
        "enable_thinking": args.enable_thinking,
        "prompt": prompt,
        "prompt_ids_0_based": prompt_ids,
        "uniforms": args.uniforms,
        "generation_config": {
            "do_sample": bool(config.do_sample),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "eos_token_id": sorted(eos_ids),
        },
        "generated_ids_0_based": generated_ids,
        "completion": tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "stop_reason": stop_reason,
        "steps": steps,
    }
    (output_dir / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
