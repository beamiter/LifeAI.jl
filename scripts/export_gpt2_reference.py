#!/usr/bin/env python3
"""Export an offline, deterministic GPT-2 tokenizer/forward/cache reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import safetensors
import tokenizers
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


MODEL_ID = "openai-community/gpt2"
DEFAULT_PROMPT = "The meaning of life is"
CORPUS = [
    "Hello, world!",
    "Hello, 世界",
    "  multiple   spaces",
    "line one\nline two\r\n",
    "\tindent\u0000\u0001",
    "café naïve",
    "emoji: 🧬🚀",
    "can't I'll we've",
    " trailing   ",
    "<|endoftext|>",
]
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_bytes(token: str, inverse: dict[str, int]) -> bytes:
    if token == "<|endoftext|>":
        return token.encode("utf-8")
    return bytes(inverse[character] for character in token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if not args.revision or args.steps < 1:
        parser.error("--revision is required and --steps must be positive")
    for filename in REQUIRED_FILES:
        path = args.model_dir / filename
        if not path.is_file():
            parser.error(f"missing model file: {path}")

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    inverse = {character: byte for byte, character in bytes_to_unicode().items()}
    corpus = []
    for text in CORPUS:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = list(encoded["input_ids"])
        tokens = tokenizer.convert_ids_to_tokens(ids)
        pretokenized = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        corpus.append(
            {
                "text": text,
                "ids": ids,
                "tokens": tokens,
                "token_bytes_hex": [token_bytes(value, inverse).hex() for value in tokens],
                "decoded": tokenizer.decode(
                    ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                ),
                "decoded_skip_special": tokenizer.decode(
                    ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                ),
                "offsets": [list(pair) for pair in encoded["offset_mapping"]],
                "pretokenized": [
                    {"text": value, "offset": list(offset)}
                    for value, offset in pretokenized
                ],
            }
        )

    inputs = tokenizer(
        args.prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    block_outputs: list[torch.Tensor] = []
    handles = []
    for block in model.transformer.h:
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, output: block_outputs.append(
                    output[0].detach().cpu()
                )
            )
        )
    with torch.no_grad():
        positions = torch.arange(inputs.input_ids.shape[1]).unsqueeze(0)
        embedding = (
            model.transformer.wte(inputs.input_ids)
            + model.transformer.wpe(positions)
        ).cpu()
        output = model(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    if len(block_outputs) != model.config.n_layer:
        raise RuntimeError("failed to capture every GPT-2 block")

    generated = inputs.input_ids.clone()
    step_logits = []
    generated_ids = []
    past_key_values = None
    current = generated
    with torch.no_grad():
        for _ in range(args.steps):
            result = model(
                input_ids=current,
                past_key_values=past_key_values,
                use_cache=True,
            )
            values = result.logits[:, -1, :].cpu()
            step_logits.append(values)
            next_id = values.argmax(dim=-1)
            generated_ids.append(int(next_id.item()))
            past_key_values = result.past_key_values
            current = next_id[:, None]
    all_ids = list(inputs.input_ids[0].tolist()) + generated_ids

    tensors = {
        "embedding": embedding,
        "final_hidden": output.logits.new_empty(0),  # replaced below
        "logits": output.logits.cpu(),
        "greedy_step_logits": torch.cat(step_logits, dim=0),
    }
    with torch.no_grad():
        final_hidden = model.transformer(
            input_ids=inputs.input_ids,
            use_cache=False,
        ).last_hidden_state.cpu()
    tensors["final_hidden"] = final_hidden
    for index, value in enumerate(block_outputs):
        tensors[f"block_{index:02d}"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output_dir / "reference.safetensors")
    metadata = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": args.revision,
        "source": str(args.model_dir.resolve()),
        "prompt": args.prompt,
        "prompt_ids": list(inputs.input_ids[0].tolist()),
        "greedy_steps": args.steps,
        "generated_ids": generated_ids,
        "completion": tokenizer.decode(
            generated_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        ),
        "text": tokenizer.decode(
            all_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        ),
        "files": {filename: sha256(args.model_dir / filename) for filename in REQUIRED_FILES},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
        },
        "dtype": "float32",
        "attention_implementation": "eager",
        "deterministic_algorithms": True,
        "corpus": corpus,
    }
    (args.output_dir / "reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
