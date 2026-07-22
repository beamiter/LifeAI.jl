#!/usr/bin/env python3
"""Export a small, offline Qwen3 Float32 parity fixture for LifeAI.jl."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


DEFAULT_TOKEN_IDS = [1, 9707, 13, 151643, 100, 42, 151645, 2]
DEFAULT_DECODE_TOKEN_ID = 17


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--revision", default="local")
    parser.add_argument("--token-ids", type=int, nargs="+", default=DEFAULT_TOKEN_IDS)
    parser.add_argument("--decode-token-id", type=int, default=DEFAULT_DECODE_TOKEN_ID)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    block_outputs = [None] * len(model.model.layers)
    handles = []

    def capture_block(index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            block_outputs[index] = value.detach().to(torch.float32).cpu().contiguous()

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(capture_block(index)))

    input_ids = torch.tensor([args.token_ids], dtype=torch.long)
    decode_ids = torch.tensor([[args.decode_token_id]], dtype=torch.long)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        for handle in handles:
            handle.remove()

        decode_outputs = model(
            input_ids=decode_ids,
            past_key_values=outputs.past_key_values,
            use_cache=True,
            return_dict=True,
        )

    tensors = {
        "embedding": outputs.hidden_states[0].detach().to(torch.float32).cpu().contiguous(),
        "final_hidden": outputs.hidden_states[-1].detach().to(torch.float32).cpu().contiguous(),
        "logits": outputs.logits.detach().to(torch.float32).cpu().contiguous(),
        "decode_logits": decode_outputs.logits.detach().to(torch.float32).cpu().contiguous(),
    }
    for index, value in enumerate(block_outputs):
        if value is None:
            raise RuntimeError(f"block {index} hook did not capture an output")
        tensors[f"block.{index}"] = value

    save_file(tensors, args.output_dir / "reference.safetensors")
    metadata = {
        "model": str(args.model_dir.resolve()),
        "revision": args.revision,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "weight_storage_dtype": "bfloat16",
        "compute_dtype": "float32",
        "token_ids_0_based": args.token_ids,
        "decode_token_id_0_based": args.decode_token_id,
    }
    (args.output_dir / "reference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
