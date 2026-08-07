#!/usr/bin/env python3
"""Export a deterministic tiny Qwen3 MoE checkpoint and parity reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_values(name: str, parameter: torch.Tensor) -> torch.Tensor:
    seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    positions = torch.arange(parameter.numel(), dtype=torch.float32).reshape(
        parameter.shape
    )
    phase = (seed % 10_000) / 997.0
    if "layernorm.weight" in name or name == "model.norm.weight":
        return 1.0 + 0.03 * torch.sin(positions * 0.41 + phase)
    return 0.08 * torch.sin(positions * 0.37 + phase) + 0.02 * torch.cos(
        positions * 0.13 + phase * 0.5
    )


def build_model() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        moe_intermediate_size=3,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        norm_topk_prob=True,
        rms_norm_eps=1.0e-6,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        sliding_window=None,
        use_sliding_window=False,
        tie_word_embeddings=False,
        torch_dtype="float32",
        use_cache=True,
    )
    model = Qwen3MoeForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(deterministic_values(name, parameter))
    return model


def export(output_dir: Path) -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model.save_pretrained(output_dir, safe_serialization=True)

    input_ids = torch.tensor([[0, 3, 6, 9]], dtype=torch.long)
    prompt_ids = input_ids[:, :3]
    decode_ids = input_ids[:, 3:]
    block_outputs: list[torch.Tensor] = []
    handles = []
    for layer in model.model.layers:
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output: block_outputs.append(
                    output[0].detach().clone()
                )
            )
        )

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_router_logits=True,
        use_cache=True,
        return_dict=True,
    )
    for handle in handles:
        handle.remove()

    prompt = model(
        input_ids=prompt_ids,
        output_router_logits=True,
        use_cache=True,
        return_dict=True,
    )
    decoded = model(
        input_ids=decode_ids,
        past_key_values=prompt.past_key_values,
        output_router_logits=True,
        use_cache=True,
        return_dict=True,
    )

    tensors: dict[str, torch.Tensor] = {
        "embedding": model.model.embed_tokens(input_ids).detach().contiguous(),
        "final_hidden": outputs.hidden_states[-1].detach().contiguous(),
        "logits": outputs.logits.detach().contiguous(),
        "prompt_logits": prompt.logits.detach().contiguous(),
        "decode_logits": decoded.logits.detach().contiguous(),
    }
    selected_experts: dict[str, list[list[int]]] = {}
    for layer, (block, router_logits) in enumerate(
        zip(block_outputs, outputs.router_logits)
    ):
        probabilities = torch.softmax(router_logits.float(), dim=-1)
        routing_weights, selected = torch.topk(
            probabilities,
            model.config.num_experts_per_tok,
            dim=-1,
        )
        if model.config.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        tensors[f"block.{layer}"] = block.contiguous()
        tensors[f"router_logits.{layer}"] = router_logits.detach().contiguous()
        tensors[f"routing_weights.{layer}"] = routing_weights.detach().contiguous()
        selected_experts[str(layer)] = selected.tolist()

    reference_path = output_dir / "reference.safetensors"
    save_file(tensors, reference_path)
    config_path = output_dir / "config.json"
    generation_config_path = output_dir / "generation_config.json"
    model_path = output_dir / "model.safetensors"
    metadata = {
        "schema_version": 1,
        "model_type": "qwen3_moe",
        "dtype": "float32",
        "input_ids_0_based": input_ids[0].tolist(),
        "prompt_ids_0_based": prompt_ids[0].tolist(),
        "decode_id_0_based": int(decode_ids[0, 0]),
        "selected_experts_0_based": selected_experts,
        "versions": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "checksums": {
            "script_sha256": sha256(Path(__file__)),
            "config_sha256": sha256(config_path),
            "generation_config_sha256": sha256(generation_config_path),
            "model_sha256": sha256(model_path),
            "reference_sha256": sha256(reference_path),
        },
    }
    (output_dir / "reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export(args.output_dir.resolve())


if __name__ == "__main__":
    main()
