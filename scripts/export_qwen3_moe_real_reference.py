#!/usr/bin/env python3
"""Export a layer-streamed Float32 Transformers reference for Qwen3 MoE.

The official checkpoint stores BF16 values.  This exporter calls the upstream
Transformers decoder layer and releases it before loading the next one.  Native
BF16 is the default; Float32 promotion remains available as an explicit
diagnostic mode.  Neither mode requires the complete parameter tree to remain
resident in host memory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)


DEFAULT_TOKEN_IDS = [1, 9707]
DEFAULT_DECODE_TOKEN_ID = 13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maxrss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class ShardedTensorReader:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        single_path = model_dir / "model.safetensors"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = {
                str(name): str(shard)
                for name, shard in index["weight_map"].items()
            }
            self.index_path: Path | None = index_path
        elif single_path.is_file():
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                self.weight_map = {str(name): single_path.name for name in handle.keys()}
            self.index_path = None
        else:
            raise FileNotFoundError(
                f"no model.safetensors or model.safetensors.index.json in {model_dir}"
            )

    def tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"checkpoint tensor is missing: {name}")
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).to(dtype=dtype).contiguous()

    def state_dict(
        self,
        prefix: str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        names = sorted(name for name in self.weight_map if name.startswith(prefix))
        if not names:
            raise KeyError(f"checkpoint contains no tensors below {prefix}")
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name in names:
            by_shard[self.weight_map[name]].append(name)
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(by_shard):
            with safe_open(
                self.model_dir / shard,
                framework="pt",
                device="cpu",
            ) as handle:
                for name in by_shard[shard]:
                    state[name.removeprefix(prefix)] = (
                        handle.get_tensor(name).to(dtype=dtype).contiguous()
                    )
        return state


def causal_mask(sequence_length: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full(
        (sequence_length, sequence_length),
        torch.finfo(dtype).min,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.reshape(1, 1, sequence_length, sequence_length)


def routing_reference(
    router_logits: torch.Tensor,
    top_k: int,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(router_logits.float(), dim=-1)
    routing_weights, selected = torch.topk(probabilities, top_k, dim=-1)
    if normalize:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # LifeAI's checkpoint/reference safetensors reader intentionally accepts
    # only floating tensors. Expert ids are <= 127 and therefore exact in F32.
    return routing_weights.contiguous(), selected.to(torch.float32).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--token-ids", type=int, nargs="+", default=DEFAULT_TOKEN_IDS)
    parser.add_argument("--decode-token-id", type=int, default=DEFAULT_DECODE_TOKEN_ID)
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    return parser.parse_args()


def export(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    compute_dtype = (
        torch.bfloat16 if args.compute_dtype == "bfloat16" else torch.float32
    )

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config._attn_implementation = "eager"
    if config.model_type != "qwen3_moe":
        raise ValueError(f"expected qwen3_moe, got {config.model_type}")
    if config.tie_word_embeddings:
        raise ValueError("the real parity exporter currently requires an untied LM head")

    token_ids = torch.tensor([args.token_ids], dtype=torch.long)
    decode_ids = torch.tensor([[args.decode_token_id]], dtype=torch.long)
    for token in args.token_ids + [args.decode_token_id]:
        if not 0 <= token < config.vocab_size:
            raise ValueError(f"token id {token} is outside the model vocabulary")

    reader = ShardedTensorReader(model_dir)
    embedding_started = time.perf_counter()
    embedding_weight = reader.tensor("model.embed_tokens.weight", compute_dtype)
    prompt_hidden = F.embedding(token_ids, embedding_weight).contiguous()
    decode_hidden = F.embedding(decode_ids, embedding_weight).contiguous()
    del embedding_weight
    gc.collect()

    tensors: dict[str, torch.Tensor] = {
        "embedding": prompt_hidden.clone(),
        "decode_embedding": decode_hidden.clone(),
    }
    prompt_length = token_ids.shape[1]
    prompt_positions = torch.arange(prompt_length, dtype=torch.long).unsqueeze(0)
    decode_positions = torch.tensor([[prompt_length]], dtype=torch.long)
    prompt_cache_position = torch.arange(prompt_length, dtype=torch.long)
    decode_cache_position = torch.tensor([prompt_length], dtype=torch.long)
    rotary = Qwen3MoeRotaryEmbedding(config)
    prompt_position_embeddings = rotary(prompt_hidden, prompt_positions)
    decode_position_embeddings = rotary(decode_hidden, decode_positions)
    prompt_attention_mask = causal_mask(prompt_length, compute_dtype)
    cache = DynamicCache()
    embedding_seconds = time.perf_counter() - embedding_started

    layer_timings: list[dict[str, float | int]] = []
    for layer_index in range(config.num_hidden_layers):
        load_started = time.perf_counter()
        prefix = f"model.layers.{layer_index}."
        state = reader.state_dict(prefix, compute_dtype)
        with torch.device("meta"):
            layer = Qwen3MoeDecoderLayer(config, layer_index)
        incompatible = layer.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"layer {layer_index} state mismatch: {incompatible}"
            )
        del state
        layer.eval()
        load_seconds = time.perf_counter() - load_started

        forward_started = time.perf_counter()
        prompt_output = layer(
            prompt_hidden,
            attention_mask=prompt_attention_mask,
            position_ids=prompt_positions,
            past_key_value=cache,
            output_router_logits=True,
            use_cache=True,
            cache_position=prompt_cache_position,
            position_embeddings=prompt_position_embeddings,
        )
        prompt_hidden, prompt_router = prompt_output
        decode_output = layer(
            decode_hidden,
            attention_mask=None,
            position_ids=decode_positions,
            past_key_value=cache,
            output_router_logits=True,
            use_cache=True,
            cache_position=decode_cache_position,
            position_embeddings=decode_position_embeddings,
        )
        decode_hidden, decode_router = decode_output
        prompt_weights, prompt_selected = routing_reference(
            prompt_router,
            config.num_experts_per_tok,
            config.norm_topk_prob,
        )
        decode_weights, decode_selected = routing_reference(
            decode_router,
            config.num_experts_per_tok,
            config.norm_topk_prob,
        )
        tensors[f"block.{layer_index}"] = prompt_hidden.clone().contiguous()
        tensors[f"router_logits.{layer_index}"] = prompt_router.clone().contiguous()
        tensors[f"routing_weights.{layer_index}"] = prompt_weights
        tensors[f"selected_experts.{layer_index}"] = prompt_selected
        tensors[f"decode_block.{layer_index}"] = decode_hidden.clone().contiguous()
        tensors[f"decode_router_logits.{layer_index}"] = decode_router.clone().contiguous()
        tensors[f"decode_routing_weights.{layer_index}"] = decode_weights
        tensors[f"decode_selected_experts.{layer_index}"] = decode_selected
        forward_seconds = time.perf_counter() - forward_started
        layer_timings.append(
            {
                "layer": layer_index,
                "load_seconds": load_seconds,
                "forward_seconds": forward_seconds,
            }
        )
        print(
            f"layer {layer_index + 1:02d}/{config.num_hidden_layers}: "
            f"load={load_seconds:.3f}s forward={forward_seconds:.3f}s "
            f"rss={maxrss_bytes() / 2**30:.2f}GiB",
            file=sys.stderr,
            flush=True,
        )
        del layer, prompt_output, decode_output, prompt_router, decode_router
        gc.collect()

    output_started = time.perf_counter()
    norm_weight = reader.tensor("model.norm.weight", compute_dtype)
    norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.load_state_dict({"weight": norm_weight}, strict=True, assign=True)
    norm.eval()
    final_hidden = norm(prompt_hidden).contiguous()
    decode_final_hidden = norm(decode_hidden).contiguous()
    del norm, norm_weight

    lm_head_weight = reader.tensor("lm_head.weight", compute_dtype)
    logits = F.linear(final_hidden, lm_head_weight).contiguous()
    decode_logits = F.linear(decode_final_hidden, lm_head_weight).contiguous()
    del lm_head_weight
    gc.collect()
    tensors.update(
        {
            "final_hidden": final_hidden,
            "decode_final_hidden": decode_final_hidden,
            "logits": logits,
            "decode_logits": decode_logits,
        }
    )
    reference_path = output_dir / "reference.safetensors"
    save_file(tensors, reference_path)
    output_seconds = time.perf_counter() - output_started

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    metadata = {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3-30B-A3B",
        "revision": args.revision,
        "source": str(model_dir),
        "implementation": "transformers Qwen3MoeDecoderLayer, layer-streamed",
        "weight_storage_dtype": str(config.torch_dtype).removeprefix("torch."),
        "compute_dtype": args.compute_dtype,
        "attention_implementation": "eager",
        "token_ids_0_based": args.token_ids,
        "decode_token_id_0_based": args.decode_token_id,
        "prompt_length": prompt_length,
        "num_hidden_layers": config.num_hidden_layers,
        "num_experts": config.num_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "runtime": {
            "threads": args.threads,
            "embedding_seconds": embedding_seconds,
            "output_projection_seconds": output_seconds,
            "total_seconds": time.perf_counter() - started,
            "maxrss_bytes": maxrss_bytes(),
            "layers": layer_timings,
        },
        "checksums": {
            "exporter_sha256": sha256(Path(__file__).resolve()),
            "config_sha256": sha256(config_path),
            "index_sha256": sha256(index_path) if index_path.is_file() else None,
            "reference_sha256": sha256(reference_path),
        },
    }
    (output_dir / "reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"reference ready: {reference_path} "
        f"total={metadata['runtime']['total_seconds']:.3f}s "
        f"maxrss={metadata['runtime']['maxrss_bytes'] / 2**30:.2f}GiB",
        file=sys.stderr,
    )


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
