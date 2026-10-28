#!/usr/bin/env python3
"""Export a small independent NumPy Flint oracle and HF checkpoint.

Requires numpy and safetensors. Run from the repository root:
  python3 scripts/export_flint_reference.py

Equations follow the Apache-2.0 Transformers Flint reference implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/flint/modeling_flint.py
The fixture deliberately exercises attention sinks, GQA, Q width != hidden
width, full/sliding attention, two dense layers, and routed/shared experts.
It is a synthetic numerical fixture, not a pretrained language model.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def make_fixture():
    config = dict(
        model_type="flint", architectures=["FlintForCausalLM"],
        vocab_size=23, hidden_size=8, intermediate_size=12,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, max_position_embeddings=32, rms_norm_eps=1e-6,
        rope_theta=10000.0, sliding_window=3,
        layer_types=["full_attention", "full_attention", "sliding_attention", "full_attention"],
        mlp_only_layers=[0, 1], decoder_sparse_step=1,
        moe_intermediate_size=8, shared_expert_intermediate_size=12,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
        tie_word_embeddings=False, bos_token_id=0, eos_token_id=[22], pad_token_id=22,
        attention_bias=False, attention_dropout=0.0, hidden_act="silu",
        router_score_function="sigmoid", softmax_type="learnable",
    )
    rng = np.random.default_rng(20261028)
    weights = {}

    def weight(name, shape, norm=False):
        if norm:
            weights[name] = rng.uniform(0.6, 1.4, shape).astype(np.float32)
        else:
            weights[name] = (rng.standard_normal(shape) * 0.23).astype(np.float32)

    def mlp(prefix, intermediate):
        weight(prefix + "gate_proj.weight", (intermediate, 8))
        weight(prefix + "up_proj.weight", (intermediate, 8))
        weight(prefix + "down_proj.weight", (8, intermediate))

    weight("model.embed_tokens.weight", (23, 8))
    weight("lm_head.weight", (23, 8))
    weight("model.norm.weight", (8,), True)
    for layer in range(4):
        p = f"model.layers.{layer}."
        weight(p + "input_layernorm.weight", (8,), True)
        weight(p + "post_attention_layernorm.weight", (8,), True)
        for proj, shape in (("q", (16, 8)), ("k", (8, 8)), ("v", (8, 8)), ("o", (8, 16))):
            weight(p + f"self_attn.{proj}_proj.weight", shape)
        weight(p + "self_attn.q_norm.weight", (4,), True)
        weight(p + "self_attn.k_norm.weight", (4,), True)
        weights[p + "self_attn.softmax_offset"] = np.array([-0.4, 0.7, 2.0, -1.0], np.float32)
        if layer < 2:
            mlp(p + "mlp.", 12)
        else:
            weight(p + "mlp.gate.weight", (4, 8))
            for expert in range(4):
                mlp(p + f"mlp.experts.{expert}.", 8)
            mlp(p + "mlp.shared_expert.", 12)
    return config, weights


def forward(config, weights, token_ids):
    """Independent token-major eager float32 reference; no LifeAI calls."""
    def norm(x, key):
        return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(config["rms_norm_eps"]))) * weights[key]

    def linear(x, key):
        return x @ weights[key].T

    def mlp(x, p):
        gate = linear(x, p + "gate_proj.weight")
        return linear((gate / (1 + np.exp(-gate))) * linear(x, p + "up_proj.weight"), p + "down_proj.weight")

    n, heads, d = len(token_ids), config["num_attention_heads"], config["head_dim"]
    x = weights["model.embed_tokens.weight"][token_ids].copy()
    positions = np.arange(n, dtype=np.float32)
    for layer, kind in enumerate(config["layer_types"]):
        p = f"model.layers.{layer}."
        a = p + "self_attn."
        hidden = norm(x, p + "input_layernorm.weight")
        q = norm(linear(hidden, a + "q_proj.weight").reshape(n, heads, d), a + "q_norm.weight")
        k = norm(linear(hidden, a + "k_proj.weight").reshape(n, config["num_key_value_heads"], d), a + "k_norm.weight")
        v = linear(hidden, a + "v_proj.weight").reshape(n, config["num_key_value_heads"], d)
        if kind == "sliding_attention":
            inv = np.float32(config["rope_theta"]) ** (-np.arange(0, d, 2, dtype=np.float32) / d)
            angle = positions[:, None] * inv[None, :]
            cosine = np.concatenate((np.cos(angle), np.cos(angle)), -1)[:, None, :]
            sine = np.concatenate((np.sin(angle), np.sin(angle)), -1)[:, None, :]
            def rotate(t):
                return np.concatenate((-t[..., d // 2:], t[..., :d // 2]), -1)
            q = q * cosine + rotate(q) * sine
            k = k * cosine + rotate(k) * sine
        k = np.repeat(k, heads // config["num_key_value_heads"], axis=1)
        v = np.repeat(v, heads // config["num_key_value_heads"], axis=1)
        scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(np.float32(d))
        allowed = positions[None, :] <= positions[:, None]
        if kind == "sliding_attention":
            allowed &= positions[None, :] > positions[:, None] - config["sliding_window"]
        scores = np.where(allowed[None, :, :], scores, -np.inf)
        sinks = np.broadcast_to(weights[a + "softmax_offset"][:, None, None], (heads, n, 1))
        augmented = np.concatenate((scores, sinks), axis=-1)
        exp = np.exp(augmented - augmented.max(axis=-1, keepdims=True))
        attention = (exp / exp.sum(axis=-1, keepdims=True))[..., :-1]
        output = np.einsum("hts,shd->thd", attention, v).reshape(n, heads * d)
        x = x + linear(output, a + "o_proj.weight")
        hidden = norm(x, p + "post_attention_layernorm.weight")
        if layer < 2:
            ff = mlp(hidden, p + "mlp.")
        else:
            router = linear(hidden, p + "mlp.gate.weight")
            scores = 1 / (1 + np.exp(-router))
            experts = np.argsort(-scores, axis=-1, kind="stable")[:, :config["num_experts_per_tok"]]
            routing = np.take_along_axis(scores, experts, axis=-1)
            routing /= routing.sum(axis=-1, keepdims=True)
            ff = np.zeros_like(hidden)
            # Token-wise reference avoids matching the implementation's dispatch.
            for token in range(n):
                for slot, expert in enumerate(experts[token]):
                    ff[token] += routing[token, slot] * mlp(hidden[token:token+1], p + f"mlp.experts.{expert}.")[0]
            ff += mlp(hidden, p + "mlp.shared_expert.")
        x = x + ff
    return linear(norm(x, "model.norm.weight"), "lm_head.weight")


def quantized_weights(weights, bits):
    """Independent weight-only round-to-nearest symmetric quantization oracle."""
    result = {}
    for name, weight in weights.items():
        if name == "lm_head.weight" or re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$", name):
            grouped = weight.reshape(weight.shape[0], -1, 4) if bits == 4 else weight[:, None, :]
            maximum = 7 if bits == 4 else 127
            scale = np.maximum(np.max(np.abs(grouped), axis=-1, keepdims=True) / maximum, np.float32(1e-12))
            rounded = np.clip(np.rint(grouped / scale), -maximum, maximum)
            result[name] = (rounded * scale).reshape(weight.shape)
        else:
            result[name] = weight
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "test/flint/fixtures/tiny")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config, weights = make_fixture()
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    names = sorted(weights)
    weight_map = {}
    for shard in range(2):
        name = f"model-{shard+1:05d}-of-00002.safetensors"
        selected = {key: weights[key] for key in names[shard::2]}
        save_file(selected, str(args.output / name))
        weight_map.update({key: name for key in selected})
    (args.output / "model.safetensors.index.json").write_text(json.dumps(dict(
        metadata={"total_size": sum(w.nbytes for w in weights.values())}, weight_map=weight_map), indent=2) + "\n")
    ids = [2, 4, 6, 9, 3, 5, 7, 4, 1]
    output = dict(source="NumPy float32 independent Flint reference", seed=20261028,
                  input_ids=ids, logits=forward(config, weights, ids).tolist(),
                  shifted_input_ids=ids[2:], shifted_logits=forward(config, weights, ids[2:]).tolist())
    context, generated = ids.copy(), []
    for _ in range(4):
        token = int(np.argmax(forward(config, weights, context)[-1]))
        generated.append(token)
        context.append(token)
    output["greedy_tokens"] = generated
    output["quantized_logits"] = {
        str(bits): forward(config, quantized_weights(weights, bits), ids).tolist()
        for bits in (8, 4)
    }
    (args.output / "reference.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Exported {len(weights)} tensors and {len(ids)} tokens to {args.output}")


if __name__ == "__main__":
    main()
