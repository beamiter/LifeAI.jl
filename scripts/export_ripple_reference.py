#!/usr/bin/env python3
"""Create synthetic fixtures with the *published* Ricursive PyTorch forward.

python export_ripple_reference.py --upstream /path/to/pinned/hf/snapshot
Requires torch (CPU is sufficient), transformers==4.57.6, safetensors, tokenizers.
No production weights are downloaded. Review upstream Python before running.
"""
import argparse
import hashlib
import importlib
import json
import pathlib
import sys
import types

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import AutoTokenizer

REPO = "ricursiveAI/Ripple-65B-Preview"
REVISION = "e27e171792f303d74c048184e8259ba663cfa0d5"
SOURCE_HASHES = {
    "configuration_ricursive.py": "002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6",
    "modeling_ricursive.py": "f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / "test/ripple/fixtures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((args.upstream / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} differs from pinned revision {REVISION}")
    package = types.ModuleType("ripple_upstream")
    package.__path__ = [str(args.upstream.resolve())]
    sys.modules[package.__name__] = package
    Config = importlib.import_module("ripple_upstream.configuration_ricursive").RicursiveConfig
    Model = importlib.import_module("ripple_upstream.modeling_ricursive").RicursiveForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(711)
    config = Config(vocab_size=279, hidden_size=8, intermediate_size=12,
                    num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=4, max_position_embeddings=64, sliding_window=3,
                    layer_types=["sliding_attention", "full_attention", "sliding_attention",
                                 "sliding_attention", "full_attention"], use_valres=True,
                    ffn_clamp_threshold=0.4, embed_scale=8**0.5,
                    bos_token_id=257, eos_token_id=[265, 257], pad_token_id=277)
    model = Model(config).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "layernorm" in name or "_norm.weight" in name or name == "model.norm.weight":
                parameter.copy_(0.7 + torch.rand_like(parameter) * 0.6)
            elif name.endswith("value_res_lambda"):
                parameter.copy_(torch.randn_like(parameter) * 1.7)
            else:
                parameter.copy_(torch.randn_like(parameter) * 0.45)
    weights = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(args.output / "model.safetensors"))
    write_json(args.output / "config.json", config.to_dict())
    tokens = torch.tensor([[3, 8, 4, 7, 9, 2, 12, 16, 6, 1, 15] * 2 + [5]])

    def forward(m, ids):
        with torch.no_grad():
            return m(ids, use_cache=False).logits[0]

    expected = forward(model, tokens)
    with torch.no_grad():
        cache = None
        steps = []
        for i in range(tokens.shape[1]):
            output = model(tokens[:, i:i+1], use_cache=True, past_key_values=cache)
            cache = output.past_key_values
            steps.append(output.logits[0])
        torch.testing.assert_close(torch.cat(steps), expected, atol=3e-5, rtol=3e-5)
    generated = []
    growing = tokens.clone()
    for _ in range(5):
        token = int(forward(model, growing)[-1].argmax())
        generated.append(token)
        growing = torch.cat([growing, torch.tensor([[token]])], dim=1)
    variants = []
    for options in [dict(use_valres=False), dict(use_gated_attention=False),
                    dict(ffn_clamp_threshold=None), dict(embed_scale=None),
                    dict(sliding_window=1), dict(use_valres=False, use_gated_attention=False,
                                                ffn_clamp_threshold=None, embed_scale=None)]:
        variant_config = Config(**(config.to_dict() | options))
        variant = Model(variant_config).float().eval()
        variant.load_state_dict(weights, strict=False)
        variants.append(dict(options=options, logits=forward(variant, tokens).tolist()))
    reference = dict(repository=REPO, revision=REVISION, torch=torch.__version__,
                     input_ids=tokens[0].tolist(), logits=expected.tolist(),
                     greedy_tokens=generated, variants=variants,
                     source_sha256={name: hashlib.sha256((args.upstream / name).read_bytes()).hexdigest()
                                    for name in ["modeling_ricursive.py", "configuration_ricursive.py"]})
    write_json(args.output / "reference.json", reference)

    # Tiny tokenizer uses the exact upstream splitting pipeline and template, a
    # complete byte alphabet and one real BPE merge. Added tokens also occur in
    # the BPE vocabulary, as they do in the production Ripple tokenizer.
    original = json.loads((args.upstream / "tokenizer.json").read_text())
    byte_vocab = {token: idx for token, idx in original["model"]["vocab"].items() if idx < 256}
    for token in original["added_tokens"]:
        token["id"] -= 100000
        byte_vocab[token["content"]] = token["id"]
    byte_vocab["hi"] = 278
    original["model"]["vocab"] = byte_vocab
    original["model"]["merges"] = [["h", "i"]]
    write_json(args.output / "tokenizer.json", original)
    tiny_tokenizer_config = json.loads((args.upstream / "tokenizer_config.json").read_text())
    tiny_tokenizer_config.pop("added_tokens_decoder", None)
    write_json(args.output / "tokenizer_config.json", tiny_tokenizer_config)
    (args.output / "chat_template.jinja").write_bytes((args.upstream / "chat_template.jinja").read_bytes())
    examples = ["", "Hello, world!", "hi hi", "你好，世界！123456789", "Café cafe\u0301",
                "\x00\n\t", "I'm WE'RE\r\n", "a < b && c > d", "🦀🚀", "foo_bar.jl",
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
                "<|extra_id_0|>|||EMAIL_ADDRESS|||<|pad|>", "  \n\n\t trailing  "]
    cases = []
    for kind, path in [("tiny", args.output), ("official", args.upstream)]:
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        for text in examples:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            cases.append(dict(kind=kind, text=text, ids=ids, decoded=tokenizer.decode(ids),
                              decoded_all=tokenizer.decode(ids, skip_special_tokens=False)))
    write_json(args.output / "tokenizer_cases.json", cases)
    tokenizer = AutoTokenizer.from_pretrained(args.output)
    tools = [dict(type="function", function=dict(name="execute_bash", description="中文 <&> 🦀",
                                                parameters=dict(type="object", properties=dict(command=dict(type="string")))))]
    messages = [dict(role="system", content="Be helpful."), dict(role="user", content="pwd"),
                dict(role="assistant", content="", reasoning_content="inspect",
                     tool_calls=[dict(function=dict(name="execute_bash", arguments=dict(command="pwd")))]),
                dict(role="tool", content="/tmp"), dict(role="tool", content="ok"),
                dict(role="assistant", content="/tmp", reasoning_content="done"),
                dict(role="user", content="next")]
    chat_cases = []
    for options in [dict(), dict(tools=tools), dict(clear_thinking=True),
                    dict(tools=tools, add_generation_prompt=False)]:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               **(dict(add_generation_prompt=True) | options))
        chat_cases.append(dict(messages=messages, options=options, text=prompt))
    write_json(args.output / "chat_cases.json", chat_cases)
    print(f"Official PyTorch reference exported to {args.output}")


if __name__ == "__main__":
    main()
