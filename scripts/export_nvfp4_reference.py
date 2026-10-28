#!/usr/bin/env python3
"""Independent NumPy oracle for rowwise NVFP4 E2M1/E4M3-FN double scaling.

Implements NVIDIA's 1D block-16 recipe with deterministic nearest-even casts;
this is not the stochastic/RHT/2D training recipe. No LifeAI code is invoked.

Sources:
https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def fp8_decode(code):
    exponent, mantissa = code >> 3, code & 7
    return np.float32(mantissa / 512 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7))


FP8 = np.array([fp8_decode(code) for code in range(127)], dtype=np.float32)


def nearest_even(value, values):
    """Compare exact binary32 input against an exhaustive finite codebook."""
    return min(range(len(values)), key=lambda code: (abs(float(value) - float(values[code])), code & 1))


def quantize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    rows, columns = matrix.shape
    blocks = (columns + 15) // 16
    padded = np.zeros((rows, blocks * 16), dtype=np.float32)
    padded[:, :columns] = matrix
    maximum = np.max(np.abs(matrix))
    global_scale = np.float32(1) if maximum == 0 else np.maximum(np.float32(maximum / np.float32(2688)), np.nextafter(np.float32(0), np.float32(1)))
    if np.float32(np.float32(2688) * global_scale) == np.float32(np.inf):
        global_scale = np.nextafter(global_scale, np.float32(0))
    block_codes = np.zeros((rows, blocks), dtype=np.uint8)
    element_codes = np.zeros_like(padded, dtype=np.uint8)
    reconstructed = np.zeros_like(padded)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for row in range(rows):
            for block in range(blocks):
                start = block * 16
                values = padded[row, start:start+16]
                maximum = np.max(np.abs(values))
                scale_code = nearest_even(np.float32(np.float32(maximum / np.float32(6)) / global_scale), FP8)
                block_codes[row, block] = scale_code
                block_scale = FP8[scale_code]
                if block_scale == 0:
                    continue
                for offset, value in enumerate(values):
                    code = nearest_even(np.float32(np.float32(abs(value) / global_scale) / block_scale), FP4)
                    code |= int(np.signbit(value)) << 3
                    element_codes[row, start+offset] = code
                    signed = np.copysign(FP4[code & 7], np.float32(-1) if code & 8 else np.float32(1))
                    reconstructed[row, start+offset] = np.float32(np.float32(signed * block_scale) * global_scale)
    packed = element_codes[:, ::2] | (element_codes[:, 1::2] << 4)
    return dict(input=matrix.tolist(), input_bits=matrix.view(np.uint32).tolist(),
                packed=packed.tolist(), block_scales=block_codes.tolist(),
                global_scale=float(global_scale), reconstructed=reconstructed[:, :columns].tolist(),
                reconstructed_bits=np.ascontiguousarray(reconstructed[:, :columns]).view(np.uint32).tolist())


def cases():
    # Anchor makes global_scale=1, so the first two rows test exact FP4 codes.
    codebook = np.vstack([np.tile(FP4, 2), -np.tile(FP4, 2), np.full(16, 2688, np.float32)])
    midpoints = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
    ties = np.vstack([np.r_[midpoints, -midpoints, 6, -6], np.full(16, 2688, np.float32)])
    # Every positive E4M3-FN code, including subnormals and maximum finite.
    scale_codebook = np.vstack([np.full(16, np.float32(value * 6), np.float32) for value in FP8])
    scale_ties = np.vstack([np.full(16, np.float32((FP8[i]+FP8[i+1])/2*6), np.float32)
                           for i in (0, 1, 6, 7, 8, 15, 55, 56, 62, 63, 64, 119, 125)] + [np.full(16, 2688, np.float32)])
    rng = np.random.default_rng(20261028)
    random = rng.normal(0, 1, (4, 37)).astype(np.float32) * np.array([1e-4, 1, 13, 1e4], np.float32)[:, None]
    tiny = np.zeros((2, 17), np.float32)
    tiny[0, :4] = [np.nextafter(np.float32(0), np.float32(1)), -np.nextafter(np.float32(0), np.float32(1)), 0, -0.0]
    tiny[1, :3] = [1e-35, -2e-35, 1e-37]
    big = np.zeros((2, 17), np.float32)
    big[0, :4] = [np.finfo(np.float32).max, -np.finfo(np.float32).max, 1e30, -1e30]
    big[1] = np.linspace(-1e37, 1e37, 17, dtype=np.float32)
    return dict(codebook=codebook, ties=ties, scale_codebook=scale_codebook,
                scale_ties=scale_ties, random_padded=random, tiny=tiny, big=big,
                zeros=np.zeros((2, 17), np.float32), single=np.array([[-3.25]], np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "test/fixtures/nvfp4/reference.json")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = repo / "scripts/export_flint_reference.py"
    spec = importlib.util.spec_from_file_location("flint_oracle", source)
    flint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flint)
    fixture = repo / "test/flint/fixtures/tiny"
    config = json.loads((fixture / "config.json").read_text())
    index = json.loads((fixture / "model.safetensors.index.json").read_text())
    weights = {}
    for shard in sorted(set(index["weight_map"].values())):
        weights.update(load_file(str(fixture / shard)))
    quantized, records = {}, {}
    for name, value in weights.items():
        if name == "lm_head.weight" or re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$", name):
            records[name] = quantize(value)
            quantized[name] = np.array(records[name]["reconstructed"], np.float32)
        else:
            quantized[name] = value
    tokens = [2, 4, 6, 9, 3, 5, 7, 4, 1]
    generated, context = [], tokens.copy()
    for _ in range(4):
        token = int(np.argmax(flint.forward(config, quantized, context)[-1]))
        generated.append(token)
        context.append(token)
    output = dict(recipe="NVFP4 1D block16, E2M1 RNE, E4M3-FN RNE, FP32 global scale",
                  matrix_cases={name: quantize(value) for name, value in cases().items()},
                  flint=dict(tensors=records, input_ids=tokens,
                             logits=flint.forward(config, quantized, tokens).tolist(), greedy_tokens=generated))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(output['matrix_cases'])} matrix cases and {len(records)} Flint projection references to {args.output}")


if __name__ == "__main__":
    main()
