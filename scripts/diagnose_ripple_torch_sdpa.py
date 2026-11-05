#!/usr/bin/env python3
"""Compare automatic CUDA FP32 attention with PyTorch's fixed MATH backend."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'benchmark_results/ripple/full/torch-sdpa-diagnostic.json')
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for tokens in (19, 28, 64):
        query = torch.randn(1, tokens, 80, 128, device='cuda').transpose(1, 2)
        key = torch.randn(1, tokens, 8, 128, device='cuda').transpose(1, 2).repeat_interleave(10, dim=1)
        value = torch.randn_like(key)
        automatic = scaled_dot_product_attention(query, key, value, is_causal=True)
        with sdpa_kernel(SDPBackend.MATH):
            mathematical = scaled_dot_product_attention(query, key, value, is_causal=True)
        records.append({'tokens': tokens,
                        'max_absolute_error': float((automatic - mathematical).abs().max())})
    result = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'seed': 42, 'cases': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
