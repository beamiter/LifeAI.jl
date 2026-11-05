#!/usr/bin/env python3
"""Export tiny long-position oracles from the pinned published rotary forward.

CPU suffices for the checked-in fixture. Also run --device cuda on DSW to check
its actual torch CUDA power/trig backend independently; no model weights load.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import torch
from safetensors.torch import save_file

REVISION = 'e27e171792f303d74c048184e8259ba663cfa0d5'
HASHES = {
    'configuration_ricursive.py': '002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6',
    'modeling_ricursive.py': 'f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3',
}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output', type=Path,
        default=Path(__file__).resolve().parents[1] / 'test/ripple/fixtures/rope/reference.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    for name, expected in HASHES.items():
        if hashlib.sha256((args.upstream / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name} differs from pinned revision {REVISION}')
    package = types.ModuleType('ripple_rope_official')
    package.__path__ = [str(args.upstream.resolve())]
    sys.modules[package.__name__] = package
    official = importlib.import_module(package.__name__ + '.modeling_ricursive')
    torch.set_num_threads(1)
    torch.manual_seed(1048576)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')
    cases = []
    tensors = {}
    groups = [
        ('sparse_long', [0, 1, 1023, 1024, 4095, 4096, 4097, 8191, 8192,
                         32767, 32768, 131071, 131072, 262143, 262144,
                         524287, 524288, 1048575, 1048576, 2097151]),
        ('window_crossing', list(range(4094, 4100))),
        ('near_128k', list(range(131070, 131076))),
        ('near_1mi', list(range(1048574, 1048580))),
    ]
    dim, heads, theta = 128, 3, 10000.0
    rotary = official.RicursiveRotaryEmbedding(dim, theta)
    for name, positions in groups:
        query = torch.randn((1, heads, len(positions), dim), dtype=torch.float32).to(args.device)
        ids = torch.tensor([positions], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            cosine, sine = rotary(query, ids)
            rotated, _ = official.apply_rotary_pos_emb(query, query, cosine, sine)
            # Diagnostic coefficients use the exact published Float32 expression;
            # the primary cosine/sine/rotated outputs above call its actual code.
            frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, device=args.device).float() / dim))
        tensors[name + '.input'] = query[0].permute(2, 0, 1).contiguous().cpu()
        tensors[name + '.rotated'] = rotated[0].permute(2, 0, 1).contiguous().cpu()
        tensors[name + '.cosine'] = cosine[0].transpose(0, 1).contiguous().cpu()
        tensors[name + '.sine'] = sine[0].transpose(0, 1).contiguous().cpu()
        tensors[name + '.inverse_frequencies'] = frequencies.cpu().contiguous()
        cases.append(dict(name=name, dim=dim, heads=heads, theta=theta,
            positions_zero_based=positions, input_shape=[dim, heads, len(positions)]))
    result = dict(repository='ricursiveAI/Ripple-65B-Preview', revision=REVISION,
        source_hashes=HASHES, torch_version=torch.__version__, device=args.device,
        operation='published RicursiveRotaryEmbedding.forward and apply_rotary_pos_emb',
        tensor_layout='input/rotated (D,H,T); cosine/sine (D,T)',
        tensor_file=args.output.with_suffix('.safetensors').name, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    save_file(tensors, str(args.output.with_suffix('.safetensors')))
    print(json.dumps(dict(output=str(args.output), device=args.device, cases=len(cases), torch=torch.__version__)))


if __name__ == '__main__':
    main()
