#!/usr/bin/env python3
"""Independent last-token global-attention oracle using actual Torch MATH SDPA.

Procedural exact Float32 hash inputs avoid committing gigabytes of K/V data.
Torch uses non-overflowing Int64 arithmetic followed by UInt32 masks; the Julia
GPU test implements the same algorithm with wrapping UInt32 arithmetic.
No model/checkpoint tensors are read or loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ALGORITHM = dict(version=1, multiplier_a=747796405, addend=2891336453,
                 multiplier_b=277803737, xor_shift=16, mask=4294967295,
                 low_mask=65535, center=32768, denominator=32768,
                 seed_query=97, seed_key=17, seed_value=43)


def hashed(index, seed):
    state = ((index + seed) * ALGORITHM['multiplier_a'] + ALGORITHM['addend']) & ALGORITHM['mask']
    state = state ^ (state >> ALGORITHM['xor_shift'])
    state = (state * ALGORITHM['multiplier_b'] + ALGORITHM['addend']) & ALGORITHM['mask']
    state = state ^ (state >> ALGORITHM['xor_shift'])
    return ((state & ALGORITHM['low_mask']).to(torch.float32) - ALGORITHM['center']) / ALGORITHM['denominator']


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', nargs='+', type=int, default=[131072, 262144, 524288, 1048576])
    parser.add_argument('--output', type=Path,
        default=Path(__file__).resolve().parents[1] / 'test/ripple/fixtures/long_attention/reference.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--head-batch', type=int, default=5, help='Bound MATH backend repetition/scaled-key scratch')
    args = parser.parse_args()
    if any(n <= 0 or n > 1048576 for n in args.lengths):
        parser.error('lengths must be in1..1048576')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')
    dim, heads, kvheads = 128, 10, 1
    if not 1 <= args.head_batch <= heads:
        parser.error('--head-batch must be1..10')
    result = dict(torch_version=torch.__version__, device=args.device, attention_backend='math',
        operation='torch.nn.functional.scaled_dot_product_attention, forced SDPBackend.MATH',
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), algorithm=ALGORITHM,
        head_dim=dim, query_heads=heads, kv_heads=kvheads, query_tokens=1, head_batch=args.head_batch,
        first_key_position=0, window=0, tensor_layout='Julia(D,H,Q); key/value Julia(D,KV,T)', cases=[])
    if args.device == 'cuda':
        result['gpu'] = torch.cuda.get_device_name()
    for length in args.lengths:
        started = time.monotonic()
        query = hashed(torch.arange(dim * heads, dtype=torch.int64, device=args.device), ALGORITHM['seed_query']).reshape(1, heads, 1, dim)
        key = torch.empty((1, kvheads, length, dim), dtype=torch.float32, device=args.device)
        value = torch.empty_like(key)
        # Bound procedural-generation scratch, independently of context length.
        for first in range(0, length, 4096):
            stop = min(first + 4096, length)
            index = torch.arange(first * dim, stop * dim, dtype=torch.int64, device=args.device)
            key[:, :, first:stop, :] = hashed(index, ALGORITHM['seed_key']).reshape(1, 1, stop - first, dim)
            value[:, :, first:stop, :] = hashed(index, ALGORITHM['seed_value']).reshape(1, 1, stop - first, dim)
        probe_indices = sorted(set([0, 1, 17, 127, 128, 129, dim * length // 2, dim * length - 1]))
        probe_indices = [i for i in probe_indices if i < dim * length]
        key_probes = key.reshape(-1)[probe_indices].cpu().tolist()
        value_probes = value.reshape(-1)[probe_indices].cpu().tolist()
        # Public MATH SDPA may allocate an additional scaled-key copy. Process
        # independent query-head batches so peak remains below~10GiB at1Mi.
        if args.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        attention_started = time.monotonic()
        pieces = []
        for first_head in range(0, heads, args.head_batch):
            stop_head = min(first_head + args.head_batch, heads)
            count = stop_head - first_head
            repeated_key = key.repeat_interleave(count, dim=1)
            repeated_value = value.repeat_interleave(count, dim=1)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                pieces.append(F.scaled_dot_product_attention(query[:, first_head:stop_head], repeated_key,
                    repeated_value, attn_mask=None, dropout_p=0.0, is_causal=False))
            del repeated_key, repeated_value
        output = torch.cat(pieces, dim=1)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        attention_seconds = time.monotonic() - attention_started
        # The single query is at the last absolute position; all keys are causal.
        record = dict(length=length, query_position=length - 1,
            output_shape=[dim, heads, 1], output=output[0, :, 0, :].contiguous().cpu().reshape(-1).tolist(),
            probe_indices_zero_based=probe_indices, key_probes=key_probes, value_probes=value_probes,
            query_probes=query.reshape(-1)[:16].cpu().tolist(),
            attention_seconds=attention_seconds, total_seconds=time.monotonic() - started)
        if args.device == 'cuda':
            record['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['cases'].append(record)
        write_json(args.output, result)
        print(json.dumps({k:v for k,v in record.items() if k not in ('output','key_probes','value_probes','query_probes')}) ,flush=True)
        del output, query, key, value, pieces, index
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(dict(output=str(args.output), cases=len(result['cases']))))


if __name__ == '__main__':
    main()
