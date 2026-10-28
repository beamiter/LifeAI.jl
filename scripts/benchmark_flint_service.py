#!/usr/bin/env python3
"""Measure the resident Flint worker without including weight load or TUI rendering."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request


def request(url, path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip('/') + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


def sample_gpu(stop, samples):
    fields = 'utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory,memory.used'
    while not stop.is_set():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=' + fields,
                                     '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=3, check=True)
            values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(',')]
            samples.append(dict(zip(fields.split(','), values)))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        stop.wait(0.5)


def run(args):
    with request(args.url, '/healthz') as response:
        health = json.load(response)
    if not health.get('ready') or health.get('model') != 'RecursiveSI/flint':
        raise RuntimeError('Select Flint and wait for ready before benchmarking; no model is switched automatically')
    if health.get('active_requests', 0):
        raise RuntimeError('Service is busy; wait for the current conversation to finish')
    runs, gpu = [], []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, gpu), daemon=True)
    if args.gpu_sample:
        sampler.start()
    try:
        for index in range(args.runs):
            started, first, final = time.perf_counter(), None, None
            pieces, events = [], []
            payload = dict(model='RecursiveSI/flint', messages=[dict(role='user', content=args.prompt)],
                           stream=True, options=dict(num_predict=args.tokens, temperature=0))
            with request(args.url, '/api/chat', payload, args.timeout) as response:
                for line in response:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    elapsed = time.perf_counter() - started
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    piece = event.get('response', '')
                    if piece:
                        first = elapsed if first is None else first
                        pieces.append(piece)
                        events.append(elapsed)
                    if event.get('done'):
                        final = event
                        break
            wall = time.perf_counter() - started
            if final is None:
                raise RuntimeError('Incomplete stream; no done record')
            count = final.get('eval_count', 0)
            total = final.get('total_duration', 0) / 1e9
            ttft = final.get('time_to_first_token')
            ttft = ttft / 1e9 if ttft is not None else first
            decode_seconds = max(0, total - ttft) if ttft is not None else None
            item = dict(run=index + 1, wall_seconds=wall, first_text_seconds=first,
                        server_seconds=total, server_ttft_seconds=ttft, tokens=count,
                        end_to_end_tokens_per_second=count / total if total else None,
                        decode_tokens_per_second=(count - 1) / decode_seconds
                        if count > 1 and decode_seconds else None,
                        text=''.join(pieces), event_seconds=events, final=final)
            runs.append(item)
            print(json.dumps({key: value for key, value in item.items()
                              if key not in ('text', 'event_seconds', 'final')}, ensure_ascii=False), flush=True)
    finally:
        stop.set()
        if sampler.is_alive():
            sampler.join(timeout=4)
    report = dict(schema_version=1, url=args.url, prompt=args.prompt, requested_tokens=args.tokens,
                  health=health, runs=runs, gpu_samples=gpu,
                  median_decode_tokens_per_second=statistics.median(
                      item['decode_tokens_per_second'] for item in runs if item['decode_tokens_per_second']))
    if gpu:
        report['gpu_mean'] = {key: statistics.mean(item[key] for item in gpu) for key in gpu[0]}
        report['gpu_max'] = {key: max(item[key] for item in gpu) for key in gpu[0]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:11435')
    p.add_argument('--prompt', default='请用中文解释什么是递归，并给出一个简单例子。')
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--runs', type=int, default=2)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--gpu-sample', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.runs < 1 or args.tokens < 2 or args.timeout <= 0:
        p.error('--runs >= 1, --tokens >= 2 and --timeout > 0 are required')
    run(args)
