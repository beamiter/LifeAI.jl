#!/usr/bin/env python3
"""Run a staged exact-capacity Flint corpus through the real streaming API."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
MAX_RECORD_BYTES = 1024 * 1024


def gpu_sampler(path, stopped, interval, samples, errors):
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['client_monotonic', 'gpu_timestamp', 'index', 'memory_used_MiB',
                         'memory_total_MiB', 'utilization_gpu_percent'])
        while not stopped.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    check=True, timeout=10)
                for row in csv.reader(result.stdout.splitlines()):
                    row = [value.strip() for value in row]
                    writer.writerow([time.monotonic(), *row])
                    if len(row) == 5:
                        samples.append({'index': int(row[1]), 'memory_used_MiB': float(row[2])})
                file.flush()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(str(error))
                return
            stopped.wait(interval)


def positive_seconds(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:11435')
    parser.add_argument('--prompt', type=Path,
                        default=ROOT / 'flint-261632-prompt-512-output.txt')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'corpus-manifest.json')
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--timeout', type=positive_seconds, default=7200)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gpu-interval', type=positive_seconds, default=2)
    parser.add_argument('--no-gpu-sampling', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate corpus and print request metadata only.')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    case = next(case for case in manifest['cases'] if case['file'] == args.prompt.name)
    raw_prompt = args.prompt.read_bytes()
    assert len(raw_prompt) <= 4 * 1024 * 1024
    assert hashlib.sha256(raw_prompt).hexdigest() == case['sha256'], 'Corpus content changed.'
    prompt = raw_prompt.decode('utf-8')
    count = args.max_tokens if args.max_tokens is not None else case['requested_output_tokens']
    assert count >= 1
    assert count + case['prompt_tokens_official_tokenizers'] <= manifest['context_tokens']
    payload = {'model': 'RecursiveSI/flint', 'messages': [{'role': 'user', 'content': prompt}],
               'stream': True, 'options': {'num_predict': count, 'temperature': 0}}
    body = json.dumps(payload, ensure_ascii=False).encode()
    assert len(body) <= 16 * 1024 * 1024
    metadata = {'url': args.url, 'prompt_file': str(args.prompt), 'prompt_sha256': case['sha256'],
                'prompt_utf8_bytes': len(raw_prompt), 'request_body_bytes': len(body),
                'expected_prompt_tokens': case['prompt_tokens_official_tokenizers'],
                'requested_output_tokens': count,
                'reserved_output_tokens': count,
                'context_budget': case['prompt_tokens_official_tokenizers'] + count,
                'expected_answer': manifest['expected_answer'], 'timeout_seconds': args.timeout,
                'gpu_memory_scope': 'Whole GPU, including all resident processes, sampled every configured interval.'}
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    output = args.output_dir or ROOT / ('run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'request-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    opener = build_opener(ProxyHandler({}))
    with opener.open(args.url.rstrip('/') + '/healthz', timeout=10) as response:
        health = json.load(response)
    (output / 'health-before.json').write_text(json.dumps(health, indent=2) + '\n')
    assert health.get('ready') is True and health.get('model') == 'RecursiveSI/flint', 'Flint is not resident and ready.'
    assert health['context_tokens'] >= metadata['context_budget'], 'Published context is too short.'
    assert health['max_new_tokens'] >= count, 'Published output budget is too short.'
    stopped = threading.Event()
    samples, sampler_errors = [], []
    sampler = None
    if not args.no_gpu_sampling:
        sampler = threading.Thread(target=gpu_sampler,
            args=(output / 'gpu.csv', stopped, args.gpu_interval, samples, sampler_errors), daemon=True)
        sampler.start()
    started = time.monotonic()
    summary = {'status': 'running', **metadata, 'prefill_events': [],
               'first_token_seconds': None, 'prefill_complete_seconds': None}
    final_packet = None
    answer = []
    timer = None
    exit_code = 0
    try:
        request = Request(args.url.rstrip('/') + '/api/chat', body,
                          {'Content-Type': 'application/json'}, method='POST')
        with opener.open(request, timeout=args.timeout) as response, (output / 'events.ndjson').open('w') as events:
            # urllib's timeout is per socket operation; add an overall deadline.
            underlying_socket = response.fp.raw._sock
            def deadline():
                try:
                    underlying_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            timer = threading.Timer(max(0.001, args.timeout - (time.monotonic() - started)), deadline)
            timer.daemon = True
            timer.start()
            while True:
                record = response.readline(MAX_RECORD_BYTES + 1)
                elapsed = time.monotonic() - started
                if elapsed >= args.timeout:
                    raise TimeoutError('Overall benchmark deadline exceeded.')
                if not record:
                    raise RuntimeError('Stream ended without a done packet.')
                if len(record) > MAX_RECORD_BYTES:
                    raise RuntimeError('Stream record exceeds 1 MiB.')
                if not record.strip():
                    continue
                packet = json.loads(record)
                events.write(json.dumps({'client_elapsed_seconds': elapsed, 'packet': packet}, ensure_ascii=False) + '\n')
                events.flush()
                if packet.get('phase') == 'prefill':
                    processed, total = packet['prefill_processed'], packet['prompt_eval_count']
                    assert total == metadata['expected_prompt_tokens'], 'Worker prompt token count differs from calibration.'
                    assert 0 <= processed <= total
                    progress = {'elapsed_seconds': elapsed, 'processed': processed, 'total': total}
                    summary['prefill_events'].append(progress)
                    print(json.dumps({'phase': 'prefill', **progress}), flush=True)
                    if processed == total:
                        summary['prefill_complete_seconds'] = elapsed
                piece = packet.get('response', '')
                if piece:
                    answer.append(piece)
                    if summary['first_token_seconds'] is None:
                        summary['first_token_seconds'] = elapsed
                        print(json.dumps({'first_token_seconds': elapsed}), flush=True)
                if packet.get('done') is True:
                    final_packet = packet
                    (output / 'final-packet.json').write_text(json.dumps(packet, indent=2, ensure_ascii=False) + '\n')
                    if packet.get('error'):
                        raise RuntimeError(str(packet['error']))
                    assert packet.get('prompt_eval_count') == metadata['expected_prompt_tokens']
                    summary['status'] = 'complete'
                    break
    except HTTPError as error:
        summary['status'] = 'error'
        summary['error'] = f'HTTP {error.code}: {error.read(MAX_RECORD_BYTES).decode(errors="replace")}'
        exit_code = 1
    except Exception as error:
        summary['status'] = 'error'
        summary['error'] = f'{type(error).__name__}: {error}'
        exit_code = 1
    finally:
        if timer is not None:
            timer.cancel()
        summary['total_seconds'] = time.monotonic() - started
        stopped.set()
        if sampler is not None:
            sampler.join(timeout=11)
        summary['peak_gpu_memory_MiB'] = max((sample['memory_used_MiB'] for sample in samples), default=None)
        summary['gpu_samples'] = len(samples)
        summary['gpu_sampler_errors'] = sampler_errors
        summary['final_packet'] = final_packet
        summary['actual_output_tokens'] = final_packet.get('eval_count') if final_packet else None
        text = ''.join(answer)
        summary['retrieved_identifiers'] = {key: value in text for key, value in manifest['expected_answer'].items()}
        (output / 'answer.txt').write_text(text)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'output_dir': str(output), **summary}, ensure_ascii=False), flush=True)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
