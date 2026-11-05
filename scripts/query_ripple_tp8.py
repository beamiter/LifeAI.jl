#!/usr/bin/env python3
"""Query a deployed Ripple TP8 service from DSW without printing its API key."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--prompt', default='Reply with exactly OK.')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--health-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ready = json.loads((args.run_dir / 'results/ready.json').read_text())
    candidates = [ready['endpoint']] + [f'http://{ip}:{ready["port"]}' for ip in ready.get('addresses', [])]
    endpoint = None
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            with urllib.request.urlopen(candidate + '/healthz', timeout=5) as response:
                health = json.load(response)
            if health['status'] == 'ready':
                endpoint = candidate
                break
        except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
            errors.append(f'{candidate}: {error}')
    if endpoint is None:
        raise RuntimeError('No healthy endpoint reachable from DSW: ' + '; '.join(errors))
    if args.health_only:
        result = {'endpoint': endpoint, 'health': health}
    else:
        key = (args.run_dir / 'credentials/api_key').read_text().strip()
        payload = {'model': ready['model_id'], 'messages': [{'role': 'user', 'content': args.prompt}],
                   'temperature': 0, 'max_tokens': args.max_tokens, 'stream': False}
        request = urllib.request.Request(endpoint + '/v1/chat/completions', data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.load(response)
        result = {'endpoint': endpoint, 'response': body}
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
