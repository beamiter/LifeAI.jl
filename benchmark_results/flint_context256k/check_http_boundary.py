#!/usr/bin/env python3
"""Verify exact-capacity-plus-one rejection before starting a long inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--router-url', default='http://127.0.0.1:11435')
    parser.add_argument('--worker-url', default='http://127.0.0.1:11437')
    parser.add_argument('--prompt', type=Path, default=ROOT / 'flint-262143-prompt-1-output.txt')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'corpus-manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'boundary-check.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    case = next(case for case in manifest['cases'] if case['prompt_tokens_official_tokenizers'] == 262143)
    content = args.prompt.read_bytes()
    assert hashlib.sha256(content).hexdigest() == case['sha256'], 'Corpus content changed.'
    request = {'model': 'RecursiveSI/flint', 'messages': [{'role': 'user', 'content': content.decode()}],
               'stream': True, 'options': {'num_predict': 2, 'temperature': 0}}
    body = json.dumps(request).encode()
    summary = {'prompt_tokens': 262143, 'reserved_output_tokens': 2, 'total_budget': 262145,
               'context_tokens': 262144, 'expected_worker_code': 'context_length_exceeded',
               'prompt_sha256': case['sha256'], 'request_body_bytes': len(body), 'results': []}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    opener = build_opener(ProxyHandler({}))
    def health(url):
        with opener.open(url.rstrip('/') + '/healthz', timeout=10) as response:
            return json.load(response)
    # Preflight both services before either POST. An old 2048 service must never
    # be mistaken for a successful 256K boundary check.
    for url in (args.router_url, args.worker_url):
        value = health(url)
        assert value.get('ready') is True and value.get('model') == 'RecursiveSI/flint'
        assert value.get('context_tokens') == 262144, 'Refusing POST: deployed context is not exactly 262144.'
        assert value.get('active_requests', 0) == 0, 'Refusing POST while another request is active.'
    for label, url in (('worker', args.worker_url), ('router', args.router_url)):
        started = time.monotonic()
        post = Request(url.rstrip('/') + '/api/chat', body,
                       {'Content-Type': 'application/json'}, method='POST')
        try:
            with opener.open(post, timeout=60) as response:
                raise AssertionError(f'{label} unexpectedly accepted overflow: HTTP {response.status}')
        except HTTPError as error:
            payload = json.load(error)
            assert error.code == 400, payload
            if label == 'worker':
                assert payload.get('code') == 'context_length_exceeded', payload
            message = payload.get('error', '')
            assert all(value in message for value in
                       ('Prompt (262143 tokens)', 'output (2)', 'exceeds 262144 tokens')), payload
            summary['results'].append({'service': label, 'status': error.code,
                'seconds': time.monotonic() - started, 'payload': payload})
    after = health(args.router_url)
    assert after.get('ready') is True and after.get('active_requests') == 0
    summary['status'] = 'passed'
    summary['inference_started'] = False
    summary['no_inference_evidence'] = ('Worker returned context_length_exceeded from prepare_request, '
        'before generate or NDJSON headers; router returned the same HTTP400 and released its reservation.')
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
