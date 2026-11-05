#!/usr/bin/env python3
"""Verify the running TP8 service from DSW; never print its bearer key."""
import argparse
import datetime
import json
from pathlib import Path
import urllib.error
import urllib.request
from query_ripple_tp8 import discover_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    ready = json.loads((args.run_dir / 'results/ready.json').read_text())
    endpoint, _ = discover_endpoint(ready, opener)
    key = (args.run_dir / 'credentials/api_key').read_text().strip()
    checks = []

    def request(name, path, expected, payload=None, authorized=True, method=None):
        headers = {'Content-Type': 'application/json'}
        if authorized:
            headers['Authorization'] = 'Bearer ' + key
        req = urllib.request.Request(endpoint + path, method=method,
            data=None if payload is None else json.dumps(payload).encode(), headers=headers)
        try:
            with opener.open(req, timeout=600) as response:
                status, body = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, body = error.code, json.load(error)
        assert status == expected, (name, status, body)
        checks.append({'name': name, 'status': status, 'passed': True})
        return body

    health = request('health', '/healthz', 200, authorized=False)
    assert health['status'] == 'ready' and health['tensor_parallel'] == 8
    request('unauthenticated_request', '/v1/models', 401, authorized=False)
    models = request('models', '/v1/models', 200)
    assert any(model['id'] == ready['model_id'] for model in models['data'])
    base = {'model': ready['model_id'], 'messages': [{'role': 'user', 'content': 'Reply with exactly OK.'}],
            'temperature': 0, 'max_tokens': 64, 'stream': False}
    for name, override in [('streaming_rejected', {'stream': True}), ('empty_messages', {'messages': []}),
            ('zero_tokens', {'max_tokens': 0}), ('output_limit', {'max_tokens': ready['max_output'] + 1}),
            ('unknown_model', {'model': 'unknown'}), ('multiple_choices', {'n': 2}),
            ('context_limit', {'messages': [{'role': 'user', 'content': 'x ' * ready['max_context']}], 'max_tokens': 1})]:
        request(name, '/v1/chat/completions', 400, {**base, **override})
    request('unknown_route', '/unknown', 404)
    body = request('real_completion', '/v1/chat/completions', 200, base)
    assert body['object'] == 'chat.completion'
    assert body['choices'][0]['message']['content'].strip() == 'OK'
    assert body['choices'][0]['finish_reason'] == 'stop'
    assert body['usage']['completion_tokens'] == 2
    assert body['lifeai']['tensor_parallel'] == 8 and body['lifeai']['weight_dtype'] == 'BF16'
    result = {'endpoint': endpoint, 'code_revision': ready['code_revision'], 'all_passed': True,
        'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'checks': checks, 'completion': body}
    # A multi-interface DLC pod may initially advertise a link-local address.
    # Save the endpoint only after the authenticated completion proved it from DSW.
    if ready['endpoint'] != endpoint:
        ready['initial_endpoint'] = ready['endpoint']
        ready['endpoint'] = endpoint
        ready['endpoint_verified_at'] = result['checked_at']
        path = args.run_dir / 'results/ready.json'
        temporary = path.with_suffix('.verified.tmp')
        temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    (args.run_dir / 'results/http-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
