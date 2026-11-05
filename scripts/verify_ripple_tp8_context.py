#!/usr/bin/env python3
"""Check an approximately 8K input and a complete final answer through DLC HTTP.

Run on DSW with /isaac-sim/python.sh for the checkpoint's tokenizers dependency.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from tokenizers import Tokenizer
from query_ripple_tp8 import discover_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    ready = json.loads((args.run_dir / 'results/ready.json').read_text())
    key = (args.run_dir / 'credentials/api_key').read_text().strip()
    if args.tokenizer is None:
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/deployment/dlc_ripple_tp8.json').read_text())
        args.tokenizer = Path(config['model']) / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    def content(count):
        return 'Context data:\n' + 'This is filler text. ' * count + '\nEnd of data. Reply with exactly OK.'

    def prompt(count):
        return '<|im_start|>user\n' + content(count) + '<|im_end|>\n<|im_start|>assistant\n'

    low, high = 0, 10000
    while low + 1 < high:
        mid = (low + high) // 2
        if len(tokenizer.encode(prompt(mid)).ids) <= 8000:
            low = mid
        else:
            high = mid
    assert ready['max_context'] >= 8128 and ready['max_output'] >= 128
    payload = {'model': ready['model_id'], 'messages': [{'role': 'user', 'content': content(low)}],
               'max_tokens': 128, 'temperature': 0, 'stream': False}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint, _ = discover_endpoint(ready, opener)
    request = urllib.request.Request(endpoint + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    started = time.monotonic()
    with opener.open(request, timeout=600) as response:
        status, body = response.status, json.load(response)
    assert status == 200 and 7900 <= body['usage']['prompt_tokens'] <= 8000
    assert body['lifeai']['tensor_parallel'] == 8
    assert body['choices'][0]['finish_reason'] == 'stop'
    assert body['choices'][0]['message']['content'].rsplit('</think>', 1)[-1].strip() == 'OK'
    result = {'passed': True, 'endpoint': endpoint, 'target_prompt_tokens': 8000, 'request_max_tokens': 128,
        'scope': 'Long-context capacity and final OK check; not a general reasoning benchmark',
        'status': status, 'seconds': time.monotonic() - started, 'response': body}
    (args.run_dir / 'results/http-long-context.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
