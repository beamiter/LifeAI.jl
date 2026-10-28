#!/usr/bin/env python3
"""Run one calibrated quality case and validate the entire final JSON answer."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def strict_answer(text, expected):
    final = text.strip()
    if final.startswith('<think>'):
        if '</think>' not in final:
            return {'final_answer_text': '', 'strict_json_pass': False,
                    'error': 'Unterminated think block.', 'identifier_matches': {k: False for k in expected}}
        final = final.split('</think>', 1)[1].strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', final, re.DOTALL | re.IGNORECASE)
    if fenced:
        final = fenced.group(1).strip()
    result = {'final_answer_text': final, 'strict_json_pass': False,
              'identifier_matches': {k: False for k in expected}}
    try:
        parsed = json.loads(final)
        result['parsed_final_json'] = parsed
        result['identifier_matches'] = {key: isinstance(parsed, dict) and parsed.get(key) == value
                                        for key, value in expected.items()}
        result['strict_json_pass'] = isinstance(parsed, dict) and parsed == expected
    except (ValueError, TypeError) as error:
        result['error'] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, choices=(8192, 32768, 262144), required=True)
    parser.add_argument('--style', choices=('repeated', 'varied'), required=True)
    parser.add_argument('--label', default='unspecified')
    parser.add_argument('--corpus-dir', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--driver', type=Path, default=ROOT / 'benchmark_http.py')
    parser.add_argument('--url', default='http://127.0.0.1:11435')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest or args.corpus_dir / 'quality-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    case = next(case for case in manifest['cases']
                if case['total_context_budget'] == args.budget and case['style'] == args.style)
    command = [sys.executable, str(args.driver), '--url', args.url,
        '--manifest', str(manifest_path), '--prompt', str(args.corpus_dir / case['file']),
        '--max-tokens', str(case['reserved_output_tokens']), '--output-dir', str(args.output_dir)]
    result = subprocess.run(command, capture_output=True, text=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'driver.stdout.log').write_text(result.stdout)
    (args.output_dir / 'driver.stderr.log').write_text(result.stderr)
    text_path = args.output_dir / 'answer.txt'
    text = text_path.read_text() if text_path.exists() else ''
    driver_summary_path = args.output_dir / 'summary.json'
    driver_summary = json.loads(driver_summary_path.read_text()) if driver_summary_path.exists() else {}
    evaluation = strict_answer(text, manifest['expected_answer'])
    summary = {'label': args.label, 'style': args.style, 'context_budget': args.budget,
        'prompt_tokens': case['prompt_tokens_official_tokenizers'],
        'reserved_output_tokens': case['reserved_output_tokens'],
        'actual_output_tokens': driver_summary.get('actual_output_tokens'),
        'transport_status': driver_summary.get('status'),
        'first_token_seconds': driver_summary.get('first_token_seconds'),
        'total_seconds': driver_summary.get('total_seconds'),
        'peak_gpu_memory_MiB': driver_summary.get('peak_gpu_memory_MiB'),
        'done_reason': (driver_summary.get('final_packet') or {}).get('done_reason'),
        'expected_answer': manifest['expected_answer'], **evaluation}
    (args.output_dir / 'quality-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
