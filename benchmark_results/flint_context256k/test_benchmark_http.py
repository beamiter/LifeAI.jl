#!/usr/bin/env python3
"""Test the benchmark transport against a CPU-only loopback fixture."""
import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get('FLINT_CORPUS_DIR', ROOT))


class PublishedBudgetTests(unittest.TestCase):
    def test_large_reserved_output_uses_published_limit_and_accepts_early_eos(self):
        for maximum in (1024, 4096):
            posted = []
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def send_json(self, payload):
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    self.send_json({'ready': True, 'model': 'RecursiveSI/flint',
                                    'context_tokens': 262144, 'max_new_tokens': maximum})

                def do_POST(self):
                    posted.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                    self.send_json({'response': 'ok', 'done': True, 'prompt_eval_count': 2,
                                    'eval_count': 1, 'done_reason': 'stop'})

            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    directory = Path(directory)
                    prompt = directory / 'fixture.txt'
                    prompt.write_text('fixture')
                    manifest = directory / 'manifest.json'
                    manifest.write_text(json.dumps({'context_tokens': 262144, 'expected_answer': {},
                        'cases': [{'file': prompt.name, 'sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
                                   'requested_output_tokens': 2048, 'prompt_tokens_official_tokenizers': 2}]}))
                    output = directory / 'run'
                    result = subprocess.run([sys.executable, str(ROOT / 'benchmark_http.py'),
                        '--url', f'http://127.0.0.1:{server.server_port}', '--no-gpu-sampling',
                        '--prompt', str(prompt), '--manifest', str(manifest), '--output-dir', str(output)],
                        capture_output=True, text=True)
                    if maximum == 1024:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(posted, [])
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(posted[0]['options']['num_predict'], 2048)
                        summary = json.loads((output / 'summary.json').read_text())
                        self.assertEqual(summary['reserved_output_tokens'], 2048)
                        self.assertEqual(summary['actual_output_tokens'], 1)
                        self.assertEqual(summary['status'], 'complete')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


@unittest.skipUnless((CORPUS / 'flint-262143-prompt-1-output.txt').exists(),
                     'Generate the corpus or set FLINT_CORPUS_DIR first.')
class BenchmarkTests(unittest.TestCase):
    def test_boundary_refuses_old_context_and_checks_both_400s(self):
        for context in (2048, 262144):
            captured = []
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def send_json(self, status, payload):
                    data = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def do_GET(self):
                    self.send_json(200, {'ready': True, 'model': 'RecursiveSI/flint',
                                        'context_tokens': context, 'active_requests': 0})

                def do_POST(self):
                    captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                    payload = {'error': 'Prompt (262143 tokens) plus output (2) exceeds 262144 tokens'}
                    if self.path.startswith('/worker/'):
                        payload['code'] = 'context_length_exceeded'
                    self.send_json(400, payload)

            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / 'boundary.json'
                    url = f'http://127.0.0.1:{server.server_port}'
                    result = subprocess.run([sys.executable, str(ROOT / 'check_http_boundary.py'),
                        '--router-url', url + '/router', '--worker-url', url + '/worker',
                        '--prompt', str(CORPUS / 'flint-262143-prompt-1-output.txt'),
                        '--manifest', str(CORPUS / 'corpus-manifest.json'),
                        '--output', str(output)], capture_output=True, text=True)
                    if context == 2048:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('Refusing POST', result.stderr)
                        self.assertEqual(captured, [])
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(len(captured), 2)
                        self.assertTrue(all(p['options']['num_predict'] == 2 for p in captured))
                        summary = json.loads(output.read_text())
                        self.assertEqual(summary['status'], 'passed')
                        self.assertEqual([r['status'] for r in summary['results']], [400, 400])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_real_stream_records_capacity_progress_timing_and_answer(self):
        manifest = json.loads((CORPUS / 'corpus-manifest.json').read_text())
        case = manifest['cases'][0]
        answer = json.dumps(manifest['expected_answer'])
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                data = json.dumps({'ready': True, 'model': 'RecursiveSI/flint',
                                   'context_tokens': 262144, 'max_new_tokens': 512}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append(payload)
                packets = [
                    {'response': '', 'done': False, 'phase': 'prefill',
                     'prefill_processed': 4096, 'prompt_eval_count': 261632},
                    {'response': '', 'done': False, 'phase': 'prefill',
                     'prefill_processed': 261632, 'prompt_eval_count': 261632},
                    {'response': answer, 'done': False},
                    {'response': '', 'done': True, 'prompt_eval_count': 261632,
                     'eval_count': 40, 'done_reason': 'stop'},
                ]
                data = ('\n'.join(json.dumps(packet) for packet in packets) + '\n').encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'run'
                result = subprocess.run([sys.executable, str(ROOT / 'benchmark_http.py'),
                    '--url', f'http://127.0.0.1:{server.server_port}', '--no-gpu-sampling',
                    '--prompt', str(CORPUS / case['file']),
                    '--manifest', str(CORPUS / 'corpus-manifest.json'),
                    '--output-dir', str(output), '--timeout', '5'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['status'], 'complete')
                self.assertEqual(summary['context_budget'], 262144)
                self.assertEqual(len(summary['prefill_events']), 2)
                self.assertGreaterEqual(summary['first_token_seconds'], summary['prefill_complete_seconds'])
                self.assertTrue(all(summary['retrieved_identifiers'].values()))
                self.assertEqual((output / 'answer.txt').read_text(), answer)
                self.assertEqual(len((output / 'events.ndjson').read_text().splitlines()), 4)
                self.assertTrue((output / 'final-packet.json').exists())
                self.assertEqual(captured[0]['options']['num_predict'], 512)
                self.assertEqual(captured[0]['messages'][0]['content'], (CORPUS / case['file']).read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
