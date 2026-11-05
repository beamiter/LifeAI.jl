import json
from pathlib import Path
import tempfile
import socket
import sys
import struct
import threading
import time
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import ripple_worker as ripple
import server as router


class Fixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.requests = []
        self.mode = 'ok'
        self.timing = {}
        self.async_enabled = False
        self.async_states = []
        self.async_polls = 0
        self.cancelled_paths = []
        self.async_result = dict(model='lifeai-ripple65b-tp8', choices=[dict(
            message=dict(role='assistant', content='异步 OK'), finish_reason='stop')],
            usage=dict(prompt_tokens=100000, completion_tokens=3))
        self.entered, self.unblock = threading.Event(), threading.Event()
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, code, value):
                raw = json.dumps(value, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == '/healthz':
                    self.reply(200, dict(status='ready', model='lifeai-ripple65b-tp8', tensor_parallel=8,
                                         async_jobs=owner.async_enabled))
                elif self.headers.get('Authorization') != 'Bearer local-test-secret':
                    self.reply(401, dict(error=dict(message='Unauthorized')))
                elif self.path.startswith('/v1/chat/completions/jobs/'):
                    index = min(owner.async_polls, len(owner.async_states) - 1)
                    owner.async_polls += 1
                    packet = dict(id='owned-1', **owner.async_states[index])
                    if packet.get('status') == 'completed':
                        packet['result'] = owner.async_result
                    self.reply(200, packet)
                else:
                    self.reply(200, dict(data=[dict(id='lifeai-ripple65b-tp8')]))

            def do_DELETE(self):
                if self.headers.get('Authorization') != 'Bearer local-test-secret':
                    self.reply(401, dict(error='Unauthorized'))
                else:
                    owner.cancelled_paths.append(self.path)
                    self.reply(200, dict(id='owned-1', status='cancelled'))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append((self.path, self.headers.get('Authorization'), body))
                if self.path == '/v1/chat/completions/jobs':
                    self.reply(202, dict(id='owned-1', status='queued'))
                    return
                if owner.mode == 'blocked':
                    owner.entered.set()
                    owner.unblock.wait(3)
                if owner.mode == 'error':
                    self.reply(400, dict(error=dict(message='request exceeds context limit')))
                    return
                reply = dict(model='lifeai-ripple65b-tp8', choices=[dict(
                    message=dict(role='assistant', content='<think>分析</think>\n你好🌟'), finish_reason='stop')],
                    usage=dict(prompt_tokens=42, completion_tokens=5), lifeai=owner.timing)
                if owner.mode == 'wrong_model':
                    reply['model'] = 'other'
                if owner.mode == 'incomplete':
                    reply['choices'][0]['finish_reason'] = None
                if owner.mode == 'invalid_usage':
                    reply['usage']['completion_tokens'] = True
                self.reply(200, reply)

        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.thread.start()
        (self.root / 'results').mkdir()
        (self.root / 'credentials').mkdir()
        (self.root / 'credentials/api_key').write_text('local-test-secret\n')
        self.ready = dict(phase='ready', model=ripple.MODEL, model_id='lifeai-ripple65b-tp8',
                          endpoint=f'http://127.0.0.1:{self.upstream.server_port}', max_context=8192,
                          max_output=512, world_size=8)
        (self.root / 'results/ready.json').write_text(json.dumps(self.ready))
        self.receipt = self.root / 'deployment.json'
        self.receipt.write_text(json.dumps(dict(run_dir=str(self.root))))
        self.backend = ripple.RippleBackend(self.receipt, timeout=3)
        self.worker = ThreadingHTTPServer(('127.0.0.1', 0), ripple.handler_for(self.backend))
        self.worker_thread = threading.Thread(target=self.worker.serve_forever, daemon=True)
        self.worker_thread.start()
        self.opener = build_opener(ProxyHandler({}))

    def request(self, payload=None, path='/api/chat'):
        request = Request(f'http://127.0.0.1:{self.worker.server_port}' + path,
                          data=None if payload is None else json.dumps(payload).encode())
        try:
            with self.opener.open(request, timeout=5) as response:
                return response.status, response.headers['Content-Type'], response.read()
        except HTTPError as error:
            return error.code, error.headers['Content-Type'], error.read()

    def close(self):
        self.unblock.set()
        for server, thread in [(self.worker, self.worker_thread), (self.upstream, self.thread)]:
            server.shutdown()
            server.server_close()
            thread.join(3)
        self.temp.cleanup()


def payload(stream=True):
    return dict(model=ripple.MODEL, stream=stream, messages=[
        dict(role='system', content='中文回答'), dict(role='user', content='记住松鼠'),
        dict(role='assistant', content='OK'), dict(role='user', content='暗号是什么？')],
        options=dict(num_predict=8, temperature=0.3))


class RippleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.addCleanup(self.fixture.close)

    def test_health_publishes_limits_but_never_the_key(self):
        code, _, raw = self.fixture.request(path='/healthz')
        value = json.loads(raw)
        self.assertEqual(code, 200)
        self.assertTrue(value['ready'])
        self.assertEqual(value['model'], ripple.MODEL)
        self.assertEqual((value['context_tokens'], value['max_new_tokens']), (8192, 512))
        self.assertFalse(value['streaming'])
        self.assertNotIn(b'local-test-secret', raw)
        self.assertNotIn('local-test-secret', repr(self.fixture.backend.connect()))
        self.assertEqual(ripple.RippleBackend(self.fixture.receipt).timeout, 21600)

    def test_native_stream_is_one_complete_event_and_forwards_all_roles(self):
        for stream in (True, False):
            with self.subTest(stream=stream):
                code, content_type, raw = self.fixture.request(payload(stream))
                self.assertEqual(code, 200)
                self.assertEqual(content_type, 'application/x-ndjson' if stream else 'application/json')
                self.assertEqual(len(raw.splitlines()), 1)
                event = json.loads(raw)
                self.assertEqual(event['response'], '<think>分析</think>\n你好🌟')
                self.assertEqual(event['model'], ripple.MODEL)
                self.assertTrue(event['done'])
                self.assertEqual(event['done_reason'], 'stop')
                self.assertEqual((event['eval_count'], event['prompt_eval_count']), (5, 42))
                self.assertNotIn('eval_duration', event)
                self.assertGreater(event['total_duration'], 0)
                path, auth, body = self.fixture.requests[-1]
                self.assertEqual(path, '/v1/chat/completions')
                self.assertEqual(auth, 'Bearer local-test-secret')
                self.assertIs(body['stream'], False)
                self.assertEqual(body['messages'], payload()['messages'])
                self.assertEqual((body['max_tokens'], body['temperature']), (8, 0.3))
                self.assertNotIn(b'local-test-secret', raw)

    def test_long_utf8_request_above_old_four_mib_limit_is_preserved(self):
        value = payload(False)
        content = '长文' * 700000
        self.assertGreater(len(content.encode()), 4 * 1024 * 1024)
        value['messages'] = [dict(role='user', content=content)]
        self.assertEqual(self.fixture.request(value)[0], 200)
        self.assertEqual(self.fixture.requests[-1][2]['messages'][0]['content'], content)

    def test_transport_limits_and_small_patched_boundaries(self):
        health = json.loads(self.fixture.request(path='/healthz')[2])
        self.assertEqual(health['max_input_bytes'], 64 * 1024 * 1024)
        self.assertEqual(health['max_request_bytes'], 128 * 1024 * 1024)
        self.assertEqual(health['max_reply_bytes'], 16 * 1024 * 1024)
        self.assertEqual(ripple.UPSTREAM_BODY, 128 * 1024 * 1024)
        with patch.object(ripple, 'MAX_MESSAGE_CHARS', 8), patch.object(ripple, 'MAX_TOTAL_MESSAGE_CHARS', 12):
            for messages in ([dict(role='user', content='x' * 9)],
                             [dict(role='user', content='x' * 7), dict(role='user', content='y' * 6)]):
                self.assertEqual(self.fixture.request({**payload(), 'messages': messages})[0], 413)
        with patch.object(ripple, 'UPSTREAM_BODY', 32):
            self.assertEqual(self.fixture.request(payload())[0], 413)
        self.assertEqual(self.fixture.requests, [])

    def test_bad_options_fail_before_chat(self):
        for override in [dict(options=dict(num_predict=True)), dict(options=dict(num_predict=513)),
                         dict(options=dict(temperature=float('nan'))), dict(messages=[]),
                         dict(messages=[dict(role='tool', content='x')]), dict(stream='yes')]:
            with self.subTest(override=override):
                self.assertEqual(self.fixture.request({**payload(), **override})[0], 400)
        self.assertEqual(self.fixture.requests, [])

    def test_upstream_error_preserves_context_message_and_releases_busy(self):
        self.fixture.mode = 'error'
        code, _, raw = self.fixture.request(payload())
        self.assertEqual(code, 400)
        self.assertIn(b'context', raw)
        self.assertNotIn(b'local-test-secret', raw)
        self.fixture.mode = 'ok'
        self.assertEqual(self.fixture.request(payload())[0], 200)

    def test_incomplete_wrong_model_and_invalid_usage_do_not_emit_success(self):
        for mode in ('incomplete', 'wrong_model', 'invalid_usage'):
            with self.subTest(mode=mode):
                self.fixture.mode = mode
                code, _, raw = self.fixture.request(payload())
                self.assertEqual(code, 502)
                self.assertNotIn('done', json.loads(raw))
        self.fixture.mode = 'ok'
        self.assertEqual(self.fixture.request(payload())[0], 200)

    def test_authentication_failure_and_missing_receipt_are_actionable(self):
        (self.fixture.root / 'credentials/api_key').write_text('wrong-secret')
        code, _, raw = self.fixture.request(path='/healthz')
        self.assertEqual(code, 200)
        self.assertFalse(json.loads(raw)['ready'])
        self.assertIn(b'authentication', raw)
        self.assertNotIn(b'wrong-secret', raw)
        self.fixture.receipt.unlink()
        code, _, raw = self.fixture.request(path='/healthz')
        self.assertEqual(code, 200)
        self.assertFalse(json.loads(raw)['ready'])
        self.assertIn(b'deployment', raw)

    def test_parallel_chat_rejected_then_backend_recovers(self):
        self.fixture.mode = 'blocked'
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(self.fixture.request, payload())
            self.assertTrue(self.fixture.entered.wait(3))
            self.assertEqual(self.fixture.request(payload())[0], 409)
            self.fixture.unblock.set()
            self.assertEqual(pending.result(5)[0], 200)
        self.fixture.mode = 'ok'
        self.assertEqual(self.fixture.request(payload())[0], 200)

    def test_verified_decode_timing_is_forwarded_without_inventing_a_rate(self):
        for value in (7.5, float('nan'), True, 'fast'):
            self.fixture.timing = dict(decode_tokens_per_second=value, prefill_seconds=.2)
            code, _, raw = self.fixture.request(payload())
            self.assertEqual(code, 200)
            event = json.loads(raw)
            self.assertEqual('decode_tokens_per_second' in event['lifeai'], type(value) is float and value==7.5)
            self.assertEqual(event['lifeai']['prefill_seconds'], .2)
            self.assertNotIn('eval_duration', event)
        self.fixture.ready['projection_dtype'] = 'Float32'
        (self.fixture.root / 'results/ready.json').write_text(json.dumps(self.fixture.ready))
        self.assertEqual(json.loads(self.fixture.request(payload())[2])['lifeai']['weight_dtype'], 'Float32')

    def enable_async(self, states):
        self.fixture.async_enabled = True
        self.fixture.async_states = states
        self.fixture.async_polls = 0

    def test_async_stream_reports_prefill_progress_then_complete_result(self):
        self.enable_async([dict(status='queued'),
            dict(status='running', phase='prefill', prefill_processed=256, prompt_eval_count=100000),
            dict(status='running', phase='prefill', prefill_processed=512, prompt_eval_count=100000),
            dict(status='running', phase='decode'), dict(status='completed')])
        with patch.object(ripple, 'POLL_INTERVAL', .001):
            code, content_type, raw = self.fixture.request(payload(True))
        self.assertEqual(code, 200)
        self.assertEqual(content_type, 'application/x-ndjson')
        packets = [json.loads(line) for line in raw.splitlines()]
        progress = [packet for packet in packets if packet.get('phase') == 'prefill']
        self.assertEqual([packet['prefill_processed'] for packet in progress], [256, 512])
        self.assertTrue(all(packet['response'] == '' and not packet['done'] for packet in progress))
        self.assertEqual(packets[-1]['response'], '异步 OK')
        self.assertTrue(packets[-1]['done'])
        self.assertEqual(self.fixture.cancelled_paths, [])
        path, auth, body = self.fixture.requests[-1]
        self.assertEqual(path, '/v1/chat/completions/jobs')
        self.assertEqual(auth, 'Bearer local-test-secret')
        self.assertFalse(body['stream'])

    def test_async_nonstream_waits_and_returns_single_native_json(self):
        self.enable_async([dict(status='running', phase='prefill', prefill_processed=128, prompt_eval_count=1000),
                           dict(status='completed')])
        with patch.object(ripple, 'POLL_INTERVAL', .001):
            code, content_type, raw = self.fixture.request(payload(False))
        self.assertEqual((code, content_type), (200, 'application/json'))
        self.assertEqual(json.loads(raw)['response'], '异步 OK')
        self.assertEqual(self.fixture.cancelled_paths, [])

    def test_async_failure_is_not_a_success_event_and_releases_busy(self):
        self.enable_async([dict(status='running', phase='decode'), dict(status='failed', error='GPU worker failed')])
        with patch.object(ripple, 'POLL_INTERVAL', .001):
            code, _, raw = self.fixture.request(payload(True))
        self.assertEqual(code, 200)
        packets = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(packets[-1]['error'], 'GPU worker failed')
        self.assertTrue(packets[-1]['done'])
        self.assertNotIn('eval_count', packets[-1])
        self.assertNotIn(b'local-test-secret', raw)
        self.assertEqual(self.fixture.cancelled_paths, ['/v1/chat/completions/jobs/owned-1'])
        self.enable_async([dict(status='completed')])
        self.assertEqual(self.fixture.request(payload(False))[0], 200)

    def test_async_disconnect_cancels_only_owned_job(self):
        self.enable_async([dict(status='running', phase='prefill', prefill_processed=128, prompt_eval_count=1000)])
        def disconnect(_packet):
            raise BrokenPipeError('client disconnected')
        with self.assertRaises(BrokenPipeError):
            self.fixture.backend.generate(payload(), on_progress=disconnect)
        self.assertEqual(self.fixture.cancelled_paths, ['/v1/chat/completions/jobs/owned-1'])
        self.assertTrue(self.fixture.backend.health()['ready'])
        self.enable_async([dict(status='completed')])
        self.assertEqual(self.fixture.request(payload(False))[0], 200)

    def test_async_http_disconnect_cancels_owned_job(self):
        self.enable_async([dict(status='running', phase='prefill', prefill_processed=128, prompt_eval_count=1000)])
        body = json.dumps(payload(True)).encode()
        with patch.object(ripple, 'POLL_INTERVAL', .005), patch.object(ripple, 'HEARTBEAT_INTERVAL', .005):
            client = socket.create_connection(('127.0.0.1', self.fixture.worker.server_port), timeout=3)
            client.sendall((f'POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n').encode() + body)
            received = b''
            while b'prefill_processed' not in received:
                received += client.recv(65536)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            client.close()
            deadline = time.monotonic() + 3
            while not self.fixture.cancelled_paths and time.monotonic() < deadline:
                time.sleep(.01)
        self.assertEqual(self.fixture.cancelled_paths, ['/v1/chat/completions/jobs/owned-1'])
        self.assertTrue(self.fixture.backend.health()['ready'])

    def test_async_deadline_cancels_owned_job(self):
        self.enable_async([dict(status='queued')])
        self.fixture.backend.timeout = .04
        with patch.object(ripple, 'POLL_INTERVAL', .01):
            with self.assertRaises(ripple.APIError) as caught:
                self.fixture.backend.generate(payload())
        self.assertEqual(caught.exception.status, 504)
        self.assertEqual(self.fixture.cancelled_paths, ['/v1/chat/completions/jobs/owned-1'])

    def manager(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        config = dict(load_timeout=5, request_timeout=5, models=[dict(
            key='ripple', id=ripple.MODEL, label='Ripple-65B TP8', streaming=False,
            context_tokens=8192, max_new_tokens=512, url=f'http://127.0.0.1:{port}',
            log=str(self.fixture.root / 'adapter.log'), command=[sys.executable,
                str(Path(ripple.__file__).resolve()), '--deployment', str(self.fixture.receipt),
                '--port', str(port)])])
        manager = router.ModelManager(config)
        self.addCleanup(manager.close)
        self.assertEqual(manager.select('ripple')[0], 202)
        deadline = time.monotonic() + 6
        while manager.health()['state'] == 'loading' and time.monotonic() < deadline:
            time.sleep(.02)
        return manager

    def test_router_owns_only_the_adapter_and_passes_native_messages(self):
        manager = self.manager()
        self.assertTrue(manager.health()['ready'], manager.health())
        self.assertFalse(manager.health()['streaming'])
        http = ThreadingHTTPServer(('127.0.0.1', 0), router.handler_for(manager))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(f'http://127.0.0.1:{http.server_port}/api/chat',
                              data=json.dumps(payload()).encode())
            with self.fixture.opener.open(request, timeout=5) as response:
                event = json.loads(response.read())
            self.assertEqual(event['model'], ripple.MODEL)
            self.assertTrue(event['done'])
            self.assertEqual(self.fixture.requests[-1][2]['messages'], payload()['messages'])
            self.assertEqual(manager.active_requests, 0)
        finally:
            http.shutdown(); http.server_close(); thread.join(3)
        process = manager.process
        manager.close()
        self.assertIsNotNone(process.poll())
        # The independently owned upstream survives local worker shutdown.
        self.assertEqual(self.fixture.backend.health()['tensor_parallel'], 8)

    def test_bad_credentials_fail_router_loading_without_waiting_load_timeout(self):
        (self.fixture.root / 'credentials/api_key').write_text('wrong-secret')
        manager = self.manager()
        self.assertEqual(manager.health()['state'], 'error')
        self.assertIn('authentication', manager.health()['last_error'])


if __name__ == '__main__':
    unittest.main()
