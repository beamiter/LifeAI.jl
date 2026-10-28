"""Model-router contracts using loopback mock workers and owned subprocesses.

No SSH, GPU, Julia, model downloads or third-party Python modules are used.
Run: python3 -m unittest -v test_server
"""
import contextlib
import http.client
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import server


WORKER = r'''
import json, os, signal, sys, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
spec=json.loads(os.environ['MOCK_WORKER_SPEC'])
started=time.monotonic()
events=Path(spec['events'])
def record(kind, **values):
    with events.open('a') as stream:
        stream.write(json.dumps(dict(event=kind, model=spec['model'], pid=os.getpid(), time=time.monotonic(), **values))+'\n')
        stream.flush()
class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args): pass
    def respond(self,status,value):
        data=json.dumps(value).encode()
        self.send_response(status); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path!='/healthz': return self.respond(404,{'error':'not found'})
        self.respond(200,dict(ready=time.monotonic()-started>=spec.get('ready_delay',0),
            model=spec.get('health_model',spec['model']),context_tokens=spec.get('context_tokens',256),
            max_new_tokens=spec.get('max_new_tokens',32)))
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        record('REQUEST',path=self.path,payload=payload)
        prompt=payload.get('prompt','')+' '.join(m.get('content','') for m in payload.get('messages',[]))
        if 'WORKER_ERROR' in prompt: return self.respond(422,{'error':'mock context limit'})
        if 'WAIT' in prompt:
            gate=Path(spec['events']+'.release')
            deadline=time.monotonic()+10
            while not gate.exists() and time.monotonic()<deadline: time.sleep(.01)
        if not payload.get('stream',True):
            if 'INVALID_JSON' in prompt:
                data=b'not-json'
                self.send_response(200); self.send_header('Content-Length',str(len(data)))
                self.end_headers(); self.wfile.write(data); return
            return self.respond(200,dict(model=spec['model'],response='hello world',done=True))
        self.send_response(200); self.send_header('Content-Type','application/x-ndjson')
        self.send_header('Connection','close'); self.end_headers(); self.close_connection=True
        packets=[dict(model=spec['model'],response='hello',done=False)]
        if 'PARTIAL' not in prompt and 'BAD_JSON' not in prompt: packets += [dict(response=' world',done=False),dict(done=True,eval_count=2)]
        for packet in packets:
            self.wfile.write(json.dumps(packet).encode()+b'\n'); self.wfile.flush()
        if 'BAD_JSON' in prompt: self.wfile.write(b'not-json\n'); self.wfile.flush()
server=ThreadingHTTPServer(('127.0.0.1',spec['port']),Handler)
server.daemon_threads=True
record('START')
def stop(*args):
    record('STOP')
    threading.Thread(target=server.shutdown,daemon=True).start()
signal.signal(signal.SIGTERM,stop)
try: server.serve_forever(poll_interval=.02)
finally: server.server_close(); record('EXIT')
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_until(predicate, timeout=6, message='condition did not become true'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.015)
    raise AssertionError(message)


class RouterFixture:
    def __init__(self, overrides=None):
        self.temporary = tempfile.TemporaryDirectory(prefix='lifeai-router-test-')
        self.root = Path(self.temporary.name)
        worker = self.root / 'worker.py'
        worker.write_text(WORKER)
        self.events = self.root / 'events.jsonl'
        models = []
        for key, model in [('qwen', 'Qwen/mock'), ('flint', 'RecursiveSI/mock')]:
            spec = dict(port=free_port(), model=model, events=str(self.events))
            spec.update((overrides or {}).get(key, {}))
            models.append(dict(key=key, id=model, label=key, context_tokens=256, max_new_tokens=32,
                command=[sys.executable, str(worker)], url=f"http://127.0.0.1:{spec['port']}",
                log=str(self.root / f'{key}.log'), env={'MOCK_WORKER_SPEC': json.dumps(spec)}))
        self.manager = server.ModelManager(dict(models=models, load_timeout=4, request_timeout=4))
        self.http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.handler_for(self.manager))
        self.http.daemon_threads = True
        self.thread = threading.Thread(target=self.http.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.http.server_port}'
        self.opener = build_opener(ProxyHandler({}))

    def records(self):
        if not self.events.exists():
            return []
        return [json.loads(line) for line in self.events.read_text().splitlines() if line]

    def request(self, path, payload=None):
        request = self.url + path if payload is None else Request(self.url + path,
            json.dumps(payload).encode(), {'Content-Type': 'application/json'}, method='POST')
        try:
            response = self.opener.open(request, timeout=6)
        except HTTPError as error:
            return error.code, json.load(error)
        with response:
            data = response.read()
            if 'ndjson' in response.headers.get('Content-Type', ''):
                return response.status, [json.loads(line) for line in data.splitlines() if line]
            return response.status, json.loads(data)

    def ready(self, model='qwen'):
        code, _ = self.request('/api/select', {'model': model})
        assert code in (200, 202), code
        wait_until(lambda: self.manager.health()['ready'], message=str(self.manager.health()))
        return self.manager.process

    def chat(self, text='hello', **kwargs):
        data = dict(messages=[dict(role='user', content=text)],
                    options=dict(num_predict=4, temperature=0))
        data.update(kwargs)
        return self.request('/api/chat', data)

    def close(self):
        # Release any intentionally blocked mock request before stopping workers.
        Path(str(self.events) + '.release').touch()
        self.manager.close()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(2)
        self.temporary.cleanup()


class RouterHTTPTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RouterFixture()
        self.addCleanup(self.fixture.close)

    def test_load_duration_stops_increasing_after_ready(self):
        self.fixture.ready()
        first = self.fixture.manager.health()['load_seconds']
        time.sleep(.03)
        self.assertIsInstance(first, float)
        self.assertEqual(first, self.fixture.manager.health()['load_seconds'])

    def test_catalog_initial_health_and_unknown_endpoint(self):
        code, health = self.fixture.request('/healthz')
        self.assertEqual(code, 200)
        self.assertFalse(health['ready'])
        self.assertEqual(health['state'], 'idle')
        code, catalog = self.fixture.request('/api/models')
        self.assertEqual(code, 200)
        self.assertEqual({entry['key'] for entry in catalog['models']}, {'qwen', 'flint'})
        self.assertEqual(self.fixture.request('/missing')[0], 404)
        self.assertEqual(self.fixture.request('/missing', {})[0], 404)

    def test_switch_terminates_owned_worker_before_new_start(self):
        old = self.fixture.ready('qwen')
        starts = len([r for r in self.fixture.records() if r['event'] == 'START'])
        self.assertEqual(self.fixture.request('/api/select', {'model': 'qwen'})[0], 200)
        self.assertEqual(len([r for r in self.fixture.records() if r['event'] == 'START']), starts)
        self.fixture.ready('flint')
        self.assertIsNotNone(old.poll())
        records = self.fixture.records()
        old_exit = next(r for r in records if r['model'] == 'Qwen/mock' and r['event'] == 'EXIT')
        new_start = next(r for r in records if r['model'] == 'RecursiveSI/mock' and r['event'] == 'START')
        self.assertLess(old_exit['time'], new_start['time'])
        health = self.fixture.manager.health()
        self.assertEqual(health['model'], 'RecursiveSI/mock')
        self.assertEqual((health['context_tokens'], health['max_new_tokens']), (256, 32))

    def test_templates_and_forwarded_limits(self):
        self.fixture.ready('qwen')
        messages = [dict(role='system', content='系统'), dict(role='user', content='你好'),
                    dict(role='assistant', content='answer'), dict(role='user', content='next')]
        code, result = self.fixture.chat(messages=messages, stream=False)
        self.assertEqual((code, result['response']), (200, 'hello world'))
        request = [r for r in self.fixture.records() if r['event'] == 'REQUEST'][-1]
        self.assertEqual(request['path'], '/api/generate')
        self.assertEqual(request['payload']['prompt'], server.qwen_prompt(messages))
        self.assertTrue(request['payload']['raw'])
        self.assertFalse(request['payload']['think'])
        self.assertEqual(request['payload']['options'], dict(num_predict=4, temperature=0,
            num_ctx=256, repeat_penalty=1, repeat_last_n=0))
        self.fixture.ready('flint')
        code, packets = self.fixture.chat(messages=messages)
        self.assertEqual(code, 200)
        self.assertTrue(packets[-1]['done'])
        request = [r for r in self.fixture.records() if r['event'] == 'REQUEST'][-1]
        self.assertEqual(request['path'], '/api/chat')
        self.assertEqual(request['payload']['messages'], messages)
        self.assertNotIn('prompt', request['payload'])
        self.assertEqual(self.fixture.request('/api/generate', dict(prompt='raw', options={'num_predict': 1}))[0], 400)

    def test_busy_rejects_switch_and_parallel_generation(self):
        self.fixture.ready('qwen')
        result = []
        thread = threading.Thread(target=lambda: result.append(self.fixture.chat('WAIT')))
        thread.start()
        self.addCleanup(lambda: thread.join(6))
        wait_until(lambda: self.fixture.manager.active_requests == 1)
        self.assertEqual(self.fixture.request('/api/select', {'model': 'flint'})[0], 409)
        self.assertEqual(self.fixture.chat('second')[0], 409)
        Path(str(self.fixture.events) + '.release').touch()
        thread.join(6)
        self.assertEqual(result[0][0], 200)
        wait_until(lambda: self.fixture.manager.active_requests == 0)
        self.assertEqual(self.fixture.chat()[0], 200)

    def test_limits_and_invalid_request_types_fail_before_worker_post(self):
        self.fixture.ready('qwen')
        invalid = [dict(options={'num_predict': value}) for value in (0, -1, 33, True, 1.5)]
        invalid += [dict(options={'num_predict': 1, 'temperature': value})
                    for value in (-1, 3, True, float('nan'), float('inf'), .1)]
        invalid += [dict(stream=1), dict(options=[]), dict(messages=[]),
                    dict(messages=[{'role': 'tool', 'content': 'x'}]),
                    dict(messages=[{'role': 'user', 'content': None}]),
                    dict(messages=[dict(role='user', content='x')] * 65), dict(model='unknown')]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(self.fixture.chat(**payload)[0], 400)
        self.assertEqual([r for r in self.fixture.records() if r['event'] == 'REQUEST'], [])
        self.assertEqual(self.fixture.manager.active_requests, 0)

    def test_long_utf8_chat_exceeds_old_body_cap_and_is_preserved(self):
        self.fixture.ready('flint')
        content = '长上下文🌟' * 100000
        self.assertGreater(len(content.encode()), 1024 * 1024)
        code, packets = self.fixture.chat(messages=[dict(role='user', content=content)])
        self.assertEqual(code, 200)
        self.assertTrue(packets[-1]['done'])
        request = [r for r in self.fixture.records() if r['event'] == 'REQUEST'][-1]
        self.assertEqual(request['payload']['messages'][0]['content'], content)
        self.assertEqual(self.fixture.manager.active_requests, 0)

    def test_chat_character_limits_fail_before_worker_post(self):
        self.fixture.ready('flint')
        with patch.object(server, 'MAX_MESSAGE_CHARS', 8), patch.object(server, 'MAX_TOTAL_MESSAGE_CHARS', 12):
            for messages in ([dict(role='user', content='长' * 9)],
                             [dict(role='user', content='长' * 7), dict(role='user', content='文' * 6)]):
                with self.subTest(lengths=[len(m['content']) for m in messages]):
                    self.assertEqual(self.fixture.chat(messages=messages)[0], 413)
        self.assertEqual([r for r in self.fixture.records() if r['event'] == 'REQUEST'], [])
        self.assertEqual(self.fixture.manager.active_requests, 0)

    def test_invalid_unhashable_model_is_client_error(self):
        self.fixture.ready('qwen')
        for model in ({}, []):
            with self.subTest(model=model):
                self.assertEqual(self.fixture.request('/api/select', {'model': model})[0], 400)
                self.assertEqual(self.fixture.chat(model=model)[0], 400)
        self.assertEqual(self.fixture.manager.active_requests, 0)

    def test_stream_done_and_errors_always_release_reservation(self):
        self.fixture.ready('flint')
        code, packets = self.fixture.chat()
        self.assertEqual(code, 200)
        self.assertEqual(''.join(packet.get('response', '') for packet in packets), 'hello world')
        self.assertEqual(sum(packet.get('done') is True for packet in packets), 1)
        for text, code in [('WORKER_ERROR', 422), ('PARTIAL', 200), ('BAD_JSON', 200)]:
            with self.subTest(text=text):
                status, result = self.fixture.chat(text)
                self.assertEqual(status, code)
                if text in ('PARTIAL', 'BAD_JSON'):
                    self.assertTrue(result[-1]['done'])
                    self.assertIn('error', result[-1])
                    if text == 'PARTIAL': self.assertIn('before completion', result[-1]['error'])
                else:
                    self.assertIn('mock context limit', result['error'])
                wait_until(lambda: self.fixture.manager.active_requests == 0)
                self.assertEqual(self.fixture.chat()[0], 200)

    def test_upstream_connection_and_nonstream_json_errors_release_reservation(self):
        self.fixture.ready('flint')
        with patch.object(self.fixture.manager.opener, 'open', side_effect=URLError('mock connection refused')):
            code, body = self.fixture.chat()
        self.assertEqual(code, 502)
        self.assertIn('mock connection refused', body['error'])
        self.assertEqual(self.fixture.manager.active_requests, 0)
        code, body = self.fixture.chat('INVALID_JSON', stream=False)
        self.assertEqual(code, 502)
        self.assertIn('error', body)
        self.assertEqual(self.fixture.manager.active_requests, 0)
        self.assertEqual(self.fixture.chat()[0], 200)

    def test_worker_crash_marks_health_unready_and_can_reload(self):
        process = self.fixture.ready('qwen')
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
        code, health = self.fixture.request('/healthz')
        self.assertEqual(code, 200)
        self.assertFalse(health['ready'])
        self.assertEqual(health['state'], 'error')
        self.assertEqual(self.fixture.chat()[0], 503)
        replacement = self.fixture.ready('qwen')
        self.assertNotEqual(replacement.pid, process.pid)
        self.assertEqual(self.fixture.chat()[0], 200)

    def test_malformed_json_and_content_length(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.fixture.http.server_port, timeout=3)
        connection.request('POST', '/api/select', b'bad-json', {'Content-Type': 'application/json'})
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        response.read(); connection.close()
        connection = http.client.HTTPConnection('127.0.0.1', self.fixture.http.server_port, timeout=3)
        connection.putrequest('POST', '/api/select')
        connection.putheader('Content-Length', str(server.MAX_BODY_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read(); connection.close()

    def test_shutdown_reaps_ready_worker_and_rejects_new_load(self):
        process = self.fixture.ready('flint')
        self.fixture.manager.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(self.fixture.manager.health()['ready'])
        with self.assertRaises(server.APIError) as caught:
            self.fixture.manager.select('qwen')
        self.assertEqual(caught.exception.status, 503)


class ManagerLifecycleTests(unittest.TestCase):
    def config(self, root):
        return dict(models=[dict(key='qwen', id='Qwen/mock', context_tokens=32,
            max_new_tokens=8, command=['missing-program'], url='http://127.0.0.1:1',
            log=str(Path(root)/'worker.log'))])

    def test_terminate_escalates_and_waits(self):
        process = MagicMock(pid=987654)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('mock', 15), 0]
        with patch.object(server.os, 'killpg') as killpg:
            server.ModelManager._terminate(process)
        self.assertEqual(killpg.call_args_list[0].args, (987654, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (987654, signal.SIGKILL))
        self.assertEqual(process.wait.call_count, 2)

    def test_spawn_failure_closes_unowned_log(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = server.ModelManager(self.config(directory))
            log = MagicMock()
            with patch.object(server.Path, 'open', return_value=log), \
                    patch.object(server.subprocess, 'Popen', side_effect=OSError('spawn failed')):
                manager._load('Qwen/mock')
            self.assertEqual(manager.state, 'error')
            self.assertIn('spawn failed', manager.error)
            log.close.assert_called()

    def test_close_waits_for_worker_detached_during_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config['models'].append(dict(config['models'][0], key='flint', id='RecursiveSI/mock'))
            manager = server.ModelManager(config)
            old = MagicMock(pid=987654)
            old.poll.return_value = None
            manager.process, manager.selected, manager.state = old, 'Qwen/mock', 'ready'
            entered, release, stopped, closed = [threading.Event() for _ in range(4)]
            def terminate(process):
                if process is old:
                    entered.set()
                    release.wait(3)
                    old.poll.return_value = 0
                    stopped.set()
            with patch.object(manager, '_terminate', side_effect=terminate):
                manager.select('flint')
                self.assertTrue(entered.wait(2))
                closer = threading.Thread(target=lambda: (manager.close(), closed.set()))
                closer.start()
                try:
                    self.assertFalse(closed.wait(.1), 'close returned while an owned worker was still stopping')
                finally:
                    release.set()
                    closer.join(3)
                self.assertTrue(stopped.is_set())
                self.assertTrue(closed.is_set())


class WorkerHealthTests(unittest.TestCase):
    def test_loading_identity_and_limits_are_validated(self):
        for override in (dict(health_model='wrong/model'), dict(context_tokens=0),
                         dict(max_new_tokens=True), dict(context_tokens=16, max_new_tokens=32)):
            with self.subTest(override=override):
                fixture = RouterFixture({'flint': override})
                try:
                    self.assertEqual(fixture.request('/api/select', {'model': 'flint'})[0], 202)
                    wait_until(lambda: fixture.manager.state == 'error')
                    self.assertFalse(fixture.manager.health()['ready'])
                    self.assertIsNotNone(fixture.manager.process.poll())
                finally:
                    fixture.close()

    def test_load_timeout_terminates_started_worker(self):
        fixture = RouterFixture({'flint': dict(ready_delay=10)})
        fixture.manager.config['load_timeout'] = .2
        try:
            fixture.request('/api/select', {'model': 'flint'})
            wait_until(lambda: fixture.manager.state == 'error')
            self.assertIn('timed out', fixture.manager.error)
            self.assertIsNotNone(fixture.manager.process.poll())
        finally:
            fixture.close()

    def test_shutdown_during_loading_reaps_new_worker(self):
        fixture = RouterFixture({'flint': dict(ready_delay=10)})
        try:
            fixture.request('/api/select', {'model': 'flint'})
            wait_until(lambda: fixture.manager.process is not None)
            process = fixture.manager.process
            fixture.manager.close()
            self.assertIsNotNone(process.poll())
            self.assertFalse(fixture.manager.loader_thread.is_alive())
            self.assertEqual(fixture.chat()[0], 503)
        finally:
            fixture.close()

    def test_while_loading_same_selection_is_idempotent_and_chat_unavailable(self):
        fixture = RouterFixture({'flint': dict(ready_delay=1)})
        try:
            self.assertEqual(fixture.request('/api/select', {'model': 'flint'})[0], 202)
            self.assertEqual(fixture.request('/api/select', {'model': 'flint'})[0], 202)
            self.assertEqual(fixture.request('/api/select', {'model': 'qwen'})[0], 409)
            self.assertEqual(fixture.chat()[0], 503)
            wait_until(lambda: fixture.manager.health()['ready'])
            self.assertEqual(len([r for r in fixture.records() if r['event'] == 'START']), 1)
        finally:
            fixture.close()


if __name__ == '__main__':
    unittest.main()
