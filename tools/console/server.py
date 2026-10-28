#!/usr/bin/env python3
"""Loopback model router: one owned GPU worker at a time, Qwen or Flint."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_MESSAGES = 64
MAX_MESSAGE_CHARS = 4 * 1024 * 1024
MAX_TOTAL_MESSAGE_CHARS = 4 * 1024 * 1024


class APIError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def qwen_prompt(messages):
    return ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'
                   for m in messages) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {m['id']: dict(m) for m in config['models']}
        self.aliases = {m['key']: m['id'] for m in config['models']}
        self.lock = threading.RLock()
        self.opener = build_opener(ProxyHandler({}))
        self.process = None
        self.log = None
        self.selected = None
        self.state = 'idle'
        self.error = None
        self.active_requests = 0
        self.closed = False
        self.load_started = None
        self.load_duration = None
        self.worker_health = {}
        self.loader_thread = None
        self.retiring_process = None

    def resolve(self, name):
        if not isinstance(name, str):
            raise APIError(400, 'model must be a string: qwen or flint.')
        result = self.aliases.get(name, name)
        if result not in self.models:
            raise APIError(400, 'Unknown model; choose qwen or flint.')
        return result

    def catalog(self):
        return [dict(key=m['key'], id=m['id'], label=m.get('label', m['id']),
                     context_tokens=m['context_tokens'], max_new_tokens=m['max_new_tokens'])
                for m in self.models.values()]

    def health(self):
        with self.lock:
            if self.state == 'ready' and (self.process is None or self.process.poll() is not None):
                self.state, self.error = 'error', 'Model worker exited; select the model to reload.'
            spec = self.models.get(self.selected, {})
            return dict(ready=self.state == 'ready', model=self.selected,
                        selected_model=self.selected, state=self.state,
                        loading_model=self.selected if self.state == 'loading' else None,
                        phase=self.worker_health.get('phase', self.state),
                        last_error=self.error, active_requests=self.active_requests,
                        context_tokens=self.worker_health.get('context_tokens', spec.get('context_tokens')),
                        max_new_tokens=self.worker_health.get('max_new_tokens', spec.get('max_new_tokens')),
                        load_seconds=round(time.monotonic() - self.load_started, 2)
                        if self.load_started and self.state == 'loading' else self.load_duration,
                        models=self.catalog())

    def select(self, name):
        model = self.resolve(name)
        with self.lock:
            if self.closed:
                raise APIError(503, 'Service is shutting down.')
            if self.active_requests:
                raise APIError(409, 'A response is in progress; finish it before switching models.')
            if self.state == 'loading':
                if self.selected == model:
                    return 202, self.health()
                raise APIError(409, 'A model is loading; wait for it before switching.')
            if self.selected == model and self.health()['ready']:
                return 200, self.health()
            self.selected, self.state, self.error = model, 'loading', None
            self.worker_health = {}
            self.load_started = time.monotonic()
            self.load_duration = None
            self.loader_thread = threading.Thread(target=self._load, args=(model,), daemon=True)
            self.loader_thread.start()
            return 202, self.health()

    @staticmethod
    def _terminate(process):
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
        except ProcessLookupError:
            process.wait()

    def _load(self, model):
        spec = self.models[model]
        try:
            # Only this manager's process group is stopped. Never kill by port/name.
            with self.lock:
                old, old_log = self.process, self.log
                self.process, self.log = None, None
                self.retiring_process = old
            self._terminate(old)
            with self.lock:
                self.retiring_process = None
            if old_log:
                old_log.close()
            if old is not None:
                time.sleep(1)  # allow driver context cleanup before the next load
            with self.lock:
                if self.closed:
                    return
                log_path = Path(spec['log'])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('ab', buffering=0)
                env = os.environ.copy()
                env.update(spec.get('env', {}))
                try:
                    process = subprocess.Popen(spec['command'], cwd=spec.get('cwd'), env=env,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               start_new_session=True)
                except Exception:
                    log.close()
                    raise
                self.process, self.log = process, log
            deadline = time.monotonic() + self.config.get('load_timeout', 1800)
            while time.monotonic() < deadline:
                with self.lock:
                    if self.closed:
                        return
                if process.poll() is not None:
                    raise RuntimeError(f"{spec['key']} worker exited ({process.returncode}); see {spec['log']}")
                try:
                    with self.opener.open(spec['url'] + '/healthz', timeout=2) as response:
                        health = json.load(response)
                    if health.get('model') != model:
                        raise RuntimeError('Worker model identity does not match its configuration.')
                    with self.lock:
                        self.worker_health = health
                    if health.get('error'):
                        raise RuntimeError(str(health['error']))
                    if health.get('ready') is True:
                        if health.get('model') != model:
                            raise RuntimeError('Worker model identity does not match its configuration.')
                        for key in ('context_tokens', 'max_new_tokens'):
                            if type(health.get(key)) is not int or health[key] <= 0:
                                raise RuntimeError('Worker did not publish valid generation limits.')
                        if health['max_new_tokens'] >= health['context_tokens']:
                            raise RuntimeError('Worker output cap must be below its context limit.')
                        with self.lock:
                            if not self.closed:
                                self.worker_health, self.state = health, 'ready'
                                self.load_duration = (round(time.monotonic() - self.load_started, 2)
                                                      if self.load_started is not None else None)
                        return
                except (URLError, OSError, ValueError):
                    pass
                time.sleep(.5)
            raise RuntimeError(f"{spec['key']} load timed out; see {spec['log']}")
        except Exception as exc:
            with self.lock:
                process = self.process
            self._terminate(process)
            with self.lock:
                self.state, self.error = 'error', str(exc)
                self.load_duration = (round(time.monotonic() - self.load_started, 2)
                                      if self.load_started is not None else None)

    def reserve(self, payload, raw=False):
        if not isinstance(payload, dict):
            raise APIError(400, 'JSON body must be an object.')
        model = self.resolve(payload.get('model', self.selected))
        if raw:
            if self.models[model]['key'] != 'qwen' or not isinstance(payload.get('prompt'), str):
                raise APIError(400, 'Use /api/chat with messages for Flint.')
            messages = None
            if len(payload['prompt']) > MAX_TOTAL_MESSAGE_CHARS:
                raise APIError(413, 'Prompt exceeds the 4 Mi character limit.')
        else:
            messages = payload.get('messages')
            if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
                raise APIError(400, f'messages must contain 1 to {MAX_MESSAGES} chat messages.')
            total_chars = 0
            for message in messages:
                if (not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant')
                        or not isinstance(message.get('content'), str)):
                    raise APIError(400, 'Each message needs a system/user/assistant role and string content.')
                total_chars += len(message['content'])
                if (len(message['content']) > MAX_MESSAGE_CHARS
                        or total_chars > MAX_TOTAL_MESSAGE_CHARS):
                    raise APIError(413, 'Chat content exceeds the 4 Mi character limit.')
        options = payload.get('options', {})
        if not isinstance(options, dict) or type(payload.get('stream', True)) is not bool:
            raise APIError(400, 'Invalid options or stream flag.')
        with self.lock:
            health = self.health()
            if self.closed or not health['ready']:
                raise APIError(503, self.error or 'Model is loading; wait for ready=true.')
            if model != self.selected:
                raise APIError(409, 'Select the requested model first.')
            if self.active_requests:
                raise APIError(409, 'Another response is in progress.')
            count = options.get('num_predict', min(256, health['max_new_tokens']))
            if type(count) is not int or not 1 <= count <= health['max_new_tokens']:
                raise APIError(400, f"num_predict must be 1..{health['max_new_tokens']}.")
            temperature = options.get('temperature', 0)
            if (type(temperature) not in (int, float) or not math.isfinite(temperature)
                    or not 0 <= temperature <= 2):
                raise APIError(400, 'temperature must be finite and between 0 and 2.')
            spec = self.models[model]
            if spec['key'] == 'qwen' and temperature != 0:
                raise APIError(400, 'This Qwen worker supports temperature=0.')
            output = dict(model=model, stream=payload.get('stream', True),
                          options=dict(num_predict=count, temperature=temperature))
            if spec['key'] == 'qwen':
                output.update(raw=True, think=False,
                              prompt=payload['prompt'] if raw else qwen_prompt(messages))
                output['options'].update(num_ctx=health['context_tokens'], repeat_penalty=1, repeat_last_n=0)
                endpoint = '/api/generate'
            else:
                output['messages'] = messages
                endpoint = '/api/chat'
            self.active_requests += 1
            return spec['url'] + endpoint, output

    def release(self):
        with self.lock:
            self.active_requests -= 1

    def close(self):
        with self.lock:
            self.closed = True
            process, log = self.process, self.log
            retiring, loader = self.retiring_process, self.loader_thread
        self._terminate(process)
        self._terminate(retiring)
        if loader and loader is not threading.current_thread():
            loader.join(timeout=40)
        if log:
            log.close()


def handler_for(manager):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def json_response(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/healthz':
                self.json_response(200, manager.health())
            elif self.path == '/api/models':
                self.json_response(200, {'models': manager.catalog()})
            else:
                self.json_response(404, {'error': 'Unknown endpoint.'})

        def do_POST(self):
            reserved = False
            streaming = False
            try:
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    raise APIError(400, 'Invalid Content-Length.')
                if not 0 < length <= MAX_BODY_BYTES:
                    self.close_connection = True
                    raise APIError(413, 'JSON request must be between 1 byte and 16 MiB.')
                try:
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeError):
                    raise APIError(400, 'Invalid JSON.')
                if not isinstance(payload, dict):
                    raise APIError(400, 'JSON body must be an object.')
                if self.path == '/api/select':
                    status, result = manager.select(payload.get('model'))
                    self.json_response(status, result)
                    return
                if self.path not in ('/api/chat', '/api/generate'):
                    raise APIError(404, 'Unknown endpoint.')
                url, forwarded = manager.reserve(payload, raw=self.path == '/api/generate')
                reserved = True
                request = Request(url, data=json.dumps(forwarded, ensure_ascii=False).encode(),
                                  headers={'Content-Type': 'application/json'}, method='POST')
                with manager.opener.open(request, timeout=manager.config.get('request_timeout', 1800)) as response:
                    if not forwarded['stream']:
                        result = json.load(response)
                        manager.release()
                        reserved = False
                        self.json_response(200, result)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection, streaming = True, True
                    done = False
                    for line in response:
                        if not line.strip():
                            continue
                        packet = json.loads(line)
                        if packet.get('done') is True:
                            manager.release()
                            reserved = False
                        self.wfile.write(json.dumps(packet, ensure_ascii=False).encode() + b'\n')
                        self.wfile.flush()
                        if packet.get('done') is True:
                            done = True
                            break
                    if not done:
                        raise APIError(502, 'Worker stream ended before completion.')
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                if reserved:
                    manager.release()
                    reserved = False
                status = exc.status if isinstance(exc, APIError) else 502
                message = str(exc)
                if isinstance(exc, HTTPError):
                    status = exc.code
                    try:
                        message = json.load(exc).get('error', message)
                    except (ValueError, OSError):
                        pass
                try:
                    if streaming:
                        self.wfile.write(json.dumps({'error': message, 'done': True}).encode() + b'\n')
                        self.wfile.flush()
                    else:
                        self.json_response(status, {'error': message})
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                if reserved:
                    manager.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1'])
    parser.add_argument('--port', type=int, default=11435)
    parser.add_argument('--default-model', default='flint')
    args = parser.parse_args()
    manager = ModelManager(json.loads(args.config.read_text()))
    server = ThreadingHTTPServer((args.host, args.port), handler_for(manager))
    server.daemon_threads = True
    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        if args.default_model != 'none':
            manager.select(args.default_model)
        print(f'LifeAI model router listening on {args.host}:{args.port}', flush=True)
        server.serve_forever()
    finally:
        manager.close()
        server.server_close()


if __name__ == '__main__':
    main()
