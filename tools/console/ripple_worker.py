#!/usr/bin/env python3
"""Loopback LifeAI protocol adapter for an existing Ripple DLC deployment.

Only this adapter process is owned by the router. Starting/stopping it never
loads weights or starts/stops the remote DLC job. Bearer credentials stay on DSW.
"""
import argparse
from dataclasses import dataclass, field
import ipaddress
import json
import math
from pathlib import Path
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MODEL = 'ricursiveAI/Ripple-65B-Preview'
MAX_BODY = 16 * 1024 * 1024
UPSTREAM_BODY = 4 * 1024 * 1024


class APIError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


@dataclass(frozen=True)
class Connection:
    endpoint: str
    model: str
    key: str = field(repr=False)
    context: int
    output: int
    world_size: int


class RippleBackend:
    def __init__(self, deployment, timeout=600):
        self.deployment = Path(deployment)
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.busy = threading.Lock()

    def json_request(self, url, payload=None, key=None, timeout=None):
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        if data is not None and len(data) > UPSTREAM_BODY:
            raise APIError(413, 'Ripple request exceeds the 4 MiB upstream limit.')
        try:
            request = Request(url, data=data, headers=headers)
            with self.opener.open(request, timeout=timeout or self.timeout) as response:
                raw = response.read(MAX_BODY + 1)
        except HTTPError as error:
            detail = 'Ripple upstream rejected the request.'
            try:
                body = json.loads(error.read(65536))
                message = body.get('error', detail)
                detail = message.get('message', detail) if isinstance(message, dict) else str(message)
            except (ValueError, OSError, AttributeError):
                pass
            if error.code in (401, 403):
                detail = 'Ripple authentication failed; check the DSW credentials/api_key file.'
            raise APIError(error.code, detail) from None
        except OSError:
            raise APIError(503, 'Ripple DLC service is unreachable; check the DLC deployment and retry.') from None
        if len(raw) > MAX_BODY:
            raise APIError(502, 'Ripple response exceeds 16 MiB.')
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            raise APIError(502, 'Ripple returned invalid JSON.') from None
        if not isinstance(result, dict):
            raise APIError(502, 'Ripple returned a non-object response.')
        return result

    def connect(self):
        try:
            root = self.deployment
            if root.is_file():
                root = Path(json.loads(root.read_text())['run_dir'])
            ready = json.loads((root / 'results/ready.json').read_text())
            key = (root / 'credentials/api_key').read_text().strip()
        except (OSError, ValueError, KeyError, TypeError):
            raise APIError(503, 'Cannot read the Ripple deployment receipt or DSW API-key file.') from None
        if not isinstance(ready, dict):
            raise APIError(503, 'Ripple deployment metadata must be an object.')
        if (ready.get('phase') != 'ready' or ready.get('model') != MODEL or not key
                or any(ch in key for ch in '\r\n')):
            raise APIError(503, 'Ripple deployment metadata or credentials are not ready.')
        context, output, world = (ready.get(k) for k in ('max_context', 'max_output', 'world_size'))
        if (any(type(v) is not int or v <= 0 for v in (context, output, world))
                or output >= context):
            raise APIError(502, 'Ripple deployment did not publish valid generation limits.')
        alias = ready.get('model_id')
        if not isinstance(alias, str) or not alias:
            raise APIError(502, 'Ripple deployment did not publish its model ID.')
        candidates = [ready.get('endpoint', '')] + [
            f'http://{ip}:{ready.get("port", 18085)}' for ip in ready.get('addresses', [])]
        for candidate in dict.fromkeys(candidates):
            parsed = urlsplit(candidate)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
                    or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
                continue
            try:
                if ipaddress.ip_address(parsed.hostname).is_link_local:
                    continue
            except ValueError:
                pass
            endpoint = candidate.rstrip('/')
            try:
                health = self.json_request(endpoint + '/healthz', timeout=2)
            except APIError:
                continue
            if health.get('status') != 'ready' or health.get('model') != alias or health.get('tensor_parallel') != world:
                continue
            models = self.json_request(endpoint + '/v1/models', key=key, timeout=3)
            entries = models.get('data')
            if not isinstance(entries, list) or not any(isinstance(m, dict) and m.get('id') == alias for m in entries):
                raise APIError(502, 'Ripple authenticated model identity does not match the deployment.')
            return Connection(endpoint, alias, key, context, output, world)
        raise APIError(503, 'No healthy Ripple DLC endpoint is reachable from DSW.')

    def health(self):
        connection = self.connect()
        return dict(ready=True, model=MODEL, phase='ready', context_tokens=connection.context,
                    max_new_tokens=connection.output, streaming=False,
                    backend='DLC tensor parallel', tensor_parallel=connection.world_size)

    def generate(self, payload):
        if not isinstance(payload, dict) or payload.get('model', MODEL) not in (MODEL, 'ripple'):
            raise APIError(400, 'Select model ripple for this worker.')
        messages = payload.get('messages')
        if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
            raise APIError(400, 'messages must contain 1..64 chat messages.')
        for message in messages:
            if (not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant')
                    or not isinstance(message.get('content'), str)):
                raise APIError(400, 'Each message needs a supported role and string content.')
        options = payload.get('options', {})
        if not isinstance(options, dict) or type(payload.get('stream', True)) is not bool:
            raise APIError(400, 'Invalid options or stream flag.')
        count, temperature = options.get('num_predict', 256), options.get('temperature', 0)
        if type(count) is not int or count <= 0:
            raise APIError(400, 'num_predict must be a positive integer.')
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise APIError(400, 'temperature must be finite and between 0 and 2.')
        if not self.busy.acquire(blocking=False):
            raise APIError(409, 'Another Ripple response is in progress.')
        try:
            connection = self.connect()
            if count > connection.output:
                raise APIError(400, f'num_predict must be 1..{connection.output}.')
            started = time.monotonic()
            body = self.json_request(connection.endpoint + '/v1/chat/completions',
                dict(model=connection.model, messages=messages, max_tokens=count,
                     temperature=temperature, stream=False), key=connection.key)
            elapsed = time.monotonic() - started
            try:
                if body.get('model') not in (connection.model, MODEL):
                    raise ValueError('model mismatch')
                choices = body['choices']
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError('choice count')
                text = choices[0]['message']['content']
                finish = choices[0]['finish_reason']
                usage = body['usage']
                prompt, tokens = usage['prompt_tokens'], usage['completion_tokens']
                if (not isinstance(text, str) or finish not in ('stop', 'length')
                        or any(type(v) is not int or v < 0 for v in (prompt, tokens))):
                    raise ValueError('invalid completion')
            except (KeyError, TypeError, IndexError, ValueError):
                raise APIError(502, 'Ripple returned an incomplete or invalid chat completion.') from None
            # Upstream is non-streaming. Publish exactly one complete native event,
            # with end-to-end duration rather than inventing a decode rate.
            return dict(model=MODEL, response=text, done=True, done_reason=finish,
                        prompt_eval_count=prompt, eval_count=tokens, total_duration=int(elapsed * 1e9),
                        lifeai=dict(tensor_parallel=connection.world_size, weight_dtype='BF16',
                                    activation_dtype='Float32', duration_seconds=elapsed, streaming=False))
        finally:
            self.busy.release()


def handler_for(backend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, value, stream=False):
            raw = json.dumps(value, ensure_ascii=False).encode() + (b'\n' if stream else b'')
            self.send_response(status)
            self.send_header('Content-Type', 'application/x-ndjson' if stream else 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                if self.path != '/healthz':
                    raise APIError(404, 'Unknown endpoint.')
                self.respond(200, backend.health())
            except APIError as error:
                self.respond(200, dict(ready=False, model=MODEL, phase='error', error=str(error)))

        def do_POST(self):
            try:
                if self.path != '/api/chat':
                    raise APIError(404, 'Use /api/chat for Ripple.')
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    raise APIError(400, 'Invalid Content-Length.')
                if not 0 < length <= MAX_BODY:
                    raise APIError(413, 'Request exceeds the 16 MiB native body limit.')
                try:
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeError):
                    raise APIError(400, 'Invalid JSON.')
                result = backend.generate(payload)
                self.respond(200, result, stream=payload.get('stream', True))
            except APIError as error:
                self.respond(error.status, dict(error=str(error)))
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--host', choices=['127.0.0.1'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11438)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 86400:
        parser.error('--timeout must be between 0 and 86400 seconds')
    server = ThreadingHTTPServer((args.host, args.port), handler_for(RippleBackend(args.deployment, args.timeout)))
    server.daemon_threads = True
    def shutdown(*_args):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
