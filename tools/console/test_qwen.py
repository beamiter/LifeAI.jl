"""Local HTTP contract tests; no SSH or remote service access."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import qwen


class ClientTests(unittest.TestCase):
    @contextlib.contextmanager
    def service(self, context=40960, cap=40896, error=None):
        requests = []
        class Opener:
            def open(self, request, timeout=None):
                if isinstance(request, str):
                    return io.BytesIO(json.dumps(dict(model=qwen.MODEL, ready=True,
                        context_tokens=context, max_new_tokens=cap)).encode())
                requests.append(json.loads(request.data))
                if error:
                    raise HTTPError(request.full_url, 400, "Bad Request", {},
                        io.BytesIO(json.dumps({'error': error}).encode()))
                return io.BytesIO(json.dumps({'response': 'ok', 'done': True}).encode())
        args = qwen.parse_args(['--url', 'http://127.0.0.1:11435',
            '--no-tunnel', '--no-stream', '-n', '1000'])
        with patch.object(qwen.urllib.request, 'build_opener', return_value=Opener()):
            yield qwen.Client(args), requests

    def test_large_request_uses_advertised_context(self):
        with self.service() as (client, requests):
            client.connect()
            self.assertEqual(client.generate([('user', 'hello')], lambda x: None)[0], 'ok')
            self.assertEqual(requests[0]['options']['num_ctx'], 40960)
            self.assertEqual(requests[0]['options']['num_predict'], 1000)

    def test_actual_cap_rejected_after_connect_without_post(self):
        with self.service(cap=512) as (client, requests):
            with self.assertRaisesRegex(qwen.ClientError, '512'):
                client.connect()
            self.assertEqual(requests, [])

    def test_no_space_for_prompt(self):
        with self.service(context=1000, cap=1000) as (client, requests):
            with self.assertRaisesRegex(qwen.ClientError, '提示词'):
                client.connect()
            self.assertEqual(requests, [])

    def test_invalid_health_limit_fails_closed(self):
        with self.service(context=True) as (client, requests):
            with self.assertRaisesRegex(qwen.ClientError, 'context_tokens'):
                client.connect()
            self.assertEqual(requests, [])

    def test_context_error_preserves_server_detail(self):
        detail = 'prompt tokens 40300 plus requested output 1000 exceeds context 40960'
        with self.service(error=detail) as (client, requests):
            client.connect()
            with self.assertRaises(qwen.urllib.error.HTTPError) as caught:
                client.generate([('user', 'long prompt')], lambda x: None)
            text = qwen.error_text(caught.exception)
            self.assertIn(detail, text)
            self.assertIn('--max-tokens', text)
            self.assertNotIn('4K', text)

    def test_help_and_positive_validation(self):
        self.assertEqual(qwen.parse_args(['-n', '40896']).max_tokens, 40896)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                qwen.parse_args(['-n', '0'])


class RouterOpener:
    """Stateful in-memory HTTP wire, including asynchronous model selection."""
    def __init__(self, *, streaming=False):
        self.selected = qwen.MODEL
        self.pending = None
        self.polls = 0
        self.requests = []
        self.fail_select = False
        self.fail_stream = False
        self.streaming = streaming
        self.blocked_once = False

    def health(self):
        if self.pending:
            self.polls += 1
            if self.polls >= 2:
                self.selected, self.pending = self.pending, None
        return dict(model=self.selected, ready=not self.pending,
                    state='loading' if self.pending else 'ready',
                    loading_model=self.pending, last_error=None,
                    context_tokens=4096, max_new_tokens=512,
                    models=[dict(key=k, id=v, context_tokens=4096, max_new_tokens=512)
                            for k, v in qwen.MODELS.items()])

    def open(self, request, timeout=None):
        if isinstance(request, str):
            if request.endswith('/healthz'):
                return io.BytesIO(json.dumps(self.health()).encode())
            raise AssertionError(request)
        path = qwen.urllib.parse.urlsplit(request.full_url).path
        payload = json.loads(request.data)
        self.requests.append((path, payload))
        if path == '/api/select':
            if self.fail_select:
                raise HTTPError(request.full_url, 500, 'failed', {},
                    io.BytesIO(b'{"error":"load failed"}'))
            if self.blocked_once:
                self.blocked_once = False
                raise HTTPError(request.full_url, 409, 'busy', {},
                    io.BytesIO(b'{"error":"busy"}'))
            self.pending, self.polls = payload['model'], 0
            return io.BytesIO(b'{"ready":false,"state":"loading"}')
        if path != '/api/chat':
            raise AssertionError(path)
        if payload['model'] != self.selected:
            raise AssertionError('client did not select requested model')
        final = dict(response='你好', done=True, eval_count=2,
                     lifeai=dict(decode_tokens_per_second=12.5))
        if payload['stream']:
            records = [dict(response='你', done=False), dict(response='好', done=False)]
            records += [dict(error='decode failed', done=True)] if self.fail_stream else [dict(final, response='')]
            return io.BytesIO(b'\n'.join(json.dumps(x, ensure_ascii=False).encode() for x in records)+b'\n')
        return io.BytesIO(json.dumps(final, ensure_ascii=False).encode())


class ImmediateThread:
    def __init__(self, target, **kwargs):
        self.target = target
    def start(self):
        self.target()


class MultiModelTests(unittest.TestCase):
    def client(self, router=None, model='qwen', stream=False):
        router = router or RouterOpener()
        options = ['--no-tunnel', '--model', model, '-n', '128']
        if not stream:
            options += ['--no-stream']
        with patch.object(qwen.urllib.request, 'build_opener', return_value=router):
            client = qwen.Client(qwen.parse_args(options))
        return client, router

    @patch.object(qwen.time, 'sleep', lambda _: None)
    def test_flint_async_selection_and_server_template(self):
        client, router = self.client(model='flint')
        statuses = []
        client.connect(progress=statuses.append)
        self.assertTrue(client.ready)
        self.assertEqual(client.model_id, qwen.MODELS['flint'])
        self.assertEqual(router.requests[0], ('/api/select', {'model': qwen.MODELS['flint']}))
        self.assertTrue(any('Flint' in status for status in statuses))
        answer, final = client.generate([('system','帮助用户'), ('user','你好')], lambda _: None)
        self.assertEqual(answer, '你好')
        path, payload = router.requests[-1]
        self.assertEqual(path, '/api/chat')
        self.assertEqual(payload['messages'], [{'role':'system','content':'帮助用户'}, {'role':'user','content':'你好'}])
        self.assertNotIn('prompt', payload)
        self.assertEqual(payload['options'], {'temperature':0.0, 'num_predict':128})

    @patch.object(qwen.time, 'sleep', lambda _: None)
    def test_busy_selection_retries_and_other_loading_waits(self):
        client, router = self.client()
        router.pending, router.polls = qwen.MODELS['flint'], -2
        router.blocked_once = True
        client.connect()
        self.assertEqual(client.model_key, 'qwen')
        self.assertTrue(client.ready)
        self.assertEqual([p for p,v in router.requests], ['/api/select','/api/select'])

    @patch.object(qwen.time, 'sleep', lambda _: None)
    def test_switch_failure_preserves_old_selection(self):
        client, router = self.client()
        client.connect()
        router.fail_select = True
        with self.assertRaises(HTTPError):
            client.select_model('flint')
        self.assertEqual(client.model_key, 'qwen')
        self.assertTrue(client.ready)

    @patch.object(qwen.time, 'sleep', lambda _: None)
    def test_external_switch_reselects_before_generation(self):
        client, router = self.client()
        client.connect()
        router.selected = qwen.MODELS['flint']
        client.generate([('user','hello')], lambda _:None)
        self.assertEqual(router.requests[0][0], '/api/select')
        self.assertEqual(router.selected, qwen.MODEL)

    @patch.object(qwen.time, 'sleep', lambda _: None)
    def test_streaming_utf8_and_final_validation(self):
        client, router = self.client(stream=True)
        client.connect()
        pieces = []
        answer, _ = client.generate([('user','hi')], pieces.append)
        self.assertEqual(pieces, ['你','好'])
        self.assertEqual(answer, '你好')
        router.fail_stream = True
        with self.assertRaisesRegex(qwen.ClientError, 'decode failed'):
            client.generate([('user','hi')], lambda _:None)

    def test_legacy_rejects_flint_without_post(self):
        with ClientTests().service() as (client, requests):
            client.connect()
            with self.assertRaisesRegex(qwen.ClientError, '旧版'):
                client.select_model('flint')
            self.assertEqual(requests, [])

    @patch.object(qwen.threading, 'Thread', ImmediateThread)
    @patch.object(qwen.time, 'sleep', lambda _: None)
    def test_ui_histories_are_separate_and_clear_is_local(self):
        client, router = self.client()
        client.connect()
        ui = qwen.ChatUI(client, None)
        ui.submit('Qwen question')
        ui.drain_events()
        qwen_history = list(ui.history)
        ui.submit('/model flint')
        ui.drain_events()
        self.assertEqual(ui.active_model, 'flint')
        self.assertEqual(ui.history, [])
        ui.submit('Flint question')
        ui.drain_events()
        self.assertEqual(router.requests[-1][1]['messages'][0]['content'], 'Flint question')
        ui.reset()
        self.assertEqual(ui.histories['flint'], [])
        self.assertEqual(ui.histories['qwen'], qwen_history)
        ui.switch_model('qwen')
        ui.drain_events()
        self.assertEqual(ui.history, qwen_history)

    @patch.object(qwen.threading, 'Thread', ImmediateThread)
    @patch.object(qwen.time, 'sleep', lambda _: None)
    def test_ui_error_never_commits_partial_reply(self):
        client, router = self.client(stream=True)
        client.connect()
        ui = qwen.ChatUI(client, None)
        router.fail_stream = True
        ui.submit('bad generation')
        ui.drain_events()
        self.assertEqual(ui.history, [])
        self.assertEqual(ui.messages[-1][0], '错误')
        self.assertFalse(ui.busy)
        self.assertIn('你好', ui.messages[-2][1])

    def test_new_options_defaults_and_validation(self):
        args = qwen.parse_args(['--model','flint'])
        self.assertEqual(args.model, 'flint')
        self.assertEqual(args.load_timeout, 1800)
        self.assertIn('dsw-l3wna78bngwh3pfple-22', args.ssh_host)
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--model','unknown'], ['--temperature','nan'], ['--timeout','inf'], ['--load-timeout','0']):
                with self.assertRaises(SystemExit):
                    qwen.parse_args(argv)


class LayoutTests(unittest.TestCase):
    def test_chinese_width_wrapping_and_controls(self):
        self.assertEqual(qwen.cell_width('中文a'), 5)
        self.assertEqual(qwen.cell_width('e\u0301'), 1)
        self.assertEqual(qwen.wrap_cells('你好世界', 4), ['你好','世界'])
        self.assertEqual(qwen.clip_cells('你好世界', 5), '你好')
        self.assertNotIn('\x1b', qwen.clean_terminal_text('\x1b[31mhello\x00'))
        self.assertEqual(qwen.previous_cursor('ae\u0301中', 3), 1)
        self.assertEqual(qwen.next_cursor('ae\u0301中', 1), 3)
        text, cursor = qwen.input_window('你好世界hello', 5, 6)
        self.assertLessEqual(qwen.cell_width(text), 6)
        self.assertLess(cursor, 6)

    def test_narrow_layout_has_no_overlap_or_out_of_bounds(self):
        class Curses:
            error = RuntimeError
            A_BOLD = 1
            A_DIM = 2
            @staticmethod
            def color_pair(_): return 0
        class Screen:
            def __init__(self, height, width): self.height, self.width, self.writes = height, width, []
            def getmaxyx(self): return self.height, self.width
            def erase(self): self.writes = []
            def addstr(self, row, col, text, style):
                assert 0 <= row < self.height
                assert col + qwen.cell_width(text) < self.width
                self.writes.append((row,col,text))
            def refresh(self): pass
            def move(self, row, col):
                assert 0 <= row < self.height and 0 <= col < self.width
        client, _ = MultiModelTests().client()
        client.ready = True
        for dimensions in ((12,40),(20,80),(30,150),(8,30)):
            ui = qwen.ChatUI(client, Curses)
            ui.messages = [('你','你好'*80),('Flint','答复'*100)]
            ui.input, ui.cursor = '中文输入'*20, 70
            screen = Screen(*dimensions)
            ui.paint(screen)
            self.assertTrue(screen.writes)


class HTTPWireTests(unittest.TestCase):
    def test_real_http_buffered_and_streaming(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        router = RouterOpener()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(router.health()).encode())
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                request = qwen.urllib.request.Request('http://localhost'+self.path, data=body)
                response = router.open(request)
                self.send_response(202 if self.path == '/api/select' else 200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                self.wfile.write(response.read())
        server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for extra in (['--no-stream'], []):
                args = qwen.parse_args(['--url', 'http://127.0.0.1:'+str(server.server_port),
                    '--no-tunnel', '--model','flint', '-n','16']+extra)
                client = qwen.Client(args)
                client.connect()
                pieces = []
                result, _ = client.generate([('user','问题')], pieces.append)
                self.assertEqual(result,'你好')
                self.assertEqual(''.join(pieces),'你好')
                client.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class RenderingTests(unittest.TestCase):
    def test_flint_history_keeps_final_answer(self):
        ui = qwen.ChatUI(qwen.Client(qwen.parse_args(['--model', 'flint'])), None)
        ui.events.put(('done', ([('user', '问题')], '<think>reason</think>答案', {'done_reason': 'stop'})))
        ui.drain_events()
        self.assertEqual(ui.history[-1], ('assistant', '答案'))
        self.assertEqual(qwen.assistant_history('qwen', '<think>x</think>y'), '<think>x</think>y')
        self.assertEqual(qwen.assistant_history('flint', '<think>partial'), '<think>partial')

    def test_output_limit_is_visible(self):
        ui = qwen.ChatUI(qwen.Client(qwen.parse_args([])), None)
        ui.events.put(('done', ([], 'partial', {'done_reason': 'length', 'eval_count': 8})))
        ui.drain_events()
        self.assertIn('达到输出上限', ui.status)
        self.assertIn('-n', ui.status)

    def test_generation_rate_supports_both_workers(self):
        self.assertEqual(qwen.generation_rate({'lifeai': {'decode_tokens_per_second': 12.5}}), 12.5)
        self.assertEqual(qwen.generation_rate({'eval_count': 8, 'eval_duration': 4_000_000_000}), 2)
        self.assertEqual(qwen.generation_rate({'eval_count': 8, 'eval_duration': 0}), 0)
        self.assertEqual(qwen.generation_rate({'lifeai': None, 'eval_count': 8, 'eval_duration': 4_000_000_000}), 2)

    def test_long_input_only_measures_the_visible_cursor_window(self):
        with patch.object(qwen, 'cell_width', wraps=qwen.cell_width) as measure:
            visible, cursor = qwen.input_window('a' * 10000, 10000, 80)
        self.assertEqual(visible, 'a' * 79)
        self.assertEqual(cursor, 79)
        self.assertLess(measure.call_count, 200)

    def test_wrapped_rows_reused_until_text_model_or_width_changes(self):
        class Colors:
            A_BOLD, A_DIM = 1, 2
            @staticmethod
            def color_pair(number): return number * 4
        args = qwen.parse_args([])
        ui = qwen.ChatUI(qwen.Client(args), Colors())
        ui.messages.append(('你', '你好，世界'))
        with patch.object(qwen, 'wrap_cells', wraps=qwen.wrap_cells) as wrapped:
            first = ui.message_rows(40)
            self.assertIs(first, ui.message_rows(40))
            self.assertEqual(wrapped.call_count, 1)
            ui.messages.append(('Qwen', '回答'))
            self.assertIsNot(first, ui.message_rows(40))
            self.assertEqual(wrapped.call_count, 3)
            ui.message_rows(60)
            self.assertEqual(wrapped.call_count, 5)
            ui.active_model = 'flint'
            self.assertNotEqual(first, ui.message_rows(60))


if __name__ == '__main__':
    unittest.main()
