#!/usr/bin/env python3
"""LifeAI terminal chat: switch between Qwen and Flint over one SSH connection.

Linux: python3 qwen.py --install; then run qwen (TUI) or qwen '你的问题'.
Requires Python 3.9+, OpenSSH, and a local SSH identity authorized by the DSW.
No model weights, Julia installation, or third-party Python packages are needed.
"""
import argparse
import contextlib
import json
import locale
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

MODEL = "Qwen/Qwen3-8B"  # Legacy import compatibility.
MODELS = {"qwen": MODEL, "flint": "RecursiveSI/flint"}
MODEL_LABELS = {"qwen": "Qwen3-8B", "flint": "Flint"}
DSW_HOST = "dsw-notebook-dsw-l3wna78bngwh3pfple-22.vpc-0jlbz3pri2042fd5xw2ov.instance-forward.dsw.cn-wulanchabu.aliyuncs.com"
DEFAULT_URL = "http://127.0.0.1:11435"
INSTALL_MARKER = "# LifeAI Qwen CLI launcher"


class ClientError(Exception):
    pass


def render_prompt(messages):
    # The frozen Qwen3 non-thinking template for ordinary system/user/assistant turns.
    text = "".join("<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
                   for role, content in messages)
    return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def error_text(error):
    if isinstance(error, urllib.error.HTTPError):
        try:
            detail = json.loads(error.read().decode("utf-8"))
            value = detail.get("error", detail)
            if isinstance(value, dict):
                value = value.get("message", value)
            detail = str(value)
        except (ValueError, UnicodeError):
            detail = str(error)
        if "context" in detail.lower() or "prompt" in detail.lower():
            return "当前提示词与请求输出超过服务可用上下文；请缩短对话、降低 --max-tokens，或按 Ctrl+L 开始新对话。服务详情: " + detail
        return f"服务返回 HTTP {error.code}: {detail}"
    return str(error)


class Client:
    def __init__(self, args):
        self.args = args
        self.url = args.url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.path or parsed.query or parsed.fragment or parsed.username):
            raise ClientError("--url 请填写服务根地址，例如 http://127.0.0.1:11435")
        self.parsed = parsed
        self.tunnel = None
        self.tunnel_log = None
        self.context_tokens = None
        self.max_new_tokens = None
        self.model_key = args.model
        self.models = dict(MODELS)
        self.router = False
        self.ready = False
        self.connected = False
        self.service_model_id = None
        self.last_health = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.connection_label = "未连接"

    @property
    def model_id(self):
        return self.models[self.model_key]

    def request_json(self, path, payload=None, timeout=None):
        request = self.url + path
        if payload is not None:
            request = urllib.request.Request(request,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
        with self.opener.open(request, timeout=timeout or self.args.timeout) as response:
            value = json.load(response)
        if not isinstance(value, (dict, list)):
            raise ClientError("服务返回了无效的 JSON 响应。")
        return value

    def _read_models(self, value):
        entries = value.get("models", []) if isinstance(value, dict) else value
        if isinstance(entries, dict):
            entries = [dict(item, key=key) if isinstance(item, dict) else
                       {"key": key, "id": item} for key, item in entries.items()]
        if not isinstance(entries, list):
            raise ClientError("服务模型列表格式无效。")
        for item in entries:
            if isinstance(item, str):
                item = {"id": item}
            if not isinstance(item, dict):
                continue
            model_id = item.get("id", item.get("model", item.get("model_id")))
            key = item.get("key", item.get("name"))
            if key not in MODELS:
                key = next((k for k, known in MODELS.items() if known == model_id), None)
            if key and isinstance(model_id, str):
                self.models[key] = model_id

    def health(self, timeout=3):
        try:
            value = self.request_json("/healthz", timeout=timeout)
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict):
            raise ClientError("服务 /healthz 响应无效。")
        if "models" in value:
            self.router = True
            self._read_models(value)
        model = value.get("model", value.get("selected_model", value.get("selected")))
        if isinstance(model, dict):
            model = model.get("id", model.get("model"))
        if model is not None and not isinstance(model, str):
            raise ClientError("服务 /healthz 模型标识无效。")
        model = MODELS.get(model, model)
        if not self.router and model != MODEL:
            raise ClientError("该端口不是兼容的 LifeAI 模型服务；请检查 --url。")
        self.service_model_id = model
        self.last_health = value
        return value

    def _apply_limits(self, health):
        limits = health.get("limits", health)
        if not isinstance(limits, dict):
            raise ClientError("服务 /healthz 限制格式无效。")
        for field in ("context_tokens", "max_new_tokens"):
            limit = limits.get(field)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ClientError(f"服务 /healthz 缺少有效 {field}，无法确认生成限制。")
            setattr(self, field, limit)
        self.validate_limits()

    def _status(self, progress, message):
        if progress and message != getattr(self, "_last_progress", None):
            self._last_progress = message
            progress(message)

    def connect(self, progress=None):
        self.ready = False
        self._status(progress, "正在连接模型服务…")
        health = self.health()
        if health is None:
            self._open_tunnel(progress)
        else:
            self.connection_label = "已连接"
        self.connected = True
        return self.select_model(self.model_key, progress=progress)

    def _open_tunnel(self, progress):
        if self.args.no_tunnel or self.parsed.hostname not in ("localhost", "127.0.0.1"):
            raise ClientError("服务无法访问，请检查地址或先启动服务。Ctrl+R 可重试。")
        ssh = shutil.which("ssh")
        if not ssh:
            raise ClientError("未找到 ssh；Linux 上请安装 openssh-client。")
        port = self.parsed.port or 80
        if self.parsed.scheme != "http":
            raise ClientError("自动 SSH 转发请使用 http://127.0.0.1:端口。")
        if self.tunnel is not None and self.tunnel.poll() is None:
            raise ClientError("SSH 通道仍在运行，但模型服务不可访问。请检查远端服务后按 Ctrl+R。")
        command = [ssh, "-N", "-T", "-p", str(self.args.ssh_port),
                   "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=10",
                   "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
                   "-o", "StrictHostKeyChecking=accept-new",
                   "-L", f"127.0.0.1:{port}:127.0.0.1:11435"]
        if self.args.ssh_key:
            key = Path(self.args.ssh_key).expanduser()
            if not key.is_file():
                raise ClientError(f"SSH 私钥文件不存在: {key}")
            command += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
        if self.args.known_hosts:
            command += ["-o", "UserKnownHostsFile=" + str(Path(self.args.known_hosts).expanduser())]
        command += [self.args.ssh_user + "@" + self.args.ssh_host]
        self.connection_label = "SSH 连接中"
        self._status(progress, "正在通过 SSH 连接 DSW…")
        if self.tunnel_log is not None:
            self.tunnel_log.close()
        self.tunnel_log = tempfile.TemporaryFile(mode="w+b")
        self.tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=self.tunnel_log)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.tunnel.poll() is not None:
                self.tunnel_log.seek(0)
                detail = self.tunnel_log.read(4096).decode("utf-8", errors="replace").strip()
                raise ClientError("SSH 连接失败。请使用已获 DSW 授权的密钥，"
                                  "可加 --ssh-key ~/.ssh/你的私钥。\n" + detail)
            if self.health(timeout=1) is not None:
                self.connection_label = "SSH 已连接"
                return
            time.sleep(0.2)
        raise ClientError("SSH 已启动，但模型服务不可访问。请确认 DSW 与 LifeAI 服务正在运行。")

    def select_model(self, key, progress=None):
        if key not in MODELS:
            raise ClientError("模型可选 qwen 或 flint。")
        previous = self.model_key
        target = self.models[key]
        health = self.health()
        if health is None:
            self.ready = False
            raise ClientError("模型服务连接中断；按 Ctrl+R 重新连接。")
        if not self.router and key != "qwen":
            raise ClientError("当前是旧版 Qwen 服务，不支持 Flint；请启动新版多模型服务。")
        self.ready = False
        deadline = time.monotonic() + self.args.load_timeout
        requested = False
        self._status(progress, f"正在准备 {MODEL_LABELS[key]}，首次加载可能需要数分钟…")
        try:
            while True:
                health = self.health()
                if health and health.get("ready") is True and self.service_model_id == target:
                    self.model_key = key
                    self._apply_limits(health)
                    self.ready = True
                    return health
                loading = bool(health and (health.get("state") == "loading" or health.get("loading")))
                if requested and health and health.get("state") == "error" and health.get("last_error"):
                    raise ClientError(str(health["last_error"]))
                if time.monotonic() >= deadline:
                    raise ClientError(f"等待 {MODEL_LABELS[key]} 加载超时；可增大 --load-timeout 或按 Ctrl+R 重试。")
                if self.router and health and not loading:
                    try:
                        response = self.request_json("/api/select", {"model": target},
                                                     timeout=max(1, deadline - time.monotonic()))
                        if isinstance(response, dict) and response.get("error"):
                            raise ClientError(str(response["error"]))
                        requested = True
                    except urllib.error.HTTPError as error:
                        if error.code != 409:
                            raise
                        error.close()
                        self._status(progress, "服务正在处理其他请求，等待后切换模型…")
                elif loading:
                    loading_id = health.get("loading_model", self.service_model_id)
                    requested = requested or loading_id == target
                    if loading_id and loading_id != target:
                        self._status(progress, f"服务正在加载 {loading_id}；完成后切换到 {MODEL_LABELS[key]}…")
                    else:
                        phase = {"quantizing": "GPU 量化", "preparing_native": "准备原生 FP4",
                                 "warming": "预热推理", "loading_artifact": "读取量化权重"}.get(health.get("phase"), "加载权重")
                        elapsed = health.get("load_seconds")
                        duration = f" · {int(elapsed)}s" if isinstance(elapsed, (int, float)) else ""
                        self._status(progress, f"{MODEL_LABELS[key]} 加载中 · {phase}{duration}（Ctrl+C 可退出客户端）")
                elif health is None:
                    self._status(progress, "等待模型服务恢复连接…")
                time.sleep(0.5)
        except Exception:
            self.model_key = previous
            self.ready = False
            health = self.health()
            if health and health.get("ready") is True and self.service_model_id == self.models[previous]:
                try:
                    self._apply_limits(health)
                    self.ready = True
                except ClientError:
                    pass
            raise

    def validate_limits(self):
        if self.context_tokens is None or self.max_new_tokens is None:
            raise ClientError("尚未取得服务上下文限制，请先连接服务。")
        if self.args.max_tokens > self.max_new_tokens:
            raise ClientError(f"--max-tokens={self.args.max_tokens} 超过服务输出上限 "
                              f"{self.max_new_tokens}（上下文 {self.context_tokens} tokens）。")
        if self.args.max_tokens >= self.context_tokens:
            raise ClientError(f"请求输出 {self.args.max_tokens} tokens 未给提示词留下空间；"
                              f"服务上下文为 {self.context_tokens} tokens，请降低 --max-tokens。")

    def close(self):
        if self.tunnel is not None and self.tunnel.poll() is None:
            self.tunnel.terminate()
            try:
                self.tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tunnel.kill()
                self.tunnel.wait()
        if self.tunnel_log is not None:
            self.tunnel_log.close()

    def generate(self, messages, emit, progress=None):
        if not self.ready:
            self.connect(progress=progress)
        health = self.health()
        if not health or health.get("ready") is not True or self.service_model_id != self.model_id:
            self.select_model(self.model_key, progress=progress)
        self._apply_limits(self.last_health)
        options = {"temperature": self.args.temperature, "num_predict": self.args.max_tokens}
        if self.router:
            payload = {"model": self.model_id,
                       "messages": [{"role": role, "content": content} for role, content in messages],
                       "stream": not self.args.no_stream, "options": options}
            endpoint = "/api/chat"
        else:
            if self.args.temperature != 0:
                raise ClientError("旧版 Qwen 服务只支持 --temperature 0。")
            payload = {"model": MODEL, "prompt": render_prompt(messages), "raw": True,
                       "think": False, "stream": not self.args.no_stream,
                       "options": dict(options, repeat_penalty=1, repeat_last_n=0, num_ctx=self.context_tokens)}
            endpoint = "/api/generate"
        request = urllib.request.Request(self.url + endpoint,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
        pieces = []
        with self.opener.open(request, timeout=self.args.timeout) as response:
            final = None
            records = [response.read()] if self.args.no_stream else response
            for line in records:
                if not line.strip():
                    continue
                item = json.loads(line.decode("utf-8"))
                if not isinstance(item, dict):
                    raise ClientError("服务返回了无效的生成事件。")
                if "error" in item:
                    raise ClientError(str(item["error"]))
                piece = item.get("response", "")
                if not isinstance(piece, str):
                    raise ClientError("服务返回了无效的文本片段。")
                if piece:
                    pieces.append(piece)
                    emit(piece)
                if item.get("done") is True:
                    final = item
                    break
            if final is None:
                raise ClientError("连接中断，服务没有返回完整回答；本轮未写入对话历史。")
        return "".join(pieces), final


def cell_width(text):
    return sum(0 if unicodedata.combining(c) else
               2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def clean_terminal_text(text):
    # Do not interpret escape sequences contained in model output.
    return "".join(c if c == "\n" or (c.isprintable() and c != "\x1b") else
                   " " if c == "\t" else "" for c in text)


def wrap_cells(text, width):
    lines, current, used = [], "", 0
    for char in clean_terminal_text(text):
        if char == "\n":
            lines.append(current)
            current, used = "", 0
            continue
        size = cell_width(char)
        if used + size > width and current:
            lines.append(current)
            current, used = "", 0
        current += char
        used += size
    lines.append(current)
    return lines


def clip_cells(text, width):
    result, used = "", 0
    for char in clean_terminal_text(text).replace("\n", " "):
        size = cell_width(char)
        if used + size > width:
            break
        result += char
        used += size
    return result


def previous_cursor(text, position):
    position = max(0, position - 1)
    while position > 0 and unicodedata.combining(text[position]):
        position -= 1
    return position


def next_cursor(text, position):
    position = min(len(text), position + 1)
    while position < len(text) and unicodedata.combining(text[position]):
        position += 1
    return position


def input_window(text, cursor, width):
    width = max(1, width)
    cursor = max(0, min(len(text), cursor))
    start, used = cursor, 0
    while start > 0:
        previous = previous_cursor(text, start)
        size = cell_width(text[previous:start])
        if used + size >= width:
            break
        start, used = previous, used + size
    return clip_cells(text[start:], width), used


def generation_rate(result):
    extra = result.get("lifeai")
    rate = extra.get("decode_tokens_per_second") if isinstance(extra, dict) else None
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and 0 <= rate < float("inf"):
        return float(rate)
    count, duration = result.get("eval_count"), result.get("eval_duration")
    if type(count) is int and type(duration) in (int, float) and count >= 0 and 0 < duration < float("inf"):
        return count * 1e9 / duration
    return 0.0


def assistant_history(model, answer):
    # Flint emits its reasoning block in the visible stream. Retain the final
    # answer as ordinary assistant content for its next chat-template turn.
    if model == "flint" and answer.startswith("<think>") and "</think>" in answer:
        return answer.split("</think>", 1)[1].lstrip()
    return answer


class ChatUI:
    def __init__(self, client, curses):
        self.client, self.c = client, curses
        self.active_model = client.model_key
        self.histories = {key: self.new_history() for key in MODELS}
        self.transcripts = {key: [] for key in MODELS}
        self.events = queue.Queue()
        self.input = ""
        self.cursor = 0
        self.scroll = 0
        self.busy = False
        self.loading_model = None
        self._rows_key = None
        self._rows_cache = []
        self.status = "输入问题后按 Enter；Ctrl+T 切换模型。历史仅保存在当前会话。"

    def new_history(self):
        return [("system", self.client.args.system)] if self.client.args.system else []

    @property
    def history(self):
        return self.histories[self.active_model]

    @history.setter
    def history(self, value):
        self.histories[self.active_model] = value

    @property
    def messages(self):
        return self.transcripts[self.active_model]

    @messages.setter
    def messages(self, value):
        self.transcripts[self.active_model] = value

    def progress(self, message):
        self.events.put(("status", message))

    def start_connection(self):
        if self.busy:
            return
        self.busy = True
        self.loading_model = self.active_model
        self.status = "正在连接 DSW 模型服务…"
        def worker():
            try:
                self.client.connect(progress=self.progress)
                self.events.put(("selected", self.client.model_key))
            except Exception as error:
                self.events.put(("connection_error", error_text(error)))
        threading.Thread(target=worker, daemon=True).start()

    def switch_model(self, key):
        if key not in MODELS:
            self.status = "用法：/model qwen 或 /model flint。"
            return
        if self.busy:
            self.status = "请等当前加载或回答完成后再切换模型。"
            return
        if key == self.active_model and self.client.ready:
            self.status = f"当前已经是 {MODEL_LABELS[key]}。"
            return
        self.busy = True
        self.loading_model = key
        self.status = f"正在切换至 {MODEL_LABELS[key]}；该模型历史将独立保留。"
        def worker():
            try:
                if not self.client.connected:
                    self.client.model_key = key
                    self.client.connect(progress=self.progress)
                else:
                    self.client.select_model(key, progress=self.progress)
                self.events.put(("selected", key))
            except Exception as error:
                self.events.put(("connection_error", error_text(error)))
        threading.Thread(target=worker, daemon=True).start()

    def submit(self, question):
        command = question.strip()
        if not command:
            return
        if command in ("/quit", "/exit"):
            raise KeyboardInterrupt
        if command == "/clear":
            self.reset()
            return
        if command in ("/help", "/model", "/models"):
            self.status = f"当前 {MODEL_LABELS[self.active_model]} · /model qwen|flint 切换 · /clear 清空当前历史"
            return
        if command.startswith("/model ") or command in ("/qwen", "/flint"):
            self.switch_model(command.split(maxsplit=1)[1].strip() if command.startswith("/model ") else command[1:])
            return
        if self.busy:
            self.status = "正在加载或回答，请稍候。"
            return
        if not self.client.ready:
            self.status = "模型尚未就绪；按 Ctrl+R 重试连接。"
            return
        self.busy = True
        self.messages += [("你", question), (MODEL_LABELS[self.active_model], "")]
        self.scroll = 0
        pending = self.history + [("user", question)]
        self.status = f"{MODEL_LABELS[self.active_model]} 正在生成…"
        def worker():
            try:
                answer, result = self.client.generate(pending,
                    lambda piece: self.events.put(("piece", piece)), progress=self.progress)
                self.events.put(("done", (pending, answer, result)))
            except Exception as error:
                self.events.put(("error", error_text(error)))
        threading.Thread(target=worker, daemon=True).start()

    def reset(self):
        if self.busy:
            self.status = "请等当前加载或回答完成后再开新对话。"
            return
        self.messages = []
        self.history = self.new_history()
        self.input, self.cursor, self.scroll = "", 0, 0
        self.status = f"已清空 {MODEL_LABELS[self.active_model]} 的历史；另一模型历史保留。"

    def drain_events(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "status":
                self.status = value
            elif kind == "piece":
                if self.messages:
                    role, current = self.messages[-1]
                    self.messages[-1] = (role, current + value)
            elif kind == "selected":
                self.active_model = value
                self.busy = False
                self.loading_model = None
                self.input, self.cursor, self.scroll = "", 0, 0
                self.status = (f"{MODEL_LABELS[value]} 已就绪 · 上下文 {self.client.context_tokens} tokens · "
                               f"输出上限 {self.client.max_new_tokens} · 当前历史 {len(self.history)} 条")
            elif kind == "done":
                pending, answer, result = value
                self.history = pending + [("assistant", assistant_history(self.active_model, answer))]
                self.busy = False
                rate = generation_rate(result)
                label = "达到输出上限" if result.get("done_reason") == "length" else "回答完成"
                self.status = f"{label} · {result.get('eval_count', 0)} tokens · {rate:.1f} tok/s"
                if result.get("done_reason") == "length":
                    self.status += " · 可增大 -n"
            else:
                self.busy = False
                self.loading_model = None
                self.messages.append(("错误", value))
                self.status = ("切换/连接失败，原历史已保留；Ctrl+R 重试。" if kind == "connection_error"
                               else "请求失败，本轮未写入历史；可重试或 Ctrl+L 清空当前对话。")

    def message_rows(self, body_width):
        key = (body_width, self.active_model, tuple(self.messages))
        if key == self._rows_key:
            return self._rows_cache
        c, rows = self.c, []
        for role, text in self.messages:
            style = c.color_pair(1) if role == "你" else c.color_pair(3) if role == "错误" else c.color_pair(2)
            rows.append((role + " ▸", style | c.A_BOLD))
            rows.extend((line, 0) for line in wrap_cells(text, body_width - 1))
            rows.append(("", 0))
        if not rows:
            rows = [("Qwen 与 Flint 共用此界面，分别保留对话历史。", c.A_DIM),
                    ("Ctrl+T 切换模型，或输入 /model flint。", c.A_DIM),
                    ("模型加载时仍可滚动和编辑；请在就绪后发送。", c.A_DIM)]
        self._rows_key, self._rows_cache = key, rows
        return rows

    def paint(self, screen):
        c = self.c
        height, width = screen.getmaxyx()
        screen.erase()
        def put(row, col, text, style=0):
            if 0 <= row < height and 0 <= col < width - 1:
                with contextlib.suppress(c.error):
                    screen.addstr(row, col, clip_cells(text, width - col - 1), style)
        if height < 12 or width < 40:
            put(0, 0, "请放大终端至至少 40 列、12 行；Ctrl+C 退出。")
            screen.refresh()
            return
        left = max(2, (width - 120) // 2)
        body_width = width - 2 * left
        put(0, left, "LifeAI · 终端聊天", c.A_BOLD | c.color_pair(1))
        connection = self.client.connection_label
        put(0, max(left + 22, width - left - cell_width(connection)), connection, c.color_pair(2))
        tabs = "  ".join(("[" + MODEL_LABELS[key] + "]") if key == self.active_model
                         else MODEL_LABELS[key] for key in MODELS)
        state = ("加载 " + MODEL_LABELS[self.loading_model] if self.loading_model
                 else "生成中" if self.busy else "就绪" if self.client.ready else "未就绪")
        if self.busy:
            state += " " + "◐◓◑◒"[int(time.monotonic() * 4) % 4]
        put(1, left, tabs + "  ·  " + state, c.A_BOLD)
        put(2, left, "─" * body_width, c.A_DIM)
        available = height - 8
        rows = self.message_rows(body_width)
        self.scroll = min(self.scroll, max(0, len(rows) - available))
        end = len(rows) - self.scroll
        visible = rows[max(0, end - available):end]
        for row, (text, style) in enumerate(visible, 3):
            put(row, left, text, style)
        put(height - 5, left, "─" * body_width, c.A_DIM)
        put(height - 4, left, "›", c.A_BOLD | c.color_pair(1))
        visible_input, cursor_column = input_window(self.input, self.cursor, body_width - 3)
        put(height - 4, left + 2, visible_input)
        put(height - 3, left, self.status, c.A_DIM)
        put(height - 2, left, "Enter 发送 · Ctrl+T 切换 · Ctrl+L 清空 · Ctrl+R 重连 · /help", c.A_DIM)
        with contextlib.suppress(c.error):
            screen.move(height - 4, min(width - 2, left + 2 + cursor_column))
            screen.refresh()

    def run(self, screen):
        c = self.c
        screen.keypad(True)
        screen.timeout(50)
        with contextlib.suppress(c.error):
            c.curs_set(1)
        if c.has_colors():
            c.start_color()
            with contextlib.suppress(c.error):
                c.use_default_colors()
                c.init_pair(1, c.COLOR_CYAN, -1)
                c.init_pair(2, c.COLOR_GREEN, -1)
                c.init_pair(3, c.COLOR_RED, -1)
        if not self.client.ready:
            self.start_connection()
        while True:
            self.drain_events()
            self.paint(screen)
            try:
                key = screen.get_wch()
            except c.error:
                continue
            if key in ("\x03", "\x04"):
                return
            if key == "\x0c":
                self.reset()
            elif key in ("\x14", getattr(c, "KEY_F2", -999)):
                self.switch_model("flint" if self.active_model == "qwen" else "qwen")
            elif key == "\x12":
                self.start_connection()
            elif key in (c.KEY_PPAGE, c.KEY_NPAGE):
                self.scroll = max(0, self.scroll + (5 if key == c.KEY_PPAGE else -5))
            elif key == c.KEY_RESIZE:
                continue
            elif key in ("\n", "\r", c.KEY_ENTER):
                if not self.busy:
                    question = self.input
                    self.input, self.cursor = "", 0
                    if question.strip() in ("/quit", "/exit"):
                        return
                    self.submit(question)
            elif key in ("\x7f", "\b", c.KEY_BACKSPACE):
                if self.cursor:
                    previous = previous_cursor(self.input, self.cursor)
                    self.input = self.input[:previous] + self.input[self.cursor:]
                    self.cursor = previous
            elif key == c.KEY_DC:
                self.input = self.input[:self.cursor] + self.input[next_cursor(self.input, self.cursor):]
            elif key in (c.KEY_LEFT, "\x02"):
                self.cursor = previous_cursor(self.input, self.cursor)
            elif key in (c.KEY_RIGHT, "\x06"):
                self.cursor = next_cursor(self.input, self.cursor)
            elif key in (c.KEY_HOME, "\x01"):
                self.cursor = 0
            elif key in (c.KEY_END, "\x05"):
                self.cursor = len(self.input)
            elif key == "\x15":
                self.input, self.cursor = "", 0
            elif isinstance(key, str) and key.isprintable():
                self.input = self.input[:self.cursor] + key + self.input[self.cursor:]
                self.cursor += len(key)


def install(prefix):
    prefix = Path(prefix).expanduser().resolve()
    target = prefix / "share/lifeai-qwen/qwen.py"
    launcher = prefix / "bin/qwen"
    if launcher.exists() and INSTALL_MARKER not in launcher.read_text(errors="replace"):
        raise ClientError(f"已有其他程序占用 {launcher}，未覆盖。请使用 --prefix 指定安装目录。")
    target.parent.mkdir(parents=True, exist_ok=True)
    launcher.parent.mkdir(parents=True, exist_ok=True)
    if Path(__file__).resolve() != target:
        shutil.copy2(__file__, target)
    launcher.write_text("#!/bin/sh\n" + INSTALL_MARKER + "\nexec " +
                        shlex.quote(sys.executable) + " " + shlex.quote(str(target)) + ' "$@"\n')
    launcher.chmod(0o755)
    print(f"已安装: {launcher}\n运行 qwen 进入聊天界面，或 qwen --model flint '你的问题' 单次提问。")
    if prefix == (Path.home() / ".local").resolve():
        shell = Path(os.environ.get("SHELL", "/bin/bash")).name
        profile_name = {"bash": ".bashrc", "zsh": ".zshrc"}.get(shell)
        if profile_name:
            profile = Path.home() / profile_name
            original = profile.read_text() if profile.exists() else ""
            marker = "# LifeAI Qwen CLI PATH"
            if marker not in original:
                backup = target.parent / (profile_name.lstrip(".") + ".before-qwen")
                if not backup.exists():
                    backup.write_text(original)
                profile.write_text(original + "\n" + marker + "\nexport PATH=" +
                                   shlex.quote(str(launcher.parent)) + ':"$PATH"\n')
                print(f"已在 {profile} 配置 PATH；原文件备份: {backup}")
    if str(launcher.parent) not in os.environ.get("PATH", "").split(os.pathsep):
        print("当前终端先执行: export PATH=" + shlex.quote(str(launcher.parent)) + ':"$PATH"')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LifeAI CLI：Qwen/Flint 模型切换、SSH 连接与终端聊天",
        epilog="直接运行 qwen 打开 TUI；qwen '问题' 单次提问；cat 文件 | qwen 处理输入。")
    p.add_argument("prompt", nargs="*", help="要提问的内容；不填写则进入 TUI")
    p.add_argument("--install", action="store_true", help="安装 qwen 命令到 ~/.local/bin")
    p.add_argument("--prefix", default=str(Path.home() / ".local"), help="安装目录前缀")
    p.add_argument("--url", default=os.environ.get("QWEN_URL", DEFAULT_URL), help="服务根地址")
    p.add_argument("--ssh-key", default=os.environ.get("QWEN_SSH_KEY"), help="DSW 已授权的 SSH 私钥路径")
    p.add_argument("--known-hosts", help="可选的 SSH known_hosts 文件")
    p.add_argument("--ssh-host", default=os.environ.get("QWEN_SSH_HOST", DSW_HOST), help="SSH 主机")
    p.add_argument("--ssh-user", default="root", help="SSH 用户，默认 root")
    p.add_argument("--ssh-port", type=int, default=22, help="SSH 端口，默认 22")
    p.add_argument("--no-tunnel", action="store_true", help="只使用现有连接，不创建 SSH 转发")
    p.add_argument("--model", choices=tuple(MODELS), default=os.environ.get("LIFEAI_MODEL", "qwen"), help="选择 qwen 或 flint（默认 qwen）")
    p.add_argument("--status", action="store_true", help="连接并显示所选模型服务状态")
    p.add_argument("-n", "--max-tokens", type=int, default=256, help="每次回答最多 tokens（正整数，上限由服务决定）")
    p.add_argument("--timeout", type=float, default=300, help="生成请求等待秒数")
    p.add_argument("--load-timeout", type=float, default=1800, help="模型加载最长等待秒数（默认 1800）")
    p.add_argument("--temperature", type=float, default=0.0, help="采样温度，0 为贪心（默认 0）")
    p.add_argument("--system", default="", help="可选的系统提示词")
    p.add_argument("--no-stream", action="store_true", help="等待完整回答后再输出")
    args = p.parse_args(argv)
    if args.model not in MODELS:
        p.error("LIFEAI_MODEL 必须为 qwen 或 flint")
    if args.max_tokens < 1:
        p.error("--max-tokens 必须为正整数")
    if not 1 <= args.ssh_port <= 65535 or not 0 < args.timeout < float("inf") or not 0 < args.load_timeout < float("inf"):
        p.error("SSH 端口必须为 1～65535；超时时间必须为有限正数")
    if not 0 <= args.temperature < float("inf"):
        p.error("--temperature 必须为有限非负数")
    return args


def main(argv=None):
    args = parse_args(argv)
    client = None
    try:
        if args.install:
            install(args.prefix)
            return 0
        question = " ".join(args.prompt)
        if not question and not args.status and not sys.stdin.isatty():
            question = sys.stdin.read().strip()
            if not question:
                raise ClientError("请输入问题，或在交互终端运行 qwen 打开聊天界面。")
        if not question and not args.status:
            if not sys.stdout.isatty() or os.environ.get("TERM", "") in ("", "dumb"):
                raise ClientError("TUI 需要交互终端；也可运行 qwen '你的问题'。")
            try:
                import curses
            except ImportError as error:
                raise ClientError("Python 缺少 curses；请安装系统 python3，或用单次提问方式。") from error
        client = Client(args)
        if args.status or question:
            client.connect(progress=lambda message: print(message, file=sys.stderr, flush=True))
        if args.status:
            print(json.dumps(client.health(), ensure_ascii=False, indent=2))
        elif question:
            messages = [("system", args.system)] if args.system else []
            client.generate(messages + [("user", question)],
                            lambda piece: (sys.stdout.write(clean_terminal_text(piece)), sys.stdout.flush()))
            print()
        else:
            with contextlib.suppress(locale.Error):
                locale.setlocale(locale.LC_ALL, "")
            curses.wrapper(ChatUI(client, curses).run)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ClientError, OSError, ValueError) as error:
        print("qwen: " + clean_terminal_text(error_text(error)), file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
