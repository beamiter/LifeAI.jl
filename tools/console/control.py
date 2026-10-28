#!/usr/bin/env python3
"""Start, stop or inspect this deployment's own model router."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'manifests/router.json'


def ticks(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text()
        return value[value.rfind(')') + 2:].split()[19]
    except (OSError, IndexError):
        return None


def saved_state():
    try:
        value = json.loads(STATE.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def owned(state):
    pid = state.get('pid')
    started = state.get('start_ticks')
    if (type(pid) is not int or pid <= 0 or not isinstance(started, str)
            or not started.isdigit() or ticks(pid) != started):
        return False
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return str(ROOT / 'server.py').encode() in command
    except OSError:
        return False


def status():
    state = saved_state()
    alive = owned(state)
    health = None
    if alive:
        try:
            with build_opener(ProxyHandler({})).open('http://127.0.0.1:11435/healthz', timeout=3) as r:
                health = json.load(r)
        except (OSError, ValueError):
            pass
    result = dict(running=alive, pid=state.get('pid'), health=health,
                  endpoint='http://127.0.0.1:11435', logs=str(ROOT / 'logs'))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def start(model):
    if owned(saved_state()):
        return status()
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 11435)) == 0:
            raise RuntimeError('Port 11435 is used by another service; it was left untouched.')
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'server.py'), '--config', str(ROOT / 'models.json'),
               '--default-model', model]
    with (ROOT / 'logs/router.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=log, cwd=ROOT, start_new_session=True)
    value = dict(pid=process.pid, start_ticks=ticks(process.pid), command=command, started_at=time.time())
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)
    return status()


def stop():
    state = saved_state()
    if owned(state):
        # The router gracefully stops its worker before exiting. Give its
        # 15-second SIGTERM-to-SIGKILL worker cleanup time to finish.
        try:
            descriptor = os.pidfd_open(state['pid'])
        except ProcessLookupError:
            return status()
        try:
            if not owned(state):
                return status()
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            except ProcessLookupError:
                return status()
        finally:
            os.close(descriptor)
        deadline = time.monotonic() + 60
        while owned(state) and time.monotonic() < deadline:
            time.sleep(.2)
        if owned(state):
            raise RuntimeError('Router did not finish cleanup; inspect logs before forcing it.')
    return status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop'])
    parser.add_argument('--model', choices=['qwen', 'flint'], default='flint')
    args = parser.parse_args()
    if args.action == 'status':
        status()
        return
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'start':
            start(args.model)
        else:
            stop()


if __name__ == '__main__':
    main()
