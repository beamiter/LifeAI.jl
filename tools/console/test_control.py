"""Ownership/lifecycle regression tests. No real process is started or signalled."""
import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import control


class OwnershipTests(unittest.TestCase):
    def test_stat_start_ticks_with_spaces_and_parentheses_in_comm(self):
        fields = ['S'] + ['0'] * 18 + ['90210'] + ['0'] * 3
        stat = '42 (worker name (nested)) ' + ' '.join(fields)
        with patch.object(Path, 'read_text', return_value=stat):
            self.assertEqual(control.ticks(42), '90210')

    def test_missing_or_truncated_stat_is_not_owned(self):
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            self.assertIsNone(control.ticks(42))
        with patch.object(Path, 'read_text', return_value='42 (worker) S 1'):
            self.assertIsNone(control.ticks(42))

    def test_same_start_ticks_and_exact_script_argument_are_required(self):
        state = dict(pid=42, start_ticks='1000')
        exact = str(control.ROOT / 'server.py').encode()
        with patch.object(control, 'ticks', return_value='1000'):
            with patch.object(Path, 'read_bytes', return_value=b'python3\0'+exact+b'\0--config\0x\0'):
                self.assertTrue(control.owned(state))
            for wrong in (exact+b'.bak', b'/another'+exact, b'--path='+exact):
                with patch.object(Path, 'read_bytes', return_value=b'python3\0'+wrong+b'\0'):
                    self.assertFalse(control.owned(state))
        with patch.object(control, 'ticks', return_value='1001'), patch.object(Path, 'read_bytes') as command:
            self.assertFalse(control.owned(state))
            command.assert_not_called()

    def test_invalid_pid_does_not_read_process(self):
        for pid in (None, '42', 42.0, True, 0, -42):
            with patch.object(control, 'ticks') as ticks:
                self.assertFalse(control.owned(dict(pid=pid,start_ticks='1000')))
                ticks.assert_not_called()

    def test_missing_manifest_and_corrupt_json_are_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'router.json'
            with patch.object(control, 'STATE', path):
                self.assertEqual(control.saved_state(), {})
                path.write_text('{not JSON')
                self.assertEqual(control.saved_state(), {})
                path.write_text(json.dumps(dict(pid=42,start_ticks='1')))
                self.assertEqual(control.saved_state(), dict(pid=42,start_ticks='1'))
                for value in (None, [], [1,2], 'string', 42, True):
                    path.write_text(json.dumps(value))
                    self.assertEqual(control.saved_state(), {})


    def test_missing_invalid_or_unreadable_ticks_fail_closed(self):
        for started in (None, '', 1000, True, 'not-digits'):
            with patch.object(control, 'ticks') as ticks, patch.object(Path, 'read_bytes') as command:
                self.assertFalse(control.owned(dict(pid=42,start_ticks=started)))
                ticks.assert_not_called()
                command.assert_not_called()
        with patch.object(control, 'ticks', return_value=None), patch.object(Path, 'read_bytes') as command:
            self.assertFalse(control.owned(dict(pid=42,start_ticks='1000')))
            command.assert_not_called()



class LifecycleTests(unittest.TestCase):
    def test_stop_never_signals_unowned_process(self):
        state = dict(pid=42,start_ticks='1000')
        with patch.object(control, 'saved_state', return_value=state), \
             patch.object(control, 'owned', return_value=False), \
             patch.object(control, 'status', return_value={'running':False}), \
             patch.object(control.os, 'kill') as kill:
            self.assertEqual(control.stop(), {'running':False})
            kill.assert_not_called()

    def test_start_does_not_replace_owned_router(self):
        with patch.object(control, 'saved_state', return_value={'pid':42}), \
             patch.object(control, 'owned', return_value=True), \
             patch.object(control, 'status', return_value={'running':True}), \
             patch.object(control.subprocess, 'Popen') as popen:
            self.assertEqual(control.start('flint'), {'running':True})
            popen.assert_not_called()

    def test_start_refuses_occupied_port_without_launching(self):
        probe = MagicMock()
        probe.__enter__.return_value.connect_ex.return_value = 0
        with patch.object(control, 'saved_state', return_value={}), \
             patch.object(control, 'owned', return_value=False), \
             patch.object(control.socket, 'socket', return_value=probe), \
             patch.object(control.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'another service'):
                control.start('flint')
            popen.assert_not_called()

    def test_status_does_not_contact_unowned_service(self):
        with patch.object(control, 'saved_state', return_value={'pid':42}), \
             patch.object(control, 'owned', return_value=False), \
             patch.object(control, 'build_opener') as opener, \
             contextlib.redirect_stdout(io.StringIO()):
            status = control.status()
            self.assertFalse(status['running'])
            self.assertIsNone(status['health'])
            opener.assert_not_called()


    def test_stop_uses_pidfd_and_rechecks_ownership(self):
        state = dict(pid=42,start_ticks='1000')
        with patch.object(control, 'saved_state', return_value=state), \
             patch.object(control, 'owned', side_effect=[True,True,False,False]), \
             patch.object(control, 'status', return_value={'running':False}), \
             patch.object(control.os, 'pidfd_open', return_value=99999) as opened, \
             patch.object(control.signal, 'pidfd_send_signal') as sent, \
             patch.object(control.os, 'close') as closed, \
             patch.object(control.os, 'kill') as kill:
            control.stop()
            opened.assert_called_once_with(42)
            sent.assert_called_once_with(99999, signal.SIGTERM)
            closed.assert_called_once_with(99999)
            kill.assert_not_called()

    def test_stop_does_not_signal_reused_pid_after_open(self):
        with patch.object(control, 'saved_state', return_value=dict(pid=42,start_ticks='1000')), \
             patch.object(control, 'owned', side_effect=[True,False]), \
             patch.object(control, 'status', return_value={'running':False}), \
             patch.object(control.os, 'pidfd_open', return_value=99999), \
             patch.object(control.signal, 'pidfd_send_signal') as sent, \
             patch.object(control.os, 'close') as closed, \
             patch.object(control.os, 'kill') as kill:
            control.stop()
            sent.assert_not_called()
            closed.assert_called_once_with(99999)
            kill.assert_not_called()

    def test_stop_tolerates_exit_during_open_or_signal(self):
        for stage in ('open','signal'):
            with patch.object(control, 'saved_state', return_value=dict(pid=42,start_ticks='1000')), \
                 patch.object(control, 'owned', return_value=True), \
                 patch.object(control, 'status', return_value={'running':False}), \
                 patch.object(control.os, 'pidfd_open', side_effect=ProcessLookupError if stage=='open' else None, return_value=99999), \
                 patch.object(control.signal, 'pidfd_send_signal', side_effect=ProcessLookupError if stage=='signal' else None) as sent, \
                 patch.object(control.os, 'close') as closed, \
                 patch.object(control.os, 'kill') as kill:
                self.assertEqual(control.stop(), {'running':False})
                kill.assert_not_called()
                if stage=='open':
                    sent.assert_not_called()
                    closed.assert_not_called()
                else:
                    closed.assert_called_once_with(99999)

    def test_stop_timeout_never_force_kills(self):
        with patch.object(control, 'saved_state', return_value=dict(pid=42,start_ticks='1000')), \
             patch.object(control, 'owned', return_value=True), \
             patch.object(control.os, 'pidfd_open', return_value=99999), \
             patch.object(control.signal, 'pidfd_send_signal') as sent, \
             patch.object(control.os, 'close') as closed, \
             patch.object(control.os, 'kill') as kill, \
             patch.object(control.time, 'monotonic', side_effect=[0,61]):
            with self.assertRaisesRegex(RuntimeError, 'cleanup'):
                control.stop()
            sent.assert_called_once_with(99999, signal.SIGTERM)
            closed.assert_called_once_with(99999)
            kill.assert_not_called()

    def test_mutating_cli_actions_hold_lock(self):
        for action in ('start','stop'):
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(control, 'STATE', Path(directory)/'manifests/router.json'), \
                     patch.object(control.sys, 'argv', ['control.py', action]), \
                     patch.object(control.fcntl, 'flock') as flock, \
                     patch.object(control, 'start') as start, \
                     patch.object(control, 'stop') as stop:
                    def check_lock(*args):
                        self.assertEqual(flock.call_count, 1)
                        self.assertEqual(flock.call_args.args[1], control.fcntl.LOCK_EX)
                    start.side_effect = check_lock
                    stop.side_effect = check_lock
                    control.main()
                    (start if action=='start' else stop).assert_called_once()
                    (stop if action=='start' else start).assert_not_called()



if __name__ == '__main__':
    unittest.main()
