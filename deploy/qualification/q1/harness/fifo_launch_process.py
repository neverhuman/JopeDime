"""FIFO launch unittest mixins, imported by ``test_fifo_launch``."""
from pathlib import Path
import fcntl
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fifo_launch as launch
import q1
import q1_cut


class FIFOProcessTests:
    def test_owned_process_deadline_and_inherited_lock_descriptor(self):
        with self.lock.open("rb") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            fd = lease.fileno()
            marker = self.root / "descriptor-was-inherited"
            code = f"import os;from pathlib import Path;os.fstat({fd});Path({str(marker)!r}).touch()"
            self.assertEqual(
                launch.run_owned(
                    [sys.executable, "-c", code],
                    self.root / "fd.log",
                    dict(os.environ),
                    1,
                    fd,
                    lambda: 0,
                    0.02,
                ),
                0,
            )
            self.assertTrue(marker.exists())
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "wall cap"):
                launch.run_owned(
                    [sys.executable, "-c", "import time;time.sleep(2)"],
                    self.root / "timeout.log",
                    dict(os.environ),
                    0.05,
                    fd,
                    lambda: 0,
                    0.02,
                )
            self.assertLess(time.monotonic() - started, 1)

    def test_leader_exit_still_kills_its_term_ignoring_descendant(self):
        pidfile = self.root / "descendant.pid"
        script = self.root / "leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(2)"
        script.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\n"
        )
        with self.lock.open("rb") as handle:
            self.assertEqual(
                launch.run_owned(
                    [sys.executable, str(script)],
                    self.root / "descendant.log",
                    dict(os.environ),
                    1,
                    handle.fileno(),
                    lambda: 0,
                    0.02,
                ),
                0,
            )
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 0.5
        while self.pid_is_active(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.pid_is_active(pid))

    @staticmethod
    def pid_is_active(pid):
        identity = q1._proc_identity(pid)
        return identity is not None and identity["state"] not in ("Z", "X", "x")

    def test_wall_timeout_cleans_an_isolated_session_descendant(self):
        pidfile = self.root / "timeout-descendant.pid"
        script = self.root / "timeout-leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(30)"
        script.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\ntime.sleep(30)\n"
        )
        with self.lock.open("rb") as handle:
            with self.assertRaisesRegex(TimeoutError, "wall cap"):
                launch.run_owned(
                    [sys.executable, str(script)],
                    self.root / "timeout-descendant.log",
                    dict(os.environ),
                    0.2,
                    handle.fileno(),
                    lambda: 0,
                    0.02,
                )
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 0.5
        while self.pid_is_active(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.pid_is_active(pid))

    def test_session_cleanup_retries_identity_error_before_returning(self):
        pidfile = self.root / "retry-descendant.pid"
        script = self.root / "retry-leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(30)"
        script.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\n"
        )
        original = q1._signal_scope
        attempts = 0

        def fail_once(session_id, process_group, signum):
            nonlocal attempts
            if attempts == 0:
                attempts += 1
                raise q1.BackendCleanupError("synthetic first cleanup failure")
            return original(session_id, process_group, signum)

        with (
            self.lock.open("rb") as handle,
            patch.object(q1, "_signal_scope", side_effect=fail_once),
        ):
            self.assertEqual(
                launch.run_owned(
                    [sys.executable, str(script)],
                    self.root / "retry-descendant.log",
                    dict(os.environ),
                    1,
                    handle.fileno(),
                    lambda: 0,
                    0.02,
                ),
                0,
            )
        self.assertGreaterEqual(attempts, 1)
        pid = int(pidfile.read_text())
        self.assertFalse(self.pid_is_active(pid))

    def test_sigterm_interrupt_cleans_a_term_ignoring_isolated_descendant(self):
        pidfile = self.root / "interrupt-descendant.pid"
        leader = self.root / "interrupt-leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(30)"
        leader.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\ntime.sleep(30)\n"
        )
        runner = self.root / "interrupt-runner.py"
        runner.write_text(
            "import os,sys\n"
            f"sys.path.insert(0,{str(Path(launch.__file__).parent)!r})\n"
            "import fifo_launch\n"
            "from pathlib import Path\n"
            f"with open({str(self.lock)!r},'rb') as handle:\n"
            " with fifo_launch.termination_handler():\n"
            f"  fifo_launch.run_owned([sys.executable,{str(leader)!r}],Path({str(self.root / 'interrupt.log')!r}),dict(os.environ),30,handle.fileno(),lambda:0,0.02)\n"
        )
        runner_stderr = (self.root / "interrupt-runner.stderr").open("wb")
        runner_process = subprocess.Popen(
            [sys.executable, str(runner)],
            stdout=subprocess.DEVNULL,
            stderr=runner_stderr,
        )
        try:
            deadline = time.monotonic() + 2
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(
                pidfile.exists(),
                "fixture driver did not launch: "
                + (self.root / "interrupt-runner.stderr").read_text(),
            )
            os.kill(runner_process.pid, signal.SIGTERM)
            self.assertNotEqual(runner_process.wait(timeout=2), 0)
        finally:
            if runner_process.poll() is None:
                runner_process.kill()
                runner_process.wait(timeout=2)
            runner_stderr.close()
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 0.5
        while self.pid_is_active(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.pid_is_active(pid))

    def test_high_load_and_termination_refuse_or_cleanup_without_other_jobs(self):
        with self.assertRaisesRegex(ValueError, "host load"):
            launch.admit(
                self.args, self.paths, source_check=lambda *_: None, load=lambda: 86
            )
        with self.lock.open("rb") as handle:
            with self.assertRaisesRegex(ValueError, "host load"):
                launch.run_owned(
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    self.root / "must-not-start.log",
                    dict(os.environ),
                    1,
                    handle.fileno(),
                    lambda: 86,
                    0.02,
                )
        self.assertFalse((self.root / "must-not-start.log").exists())
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(InterruptedError):
            with launch.termination_handler():
                signal.raise_signal(signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_prepare_output_and_run_directory_never_overwrite(self):
        out = self.root / "intent.json"
        launch.write_new(out, {"fixture": True})
        with self.assertRaises(FileExistsError):
            launch.write_new(out, {"overwrite": True})
        self.assertEqual(q1.load(out), {"fixture": True})
        self.args.run_dir.mkdir()
        sentinel = self.args.run_dir / "prior-evidence"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "new canonical run"):
            self.run_launch()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(self.queue.iterdir()), [])
