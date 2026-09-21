import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import stopped_unit as unit
from promotion_gate import immutable


class Manager:
    """Model systemd's actual Stop ordering, including its unconditional CONT."""

    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.restart = "on-failure"
        self.commands = []

    def install(self, path, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)

    def property(self, service, name):
        if name == "Restart":
            return self.restart
        return str(self.parent.pid if self.parent.poll() is None else 0)

    def run(self, command, **kwargs):
        self.commands.append(command)
        if "daemon-reload" in command:
            files = list(self.root.glob("*.service.d/*"))
            self.restart = "no" if any("Restart=no" in p.read_text() for p in files) else "on-failure"
        elif "rm" in command:
            Path(command[-1]).unlink()
        elif "stop" in command and self.parent.poll() is None:
            os.kill(self.parent.pid, signal.SIGTERM)
            os.kill(self.parent.pid, signal.SIGCONT)
            self.parent.wait(timeout=2)

    def patch(self):
        stack = ExitStack()
        for name, value in [
            ("RUNTIME_ROOT", self.root),
            ("install", self.install),
            ("property", self.property),
        ]:
            stack.enter_context(patch.object(unit, name, value))
        stack.enter_context(patch.object(unit.subprocess, "run", self.run))
        return stack


class StopTests(unittest.TestCase):
    def test_interrupted_atomic_config_never_publishes_partial_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "override.conf"
            for before in (None, b"original\n"):
                if before is not None:
                    path.write_bytes(before)
                with patch.object(unit.os, "replace", side_effect=OSError("interrupted before rename")):
                    with self.assertRaises(OSError):
                        unit.atomic_config(path, b"complete new override\n")
                self.assertEqual(path.read_bytes() if path.exists() else None, before)
                unit.atomic_config(path, b"complete new override\n")
                self.assertEqual(path.read_bytes(), b"complete new override\n")
                path.unlink()

    def test_interrupted_owned_staging_can_retry_without_resuming_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, marker = self.parent(root)
            manager = Manager(root / "runtime", parent)
            real = unit.atomic_config

            def interrupted(path, data):
                if path.name == "restart-disabled.conf":
                    with patch.object(unit.os, "replace", side_effect=OSError("staging interrupted")):
                        real(path, data)
                else:
                    real(path, data)

            with manager.patch(), patch.object(unit, "atomic_config", interrupted):
                with self.assertRaises(OSError):
                    unit.stop("fixture.service", parent.pid, root / "journal", lambda: None)
            self.assertFalse((root / "journal/restart-disabled.conf").exists())
            self.assertEqual(unit.process(parent.pid)["state"], "T")
            with manager.patch():
                unit.stop("fixture.service", parent.pid, root / "journal", lambda: None)
            self.assertFalse(marker.exists())
            self.assertEqual(manager.restart, "on-failure")

    def parent(self, root):
        marker = root / "dispatched"
        code = "import signal,sys,time;signal.signal(signal.SIGTERM,lambda *_:open(sys.argv[1],'w').write('resumed'));print('ready',flush=True);time.sleep(100)"
        parent = subprocess.Popen(
            [sys.executable, "-c", code, str(marker)], stdout=subprocess.PIPE, text=True
        )
        parent.stdout.readline()
        os.kill(parent.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 2
        while unit.process(parent.pid)["state"] != "T":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.addCleanup(parent.stdout.close)

        def cleanup():
            if parent.poll() is None:
                parent.kill()
                parent.wait()

        self.addCleanup(cleanup)
        return parent, marker

    def test_direct_pidfd_kill_precedes_systemd_stop_and_restores_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, marker = self.parent(root)
            manager = Manager(root / "runtime", parent)
            with manager.patch():
                unit.stop("fixture.service", parent.pid, root / "journal", lambda: None)
            self.assertEqual(parent.returncode, -signal.SIGKILL)
            self.assertFalse(marker.exists())
            self.assertEqual(manager.restart, "on-failure")
            self.assertTrue((root / "journal/complete.json").exists())

    def test_refused_restart_policy_keeps_dispatch_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, marker = self.parent(root)
            manager = Manager(root / "runtime", parent)
            with (
                manager.patch(),
                patch.object(
                    unit,
                    "property",
                    side_effect=lambda _, n: "on-failure" if n == "Restart" else str(parent.pid),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "restart was not disabled"):
                    unit.stop("fixture.service", parent.pid, root / "journal", lambda: None)
            self.assertEqual(unit.process(parent.pid)["state"], "T")
            self.assertFalse(marker.exists())

    def test_retry_reloads_after_override_bytes_were_already_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, marker = self.parent(root)
            manager = Manager(root / "runtime", parent)
            original = manager.run
            failed = False

            def fail(command, **kwargs):
                nonlocal failed
                if "daemon-reload" in command and not list(manager.root.glob("*.service.d/*")) and not failed:
                    failed = True
                    raise OSError("interrupted before restoration reload")
                original(command, **kwargs)

            with manager.patch(), patch.object(unit.subprocess, "run", fail):
                with self.assertRaises(OSError):
                    unit.stop("fixture.service", parent.pid, root / "journal", lambda: None)
            self.assertEqual(manager.restart, "no")
            with manager.patch():
                unit.recover(root / "journal")
            self.assertEqual(manager.restart, "on-failure")
            self.assertFalse(marker.exists())

    def test_retry_refuses_a_competing_override_before_install(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, _ = self.parent(root)
            manager = Manager(root / "runtime", parent)
            with manager.patch():
                with self.assertRaisesRegex(RuntimeError, "boundary"):
                    unit.stop(
                        "fixture.service",
                        parent.pid,
                        root / "journal",
                        lambda: (_ for _ in ()).throw(RuntimeError("boundary")),
                    )
                override = next(manager.root.glob("*.service.d/*"))
                override.write_text("competitor\n")
                with self.assertRaisesRegex(RuntimeError, "another operation"):
                    unit.stop("fixture.service", parent.pid, root / "journal", lambda: None)
                self.assertEqual(override.read_text(), "competitor\n")

    def test_reused_pid_never_stops_a_replacement_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            journal = root / "journal"
            journal.mkdir()
            override = root / "absent.conf"
            request = dict(
                unit="fixture.service",
                pid=123,
                birth="old",
                override=str(override),
                restart_before="on-failure",
                driver_sha256=unit.sha(Path(unit.__file__)),
            )
            immutable(journal / "request.json", request)
            immutable(journal / "override-before.json", dict(content=None))
            immutable(journal / "kill-intent.json", dict(request_sha256=unit.sha(journal / "request.json")))
            (journal / "restart-disabled.conf").write_text("owned\n")
            with (
                patch.object(unit, "process", return_value=dict(state="S", birth="new")),
                patch.object(
                    unit, "property", side_effect=lambda _, name: "123" if name == "MainPID" else "on-failure"
                ),
                patch.object(unit.subprocess, "run") as command,
            ):
                unit.recover(journal)
            self.assertFalse(any("stop" in call.args[0] for call in command.call_args_list))
