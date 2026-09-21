import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from watch_campaign_reports import run_report


class ReportProcessTests(unittest.TestCase):
    def test_heartbeat_continues_while_child_runs(self):
        ticks = []
        with tempfile.TemporaryFile() as log:
            run_report(
                [sys.executable, "-c", "import time; time.sleep(.12)"],
                log,
                ticks.append,
                timeout=5,
                interval=0.02,
            )
        self.assertTrue(ticks)
        self.assertEqual(ticks, sorted(ticks))

    def test_failure_is_propagated(self):
        with tempfile.TemporaryFile() as log:
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run_report([sys.executable, "-c", "raise SystemExit(7)"], log, lambda _: None, timeout=5)
        self.assertEqual(error.exception.returncode, 7)

    def test_timeout_reaps_owned_child(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as log:
            pid_path = Path(directory) / "pid"
            command = [
                sys.executable,
                "-c",
                'import os,time,sys; open(sys.argv[1],"w").write(str(os.getpid())); time.sleep(60)',
                str(pid_path),
            ]
            with self.assertRaises(subprocess.TimeoutExpired):
                run_report(command, log, lambda _: None, timeout=0.5, interval=0.02)
            self.assertTrue(pid_path.exists())
            self.assertFalse(Path("/proc", pid_path.read_text()).exists())


if __name__ == "__main__":
    unittest.main()
