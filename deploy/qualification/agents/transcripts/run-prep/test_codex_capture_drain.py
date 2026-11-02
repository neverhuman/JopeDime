"""Regression tests for Codex MCP recorder shutdown after normal client exit."""

import importlib.util
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


RUNNER = Path(__file__).with_name("capture_three_clients.py")
SPEC = importlib.util.spec_from_file_location("capture_three_clients_drain", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normal_codex_exit_allows_recorder_flush_before_bounded_cleanup():
    read_fd, write_fd = os.pipe()
    recorder = "import os,sys,time; time.sleep(3); os.write(int(sys.argv[1]), b'done')"
    client = (
        "import subprocess, sys; "
        "fd=int(sys.argv[1]); "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], str(fd)], "
        "pass_fds=(fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", client, str(write_fd), recorder],
        start_new_session=True,
        pass_fds=(write_fd,),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.close(write_fd)
    drained = False
    try:
        assert process.wait(timeout=5) == 0
        MODULE.drain_group(process, time.monotonic() + 20)
        ready, _writable, _errors = select.select([read_fd], [], [], 0)
        assert ready, "recorder descendant was killed before its flush marker"
        assert os.read(read_fd, 4) == b"done"
        drained = True
    finally:
        if not drained:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os.close(read_fd)
