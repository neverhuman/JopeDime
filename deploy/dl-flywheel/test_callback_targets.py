"""Executable coverage-family adapter; adds no fixture identities.

The frozen runner owns all 37 + 7 + 11 + 8 isolated fixture executions.
This adapter must not be placed within any frozen package's closed roots.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
from pathlib import Path
import runpy

RUNNER = Path(__file__).resolve().with_name("run_callback_targets_offline.py")
RUNNER_SHA256 = "17c034261de03499533d35ffc308bb56b5adb82008b2a53ab73f52429cbb8252"

if __name__ == "__main__":
    if RUNNER.is_symlink() or hashlib.sha256(RUNNER.read_bytes()).hexdigest() != RUNNER_SHA256:
        raise ValueError("frozen callback fixture runner source pin")
    runpy.run_path(str(RUNNER), run_name="__main__")
