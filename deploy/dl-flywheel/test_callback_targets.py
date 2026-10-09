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
RUNNER_SHA256 = "c31178eed46d2255f69ba3558350b5ccb9ae4936e3f45d85a3b1483c9fb0c4f8"

if __name__ == "__main__":
    if RUNNER.is_symlink() or hashlib.sha256(RUNNER.read_bytes()).hexdigest() != RUNNER_SHA256:
        raise ValueError("frozen callback fixture runner source pin")
    runpy.run_path(str(RUNNER), run_name="__main__")
