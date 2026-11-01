"""Data-free tests for the A5 custody packager."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("capture_clients.py")
SPEC = importlib.util.spec_from_file_location("capture_clients", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_synthetic_package_contract():
    MODULE.self_test()


def test_roster_names_are_exact():
    assert set(MODULE.CLIENTS) == {"claude_code", "gemini_cli", "codex_cli"}


def test_digest_is_lowercase_sha256():
    assert MODULE.digest(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
