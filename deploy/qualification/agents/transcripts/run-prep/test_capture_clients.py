"""Data-free tests for the A5 custody packager."""

import importlib.util
from pathlib import Path

import pytest


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
    assert (
        MODULE.digest(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_copy_drift_cannot_publish_a_manifest(monkeypatch):
    original = MODULE.shutil.copyfile

    def tamper(source, destination, **kwargs):
        result = original(source, destination, **kwargs)
        Path(destination).write_bytes(b"changed after sidecar validation")
        return result

    monkeypatch.setattr(MODULE.shutil, "copyfile", tamper)
    with pytest.raises(MODULE.CapturePackageError, match="sidecar.*changed"):
        MODULE.self_test()


def test_candidate_drift_during_copy_cannot_publish(monkeypatch):
    original = MODULE.shutil.copyfile

    def tamper(source, destination, **kwargs):
        result = original(source, destination, **kwargs)
        (Path(source).parent.parent / "candidate.bin").write_bytes(b"changed candidate")
        return result

    monkeypatch.setattr(MODULE.shutil, "copyfile", tamper)
    with pytest.raises(MODULE.CapturePackageError, match="candidate"):
        MODULE.self_test()
