"""Synthetic CPU proof of the read-only staging fault and bounded operator recovery."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "h82_operator_recovery", Path(__file__).with_name("recover.py")
)
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def fixture_prefix(root):
    rows = [
        {"arm": "K8", "seed": 24, "attempt": 1, "timing": True},
        {"arm": "CONTROL", "seed": 24, "attempt": 1, "timing": False},
        {"arm": "K8", "seed": 24, "attempt": 1, "timing": False},
        {
            "arm": "SAM",
            "seed": 24,
            "attempt": 1,
            "timing": False,
            "error_kind": "host_load",
            "outcome_started": False,
        },
        {
            "arm": "SAM",
            "seed": 24,
            "attempt": 2,
            "timing": False,
            "error_kind": "fatal",
            "outcome_started": False,
            "stage": "loading",
        },
    ]
    ledger = {"attempts": rows, "cap_seconds": 2700, "in_progress": None}
    (root / "budget.json").write_text(json.dumps(ledger))
    (root / "SAM-seed24-attempt2.log").write_text(
        "PermissionError: synthetic immutable input collision\n"
    )
    return ledger, r.sha(root / "budget.json"), r.sha(root / "SAM-seed24-attempt2.log")


def test_reproduces_readonly_copy_failure_and_preserved_cleanup_success(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"synthetic checkpoint bytes")
    source.chmod(0o444)
    root = tmp_path / "run"
    staged = root / "scratch/r882_adapter_copy.pt"
    staged.parent.mkdir(parents=True)
    shutil.copy2(source, staged)
    with pytest.raises(PermissionError):
        shutil.copy2(source, staged)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert r.clear_staging(root, tmp_path / "custody", expected_sha=digest)
    assert not staged.exists()
    assert (
        tmp_path / "custody/preserved-stale-r882-copy.pt"
    ).read_bytes() == source.read_bytes()
    shutil.copy2(source, staged)
    assert staged.read_bytes() == source.read_bytes()
    assert r.clear_staging(root, tmp_path / "custody", expected_sha=digest)
    assert not r.clear_staging(root, tmp_path / "custody", expected_sha=digest)


def test_unknown_staged_bytes_or_symlinks_are_retained(tmp_path):
    root = tmp_path / "run"
    staged = root / "scratch/r882_adapter_copy.pt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"unrecognized")
    with pytest.raises(RuntimeError):
        r.clear_staging(root, tmp_path / "custody")
    assert staged.read_bytes() == b"unrecognized"
    staged.unlink()
    outside = tmp_path / "external"
    outside.write_bytes(b"input")
    staged.symlink_to(outside)
    with pytest.raises(RuntimeError):
        r.clear_staging(root, tmp_path / "custody", expected_sha=r.sha(outside))
    assert staged.is_symlink() and outside.exists()


def test_exact_prefix_is_retained_and_remaining_sam_attempt_is_three(tmp_path):
    original, budget_sha, log_sha = fixture_prefix(tmp_path)
    loaded = r.validate_prefix(
        tmp_path, expected_sha=budget_sha, expected_log_sha=log_sha
    )
    assert loaded == original
    assert loaded["attempts"][-1]["attempt"] == 2
    assert len(loaded["attempts"]) == 5 and loaded["cap_seconds"] == 2700
    assert json.loads((tmp_path / "budget.json").read_text()) == original


@pytest.mark.parametrize(
    "mutation",
    [
        "prefix",
        "uncertain",
        "cap",
        "scored",
        "wrong_failure",
        "unknown_outcome",
        "unknown_phase",
        "unknown_adapter",
        "log",
    ],
)
def test_recovery_refuses_other_states(tmp_path, mutation):
    ledger, budget_sha, log_sha = fixture_prefix(tmp_path)
    if mutation == "prefix":
        ledger["extra"] = "changed"
    elif mutation == "uncertain":
        ledger["in_progress"] = {"attempt": 3}
    elif mutation == "cap":
        ledger["cap_seconds"] = 5400
    elif mutation == "scored":
        ledger["attempts"][-1]["outcome_started"] = True
    elif mutation == "wrong_failure":
        ledger["attempts"][-2]["error_kind"] = "integrity"
    elif mutation == "unknown_outcome":
        (tmp_path / "ASAM-seed24-attempt1.json").write_text("{}")
    elif mutation == "unknown_phase":
        (tmp_path / "ASAM-seed24-attempt1.phase.json").write_text("{}")
    elif mutation == "unknown_adapter":
        (tmp_path / "ASAM-seed24-attempt1.adapter.pt").write_bytes(b"unknown")
    else:
        (tmp_path / "SAM-seed24-attempt2.log").write_text("different error")
    (tmp_path / "budget.json").write_text(json.dumps(ledger))
    if mutation not in ("prefix", "log"):
        budget_sha = r.sha(
            tmp_path / "budget.json"
        )  # exercise semantic refusal beyond the immutable seal
    with pytest.raises(RuntimeError):
        r.validate_prefix(tmp_path, expected_sha=budget_sha, expected_log_sha=log_sha)


def test_dangling_staging_symlink_is_rejected_before_worker_can_follow_it(tmp_path):
    root = tmp_path / "run"
    staged = root / "scratch/r882_adapter_copy.pt"
    staged.parent.mkdir(parents=True)
    staged.symlink_to(tmp_path / "missing-external-input")
    with pytest.raises(RuntimeError):
        r.clear_staging(root, tmp_path / "custody")
    assert staged.is_symlink()


def test_symlinked_staging_directory_cannot_remove_checkpoint_file(tmp_path):
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "r882_adapter_copy.pt"
    checkpoint.write_bytes(b"synthetic pinned adapter")
    (root / "scratch").symlink_to(checkpoints, target_is_directory=True)
    with pytest.raises(RuntimeError, match="staging directory"):
        r.clear_staging(
            root,
            tmp_path / "custody",
            expected_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
    assert checkpoint.read_bytes() == b"synthetic pinned adapter"
