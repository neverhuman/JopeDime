import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

import jope.research

SPEC = importlib.util.spec_from_file_location(
    "dlgp_safe_cleanup", Path(jope.research.__file__).parent / "dlgp_safe_cleanup.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cleanup_candidates_are_exact_and_quarantined_paths_are_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    sccache = tmp_path / "sccache"
    assert (
        MODULE.validate_candidate(workspace / "target/debug/incremental", workspace, sccache)
        == "rust_incremental"
    )
    with pytest.raises(ValueError, match="not an exact rebuildable path"):
        MODULE.validate_candidate(workspace / "target", workspace, sccache)
    with pytest.raises(ValueError):
        MODULE.validate_candidate(Path("/home/ubuntu/TO_DELETE"), workspace, sccache)


def test_cleanup_evidence_verification_detects_drift(tmp_path):
    bundle = tmp_path / "evidence"
    bundle.mkdir()
    payload = bundle / "result.json"
    payload.write_text("frozen\n")
    manifest = {
        "schema": "JopeDlgpRound2Evidence.v1",
        "payload_sha256": "a" * 64,
        "files": [
            {
                "path": "result.json",
                "bytes": payload.stat().st_size,
                "sha256": MODULE.sha256_file(payload),
            }
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    assert MODULE.verify_evidence_bundle(bundle) == "a" * 64
    payload.write_text("changed\n")
    with pytest.raises(ValueError, match="frozen input drift"):
        MODULE.verify_evidence_bundle(bundle)


def test_log_pruning_needs_manifest_membership_and_age(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    old = root / "old.log"
    unlisted = root / "unlisted.log"
    old.write_text("old")
    unlisted.write_text("keep")
    timestamp = time.time() - 40 * 86_400
    os.utime(old, (timestamp, timestamp))
    os.utime(unlisted, (timestamp, timestamp))
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "JopeDlgpLogCleanupPolicy.v1",
                "rebuildable_unreferenced_logs": ["old.log"],
            }
        )
    )
    removed = MODULE.prune_old_manifested_logs(root, policy, 30, True)
    assert removed == [str(old.resolve())]
    assert not old.exists()
    assert unlisted.exists()
