from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jope.research

import pytest


SCRIPTS = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "backfill_hyperion_stale_confirmation_cells",
    SCRIPTS / "backfill_hyperion_stale_confirmation_cells.py",
)
assert SPEC and SPEC.loader
backfill = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = backfill
SPEC.loader.exec_module(backfill)


def test_synthetic_hold_reverses_features_and_replaces_targets() -> None:
    source = b"1,2,99\n3,4,-7\n"
    assert backfill.synthetic_hold_bytes(source, Path("train.csv")) == (
        b"3,4,0\n1,2,0\n"
    )


def test_stale_projection_matches_replay_fallback_accounting() -> None:
    report = {
        "schema": backfill.replay.SCHEMA,
        "complete_dataset_count": 20,
        "raw_fallback_count": 2,
        "results": [
            {"dataset_id": "a", "state": "encoder_authorized"},
            {"dataset_id": "b", "state": "stale_or_foreign_cache"},
            {"dataset_id": "c", "state": "encoder_failure_or_invalid_cuda_proof"},
        ],
        "test_artifacts_read": False,
    }
    assert [row["dataset_id"] for row in backfill.stale_results(report)] == ["b", "c"]
    report["raw_fallback_count"] = 1
    with pytest.raises(ValueError, match="accounting"):
        backfill.stale_results(report)


def test_immutable_write_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "dataset" / "train3.vs"
    backfill.immutable_write(path, b"first")
    backfill.immutable_write(path, b"first")
    with pytest.raises(ValueError, match="differs"):
        backfill.immutable_write(path, b"second")
