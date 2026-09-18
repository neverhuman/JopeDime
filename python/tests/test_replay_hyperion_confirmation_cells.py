from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import jope.research

SCRIPTS = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "replay_hyperion_confirmation_cells",
    SCRIPTS / "replay_hyperion_confirmation_cells.py",
)
assert SPEC and SPEC.loader
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def test_cuda_proof_requires_exactly_one_isolated_device_report() -> None:
    assert replay.device_proof("device=cuda:0\n", "")
    assert not replay.device_proof("device=cuda:0\ndevice=cuda:0\n", "")
    assert not replay.device_proof("device=cuda:1\n", "")
    assert not replay.device_proof("device=cuda:0\n", "fallback device=cpu")


def test_confirmation_projection_is_exact_ordered_and_train_only() -> None:
    datasets = [
        {
            "dataset_id": f"{index:016x}",
            "cohort": "confirmation",
            "test_artifacts_read": False,
        }
        for index in range(20)
    ]
    coverage = {
        "schema": replay.physical.SCHEMA,
        "dataset_count": 42,
        "datasets": [
            *datasets,
            *[
                {
                    "dataset_id": f"d{index:015x}",
                    "cohort": "development",
                    "test_artifacts_read": False,
                }
                for index in range(22)
            ],
        ],
        "test_artifacts_read": False,
    }
    assert replay.confirmation_entries(coverage) == datasets
    coverage["datasets"][1], coverage["datasets"][2] = (
        coverage["datasets"][2],
        coverage["datasets"][1],
    )
    with pytest.raises(ValueError, match="ordered"):
        replay.confirmation_entries(coverage)


def test_replay_freeze_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "replay.json"
    report = {"schema": replay.SCHEMA, "complete_dataset_count": 20}
    replay.freeze(output, report)
    first = output.read_bytes()
    replay.freeze(output, report)
    assert output.read_bytes() == first
    with pytest.raises(ValueError, match="other bytes"):
        replay.freeze(output, {**report, "complete_dataset_count": 19})
