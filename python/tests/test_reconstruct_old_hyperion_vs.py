from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import jope.research

SCRIPT = Path(jope.research.__file__).parent / "reconstruct_old_hyperion_vs.py"
SPEC = importlib.util.spec_from_file_location("reconstruct_old_hyperion_vs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_target_column_never_enters_joint_features(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "train.csv").write_text("1,2,10\n3,4,20\n", encoding="utf-8")
    (root_a / "test.csv").write_text("5,6,30\n", encoding="utf-8")
    (root_b / "train.csv").write_text("1,2,-999\n3,4,999\n", encoding="utf-8")
    (root_b / "test.csv").write_text("5,6,12345\n", encoding="utf-8")

    joint_a, train_a, test_a, receipt_a = MODULE.load_joint_features(root_a)
    joint_b, train_b, test_b, receipt_b = MODULE.load_joint_features(root_b)

    assert np.array_equal(joint_a, joint_b)
    assert (train_a, test_a) == (train_b, test_b) == (2, 1)
    assert receipt_a["joint_feature_sha256"] == receipt_b["joint_feature_sha256"]
    assert receipt_a["train"]["target_policy"] == "excluded before any encoder call"


def test_all100_collapse_matches_two_stage_historical_mean() -> None:
    stack = np.arange(100 * 4 * 3 * 2, dtype=np.float32).reshape(100, 4, 3, 2)
    expected = np.mean(np.mean(stack, axis=1).transpose(1, 0, 2), axis=1)
    actual = MODULE.collapse_all100_mean(stack)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_matrix_publish_round_trips_and_refuses_overwrite(tmp_path: Path) -> None:
    matrix = np.array([[-0.12345679, 0.0], [np.float32(1.0 / 3.0), 12.5]], dtype=np.float32)
    path = tmp_path / "train3.vs.orig"
    receipt = MODULE.write_matrix_exclusive(path, matrix)
    replay = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    assert np.array_equal(replay, matrix)
    assert receipt["shape"] == [2, 2]
    assert receipt["finite"] is True
    with pytest.raises(FileExistsError):
        MODULE.write_matrix_exclusive(path, matrix)


def test_frozen_default_cohort_is_unique_and_complete() -> None:
    ids = MODULE.parse_dataset_ids([])
    assert len(ids) == 20
    assert len(set(ids)) == 20
