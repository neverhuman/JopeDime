from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "hyperion_train_audit.py"
SPEC = importlib.util.spec_from_file_location("hyperion_train_audit", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_data_boundary_rejects_every_non_train_sidecar(tmp_path: Path) -> None:
    dataset = tmp_path / "opaque"
    dataset.mkdir()
    for name in ("test.csv", "test3.vs", "hyperion3test.vs", "train_target.vs"):
        (dataset / name).write_text("1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="forbidden"):
            audit.checked_data_path(dataset, name)


def test_grouped_folds_keep_duplicates_atomic_and_cover_rows_once() -> None:
    groups = ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
    folds = audit.grouped_folds(groups, 5, audit.CV5_SEED)
    assert folds == audit.grouped_folds(groups, 5, audit.CV5_SEED)
    assert sorted(row for fold in folds for row in fold.hold_rows) == list(range(len(groups)))
    location = {row: fold.index for fold in folds for row in fold.hold_rows}
    assert location[0] == location[1]
    assert all(set(fold.fit_rows).isdisjoint(fold.hold_rows) for fold in folds)


def test_preprocessing_is_fit_only() -> None:
    cell = np.asarray([[0.0], [1.0], [2.0], [100.0]])
    row = np.tile(np.arange(128, dtype=float), (4, 1))
    row[1] += 1.0
    row[2] += 2.0
    row[3] += 1000.0
    transformed_cell, transformed_row = audit.transform_blocks(cell, row, (0, 1, 2))
    assert transformed_cell[:, 0].tolist() == pytest.approx([1 / 6, 0.5, 5 / 6, 1.0])
    assert transformed_row[3, 0] > 100.0
    mutated = cell.copy()
    mutated[3, 0] = -100.0
    transformed_mutated, _ = audit.transform_blocks(mutated, row, (0, 1, 2))
    assert transformed_mutated[:3] == pytest.approx(transformed_cell[:3])


def _fold(index: int, raw_error: float, arm_error: float) -> object:
    return audit.FoldEvidence(
        index=index,
        row_groups=[f"g{index}-{row}" for row in range(2)],
        actual=[0.0, 0.0],
        raw=[raw_error, raw_error],
        arms={"hyperion_cell": [arm_error, arm_error]},
    )


def test_gate_accepts_stable_lift_and_rejects_one_bad_fold() -> None:
    thresholds = audit.Thresholds()
    stable = [_fold(index, 1.0, 0.9) for index in range(5)]
    accepted = audit.evaluate_arm("cv5", stable, "hyperion_cell", thresholds)
    assert accepted["passed"] is True
    harmful = [_fold(index, 1.0, 0.9) for index in range(4)] + [_fold(4, 1.0, 1.03)]
    rejected = audit.evaluate_arm("cv5", harmful, "hyperion_cell", thresholds)
    assert rejected["passed"] is False
    assert rejected["worst_fold_regression"] > thresholds.maximum_fold_regression


def test_staged_csv_replaces_only_hold_targets(tmp_path: Path) -> None:
    path = tmp_path / "staged.csv"
    audit.stage_csv(path, np.asarray([[1.0], [2.0], [3.0]]), np.asarray([4.0, 5.0, 6.0]), (1,))
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]
    assert [float(row[-1]) for row in rows] == [4.0, 0.0, 6.0]
