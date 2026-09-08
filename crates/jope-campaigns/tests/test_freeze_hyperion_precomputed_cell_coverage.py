from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "freeze_hyperion_precomputed_cell_coverage.py"
)
SPEC = importlib.util.spec_from_file_location("freeze_hyperion_cell", SCRIPT)
assert SPEC and SPEC.loader
coverage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coverage
SPEC.loader.exec_module(coverage)


def fixture_dataset(root: Path) -> Path:
    dataset = root / "abcd1234"
    dataset.mkdir()
    (dataset / "train.csv").write_text("1,2,0\n1,2,1\n3,4,2\n", encoding="utf-8")
    (dataset / "train3.vs").write_text("0.1,0.2\n0.3,0.4\n0.5,0.6\n", encoding="utf-8")
    (dataset / "test.csv").write_text("forbidden\n", encoding="utf-8")
    return dataset


def test_inspection_is_exact_train_only_and_groups_duplicate_features(
    tmp_path: Path,
) -> None:
    fixture_dataset(tmp_path)
    actual = coverage.inspect_dataset(tmp_path, "abcd1234", "development")
    assert actual["rows"] == 3
    assert actual["raw_features"] == 2
    assert actual["cell_rows"] == 3
    assert actual["cell_columns"] == 2
    assert actual["atomic_group_count"] == 2
    assert "target_statistics_emitted" not in actual
    assert actual["test_artifacts_read"] is False
    assert "target" not in actual
    assert "test" not in actual["train_relative_path"]
    with pytest.raises(ValueError, match="forbidden"):
        coverage.checked_path(tmp_path / "abcd1234", "test.csv")


def test_cell_shape_range_and_mutation_are_rejected(tmp_path: Path) -> None:
    dataset = fixture_dataset(tmp_path)
    (dataset / "train3.vs").write_text("0.1,0.2\n0.3,0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rows"):
        coverage.inspect_dataset(tmp_path, "abcd1234", "confirmation")
    (dataset / "train3.vs").write_text("0.1,0.2\n0.3,1.4\n0.5,0.6\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        coverage.inspect_dataset(tmp_path, "abcd1234", "confirmation")


def test_immutable_freeze_and_verify_bytes(tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"
    report = {"schema": coverage.SCHEMA, "datasets": 42}
    coverage.freeze(output, report)
    first = output.read_bytes()
    coverage.freeze(output, report)
    assert output.read_bytes() == first
    with pytest.raises(ValueError, match="other bytes"):
        coverage.freeze(output, {**report, "datasets": 41})
    assert json.loads(output.read_text()) == report
