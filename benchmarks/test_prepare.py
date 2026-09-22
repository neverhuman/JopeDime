"""Real split and byte-identity tests. Synthetic fixtures; no network or fitting."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location("asm_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_groups_and_duplicate_features_cannot_cross_partitions():
    rows = [[str(i // 2), str((i // 2) % 3), str(i)] for i in range(300)]
    # Same features carry different labels; label differences must not split duplicates.
    groups = prepare.components(rows, [0], [])
    assert all(groups[i] == groups[i + 1] for i in range(0, 300, 2))
    labels = [row[1] for row in rows]
    folds = prepare.assignments(labels, groups, "multiclass", 123)
    assert folds == prepare.assignments(labels, groups, "multiclass", 123)
    assert set(folds) == {None, 0, 1, 2, 3, 4}
    for fold in set(folds):
        assert {label for label, f in zip(labels, folds) if f == fold} == {"0", "1", "2"}
    assert all(folds[i] == folds[i + 1] for i in range(0, 300, 2))


def test_declared_groups_and_duplicates_form_transitive_components():
    rows = [["a", "x"], ["a", "y"], ["b", "y"], ["c", "z"]]
    groups = prepare.components(rows, [1], [0])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_preparation_preserves_labels_missing_values_and_sealed_identity(tmp_path, monkeypatch):
    raw_rows = [
        [str(i), "?" if i % 7 == 0 else str(i % 4), "positive" if i % 2 else "negative"] for i in range(300)
    ]
    raw = "number\tcategory\ttarget\n" + "\n".join("\t".join(r) for r in raw_rows) + "\n"
    compressed = gzip.compress(raw.encode(), mtime=0)
    metadata = yaml.safe_dump(
        {
            "task": "classification",
            "features": [{"name": "number", "type": "continuous"}, {"name": "category", "type": "ordinal"}],
        }
    ).encode()
    monkeypatch.setattr(prepare, "fetch", lambda url: metadata if url.endswith(".yaml") else compressed)
    config = {"repository": "fixture", "revision": "a" * 40, "seed": 123}
    selection = {"id": "fixture", "task": "binary", "group_columns": []}
    frozen = prepare.prepare_one(selection, config, tmp_path)
    assert prepare.prepare_one(frozen, config, tmp_path, frozen=True) == frozen
    dev = json.loads((tmp_path / "development/fixture.json").read_text())
    sealed = json.loads((tmp_path / "sealed/fixture.json").read_text())
    assert {r["id"] for r in dev["rows"]}.isdisjoint(r["id"] for r in sealed["rows"])
    assert {r["group"] for r in dev["rows"]}.isdisjoint(r["group"] for r in sealed["rows"])
    assert any(r["values"][1] is None for r in dev["rows"])
    assert dev["features"][1]["kind"] == "categorical"
    for row in dev["rows"] + sealed["rows"]:
        assert row["target"] == raw_rows[row["source_index"]][2]
    (tmp_path / "development/fixture.json").write_text("tampered")
    with pytest.raises(ValueError, match="changed immutable input"):
        prepare.prepare_one(frozen, config, tmp_path, frozen=True)


def test_wrong_split_version_refuses_repreparation(monkeypatch):
    monkeypatch.setattr(prepare.sklearn, "__version__", "different")
    with pytest.raises(ValueError, match="split identity"):
        prepare.assignments(["0", "1"], [0, 1], "binary", 1)


def test_missing_class_is_not_reported_as_valid_stratification():
    with pytest.raises(ValueError, match="lacks a class"):
        prepare.assignments(["rare"] + ["common"] * 99, list(range(100)), "binary", 1)


def test_frozen_panel_selection_contains_no_result_based_membership():
    root = Path(__file__).parent
    selection = json.loads((root / "panel-selection.json").read_text())
    frozen = json.loads((root / "panel.json").read_text())
    assert [(d["id"], d["task"]) for d in frozen["datasets"]] == [
        (d["id"], d["task"]) for d in selection["datasets"]
    ]
    assert all(
        sum(d["task"] == task for d in frozen["datasets"]) == 10
        for task in ("regression", "binary", "multiclass")
    )
    assert "fit" not in frozen and "scores" not in frozen
