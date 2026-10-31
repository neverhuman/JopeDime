"""Synthetic CPU controls for the H124 fixed-lambda shrink protocol."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h124_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def _rows():
    return [
        {
            "context_sha256": f"{index:064x}",
            "family_id": f"family-{index:03d}",
            "dl_split": "DL-Train",
            "source_split": "train",
            "task": "regression",
        }
        for index in range(205)
    ]


def _packets(blend_gain=0.002, pure_gain=0.0, harmed=0):
    rows = _rows()
    packets = {}
    for seed in r.SEEDS:
        packets[str(seed)] = {}
        for index, row in enumerate(rows):
            blend = 0.5 + (-0.02 if index < harmed else blend_gain)
            packets[str(seed)][row["context_sha256"]] = {
                "r882": {"u4": 0.5},
                "blend": {"u4": blend},
                "pure_gbm": {"u4": 0.5 + pure_gain},
            }
    return rows, packets


def test_fold_is_fresh_deterministic_family_grouped_and_disjoint():
    rows = _rows()
    groups, mapping = r.folds(rows)
    reversed_groups, reversed_mapping = r.folds(list(reversed(rows)))
    assert groups == reversed_groups
    assert mapping == reversed_mapping
    assert [len(group) for group in groups] == [41] * 5
    assert len(set(mapping.values())) == 5
    old = [
        sorted(group.tolist())
        for group in np.array_split(
            np.random.default_rng(11900).permutation(
                sorted({row["family_id"] for row in rows})
            ),
            5,
        )
    ]
    assert groups != old
    duplicate = rows + [rows[0]]
    with pytest.raises(ValueError):
        r.folds(duplicate)
    rows[0]["dl_split"] = "DL-Val"
    with pytest.raises(ValueError):
        r.folds(rows)


def test_score_blend_is_fixed_and_matches_registered_utility_formula():
    r882_cost = np.asarray([3.0, 0.5, -1.0])
    gbm_cost = np.asarray([1.0, 1.5, -3.0])
    blended = r.blend_cost(r882_cost, gbm_cost)
    expected_utility = -r882_cost + 0.5 * ((-gbm_cost) - (-r882_cost))
    np.testing.assert_array_equal(blended, -expected_utility.astype(np.float32))
    np.testing.assert_array_equal(r.blend_cost(r882_cost, gbm_cost, 1.0), gbm_cost)
    with pytest.raises(ValueError, match="fixed"):
        r.blend_cost(r882_cost, gbm_cost, 0.75)
    with pytest.raises(ValueError, match="finite"):
        r.blend_cost([0.0], [float("nan")])


def test_family_values_refuse_incomplete_or_duplicate_panels():
    rows, packets = _packets()
    families, values = r.family_values(rows, packets, "blend")
    assert len(families) == 205
    assert values.shape == (3, 205)
    packets["64"].pop(rows[0]["context_sha256"])
    with pytest.raises(ValueError, match="every DL-Train context"):
        r.family_values(rows, packets, "blend")
    with pytest.raises(ValueError, match="unique"):
        r.family_values(rows + [rows[0]], packets, "blend")


def test_fixed_lambda_primary_gate_and_secondary_is_descriptive(monkeypatch):
    monkeypatch.setattr(r, "elo_h124", lambda *_args, **_kwargs: {"ratings": {}})
    rows, packets = _packets(blend_gain=0.002, pure_gain=0.0)
    result = r.h124_summary(rows, packets)
    assert result["primary_vs_r882"]["admitted"]
    assert result["primary_vs_r882"]["lcb95"] == pytest.approx(0.002)
    assert result["secondary_pure_gbm_vs_r882"]["admission_assessed"] is False
    assert result["secondary_pure_gbm_vs_r882"]["admitted"] is None

    rows, tied = _packets(blend_gain=0.001)
    assert not r.h124_summary(rows, tied)["primary_vs_r882"]["admitted"]

    rows, harmed = _packets(blend_gain=0.005, harmed=21)
    rejected = r.h124_summary(rows, harmed)["primary_vs_r882"]
    assert rejected["median_seed_gain"] > 0
    assert rejected["harm_count"] == 21
    assert not rejected["admitted"]


def test_unregistered_fit_refuses_before_input_access(tmp_path):
    for task in ((63, (0,)), (67, (0,)), (64, ()), (64, (1, 0))):
        with pytest.raises(ValueError):
            r.fit_one(task, tmp_path, [], {}, {}, {}, {})


def test_cpu_dry_run_uses_only_synthetic_metadata_and_no_fits():
    receipt = r.cpu_dry_run()
    assert receipt["status"] == "CPU_DRY_RUN_PASS"
    assert receipt["synthetic_folds"] == [41] * 5
    assert receipt["fit_count_planned"] == 45
    assert receipt["data_reads"] == receipt["checkpoint_reads"] == 0
    assert receipt["estimator_fits"] == receipt["outcome_cells"] == 0
    assert not receipt["GPU_used"]
    assert not receipt["qualification_lock_used"]
    assert not receipt["FIFO_ticket_created"]
