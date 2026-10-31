"""Synthetic CPU controls for the H129 blend / harm-gate protocol (no corpus, checkpoint or estimator fits)."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h129_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def test_h124_module_is_pinned_and_rebound_to_fresh_fold_and_seeds():
    h124 = r.load_h124()
    assert (h124.SEEDS, h124.FOLD_SEED, h124.SAMPLE_SEED) == (
        (91, 92, 93),
        129029,
        129029,
    )
    assert not set(r.SEEDS) & {64, 65, 66, 51, 52, 53} and r.FOLD_SEED not in (
        12400,
        11900,
    )
    rows = [
        {
            "context_sha256": f"{i:064x}",
            "family_id": f"family-{i:03d}",
            "dl_split": "DL-Train",
            "source_split": "train",
            "task": "regression",
        }
        for i in range(205)
    ]
    groups, _ = h124.folds(rows)
    h124.FOLD_SEED = 12400
    old, _ = h124.folds(rows)
    assert [len(g) for g in groups] == [41] * 5 and groups != old


def test_blend_lambdas_are_fixed():
    ref, gbm = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    np.testing.assert_allclose(r.blend(ref, gbm, 0.25), [1.5, 2.0, 2.5])
    np.testing.assert_allclose(r.blend(ref, gbm, 0.5), [2.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        r.blend(ref, gbm, 1.0)


def test_disagreement_is_rank_based_outcome_free_and_ignores_revealed_slots():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(50)
    assert r.disagreement(ref, ref) == pytest.approx(0.0)
    assert r.disagreement(-ref, ref) == pytest.approx(2.0)
    calibrated = -1.0 + 0.7 * ref  # positive-slope calibration leaves ranks unchanged
    assert r.disagreement(calibrated, ref) == pytest.approx(0.0)
    moved = ref.copy()
    moved[:4] = [9, -9, 9, -9]
    assert r.disagreement(moved, ref) == pytest.approx(0.0)
    assert math.isinf(r.disagreement(np.ones(50), ref))


def test_threshold_rule_uses_fit_harm_target_and_prefers_smaller_quantile_on_ties():
    # 40 families; high-disagreement contexts carry the harm.
    families = [f"f{i}" for i in range(40)]
    d = np.linspace(0.0, 1.0, 40)
    deltas = np.where(d > 0.6, -0.05, 0.02)
    rule = r.choose_threshold(d, deltas, families)
    assert rule["quantile"] == 0.5  # 0.75/1.0 harm 15%/40% > 5%; 0.5 beats 0.25 on gain
    assert rule["tau"] == pytest.approx(np.quantile(d, 0.5))
    flat = r.choose_threshold(d, np.zeros(40), families)
    assert flat["quantile"] == 0.25  # all gains tie at 0: smallest quantile
    closed = r.choose_threshold(d, np.full(40, -0.05), families)
    assert closed["quantile"] is None and closed["tau"] == -math.inf


def test_holm_step_down():
    assert r.holm({"a": 0.010, "b": 0.020, "c": 0.030}) == {
        "a": True,
        "b": True,
        "c": True,
    }
    assert r.holm({"a": 0.010, "b": 0.030, "c": 0.040}) == {
        "a": True,
        "b": False,
        "c": False,
    }
    assert r.holm({"a": 0.030, "b": 0.001, "c": 0.5}) == {
        "a": False,
        "b": True,
        "c": False,
    }


def _packets(gains, harmed=0):
    rows = [
        {"context_sha256": f"{i:064x}", "family_id": f"family-{i:03d}"}
        for i in range(205)
    ]
    packets = {}
    for seed in r.SEEDS:
        packets[str(seed)] = {}
        for i, row in enumerate(rows):
            cell = {"r882": {"u4": 0.5}}
            for arm, g in gains.items():
                cell[arm] = {
                    "u4": 0.5 + (-0.02 if i < harmed and arm == "BLEND_050" else g)
                }
            cell["gate_open"] = True
            packets[str(seed)][row["context_sha256"]] = cell
    return rows, packets


def test_summary_gates_holm_harm_and_report_only(monkeypatch):
    h124 = r.load_h124()
    monkeypatch.setattr(
        h124,
        "elo_h124",
        lambda *a, **k: {
            "registered_estimator": "median_seed_elo",
            "median_seed_elo": {},
            "pooled_score_elo_additional": {},
            "reference_elo": 1500,
        },
    )
    rows, packets = _packets(
        {
            "BLEND_050": 0.004,
            "BLEND_025": 0.0,
            "HARM_GATED_050": 0.003,
            "PURE_GBM": 0.01,
        },
        harmed=21,
    )
    result, arrays = r.summary(h124, rows, packets)
    arms = result["arms"]
    assert (
        arms["BLEND_050"]["harm_count"] == 21 and not arms["BLEND_050"]["admitted"]
    )  # 10.24% > 10%
    assert not arms["BLEND_025"]["admitted"] and not arms["BLEND_025"]["holm_reject"]
    assert arms["HARM_GATED_050"]["admitted"]
    assert arms["PURE_GBM"]["admitted"] is None
    assert (
        result["verdict"] == "CANDIDATE: HARM_GATED_050"
        and result["candidate_breakthrough"]
    )
    assert len(arrays["family_gain_by_seed"]["BLEND_050"]) == 3


def test_metrics_size_is_checked_before_writing():
    assert r.compact_metrics({"a": 1}).endswith("\n")
    with pytest.raises(RuntimeError, match="100 KB"):
        r.compact_metrics({"a": "x" * 100_000})


def test_dry_run_is_data_free():
    out = r.cpu_dry_run()
    assert (
        out["status"] == "CPU_DRY_RUN_PASS"
        and out["data_reads"] == 0
        and out["estimator_fits"] == 0
    )
