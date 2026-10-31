"""CPU fixture checks for the prospective H119 protocol; no corpus access."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "h119", Path(__file__).with_name("runner.py")
)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def panel(u4=0.02, legacy=0):
    rows = [
        {
            "context_sha256": str(i),
            "family_id": f"f{i:03d}",
            "dl_split": "DL-Train",
            "source_split": "train",
            "task": "regression",
        }
        for i in range(205)
    ]
    packets = {
        str(s): {
            x["context_sha256"]: {
                "GBM": {"u4": 1 + u4, "legacy": 1 + legacy},
                "r882": {"u4": 1.0, "legacy": 1.0},
            }
            for x in rows
        }
        for s in r.SEEDS
    }
    return rows, packets


def test_split_is_fresh_stable_family_grouped_and_order_invariant():
    rows, _ = panel()
    a, mapping = r.folds(rows)
    b, other = r.folds(list(reversed(rows)))
    assert a == b and mapping == other and [len(x) for x in a] == [41] * 5
    old = [
        sorted(x.tolist())
        for x in np.array_split(
            np.random.default_rng(4900).permutation(
                sorted({row["family_id"] for row in rows})
            ),
            5,
        )
    ]
    assert a != old
    with pytest.raises(ValueError):
        r.folds(rows + [rows[0]])
    rows[0]["dl_split"] = "DL-Val"
    with pytest.raises(ValueError):
        r.folds(rows)


def test_both_tracks_match_h108_revealed_and_unrevealed_semantics():
    u = np.array([2.0, 0, 0, 0, 0.2, 0.5, 0.7, 0.9, 1.0])
    cost = np.array([-10.0, 0, 0, 0, 4, 3, 2, 1, 0.0])
    assert r.endpoints(cost, u) == {"u4": 2.0, "legacy": 2.0}
    u[:4] = 0
    assert r.endpoints(cost, u) == {"u4": 1.0, "legacy": 0.0}
    with pytest.raises(ValueError):
        r.endpoints(np.full(9, np.nan), u)


def test_positive_primary_and_nonharm_both_required():
    rows, packets = panel()
    out = r.readout(rows, packets)
    assert out["admitted"] and out["endpoints"]["u4"][
        "median_across_seed_means"
    ] == pytest.approx(0.02)
    assert out["endpoints"]["u4"]["holm_alpha"] == 0.025
    assert out["endpoints"]["legacy"]["holm_pass"]
    assert out["endpoints"]["u4"]["identical_seed_delta_groups"] == [[51, 52, 53]]


@pytest.mark.parametrize(
    "u4,legacy", [(0, 0), (0.001, 0), (-0.01, 0), (0.02, -0.01), (0.02, -0.02)]
)
def test_ties_strict_margins_and_legacy_harm_reject(u4, legacy):
    assert not r.readout(*panel(u4, legacy))["admitted"]


def test_harm_uses_exact_float64_family_count_and_blocks_positive_mean():
    rows, packets = panel(0.2)
    for s in packets:
        for x in rows[:21]:
            packets[s][x["context_sha256"]]["GBM"]["u4"] = 0.98
    out = r.readout(rows, packets)
    assert out["endpoints"]["u4"]["harm_count"] == 21
    assert out["endpoints"]["u4"]["median_across_seed_means"] > 0
    assert not out["admitted"]


@pytest.mark.parametrize("kind", ["missing", "seed", "nonfinite", "duplicate"])
def test_incomplete_panels_do_not_become_zero_or_a_verdict(kind):
    rows, packets = panel()
    if kind == "missing":
        packets["51"].pop("0")
    elif kind == "seed":
        packets["50"] = packets.pop("51")
    elif kind == "nonfinite":
        packets["51"]["0"]["GBM"]["u4"] = float("nan")
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        r.readout(rows, packets)


def test_fit_only_positive_slope_calibration_and_exact_h85_parameters():
    x = np.arange(30, dtype=np.float32)
    y = 2.0 - 3 * x
    a, b = r.calibration(x, y, np.ones(30, bool))
    assert a == pytest.approx(2) and b == pytest.approx(3)
    assert r.calibration(x, 3 * x, np.ones(30, bool))[1] == 0.05
    assert r.calibration(x[:3], y[:3], np.ones(3, bool)) == (0.0, 1.0)
    assert r.MODEL == {
        "max_iter": 300,
        "learning_rate": 0.08,
        "max_leaf_nodes": 63,
        "l2_regularization": 1.0,
    }
    # Positive calibration must preserve the fixed selection when costs differ.
    u = np.arange(30, dtype=float)
    assert r.endpoints((a - b * x).astype(np.float32), u) == r.endpoints(-x, u)


def test_family_bt_has_anchor1500_and_zero_ties_are_not_wins():
    rows, packets = panel(0, 0)
    result = r.elo(rows, packets, "u4", 119001, draws=10)
    assert result["ratings"] == {"GBM_PROG412_CTX": 1500.0, "R882_CONTINUITY": 1500.0}
    assert result["elo_ci95"]["GBM_PROG412_CTX"] == [1500.0, 1500.0]
    assert result["pairwise_win_tie_loss_vs_anchor"]["GBM_PROG412_CTX"]["wins"] == 0
    assert result["pairwise_win_tie_loss_vs_anchor"]["GBM_PROG412_CTX"]["ties"] == 205


def test_unregistered_fits_fail_before_data_or_training(tmp_path):
    for task in [(50, (0,)), (51, ()), (51, (1, 0)), (51, (5,)), (51, (0, 0))]:
        with pytest.raises(ValueError):
            r.fit_one(task, tmp_path, [], {}, {}, {}, {})


def test_vectorized_bt_bootstrap_matches_actual_canonical_helper():
    rows, packets = panel()
    for s in packets:
        for i, row in enumerate(rows):
            packets[s][row["context_sha256"]]["GBM"]["u4"] = 1 + (
                0.2 if i % 3 == 0 else -0.1 if i % 3 == 1 else 0
            )
    result = r.elo(rows, packets, "u4", 123, draws=20)
    helper = r.source_module(
        "python/src/jope/north_star/elo_tracker.py", "h119_actual_test_elo"
    )
    records = [
        helper.DatasetEvalRecord(
            dataset_id=x["context_sha256"],
            family_id=x["family_id"],
            task_type="regression",
            scores_by_contender={
                "GBM_PROG412_CTX": packets["51"][x["context_sha256"]]["GBM"]["u4"],
                "R882_CONTINUITY": 1.0,
            },
        )
        for x in rows
    ]
    ratings = []
    for indices in np.random.default_rng(123).integers(0, 205, (20, 205)):
        sampled = [
            helper.DatasetEvalRecord(
                dataset_id=str(k),
                family_id=str(k),
                task_type="regression",
                scores_by_contender=records[i].scores_by_contender,
            )
            for k, i in enumerate(indices)
        ]
        ratings.append(
            helper.compute_family_grouped_bradley_terry_elo(
                sampled, "R882_CONTINUITY", prior=0.01
            )["ratings"]["GBM_PROG412_CTX"]
        )
    assert (
        result["elo_ci95"]["GBM_PROG412_CTX"]
        == np.quantile(ratings, [0.025, 0.975]).round(2).tolist()
    )
