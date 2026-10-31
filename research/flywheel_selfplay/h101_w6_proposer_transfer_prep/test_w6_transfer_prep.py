"""Unit tests for H101 W6 cross-family outcome transfer preparation and fail-closed verdict gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from w6_transfer_prep import (  # noqa: E402
    analyze_w6_panel,
    eval_picks,
    eval_scorer,
    fit_empirical_bayes_group_mean,
    fit_weighted_bilinear,
    hedge_picks,
)


def test_fit_weighted_bilinear_rank_and_weights() -> None:
    rng = np.random.default_rng(701)
    n, dz, dx = 120, 6, 8
    z = rng.standard_normal((n, dz))
    x = rng.standard_normal((n, dx))
    true_w = rng.standard_normal((dz, 2)) @ rng.standard_normal((2, dx))
    r = np.sum((z @ true_w) * x, axis=1) + 0.01 * rng.standard_normal(n)
    weights = rng.uniform(0.5, 2.0, size=n)

    w_hat = fit_weighted_bilinear(z, x, r, weights=weights, ridge=1.0, rank=3)
    assert w_hat.shape == (dz, dx)
    _, s_vals, _ = np.linalg.svd(w_hat, full_matrices=False)
    assert np.all(s_vals[3:] < 1e-10)
    pred = np.sum((z @ w_hat) * x, axis=1)
    corr = float(np.corrcoef(r, pred)[0, 1])
    assert corr > 0.90


def test_fit_empirical_bayes_group_mean_shrinkage_and_unseen_fallback() -> None:
    gids = ["prog_a", "prog_a", "prog_b"]
    resids = np.array([0.4, 0.4, -0.6], dtype=np.float64)
    weights = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    eb = fit_empirical_bayes_group_mean(gids, resids, weights=weights, tau=2.0)

    # prog_a seen 2 times with mean +0.4 -> shrink = 2 / (2 + 2) = 0.5 -> +0.2
    assert np.isclose(eb["prog_a"], 0.2)
    # prog_b seen 1 time with mean -0.6 -> shrink = 1 / (1 + 2) = 1/3 -> -0.2
    assert np.isclose(eb["prog_b"], -0.2)
    # unseen program ID is absent (caller uses .get(id, 0.0) on top of PRIOR, never fabricating 0 utility)
    assert eb.get("prog_unseen", 0.0) == 0.0


def test_hedge_picks_and_eval_scorer_respect_anchor_boundary() -> None:
    u = np.array([1.0, 1.2, 0.8, 1.1, 0.5, 1.4, 1.6, 0.9, 1.3, 0.7], dtype=np.float64)
    ids = [f"{i:064x}" for i in range(len(u))]
    prior = np.array([9.0, 9.0, 9.0, 9.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.4], dtype=np.float64)
    sec = np.array([9.0, 9.0, 9.0, 9.0, 0.2, 0.7, 0.95, 0.1, 0.85, 0.3], dtype=np.float64)

    hp = hedge_picks(prior, sec, ids)
    assert len(hp) == 4
    assert len(set(hp)) == 4
    assert all(i >= 4 for i in hp)
    # prior top-2 unrevealed are indices 5 (0.9) and 7 (0.8); sec top-2 among rest are 6 (0.95) and 8 (0.85)
    assert hp == [5, 7, 6, 8]

    u0, u4_val, leg_val = eval_picks(u, hp)
    assert np.isclose(u0, 1.2)
    assert np.isclose(u4_val, 1.6)
    assert np.isclose(leg_val, 1.4)

    u0_s, u4_s, leg_s, top4_s = eval_scorer(u, sec, ids)
    assert top4_s[:2] == [6, 8]
    assert np.isclose(u0_s, 1.2)
    assert np.isclose(u4_s, 1.6)
    assert np.isclose(leg_s, 1.6)


def _synthetic_complete_rows() -> tuple[list[dict], set[str], set[str]]:
    contexts = {f"ctx_{i:02d}" for i in range(20)}
    families = {f"fam_{i:02d}" for i in range(20)}
    rows = []
    for i in range(20):
        c = f"ctx_{i:02d}"
        f = f"fam_{i:02d}"
        for seed in (701, 702, 703):
            rows.append(
                {
                    "context": c,
                    "family": f,
                    "seed": seed,
                    "arm": "PRIOR",
                    "status": "completed",
                    "u4": 1.50,
                }
            )
            rows.append(
                {
                    "context": c,
                    "family": f,
                    "seed": seed,
                    "arm": "TREAT_FAMBAL",
                    "status": "completed",
                    "u4": 1.55,
                }
            )
    return rows, contexts, families


def test_analyze_w6_panel_passes_complete_positive_signal() -> None:
    rows, contexts, families = _synthetic_complete_rows()
    res = analyze_w6_panel(
        rows=rows,
        expected_contexts=contexts,
        expected_families=families,
        expected_arms=("PRIOR", "TREAT_FAMBAL"),
        expected_seeds={701, 702, 703},
        primary_contrasts=(("TREAT_FAMBAL", "PRIOR"),),
        secondary_contrasts=(),
        anchor_arm="PRIOR",
        shortfalls=0,
    )
    assert res["completeness"]["complete"] is True
    assert res["primary"]["TREAT_FAMBAL - PRIOR"]["common_gate_passed"] is True
    assert res["verdict"].startswith("ADVANCE_CACHED_SCREEN")


def test_analyze_w6_panel_rejects_missing_seeds_duplicate_seeds_incompletes_and_shortfalls() -> None:
    rows, contexts, families = _synthetic_complete_rows()

    # Case 1: missing seeds 702/703 on 3 families
    missing_seed_rows = [
        r
        for r in rows
        if not (r["family"] in {"fam_00", "fam_01", "fam_02"} and r["seed"] in {702, 703})
    ]
    res_missing = analyze_w6_panel(
        rows=missing_seed_rows,
        expected_contexts=contexts,
        expected_families=families,
        expected_arms=("PRIOR", "TREAT_FAMBAL"),
        expected_seeds={701, 702, 703},
        primary_contrasts=(("TREAT_FAMBAL", "PRIOR"),),
        secondary_contrasts=(),
    )
    assert res_missing["completeness"]["complete"] is False
    assert res_missing["verdict"] == "INCOMPLETE"
    assert res_missing["primary"]["TREAT_FAMBAL - PRIOR"]["common_gate_passed"] is False

    # Case 2: seed 701 repeated x3 instead of {701, 702, 703}
    dup_seed_rows = [{**r, "seed": 701} for r in rows]
    res_dup = analyze_w6_panel(
        rows=dup_seed_rows,
        expected_contexts=contexts,
        expected_families=families,
        expected_arms=("PRIOR", "TREAT_FAMBAL"),
        expected_seeds={701, 702, 703},
        primary_contrasts=(("TREAT_FAMBAL", "PRIOR"),),
        secondary_contrasts=(),
    )
    assert res_dup["completeness"]["complete"] is False
    assert res_dup["completeness"]["duplicate_key_count"] > 0
    assert res_dup["verdict"] == "INCOMPLETE"

    # Case 3: incomplete rows present
    inc_rows = [dict(r) for r in rows]
    inc_rows[0]["status"] = "incomplete"
    res_inc = analyze_w6_panel(
        rows=inc_rows,
        expected_contexts=contexts,
        expected_families=families,
        expected_arms=("PRIOR", "TREAT_FAMBAL"),
        expected_seeds={701, 702, 703},
        primary_contrasts=(("TREAT_FAMBAL", "PRIOR"),),
        secondary_contrasts=(),
    )
    assert res_inc["completeness"]["complete"] is False
    assert res_inc["completeness"]["incomplete_rows"] == 1
    assert res_inc["verdict"] == "INCOMPLETE"

    # Case 4: proposal shortfalls > 0
    res_short = analyze_w6_panel(
        rows=rows,
        expected_contexts=contexts,
        expected_families=families,
        expected_arms=("PRIOR", "TREAT_FAMBAL"),
        expected_seeds={701, 702, 703},
        primary_contrasts=(("TREAT_FAMBAL", "PRIOR"),),
        secondary_contrasts=(),
        shortfalls=2,
    )
    assert res_short["completeness"]["complete"] is False
    assert res_short["verdict"] == "INCOMPLETE"


def test_eval_picks_rejects_anchor_or_duplicate_indices() -> None:
    u = np.ones(10, dtype=np.float64)
    with pytest.raises(ValueError, match="4 distinct unrevealed"):
        eval_picks(u, [3, 4, 5, 6])
    with pytest.raises(ValueError, match="4 distinct unrevealed"):
        eval_picks(u, [4, 4, 5, 6])
