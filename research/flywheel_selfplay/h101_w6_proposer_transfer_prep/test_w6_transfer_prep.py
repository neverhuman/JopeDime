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
    compute_template_hashes,
    derive_registered_w6_verdict,
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
    prior = np.array(
        [9.0, 9.0, 9.0, 9.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.4], dtype=np.float64
    )
    sec = np.array(
        [9.0, 9.0, 9.0, 9.0, 0.2, 0.7, 0.95, 0.1, 0.85, 0.3], dtype=np.float64
    )

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


def test_analyze_w6_panel_rejects_missing_seeds_duplicate_seeds_incompletes_and_shortfalls() -> (
    None
):
    rows, contexts, families = _synthetic_complete_rows()

    # Case 1: missing seeds 702/703 on 3 families
    missing_seed_rows = [
        r
        for r in rows
        if not (
            r["family"] in {"fam_00", "fam_01", "fam_02"} and r["seed"] in {702, 703}
        )
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


def test_compute_template_hashes_ignores_numeric_constant_columns_and_detects_structure() -> (
    None
):
    rng = np.random.default_rng(702)
    base = rng.standard_normal((2, 412)).astype(np.float64)
    # Row 1 differs from Row 0 ONLY in constant-magnitude columns 338..363
    base[1] = base[0].copy()
    base[1, 338:364] += 5.0

    hashes_same_struct = compute_template_hashes(base)
    assert len(hashes_same_struct) == 2
    assert hashes_same_struct[0] == hashes_same_struct[1]

    # Perturbing a structural column outside 338..363 changes the proxy hash
    diff_struct = base.copy()
    diff_struct[1, 10] += 0.25
    hashes_diff_struct = compute_template_hashes(diff_struct)
    assert hashes_diff_struct[0] != hashes_diff_struct[1]


def test_derive_registered_w6_verdict_enforces_retrieval_unseen_and_shuffle_kill_gates() -> (
    None
):
    contexts = {f"ctx_{i:02d}" for i in range(20)}
    families = {f"fam_{i:02d}" for i in range(20)}
    arms = ("PRIOR", "RETR", "SHUF", "TREAT_FAMBAL")

    def make_panel(
        prior_val: float, retr_val: float, shuf_val: float, treat_val: float
    ) -> dict:
        rows = []
        for i in range(20):
            c = f"ctx_{i:02d}"
            f = f"fam_{i:02d}"
            for a, val in (
                ("PRIOR", prior_val),
                ("RETR", retr_val),
                ("SHUF", shuf_val),
                ("TREAT_FAMBAL", treat_val),
            ):
                rows.append(
                    {
                        "context": c,
                        "family": f,
                        "seed": 0,
                        "arm": a,
                        "status": "completed",
                        "u4": val,
                    }
                )
        return analyze_w6_panel(
            rows=rows,
            expected_contexts=contexts,
            expected_families=families,
            expected_arms=arms,
            expected_seeds={0},
            primary_contrasts=(("TREAT_FAMBAL", "PRIOR"), ("TREAT_FAMBAL", "RETR")),
            secondary_contrasts=(("TREAT_FAMBAL", "SHUF"),),
            anchor_arm="PRIOR",
            shortfalls=0,
        )

    # Case A: TREAT_FAMBAL beats PRIOR, RETR, and SHUF on both full and unseen-template panels
    panel_all_pass = make_panel(1.50, 1.48, 1.50, 1.55)
    assert panel_all_pass["verdict"].startswith("ADVANCE_CACHED_SCREEN")
    v_pass = derive_registered_w6_verdict(
        track2_u4_panel=panel_all_pass,
        track1_leg_panel=panel_all_pass,
        unseen_tpl_u4_panel=panel_all_pass,
        prospective_panel=None,
    )
    assert v_pass["overall_w6_study_verdict"] == "ADVANCE_TO_PROSPECTIVE"
    assert all(
        v_pass["registered_gate_checks"][k]
        for k in (
            "advance_over_generic_prior_u4",
            "advance_over_current_retrieval_u4",
            "survives_unseen_template_proxy_u4",
            "beats_context_shuffle_placebo_u4",
        )
    )

    # Case B: TREAT_FAMBAL beats PRIOR, but fails RETR and SHUF (as observed on DL-Train)
    panel_retr_shuf_fail = make_panel(1.50, 1.56, 1.55, 1.55)
    assert panel_retr_shuf_fail["verdict"].startswith("KILL_NO_ADVANCE")
    assert panel_retr_shuf_fail["diagnostic_primaries_passing"] == [
        "TREAT_FAMBAL - PRIOR"
    ]
    v_kill_retr = derive_registered_w6_verdict(
        track2_u4_panel=panel_retr_shuf_fail,
        track1_leg_panel=panel_retr_shuf_fail,
        unseen_tpl_u4_panel=panel_all_pass,
        prospective_panel=None,
    )
    assert v_kill_retr["overall_w6_study_verdict"] == "KILL_NO_ADVANCE"
    assert (
        v_kill_retr["registered_gate_checks"]["advance_over_generic_prior_u4"] is True
    )
    assert (
        v_kill_retr["registered_gate_checks"]["advance_over_current_retrieval_u4"]
        is False
    )
    assert (
        v_kill_retr["registered_gate_checks"]["beats_context_shuffle_placebo_u4"]
        is False
    )

    # Case C: Full panel passes prior/retr/shuf, but unseen-template panel fails
    panel_unseen_fail = make_panel(1.50, 1.48, 1.50, 1.50)
    v_kill_unseen = derive_registered_w6_verdict(
        track2_u4_panel=panel_all_pass,
        track1_leg_panel=panel_all_pass,
        unseen_tpl_u4_panel=panel_unseen_fail,
        prospective_panel=None,
    )
    assert v_kill_unseen["overall_w6_study_verdict"] == "KILL_NO_ADVANCE"
    assert (
        v_kill_unseen["registered_gate_checks"]["survives_unseen_template_proxy_u4"]
        is False
    )
