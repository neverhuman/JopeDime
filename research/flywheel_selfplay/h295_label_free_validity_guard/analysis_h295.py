"""Statistical analysis and verdict evaluation for H295 (h295_label_free_validity_guard):
- Family-equal mean gain vs slot 0
- 10,000-draw family bootstrap (seed 295150) for per-seed CI95 and 3-seed median CI95
- Family-median gain, W/T/L (|d| <= 1e-4 tie band; ties never win), one-sided exact binomial sign test
- Harmed families (family loss > 0.01 vs slot 0), invalid pick counts (~gate_k), switched-to-invalid counts
- Hunter (2004) Minorization-Maximization Bradley-Terry Elo anchored at slot 0 = 1500.0 (1e-4 tie band).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import binomtest

import h265_core as core

BOOT_DRAWS: int = 10000
BOOT_SEED: int = 295150
TIE_BAND: float = 1e-4
HARM_THRESH: float = 0.01


def compute_mm_bradley_terry_elo(
    fam_u_by_arm: np.ndarray,
    tie_band: float = TIE_BAND,
    anchor_idx: int = 0,
    anchor_elo: float = 1500.0,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> List[float]:
    """Hunter (2004) MM algorithm for Bradley-Terry Elo on family utilities (n_fams, n_arms).
    Pairwise comparisons within each family use `tie_band = 1e-4` (ties contribute 0.5 win to each arm)
    plus a tiny Laplace smoothing `1e-3` so disconnected extremes remain finite.
    """
    n_fams, n_arms = fam_u_by_arm.shape
    wins = np.zeros((n_arms, n_arms), dtype=np.float64)
    for i in range(n_arms):
        for j in range(i + 1, n_arms):
            diff = fam_u_by_arm[:, i] - fam_u_by_arm[:, j]
            w_ij = float(np.sum(diff > tie_band))
            w_ji = float(np.sum(diff < -tie_band))
            ties = float(n_fams) - w_ij - w_ji
            wins[i, j] = w_ij + 0.5 * ties + 1e-3
            wins[j, i] = w_ji + 0.5 * ties + 1e-3

    n_ij = wins + wins.T
    w_i = wins.sum(axis=1)
    gamma = np.ones(n_arms, dtype=np.float64)
    for _ in range(max_iter):
        g_sum = gamma[:, None] + gamma[None, :]
        denom = np.where(g_sum > 0, n_ij / g_sum, 0.0)
        np.fill_diagonal(denom, 0.0)
        d_i = denom.sum(axis=1)
        new_gamma = np.where(d_i > 0, w_i / d_i, gamma)
        new_gamma = new_gamma / max(float(new_gamma[anchor_idx]), 1e-30)
        if float(np.max(np.abs(new_gamma - gamma))) < tol:
            gamma = new_gamma
            break
        gamma = new_gamma

    elos = [
        float(anchor_elo + 400.0 * math.log10(max(float(g), 1e-30)))
        for g in gamma
    ]
    return elos


def evaluate_panel(
    u_panel: np.ndarray,
    picks_panel: np.ndarray,
    gate_k: np.ndarray,
    fams: Sequence[str],
    arm_names: Sequence[str],
    seeds: Sequence[int],
    boot_seed: int = BOOT_SEED,
    n_boot: int = BOOT_DRAWS,
) -> Dict[str, Any]:
    """Evaluate a (n_seeds, n_ctx, n_arms) utility & pick panel on DL-Train."""
    n_seeds, n_ctx, n_arms = u_panel.shape
    fams_str = [str(f) for f in fams]
    fam_ids = sorted(set(fams_str))
    n_fams = len(fam_ids)
    fams_arr = np.asarray(fams_str)
    fam_masks = [fams_arr == f for f in fam_ids]

    # Per-seed family utility matrix: (n_seeds, n_fams, n_arms)
    fam_u = np.zeros((n_seeds, n_fams, n_arms), dtype=np.float64)
    for r in range(n_seeds):
        for a in range(n_arms):
            col = u_panel[r, :, a]
            fam_u[r, :, a] = [float(col[m].mean()) for m in fam_masks]

    # MM Bradley-Terry Elo per seed
    elo_by_seed: List[Dict[str, float]] = []
    for r in range(n_seeds):
        elos = compute_mm_bradley_terry_elo(fam_u[r], tie_band=TIE_BAND, anchor_idx=0, anchor_elo=1500.0)
        elo_by_seed.append({arm_names[a]: round(float(elos[a]), 6) for a in range(n_arms)})
    median_elo = {
        arm_names[a]: round(float(np.median([elo_by_seed[r][arm_names[a]] for r in range(n_seeds)])), 6)
        for a in range(n_arms)
    }

    arms_summary: Dict[str, Any] = {}
    for a_idx, arm in enumerate(arm_names):
        per_seed_gain: List[float] = []
        per_seed_ci95: List[List[float]] = []
        per_seed_lcb_gt0: List[bool] = []
        per_seed_fam_median: List[float] = []
        per_seed_wtl: List[Dict[str, int]] = []
        per_seed_sign_p: List[float] = []
        per_seed_harmed: List[int] = []
        per_seed_invalid: List[int] = []
        per_seed_switched_invalid: List[int] = []
        per_seed_changed: List[int] = []
        per_seed_abstain: List[float] = []
        boot_means_all_seeds = np.zeros((n_seeds, n_boot), dtype=np.float64)

        for r in range(n_seeds):
            diff_f = fam_u[r, :, a_idx] - fam_u[r, :, 0]
            gain = float(np.mean(diff_f))
            f_med = float(np.median(diff_f))

            rng = np.random.default_rng(core.stream_seed(boot_seed, "h295_boot", int(seeds[r])))
            boot_ix = rng.integers(0, n_fams, size=(n_boot, n_fams))
            b_means = diff_f[boot_ix].mean(axis=1)
            boot_means_all_seeds[r] = b_means
            lcb, ucb = [float(x) for x in np.quantile(b_means, [0.025, 0.975])]

            wins = int(np.sum(diff_f > TIE_BAND))
            losses = int(np.sum(diff_f < -TIE_BAND))
            ties = int(n_fams - wins - losses)
            n_non_tied = wins + losses
            if n_non_tied > 0:
                sign_p = float(binomtest(wins, n_non_tied, 0.5, alternative="greater").pvalue)
            else:
                sign_p = 1.0

            harmed = int(np.sum(diff_f < -HARM_THRESH - 1e-12))
            p_r = picks_panel[r, :, a_idx]
            inv = int(np.sum(~gate_k[np.arange(n_ctx), p_r]))
            sw_inv = int(np.sum((p_r != 0) & (~gate_k[np.arange(n_ctx), p_r])))
            chg = int(np.sum(p_r != 0))
            abst = float(np.mean(p_r == 0))

            per_seed_gain.append(round(gain, 6))
            per_seed_ci95.append([round(lcb, 6), round(ucb, 6)])
            per_seed_lcb_gt0.append(bool(lcb > 0.0))
            per_seed_fam_median.append(round(f_med, 6))
            per_seed_wtl.append({"W": wins, "T": ties, "L": losses})
            per_seed_sign_p.append(float(sign_p))
            per_seed_harmed.append(harmed)
            per_seed_invalid.append(inv)
            per_seed_switched_invalid.append(sw_inv)
            per_seed_changed.append(chg)
            per_seed_abstain.append(round(abst, 6))

        med_boot_dist = np.median(boot_means_all_seeds, axis=0)
        med_lcb, med_ucb = [float(x) for x in np.quantile(med_boot_dist, [0.025, 0.975])]
        exact_gains = [float( np.mean(fam_u[r, :, a_idx] - fam_u[r, :, 0]) ) for r in range(n_seeds)]

        arms_summary[arm] = {
            "median_gain": round(float(np.median(exact_gains)), 6),
            "median_gain_exact": float(np.median(exact_gains)),
            "mean_gain": round(float(np.mean(exact_gains)), 6),
            "median_gain_ci95": [round(med_lcb, 6), round(med_ucb, 6)],
            "per_seed_gain": per_seed_gain,
            "per_seed_gain_exact": exact_gains,
            "per_seed_ci95": per_seed_ci95,
            "per_seed_lcb95_gt_0": per_seed_lcb_gt0,
            "lcb95_positive_seeds": int(sum(per_seed_lcb_gt0)),
            "family_median_gain_median": round(float(np.median(per_seed_fam_median)), 6),
            "per_seed_family_median_gain": per_seed_fam_median,
            "wtl_by_seed": per_seed_wtl,
            "wtl_median": {
                "W": float(np.median([w["W"] for w in per_seed_wtl])),
                "T": float(np.median([w["T"] for w in per_seed_wtl])),
                "L": float(np.median([w["L"] for w in per_seed_wtl])),
            },
            "sign_test_p_by_seed": per_seed_sign_p,
            "sign_test_p_median": float(np.median(per_seed_sign_p)),
            "harmed_families_by_seed": per_seed_harmed,
            "harmed_families_median": float(np.median(per_seed_harmed)),
            "harmed_families_mean": round(float(np.mean(per_seed_harmed)), 4),
            "invalid_picks_by_seed": per_seed_invalid,
            "invalid_picks_median": float(np.median(per_seed_invalid)),
            "switched_to_invalid_by_seed": per_seed_switched_invalid,
            "switched_to_invalid_median": float(np.median(per_seed_switched_invalid)),
            "changed_by_seed": per_seed_changed,
            "changed_median": float(np.median(per_seed_changed)),
            "abstain_rate_by_seed": per_seed_abstain,
            "abstain_rate_median": round(float(np.median(per_seed_abstain)), 6),
            "elo_by_seed": [elo_by_seed[r][arm] for r in range(n_seeds)],
            "elo_median": median_elo[arm],
        }

    return {
        "n_seeds": int(n_seeds),
        "n_contexts": int(n_ctx),
        "n_families": int(n_fams),
        "slot0_family_mean": round(float(np.mean(fam_u[0, :, 0])), 6),
        "slot0_context_mean": round(float(np.mean(u_panel[0, :, 0])), 6),
        "arms": arms_summary,
        "elo_by_seed": elo_by_seed,
        "median_elo": median_elo,
    }


def evaluate_h295_verdict(panel_eval: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate H295 strict decision rule on DL-Train 4-fold family-grouped outer CV:
    Primary (`ENS_VOTE2+VALGUARD`) passes iff on 3/3 seeds:
      1. `invalid picks <= slot 0` (and cuts invalid picks relative to `ENS_VOTE2`: `invalid_median < ENS_VOTE2`
         or `invalid_by_seed <= ENS_VOTE2` with strict cut on >= 1 seed)
      2. `family-bootstrap LCB95 of gain vs slot 0 > 0` on 3/3 seeds
      3. `median gain >= 0.85 x ENS_VOTE2 median`
      4. `harmed families not above ENS_VOTE2` (`harmed_by_seed <= ENS_VOTE2` on 3/3 seeds and `harmed_median <= ENS_VOTE2`)
      5. `shuffled-label null fails` (does not reproduce the invalid-pick cut and/or fails the primary clauses).
    """
    arms = panel_eval["arms"]
    slot0 = arms["SLOT0"]
    ref = arms["ENS_VOTE2"]
    prim = arms["ENS_VOTE2+VALGUARD"]
    sec = arms["ENS_VOTE2+VALGUARD+HARMVETO"]
    shuf = arms["SHUFFLED_NULL"]

    slot0_inv = slot0["invalid_picks_by_seed"]
    ref_inv = ref["invalid_picks_by_seed"]
    ref_harm = ref["harmed_families_by_seed"]
    ref_med_gain = float(ref["median_gain_exact"])
    gain_85_thresh = 0.85 * ref_med_gain

    def _check_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
        c_inv = cand["invalid_picks_by_seed"]
        c_harm = cand["harmed_families_by_seed"]
        c_med_gain = float(cand["median_gain_exact"])
        inv_le_slot0_3of3 = all(int(c_inv[r]) <= int(slot0_inv[r]) for r in range(3))
        inv_cut_vs_ref = bool(
            all(int(c_inv[r]) <= int(ref_inv[r]) for r in range(3))
            and sum(int(c_inv[r]) for r in range(3)) < sum(int(ref_inv[r]) for r in range(3))
        )
        lcb_gt0_3of3 = bool(int(cand["lcb95_positive_seeds"]) == 3)
        gain_ge_85pct = bool(c_med_gain >= gain_85_thresh - 1e-12)
        harm_not_above_ref_3of3 = all(int(c_harm[r]) <= int(ref_harm[r]) for r in range(3))
        harm_med_not_above_ref = bool(float(cand["harmed_families_median"]) <= float(ref["harmed_families_median"]))
        passes_all = bool(
            inv_le_slot0_3of3
            and inv_cut_vs_ref
            and lcb_gt0_3of3
            and gain_ge_85pct
            and harm_not_above_ref_3of3
            and harm_med_not_above_ref
        )
        return {
            "median_gain": round(c_med_gain, 6),
            "ens_vote2_median_gain": round(ref_med_gain, 6),
            "gain_ratio_vs_ens_vote2": round(c_med_gain / max(ref_med_gain, 1e-12), 6),
            "gain_threshold_85pct": round(gain_85_thresh, 6),
            "pass_gain_ge_85pct": gain_ge_85pct,
            "pass_lcb95_gt_0_on_3of3_seeds": lcb_gt0_3of3,
            "invalid_picks_by_seed": c_inv,
            "slot0_invalid_picks_by_seed": slot0_inv,
            "ens_vote2_invalid_picks_by_seed": ref_inv,
            "pass_invalid_le_slot0_3of3": inv_le_slot0_3of3,
            "pass_invalid_cut_vs_ens_vote2": inv_cut_vs_ref,
            "harmed_families_by_seed": c_harm,
            "ens_vote2_harmed_families_by_seed": ref_harm,
            "pass_harmed_not_above_ens_vote2_3of3": harm_not_above_ref_3of3,
            "pass_harmed_median_not_above_ens_vote2": harm_med_not_above_ref,
            "passes_all_candidate_clauses": passes_all,
        }

    prim_chk = _check_candidate(prim)
    sec_chk = _check_candidate(sec)
    shuf_chk = _check_candidate(shuf)

    shuf_fails = bool(not shuf_chk["passes_all_candidate_clauses"] and not shuf_chk["pass_invalid_cut_vs_ens_vote2"])
    verdict = "VALGUARD_ADVANCE" if (prim_chk["passes_all_candidate_clauses"] and shuf_fails) else "VALGUARD_NOT_ADVANCED"

    return {
        "verdict": verdict,
        "primary_arm": "ENS_VOTE2+VALGUARD",
        "secondary_arm": "ENS_VOTE2+VALGUARD+HARMVETO",
        "shuffled_null_arm": "SHUFFLED_NULL",
        "shuffled_null_fails": shuf_fails,
        "primary_check": prim_chk,
        "secondary_check": sec_chk,
        "shuffled_null_check": shuf_chk,
    }
