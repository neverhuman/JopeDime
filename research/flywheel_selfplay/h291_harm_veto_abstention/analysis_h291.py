"""Metrics, 10,000-draw family bootstrap (seed 291150), non-tied sign test, MM Bradley-Terry Elo,
and Step 6 verdict evaluation for H291 (h291_harm_veto_abstention)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import h265_core as core

BOOT_SEED = 291150
N_BOOT = 10_000
TIE_BAND = 1e-4
HARM_THRESH = 0.01


def mm_bradley_terry_elo(
    fam_mat: np.ndarray,
    anchor_idx: int = 0,
    anchor_elo: float = 1500.0,
    tie_band: float = TIE_BAND,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Hunter (2004) Minorization-Maximization (MM) Bradley-Terry Elo on family-mean utilities,
    with ties |d| <= tie_band split 0.5/0.5 and slot 0 anchored at 1500.0."""
    n_fam, n_arms = fam_mat.shape
    w_mat = np.zeros((n_arms, n_arms), dtype=np.float64)
    n_mat = np.zeros((n_arms, n_arms), dtype=np.float64)
    for i in range(n_arms):
        for j in range(n_arms):
            if i == j:
                continue
            d = fam_mat[:, i] - fam_mat[:, j]
            wins = float(np.sum(d > tie_band))
            ties = float(np.sum(np.abs(d) <= tie_band))
            w_mat[i, j] = wins + 0.5 * ties + 1e-3  # tiny Laplace regularization for identifiability
            n_mat[i, j] = float(n_fam) + 2e-3

    gamma = np.ones(n_arms, dtype=np.float64)
    w_i = np.sum(w_mat, axis=1)
    for _ in range(max_iter):
        gamma_prev = gamma.copy()
        for i in range(n_arms):
            denom = 0.0
            for j in range(n_arms):
                if i == j:
                    continue
                denom += n_mat[i, j] / (gamma[i] + gamma[j])
            if denom > 0.0:
                gamma[i] = w_i[i] / denom
        gamma /= np.exp(np.mean(np.log(np.maximum(gamma, 1e-15))))
        if float(np.max(np.abs(gamma - gamma_prev))) < tol:
            break

    elo = 400.0 * np.log10(np.maximum(gamma, 1e-15))
    elo = elo - elo[anchor_idx] + float(anchor_elo)
    return elo


def binom_two_sided_pvalue(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign-test p-value on non-tied families (H0: p = 0.5)."""
    n = int(wins + losses)
    if n <= 0:
        return 1.0
    k = min(int(wins), int(losses))
    # Sum_{j=0..k} C(n, j) * 2^{-n} in log space for numerical stability
    log2_n = n * math.log(2.0)
    tail = 0.0
    for j in range(k + 1):
        log_prob = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) - log2_n
        tail += math.exp(log_prob)
    return float(min(1.0, 2.0 * tail))


def binom_one_sided_greater_pvalue(wins: int, losses: int) -> float:
    """Exact one-sided binomial sign-test p-value P(W >= wins | n = wins + losses, p = 0.5)."""
    n = int(wins + losses)
    if n <= 0:
        return 1.0
    log2_n = n * math.log(2.0)
    tail = 0.0
    for j in range(int(wins), n + 1):
        log_prob = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) - log2_n
        tail += math.exp(log_prob)
    return float(min(1.0, tail))


def compute_arm_seed_metrics(
    u_slot0: np.ndarray,
    u_arm: np.ndarray,
    picks: np.ndarray,
    fams: Sequence[str],
    boot_idx: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-seed metrics for one arm against slot 0 on a given cohort."""
    n_ctx = int(len(u_slot0))
    fams_str = np.asarray([str(f) for f in fams])
    unique_fams = sorted(set(fams_str.tolist()))
    n_fam = len(unique_fams)

    diff_ctx = np.asarray(u_arm, dtype=np.float64) - np.asarray(u_slot0, dtype=np.float64)
    fam_gains = np.zeros(n_fam, dtype=np.float64)
    fam_slot0 = np.zeros(n_fam, dtype=np.float64)
    fam_arm = np.zeros(n_fam, dtype=np.float64)
    for fi, f in enumerate(unique_fams):
        mask = fams_str == f
        fam_gains[fi] = float(np.mean(diff_ctx[mask]))
        fam_slot0[fi] = float(np.mean(u_slot0[mask]))
        fam_arm[fi] = float(np.mean(u_arm[mask]))

    mean_gain = float(np.mean(fam_gains))
    median_fam_gain = float(np.median(fam_gains))
    iqr_fam_gain = [
        float(np.percentile(fam_gains, 25.0)),
        float(np.percentile(fam_gains, 75.0)),
    ]

    # 10,000 family bootstrap draws (boot_idx shape: (N_BOOT, n_fam))
    boot_means = np.mean(fam_gains[boot_idx], axis=1)
    ci95 = [
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    ]

    wins = int(np.sum(fam_gains > TIE_BAND))
    losses = int(np.sum(fam_gains < -TIE_BAND))
    ties = int(n_fam - wins - losses)
    harmed = int(np.sum(fam_gains < -HARM_THRESH))
    helped = int(np.sum(fam_gains > HARM_THRESH))

    raw_neg2_arm = int(np.sum(np.asarray(u_arm, dtype=np.float64) <= -1.999))
    raw_neg2_slot0 = int(np.sum(np.asarray(u_slot0, dtype=np.float64) <= -1.999))
    changed_ctx = int(np.sum(np.asarray(picks, dtype=np.int64) > 0))
    abstain_ctx = int(np.sum(np.asarray(picks, dtype=np.int64) == 0))
    abstain_rate = float(abstain_ctx / max(1, n_ctx))

    return {
        "n_families": int(n_fam),
        "n_contexts": int(n_ctx),
        "family_equal_mean_gain": mean_gain,
        "family_equal_slot0_mean": float(np.mean(fam_slot0)),
        "family_equal_arm_mean": float(np.mean(fam_arm)),
        "ci95": ci95,
        "lcb95": float(ci95[0]),
        "ucb95": float(ci95[1]),
        "lcb95_gt_0": bool(ci95[0] > 0.0),
        "family_median_gain": median_fam_gain,
        "family_iqr_gain": iqr_fam_gain,
        "non_tied_wins": wins,
        "non_tied_losses": losses,
        "ties": ties,
        "sign_test_p_two_sided": binom_two_sided_pvalue(wins, losses),
        "sign_test_p_one_sided_greater": binom_one_sided_greater_pvalue(wins, losses),
        "harmed_families_gt_0p01": harmed,
        "helped_families_gt_0p01": helped,
        "raw_neg2_picks": raw_neg2_arm,
        "slot0_raw_neg2_picks": raw_neg2_slot0,
        "raw_neg2_delta_vs_slot0": int(raw_neg2_arm - raw_neg2_slot0),
        "changed_contexts": changed_ctx,
        "abstain_contexts": abstain_ctx,
        "abstain_rate": abstain_rate,
        "_boot_means": boot_means,
        "_fam_arm": fam_arm,
    }


def summarize_panel_across_seeds(
    arm_names: Sequence[str],
    u_panel: np.ndarray,
    picks_panel: np.ndarray,
    fams: Sequence[str],
    seed_labels: Sequence[int],
    boot_seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> Dict[str, Any]:
    """Summarize a 3-seed panel of arms (including SLOT0 at index 0) with 10k family bootstrap and MM BT Elo.
    u_panel: shape (n_seeds, n_ctx, n_arms)
    picks_panel: shape (n_seeds, n_ctx, n_arms)
    """
    n_seeds, n_ctx, n_arms = u_panel.shape
    fams_str = np.asarray([str(f) for f in fams])
    unique_fams = sorted(set(fams_str.tolist()))
    n_fam = len(unique_fams)

    rng = np.random.default_rng(int(boot_seed))
    boot_idx = rng.integers(0, n_fam, size=(int(n_boot), n_fam), dtype=np.int32)

    per_seed_by_arm: Dict[str, List[Dict[str, Any]]] = {name: [] for name in arm_names}
    elo_per_seed: List[Dict[str, float]] = []
    elo_boot_by_seed: List[np.ndarray] = []

    for r in range(n_seeds):
        u_s0 = u_panel[r, :, 0]
        fam_mat = np.zeros((n_fam, n_arms), dtype=np.float64)
        for a_idx, name in enumerate(arm_names):
            m = compute_arm_seed_metrics(
                u_slot0=u_s0,
                u_arm=u_panel[r, :, a_idx],
                picks=picks_panel[r, :, a_idx],
                fams=fams_str,
                boot_idx=boot_idx,
            )
            fam_mat[:, a_idx] = m.pop("_fam_arm")
            per_seed_by_arm[name].append(m)

        # Compute MM Bradley-Terry Elo on family-mean utilities for seed r (slot 0 anchored at 1500, 1e-4 tie band)
        elo_r = mm_bradley_terry_elo(fam_mat, anchor_idx=0, anchor_elo=1500.0, tie_band=TIE_BAND)
        elo_per_seed.append({arm_names[a]: float(elo_r[a]) for a in range(n_arms)})
        for a_idx, name in enumerate(arm_names):
            per_seed_by_arm[name][r]["mm_bt_elo"] = float(elo_r[a_idx])

    # Aggregate across seeds for each arm
    arms_summary: Dict[str, Any] = {}
    for a_idx, name in enumerate(arm_names):
        seed_list = per_seed_by_arm[name]
        boot_stack = np.stack([s.pop("_boot_means") for s in seed_list], axis=0)  # (n_seeds, n_boot)
        median_boot = np.median(boot_stack, axis=0)
        mean_boot = np.mean(boot_stack, axis=0)

        seed_gains = [float(s["family_equal_mean_gain"]) for s in seed_list]
        seed_med_fam_gains = [float(s["family_median_gain"]) for s in seed_list]
        seed_harms = [int(s["harmed_families_gt_0p01"]) for s in seed_list]
        seed_helped = [int(s["helped_families_gt_0p01"]) for s in seed_list]
        seed_neg2 = [int(s["raw_neg2_picks"]) for s in seed_list]
        seed_abst = [float(s["abstain_rate"]) for s in seed_list]
        seed_elos = [float(s["mm_bt_elo"]) for s in seed_list]
        seed_wins = [int(s["non_tied_wins"]) for s in seed_list]
        seed_losses = [int(s["non_tied_losses"]) for s in seed_list]
        seed_ties = [int(s["ties"]) for s in seed_list]
        seed_p_two = [float(s["sign_test_p_two_sided"]) for s in seed_list]
        seed_p_one = [float(s["sign_test_p_one_sided_greater"]) for s in seed_list]

        arms_summary[name] = {
            "median_of_seeds_gain": float(np.median(seed_gains)),
            "median_of_seeds_ci95": [
                float(np.percentile(median_boot, 2.5)),
                float(np.percentile(median_boot, 97.5)),
            ],
            "mean_of_seeds_gain": float(np.mean(seed_gains)),
            "mean_of_seeds_ci95": [
                float(np.percentile(mean_boot, 2.5)),
                float(np.percentile(mean_boot, 97.5)),
            ],
            "seed_gains": seed_gains,
            "per_seed_ci95": [s["ci95"] for s in seed_list],
            "lcb95_positive_seeds": int(sum(1 for s in seed_list if s["lcb95_gt_0"])),
            "all_seeds_lcb95_gt_0": bool(all(s["lcb95_gt_0"] for s in seed_list)),
            "median_of_seeds_family_median_gain": float(np.median(seed_med_fam_gains)),
            "seed_family_median_gains": seed_med_fam_gains,
            "harmed_families_median": float(np.median(seed_harms)),
            "harmed_families_per_seed": seed_harms,
            "helped_families_median": float(np.median(seed_helped)),
            "helped_families_per_seed": seed_helped,
            "raw_neg2_median": float(np.median(seed_neg2)),
            "raw_neg2_per_seed": seed_neg2,
            "slot0_raw_neg2": int(seed_list[0]["slot0_raw_neg2_picks"]),
            "raw_neg2_le_slot0_all_seeds": bool(
                all(int(s["raw_neg2_picks"]) <= int(s["slot0_raw_neg2_picks"]) for s in seed_list)
            ),
            "abstain_rate_median": float(np.median(seed_abst)),
            "abstain_rate_per_seed": seed_abst,
            "mm_bt_elo_median": float(np.median(seed_elos)),
            "mm_bt_elo_per_seed": seed_elos,
            "non_tied_wins_per_seed": seed_wins,
            "non_tied_losses_per_seed": seed_losses,
            "ties_per_seed": seed_ties,
            "sign_test_p_two_sided_per_seed": seed_p_two,
            "sign_test_p_one_sided_per_seed": seed_p_one,
            "per_seed": {str(seed_labels[r]): seed_list[r] for r in range(n_seeds)},
        }

    return {
        "n_families": int(n_fam),
        "n_contexts": int(n_ctx),
        "n_seeds": int(n_seeds),
        "seed_labels": [int(s) for s in seed_labels],
        "bootstrap": {"seed": int(boot_seed), "n_draws": int(n_boot)},
        "arms": arms_summary,
    }


def evaluate_part_a_gate(part_a_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate Step 6 HARMVETO_ADVANCE gate on Part A (DL-Train 4-fold inner CV):
    HARMVETO_ADVANCE iff Part A:
      1. median gain >= 0.7 x CLIP median
      2. LCB95 > 0 on 3/3 seeds
      3. harmed median <= ENS_VOTE3 harmed median
      4. raw -2 <= slot 0 (on all 3 seeds / median)
      5. PLACEBO and NULL_LF fail the same gate.
    Otherwise HARMVETO_NOT_ADVANCED.
    """
    arms = part_a_summary["arms"]
    clip_med_gain = float(arms["CLIP"]["median_of_seeds_gain"])
    gain_floor = float(0.70 * clip_med_gain)
    ens3_harm_med = float(arms["ENS_VOTE3"]["harmed_families_median"])
    slot0_neg2 = int(arms["SLOT0"]["slot0_raw_neg2"])

    def _check_single_arm(arm_name: str) -> Dict[str, Any]:
        a = arms[arm_name]
        med_gain = float(a["median_of_seeds_gain"])
        c1_gain = bool(med_gain >= gain_floor)
        c2_lcb = bool(int(a["lcb95_positive_seeds"]) == 3)
        c3_harm = bool(float(a["harmed_families_median"]) <= ens3_harm_med)
        c4_neg2 = bool(
            float(a["raw_neg2_median"]) <= float(slot0_neg2)
            and bool(a["raw_neg2_le_slot0_all_seeds"])
        )
        passed = bool(c1_gain and c2_lcb and c3_harm and c4_neg2)
        return {
            "arm": arm_name,
            "median_gain": med_gain,
            "gain_retention_vs_clip": float(med_gain / clip_med_gain) if clip_med_gain > 0 else 0.0,
            "gain_ge_0p7_clip": c1_gain,
            "lcb95_positive_seeds": int(a["lcb95_positive_seeds"]),
            "lcb95_gt_0_all_3_seeds": c2_lcb,
            "harmed_median": float(a["harmed_families_median"]),
            "harmed_median_le_ens_vote3": c3_harm,
            "raw_neg2_median": float(a["raw_neg2_median"]),
            "raw_neg2_per_seed": a["raw_neg2_per_seed"],
            "raw_neg2_le_slot0": c4_neg2,
            "passes_arm_criteria": passed,
        }

    eval_by_arm = {
        name: _check_single_arm(name)
        for name in ("CLIP+HARMVETO", "ENS_VOTE2+HARMVETO", "PLACEBO", "NULL_LF")
        if name in arms
    }
    placebo_fails = not eval_by_arm["PLACEBO"]["passes_arm_criteria"]
    nulllf_fails = not eval_by_arm["NULL_LF"]["passes_arm_criteria"]

    candidate_arms = ["CLIP+HARMVETO", "ENS_VOTE2+HARMVETO"]
    passing_candidates = [
        name for name in candidate_arms if eval_by_arm[name]["passes_arm_criteria"]
    ]
    verdict = (
        "HARMVETO_ADVANCE"
        if (len(passing_candidates) > 0 and placebo_fails and nulllf_fails)
        else "HARMVETO_NOT_ADVANCED"
    )

    # Select best-by-prereg arm to score descriptively in Part B:
    # Prefer a passing candidate arm; otherwise pick the candidate arm with lowest harmed_median
    # among those with LCB95 > 0 on all 3 seeds (tie-broken by highest median_gain).
    best_arm = sorted(
        candidate_arms,
        key=lambda n: (
            1 if eval_by_arm[n]["passes_arm_criteria"] else 0,
            1 if eval_by_arm[n]["lcb95_gt_0_all_3_seeds"] else 0,
            1 if eval_by_arm[n]["harmed_median_le_ens_vote3"] else 0,
            eval_by_arm[n]["median_gain"],
        ),
        reverse=True,
    )[0]

    return {
        "verdict": verdict,
        "clip_median_gain": clip_med_gain,
        "required_gain_floor_0p7_clip": gain_floor,
        "ens_vote3_harmed_median": ens3_harm_med,
        "slot0_raw_neg2": slot0_neg2,
        "placebo_fails_gate": bool(placebo_fails),
        "null_lf_fails_gate": bool(nulllf_fails),
        "passing_candidates": passing_candidates,
        "best_by_prereg_arm": best_arm,
        "arm_checks": eval_by_arm,
    }
