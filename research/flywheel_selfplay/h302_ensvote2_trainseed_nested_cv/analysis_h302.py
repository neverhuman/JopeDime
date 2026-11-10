"""H302 (h302_ensvote2_trainseed_nested_cv): statistical evaluator, H300/H294 gate, and seed-spread red-team."""

from __future__ import annotations

import hashlib
import math
from math import comb
from typing import Any

import numpy as np

OUTER_SPLIT_SALT = b"H302-outer-v1\x00"
ROOT_TRAINING_SEEDS = (30001, 30002, 30003)
EXPECTED_MEMBER_SEEDS = {
    30001: (685781424, 1430007429, 1330439319),
    30002: (1869595753, 786733481, 1272701215),
    30003: (1510579523, 89540748, 1415857569),
}
BOOT_SEED = 302150
VALID_NULL_SEEDS = (302101, 302102, 302103)
NOLABEL_NULL_SEEDS = (302201, 302202, 302203)
DRAWS = 10000
ALPHA = 0.05
TIE = 1e-4
HARM_LOSS = 0.01
HARM_MAX_SHARE = 0.35
H300_MEDIAN_MIN = 0.001
H294_MEDIAN_MIN = 0.010

# Published H294 ENS_VOTE2 bootstrap-seed (294150, 294151, 294152) reference for seed-noise red-team comparison
H294_PUBLISHED_REF = {
    "wave": "H294",
    "cohort": "fresh_cohort_v5_H294 (358 EVEN families)",
    "seed_type": "bootstrap_seeds_on_single_deterministic_policy",
    "seeds": [294150, 294151, 294152],
    "per_seed_family_mean_gain": [0.04633032459283523, 0.04633032459283523, 0.04633032459283523],
    "per_seed_lcb_one_sided_95": [0.024272312712920317, 0.024316613057566595, 0.024649034372398116],
    "changed_picks_by_seed": [89, 89, 89],
    "elo_by_seed": [1635.2351664675277, 1635.2351664675277, 1635.2351664675277],
}


def outer_fold_of_family(family_id: str) -> int:
    """Deterministic outer fold assignment: SHA256('H302-outer-v1' NUL family) last byte mod 4."""
    digest = hashlib.sha256(OUTER_SPLIT_SALT + family_id.encode("utf-8")).digest()
    return int(digest[-1] % 4)


def bt_two(wins_ab: float, wins_ba: float, prior: float = 0.01, iters: int = 400) -> float:
    """Two-contender MM Bradley-Terry (incumbent slot 0 anchored at 1500); returns challenger Elo."""
    s_a, s_b = 1.0, 1.0
    games = wins_ab + wins_ba
    for _ in range(iters):
        s_b = (wins_ba + prior) / ((games + 2.0 * prior) / (s_a + s_b))
        s_b = float(np.clip(s_b / s_a, 1e-6, 1e6))
        s_a = 1.0
    return float(1500.0 + 400.0 * np.log10(s_b))


def sign_test_one_sided(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(int(wins), n + 1)) / (2**n))


def invalid_noninferiority(inv: np.ndarray, n_families: int) -> list[dict[str, Any]]:
    """inv: (3, contexts, 2) booleans [slot0, arm]. Per seed: excess, tolerance, McNemar discordants and p."""
    tol = max(2, math.ceil(0.01 * int(n_families)))
    out = []
    for v in np.asarray(inv, dtype=bool):
        s0, arm = v[:, 0], v[:, 1]
        b = int((arm & ~s0).sum())
        c = int((s0 & ~arm).sum())
        p = sign_test_one_sided(b, c)
        excess = int(arm.sum()) - int(s0.sum())
        out.append(
            {
                "slot0_invalid": int(s0.sum()),
                "arm_invalid": int(arm.sum()),
                "excess": excess,
                "tolerance": tol,
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_p": p,
                "ok": bool(excess <= tol and p >= ALPHA),
            }
        )
    return out


def summarize_three_seeds(
    u: np.ndarray,
    inv: np.ndarray,
    families: list[str],
    picks: list[list[int]] | np.ndarray,
    boot_seed: int = BOOT_SEED,
    n_draws: int = DRAWS,
) -> dict[str, Any]:
    """Summarize 3 training seeds with shared whole-family bootstrap (boot_seed = 302150).

    u: (3, n_ctx, 2) = [slot0_utility, arm_utility]
    inv: (3, n_ctx, 2) = [slot0_invalid, arm_invalid]
    """
    u_arr = np.asarray(u, dtype=np.float64)
    inv_arr = np.asarray(inv, dtype=bool)
    fams = sorted(set(families))
    n_fam = len(fams)
    f_arr = np.asarray(families)

    # d[s, f]: seed s's mean utility difference vs slot 0 within family f
    D = np.stack(
        [
            np.array([u_arr[s, f_arr == x, 1].mean() - u_arr[s, f_arr == x, 0].mean() for x in fams], dtype=np.float64)
            for s in range(3)
        ],
        axis=0,
    )

    # Shared family bootstrap indices across all 3 training seeds (H300_PREREG Section "Estimand, uncertainty, and decision")
    rng = np.random.default_rng(int(boot_seed))
    boot_idx = rng.integers(0, n_fam, size=(int(n_draws), n_fam))

    # B_mean: (3, n_draws), B_median: (3, n_draws)
    B_mean = np.stack([D[s][boot_idx].mean(axis=1) for s in range(3)], axis=0)
    B_med = np.stack([np.median(D[s][boot_idx], axis=1) for s in range(3)], axis=0)

    wtl = [
        [int((d > TIE).sum()), int((np.abs(d) <= TIE).sum()), int((d < -TIE).sum())]
        for d in D
    ]
    gains = D.mean(axis=1)
    med_gains = np.median(D, axis=1)

    # Family-level MM Bradley-Terry Elo and shared-bootstrap Elo CI95
    win_ind = (D > TIE).astype(np.float64)
    loss_ind = (D < -TIE).astype(np.float64)
    elo_point_by_seed = [float(bt_two(float(lo), float(w))) for w, _, lo in wtl]
    B_elo = np.zeros((3, int(n_draws)), dtype=np.float64)
    for s in range(3):
        w_boot = win_ind[s][boot_idx].sum(axis=1)
        l_boot = loss_ind[s][boot_idx].sum(axis=1)
        B_elo[s] = 1500.0 + 400.0 * np.log10((w_boot + 0.01) / (l_boot + 0.01))
    elo_ci95_by_seed = [np.quantile(B_elo[s], [0.025, 0.975]).tolist() for s in range(3)]
    pooled_elo_boot = np.median(B_elo, axis=0)
    pooled_elo_ci95 = np.quantile(pooled_elo_boot, [0.025, 0.975]).tolist()

    pooled_mean_boot = np.median(B_mean, axis=0)
    pooled_med_boot = np.median(B_med, axis=0)

    out = {
        "families": n_fam,
        "contexts": int(u_arr.shape[1]),
        "shared_boot_seed": int(boot_seed),
        "boot_draws": int(n_draws),
        "per_seed_family_mean_gain": gains.tolist(),
        "per_seed_ci95": [np.quantile(B_mean[s], [0.025, 0.975]).tolist() for s in range(3)],
        "per_seed_lcb_one_sided": {
            "0.025": [float(np.quantile(B_mean[s], 0.025)) for s in range(3)],
            "0.05": [float(np.quantile(B_mean[s], 0.05)) for s in range(3)],
        },
        "per_seed_p_boot": [float((B_mean[s] <= 0.0).mean()) for s in range(3)],
        "median_gain_across_seeds": float(np.median(gains)),
        "pooled_ci95": np.quantile(pooled_mean_boot, [0.025, 0.975]).tolist(),
        "pooled_lcb_one_sided_95": float(np.quantile(pooled_mean_boot, 0.05)),
        "secondary_family_median_gain": {
            "per_seed_family_median_gain": med_gains.tolist(),
            "per_seed_family_median_ci95": [np.quantile(B_med[s], [0.025, 0.975]).tolist() for s in range(3)],
            "median_of_family_median_gain_across_seeds": float(np.median(med_gains)),
            "pooled_family_median_ci95": np.quantile(pooled_med_boot, [0.025, 0.975]).tolist(),
        },
        "wtl_by_seed": wtl,
        "sign_test_p_by_seed": [sign_test_one_sided(w, lo) for w, _, lo in wtl],
        "elo_by_seed": elo_point_by_seed,
        "elo_ci95_by_seed": elo_ci95_by_seed,
        "median_elo_across_seeds": float(np.median(elo_point_by_seed)),
        "pooled_elo_ci95": pooled_elo_ci95,
        "harmed_share_by_seed": [float((d < -HARM_LOSS).mean()) for d in D],
        "harmed_count_by_seed": [int((d < -HARM_LOSS).sum()) for d in D],
        "harmed_families_by_seed": [[x for x, v in zip(fams, d) if v < -HARM_LOSS] for d in D],
        "invalid_by_seed": [[int(v[:, 0].sum()), int(v[:, 1].sum())] for v in inv_arr],
        "changed_picks_by_seed": [int(sum(int(j) != 0 for j in p)) for p in picks],
        "slot0_context_mean": float(u_arr[0, :, 0].mean()),
        "slot0_family_equal_mean": float(
            np.mean([u_arr[0, f_arr == x, 0].mean() for x in fams])
        ),
        "arm_context_mean_by_seed": [float(u_arr[s, :, 1].mean()) for s in range(3)],
        "arm_family_equal_mean_by_seed": [
            float(np.mean([u_arr[s, f_arr == x, 1].mean() for x in fams])) for s in range(3)
        ],
        "_B": B_mean,
        "_D": D,
    }
    out["arm_p"] = float(
        max(max(a, b) for a, b in zip(out["per_seed_p_boot"], out["sign_test_p_by_seed"]))
    )
    return out


def evaluate_h302_criteria(summary: dict[str, Any], inv: np.ndarray, n_families: int) -> dict[str, Any]:
    """Evaluate H300/H302 primary conditions 1-4 and descriptive H294 conditions."""
    lcb95 = summary["per_seed_lcb_one_sided"]["0.05"]
    ni = invalid_noninferiority(inv, n_families)

    cond1_median_gt_0001 = bool(summary["median_gain_across_seeds"] > H300_MEDIAN_MIN)
    cond2_lcb95_pos_3of3 = bool(all(x > 0.0 for x in lcb95))
    cond3_harmed_le_35pct = bool(all(h <= HARM_MAX_SHARE for h in summary["harmed_share_by_seed"]))
    cond4_invalid_ok = bool(all(x["ok"] for x in ni))

    passes_h302_primary = bool(
        cond1_median_gt_0001 and cond2_lcb95_pos_3of3 and cond3_harmed_le_35pct and cond4_invalid_ok
    )

    # Descriptive H294 stricter threshold (+0.010) + sign test < 0.05 on 3/3 seeds
    desc_median_gt_0010 = bool(summary["median_gain_across_seeds"] > H294_MEDIAN_MIN)
    desc_sign_test_3of3 = bool(all(p < ALPHA for p in summary["sign_test_p_by_seed"]))
    passes_h294_stricter = bool(
        passes_h302_primary and desc_median_gt_0010 and desc_sign_test_3of3
    )

    return {
        "condition_1_median_gain_gt_0_001": cond1_median_gt_0001,
        "condition_2_lcb95_gt_0_on_3of3_seeds": cond2_lcb95_pos_3of3,
        "condition_3_harmed_share_le_35pct_each_seed": cond3_harmed_le_35pct,
        "condition_4_invalid_noninferior_each_seed": cond4_invalid_ok,
        "passes_h302_primary": passes_h302_primary,
        "verdict": (
            "INNERCV_CONFIRMED_TRAINING_SEEDS"
            if passes_h302_primary
            else "NOT_CONFIRMED_TRAINING_SEEDS"
        ),
        "lcb_one_sided_95": lcb95,
        "invalid_detail": ni,
        "strict_invalid_not_above_slot0_descriptive": bool(all(x["excess"] <= 0 for x in ni)),
        "descriptive_h294_rule_check": {
            "median_gain_gt_0_010": desc_median_gt_0010,
            "sign_test_p_lt_0_05_on_3of3": desc_sign_test_3of3,
            "passes_all_h294_criteria": passes_h294_stricter,
        },
    }


def analyze_policy_pick_agreement_and_seed_spread(
    picks_by_seed: list[list[int]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Analyze whether the 3 fitted ENS_VOTE2 policies make identical picks and compare training-seed
    spread against H294's bootstrap-seed spread."""
    p = np.asarray(picks_by_seed, dtype=np.int32)
    assert p.shape[0] == 3
    n_ctx = int(p.shape[1])

    all_three_identical = bool(np.array_equal(p[0], p[1]) and np.array_equal(p[1], p[2]))
    all_three_agree_mask = (p[0] == p[1]) & (p[1] == p[2])
    n_all_three_agree = int(np.sum(all_three_agree_mask))

    pairwise = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        si, sj = ROOT_TRAINING_SEEDS[i], ROOT_TRAINING_SEEDS[j]
        eq_all = int(np.sum(p[i] == p[j]))
        sw_i = p[i] != 0
        sw_j = p[j] != 0
        union_sw = int(np.sum(sw_i | sw_j))
        inter_same_slot = int(np.sum(sw_i & sw_j & (p[i] == p[j])))
        inter_both_sw = int(np.sum(sw_i & sw_j))
        pairwise[f"{si}_vs_{sj}"] = {
            "identical_picks": bool(eq_all == n_ctx),
            "agree_contexts": eq_all,
            "agree_fraction": float(eq_all / n_ctx),
            "differ_contexts": int(n_ctx - eq_all),
            "switched_seed_a": int(np.sum(sw_i)),
            "switched_seed_b": int(np.sum(sw_j)),
            "both_switched_contexts": inter_both_sw,
            "both_switched_same_slot": inter_same_slot,
            "switched_same_slot_jaccard": float(inter_same_slot / union_sw) if union_sw > 0 else 1.0,
        }

    any_switched = (p[0] != 0) | (p[1] != 0) | (p[2] != 0)
    all_switched_same = (p[0] != 0) & (p[0] == p[1]) & (p[1] == p[2])

    gains = np.asarray(summary["per_seed_family_mean_gain"], dtype=np.float64)
    lcbs = np.asarray(summary["per_seed_lcb_one_sided"]["0.05"], dtype=np.float64)
    elos = np.asarray(summary["elo_by_seed"], dtype=np.float64)
    chgs = np.asarray(summary["changed_picks_by_seed"], dtype=np.float64)

    h294_gains = np.asarray(H294_PUBLISHED_REF["per_seed_family_mean_gain"], dtype=np.float64)
    h294_lcbs = np.asarray(H294_PUBLISHED_REF["per_seed_lcb_one_sided_95"], dtype=np.float64)

    return {
        "three_policies_make_identical_picks": all_three_identical,
        "n_contexts_all_3_seeds_agree": n_all_three_agree,
        "fraction_contexts_all_3_seeds_agree": float(n_all_three_agree / n_ctx),
        "n_contexts_any_seed_switches": int(np.sum(any_switched)),
        "n_contexts_all_3_seeds_switch_to_same_slot": int(np.sum(all_switched_same)),
        "picks_sha256_by_seed": {
            str(ROOT_TRAINING_SEEDS[s]): hashlib.sha256(
                np.asarray(p[s], dtype=np.int32).tobytes()
            ).hexdigest()
            for s in range(3)
        },
        "pairwise_seed_agreement": pairwise,
        "seed_noise_red_team_vs_h294": {
            "h302_training_seeds": {
                "seeds": list(ROOT_TRAINING_SEEDS),
                "gain_by_seed": gains.tolist(),
                "gain_range_max_minus_min": float(np.max(gains) - np.min(gains)),
                "gain_std_ddof1": float(np.std(gains, ddof=1)),
                "lcb95_by_seed": lcbs.tolist(),
                "lcb95_range_max_minus_min": float(np.max(lcbs) - np.min(lcbs)),
                "lcb95_std_ddof1": float(np.std(lcbs, ddof=1)),
                "changed_picks_by_seed": [int(x) for x in chgs],
                "changed_picks_range": int(np.max(chgs) - np.min(chgs)),
                "elo_by_seed": elos.tolist(),
                "elo_range_max_minus_min": float(np.max(elos) - np.min(elos)),
            },
            "h294_bootstrap_seeds_reference": {
                "seeds": H294_PUBLISHED_REF["seeds"],
                "gain_by_seed": h294_gains.tolist(),
                "gain_range_max_minus_min": float(np.max(h294_gains) - np.min(h294_gains)),
                "gain_std_ddof1": float(np.std(h294_gains, ddof=1)),
                "lcb95_by_seed": h294_lcbs.tolist(),
                "lcb95_range_max_minus_min": float(np.max(h294_lcbs) - np.min(h294_lcbs)),
                "lcb95_std_ddof1": float(np.std(h294_lcbs, ddof=1)),
                "changed_picks_by_seed": H294_PUBLISHED_REF["changed_picks_by_seed"],
                "changed_picks_range": 0,
            },
        },
    }
