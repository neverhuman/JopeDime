"""H284 statistical analysis and primary fresh-pool decision rule for frozen BAG_SWITCH (q = 6)
and BAG_SWITCH_NESTED (q* in {5, 6, 7}) vs slot 0 and vs frozen RERANK8_VETO."""

import numpy as np

ALL_ARMS = (
    "slot0",
    "RERANK8_VETO",
    "H278_RERANK8_VETO",
    "BAG_SWITCH",
    "BAG_SWITCH_NESTED",
)
EVAL_ARMS = (
    "RERANK8_VETO",
    "H278_RERANK8_VETO",
    "BAG_SWITCH",
    "BAG_SWITCH_NESTED",
)
DRAWS = 10_000
BOOT_SEED = 20261103
TIE_EPS = 1e-4
HARM_001 = 0.01
HARM_005 = 0.05
NUM_EPS = 1e-12
MIN_FRESH_GAIN = 0.010
MAX_HARM_RATIO_VS_VETO = 0.75


def family_matrix(values: np.ndarray, fams: list[str]) -> tuple[list[str], np.ndarray]:
    """Equal-weight per-family mean across contexts for each arm column."""
    ids = sorted(set(fams))
    arr = np.asarray(fams)
    mat = np.stack([values[arr == f].mean(axis=0) for f in ids])
    return ids, mat


def wtl_counts(d: np.ndarray, tie_eps: float = TIE_EPS) -> dict[str, int]:
    """Family win/tie/loss counts where ties |d| <= 1e-4 are not wins."""
    wins = int((d > tie_eps).sum())
    losses = int((d < -tie_eps).sum())
    ties = int(len(d) - wins - losses)
    return {"W": wins, "T": ties, "L": losses}


def elo_from_family_matrix(fm: np.ndarray, anchor_col: int = 0) -> dict[str, float]:
    """Pairwise family win-rate Elo with tie = 0.5 for |d| <= 1e-4, anchored at slot0 = 1500."""
    n_arms = fm.shape[1]
    s = np.zeros(n_arms, dtype=np.float64)
    for a in range(n_arms):
        vals = []
        for b in range(n_arms):
            if a == b:
                continue
            d = fm[:, a] - fm[:, b]
            w = float((d > TIE_EPS).sum())
            t = float((np.abs(d) <= TIE_EPS).sum())
            p = np.clip((w + 0.5 * t) / len(d), 1e-6, 1.0 - 1e-6)
            vals.append(400.0 * np.log10(p / (1.0 - p)))
        s[a] = float(np.mean(vals))
    s = s - s[anchor_col] + 1500.0
    return {ALL_ARMS[i]: float(s[i]) for i in range(n_arms)}


def summarize_pool_panel(
    panel: np.ndarray,
    fams: list[str],
    invalid_by_seed: dict[str, list[int]],
    changed_by_seed: dict[str, list[int]],
    seed_base: int = BOOT_SEED,
) -> dict:
    """Summarize a (n_seeds, n_ctx, n_arms) utility panel on a pool or union of pools."""
    n_seeds = panel.shape[0]
    fam_ids, fm0 = family_matrix(panel[0], fams)
    n_fams = len(fam_ids)
    fm = np.stack([family_matrix(panel[s], fams)[1] for s in range(n_seeds)], axis=0)  # (S, F, A)

    rng_med = np.random.default_rng(seed_base)
    ix_med = rng_med.integers(0, n_fams, size=(DRAWS, n_fams))

    contrasts = {}
    references = {
        "RERANK8_VETO": ("slot0",),
        "H278_RERANK8_VETO": ("slot0",),
        "BAG_SWITCH": ("slot0", "RERANK8_VETO", "H278_RERANK8_VETO"),
        "BAG_SWITCH_NESTED": ("slot0", "RERANK8_VETO", "H278_RERANK8_VETO", "BAG_SWITCH"),
    }

    for arm, refs in references.items():
        col_a = ALL_ARMS.index(arm)
        for ref in refs:
            col_r = ALL_ARMS.index(ref)
            d = fm[:, :, col_a] - fm[:, :, col_r]  # (S, F)
            per_seed_gain = [float(d[s].mean()) for s in range(n_seeds)]
            per_seed_ci95 = []
            for s in range(n_seeds):
                rng_s = np.random.default_rng(seed_base + 1000 + s * 37 + col_a * 7 + col_r)
                ix_s = rng_s.integers(0, n_fams, size=(DRAWS, n_fams))
                boot_s = d[s][ix_s].mean(axis=1)
                per_seed_ci95.append([float(x) for x in np.quantile(boot_s, [0.025, 0.975])])

            boot_med = np.median(d[:, ix_med].mean(axis=2), axis=0)
            median_ci95 = [float(x) for x in np.quantile(boot_med, [0.025, 0.975])]
            wtl_by_seed = [wtl_counts(d[s]) for s in range(n_seeds)]
            harmed_001_by_seed = [int((d[s] < -HARM_001 - NUM_EPS).sum()) for s in range(n_seeds)]
            harmed_005_by_seed = [int((d[s] < -HARM_005 - NUM_EPS).sum()) for s in range(n_seeds)]
            harmed_001_fams_by_seed = [
                [fam_ids[i] for i in range(n_fams) if d[s, i] < -HARM_001 - NUM_EPS]
                for s in range(n_seeds)
            ]

            contrasts[f"{arm}_minus_{ref}"] = {
                "per_seed_gain": per_seed_gain,
                "per_seed_ci95": per_seed_ci95,
                "lcb95_positive_seeds": int(sum(ci[0] > 0.0 for ci in per_seed_ci95)),
                "positive_seeds": int(sum(g > 0.0 for g in per_seed_gain)),
                "median_gain": float(np.median(per_seed_gain)),
                "mean_gain": float(np.mean(per_seed_gain)),
                "ci95": median_ci95,
                "wtl_by_seed": wtl_by_seed,
                "wtl_median": {
                    k: float(np.median([w[k] for w in wtl_by_seed])) for k in ("W", "T", "L")
                },
                "harmed_001_by_seed": harmed_001_by_seed,
                "harmed_001_median": float(np.median(harmed_001_by_seed)),
                "harmed_005_by_seed": harmed_005_by_seed,
                "harmed_005_median": float(np.median(harmed_005_by_seed)),
                "harmed_001_families_by_seed": harmed_001_fams_by_seed,
            }

    elo_by_seed = [elo_from_family_matrix(fm[s]) for s in range(n_seeds)]
    median_elo = {
        a: float(np.median([e[a] for e in elo_by_seed])) for a in ALL_ARMS
    }

    return {
        "n_contexts": int(panel.shape[1]),
        "n_families": int(n_fams),
        "family_ids": fam_ids,
        "slot0_family_mean": float(fm[:, :, 0].mean()),
        "slot0_context_mean": float(panel[:, :, 0].mean()),
        "contrasts": contrasts,
        "elo_by_seed": elo_by_seed,
        "median_elo": median_elo,
        "invalid_by_seed": invalid_by_seed,
        "invalid_median": {a: float(np.median(v)) for a, v in invalid_by_seed.items()},
        "invalid_total": {a: int(sum(v)) for a, v in invalid_by_seed.items()},
        "changed_by_seed": changed_by_seed,
        "changed_median": {a: float(np.median(v)) for a, v in changed_by_seed.items()},
    }


def evaluate_arm_fresh_gates(summary: dict, arm: str) -> dict:
    """Evaluate the 4 preregistered fresh-pool primary gates for arm in {BAG_SWITCH, BAG_SWITCH_NESTED}."""
    c_vs_0 = summary["contrasts"][f"{arm}_minus_slot0"]
    c_veto_279 = summary["contrasts"]["RERANK8_VETO_minus_slot0"]
    c_veto_278 = summary["contrasts"]["H278_RERANK8_VETO_minus_slot0"]

    lcb_pos_seeds = int(c_vs_0["lcb95_positive_seeds"])
    med_gain = float(c_vs_0["median_gain"])
    h_arm = float(c_vs_0["harmed_001_median"])
    h_veto_279 = float(c_veto_279["harmed_001_median"])
    h_veto_278 = float(c_veto_278["harmed_001_median"])

    inv_arm = summary["invalid_by_seed"][arm]
    inv_0 = summary["invalid_by_seed"]["slot0"]

    g1_lcb = lcb_pos_seeds == 3
    g2_gain = med_gain > MIN_FRESH_GAIN
    g3_harm_vs_paired_veto = h_arm <= MAX_HARM_RATIO_VS_VETO * h_veto_279 + NUM_EPS
    g3_harm_vs_h278_veto = h_arm <= MAX_HARM_RATIO_VS_VETO * h_veto_278 + NUM_EPS
    g4_inv = all(int(ia) <= int(i0) for ia, i0 in zip(inv_arm, inv_0))

    return {
        "arm": arm,
        "lcb95_positive_seeds_3_of_3": g1_lcb,
        "lcb95_positive_seeds": lcb_pos_seeds,
        "per_seed_ci95": c_vs_0["per_seed_ci95"],
        "median_gain_gt_0p010": g2_gain,
        "median_gain": med_gain,
        "ci95": c_vs_0["ci95"],
        "harm_le_0p75_paired_veto": g3_harm_vs_paired_veto,
        "harm_le_0p75_h278_veto": g3_harm_vs_h278_veto,
        "harmed_001_median": h_arm,
        "paired_veto_harmed_001_median": h_veto_279,
        "h278_veto_harmed_001_median": h_veto_278,
        "harm_ratio_vs_paired_veto": float(h_arm / h_veto_279) if h_veto_279 > 0 else (0.0 if h_arm == 0 else float("inf")),
        "harm_ratio_vs_h278_veto": float(h_arm / h_veto_278) if h_veto_278 > 0 else (0.0 if h_arm == 0 else float("inf")),
        "invalid_not_above_slot0": g4_inv,
        "invalid_by_seed": inv_arm,
        "slot0_invalid_by_seed": inv_0,
        "pass_all_vs_paired_veto": bool(g1_lcb and g2_gain and g3_harm_vs_paired_veto and g4_inv),
        "pass_all_vs_h278_veto": bool(g1_lcb and g2_gain and g3_harm_vs_h278_veto and g4_inv),
    }


def decide_fresh(union_summary: dict | None) -> dict:
    """Compute overall H284 verdict from the union of available fresh pools."""
    if union_summary is None or union_summary.get("n_contexts", 0) == 0:
        return {
            "verdict": "DESCRIPTIVE_ONLY",
            "reason": "No fresh pools (fresh_pools_H283.json / fresh_pools_H285.json) appeared before cutoff; reporting r782/r785 descriptive evaluation only.",
        }
    bag6 = evaluate_arm_fresh_gates(union_summary, "BAG_SWITCH")
    nested = evaluate_arm_fresh_gates(union_summary, "BAG_SWITCH_NESTED")
    adv = bool(
        bag6["pass_all_vs_paired_veto"]
        or bag6["pass_all_vs_h278_veto"]
        or nested["pass_all_vs_paired_veto"]
        or nested["pass_all_vs_h278_veto"]
    )
    return {
        "verdict": "FRESH_ADVANCE" if adv else "NOT_ADVANCED_FRESH",
        "BAG_SWITCH": bag6,
        "BAG_SWITCH_NESTED": nested,
    }
