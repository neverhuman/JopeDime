"""H287 statistical analysis and inner-CV export advancement decision rule for BAG_SWITCH seed ensembles
(ENS_MEAN, ENS_MEAN_PROPOSER, ENS_VOTE2, ENS_VOTE3) and CLIP vs slot 0 and vs single-seed BAG_SWITCH."""

import numpy as np

ALL_ARMS = (
    "slot0",
    "RERANK8_VETO",
    "BAG_SWITCH",
    "ENS_MEAN",
    "ENS_MEAN_PROPOSER",
    "ENS_VOTE2",
    "ENS_VOTE3",
    "CLIP",
    "CLIP_ENS_VOTE2",
)
ENS_PRIMARY_ARMS = (
    "ENS_MEAN",
    "ENS_VOTE2",
    "ENS_VOTE3",
)
DRAWS = 10_000
BOOT_SEED = 287150
TIE_EPS = 1e-4
HARM_001 = 0.01
HARM_005 = 0.05
NUM_EPS = 1e-12


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


def elo_from_family_matrix(
    fm: np.ndarray, arm_names: tuple[str, ...] = ALL_ARMS, anchor_col: int = 0
) -> dict[str, float]:
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
    return {arm_names[i]: float(s[i]) for i in range(n_arms)}


def summarize_panel(
    panel: np.ndarray,
    fams: list[str],
    invalid_by_seed: dict[str, list[int]],
    changed_by_seed: dict[str, list[int]],
    arm_names: tuple[str, ...] = ALL_ARMS,
    seed_base: int = BOOT_SEED,
    label: str = "inner_cv_dl_train",
    reused: bool = False,
) -> dict:
    """Summarize a (n_seeds, n_ctx, n_arms) utility panel on DL-Train inner CV or reused DL-Val91."""
    n_seeds = panel.shape[0]
    fam_ids, _ = family_matrix(panel[0], fams)
    n_fams = len(fam_ids)
    fm = np.stack([family_matrix(panel[s], fams)[1] for s in range(n_seeds)], axis=0)  # (S, F, A)

    rng_med = np.random.default_rng(seed_base)
    ix_med = rng_med.integers(0, n_fams, size=(DRAWS, n_fams))

    contrasts = {}
    for col_a, arm in enumerate(arm_names):
        if arm == "slot0":
            continue
        refs = ["slot0"]
        if arm != "BAG_SWITCH" and "BAG_SWITCH" in arm_names:
            refs.append("BAG_SWITCH")
        for ref in refs:
            col_r = arm_names.index(ref)
            d = fm[:, :, col_a] - fm[:, :, col_r]  # (S, F)
            per_seed_gain = [float(d[s].mean()) for s in range(n_seeds)]
            per_seed_fam_median = [float(np.median(d[s])) for s in range(n_seeds)]
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
                "per_seed_family_median_gain": per_seed_fam_median,
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
                "harmed_001_mean": float(np.mean(harmed_001_by_seed)),
                "harmed_005_by_seed": harmed_005_by_seed,
                "harmed_005_median": float(np.median(harmed_005_by_seed)),
                "harmed_001_families_by_seed": harmed_001_fams_by_seed,
            }

    elo_by_seed = [elo_from_family_matrix(fm[s], arm_names=arm_names) for s in range(n_seeds)]
    median_elo = {
        a: float(np.median([e[a] for e in elo_by_seed])) for a in arm_names
    }

    return {
        "split_label": label,
        "reused": bool(reused),
        "role": "descriptive_only_reused" if reused else "primary_inner_cv",
        "n_seeds": int(n_seeds),
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


def evaluate_ens_arm_gate(inner_cv_summary: dict, arm: str) -> dict:
    """Evaluate H287 Step 5 gate for an arm on DL-Train inner CV:
    inner-CV median gain >= single-seed BAG_SWITCH median AND per-seed LCB95 > 0 on 3/3 AND harm <= single-seed."""
    c_arm = inner_cv_summary["contrasts"][f"{arm}_minus_slot0"]
    c_bag = inner_cv_summary["contrasts"]["BAG_SWITCH_minus_slot0"]

    med_gain_arm = float(c_arm["median_gain"])
    med_gain_bag = float(c_bag["median_gain"])
    lcb_pos_arm = int(c_arm["lcb95_positive_seeds"])
    harm_med_arm = float(c_arm["harmed_001_median"])
    harm_med_bag = float(c_bag["harmed_001_median"])
    harm_mean_arm = float(c_arm["harmed_001_mean"])
    harm_mean_bag = float(c_bag["harmed_001_mean"])

    inv_arm = inner_cv_summary["invalid_by_seed"][arm]
    inv_0 = inner_cv_summary["invalid_by_seed"]["slot0"]

    g_gain = med_gain_arm >= med_gain_bag - NUM_EPS
    g_lcb = lcb_pos_arm == 3
    g_harm_med = harm_med_arm <= harm_med_bag + NUM_EPS
    g_harm_mean = harm_mean_arm <= harm_mean_bag + NUM_EPS
    g_inv = all(int(ia) <= int(i0) for ia, i0 in zip(inv_arm, inv_0))

    return {
        "arm": arm,
        "median_gain": med_gain_arm,
        "bag_switch_median_gain": med_gain_bag,
        "gain_ge_bag_switch_median": bool(g_gain),
        "lcb95_positive_seeds": lcb_pos_arm,
        "lcb95_positive_3_of_3": bool(g_lcb),
        "per_seed_ci95": c_arm["per_seed_ci95"],
        "harmed_001_median": harm_med_arm,
        "bag_switch_harmed_001_median": harm_med_bag,
        "harm_median_le_bag_switch": bool(g_harm_med),
        "harmed_001_mean": harm_mean_arm,
        "bag_switch_harmed_001_mean": harm_mean_bag,
        "harm_mean_le_bag_switch": bool(g_harm_mean),
        "invalid_by_seed": inv_arm,
        "slot0_invalid_by_seed": inv_0,
        "invalid_le_slot0": bool(g_inv),
        "pass_exported_advance": bool(g_gain and g_lcb and g_harm_med),
    }


def decide_h287(inner_cv_summary: dict) -> dict:
    """Compute overall H287 verdict on DL-Train inner CV."""
    by_arm = {
        arm: evaluate_ens_arm_gate(inner_cv_summary, arm)
        for arm in ("ENS_MEAN", "ENS_MEAN_PROPOSER", "ENS_VOTE2", "ENS_VOTE3", "CLIP", "CLIP_ENS_VOTE2")
        if f"{arm}_minus_slot0" in inner_cv_summary["contrasts"]
    }
    adv_ens = [
        arm for arm in ENS_PRIMARY_ARMS if by_arm.get(arm, {}).get("pass_exported_advance", False)
    ]
    return {
        "verdict": "EXPORTED_ADVANCE" if adv_ens else "EXPORTED_NO_ADVANCE",
        "advancing_ens_arms": adv_ens,
        "by_arm": by_arm,
    }
