"""Prospective H154 statistics: trajectory EMA/SWA weight averaging + train-only stat recalibration on r882 continuation."""

import hashlib
import random
import numpy as np

SEEDS = (154101, 154102, 154103)
FOLD_SEED = 154140
BOOT_SEED = 154141
DRAWS = 10000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EMA_DECAYS = {
    "EMA_099": 0.99,
    "EMA_0995": 0.995,
    "EMA_0999": 0.999,
}
SWA_WINDOWS = {
    "SWA_K5": 5,
    "SWA_K10": 10,
    "SWA_K20": 20,
}
CANDIDATE_ARMS = (
    "EMA_099",
    "EMA_0995",
    "EMA_0999",
    "SWA_K5",
    "SWA_K10",
    "SWA_K20",
)
ARMS = {
    "CTRL": {"kind": "raw_final_step40", "decay": None, "swa_k": None, "recal": False},
    "EMA_099": {"kind": "ema", "decay": 0.99, "swa_k": None, "recal": True},
    "EMA_0995": {"kind": "ema", "decay": 0.995, "swa_k": None, "recal": True},
    "EMA_0999": {"kind": "ema", "decay": 0.999, "swa_k": None, "recal": True},
    "SWA_K5": {"kind": "swa", "decay": None, "swa_k": 5, "recal": True},
    "SWA_K10": {"kind": "swa", "decay": None, "swa_k": 10, "recal": True},
    "SWA_K20": {"kind": "swa", "decay": None, "swa_k": 20, "recal": True},
}
HOLM_M = 7  # All seven executed arms (CTRL + 6 trajectory-averaged arms) vs r882.
HOLM_M_CTRL = 6  # Six trajectory-averaged arms vs CTRL.
RECAL_BOUNDS = (0.85, 1.15)
RECAL_SLATES = 8
DOSE_INVALID_BOUNDS = (0.95, 1.05)


def folds(rows):
    if not rows or len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("unique contexts required")
    if any(
        r["dl_split"] != "DL-Train"
        or r.get("has_meta_test") is True
        or r.get("source_split") not in ("train", "inner")
        for r in rows
    ):
        raise ValueError("DL-Train inner rows only")
    fams = sorted({r["family_id"] for r in rows})
    return [
        sorted(x.tolist())
        for x in np.array_split(np.random.default_rng(FOLD_SEED).permutation(fams), 5)
    ]


def batch_plan(fit_ids, orders, seed):
    """H140/H146/H151 paired stream: 20 rounds, consume 8 draws but retain first 2 steps (40 total updates)."""
    if seed not in SEEDS or not fit_ids:
        raise ValueError("registered seed and fit contexts required")
    ids = tuple(sorted(fit_ids))
    sampler = seed * 1_000_003 + 10_007
    plan = []
    for rnd in range(1, 21):
        rng = np.random.RandomState((sampler + rnd * 104729) % 2**32)
        rot = rnd % len(ids)
        rotated = ids[rot:] + ids[:rot]
        for step in range(8):
            stream = int.from_bytes(
                hashlib.sha256(
                    f"flywheel-batch/v1:{sampler}:{rnd}:{step}".encode()
                ).digest()[:8],
                "little",
            )
            cids = random.Random(stream).choices(rotated, k=16)
            indexes = []
            for cid in cids:
                order = orders[cid]
                k = min(48, 192, len(order))
                rest = order[k:] if len(order) > k else order
                indexes.append(
                    np.concatenate(
                        [
                            order[:k],
                            rng.choice(rest, 192 - k, replace=len(rest) < 192 - k),
                        ]
                    )
                )
            if step < 2:
                plan.append((stream, cids, indexes))
    return plan


def family_matrix(rows, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3, len(rows)) or not np.isfinite(values).all():
        raise ValueError("all three complete finite seed panels required")
    families = sorted({r["family_id"] for r in rows})
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context")
    return families, np.stack(
        [
            values[:, [i for i, r in enumerate(rows) if r["family_id"] == f]].mean(
                axis=1
            )
            for f in families
        ],
        axis=1,
    )


def contrast(rows, values, reference):
    fams, delta = family_matrix(rows, np.asarray(values) - np.asarray(reference))
    gains = delta.mean(axis=1)
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(fams), (DRAWS, len(fams)))
    boot = np.median(delta[:, ix].mean(axis=2), axis=0)
    mid = sorted(range(3), key=lambda i: (gains[i], SEEDS[i]))[1]
    positive = np.maximum(delta[mid], 0)
    share = (
        float(np.sort(positive)[-3:].sum() / positive.sum()) if positive.sum() else 1.0
    )
    harmed = int((delta[mid] < 0).sum())
    score = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5))

    def elo(x):
        wins = x.sum(axis=-1)
        n = x.shape[-1]
        return 1500 + 400 * np.log10((wins + 0.01) / (n - wins + 0.01))

    elo_draws = np.median(elo(score[:, ix]), axis=0)
    return dict(
        families=len(fams),
        per_seed_gain=gains.tolist(),
        median_gain=float(np.median(gains)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
        seeds_beating=int((gains > 0).sum()),
        median_seed=SEEDS[mid],
        top3_positive_share=share,
        harmed_families=harmed,
        elo=float(np.median(elo(score))),
        elo_ci95=np.quantile(elo_draws, [0.025, 0.975]).tolist(),
    )


def dose_check(drifts, utilities):
    """Verify every arm has positive finite adapter drift and trajectory-averaged arms are not no-op copies of CTRL."""
    if set(drifts) != set(ARMS) or set(utilities) != set(ARMS):
        raise ValueError("complete registered seven-arm inventory required")
    values = {a: np.asarray(v, dtype=float) for a, v in drifts.items()}
    panels = {a: np.asarray(v, dtype=np.float64) for a, v in utilities.items()}
    if any(
        v.shape != (3,) or not np.isfinite(v).all() or (v <= 0).any()
        for v in values.values()
    ):
        raise ValueError("three finite positive drifts per arm required")
    if any(v.shape != (3, 91) or not np.isfinite(v).all() for v in panels.values()):
        raise ValueError("three complete finite 91-context utility panels required")
    med = {a: float(np.median(v)) for a, v in values.items()}
    ctrl = med["CTRL"]
    result = {}
    for arm in ARMS:
        if arm == "CTRL":
            result[arm] = dict(
                median_adapter_l2_drift=med[arm],
                control_median_adapter_l2_drift=ctrl,
                relative_difference=0.0,
                drift_within_five_percent=True,
                identical_utilities_to_ctrl=True,
                status="DOSE_VALID",
            )
            continue
        relative = abs(med[arm] - ctrl) / ctrl
        within_band = (
            DOSE_INVALID_BOUNDS[0] * ctrl <= med[arm] <= DOSE_INVALID_BOUNDS[1] * ctrl
        )
        identical = bool(np.array_equal(panels[arm], panels["CTRL"]))
        invalid = within_band and identical
        result[arm] = dict(
            median_adapter_l2_drift=med[arm],
            control_median_adapter_l2_drift=ctrl,
            relative_difference=relative,
            drift_within_five_percent=within_band,
            identical_utilities_to_ctrl=identical,
            status="DOSE_INVALID" if invalid else "DOSE_VALID",
        )
    return result


def decide(contrasts_vs_r882, contrasts_vs_ctrl, drifts, utilities):
    """Holm step-down vs r882 across all 7 executed arms (HOLM_M = 7), and vs CTRL across the 6 averaged arms (HOLM_M_CTRL = 6)."""
    if set(contrasts_vs_r882) != set(ARMS):
        raise ValueError(
            "complete registered contrasts vs r882 required for all 7 arms"
        )
    if set(contrasts_vs_ctrl) != set(CANDIDATE_ARMS):
        raise ValueError(
            "complete registered contrasts vs CTRL required for all 6 candidate arms"
        )
    doses = dose_check(drifts, utilities)

    # 1. Unified 7-arm Holm step-down vs r882
    p_r882 = {a: float(contrasts_vs_r882[a]["p_one_sided"]) for a in ARMS}
    vs_r882 = {}
    adjusted = 0.0
    for rank, a in enumerate(sorted(p_r882, key=lambda k: (p_r882[k], k))):
        adjusted = max(adjusted, min(1.0, p_r882[a] * (HOLM_M - rank)))
        v = contrasts_vs_r882[a]
        efficacy_gates = dict(
            margin=v["median_gain"] > 0.001,
            ci=v["ci95"][0] > 0.0,
            three_seeds=v["seeds_beating"] == 3,
            concentration=v["top3_positive_share"] <= 0.8,
            holm=adjusted < 0.05,
        )
        gates = dict(efficacy_gates, dose=doses[a]["status"] == "DOSE_VALID")
        vs_r882[a] = dict(
            holm_adjusted_p=adjusted,
            gates=gates,
            dose=doses[a],
            efficacy_only_advance=all(efficacy_gates.values()),
            advance=all(gates.values()),
        )

    # 2. 6-arm Holm step-down vs CTRL (trajectory-averaging mechanism readout)
    p_ctrl = {a: float(contrasts_vs_ctrl[a]["p_one_sided"]) for a in CANDIDATE_ARMS}
    vs_ctrl = {}
    adj_c = 0.0
    for rank, a in enumerate(sorted(p_ctrl, key=lambda k: (p_ctrl[k], k))):
        adj_c = max(adj_c, min(1.0, p_ctrl[a] * (HOLM_M_CTRL - rank)))
        v = contrasts_vs_ctrl[a]
        efficacy_gates = dict(
            margin=v["median_gain"] > 0.001,
            ci=v["ci95"][0] > 0.0,
            three_seeds=v["seeds_beating"] == 3,
            concentration=v["top3_positive_share"] <= 0.8,
            holm=adj_c < 0.05,
        )
        gates = dict(efficacy_gates, dose=doses[a]["status"] == "DOSE_VALID")
        vs_ctrl[a] = dict(
            holm_adjusted_p=adj_c,
            gates=gates,
            dose=doses[a],
            efficacy_only_advance=all(efficacy_gates.values()),
            advance=all(gates.values()),
        )

    return dict(vs_r882=vs_r882, vs_ctrl=vs_ctrl, doses=doses)


def study_verdict(decision):
    """Return (top_verdict, r882_verdict, trajectory_vs_ctrl_verdict, best_r882_arm)."""
    if (
        not isinstance(decision, dict)
        or "vs_r882" not in decision
        or "vs_ctrl" not in decision
    ):
        raise ValueError("complete decision dictionary required")
    vs_r882 = decision["vs_r882"]
    vs_ctrl = decision["vs_ctrl"]
    if set(vs_r882) != set(ARMS) or set(vs_ctrl) != set(CANDIDATE_ARMS):
        raise ValueError("incomplete arm sets in decision")
    for a, d in vs_r882.items():
        if d["advance"] and not d["gates"].get("dose", False):
            raise ValueError(f"{a} cannot advance with invalid dose")
    adv_r882 = [a for a in ARMS if vs_r882[a]["advance"]]
    adv_ctrl = [a for a in CANDIDATE_ARMS if vs_ctrl[a]["advance"]]
    r882_v = "ADVANCE" if adv_r882 else "NO_ADVANCE"
    ctrl_v = (
        "TRAJECTORY_AVERAGING_BEATS_CTRL"
        if adv_ctrl
        else "NO_TRAJECTORY_AVERAGING_GAIN_OVER_CTRL"
    )
    top_v = "ADVANCE" if adv_r882 else "NO_ADVANCE"
    best_arm = adv_r882[0] if adv_r882 else None
    return top_v, r882_v, ctrl_v, best_arm
