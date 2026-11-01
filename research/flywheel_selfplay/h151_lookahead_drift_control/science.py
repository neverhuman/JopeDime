"""Prospective H151 statistics: Lookahead vs drift-matched AdamW step-size controls (reused from merged H146)."""

import hashlib
import random
import numpy as np

SEEDS = (151101, 151102, 151103)
FOLD_SEED = 151140
# arm -> (learning_rate, lookahead_k). Every arm uses decoupled weight_decay=1e-4 and 40 fast updates.
ARMS = {
    "CTRL": (1e-4, None),
    "CTRL_HALF": (5e-5, None),
    "CTRL_QTR": (2.5e-5, None),
    "LA": (1e-4, 5),
}
CTRL_CANDIDATES = ("CTRL_HALF", "CTRL_QTR", "CTRL")
WEIGHT_DECAY = 1e-4
LA_ALPHA = 0.5
DRIFT_MATCH_MAX_REL_GAP = 0.15
DRIFT_MATCH_BOUNDS = (0.85, 1.15)
DRAWS = 10000
BOOT_SEED = 151141
HOLM_M = 4  # All four executed arms (CTRL, CTRL_HALF, CTRL_QTR, LA) vs r882.


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
    """H140/H146 paired stream: 20 rounds, consume 8 draws but retain first 2 steps."""
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
    # The family contribution vector comes from the seed defining the median point estimate.
    mid = sorted(range(3), key=lambda i: (gains[i], SEEDS[i]))[1]
    positive = np.maximum(delta[mid], 0)
    share = (
        float(np.sort(positive)[-3:].sum() / positive.sum()) if positive.sum() else 1.0
    )
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
        elo=float(np.median(elo(score))),
        elo_ci95=np.quantile(elo_draws, [0.025, 0.975]).tolist(),
    )


def match_drift_control(drifts):
    """Select the CTRL_* arm whose median adapter_l2_drift is closest to LA's (inclusive 0.85*LA <= CTRL_* <= 1.15*LA)."""
    if set(drifts) != set(ARMS):
        raise ValueError("complete arm drift inventory required")
    values = {a: np.asarray(v, dtype=np.float64) for a, v in drifts.items()}
    if any(
        v.shape != (3,) or not np.isfinite(v).all() or (v <= 0).any()
        for v in values.values()
    ):
        raise ValueError("three finite positive drifts per arm required")
    medians = {a: float(np.median(v)) for a, v in values.items()}
    la_med = medians["LA"]
    abs_gaps = {c: float(abs(medians[c] - la_med)) for c in CTRL_CANDIDATES}
    rel_gaps = {c: float(abs_gaps[c] / la_med) for c in CTRL_CANDIDATES}
    matched = min(
        CTRL_CANDIDATES, key=lambda c: (abs_gaps[c], CTRL_CANDIDATES.index(c))
    )
    matched_med = medians[matched]
    rel_gap = rel_gaps[matched]
    low_bound = DRIFT_MATCH_BOUNDS[0] * la_med
    high_bound = DRIFT_MATCH_BOUNDS[1] * la_med
    valid = bool(low_bound <= matched_med <= high_bound)
    return dict(
        matched_control_arm=matched,
        la_median_drift=la_med,
        matched_control_median_drift=matched_med,
        allowed_drift_band=[float(low_bound), float(high_bound)],
        median_drift_by_arm=medians,
        abs_gap_by_control=abs_gaps,
        rel_gap_by_control=rel_gaps,
        relative_drift_gap=rel_gap,
        max_allowed_relative_gap=DRIFT_MATCH_MAX_REL_GAP,
        drift_match_valid=valid,
        status="DRIFT_MATCH_VALID" if valid else "DRIFT_MATCH_INVALID",
    )


def decide(contrasts, drift_match):
    """Holm step-down vs r882 across all four executed arms; LA per-arm advance also requires drift_match_valid."""
    if set(contrasts) != set(ARMS):
        raise ValueError("complete registered contrasts required for all four arms")
    if not isinstance(drift_match, dict) or "drift_match_valid" not in drift_match:
        raise ValueError("drift_match receipt required")
    dm_valid = bool(drift_match["drift_match_valid"])
    p = {a: contrasts[a]["p_one_sided"] for a in contrasts}
    result = {}
    adjusted = 0.0
    for rank, a in enumerate(sorted(p, key=lambda a: (p[a], a))):
        adjusted = max(adjusted, min(1.0, p[a] * (HOLM_M - rank)))
        v = contrasts[a]
        efficacy_gates = dict(
            margin=v["median_gain"] > 0.001,
            ci=v["ci95"][0] > 0,
            three_seeds=v["seeds_beating"] == 3,
            concentration=v["top3_positive_share"] <= 0.8,
            holm=adjusted < 0.05,
        )
        gates = dict(efficacy_gates)
        if a == "LA":
            gates["drift_match"] = dm_valid
        result[a] = dict(
            holm_adjusted_p=adjusted,
            gates=gates,
            efficacy_only_advance=all(efficacy_gates.values()),
            advance=all(gates.values()),
        )
    return result


def mechanism_verdict(la_vs_matched, drift_match, r882_decision):
    """Return (verdict, mechanism_verdict, r882_verdict) per h151_rd_argon_t2055.md and h151_rd_argon_t2130.md."""
    if set(r882_decision) != set(ARMS):
        raise ValueError("complete four-arm r882 decision required")
    if r882_decision["LA"]["advance"] and not drift_match["drift_match_valid"]:
        raise ValueError("LA cannot advance when drift_match is invalid")
    any_adv = any(d["advance"] for d in r882_decision.values())
    r882_v = "ADVANCE" if any_adv else "NO_ADVANCE"
    if not drift_match["drift_match_valid"]:
        mech_v = "DRIFT_MATCH_INVALID"
    elif la_vs_matched["median_gain"] > 0.0 and la_vs_matched["ci95"][0] > 0.0:
        mech_v = "LOOKAHEAD_SPECIFIC"
    else:
        mech_v = "STEP_SIZE_ONLY"
    top_v = "ADVANCE" if any_adv else mech_v
    return top_v, mech_v, r882_v
