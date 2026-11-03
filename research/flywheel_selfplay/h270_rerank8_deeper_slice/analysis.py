"""H270 deeper-slice re-ranker: the 45-cell repeated nested family CV protocol (3 fold seeds x 5 folds x 3 model
seeds), per-seed medians across fold seeds, family bootstrap inference, Holm m = 2 over PRIMARY and SECONDARY, and
the invalid-pick guard (hw_t1830)."""

import hashlib
import numpy as np

SEEDS = (270101, 270102, 270103)
FOLD_SEEDS = (270140, 270141, 270142)
SECONDARY_FOLD_SEED = 270140
BOOT_SEED = 270150
K_DEEP = 8
ARMS = ("RERANK4", "RERANK8", "RERANK8_SHUF")
REFERENCES = {
    "RERANK4": ("r882",),
    "RERANK8": ("r882", "RERANK4", "RERANK8_SHUF"),
    "RERANK8_SHUF": ("r882",),
}
ALL_ARMS = ("r882",) + ARMS
DRAWS, EPS, HARM = 10000, 1e-12, 1e-4
ADVANCE_MARGIN = 0.002
CLIFF_CONTEXT = 27


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows, fold_seed):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(fold_seed).permutation(205)
    return [sorted(fams[i] for i in g) for g in np.array_split(order, 5)]


def cids_by_fold(rows, fold_seed):
    where = {f: k for k, g in enumerate(folds(rows, fold_seed)) for f in g}
    out = {k: [] for k in range(5)}
    for r in rows:
        out[where[r["family_id"]]].append(r["context_sha256"])
    return out


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(families) or not np.isfinite(values).all():
        raise ValueError("invalid context panel")
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def bt_ratings(wins, anchor=0, prior=0.01):
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2 * prior * (1 - np.eye(n)))
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > HARM, 1.0, np.where(diff < -HARM, 0.0, 0.5)) * (
        1 - np.eye(values.shape[1])
    )
    return family_matrix(outcomes, families)[1]


def holm(pvals):
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(pvals, key=pvals.get)):
        previous = max(previous, min(1.0, pvals[key] * (len(pvals) - rank)))
        result[key] = previous
    return result


def summarize(rows, panel, seed=BOOT_SEED):
    """panel: (3 model seeds, R repetitions, contexts, policies); R = 3 fold seeds (PRIMARY) or 1 (SECONDARY mixture).
    Per model seed the gain is the MEDIAN over repetitions of the equal-family gain; the statistic is the median over
    seeds; the family bootstrap resamples families jointly across all seeds and repetitions."""
    values = np.asarray(panel, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[0] != 3
        or values.shape[2] != len(rows)
        or values.shape[3] != len(ALL_ARMS)
    ):
        raise ValueError("panel shape")
    if not np.isfinite(values).all():
        raise ValueError("incomplete panel")
    families = [r["family_id"] for r in rows]
    fm = np.stack(
        [np.stack([family_matrix(v, families)[1] for v in rep]) for rep in values]
    )  # (3, R, F, P)
    fam_ids = family_matrix(values[0, 0], families)[0]
    n = fm.shape[2]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    contrasts = {}
    for col, arm in enumerate(ARMS, 1):
        for label in REFERENCES[arm]:
            ref = ALL_ARMS.index(label)
            diff = fm[..., col] - fm[..., ref]  # (3, R, F)
            per_rep = diff.mean(axis=2)
            per_seed = np.median(per_rep, axis=1)
            boot = np.median(np.median(diff[:, :, ix].mean(axis=3), axis=1), axis=0)
            mean_diff = diff.mean(
                axis=1
            )  # (3, F): per-seed family delta averaged over repetitions
            contrasts[arm + "_minus_" + label] = dict(
                per_seed_gain=per_seed.tolist(),
                per_seed_per_repetition_gain=per_rep.tolist(),
                median_gain=float(np.median(per_seed)),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                ci_holm_m2=np.quantile(boot, [0.0125, 0.9875]).tolist(),
                positive_seeds=int((per_seed > 0).sum()),
                p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
                wtl_by_seed=[
                    [
                        int((d > HARM).sum()),
                        int((abs(d) <= HARM).sum()),
                        int((d < -HARM).sum()),
                    ]
                    for d in mean_diff
                ],
                harm_families_by_seed=[
                    [f for f, d in zip(fam_ids, row) if d < -HARM] for row in mean_diff
                ],
            )
    pooled = values.mean(
        axis=1
    )  # (3, contexts, policies) averaged over repetitions for Elo
    fw = np.stack([family_wins(v, families) for v in pooled])
    elo = bt_ratings(fw.sum(axis=1))
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2)), axis=0)
    return dict(
        contexts=len(rows),
        families=n,
        repetitions=values.shape[1],
        contrasts=contrasts,
        family_utility_by_seed=fm.mean(axis=(1, 2)).tolist(),
        context_utility_by_seed=values.mean(axis=(1, 2)).tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
    )


def decide(primary, secondary, invalid_rerank8, invalid_control):
    """ADVANCE iff PRIMARY median family gain > +0.002 AND PRIMARY Holm-level LCB > 0 AND 3/3 seeds positive AND
    RERANK8 beats RERANK8_SHUF (PRIMARY median > 0) AND SECONDARY > 0 on 3/3 seeds AND RERANK8's DL-Train -2.0 picks
    <= CONTROL's. Holm m = 2 over {PRIMARY, SECONDARY} p-values for RERANK8 - r882."""
    p = primary["contrasts"]["RERANK8_minus_r882"]
    s = secondary["contrasts"]["RERANK8_minus_r882"]
    adj = holm({"PRIMARY": p["p_one_sided"], "SECONDARY": s["p_one_sided"]})
    gates = dict(
        primary_median_gt_margin=bool(p["median_gain"] > ADVANCE_MARGIN + EPS),
        primary_holm_lcb_gt_0=bool(p["ci_holm_m2"][0] > 0),
        primary_three_seeds=bool(p["positive_seeds"] == 3),
        beats_placebo=bool(
            primary["contrasts"]["RERANK8_minus_RERANK8_SHUF"]["median_gain"] > EPS
        ),
        secondary_three_seeds=bool(s["positive_seeds"] == 3),
        invalid_not_worse=bool(invalid_rerank8 <= invalid_control),
    )
    return dict(
        holm_p=adj,
        gates=gates,
        invalid_train=dict(RERANK8=invalid_rerank8, CONTROL=invalid_control),
        advance=bool(all(gates.values())),
    )
