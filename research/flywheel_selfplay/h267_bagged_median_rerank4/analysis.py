"""H267 bagged median-rank re-ranker: repeated nested family CV protocol (fold seed per model seed), family bootstrap
bags, median within-context rank aggregation, out-of-bag strict admission, and model-free inference (hw_t1805)."""

import hashlib
import numpy as np

SEEDS = (267101, 267102, 267103)
FOLD_SEEDS = {
    267101: 267140,
    267102: 267141,
    267103: 267142,
}  # one fresh partition per model seed (paired)
BOOT_SEED = 267150
BAG_SEED_BASE = 267160  # bag b uses bootstrap seed 267160 + b (streamed per model seed and assess fold)
N_BAGS = 25
ARMS = ("RERANK4", "BAG_MED", "BAG_SHUF")
REFERENCES = {
    "RERANK4": ("r882",),
    "BAG_MED": ("r882", "RERANK4", "BAG_SHUF"),
    "BAG_SHUF": ("r882",),
}
ALL_ARMS = ("r882",) + ARMS
DRAWS, INNER_DRAWS, EPS = 10000, 2000, 1e-12
ADVANCE_MARGIN = 0.002
STRICT_MIN_GAIN = 0.001
PLATEAU_BEST = 1.1389572  # r3175
CLIFF_CONTEXT = 27  # H152 gate-cliff DL-Val context (fam-9f67f144...): reported separately, never excluded
HARM = 1e-4


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows, seed):
    """Five family-grouped folds (41 families each) from the model seed's own fold seed."""
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(FOLD_SEEDS[seed]).permutation(205)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def cids_by_fold(rows, seed):
    groups = folds(rows, seed)
    where = {f: k for k, g in enumerate(groups) for f in g}
    out = {k: [] for k in range(5)}
    for r in rows:
        out[where[r["family_id"]]].append(r["context_sha256"])
    return out


def bag_families(fit_fams, seed, assess_fold, b):
    """Family-level bootstrap resample (with replacement) of the sorted FIT families for bag b."""
    fams = sorted(fit_fams)
    rng = np.random.default_rng(stream_seed(BAG_SEED_BASE + b, seed, assess_fold))
    return [fams[i] for i in rng.integers(0, len(fams), len(fams))]


def within_context_ranks(scores):
    """Rank of each top-4 slot inside its context, 0 = highest score; ties go to the smaller slot (r882 order)."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order, np.arange(s.shape[1])[None, :].repeat(len(s), 0), axis=1
    )
    return ranks


def median_rank_pick(rank_stack):
    """rank_stack: (bags, 4) ranks for one context. The slot with the smallest median rank; ties keep r882 order
    (smallest slot), so slot 0 is kept unless another slot's median rank is strictly smaller. Empty -> slot 0."""
    r = np.asarray(rank_stack, dtype=np.float64)
    if r.size == 0:
        return 0
    med = np.median(r, axis=0)
    return int(np.argmin(med))  # argmin returns the first (smallest) slot among ties


def admit(fit_fam_deltas, seed, assess_fold, arm):
    """Strict inner admission: mean family gain over r882 > .001 AND family-bootstrap LCB95 > 0 (2,000 draws)."""
    d = np.asarray(fit_fam_deltas, dtype=np.float64)
    ix = np.random.default_rng(
        stream_seed(BOOT_SEED, "inner", seed, assess_fold, arm)
    ).integers(0, len(d), (INNER_DRAWS, len(d)))
    lcb, ucb = np.quantile(d[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(d.mean())
    return dict(
        gain=gain,
        ci95=[lcb, ucb],
        admitted=bool(gain > STRICT_MIN_GAIN + EPS and lcb > 0),
    )


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(families) or not np.isfinite(values).all():
        raise ValueError("invalid context panel")
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def bt_ratings(wins, anchor=0, prior=0.01):
    """H148 canonical 400-iteration MM, r882 anchor, pseudo-win prior .01."""
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
    """Per-seed equal-family gains, the median over seeds, the family-bootstrap median CI95 and the Holm-level
    (m = 2) CI [1.25%, 98.75%], one-sided p, harm families, W/T/L and MM Bradley-Terry Elo (r882 = 1500)."""
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(values).all():
        raise ValueError("complete 3-seed 4-policy panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    fam_ids = family_matrix(values[0], families)[0]
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    contrasts = {}
    for col, arm in enumerate(ARMS, 1):
        for label in REFERENCES[arm]:
            ref = ALL_ARMS.index(label)
            diff = fm[:, :, col] - fm[:, :, ref]
            per_seed = diff.mean(axis=1)
            boot = np.median(diff[:, ix].mean(axis=2), axis=0)
            contrasts[arm + "_minus_" + label] = dict(
                per_seed_gain=per_seed.tolist(),
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
                    for d in diff
                ],
                harm_families_by_seed=[
                    [f for f, d in zip(fam_ids, row) if d < -HARM] for row in diff
                ],
            )
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1))
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2)), axis=0)
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=contrasts,
        family_utility_by_seed=fm.mean(axis=1).tolist(),
        context_utility_by_seed=values.mean(axis=1).tolist(),
        vs_plateau_best_by_seed=(values.mean(axis=1) - PLATEAU_BEST).tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
    )


def decide(primary, secondary):
    """hw_t1805: Holm m = 2 over {PRIMARY (TRAIN OOF), SECONDARY (DL-Val91)} for BAG_MED - r882. ADVANCE iff PRIMARY
    median family gain > +0.002 AND PRIMARY Holm-level LCB > 0 AND PRIMARY 3/3 seeds positive AND BAG_MED beats
    BAG_SHUF on PRIMARY (median gain > 0) AND SECONDARY gain > 0 on 3/3 seeds."""
    p = primary["contrasts"]["BAG_MED_minus_r882"]
    s = secondary["contrasts"]["BAG_MED_minus_r882"]
    adj = holm({"PRIMARY": p["p_one_sided"], "SECONDARY": s["p_one_sided"]})
    gates = dict(
        primary_median_gt_margin=bool(p["median_gain"] > ADVANCE_MARGIN + EPS),
        primary_holm_lcb_gt_0=bool(p["ci_holm_m2"][0] > 0),
        primary_three_seeds=bool(p["positive_seeds"] == 3),
        beats_placebo=bool(
            primary["contrasts"]["BAG_MED_minus_BAG_SHUF"]["median_gain"] > EPS
        ),
        secondary_three_seeds=bool(s["positive_seeds"] == 3),
    )
    return dict(holm_p=adj, gates=gates, advance=bool(all(gates.values())))
