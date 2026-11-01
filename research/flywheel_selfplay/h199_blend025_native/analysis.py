"""H199 BLEND_025 on the flywheel native first-argmin protocol (H195 machinery): folds, fit plan, SHUF, blend, statistics, verdict.

Pure numpy; no data access or model code. The fold algorithm is H124 `folds` (sorted families, default_rng(seed)
permutation, 5-way array_split) with a fold seed per outer seed; the blend, endpoint and gain/harm/p rules are H129's.
"""

import hashlib
import itertools

import numpy as np

SEEDS = (199101, 199102, 199103)
FOLD_SEEDS = {199101: 199141, 199102: 199142, 199103: 199143}
SAMPLE_SEED = 199140
BOOT_SEED = 199150
SHUF_SEED = 199160
# 4 h cap: 60 HistGBM fits (~2 min each at 4 threads, H124 timing). The descriptive SHUF placebo uses the first seed only.
FIT_SEEDS = {"BLEND": SEEDS, "SHUF": SEEDS[:1]}
ARM_SEEDS = {"BLEND_025_SHUF": SEEDS[:1]}
DRAWS = 10_000
ELO_DRAWS = 1_000
LAMBDA = 0.25
LAMBDAS = {
    "BLEND_015": 0.15,
    "BLEND_025": 0.25,
    "BLEND_035": 0.35,
}  # 0.15/0.35 report-only dose
PRIMARY = "BLEND_025"
ARMS = ("R882", "BLEND_025", "BLEND_015", "BLEND_035", "BLEND_025_SHUF")
DESCRIPTIVE = ARMS[2:]
LCB_MARGIN = 0.001  # median gate and the report-only H129 one-sided p
HARM_DELTA, HARM_CAP = -0.01, 0.10
TIE_MARGIN = 1e-4


# ------------------------------------------------------------------ folds and fit plan


def seed_folds(family_ids, seed):
    """H124 `folds` algorithm with this seed's fold seed: sorted families, permutation, five groups."""
    families = sorted(set(family_ids))
    if len(families) != 205:
        raise ValueError("205 DL-Train families required")
    return [
        sorted(x.tolist())
        for x in np.array_split(
            np.random.default_rng(FOLD_SEEDS[seed]).permutation(families), 5
        )
    ]


def fold_jaccard(folds_by_seed):
    """Pairwise across seeds: mean over k of the Jaccard of A-fold k, and the mean best-match Jaccard."""
    out = {}
    for s, t in itertools.combinations(sorted(folds_by_seed), 2):
        a, b = folds_by_seed[s], folds_by_seed[t]
        same = [
            len(set(a[k]) & set(b[k])) / len(set(a[k]) | set(b[k])) for k in range(5)
        ]
        best = [max(len(set(x) & set(y)) / len(set(x) | set(y)) for y in b) for x in a]
        out[f"{s}-{t}"] = dict(
            same_index=float(np.mean(same)), best_match=float(np.mean(best))
        )
    return out


def cell(k):
    """Outer fold k: A = k, B = (k+1) % 5, FIT = the other three."""
    a, b = k, (k + 1) % 5
    return a, b, tuple(j for j in range(5) if j not in (a, b))


def fit_plan():
    """Per seed and model type: 5 outer models (exclude {A, B}) + 10 inner models (exclude {A, B, j}, j in FIT),
    i.e. all 5 adjacent pairs and all C(5,3) triples (every triple of a 5-cycle contains an adjacent pair)."""
    outers = sorted({tuple(sorted(cell(k)[:2])) for k in range(5)})
    inners = sorted(
        {tuple(sorted((*cell(k)[:2], j))) for k in range(5) for j in cell(k)[2]}
    )
    return outers, inners


def inner_model_for(k, j):
    """The inner model whose out-of-fold prediction calibrates FIT fold j of outer fold k."""
    a, b, fit = cell(k)
    if j not in fit:
        raise ValueError("j must be a FIT fold of k")
    return tuple(sorted((a, b, j)))


def shuffle_within_family(targets, row_families, seed):
    """SHUF placebo: permute the training targets within each family (all sampled rows of that family's contexts)."""
    y = np.asarray(targets).copy()
    fam = np.asarray(row_families)
    rng = np.random.default_rng(seed)
    for f in sorted(set(fam.tolist())):
        idx = np.flatnonzero(fam == f)
        y[idx] = y[rng.permutation(idx)]
    return y


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def blend(reference_cost, gbm_cost, lam=LAMBDA):
    """H129: cost-space interpolation r882 + lam * (GBM - r882) (equal to the registered utility interpolation)."""
    ref, prior = np.asarray(reference_cost), np.asarray(gbm_cost)
    if (
        lam not in LAMBDAS.values()
        or ref.shape != prior.shape
        or not (np.isfinite(ref).all() and np.isfinite(prior).all())
    ):
        raise ValueError(
            "H199 lambdas are 0.15, 0.25, 0.35; finite matching costs required"
        )
    return (ref + lam * (prior - ref)).astype(np.float32)


# ------------------------------------------------------------------ statistics


def family_matrix(values, families):
    order = sorted(set(families))
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray([v[..., np.asarray(families) == f].mean(axis=-1) for f in order])
    return order, np.moveaxis(m, 0, -1)


def contrast(panel, base, families, seed=BOOT_SEED):
    """panel/base: [3 seeds, contexts]. Per-seed equal-family gain, median over seeds, 10,000 shared family draws
    reduced over seeds (axis 0), H129 one-sided p against +0.001, harm = families whose three-seed mean delta < -0.01."""
    _, fp = family_matrix(panel, families)
    _, fb = family_matrix(base, families)
    delta = fp - fb  # (3, F)
    n = delta.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    boot = np.median(delta[:, draw].mean(axis=2), axis=0)
    per_seed = delta.mean(axis=1)
    fam_mean = delta.mean(axis=0)
    return dict(
        per_seed_gain=per_seed.tolist(),
        median_gain=float(np.median(per_seed)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        p_one_sided_vs_margin=float(
            (1 + np.count_nonzero(boot <= LCB_MARGIN)) / (DRAWS + 1)
        ),
        positive_seeds=int((per_seed > 0).sum()),
        harm_fraction=float((fam_mean < HARM_DELTA).mean()),
        harm_families=int((fam_mean < HARM_DELTA).sum()),
        families=int(n),
    )


def family_win_matrix(values_by_context, family_ids, tie_margin=TIE_MARGIN):
    v = np.asarray(values_by_context, dtype=np.float64)
    diff = v[:, :, None] - v[:, None, :]
    wins = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5)) * (
        1.0 - np.eye(v.shape[1])
    )
    fams = sorted(set(family_ids))
    return np.array([wins[np.asarray(family_ids) == f].mean(axis=0) for f in fams])


def bt_ratings(wins, anchor=0, prior=0.01):
    """Canonical 400-iteration MM (H148/H176), with an optional leading bootstrap dimension."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    s = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2 * prior * (1 - np.eye(n)))
            / np.maximum(s[..., :, None] + s[..., None, :], 1e-12)
        ).sum(axis=-1)
        s = np.clip(totals / denom, 1e-6, 1e6)
        s /= np.maximum(s[..., anchor, None], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(s, 1e-9))


def elo_vs_reference(panel, base, families, seed=BOOT_SEED + 1):
    wins = np.stack(
        [
            family_win_matrix(np.stack([base[s], panel[s]], axis=1), families)
            for s in range(len(panel))
        ]
    )
    elo = bt_ratings(wins.sum(axis=1))[:, 1]
    n = wins.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (ELO_DRAWS, n))
    boot = np.median(
        bt_ratings(np.take(wins, draw, axis=1).sum(axis=2))[..., 1], axis=0
    )
    return dict(
        per_seed=elo.tolist(),
        median=float(np.median(elo)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def verdict(primary):
    """CANDIDATE (shadow-only adoption note; the flywheel owner decides) iff, on pooled-A native first-argmin utility
    (Holm m=1): LCB95 > 0 AND median > +0.001 AND 3/3 seeds positive AND harm <= 10%. Exact ties are zero gain;
    a tie fails the strict gates. The H129 p against +0.001 is reported, not gated."""
    gates = dict(
        lcb95_gt_0=primary["ci95"][0] > 0,
        median_gt_0001=primary["median_gain"] > LCB_MARGIN,
        all_seeds_positive=primary["positive_seeds"] == len(SEEDS),
        harm_le_10pct=primary["harm_fraction"] <= HARM_CAP,
    )
    return dict(
        verdict="CANDIDATE: BLEND_025" if all(gates.values()) else "NO_ADVANCE",
        gates=gates,
    )


def fold_manifest(rows):
    """Registered per-seed family partitions (TRAIN identity metadata only: context and family ids)."""
    fams = [r["family_id"] for r in rows]
    folds = {s: seed_folds(fams, s) for s in SEEDS}
    of = {s: {f: k for k, g in enumerate(folds[s]) for f in g} for s in SEEDS}
    return dict(
        wave="H199",
        fold_seeds={str(s): FOLD_SEEDS[s] for s in SEEDS},
        algorithm="per seed s: H124 folds (sorted 205 DL-Train families, default_rng(FOLD_SEEDS[s]).permutation, "
        "5-way array_split); cyclic A=g[k], B=g[(k+1)%5], FIT=other three",
        family_count=len(set(fams)),
        context_count=len(rows),
        fold_context_counts_by_seed={
            str(s): [sum(of[s][f] == k for f in fams) for k in range(5)] for s in SEEDS
        },
        pairwise_fold_jaccard=fold_jaccard(folds),
        families_by_seed={str(s): g for s, g in folds.items()},
    )
