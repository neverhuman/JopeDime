"""H226 fresh-fold confirmation of frozen GATED_025 on the retained-best r3175 base (zero adapter updates): folds,
fit plan, blend, gate, statistics, top-1 flip diagnostics, and verdict.

Pure numpy; no data access or model code. The fold algorithm is H124 `folds` (sorted families, default_rng(seed)
permutation, 5-way array_split) with a fresh fold seed per outer seed; the blend, endpoint and gain/harm/p rules are
H129's. The gate and its FIT-only tau procedure are H210/H217's, frozen (same grid, same 5% FIT harm target, no
retuning). New in H226: GATED_025 is applied to the retained-best r3175 base (G3175, with FIT-only tau chosen over
R3175) alongside R3175 alone, R882 reference, G882 replication anchor, and G3175_DIS disagreement placebo; co-primary
PLATEAU_WIN requires both G3175-vs-R882 DL-Val91 gain > +0.002 with family-bootstrap LCB95 > 0 AND G3175-vs-R3175
DL-Val91 family-bootstrap LCB95 > 0 (eliminating winner's-curse selection bias of r3175 on DL-Val).
"""

import hashlib
import itertools
import json
import math

import numpy as np

SEEDS = (226101, 226102, 226103)
FOLD_SEEDS = {226101: 226141, 226102: 226142, 226103: 226143}
SAMPLE_SEED = 226140
BOOT_SEED = 226150
SHUF_SEED = 226160  # unused (no SHUF arm in H226)
# 45 HistGBM fits (~2 min each at 4 threads, H195/H199/H217 timing); no SHUF placebo in H226.
FIT_SEEDS = {"BLEND": SEEDS, "SHUF": ()}
ARM_SEEDS = {}
DRAWS = 10_000
ELO_DRAWS = 1_000
LAMBDA = 0.25
LAMBDAS = {
    "BLEND_025": 0.25,
}  # lambda fixed at 0.25 (H129/H210/H217); no dose in H226
PRIMARY = "G3175"
ARMS = ("R882", "G3175", "R3175", "G882", "G3175_DIS", "B3175", "B882")
BASES = ("R882", "R3175")
BASE_OF = {
    "G3175": "R3175",
    "G3175_DIS": "R3175",
    "B3175": "R3175",
    "G882": "R882",
    "B882": "R882",
}
BLEND_OF = {
    "R3175": "B3175",
    "R882": "B882",
}
GATES = {
    "G3175": "zgap",
    "G3175_DIS": "disagreement",
    "G882": "zgap",
}  # gated arm -> outcome-free gate statistic on its own base (BASE_OF[arm])
FIT_HARM_TARGET = 0.05  # tau is chosen on FIT under half the 10% A cap
DESCRIPTIVE = ARMS[2:]
LCB_MARGIN = 0.001  # H129 one-sided p is computed against this margin
HARM_DELTA, HARM_CAP = -0.01, 0.10
TIE_MARGIN = 1e-4
R882_DL_VAL = (
    1.1377941898164525  # flywheel91 native context mean of r882 (H135/H152 contract)
)
R3175_DL_VAL = 1.1389572403748425  # flywheel91 native context mean of r3175 (adapter_admissions.jsonl round 3175)
PLATEAU_MARGIN = 0.002  # co-primary: DL-Val context-mean gain vs R882 must exceed this, with LCB95 > 0 vs both R882 and R3175
PLATEAU_SEED = BOOT_SEED + 2
JACCARD_CAP = 0.20  # preregistered best-match Jaccard target vs prior waves
EARLIER_FOLD_SEEDS = {
    "H203": {203101: 203141, 203102: 203142, 203103: 203143},
    "H210": {210101: 210141, 210102: 210142, 210103: 210143},
    "H217": {217101: 217141, 217102: 217142, 217103: 217143},
    "H218": {218101: 218141, 218102: 218142, 218103: 218143},
    "H222": {222101: 222141, 222102: 222142, 222103: 222143},
    "H223": {223101: 223141, 223102: 223142, 223103: 223143},
    "H224": {224101: 224141, 224102: 224142, 224103: 224143},
    "H225": {225101: 225141, 225102: 225142, 225103: 225143},
}


# ------------------------------------------------------------------ folds and fit plan


def folds_for_seed(family_ids, fold_seed):
    """H124 `folds` algorithm for an arbitrary fold_seed: sorted 205 families, permutation, five groups."""
    families = sorted(set(family_ids))
    if len(families) != 205:
        raise ValueError("205 DL-Train families required")
    return [
        sorted(x.tolist())
        for x in np.array_split(
            np.random.default_rng(fold_seed).permutation(families), 5
        )
    ]


def seed_folds(family_ids, seed):
    """H124 `folds` algorithm with this seed's fold seed: sorted families, permutation, five groups."""
    return folds_for_seed(family_ids, FOLD_SEEDS[seed])


def partition_jaccard(a, b):
    """Two five-fold partitions: mean over k of the Jaccard of fold k, and the mean best-match Jaccard."""
    same = [len(set(a[k]) & set(b[k])) / len(set(a[k]) | set(b[k])) for k in range(5)]
    best = [max(len(set(x) & set(y)) / len(set(x) | set(y)) for y in b) for x in a]
    return dict(same_index=float(np.mean(same)), best_match=float(np.mean(best)))


def fold_jaccard(folds_by_seed):
    """Pairwise across seeds (partition_jaccard)."""
    return {
        f"{s}-{t}": partition_jaccard(folds_by_seed[s], folds_by_seed[t])
        for s, t in itertools.combinations(sorted(folds_by_seed), 2)
    }


def cross_wave_jaccard(folds_by_seed, other_waves):
    """Every H226 seed partition against every seed partition of earlier waves (wave -> {seed: folds})."""
    return {
        wave: {
            f"{s}-{t}": partition_jaccard(folds_by_seed[s], other[t])
            for s in sorted(folds_by_seed)
            for t in sorted(other)
        }
        for wave, other in sorted(other_waves.items())
    }


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
        raise ValueError("H226 lambda is fixed to 0.25; finite matching costs required")
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


# ------------------------------------------------------------------ outcome-free gate


def zgap(r882_cost, blend_cost):
    """Primary gate statistic: how strongly r882 disfavours the blend's pick. (r882[b] - r882[a]) / std(r882) for
    a = first argmin of r882 and b = first argmin of the blend; 0 when the picks agree, +inf for a constant r882.
    Outcome-free: costs only (no fitness, gate or utility)."""
    r, c = np.asarray(r882_cost, np.float64), np.asarray(blend_cost, np.float64)
    a, b = int(np.argmin(r)), int(np.argmin(c))
    if a == b:
        return 0.0
    sd = float(np.std(r))
    return math.inf if sd == 0 else float((r[b] - r[a]) / sd)


def disagreement(gbm_cost, r882_cost):
    """Report-only gate statistic, H129's: 1 - Spearman(GBM, r882) over unrevealed slots 4..P-1 (inf if constant)."""
    a, b = np.asarray(gbm_cost, np.float64)[4:], np.asarray(r882_cost, np.float64)[4:]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.inf

    def ranks(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        return (np.bincount(inverse, weights=r) / counts)[inverse]

    return float(1.0 - np.corrcoef(ranks(a), ranks(b))[0, 1])


def tau_grid(values):
    """Closed gate (-inf = always R882), FIT quantiles q10..q90 of the statistic among contexts where it is > 0 and
    finite (switching contexts for zgap), and +inf (always blend). Linear interpolation."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    qs = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if len(v) else []
    return [-math.inf, *qs, math.inf]


def gated(stat_value, tau):
    """Gate open (use the blend) iff the statistic is <= tau; a closed gate falls back to r882."""
    return bool(stat_value <= tau)


def choose_tau(
    stat_values, r882_utility, blend_utility, families, target=FIT_HARM_TARGET
):
    """FIT-only selection: the tau maximising the FIT equal-family native gain over r882 among taus whose FIT harm
    fraction (families with mean delta < -0.01) is <= target. Ties go to the smaller tau (more conservative); the
    closed gate (gain 0, harm 0) is always feasible, so the rule falls back to r882 when nothing qualifies."""
    s = np.asarray(stat_values, np.float64)
    r, b = np.asarray(r882_utility, np.float64), np.asarray(blend_utility, np.float64)
    fams = np.asarray(families)
    order = sorted(set(fams.tolist()))
    rows = []
    for tau in tau_grid(s):
        delta = np.where(s <= tau, b - r, 0.0)
        fam = np.array([delta[fams == f].mean() for f in order])
        rows.append(
            dict(
                tau=tau,
                gain=float(fam.mean()),
                harm=float((fam < HARM_DELTA).mean()),
                open_rate=float((s <= tau).mean()),
            )
        )
    feasible = [x for x in rows if x["harm"] <= target]
    best = max(feasible, key=lambda x: (x["gain"], -x["tau"]))
    return dict(
        tau=best["tau"],
        fit_gain=best["gain"],
        fit_harm=best["harm"],
        fit_open_rate=best["open_rate"],
        grid=rows,
    )


def plateau(arm_values, ref_values, families, seed=PLATEAU_SEED):
    """DL-Val flywheel91 contrast: per-context native utilities of the arm and of a reference arm (91 contexts, each
    averaged over the 15 cells). Gain = context-mean delta. CI95 = 10,000 family-cluster draws (families resampled with
    replacement, every drawn family's contexts pooled, context-mean delta)."""
    a, r = np.asarray(arm_values, np.float64), np.asarray(ref_values, np.float64)
    fams = np.asarray(families)
    if a.shape != r.shape or a.shape != fams.shape or not np.isfinite(a - r).all():
        raise ValueError("matching finite per-context values required")
    delta = a - r
    order = sorted(set(fams.tolist()))
    sums = np.array([delta[fams == f].sum() for f in order])
    counts = np.array([(fams == f).sum() for f in order])
    draw = np.random.default_rng(seed).integers(0, len(order), (DRAWS, len(order)))
    boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    gain = float(delta.mean())
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return dict(
        arm_context_mean=float(a.mean()),
        ref_context_mean=float(r.mean()),
        r882_context_mean=float(r.mean()),
        gain=gain,
        ci95=ci,
        contexts=int(len(a)),
        families=len(order),
        lcb95_gt_0=bool(ci[0] > 0 and gain > 0),
        win_vs_margin=bool(gain > PLATEAU_MARGIN and ci[0] > 0),
    )


def top1_flips(arm_picks_by_cell, ref_picks_by_cell):
    """DL-Val91 top-1 pick flip diagnostics across the 15 cells (3 seeds x 5 outer folds).
    Both inputs have shape (15, C) holding integer first-argmin indices per cell and context."""
    a = np.asarray(arm_picks_by_cell, dtype=np.int64)
    r = np.asarray(ref_picks_by_cell, dtype=np.int64)
    if a.shape != r.shape or a.ndim != 2 or a.shape[0] != 15:
        raise ValueError("expected (15, C) pick matrices")
    diff = a != r  # (15, C)
    per_cell = diff.sum(axis=1)
    per_seed = [float(per_cell[s * 5 : (s + 1) * 5].mean()) for s in range(3)]
    return dict(
        contexts_any_cell=int(diff.any(axis=0).sum()),
        contexts_majority_cells=int((diff.sum(axis=0) >= 8).sum()),
        contexts_all_cells=int(diff.all(axis=0).sum()),
        mean_across_cells=float(per_cell.mean()),
        median_across_seeds=float(np.median(per_seed)),
        per_seed_mean=per_seed,
        per_cell_counts=per_cell.tolist(),
        total_contexts=int(a.shape[1]),
    )


def verdict(primary, plateau_vs_r882, plateau_vs_r3175):
    """Primary (G3175 vs R882, pooled-A native first-argmin, 205 families): CONFIRMED: G3175 iff LCB95 > 0 AND 3/3 seeds
    positive AND harm <= 10%. The H129 one-sided p is reported, not gated. Exact ties are zero gain and fail.
    Co-primary (DL-Val flywheel91 plateau): PLATEAU_WIN iff G3175-vs-R882 gain > +0.002 AND family-bootstrap LCB95 > 0
    AND G3175-vs-R3175 family-bootstrap LCB95 > 0 (eliminating winner's-curse selection bias of r3175 on DL-Val)."""
    gates = dict(
        lcb95_gt_0=primary["ci95"][0] > 0,
        all_3_seeds_positive=primary["positive_seeds"] == len(SEEDS)
        and len(primary["per_seed_gain"]) == len(SEEDS),
        harm_le_10pct=primary["harm_fraction"] <= HARM_CAP,
    )
    plateau_gates = dict(
        vs_r882_gain_gt_0002=plateau_vs_r882["gain"] > PLATEAU_MARGIN,
        vs_r882_lcb95_gt_0=plateau_vs_r882["ci95"][0] > 0,
        vs_r3175_lcb95_gt_0=plateau_vs_r3175["ci95"][0] > 0
        and plateau_vs_r3175["gain"] > 0,
    )
    confirmed = all(gates.values())
    plateau_win = all(plateau_gates.values())
    head = "CONFIRMED: G3175" if confirmed else "NOT_CONFIRMED"
    tail = "PLATEAU_WIN" if plateau_win else "NO_PLATEAU_WIN"
    return dict(
        verdict=f"{head} + {tail}",
        primary=head,
        co_primary_plateau=tail,
        gates=gates,
        plateau_gates=plateau_gates,
    )


def fold_manifest(rows, other_waves=None):
    """Registered per-seed family partitions (TRAIN identity metadata only: context and family ids), with the overlap
    against earlier waves' registered partitions (H203, H210, H217, H218, H222, H223, H224, H225)."""
    fams = [r["family_id"] for r in rows]
    folds = {s: seed_folds(fams, s) for s in SEEDS}
    of = {s: {f: k for k, g in enumerate(folds[s]) for f in g} for s in SEEDS}
    waves = {
        w: {s: folds_for_seed(fams, fs) for s, fs in fseeds.items()}
        for w, fseeds in EARLIER_FOLD_SEEDS.items()
    }
    if other_waves:
        for w, o in other_waves.items():
            waves[w] = {int(s): g for s, g in o.items()}
    earlier = {json.dumps(g) for other in waves.values() for g in other.values()}
    if len({json.dumps(g) for g in folds.values()}) != 3 or earlier & {
        json.dumps(g) for g in folds.values()
    }:
        raise ValueError(
            "H226 partitions must be distinct per seed and from earlier waves"
        )
    cw = cross_wave_jaccard(folds, waves)
    pw = fold_jaccard(folds)
    max_within = max(v["best_match"] for v in pw.values())
    max_by_wave = {
        w: max(v["best_match"] for v in pairs.values()) for w, pairs in cw.items()
    }
    if max_within > JACCARD_CAP or any(v > JACCARD_CAP for v in max_by_wave.values()):
        raise ValueError("fold best-match Jaccard exceeds 0.20 cap")
    return dict(
        wave="H226",
        fold_seeds={str(s): FOLD_SEEDS[s] for s in SEEDS},
        algorithm="per seed s: H124 folds (sorted 205 DL-Train families, default_rng(FOLD_SEEDS[s]).permutation, "
        "5-way array_split); cyclic A=g[k], B=g[(k+1)%5], FIT=other three",
        family_count=len(set(fams)),
        context_count=len(rows),
        fold_context_counts_by_seed={
            str(s): [sum(of[s][f] == k for f in fams) for k in range(5)] for s in SEEDS
        },
        pairwise_fold_jaccard=pw,
        cross_wave_fold_jaccard=cw,
        max_best_match_jaccard_by_wave=max_by_wave,
        families_by_seed={str(s): g for s, g in folds.items()},
    )
