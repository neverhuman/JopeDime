"""H223 bagged blend component inside the frozen GATED_025 recipe (H199/H210/H217 machinery): folds, fit plan,
family-bootstrap member weights, blend, gate, statistics, verdict.

Pure numpy; no data access or model code. The fold algorithm is H124 `folds` (sorted families, default_rng(seed)
permutation, 5-way array_split) with a fold seed per outer seed; the blend, endpoint and gain/harm/p rules are H129's.
The gate and its FIT-only tau procedure are H210's, frozen (same grid, same 5% FIT harm target, no retuning).
New in H223: the blend component of the primary is the mean of three HistGBM members per fit unit (member 0 = the
unweighted H217 fit, members 1-2 = family-bootstrap refits), calibrated and gated exactly as GATED_025.
"""

import hashlib
import itertools
import json
import math

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h195_blend025_kernel_confirm.analysis import family_matrix as _shared_family_matrix
from h195_blend025_kernel_confirm.analysis import _contrast as _shared_contrast
from h217_gated025_confirm.analysis import _plateau


SEEDS = (223101, 223102, 223103)
FOLD_SEEDS = {223101: 223141, 223102: 223142, 223103: 223143}
SAMPLE_SEED = 223140
BOOT_SEED = 223150
SHUF_SEED = 223160  # unused (no SHUF arm in H223)
BAG_SEED = 223170  # family-bootstrap draws of members 1-2
MEMBERS = 3  # member 0 is the unweighted fit (GATED_025's component); 1-2 are family-bootstrap refits
# 45 fit units x 3 members = 135 HistGBM fits; no SHUF placebo in H223.
FIT_SEEDS = {"BLEND": SEEDS, "SHUF": ()}
ARM_SEEDS = {}
DRAWS = 10_000
ELO_DRAWS = 1_000
LAMBDA = 0.25
LAMBDAS = {
    "BLEND_025": 0.25,
}  # lambda fixed at 0.25 (H129); no dose in H223 (BLEND_BAG3 uses the same 0.25)
PRIMARY = "GATED_BAG3"
REFERENCE = "GATED_025"
ARMS = (
    "R882",
    "GATED_BAG3",
    "GATED_025",
    "GATED_BAG3_DIS",
    "BLEND_BAG3",
    "BLEND_025",
)
GATES = {
    "GATED_BAG3": "zgap",
    "GATED_025": "zgap",
    "GATED_BAG3_DIS": "disagreement",
}  # arm -> outcome-free gate statistic
PRIORS = {
    "GATED_BAG3": "BAG",
    "GATED_025": "SINGLE",
    "GATED_BAG3_DIS": "BAG",
    "BLEND_BAG3": "BAG",
    "BLEND_025": "SINGLE",
}  # arm -> calibrated prior it blends with r882 (SINGLE = member 0, BAG = mean of the 3 members)
FIT_HARM_TARGET = 0.05  # tau is chosen on FIT under half the 10% A cap
JACCARD_MAX = 0.20  # best-match fold Jaccard against every earlier registered partition (hw_t1325)
DESCRIPTIVE = ARMS[2:]
LCB_MARGIN = 0.001  # H129 one-sided p is computed against this margin
HARM_DELTA, HARM_CAP = -0.01, 0.10
TIE_MARGIN = 1e-4
R882_DL_VAL = (
    1.1377941898164525  # flywheel91 native context mean of r882 (H135/H152 contract)
)
PLATEAU_MARGIN = 0.002  # co-primary: DL-Val context-mean gain must exceed this, with family-bootstrap LCB95 > 0
PLATEAU_SEED = BOOT_SEED + 2


def plateau(arm_values, ref_values, families, seed=PLATEAU_SEED):
    return _plateau(arm_values, ref_values, families, seed, DRAWS, PLATEAU_MARGIN)


def family_win_matrix(values_by_context, family_ids, tie_margin=TIE_MARGIN):
    from h195_blend025_kernel_confirm.analysis import family_win_matrix as shared
    return shared(values_by_context, family_ids, tie_margin)


def contrast(panel, base, families, seed=BOOT_SEED):
    from h195_blend025_kernel_confirm.analysis import _contrast as shared
    return shared(panel, base, families, seed, DRAWS, LCB_MARGIN, HARM_DELTA)


def family_matrix(values, families):
    return _shared_family_matrix(values, families)


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
    """Every H223 seed partition against every seed partition of earlier waves (wave -> {seed: folds})."""
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


def member_weights(row_families, seed, excluded, member):
    """Per-row sample weights of one fit-unit member. Member 0 is unweighted (None: the H217 fit, bit for bit).
    Members 1-2 draw the unit's FIT families with replacement (as many draws as families) and weight every row of a
    family by its multiplicity, so undrawn families get weight 0: a family-cluster bootstrap of the fit rows."""
    if member == 0:
        return None
    if not 0 < member < MEMBERS:
        raise ValueError("unregistered member")
    fam = np.asarray(row_families)
    order = sorted(set(fam.tolist()))
    rng = np.random.default_rng(stream_seed(BAG_SEED, seed, member, *excluded))
    counts = np.bincount(rng.integers(0, len(order), len(order)), minlength=len(order))
    index = {f: i for i, f in enumerate(order)}
    return counts[[index[f] for f in fam.tolist()]].astype(np.float64)


def blend(reference_cost, gbm_cost, lam=LAMBDA):
    """H129: cost-space interpolation r882 + lam * (GBM - r882) (equal to the registered utility interpolation)."""
    ref, prior = np.asarray(reference_cost), np.asarray(gbm_cost)
    if (
        lam not in LAMBDAS.values()
        or ref.shape != prior.shape
        or not (np.isfinite(ref).all() and np.isfinite(prior).all())
    ):
        raise ValueError("H223 lambda is fixed to 0.25; finite matching costs required")
    return (ref + lam * (prior - ref)).astype(np.float32)


# ------------------------------------------------------------------ statistics






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


def verdict(primary, plateau_result):
    """Primary (GATED_BAG3 vs R882, pooled-A native first-argmin, 205 families): CANDIDATE iff LCB95 > 0 AND 3/3
    seeds positive AND harm <= 10%. The H129 one-sided p is reported, not gated. Exact ties are zero gain and fail.
    Co-primary (DL-Val flywheel91 plateau): PLATEAU_WIN iff gain > +0.002 AND family-bootstrap LCB95 > 0.
    Neither result advances anything: CANDIDATE earns a shadow-only note; the flywheel owner decides."""
    gates = dict(
        lcb95_gt_0=primary["ci95"][0] > 0,
        all_3_seeds_positive=primary["positive_seeds"] == len(SEEDS)
        and len(primary["per_seed_gain"]) == len(SEEDS),
        harm_le_10pct=primary["harm_fraction"] <= HARM_CAP,
    )
    confirmed = all(gates.values())
    head = f"CANDIDATE: {PRIMARY}" if confirmed else "NO_CANDIDATE"
    tail = "PLATEAU_WIN" if plateau_result["win"] else "NO_PLATEAU_WIN"
    return dict(
        verdict=f"{head} + {tail}",
        primary=head,
        co_primary_plateau=tail,
        gates=gates,
        plateau_gates=dict(
            gain_gt_0002=plateau_result["gain"] > PLATEAU_MARGIN,
            lcb95_gt_0=plateau_result["ci95"][0] > 0,
        ),
    )


def fold_manifest(rows, other_waves):
    """Registered per-seed family partitions (TRAIN identity metadata only: context and family ids), with the overlap
    against the earlier waves' registered partitions (wave -> {seed: folds})."""
    fams = [r["family_id"] for r in rows]
    folds = {s: seed_folds(fams, s) for s in SEEDS}
    of = {s: {f: k for k, g in enumerate(folds[s]) for f in g} for s in SEEDS}
    earlier = {json.dumps(g) for other in other_waves.values() for g in other.values()}
    if len({json.dumps(g) for g in folds.values()}) != 3 or earlier & {
        json.dumps(g) for g in folds.values()
    }:
        raise ValueError(
            "H223 partitions must be distinct per seed and from earlier waves"
        )
    cross = cross_wave_jaccard(
        folds, {w: {int(s): g for s, g in o.items()} for w, o in other_waves.items()}
    )
    worst = max(
        [v["best_match"] for pairs in cross.values() for v in pairs.values()]
        + [v["best_match"] for v in fold_jaccard(folds).values()]
    )
    if worst > JACCARD_MAX:
        raise ValueError(f"best-match fold Jaccard {worst} > {JACCARD_MAX}")
    return dict(
        wave="H223",
        fold_seeds={str(s): FOLD_SEEDS[s] for s in SEEDS},
        algorithm="per seed s: H124 folds (sorted 205 DL-Train families, default_rng(FOLD_SEEDS[s]).permutation, "
        "5-way array_split); cyclic A=g[k], B=g[(k+1)%5], FIT=other three",
        family_count=len(set(fams)),
        context_count=len(rows),
        fold_context_counts_by_seed={
            str(s): [sum(of[s][f] == k for f in fams) for k in range(5)] for s in SEEDS
        },
        pairwise_fold_jaccard=fold_jaccard(folds),
        cross_wave_fold_jaccard=cross,
        max_best_match_jaccard=worst,
        families_by_seed={str(s): g for s, g in folds.items()},
    )
