"""H218 nested blend-weight (lambda in {0.125, 0.25, 0.5}) + outcome-free zgap gate on FIT: folds, fit plan, blend, gate, statistics, verdict.

Pure numpy; no data access or model code. The fold algorithm is H124 `folds` (sorted families, default_rng(seed)
permutation, 5-way array_split) with a fresh fold seed per outer seed; the blend, endpoint and gain/harm/p rules are
H129's. The functions `zgap`, `disagreement`, `tau_grid`, `gated`, `choose_tau`, and `blend` are byte-identical to
H210/H217; only `LAMBDAS` expands from `{0.25}` to `{0.125, 0.25, 0.5}` and `choose_lambda_tau` selects `(lambda, tau)`
nested on FIT only (inner 3-fold family CV out-of-fold predictions) maximizing FIT equal-family native gain subject to
FIT harm <= 5%, else falling back to r882 (`tau = -inf`).
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
from h217_gated025_confirm.analysis import _plateau as _shared_plateau

SEEDS = (218101, 218102, 218103)
FOLD_SEEDS = {218101: 218141, 218102: 218142, 218103: 218143}
SAMPLE_SEED = 218140
BOOT_SEED = 218150
SHUF_SEED = 218160  # deterministic stream seed for permuting zgap across contexts in GATED_NEST_SHUF
# 45 HistGBM fits (~2 min each at 4 threads, H195/H199/H210 timing); no SHUF HistGBM fits needed.
FIT_SEEDS = {"BLEND": SEEDS, "SHUF": ()}
ARM_SEEDS = {}
DRAWS = 10_000
ELO_DRAWS = 1_000
LAMBDA = 0.25
LAMBDA_GRID = (0.125, 0.25, 0.5)
LAMBDAS = {
    "BLEND_125": 0.125,
    "BLEND_025": 0.25,
    "BLEND_500": 0.5,
}
PRIMARY = "GATED_NEST"
ARMS = ("R882", "GATED_NEST", "GATED_NEST_SHUF", "GATED_025")
GATES = {
    "GATED_NEST": "zgap_nest",
    "GATED_NEST_SHUF": "zgap_nest_shuf",
    "GATED_025": "zgap_025",
}
FIT_HARM_TARGET = 0.05  # (lambda, tau) chosen on FIT under half the 10% A cap
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
    return _shared_plateau(arm_values, ref_values, families, seed, DRAWS, PLATEAU_MARGIN)


def family_win_matrix(values_by_context, family_ids, tie_margin=TIE_MARGIN):
    from h195_blend025_kernel_confirm.analysis import family_win_matrix as shared
    return shared(values_by_context, family_ids, tie_margin)


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
    """Every H218 seed partition against every seed partition of earlier waves (wave -> {seed: folds})."""
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
        raise ValueError("H218 lambda is fixed to 0.25; finite matching costs required")
    return (ref + lam * (prior - ref)).astype(np.float32)


# ------------------------------------------------------------------ statistics






def family_matrix(values, families):
    return _shared_family_matrix(values, families)


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
    p_margin = float((1 + np.count_nonzero(boot <= LCB_MARGIN)) / (DRAWS + 1))
    p_zero = float((1 + np.count_nonzero(boot <= 0.0)) / (DRAWS + 1))
    return dict(
        per_seed_gain=per_seed.tolist(),
        median_gain=float(np.median(per_seed)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        p_one_sided_vs_margin=p_margin,
        p_one_sided_vs_zero=p_zero,
        holm_m1_p=p_zero,
        positive_seeds=int((per_seed > 0).sum()),
        harm_fraction=float((fam_mean < HARM_DELTA).mean()),
        harm_families=int((fam_mean < HARM_DELTA).sum()),
        families=int(n),
    )




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


def permute_zgap(stat_values, seed):
    """Permute outcome-free zgap values across contexts using a deterministic stream seed."""
    v = np.asarray(stat_values, np.float64)
    perm = np.random.default_rng(seed).permutation(len(v))
    return v[perm].tolist()


def choose_lambda_tau(
    stat_values_by_lam,
    r882_utility,
    blend_utility_by_lam,
    families,
    target=FIT_HARM_TARGET,
):
    """FIT-only nested (lambda, tau) selection across LAMBDA_GRID = (0.125, 0.25, 0.5).

    For each candidate lambda in LAMBDA_GRID, applies the frozen H210/H217 choose_tau rule on FIT.
    Selects the (lambda, tau) maximising FIT equal-family native gain over r882 subject to FIT harm <= target (5%).
    Ties go to the smaller tau (more conservative) and then smaller lambda; when no open threshold qualifies with
    positive gain, falls back to r882 (tau = -inf, selected_lambda = 'R882').
    """
    by_lambda = {}
    candidates = []
    for lam in LAMBDA_GRID:
        c = choose_tau(
            stat_values_by_lam[lam],
            r882_utility,
            blend_utility_by_lam[lam],
            families,
            target=target,
        )
        by_lambda[str(lam)] = c
        candidates.append((lam, c))
    best_lam, best = max(
        candidates, key=lambda item: (item[1]["fit_gain"], -item[1]["tau"], -item[0])
    )
    closed = best["tau"] == -math.inf
    return dict(
        lam=0.25 if closed else float(best_lam),
        selected_lambda="R882" if closed else str(best_lam),
        tau=best["tau"],
        fit_gain=best["fit_gain"],
        fit_harm=best["fit_harm"],
        fit_open_rate=best["fit_open_rate"],
        by_lambda={
            k: {gk: gv for gk, gv in v.items() if gk != "grid"}
            for k, v in by_lambda.items()
        },
        grid=best["grid"],
    )


def lambda_selection_counts(cells, arm="GATED_NEST"):
    """Count how many cells selected each lambda in {'0.125', '0.25', '0.5', 'R882'}."""
    counts = {"0.125": 0, "0.25": 0, "0.5": 0, "R882": 0}
    for c in cells:
        sel = str(c["gate_choice"][arm].get("selected_lambda", "0.25"))
        counts[sel] = counts.get(sel, 0) + 1
    return counts




def verdict(primary, plateau_result):
    """Primary (GATED_NEST, pooled-A native first-argmin, 205 families): CANDIDATE: GATED_NEST iff LCB95 > 0 AND 3/3
    seeds positive AND harm <= 10% (Holm over primary only, m=1). Exact ties are zero gain and fail.
    Co-primary (DL-Val flywheel91 plateau): PLATEAU_WIN iff gain > +0.002 AND family-bootstrap LCB95 > 0."""
    gates = dict(
        lcb95_gt_0=primary["ci95"][0] > 0,
        all_3_seeds_positive=primary["positive_seeds"] == len(SEEDS)
        and len(primary["per_seed_gain"]) == len(SEEDS),
        harm_le_10pct=primary["harm_fraction"] <= HARM_CAP,
    )
    confirmed = all(gates.values())
    head = "CANDIDATE: GATED_NEST" if confirmed else "NO_ADVANCE"
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
            "H218 partitions must be distinct per seed and from earlier waves"
        )
    return dict(
        wave="H218",
        fold_seeds={str(s): FOLD_SEEDS[s] for s in SEEDS},
        algorithm="per seed s: H124 folds (sorted 205 DL-Train families, default_rng(FOLD_SEEDS[s]).permutation, "
        "5-way array_split); cyclic A=g[k], B=g[(k+1)%5], FIT=other three",
        family_count=len(set(fams)),
        context_count=len(rows),
        fold_context_counts_by_seed={
            str(s): [sum(of[s][f] == k for f in fams) for k in range(5)] for s in SEEDS
        },
        pairwise_fold_jaccard=fold_jaccard(folds),
        cross_wave_fold_jaccard=cross_wave_jaccard(
            folds,
            {w: {int(s): g for s, g in o.items()} for w, o in other_waves.items()},
        ),
        families_by_seed={str(s): g for s, g in folds.items()},
    )
