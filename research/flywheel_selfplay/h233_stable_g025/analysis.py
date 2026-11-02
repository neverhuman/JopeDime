"""H233 pick-stability unanimity vote gate inside the frozen H217 GATED_025 recipe (0 optimizer updates, xbabe2 CPU):
folds, 75-fit plan (5 outer x 3 members + 10 inner x 1 member = 25 fits/seed), family-bootstrap member weights,
pick-unanimity vote, FIT-cell agreement-mask permutation (STABLE_SHUF), statistics, and admission verdict.

Pure numpy; no data access or model code. The fold algorithm is H124 `folds` (sorted families, default_rng(seed)
permutation, 5-way array_split) with a fresh fold seed per outer seed (233141-233143); the blend, endpoint and
gain/harm/bootstrap rules are H129/H195/H217's. The zgap gate and its FIT-only tau procedure are H210/H217's, frozen
(same grid, same 5% FIT harm target, no retuning).
New in H233: STABLE_G025 applies the frozen H217 GATED_025 blend to a context ONLY when (a) the H217 zgap gate passes
AND (b) the blended first-argmin pick is unanimous across 3 outer HistGBM members (member 0 = H217 unweighted fit,
members 1-2 = FIT-family bootstrap refits with the same recipe); otherwise it falls back to r882.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h195_blend025_kernel_confirm.analysis import _contrast as _shared_contrast
from h195_blend025_kernel_confirm.analysis import family_matrix as _shared_family_matrix
from h217_gated025_confirm.analysis import _plateau

SEEDS = (233101, 233102, 233103)
FOLD_SEEDS = {233101: 233141, 233102: 233142, 233103: 233143}
SAMPLE_SEED = 233140
BOOT_SEED = 233150
SHUF_SEED = 233160  # deterministic permutation of member-agreement mask within each FIT-calibrated cell
BAG_SEED = 233170  # family-cluster bootstrap draws of outer members 1-2
MEMBERS = 3  # outer units fit 3 members (0 = unweighted H217 fit, 1-2 = FIT-family bootstrap refits)
INNER_MEMBERS = 1  # inner units fit member 0 only (H217 FIT calibration & tau selection): 5*3 + 10*1 = 25 fits/seed
FIT_SEEDS = {"BLEND": SEEDS, "SHUF": ()}
ARM_SEEDS = {}
DRAWS = 10_000
ELO_DRAWS = 1_000
LAMBDA = 0.25
LAMBDAS = {"BLEND_025": 0.25}
PRIMARY = "STABLE_G025"
REFERENCE = "G025"
PLACEBO = "STABLE_SHUF"
ARMS = (
    "R882",
    "G025",
    "STABLE_G025",
    "STABLE_SHUF",
    "BAG_VOTE_UNGATED",
)
GATES = {
    "G025": "zgap",
    "STABLE_G025": "zgap_and_unanimous",
    "STABLE_SHUF": "zgap_and_shuf_unanimous",
    "BAG_VOTE_UNGATED": "unanimous",
}
FIT_HARM_TARGET = 0.05  # tau is chosen on FIT under half the 10% A cap (frozen H210/H217 rule)
JACCARD_MAX = 0.20  # best-match fold Jaccard bound vs H203-H231 (hw_t2205)
DESCRIPTIVE = ("G025", "STABLE_SHUF", "BAG_VOTE_UNGATED")
GAIN_MARGIN = 0.002  # prereg admission: median equal-family gain over 3 seeds vs R882 > +0.002
LCB_MARGIN = 0.001  # H129 one-sided p is computed against this margin
HARM_DELTA, HARM_CAP = -0.01, 0.10
TIE_MARGIN = 1e-4
R882_DL_VAL = 1.1377941898164525
PLATEAU_MARGIN = 0.002
PLATEAU_SEED = BOOT_SEED + 2


def plateau(arm_values, ref_values, families, seed=PLATEAU_SEED):
    return _plateau(arm_values, ref_values, families, seed, DRAWS, PLATEAU_MARGIN)


def family_win_matrix(values_by_context, family_ids, tie_margin=TIE_MARGIN):
    from h195_blend025_kernel_confirm.analysis import family_win_matrix as shared

    return shared(values_by_context, family_ids, tie_margin)


def contrast(panel, base, families, seed=BOOT_SEED):
    return _shared_contrast(panel, base, families, seed, DRAWS, LCB_MARGIN, HARM_DELTA)


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
    """Every H233 seed partition against every seed partition of earlier waves (wave -> {seed: folds})."""
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
    """Per seed: 5 outer exclusion pairs (exclude {A, B}) + 10 inner exclusion triples (exclude {A, B, j}, j in FIT),
    i.e. all 5 adjacent pairs and all C(5,3) triples (every triple of a 5-cycle contains an adjacent pair)."""
    outers = sorted({tuple(sorted(cell(k)[:2])) for k in range(5)})
    inners = sorted(
        {tuple(sorted((*cell(k)[:2], j))) for k in range(5) for j in cell(k)[2]}
    )
    return outers, inners


def members_for_exclusion(excluded):
    """Outer units (exclude {A, B}, len==2) fit 3 members (0, 1, 2); inner units (exclude {A, B, j}, len==3) fit
    member 0 only (the H217 recipe used for FIT calibration and tau selection). Total: 5*3 + 10*1 = 25 fits/seed."""
    if tuple(sorted(set(excluded))) != tuple(excluded):
        raise ValueError("excluded folds must be sorted and distinct")
    if len(excluded) == 2:
        return tuple(range(MEMBERS))
    if len(excluded) == 3:
        return (0,)
    raise ValueError("exclusion size must be 2 (outer) or 3 (inner)")


def inner_model_for(k, j):
    """The inner model whose out-of-fold prediction calibrates FIT fold j of outer fold k."""
    a, b, fit = cell(k)
    if j not in fit:
        raise ValueError("j must be a FIT fold of k")
    return tuple(sorted((a, b, j)))


def shuffle_within_family(targets, row_families, seed):
    """SHUF helper: permute training targets within each family."""
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
        raise ValueError("H233 lambda is fixed to 0.25; finite matching costs required")
    return (ref + lam * (prior - ref)).astype(np.float32)


def unanimous_pick(blended_members):
    """Check whether the blended first-argmin pick is unanimous across all 3 HistGBM members."""
    if len(blended_members) != MEMBERS:
        raise ValueError("exactly 3 blended member cost vectors required")
    picks = tuple(int(np.argmin(np.asarray(b))) for b in blended_members)
    return bool(picks[0] == picks[1] == picks[2]), picks


def shuffle_agreement_mask(agree_by_cid, seed, outer_k, stage_tag="A"):
    """Permute the boolean member-agreement mask across contexts within a single FIT-calibrated cell (seed, outer_k),
    preserving the exact vote count and vote rate."""
    if seed not in SEEDS or not 0 <= int(outer_k) < 5:
        raise ValueError("unregistered seed or outer fold")
    cids = sorted(agree_by_cid)
    mask = np.array([bool(agree_by_cid[c]) for c in cids], dtype=bool)
    rng = np.random.default_rng(stream_seed(SHUF_SEED, seed, int(outer_k), str(stage_tag)))
    perm = rng.permutation(mask)
    if int(perm.sum()) != int(mask.sum()):
        raise RuntimeError("permuted agreement mask changed vote count")
    return {c: bool(perm[i]) for i, c in enumerate(cids)}


def context_arm_costs(base_cost, member_priors, tau, shuf_agree):
    """Compute all 5 H233 arms' cost vectors, gate decisions, and outcome-free signals for one context."""
    base = np.asarray(base_cost, dtype=np.float32)
    if len(member_priors) != MEMBERS:
        raise ValueError("3 member priors required")
    blended = [blend(base, p) for p in member_priors]
    b0 = blended[0]
    z = zgap(base, b0)
    z_open = gated(z, tau)
    unanimous, picks = unanimous_pick(blended)
    shuf_bool = bool(shuf_agree)
    opened = {
        "G025": bool(z_open),
        "STABLE_G025": bool(z_open and unanimous),
        "STABLE_SHUF": bool(z_open and shuf_bool),
        "BAG_VOTE_UNGATED": bool(unanimous),
    }
    costs = {
        "R882": base,
        "G025": b0 if opened["G025"] else base,
        "STABLE_G025": b0 if opened["STABLE_G025"] else base,
        "STABLE_SHUF": b0 if opened["STABLE_SHUF"] else base,
        "BAG_VOTE_UNGATED": b0 if opened["BAG_VOTE_UNGATED"] else base,
    }
    signal = dict(
        zgap=float(z),
        zgap_open=bool(z_open),
        unanimous=bool(unanimous),
        shuf_unanimous=shuf_bool,
        member_picks=list(picks),
        r882_pick=int(np.argmin(base)),
    )
    return costs, opened, signal


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
        bt_ratings(wins[:, draw].sum(axis=2))[..., 1], axis=0
    )  # (ELO_DRAWS,)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return dict(
        per_seed=[float(x) for x in elo], median=float(np.median(elo)), ci95=ci
    )


def joint_elo(panels, names, families, seed=BOOT_SEED + 1):
    if names[0] != "R882":
        raise ValueError("R882 must anchor index 0")
    wins = np.stack(
        [
            family_win_matrix(np.stack([panels[a][s] for a in names], axis=1), families)
            for s in range(len(panels["R882"]))
        ]
    )
    elo = bt_ratings(wins.sum(axis=1))  # (S, A)
    n = wins.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (ELO_DRAWS, n))
    boot = np.median(bt_ratings(wins[:, draw].sum(axis=2)), axis=0)  # (ELO_DRAWS, A)
    return {
        a: dict(
            per_seed=[float(x) for x in elo[:, i]],
            median=float(np.median(elo[:, i])),
            ci95=np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
        )
        for i, a in enumerate(names)
    }


def holm(p_by_arm):
    """Holm step-down adjusted p-values and rejection indicators at alpha = 0.05."""
    m = len(p_by_arm)
    order = sorted(p_by_arm, key=lambda a: (p_by_arm[a], a))
    adj = {}
    running = 0.0
    for i, a in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_by_arm[a]))
        adj[a] = running
    rejected = {}
    still = True
    for a in order:
        still = still and adj[a] < 0.05
        rejected[a] = still
    return dict(m=m, order=order, adjusted_p=adj, rejected_005=rejected)


# ------------------------------------------------------------------ gates and FIT-only threshold selection


def zgap(r882_cost, blend_cost):
    """Standardized r882 cost increase from switching r882's first-argmin pick to the blend's first-argmin pick.
    Zero when the two picks agree; small positive when the blend switches to a candidate r882 also scored nearly best;
    large when the blend switches to a candidate r882 scored poorly. Outcome-free (costs only)."""
    r = np.asarray(r882_cost, np.float64)
    b = np.asarray(blend_cost, np.float64)
    if r.shape != b.shape or r.ndim != 1 or not (np.isfinite(r).all() and np.isfinite(b).all()):
        raise ValueError("finite 1-D matching cost vectors required")
    ia, ib = int(np.argmin(r)), int(np.argmin(b))
    if ia == ib:
        return 0.0
    sd = float(np.std(r))
    if sd <= 1e-12:
        return math.inf
    return float((r[ib] - r[ia]) / sd)


def disagreement(gbm_cost, r882_cost):
    """H129 placebo statistic: std((GBM_cal - r882) / std(r882)) within context; outcome-free."""
    g = np.asarray(gbm_cost, np.float64)
    r = np.asarray(r882_cost, np.float64)
    if g.shape != r.shape or r.ndim != 1 or not (np.isfinite(g).all() and np.isfinite(r).all()):
        raise ValueError("finite 1-D matching cost vectors required")
    sd = float(np.std(r))
    if sd <= 1e-12:
        return math.inf
    return float(np.std((g - r) / sd))


def tau_grid(values):
    """Candidate thresholds from FIT contexts: -inf (never blend), quantiles among contexts where it is > 0 and
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


def verdict(primary_vs_r882, stable_vs_shuf, stable_vs_g025, secondary_val_vs_r882, plateau_result):
    """Preregistered admission rules (hw_t2205):
    STABLE_G025 (sole primary, Holm m=1) is admitted as a candidate iff on the 205-family pooled-A OOF contrast:
      1. median equal-family gain over 3 seeds vs R882 > +0.002
      2. family-bootstrap 10k CI95 lower bound > 0
      3. 3/3 positive seeds vs R882
      4. family harm fraction (< -0.01) <= 10%
      5. strictly beats STABLE_SHUF (median equal-family gain of STABLE_G025 - STABLE_SHUF > 0; ties = NO_ADVANCE)
      6. strictly beats G025 (median equal-family gain of STABLE_G025 - G025 > 0; ties = NO_ADVANCE)
    Secondary endpoint: DL-Val91 (19 families, 91 contexts) equal-family contrast + 15-cell plateau readout.
    """
    gates = dict(
        median_gain_gt_0002=bool(primary_vs_r882["median_gain"] > GAIN_MARGIN),
        lcb95_gt_0=bool(primary_vs_r882["ci95"][0] > 0.0),
        all_3_seeds_positive=bool(
            primary_vs_r882["positive_seeds"] == len(SEEDS)
            and len(primary_vs_r882["per_seed_gain"]) == len(SEEDS)
        ),
        harm_le_10pct=bool(primary_vs_r882["harm_fraction"] <= HARM_CAP),
        beats_stable_shuf=bool(stable_vs_shuf["median_gain"] > 0.0),
        beats_g025=bool(stable_vs_g025["median_gain"] > 0.0),
    )
    confirmed = all(gates.values())
    head = f"CANDIDATE: {PRIMARY}" if confirmed else "NO_ADVANCE"
    tail = "PLATEAU_WIN" if plateau_result["win"] else "NO_PLATEAU_WIN"
    return dict(
        verdict=head,
        combined=f"{head} + {tail}",
        primary=head,
        secondary_dl_val91=tail,
        holm_m=1,
        gates=gates,
        secondary_dl_val91_gates=dict(
            equal_family_median_gain_gt_0002=bool(
                secondary_val_vs_r882["median_gain"] > GAIN_MARGIN
            ),
            equal_family_lcb95_gt_0=bool(secondary_val_vs_r882["ci95"][0] > 0.0),
            plateau_context_mean_gain_gt_0002=bool(
                plateau_result["gain"] > PLATEAU_MARGIN
            ),
            plateau_lcb95_gt_0=bool(plateau_result["ci95"][0] > 0.0),
        ),
    )


def fold_manifest(rows, other_waves):
    """Registered per-seed family partitions (TRAIN identity metadata only: context and family ids), with the overlap
    against all earlier waves' registered partitions (H203-H231)."""
    fams = [r["family_id"] for r in rows]
    folds = {s: seed_folds(fams, s) for s in SEEDS}
    of = {s: {f: k for k, g in enumerate(folds[s]) for f in g} for s in SEEDS}
    earlier = {json.dumps(g) for other in other_waves.values() for g in other.values()}
    if len({json.dumps(g) for g in folds.values()}) != 3 or earlier & {
        json.dumps(g) for g in folds.values()
    }:
        raise ValueError(
            "H233 partitions must be distinct per seed and from earlier waves"
        )
    cross = cross_wave_jaccard(
        folds, {w: {str(s): g for s, g in o.items()} for w, o in other_waves.items()}
    )
    max_by_wave = {
        w: float(max(v["best_match"] for v in pairs.values()))
        for w, pairs in cross.items()
    }
    worst = max(
        list(max_by_wave.values())
        + [v["best_match"] for v in fold_jaccard(folds).values()]
    )
    if worst > JACCARD_MAX:
        raise ValueError(f"best-match fold Jaccard {worst} > {JACCARD_MAX}")
    return dict(
        wave="H233",
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
        max_best_match_jaccard_by_wave=max_by_wave,
        max_best_match_jaccard=float(worst),
        families_by_seed={str(s): g for s, g in folds.items()},
    )
