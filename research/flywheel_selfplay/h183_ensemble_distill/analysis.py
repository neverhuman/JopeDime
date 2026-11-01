"""H183 fixed-fold distillation estimands and unchanged strict primary gate."""

import math
import numpy as np
from parents import numerical

BASE = numerical()
SEEDS = (183101, 183102, 183103)
STUDENT_SEEDS = (183501, 183502, 183503)
FOLD_SEED, BOOT_SEED = 183140, 183141
ARMS = ("R882", "FRESH1", "RENS10", "ENS10", "DIST10", "DIST10_80")
PRIMARY_FAMILY = ("DIST10", "DIST10_80")
DRAWS, EPS, MARGIN = 10000, 1e-12, 0.001
stream_seed = BASE.stream_seed
batch_plan = BASE.batch_plan
validate_fit_plan = BASE.validate_fit_plan
context_utility = BASE.context_utility
family_win_matrix = BASE.family_win_matrix
bt_ratings = BASE.bt_ratings
holm_adjust = BASE.holm_adjust
paired_family_bootstrap = BASE.paired_family_bootstrap


def family_folds(family_ids):
    return BASE.family_folds(family_ids, seed=FOLD_SEED)


def partition(family_ids, outer, folds=None):
    return BASE.partition(
        family_ids, outer, family_folds(family_ids) if folds is None else folds
    )


def teacher_seeds(seed):
    if seed not in SEEDS:
        raise ValueError("unknown H183 seed")
    return tuple(183200 + 100 * SEEDS.index(seed) + i for i in range(1, 11))


def student_seed(seed):
    return STUDENT_SEEDS[SEEDS.index(seed)]


def execution_plan(cell_seconds, cap=14400.0, reserve=1200.0):
    if not math.isfinite(cell_seconds) or cell_seconds <= 0:
        raise ValueError("positive finite FIT-only cell timing required")
    for n in (5, 3):
        projection = n * len(SEEDS) * cell_seconds + reserve
        if projection <= cap:
            return dict(
                outer_folds=n,
                seeds=3,
                projected_seconds=projection,
                within_cap=True,
                assessed_families=n * 41,
            )
    return dict(
        outer_folds=3,
        seeds=3,
        projected_seconds=9 * cell_seconds + reserve,
        within_cap=False,
        assessed_families=123,
    )


def _ci(x):
    return np.quantile(x, [0.025, 0.975], axis=0).tolist()


def summarize(context_panel, family_ids, agreement=None, draws=DRAWS):
    panel = np.asarray(context_panel, dtype=np.float64)
    if panel.shape != (3, len(family_ids), len(ARMS)) or not np.isfinite(panel).all():
        raise ValueError("complete three-seed/context/arm utility panel required")
    fams = sorted(set(family_ids))
    masks = [np.asarray(family_ids) == f for f in fams]
    fp = np.stack([panel[:, mask, :].mean(axis=1) for mask in masks], axis=1)
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(fams), (draws, len(fams)))
    out = dict(
        arms={},
        gain_vs_r882={},
        contrasts={},
        elo_vs_r882={},
        assessed_contexts=len(family_ids),
        assessed_families=len(fams),
        family_bootstrap_draws=draws,
        seed_ids=list(SEEDS),
    )
    for j, arm in enumerate(ARMS):
        mean = fp[:, :, j].mean(axis=1)
        boot = np.median(fp[:, ix, j].mean(axis=2), axis=0)
        out["arms"][arm] = dict(
            per_seed_equal_family_utility=mean.tolist(),
            median_over_seeds=float(np.median(mean)),
            ci95=_ci(boot),
            minus2_count_by_seed=(panel[:, :, j] == -2.0).sum(axis=1).tolist(),
        )
    boots = {}
    pairs = [(a + "_minus_R882", a, "R882") for a in ARMS[1:]]
    pairs += [
        (a + "_minus_" + b, a, b)
        for a, b in [
            ("ENS10", "RENS10"),
            ("DIST10", "RENS10"),
            ("DIST10_80", "RENS10"),
            ("DIST10", "ENS10"),
            ("DIST10_80", "ENS10"),
            ("DIST10", "FRESH1"),
        ]
    ]
    for name, a, b in pairs:
        delta = fp[:, :, ARMS.index(a)] - fp[:, :, ARMS.index(b)]
        per_seed = delta.mean(axis=1)
        point = float(np.median(per_seed))
        boot = paired_family_bootstrap(delta, ix)
        record = dict(
            per_seed=per_seed.tolist(),
            median_over_seeds=point,
            ci95=_ci(boot),
            p_one_sided_centered=float(
                (1 + (boot - point >= point).sum()) / (draws + 1)
            ),
        )
        out["contrasts"][name] = record
        boots[name] = boot
        if b == "R882":
            out["gain_vs_r882"][a] = record
    adj = holm_adjust(
        [out["gain_vs_r882"][a]["p_one_sided_centered"] for a in PRIMARY_FAMILY]
    )
    for a, p in zip(PRIMARY_FAMILY, adj, strict=True):
        out["gain_vs_r882"][a]["holm_m2_adjusted_p"] = float(p)
    wins = np.stack([family_win_matrix(panel[s], family_ids) for s in range(3)])
    ratings = bt_ratings(wins.sum(axis=1))
    elo_boot = []
    for start in range(0, draws, 100):
        resampled = np.take(wins, ix[start : start + 100], axis=1).sum(axis=2)
        elo_boot.append(np.median(bt_ratings(resampled), axis=0))
    elo_ci = _ci(np.concatenate(elo_boot, axis=0))
    for j, a in enumerate(ARMS):
        out["elo_vs_r882"][a] = dict(
            per_seed=ratings[:, j].tolist(),
            median_over_seeds=float(np.median(ratings[:, j])),
            ci95=[elo_ci[0][j], elo_ci[1][j]],
        )
    teacher = out["gain_vs_r882"]["ENS10"]
    out["retained_teacher_gain"] = {}
    for arm in PRIMARY_FAMILY:
        gain = out["gain_vs_r882"][arm]
        ratio = (
            gain["median_over_seeds"] / teacher["median_over_seeds"]
            if teacher["median_over_seeds"] > 0
            else None
        )
        stable = teacher["ci95"][0] > 0
        ratio_ci = (
            _ci(boots[arm + "_minus_R882"] / boots["ENS10_minus_R882"])
            if stable
            else None
        )
        out["retained_teacher_gain"][arm] = dict(
            point_ratio=ratio,
            ci95=ratio_ci,
            stable_teacher_denominator=stable,
            reason=None
            if stable
            else "teacher gain CI crosses zero; retention interval is unstable",
        )
    if agreement is not None:
        out["student_teacher_agreement"] = {}
        for arm, v in agreement.items():
            v = np.asarray(v, dtype=float)
            if v.shape != (3, len(family_ids)):
                raise ValueError("agreement context roster mismatch")
            fm = np.stack([v[:, mask].mean(axis=1) for mask in masks], axis=1)
            boot = paired_family_bootstrap(fm, ix)
            out["student_teacher_agreement"][arm] = dict(
                per_seed=fm.mean(axis=1).tolist(),
                median_over_seeds=float(np.median(fm.mean(axis=1))),
                ci95=_ci(boot),
                per_seed_context_mean=v.mean(axis=1).tolist(),
            )
    return decide(out)


def decide(summary):
    for arm in PRIMARY_FAMILY:
        g = summary["gain_vs_r882"][arm]
        g["advance_candidate"] = bool(
            g["median_over_seeds"] > MARGIN + EPS
            and g["ci95"][0] > 0
            and all(x > 0 for x in g["per_seed"])
            and g["holm_m2_adjusted_p"] < 0.05
        )
    summary["verdict"] = (
        "ADVANCE_CANDIDATE"
        if summary["gain_vs_r882"]["DIST10"]["advance_candidate"]
        else "NO_ADVANCE"
    )
    summary["secondary_only"] = bool(
        summary["gain_vs_r882"]["DIST10_80"]["advance_candidate"]
        and not summary["gain_vs_r882"]["DIST10"]["advance_candidate"]
    )
    summary["H175_power_reference"] = dict(
        measured_family_clustered_41_B_MDE80=0.09467790954714358,
        heuristic_205_family_MDE80=0.04234124834299788,
        measured_for_H183=False,
        scope="Conditional oracle-bank/planning references, not a measured power or ceiling conclusion for H183.",
    )
    summary["diagnostics_unadjusted"] = True
    summary["serving_promotion"] = False
    summary["promotion_requires_H177_H178_qualification"] = True
    return summary
