"""Shared whole-family bootstrap of all nine frozen policies; no policy selection."""

import numpy as np
from analysis_h302 import bt_two

RNG = 356191
DRAWS = 10000
TIE = 1e-4


def decide(primary):
    criteria = dict(
        median_elo_vs_r3175_gt_1500=bool(primary["median_elo"] > 1500.0),
        paired_family_lift_lcb95_gt_zero=bool(primary["paired_lift_lcb95"] > 0.0),
    )
    return criteria, "NL15_DEV91_CONFIRMED" if all(
        criteria.values()
    ) else "NL15_DEV91_NOT_CONFIRMED"


def summarize(candidate, baseline, families, draws=DRAWS, rng=RNG):
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    assert candidate.ndim == 2 and candidate.shape[1] == len(families) == len(baseline)
    names = sorted(set(families))
    f = np.asarray(families)
    diffs = np.stack(
        [
            np.array(
                [(u[f == family] - baseline[f == family]).mean() for family in names]
            )
            for u in candidate
        ]
    )
    ix = np.random.default_rng(rng).integers(0, len(names), (draws, len(names)))
    boot = diffs[:, ix].mean(axis=2)
    literal = np.median(diffs[:, ix], axis=2)
    w = (diffs > TIE).sum(axis=1)
    losses = (diffs < -TIE).sum(axis=1)
    t = len(names) - w - losses
    elo = np.asarray([bt_two(float(loss), float(win)) for win, loss in zip(w, losses)])
    wb = (diffs[:, ix] > TIE).sum(axis=2)
    lb = (diffs[:, ix] < -TIE).sum(axis=2)
    # Same smoothed two-player BT bootstrap formula as frozen analysis_h302.
    elo_boot = 1500 + 400 * np.log10((wb + 0.01) / (lb + 0.01))
    pooled = np.median(boot, axis=0)
    pooled_elo = np.median(elo_boot, axis=0)
    cells = []
    for i, d in enumerate(diffs):
        cells.append(
            dict(
                policy=i,
                family_equal_mean_lift=float(d.mean()),
                family_median_lift=float(np.median(d)),
                lift_ci95=np.quantile(boot[i], [0.025, 0.975]).tolist(),
                lift_lcb95=float(np.quantile(boot[i], 0.05)),
                family_median_ci95=np.quantile(literal[i], [0.025, 0.975]).tolist(),
                elo=float(elo[i]),
                elo_ci95=np.quantile(elo_boot[i], [0.025, 0.975]).tolist(),
                family_wtl=[int(w[i]), int(t[i]), int(losses[i])],
                harmed_share=float(np.mean(d < -0.01)),
            )
        )
    roots = []
    if len(diffs) == 9:
        for s, root in enumerate((30001, 30002, 30003)):
            selected = [s, s + 3, s + 6]
            b = np.median(boot[selected], axis=0)
            eb = np.median(elo_boot[selected], axis=0)
            roots.append(
                dict(
                    root_seed=root,
                    median_lift=float(np.median(diffs[selected].mean(axis=1))),
                    lift_ci95=np.quantile(b, [0.025, 0.975]).tolist(),
                    lift_lcb95=float(np.quantile(b, 0.05)),
                    median_elo=float(np.median(elo[selected])),
                    elo_ci95=np.quantile(eb, [0.025, 0.975]).tolist(),
                )
            )
    return dict(
        median_family_equal_lift=float(np.median(diffs.mean(axis=1))),
        paired_lift_ci95=np.quantile(pooled, [0.025, 0.975]).tolist(),
        paired_lift_lcb95=float(np.quantile(pooled, 0.05)),
        literal_family_median_lift=float(np.median(np.median(diffs, axis=1))),
        literal_family_median_ci95=np.quantile(
            np.median(literal, axis=0), [0.025, 0.975]
        ).tolist(),
        median_elo=float(np.median(elo)),
        median_elo_ci95=np.quantile(pooled_elo, [0.025, 0.975]).tolist(),
        median_elo_lcb95=float(np.quantile(pooled_elo, 0.05)),
        per_policy=cells,
        per_root=roots,
        families=len(names),
        dependent_policies=len(diffs),
        bootstrap=dict(
            draws=draws,
            rng=rng,
            unit="whole family; same draw shared across all nine policies; no seed independence claim",
        ),
    )
