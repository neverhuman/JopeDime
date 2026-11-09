"""Frozen H297 paired family bootstrap, concentration and MM Bradley-Terry."""

import numpy as np

ARMS = ("r3175", "CONTROL", "CTXNORM_CLIP", "CTXNORM_CLIP_NOSMOOTH")
SEEDS = (29701, 29702, 29703)
BOOT_SEED = 29750
DRAWS = 10000
TIE = 1e-4


def bt_mm(wins):
    w = np.asarray(wins, dtype=float) + 0.01
    i = np.arange(w.shape[-1])
    w[..., i, i] = 0
    n = w + w.swapaxes(-1, -2)
    total = w.sum(-1)
    s = np.ones_like(total)
    for _ in range(5000):
        new = total / (n / (s[..., :, None] + s[..., None, :])).sum(-1)
        new /= new[..., :1]
        if np.max(np.abs(np.log(new / s))) < 1e-10:
            return 1500 + 400 * np.log10(new)
        s = new
    raise RuntimeError("MM nonconvergence")


def summarize(results, baseline, families):
    panel = np.empty((3, 91, 4))
    means = np.empty((3, 4))
    changed = np.zeros((3, 4), int)
    for si, seed in enumerate(SEEDS):
        panel[si, :, 0] = baseline["per_context"]
        means[si, 0] = baseline["mean"]
        for ai, arm in enumerate(ARMS[1:], 1):
            r = next(r for r in results if r["seed"] == seed and r["arm"] == arm)
            panel[si, :, ai] = r["per_context"]
            means[si, ai] = r["mean"]
            changed[si, ai] = sum(
                a != b
                for a, b in zip(r["selected_programs"], baseline["selected_programs"])
            )
    assert np.isfinite(panel).all()
    families = np.asarray(families)
    ids = sorted(set(families))
    assert len(ids) == 19
    groups = [np.flatnonzero(families == f) for f in ids]
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(BOOT_SEED)
    ix = rng.integers(0, 19, (DRAWS, 19))
    counts = np.zeros((DRAWS, 19))
    np.add.at(counts, (np.arange(DRAWS)[:, None], ix), 1)
    den = counts @ sizes
    elo = []
    eboot = []
    for si in range(3):
        d = panel[si, :, :, None] - panel[si, :, None, :]
        w = np.where(d > TIE, 1.0, np.where(d < -TIE, 0.0, 0.5))
        elo.append(bt_mm(w.sum(0)))
        fw = np.stack([w[g].sum(0) for g in groups])
        eboot.append(bt_mm(np.einsum("df,fab->dab", counts, fw, optimize=True)))
    elo = np.stack(elo)
    eboot = np.stack(eboot)
    arms = {}
    for ai, arm in enumerate(ARMS):
        delta = panel[:, :, ai] - panel[:, :, 0]
        sums = np.stack([delta[:, g].sum(1) for g in groups], axis=1)
        boot = np.einsum("df,sf->sd", counts, sums, optimize=True) / den
        gain = means[:, ai] - means[:, 0]
        rows = []
        for si, seed in enumerate(SEEDS):
            positive = np.maximum(delta[si], 0)
            positive_sum = float(positive.sum())
            share = (
                float(np.sort(positive)[-4:].sum() / positive_sum)
                if positive_sum > 0
                else None
            )
            ci = np.quantile(boot[si], [0.025, 0.975])
            criteria = dict(
                mean_exceeds_baseline_plus_0_001=bool(
                    means[si, ai] > baseline["mean"] + 0.001
                ),
                mean_exceeds_paired_control=bool(means[si, ai] > means[si, 1]),
                paired_family_bootstrap_lcb_positive=bool(ci[0] > 0),
                top4_positive_gain_share_lt_0_8=bool(share is not None and share < 0.8),
            )
            rows.append(
                dict(
                    seed=seed,
                    mean=float(means[si, ai]),
                    paired_control_mean=float(means[si, 1]),
                    lift=float(gain[si]),
                    ci95=ci.tolist(),
                    context_wtl=dict(
                        wins=int((delta[si] > TIE).sum()),
                        ties=int((np.abs(delta[si]) <= TIE).sum()),
                        losses=int((delta[si] < -TIE).sum()),
                    ),
                    changed_contexts=int(changed[si, ai]),
                    top4_positive_gain_share=share,
                    positive_gain_sum=positive_sum,
                    elo=float(elo[si, ai]),
                    elo_ci95=np.quantile(eboot[si, :, ai], [0.025, 0.975]).tolist(),
                    criteria=criteria,
                    passes=all(criteria.values()),
                )
            )
        arms[arm] = dict(
            by_seed=rows,
            median_lift=float(np.median(gain)),
            median_lift_ci95=np.quantile(
                np.median(boot, axis=0), [0.025, 0.975]
            ).tolist(),
            median_elo=float(np.median(elo[:, ai])),
            median_elo_ci95=np.quantile(
                np.median(eboot[:, :, ai], axis=0), [0.025, 0.975]
            ).tolist(),
        )
    verdict = (
        "DEV_ADVANCE"
        if all(r["passes"] for r in arms["CTXNORM_CLIP"]["by_seed"])
        else "NOT_ADVANCED"
    )
    return dict(
        status="COMPLETE",
        verdict=verdict,
        primary="CTXNORM_CLIP",
        arms=arms,
        bootstrap=dict(
            seed=BOOT_SEED,
            draws=DRAWS,
            families=19,
            unit="paired context deltas, resample whole families with replacement; context-weighted ratio",
            median_ci="same family draws across seeds; percentile of per-draw median seed lift",
        ),
        elo=dict(
            method="MM Bradley-Terry",
            unit="context",
            anchor=1500,
            tie_band=TIE,
            symmetric_pseudocount=0.01,
        ),
        adoption=False,
        fresh_generalization=False,
        optimizer_updates=72,
        regression_updates=2,
    ), panel
