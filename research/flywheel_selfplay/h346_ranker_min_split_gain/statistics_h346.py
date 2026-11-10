"""Paired family resampling shared across all nine dependent partition/seed cells."""

import numpy as np


def drop(x, k):
    return np.sort(x, axis=-1)[..., : x.shape[-1] - k].mean(axis=-1)


def elo(w, l):
    a = b = 1.0
    for _ in range(400):
        b = float(np.clip(((w + 0.01) / ((w + l + 0.02) / (a + b))) / a, 1e-6, 1e6))
        a = 1.0
    return float(1500 + 400 * np.log10(b))


def elo_draws(w, l):
    """Same 400-step MM recurrence, vectorized over shared family draws."""
    w, l = np.asarray(w, dtype=float), np.asarray(l, dtype=float)
    b = np.ones_like(w)
    for _ in range(400):
        b = np.clip((w + .01) * (1 + b) / (w + l + .02), 1e-6, 1e6)
    return 1500 + 400 * np.log10(b)


def decide(paired, cap, base):
    assert len(paired["per_seed_medians"]) == 3
    c = dict(
        seed_consistency=bool(sum(x > 0 for x in paired["per_seed_medians"]) >= 2),
        paired_drop10_margin=bool(paired["median"] >= 0.002 and paired["lcb95"] > 0),
        gain_retention_90percent=bool(cap["median_gain"] >= 0.9 * base["median_gain"]),
        harmed_nonincrease=bool(
            cap["median_harmed_rate"] <= base["median_harmed_rate"]
        ),
    )
    return dict(
        verdict="MINGAIN_ADVANCE" if all(c.values()) else "MINGAIN_NOT_ADVANCED",
        criteria=c,
        failed_clauses=[k for k, v in c.items() if not v],
    )


def analyze(base, cap, draws=10000, rng=346191):
    assert base.shape == cap.shape and base.ndim == 2
    n = base.shape[1]
    ix = np.random.default_rng(rng).integers(0, n, size=(draws, n))
    summaries = {}
    boot_by = {}
    drop_by = {}
    for arm, d in [("BASE", base), ("MINGAIN", cap)]:
        boots = np.array([v[ix].mean(axis=1) for v in d])
        family_medians = np.array([np.median(v[ix],axis=1) for v in d])
        elo_boot = np.array([elo_draws((v[ix]>1e-4).sum(axis=1),
                                      (v[ix]<-1e-4).sum(axis=1)) for v in d])
        boot_by[arm] = boots
        drop_by[arm] = np.array(
            [np.sort(v[ix], axis=1)[:, :-10].mean(axis=1) for v in d]
        )
        rows = []
        for i, x in enumerate(d):
            w = int((x > 1e-4).sum())
            l = int((x < -1e-4).sum())
            ties = n - w - l
            rows.append(
                dict(
                    family_mean_gain=float(x.mean()),
                    family_median_gain=float(np.median(x)),
                    family_median_ci95=np.quantile(family_medians[i],[.025,.975]).tolist(),
                    ci95=np.quantile(boots[i], [0.025, 0.975]).tolist(),
                    wins=w,
                    ties=ties,
                    losses=l,
                    harmed=int((x < -0.01).sum()),
                    harmed_rate=float(np.mean(x < -0.01)),
                    drop_top={str(k): float(drop(x, k)) for k in [1, 5, 10]},
                    elo=elo(w, l),
                    elo_ci95=np.quantile(elo_boot[i],[.025,.975]).tolist(),
                )
            )
        median_boot = np.median(boots, axis=0)
        summaries[arm] = dict(
            cells=rows,
            aggregate=dict(
                median_gain=float(np.median(d.mean(axis=1))),
                median_family_median_gain=float(np.median(np.median(d, axis=1))),
                median_family_median_ci95=np.quantile(np.median(family_medians,axis=0),[.025,.975]).tolist(),
                ci95=np.quantile(median_boot, [0.025, 0.975]).tolist(),
                lcb95=float(np.quantile(median_boot, 0.05)),
                median_harmed_rate=float(np.median(np.mean(d < -0.01, axis=1))),
                median_harmed=float(np.median(np.sum(d < -0.01, axis=1))),
                median_elo=float(np.median([r["elo"] for r in rows])),
                median_elo_ci95=np.quantile(np.median(elo_boot,axis=0),[.025,.975]).tolist(),
                median_counts={
                    k: float(np.median([r[k] for r in rows]))
                    for k in ["wins", "ties", "losses"]
                },
                drop_top={str(k): float(np.median(drop(d, k))) for k in [1, 5, 10]},
            ),
        )
        if len(d)==9:
            summaries[arm]['per_root']={str(30001+s):dict(
                median_gain=float(np.median(d[s::3].mean(axis=1))),
                ci95=np.quantile(np.median(boots[s::3],axis=0),[.025,.975]).tolist(),
                median_family_median_gain=float(np.median(np.median(d[s::3],axis=1))),
                family_median_ci95=np.quantile(np.median(family_medians[s::3],axis=0),[.025,.975]).tolist(),
                median_elo=float(np.median([rows[c]['elo'] for c in range(s,9,3)])),
                elo_ci95=np.quantile(np.median(elo_boot[s::3],axis=0),[.025,.975]).tolist()
            ) for s in range(3)}
    per = drop(cap, 10) - drop(base, 10)
    dist = np.median(drop_by["MINGAIN"] - drop_by["BASE"], axis=0)
    paired = dict(
        per_cell=per.tolist(),
        median=float(np.median(per)),
        ci95=np.quantile(dist, [0.025, 0.975]).tolist(),
        lcb95=float(np.quantile(dist, 0.05)),
        rng=rng,
        draws=draws,
        definition="difference of arm-specific drop-top10 means, then median across9 cells; paired family resamples",
    )
    paired["per_seed_medians"] = np.median(per.reshape(3,3),axis=0).tolist() if len(per)==9 else None
    if len(per)==9:
        delta_boot=drop_by['MINGAIN']-drop_by['BASE']
        paired['per_seed_ci95']=[np.quantile(np.median(delta_boot[s::3],axis=0),[.025,.975]).tolist() for s in range(3)]
    delta = cap - base
    alt = np.median(
        np.array([np.sort(v[ix], axis=1)[:, :-10].mean(axis=1) for v in delta]), axis=0
    )
    secondary = dict(
        per_cell=drop(delta, 10).tolist(),
        median=float(np.median(drop(delta, 10))),
        ci95=np.quantile(alt, [0.025, 0.975]).tolist(),
        lcb95=float(np.quantile(alt, 0.05)),
        decision_authority=False,
    )
    return summaries, paired, secondary, ix


def config_status(matches):
    assert 0 <= matches <= 36
    return "CONFIG_INSENSITIVE" if matches >= 30 else "CONFIG_SENSITIVE"
