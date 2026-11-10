"""Preregistered 19-family sensitivity inference; shared family draws across roots."""

import numpy as np
from scipy.stats import t


def exact_flip(d):
    d = np.asarray(d, dtype=float)
    n = d.shape[1]
    obs = d.mean(axis=1)
    pool = float(np.median(obs))
    counts = np.zeros(len(d), dtype=np.int64)
    pc = 0
    for start in range(0, 2**n, 8192):
        bits = (
            np.arange(start, min(start + 8192, 2**n), dtype=np.uint64)[:, None]
            >> np.arange(n, dtype=np.uint64)
        ) & 1
        values = (bits.astype(np.float64) * 2 - 1) @ d.T / n
        counts += (values >= obs - 1e-14).sum(axis=0)
        pc += int((np.median(values, axis=1) >= pool - 1e-14).sum())
    return counts / (2**n), pc / (2**n)


def jk(values, point, df):
    values = np.asarray(values)
    n = len(values)
    se = float(np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2)))
    return dict(
        min_gain=float(values.min()),
        se=se,
        ci95=[
            float(point - t.ppf(0.975, df) * se),
            float(point + t.ppf(0.975, df) * se),
        ],
    )


def wild_summary(point, samples):
    delta = np.asarray(samples) - point
    q = np.quantile(delta, [0.025, 0.95, 0.975])
    return dict(
        ci95=[float(point - q[2]), float(point - q[0])], lcb95=float(point - q[1])
    )


def analyze(d):
    d = np.asarray(d, dtype=float)
    ns, n = d.shape
    assert n >= 2
    points = d.mean(axis=1)
    pooled = float(np.median(points))
    p, pp = exact_flip(d)
    signs = np.random.default_rng(312150).choice([-1.0, 1.0], size=(10000, n))
    draws = signs @ (d - points[:, None]).T / n + points
    lofo = (d.sum(axis=1)[:, None] - d) / (n - 1)
    rows = []
    for i in range(ns):
        se = float(d[i].std(ddof=1) / np.sqrt(n))
        j = jk(lofo[i], points[i], n - 1)
        desc = np.sort(d[i], kind="stable")[::-1]
        rows.append(
            dict(
                gain=float(points[i]),
                exact_signflip_p=float(p[i]),
                signflip_assignments=2**n,
                t_df=n - 1,
                t_se=se,
                t_ci95=[
                    float(points[i] - t.ppf(0.975, n - 1) * se),
                    float(points[i] + t.ppf(0.975, n - 1) * se),
                ],
                t_lcb95=float(points[i] - t.ppf(0.95, n - 1) * se),
                wild=wild_summary(points[i], draws[:, i]),
                jackknife=j,
                drop_top={str(k): float(desc[k:].mean()) for k in (1, 5, 10) if k < n},
                wins=int((d[i] > 1e-4).sum()),
                ties=int((abs(d[i]) <= 1e-4).sum()),
                losses=int((d[i] < -1e-4).sum()),
                harmed=int((d[i] < -0.01).sum()),
                family_median=float(np.median(d[i])),
            )
        )
    pj = jk(np.median(lofo, axis=0), pooled, n - 1)
    pooled_result = dict(
        gain=pooled,
        exact_signflip_p=float(pp),
        signflip_assignments=2**n,
        t_df=n - 1,
        t_se=pj["se"],
        t_ci95=pj["ci95"],
        t_lcb95=float(pooled - t.ppf(0.95, n - 1) * pj["se"]),
        jackknife=pj,
        wild=wild_summary(pooled, np.median(draws, axis=1)),
        drop_top={
            str(k): float(np.median([r["drop_top"][str(k)] for r in rows]))
            for k in (1, 5, 10)
            if k < n
        },
        note="Median across dependent seeds. Pooled t interval uses jackknife SE of nonlinear median; descriptive approximation.",
    )
    return rows, pooled_result


def decision(rows, overlap, nonoverlap_median):
    criteria = dict(
        a_fresh3_median_gt_001=bool(np.median([x["gain"] for x in rows[3:]]) > 0.001),
        b_exact_p_lt05_ge4=sum(x["exact_signflip_p"] < 0.05 for x in rows) >= 4,
        c_t18_lcb_positive_ge4=sum(x["t_lcb95"] > 0 for x in rows) >= 4,
        d_lofo_min_positive_ge4=sum(x["jackknife"]["min_gain"] > 0 for x in rows) >= 4,
        e_nonoverlap_positive=bool(
            not overlap or nonoverlap_median is not None and nonoverlap_median > 0
        ),
    )
    return dict(
        verdict="H310_SMALLCLUSTER_ROBUST"
        if all(criteria.values())
        else "H310_FRAGILE",
        criteria=criteria,
        failed_clauses=[k for k, v in criteria.items() if not v],
    )
