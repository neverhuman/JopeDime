"""Fixed archived-family inference. No dataset readers or model dependencies."""

import numpy as np


def exact_flip(d):
    d = np.asarray(d, dtype=float)
    n = d.shape[1]
    if n == 0:
        return None, None
    obs = d.mean(axis=1)
    counts = np.zeros(len(d), dtype=np.int64)
    pc = 0
    for start in range(0, 2**n, 8192):
        bits = (
            np.arange(start, min(start + 8192, 2**n), dtype=np.uint64)[:, None]
            >> np.arange(n, dtype=np.uint64)
        ) & 1
        vals = (bits.astype(float) * 2 - 1) @ d.T / n
        counts += (vals >= obs - 1e-14).sum(axis=0)
        pc += int((np.median(vals, axis=1) >= np.median(obs) - 1e-14).sum())
    return counts / (2**n), pc / (2**n)


def bt_two(losses, wins):
    a = b = 1.0
    games = losses + wins
    for _ in range(400):
        b = float(np.clip(((wins + 0.01) / ((games + 0.02) / (a + b))) / a, 1e-6, 1e6))
        a = 1.0
    return float(1500 + 400 * np.log10(b))


def interval(point, values):
    q = np.quantile(values - point, [0.025, 0.95, 0.975])
    return dict(
        ci95=[float(point - q[2]), float(point - q[0])],
        lcb95_one_sided=float(point - q[1]),
    )


def summarize(d):
    d = np.asarray(d, dtype=float)
    n = d.shape[1]
    if n == 0:
        return dict(
            families=0,
            per_seed=[],
            pooled=None,
            unmeasured_reason="empty group; no zero imputation",
        )
    p, pp = exact_flip(d)
    points = d.mean(axis=1)
    signs = np.random.default_rng(314150).choice([-1.0, 1.0], size=(10000, n))
    boot = signs @ (d - points[:, None]).T / n + points
    rows = []
    for s in range(6):
        x = d[s]
        w = int((x > 1e-4).sum())
        l = int((x < -1e-4).sum())
        rows.append(
            dict(
                root_seed=30001 + s,
                gain=float(points[s]),
                family_median_gain=float(np.median(x)),
                wins=w,
                ties=int((abs(x) <= 1e-4).sum()),
                losses=l,
                harmed=int((x < -0.01).sum()),
                elo=bt_two(l, w),
                exact_signflip_p=float(p[s]),
                wild=interval(points[s], boot[:, s]),
                drop_top={
                    str(k): float(np.mean(np.sort(x)[:-k])) if n > k else None
                    for k in [1, 5]
                },
            )
        )
    point = float(np.median(points))
    pooled = dict(
        gain=point,
        median_family_median_gain=float(np.median(np.median(d, axis=1))),
        exact_signflip_p=float(pp),
        wild=interval(point, np.median(boot, axis=1)),
        drop_top={
            str(k): float(np.median([r["drop_top"][str(k)] for r in rows]))
            if n > k
            else None
            for k in [1, 5]
        },
        elo=float(np.median([r["elo"] for r in rows])),
        median_counts={
            key: float(np.median([r[key] for r in rows]))
            for key in ["wins", "ties", "losses", "harmed"]
        },
    )
    return dict(
        families=n,
        per_seed=rows,
        pooled=pooled,
        signflip_assignments=2**n,
        wild_rng=314150,
        wild_draws=10000,
        degenerate_single_family=n == 1,
        count_note="Pooled counts are componentwise medians across dependent seed policies, not a seventh policy.",
    )


def decide(groups):
    clean = groups["CLEAN"]
    neigh = groups["NEIGHBORED"]["families"]
    pool = clean["pooled"]
    count = sum(r["wild"]["ci95"][0] > 0 for r in clean["per_seed"])
    confounded = bool(
        neigh >= 1
        and pool is not None
        and (pool["gain"] <= 0.001 or pool["wild"]["ci95"][0] <= 0)
    )
    independent = bool(
        clean["families"] >= 10
        and pool is not None
        and pool["gain"] > 0.001
        and count >= 4
    )
    verdict = (
        "H310_LINEAGE_CONFOUNDED"
        if confounded
        else "H310_LINEAGE_INDEPENDENT"
        if independent
        else "INCONCLUSIVE"
    )
    reasons = []
    if verdict == "INCONCLUSIVE":
        if clean["families"] < 10:
            reasons.append("CLEAN<10")
        if pool is None:
            reasons.append("CLEAN outcome unmeasured because group is empty")
        elif pool["gain"] <= 0.001:
            reasons.append("CLEAN pooled gain<=.001")
        if count < 4:
            reasons.append("CLEAN positive CI lower endpoints on fewer than4/6")
    return dict(
        verdict=verdict,
        confounded_clause=confounded,
        independent_clause=independent,
        clean_positive_wild_ci_seed_count=count,
        decision_bound="two-sided95% basic CI lower endpoint",
        reasons=reasons,
    )
