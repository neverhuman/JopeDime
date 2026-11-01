"""H181 fixed-cohort cross-fit control variates, paired bootstrap and calibration."""

import hashlib
import numpy as np
from scipy.stats import beta

SEEDS = (181101, 181102, 181103)
BANK_SEEDS = (175101, 175102, 175103)
FOLD_SEED, BOOT_SEED = 181140, 181141
METHODS = ("STRICT", "CUPED1", "CUPED_K", "POSTSTRAT")
FRACTIONS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
TARGETS = (0.016, 0.028, 0.042)
DRAWS, BOOTSTRAPS = 200, 10000
EPS, MARGIN = 1e-12, 0.001


def stream(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def groups(n=205):
    if n != 205:
        raise ValueError("205 TRAIN families required")
    return [
        np.sort(x)
        for x in np.array_split(np.random.default_rng(FOLD_SEED).permutation(n), 5)
    ]


def roles(group, outer):
    a, b = group[outer], group[(outer + 1) % 5]
    fit = np.setdiff1d(np.arange(sum(map(len, group))), np.r_[a, b])
    if set(fit) & set(a) | set(fit) & set(b) | set(a) & set(b):
        raise ValueError("family role overlap")
    return fit, b, a


def slate_indices(cid, n, slate, seed):
    if slate == "full":
        return np.arange(n)
    if slate != "24" or n < 24:
        raise ValueError("invalid slate")
    return np.r_[
        np.arange(4),
        np.random.default_rng(stream(seed, cid, "slate24")).choice(
            np.arange(4, n), 20, replace=False
        ),
    ]


def family_mean(values, rows):
    labels = np.asarray([r["family_id"] for r in rows])
    fams = sorted(set(labels))
    value = np.asarray(values, dtype=np.float64)
    if value.shape[-1] != len(rows) or not np.isfinite(value).all():
        raise ValueError("finite context panel required")
    return np.stack([value[..., labels == f].mean(axis=-1) for f in fams], axis=-1)


def covariates(rows, costs, utilities, slate, bank_seed):
    """ONLY frozen r882 + fixed data; no candidate output argument or access."""
    fams = sorted({r["family_id"] for r in rows})
    base = []
    margin = []
    for row in rows:
        cid = row["context_sha256"]
        ix = slate_indices(cid, len(costs[cid]), slate, bank_seed)
        cost = np.asarray(costs[cid][ix], dtype=np.float64)
        u = np.asarray(utilities[cid][ix], dtype=np.float64)
        if not np.isfinite(cost).all() or not np.isfinite(u).all():
            raise ValueError("nonfinite input")
        base.append(float(u[np.argmin(cost)]))
        two = np.partition(cost, 1)[:2]
        margin.append(float(two.max() - two.min()))
    base = np.array(base)
    margin = np.array(margin)
    labels = np.array([r["family_id"] for r in rows])
    return np.array(
        [
            [
                base[labels == f].mean(),
                np.sum(base[labels == f] == -2.0),
                margin[labels == f].mean(),
                np.log(np.sum(labels == f)),
            ]
            for f in fams
        ]
    )


def strata(X):
    order = np.lexsort((np.arange(len(X)), X[:, 0]))
    result = np.empty(len(X), dtype=int)
    for h, ix in enumerate(np.array_split(order, 5)):
        result[ix] = h
    return result


def fit_coefficient(X, Y, fit, columns):
    """FIT only, centered OLS/minimum norm. No optimizer, no evaluated Y."""
    xf = np.array(X[np.asarray(fit)][:, columns], copy=True)
    scale = xf.std(axis=0)
    active = scale > 1e-12
    design = np.zeros_like(xf)
    design[:, active] = (xf[:, active] - xf[:, active].mean(axis=0)) / scale[active]
    response = np.array(Y[:, fit], copy=True)
    response -= response.mean(axis=1, keepdims=True)
    theta = (np.linalg.pinv(design, rcond=1e-10) @ response.T).T
    theta[:, active] /= scale[active]
    theta[:, ~active] = 0
    return theta


def adjusted_panel(X, Y, group):
    if X.shape != (205, 4) or Y.ndim != 2 or Y.shape[1] != 205:
        raise ValueError("panel axes")
    result = np.broadcast_to(Y, (4, *Y.shape)).copy()
    theta = np.zeros((2, 5, len(Y), 4))
    h = strata(X)
    fallback = []
    for outer in range(5):
        fit, b, a = roles(group, outer)
        for mi, columns in [(1, [0]), (2, [0, 1, 2, 3])]:
            co = fit_coefficient(X, Y, fit, columns)
            mu = X[:, columns].mean(axis=0)
            result[mi][:, b] = Y[:, b] - co @ (X[b][:, columns] - mu).T
            theta[mi - 1, outer][:, columns] = co
        point, empty = post_weights(
            np.ones((1, len(b))), h[b], np.bincount(h, minlength=5) / len(h)
        )
        result[3][:, b] = Y[:, b] * (len(b) * point[0])
        fallback.append(empty)
    return result, theta, np.asarray(fallback)


def counts(n, seed, draws=BOOTSTRAPS):
    indices = np.random.default_rng(seed).integers(0, n, (draws, n))
    result = np.empty((draws, n))
    for j in range(n):
        result[:, j] = (indices == j).sum(axis=1)
    return result


def post_weights(count, labels, population):
    """Same paired resamples, reweight to fixed pre-treatment strata masses.

    Missing stratum => original STRICT draw, never hidden extrapolation.
    """
    count = np.asarray(count, dtype=float)
    weights = np.zeros_like(count)
    bad = np.zeros(len(count), dtype=bool)
    for h, target in enumerate(population):
        mask = np.asarray(labels) == h
        mass = count[:, mask].sum(axis=1)
        bad |= (mass == 0) & (target > 0)
        weights[:, mask] = np.divide(
            count[:, mask] * target,
            mass[:, None],
            out=np.zeros_like(count[:, mask]),
            where=mass[:, None] > 0,
        )
    weights[bad] = count[bad] / count[bad].sum(axis=1, keepdims=True)
    return weights, int(bad.sum())


def selector(values, weights):
    gain = values.mean(axis=1)
    bootstrap = values @ weights.T
    ci = np.quantile(bootstrap, [0.025, 0.975], axis=1).T
    return gain, ci, (gain > MARGIN + EPS) & (ci[:, 0] > 0), bootstrap.var(axis=1)


def cp_upper(k, n):
    if n < 1:
        return None
    return 1.0 if k == n else float(beta.ppf(0.95, k + 1, n - k))


def cp_interval(k, n):
    if n < 1:
        return [None, None]
    return [
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    ]


def family_weights(n=205, seed=BOOT_SEED):
    return counts(n, seed) / n


def seed_bootstrap(values, weights):
    """[3 seeds, families] -> exactly10k draws; reduce SEED axis0."""
    values = np.asarray(values, dtype=float)
    if values.shape != (3, weights.shape[1]):
        raise ValueError("three seed/family panels required")
    return np.median(values @ weights.T, axis=0)


def summarize(values, weights):
    point = values.mean(axis=1)
    boot = seed_bootstrap(values, weights)
    return dict(
        per_seed=point.tolist(),
        median=float(np.median(point)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def variance_ratio(first, second, strict_first, strict_second, weights):
    def ratio(a, b, c, d):
        v = np.maximum(0, b - a * a)
        r = np.maximum(0, d - c * c)

        if np.any((r <= 1e-18) & (v > 1e-18)):
            raise ValueError("undefined positive variance / zero STRICT variance")
        return np.divide(v, r, out=np.ones_like(v), where=r > 1e-18)

    per = ratio(
        first.mean(axis=1),
        second.mean(axis=1),
        strict_first.mean(axis=1),
        strict_second.mean(axis=1),
    )
    boot = np.median(
        ratio(
            first @ weights.T,
            second @ weights.T,
            strict_first @ weights.T,
            strict_second @ weights.T,
        ),
        axis=0,
    )
    return dict(
        per_seed=per.tolist(),
        median=float(np.median(per)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        definition="Variance of B estimator error minus same-policy finite205-family true gain, across fixed folds/MC draws; STRICT variance denominator. Conditional family-resample interval; zero/zero=1.",
    )


def two_policy_elo(winprob, weights):
    n = winprob.shape[1]

    def rating(q):
        return 1500 + 400 * np.log10((n * q + 0.01) / (n * (1 - q) + 0.01))

    ps = rating(winprob.mean(axis=1))
    bb = np.median(rating(winprob @ weights.T), axis=0)
    return dict(
        r882=1500.0,
        per_seed=ps.tolist(),
        median=float(np.median(ps)),
        ci95=np.quantile(bb, [0.025, 0.975]).tolist(),
    )


def decision(reductions, ucb, bias, method):
    return bool(
        method in ["CUPED1", "CUPED_K"]
        and all(x is not None and x >= 0.2 - EPS for x in reductions)
        and ucb <= 0.05
        and bias < 0.001 - EPS
    )
