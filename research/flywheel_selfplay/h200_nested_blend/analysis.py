"""Data-free H200 fold, nested-choice and native-utility statistics."""

import hashlib
import numpy as np

SEEDS = (200101, 200102, 200103)
FOLD_SEEDS = {200101: 200141, 200102: 200142, 200103: 200143}
SAMPLE_SEED = 200140
BOOT_SEED = 200150
ELO_BOOT_SEED = 200151
SHUF_SEED = 200160
LAMBDA_GRID = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
PRIMARY, ARMS = "NEST_L", ("R882", "FIX025", "NEST_L", "RANK025", "NEST_SHUF")
DRAWS, ELO_DRAWS = 10_000, 1_000
LCB_MARGIN, HARM_DELTA, HARM_CAP = 0.001, -0.01, 0.10
TIE_MARGIN = 1e-4


def seed_folds(family_ids, seed):
    if seed not in FOLD_SEEDS:
        raise ValueError("unregistered H200 seed")
    families = sorted(set(family_ids))
    if len(families) != 205:
        raise ValueError("205 DL-Train families required")
    return [sorted(x.tolist()) for x in np.array_split(
        np.random.default_rng(FOLD_SEEDS[seed]).permutation(families), 5
    )]


def cell(k):
    if k not in range(5):
        raise ValueError("outer fold must be 0..4")
    a, b = k, (k + 1) % 5
    return a, b, tuple(j for j in range(5) if j not in (a, b))


def fit_plan():
    outers = sorted({tuple(sorted(cell(k)[:2])) for k in range(5)})
    inners = sorted({tuple(sorted((*cell(k)[:2], j))) for k in range(5) for j in cell(k)[2]})
    calibrators = sorted({tuple(i for i in range(5) if i != train) for train in range(5)})
    return outers, inners, calibrators


def inner_model_for(k, j):
    a, b, fit = cell(k)
    if j not in fit:
        raise ValueError("inner fold must belong to FIT")
    return tuple(sorted((a, b, j)))


def calibrator_model_for(train_fold):
    if train_fold not in range(5):
        raise ValueError("training fold must be 0..4")
    return tuple(i for i in range(5) if i != train_fold)


def stream_seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little") & 0x7FFFFFFF


def permute_scores(scores, seed):
    """Deterministically permute scores within one context, preserving its score multiset."""
    a = np.asarray(scores)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("finite one-dimensional candidate scores required")
    return a[np.random.default_rng(seed).permutation(len(a))]


def blend(reference_cost, gbm_cost, lam):
    ref, prior = np.asarray(reference_cost), np.asarray(gbm_cost)
    if lam not in LAMBDA_GRID or ref.shape != prior.shape or not (np.isfinite(ref).all() and np.isfinite(prior).all()):
        raise ValueError("registered lambda and finite matching costs required")
    if lam == 0.0:
        return ref.copy()
    return (ref + lam * (prior - ref)).astype(np.float32)


def average_rank01(values):
    """Lower values rank first; average ties and map best/worst to 0/1."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("finite one-dimensional values required")
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks / (n - 1)


def rank_blend(reference_cost, gbm_cost):
    ref, prior = np.asarray(reference_cost), np.asarray(gbm_cost)
    if ref.shape != prior.shape or ref.ndim != 1:
        raise ValueError("matching one-dimensional cost vectors required")
    return 0.75 * average_rank01(ref) + 0.25 * average_rank01(prior)


def choose_lambda(gains_by_lambda, margin=0.001):
    """Largest mean-gain lambda only if it beats lambda 0 by strictly > margin; ties choose smaller lambda."""
    gains = {float(k): float(v) for k, v in gains_by_lambda.items()}
    if set(gains) != set(LAMBDA_GRID) or not all(np.isfinite(v) for v in gains.values()):
        raise ValueError("complete finite lambda grid required")
    best = max(LAMBDA_GRID, key=lambda lam: (gains[lam], -lam))
    if best == 0.0 or gains[best] - gains[0.0] <= margin:
        return 0.0
    return best


def family_matrix(values, families):
    order = sorted(set(families))
    v = np.asarray(values, dtype=np.float64)
    if v.shape[-1] != len(families):
        raise ValueError("context panel and family ids differ")
    m = np.asarray([v[..., np.asarray(families) == f].mean(axis=-1) for f in order])
    return order, np.moveaxis(m, 0, -1)


def contrast(panel, base, families, seed=BOOT_SEED):
    _, fp = family_matrix(panel, families)
    _, fb = family_matrix(base, families)
    delta = fp - fb
    n = delta.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    boot = np.median(delta[:, draw].mean(axis=2), axis=0)
    per_seed = delta.mean(axis=1)
    fam_mean = delta.mean(axis=0)
    p = (1 + np.count_nonzero(boot <= LCB_MARGIN)) / (DRAWS + 1)
    return dict(per_seed_gain=per_seed.tolist(), median_gain=float(np.median(per_seed)),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist(), p_one_sided_vs_margin=float(p),
                positive_seeds=int((per_seed > 0).sum()),
                harm_fraction=float((fam_mean < HARM_DELTA).mean()),
                harm_families=int((fam_mean < HARM_DELTA).sum()), families=int(n),
                exact_tie_families=int((fam_mean == 0).sum()),
                exact_tie_families_all_seeds=int(np.all(delta == 0, axis=0).sum()),
                family_win_tie_loss=[int((fam_mean > 0).sum()), int((fam_mean == 0).sum()),
                                     int((fam_mean < 0).sum())])


def holm(p_values):
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    m, result, running = len(order), {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[key]))
        result[key] = running
    return result


def family_win_matrix(values_by_context, family_ids):
    v = np.asarray(values_by_context, dtype=np.float64)
    diff = v[:, :, None] - v[:, None, :]
    wins = np.where(diff > TIE_MARGIN, 1.0, np.where(diff < -TIE_MARGIN, 0.0, 0.5)) * (1.0 - np.eye(v.shape[1]))
    return np.array([wins[np.asarray(family_ids) == f].mean(axis=0) for f in sorted(set(family_ids))])


def bt_ratings(wins, anchor=0, prior=0.01):
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    s = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = ((games + 2 * prior * (1 - np.eye(n))) / np.maximum(s[..., :, None] + s[..., None, :], 1e-12)).sum(axis=-1)
        s = np.clip(totals / denom, 1e-6, 1e6)
        s /= np.maximum(s[..., anchor, None], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(s, 1e-9))


def elo_vs_reference(panel, base, families, seed=ELO_BOOT_SEED):
    wins = np.stack([family_win_matrix(np.stack([base[s], panel[s]], axis=1), families) for s in range(len(panel))])
    elo = bt_ratings(wins.sum(axis=1))[:, 1]
    n = wins.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (ELO_DRAWS, n))
    boot = np.median(bt_ratings(np.take(wins, draw, axis=1).sum(axis=2))[..., 1], axis=0)
    return dict(per_seed=elo.tolist(), median=float(np.median(elo)), ci95=np.quantile(boot, [0.025, 0.975]).tolist())


def verdict(primary, adjusted_p):
    gates = dict(lcb95_gt_0=primary["ci95"][0] > 0, holm_p_lt_005=adjusted_p < 0.05,
                 harm_le_10pct=primary["harm_fraction"] <= HARM_CAP)
    return dict(verdict="CANDIDATE: NEST_L" if all(gates.values()) else "NO_ADVANCE", gates=gates)
