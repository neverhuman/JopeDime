"""H264 zero-update kNN residual memory on frozen r882 embeddings: fixed cross-fit protocol (one fresh fold seed 264140),
the exact shortlist kNN search and residual, the strict inner config selector and model-free inference only."""

import hashlib
import numpy as np

SEEDS = (264101, 264102, 264103)
FOLD_SEED = 264140  # hw_t1658: one fresh family partition shared by all seeds
FOLD_SEEDS = {s: FOLD_SEED for s in SEEDS}
BOOT_SEED = 264150
ARMS = ("CONTROL", "KNN", "KNN_SHUF")
# Contrasts: every arm vs r882; KNN and KNN_SHUF also vs CONTROL; KNN vs its placebo KNN_SHUF.
REFERENCES = {
    "CONTROL": ("r882",),
    "KNN": ("r882", "CONTROL", "KNN_SHUF"),
    "KNN_SHUF": ("r882", "CONTROL"),
}
SEEDS_POSITIVE_REQUIRED = 3  # hw_t1658: 3/3 seeds positive
ALL_ARMS = ("r882",) + ARMS
DRAWS, EPS = 10000, 1e-12
ADVANCE_MARGIN = 0.002  # median family gain > +0.002 AND LCB95 > 0 AND Holm p < .05 (m=2) AND 3/3 AND KNN > KNN_SHUF
PLATEAU_BEST = 1.1389572  # r3175
CLIFF_CONTEXT = 27  # H152 gate-cliff DL-Val context (fam-9f67f144...): reported separately, never excluded
SHORTLIST = 64  # r882 top-64 candidates per context form both the memory bank and the scored query set
K_GRID, TAU_GRID, LAMBDA_GRID = (8, 32, 128), (0.05, 0.2), (0.25, 0.5, 1.0)
K_MAX = max(K_GRID)
HUBER_DELTA = 1.0  # H161 Huber scale: residuals of gate-failed (cost 2.0) candidates are clipped to [-1, 1]
CONFIGS = [(0, 0.0, 0.0)] + [
    (k, t, lam) for k in K_GRID for t in TAU_GRID for lam in LAMBDA_GRID
]  # index 0 = r882


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows, seed):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(FOLD_SEEDS[seed]).permutation(205)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer, seed):
    groups = folds(rows, seed)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(families) or not np.isfinite(values).all():
        raise ValueError("invalid context panel")
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def choose_config(rows, panel, seed):
    """Strict inner selector over CONFIGS on the SELECT fold: the earliest config with the best family-mean utility
    (config 0 = lambda 0 = r882 wins ties), accepted only if its gain over config 0 is > .001 AND the family-bootstrap
    LCB95 > 0; otherwise config 0."""
    mat = np.asarray(panel, dtype=np.float64)
    if mat.shape != (len(rows), len(CONFIGS)) or not np.isfinite(mat).all():
        raise ValueError("finite SELECT utilities for every config required")
    _, fm = family_matrix(mat, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    best = next(i for i, v in enumerate(means) if v >= means.max() - EPS)
    diff = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(diff[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(diff.mean())
    accepted = gain > 0.001 + EPS and ci[0] > 0
    return dict(
        best_config=best,
        selected_config=best if accepted else 0,
        best=dict(zip(("k", "tau", "lambda"), CONFIGS[best])),
        selected=dict(zip(("k", "tau", "lambda"), CONFIGS[best if accepted else 0])),
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=means.tolist(),
    )


def bt_ratings(wins, anchor=0, prior=0.01):
    """H148 canonical 400-iteration MM, r882 anchor, pseudo-win prior .01."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2 * prior * (1 - np.eye(n)))
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > 1e-4, 1.0, np.where(diff < -1e-4, 0.0, 0.5)) * (
        1 - np.eye(values.shape[1])
    )
    return family_matrix(outcomes, families)[1]


def holm(pvals):
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(pvals, key=pvals.get)):
        previous = max(previous, min(1.0, pvals[key] * (len(pvals) - rank)))
        result[key] = previous
    return result


def summarize(rows, panel, seed=BOOT_SEED):
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(values).all():
        raise ValueError("complete 3-seed 4-policy panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    fam_ids = family_matrix(values[0], families)[0]
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    contrasts = {}
    # Avoid materializing a 3 x 10000 x 205 x 5 panel.
    for col, arm in enumerate(ARMS, 1):
        for label in REFERENCES[arm]:
            ref = ALL_ARMS.index(label)
            diff = fm[:, :, col] - fm[:, :, ref]
            per_seed = diff.mean(axis=1)
            boot = np.median(diff[:, ix].mean(axis=2), axis=0)
            contrasts[arm + "_minus_" + label] = dict(
                per_seed_gain=per_seed.tolist(),
                median_gain=float(np.median(per_seed)),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                positive_seeds=int((per_seed > 0).sum()),
                p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
                harm_families_by_seed=[
                    [f for f, d in zip(fam_ids, row) if d < -1e-4] for row in diff
                ],
            )
    adjusted = holm(
        {arm: contrasts[arm + "_minus_r882"]["p_one_sided"] for arm in ARMS[1:]}
    )
    for arm in ARMS[1:]:
        c = contrasts[arm + "_minus_r882"]
        c["holm_p"] = adjusted[arm]
        c["advance"] = bool(
            c["median_gain"] > ADVANCE_MARGIN + EPS
            and c["ci95"][0] > 0
            and adjusted[arm] < 0.05
            and c["positive_seeds"] >= SEEDS_POSITIVE_REQUIRED
        )
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1))
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    # Average family weights; pseudo-prior follows H148 sum-of-families convention.
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2)), axis=0)
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=contrasts,
        family_utility_by_seed=means.tolist(),
        context_utility_by_seed=values.mean(axis=1).tolist(),
        vs_plateau_best_by_seed=(values.mean(axis=1) - PLATEAU_BEST).tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def shortlist(cost, size=SHORTLIST):
    """Indices of the `size` lowest r882 costs (stable order). Outcome-free."""
    return np.argsort(np.asarray(cost), kind="stable")[:size]


def residual_values(pred, fitness, gate, delta=HUBER_DELTA):
    """Memory values: true cost (the H146/H170 CTRL target: clip(fitness, -4, 2) if gate else 2.0) minus the r882
    predicted cost; residuals of gate-failed candidates are clipped to [-delta, delta] (H161 Huber scale)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.where(gate, np.clip(fitness, -4.0, 2.0), 2.0)
    r = target - pred
    return np.where(gate, r, np.clip(r, -delta, delta))


def keys(ctx_vec, cand_mat):
    """L2-normalised concat(context embedding, candidate embedding), float32, one row per candidate."""
    ctx = np.broadcast_to(
        np.asarray(ctx_vec, dtype=np.float32), (len(cand_mat), len(ctx_vec))
    )
    k = np.concatenate([ctx, np.asarray(cand_mat, dtype=np.float32)], axis=1)
    return (k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-12)).astype(
        np.float32
    )


def knn_search(qkeys, qfam, mkeys, mfam, kmax=K_MAX, chunk=1024, cooperate=None):
    """Exact chunked search: for each query the kmax nearest memory keys by Euclidean distance between normalised
    keys, excluding every memory pair of the query's own family. Returns (distances ascending, memory indices)."""
    import torch

    mk = torch.as_tensor(mkeys)
    mf = np.asarray(mfam)
    dist = np.empty((len(qkeys), kmax), dtype=np.float32)
    idx = np.empty((len(qkeys), kmax), dtype=np.int64)
    for s in range(0, len(qkeys), chunk):
        if cooperate is not None:
            cooperate()
        q = torch.as_tensor(qkeys[s : s + chunk])
        sim = q @ mk.T
        same = torch.as_tensor(np.asarray(qfam[s : s + chunk])[:, None] == mf[None, :])
        sim = sim.masked_fill(same, -np.inf)
        v, i = torch.topk(sim, kmax, dim=1, largest=True, sorted=True)
        if not torch.isfinite(v).all():
            raise RuntimeError("fewer than kmax out-of-family memory pairs")
        dist[s : s + chunk] = torch.sqrt(torch.clamp(2.0 - 2.0 * v, min=0.0)).numpy()
        idx[s : s + chunk] = i.numpy()
    return dist, idx


def knn_residual(dist, idx, values, k, tau):
    """softmax(-d / tau)-weighted mean of the memory values over the k nearest neighbours."""
    d = np.asarray(dist[:, :k], dtype=np.float64)
    w = np.exp(-(d - d[:, :1]) / tau)
    w /= w.sum(axis=1, keepdims=True)
    return (w * np.asarray(values, dtype=np.float64)[idx[:, :k]]).sum(axis=1)


def pick(cost, short_idx, residual, lam):
    """Single pick of r882 cost + lambda x kNN residual on the shortlist (other candidates keep their r882 cost);
    first index on ties. lambda 0 is exactly r882's argmin."""
    adjusted = np.asarray(cost, dtype=np.float64).copy()
    if lam:
        adjusted[short_idx] += lam * np.asarray(residual, dtype=np.float64)
    return int(np.argmin(adjusted))


def utility(fitness, gate, choice):
    return float(-fitness[choice] if gate[choice] else -2.0)


def decide(summary):
    """hw_t1658 advance rule: median > +0.002, LCB95 > 0, Holm p < .05, 3/3 seeds (inside summarize) AND KNN beats its
    placebo (median KNN - KNN_SHUF gain > 0). KNN_SHUF is a placebo and cannot advance."""
    c = summary["contrasts"]
    c["KNN_minus_r882"]["beats_placebo"] = bool(
        c["KNN_minus_KNN_SHUF"]["median_gain"] > EPS
    )
    c["KNN_minus_r882"]["advance"] = bool(
        c["KNN_minus_r882"]["advance"] and c["KNN_minus_r882"]["beats_placebo"]
    )
    c["KNN_SHUF_minus_r882"]["advance"] = False
    return summary
