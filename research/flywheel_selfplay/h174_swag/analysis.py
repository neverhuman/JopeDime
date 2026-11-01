"""H174 family cross-fit, selection, bootstrap and Elo; model-free inference only."""

import hashlib
import numpy as np

SEEDS = (174101, 174102, 174103)
FOLD_SEED, BOOT_SEED = 174140, 174141
STEPS = (0, 10, 20, 30, 40)
ARMS = ("CTRL", "SWA", "BMA05", "BMA10")
ALL_ARMS = ("r882",) + ARMS
DRAWS, EPS = 10000, 1e-12


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(FOLD_SEED).permutation(205)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer):
    groups = folds(rows)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def batch_plan(cids, orders, seed):
    """Merged H156: 8 contexts, top-fitness four reveals plus 20 sampled candidates."""
    cids, plan = sorted(cids), []
    for step in range(1, 41):
        stream = stream_seed(seed, step)
        rng = np.random.default_rng(stream)
        chosen = [
            cids[i] for i in rng.choice(len(cids), min(8, len(cids)), replace=False)
        ]
        indexes = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            indexes.append(
                np.concatenate([order[:4], rng.choice(order[4:], 20, replace=False)])
            )
        plan.append((stream, chosen, indexes))
    return plan


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(families) or not np.isfinite(values).all():
        raise ValueError("invalid context panel")
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def choose(rows, panel, seed):
    mat = np.asarray(panel, dtype=np.float64)
    if mat.shape != (len(rows), 5) or not np.isfinite(mat).all():
        raise ValueError("five finite selector snapshots required")
    _, fm = family_matrix(mat, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    # Prespecified numerical ties select the earliest step, including r882.
    best = next(i for i, v in enumerate(means) if v >= means.max() - EPS)
    diff = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(diff[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(diff.mean())
    accepted = gain > 0.001 + EPS and ci[0] > 0
    return dict(
        best_step=STEPS[best],
        selected_step=STEPS[best] if accepted else 0,
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
    if values.shape != (3, len(rows), 5) or not np.isfinite(values).all():
        raise ValueError("complete 3-seed 5-policy panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    contrasts = {}
    # Avoid materializing a 3 x 10000 x 205 x 5 panel.
    for col, arm in enumerate(ARMS, 1):
        for ref, label in [(0, "r882")] + ([(1, "CTRL")] if col > 1 else []):
            diff = fm[:, :, col] - fm[:, :, ref]
            per_seed = diff.mean(axis=1)
            boot = np.median(diff[:, ix].mean(axis=2), axis=0)
            contrasts[arm + "_minus_" + label] = dict(
                per_seed_gain=per_seed.tolist(),
                median_gain=float(np.median(per_seed)),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                positive_seeds=int((per_seed > 0).sum()),
                p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
            )
    adjusted = holm(
        {arm: contrasts[arm + "_minus_r882"]["p_one_sided"] for arm in ARMS[1:]}
    )
    for arm in ARMS[1:]:
        c = contrasts[arm + "_minus_r882"]
        c["holm_p"] = adjusted[arm]
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
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def duplicate_groups(items):
    """Exact equal-vector/identity groups; only compare a shared ordered cohort."""
    grouped = {}
    for label, value in items.items():
        if isinstance(value, str):
            key = value
        else:
            arr = np.array(value, dtype=np.float64, order="C", copy=True)
            if not np.isfinite(arr).all():
                raise ValueError("nonfinite utility vector")
            arr[arr == 0] = 0.0  # Normalize signed zero for numeric utility identity.
            key = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
        grouped.setdefault(key, []).append(label)
    return [labels for labels in grouped.values() if len(labels) > 1]


def choose_candidate(rows, panel, seed, draws=1000):
    """Paired binary candidate-vs-r882 B selector; ties fail closed."""
    mat = np.asarray(panel, dtype=np.float64)
    if mat.shape != (len(rows), 2) or not np.isfinite(mat).all():
        raise ValueError("baseline and candidate selector panels required")
    _, fm = family_matrix(mat, [r["family_id"] for r in rows])
    diff = fm[:, 1] - fm[:, 0]
    gain = float(diff.mean())
    ix = np.random.default_rng(seed).integers(0, len(diff), (draws, len(diff)))
    boot = diff[ix].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    accepted = gain > 0.001 + EPS and ci[0] > 0
    return dict(
        best_candidate=True,
        selected_candidate=bool(accepted),
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=[float(fm[:, 0].mean()), float(fm[:, 1].mean())],
        bootstrap_draws=draws,
    )
