"""H263 fixed cross-fit protocol (merged H237/H232/H170 machinery; one fresh fold seed 263140); model-free inference
only."""

import hashlib
import numpy as np

SEEDS = (263101, 263102, 263103)
FOLD_SEED = 263140  # hw_t1335: one fresh family partition shared by all seeds
FOLD_SEEDS = {s: FOLD_SEED for s in SEEDS}
BOOT_SEED = 263150
STEPS = (0, 10, 20, 30, 40)
ARMS = ("CONTROL", "L2SP", "HEAD4", "HEAD4_NOTR")
ALL_ARMS = ("r882",) + ARMS
DRAWS, EPS = 10000, 1e-12
ADVANCE_MARGIN = 0.002  # hw_t1335: median gain vs r882 > +0.002 AND LCB95 > 0 AND Holm p < .05 AND 3/3 seeds positive
PLATEAU_BEST = 1.1389572  # r3175
CLIFF_CONTEXT = 27  # H152 gate-cliff DL-Val context (fam-9f67f144...): reported separately, never excluded


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


def plan_digest(plan):
    digest = hashlib.sha256()
    for stream, cids, indexes in plan:
        digest.update(
            __import__("json")
            .dumps([stream, cids, [ix.tolist() for ix in indexes]])
            .encode()
        )
    return digest.hexdigest()


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


def compute_r882_top_zgap(rcost):
    part = np.partition(rcost, 1)
    sd = float(np.std(rcost))
    if sd <= 1e-12:
        return 999.0
    return float((part[1] - part[0]) / sd)


def top4_pick(cost, fitness, gate):
    # Exact H258 gated TOP4_PICK: stable cost top4, raw gated cost tie by rank.
    order = np.argsort(cost, kind="stable")
    revealed = order[:4] if compute_r882_top_zgap(cost) < 0.25 else order[:1]
    choice = int(revealed[np.argmin(np.where(gate[revealed], fitness[revealed], 2.0))])
    return dict(
        choice=choice,
        utility=float(-fitness[choice] if gate[choice] else -2.0),
        reveals=4 if len(revealed) == 4 else 0,
        indices=revealed.tolist(),
    )


def summarize(rows, panel):
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(values).all():
        raise ValueError("complete final three-seed three-arm panel required")
    families = [r["family_id"] for r in rows]
    ids = sorted(set(families))
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(ids), (DRAWS, len(ids)))
    contrasts = {}
    bootstraps = {}
    pairs = [
        ("HEAD4_minus_r882", 3, 0),
        ("HEAD4_NOTR_minus_r882", 4, 0),
        ("CONTROL_minus_r882", 1, 0),
        ("L2SP_minus_r882", 2, 0),
        ("HEAD4_minus_L2SP", 3, 2),
    ]
    for key, col, ref in pairs:
        diff = fm[:, :, col] - fm[:, :, ref]
        per_seed = diff.mean(axis=1)
        boot = np.median(diff[:, ix].mean(axis=2), axis=0)
        bootstraps[key] = boot
        contrasts[key] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
            harm_families_by_seed=[
                [f for f, d in zip(ids, row, strict=True) if d < -1e-4] for row in diff
            ],
            family_wtl_by_seed=[
                [
                    int((row > 1e-4).sum()),
                    int((abs(row) <= 1e-4).sum()),
                    int((row < -1e-4).sum()),
                ]
                for row in diff
            ],
        )
    tested = [pairs[0][0], pairs[1][0]]
    adjusted = holm({key: contrasts[key]["p_one_sided"] for key in tested})
    ordered = sorted(tested, key=lambda key: contrasts[key]["p_one_sided"])
    preceding_pass = True
    for rank, key in enumerate(ordered):
        quantile = 0.025 / (2 - rank) if preceding_pass else 0.0125
        c = contrasts[key]
        c["holm_p"] = adjusted[key]
        c["holm_lcb_quantile"] = quantile
        c["holm_lcb95"] = float(np.quantile(bootstraps[key], quantile))
        c["advance"] = bool(
            c["median_gain"] > 0.002
            and c["holm_lcb95"] > 0
            and c["holm_p"] < 0.05
            and c["positive_seeds"] == 3
        )
        preceding_pass = preceding_pass and adjusted[key] < 0.05
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1))
    ei = np.random.default_rng(BOOT_SEED + 1).integers(0, len(ids), (1000, len(ids)))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2)), axis=0)
    return dict(
        arms=list(ALL_ARMS),
        contexts=len(rows),
        families=len(ids),
        contrasts=contrasts,
        context_utility_by_seed=values.mean(axis=1).tolist(),
        family_utility_by_seed=fm.mean(axis=1).tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
        bootstrap_draws=DRAWS,
        bootstrap_seed=BOOT_SEED,
        elo_bootstrap_draws=1000,
    )
