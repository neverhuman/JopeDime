"""H262 fixed split, paired streams, decision rules and final-only inference."""

import hashlib
import numpy as np

SEEDS = (262101, 262102, 262103)
FOLD_SEED = 262140
BOOT_SEED = 262150
DRAWS = 10000
ARMS = ("NEVER", "CURRENT", "STRICT_LCB")


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
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


def partition(rows):
    families = sorted({r["family_id"] for r in rows})
    if len(families) != 205:
        raise ValueError("205 training families required")
    shuffled = np.random.default_rng(FOLD_SEED).permutation(families)
    # floor(0.70*205)=143 train, remaining 62 acceptance families.
    fit = set(shuffled[:143])
    accept = set(shuffled[143:])
    return (
        [r for r in rows if r["family_id"] in fit],
        [r for r in rows if r["family_id"] in accept],
    )


def current_accept(candidate, incumbent):
    candidate = np.asarray(candidate, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    if candidate.shape != incumbent.shape or not candidate.size:
        raise ValueError("matched nonempty acceptance panel required")
    new, old = float(candidate.mean()), float(incumbent.mean())
    if not np.isfinite([new, old]).all():
        raise ValueError("non-finite development utility")
    return dict(
        accepted=bool(new >= old),
        candidate_mean=new,
        incumbent_mean=old,
        gain=new - old,
        action="accept_improvement"
        if new > old
        else "accept_tie"
        if new == old
        else "restore_incumbent_model_and_optimizer",
    )


def strict_accept(candidate, incumbent, families, seed):
    candidate = np.asarray(candidate, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    if candidate.shape != incumbent.shape or candidate.shape != (len(families),):
        raise ValueError("matched acceptance panel required")
    if not np.isfinite(candidate).all() or not np.isfinite(incumbent).all():
        return dict(accepted=False, reason="nonfinite", ci95=None, gain=None)
    _, fm = family_matrix(candidate - incumbent, families)
    ix = np.random.default_rng(seed).integers(0, len(fm), (2000, len(fm)))
    ci = np.quantile(fm[ix].mean(axis=1), [0.025, 0.975]).tolist()
    accepted = bool(ci[0] > 0.002)
    return dict(
        accepted=accepted,
        gain=float(fm.mean()),
        ci95=ci,
        draws=2000,
        bootstrap_seed=seed,
        action="accept_LCB" if accepted else "restore_incumbent_model_and_optimizer",
    )


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
    if values.shape != (3, len(rows), 3) or not np.isfinite(values).all():
        raise ValueError("complete final three-seed three-arm panel required")
    families = [r["family_id"] for r in rows]
    ids = sorted(set(families))
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(ids), (DRAWS, len(ids)))
    contrasts = {}
    bootstraps = {}
    pairs = [
        ("STRICT_LCB_minus_r882", 2, 0),
        ("STRICT_LCB_minus_CURRENT", 2, 1),
        ("CURRENT_minus_r882", 1, 0),
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
        arms=list(ARMS),
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
