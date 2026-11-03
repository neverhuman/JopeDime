"""H250 fixed cross-fit protocol (merged H237/H232/H170 machinery; one fresh fold seed 250140); model-free inference
only."""

import hashlib
import numpy as np

SEEDS = (250101, 250102, 250103)
FOLD_SEED = 250140  # hw_t1131: one fresh family partition shared by all seeds
FOLD_SEEDS = {s: FOLD_SEED for s in SEEDS}
BOOT_SEED = 250150
STEPS = (0, 10, 20, 30, 40)
ARMS = ("CONTROL", "CTXZ_FB", "CTXZ")
ALL_ARMS = ("r882",) + ARMS
DRAWS, EPS = 10000, 1e-12
ADVANCE_MARGIN = 0.002  # hw_t1131: median gain vs r882 > +0.002 AND LCB95 > 0 AND Holm p < .05 AND >= 2/3 seeds positive
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


def arm_batch_plan(rows, orders, seed, outer, arm):
    """CONTROL/CTXZ retain H170 order; CTXZ_FB draws eight distinct uniform families."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if arm != "CTXZ_FB":
        return batch_plan([r["context_sha256"] for r in rows], orders, seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["family_id"], []).append(row["context_sha256"])
    families = sorted(groups)
    if len(families) != 123:
        raise ValueError("123 FIT families required")
    groups = {f: sorted(cids) for f, cids in groups.items()}
    plan = []
    for step in range(1, 41):
        stream = stream_seed(seed, outer, "CTXZ_FB", step)
        rng = np.random.default_rng(stream)
        chosen_families = [
            families[i] for i in rng.choice(len(families), 8, replace=False)
        ]
        cids = [groups[f][rng.integers(len(groups[f]))] for f in chosen_families]
        indexes = []
        for cid in cids:
            order = np.asarray(orders[cid], dtype=np.int64)
            indexes.append(
                np.concatenate([order[:4], rng.choice(order[4:], 20, replace=False)])
            )
        plan.append((stream, cids, indexes))
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
        for ref, label in [(0, "r882")] + ([(1, "CONTROL")] if col > 1 else []):
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
            and c["positive_seeds"] >= 2
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


def pseudoreplication(primary, secondary, unconstrained, unconstrained_val, cells):
    """Disclose shared predictions without pretending distinct seeds replicate predictions."""

    def panel_groups(panel):
        return duplicate_groups(
            {
                f"seed{s}:{arm}": panel[si, :, ai]
                for si, s in enumerate(SEEDS)
                for ai, arm in enumerate(ALL_ARMS)
            }
        )

    report = dict(
        strict_OOF_identical=panel_groups(primary),
        strict_DLVal_identical=panel_groups(secondary),
        unconstrained_OOF_identical=panel_groups(unconstrained),
        unconstrained_DLVal_identical=panel_groups(unconstrained_val),
        step40_adapter_identical=duplicate_groups(
            {
                f"A{c['outer_fold']}:seed{c['seed']}:{arm}": c["adapter_sha256"][arm][
                    "40"
                ]
                for c in cells
                for arm in ARMS
            }
        ),
        selected_adapter_identical=duplicate_groups(
            {
                f"A{c['outer_fold']}:seed{c['seed']}:{arm}": c["adapter_sha256"][arm][
                    str(c["decisions"][arm]["selected_step"])
                ]
                for c in cells
                for arm in ARMS
            }
        ),
        selector_snapshot_identical=[],
        note="Exact vectors on matched ordered contexts only; step0 and fallback r882 duplication is expected. Distinct adapter hashes do not imply independent prediction replication.",
    )
    for outer in range(5):
        group = [c for c in cells if c["outer_fold"] == outer]
        for j, step in enumerate(STEPS[1:], 1):
            items = {
                f"seed{c['seed']}:{arm}": c["selector_panels"][arm][j]["values"]
                for c in group
                for arm in ARMS
            }
            report["selector_snapshot_identical"].append(
                dict(outer=outer, step=step, groups=duplicate_groups(items))
            )
    report["all_arms_selector_identical_to_CONTROL"] = all(
        all(
            np.array_equal(
                c["selector_panels"][arm][j]["values"],
                c["selector_panels"]["CONTROL"][j]["values"],
            )
            for arm in ARMS[1:]
            for j in range(1, len(STEPS))
        )
        for c in cells
    )
    return report


def archived_plateau_elo(rows, panel, reference, seed=BOOT_SEED + 1):
    """Descriptive archived-only comparator, never called for optimization/selection/advancement."""
    values = np.asarray(panel, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (
        values.shape != (3, len(rows), len(ALL_ARMS))
        or reference.shape != (len(rows),)
        or not np.isfinite(values).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("matched finite plateau panel required")
    families = [r["family_id"] for r in rows]
    n = len(set(families))
    ix = np.random.default_rng(seed).integers(0, n, (1000, n))
    out = {}
    for ai, arm in enumerate(ALL_ARMS):
        wins = np.stack(
            [
                family_wins(np.column_stack([reference, values[si, :, ai]]), families)
                for si in range(3)
            ]
        )
        observed = bt_ratings(wins.sum(axis=1))[:, 1]
        boot = np.median(bt_ratings(wins[:, ix].sum(axis=2))[:, :, 1], axis=0)
        out[arm] = dict(
            per_seed_rating=observed.tolist(),
            median_rating=float(np.median(observed)),
            ci95_rating=np.quantile(boot, [0.025, 0.975]).tolist(),
            per_seed_lift=(observed - 1500).tolist(),
            median_lift=float(np.median(observed) - 1500),
            ci95_lift=np.quantile(boot - 1500, [0.025, 0.975]).tolist(),
        )
    return dict(
        arms=out,
        reference_anchor=1500,
        bootstrap_draws=1000,
        boot_seed=seed,
        role="DESCRIPTIVE_ARCHIVED_ONLY; not a new CPU r3175 qualification; historical winner selection remains",
    )
