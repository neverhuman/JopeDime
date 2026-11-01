"""H188 kernel-INPUT ablation on the r882 adapter (0 updates): pooled-B selection then pooled-A confirmation.

Fold/utility/win-matrix/Bradley-Terry helpers are copied verbatim from merged H184/H180/H176 (axis=0 paired bootstrap,
#1512). `pooled_b_select_m3` is merged H177 `analysis.pooled_b_select` (the rule H178 calibrated; receipt pinned in the
manifest) copied operation for operation with the arm set made a parameter for H188's prespecified m=3 family.
"""

import hashlib

import numpy as np

SEEDS = (188101, 188102, 188103)  # three fold reshuffles
FOLD_SEED, BOOT_SEED = 188140, 188141
ARMS = ("R882", "KIN_ANCHOR", "KIN_RANKRES", "KIN_CLIPRES")
CANDIDATES = ARMS[1:]  # Holm m=3
DRAWS, EPS = 10_000, 1e-12
TIE_MARGIN = 1e-4
MARGIN = 0.001


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def family_folds(family_ids, seed=FOLD_SEED):
    families = sorted(set(family_ids))
    if len(families) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(seed).permutation(len(families))
    return [sorted(families[i] for i in group) for group in np.array_split(order, 5)]


def partition(family_ids, outer, folds=None):
    groups = family_folds(family_ids) if folds is None else folds
    if len(groups) != 5 or not 0 <= outer < 5:
        raise ValueError("five family folds required")
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = set().union(
        *(set(groups[i]) for i in range(5) if i not in (outer, (outer + 1) % 5))
    )
    if fit & (a | b) or a & b:
        raise ValueError("family fold overlap")
    return fit, b, a


def context_utility(fitness, gate):
    fitness = np.asarray(fitness, dtype=np.float64)
    gate = np.asarray(gate, dtype=bool)
    if fitness.shape != gate.shape or not np.isfinite(fitness).all():
        raise ValueError("invalid outcome arrays")
    return np.where(gate, -fitness, -2.0)


def family_values(rows, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("one finite utility per context required")
    families = sorted({r["family_id"] for r in rows})
    by_family = np.array(
        [values[[r["family_id"] == f for r in rows]].mean() for f in families],
        dtype=np.float64,
    )
    return families, by_family


def zero_update_gap(
    train_rows, train_r882, train_oracle, val_rows, val_r882, val_oracle
):
    train_families, train_f = family_values(train_rows, train_r882)
    val_families, val_f = family_values(val_rows, val_r882)
    _, train_o = family_values(train_rows, train_oracle)
    _, val_o = family_values(val_rows, val_oracle)
    train_mean, val_mean = float(train_f.mean()), float(val_f.mean())
    train_oracle_mean, val_oracle_mean = float(train_o.mean()), float(val_o.mean())
    headroom = train_oracle_mean - train_mean
    gap = train_mean - val_mean
    return dict(
        train=dict(
            contexts=len(train_rows),
            families=len(train_families),
            equal_family_utility=train_mean,
            family_q05=float(np.quantile(train_f, 0.05)),
            family_q25=float(np.quantile(train_f, 0.25)),
            family_median=float(np.median(train_f)),
            family_q75=float(np.quantile(train_f, 0.75)),
            family_q95=float(np.quantile(train_f, 0.95)),
            minus2_contexts=int(np.count_nonzero(np.asarray(train_r882) == -2.0)),
            oracle_equal_family_utility=train_oracle_mean,
        ),
        dl_val=dict(
            contexts=len(val_rows),
            families=len(val_families),
            equal_family_utility=val_mean,
            family_q05=float(np.quantile(val_f, 0.05)),
            family_q25=float(np.quantile(val_f, 0.25)),
            family_median=float(np.median(val_f)),
            family_q75=float(np.quantile(val_f, 0.75)),
            family_q95=float(np.quantile(val_f, 0.95)),
            minus2_contexts=int(np.count_nonzero(np.asarray(val_r882) == -2.0)),
            oracle_equal_family_utility=val_oracle_mean,
        ),
        generalization_gap_train_minus_val=gap,
        oracle_headroom_train=headroom,
        oracle_headroom_normalized_gap=(gap / headroom if headroom > 0 else None),
        normalization_definition="(r882 equal-family TRAIN utility - r882 equal-family DL-Val utility) / (TRAIN oracle utility - r882 TRAIN utility)",
        family_values={
            "train": dict(zip(train_families, train_f.tolist())),
            "dl_val": dict(zip(val_families, val_f.tolist())),
        },
    )


def family_win_matrix(values_by_context, family_ids, tie_margin=TIE_MARGIN):
    v = np.asarray(values_by_context, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != len(family_ids) or not np.isfinite(v).all():
        raise ValueError("context x policy finite matrix required")
    diff = v[:, :, None] - v[:, None, :]
    wins = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    wins *= 1.0 - np.eye(v.shape[1])
    families = sorted(set(family_ids))
    return np.array(
        [wins[np.asarray(family_ids) == family].mean(axis=0) for family in families]
    )


def bt_ratings(wins, anchor=0, prior=0.01):
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


def _percentile_ci(x):
    return np.quantile(np.asarray(x), [0.025, 0.975], axis=0).tolist()


def choice_agreement(choices, reference):
    """Fraction of contexts where an arm's argmin candidate equals R882's."""
    keys = sorted(reference)
    if set(choices) != set(keys) or not keys:
        raise ValueError("choice closure")
    return float(np.mean([choices[k] == reference[k] for k in keys]))


def seed_folds(family_ids, seed):
    """One fresh 5-fold family reshuffle per seed (cyclic A=g[j], B=g[j+1], FIT=rest; FIT unused at 0 updates)."""
    return family_folds(family_ids, seed=stream_seed(FOLD_SEED, seed))


def family_matrix(values, families):
    order = sorted(set(families))
    matrix = np.asarray(
        [np.asarray(values)[np.asarray(families) == f].mean(axis=0) for f in order]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite complete panel")
    return order, matrix


def holm(pvals):
    """Merged H177 holm, verbatim."""
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(pvals, key=pvals.get)):
        previous = max(previous, min(1.0, pvals[key] * (len(pvals) - rank)))
        result[key] = previous
    return result


def pooled_b_select_m3(
    rows, r882_values, candidate_panels_by_arm, arms=CANDIDATES, seed=BOOT_SEED
):
    """H177 pooled-B rule for H188's m=3 family: per family, average the three seeds' (arm - r882) gains, then a
    10,000-draw paired family bootstrap; admit iff mean gain > 0.001 + 1e-12 AND lower 95% bound > 0 AND Holm(m=3)
    p < 0.05. Ties reject. Exactly one candidate panel (3, N_ctx) per arm."""
    if set(candidate_panels_by_arm) != set(arms) or len(arms) != 3:
        raise ValueError("exact prespecified m=3 arm family required")
    families = [r["family_id"] for r in rows]
    n_fams = len(set(families))
    if n_fams != 205:
        raise ValueError(f"pooled B selector requires 205 families, got {n_fams}")
    base = np.asarray(r882_values, dtype=np.float64)
    if base.shape != (3, len(rows)) or not np.isfinite(base).all():
        raise ValueError("r882_values must be a finite (3, N_ctx) panel")
    fm_base = np.stack([family_matrix(v, families)[1] for v in base])
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))
    raw, pvals = {}, {}
    for arm in arms:
        arr = np.asarray(candidate_panels_by_arm[arm], dtype=np.float64)
        if arr.shape != (3, len(rows)) or not np.isfinite(arr).all():
            raise ValueError(f"arm {arm} must be one finite (3, N_ctx) candidate panel")
        diff = np.stack([family_matrix(v, families)[1] for v in arr]) - fm_base
        per_seed = diff.mean(axis=1)
        seed_avg = diff.mean(axis=0)
        mean_gain = float(seed_avg.mean())
        boot = seed_avg[ix].mean(axis=1)
        ci95 = np.quantile(boot, [0.025, 0.975]).tolist()
        p = float((1 + (boot <= 0.0).sum()) / (DRAWS + 1))
        boot_med = np.median(diff[:, ix].mean(axis=2), axis=0)
        pvals[arm] = p
        raw[arm] = dict(
            mean_gain=mean_gain,
            ci95=ci95,
            per_seed_gain=per_seed.tolist(),
            median_seed_gain=float(np.median(per_seed)),
            median_seed_ci95=np.quantile(boot_med, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
            p_one_sided=p,
        )
    adj = holm(pvals)
    out = {}
    for arm, d in raw.items():
        unadj = bool(d["mean_gain"] > MARGIN + EPS and d["ci95"][0] > 0.0)
        out[arm] = dict(
            **d,
            holm_multiplicity=3,
            holm_p=float(adj[arm]),
            unadjusted_admitted=unadj,
            admitted=bool(unadj and adj[arm] < 0.05),
        )
    return out


def summarize_a(rows, panels, minus2, agreement, seed_ids=SEEDS, boot_seed=BOOT_SEED):
    """Pooled-A panel [seed, context, arm] (ARMS order, column 0 = R882): equal-family utility, per-seed gains, median
    over seeds with a 10,000-draw paired family bootstrap reduced over seeds (axis=0), Holm(m=3) on the one-sided p,
    canonical BT Elo vs r882 with CI, exact -2.0 counts and choice agreement."""
    values = np.asarray(panels, dtype=np.float64)
    if (
        values.shape != (len(seed_ids), len(rows), len(ARMS))
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete finite pooled-A panel required")
    families = [r["family_id"] for r in rows]
    fams, _ = family_matrix(values[0, :, 0], families)
    fm = np.stack(
        [family_matrix(values[s], families)[1] for s in range(len(seed_ids))]
    )  # (S, F, A)
    n = len(fams)
    draw = np.random.default_rng(boot_seed).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    out = {"family_count": n, "context_count": len(rows), "arms": {}, "contrasts": {}}
    for j, arm in enumerate(ARMS):
        out["arms"][arm] = dict(
            per_seed_equal_family_utility=means[:, j].tolist(),
            median_over_seeds=float(np.median(means[:, j])),
            minus2_contexts_per_seed=list(minus2[arm]),
            choice_agreement_with_r882_per_seed=list(agreement[arm]),
            identical_across_seeds=bool(
                all(
                    np.array_equal(values[0, :, j], values[s, :, j])
                    for s in range(len(seed_ids))
                )
            ),
        )
    pvals = {}
    for arm in CANDIDATES:
        j = ARMS.index(arm)
        delta = fm[:, :, j] - fm[:, :, 0]
        per_seed = delta.mean(axis=1)
        boot = np.median(delta[:, draw].mean(axis=2), axis=0)
        p = float((1 + np.count_nonzero(boot <= 0)) / (DRAWS + 1))
        pvals[arm] = p
        out["contrasts"][f"{arm}_minus_R882"] = dict(
            per_seed_equal_family_gain=per_seed.tolist(),
            median_over_seeds=float(np.median(per_seed)),
            family_bootstrap_ci95=_percentile_ci(boot),
            p_one_sided=p,
            positive_seeds=int((per_seed > 0).sum()),
        )
    adj = holm(pvals)
    for arm in CANDIDATES:
        out["contrasts"][f"{arm}_minus_R882"]["holm_p"] = float(adj[arm])
    win = np.stack(
        [family_win_matrix(values[s], families) for s in range(len(seed_ids))]
    )
    elo = bt_ratings(win.sum(axis=1))
    parts = [
        bt_ratings(np.take(win, draw[st : st + 100], axis=1).sum(axis=2))
        for st in range(0, DRAWS, 100)
    ]
    med_boot = np.median(np.concatenate(parts, axis=1), axis=0)
    out["elo_vs_r882"] = {
        arm: dict(
            per_seed=elo[:, j].tolist(),
            median_over_seeds=float(np.median(elo[:, j])),
            family_bootstrap_ci95=_percentile_ci(med_boot[:, j]),
        )
        for j, arm in enumerate(ARMS)
    }
    out["identical_seed_panels"] = bool(
        all(np.array_equal(values[0], values[s]) for s in range(len(seed_ids)))
    )
    return out


def verdict(b_decisions, a_summary):
    """ADVANCE for an arm only if admitted on pooled B AND pooled-A median gain > 0 AND 3/3 seeds positive AND pooled-A
    median-seed LCB95 > 0; ties reject. Never adopted (flagged to the flywheel owner)."""
    per_arm = {}
    for arm in CANDIDATES:
        a = a_summary["contrasts"][f"{arm}_minus_R882"]
        gates = dict(
            admitted_on_pooled_b=bool(b_decisions[arm]["admitted"]),
            a_gain_gt_0=a["median_over_seeds"] > EPS,
            three_of_three_seeds=a["positive_seeds"] == 3,
            a_lcb95_gt_0=a["family_bootstrap_ci95"][0] > 0,
        )
        per_arm[arm] = dict(gates=gates, advance=all(gates.values()))
    adv = sorted(a for a in per_arm if per_arm[a]["advance"])
    return dict(
        verdict=("ADVANCE:" + "+".join(adv)) if adv else "NO_ADVANCE", per_arm=per_arm
    )
