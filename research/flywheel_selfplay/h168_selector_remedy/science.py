"""H168 statistical protocol: 5-fold family cross-fit, OLD vs NEW inner-B selector remedy, pinned median-seed OOF confirmation, and canonical BT Elo."""

import hashlib
import numpy as np

SEEDS = (168101, 168102, 168103)
FOLD_SEED = 168140
BOOT_SEED = 168141
STEPS = (0, 10, 20, 30, 40)
WIDTH_ARMS = ("W10", "W20")
WIDTH_MULTS = {"W10": 1.0, "W20": 2.0}
PRIMARY_ARMS = ("W10_OLD", "W10_NEW", "W20_OLD", "W20_NEW")
DIAGNOSTIC_ARMS = ("W10_RELAXED", "W10_NONZERO", "W20_RELAXED", "W20_NONZERO")
COLUMNS = ("r882",) + PRIMARY_ARMS
DIAG_COLUMNS = ("r882",) + DIAGNOSTIC_ARMS
MARGIN = 0.001
TIE = 1e-12
DRAWS = 10000


def stream_seed(*parts):
    raw = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & 0x7FFFFFFF


def folds(rows, seed=FOLD_SEED):
    ids = [r["context_sha256"] for r in rows]
    fams = sorted({r["family_id"] for r in rows})
    if (
        len(ids) != len(set(ids))
        or len(fams) < 5
        or len(fams) % 5 != 0
        or any(
            r.get("dl_split", "DL-Train") != "DL-Train"
            or r.get("has_meta_test") is True
            for r in rows
        )
    ):
        raise ValueError(
            "unique DL-Train contexts and family count divisible by 5 required"
        )
    order = np.random.default_rng(seed).permutation(len(fams))
    size = len(fams) // 5
    return [sorted(fams[i] for i in order[k * size : (k + 1) * size]) for k in range(5)]


def partition(rows, outer, seed=FOLD_SEED):
    groups = folds(rows, seed)
    select = (outer + 1) % 5
    fit_fams = {f for k, g in enumerate(groups) if k not in (outer, select) for f in g}
    sel_fams = set(groups[select])
    out_fams = set(groups[outer])
    assert not (fit_fams & sel_fams or fit_fams & out_fams or sel_fams & out_fams)
    fit = [r for r in rows if r["family_id"] in fit_fams]
    sel = [r for r in rows if r["family_id"] in sel_fams]
    out = [r for r in rows if r["family_id"] in out_fams]
    return fit, sel, out


def batch_plan(fit_cids, orders, seed, steps=40, batch=8, top=24):
    """H156 common-random-numbers fit batch plan: 8 contexts/step, top-4 revealed + 20 sampled candidates."""
    if seed not in SEEDS or not fit_cids:
        raise ValueError("registered seed and non-empty fit_cids required")
    cids = sorted(fit_cids)
    plan = []
    for step in range(1, steps + 1):
        s = stream_seed(seed, step)
        rng = np.random.default_rng(s)
        chosen = [
            cids[i]
            for i in rng.choice(len(cids), size=min(batch, len(cids)), replace=False)
        ]
        ixs = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            head = order[: min(4, len(order))]
            rest = order[min(4, len(order)) :]
            need = min(top, len(order)) - len(head)
            tail = (
                rng.choice(rest, size=need, replace=False)
                if need > 0
                else np.empty(0, dtype=np.int64)
            )
            ixs.append(np.concatenate([head, tail]))
        plan.append((s, chosen, ixs))
    return plan


def family_matrix(values, families):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] != len(families) or not np.isfinite(arr).all():
        raise ValueError("nonfinite or mismatched context values")
    fams = sorted(set(families))
    fam_arr = np.asarray(families)
    mat = np.array([arr[fam_arr == f].mean(axis=0) for f in fams], dtype=np.float64)
    if not np.isfinite(mat).all():
        raise ValueError("nonfinite family matrix")
    return fams, mat


def bootstrap(a, b, draws=DRAWS, seed=BOOT_SEED):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.ndim != 1 or len(diff) == 0 or not np.isfinite(diff).all():
        raise ValueError("invalid family contrast")
    point = float(diff.mean())
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(draws, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = float((int(np.sum(boot <= 0.0)) + 1) / (draws + 1))
    return dict(
        delta=point,
        family_median=float(np.median(diff)),
        ci_low_95=float(lo),
        ci_high_95=float(hi),
        p_one_sided=p,
        draws=int(draws),
    )


def choose_dual(rows, panel, seed, r882_col=None, draws=DRAWS):
    """Evaluates OLD (strict inner-B LCB>0 & gain>0.001) vs NEW (prospective remedy: inner-B mean gain>0.001 with strict baseline ties)
    plus pre-locked RELAXED (gain>1e-12) and NONZERO (best step in 10..40) diagnostics on the exact same trajectory snapshots."""
    mat = np.asarray(panel, dtype=np.float64)
    if mat.shape != (len(rows), len(STEPS)) or not np.isfinite(mat).all():
        raise ValueError("selector panel shape/finiteness mismatch")
    fams = [r["family_id"] for r in rows]
    _, fm = family_matrix(mat, fams)
    if r882_col is None:
        base = fm[:, 0]
    else:
        _, base = family_matrix(np.asarray(r882_col, dtype=np.float64), fams)

    means = fm.mean(axis=0)
    gains = [float((fm[:, j] - base).mean()) for j in range(len(STEPS))]

    # Earliest non-zero step maximizing equal-family gain (ties within TIE resolved to earlier step)
    nonzero_gains = np.asarray(gains[1:], dtype=np.float64)
    best_nz_idx = 1 + int(np.flatnonzero(nonzero_gains >= nonzero_gains.max() - TIE)[0])
    nonzero_step = STEPS[best_nz_idx]

    old_step, old_gain = 0, 0.0
    new_step, new_gain = 0, 0.0
    relaxed_step, relaxed_gain = 0, 0.0
    candidates = {}

    for j, step in enumerate(STEPS):
        if step == 0:
            candidates["0"] = dict(
                delta=0.0,
                family_median=0.0,
                ci_low_95=0.0,
                ci_high_95=0.0,
                p_one_sided=1.0,
                draws=int(draws),
                eligible_old=False,
                eligible_new=False,
                eligible_relaxed=False,
            )
            continue
        st = bootstrap(fm[:, j], base, draws=draws, seed=seed + step)
        ok_old = bool(st["delta"] > MARGIN + TIE and st["ci_low_95"] > 0.0)
        ok_new = bool(st["delta"] > MARGIN + TIE)
        ok_relaxed = bool(st["delta"] > TIE)
        candidates[str(step)] = dict(
            **st,
            eligible_old=ok_old,
            eligible_new=ok_new,
            eligible_relaxed=ok_relaxed,
        )
        if ok_old and st["delta"] > old_gain + TIE:
            old_step, old_gain = step, st["delta"]
        if ok_new and st["delta"] > new_gain + TIE:
            new_step, new_gain = step, st["delta"]
        if ok_relaxed and st["delta"] > relaxed_gain + TIE:
            relaxed_step, relaxed_gain = step, st["delta"]

    nz_cand = candidates[str(nonzero_step)]
    return dict(
        old_selected_step=old_step,
        old_selected_gain=float(old_gain),
        old_selected_ci95=[
            candidates[str(old_step)]["ci_low_95"],
            candidates[str(old_step)]["ci_high_95"],
        ],
        new_selected_step=new_step,
        new_selected_gain=float(new_gain),
        new_selected_ci95=[
            candidates[str(new_step)]["ci_low_95"],
            candidates[str(new_step)]["ci_high_95"],
        ],
        relaxed_selected_step=relaxed_step,
        relaxed_selected_gain=float(relaxed_gain),
        relaxed_selected_ci95=[
            candidates[str(relaxed_step)]["ci_low_95"],
            candidates[str(relaxed_step)]["ci_high_95"],
        ],
        nonzero_selected_step=nonzero_step,
        nonzero_selected_gain=float(nz_cand["delta"]),
        nonzero_selected_ci95=[nz_cand["ci_low_95"], nz_cand["ci_high_95"]],
        family_means=means.tolist(),
        candidates=candidates,
    )


def bt_ratings(wins, anchor=0, prior=0.01):
    """Canonical H148 400-iteration MM Bradley-Terry solver, anchored at r882=1500.0."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    offdiag = 1.0 - np.eye(n)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2.0 * prior * offdiag)
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500.0 + 400.0 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families, tie_margin=1e-4):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(values.shape[1])
    return family_matrix(outcomes, families)[1]


def holm(pvals):
    keys = list(pvals.keys())
    m = len(keys)
    order = sorted(keys, key=lambda k: (pvals[k], k))
    adj = {}
    running_max = 0.0
    for rank, k in enumerate(order):
        val = min(1.0, float(pvals[k]) * (m - rank))
        running_max = max(running_max, val)
        adj[k] = running_max
    return adj


def summarize(rows, panel, col_names=COLUMNS, seed=BOOT_SEED):
    """Summarizes a [3 seeds, contexts, cols] panel with the pinned median-seed estimand and 10,000-draw shared-family bootstrap."""
    values = np.asarray(panel, dtype=np.float64)
    if (
        values.shape != (len(SEEDS), len(rows), len(col_names))
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete finite 3-seed panel required")
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context in panel")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])  # [3, F, C]
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)  # [3, C]
    boot = fm[:, ix].mean(axis=2)  # [3, DRAWS, C]

    pairs = [(f"{a}_minus_r882", a, "r882") for a in col_names[1:]]
    if tuple(col_names) == COLUMNS:
        pairs.extend(
            [
                ("W10_NEW_minus_W10_OLD", "W10_NEW", "W10_OLD"),
                ("W20_NEW_minus_W20_OLD", "W20_NEW", "W20_OLD"),
                ("W20_OLD_minus_W10_OLD", "W20_OLD", "W10_OLD"),
                ("W20_NEW_minus_W10_NEW", "W20_NEW", "W10_NEW"),
            ]
        )
    elif tuple(col_names) == DIAG_COLUMNS:
        pairs.extend(
            [
                ("W10_NONZERO_minus_W10_RELAXED", "W10_NONZERO", "W10_RELAXED"),
                ("W20_NONZERO_minus_W20_RELAXED", "W20_NONZERO", "W20_RELAXED"),
            ]
        )

    contrasts = {}
    for name, left, right in pairs:
        li, ri = col_names.index(left), col_names.index(right)
        fam_diff = fm[:, :, li] - fm[:, :, ri]
        per_seed = means[:, li] - means[:, ri]
        draws = np.median(boot[:, :, li] - boot[:, :, ri], axis=0)
        contrasts[name] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
            p_one_sided=float((1 + (draws <= 0.0).sum()) / (DRAWS + 1)),
            positive_seeds=int((per_seed > TIE).sum()),
            legacy_mean_family_seed_medians=float(np.median(fam_diff, axis=0).mean()),
        )

    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1), 0)
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2), 0), axis=0)

    inv_by_seed = {
        c: [int(np.sum(values[s, :, ci] <= -1.999999)) for s in range(len(SEEDS))]
        for ci, c in enumerate(col_names)
    }
    inv_median = {c: float(np.median(v)) for c, v in inv_by_seed.items()}

    return dict(
        contexts=len(rows),
        families=n,
        columns=list(col_names),
        contrasts=contrasts,
        family_utility_by_seed={
            c: means[:, ci].tolist() for ci, c in enumerate(col_names)
        },
        context_utility_by_seed={
            c: values[:, :, ci].mean(axis=1).tolist() for ci, c in enumerate(col_names)
        },
        invalid_counts_by_seed=inv_by_seed,
        invalid_counts_median=inv_median,
        elo_by_seed={c: elo[:, ci].tolist() for ci, c in enumerate(col_names)},
        median_elo={c: float(np.median(elo[:, ci])) for ci, c in enumerate(col_names)},
        elo_ci95={
            c: np.quantile(elo_boot[:, ci], [0.025, 0.975]).tolist()
            for ci, c in enumerate(col_names)
        },
        identical_seed_panels=bool(
            np.array_equal(values[0], values[1])
            and np.array_equal(values[1], values[2])
        ),
    )


def decide(primary_summary, nonzero_selections):
    """Applies full 205-family OOF confirmation (+0.001, LCB>0, >=2/3 seeds, Holm m=4) across PRIMARY_ARMS."""
    cs = primary_summary["contrasts"]
    raw_p = {arm: cs[f"{arm}_minus_r882"]["p_one_sided"] for arm in PRIMARY_ARMS}
    adj_p = holm(raw_p)
    arms_out = {}
    for arm in PRIMARY_ARMS:
        c = cs[f"{arm}_minus_r882"]
        gates = dict(
            margin=bool(c["median_gain"] > MARGIN + TIE),
            ci95_lower_gt_0=bool(c["ci95"][0] > 0.0),
            two_of_three_seeds_positive=bool(c["positive_seeds"] >= 2),
            holm=bool(adj_p[arm] < 0.05),
            nonzero_selection=bool(nonzero_selections.get(arm, 0) > 0),
        )
        arms_out[arm] = dict(
            holm_adjusted_p=float(adj_p[arm]),
            gates=gates,
            advance=bool(all(gates.values())),
        )

    mechanism = {}
    for w in WIDTH_ARMS:
        m_c = cs[f"{w}_NEW_minus_{w}_OLD"]
        nz_old = nonzero_selections.get(f"{w}_OLD", 0)
        nz_new = nonzero_selections.get(f"{w}_NEW", 0)
        if nz_old == 0 and nz_new == 0:
            label = "IDENTICAL_ZERO_SELECTION"
        elif (
            abs(m_c["median_gain"]) <= TIE
            and abs(m_c["ci95"][0]) <= TIE
            and abs(m_c["ci95"][1]) <= TIE
        ):
            label = "IDENTICAL_SELECTION"
        elif m_c["median_gain"] > TIE and m_c["ci95"][0] > 0.0:
            label = "NEW_SELECTOR_IMPROVES_OOF"
        elif m_c["median_gain"] < -TIE and m_c["ci95"][1] < 0.0:
            label = "NEW_SELECTOR_DEGRADES_OOF"
        else:
            label = "NEW_SELECTOR_NOISE_NEUTRAL"
        mechanism[w] = dict(
            classification=label,
            old_nonzero_cells=int(nz_old),
            new_nonzero_cells=int(nz_new),
            median_gain_new_minus_old=float(m_c["median_gain"]),
            ci95_new_minus_old=m_c["ci95"],
            p_one_sided=float(m_c["p_one_sided"]),
        )

    if all(nonzero_selections.get(a, 0) == 0 for a in PRIMARY_ARMS):
        verdict = "RULE_NEUTRAL"
    elif any(arms_out[a]["advance"] for a in PRIMARY_ARMS):
        verdict = "ADVANCE"
    else:
        verdict = "NO_ADVANCE"

    return dict(
        verdict=verdict,
        primary_arms=arms_out,
        selector_mechanism=mechanism,
    )
