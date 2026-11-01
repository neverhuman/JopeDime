"""H173 scientific logic: FIT-only r882 regret weights, 5-fold family cross-fit, strict inner-B selector, and 205-family OOF confirmation."""

import numpy as np

SEEDS = (173101, 173102, 173103)
FOLD_SEED = 173140
BOOT_SEED = 173141
STEPS = (0, 10, 20, 30, 40)
MARGIN = 0.001
TIE = 1e-12
EPS_FLOOR = 1e-6
ELO_TIE_MARGIN = 1e-4
WEIGHT_CLIP_MIN = 0.2
WEIGHT_CLIP_MAX = 5.0

ARMS = ("CTRL", "RW05", "RW10", "RW10_SHUF")
TREATMENT_ARMS = ("RW05", "RW10", "RW10_SHUF")
HARDNESS_ARMS = ("RW05", "RW10")
COLUMNS = ("r882",) + ARMS

SHUF_SEED_FORMULA = "seed + 1000 * outer + 77"
SELECTOR_SEED_FORMULA = "boot_seed + 1000 * outer + (seed % 1000)"


def shuf_seed_for(seed, outer):
    return int(seed) + 1000 * int(outer) + 77


def selector_seed_for(outer, seed, boot_seed=BOOT_SEED):
    return int(boot_seed) + 1000 * int(outer) + (int(seed) % 1000)


DIAG_UNCONSTRAINED_ARMS = (
    "CTRL_UNCONSTRAINED",
    "RW05_UNCONSTRAINED",
    "RW10_UNCONSTRAINED",
    "RW10_SHUF_UNCONSTRAINED",
)
DIAG_UNCONSTRAINED_COLUMNS = ("r882",) + DIAG_UNCONSTRAINED_ARMS

DIAG_NONZERO_ARMS = (
    "CTRL_NONZERO",
    "RW05_NONZERO",
    "RW10_NONZERO",
    "RW10_SHUF_NONZERO",
)
DIAG_NONZERO_COLUMNS = ("r882",) + DIAG_NONZERO_ARMS

GAMMAS = {
    "CTRL": 0.0,
    "RW05": 0.5,
    "RW10": 1.0,
}


def folds(rows, seed=FOLD_SEED):
    cids = [r["context_sha256"] for r in rows]
    if len(cids) != len(set(cids)):
        raise ValueError("duplicate context_sha256 in rows")
    if any(r.get("dl_split") != "DL-Train" or r.get("has_meta_test") for r in rows):
        raise ValueError("non-train or meta-test row in folds")
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError(f"expected 205 families, got {len(fams)}")
    perm = list(np.random.default_rng(seed).permutation(fams))
    return [sorted(perm[i * 41 : (i + 1) * 41]) for i in range(5)]


def partition(rows, outer, seed=FOLD_SEED):
    groups = folds(rows, seed)
    assess = set(groups[outer])
    select = set(groups[(outer + 1) % 5])
    fit = {
        f for i, g in enumerate(groups) if i not in (outer, (outer + 1) % 5) for f in g
    }
    assert len(fit) == 123 and len(select) == 41 and len(assess) == 41
    assert not (fit & select or fit & assess or select & assess)

    def by(s):
        return [r for r in rows if r["family_id"] in s]

    return by(fit), by(select), by(assess)


def batch_plan(fit_ids, orders, seed, steps=40, batch_size=8, top=24):
    cids = sorted(fit_ids)
    plan = []
    for step in range(1, steps + 1):
        stream = seed * 1000 + step
        rng = np.random.default_rng(stream)
        chosen = [
            cids[i] for i in rng.choice(len(cids), size=batch_size, replace=False)
        ]
        indices = []
        for cid in chosen:
            order = np.asarray(orders[cid], dtype=np.int64)
            if len(order) <= top:
                idx = order.copy()
            else:
                head = order[:4]
                tail = rng.choice(order[4:], size=top - 4, replace=False)
                idx = np.concatenate([head, np.sort(tail)])
            indices.append(idx)
        plan.append((stream, chosen, indices))
    return plan


def _power_clip_renorm(regrets, eps, gamma):
    if abs(gamma) <= 1e-12:
        return np.ones(len(regrets), dtype=np.float64)
    u = np.power(np.asarray(regrets, dtype=np.float64) + float(eps), float(gamma))
    m1 = float(np.mean(u))
    if not np.isfinite(m1) or m1 <= 0.0:
        raise ValueError("non-positive initial weight mean")
    w1 = u / m1
    w2 = np.clip(w1, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
    m2 = float(np.mean(w2))
    if not np.isfinite(m2) or m2 <= 0.0:
        raise ValueError("non-positive clipped weight mean")
    return w2 / m2


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("weights must be non-empty 1D array")
    denom = float(np.sum(w * w))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError("invalid weight sum of squares")
    ess = float((np.sum(w) ** 2) / denom)
    return dict(
        ess=ess,
        ess_ratio=float(ess / len(w)),
        n_contexts=int(len(w)),
        mean=float(np.mean(w)),
        std=float(np.std(w)),
        min=float(np.min(w)),
        median=float(np.median(w)),
        max=float(np.max(w)),
    )


def compute_fit_regret_weights(
    fit_ids, pool_regret_sources, seed, outer, excluded_ids=None
):
    """Computes per-context weights w_c ONCE per cell from frozen r882 step-0 costs on the FIT split ONLY.

    regret_c = cost(r882 argmin) - min cost in the pool (>= 0), where realized candidate cost is
    fitness[j] if gate[j] else 2.0 (= -utility).
    w_c = (regret_c + eps)^gamma with eps = median FIT regret, normalized to FIT mean 1,
    clipped to [0.2, 5.0], and renormalized to FIT mean 1.
    RW10_SHUF permutes the RW10 weight vector across FIT contexts using a deterministic seed-fixed permutation.
    Never reads or depends on B, A, or DL-Val contexts.
    """
    fit_cids = sorted(set(fit_ids))
    if len(fit_cids) != len(list(fit_ids)):
        raise ValueError("duplicate context IDs in fit_ids")
    if len(fit_cids) == 0:
        raise ValueError("empty fit_ids")
    if excluded_ids is not None:
        overlap = set(fit_cids) & set(excluded_ids)
        if overlap:
            raise ValueError(
                f"FIT split overlaps excluded B/A/Val contexts: {sorted(overlap)[:3]}"
            )

    regrets = np.empty(len(fit_cids), dtype=np.float64)
    regret_by_cid = {}
    for i, cid in enumerate(fit_cids):
        if cid not in pool_regret_sources:
            raise KeyError(f"missing FIT context in pool_regret_sources: {cid}")
        entry = pool_regret_sources[cid]
        fitness = np.asarray(entry["fitness"], dtype=np.float64)
        gate = np.asarray(entry["gate"], dtype=bool)
        if fitness.shape != gate.shape or fitness.ndim != 1 or len(fitness) == 0:
            raise ValueError(f"invalid fitness/gate shape for {cid}")
        choice = int(entry["r882_choice"])
        if choice < 0 or choice >= len(fitness):
            raise ValueError(f"r882_choice out of bounds for {cid}: {choice}")
        realized_costs = np.where(gate, fitness, 2.0)
        r882_cost = float(realized_costs[choice])
        min_cost = float(np.min(realized_costs))
        reg = max(0.0, float(r882_cost - min_cost))
        regrets[i] = reg
        regret_by_cid[cid] = reg

    raw_median_regret = float(np.median(regrets))
    eps = max(raw_median_regret, EPS_FLOOR)

    w_ctrl = _power_clip_renorm(regrets, eps, GAMMAS["CTRL"])
    w_rw05 = _power_clip_renorm(regrets, eps, GAMMAS["RW05"])
    w_rw10 = _power_clip_renorm(regrets, eps, GAMMAS["RW10"])

    shuf_seed = shuf_seed_for(seed, outer)
    perm = np.random.default_rng(shuf_seed).permutation(len(fit_cids))
    w_rw10_shuf = w_rw10[perm].copy()

    vectors = {
        "CTRL": w_ctrl,
        "RW05": w_rw05,
        "RW10": w_rw10,
        "RW10_SHUF": w_rw10_shuf,
    }
    weights_by_arm = {
        arm: {cid: float(vec[i]) for i, cid in enumerate(fit_cids)}
        for arm, vec in vectors.items()
    }
    ess_by_arm = {arm: effective_sample_size(vec) for arm, vec in vectors.items()}

    return dict(
        fit_cids=fit_cids,
        raw_median_fit_regret=raw_median_regret,
        eps=eps,
        zero_regret_count=int(np.sum(regrets <= 1e-12)),
        positive_regret_count=int(np.sum(regrets > 1e-12)),
        max_fit_regret=float(np.max(regrets)),
        mean_fit_regret=float(np.mean(regrets)),
        regret_by_cid=regret_by_cid,
        weights_by_arm=weights_by_arm,
        ess_by_arm=ess_by_arm,
        shuf_seed=shuf_seed,
        shuf_perm=perm.tolist(),
    )


def family_matrix(rows, values):
    arr = np.asarray(values, dtype=np.float64)
    fam_ids = [r["family_id"] if isinstance(r, dict) else str(r) for r in rows]
    fams = sorted(set(fam_ids))
    idx = {f: [i for i, fid in enumerate(fam_ids) if fid == f] for f in fams}
    out = np.empty((len(fams),) + arr.shape[1:], dtype=np.float64)
    for j, f in enumerate(fams):
        out[j] = arr[idx[f]].mean(axis=0)
    return fams, out


def choose_strict_and_diagnostics(rows, panel, seed=BOOT_SEED, draws=1000):
    """Evaluates candidate steps (0, 10, 20, 30, 40) on inner fold B (41 families).

    Strict primary selector:
      - earliest best step with gain > +0.001 + 1e-12 AND 1,000-draw paired family-bootstrap LCB95 > 0, else 0.
    Descriptive selectors locked simultaneously from the same inner-B panel:
      - unconstrained_selected_step: argmax_{t in {0,10,20,30,40}} gain(t) (ties within 1e-12 -> earliest step).
      - nonzero_selected_step: argmax_{t in {10,20,30,40}} gain(t) (ties within 1e-12 -> earliest step).
    """
    _, fam = family_matrix(rows, panel)
    base = fam[:, 0]
    strict_best_step, strict_best_gain = 0, 0.0
    unconstrained_best_step, unconstrained_best_gain = 0, 0.0
    nonzero_best_step, nonzero_best_gain = 10, -np.inf

    candidates = {
        "0": dict(
            gain=0.0,
            ci_low_95=0.0,
            ci_high_95=0.0,
            p_one_sided=1.0,
            strict_eligible=False,
        )
    }
    for j, step in enumerate(STEPS[1:], start=1):
        diff = fam[:, j] - base
        gain = float(diff.mean())
        rng = np.random.default_rng(seed + j)
        sample = rng.integers(0, len(diff), size=(draws, len(diff)))
        means = diff[sample].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        p = float((1 + np.sum(means <= 0.0)) / (draws + 1))
        strict_ok = bool(gain > MARGIN + TIE and low > 0.0)
        candidates[str(step)] = dict(
            gain=gain,
            ci_low_95=float(low),
            ci_high_95=float(high),
            p_one_sided=p,
            strict_eligible=strict_ok,
        )
        if strict_ok and gain > strict_best_gain + TIE:
            strict_best_step, strict_best_gain = step, gain
        if gain > unconstrained_best_gain + TIE:
            unconstrained_best_step, unconstrained_best_gain = step, gain
        if gain > nonzero_best_gain + TIE:
            nonzero_best_step, nonzero_best_gain = step, gain

    return dict(
        selected_step=strict_best_step,
        selected_gain=float(strict_best_gain),
        unconstrained_selected_step=unconstrained_best_step,
        unconstrained_selected_gain=float(unconstrained_best_gain),
        nonzero_selected_step=nonzero_best_step,
        nonzero_selected_gain=float(nonzero_best_gain),
        candidates=candidates,
    )


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj


def bt_ratings(wins, anchor=0, base=1500.0, scale=400.0, iters=400, prior=0.01):
    """Canonical H148 400-iteration MM Bradley-Terry solver, anchored at r882=1500.0."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    offdiag = 1.0 - np.eye(n)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1], dtype=np.float64)
    for _ in range(iters):
        denom = (
            (games + 2.0 * prior * offdiag)
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / np.maximum(denom, 1e-12), 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return base + scale * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, rows, tie_margin=ELO_TIE_MARGIN):
    """Context-level pairwise win/tie/loss outcomes (1e-4 tie margin) aggregated equally by family (H148 canonical)."""
    val = np.asarray(values, dtype=np.float64)
    diff = val[:, :, None] - val[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(val.shape[1])
    return family_matrix(rows, outcomes)[1]


def _pair_wins(values, rows=None, tie_margin=ELO_TIE_MARGIN):
    val = np.asarray(values, dtype=np.float64)
    if rows is None:
        rows = [f"fam_{i:04d}" for i in range(len(val))]
    return family_wins(val, rows, tie_margin=tie_margin).sum(axis=0)


def elo_panel(
    rows, panel, seed=BOOT_SEED, draws=1000, cols=COLUMNS, tie_margin=ELO_TIE_MARGIN
):
    val_panel = np.asarray(panel, dtype=np.float64)
    fw = np.stack(
        [
            family_wins(val_panel[s], rows, tie_margin=tie_margin)
            for s in range(val_panel.shape[0])
        ]
    )
    n_fams = fw.shape[1]
    point = bt_ratings(fw.sum(axis=1), anchor=0)
    med = np.median(point, axis=0)
    rng = np.random.default_rng(seed + 900)
    idx = rng.integers(0, n_fams, size=(draws, n_fams))
    boot = np.median(bt_ratings(fw[:, idx].sum(axis=2), anchor=0), axis=0)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return dict(
        elo_by_seed={c: point[:, i].tolist() for i, c in enumerate(cols)},
        median_elo={c: float(med[i]) for i, c in enumerate(cols)},
        elo_ci95={c: [float(low[i]), float(high[i])] for i, c in enumerate(cols)},
    )


def summarize(rows, panel, cols=COLUMNS, seed=BOOT_SEED, draws=10000):
    """Computes seed-level equal-family gains, 10,000-draw shared-family bootstrap CIs, and BT Elo ratings."""
    if panel.ndim == 4:
        panel = panel.mean(axis=1)
    fams, fam = family_matrix(rows, np.swapaxes(panel, 0, 1))
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(fams), size=(draws, len(fams)))

    if cols == COLUMNS:
        pairs = [
            ("CTRL_minus_r882", 1, 0),
            ("RW05_minus_r882", 2, 0),
            ("RW10_minus_r882", 3, 0),
            ("RW10_SHUF_minus_r882", 4, 0),
            ("RW05_minus_CTRL", 2, 1),
            ("RW10_minus_CTRL", 3, 1),
            ("RW10_SHUF_minus_CTRL", 4, 1),
            ("RW10_minus_RW10_SHUF", 3, 4),
            ("RW05_minus_RW10_SHUF", 2, 4),
            ("RW10_minus_RW05", 3, 2),
        ]
    else:
        pairs = [
            (f"{cols[1]}_minus_r882", 1, 0),
            (f"{cols[2]}_minus_r882", 2, 0),
            (f"{cols[3]}_minus_r882", 3, 0),
            (f"{cols[4]}_minus_r882", 4, 0),
            (f"{cols[2]}_minus_{cols[1]}", 2, 1),
            (f"{cols[3]}_minus_{cols[1]}", 3, 1),
            (f"{cols[4]}_minus_{cols[1]}", 4, 1),
            (f"{cols[3]}_minus_{cols[4]}", 3, 4),
        ]

    contrasts = {}
    for name, a, b in pairs:
        diff = fam[:, :, a] - fam[:, :, b]
        per_seed = diff.mean(axis=0)
        med = float(np.median(per_seed))
        boot_seed_means = diff[sample].mean(axis=1)
        boot_med = np.median(boot_seed_means, axis=1)
        low, high = np.quantile(boot_med, [0.025, 0.975])
        p = float((1 + np.sum(boot_med <= 0.0)) / (draws + 1))
        legacy_fam_med = float(np.median(diff, axis=1).mean())
        contrasts[name] = dict(
            per_seed_gain=[float(x) for x in per_seed],
            median_gain=med,
            ci95=[float(low), float(high)],
            p_one_sided=p,
            positive_seeds=int(np.sum(per_seed > 0.0)),
            legacy_mean_family_seed_medians=legacy_fam_med,
        )

    elo = elo_panel(rows, panel, seed=seed, cols=cols)
    invalids = (panel <= -2.0 + 1e-9).sum(axis=1)
    return dict(
        contexts=len(rows),
        families=len(fams),
        columns=list(cols),
        contrasts=contrasts,
        family_utility_by_seed={
            c: [float(x) for x in fam[:, :, i].mean(axis=0)] for i, c in enumerate(cols)
        },
        context_utility_by_seed={
            c: [float(x) for x in panel[:, :, i].mean(axis=1)]
            for i, c in enumerate(cols)
        },
        invalid_counts_by_seed={
            c: [int(x) for x in invalids[:, i]] for i, c in enumerate(cols)
        },
        invalid_counts_median={
            c: float(np.median(invalids[:, i])) for i, c in enumerate(cols)
        },
        **elo,
        identical_seed_panels=bool(
            np.allclose(panel[0], panel[1]) and np.allclose(panel[1], panel[2])
        ),
    )


def pseudoreplication_report(panel, selected_steps_by_arm_and_seed):
    """Checks per-arm whether seeds produce identical selector step choices or identical OOF utility panels (H160 lesson)."""
    report = {}
    r882_vec = panel[0, :, 0]
    for arm_idx, arm in enumerate(ARMS, start=1):
        steps_3 = selected_steps_by_arm_and_seed[arm]
        s0, s1, s2 = steps_3[0], steps_3[1], steps_3[2]
        u0, u1, u2 = panel[0, :, arm_idx], panel[1, :, arm_idx], panel[2, :, arm_idx]
        c01 = bool(s0 == s1)
        c02 = bool(s0 == s2)
        c12 = bool(s1 == s2)
        p01 = bool(np.allclose(u0, u1, atol=0.0, rtol=0.0))
        p02 = bool(np.allclose(u0, u2, atol=0.0, rtol=0.0))
        p12 = bool(np.allclose(u1, u2, atol=0.0, rtol=0.0))
        report[arm] = dict(
            selected_steps_by_seed=steps_3,
            nonzero_cells_total=int(sum(1 for s in steps_3 for step in s if step > 0)),
            all_baseline_selected=bool(all(step == 0 for s in steps_3 for step in s)),
            identical_choices_all_3_seeds=bool(c01 and c12),
            identical_choices_any_seed_pair=bool(c01 or c02 or c12),
            identical_choices_pairs=dict(s0_s1=c01, s0_s2=c02, s1_s2=c12),
            identical_oof_panel_all_3_seeds=bool(p01 and p12),
            identical_oof_panel_any_seed_pair=bool(p01 or p02 or p12),
            identical_oof_panel_pairs=dict(s0_s1=p01, s0_s2=p02, s1_s2=p12),
            identical_oof_panel_to_r882_by_seed=[
                bool(np.allclose(panel[si, :, arm_idx], r882_vec, atol=0.0, rtol=0.0))
                for si in range(3)
            ],
            identical_oof_panel_to_ctrl_by_seed=[
                bool(
                    np.allclose(
                        panel[si, :, arm_idx], panel[si, :, 1], atol=0.0, rtol=0.0
                    )
                )
                for si in range(3)
            ],
        )
    return report


def decide(primary, nonzero_selections):
    """Evaluates Holm-Bonferroni over treatment arms (m=3: RW05, RW10, RW10_SHUF) and all 4 arms (m=4)."""
    treat_names = [f"{a}_minus_r882" for a in TREATMENT_ARMS]
    treat_pvals = [primary["contrasts"][k]["p_one_sided"] for k in treat_names]
    adj_m3 = holm(treat_pvals)
    adj_m3_map = dict(zip(TREATMENT_ARMS, adj_m3, strict=True))

    all_names = [f"{a}_minus_r882" for a in ARMS]
    all_pvals = [primary["contrasts"][k]["p_one_sided"] for k in all_names]
    adj_m4 = holm(all_pvals)
    adj_m4_map = dict(zip(ARMS, adj_m4, strict=True))

    arms_out = {}
    for arm in ARMS:
        c = primary["contrasts"][f"{arm}_minus_r882"]
        holm_p = adj_m3_map.get(arm, adj_m4_map[arm])
        qualifies_as_hardness_arm = arm in HARDNESS_ARMS
        gates = dict(
            qualifies_as_hardness_arm=qualifies_as_hardness_arm,
            margin=c["median_gain"] > MARGIN + TIE,
            ci95_lower_gt_0=c["ci95"][0] > 0.0,
            two_of_three_seeds_positive=c["positive_seeds"] >= 2,
            holm_m3=holm_p < 0.05,
            holm_m4=adj_m4_map[arm] < 0.05,
            nonzero_selection=nonzero_selections[arm] > 0,
        )
        arms_out[arm] = dict(
            holm_m3_adjusted_p=float(holm_p) if arm in adj_m3_map else None,
            holm_m4_adjusted_p=float(adj_m4_map[arm]),
            gates=gates,
            advance=bool(
                qualifies_as_hardness_arm
                and gates["margin"]
                and gates["ci95_lower_gt_0"]
                and gates["two_of_three_seeds_positive"]
                and gates["holm_m3"]
                and gates["nonzero_selection"]
            ),
        )

    rw10_vs_shuf = primary["contrasts"]["RW10_minus_RW10_SHUF"]
    if arms_out["RW10"]["advance"] and rw10_vs_shuf["ci95"][0] > 0.0:
        mech = "HARDNESS_SIGNAL_ADVANCES_OVER_SHUFFLED_PLACEBO"
    elif arms_out["RW10"]["advance"] or arms_out["RW05"]["advance"]:
        mech = "REWEIGHTING_ADVANCES_WITHOUT_CLEAN_SHUFFLE_SEPARATION"
    elif all(nonzero_selections[a] == 0 for a in ARMS):
        mech = "RULE_NEUTRAL_ALL_ZERO_SELECTION"
    else:
        mech = "NOISE_NEUTRAL_NO_OOF_GAIN"

    verdict = (
        "ADVANCE"
        if any(arms_out[a]["advance"] for a in HARDNESS_ARMS)
        else "NO_ADVANCE"
    )
    return dict(
        verdict=verdict,
        mechanism_classification=mech,
        primary_arms=arms_out,
        hardness_vs_placebo=dict(
            median_gain_rw10_minus_shuf=rw10_vs_shuf["median_gain"],
            ci95_rw10_minus_shuf=rw10_vs_shuf["ci95"],
            p_one_sided=rw10_vs_shuf["p_one_sided"],
        ),
    )
