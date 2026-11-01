"""Pure NumPy statistical, C-Mixup target-similarity kernel, cross-fit selection, and canonical H148 BT Elo helpers for H166."""

import hashlib
import math
import numpy as np

FOLD_SEED = 166140
SEEDS = (166101, 166102, 166103)
BOOT_SEED = 166141
ARMS = ("CTRL", "MIX02", "MIX04", "RANDMIX02")
TREATMENT_ARMS = ("MIX02", "MIX04", "RANDMIX02")
CMIX_ARMS = ("MIX02", "MIX04")
ALL_ARMS = ("r882", "CTRL", "MIX02", "MIX04", "RANDMIX02")
ARM_STEPS = {
    "CTRL": (0, 10, 20, 30, 40),
    "MIX02": (0, 10, 20, 30, 40),
    "MIX04": (0, 10, 20, 30, 40),
    "RANDMIX02": (0, 10, 20, 30, 40),
}
ARM_ALPHA = {
    "CTRL": 0.0,
    "MIX02": 0.2,
    "MIX04": 0.4,
    "RANDMIX02": 0.2,
}
ARM_PARTNER_MODE = {
    "CTRL": "none",
    "MIX02": "gaussian",
    "MIX04": "gaussian",
    "RANDMIX02": "uniform",
}
LR = 1e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.001
EPS = 1e-12
ELO_TIE_MARGIN = 1e-4
ELO_PRIOR = 0.01
ELO_ITERS = 400


def resolve_effective_cost_target(fitness, gate):
    """Resolve H59/H140 effective regression cost target: valid clamped to [-4.0, 2.0], failed-gate mapped to 2.0."""
    fit_arr = np.asarray(fitness, dtype=np.float64)
    gate_arr = np.asarray(gate, dtype=bool)
    if fit_arr.shape != gate_arr.shape:
        raise ValueError("fitness and gate shapes must match")
    return np.where(gate_arr, np.clip(fit_arr, -4.0, 2.0), 2.0)


def context_target_summary(fitness, gate):
    """Return 4D candidate-utility summary (mean, std, max, median) from gate-resolved effective utilities."""
    fit_arr = np.asarray(fitness, dtype=np.float64)
    gate_arr = np.asarray(gate, dtype=bool)
    if fit_arr.ndim != 1 or fit_arr.size == 0 or gate_arr.shape != fit_arr.shape:
        raise ValueError("invalid context fitness/gate shape")
    eff_u = -resolve_effective_cost_target(fit_arr, gate_arr)
    if not np.isfinite(eff_u).all():
        raise ValueError("nonfinite context utility summary")
    return np.array(
        [
            float(np.mean(eff_u)),
            float(np.std(eff_u)),
            float(np.max(eff_u)),
            float(np.median(eff_u)),
        ],
        dtype=np.float64,
    )


def fit_fold_cmixup_geometry(fit_cids, raw_summaries):
    """Compute fit-fold standardized target descriptors and median pairwise Euclidean bandwidth."""
    cids = sorted(fit_cids)
    if len(cids) < 2:
        raise ValueError("fit_cids must contain at least 2 contexts")
    mat = np.stack(
        [np.asarray(raw_summaries[c], dtype=np.float64) for c in cids], axis=0
    )
    if mat.ndim != 2 or not np.isfinite(mat).all():
        raise ValueError("invalid fit-fold target summaries")
    mu = np.mean(mat, axis=0)
    sigma = np.maximum(np.std(mat, axis=0), 1e-6)
    z = (mat - mu) / sigma
    diff = z[:, None, :] - z[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    triu_i, triu_j = np.triu_indices(len(cids), k=1)
    pairwise = dists[triu_i, triu_j]
    bandwidth = float(max(float(np.median(pairwise)), 1e-6))
    standardized = {c: z[idx] for idx, c in enumerate(cids)}
    return dict(
        standardized=standardized,
        bandwidth=bandwidth,
        mean=mu.tolist(),
        std=sigma.tolist(),
        pair_count=int(pairwise.size),
    )


def cmixup_partner_probs(batch_cids, standardized_map, bandwidth, mode):
    """Return (B, B) partner probability matrix for a minibatch of fit-fold contexts."""
    b = len(batch_cids)
    if b < 1:
        raise ValueError("empty batch_cids")
    if mode == "none" or b == 1:
        return np.eye(b, dtype=np.float64)
    if mode == "uniform":
        probs = np.full((b, b), 1.0 / float(b - 1), dtype=np.float64)
        np.fill_diagonal(probs, 0.0)
        return probs
    if mode != "gaussian":
        raise ValueError(f"unknown partner mode: {mode}")
    sigma = max(float(bandwidth), 1e-6)
    z = np.stack(
        [np.asarray(standardized_map[c], dtype=np.float64) for c in batch_cids],
        axis=0,
    )
    diff = z[:, None, :] - z[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    weights = np.exp(-sq_dist / (2.0 * sigma * sigma))
    np.fill_diagonal(weights, 0.0)
    probs = np.zeros((b, b), dtype=np.float64)
    for i in range(b):
        row_sum = float(np.sum(weights[i]))
        if row_sum > EPS and np.isfinite(row_sum):
            probs[i] = weights[i] / row_sum
        else:
            probs[i] = 1.0 / float(b - 1)
            probs[i, i] = 0.0
    return probs


def sample_cmixup_plan(batch_cids, standardized_map, bandwidth, arm, seed, outer, step):
    """Deterministically sample partner indices and Beta(alpha, alpha) mixing weights."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    b = len(batch_cids)
    mode = ARM_PARTNER_MODE[arm]
    alpha = ARM_ALPHA[arm]
    probs = cmixup_partner_probs(batch_cids, standardized_map, bandwidth, mode)
    if arm == "CTRL" or b == 1:
        return dict(
            partners=np.arange(b, dtype=np.int64),
            lambdas=np.ones(b, dtype=np.float64),
            probs=probs,
        )
    # Seed depends on (seed, outer, step) only so MIX02 and RANDMIX02 share identical Beta draws.
    rng = np.random.default_rng(stream_seed("cmixup", seed, outer, step))
    u_partner = rng.random(b)
    lambdas = rng.beta(alpha, alpha, size=b).astype(np.float64)
    partners = np.zeros(b, dtype=np.int64)
    for i in range(b):
        cdf = np.cumsum(probs[i])
        pick = int(np.searchsorted(cdf, u_partner[i], side="right"))
        if pick >= b:
            pick = b - 1
        if pick == i:
            pick = (i + 1) % b
        partners[i] = pick
    return dict(
        partners=partners,
        lambdas=lambdas,
        probs=probs,
    )


def mix_slate_arrays(
    progs,
    sos,
    tok,
    mask,
    rev_raw,
    rev_gate,
    a_tot,
    a_eff,
    fitness,
    gate,
    partners,
    lambdas,
):
    """Pure NumPy reference implementation of slate-level convex C-Mixup on gate-resolved targets."""
    idx = np.asarray(partners, dtype=np.int64)
    lam = np.asarray(lambdas, dtype=np.float64)
    b = len(idx)
    l1 = lam.reshape(b, 1)
    l2 = lam.reshape(b, 1, 1)
    prog_m = l2 * progs + (1.0 - l2) * progs[idx]
    sos_m = l2 * sos + (1.0 - l2) * sos[idx]
    tok_m = l2 * tok + (1.0 - l2) * tok[idx]
    mask_m = np.logical_or(mask, mask[idx])
    rev_eff = resolve_effective_cost_target(rev_raw, rev_gate > 0.5)
    rev_raw_m = l1 * rev_eff + (1.0 - l1) * rev_eff[idx]
    rev_gate_m = l1 * rev_gate + (1.0 - l1) * rev_gate[idx]
    a_tot_m = l1 * a_tot + (1.0 - l1) * a_tot[idx]
    a_eff_m = l1 * a_eff + (1.0 - l1) * a_eff[idx]
    eff_target = resolve_effective_cost_target(fitness, gate)
    fit_m = l1 * eff_target + (1.0 - l1) * eff_target[idx]
    gate_m = np.ones_like( np.asarray(gate, dtype=bool), dtype=bool)
    return dict(
        progs=prog_m,
        sos=sos_m,
        tok=tok_m,
        mask=mask_m,
        rev_raw=rev_raw_m,
        rev_gate=rev_gate_m,
        a_tot=a_tot_m,
        a_eff=a_eff_m,
        fitness=fit_m,
        gate=gate_m,
    )


def folds(rows, seed=FOLD_SEED):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) % 5 or not fams:
        raise ValueError("family count must be a positive multiple of 5")
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


def stream_seed(*parts):
    raw = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & 0x7FFFFFFF


def batch_plan(fit_cids, orders, seed, steps=40, batch=8, top=24):
    """Build 40 minibatches of same-shape (24-candidate) rank-aligned slates."""
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
            if need > 0:
                picked_pos = np.sort(rng.choice(len(rest), size=need, replace=False))
                tail = rest[picked_pos]
            else:
                tail = np.empty(0, dtype=np.int64)
            ixs.append(np.concatenate([head, tail]))
        plan.append((s, chosen, ixs))
    return plan


def family_means(rows, values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] != len(rows) or not np.isfinite(arr).all():
        raise ValueError("nonfinite or mismatched context values")
    fams = sorted({r["family_id"] for r in rows})
    idx = {f: i for i, f in enumerate(fams)}
    buckets = [[] for _ in fams]
    for r, v in zip(rows, arr):
        buckets[idx[r["family_id"]]].append(v)
    return fams, np.array([np.mean(b, axis=0) for b in buckets], dtype=np.float64)


def bootstrap(a, b, draws, seed):
    """Single-seed or 1D family bootstrap contrast."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.ndim != 1 or len(diff) == 0 or not np.isfinite(diff).all():
        raise ValueError("invalid family contrast")
    point = float(diff.mean())
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(draws, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    # Ties count against progress.
    p = float((int(np.sum(boot <= 0.0)) + 1) / (draws + 1))
    return dict(
        delta=point,
        ci_low_95=float(lo),
        ci_high_95=float(hi),
        p_one_sided=p,
        draws=draws,
    )


def bootstrap_median_of_seed_gains(fam_left, fam_right, draws, seed):
    """Shared-family bootstrap of the median over seeds of per-seed equal-family mean gains (H148)."""
    diff = np.asarray(fam_left, dtype=np.float64) - np.asarray(
        fam_right, dtype=np.float64
    )
    if diff.ndim != 2 or diff.shape[1] == 0 or not np.isfinite(diff).all():
        raise ValueError("invalid (n_seeds, n_families) contrast matrix")
    n_fams = diff.shape[1]
    per_seed = diff.mean(axis=1)
    point = float(np.median(per_seed))
    idx = np.random.default_rng(seed).integers(0, n_fams, size=(draws, n_fams))
    boot_seed_means = diff[:, idx].mean(axis=2)
    boot_med = np.median(boot_seed_means, axis=0)
    lo, hi = np.quantile(boot_med, [0.025, 0.975])
    p = float((int(np.sum(boot_med <= 0.0)) + 1) / (draws + 1))
    return dict(
        delta=point,
        ci_low_95=float(lo),
        ci_high_95=float(hi),
        p_one_sided=p,
        draws=draws,
        per_seed=per_seed.tolist(),
    )


def choose(rows, panel, steps, seed):
    steps_tuple = tuple(int(s) for s in steps)
    if not steps_tuple or steps_tuple[0] != 0:
        raise ValueError("steps must start with 0")
    mat = np.asarray(panel, dtype=np.float64)
    if mat.shape != (len(rows), len(steps_tuple)) or not np.isfinite(mat).all():
        raise ValueError("selector panel shape/finiteness mismatch")
    _, fam = family_means(rows, mat)
    base = fam[:, 0]
    best_step = 0
    best_gain = 0.0
    unconstrained_step = 0
    unconstrained_gain = 0.0
    best_nonzero_step = int(steps_tuple[1]) if len(steps_tuple) > 1 else 0
    best_nonzero_gain = -float("inf")
    candidates = {}
    for j, step in enumerate(steps_tuple):
        if step == 0:
            candidates["0"] = dict(
                delta=0.0,
                ci_low_95=0.0,
                ci_high_95=0.0,
                p_one_sided=1.0,
                draws=1000,
                eligible_strict=False,
                eligible_unconstrained=False,
            )
            continue
        stat = bootstrap(fam[:, j], base, 1000, seed + step)
        ok_unconstrained = stat["delta"] > EPS
        ok_strict = stat["delta"] > THRESHOLD + EPS and stat["ci_low_95"] > 0.0
        candidates[str(step)] = dict(
            **stat,
            eligible_strict=bool(ok_strict),
            eligible_unconstrained=bool(ok_unconstrained),
        )
        if ok_strict and stat["delta"] > best_gain + EPS:
            best_step, best_gain = step, stat["delta"]
        if ok_unconstrained and stat["delta"] > unconstrained_gain + EPS:
            unconstrained_step, unconstrained_gain = step, stat["delta"]
        if stat["delta"] > best_nonzero_gain + EPS:
            best_nonzero_step, best_nonzero_gain = step, float(stat["delta"])
    if not math.isfinite(best_nonzero_gain):
        best_nonzero_gain = 0.0
    return dict(
        steps=list(steps_tuple),
        selected_step=best_step,
        selected_gain=best_gain,
        unconstrained_selected_step=unconstrained_step,
        unconstrained_selected_gain=unconstrained_gain,
        best_nonzero_step=best_nonzero_step,
        best_nonzero_gain=best_nonzero_gain,
        candidates=candidates,
    )


def bt_ratings(wins, anchor=0, prior=ELO_PRIOR, iters=ELO_ITERS):
    """Canonical H145/H148 400-iteration MM solver with prior=0.01 and strength clipping [1e-6, 1e6]."""
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
    return 1500.0 + 400.0 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(rows, values, tie_margin=ELO_TIE_MARGIN):
    """Canonical H148 per-family pairwise win matrix with tie_margin=1e-4."""
    mat = np.asarray(values, dtype=np.float64)
    diff = mat[:, :, None] - mat[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(mat.shape[1])
    _, fam_mat = family_means(rows, outcomes)
    return fam_mat


def elo(rows, matrix, anchor=1500.0, iters=ELO_ITERS):
    """Single-panel canonical H148 BT Elo ratings anchored at r882=1500.0."""
    fw = family_wins(rows, matrix, tie_margin=ELO_TIE_MARGIN)
    ratings = bt_ratings(fw.sum(axis=0), anchor=0, prior=ELO_PRIOR, iters=iters)
    shift = float(anchor) - 1500.0
    return [float(r + shift) for r in ratings]


def holm(pvals):
    keys = list(pvals.keys())
    m = len(keys)
    order = sorted(keys, key=lambda k: pvals[k])
    adj = {}
    running_max = 0.0
    for rank, k in enumerate(order):
        val = min(1.0, float(pvals[k]) * (m - rank))
        running_max = max(running_max, val)
        adj[k] = running_max
    return adj


def invalid_counts(arr_col):
    col = np.asarray(arr_col, dtype=np.float64)
    per_seed = [int(np.sum(col[s] <= -2.0 + EPS)) for s in range(col.shape[0])]
    return dict(
        per_seed=per_seed,
        median=float(np.median(per_seed)),
        mean=float(np.mean(per_seed)),
    )


def summarize(rows, panel, arm_names=ALL_ARMS, seed=BOOT_SEED, drift_by_arm=None):
    arr = np.asarray(panel, dtype=np.float64)
    if (
        arr.shape != (len(SEEDS), len(rows), len(arm_names))
        or not np.isfinite(arr).all()
    ):
        raise ValueError("assessment panel shape/finiteness mismatch")
    fams, _ = family_means(rows, arr[0, :, 0])
    # fam has shape (n_seeds, n_arms, n_fams)
    fam = np.stack(
        [
            [family_means(rows, arr[s, :, m])[1] for m in range(len(arm_names))]
            for s in range(len(SEEDS))
        ],
        axis=0,
    )
    n_fams = len(fams)
    # Canonical H148 per-seed Elo and 1,000-draw shared-family bootstrap of median-over-seeds Elo
    fw = np.stack(
        [family_wins(rows, arr[s], tie_margin=ELO_TIE_MARGIN) for s in range(len(SEEDS))],
        axis=0,
    )
    elo_by_seed = bt_ratings(fw.sum(axis=1), anchor=0, prior=ELO_PRIOR, iters=ELO_ITERS)
    elo_med = np.median(elo_by_seed, axis=0)
    ei = np.random.default_rng(seed + 500).integers(0, n_fams, size=(1000, n_fams))
    elo_boot = np.median(
        bt_ratings(fw[:, ei].sum(axis=2), anchor=0, prior=ELO_PRIOR, iters=ELO_ITERS),
        axis=0,
    )
    elo_lo, elo_hi = np.quantile(elo_boot, [0.025, 0.975], axis=0)

    arm_index = {name: i for i, name in enumerate(arm_names)}
    idx_ctrl = arm_index["CTRL"]
    idx_randmix = arm_index["RANDMIX02"]

    contrasts = {}
    raw_p_all = {}
    raw_p_treatment = {}
    for idx_m, name in enumerate(arm_names[1:], start=1):
        st = bootstrap_median_of_seed_gains(
            fam[:, idx_m, :], fam[:, 0, :], 10000, seed + idx_m
        )
        st_ctrl = bootstrap_median_of_seed_gains(
            fam[:, idx_m, :], fam[:, idx_ctrl, :], 10000, seed + 100 + idx_m
        )
        st_randmix = bootstrap_median_of_seed_gains(
            fam[:, idx_m, :], fam[:, idx_randmix, :], 10000, seed + 200 + idx_m
        )
        per_seed = st["per_seed"]
        per_seed_vs_ctrl = st_ctrl["per_seed"]
        per_seed_vs_randmix = st_randmix["per_seed"]
        per_seed_fam_means = [float(fam[s, idx_m].mean()) for s in range(len(SEEDS))]
        per_seed_ctx_means = [float(arr[s, :, idx_m].mean()) for s in range(len(SEEDS))]
        inv = invalid_counts(arr[:, :, idx_m])
        entry = dict(
            delta=float(st["delta"]),
            ci_low_95=float(st["ci_low_95"]),
            ci_high_95=float(st["ci_high_95"]),
            p_one_sided=float(st["p_one_sided"]),
            draws=int(st["draws"]),
            per_seed_equal_family_delta=per_seed,
            positive_seeds=int(sum(x > EPS for x in per_seed)),
            raw_context_mean=float(np.median(per_seed_ctx_means)),
            equal_family_mean=float(np.median(per_seed_fam_means)),
            per_seed_equal_family_mean=per_seed_fam_means,
            elo=float(elo_med[idx_m]),
            elo_by_seed=elo_by_seed[:, idx_m].tolist(),
            elo_delta_vs_r882=float(elo_med[idx_m] - elo_med[0]),
            elo_ci_low_95=float(elo_lo[idx_m]),
            elo_ci_high_95=float(elo_hi[idx_m]),
            delta_vs_ctrl=float(st_ctrl["delta"]),
            ci_low_95_vs_ctrl=float(st_ctrl["ci_low_95"]),
            ci_high_95_vs_ctrl=float(st_ctrl["ci_high_95"]),
            p_one_sided_vs_ctrl=float(st_ctrl["p_one_sided"]),
            per_seed_delta_vs_ctrl=per_seed_vs_ctrl,
            delta_vs_randmix02=float(st_randmix["delta"]),
            ci_low_95_vs_randmix02=float(st_randmix["ci_low_95"]),
            ci_high_95_vs_randmix02=float(st_randmix["ci_high_95"]),
            p_one_sided_vs_randmix02=float(st_randmix["p_one_sided"]),
            per_seed_delta_vs_randmix02=per_seed_vs_randmix,
            minus2_invalid_counts=inv,
        )
        if drift_by_arm and name in drift_by_arm:
            entry["adapter_l2_drift"] = drift_by_arm[name]
        contrasts[name] = entry
        raw_p_all[name] = st["p_one_sided"]
        if name in TREATMENT_ARMS:
            raw_p_treatment[name] = st["p_one_sided"]

    adj_p_treatment = holm(raw_p_treatment)
    adj_p_all = holm(raw_p_all)
    for name in contrasts:
        contrasts[name]["p_holm_all4"] = float(adj_p_all[name])
        contrasts[name]["p_holm"] = float(
            adj_p_treatment.get(name, adj_p_all[name])
        )
        is_treatment = name in TREATMENT_ARMS
        contrasts[name]["advance"] = bool(
            is_treatment
            and contrasts[name]["delta"] > THRESHOLD + EPS
            and contrasts[name]["ci_low_95"] > 0.0
            and contrasts[name]["p_holm"] < 0.05
        )

    r882_fam_means = [float(fam[s, 0].mean()) for s in range(len(SEEDS))]
    r882_ctx_means = [float(arr[s, :, 0].mean()) for s in range(len(SEEDS))]
    return dict(
        family_count=n_fams,
        context_count=len(rows),
        r882_raw_context_mean=float(np.median(r882_ctx_means)),
        r882_equal_family_mean=float(np.median(r882_fam_means)),
        r882_elo=float(elo_med[0]),
        r882_minus2_invalid_counts=invalid_counts(arr[:, :, 0]),
        arms=contrasts,
    )


def verdict(train_summary, val_summary, selections):
    if any(train_summary["arms"][a]["advance"] for a in TREATMENT_ARMS):
        return "ADVANCE"
    return "NO_ADVANCE"
