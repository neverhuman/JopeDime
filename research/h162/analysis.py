"""Pure NumPy statistical, cosine warm-restart LR schedule, cross-fit selection, and Elo helpers for H162."""

import hashlib
import math
import numpy as np

FOLD_SEED = 162140
SEEDS = (162101, 162102, 162103)
BOOT_SEED = 162141
ARMS = ("CTRL", "COS_R2", "COS_R4", "CTRL_LRMATCH")
COS_ARMS = ("COS_R2", "COS_R4")
ALL_ARMS = ("r882", "CTRL", "COS_R2", "COS_R4", "CTRL_LRMATCH")
ARM_STEPS = {
    "CTRL": (0, 10, 20, 30, 40),
    "COS_R2": (0, 20, 40),
    "COS_R4": (0, 10, 20, 30, 40),
    "CTRL_LRMATCH": (0, 10, 20, 30, 40),
}
ALL_SNAPSHOT_STEPS = (0, 10, 20, 30, 40)
CTRL_LR = 1e-4
COS_PEAK_LR = 2e-4
COS_MIN_LR = 1e-6
CTRL_LRMATCH_LR = 0.5 * (COS_PEAK_LR + COS_MIN_LR)  # 1.005e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.001
EPS = 1e-12


def cosine_cycle_lr(step_in_cycle, cycle_len, lr_max=COS_PEAK_LR, lr_min=COS_MIN_LR):
    if cycle_len <= 1:
        raise ValueError("cycle_len must be > 1")
    if not (0 <= step_in_cycle < cycle_len):
        raise ValueError("step_in_cycle out of bounds")
    cos_val = math.cos(math.pi * float(step_in_cycle) / float(cycle_len - 1))
    return float(lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos_val))


def lr_schedule(arm, total_steps=40):
    if total_steps <= 0 or total_steps % 20 != 0:
        raise ValueError("total_steps must be a positive multiple of 20")
    if arm == "CTRL":
        return [float(CTRL_LR) for _ in range(total_steps)]
    if arm == "COS_R2":
        return [cosine_cycle_lr(s % 20, 20) for s in range(total_steps)]
    if arm == "COS_R4":
        return [cosine_cycle_lr(s % 10, 10) for s in range(total_steps)]
    if arm == "CTRL_LRMATCH":
        mean_r2 = float(np.mean(lr_schedule("COS_R2", total_steps)))
        mean_r4 = float(np.mean(lr_schedule("COS_R4", total_steps)))
        if abs(mean_r2 - mean_r4) > EPS or abs(mean_r2 - CTRL_LRMATCH_LR) > EPS:
            raise ValueError("cosine schedule mean LR mismatch")
        return [float(mean_r2) for _ in range(total_steps)]
    raise ValueError(f"unknown arm: {arm}")


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
    return dict(
        steps=list(steps_tuple),
        selected_step=best_step,
        selected_gain=best_gain,
        unconstrained_selected_step=unconstrained_step,
        unconstrained_selected_gain=unconstrained_gain,
        candidates=candidates,
    )


def elo(rows, matrix, anchor=1500.0, iters=400):
    mat = np.asarray(matrix, dtype=np.float64)
    _, fam_idx = np.unique([r["family_id"] for r in rows], return_inverse=True)
    counts = np.bincount(fam_idx)
    weights = 1.0 / (len(counts) * counts[fam_idx])
    m = mat.shape[1]
    wins = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            d = mat[:, i] - mat[:, j]
            w_ij = float(
                np.sum(weights[d > EPS]) + 0.5 * np.sum(weights[np.abs(d) <= EPS])
            )
            wins[i, j] = w_ij
            wins[j, i] = 1.0 - w_ij
    p = np.ones(m, dtype=np.float64)
    for _ in range(iters):
        nxt = np.ones(m, dtype=np.float64)
        for i in range(m):
            w = wins[i].sum()
            denom = sum(
                (wins[i, j] + wins[j, i]) / (p[i] + p[j]) for j in range(m) if i != j
            )
            nxt[i] = max(w / max(denom, EPS), EPS)
        p = nxt / nxt[0]
    scale = 400.0 / math.log(10.0)
    return [float(anchor + scale * math.log(max(x, EPS))) for x in p]


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
    fam = np.stack(
        [
            [family_means(rows, arr[s, :, m])[1] for m in range(len(arm_names))]
            for s in range(len(SEEDS))
        ],
        axis=0,
    )
    med = np.median(fam, axis=0)
    med_ctx = np.median(arr, axis=0)
    elo_med = elo(rows, med_ctx)

    # 200-draw family bootstrap for Elo CIs
    rng = np.random.default_rng(seed + 500)
    fam_to_indices = {}
    for idx_r, r in enumerate(rows):
        fam_to_indices.setdefault(r["family_id"], []).append(idx_r)
    elo_draws = np.zeros((200, len(arm_names)), dtype=np.float64)
    for d in range(200):
        picked_fams = rng.choice(len(fams), size=len(fams), replace=True)
        boot_rows = []
        boot_vals = []
        for b_idx, fi in enumerate(picked_fams):
            f_id = fams[fi]
            for r_i in fam_to_indices[f_id]:
                boot_rows.append({"family_id": f"bfam_{b_idx}"})
                boot_vals.append(med_ctx[r_i])
        elo_draws[d] = elo(boot_rows, np.asarray(boot_vals, dtype=np.float64))
    elo_lo = np.quantile(elo_draws, 0.025, axis=0)
    elo_hi = np.quantile(elo_draws, 0.975, axis=0)

    arm_index = {name: i for i, name in enumerate(arm_names)}
    idx_ctrl = arm_index["CTRL"]
    idx_lrmatch = arm_index["CTRL_LRMATCH"]

    contrasts = {}
    raw_p = {}
    for idx_m, name in enumerate(arm_names[1:], start=1):
        st = bootstrap(med[idx_m], med[0], 10000, seed + idx_m)
        st_lrmatch = bootstrap(med[idx_m], med[idx_lrmatch], 10000, seed + 100 + idx_m)
        st_ctrl = bootstrap(med[idx_m], med[idx_ctrl], 10000, seed + 200 + idx_m)
        per_seed = [
            float((fam[s, idx_m] - fam[s, 0]).mean()) for s in range(len(SEEDS))
        ]
        per_seed_vs_lrmatch = [
            float((fam[s, idx_m] - fam[s, idx_lrmatch]).mean())
            for s in range(len(SEEDS))
        ]
        inv = invalid_counts(arr[:, :, idx_m])
        entry = dict(
            **st,
            per_seed_equal_family_delta=per_seed,
            positive_seeds=int(sum(x > EPS for x in per_seed)),
            raw_context_mean=float(med_ctx[:, idx_m].mean()),
            equal_family_mean=float(med[idx_m].mean()),
            elo=float(elo_med[idx_m]),
            elo_delta_vs_r882=float(elo_med[idx_m] - elo_med[0]),
            elo_ci_low_95=float(elo_lo[idx_m]),
            elo_ci_high_95=float(elo_hi[idx_m]),
            delta_vs_ctrl_lrmatch=float(st_lrmatch["delta"]),
            ci_low_95_vs_ctrl_lrmatch=float(st_lrmatch["ci_low_95"]),
            ci_high_95_vs_ctrl_lrmatch=float(st_lrmatch["ci_high_95"]),
            p_one_sided_vs_ctrl_lrmatch=float(st_lrmatch["p_one_sided"]),
            per_seed_delta_vs_ctrl_lrmatch=per_seed_vs_lrmatch,
            delta_vs_ctrl=float(st_ctrl["delta"]),
            ci_low_95_vs_ctrl=float(st_ctrl["ci_low_95"]),
            ci_high_95_vs_ctrl=float(st_ctrl["ci_high_95"]),
            p_one_sided_vs_ctrl=float(st_ctrl["p_one_sided"]),
            minus2_invalid_counts=inv,
        )
        if drift_by_arm and name in drift_by_arm:
            entry["adapter_l2_drift"] = drift_by_arm[name]
        contrasts[name] = entry
        raw_p[name] = st["p_one_sided"]
    adj_p = holm(raw_p)
    for name, p_adj in adj_p.items():
        contrasts[name]["p_holm"] = float(p_adj)
        is_cos = name in COS_ARMS
        contrasts[name]["advance"] = bool(
            is_cos
            and contrasts[name]["delta"] > THRESHOLD + EPS
            and contrasts[name]["ci_low_95"] > 0.0
            and contrasts[name]["delta_vs_ctrl_lrmatch"] > EPS
        )

    return dict(
        family_count=len(fams),
        context_count=len(rows),
        r882_raw_context_mean=float(med_ctx[:, 0].mean()),
        r882_equal_family_mean=float(med[0].mean()),
        r882_elo=float(elo_med[0]),
        r882_minus2_invalid_counts=invalid_counts(arr[:, :, 0]),
        arms=contrasts,
    )


def verdict(train_summary, val_summary, selections):
    if any(train_summary["arms"][a]["advance"] for a in COS_ARMS):
        return "ADVANCE"
    return "NO_ADVANCE"
