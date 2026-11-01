"""Pure NumPy statistical, cross-fit selection, capacity-sweep, and Elo helpers for H156."""

import hashlib
import math
import numpy as np

FOLD_SEED = 156140
SEEDS = (156101, 156102, 156103)
BOOT_SEED = 156141
STEPS = (0, 10, 20, 30, 40)
WIDTH_ARMS = ("W05", "W10", "W20")
WIDTH_MULTS = {"W05": 0.5, "W10": 1.0, "W20": 2.0}
CANDIDATE_ARMS = ("W05_XFIT", "W10_XFIT", "W20_XFIT", "XFIT_JOINT")
ALL_ARMS = ("r882", "W05_XFIT", "W10_XFIT", "W20_XFIT", "XFIT_JOINT")
THRESHOLD = 0.001
EPS = 1e-12


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


def choose(rows, panel, seed, r882_col=None):
    mat = np.asarray(panel, dtype=np.float64)
    if mat.shape != (len(rows), len(STEPS)) or not np.isfinite(mat).all():
        raise ValueError("selector panel shape/finiteness mismatch")
    _, fam = family_means(rows, mat)
    if r882_col is None:
        base = fam[:, 0]
    else:
        _, base = family_means(rows, np.asarray(r882_col, dtype=np.float64))
    best_xfit = 0
    best_xfit_gain = 0.0
    best_lcb = 0
    best_lcb_gain = 0.0
    candidates = {}
    for j, step in enumerate(STEPS):
        if step == 0 and r882_col is None:
            candidates["0"] = dict(
                gain=0.0,
                ci_low_95=0.0,
                ci_high_95=0.0,
                eligible_xfit=False,
                eligible_lcb=False,
            )
            continue
        stat = bootstrap(fam[:, j], base, 1000, seed + step)
        ok_xfit = stat["delta"] > EPS
        ok_lcb = stat["delta"] > THRESHOLD + EPS and stat["ci_low_95"] > 0.0
        candidates[str(step)] = dict(
            **stat,
            eligible_xfit=bool(ok_xfit),
            eligible_lcb=bool(ok_lcb),
        )
        if ok_xfit and stat["delta"] > best_xfit_gain + EPS:
            best_xfit, best_xfit_gain = step, stat["delta"]
        if ok_lcb and stat["delta"] > best_lcb_gain + EPS:
            best_lcb, best_lcb_gain = step, stat["delta"]
    return dict(
        selected_step=best_xfit,
        selected_gain=best_xfit_gain,
        lcb_selected_step=best_lcb,
        lcb_selected_gain=best_lcb_gain,
        candidates=candidates,
    )


def choose_joint(rows, arm_panels, r882_col, seed):
    _, base = family_means(rows, np.asarray(r882_col, dtype=np.float64))
    best_arm, best_step, best_gain = "W10", 0, 0.0
    lcb_arm, lcb_step, lcb_gain = "W10", 0, 0.0
    grid = {}
    for ai, arm in enumerate(WIDTH_ARMS):
        mat = np.asarray(arm_panels[arm], dtype=np.float64)
        if mat.shape != (len(rows), len(STEPS)) or not np.isfinite(mat).all():
            raise ValueError(f"invalid panel for {arm}")
        _, fam = family_means(rows, mat)
        grid[arm] = {}
        for j, step in enumerate(STEPS):
            if arm == "W10" and step == 0:
                grid[arm]["0"] = dict(
                    delta=0.0,
                    ci_low_95=0.0,
                    ci_high_95=0.0,
                    p_one_sided=1.0,
                    draws=1000,
                    eligible_xfit=False,
                    eligible_lcb=False,
                )
                continue
            st = bootstrap(fam[:, j], base, 1000, seed + 1000 * ai + step)
            ok_xfit = st["delta"] > EPS
            ok_lcb = st["delta"] > THRESHOLD + EPS and st["ci_low_95"] > 0.0
            grid[arm][str(step)] = dict(
                **st,
                eligible_xfit=bool(ok_xfit),
                eligible_lcb=bool(ok_lcb),
            )
            if ok_xfit and st["delta"] > best_gain + EPS:
                best_arm, best_step, best_gain = arm, step, st["delta"]
            if ok_lcb and st["delta"] > lcb_gain + EPS:
                lcb_arm, lcb_step, lcb_gain = arm, step, st["delta"]
    return dict(
        selected_arm=best_arm,
        selected_step=best_step,
        selected_gain=best_gain,
        lcb_selected_arm=lcb_arm,
        lcb_selected_step=lcb_step,
        lcb_selected_gain=lcb_gain,
        grid=grid,
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


def summarize(rows, panel, arm_names=ALL_ARMS, seed=BOOT_SEED):
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

    # 1,000-draw family bootstrap for Elo CIs
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

    contrasts = {}
    raw_p = {}
    for idx_m, name in enumerate(arm_names[1:], start=1):
        st = bootstrap(med[idx_m], med[0], 10000, seed + idx_m)
        per_seed = [
            float((fam[s, idx_m] - fam[s, 0]).mean()) for s in range(len(SEEDS))
        ]
        contrasts[name] = dict(
            **st,
            per_seed_equal_family_delta=per_seed,
            positive_seeds=int(sum(x > EPS for x in per_seed)),
            raw_context_mean=float(med_ctx[:, idx_m].mean()),
            equal_family_mean=float(med[idx_m].mean()),
            elo=float(elo_med[idx_m]),
            elo_delta_vs_r882=float(elo_med[idx_m] - elo_med[0]),
            elo_ci_low_95=float(elo_lo[idx_m]),
            elo_ci_high_95=float(elo_hi[idx_m]),
        )
        raw_p[name] = st["p_one_sided"]
    adj_p = holm(raw_p)
    for name, p_adj in adj_p.items():
        contrasts[name]["p_holm"] = float(p_adj)
        contrasts[name]["advance"] = bool(
            contrasts[name]["delta"] > EPS
            and contrasts[name]["ci_low_95"] > 0.0
            and contrasts[name]["positive_seeds"] >= 2
            and p_adj < 0.05
        )

    return dict(
        family_count=len(fams),
        context_count=len(rows),
        r882_raw_context_mean=float(med_ctx[:, 0].mean()),
        r882_equal_family_mean=float(med[0].mean()),
        r882_elo=float(elo_med[0]),
        arms=contrasts,
    )


def verdict(train_summary, val_summary, selections):
    if all(
        d["XFIT_JOINT"]["selected_step"] == 0
        and all(d[w]["selected_step"] == 0 for w in WIDTH_ARMS)
        for d in selections
    ):
        return "RULE_NEUTRAL"
    if any(c["advance"] for c in train_summary["arms"].values()) and any(
        c["delta"] > EPS and c["ci_low_95"] > 0.0 for c in val_summary["arms"].values()
    ):
        return "ADVANCE"
    return "NULL"
