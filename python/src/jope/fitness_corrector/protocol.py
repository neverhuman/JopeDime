"""Fair-evaluation protocol: folds, fold-dependent features, metrics, baselines.

Group = dataset_hash (never task: datasets repeat across tasks). PCA features
are fit per fold on train-fold datasets only (table carries raw embeddings).
All metrics are tie-aware; ties break by original candidate order, mirroring
production's stable sorts.
"""

from __future__ import annotations

import numpy as np

from .table import fit_embedding_pca, project_embedding

PCA_K_H = 16
PCA_K_T = 8


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def dataset_group_folds(datasets: np.ndarray, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic sorted-modulo group folds (selector_calibration precedent).
    Returns [(train_idx, test_idx)] over row indices."""
    uniq = sorted(set(datasets.tolist()))
    fold_of = {d: i % folds for i, d in enumerate(uniq)}
    assignments = np.array([fold_of[d] for d in datasets])
    out = []
    for f in range(folds):
        test = np.where(assignments == f)[0]
        train = np.where(assignments != f)[0]
        if len(test) and len(train):
            out.append((train, test))
    return out


def temporal_split(
    started_ms: np.ndarray, datasets: np.ndarray, quantile: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Train on tasks started before the quantile timestamp; drop test rows
    whose dataset also appears in train (keeps the split group-clean)."""
    valid = started_ms[started_ms > 0]
    if not len(valid):
        return np.arange(0), np.arange(0)
    cut = np.quantile(valid, quantile)
    train = np.where(started_ms <= cut)[0]
    test = np.where(started_ms > cut)[0]
    train_ds = set(datasets[train].tolist())
    test = np.array([i for i in test if datasets[i] not in train_ds], dtype=int)
    return train, test


# ---------------------------------------------------------------------------
# fold-dependent feature augmentation (PCA + interactions)
# ---------------------------------------------------------------------------


def fold_features(
    tbl: dict,
    train_idx: np.ndarray,
    embeds_all: dict | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Return (X_aug, names, pca_bundle) where PCA is fit on train-fold
    datasets only and applied to every row."""
    x = tbl["X"]
    names = list(tbl["feature_names"])
    train_datasets = sorted(set(tbl["groups"][train_idx].tolist()))
    ds_emb: dict[str, dict[str, np.ndarray]] = {}
    for i in np.where(tbl["emb_h_ok"])[0]:
        ds_emb.setdefault(tbl["groups"][i], {})["hyperion"] = tbl["emb_h"][i]
    for i in np.where(tbl["emb_t_ok"])[0]:
        ds_emb.setdefault(tbl["groups"][i], {})["tabpfn_v3"] = tbl["emb_t"][i]
    pca_h = fit_embedding_pca(ds_emb, train_datasets, "hyperion", PCA_K_H)
    pca_t = fit_embedding_pca(ds_emb, train_datasets, "tabpfn_v3", PCA_K_T)
    n = x.shape[0]
    pc_h = np.zeros((n, PCA_K_H))
    pc_t = np.zeros((n, PCA_K_T))
    for i in range(n):
        pc_h[i] = project_embedding(pca_h, tbl["emb_h"][i] if tbl["emb_h_ok"][i] else None, PCA_K_H)
        pc_t[i] = project_embedding(pca_t, tbl["emb_t"][i] if tbl["emb_t_ok"][i] else None, PCA_K_T)
    pc_names = [f"hyp_pc{j}" for j in range(PCA_K_H)] + [f"tab_pc{j}" for j in range(PCA_K_T)]
    # interactions of top-2 hyperion PCs with the key candidate-varying features
    ix_cols = []
    ix_names = []
    cand = {"proxy_score", "dispersion", "n_stages"}
    for c in sorted(cand):
        if c not in names:
            continue
        cx = x[:, names.index(c)]
        for j in (0, 1):
            ix_cols.append(cx * pc_h[:, j])
            ix_names.append(f"ix_{c}__hyp_pc{j}")
    x_aug = np.hstack([x, pc_h, pc_t] + ([np.column_stack(ix_cols)] if ix_cols else []))
    return x_aug, names + pc_names + ix_names, {"hyperion": pca_h, "tabpfn_v3": pca_t}


# ---------------------------------------------------------------------------
# metrics (tie-aware, per task, aggregated)
# ---------------------------------------------------------------------------


def _mid_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v), dtype=float)
    for val in np.unique(v):
        m = v == val
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _mid_ranks(np.asarray(a, float)), _mid_ranks(np.asarray(b, float))
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def per_task_metrics(pred: np.ndarray, honest: np.ndarray, task_keys: np.ndarray, min_group: int = 5) -> dict:
    """Within-task ranking metrics. Ties in pred break by original order
    (stable argsort on -pred), mirroring production stable sorts."""
    rows = []
    for t in np.unique(task_keys):
        idx = np.where(task_keys == t)[0]
        if len(idx) < min_group:
            continue
        p, h = pred[idx], honest[idx]
        sp = spearman(p, h)
        order = np.argsort(-p, kind="stable")
        best = float(np.max(h))
        top1 = float(h[order[0]])
        k3 = order[:3]
        rows.append(
            {
                "task": str(t),
                "n": int(len(idx)),
                "spearman": sp,
                "capture1": 1.0 if top1 >= best - 1e-12 else 0.0,
                "capture3": 1.0 if float(np.max(h[k3])) >= best - 1e-12 else 0.0,
                "regret1": best - top1,
                "ndcg3": _ndcg_at_k(p, h, 3),
            }
        )
    if not rows:
        return {"n_tasks": 0}
    agg = {"n_tasks": len(rows)}
    for key in ("spearman", "capture1", "capture3", "regret1", "ndcg3"):
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_median"] = float(np.median(vals))
    agg["_per_task"] = rows
    return agg


def _ndcg_at_k(pred: np.ndarray, honest: np.ndarray, k: int) -> float:
    order = np.argsort(-pred, kind="stable")[:k]
    ideal = np.sort(honest)[::-1][:k]
    rel = honest[order]
    # shift to non-negative gains within the group
    lo = honest.min()
    gains = rel - lo
    ideal_gains = ideal - lo
    denom = np.log2(np.arange(2, len(order) + 2))
    dcg = float((gains / denom).sum())
    idcg = float((ideal_gains / denom[: len(ideal_gains)]).sum())
    return dcg / idcg if idcg > 1e-12 else 1.0


def pooled_metrics(pred: np.ndarray, honest: np.ndarray) -> dict:
    return {"spearman": spearman(pred, honest), "pearson": pearson(pred, honest), "n": int(len(pred))}


def paired_bootstrap_delta(
    per_task_a: list[dict], per_task_b: list[dict], key: str, n_boot: int = 2000, seed: int = 20260714
) -> dict:
    """Bootstrap-by-task 95% CI for mean(metric_a - metric_b)."""
    b_by_task = {r["task"]: r for r in per_task_b}
    deltas = np.array(
        [
            r[key] - b_by_task[r["task"]][key]
            for r in per_task_a
            if r["task"] in b_by_task and r[key] is not None and b_by_task[r["task"]][key] is not None
        ]
    )
    if not len(deltas):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n_boot)])
    return {
        "n": int(len(deltas)),
        "delta_mean": float(deltas.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def _col(tbl: dict, name: str) -> np.ndarray:
    return tbl["X"][:, tbl["feature_names"].index(name)]


def shaped_proxy_grid(tbl: dict, train_idx: np.ndarray) -> tuple[float, float]:
    """Pick (lambda, gamma) for proxy_mean − λ·dispersion − γ·n_stages on the
    TRAIN fold only (mean within-task spearman objective)."""
    best = (0.0, 0.0)
    best_val = -2.0
    honest = tbl["y_abs"]
    tasks = tbl["task_keys"]
    proxy = _col(tbl, "proxy_score")
    disp = _col(tbl, "dispersion")
    stages = _col(tbl, "n_stages")
    for lam in (0.0, 0.1, 0.25, 0.5):
        for gam in (0.0, 0.001, 0.01, 0.03):
            score = proxy - lam * disp - gam * stages
            m = per_task_metrics(score[train_idx], honest[train_idx], tasks[train_idx])
            v = m.get("spearman_mean")
            if v is not None and v > best_val:
                best_val = v
                best = (lam, gam)
    return best


def baseline_scores(tbl: dict, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Baseline score vectors over ALL rows; any tuning uses train_idx only."""
    n = tbl["X"].shape[0]
    out: dict[str, np.ndarray] = {}
    proxy = _col(tbl, "proxy_score")
    out["raw_proxy"] = proxy.copy()
    if tbl["engine"] == "featmath":
        r2 = _col(tbl, "rung2")
        r2m = _col(tbl, "rung2_missing")
        r1 = _col(tbl, "rung1")
        r1m = _col(tbl, "rung1_missing")
        r0 = _col(tbl, "rung0")
        rf = np.where(r2m < 0.5, r2, np.where(r1m < 0.5, r1, r0))
        out["rung_fallback"] = rf
        lam, gam = shaped_proxy_grid(tbl, train_idx)
        out["shaped_proxy"] = proxy - lam * _col(tbl, "dispersion") - gam * _col(tbl, "n_stages")
        out["_shaped_params"] = np.array([lam, gam])
    else:
        out["production_order"] = np.zeros(n)  # candidate_priority ≡ 0 → ties
        out["acquisition"] = _col(tbl, "acq_score")
        out["pred_percentile"] = _col(tbl, "pred_percentile")
    return out


# ---------------------------------------------------------------------------
# budget model
# ---------------------------------------------------------------------------


def budget_capture_curve(
    pred: np.ndarray, honest: np.ndarray, task_keys: np.ndarray, target: float = 0.95
) -> dict:
    """Smallest m with capture@m >= target over test tasks + savings estimate."""
    curves = {}
    for m in range(1, 11):
        caps = []
        for t in np.unique(task_keys):
            idx = np.where(task_keys == t)[0]
            if len(idx) < 5:
                continue
            order = np.argsort(-pred[idx], kind="stable")[:m]
            caps.append(1.0 if honest[idx][order].max() >= honest[idx].max() - 1e-12 else 0.0)
        if caps:
            curves[m] = float(np.mean(caps))
    m_star = next((m for m in sorted(curves) if curves[m] >= target), None)
    return {
        "capture_curve": curves,
        "m_at_95": m_star,
        "cv_fraction_saved": (10 - m_star) / 10.0 if m_star else 0.0,
    }
