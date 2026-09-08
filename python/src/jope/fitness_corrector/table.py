"""Modeling table: pairs + genome features + dataset features, leak-guarded.

The table carries BASE features (candidate-varying + dataset shape + shape
interactions) plus raw dataset embeddings per row. PCA over embeddings is
fold-dependent (fit on train-fold datasets only) and is appended by
protocol.fold_features — never here — so unseen-dataset evaluation stays
honest. Honest-side quantities never become features (prefix guard).
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from .genome_features import structural_features

LEAKY_PREFIXES = ("honest", "oof", "fold_r2", "robust", "cv_", "label_",
                  "actual", "mean_r2", "test_")

NEWMATH_ORIGINS = ("known_good", "gp_seed", "league", "gp", "other")
NEWMATH_MUTATIONS = ("anchor_replay_baseline", "seed_concept", "insert_stage",
                     "parameter_jitter", "typed_crossover", "subprogram_union",
                     "adf_insertion", "expr_mutation", "other")

EMB_KINDS = ("hyperion", "tabpfn_v3")
PCA_K = {"hyperion": 16, "tabpfn_v3": 8}


def assert_no_leaky_features(names: list[str]) -> None:
    bad = [n for n in names
           if any(n.lower().startswith(p) for p in LEAKY_PREFIXES)]
    if bad:
        raise ValueError(f"leaky feature names: {bad}")


def _finite(x, default=None):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else default


# ---------------------------------------------------------------------------
# dataset-side inputs
# ---------------------------------------------------------------------------

def load_dataset_embeddings(snapshot_db: Path) -> dict[str, dict[str, np.ndarray]]:
    db = sqlite3.connect(f"file:{snapshot_db}?mode=ro", uri=True)
    out: dict[str, dict[str, np.ndarray]] = {}
    for ds, hyp, tab in db.execute(
            "SELECT dataset, hyperion, tabpfn_v3 FROM dataset_bank"):
        rec: dict[str, np.ndarray] = {}
        if hyp is not None:
            rec["hyperion"] = np.frombuffer(hyp, dtype=np.float32).astype(np.float64)
        if tab is not None:
            rec["tabpfn_v3"] = np.frombuffer(tab, dtype=np.float32).astype(np.float64)
        out[ds] = rec
    db.close()
    return out


def dataset_shapes_from_reports(cache_dir: Path) -> dict[str, dict]:
    """rows/features per dataset from featmath reports (newmath reports carry
    neither; datasets overlap across engines so coverage is near-complete)."""
    out: dict[str, dict] = {}
    for p in sorted((cache_dir / "reports_freeze").glob("*.report.json")):
        try:
            with p.open() as fh:
                head = fh.read(2048)
            if '"jain_regressor_gen1000"' not in head:
                continue
            with p.open() as fh:
                rep = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        ds = rep.get("dataset_hash")
        if not ds or ds in out:
            continue
        rows = rep.get("rows")
        if isinstance(rows, dict):
            rows = rows.get("train")
        out[ds] = {"rows": _finite(rows), "features": _finite(rep.get("features"))}
    return out


# ---------------------------------------------------------------------------
# PCA helpers (called by protocol per fold, and by the final artifact fit)
# ---------------------------------------------------------------------------

def fit_embedding_pca(embeds: dict[str, dict[str, np.ndarray]],
                      datasets: list[str], kind: str, k: int) -> dict | None:
    mats = [embeds[d][kind] for d in sorted(set(datasets))
            if d in embeds and kind in embeds[d]]
    if len(mats) < max(8, k + 2):
        return None
    x = np.stack(mats)
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    comps = vt[:k].copy()
    for i in range(comps.shape[0]):
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    return {"mean": mean, "components": comps}


def project_embedding(pca: dict | None, vec: np.ndarray | None, k: int) -> np.ndarray:
    if pca is None or vec is None or vec.shape[0] != pca["mean"].shape[0]:
        return np.zeros(k)
    return (pca["components"] @ (vec - pca["mean"]))[:k]


# ---------------------------------------------------------------------------
# per-engine candidate-varying features
# ---------------------------------------------------------------------------

def _clip_r2(v: float | None) -> float | None:
    """Proxy scores are fold R²s: values outside [-1, 1] are degenerate
    telemetry (observed up to 627 in reports) — clip per house convention."""
    if v is None:
        return None
    return max(-1.0, min(1.0, v))


def featmath_features(row: dict) -> dict[str, float] | None:
    s = structural_features(row.get("genome") or "")
    if s is None:
        return None
    fold = [_clip_r2(v) for v in (row.get("search_fold_scores") or [])
            if isinstance(v, (int, float)) and math.isfinite(v)]
    r0 = _clip_r2(_finite(row.get("search_score_rung0")))
    r1 = _clip_r2(_finite(row.get("search_score_rung1")))
    r2 = _clip_r2(_finite(row.get("search_score_rung2")))
    ss = _clip_r2(_finite(row.get("search_score"), 0.0))
    f: dict[str, float] = {
        "proxy_score": ss,
        "rung0": r0 if r0 is not None else 0.0,
        "rung0_missing": 0.0 if r0 is not None else 1.0,
        "rung1": r1 if r1 is not None else 0.0,
        "rung1_missing": 0.0 if r1 is not None else 1.0,
        "rung2": r2 if r2 is not None else 0.0,
        "rung2_missing": 0.0 if r2 is not None else 1.0,
        "rung_delta_10": (r1 - r0) if (r0 is not None and r1 is not None) else 0.0,
        "rung_delta_21": (r2 - r1) if (r1 is not None and r2 is not None) else 0.0,
        "n_rungs": float(sum(v is not None for v in (r0, r1, r2))),
        "n_fold_scores": float(len(fold)),
        "fold_std": float(np.std(fold)) if len(fold) >= 2 else 0.0,
        "fold_min": float(min(fold)) if fold else 0.0,
        "fold_range": float(max(fold) - min(fold)) if len(fold) >= 2 else 0.0,
        "dispersion": float(np.std(fold)) if len(fold) >= 2 else 0.0,
    }
    f.update(s)
    return f


def newmath_features(row: dict) -> dict[str, float] | None:
    s = structural_features(row.get("genome") or "")
    if s is None:
        return None
    max_gen = _finite(row.get("task_max_generation"), 0.0) or 0.0
    first_gen = _finite(row.get("first_generation"), 0.0) or 0.0
    f: dict[str, float] = {
        "proxy_score": _clip_r2(_finite(row.get("predicted_score"), 0.0)),
        "acq_score": _clip_r2(_finite(row.get("acquisition_score"), 0.0)),
        "acq_max": _clip_r2(_finite(row.get("acquisition_score_max"), 0.0)),
        "pred_percentile": _finite(row.get("within_task_pred_percentile"), 0.5),
        "complexity_field": _finite(row.get("complexity"), 0.0),
        "first_gen_frac": (first_gen / max_gen) if max_gen > 0 else 0.0,
        "log1p_occurrences": math.log1p(_finite(row.get("occurrences"), 0.0) or 0.0),
        "pred_range": _finite(row.get("predicted_score_range"), 0.0),
        "dispersion": _finite(row.get("predicted_score_range"), 0.0),
    }
    origin = row.get("origin") or "other"
    if origin not in NEWMATH_ORIGINS:
        origin = "other"
    for o in NEWMATH_ORIGINS:
        f[f"origin_{o}"] = 1.0 if origin == o else 0.0
    mut = row.get("mutation") or "other"
    if mut not in NEWMATH_MUTATIONS:
        mut = "other"
    for m in NEWMATH_MUTATIONS:
        f[f"mut_{m}"] = 1.0 if mut == m else 0.0
    f.update(s)
    return f


# ---------------------------------------------------------------------------
# table assembly
# ---------------------------------------------------------------------------

def within_group_z(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    z = np.zeros_like(values)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        v = values[idx]
        std = v.std()
        z[idx] = (v - v.mean()) / std if std > 1e-12 else 0.0
    return z


def within_group_midrank_pct(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pct = np.zeros_like(values)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        v = values[idx]
        if len(v) < 2:
            pct[idx] = 0.5
            continue
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):
            m = v == val
            ranks[m] = ranks[m].mean()
        pct[idx] = ranks / (len(v) - 1)
    return pct


# columns that get within-task z + mid-rank companions (range-restriction fix)
WTZ_COLS = {
    "featmath": ("proxy_score", "rung0", "rung1", "rung2", "fold_std"),
    "newmath": ("proxy_score", "acq_score", "complexity_field",
                "first_gen_frac"),
}


def build_modeling_table(pairs_path: Path, engine: str,
                         embeds: dict[str, dict[str, np.ndarray]],
                         shapes: dict[str, dict]) -> dict:
    builder = featmath_features if engine == "featmath" else newmath_features
    honest_key = "honest_oof_r2" if engine == "featmath" else "honest_robust_score"
    rows = []
    n_unparseable = 0
    with pairs_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("engine") != engine:
                continue
            y = _finite(r.get(honest_key))
            if y is None:
                continue
            f = builder(r)
            if f is None:
                n_unparseable += 1
                continue
            rows.append((r, f, y))
    if not rows:
        raise RuntimeError(f"no rows for engine {engine}")

    cand_names = sorted(rows[0][1].keys())
    ds_names = ["log_rows", "log_cols"]
    inter_cand = ["proxy_score", "dispersion", "n_stages"]
    inter_names = [f"ix_{c}__{d}" for c in inter_cand for d in ds_names]
    base_names = cand_names + ds_names + inter_names
    assert_no_leaky_features(base_names)

    n = len(rows)
    x = np.zeros((n, len(base_names)))
    y_abs = np.zeros(n)
    emb_h = np.zeros((n, 1536))
    emb_t = np.zeros((n, 512))
    emb_h_ok = np.zeros(n, dtype=bool)
    emb_t_ok = np.zeros(n, dtype=bool)
    groups: list[str] = []
    task_keys: list[str] = []
    started: list[float] = []
    metas: list[dict] = []
    for i, (r, f, y) in enumerate(rows):
        ds = r["dataset"]
        shape = shapes.get(ds) or {}
        vals = dict(f)
        vals["log_rows"] = math.log1p(shape.get("rows") or 0.0)
        vals["log_cols"] = math.log1p(shape.get("features") or 0.0)
        for c in inter_cand:
            for d in ds_names:
                vals[f"ix_{c}__{d}"] = vals[c] * vals[d]
        x[i] = [vals[nm] for nm in base_names]
        y_abs[i] = y
        emb = embeds.get(ds) or {}
        if "hyperion" in emb and emb["hyperion"].shape[0] == 1536:
            emb_h[i] = emb["hyperion"]
            emb_h_ok[i] = True
        if "tabpfn_v3" in emb and emb["tabpfn_v3"].shape[0] == 512:
            emb_t[i] = emb["tabpfn_v3"]
            emb_t_ok[i] = True
        groups.append(ds)
        task_keys.append(r["task_key"])
        started.append(_finite(r.get("started_at_unix_ms"), 0.0) or 0.0)
        metas.append({"dataset": ds, "task_key": r["task_key"],
                      "genome": r["genome"], "honest": y,
                      "proxy_score": vals["proxy_score"]})
    task_arr = np.array(task_keys)
    # within-task z + mid-rank companions for the key proxy columns
    wtz_names: list[str] = []
    wtz_cols: list[np.ndarray] = []
    for col in WTZ_COLS[engine]:
        if col not in base_names:
            continue
        v = x[:, base_names.index(col)]
        wtz_cols.append(within_group_z(v, task_arr))
        wtz_names.append(f"wtz_{col}")
        wtz_cols.append(within_group_midrank_pct(v, task_arr))
        wtz_names.append(f"wtr_{col}")
    if wtz_cols:
        x = np.hstack([x, np.column_stack(wtz_cols)])
        base_names = base_names + wtz_names
    # house convention: honest scores clip at -1 (active_v3.robust_score);
    # unclipped catastrophic negatives destroy absolute-head fits and regret.
    y_abs = np.maximum(y_abs, -1.0)
    return {
        "X": x, "y_abs": y_abs,
        "y_z": within_group_z(y_abs, task_arr),
        "groups": np.array(groups), "task_keys": task_arr,
        "started_ms": np.array(started),
        "emb_h": emb_h, "emb_t": emb_t,
        "emb_h_ok": emb_h_ok, "emb_t_ok": emb_t_ok,
        "feature_names": base_names, "meta": metas,
        "n_unparseable": n_unparseable, "engine": engine,
    }


def table_content_hash(tbl: dict) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tbl["X"]).tobytes())
    h.update(np.ascontiguousarray(tbl["y_abs"]).tobytes())
    h.update("\n".join(tbl["feature_names"]).encode())
    h.update(",".join(tbl["groups"].tolist()).encode())
    return h.hexdigest()
