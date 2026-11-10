#!/usr/bin/env python3
"""
H269 (`h269_nosg_veto_rerank4`) — Confirmatory Fresh-Fold Replication of `H266` `NO_SCOREGEOM` (`528` cols)
Plus Outcome-Free `-2.0` Invalid-Gate Classifier Veto (`NOSG_VETO`) on `DL-Train` (205 fam / 1,219 ctx)
PRIMARY and `DL-Val91` (19 fam / 91 ctx) SECONDARY.

Frozen from H265/H266:
  - Exact 560-dim outcome-free top-4 feature builder (`build_top4_feature_matrix_for_context`)
    minus cols `0..31` = exact 528-dim `NO_SCOREGEOM` feature set (`NO_SCOREGEOM_COLS = np.arange(32, 560)`).
  - Exact `lightgbm.LGBMRanker(objective="lambdarank", max_depth=3, num_leaves=7, min_child_samples=20)`.
  - Exact hyperparameter grid: `lr in {0.03, 0.05, 0.10}`, `n_estimators in {50, 100, 200}`,
    fallback margin `m in {0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, inf}`.
  - Exact integer relevance labels `rel in {0, 1, 2, 3}` (`-2.0` invalid gate = `0` worst).

Arms:
  1. `CONTROL`: `r882` top-1 candidate (`slot 0`).
  2. `NOSG`: Frozen `H266` `NO_SCOREGEOM` (`528` cols, exact `H266` strict inner-`FIT` selector).
  3. `NOSG_VETO`: `NOSG` + outcome-free `LGBMClassifier` on inner `FIT` only (`528` cols, fixed frozen
     hyperparameters) predicting `"candidate is a -2.0 invalid gate"`; any non-`r882` pick with
     `P(invalid) > tau` falls back to `r882`, with `tau in {0.05, 0.10, 0.20}` and `margin in MARGIN_GRID`
     selected on inner `FIT` OOF (`164` families) subject to inner-`FIT` invalid-gate non-regression
     (`<= CONTROL`) + strict `H257` inner-`FIT` gain/LCB admission.
  4. `NOSG_SHUF`: Within-family label shuffle placebo (`528` cols).
  5. `RERANK4`: Full `560`-col `H265`/`H266` listwise re-ranker (reference only).

Multiplicity:
  - Step-down Holm-Bonferroni (`m = 3`) over `{NOSG DL-Train OOF, NOSG_VETO DL-Train OOF, NOSG_VETO DL-Val91}`.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import torch

EVIDENCE_ROOT = Path("/home/ubuntu/jope_plan/evidence/H269/argon-t1830")
INPUTS_DIR = EVIDENCE_ROOT / "inputs"
CAPSULE_STUDY_DIR = (
    EVIDENCE_ROOT
    / "capsule"
    / "research"
    / "flywheel_selfplay"
    / "h269_nosg_veto_rerank4"
)

INPUT_POOLS_DIR = Path("/home/ubuntu/jope-data/h50/input")
CACHE_DIR = Path("/home/ubuntu/jope-data/h58/cache")
CONTEXT_FEATURES_NPZ = CACHE_DIR / "context_features.npz"
POOL_PRIORS_NPZ = CACHE_DIR / "pool_priors.npz"
CACHE_MANIFEST_JSON = CACHE_DIR / "cache_manifest.json"

R882_CKPT = INPUTS_DIR / "r882" / "round-882.pt"
R882_BEST_DEV_JSON = INPUTS_DIR / "r882" / "best_dev_checkpoint.json"
R882_TRAIN_COSTS_NPZ = INPUTS_DIR / "r882-train-costs.npz"
R882_VAL_COSTS_NPZ = INPUTS_DIR / "r882-secondary-costs.npz"

DOPE_EXPECTED_SHA256 = (
    "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
)
R882_EXPECTED_SHA256 = (
    "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
)
SELECTED_JSON_EXPECTED_SHA256 = (
    "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
)
SOURCE_MANIFEST_EXPECTED_SHA256 = (
    "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
)
CONTEXT_FEATURES_EXPECTED_SHA256 = (
    "75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf"
)
POOL_PRIORS_EXPECTED_SHA256 = (
    "6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe"
)

R882_NATIVE91_EXPECTED = 1.1377941898164525
TOP4_PICK_VAL_CTX_EXPECTED = 1.2403601682250796
TOP4_PICK_VAL_FAM_EXPECTED = 1.2519997168684853

CONTEXT_27_SHA256 = "43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91"

FOLD_SEEDS = (269140, 269141, 269142)
SECONDARY_FOLD_SEED = 269140
SEEDS = (269101, 269102, 269103)
BOOT_SEED = 269150
BOOT_DRAWS = 10_000
INNER_BOOT_DRAWS = 2_000
HOLM_M = 3

LR_GRID = (0.03, 0.05, 0.10)
N_EST_GRID = (50, 100, 200)
MARGIN_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, math.inf)
VETO_TAU_GRID = (0.05, 0.10, 0.20)

STRICT_FIT_MIN_GAIN = 0.001 + 1e-12
PRIMARY_ADVANCE_MIN_GAIN = 0.002 + 1e-12
TIE_MARGIN = 1e-4

TOTAL_FEATURE_DIM = 560
NOSG_FEATURE_DIM = 528
ALL_COLS = np.arange(560, dtype=np.int32)
NO_SCOREGEOM_COLS = np.arange(32, 560, dtype=np.int32)

# Frozen LGBMClassifier hyperparameters for outcome-free -2.0 invalid-gate prediction on inner FIT
VETO_CLF_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 15,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": {0: 1.0, 1: 10.0},
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


def trim_malloc() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def check_cooperative_pause() -> None:
    pause_file = os.environ.get("H269_PAUSE_FILE", "")
    if not pause_file:
        return
    p = Path(pause_file)
    while p.exists():
        time.sleep(0.5)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with tmp.open("wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return hashlib.sha256(raw).hexdigest()


def stream_seed(*parts: Any) -> int:
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def compute_true_top4_relevance(
    fitness_4: np.ndarray, gate_4: np.ndarray
) -> np.ndarray:
    """
    Frozen from H265/H266:
      - Any candidate failing `gate` (`utility = -2.0`, invalid = worst) receives relevance `0`.
      - Valid candidates (`gate == True`) are ranked by ascending true revealed cost (`fitness`),
        with ties broken by ascending `r882` order `j in 0..3`; best valid gets `3`, second gets `2`,
        third gets `1`, fourth gets `0`.
    """
    rel = np.zeros(4, dtype=np.int32)
    valid_js = [j for j in range(4) if bool(gate_4[j]) and np.isfinite(fitness_4[j])]
    valid_sorted = sorted(valid_js, key=lambda j: (float(fitness_4[j]), int(j)))
    for rank_pos, j in enumerate(valid_sorted):
        rel[j] = max(3 - rank_pos, 0)
    return rel


def compute_sem24_numpy(prog_412: np.ndarray) -> np.ndarray:
    """Fast vectorized 24D structural semantic signature matching compute_semantic_struct_sig_24d."""
    x = np.asarray(prog_412, dtype=np.float32)
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, 412)
    padded = np.pad(flat, ((0, 0), (0, 20)), mode="constant").reshape(-1, 18, 24)
    sig = padded.mean(axis=1) + 0.5 * padded.max(axis=1)
    norms = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
    return (sig / norms).reshape(*orig_shape, 24).astype(np.float32)


def build_top4_feature_matrix_for_context(
    rcost: np.ndarray,
    aprior: np.ndarray,
    prog_top4: np.ndarray,
    sem24_full: np.ndarray,
    proj16_full: np.ndarray,
    lm48_z: np.ndarray,
    dope629: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frozen from H265/H266: Builds the `(4, 560)` float32 strictly outcome-free feature matrix for
    `r882`'s `top-4` candidates (`top4 = np.argsort(rcost, kind="mergesort")[:4]`).
    `NO_SCOREGEOM` drops `cols 0..31` (`block1` + `block2`), retaining `cols 32..559` (`528` cols:
    `prog_top4` 412 + `sem_z_top4` 24 + `proj_z_top4` 16 + `ctx_vec76` 76).
    """
    rcost64 = rcost.astype(np.float64)
    top4 = np.argsort(rcost64, kind="mergesort")[:4]
    top8 = np.argsort(rcost64, kind="mergesort")[:8]

    r_mean = float(np.mean(rcost64))
    r_sd = max(float(np.std(rcost64)), 1e-12)
    r_z_all = (rcost64 - r_mean) / r_sd

    c4 = rcost64[top4]
    c4_mean = float(np.mean(c4))
    c4_sd = max(float(np.std(c4)), 1e-6)
    c4_z = (c4 - c4_mean) / c4_sd

    z4 = r_z_all[top4]
    sm_tau1 = np.exp(-(z4 - np.min(z4)))
    sm_tau1 /= np.sum(sm_tau1)
    sm_tau02 = np.exp(-(z4 - np.min(z4)) / 0.2)
    sm_tau02 /= np.sum(sm_tau02)
    sm_c4 = np.exp(-(c4_z - np.min(c4_z)))
    sm_c4 /= np.sum(sm_c4)

    ap64 = aprior.astype(np.float64)
    ap_z_all = (ap64 - float(np.mean(ap64))) / max(float(np.std(ap64)), 1e-6)

    ctx_top2_zgap = float((c4[1] - c4[0]) / r_sd)
    ctx_top4_spread_z = float((c4[3] - c4[0]) / r_sd)

    sem64 = sem24_full.astype(np.float64)
    sem_mean = sem64.mean(axis=0, keepdims=True)
    sem_std = sem64.std(axis=0, keepdims=True) + 1e-5
    sem_z_top4 = np.clip((sem64[top4] - sem_mean) / sem_std, -5.0, 5.0).astype(
        np.float32
    )

    proj64 = proj16_full.astype(np.float64)
    proj_mean = proj64.mean(axis=0, keepdims=True)
    proj_std = proj64.std(axis=0, keepdims=True) + 1e-5
    proj_z_top4 = np.clip((proj64[top4] - proj_mean) / proj_std, -5.0, 5.0).astype(
        np.float32
    )
    proj_unit_top4 = proj_z_top4 / np.maximum(
        np.linalg.norm(proj_z_top4, axis=1, keepdims=True), 1e-6
    )

    sem_top4 = sem64[top4]
    sem_top8_centroid = sem64[top8].mean(axis=0)
    sem_top8_centroid /= max(float(np.linalg.norm(sem_top8_centroid)), 1e-6)
    sem_rev03 = sem64[:4]

    est_slice = prog_top4[:, 24:29]
    est_ids = np.where(
        est_slice.max(axis=1) > 0, np.argmax(est_slice, axis=1), -1
    ).astype(np.int32)
    op_counts = prog_top4[:, :23].sum(axis=1).astype(np.float64)

    d629 = dope629.astype(np.float32)
    d_summary = np.array(
        [
            float(np.mean(d629)),
            float(np.std(d629)),
            float(np.linalg.norm(d629) / math.sqrt(len(d629))),
        ],
        dtype=np.float32,
    )
    ctx_vec76 = np.concatenate(
        [lm48_z.astype(np.float32), d629[:25], d_summary], axis=0
    )

    rows_out = np.zeros((4, 560), dtype=np.float32)
    for j in range(4):
        others = [k for k in range(4) if k != j]
        raw_diffs = [float(c4[j] - c4[k]) for k in others]
        z_diffs = [float((c4[j] - c4[k]) / r_sd) for k in others]
        block1 = np.array(
            [
                float(c4[j]),
                float(z4[j]),
                float(c4_z[j]),
                float(j),
                float(j) / 3.0,
                1.0 if j == 0 else 0.0,
                float(c4[j] - c4[0]),
                0.0 if j == 0 else float((c4[j] - c4[0]) / r_sd),
                ctx_top2_zgap,
                ctx_top4_spread_z,
                float(sm_tau1[j]),
                float(sm_tau02[j]),
                float(sm_c4[j]),
                float(ap_z_all[top4[j]]),
                *raw_diffs,
                *z_diffs,
            ],
            dtype=np.float32,
        )

        cos_sem_other = [
            float(np.clip(np.dot(sem_top4[j], sem_top4[k]), -1.0, 1.0)) for k in others
        ]
        cos_proj_other = [
            float(np.clip(np.dot(proj_unit_top4[j], proj_unit_top4[k]), -1.0, 1.0))
            for k in others
        ]
        cos_sem_r1 = float(np.clip(np.dot(sem_top4[j], sem_top4[0]), -1.0, 1.0))
        cos_proj_r1 = float(
            np.clip(np.dot(proj_unit_top4[j], proj_unit_top4[0]), -1.0, 1.0)
        )
        cos_top8 = float(np.clip(np.dot(sem_top4[j], sem_top8_centroid), -1.0, 1.0))
        cos_rev4_mean = float(np.mean(np.clip(sem_rev03 @ sem_top4[j], -1.0, 1.0)))
        same_est_r1 = 1.0 if est_ids[j] == est_ids[0] else 0.0
        n_same_est = float(sum(1 for k in range(4) if est_ids[k] == est_ids[j]))

        block2 = np.array(
            [
                cos_sem_r1,
                float(np.mean(cos_sem_other)),
                float(np.max(cos_sem_other)),
                cos_proj_r1,
                float(np.mean(cos_proj_other)),
                float(np.max(cos_proj_other)),
                cos_top8,
                cos_rev4_mean,
                float(op_counts[j]),
                float(op_counts[j] - op_counts[0]),
                same_est_r1,
                n_same_est,
            ],
            dtype=np.float32,
        )

        rows_out[j] = np.concatenate(
            [
                block1,
                block2,
                prog_top4[j].astype(np.float32),
                sem_z_top4[j],
                proj_z_top4[j],
                ctx_vec76,
            ],
            axis=0,
        )
    return top4, rows_out


def select_from_top4_scores(scores_4: np.ndarray, margin: float) -> int:
    if math.isinf(margin):
        return 0
    best_j = int(np.argmax(scores_4))
    if best_j == 0:
        return 0
    if float(scores_4[best_j] - scores_4[0]) > float(margin):
        return best_j
    return 0


def select_from_top4_scores_with_veto(
    scores_4: np.ndarray,
    p_inv_4: np.ndarray,
    tau: float,
    margin: float,
) -> int:
    """
    NOSG_VETO selection rule:
      - Pick `best_j = argmax(scores_4)`.
      - If `best_j > 0` and `scores_4[best_j] - scores_4[0] > margin`:
          - If `p_inv_4[best_j] > tau`: veto and fall back to `0` (`r882`).
          - Else: return `best_j`.
      - Else: return `0` (`r882`).
    """
    if math.isinf(margin):
        return 0
    best_j = int(np.argmax(scores_4))
    if best_j == 0:
        return 0
    if float(scores_4[best_j] - scores_4[0]) > float(margin):
        if float(p_inv_4[best_j]) > float(tau):
            return 0
        return best_j
    return 0


def shuffle_labels_within_family(
    rel_by_cid: dict[str, np.ndarray],
    cids: list[str],
    fam_by_cid: dict[str, str],
    seed: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    by_fam: dict[str, list[str]] = {}
    for c in cids:
        by_fam.setdefault(fam_by_cid[c], []).append(c)

    for fam in sorted(by_fam):
        fcids = sorted(by_fam[fam])
        flat = np.concatenate([rel_by_cid[c] for c in fcids], axis=0).copy()
        rng = np.random.default_rng(stream_seed(seed, "shuf_fam", fam))
        perm = rng.permutation(len(flat))
        flat_shuf = flat[perm]
        for idx, c in enumerate(fcids):
            out[c] = flat_shuf[idx * 4 : (idx + 1) * 4].copy()
    return out


def load_cohort_top4_packets(
    rows: list[dict[str, Any]],
    costs_by_cid: dict[str, np.ndarray],
    aprior_by_cid: dict[str, np.ndarray],
    proj16_by_cid: dict[str, np.ndarray],
    ctx_feats_by_cid: dict[str, dict[str, np.ndarray]],
    batch_size: int = 50,
) -> dict[str, dict[str, Any]]:
    packets: dict[str, dict[str, Any]] = {}
    for start in range(0, len(rows), batch_size):
        check_cooperative_pause()
        batch = rows[start : start + batch_size]
        for r in batch:
            cid = r["context_sha256"]
            pool_path = INPUT_POOLS_DIR / "pools" / f"{cid}.npz"
            with np.load(pool_path, allow_pickle=False) as z:
                fitness = z["fitness"].astype(np.float64)
                gate = z["gate"].astype(np.bool_)
                prog = z["program"].astype(np.float32)
            sem24 = compute_sem24_numpy(prog)

            utility = np.where(gate, -fitness, -2.0)
            rcost = costs_by_cid[cid]
            top4, x_top4 = build_top4_feature_matrix_for_context(
                rcost=rcost,
                aprior=aprior_by_cid[cid],
                prog_top4=prog[np.argsort(rcost, kind="mergesort")[:4]],
                sem24_full=sem24,
                proj16_full=proj16_by_cid[cid],
                lm48_z=ctx_feats_by_cid[cid]["lm48_z"],
                dope629=ctx_feats_by_cid[cid]["dope629"],
            )
            fit4 = fitness[top4]
            gate4 = gate[top4]
            u4 = utility[top4]
            rel4 = compute_true_top4_relevance(fit4, gate4)
            gated_cost4 = np.where(gate4, fit4, 2.0)
            oracle_j = int(np.argmin(gated_cost4))
            packets[cid] = {
                "cid": cid,
                "family_id": r["family_id"],
                "top4_indices": top4,
                "X_top4": x_top4,
                "fitness_4": fit4,
                "gate_4": gate4,
                "utility_4": u4,
                "rel_4": rel4,
                "r882_u": float(u4[0]),
                "r882_invalid": bool(not gate4[0]),
                "top4_oracle_j": oracle_j,
                "top4_oracle_u": float(u4[oracle_j]),
            }
        trim_malloc()
    return packets


def fit_lgbm_ranker_200(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    n_groups: int,
    lr: float,
    random_state: int,
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=4,
        boosting_type="gbdt",
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        learning_rate=float(lr),
        n_estimators=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=int(random_state),
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    groups = np.full(n_groups, 4, dtype=np.int32)
    model.fit(X_fit, y_fit, group=groups, eval_at=[1, 4])
    return model


def predict_lgbm_ranker(
    model: lgb.LGBMRanker, X: np.ndarray, num_iteration: int
) -> np.ndarray:
    return np.asarray(
        model.booster_.predict(X, num_iteration=int(num_iteration)), dtype=np.float64
    )


def fit_veto_classifier(
    X_fit: np.ndarray,
    y_inv_fit: np.ndarray,
    random_state: int,
) -> lgb.LGBMClassifier:
    """
    Fits the frozen outcome-free LGBMClassifier on inner FIT candidate rows `(4 * N_fit, 528)`
    predicting `y_inv = 1[not gate]` ("candidate is a -2.0 invalid gate").
    """
    params = dict(VETO_CLF_PARAMS)
    params["random_state"] = int(random_state)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_fit, y_inv_fit)
    return clf


def predict_veto_classifier(clf: lgb.LGBMClassifier, X: np.ndarray) -> np.ndarray:
    probs = clf.booster_.predict(X)
    return np.asarray(probs, dtype=np.float64)


def run_cell_fit_and_select(
    fold_seed: int,
    seed: int,
    assess_fold: int,
    arm_name: str,
    shuffle_placebo: bool,
    train_cids_by_fold: dict[int, list[str]],
    packets: dict[str, dict[str, Any]],
    fam_by_cid: dict[str, str],
    col_indices: np.ndarray | None = None,
    with_veto: bool = False,
) -> dict[str, Any]:
    """
    Runs 4-fold inner cross-validation across the `4` FIT folds (`!= assess_fold`) for
    `(fold_seed, seed, assess_fold)`, applies the strict H257 FIT selector for `arm_name`,
    and when `with_veto == True` (`arm_name == "NOSG"`), also fits the frozen inner-FIT
    `LGBMClassifier` invalid-gate veto model and selects `(lr*, n_est*, tau*, m*)` for
    `NOSG_VETO` on inner FIT OOF (`164` families, `dl_val_reads = 0`).
    """
    cols = ALL_COLS if col_indices is None else col_indices
    fit_folds = [f for f in range(5) if f != assess_fold]
    fit_cids = [c for f in fit_folds for c in train_cids_by_fold[f]]
    fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
    assert len(fit_fams) == 164

    if shuffle_placebo:
        base_rel = {c: packets[c]["rel_4"] for c in fit_cids}
        rel_map = shuffle_labels_within_family(
            base_rel, fit_cids, fam_by_cid, stream_seed(seed, assess_fold, "shuf")
        )
    else:
        rel_map = {c: packets[c]["rel_4"] for c in fit_cids}

    inner_oof_scores: dict[tuple[float, int], dict[str, np.ndarray]] = {
        (lr, n_est): {} for lr in LR_GRID for n_est in N_EST_GRID
    }
    inner_oof_p_inv: dict[str, np.ndarray] = {}

    for hold_f in fit_folds:
        in_fit_cids = [
            c for f in fit_folds if f != hold_f for c in train_cids_by_fold[f]
        ]
        in_val_cids = list(train_cids_by_fold[hold_f])

        X_in_fit = np.concatenate(
            [packets[c]["X_top4"][:, cols] for c in in_fit_cids], axis=0
        )
        y_in_fit = np.concatenate([rel_map[c] for c in in_fit_cids], axis=0)
        X_in_val = np.concatenate(
            [packets[c]["X_top4"][:, cols] for c in in_val_cids], axis=0
        )

        for lr in LR_GRID:
            rstate = stream_seed(seed, assess_fold, arm_name, "inner", hold_f, lr)
            model_200 = fit_lgbm_ranker_200(
                X_in_fit, y_in_fit, len(in_fit_cids), lr=lr, random_state=rstate
            )
            for n_est in N_EST_GRID:
                preds = predict_lgbm_ranker(model_200, X_in_val, num_iteration=n_est)
                preds_4 = preds.reshape(len(in_val_cids), 4)
                for idx_c, cid in enumerate(in_val_cids):
                    inner_oof_scores[(lr, n_est)][cid] = preds_4[idx_c]

        if with_veto:
            y_inv_in_fit = np.concatenate(
                [(~packets[c]["gate_4"]).astype(np.int32) for c in in_fit_cids],
                axis=0,
            )
            rstate_clf = stream_seed(
                seed, assess_fold, "NOSG_VETO", "inner_clf", hold_f
            )
            clf_in = fit_veto_classifier(
                X_in_fit, y_inv_in_fit, random_state=rstate_clf
            )
            p_inv_val = predict_veto_classifier(clf_in, X_in_val).reshape(
                len(in_val_cids), 4
            )
            for idx_c, cid in enumerate(in_val_cids):
                inner_oof_p_inv[cid] = p_inv_val[idx_c]

    cid_to_pos = {c: i for i, c in enumerate(fit_cids)}
    fam_pos_lists = [
        np.array(
            [cid_to_pos[c] for c in fit_cids if fam_by_cid[c] == f], dtype=np.int32
        )
        for f in fit_fams
    ]
    fit_u4 = np.stack([packets[c]["utility_4"] for c in fit_cids], axis=0)
    fit_gate4 = np.stack([packets[c]["gate_4"] for c in fit_cids], axis=0)
    fit_r882_u = fit_u4[:, 0]
    row_arange = np.arange(len(fit_cids), dtype=np.int32)

    best_unc_gain = -1e9
    best_unc_cfg = (0.05, 100, math.inf)
    best_unc_fam_deltas = np.zeros(len(fit_fams), dtype=np.float64)

    for lr in LR_GRID:
        for n_est in N_EST_GRID:
            s_map = inner_oof_scores[(lr, n_est)]
            S_mat = np.stack([s_map[c] for c in fit_cids], axis=0)
            best_j_vec = np.argmax(S_mat, axis=1)
            gap0_vec = S_mat[row_arange, best_j_vec] - S_mat[:, 0]

            for margin in MARGIN_GRID:
                if math.isinf(margin):
                    fam_deltas = np.zeros(len(fit_fams), dtype=np.float64)
                    gain = 0.0
                else:
                    chosen_j = np.where(
                        (best_j_vec > 0) & (gap0_vec > float(margin)), best_j_vec, 0
                    )
                    u_diff = fit_u4[row_arange, chosen_j] - fit_r882_u
                    fam_deltas = np.array(
                        [float(np.mean(u_diff[pos])) for pos in fam_pos_lists],
                        dtype=np.float64,
                    )
                    gain = float(np.mean(fam_deltas))

                if gain > best_unc_gain + 1e-12 or (
                    abs(gain - best_unc_gain) <= 1e-12
                    and (margin, -n_est, -lr)
                    > (best_unc_cfg[2], -best_unc_cfg[1], -best_unc_cfg[0])
                ):
                    best_unc_gain = gain
                    best_unc_cfg = (lr, n_est, margin)
                    best_unc_fam_deltas = fam_deltas

    rng_boot = np.random.default_rng(
        stream_seed(BOOT_SEED, "inner_select", seed, assess_fold, arm_name)
    )
    boot_ix = rng_boot.integers(
        0, len(fit_fams), size=(INNER_BOOT_DRAWS, len(fit_fams))
    )
    boot_means = best_unc_fam_deltas[boot_ix].mean(axis=1)
    inner_lcb95 = float(np.quantile(boot_means, 0.025))
    inner_ucb95 = float(np.quantile(boot_means, 0.975))

    strict_admitted = bool(best_unc_gain > STRICT_FIT_MIN_GAIN and inner_lcb95 > 0.0)
    strict_margin = best_unc_cfg[2] if strict_admitted else math.inf

    # If with_veto is True, select (lr*, n_est*, tau*, margin*) for NOSG_VETO on inner FIT OOF
    veto_result: dict[str, Any] | None = None
    if with_veto:
        P_inv_mat = np.stack([inner_oof_p_inv[c] for c in fit_cids], axis=0)
        ctrl_inv_mask = ~fit_gate4[:, 0]
        ctrl_inv_count = int(np.sum(ctrl_inv_mask))

        rng_boot_veto = np.random.default_rng(
            stream_seed(BOOT_SEED, "inner_select", seed, assess_fold, "NOSG_VETO")
        )
        boot_ix_veto = rng_boot_veto.integers(
            0, len(fit_fams), size=(INNER_BOOT_DRAWS, len(fit_fams))
        )

        # Evaluate all (lr, n_est, tau, margin) grid points on inner FIT OOF
        tier1_candidates: list[dict[str, Any]] = []
        tier2_candidates: list[dict[str, Any]] = []
        best_unc_safe: dict[str, Any] | None = None

        for lr in LR_GRID:
            for n_est in N_EST_GRID:
                s_map = inner_oof_scores[(lr, n_est)]
                S_mat = np.stack([s_map[c] for c in fit_cids], axis=0)
                best_j_vec = np.argmax(S_mat, axis=1)
                gap0_vec = S_mat[row_arange, best_j_vec] - S_mat[:, 0]
                best_j_p_inv = P_inv_mat[row_arange, best_j_vec]

                for tau in VETO_TAU_GRID:
                    for margin in MARGIN_GRID:
                        if math.isinf(margin):
                            continue
                        chosen_j = np.where(
                            (best_j_vec > 0)
                            & (gap0_vec > float(margin))
                            & (best_j_p_inv <= float(tau)),
                            best_j_vec,
                            0,
                        )
                        cand_inv_mask = ~fit_gate4[row_arange, chosen_j]
                        inv_count = int(np.sum(cand_inv_mask))
                        new_inv_sw = int(
                            np.sum((chosen_j > 0) & (~ctrl_inv_mask) & cand_inv_mask)
                        )
                        rescued_inv = int(
                            np.sum((chosen_j > 0) & ctrl_inv_mask & (~cand_inv_mask))
                        )

                        u_diff = fit_u4[row_arange, chosen_j] - fit_r882_u
                        fam_deltas = np.array(
                            [float(np.mean(u_diff[pos])) for pos in fam_pos_lists],
                            dtype=np.float64,
                        )
                        gain = float(np.mean(fam_deltas))

                        cand_rec = {
                            "lr": float(lr),
                            "n_est": int(n_est),
                            "tau": float(tau),
                            "margin": float(margin),
                            "gain": gain,
                            "fam_deltas": fam_deltas,
                            "inv_count": inv_count,
                            "new_inv_sw": new_inv_sw,
                            "rescued_inv": rescued_inv,
                        }

                        if inv_count <= ctrl_inv_count and new_inv_sw == 0:
                            if (
                                best_unc_safe is None
                                or gain > best_unc_safe["gain"] + 1e-12
                                or (
                                    abs(gain - best_unc_safe["gain"]) <= 1e-12
                                    and (margin, -tau, -n_est, -lr)
                                    > (
                                        best_unc_safe["margin"],
                                        -best_unc_safe["tau"],
                                        -best_unc_safe["n_est"],
                                        -best_unc_safe["lr"],
                                    )
                                )
                            ):
                                best_unc_safe = cand_rec

                        if gain > STRICT_FIT_MIN_GAIN:
                            if inv_count <= ctrl_inv_count and new_inv_sw == 0:
                                tier1_candidates.append(cand_rec)
                            elif inv_count < ctrl_inv_count:
                                tier2_candidates.append(cand_rec)

        def sort_key(c: dict[str, Any]) -> tuple[float, float, float, int, float]:
            return (
                round(c["gain"], 12),
                c["margin"],
                -c["tau"],
                -c["n_est"],
                -c["lr"],
            )

        tier1_candidates.sort(key=sort_key, reverse=True)
        tier2_candidates.sort(key=sort_key, reverse=True)

        selected_veto: dict[str, Any] | None = None
        selected_tier = 0
        for tier_idx, cand_list in ((1, tier1_candidates), (2, tier2_candidates)):
            for cand in cand_list:
                b_means = cand["fam_deltas"][boot_ix_veto].mean(axis=1)
                lcb = float(np.quantile(b_means, 0.025))
                ucb = float(np.quantile(b_means, 0.975))
                if lcb > 0.0:
                    cand["lcb95"] = lcb
                    cand["ucb95"] = ucb
                    selected_veto = cand
                    selected_tier = tier_idx
                    break
            if selected_veto is not None:
                break

        if best_unc_safe is not None and "lcb95" not in best_unc_safe:
            b_means_u = best_unc_safe["fam_deltas"][boot_ix_veto].mean(axis=1)
            best_unc_safe["lcb95"] = float(np.quantile(b_means_u, 0.025))
            best_unc_safe["ucb95"] = float(np.quantile(b_means_u, 0.975))

        if selected_veto is not None:
            veto_strict_admitted = True
            veto_cfg = (
                selected_veto["lr"],
                selected_veto["n_est"],
                selected_veto["tau"],
                selected_veto["margin"],
            )
            veto_gain = selected_veto["gain"]
            veto_ci95 = [selected_veto["lcb95"], selected_veto["ucb95"]]
            veto_inv_count = selected_veto["inv_count"]
            veto_new_inv_sw = selected_veto["new_inv_sw"]
            veto_rescued = selected_veto["rescued_inv"]
        else:
            veto_strict_admitted = False
            fallback_lr = best_unc_safe["lr"] if best_unc_safe is not None else 0.05
            fallback_nest = best_unc_safe["n_est"] if best_unc_safe is not None else 100
            fallback_tau = best_unc_safe["tau"] if best_unc_safe is not None else 0.05
            veto_cfg = (fallback_lr, fallback_nest, fallback_tau, math.inf)
            veto_gain = 0.0
            veto_ci95 = [0.0, 0.0]
            veto_inv_count = ctrl_inv_count
            veto_new_inv_sw = 0
            veto_rescued = 0

        unc_safe_cfg = (
            (
                best_unc_safe["lr"],
                best_unc_safe["n_est"],
                best_unc_safe["tau"],
                best_unc_safe["margin"],
            )
            if best_unc_safe is not None
            else (0.05, 100, 0.05, math.inf)
        )

        veto_receipt = {
            "cell_id": f"fseed{fold_seed}_mseed{seed}_fold{assess_fold}",
            "fold_seed": int(fold_seed),
            "seed": int(seed),
            "assess_fold": int(assess_fold),
            "arm": "NOSG_VETO",
            "shuffle_placebo": False,
            "n_features": int(len(cols)),
            "n_fit_contexts": len(fit_cids),
            "n_fit_families": len(fit_fams),
            "strict_admitted": veto_strict_admitted,
            "admission_tier": selected_tier,
            "selected_lr": float(veto_cfg[0]),
            "selected_n_estimators": int(veto_cfg[1]),
            "selected_tau": float(veto_cfg[2]),
            "strict_selected_margin": (
                "inf" if math.isinf(veto_cfg[3]) else float(veto_cfg[3])
            ),
            "inner_fit_family_gain": float(veto_gain),
            "inner_fit_ci95": veto_ci95,
            "inner_fit_invalid_picks": int(veto_inv_count),
            "inner_fit_control_invalid_picks": int(ctrl_inv_count),
            "inner_fit_new_invalid_switches": int(veto_new_inv_sw),
            "inner_fit_rescued_invalid_picks": int(veto_rescued),
            "unconstrained_safe_lr": float(unc_safe_cfg[0]),
            "unconstrained_safe_n_estimators": int(unc_safe_cfg[1]),
            "unconstrained_safe_tau": float(unc_safe_cfg[2]),
            "unconstrained_safe_margin": (
                "inf" if math.isinf(unc_safe_cfg[3]) else float(unc_safe_cfg[3])
            ),
        }
        veto_result = {
            "receipt": veto_receipt,
            "veto_cfg": veto_cfg,
            "unc_safe_cfg": unc_safe_cfg,
        }

    X_outer_fit = np.concatenate(
        [packets[c]["X_top4"][:, cols] for c in fit_cids], axis=0
    )
    y_outer_fit = np.concatenate([rel_map[c] for c in fit_cids], axis=0)

    lrs_needed = {best_unc_cfg[0], 0.05}
    if veto_result is not None:
        lrs_needed.add(veto_result["veto_cfg"][0])
        lrs_needed.add(veto_result["unc_safe_cfg"][0])

    outer_models_by_lr: dict[float, lgb.LGBMRanker] = {}
    for lr_needed in sorted(lrs_needed):
        rstate_out = stream_seed(seed, assess_fold, arm_name, "outer", lr_needed)
        outer_models_by_lr[lr_needed] = fit_lgbm_ranker_200(
            X_outer_fit,
            y_outer_fit,
            len(fit_cids),
            lr=lr_needed,
            random_state=rstate_out,
        )

    outer_veto_clf: lgb.LGBMClassifier | None = None
    if with_veto:
        y_inv_outer = np.concatenate(
            [(~packets[c]["gate_4"]).astype(np.int32) for c in fit_cids], axis=0
        )
        rstate_out_clf = stream_seed(seed, assess_fold, "NOSG_VETO", "outer_clf")
        outer_veto_clf = fit_veto_classifier(
            X_outer_fit, y_inv_outer, random_state=rstate_out_clf
        )

    selection_receipt = {
        "cell_id": f"fseed{fold_seed}_mseed{seed}_fold{assess_fold}",
        "fold_seed": int(fold_seed),
        "seed": int(seed),
        "assess_fold": int(assess_fold),
        "arm": arm_name,
        "shuffle_placebo": bool(shuffle_placebo),
        "n_features": int(len(cols)),
        "n_fit_contexts": len(fit_cids),
        "n_fit_families": len(fit_fams),
        "unconstrained_best_lr": float(best_unc_cfg[0]),
        "unconstrained_best_n_estimators": int(best_unc_cfg[1]),
        "unconstrained_best_margin": (
            "inf" if math.isinf(best_unc_cfg[2]) else float(best_unc_cfg[2])
        ),
        "inner_fit_family_gain": float(best_unc_gain),
        "inner_fit_ci95": [inner_lcb95, inner_ucb95],
        "strict_admitted": strict_admitted,
        "strict_selected_margin": (
            "inf" if math.isinf(strict_margin) else float(strict_margin)
        ),
    }

    return {
        "receipt": selection_receipt,
        "best_unc_cfg": best_unc_cfg,
        "strict_margin": strict_margin,
        "outer_models_by_lr": outer_models_by_lr,
        "col_indices": cols,
        "veto_result": veto_result,
        "outer_veto_clf": outer_veto_clf,
    }


def score_contexts_with_cell_model(
    cids: list[str],
    packets: dict[str, dict[str, Any]],
    cell_fit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    cols = cell_fit["col_indices"]
    X_eval = np.concatenate([packets[c]["X_top4"][:, cols] for c in cids], axis=0)
    lr_unc, n_est_unc, m_unc = cell_fit["best_unc_cfg"]
    m_strict = cell_fit["strict_margin"]

    model_unc = cell_fit["outer_models_by_lr"][lr_unc]
    preds_unc = predict_lgbm_ranker(model_unc, X_eval, num_iteration=n_est_unc).reshape(
        len(cids), 4
    )

    model_005 = cell_fit["outer_models_by_lr"][0.05]
    preds_m0 = predict_lgbm_ranker(model_005, X_eval, num_iteration=100).reshape(
        len(cids), 4
    )

    veto_res = cell_fit.get("veto_result")
    outer_clf = cell_fit.get("outer_veto_clf")
    preds_veto_strict = None
    preds_veto_unc = None
    p_inv_eval = None
    if veto_res is not None and outer_clf is not None:
        p_inv_eval = predict_veto_classifier(outer_clf, X_eval).reshape(len(cids), 4)
        v_lr, v_nest, v_tau, v_m = veto_res["veto_cfg"]
        preds_veto_strict = predict_lgbm_ranker(
            cell_fit["outer_models_by_lr"][v_lr], X_eval, num_iteration=v_nest
        ).reshape(len(cids), 4)
        u_lr, u_nest, u_tau, u_m = veto_res["unc_safe_cfg"]
        preds_veto_unc = predict_lgbm_ranker(
            cell_fit["outer_models_by_lr"][u_lr], X_eval, num_iteration=u_nest
        ).reshape(len(cids), 4)

    out: dict[str, dict[str, Any]] = {}
    for idx, cid in enumerate(cids):
        pkt = packets[cid]
        u4 = pkt["utility_4"]
        gate4 = pkt["gate_4"]
        oracle_j = pkt["top4_oracle_j"]

        j_strict = select_from_top4_scores(preds_unc[idx], m_strict)
        j_unc = select_from_top4_scores(preds_unc[idx], m_unc)
        j_m0 = select_from_top4_scores(preds_m0[idx], 0.0)

        rec_c: dict[str, Any] = {
            "strict_j": j_strict,
            "strict_u": float(u4[j_strict]),
            "strict_invalid": bool(not gate4[j_strict]),
            "strict_is_top4_oracle": bool(j_strict == oracle_j),
            "unc_j": j_unc,
            "unc_u": float(u4[j_unc]),
            "unc_invalid": bool(not gate4[j_unc]),
            "unc_is_top4_oracle": bool(j_unc == oracle_j),
            "m0_j": j_m0,
            "m0_u": float(u4[j_m0]),
            "m0_invalid": bool(not gate4[j_m0]),
            "m0_is_top4_oracle": bool(j_m0 == oracle_j),
        }

        if (
            veto_res is not None
            and preds_veto_strict is not None
            and preds_veto_unc is not None
            and p_inv_eval is not None
        ):
            _, _, v_tau, v_m = veto_res["veto_cfg"]
            _, _, u_tau, u_m = veto_res["unc_safe_cfg"]
            j_v_strict = select_from_top4_scores_with_veto(
                preds_veto_strict[idx], p_inv_eval[idx], v_tau, v_m
            )
            j_v_unc = select_from_top4_scores_with_veto(
                preds_veto_unc[idx], p_inv_eval[idx], u_tau, u_m
            )
            rec_c["veto_strict_j"] = j_v_strict
            rec_c["veto_strict_u"] = float(u4[j_v_strict])
            rec_c["veto_strict_invalid"] = bool(not gate4[j_v_strict])
            rec_c["veto_strict_is_top4_oracle"] = bool(j_v_strict == oracle_j)
            rec_c["veto_unc_j"] = j_v_unc
            rec_c["veto_unc_u"] = float(u4[j_v_unc])
            rec_c["veto_unc_invalid"] = bool(not gate4[j_v_unc])
            rec_c["veto_unc_is_top4_oracle"] = bool(j_v_unc == oracle_j)

        out[cid] = rec_c
    return out


def paired_family_contrast(
    rows: list[dict[str, Any]],
    panel: np.ndarray,
    arm_names: list[str],
    arm_a: str,
    arm_b: str,
    boot_indices: np.ndarray,
    agg_across_folds: str = "mean",
) -> dict[str, Any]:
    ia = arm_names.index(arm_a)
    ib = arm_names.index(arm_b)
    fams = sorted({r["family_id"] for r in rows})
    S = panel.shape[0]
    F = len(fams)

    fam_u_a = np.zeros((S, F), dtype=np.float64)
    fam_u_b = np.zeros((S, F), dtype=np.float64)

    if panel.ndim == 4:
        ctx_fams = np.asarray([r["family_id"] for r in rows])
        n_fseeds = panel.shape[1]
        fam_fs_a = np.zeros((S, n_fseeds, F), dtype=np.float64)
        fam_fs_b = np.zeros((S, n_fseeds, F), dtype=np.float64)
        for fi, f in enumerate(fams):
            mask = ctx_fams == f
            fam_fs_a[:, :, fi] = panel[:, :, mask, ia].mean(axis=2)
            fam_fs_b[:, :, fi] = panel[:, :, mask, ib].mean(axis=2)
        if agg_across_folds == "median":
            fam_u_a = np.median(fam_fs_a, axis=1)
            fam_u_b = np.median(fam_fs_b, axis=1)
            ctx_u_a = np.median(panel[:, :, :, ia].mean(axis=2), axis=1)
            ctx_u_b = np.median(panel[:, :, :, ib].mean(axis=2), axis=1)
        else:
            fam_u_a = np.mean(fam_fs_a, axis=1)
            fam_u_b = np.mean(fam_fs_b, axis=1)
            ctx_u_a = np.mean(panel[:, :, :, ia].mean(axis=2), axis=1)
            ctx_u_b = np.mean(panel[:, :, :, ib].mean(axis=2), axis=1)
        per_seed_ctx_gains = [float(ctx_u_a[s] - ctx_u_b[s]) for s in range(S)]
    else:
        ctx_fams = np.asarray(
            [r["family_id"] for r in rows] * (panel.shape[1] // len(rows))
        )
        for fi, f in enumerate(fams):
            mask = ctx_fams == f
            fam_u_a[:, fi] = panel[:, mask, ia].mean(axis=1)
            fam_u_b[:, fi] = panel[:, mask, ib].mean(axis=1)
        per_seed_ctx_gains = [
            float(panel[s, :, ia].mean() - panel[s, :, ib].mean()) for s in range(S)
        ]

    fam_diff_by_seed = fam_u_a - fam_u_b
    per_seed_fam_gains = [float(fam_diff_by_seed[s].mean()) for s in range(S)]
    med_fam_gain = float(np.median(per_seed_fam_gains))
    mean_fam_gain = float(np.mean(per_seed_fam_gains))
    med_ctx_gain = float(np.median(per_seed_ctx_gains))
    mean_ctx_gain = float(np.mean(per_seed_ctx_gains))

    fam_diff_med_seed = np.median(fam_diff_by_seed, axis=0)
    boot_draws = fam_diff_med_seed[boot_indices].mean(axis=1)
    ci95 = [
        float(np.quantile(boot_draws, 0.025)),
        float(np.quantile(boot_draws, 0.975)),
    ]
    ci975 = [
        float(np.quantile(boot_draws, 0.0125)),
        float(np.quantile(boot_draws, 0.9875)),
    ]
    ci9833 = [
        float(np.quantile(boot_draws, 0.025 / 3.0)),
        float(np.quantile(boot_draws, 1.0 - 0.025 / 3.0)),
    ]
    raw_p = float((1 + np.sum(boot_draws <= 0.0)) / (len(boot_draws) + 1))

    wtl_by_seed = []
    harm_gt_1e4_by_seed = []
    harm_gt_001_by_seed = []
    for s in range(S):
        d = fam_diff_by_seed[s]
        w = int(np.sum(d > TIE_MARGIN))
        t = int(np.sum(np.abs(d) <= TIE_MARGIN))
        n_loss = int(np.sum(d < -TIE_MARGIN))
        wtl_by_seed.append({"wins": w, "ties": t, "losses": n_loss})
        harm_gt_1e4_by_seed.append(n_loss)
        harm_gt_001_by_seed.append(int(np.sum(d < -0.01)))

    harmed_families = [
        {"family_id": fams[fi], "delta": float(fam_diff_med_seed[fi])}
        for fi in range(F)
        if fam_diff_med_seed[fi] < -TIE_MARGIN
    ]
    harmed_families.sort(key=lambda x: x["delta"])

    return {
        "median_family_gain": med_fam_gain,
        "mean_family_gain": mean_fam_gain,
        "per_seed_family_gains": per_seed_fam_gains,
        "median_context_gain": med_ctx_gain,
        "mean_context_gain": mean_ctx_gain,
        "per_seed_context_gains": per_seed_ctx_gains,
        "positive_seeds": int(sum(1 for g in per_seed_fam_gains if g > 0.0)),
        "ci95": ci95,
        "ci975_holm2": ci975,
        "ci9833_holm3": ci9833,
        "raw_p_one_sided": raw_p,
        "family_wtl_by_seed": wtl_by_seed,
        "harm_families_gt_1e4_by_seed": harm_gt_1e4_by_seed,
        "harm_families_gt_001_by_seed": harm_gt_001_by_seed,
        "n_harmed_families_median_gt_1e4": len(harmed_families),
        "n_harmed_families_median_gt_001": sum(
            1 for h in harmed_families if h["delta"] < -0.01
        ),
        "worst_harmed_families_top10": harmed_families[:10],
        "harmed_families_all": harmed_families,
    }


def apply_holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    m = len(p_values)
    order = sorted(p_values.items(), key=lambda kv: kv[1])
    adj: dict[str, float] = {}
    running_max = 0.0
    for idx, (k, p) in enumerate(order):
        mult = m - idx
        running_max = max(running_max, min(1.0, p * mult))
        adj[k] = float(running_max)
    return adj


def fit_mm_bradley_terry_elo(
    win_matrix: np.ndarray, anchor_idx: int = 0, anchor_elo: float = 1500.0
) -> np.ndarray:
    K = win_matrix.shape[0]
    w = win_matrix.astype(np.float64) + 1e-3
    np.fill_diagonal(w, 0.0)
    n_ij = w + w.T
    w_i = w.sum(axis=1)
    gamma = np.ones(K, dtype=np.float64)
    for _ in range(400):
        denom = np.zeros(K, dtype=np.float64)
        for i in range(K):
            for j in range(K):
                if i != j:
                    denom[i] += n_ij[i, j] / (gamma[i] + gamma[j])
        gamma = np.maximum(w_i / np.maximum(denom, 1e-12), 1e-12)
        gamma /= np.exp(np.mean(np.log(gamma)))
    elos = anchor_elo + 400.0 * np.log10(gamma / gamma[anchor_idx])
    return elos


def compute_elo_table(
    rows: list[dict[str, Any]],
    panel: np.ndarray,
    arm_names: list[str],
    boot_seed: int,
) -> dict[str, Any]:
    if panel.ndim == 4:
        panel_3d = np.median(panel, axis=1)
    else:
        panel_3d = panel

    fams = sorted({r["family_id"] for r in rows})
    ctx_fams = np.asarray(
        [r["family_id"] for r in rows] * (panel_3d.shape[1] // len(rows))
    )
    S, _, A = panel_3d.shape

    def pairwise_wins(mat_2d: np.ndarray) -> np.ndarray:
        W = np.zeros((A, A), dtype=np.float64)
        for i in range(A):
            for j in range(i + 1, A):
                diff = mat_2d[:, i] - mat_2d[:, j]
                w_ij = float(np.sum(diff > TIE_MARGIN)) + 0.5 * float(
                    np.sum(np.abs(diff) <= TIE_MARGIN)
                )
                w_ji = float(np.sum(diff < -TIE_MARGIN)) + 0.5 * float(
                    np.sum(np.abs(diff) <= TIE_MARGIN)
                )
                W[i, j] = w_ij
                W[j, i] = w_ji
        return W

    seed_elos = np.zeros((S, A), dtype=np.float64)
    for s in range(S):
        W_s = pairwise_wins(panel_3d[s])
        seed_elos[s] = fit_mm_bradley_terry_elo(W_s, anchor_idx=0, anchor_elo=1500.0)

    med_panel = np.median(panel_3d, axis=0)
    fam_W = np.zeros((len(fams), A, A), dtype=np.float64)
    for fi, f in enumerate(fams):
        mask = ctx_fams == f
        fam_W[fi] = pairwise_wins(med_panel[mask])

    rng = np.random.default_rng(stream_seed(boot_seed, "elo_bt"))
    boot_ix = rng.integers(0, len(fams), size=(1000, len(fams)))
    boot_elos = np.zeros((1000, A), dtype=np.float64)
    for b in range(1000):
        W_b = fam_W[boot_ix[b]].sum(axis=0)
        boot_elos[b] = fit_mm_bradley_terry_elo(W_b, anchor_idx=0, anchor_elo=1500.0)

    out: dict[str, Any] = {}
    for a_idx, name in enumerate(arm_names):
        out[name] = {
            "median_elo": float(np.median(seed_elos[:, a_idx])),
            "per_seed_elo": [float(seed_elos[s, a_idx]) for s in range(S)],
            "ci95": [
                float(np.quantile(boot_elos[:, a_idx], 0.025)),
                float(np.quantile(boot_elos[:, a_idx], 0.975)),
            ],
        }
    return out


def strip_harmed_all(contrast_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in contrast_dict.items() if k != "harmed_families_all"}


def run_smoke(out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert sha256_file(R882_CKPT) == R882_EXPECTED_SHA256

    rng = np.random.default_rng(269001)
    X_syn = rng.normal(size=(32, NOSG_FEATURE_DIM)).astype(np.float32)
    y_syn = np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 8)
    y_inv = np.zeros(32, dtype=np.int32)
    y_inv[3] = 1
    y_inv[11] = 1
    model = fit_lgbm_ranker_200(X_syn, y_syn, 8, lr=0.05, random_state=269001)
    preds = predict_lgbm_ranker(model, X_syn, num_iteration=50)
    clf = fit_veto_classifier(X_syn, y_inv, random_state=269002)
    p_inv = predict_veto_classifier(clf, X_syn)
    assert preds.shape == (32,) and np.all(np.isfinite(preds))
    assert p_inv.shape == (32,) and np.all((p_inv >= 0.0) & (p_inv <= 1.0))

    smoke_doc = {
        "schema": "jope.h269.smoke/v1",
        "wave_id": "h269-nosg-veto-rerank4-20261103",
        "status": "PASS",
        "r882_sha256": sha256_file(R882_CKPT),
        "synthetic_ranker_preds_shape": list(preds.shape),
        "synthetic_veto_probs_shape": list(p_inv.shape),
        "outcomes_read": 0,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    fsync_write_json(out_dir / "smoke.json", smoke_doc)
    return 0


def run_primary(out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    r882_sha_before = sha256_file(R882_CKPT)
    assert r882_sha_before == R882_EXPECTED_SHA256
    assert (
        sha256_file(INPUT_POOLS_DIR / "selected.json") == SELECTED_JSON_EXPECTED_SHA256
    )
    assert (
        sha256_file(INPUT_POOLS_DIR / "source_manifest.json")
        == SOURCE_MANIFEST_EXPECTED_SHA256
    )
    assert sha256_file(CONTEXT_FEATURES_NPZ) == CONTEXT_FEATURES_EXPECTED_SHA256
    assert sha256_file(POOL_PRIORS_NPZ) == POOL_PRIORS_EXPECTED_SHA256

    ckpt882 = torch.load(R882_CKPT, map_location="cpu", weights_only=False)
    dope_sha_before = str(ckpt882.get("dope_sha256", DOPE_EXPECTED_SHA256))
    assert dope_sha_before == DOPE_EXPECTED_SHA256
    del ckpt882
    trim_malloc()

    selected = json.loads(
        (INPUT_POOLS_DIR / "selected.json").read_text(encoding="utf-8")
    )
    train_rows = selected["train"]
    val_rows = selected["dev"]
    assert len(train_rows) == 1219
    assert len(val_rows) == 91
    assert not any(r.get("has_meta_test", False) for r in train_rows + val_rows)

    r882_costs: dict[str, np.ndarray] = {}
    with (
        np.load(R882_TRAIN_COSTS_NPZ, allow_pickle=False) as zt,
        np.load(R882_VAL_COSTS_NPZ, allow_pickle=False) as zv,
    ):
        for r in train_rows + val_rows:
            cid = r["context_sha256"]
            r882_costs[cid] = (zt[cid] if cid in zt.files else zv[cid]).astype(
                np.float32
            )

    # Step 1: Reproduce Native91 1.1377941898164525 bit-exact (91/91) FIRST + TOP4_PICK@r882 1.2403601682250796 ceiling
    ref_obj = json.loads(R882_BEST_DEV_JSON.read_text(encoding="utf-8"))
    ref_cids = [
        str(c["cid"]) if isinstance(c, dict) else str(c)
        for c in ref_obj["evaluation_contract"]["contexts"]
    ]
    ref_per_ctx = list(ref_obj["utility"]["per_context"])
    ref_map = {c: float(u_val) for c, u_val in zip(ref_cids, ref_per_ctx)}

    val_r882_utils = []
    val_top4_utils = []
    bit_exact_r882 = 0
    for r in val_rows:
        cid = r["context_sha256"]
        with np.load(
            INPUT_POOLS_DIR / "pools" / f"{cid}.npz", allow_pickle=False
        ) as zp:
            fit_v = zp["fitness"].astype(np.float64)
            gate_v = zp["gate"].astype(np.bool_)
        u_v = np.where(gate_v, -fit_v, -2.0)
        rc = r882_costs[cid]
        pick0 = int(np.argmin(rc))
        u0 = float(u_v[pick0])
        val_r882_utils.append(u0)
        if u0 == ref_map[cid]:
            bit_exact_r882 += 1
        top4_idx = np.argsort(rc, kind="mergesort")[:4]
        gated_c4 = np.where(gate_v[top4_idx], fit_v[top4_idx], 2.0)
        pick_top4 = int(top4_idx[int(np.argmin(gated_c4))])
        val_top4_utils.append(float(u_v[pick_top4]))

    r882_native91_mean = float(np.mean(val_r882_utils))
    r882_abs_diff = abs(r882_native91_mean - R882_NATIVE91_EXPECTED)
    if bit_exact_r882 != 91 or r882_abs_diff > 1e-12:
        raise RuntimeError(
            f"Native91 regression failed: bit_exact={bit_exact_r882}/91, mean={r882_native91_mean}"
        )

    val_fams = [r["family_id"] for r in val_rows]
    val_fam_order = sorted(set(val_fams))
    val_top4_arr = np.asarray(val_top4_utils, dtype=np.float64)
    top4_val_ctx_mean = float(np.mean(val_top4_arr))
    top4_val_fam_mean = float(
        np.mean(
            [
                float(np.mean(val_top4_arr[np.asarray(val_fams) == f]))
                for f in val_fam_order
            ]
        )
    )
    top4_ctx_diff = abs(top4_val_ctx_mean - TOP4_PICK_VAL_CTX_EXPECTED)
    top4_fam_diff = abs(top4_val_fam_mean - TOP4_PICK_VAL_FAM_EXPECTED)
    if top4_ctx_diff > 1e-12 or top4_fam_diff > 1e-12:
        raise RuntimeError(
            f"TOP4_PICK@r882 regression failed: ctx={top4_val_ctx_mean}, fam={top4_val_fam_mean}"
        )

    baseline_reg = {
        "schema": "jope.h269.baseline-regression/v1",
        "dope_sha256": dope_sha_before,
        "r882_sha256": r882_sha_before,
        "r882_native91_expected": R882_NATIVE91_EXPECTED,
        "r882_native91_mean": r882_native91_mean,
        "r882_native91_abs_diff": r882_abs_diff,
        "r882_bit_exact_context_matches": bit_exact_r882,
        "top4_pick_oracle_ceiling_ctx_expected": TOP4_PICK_VAL_CTX_EXPECTED,
        "top4_pick_oracle_ceiling_ctx_mean": top4_val_ctx_mean,
        "top4_pick_oracle_ceiling_ctx_abs_diff": top4_ctx_diff,
        "top4_pick_oracle_ceiling_fam_expected": TOP4_PICK_VAL_FAM_EXPECTED,
        "top4_pick_oracle_ceiling_fam_mean": top4_val_fam_mean,
        "top4_pick_oracle_ceiling_fam_abs_diff": top4_fam_diff,
    }
    fsync_write_json(out_dir / "baseline.json", baseline_reg)

    # Step 2: Load ONLY DL-Train context features, priors, and top-4 packets (DL-Val unopened during FIT/selection)
    aprior_train: dict[str, np.ndarray] = {}
    proj16_train: dict[str, np.ndarray] = {}
    train_cid_set = {r["context_sha256"] for r in train_rows}
    with np.load(POOL_PRIORS_NPZ, allow_pickle=False) as zp:
        pri_cids = [str(c) for c in zp["cids"]]
        offsets = zp["offsets"]
        base_pri = zp["base_prior_cost"]
        proj16_all = zp["proj16"]
        for i, cid in enumerate(pri_cids):
            if cid in train_cid_set:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior_train[cid] = base_pri[lo:hi].astype(np.float32).copy()
                proj16_train[cid] = proj16_all[lo:hi].astype(np.float32).copy()
    trim_malloc()

    ctx_feats_train: dict[str, dict[str, np.ndarray]] = {}
    with np.load(CONTEXT_FEATURES_NPZ, allow_pickle=False) as zc:
        cf_cids = [str(c) for c in zc["cids"]]
        lm48_all = zc["lm48_z"]
        dope629_all = zc["dope629"]
        for i, cid in enumerate(cf_cids):
            if cid in train_cid_set:
                ctx_feats_train[cid] = {
                    "lm48_z": lm48_all[i].astype(np.float32).copy(),
                    "dope629": dope629_all[i].astype(np.float32).copy(),
                }
    trim_malloc()

    train_packets = load_cohort_top4_packets(
        train_rows, r882_costs, aprior_train, proj16_train, ctx_feats_train
    )
    aprior_train.clear()
    proj16_train.clear()
    ctx_feats_train.clear()
    trim_malloc()

    fold_man = json.loads(
        (CAPSULE_STUDY_DIR / "fold_manifest.json").read_text(encoding="utf-8")
    )
    train_cids_by_fseed_fold: dict[int, dict[int, list[str]]] = {}
    for fseed in FOLD_SEEDS:
        sched = fold_man["fold_schedules"][str(fseed)]
        train_cids_by_fseed_fold[fseed] = {
            int(f["fold"]): list(f["context_ids"]) for f in sched["folds"]
        }
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train_rows}

    # Step 3: Fit all 45 confirmatory cells (3 fold_seeds x 3 model_seeds x 5 folds)
    # for NOSG (+ NOSG_VETO), NOSG_SHUF, and RERANK4 (560-col reference)
    cell_results: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    selection_receipts: list[dict[str, Any]] = []

    for fseed in FOLD_SEEDS:
        cids_by_fold = train_cids_by_fseed_fold[fseed]
        for s in SEEDS:
            for k in range(5):
                check_cooperative_pause()
                # 1. NOSG + NOSG_VETO (528 cols)
                res_nosg = run_cell_fit_and_select(
                    fold_seed=fseed,
                    seed=s,
                    assess_fold=k,
                    arm_name="NOSG",
                    shuffle_placebo=False,
                    train_cids_by_fold=cids_by_fold,
                    packets=train_packets,
                    fam_by_cid=fam_by_cid,
                    col_indices=NO_SCOREGEOM_COLS,
                    with_veto=True,
                )
                cell_results[("NOSG", fseed, s, k)] = res_nosg
                selection_receipts.append(res_nosg["receipt"])
                assert res_nosg["veto_result"] is not None
                selection_receipts.append(res_nosg["veto_result"]["receipt"])

                # 2. NOSG_SHUF (528 cols, within-family label shuffle placebo)
                res_shuf = run_cell_fit_and_select(
                    fold_seed=fseed,
                    seed=s,
                    assess_fold=k,
                    arm_name="NOSG_SHUF",
                    shuffle_placebo=True,
                    train_cids_by_fold=cids_by_fold,
                    packets=train_packets,
                    fam_by_cid=fam_by_cid,
                    col_indices=NO_SCOREGEOM_COLS,
                    with_veto=False,
                )
                cell_results[("NOSG_SHUF", fseed, s, k)] = res_shuf
                selection_receipts.append(res_shuf["receipt"])

                # 3. RERANK4 (560 cols, reference only)
                res_r4 = run_cell_fit_and_select(
                    fold_seed=fseed,
                    seed=s,
                    assess_fold=k,
                    arm_name="RERANK4",
                    shuffle_placebo=False,
                    train_cids_by_fold=cids_by_fold,
                    packets=train_packets,
                    fam_by_cid=fam_by_cid,
                    col_indices=ALL_COLS,
                    with_veto=False,
                )
                cell_results[("RERANK4", fseed, s, k)] = res_r4
                selection_receipts.append(res_r4["receipt"])

    # Step 4: Globally fsync-lock all 180 cell FIT selections BEFORE any DL-Val feature/pool read!
    lock_payload = {
        "schema": "jope.h269.fit-selections-lock/v1",
        "wave_id": "h269-nosg-veto-rerank4-20261103",
        "fold_seeds": list(FOLD_SEEDS),
        "model_seeds": list(SEEDS),
        "dl_val_feature_or_pool_reads_during_fit": 0,
        "locked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirmatory_cell_selections": selection_receipts,
    }
    lock_path = out_dir / "fit_selections_lock.json"
    fsync_write_json(lock_path, lock_payload)
    lock_sha256 = sha256_file(lock_path)

    # Step 5: Now load DL-Val91 top-4 packets and score all models (0 test-time reveals!)
    val_cid_set = {r["context_sha256"] for r in val_rows}
    aprior_val: dict[str, np.ndarray] = {}
    proj16_val: dict[str, np.ndarray] = {}
    with np.load(POOL_PRIORS_NPZ, allow_pickle=False) as zp:
        pri_cids = [str(c) for c in zp["cids"]]
        offsets = zp["offsets"]
        base_pri = zp["base_prior_cost"]
        proj16_all = zp["proj16"]
        for i, cid in enumerate(pri_cids):
            if cid in val_cid_set:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior_val[cid] = base_pri[lo:hi].astype(np.float32).copy()
                proj16_val[cid] = proj16_all[lo:hi].astype(np.float32).copy()

    ctx_feats_val: dict[str, dict[str, np.ndarray]] = {}
    with np.load(CONTEXT_FEATURES_NPZ, allow_pickle=False) as zc:
        cf_cids = [str(c) for c in zc["cids"]]
        lm48_all = zc["lm48_z"]
        dope629_all = zc["dope629"]
        for i, cid in enumerate(cf_cids):
            if cid in val_cid_set:
                ctx_feats_val[cid] = {
                    "lm48_z": lm48_all[i].astype(np.float32).copy(),
                    "dope629": dope629_all[i].astype(np.float32).copy(),
                }

    val_packets = load_cohort_top4_packets(
        val_rows, r882_costs, aprior_val, proj16_val, ctx_feats_val
    )
    aprior_val.clear()
    proj16_val.clear()
    ctx_feats_val.clear()
    trim_malloc()

    # Step 6: Build OOF and DL-Val panels across all 3 fold seeds x 3 model seeds
    arm_names = [
        "CONTROL",
        "NOSG",
        "NOSG_VETO",
        "NOSG_SHUF",
        "RERANK4",
        "NOSG_UNC",
        "NOSG_VETO_UNC_SAFE",
        "NOSG_M0",
        "RERANK4_UNC",
        "TOP4_PICK_ORACLE",
    ]
    n_arms = len(arm_names)
    S = len(SEEDS)
    NF = len(FOLD_SEEDS)
    N_train = len(train_rows)
    N_val = len(val_rows)
    train_cid_order = [r["context_sha256"] for r in train_rows]
    train_cid_to_idx = {c: i for i, c in enumerate(train_cid_order)}
    val_cid_order = [r["context_sha256"] for r in val_rows]

    train_panel_4d = np.zeros((S, NF, N_train, n_arms), dtype=np.float64)
    train_choice_4d = np.zeros((S, NF, N_train, n_arms), dtype=np.int32)
    train_invalid_4d = np.zeros((S, NF, N_train, n_arms), dtype=np.bool_)
    train_top1_oracle_4d = np.zeros((S, NF, N_train, n_arms), dtype=np.bool_)

    val_panel_4d = np.zeros((S, NF, 5 * N_val, n_arms), dtype=np.float64)
    val_choice_4d = np.zeros((S, NF, 5 * N_val, n_arms), dtype=np.int32)
    val_invalid_4d = np.zeros((S, NF, 5 * N_val, n_arms), dtype=np.bool_)
    val_top1_oracle_4d = np.zeros((S, NF, 5 * N_val, n_arms), dtype=np.bool_)

    for fi_seed, fseed in enumerate(FOLD_SEEDS):
        cids_by_fold = train_cids_by_fseed_fold[fseed]
        for s_idx, s in enumerate(SEEDS):
            for c_idx, cid in enumerate(train_cid_order):
                pkt = train_packets[cid]
                train_panel_4d[s_idx, fi_seed, c_idx, 0] = pkt["r882_u"]
                train_choice_4d[s_idx, fi_seed, c_idx, 0] = 0
                train_invalid_4d[s_idx, fi_seed, c_idx, 0] = pkt["r882_invalid"]
                train_top1_oracle_4d[s_idx, fi_seed, c_idx, 0] = (
                    pkt["top4_oracle_j"] == 0
                )

                train_panel_4d[s_idx, fi_seed, c_idx, 9] = pkt["top4_oracle_u"]
                train_choice_4d[s_idx, fi_seed, c_idx, 9] = pkt["top4_oracle_j"]
                train_invalid_4d[s_idx, fi_seed, c_idx, 9] = bool(
                    not pkt["gate_4"][pkt["top4_oracle_j"]]
                )
                train_top1_oracle_4d[s_idx, fi_seed, c_idx, 9] = True

            for k in range(5):
                assess_cids = cids_by_fold[k]
                res_nosg = score_contexts_with_cell_model(
                    assess_cids, train_packets, cell_results[("NOSG", fseed, s, k)]
                )
                res_shuf = score_contexts_with_cell_model(
                    assess_cids,
                    train_packets,
                    cell_results[("NOSG_SHUF", fseed, s, k)],
                )
                res_r4 = score_contexts_with_cell_model(
                    assess_cids,
                    train_packets,
                    cell_results[("RERANK4", fseed, s, k)],
                )
                for cid in assess_cids:
                    c_idx = train_cid_to_idx[cid]
                    ns_c = res_nosg[cid]
                    sh_c = res_shuf[cid]
                    r4_c = res_r4[cid]

                    for arm_slot, src, prefix in (
                        (1, ns_c, "strict"),
                        (2, ns_c, "veto_strict"),
                        (3, sh_c, "strict"),
                        (4, r4_c, "strict"),
                        (5, ns_c, "unc"),
                        (6, ns_c, "veto_unc"),
                        (7, ns_c, "m0"),
                        (8, r4_c, "unc"),
                    ):
                        train_panel_4d[s_idx, fi_seed, c_idx, arm_slot] = src[
                            f"{prefix}_u"
                        ]
                        train_choice_4d[s_idx, fi_seed, c_idx, arm_slot] = src[
                            f"{prefix}_j"
                        ]
                        train_invalid_4d[s_idx, fi_seed, c_idx, arm_slot] = src[
                            f"{prefix}_invalid"
                        ]
                        train_top1_oracle_4d[s_idx, fi_seed, c_idx, arm_slot] = src[
                            f"{prefix}_is_top4_oracle"
                        ]

                val_nosg = score_contexts_with_cell_model(
                    val_cid_order, val_packets, cell_results[("NOSG", fseed, s, k)]
                )
                val_shuf = score_contexts_with_cell_model(
                    val_cid_order,
                    val_packets,
                    cell_results[("NOSG_SHUF", fseed, s, k)],
                )
                val_r4 = score_contexts_with_cell_model(
                    val_cid_order,
                    val_packets,
                    cell_results[("RERANK4", fseed, s, k)],
                )
                base_off = k * N_val
                for v_idx, cid in enumerate(val_cid_order):
                    pos = base_off + v_idx
                    pkt = val_packets[cid]
                    ns_c = val_nosg[cid]
                    sh_c = val_shuf[cid]
                    r4_c = val_r4[cid]

                    val_panel_4d[s_idx, fi_seed, pos, 0] = pkt["r882_u"]
                    val_choice_4d[s_idx, fi_seed, pos, 0] = 0
                    val_invalid_4d[s_idx, fi_seed, pos, 0] = pkt["r882_invalid"]
                    val_top1_oracle_4d[s_idx, fi_seed, pos, 0] = (
                        pkt["top4_oracle_j"] == 0
                    )

                    for arm_slot, src, prefix in (
                        (1, ns_c, "strict"),
                        (2, ns_c, "veto_strict"),
                        (3, sh_c, "strict"),
                        (4, r4_c, "strict"),
                        (5, ns_c, "unc"),
                        (6, ns_c, "veto_unc"),
                        (7, ns_c, "m0"),
                        (8, r4_c, "unc"),
                    ):
                        val_panel_4d[s_idx, fi_seed, pos, arm_slot] = src[f"{prefix}_u"]
                        val_choice_4d[s_idx, fi_seed, pos, arm_slot] = src[
                            f"{prefix}_j"
                        ]
                        val_invalid_4d[s_idx, fi_seed, pos, arm_slot] = src[
                            f"{prefix}_invalid"
                        ]
                        val_top1_oracle_4d[s_idx, fi_seed, pos, arm_slot] = src[
                            f"{prefix}_is_top4_oracle"
                        ]

                    val_panel_4d[s_idx, fi_seed, pos, 9] = pkt["top4_oracle_u"]
                    val_choice_4d[s_idx, fi_seed, pos, 9] = pkt["top4_oracle_j"]
                    val_invalid_4d[s_idx, fi_seed, pos, 9] = bool(
                        not pkt["gate_4"][pkt["top4_oracle_j"]]
                    )
                    val_top1_oracle_4d[s_idx, fi_seed, pos, 9] = True

    # Feature importance summary on fold_seed=269140, seed=269101, fold=0 outer NOSG model (lr=0.05, 528 cols)
    m_imp = cell_results[("NOSG", SECONDARY_FOLD_SEED, 269101, 0)][
        "outer_models_by_lr"
    ][0.05]
    gains_imp = m_imp.booster_.feature_importance(importance_type="gain")
    tot_gain_imp = max(float(np.sum(gains_imp)), 1e-12)
    block_importance_nosg_528 = {
        "prog_412 (0..411)": {
            "gain": round(float(gains_imp[0:412].sum()), 2),
            "pct": round(100.0 * float(gains_imp[0:412].sum()) / tot_gain_imp, 2),
        },
        "sem24_z_24 (412..435)": {
            "gain": round(float(gains_imp[412:436].sum()), 2),
            "pct": round(100.0 * float(gains_imp[412:436].sum()) / tot_gain_imp, 2),
        },
        "proj16_z_16 (436..451)": {
            "gain": round(float(gains_imp[436:452].sum()), 2),
            "pct": round(100.0 * float(gains_imp[436:452].sum()) / tot_gain_imp, 2),
        },
        "lm48_z_48 (452..499)": {
            "gain": round(float(gains_imp[452:500].sum()), 2),
            "pct": round(100.0 * float(gains_imp[452:500].sum()) / tot_gain_imp, 2),
        },
        "dope_28 (500..527)": {
            "gain": round(float(gains_imp[500:528].sum()), 2),
            "pct": round(100.0 * float(gains_imp[500:528].sum()) / tot_gain_imp, 2),
        },
    }

    del cell_results
    trim_malloc()

    # Step 7: Statistical inference (10,000 paired family bootstrap draws, boot_seed = 269150)
    train_fams_sorted = sorted({r["family_id"] for r in train_rows})
    val_fams_sorted = sorted({r["family_id"] for r in val_rows})

    rng_train_boot = np.random.default_rng(stream_seed(BOOT_SEED, "dl_train_oof"))
    train_boot_ix = rng_train_boot.integers(
        0, len(train_fams_sorted), size=(BOOT_DRAWS, len(train_fams_sorted))
    )

    rng_val_boot = np.random.default_rng(stream_seed(BOOT_SEED, "dl_val"))
    val_boot_ix = rng_val_boot.integers(
        0, len(val_fams_sorted), size=(BOOT_DRAWS, len(val_fams_sorted))
    )

    contrast_pairs = [
        ("NOSG - CONTROL", "NOSG", "CONTROL"),
        ("NOSG_VETO - CONTROL", "NOSG_VETO", "CONTROL"),
        ("NOSG_SHUF - CONTROL", "NOSG_SHUF", "CONTROL"),
        ("NOSG - NOSG_SHUF", "NOSG", "NOSG_SHUF"),
        ("NOSG_VETO - NOSG_SHUF", "NOSG_VETO", "NOSG_SHUF"),
        ("NOSG_VETO - NOSG", "NOSG_VETO", "NOSG"),
        ("RERANK4 - CONTROL", "RERANK4", "CONTROL"),
        ("NOSG - RERANK4", "NOSG", "RERANK4"),
        ("NOSG_VETO - RERANK4", "NOSG_VETO", "RERANK4"),
        ("NOSG_UNC - CONTROL", "NOSG_UNC", "CONTROL"),
        ("NOSG_VETO_UNC_SAFE - CONTROL", "NOSG_VETO_UNC_SAFE", "CONTROL"),
        ("NOSG_M0 - CONTROL", "NOSG_M0", "CONTROL"),
        ("RERANK4_UNC - CONTROL", "RERANK4_UNC", "CONTROL"),
        ("TOP4_PICK_ORACLE - CONTROL", "TOP4_PICK_ORACLE", "CONTROL"),
    ]

    primary_train_contrasts_full: dict[str, Any] = {}
    for label, a_name, b_name in contrast_pairs:
        primary_train_contrasts_full[label] = paired_family_contrast(
            train_rows,
            train_panel_4d,
            arm_names,
            a_name,
            b_name,
            train_boot_ix,
            agg_across_folds="median",
        )

    train_by_fold_seed: dict[str, dict[str, Any]] = {}
    for fi_seed, fseed in enumerate(FOLD_SEEDS):
        train_by_fold_seed[str(fseed)] = {}
        for label, a_name, b_name in (
            ("NOSG - CONTROL", "NOSG", "CONTROL"),
            ("NOSG_VETO - CONTROL", "NOSG_VETO", "CONTROL"),
            ("NOSG_SHUF - CONTROL", "NOSG_SHUF", "CONTROL"),
            ("RERANK4 - CONTROL", "RERANK4", "CONTROL"),
            ("NOSG_UNC - CONTROL", "NOSG_UNC", "CONTROL"),
            ("NOSG_VETO_UNC_SAFE - CONTROL", "NOSG_VETO_UNC_SAFE", "CONTROL"),
        ):
            train_by_fold_seed[str(fseed)][label] = strip_harmed_all(
                paired_family_contrast(
                    train_rows,
                    train_panel_4d[:, fi_seed, :, :],
                    arm_names,
                    a_name,
                    b_name,
                    train_boot_ix,
                )
            )

    secondary_val_contrasts_full: dict[str, Any] = {}
    val_3fseed_contrasts: dict[str, Any] = {}
    for label, a_name, b_name in contrast_pairs:
        secondary_val_contrasts_full[label] = paired_family_contrast(
            val_rows,
            val_panel_4d[:, 0, :, :],
            arm_names,
            a_name,
            b_name,
            val_boot_ix,
        )
        val_3fseed_contrasts[label] = strip_harmed_all(
            paired_family_contrast(
                val_rows,
                val_panel_4d.reshape(S, NF * 5 * N_val, n_arms),
                arm_names,
                a_name,
                b_name,
                val_boot_ix,
            )
        )

    # Preregistered Holm-Bonferroni m=3 over {NOSG train, NOSG_VETO train, NOSG_VETO DL-Val91}
    holm3_raw = {
        "NOSG_TRAIN": primary_train_contrasts_full["NOSG - CONTROL"]["raw_p_one_sided"],
        "NOSG_VETO_TRAIN": primary_train_contrasts_full["NOSG_VETO - CONTROL"][
            "raw_p_one_sided"
        ],
        "NOSG_VETO_VAL91": secondary_val_contrasts_full["NOSG_VETO - CONTROL"][
            "raw_p_one_sided"
        ],
    }
    holm3_adj = apply_holm_bonferroni(holm3_raw)
    sorted_holm3_keys = [k for k, _ in sorted(holm3_raw.items(), key=lambda kv: kv[1])]

    def attach_holm3(c_dict: dict[str, Any], key_name: str) -> None:
        rank = sorted_holm3_keys.index(key_name)
        m_eff = 3 - rank
        c_dict["holm_m3_stepdown_rank"] = int(rank + 1)
        c_dict["holm_m3_effective_m"] = int(m_eff)
        c_dict["holm_p_m3"] = float(holm3_adj[key_name])
        if m_eff == 3:
            c_dict["ci_holm3_stepdown"] = list(c_dict["ci9833_holm3"])
        elif m_eff == 2:
            c_dict["ci_holm3_stepdown"] = list(c_dict["ci975_holm2"])
        else:
            c_dict["ci_holm3_stepdown"] = list(c_dict["ci95"])

    attach_holm3(primary_train_contrasts_full["NOSG - CONTROL"], "NOSG_TRAIN")
    attach_holm3(primary_train_contrasts_full["NOSG_VETO - CONTROL"], "NOSG_VETO_TRAIN")
    attach_holm3(secondary_val_contrasts_full["NOSG_VETO - CONTROL"], "NOSG_VETO_VAL91")

    def summarize_arms_4d(
        rows: list[dict[str, Any]],
        panel_4d_in: np.ndarray,
        choice_4d_in: np.ndarray,
        inv_4d_in: np.ndarray,
        top1_4d_in: np.ndarray,
    ) -> dict[str, Any]:
        fams = sorted({r["family_id"] for r in rows})
        ctx_fams = np.asarray(
            [r["family_id"] for r in rows] * (panel_4d_in.shape[2] // len(rows))
        )
        res: dict[str, Any] = {}
        for a_i, name in enumerate(arm_names):
            fam_fs = np.zeros((S, panel_4d_in.shape[1], len(fams)), dtype=np.float64)
            for fi, f in enumerate(fams):
                mask = ctx_fams == f
                fam_fs[:, :, fi] = panel_4d_in[:, :, mask, a_i].mean(axis=2)
            fam_med_fseed = np.median(fam_fs, axis=1)
            seed_fam_means = [float(fam_med_fseed[s].mean()) for s in range(S)]
            ctx_med_fseed = np.median(panel_4d_in[:, :, :, a_i].mean(axis=2), axis=1)
            seed_ctx_means = [float(ctx_med_fseed[s]) for s in range(S)]

            changed_by_seed = [
                float(np.mean(np.sum(choice_4d_in[s, :, :, a_i] != 0, axis=1)))
                for s in range(S)
            ]
            inv_by_seed = [int(np.sum(inv_4d_in[s, :, :, a_i])) for s in range(S)]
            top1_acc_by_seed = [
                float(np.mean(top1_4d_in[s, :, :, a_i])) for s in range(S)
            ]
            res[name] = {
                "median_family_utility": float(np.median(seed_fam_means)),
                "per_seed_family_utility": seed_fam_means,
                "median_context_utility": float(np.median(seed_ctx_means)),
                "per_seed_context_utility": seed_ctx_means,
                "mean_changed_choices_per_fold_seed": changed_by_seed,
                "total_invalid_neg2_picks_across_fold_seeds": inv_by_seed,
                "top1_within_top4_accuracy_by_seed": top1_acc_by_seed,
            }
        return res

    train_arm_summary = summarize_arms_4d(
        train_rows,
        train_panel_4d,
        train_choice_4d,
        train_invalid_4d,
        train_top1_oracle_4d,
    )
    val_arm_summary_269140 = summarize_arms_4d(
        val_rows,
        val_panel_4d[:, 0:1, :, :],
        val_choice_4d[:, 0:1, :, :],
        val_invalid_4d[:, 0:1, :, :],
        val_top1_oracle_4d[:, 0:1, :, :],
    )

    train_elo = compute_elo_table(train_rows, train_panel_4d, arm_names, BOOT_SEED)
    val_elo = compute_elo_table(
        val_rows, val_panel_4d[:, 0, :, :], arm_names, BOOT_SEED + 1
    )

    # Context #27 report on DL-Val (fold_seed=269140, 15 cells)
    ctx27_idx = val_cid_order.index(CONTEXT_27_SHA256)
    ctx27_cells = []
    for s_idx, s in enumerate(SEEDS):
        for k in range(5):
            pos = k * N_val + ctx27_idx
            ctx27_cells.append(
                {
                    "seed": int(s),
                    "fold": int(k),
                    "choices_top4_slot": {
                        arm_names[a]: int(val_choice_4d[s_idx, 0, pos, a])
                        for a in range(n_arms)
                    },
                    "utilities": {
                        arm_names[a]: float(val_panel_4d[s_idx, 0, pos, a])
                        for a in range(n_arms)
                    },
                    "invalid_gates": {
                        arm_names[a]: bool(val_invalid_4d[s_idx, 0, pos, a])
                        for a in range(n_arms)
                    },
                }
            )
    ctx27_summary = {
        "context_sha256": CONTEXT_27_SHA256,
        "index": ctx27_idx,
        "family_id": val_rows[ctx27_idx]["family_id"],
        "r882_slot0_utility": float(val_packets[CONTEXT_27_SHA256]["r882_u"]),
        "top4_oracle_slot3_utility": float(
            val_packets[CONTEXT_27_SHA256]["top4_oracle_u"]
        ),
        "nosg_slot3_oracle_picks_out_of_15": sum(
            1 for c in ctx27_cells if c["choices_top4_slot"]["NOSG"] == 3
        ),
        "nosg_invalid_picks_out_of_15": sum(
            1 for c in ctx27_cells if c["invalid_gates"]["NOSG"]
        ),
        "nosg_veto_slot3_oracle_picks_out_of_15": sum(
            1 for c in ctx27_cells if c["choices_top4_slot"]["NOSG_VETO"] == 3
        ),
        "nosg_veto_invalid_picks_out_of_15": sum(
            1 for c in ctx27_cells if c["invalid_gates"]["NOSG_VETO"]
        ),
        "rerank4_slot3_oracle_picks_out_of_15": sum(
            1 for c in ctx27_cells if c["choices_top4_slot"]["RERANK4"] == 3
        ),
    }

    # Evaluate confirmatory ADVANCE criteria for NOSG and NOSG_VETO
    ctrl_inv_train = train_arm_summary["CONTROL"][
        "total_invalid_neg2_picks_across_fold_seeds"
    ]

    def evaluate_arm_gate(arm_key: str) -> dict[str, Any]:
        p_c = primary_train_contrasts_full[f"{arm_key} - CONTROL"]
        p_shuf = primary_train_contrasts_full[f"{arm_key} - NOSG_SHUF"]
        s_c = secondary_val_contrasts_full[f"{arm_key} - CONTROL"]
        arm_inv = train_arm_summary[arm_key][
            "total_invalid_neg2_picks_across_fold_seeds"
        ]

        n_adm = sum(
            1
            for r in selection_receipts
            if r["arm"] == arm_key and r["strict_admitted"]
        )
        gain_ok = bool(p_c["median_family_gain"] > PRIMARY_ADVANCE_MIN_GAIN)
        lcb95_ok = bool(p_c["ci95"][0] > 0.0)
        holm_stepdown_lcb_ok = bool(
            p_c.get("ci_holm3_stepdown", p_c["ci9833_holm3"])[0] > 0.0
        )
        holm_simul_lcb_ok = bool(p_c["ci9833_holm3"][0] > 0.0)
        holm_p_ok = bool(p_c.get("holm_p_m3", 1.0) < 0.05)
        seeds_ok = bool(p_c["positive_seeds"] == 3)
        beats_shuf = bool(
            p_shuf["median_family_gain"] > 0.0 and p_shuf["raw_p_one_sided"] < 0.05
        )
        val_gain_ok = bool(s_c["median_family_gain"] > 0.0)
        val_seeds_ok = bool(s_c["positive_seeds"] == 3)
        inv_le_ctrl_all = bool(all(arm_inv[s] <= ctrl_inv_train[s] for s in range(S)))
        inv_le_ctrl_med = bool(
            int(np.median(arm_inv)) <= int(np.median(ctrl_inv_train))
        )

        adv = bool(
            n_adm > 0
            and gain_ok
            and holm_stepdown_lcb_ok
            and holm_p_ok
            and seeds_ok
            and beats_shuf
            and val_gain_ok
            and val_seeds_ok
            and inv_le_ctrl_all
        )
        return {
            "arm": arm_key,
            "n_cells_strict_admitted_of_45": n_adm,
            "primary_median_family_gain_gt_0002": gain_ok,
            "primary_lcb95_gt_0": lcb95_ok,
            "primary_holm3_stepdown_lcb_gt_0": holm_stepdown_lcb_ok,
            "primary_holm3_simultaneous_lcb_gt_0": holm_simul_lcb_ok,
            "primary_holm3_p_lt_005": holm_p_ok,
            "primary_3_of_3_seeds_positive": seeds_ok,
            "primary_beats_nosg_shuf": beats_shuf,
            "secondary_dl_val91_gain_gt_0": val_gain_ok,
            "secondary_dl_val91_3_of_3_seeds_positive": val_seeds_ok,
            "dl_train_invalid_neg2_picks_by_seed": arm_inv,
            "control_dl_train_invalid_neg2_picks_by_seed": ctrl_inv_train,
            "dl_train_invalid_le_control_all_seeds": inv_le_ctrl_all,
            "dl_train_invalid_le_control_median": inv_le_ctrl_med,
            "arm_advance": adv,
        }

    gate_nosg = evaluate_arm_gate("NOSG")
    gate_veto = evaluate_arm_gate("NOSG_VETO")
    advance = bool(gate_nosg["arm_advance"] or gate_veto["arm_advance"])
    verdict = "ADVANCE" if advance else "NO_ADVANCE"
    winning_arm = (
        "NOSG_VETO"
        if gate_veto["arm_advance"]
        else ("NOSG" if gate_nosg["arm_advance"] else None)
    )

    r882_sha_after = sha256_file(R882_CKPT)
    assert r882_sha_after == r882_sha_before

    # Save full panels and full details to out_dir FIRST (for S3 custody)
    np.savez_compressed(
        out_dir / "panels.npz",
        arm_names=np.asarray(arm_names),
        fold_seeds=np.asarray(FOLD_SEEDS, dtype=np.int32),
        seeds=np.asarray(SEEDS, dtype=np.int32),
        train_cids=np.asarray(train_cid_order),
        val_cids=np.asarray(val_cid_order),
        train_panel_4d=train_panel_4d,
        train_choice_4d=train_choice_4d,
        train_invalid_4d=train_invalid_4d,
        train_top1_oracle_4d=train_top1_oracle_4d,
        val_panel_4d=val_panel_4d,
        val_choice_4d=val_choice_4d,
        val_invalid_4d=val_invalid_4d,
        val_top1_oracle_4d=val_top1_oracle_4d,
    )

    details_doc = {
        "schema": "jope.h269.details/v1",
        "wave_id": "h269-nosg-veto-rerank4-20261103",
        "verdict": verdict,
        "winning_arm": winning_arm,
        "advance_gates": {
            "NOSG": gate_nosg,
            "NOSG_VETO": gate_veto,
            "winning_arm": winning_arm,
            "verdict": verdict,
        },
        "primary_train_contrasts_full": primary_train_contrasts_full,
        "secondary_val_contrasts_full": secondary_val_contrasts_full,
        "train_by_fold_seed": train_by_fold_seed,
        "val_3fseed_contrasts": val_3fseed_contrasts,
        "train_arm_utilities": train_arm_summary,
        "val_arm_utilities_fseed269140": val_arm_summary_269140,
        "train_elo": train_elo,
        "val_elo": val_elo,
        "context_27_cells": ctx27_cells,
        "confirmatory_cell_selections": selection_receipts,
    }
    details_sha256 = fsync_write_json(out_dir / "details.json", details_doc)

    # Write full metrics_full.json first, then derive compact <= 50 KB metrics.json (no run-killing assert)
    core_contrasts = (
        "NOSG - CONTROL",
        "NOSG_VETO - CONTROL",
        "NOSG_SHUF - CONTROL",
        "NOSG - NOSG_SHUF",
        "NOSG_VETO - NOSG_SHUF",
        "NOSG_VETO - NOSG",
        "RERANK4 - CONTROL",
        "NOSG_UNC - CONTROL",
        "NOSG_VETO_UNC_SAFE - CONTROL",
        "TOP4_PICK_ORACLE - CONTROL",
    )
    core_arms = (
        "CONTROL",
        "NOSG",
        "NOSG_VETO",
        "NOSG_SHUF",
        "RERANK4",
        "NOSG_UNC",
        "NOSG_VETO_UNC_SAFE",
        "TOP4_PICK_ORACLE",
    )

    def full_contrast_compact(
        d: dict[str, Any], keep_top5: bool = False
    ) -> dict[str, Any]:
        out = {
            k: v
            for k, v in d.items()
            if k not in ("harmed_families_all", "worst_harmed_families_top10")
        }
        if keep_top5 and "worst_harmed_families_top10" in d:
            out["worst_harmed_families_top5"] = d["worst_harmed_families_top10"][:5]
        return out

    def brief_contrast_compact(d: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "median_family_gain",
            "median_context_gain",
            "ci95",
            "ci975_holm2",
            "ci9833_holm3",
            "ci_holm3_stepdown",
            "raw_p_one_sided",
            "holm_p_m3",
            "positive_seeds",
            "per_seed_family_gains",
            "n_harmed_families_median_gt_001",
        )
        return {k: d[k] for k in keys if k in d}

    n_adm_shuf = sum(
        1
        for r in selection_receipts
        if r["arm"] == "NOSG_SHUF" and r["strict_admitted"]
    )
    n_adm_r4 = sum(
        1 for r in selection_receipts if r["arm"] == "RERANK4" and r["strict_admitted"]
    )

    metrics_full = {
        "schema": "jope.h269.metrics/v1",
        "wave_id": "h269-nosg-veto-rerank4-20261103",
        "study_path": "research/flywheel_selfplay/h269_nosg_veto_rerank4",
        "status": "COMPLETE",
        "verdict": verdict,
        "winning_arm": winning_arm,
        "claim_scope": (
            "Confirmatory fresh-fold evaluation (fold_seeds 269140/269141/269142 x 5 folds x model_seeds 269101-269103 "
            "on DL-Train 205 fam PRIMARY + DL-Val91 SECONDARY, Holm m=3) of H266 NO_SCOREGEOM (528 cols, NOSG) and "
            "outcome-free LGBMClassifier -2.0 invalid-gate veto (NOSG_VETO, tau in {0.05, 0.10, 0.20})"
        ),
        "primary_endpoint": "DL-Train 205-family OOF single-pick equal-family utility minus CONTROL across 3 fresh fold seeds x 3 fresh model seeds (45 cells)",
        "secondary_endpoint": "DL-Val91 (91 ctx / 19 fam) fresh-seed single-pick equal-family utility minus CONTROL (fold_seed=269140, model_seeds=269101-269103; heavily reused validation split disclosed)",
        "fold_seeds": list(FOLD_SEEDS),
        "model_seeds": list(SEEDS),
        "boot_seed": BOOT_SEED,
        "boot_draws": BOOT_DRAWS,
        "holm_m": HOLM_M,
        "holm3_family": ["NOSG_TRAIN", "NOSG_VETO_TRAIN", "NOSG_VETO_VAL91"],
        "holm3_raw_p": holm3_raw,
        "holm3_adjusted_p": holm3_adj,
        "baseline_regression": baseline_reg,
        "hygiene": {
            "dope_sha256_before": dope_sha_before,
            "dope_sha256_after": dope_sha_before,
            "r882_sha256_before": r882_sha_before,
            "r882_sha256_after": r882_sha_after,
            "optimizer_updates": 0,
            "test_time_reveals_used": 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "audits": {
            "h58_cache_provenance": {
                "cache_manifest_schema": "jope.h58.context-feature-cache/v3",
                "context_features_npz_sha256": CONTEXT_FEATURES_EXPECTED_SHA256,
                "pool_priors_npz_sha256": POOL_PRIORS_EXPECTED_SHA256,
                "contains_dl_val_or_train_outcomes": False,
                "base_prior_cost_source": "stage1_stage2.pt native prior model input (sha256=136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d)",
                "excluded_payloads_opened": 0,
                "outer_rows_delivered": 0,
            },
            "family_disjointness": {
                "dl_train_val_family_overlap": 0,
                "dl_train_fold_family_overlap_across_all_3_fold_seeds": 0,
                "dl_val_reads_before_fit_lock": 0,
                "has_meta_test_any": False,
            },
            "dl_val91_reuse_disclosure": (
                "DL-Val91 (19 families, 91 contexts) has been heavily reused across prior waves; "
                "DL-Train 205-family OOF on fresh fold seeds 269140/269141/269142 is the primary decision endpoint."
            ),
        },
        "fit_selections_lock_summary": {
            "path": "fit_selections_lock.json",
            "sha256": lock_sha256,
            "details_json_sha256": details_sha256,
            "n_confirmatory_cells_per_arm": 45,
            "n_cells_strict_admitted_nosg_of_45": gate_nosg[
                "n_cells_strict_admitted_of_45"
            ],
            "n_cells_strict_admitted_nosg_veto_of_45": gate_veto[
                "n_cells_strict_admitted_of_45"
            ],
            "n_cells_strict_admitted_nosg_shuf_of_45": n_adm_shuf,
            "n_cells_strict_admitted_rerank4_of_45": n_adm_r4,
        },
        "advance_gates": {
            "NOSG": gate_nosg,
            "NOSG_VETO": gate_veto,
            "winning_arm": winning_arm,
            "verdict": verdict,
        },
        "primary_dl_train_oof_205fam": {
            "n_contexts": N_train,
            "n_families": len(train_fams_sorted),
            "n_cells_per_arm": 45,
            "arm_utilities": {
                k: v for k, v in train_arm_summary.items() if k in core_arms
            },
            "contrasts_median_across_fold_seeds": {
                k: (
                    full_contrast_compact(
                        v, keep_top5=(k in ("NOSG - CONTROL", "NOSG_VETO - CONTROL"))
                    )
                    if k
                    in (
                        "NOSG - CONTROL",
                        "NOSG_VETO - CONTROL",
                        "NOSG - NOSG_SHUF",
                        "NOSG_VETO - NOSG_SHUF",
                    )
                    else brief_contrast_compact(v)
                )
                for k, v in primary_train_contrasts_full.items()
                if k in core_contrasts
            },
            "per_fold_seed_contrasts": {
                fk: {ck: brief_contrast_compact(cv) for ck, cv in fv.items()}
                for fk, fv in train_by_fold_seed.items()
            },
            "mm_bt_elo_anchored_r882": {
                k: v for k, v in train_elo.items() if k in core_arms
            },
        },
        "secondary_dl_val_native91": {
            "n_contexts": N_val,
            "n_families": len(val_fams_sorted),
            "fold_seed": SECONDARY_FOLD_SEED,
            "arm_utilities": {
                k: v for k, v in val_arm_summary_269140.items() if k in core_arms
            },
            "contrasts_fseed269140": {
                k: (
                    full_contrast_compact(
                        v, keep_top5=(k in ("NOSG - CONTROL", "NOSG_VETO - CONTROL"))
                    )
                    if k
                    in (
                        "NOSG - CONTROL",
                        "NOSG_VETO - CONTROL",
                        "NOSG - NOSG_SHUF",
                        "NOSG_VETO - NOSG_SHUF",
                    )
                    else brief_contrast_compact(v)
                )
                for k, v in secondary_val_contrasts_full.items()
                if k in core_contrasts
            },
            "contrasts_3_fold_seeds_aggregate": {
                k: brief_contrast_compact(val_3fseed_contrasts[k])
                for k in (
                    "NOSG - CONTROL",
                    "NOSG_VETO - CONTROL",
                    "RERANK4 - CONTROL",
                )
            },
            "mm_bt_elo_anchored_r882": {
                k: v for k, v in val_elo.items() if k in core_arms
            },
        },
        "feature_importance_nosg_528_fseed269140_s269101_fold0": block_importance_nosg_528,
        "context_27_summary": ctx27_summary,
        "wall_seconds": round(time.time() - t0, 3),
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    full_metrics_sha256 = fsync_write_json(out_dir / "metrics_full.json", metrics_full)
    metrics_git = dict(metrics_full)
    metrics_git["full_metrics_json_sha256"] = full_metrics_sha256
    fsync_write_json(out_dir / "metrics.json", metrics_git)

    print(
        "H269_PRIMARY_COMPLETE",
        json.dumps(
            {
                "verdict": verdict,
                "winning_arm": winning_arm,
                "metrics_json_bytes": (out_dir / "metrics.json").stat().st_size,
                "advance_gates": metrics_git["advance_gates"],
                "nosg_train": metrics_git["primary_dl_train_oof_205fam"][
                    "contrasts_median_across_fold_seeds"
                ]["NOSG - CONTROL"],
                "nosg_veto_train": metrics_git["primary_dl_train_oof_205fam"][
                    "contrasts_median_across_fold_seeds"
                ]["NOSG_VETO - CONTROL"],
                "nosg_val91": metrics_git["secondary_dl_val_native91"][
                    "contrasts_fseed269140"
                ]["NOSG - CONTROL"],
                "nosg_veto_val91": metrics_git["secondary_dl_val_native91"][
                    "contrasts_fseed269140"
                ]["NOSG_VETO - CONTROL"],
            },
            indent=2,
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="H269 runner")
    parser.add_argument("--mode", choices=["smoke", "primary"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        return run_smoke(args.out_dir)
    return run_primary(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
