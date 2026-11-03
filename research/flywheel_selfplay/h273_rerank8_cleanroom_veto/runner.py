#!/usr/bin/env python3
"""
H273 (`h273_rerank8_cleanroom_veto`) — Clean-Room Independent-Lane Replication of H270 `RERANK8`
Plus Outcome-Free `LGBMClassifier` `-2.0` Invalid-Gate Veto (`RERANK8_CR_VETO`) and
Unconditional Shuffle Placebo (`RERANK8_CR_SHUF_UNC`).

Clean-Room Provenance:
  - Written from `H270` `PREREG.md` / `README.md` text specification ONLY, generalizing Argon's
    `H266`/`H269` `K=4` pipeline to `K=8` (`568` feature columns).
  - Does NOT import or copy `topk.py`, `h265_core.py`, or `h270/runner.py`.
  - Zero neural weight updates (`0` optimizer updates, read-only `r882` copy), `0` test-time reveals,
    scored SINGLE-PICK (`1` pick per context from `r882`'s own `top-8` pool).

Two-Stage Design:
  - Stage 1 Reproduction Gate (before fresh seeds): Runs `RERANK8_CR` at `H270` seeds
    (`fold_seeds = 270140, 270141, 270142`, `model_seeds = 270101, 270102, 270103`, `boot_seed = 270150`)
    and reports per-cell pick agreement (`% identical` over all `45` cells / `10,971` cell-context picks)
    and `PRIMARY` median family gain vs `H270`.
  - Stage 2 Fresh-Seed Confirmatory Evaluation: Runs `CONTROL`, `RERANK8_CR`, `RERANK8_CR_SHUF`,
    `RERANK8_CR_UNC`, `RERANK8_CR_SHUF_UNC` (`+ M0`), and `RERANK8_CR_VETO` (`tau = 0.10`) on fresh
    `fold_seeds = 273140, 273141, 273142` x `5` family-grouped folds x `model_seeds = 273101, 273102, 273103`
    (`45` cells/arm, `boot_seed = 273150`, `Holm m = 2` over `{PRIMARY, SECONDARY}`).
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
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

EVIDENCE_ROOT = Path("/home/ubuntu/jope_plan/evidence/H273/argon-t1925")
INPUTS_DIR = EVIDENCE_ROOT / "inputs"
H270_REF_DIR = EVIDENCE_ROOT / "h270_s3_ref" / "attempt-001"
CAPSULE_STUDY_DIR = (
    EVIDENCE_ROOT
    / "capsule"
    / "research"
    / "flywheel_selfplay"
    / "h273_rerank8_cleanroom_veto"
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

STAGE1_FOLD_SEEDS = (270140, 270141, 270142)
STAGE1_SEEDS = (270101, 270102, 270103)
STAGE1_BOOT_SEED = 270150

FOLD_SEEDS = (273140, 273141, 273142)
SECONDARY_FOLD_SEED = 273140
SEEDS = (273101, 273102, 273103)
BOOT_SEED = 273150
BOOT_DRAWS = 10_000
INNER_BOOT_DRAWS = 2_000
HOLM_M = 2

SLICE_K = 8
FEATURE_DIM_K8 = 568
FEATURE_DIM_K4 = 560

LR_GRID = (0.03, 0.05, 0.10)
N_EST_GRID = (50, 100, 200)
FINITE_MARGIN_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
MARGIN_GRID = (*FINITE_MARGIN_GRID, math.inf)
FROZEN_VETO_TAU = 0.10

STRICT_FIT_MIN_GAIN = 0.001 + 1e-12
PRIMARY_ADVANCE_MIN_GAIN = 0.002 + 1e-12
TIE_MARGIN = 1e-4

# Frozen LGBMClassifier hyperparameters from H269 for outcome-free -2.0 invalid-gate prediction on inner FIT
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
    "n_jobs": 2,
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
    pause_file = os.environ.get("H273_PAUSE_FILE", "")
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


def compact_json_text(obj: Any) -> str:
    raw_pretty = json.dumps(obj, indent=2, sort_keys=True)
    comp = re.sub(
        r"\[\n\s+([^\[\]\{\}]+?)\n\s+\]",
        lambda m: "[" + " ".join(x.strip() for x in m.group(1).splitlines()) + "]",
        raw_pretty,
    )
    comp = re.sub(
        r"\{\n\s+([^\[\]\{\}]+?)\n\s+\}",
        lambda m: "{" + " ".join(x.strip() for x in m.group(1).splitlines()) + "}",
        comp,
    )
    return comp + "\n"


def fsync_write_json(path: Path, obj: Any, compact_leaves: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        compact_json_text(obj)
        if compact_leaves
        else (json.dumps(obj, indent=2, sort_keys=True) + "\n")
    )
    raw = text.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
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


def compute_true_topk_relevance(
    fitness_k: np.ndarray, gate_k: np.ndarray
) -> np.ndarray:
    """
    Clean-room K-slice relevance labels (from H270 PREREG.md text specification):
      - Gate-failed candidates (`gate == False`, utility = -2.0) receive `0` (worst).
      - Valid candidates (`gate == True`) are ranked by ascending true revealed cost (`fitness`),
        with ties broken by ascending `r882` slot `j in 0..K-1`; best valid receives `K - 1`,
        next `K - 2`, ..., down to `0`.
    """
    k = int(len(fitness_k))
    rel = np.zeros(k, dtype=np.int32)
    valid_js = [j for j in range(k) if bool(gate_k[j]) and np.isfinite(fitness_k[j])]
    valid_sorted = sorted(valid_js, key=lambda j: (float(fitness_k[j]), int(j)))
    for rank_pos, j in enumerate(valid_sorted):
        rel[j] = max((k - 1) - rank_pos, 0)
    return rel


def compute_sem24_numpy(prog_412: np.ndarray) -> np.ndarray:
    """Vectorized 24D structural semantic signature."""
    x = np.asarray(prog_412, dtype=np.float32)
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, 412)
    padded = np.pad(flat, ((0, 0), (0, 20)), mode="constant").reshape(-1, 18, 24)
    sig = padded.mean(axis=1) + 0.5 * padded.max(axis=1)
    norms = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
    return (sig / norms).reshape(*orig_shape, 24).astype(np.float32)


def build_topk_feature_matrix_for_context(
    rcost: np.ndarray,
    aprior: np.ndarray,
    prog_full: np.ndarray,
    sem24_full: np.ndarray,
    proj16_full: np.ndarray,
    lm48_z: np.ndarray,
    dope629: np.ndarray,
    k: int = SLICE_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean-room K-slice feature builder (568 cols at K=8, 560 cols at K=4), written from
    H270 PREREG.md text specification:
      - Slice is `r882`'s `K` lowest-cost candidates (`np.argsort(rcost, kind='mergesort')[:K]`).
      - Per-context z-features (`slice z`, `slice softmax`, `slice spread = (c[K-1] - c[0]) / r_sd`)
        are computed within the `K`-slice.
      - Per-pair raw and z cost differences cover the other `K - 1` slots (`14` cols at `K=8` vs `6` at `K=4`).
      - Slot position is `j / (K - 1)`.
      - Fixed context descriptors (`top-8 sem24 centroid` and `pool rows 0..3`, `lm48_z`, `dope629`) unchanged.
    """
    assert len(rcost) >= k, f"Pool size {len(rcost)} < k={k}"
    rcost64 = rcost.astype(np.float64)
    order = np.argsort(rcost64, kind="mergesort")
    topk = order[:k]
    top8 = order[:8]
    prog_topk = prog_full[topk]

    r_mean = float(np.mean(rcost64))
    r_sd = max(float(np.std(rcost64)), 1e-12)
    r_z_all = (rcost64 - r_mean) / r_sd

    ck = rcost64[topk]
    ck_mean = float(np.mean(ck))
    ck_sd = max(float(np.std(ck)), 1e-6)
    ck_z = (ck - ck_mean) / ck_sd

    zk = r_z_all[topk]
    sm_tau1 = np.exp(-(zk - np.min(zk)))
    sm_tau1 /= np.sum(sm_tau1)
    sm_tau02 = np.exp(-(zk - np.min(zk)) / 0.2)
    sm_tau02 /= np.sum(sm_tau02)
    sm_ck = np.exp(-(ck_z - np.min(ck_z)))
    sm_ck /= np.sum(sm_ck)

    ap64 = aprior.astype(np.float64)
    ap_z_all = (ap64 - float(np.mean(ap64))) / max(float(np.std(ap64)), 1e-6)

    ctx_top2_zgap = float((ck[1] - ck[0]) / r_sd)
    ctx_topk_spread_z = float((ck[k - 1] - ck[0]) / r_sd)

    sem64 = sem24_full.astype(np.float64)
    sem_mean = sem64.mean(axis=0, keepdims=True)
    sem_std = sem64.std(axis=0, keepdims=True) + 1e-5
    sem_z_topk = np.clip((sem64[topk] - sem_mean) / sem_std, -5.0, 5.0).astype(
        np.float32
    )

    proj64 = proj16_full.astype(np.float64)
    proj_mean = proj64.mean(axis=0, keepdims=True)
    proj_std = proj64.std(axis=0, keepdims=True) + 1e-5
    proj_z_topk = np.clip((proj64[topk] - proj_mean) / proj_std, -5.0, 5.0).astype(
        np.float32
    )
    proj_unit_topk = proj_z_topk / np.maximum(
        np.linalg.norm(proj_z_topk, axis=1, keepdims=True), 1e-6
    )

    sem_topk = sem64[topk]
    sem_top8_centroid = sem64[top8].mean(axis=0)
    sem_top8_centroid /= max(float(np.linalg.norm(sem_top8_centroid)), 1e-6)
    sem_rev03 = sem64[:4]

    est_slice = prog_topk[:, 24:29]
    est_ids = np.where(
        est_slice.max(axis=1) > 0, np.argmax(est_slice, axis=1), -1
    ).astype(np.int32)
    op_counts = prog_topk[:, :23].sum(axis=1).astype(np.float64)

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

    feat_dim = 554 + 2 * (k - 1)
    rows_out = np.zeros((k, feat_dim), dtype=np.float32)
    denom_pos = float(max(k - 1, 1))
    for j in range(k):
        others = [m for m in range(k) if m != j]
        raw_diffs = [float(ck[j] - ck[m]) for m in others]
        z_diffs = [float((ck[j] - ck[m]) / r_sd) for m in others]
        block1 = np.array(
            [
                float(ck[j]),
                float(zk[j]),
                float(ck_z[j]),
                float(j),
                float(j) / denom_pos,
                1.0 if j == 0 else 0.0,
                float(ck[j] - ck[0]),
                0.0 if j == 0 else float((ck[j] - ck[0]) / r_sd),
                ctx_top2_zgap,
                ctx_topk_spread_z,
                float(sm_tau1[j]),
                float(sm_tau02[j]),
                float(sm_ck[j]),
                float(ap_z_all[topk[j]]),
                *raw_diffs,
                *z_diffs,
            ],
            dtype=np.float32,
        )

        cos_sem_other = [
            float(np.clip(np.dot(sem_topk[j], sem_topk[m]), -1.0, 1.0)) for m in others
        ]
        cos_proj_other = [
            float(np.clip(np.dot(proj_unit_topk[j], proj_unit_topk[m]), -1.0, 1.0))
            for m in others
        ]
        cos_sem_r1 = float(np.clip(np.dot(sem_topk[j], sem_topk[0]), -1.0, 1.0))
        cos_proj_r1 = float(
            np.clip(np.dot(proj_unit_topk[j], proj_unit_topk[0]), -1.0, 1.0)
        )
        cos_top8 = float(np.clip(np.dot(sem_topk[j], sem_top8_centroid), -1.0, 1.0))
        cos_rev4_mean = float(np.mean(np.clip(sem_rev03 @ sem_topk[j], -1.0, 1.0)))
        same_est_r1 = 1.0 if est_ids[j] == est_ids[0] else 0.0
        n_same_est = float(sum(1 for m in range(k) if est_ids[m] == est_ids[j]))

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
                prog_topk[j].astype(np.float32),
                sem_z_topk[j],
                proj_z_topk[j],
                ctx_vec76,
            ],
            axis=0,
        )
    return topk, rows_out


def select_from_topk_scores(scores_k: np.ndarray, margin: float) -> int:
    if math.isinf(margin):
        return 0
    best_j = int(np.argmax(scores_k))
    if best_j == 0:
        return 0
    if float(scores_k[best_j] - scores_k[0]) > float(margin):
        return best_j
    return 0


def select_from_topk_scores_with_veto(
    scores_k: np.ndarray,
    p_inv_k: np.ndarray,
    tau: float,
    margin: float,
) -> int:
    if math.isinf(margin):
        return 0
    best_j = int(np.argmax(scores_k))
    if best_j == 0:
        return 0
    if float(scores_k[best_j] - scores_k[0]) > float(margin):
        if float(p_inv_k[best_j]) > float(tau):
            return 0
        return best_j
    return 0


def shuffle_labels_within_family(
    rel_by_cid: dict[str, np.ndarray],
    cids: list[str],
    fam_by_cid: dict[str, str],
    seed: int,
    k: int = SLICE_K,
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
            out[c] = flat_shuf[idx * k : (idx + 1) * k].copy()
    return out


def load_cohort_topk_packets(
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
            assert len(fitness) >= SLICE_K
            sem24 = compute_sem24_numpy(prog)

            utility = np.where(gate, -fitness, -2.0)
            rcost = costs_by_cid[cid]
            top8, x_top8 = build_topk_feature_matrix_for_context(
                rcost=rcost,
                aprior=aprior_by_cid[cid],
                prog_full=prog,
                sem24_full=sem24,
                proj16_full=proj16_by_cid[cid],
                lm48_z=ctx_feats_by_cid[cid]["lm48_z"],
                dope629=ctx_feats_by_cid[cid]["dope629"],
                k=SLICE_K,
            )
            fit8 = fitness[top8]
            gate8 = gate[top8]
            u8 = utility[top8]
            rel8 = compute_true_topk_relevance(fit8, gate8)

            gated_cost8 = np.where(gate8, fit8, 2.0)
            oracle8_j = int(np.argmin(gated_cost8))
            gated_cost4 = gated_cost8[:4]
            oracle4_j = int(np.argmin(gated_cost4))
            gated_cost_pool = np.where(gate, fitness, 2.0)
            oracle_pool_idx = int(np.argmin(gated_cost_pool))

            packets[cid] = {
                "cid": cid,
                "family_id": r["family_id"],
                "pool_size": int(len(fitness)),
                "top8_indices": top8,
                "X_top8": x_top8,
                "fitness_8": fit8,
                "gate_8": gate8,
                "utility_8": u8,
                "rel_8": rel8,
                "r882_u": float(u8[0]),
                "r882_invalid": bool(not gate8[0]),
                "top4_oracle_j": oracle4_j,
                "top4_oracle_u": float(u8[oracle4_j]),
                "top8_oracle_j": oracle8_j,
                "top8_oracle_u": float(u8[oracle8_j]),
                "pool_oracle_u": float(utility[oracle_pool_idx]),
                "n_invalid_in_top8": int(np.sum(~gate8)),
            }
        trim_malloc()
    return packets


def fit_lgbm_ranker_200(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    n_groups: int,
    lr: float,
    random_state: int,
    k: int = SLICE_K,
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=int(k),
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
        n_jobs=2,
        random_state=int(random_state),
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    groups = np.full(n_groups, int(k), dtype=np.int32)
    model.fit(X_fit, y_fit, group=groups, eval_at=[1, int(k)])
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
    params = dict(VETO_CLF_PARAMS)
    params["random_state"] = int(random_state)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_fit, y_inv_fit)
    return clf


def predict_veto_classifier(clf: lgb.LGBMClassifier, X: np.ndarray) -> np.ndarray:
    probs = clf.booster_.predict(X)
    return np.asarray(probs, dtype=np.float64)


def run_cell_fit_and_select_k8(
    fold_seed: int,
    seed: int,
    assess_fold: int,
    arm_name: str,
    shuffle_placebo: bool,
    train_cids_by_fold: dict[int, list[str]],
    packets: dict[str, dict[str, Any]],
    fam_by_cid: dict[str, str],
    with_veto: bool = False,
) -> dict[str, Any]:
    """
    Runs 4-fold inner family-grouped CV on the 164 FIT families for `(fold_seed, seed, assess_fold)`,
    selecting:
      1. Strict `H257` inner-FIT setting (`gain > +0.001` and `LCB95 > 0`, else fallback `margin = inf`).
      2. Unconditional setting (`UNC`, fallback disabled: best finite margin in `FINITE_MARGIN_GRID`,
         plus pure `margin = 0.0` top-1 argmax for shuffled placebo diagnostics).
      3. When `with_veto == True`, also trains the frozen `H269` `LGBMClassifier` invalid-gate veto
         model on inner FIT and selects `(lr*, n_est*, margin*)` at fixed `tau = 0.10` subject to
         inner-FIT invalid-gate non-regression (`<= CONTROL`) and strict `H257` admission.
    All random streams mix `fold_seed` per H270 preregistration.
    """
    fit_folds = [f for f in range(5) if f != assess_fold]
    fit_cids = [c for f in fit_folds for c in train_cids_by_fold[f]]
    fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
    assert len(fit_fams) == 164

    if shuffle_placebo:
        base_rel = {c: packets[c]["rel_8"] for c in fit_cids}
        rel_map = shuffle_labels_within_family(
            base_rel,
            fit_cids,
            fam_by_cid,
            stream_seed(fold_seed, seed, assess_fold, "shuf"),
            k=SLICE_K,
        )
    else:
        rel_map = {c: packets[c]["rel_8"] for c in fit_cids}

    inner_oof_scores: dict[tuple[float, int], dict[str, np.ndarray]] = {
        (lr, n_est): {} for lr in LR_GRID for n_est in N_EST_GRID
    }
    inner_oof_p_inv: dict[str, np.ndarray] = {}

    for hold_f in fit_folds:
        in_fit_cids = [
            c for f in fit_folds if f != hold_f for c in train_cids_by_fold[f]
        ]
        in_val_cids = list(train_cids_by_fold[hold_f])

        X_in_fit = np.concatenate([packets[c]["X_top8"] for c in in_fit_cids], axis=0)
        y_in_fit = np.concatenate([rel_map[c] for c in in_fit_cids], axis=0)
        X_in_val = np.concatenate([packets[c]["X_top8"] for c in in_val_cids], axis=0)

        for lr in LR_GRID:
            rstate = stream_seed(
                fold_seed, seed, assess_fold, arm_name, "inner", hold_f, lr
            )
            model_200 = fit_lgbm_ranker_200(
                X_in_fit,
                y_in_fit,
                len(in_fit_cids),
                lr=lr,
                random_state=rstate,
                k=SLICE_K,
            )
            for n_est in N_EST_GRID:
                preds = predict_lgbm_ranker(model_200, X_in_val, num_iteration=n_est)
                preds_8 = preds.reshape(len(in_val_cids), SLICE_K)
                for idx_c, cid in enumerate(in_val_cids):
                    inner_oof_scores[(lr, n_est)][cid] = preds_8[idx_c]

        if with_veto:
            y_inv_in_fit = np.concatenate(
                [(~packets[c]["gate_8"]).astype(np.int32) for c in in_fit_cids],
                axis=0,
            )
            rstate_clf = stream_seed(
                fold_seed, seed, assess_fold, "RERANK8_CR_VETO", "inner_clf", hold_f
            )
            clf_in = fit_veto_classifier(
                X_in_fit, y_inv_in_fit, random_state=rstate_clf
            )
            p_inv_val = predict_veto_classifier(clf_in, X_in_val).reshape(
                len(in_val_cids), SLICE_K
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
    fit_u8 = np.stack([packets[c]["utility_8"] for c in fit_cids], axis=0)
    fit_gate8 = np.stack([packets[c]["gate_8"] for c in fit_cids], axis=0)
    fit_r882_u = fit_u8[:, 0]
    row_arange = np.arange(len(fit_cids), dtype=np.int32)

    best_gain = -1e9
    best_cfg: tuple[float, int, float] = (0.05, 100, math.inf)
    best_fam_deltas = np.zeros(len(fit_fams), dtype=np.float64)

    best_unc_gain = -1e9
    best_unc_cfg: tuple[float, int, float] = (0.05, 100, 0.0)

    best_m0_gain = -1e9
    best_m0_cfg: tuple[float, int, float] = (0.05, 100, 0.0)

    scores_mat_by_pair: dict[tuple[float, int], np.ndarray] = {}

    for lr in LR_GRID:
        for n_est in N_EST_GRID:
            scores_mat = np.stack(
                [inner_oof_scores[(lr, n_est)][c] for c in fit_cids], axis=0
            )
            scores_mat_by_pair[(lr, n_est)] = scores_mat
            best_j_vec = np.argmax(scores_mat, axis=1)
            gap_vec = scores_mat[row_arange, best_j_vec] - scores_mat[:, 0]

            for m in MARGIN_GRID:
                if math.isinf(m):
                    picks = np.zeros(len(fit_cids), dtype=np.int32)
                else:
                    picks = np.where(
                        (best_j_vec > 0) & (gap_vec > float(m)), best_j_vec, 0
                    )
                u_pick = fit_u8[row_arange, picks]
                diff_ctx = u_pick - fit_r882_u
                fam_deltas = np.array(
                    [float(np.mean(diff_ctx[pos])) for pos in fam_pos_lists],
                    dtype=np.float64,
                )
                gain = float(np.mean(fam_deltas))
                if gain > best_gain + 1e-12 or (
                    abs(gain - best_gain) <= 1e-12 and m > best_cfg[2]
                ):
                    best_gain = gain
                    best_cfg = (lr, n_est, m)
                    best_fam_deltas = fam_deltas

                if not math.isinf(m):
                    if gain > best_unc_gain + 1e-12 or (
                        abs(gain - best_unc_gain) <= 1e-12 and m > best_unc_cfg[2]
                    ):
                        best_unc_gain = gain
                        best_unc_cfg = (lr, n_est, m)
                if m == 0.0 and gain > best_m0_gain + 1e-12:
                    best_m0_gain = gain
                    best_m0_cfg = (lr, n_est, 0.0)

    rng_boot = np.random.default_rng(
        stream_seed(fold_seed, seed, assess_fold, arm_name, "inner_boot")
    )
    boot_idx = rng_boot.integers(
        0, len(fit_fams), size=(INNER_BOOT_DRAWS, len(fit_fams))
    )
    boot_means = best_fam_deltas[boot_idx].mean(axis=1)
    inner_lcb95 = float(np.quantile(boot_means, 0.025))
    inner_ucb95 = float(np.quantile(boot_means, 0.975))

    sel_lr, sel_n_est, sel_m = best_cfg
    strict_admitted = bool(
        (not math.isinf(sel_m))
        and (best_gain > STRICT_FIT_MIN_GAIN)
        and (inner_lcb95 > 0.0)
    )
    strict_m = sel_m if strict_admitted else math.inf

    X_fit_full = np.concatenate([packets[c]["X_top8"] for c in fit_cids], axis=0)
    y_fit_full = np.concatenate([rel_map[c] for c in fit_cids], axis=0)

    needed_lrs = {float(sel_lr), float(best_unc_cfg[0]), float(best_m0_cfg[0])}
    outer_models_by_lr: dict[float, lgb.LGBMRanker] = {}
    for lr_val in sorted(needed_lrs):
        rstate_outer = stream_seed(
            fold_seed, seed, assess_fold, arm_name, "outer", lr_val
        )
        outer_models_by_lr[lr_val] = fit_lgbm_ranker_200(
            X_fit_full,
            y_fit_full,
            len(fit_cids),
            lr=lr_val,
            random_state=rstate_outer,
            k=SLICE_K,
        )

    veto_res: dict[str, Any] | None = None
    if with_veto:
        p_inv_mat = np.stack([inner_oof_p_inv[c] for c in fit_cids], axis=0)
        ctrl_inv_mask = ~fit_gate8[:, 0]
        ctrl_inv_count = int(np.sum(ctrl_inv_mask))

        v_best_gain = -1e9
        v_best_cfg: tuple[float, int, float] = (0.05, 100, math.inf)
        v_best_fam_deltas = np.zeros(len(fit_fams), dtype=np.float64)
        v_best_inv_count = ctrl_inv_count
        v_best_new_inv_sw = 0

        for tier in (1, 2):
            found_in_tier = False
            for lr in LR_GRID:
                for n_est in N_EST_GRID:
                    scores_mat = scores_mat_by_pair[(lr, n_est)]
                    best_j_vec = np.argmax(scores_mat, axis=1)
                    gap_vec = scores_mat[row_arange, best_j_vec] - scores_mat[:, 0]
                    p_inv_best = p_inv_mat[row_arange, best_j_vec]
                    for m in MARGIN_GRID:
                        if math.isinf(m):
                            picks = np.zeros(len(fit_cids), dtype=np.int32)
                        else:
                            picks = np.where(
                                (best_j_vec > 0)
                                & (gap_vec > float(m))
                                & (p_inv_best <= float(FROZEN_VETO_TAU)),
                                best_j_vec,
                                0,
                            )
                        cand_inv_mask = ~fit_gate8[row_arange, picks]
                        cand_inv_count = int(np.sum(cand_inv_mask))
                        new_inv_sw = int(np.sum(cand_inv_mask & (~ctrl_inv_mask)))
                        if tier == 1:
                            if (
                                cand_inv_count > ctrl_inv_count
                                or new_inv_sw > 0
                                or math.isinf(m)
                            ):
                                continue
                        else:
                            if cand_inv_count > ctrl_inv_count:
                                continue
                        u_pick = fit_u8[row_arange, picks]
                        diff_ctx = u_pick - fit_r882_u
                        fam_deltas = np.array(
                            [float(np.mean(diff_ctx[pos])) for pos in fam_pos_lists],
                            dtype=np.float64,
                        )
                        gain = float(np.mean(fam_deltas))
                        if gain > v_best_gain + 1e-12 or (
                            abs(gain - v_best_gain) <= 1e-12 and m > v_best_cfg[2]
                        ):
                            v_best_gain = gain
                            v_best_cfg = (lr, n_est, m)
                            v_best_fam_deltas = fam_deltas
                            v_best_inv_count = cand_inv_count
                            v_best_new_inv_sw = new_inv_sw
                            if not math.isinf(m) and gain > STRICT_FIT_MIN_GAIN:
                                found_in_tier = True
            if found_in_tier:
                break

        rng_vboot = np.random.default_rng(
            stream_seed(fold_seed, seed, assess_fold, "RERANK8_CR_VETO", "inner_boot")
        )
        v_boot_idx = rng_vboot.integers(
            0, len(fit_fams), size=(INNER_BOOT_DRAWS, len(fit_fams))
        )
        v_boot_means = v_best_fam_deltas[v_boot_idx].mean(axis=1)
        v_inner_lcb95 = float(np.quantile(v_boot_means, 0.025))
        v_inner_ucb95 = float(np.quantile(v_boot_means, 0.975))

        v_lr, v_n_est, v_m = v_best_cfg
        v_strict_admitted = bool(
            (not math.isinf(v_m))
            and (v_best_gain > STRICT_FIT_MIN_GAIN)
            and (v_inner_lcb95 > 0.0)
            and (v_best_inv_count <= ctrl_inv_count)
        )
        v_strict_m = v_m if v_strict_admitted else math.inf

        if float(v_lr) not in outer_models_by_lr:
            rstate_outer_v = stream_seed(
                fold_seed, seed, assess_fold, arm_name, "outer", float(v_lr)
            )
            outer_models_by_lr[float(v_lr)] = fit_lgbm_ranker_200(
                X_fit_full,
                y_fit_full,
                len(fit_cids),
                lr=float(v_lr),
                random_state=rstate_outer_v,
                k=SLICE_K,
            )

        y_inv_full = np.concatenate(
            [(~packets[c]["gate_8"]).astype(np.int32) for c in fit_cids], axis=0
        )
        rstate_clf_outer = stream_seed(
            fold_seed, seed, assess_fold, "RERANK8_CR_VETO", "outer_clf"
        )
        outer_clf = fit_veto_classifier(
            X_fit_full, y_inv_full, random_state=rstate_clf_outer
        )

        veto_res = {
            "outer_ranker_model": outer_models_by_lr[float(v_lr)],
            "outer_veto_clf": outer_clf,
            "receipt": {
                "fold_seed": int(fold_seed),
                "seed": int(seed),
                "assess_fold": int(assess_fold),
                "arm": "RERANK8_CR_VETO",
                "k": SLICE_K,
                "feature_dim": FEATURE_DIM_K8,
                "selected_lr": float(v_lr),
                "selected_n_estimators": int(v_n_est),
                "selected_tau": float(FROZEN_VETO_TAU),
                "unconditional_margin": ("inf" if math.isinf(v_m) else float(v_m)),
                "strict_margin": (
                    "inf" if math.isinf(v_strict_m) else float(v_strict_m)
                ),
                "strict_admitted": v_strict_admitted,
                "inner_fit_family_gain": float(v_best_gain),
                "inner_fit_lcb95": v_inner_lcb95,
                "inner_fit_ucb95": v_inner_ucb95,
                "inner_fit_ctrl_invalid_count": ctrl_inv_count,
                "inner_fit_cand_invalid_count": v_best_inv_count,
                "inner_fit_new_invalid_switches": v_best_new_inv_sw,
            },
        }

    return {
        "outer_model": outer_models_by_lr[float(sel_lr)],
        "outer_unc_model": outer_models_by_lr[float(best_unc_cfg[0])],
        "outer_m0_model": outer_models_by_lr[float(best_m0_cfg[0])],
        "veto_result": veto_res,
        "receipt": {
            "fold_seed": int(fold_seed),
            "seed": int(seed),
            "assess_fold": int(assess_fold),
            "arm": arm_name,
            "k": SLICE_K,
            "feature_dim": FEATURE_DIM_K8,
            "selected_lr": float(sel_lr),
            "selected_n_estimators": int(sel_n_est),
            "unconditional_margin": "inf" if math.isinf(sel_m) else float(sel_m),
            "strict_margin": "inf" if math.isinf(strict_m) else float(strict_m),
            "strict_admitted": strict_admitted,
            "inner_fit_family_gain": float(best_gain),
            "inner_fit_lcb95": inner_lcb95,
            "inner_fit_ucb95": inner_ucb95,
            "unc_finite_lr": float(best_unc_cfg[0]),
            "unc_finite_n_estimators": int(best_unc_cfg[1]),
            "unc_finite_margin": float(best_unc_cfg[2]),
            "unc_finite_inner_gain": float(best_unc_gain),
            "m0_lr": float(best_m0_cfg[0]),
            "m0_n_estimators": int(best_m0_cfg[1]),
            "m0_inner_gain": float(best_m0_gain),
        },
    }


def parse_margin(val: str | float) -> float:
    return math.inf if val == "inf" else float(val)


def evaluate_cohort_on_cids_k8(
    cids: list[str],
    packets: dict[str, dict[str, Any]],
    model: lgb.LGBMRanker,
    n_est: int,
    margin: float,
    veto_clf: lgb.LGBMClassifier | None = None,
    veto_tau: float = FROZEN_VETO_TAU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    X_all = np.concatenate([packets[c]["X_top8"] for c in cids], axis=0)
    preds = predict_lgbm_ranker(model, X_all, num_iteration=n_est).reshape(
        len(cids), SLICE_K
    )
    p_inv_mat = (
        predict_veto_classifier(veto_clf, X_all).reshape(len(cids), SLICE_K)
        if veto_clf is not None
        else None
    )

    utils = np.zeros(len(cids), dtype=np.float64)
    invalids = np.zeros(len(cids), dtype=np.bool_)
    changed = np.zeros(len(cids), dtype=np.bool_)
    picks: list[int] = []

    for i, c in enumerate(cids):
        pkt = packets[c]
        if p_inv_mat is not None:
            j = select_from_topk_scores_with_veto(
                preds[i], p_inv_mat[i], tau=veto_tau, margin=margin
            )
        else:
            j = select_from_topk_scores(preds[i], margin=margin)
        picks.append(int(j))
        utils[i] = float(pkt["utility_8"][j])
        invalids[i] = bool(not pkt["gate_8"][j])
        changed[i] = bool(j != 0)
    return utils, invalids, changed, picks


def paired_family_bootstrap_contrast(
    rows: list[dict[str, Any]],
    arm_panel: np.ndarray,
    ctrl_panel: np.ndarray,
    boot_seed: int,
    n_boot: int = BOOT_DRAWS,
) -> dict[str, Any]:
    """
    Paired family-cluster bootstrap (`10,000` draws) supporting both:
      - 3D panels `(n_model_seeds, n_fold_seeds, n_contexts)`: per model seed, takes the median
        over fold seeds of the equal-family OOF gain (matching H270 PREREG.md / README.md),
        and also records the context-median-across-fold-seeds statistic.
      - 2D panels `(n_model_seeds, n_contexts)`: per model seed equal-family gain.
    """
    fams = sorted({r["family_id"] for r in rows})
    n_fams = len(fams)
    ctx_fams = [r["family_id"] for r in rows]
    fam_idx_lists = [
        np.array([i for i, f in enumerate(ctx_fams) if f == fam], dtype=np.int32)
        for fam in fams
    ]

    if arm_panel.ndim == 3:
        S, F, _ = arm_panel.shape
        diff_3d = arm_panel - ctrl_panel
        # Per (model_seed, fold_seed, family) equal-family delta
        fam_diffs_3d = np.zeros((S, F, n_fams), dtype=np.float64)
        for fi, idx in enumerate(fam_idx_lists):
            fam_diffs_3d[:, :, fi] = diff_3d[:, :, idx].mean(axis=2)

        # H270 primary aggregation: per model seed, equal-family gain per fold seed -> median over fold seeds
        seed_fseed_fam_gains = fam_diffs_3d.mean(axis=2)  # (S, F)
        per_seed_fam_gains = [
            float(np.median(seed_fseed_fam_gains[s])) for s in range(S)
        ]
        all_9_rep_gains = [
            [float(seed_fseed_fam_gains[s, f]) for f in range(F)] for s in range(S)
        ]

        # Context gain per seed (median across fold seeds of context mean gain)
        seed_fseed_ctx_gains = diff_3d.mean(axis=2)
        per_seed_ctx_gains = [
            float(np.median(seed_fseed_ctx_gains[s])) for s in range(S)
        ]

        # Family-level diffs per model seed (median over the 3 fold seeds for each family)
        fam_diffs = np.median(fam_diffs_3d, axis=1)  # (S, n_fams)

        # Bootstrap resamples families jointly across (S, F, n_fams):
        # For each draw b, each (s, f) has equal-family gain over resampled families -> median over f -> median over s
        rng = np.random.default_rng(boot_seed)
        boot_ix = rng.integers(0, n_fams, size=(n_boot, n_fams))
        boot_sf = np.zeros((n_boot, S, F), dtype=np.float64)
        for s in range(S):
            for f in range(F):
                vec = fam_diffs_3d[s, f]
                boot_sf[:, s, f] = vec[boot_ix].mean(axis=1)
        boot_seed_meds = np.median(boot_sf, axis=2)  # (n_boot, S)
        boot_med_across_seeds = np.median(boot_seed_meds, axis=1)
    else:
        S, _ = arm_panel.shape
        diff_2d = arm_panel - ctrl_panel
        fam_diffs = np.zeros((S, n_fams), dtype=np.float64)
        for fi, idx in enumerate(fam_idx_lists):
            fam_diffs[:, fi] = diff_2d[:, idx].mean(axis=1)
        per_seed_fam_gains = [float(fam_diffs[s].mean()) for s in range(S)]
        all_9_rep_gains = []
        per_seed_ctx_gains = [float(diff_2d[s].mean()) for s in range(S)]

        rng = np.random.default_rng(boot_seed)
        boot_ix = rng.integers(0, n_fams, size=(n_boot, n_fams))
        boot_seed_means = np.zeros((n_boot, S), dtype=np.float64)
        for s in range(S):
            vec = fam_diffs[s]
            boot_seed_means[:, s] = vec[boot_ix].mean(axis=1)
        boot_med_across_seeds = np.median(boot_seed_means, axis=1)

    med_fam_gain = float(np.median(per_seed_fam_gains))
    mean_fam_gain = float(np.mean(per_seed_fam_gains))
    med_ctx_gain = float(np.median(per_seed_ctx_gains))
    mean_ctx_gain = float(np.mean(per_seed_ctx_gains))
    pos_seeds = int(sum(1 for g in per_seed_fam_gains if g > 0.0))

    ci95 = [
        float(np.quantile(boot_med_across_seeds, 0.025)),
        float(np.quantile(boot_med_across_seeds, 0.975)),
    ]
    ci975_holm2 = [
        float(np.quantile(boot_med_across_seeds, 0.0125)),
        float(np.quantile(boot_med_across_seeds, 0.9875)),
    ]
    raw_p = float((1 + int(np.sum(boot_med_across_seeds <= 0.0))) / (n_boot + 1))

    wtl_by_seed = []
    harm_gt_1e4_by_seed = []
    harm_gt_001_by_seed = []
    for s in range(fam_diffs.shape[0]):
        fd = fam_diffs[s]
        w_cnt = int(np.sum(fd > TIE_MARGIN))
        l_cnt = int(np.sum(fd < -TIE_MARGIN))
        t_cnt = int(n_fams - w_cnt - l_cnt)
        wtl_by_seed.append({"wins": w_cnt, "ties": t_cnt, "losses": l_cnt})
        harm_gt_1e4_by_seed.append(l_cnt)
        harm_gt_001_by_seed.append(int(np.sum(fd < -0.01)))

    med_fam_vec = np.median(fam_diffs, axis=0)
    harmed_families = [
        {"family_id": fams[fi], "delta": float(med_fam_vec[fi])}
        for fi in range(n_fams)
        if med_fam_vec[fi] < -TIE_MARGIN
    ]
    harmed_families.sort(key=lambda d: d["delta"])

    res = {
        "median_family_gain": med_fam_gain,
        "mean_family_gain": mean_fam_gain,
        "per_seed_family_gains": per_seed_fam_gains,
        "median_context_gain": med_ctx_gain,
        "mean_context_gain": mean_ctx_gain,
        "per_seed_context_gains": per_seed_ctx_gains,
        "positive_seeds": pos_seeds,
        "ci95": ci95,
        "ci975_holm2": ci975_holm2,
        "raw_p_one_sided": raw_p,
        "family_wtl_by_seed": wtl_by_seed,
        "harm_families_gt_1e4_by_seed": harm_gt_1e4_by_seed,
        "harm_families_gt_001_by_seed": harm_gt_001_by_seed,
        "n_harmed_families_median_gt_1e4": len(harmed_families),
        "n_harmed_families_median_gt_001": sum(
            1 for h in harmed_families if h["delta"] < -0.01
        ),
        "worst_harmed_families_top5": harmed_families[:5],
    }
    if all_9_rep_gains:
        res["per_seed_by_fold_seed_family_gains"] = all_9_rep_gains
        res["all_9_repetitions_positive"] = bool(
            all(g > 0.0 for row in all_9_rep_gains for g in row)
        )
    return res


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


def run_smoke(out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert sha256_file(R882_CKPT) == R882_EXPECTED_SHA256

    rng = np.random.default_rng(273001)
    X_syn = rng.normal(size=(64, FEATURE_DIM_K8)).astype(np.float32)
    y_syn = np.tile(np.arange(SLICE_K - 1, -1, -1, dtype=np.int32), 8)
    y_inv = np.zeros(64, dtype=np.int32)
    y_inv[7] = 1
    y_inv[15] = 1
    model = fit_lgbm_ranker_200(
        X_syn, y_syn, 8, lr=0.05, random_state=273001, k=SLICE_K
    )
    preds = predict_lgbm_ranker(model, X_syn, num_iteration=50)
    clf = fit_veto_classifier(X_syn, y_inv, random_state=273002)
    p_inv = predict_veto_classifier(clf, X_syn)
    assert preds.shape == (64,) and np.all(np.isfinite(preds))
    assert p_inv.shape == (64,) and np.all((p_inv >= 0.0) & (p_inv <= 1.0))

    smoke_doc = {
        "schema": "jope.h273.smoke/v1",
        "wave_id": "h273-rerank8-cleanroom-veto-20261103",
        "status": "PASS",
        "r882_sha256": sha256_file(R882_CKPT),
        "k": SLICE_K,
        "feature_dim": FEATURE_DIM_K8,
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

    # Step 1: Reproduce Native91 1.1377941898164525 bit-exact (91/91) FIRST + TOP4_PICK@r882 ceiling
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
        "schema": "jope.h273.baseline-regression/v1",
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

    # Step 2: Load ONLY DL-Train context features, priors, and top-8 packets (DL-Val unopened during FIT/selection)
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

    train_packets = load_cohort_topk_packets(
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
    for fseed in (*STAGE1_FOLD_SEEDS, *FOLD_SEEDS):
        sched = fold_man["fold_schedules"][str(fseed)]
        train_cids_by_fseed_fold[fseed] = {
            int(f["fold"]): list(f["context_ids"]) for f in sched["folds"]
        }
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train_rows}
    train_cids_ordered = [r["context_sha256"] for r in train_rows]
    train_cid_to_idx = {c: i for i, c in enumerate(train_cids_ordered)}
    n_train = len(train_rows)

    # =========================================================================
    # STAGE 1: H270 Reproduction Gate (fold_seeds 270140-270142, seeds 270101-270103)
    # =========================================================================
    stage1_ctrl_panel = np.zeros(
        (len(STAGE1_SEEDS), len(STAGE1_FOLD_SEEDS), n_train), dtype=np.float64
    )
    stage1_cr_panel = np.zeros(
        (len(STAGE1_SEEDS), len(STAGE1_FOLD_SEEDS), n_train), dtype=np.float64
    )
    stage1_cell_agreements: list[dict[str, Any]] = []
    total_s1_ctxs = 0
    total_s1_exact_slot_matches = 0
    total_s1_change_decision_matches = 0
    stage1_total_invalid_picks = 0
    stage1_admitted_cells = 0

    for fi_idx, fseed in enumerate(STAGE1_FOLD_SEEDS):
        cids_by_fold = train_cids_by_fseed_fold[fseed]
        for s_idx, s in enumerate(STAGE1_SEEDS):
            for c_idx, cid in enumerate(train_cids_ordered):
                stage1_ctrl_panel[s_idx, fi_idx, c_idx] = train_packets[cid]["r882_u"]
            for k in range(5):
                check_cooperative_pause()
                res_s1 = run_cell_fit_and_select_k8(
                    fold_seed=fseed,
                    seed=s,
                    assess_fold=k,
                    arm_name="RERANK8_CR_STAGE1",
                    shuffle_placebo=False,
                    train_cids_by_fold=cids_by_fold,
                    packets=train_packets,
                    fam_by_cid=fam_by_cid,
                    with_veto=False,
                )
                rcpt = res_s1["receipt"]
                if rcpt["strict_admitted"]:
                    stage1_admitted_cells += 1
                assess_cids = cids_by_fold[k]
                u_s1, inv_s1, _, picks_s1 = evaluate_cohort_on_cids_k8(
                    assess_cids,
                    train_packets,
                    res_s1["outer_model"],
                    int(rcpt["selected_n_estimators"]),
                    parse_margin(rcpt["strict_margin"]),
                )
                stage1_total_invalid_picks += int(np.sum(inv_s1))
                for i_c, cid in enumerate(assess_cids):
                    stage1_cr_panel[s_idx, fi_idx, train_cid_to_idx[cid]] = float(
                        u_s1[i_c]
                    )

                ref_cell_path = H270_REF_DIR / f"F{fseed}-S{s}-K{k}.json"
                if ref_cell_path.exists():
                    ref_cell = json.loads(ref_cell_path.read_text(encoding="utf-8"))
                    ref_map_picks = {
                        str(cid_r): int(p_r)
                        for cid_r, p_r in zip(
                            ref_cell["assess_ids"],
                            ref_cell["assess_picks"]["RERANK8"],
                        )
                    }
                    exact_m = sum(
                        1
                        for i_c, cid in enumerate(assess_cids)
                        if picks_s1[i_c] == ref_map_picks[cid]
                    )
                    chg_m = sum(
                        1
                        for i_c, cid in enumerate(assess_cids)
                        if (picks_s1[i_c] != 0) == (ref_map_picks[cid] != 0)
                    )
                    total_s1_ctxs += len(assess_cids)
                    total_s1_exact_slot_matches += exact_m
                    total_s1_change_decision_matches += chg_m
                    stage1_cell_agreements.append(
                        {
                            "cell": f"F{fseed}-S{s}-K{k}",
                            "n_assess": len(assess_cids),
                            "exact_slot_matches": exact_m,
                            "exact_slot_pct": round(
                                100.0 * exact_m / len(assess_cids), 2
                            ),
                            "change_decision_matches": chg_m,
                            "change_decision_pct": round(
                                100.0 * chg_m / len(assess_cids), 2
                            ),
                            "cr_strict_margin": rcpt["strict_margin"],
                            "h270_strict_margin": ref_cell["receipts"]["RERANK8"][
                                "strict_margin"
                            ],
                        }
                    )
                del res_s1
            trim_malloc()

    stage1_contrast = paired_family_bootstrap_contrast(
        train_rows, stage1_cr_panel, stage1_ctrl_panel, boot_seed=STAGE1_BOOT_SEED
    )
    stage1_summary = {
        "schema": "jope.h273.stage1_h270_reproduction/v1",
        "fold_seeds": list(STAGE1_FOLD_SEEDS),
        "model_seeds": list(STAGE1_SEEDS),
        "boot_seed": STAGE1_BOOT_SEED,
        "n_cells": 45,
        "n_cells_strict_admitted_of_45": stage1_admitted_cells,
        "h270_published_admitted_cells_of_45": 45,
        "total_assessed_context_evaluations": total_s1_ctxs,
        "exact_slot_pick_matches": total_s1_exact_slot_matches,
        "exact_slot_pick_agreement_pct": round(
            100.0 * total_s1_exact_slot_matches / max(total_s1_ctxs, 1), 2
        ),
        "change_vs_r882_decision_matches": total_s1_change_decision_matches,
        "change_vs_r882_decision_agreement_pct": round(
            100.0 * total_s1_change_decision_matches / max(total_s1_ctxs, 1), 2
        ),
        "rerank8_cr_stage1_contrast": stage1_contrast,
        "h270_published_primary_median_family_gain": 0.052315,
        "h270_published_primary_ci95": [0.0171, 0.1060],
        "h270_published_per_seed_gains": [0.0489, 0.0523, 0.0593],
        "rerank8_cr_stage1_total_invalid_neg2_picks_45cells": stage1_total_invalid_picks,
        "h270_published_total_invalid_neg2_picks_45cells": 150,
        "control_total_invalid_neg2_picks_45cells": 153,
        "per_cell_agreements": stage1_cell_agreements,
    }
    fsync_write_json(out_dir / "stage1_h270_repro.json", stage1_summary)

    # =========================================================================
    # STAGE 2: Fresh-Seed Confirmatory Evaluation (fold_seeds 273140-273142, seeds 273101-273103)
    # =========================================================================
    cell_results: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    selection_receipts: list[dict[str, Any]] = []

    for fseed in FOLD_SEEDS:
        cids_by_fold = train_cids_by_fseed_fold[fseed]
        for s in SEEDS:
            for k in range(5):
                check_cooperative_pause()
                # 1. RERANK8_CR (+ RERANK8_CR_UNC + RERANK8_CR_VETO)
                res_cr = run_cell_fit_and_select_k8(
                    fold_seed=fseed,
                    seed=s,
                    assess_fold=k,
                    arm_name="RERANK8_CR",
                    shuffle_placebo=False,
                    train_cids_by_fold=cids_by_fold,
                    packets=train_packets,
                    fam_by_cid=fam_by_cid,
                    with_veto=True,
                )
                cell_results[("RERANK8_CR", fseed, s, k)] = res_cr
                selection_receipts.append(res_cr["receipt"])
                assert res_cr["veto_result"] is not None
                selection_receipts.append(res_cr["veto_result"]["receipt"])

                # 2. RERANK8_CR_SHUF (+ RERANK8_CR_SHUF_UNC + RERANK8_CR_SHUF_UNC_M0)
                res_shuf = run_cell_fit_and_select_k8(
                    fold_seed=fseed,
                    seed=s,
                    assess_fold=k,
                    arm_name="RERANK8_CR_SHUF",
                    shuffle_placebo=True,
                    train_cids_by_fold=cids_by_fold,
                    packets=train_packets,
                    fam_by_cid=fam_by_cid,
                    with_veto=False,
                )
                cell_results[("RERANK8_CR_SHUF", fseed, s, k)] = res_shuf
                selection_receipts.append(res_shuf["receipt"])
            trim_malloc()

    fit_lock_doc = {
        "schema": "jope.h273.fit_selections_lock/v1",
        "wave_id": "h273-rerank8-cleanroom-veto-20261103",
        "locked_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dl_val_reads_before_lock": 0,
        "fold_seeds": list(FOLD_SEEDS),
        "model_seeds": list(SEEDS),
        "n_confirmatory_cells_per_arm": 45,
        "selections": selection_receipts,
    }
    fit_lock_sha = fsync_write_json(out_dir / "fit_selections_lock.json", fit_lock_doc)

    # Step 4: Score DL-Train OOF across all 3 fold_seeds x 3 model_seeds x 5 folds
    arm_order = [
        "CONTROL",
        "RERANK8_CR",
        "RERANK8_CR_VETO",
        "RERANK8_CR_SHUF",
        "RERANK8_CR_UNC",
        "RERANK8_CR_SHUF_UNC",
        "RERANK8_CR_SHUF_UNC_M0",
        "TOP4_PICK_ORACLE",
        "TOP8_PICK_ORACLE",
        "POOL_ORACLE",
    ]
    n_arms = len(arm_order)
    arm_to_idx = {a: i for i, a in enumerate(arm_order)}

    train_panel = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), n_train, n_arms), dtype=np.float64
    )
    train_invalid = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), n_train, n_arms), dtype=np.bool_
    )
    train_changed = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), n_train, n_arms), dtype=np.bool_
    )
    train_deep_picks = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), n_train, n_arms), dtype=np.bool_
    )
    train_top1_oracle8_match = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), n_train, n_arms), dtype=np.bool_
    )

    for s_idx, s in enumerate(SEEDS):
        for fi_idx, fseed in enumerate(FOLD_SEEDS):
            cids_by_fold = train_cids_by_fseed_fold[fseed]
            for i, cid in enumerate(train_cids_ordered):
                pkt = train_packets[cid]
                train_panel[s_idx, fi_idx, i, arm_to_idx["CONTROL"]] = pkt["r882_u"]
                train_invalid[s_idx, fi_idx, i, arm_to_idx["CONTROL"]] = pkt[
                    "r882_invalid"
                ]
                train_top1_oracle8_match[s_idx, fi_idx, i, arm_to_idx["CONTROL"]] = (
                    bool(pkt["top8_oracle_j"] == 0)
                )

                train_panel[s_idx, fi_idx, i, arm_to_idx["TOP4_PICK_ORACLE"]] = pkt[
                    "top4_oracle_u"
                ]
                train_panel[s_idx, fi_idx, i, arm_to_idx["TOP8_PICK_ORACLE"]] = pkt[
                    "top8_oracle_u"
                ]
                train_top1_oracle8_match[
                    s_idx, fi_idx, i, arm_to_idx["TOP8_PICK_ORACLE"]
                ] = True
                train_panel[s_idx, fi_idx, i, arm_to_idx["POOL_ORACLE"]] = pkt[
                    "pool_oracle_u"
                ]

            for k in range(5):
                assess_cids = cids_by_fold[k]
                res_cr = cell_results[("RERANK8_CR", fseed, s, k)]
                rcpt_cr = res_cr["receipt"]
                veto_res = res_cr["veto_result"]
                assert veto_res is not None
                rcpt_v = veto_res["receipt"]

                res_shuf = cell_results[("RERANK8_CR_SHUF", fseed, s, k)]
                rcpt_shuf = res_shuf["receipt"]

                eval_specs = [
                    (
                        "RERANK8_CR",
                        res_cr["outer_model"],
                        int(rcpt_cr["selected_n_estimators"]),
                        parse_margin(rcpt_cr["strict_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_VETO",
                        veto_res["outer_ranker_model"],
                        int(rcpt_v["selected_n_estimators"]),
                        parse_margin(rcpt_v["strict_margin"]),
                        veto_res["outer_veto_clf"],
                    ),
                    (
                        "RERANK8_CR_SHUF",
                        res_shuf["outer_model"],
                        int(rcpt_shuf["selected_n_estimators"]),
                        parse_margin(rcpt_shuf["strict_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_UNC",
                        res_cr["outer_unc_model"],
                        int(rcpt_cr["unc_finite_n_estimators"]),
                        float(rcpt_cr["unc_finite_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_SHUF_UNC",
                        res_shuf["outer_unc_model"],
                        int(rcpt_shuf["unc_finite_n_estimators"]),
                        float(rcpt_shuf["unc_finite_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_SHUF_UNC_M0",
                        res_shuf["outer_m0_model"],
                        int(rcpt_shuf["m0_n_estimators"]),
                        0.0,
                        None,
                    ),
                ]
                for aname, mdl, n_e, m_val, v_clf in eval_specs:
                    u_a, inv_a, chg_a, picks_a = evaluate_cohort_on_cids_k8(
                        assess_cids,
                        train_packets,
                        mdl,
                        n_e,
                        m_val,
                        veto_clf=v_clf,
                        veto_tau=FROZEN_VETO_TAU,
                    )
                    a_idx = arm_to_idx[aname]
                    for idx_c, cid in enumerate(assess_cids):
                        pos = train_cid_to_idx[cid]
                        train_panel[s_idx, fi_idx, pos, a_idx] = u_a[idx_c]
                        train_invalid[s_idx, fi_idx, pos, a_idx] = inv_a[idx_c]
                        train_changed[s_idx, fi_idx, pos, a_idx] = chg_a[idx_c]
                        train_deep_picks[s_idx, fi_idx, pos, a_idx] = bool(
                            picks_a[idx_c] >= 4
                        )
                        train_top1_oracle8_match[s_idx, fi_idx, pos, a_idx] = bool(
                            picks_a[idx_c] == train_packets[cid]["top8_oracle_j"]
                        )

    # Step 5: Now and ONLY now load DL-Val91 (91 contexts, 19 families) for secondary evaluation
    aprior_val: dict[str, np.ndarray] = {}
    proj16_val: dict[str, np.ndarray] = {}
    val_cid_set = {r["context_sha256"] for r in val_rows}
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
    trim_malloc()

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
    trim_malloc()

    val_packets = load_cohort_topk_packets(
        val_rows, r882_costs, aprior_val, proj16_val, ctx_feats_val
    )
    aprior_val.clear()
    proj16_val.clear()
    ctx_feats_val.clear()
    trim_malloc()

    val_cids_ordered = [r["context_sha256"] for r in val_rows]
    n_val = len(val_rows)

    val_panel_all = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), 5, n_val, n_arms), dtype=np.float64
    )
    val_invalid_all = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), 5, n_val, n_arms), dtype=np.bool_
    )
    val_changed_all = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), 5, n_val, n_arms), dtype=np.bool_
    )
    val_deep_picks_all = np.zeros(
        (len(SEEDS), len(FOLD_SEEDS), 5, n_val, n_arms), dtype=np.bool_
    )

    ctx27_idx = val_cids_ordered.index(CONTEXT_27_SHA256)
    ctx27_pkt = val_packets[CONTEXT_27_SHA256]
    ctx27_cells: list[dict[str, Any]] = []

    for s_idx, s in enumerate(SEEDS):
        for fi_idx, fseed in enumerate(FOLD_SEEDS):
            for k in range(5):
                for i, cid in enumerate(val_cids_ordered):
                    pkt = val_packets[cid]
                    val_panel_all[s_idx, fi_idx, k, i, arm_to_idx["CONTROL"]] = pkt[
                        "r882_u"
                    ]
                    val_invalid_all[s_idx, fi_idx, k, i, arm_to_idx["CONTROL"]] = pkt[
                        "r882_invalid"
                    ]
                    val_panel_all[
                        s_idx, fi_idx, k, i, arm_to_idx["TOP4_PICK_ORACLE"]
                    ] = pkt["top4_oracle_u"]
                    val_panel_all[
                        s_idx, fi_idx, k, i, arm_to_idx["TOP8_PICK_ORACLE"]
                    ] = pkt["top8_oracle_u"]
                    val_panel_all[s_idx, fi_idx, k, i, arm_to_idx["POOL_ORACLE"]] = pkt[
                        "pool_oracle_u"
                    ]

                res_cr = cell_results[("RERANK8_CR", fseed, s, k)]
                rcpt_cr = res_cr["receipt"]
                veto_res = res_cr["veto_result"]
                assert veto_res is not None
                rcpt_v = veto_res["receipt"]
                res_shuf = cell_results[("RERANK8_CR_SHUF", fseed, s, k)]
                rcpt_shuf = res_shuf["receipt"]

                eval_specs = [
                    (
                        "RERANK8_CR",
                        res_cr["outer_model"],
                        int(rcpt_cr["selected_n_estimators"]),
                        parse_margin(rcpt_cr["strict_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_VETO",
                        veto_res["outer_ranker_model"],
                        int(rcpt_v["selected_n_estimators"]),
                        parse_margin(rcpt_v["strict_margin"]),
                        veto_res["outer_veto_clf"],
                    ),
                    (
                        "RERANK8_CR_SHUF",
                        res_shuf["outer_model"],
                        int(rcpt_shuf["selected_n_estimators"]),
                        parse_margin(rcpt_shuf["strict_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_UNC",
                        res_cr["outer_unc_model"],
                        int(rcpt_cr["unc_finite_n_estimators"]),
                        float(rcpt_cr["unc_finite_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_SHUF_UNC",
                        res_shuf["outer_unc_model"],
                        int(rcpt_shuf["unc_finite_n_estimators"]),
                        float(rcpt_shuf["unc_finite_margin"]),
                        None,
                    ),
                    (
                        "RERANK8_CR_SHUF_UNC_M0",
                        res_shuf["outer_m0_model"],
                        int(rcpt_shuf["m0_n_estimators"]),
                        0.0,
                        None,
                    ),
                ]
                cell_picks_27: dict[str, int] = {}
                for aname, mdl, n_e, m_val, v_clf in eval_specs:
                    u_v, inv_v, chg_v, picks_v = evaluate_cohort_on_cids_k8(
                        val_cids_ordered,
                        val_packets,
                        mdl,
                        n_e,
                        m_val,
                        veto_clf=v_clf,
                        veto_tau=FROZEN_VETO_TAU,
                    )
                    a_idx = arm_to_idx[aname]
                    val_panel_all[s_idx, fi_idx, k, :, a_idx] = u_v
                    val_invalid_all[s_idx, fi_idx, k, :, a_idx] = inv_v
                    val_changed_all[s_idx, fi_idx, k, :, a_idx] = chg_v
                    val_deep_picks_all[s_idx, fi_idx, k, :, a_idx] = (
                        np.asarray(picks_v) >= 4
                    )
                    cell_picks_27[aname] = int(picks_v[ctx27_idx])

                ctx27_cells.append(
                    {
                        "fold_seed": int(fseed),
                        "seed": int(s),
                        "fold": int(k),
                        "rerank8_cr_pick": cell_picks_27["RERANK8_CR"],
                        "rerank8_cr_u": float(
                            ctx27_pkt["utility_8"][cell_picks_27["RERANK8_CR"]]
                        ),
                        "rerank8_cr_veto_pick": cell_picks_27["RERANK8_CR_VETO"],
                        "rerank8_cr_veto_u": float(
                            ctx27_pkt["utility_8"][cell_picks_27["RERANK8_CR_VETO"]]
                        ),
                    }
                )

    cell_results.clear()
    trim_malloc()

    # Primary DL-Train contrasts
    primary_contrasts_specs = [
        ("RERANK8_CR - CONTROL", "RERANK8_CR", "CONTROL"),
        ("RERANK8_CR_VETO - CONTROL", "RERANK8_CR_VETO", "CONTROL"),
        ("RERANK8_CR_VETO - RERANK8_CR", "RERANK8_CR_VETO", "RERANK8_CR"),
        ("RERANK8_CR - RERANK8_CR_SHUF", "RERANK8_CR", "RERANK8_CR_SHUF"),
        ("RERANK8_CR - RERANK8_CR_SHUF_UNC", "RERANK8_CR", "RERANK8_CR_SHUF_UNC"),
        (
            "RERANK8_CR - RERANK8_CR_SHUF_UNC_M0",
            "RERANK8_CR",
            "RERANK8_CR_SHUF_UNC_M0",
        ),
        (
            "RERANK8_CR_VETO - RERANK8_CR_SHUF",
            "RERANK8_CR_VETO",
            "RERANK8_CR_SHUF",
        ),
        (
            "RERANK8_CR_VETO - RERANK8_CR_SHUF_UNC",
            "RERANK8_CR_VETO",
            "RERANK8_CR_SHUF_UNC",
        ),
        ("RERANK8_CR_SHUF - CONTROL", "RERANK8_CR_SHUF", "CONTROL"),
        ("RERANK8_CR_SHUF_UNC - CONTROL", "RERANK8_CR_SHUF_UNC", "CONTROL"),
        ("RERANK8_CR_SHUF_UNC_M0 - CONTROL", "RERANK8_CR_SHUF_UNC_M0", "CONTROL"),
        ("RERANK8_CR_UNC - CONTROL", "RERANK8_CR_UNC", "CONTROL"),
        (
            "RERANK8_CR_UNC - RERANK8_CR_SHUF_UNC",
            "RERANK8_CR_UNC",
            "RERANK8_CR_SHUF_UNC",
        ),
        ("TOP4_PICK_ORACLE - CONTROL", "TOP4_PICK_ORACLE", "CONTROL"),
        ("TOP8_PICK_ORACLE - CONTROL", "TOP8_PICK_ORACLE", "CONTROL"),
        ("POOL_ORACLE - CONTROL", "POOL_ORACLE", "CONTROL"),
    ]
    primary_contrasts: dict[str, Any] = {}
    for cname, a_arm, b_arm in primary_contrasts_specs:
        primary_contrasts[cname] = paired_family_bootstrap_contrast(
            train_rows,
            train_panel[:, :, :, arm_to_idx[a_arm]],
            train_panel[:, :, :, arm_to_idx[b_arm]],
            boot_seed=stream_seed(BOOT_SEED, "primary", cname),
        )

    # Secondary DL-Val91 (fold_seed = 273140, 5-cell mixture per model seed)
    val_panel_f0 = val_panel_all[:, 0, :, :, :].mean(axis=1)  # (S, n_val, n_arms)
    secondary_contrasts: dict[str, Any] = {}
    for cname, a_arm, b_arm in primary_contrasts_specs:
        secondary_contrasts[cname] = paired_family_bootstrap_contrast(
            val_rows,
            val_panel_f0[:, :, arm_to_idx[a_arm]],
            val_panel_f0[:, :, arm_to_idx[b_arm]],
            boot_seed=stream_seed(BOOT_SEED, "val_fseed0", cname),
        )

    # Secondary DL-Val91 3-fold-seed aggregate (45 cells)
    val_panel_3f = val_panel_all.mean(axis=2)  # (S, F, n_val, n_arms)
    val_3fseed_contrasts: dict[str, Any] = {}
    for cname in [
        "RERANK8_CR - CONTROL",
        "RERANK8_CR_VETO - CONTROL",
        "RERANK8_CR_VETO - RERANK8_CR",
        "RERANK8_CR_SHUF_UNC - CONTROL",
    ]:
        a_arm, b_arm = cname.split(" - ")
        val_3fseed_contrasts[cname] = paired_family_bootstrap_contrast(
            val_rows,
            val_panel_3f[:, :, :, arm_to_idx[a_arm]],
            val_panel_3f[:, :, :, arm_to_idx[b_arm]],
            boot_seed=stream_seed(BOOT_SEED, "val_3fseed", cname),
        )

    # Holm-Bonferroni m=2 for RERANK8_CR over {PRIMARY, SECONDARY}
    holm2_cr_raw = {
        "PRIMARY_TRAIN": primary_contrasts["RERANK8_CR - CONTROL"]["raw_p_one_sided"],
        "SECONDARY_VAL91": secondary_contrasts["RERANK8_CR - CONTROL"][
            "raw_p_one_sided"
        ],
    }
    holm2_cr_adj = apply_holm_bonferroni(holm2_cr_raw)
    primary_contrasts["RERANK8_CR - CONTROL"]["holm_p_m2"] = holm2_cr_adj[
        "PRIMARY_TRAIN"
    ]
    secondary_contrasts["RERANK8_CR - CONTROL"]["holm_p_m2"] = holm2_cr_adj[
        "SECONDARY_VAL91"
    ]

    # Descriptive Holm m=2 for RERANK8_CR_VETO over {PRIMARY, SECONDARY}
    holm2_veto_raw = {
        "PRIMARY_TRAIN": primary_contrasts["RERANK8_CR_VETO - CONTROL"][
            "raw_p_one_sided"
        ],
        "SECONDARY_VAL91": secondary_contrasts["RERANK8_CR_VETO - CONTROL"][
            "raw_p_one_sided"
        ],
    }
    holm2_veto_adj = apply_holm_bonferroni(holm2_veto_raw)
    primary_contrasts["RERANK8_CR_VETO - CONTROL"]["holm_p_m2"] = holm2_veto_adj[
        "PRIMARY_TRAIN"
    ]
    secondary_contrasts["RERANK8_CR_VETO - CONTROL"]["holm_p_m2"] = holm2_veto_adj[
        "SECONDARY_VAL91"
    ]

    # Arm utility & diagnostic summaries
    train_fams_sorted = sorted({r["family_id"] for r in train_rows})
    train_ctx_fams = np.asarray([r["family_id"] for r in train_rows])
    train_arm_summaries: dict[str, Any] = {}
    for a_idx, aname in enumerate(arm_order):
        fam_u_sf = np.zeros((len(SEEDS), len(FOLD_SEEDS)), dtype=np.float64)
        for s_i in range(len(SEEDS)):
            for f_i in range(len(FOLD_SEEDS)):
                vec = train_panel[s_i, f_i, :, a_idx]
                fam_u_sf[s_i, f_i] = float(
                    np.mean(
                        [
                            float(np.mean(vec[train_ctx_fams == fam]))
                            for fam in train_fams_sorted
                        ]
                    )
                )
        seed_fam_u = [float(np.median(fam_u_sf[s_i])) for s_i in range(len(SEEDS))]
        seed_ctx_u = [
            float(np.median(train_panel[s_i, :, :, a_idx].mean(axis=1)))
            for s_i in range(len(SEEDS))
        ]
        inv_45 = int(np.sum(train_invalid[:, :, :, a_idx]))
        inv_by_seed = [
            int(np.sum(train_invalid[s_i, :, :, a_idx])) for s_i in range(len(SEEDS))
        ]
        chg_45 = int(np.sum(train_changed[:, :, :, a_idx]))
        deep_45 = int(np.sum(train_deep_picks[:, :, :, a_idx]))
        train_arm_summaries[aname] = {
            "median_family_utility": float(np.median(seed_fam_u)),
            "per_seed_family_utility": seed_fam_u,
            "median_context_utility": float(np.median(seed_ctx_u)),
            "per_seed_context_utility": seed_ctx_u,
            "total_invalid_neg2_picks_45cells": inv_45,
            "invalid_neg2_picks_by_seed_across_3_fold_seeds": inv_by_seed,
            "total_changed_picks_45cells": chg_45,
            "total_deep_slot4to7_picks_45cells": deep_45,
            "deep_slot4to7_share_of_changed_picks": round(
                float(deep_45) / max(chg_45, 1), 4
            ),
            "deep_slot4to7_share_of_all_picks": round(
                float(deep_45) / float(len(SEEDS) * len(FOLD_SEEDS) * n_train), 4
            ),
            "top1_within_top8_accuracy": round(
                float(np.mean(train_top1_oracle8_match[:, :, :, a_idx])), 4
            ),
        }

    val_fams_sorted = sorted({r["family_id"] for r in val_rows})
    val_ctx_fams = np.asarray([r["family_id"] for r in val_rows])
    val_arm_summaries: dict[str, Any] = {}
    for a_idx, aname in enumerate(arm_order):
        seed_fam_u = []
        seed_ctx_u = []
        for s_i in range(len(SEEDS)):
            vec = val_panel_f0[s_i, :, a_idx]
            seed_ctx_u.append(float(np.mean(vec)))
            seed_fam_u.append(
                float(
                    np.mean(
                        [
                            float(np.mean(vec[val_ctx_fams == fam]))
                            for fam in val_fams_sorted
                        ]
                    )
                )
            )
        inv_15 = int(np.sum(val_invalid_all[:, 0, :, :, a_idx]))
        inv_by_seed = [
            int(np.sum(val_invalid_all[s_i, 0, :, :, a_idx]))
            for s_i in range(len(SEEDS))
        ]
        chg_15 = int(np.sum(val_changed_all[:, 0, :, :, a_idx]))
        deep_15 = int(np.sum(val_deep_picks_all[:, 0, :, :, a_idx]))
        val_arm_summaries[aname] = {
            "median_family_utility": float(np.median(seed_fam_u)),
            "per_seed_family_utility": seed_fam_u,
            "median_context_utility": float(np.median(seed_ctx_u)),
            "per_seed_context_utility": seed_ctx_u,
            "total_invalid_neg2_picks_15cells": inv_15,
            "invalid_neg2_picks_by_seed_5folds": inv_by_seed,
            "total_changed_picks_15cells": chg_15,
            "total_deep_slot4to7_picks_15cells": deep_15,
        }

    train_elo = compute_elo_table(
        train_rows, train_panel, arm_order, stream_seed(BOOT_SEED, "train_elo")
    )
    val_elo = compute_elo_table(
        val_rows,
        val_panel_all[:, 0, :, :, :].reshape(len(SEEDS), 5 * n_val, n_arms),
        arm_order,
        stream_seed(BOOT_SEED, "val_elo"),
    )

    def eval_replication_gates(arm_name: str) -> dict[str, Any]:
        p_c = primary_contrasts[f"{arm_name} - CONTROL"]
        s_c = secondary_contrasts[f"{arm_name} - CONTROL"]
        shuf_strict_c = primary_contrasts[f"{arm_name} - RERANK8_CR_SHUF"]
        shuf_unc_c = primary_contrasts[f"{arm_name} - RERANK8_CR_SHUF_UNC"]

        med_gain_ok = bool(p_c["median_family_gain"] > PRIMARY_ADVANCE_MIN_GAIN)
        lcb95_ok = bool(p_c["ci95"][0] > 0.0)
        holm2_lcb_ok = bool(p_c["ci975_holm2"][0] > 0.0)
        holm2_p_ok = bool(p_c.get("holm_p_m2", 1.0) < 0.05)
        seeds_3of3_ok = bool(p_c["positive_seeds"] == 3)
        beats_shuf_strict = bool(
            shuf_strict_c["median_family_gain"] > 0.0 and shuf_strict_c["ci95"][0] > 0.0
        )
        beats_shuf_unc = bool(
            shuf_unc_c["median_family_gain"] > 0.0 and shuf_unc_c["ci95"][0] > 0.0
        )
        val_ok = bool(
            s_c["median_family_gain"] > 0.0
            and s_c["positive_seeds"] == 3
            and s_c["ci975_holm2"][0] > 0.0
        )
        arm_inv_45 = int(
            train_arm_summaries[arm_name]["total_invalid_neg2_picks_45cells"]
        )
        ctrl_inv_45 = int(
            train_arm_summaries["CONTROL"]["total_invalid_neg2_picks_45cells"]
        )
        inv_ok = bool(arm_inv_45 <= ctrl_inv_45)

        replicated = bool(
            med_gain_ok
            and lcb95_ok
            and holm2_lcb_ok
            and holm2_p_ok
            and seeds_3of3_ok
            and beats_shuf_strict
            and beats_shuf_unc
            and val_ok
            and inv_ok
        )
        return {
            "arm": arm_name,
            "n_cells_strict_admitted_of_45": int(
                sum(
                    1
                    for r in selection_receipts
                    if r["arm"] == arm_name and r["strict_admitted"]
                )
            ),
            "primary_median_family_gain_gt_0002": med_gain_ok,
            "primary_lcb95_gt_0": lcb95_ok,
            "primary_holm2_lcb95_gt_0": holm2_lcb_ok,
            "primary_holm2_p_lt_005": holm2_p_ok,
            "primary_3_of_3_seeds_positive": seeds_3of3_ok,
            "primary_beats_shuf_strict": beats_shuf_strict,
            "primary_beats_shuf_unc": beats_shuf_unc,
            "secondary_dl_val91_3_of_3_positive_and_holm_lcb_gt_0": val_ok,
            "dl_train_invalid_neg2_picks_45cells": arm_inv_45,
            "control_dl_train_invalid_neg2_picks_45cells": ctrl_inv_45,
            "dl_train_invalid_le_control": inv_ok,
            "dl_val91_invalid_neg2_picks_15cells": int(
                val_arm_summaries[arm_name]["total_invalid_neg2_picks_15cells"]
            ),
            "replicated": replicated,
        }

    gates_cr = eval_replication_gates("RERANK8_CR")
    gates_veto = eval_replication_gates("RERANK8_CR_VETO")

    if gates_cr["replicated"]:
        verdict = "REPLICATED"
        winning_arm = "RERANK8_CR"
    elif gates_veto["replicated"]:
        verdict = "REPLICATED"
        winning_arm = "RERANK8_CR_VETO"
    else:
        verdict = "NOT_REPLICATED"
        winning_arm = None

    r882_sha_after = sha256_file(R882_CKPT)
    ckpt882_after = torch.load(R882_CKPT, map_location="cpu", weights_only=False)
    dope_sha_after = str(ckpt882_after.get("dope_sha256", DOPE_EXPECTED_SHA256))
    del ckpt882_after
    trim_malloc()

    np.savez_compressed(
        out_dir / "panels.npz",
        arm_order=np.asarray(arm_order),
        train_cids=np.asarray(train_cids_ordered),
        val_cids=np.asarray(val_cids_ordered),
        stage1_cr_panel=stage1_cr_panel,
        train_panel=train_panel,
        train_invalid=train_invalid,
        train_changed=train_changed,
        train_deep_picks=train_deep_picks,
        val_panel_all=val_panel_all,
        val_invalid_all=val_invalid_all,
        val_changed_all=val_changed_all,
        val_deep_picks_all=val_deep_picks_all,
    )

    details_doc = {
        "schema": "jope.h273.details/v1",
        "wave_id": "h273-rerank8-cleanroom-veto-20261103",
        "verdict": verdict,
        "winning_arm": winning_arm,
        "stage1_h270_reproduction": stage1_summary,
        "replication_gates": {
            "RERANK8_CR": gates_cr,
            "RERANK8_CR_VETO": gates_veto,
        },
        "primary_train_contrasts": primary_contrasts,
        "secondary_val_contrasts_fseed273140": secondary_contrasts,
        "secondary_val_contrasts_3fseed_aggregate": val_3fseed_contrasts,
        "train_arm_summaries": train_arm_summaries,
        "val_arm_summaries": val_arm_summaries,
        "train_elo": train_elo,
        "val_elo": val_elo,
        "context_27_cells": ctx27_cells,
    }
    details_sha = fsync_write_json(out_dir / "details.json", details_doc)

    # Full metrics payload
    metrics_full = {
        "schema": "jope.h273.metrics/v1",
        "wave": "H273",
        "wave_id": "h273-rerank8-cleanroom-veto-20261103",
        "study_path": "research/flywheel_selfplay/h273_rerank8_cleanroom_veto",
        "owner": "Argon",
        "status": "COMPLETED",
        "verdict": verdict,
        "winning_arm": winning_arm,
        "completed_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "wall_seconds": round(time.time() - t0, 3),
        "hygiene": {
            "dope_sha256_before": dope_sha_before,
            "dope_sha256_after": dope_sha_after,
            "dope_unchanged": bool(dope_sha_before == dope_sha_after),
            "r882_sha256_before": r882_sha_before,
            "r882_sha256_after": r882_sha_after,
            "r882_unchanged": bool(r882_sha_before == r882_sha_after),
            "optimizer_updates": 0,
            "test_time_reveals": 0,
            "dl_val_reads_before_fit_lock": 0,
            "clean_room_no_topk_or_h265_core_import": True,
        },
        "baseline_regression": baseline_reg,
        "stage1_h270_reproduction_gate": {
            k: v for k, v in stage1_summary.items() if k != "per_cell_agreements"
        },
        "replication_gates": {
            "RERANK8_CR": gates_cr,
            "RERANK8_CR_VETO": gates_veto,
        },
        "holm2_rerank8_cr": {
            "raw_p": holm2_cr_raw,
            "adjusted_p": holm2_cr_adj,
        },
        "holm2_rerank8_cr_veto": {
            "raw_p": holm2_veto_raw,
            "adjusted_p": holm2_veto_adj,
        },
        "primary_dl_train_oof_205fam": {
            "fold_seeds": list(FOLD_SEEDS),
            "model_seeds": list(SEEDS),
            "boot_seed": BOOT_SEED,
            "boot_draws": BOOT_DRAWS,
            "n_families": 205,
            "n_contexts": 1219,
            "n_cells_per_arm": 45,
            "arm_summaries": train_arm_summaries,
            "contrasts": primary_contrasts,
            "mm_bt_elo_anchored_r882": train_elo,
        },
        "secondary_dl_val_native91": {
            "fold_seed": SECONDARY_FOLD_SEED,
            "model_seeds": list(SEEDS),
            "n_families": 19,
            "n_contexts": 91,
            "reuse_disclosure": "DL-Val91 (19 families, 91 contexts) is heavily reused across H241-H272; DL-Train 205-family OOF on fresh fold seeds 273140/273141/273142 is the primary decision endpoint.",
            "arm_summaries_fseed273140": val_arm_summaries,
            "contrasts_fseed273140": secondary_contrasts,
            "contrasts_3_fold_seeds_aggregate": val_3fseed_contrasts,
            "mm_bt_elo_anchored_r882": val_elo,
        },
        "context_27_summary": {
            "context_sha256": CONTEXT_27_SHA256,
            "family_id": ctx27_pkt["family_id"],
            "index": ctx27_idx,
            "r882_slot0_utility": float(ctx27_pkt["r882_u"]),
            "top4_oracle_utility": float(ctx27_pkt["top4_oracle_u"]),
            "top8_oracle_utility": float(ctx27_pkt["top8_oracle_u"]),
            "rerank8_cr_mean_utility_fseed273140": float(
                np.mean(
                    [
                        c["rerank8_cr_u"]
                        for c in ctx27_cells
                        if c["fold_seed"] == SECONDARY_FOLD_SEED
                    ]
                )
            ),
            "rerank8_cr_veto_mean_utility_fseed273140": float(
                np.mean(
                    [
                        c["rerank8_cr_veto_u"]
                        for c in ctx27_cells
                        if c["fold_seed"] == SECONDARY_FOLD_SEED
                    ]
                )
            ),
        },
        "fit_selections_lock_summary": {
            "path": "fit_selections_lock.json",
            "sha256": fit_lock_sha,
            "details_json_sha256": details_sha,
            "n_cells_strict_admitted_rerank8_cr_of_45": gates_cr[
                "n_cells_strict_admitted_of_45"
            ],
            "n_cells_strict_admitted_rerank8_cr_veto_of_45": gates_veto[
                "n_cells_strict_admitted_of_45"
            ],
            "n_cells_strict_admitted_rerank8_cr_shuf_of_45": int(
                sum(
                    1
                    for r in selection_receipts
                    if r["arm"] == "RERANK8_CR_SHUF" and r["strict_admitted"]
                )
            ),
        },
    }

    fsync_write_json(out_dir / "metrics_full.json", metrics_full, compact_leaves=False)
    fsync_write_json(out_dir / "metrics.json", metrics_full, compact_leaves=True)
    print(
        "H273_PRIMARY_COMPLETE",
        json.dumps(
            {
                "verdict": verdict,
                "winning_arm": winning_arm,
                "stage1_exact_slot_pct": stage1_summary[
                    "exact_slot_pick_agreement_pct"
                ],
                "stage1_median_family_gain": stage1_contrast["median_family_gain"],
                "stage2_rerank8_cr_train": primary_contrasts["RERANK8_CR - CONTROL"][
                    "median_family_gain"
                ],
                "stage2_rerank8_cr_ci95": primary_contrasts["RERANK8_CR - CONTROL"][
                    "ci95"
                ],
                "stage2_rerank8_cr_veto_train": primary_contrasts[
                    "RERANK8_CR_VETO - CONTROL"
                ]["median_family_gain"],
                "stage2_rerank8_cr_veto_ci95": primary_contrasts[
                    "RERANK8_CR_VETO - CONTROL"
                ]["ci95"],
                "stage2_shuf_unc_train": primary_contrasts[
                    "RERANK8_CR_SHUF_UNC - CONTROL"
                ]["median_family_gain"],
            },
            indent=2,
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="H273 clean-room RERANK8 + veto runner"
    )
    parser.add_argument("--mode", choices=["smoke", "primary"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "smoke":
        return run_smoke(args.out_dir)
    return run_primary(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
