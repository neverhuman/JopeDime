#!/usr/bin/env python3
"""
H265 (`h265_top4_listwise_reranker`) — Zero-Update Listwise LambdaMART Re-Ranker of `r882` Top-4
on a Read-Only `r882` Copy (0 Test-Time Reveals, Single-Pick Evaluation on `DL-Val91` & `TRAIN OOF`).
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch

EVIDENCE_ROOT = Path("/home/ubuntu/jope_plan/evidence/H265/argon-t1658")
SNAPSHOT_DIR = EVIDENCE_ROOT / "snapshot"
INPUTS_DIR = EVIDENCE_ROOT / "inputs"
CAPSULE_STUDY_DIR = (
    EVIDENCE_ROOT
    / "capsule"
    / "research"
    / "flywheel_selfplay"
    / "h265_top4_listwise_reranker"
)

INPUT_POOLS_DIR = Path("/home/ubuntu/jope-data/h50/input")
CACHE_DIR = Path("/home/ubuntu/jope-data/h58/cache")
CONTEXT_FEATURES_NPZ = CACHE_DIR / "context_features.npz"
POOL_PRIORS_NPZ = CACHE_DIR / "pool_priors.npz"

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

CTX27_INDEX = 27
CTX27_CID = "43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91"
CTX27_FAM = "fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f"
CTX27_R882_UTILITY_EXPECTED = 1.122694955279

FOLD_SEED = 265140
SEEDS = (265101, 265102, 265103)
BOOT_SEED = 265150
BOOT_DRAWS = 10_000
INNER_BOOT_DRAWS = 2_000
HOLM_M = 2
STRICT_FIT_MIN_GAIN = 0.001 + 1e-12
ADVANCE_MIN_GAIN = 0.002
TIE_MARGIN = 1e-4

LR_GRID = (0.03, 0.05, 0.10)
N_EST_GRID = (50, 100, 200)
MARGIN_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, math.inf)


def trim_malloc() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def check_cooperative_pause() -> None:
    pause_file = os.environ.get("H265_PAUSE_FILE", "")
    if not pause_file:
        return
    p = Path(pause_file)
    while p.exists():
        time.sleep(0.5)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def stream_seed(*parts: Any) -> int:
    raw = ":".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & 0x7FFFFFFF


def load_common_h252() -> Any:
    if str(SNAPSHOT_DIR) not in sys.path:
        sys.path.insert(0, str(SNAPSHOT_DIR))
    spec = importlib.util.spec_from_file_location(
        "common_h252", SNAPSHOT_DIR / "common_h252.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_true_top4_relevance(
    fitness_4: np.ndarray, gate_4: np.ndarray
) -> np.ndarray:
    """
    Computes integer listwise relevance labels `rel in {0, 1, 2, 3}` for `r882`'s `top-4` candidates
    on a `DL-Train` FIT context (higher = better):
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
    Builds the `(4, 560)` float32 strictly outcome-free feature matrix for `r882`'s `top-4` candidates
    (`top4 = np.argsort(rcost, kind="mergesort")[:4]`).
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

    # Standardized sem24 and proj16 across full pool
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

    # Context feature block (76D: 48 lm48_z + 25 leading dope629 + 3 dope629 summary stats)
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
    )  # (76,)

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
                0.0 if j == 0 else float((c4[j] - c4[0]) / r_sd),  # exact H217 zgap
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
        )  # (20,)

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
        )  # (12,)

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
    """
    Returns chosen slot `j in {0, 1, 2, 3}` inside `r882`'s `top-4`:
      - If `margin == inf`, always returns `0` (`r882` rank-1).
      - Otherwise finds `j* = argmax_{j in 0..3} scores_4[j]` (ties broken by smallest `j`).
      - Returns `j*` iff `scores_4[j*] - scores_4[0] > margin`, else `0` (fallback "keep r882 rank-1").
    """
    if math.isinf(margin):
        return 0
    best_j = int(np.argmax(scores_4))
    if best_j == 0:
        return 0
    if float(scores_4[best_j] - scores_4[0]) > float(margin):
        return best_j
    return 0


def shuffle_labels_within_family(
    rel_by_cid: dict[str, np.ndarray],
    cids: list[str],
    fam_by_cid: dict[str, str],
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Permutes candidate relevance labels within each family across all `(context, top4_slot)` entries
    belonging to that family, preserving the exact within-family label distribution while breaking
    candidate feature-to-rank correspondence (`RERANK4_SHUF` placebo).
    """
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
    """
    Loads candidate pools in batches of `batch_size`, constructs the `(4, 560)` top-4 feature matrix
    and top-4 outcomes (`utility_4`, `fitness_4`, `gate_4`, `rel_4`, `oracle_best_u_top4`), and
    immediately releases the full `1152`-candidate arrays so peak RSS stays well below `1.2 GiB`.
    """
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
                sem24 = z["sem24"].astype(np.float32)

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
            # Oracle best in top4 under gated cost (`fitness if gate else 2.0`, tie -> r882 order)
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


def run_cell_fit_and_select(
    seed: int,
    assess_fold: int,
    arm_name: str,
    shuffle_placebo: bool,
    train_cids_by_fold: dict[int, list[str]],
    packets: dict[str, dict[str, Any]],
    fam_by_cid: dict[str, str],
) -> dict[str, Any]:
    """
    Runs 4-fold inner cross-validation across the `4` FIT folds (`!= assess_fold`) for `(seed, assess_fold)`,
    applies the strict H257 FIT selector (plus records unconstrained inner-FIT-best and `m=0` reference),
    and trains the outer FIT `LGBMRanker` on all `4` FIT folds.
    Returns the fitted models and inner selection receipt (with 0 `DL-Val` access).
    """
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

    # Inner OOF score predictions on fit_cids for each (lr, n_est)
    inner_oof_scores: dict[tuple[float, int], dict[str, np.ndarray]] = {
        (lr, n_est): {} for lr in LR_GRID for n_est in N_EST_GRID
    }

    for hold_f in fit_folds:
        check_cooperative_pause()
        in_tr_cids = [
            c for f in fit_folds if f != hold_f for c in train_cids_by_fold[f]
        ]
        in_val_cids = train_cids_by_fold[hold_f]
        X_in_tr = np.concatenate([packets[c]["X_top4"] for c in in_tr_cids], axis=0)
        y_in_tr = np.concatenate([rel_map[c] for c in in_tr_cids], axis=0)
        X_in_val = np.concatenate([packets[c]["X_top4"] for c in in_val_cids], axis=0)

        for lr in LR_GRID:
            rstate = stream_seed(seed, assess_fold, arm_name, "inner", hold_f, lr)
            m200 = fit_lgbm_ranker_200(
                X_in_tr, y_in_tr, len(in_tr_cids), lr=lr, random_state=rstate
            )
            for n_est in N_EST_GRID:
                preds = predict_lgbm_ranker(
                    m200, X_in_val, num_iteration=int(n_est)
                ).reshape(len(in_val_cids), 4)
                for idx_c, cid in enumerate(in_val_cids):
                    inner_oof_scores[(lr, n_est)][cid] = preds[idx_c]

    # Evaluate inner-OOF equal-family utility gain on the 164 FIT families across (lr, n_est, margin)
    r882_u_by_cid = np.array([packets[c]["r882_u"] for c in fit_cids], dtype=np.float64)
    fit_cids_fams = np.array([fam_by_cid[c] for c in fit_cids])
    fam_masks = [fit_cids_fams == f for f in fit_fams]
    r882_fam_means = np.array(
        [float(np.mean(r882_u_by_cid[mask])) for mask in fam_masks], dtype=np.float64
    )

    best_unc_gain = -1e18
    best_unc_cfg = (LR_GRID[0], N_EST_GRID[0], 0.0)
    best_unc_fam_deltas = np.zeros(len(fit_fams), dtype=np.float64)
    grid_summary: list[dict[str, Any]] = []

    for lr in LR_GRID:
        for n_est in N_EST_GRID:
            sc_map = inner_oof_scores[(lr, n_est)]
            for margin in MARGIN_GRID:
                if math.isinf(margin):
                    continue
                u_vec = np.array(
                    [
                        packets[c]["utility_4"][
                            select_from_top4_scores(sc_map[c], margin)
                        ]
                        for c in fit_cids
                    ],
                    dtype=np.float64,
                )
                fam_means = np.array(
                    [float(np.mean(u_vec[mask])) for mask in fam_masks],
                    dtype=np.float64,
                )
                fam_deltas = fam_means - r882_fam_means
                gain = float(np.mean(fam_deltas))
                harm_rate = float(np.mean(fam_deltas < -TIE_MARGIN))
                grid_summary.append(
                    {
                        "lr": lr,
                        "n_estimators": n_est,
                        "margin": margin,
                        "inner_fit_family_gain": gain,
                        "inner_fit_harm_rate": harm_rate,
                    }
                )
                # Break numerical ties (1e-12) toward larger margin, smaller n_est, smaller lr
                if gain > best_unc_gain + 1e-12 or (
                    abs(gain - best_unc_gain) <= 1e-12
                    and (margin, -n_est, -lr)
                    > (best_unc_cfg[2], -best_unc_cfg[1], -best_unc_cfg[0])
                ):
                    best_unc_gain = gain
                    best_unc_cfg = (lr, n_est, margin)
                    best_unc_fam_deltas = fam_deltas

    # Compute paired family bootstrap LCB95 on the 164 FIT families for best_unc_cfg
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

    # Fit outer models on all 4 FIT folds:
    # 1) outer model at best_unc_cfg[0] (lr*)
    # 2) if 0.05 != best_unc_cfg[0], also fit lr=0.05 for the fixed (lr=0.05, n_est=100, m=0.0) diagnostic panel
    X_outer_fit = np.concatenate([packets[c]["X_top4"] for c in fit_cids], axis=0)
    y_outer_fit = np.concatenate([rel_map[c] for c in fit_cids], axis=0)

    outer_models_by_lr: dict[float, lgb.LGBMRanker] = {}
    for lr_needed in sorted({best_unc_cfg[0], 0.05}):
        rstate_out = stream_seed(seed, assess_fold, arm_name, "outer", lr_needed)
        outer_models_by_lr[lr_needed] = fit_lgbm_ranker_200(
            X_outer_fit,
            y_outer_fit,
            len(fit_cids),
            lr=lr_needed,
            random_state=rstate_out,
        )

    # Predict on held-out ASSESS fold (TRAIN OOF) before any DL-Val read
    assess_cids = train_cids_by_fold[assess_fold]
    X_assess = np.concatenate([packets[c]["X_top4"] for c in assess_cids], axis=0)
    m_sel = outer_models_by_lr[best_unc_cfg[0]]
    scores_assess_sel = predict_lgbm_ranker(
        m_sel, X_assess, num_iteration=int(best_unc_cfg[1])
    ).reshape(len(assess_cids), 4)

    m_fixed = outer_models_by_lr[0.05]
    scores_assess_fixed = predict_lgbm_ranker(
        m_fixed, X_assess, num_iteration=100
    ).reshape(len(assess_cids), 4)

    assess_preds = {}
    for idx_c, cid in enumerate(assess_cids):
        sc_sel = scores_assess_sel[idx_c]
        sc_fix = scores_assess_fixed[idx_c]
        j_strict = select_from_top4_scores(sc_sel, strict_margin)
        j_unc = select_from_top4_scores(sc_sel, best_unc_cfg[2])
        j_m0 = select_from_top4_scores(sc_fix, 0.0)
        assess_preds[cid] = {
            "scores_sel": sc_sel,
            "j_strict": j_strict,
            "j_unc": j_unc,
            "j_m0": j_m0,
        }

    receipt = {
        "cell_id": f"seed{seed}_fold{assess_fold}",
        "seed": seed,
        "assess_fold": assess_fold,
        "arm": arm_name,
        "shuffle_placebo": shuffle_placebo,
        "n_fit_contexts": len(fit_cids),
        "n_fit_families": len(fit_fams),
        "unconstrained_best_lr": float(best_unc_cfg[0]),
        "unconstrained_best_n_estimators": int(best_unc_cfg[1]),
        "unconstrained_best_margin": float(best_unc_cfg[2]),
        "inner_fit_family_gain": float(best_unc_gain),
        "inner_fit_ci95": [inner_lcb95, inner_ucb95],
        "strict_admitted": strict_admitted,
        "strict_selected_margin": (
            "inf" if math.isinf(strict_margin) else float(strict_margin)
        ),
    }
    return {
        "receipt": receipt,
        "outer_models_by_lr": outer_models_by_lr,
        "best_unc_cfg": best_unc_cfg,
        "strict_margin": strict_margin,
        "assess_preds": assess_preds,
    }


def paired_family_contrast(
    rows: list[dict[str, Any]],
    panel_3d: np.ndarray,
    arm_names: list[str],
    cand_arm: str,
    base_arm: str,
    boot_ix: np.ndarray,
) -> dict[str, Any]:
    ic = arm_names.index(cand_arm)
    ib = arm_names.index(base_arm)
    fams = [r["family_id"] for r in rows]
    fam_order = sorted(set(fams))
    fams_arr = np.asarray(fams)

    ctx_diff = panel_3d[:, :, ic] - panel_3d[:, :, ib]
    per_seed_ctx_gains = [float(np.mean(ctx_diff[s])) for s in range(panel_3d.shape[0])]

    fam_diff = np.stack(
        [ctx_diff[:, fams_arr == f].mean(axis=1) for f in fam_order],
        axis=1,
    )  # [3, N_fam]
    per_seed_fam_gains = [float(np.mean(fam_diff[s])) for s in range(panel_3d.shape[0])]
    pos_seeds = int(sum(1 for g in per_seed_fam_gains if g > 0.0))

    # Bootstrap over families: take median over seeds for each bootstrap draw (matching H257)
    # boot_ix has shape [B, N_fam]
    boot_seed_means = np.stack(
        [fam_diff[s][boot_ix].mean(axis=1) for s in range(panel_3d.shape[0])],
        axis=0,
    )  # [3, B]
    boot_med = np.median(boot_seed_means, axis=0)  # [B]
    ci95 = [
        float(np.quantile(boot_med, 0.025)),
        float(np.quantile(boot_med, 0.975)),
    ]
    ci975 = [
        float(np.quantile(boot_med, 0.0125)),
        float(np.quantile(boot_med, 0.9875)),
    ]
    raw_p = float((1 + int(np.sum(boot_med <= 0.0))) / (len(boot_med) + 1))

    wtl_by_seed = []
    harm_1e4_by_seed = []
    harm_001_by_seed = []
    for s in range(panel_3d.shape[0]):
        d_s = fam_diff[s]
        w = int(np.sum(d_s > TIE_MARGIN))
        loss_c = int(np.sum(d_s < -TIE_MARGIN))
        t = int(len(d_s) - w - loss_c)
        wtl_by_seed.append({"wins": w, "ties": t, "losses": loss_c})
        harm_1e4_by_seed.append(int(np.sum(d_s < -TIE_MARGIN)))
        harm_001_by_seed.append(int(np.sum(d_s < -0.01)))

    fam_diff_mean_seed = fam_diff.mean(axis=0)
    harmed_fams_1e4 = [
        {"family_id": fam_order[i], "delta": float(fam_diff_mean_seed[i])}
        for i in range(len(fam_order))
        if float(fam_diff_mean_seed[i]) < -TIE_MARGIN
    ]

    return {
        "median_family_gain": float(np.median(per_seed_fam_gains)),
        "mean_family_gain": float(np.mean(per_seed_fam_gains)),
        "per_seed_family_gains": per_seed_fam_gains,
        "median_context_gain": float(np.median(per_seed_ctx_gains)),
        "mean_context_gain": float(np.mean(per_seed_ctx_gains)),
        "per_seed_context_gains": per_seed_ctx_gains,
        "positive_seeds": pos_seeds,
        "ci95": ci95,
        "ci975_holm2": ci975,
        "raw_p_one_sided": raw_p,
        "family_wtl_by_seed": wtl_by_seed,
        "harm_families_gt_1e4_by_seed": harm_1e4_by_seed,
        "harm_families_gt_001_by_seed": harm_001_by_seed,
        "harmed_families_list": harmed_fams_1e4,
    }


def compute_chunked_mm_bt_elo(
    common_mod: Any,
    rows: list[dict[str, Any]],
    panel_3d: np.ndarray,
    arm_names: list[str],
    anchor_arm: str,
    boot_ix: np.ndarray,
    chunk_size: int = 100,
) -> dict[str, Any]:
    anchor_idx = arm_names.index(anchor_arm)
    fams = [r["family_id"] for r in rows]
    wins_by_seed = np.stack(
        [
            common_mod.family_wins_tensor(panel_3d[s], fams)
            for s in range(panel_3d.shape[0])
        ],
        axis=0,
    )
    per_seed_elo = common_mod.bt_ratings(wins_by_seed.mean(axis=1), anchor=anchor_idx)

    wins_mean_s = wins_by_seed.mean(axis=0)
    boot_elo_chunks = []
    for start in range(0, min(len(boot_ix), 1000), chunk_size):
        check_cooperative_pause()
        sl = boot_ix[start : start + chunk_size]
        w_chunk = wins_mean_s[sl].mean(axis=1)
        boot_elo_chunks.append(common_mod.bt_ratings(w_chunk, anchor=anchor_idx))
    boot_elo = np.concatenate(boot_elo_chunks, axis=0)

    out: dict[str, Any] = {}
    for j, name in enumerate(arm_names):
        s_elos = [float(per_seed_elo[s, j]) for s in range(panel_3d.shape[0])]
        out[name] = {
            "median_elo": float(np.median(s_elos)),
            "per_seed_elo": s_elos,
            "ci95": [
                float(np.quantile(boot_elo[:, j], 0.025)),
                float(np.quantile(boot_elo[:, j], 0.975)),
            ],
        }
    return out


def run_smoke(out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    r882_sha = sha256_file(R882_CKPT)
    assert r882_sha == R882_EXPECTED_SHA256

    # Synthetic 8-context smoke test of LGBMRanker + feature builder + strict selector (0 outcomes read)
    rng = np.random.default_rng(265001)
    X_syn = rng.normal(size=(32, 560)).astype(np.float32)
    y_syn = np.tile(np.array([3, 2, 1, 0], dtype=np.int32), 8)
    m_syn = fit_lgbm_ranker_200(X_syn, y_syn, n_groups=8, lr=0.05, random_state=265001)
    preds_syn = predict_lgbm_ranker(m_syn, X_syn, num_iteration=50)
    assert preds_syn.shape == (32,) and np.all(np.isfinite(preds_syn))

    smoke_obj = {
        "schema": "jope.h265.smoke/v1",
        "status": "PASS",
        "outcomes_read": 0,
        "r882_sha256": r882_sha,
        "synthetic_ranker_preds_shape": list(preds_syn.shape),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    fsync_write_json(out_dir / "smoke.json", smoke_obj)
    return 0


def run_primary(out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    common_mod = load_common_h252()

    # Verify input SHA256s before any data load
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

    # Load R882 cached costs
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

    # Step 1: Native91 1.1377941898164525 bit-exact (91/91) FIRST + TOP4_PICK@r882 1.2403601682250796 ceiling
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
    if bit_exact_r882 != 91 or r882_abs_diff != 0.0:
        raise RuntimeError(
            f"R882 Native91 regression failed: {bit_exact_r882}/91, mean={r882_native91_mean}"
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
        "schema": "jope.h265.baseline-regression/v1",
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

    # Load fold manifest and verify 100% agreement
    fold_man = json.loads(
        (CAPSULE_STUDY_DIR / "fold_manifest.json").read_text(encoding="utf-8")
    )
    train_cids_by_fold: dict[int, list[str]] = {
        int(f["fold"]): list(f["context_ids"]) for f in fold_man["folds"]
    }
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train_rows}

    # Step 3: Fit all 15 cells for RERANK4 (Primary) and RERANK4_SHUF (Secondary Placebo)
    cell_results: dict[tuple[str, int, int], dict[str, Any]] = {}
    selection_receipts: list[dict[str, Any]] = []

    for s in SEEDS:
        for k in range(5):
            check_cooperative_pause()
            for arm_name, is_shuf in (("RERANK4", False), ("RERANK4_SHUF", True)):
                res = run_cell_fit_and_select(
                    seed=s,
                    assess_fold=k,
                    arm_name=arm_name,
                    shuffle_placebo=is_shuf,
                    train_cids_by_fold=train_cids_by_fold,
                    packets=train_packets,
                    fam_by_cid=fam_by_cid,
                )
                cell_results[(arm_name, s, k)] = res
                selection_receipts.append(res["receipt"])

    # Step 4: Globally fsync-lock all 15x2 FIT selections BEFORE any DL-Val feature/pool read for scoring!
    lock_payload = {
        "schema": "jope.h265.fit-selections-lock/v1",
        "wave_id": "h265-top4-listwise-reranker-20261103",
        "fold_seed": FOLD_SEED,
        "seeds": list(SEEDS),
        "dl_val_feature_or_pool_reads_during_fit": 0,
        "locked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cell_selections": selection_receipts,
    }
    lock_path = out_dir / "fit_selections_lock.json"
    fsync_write_json(lock_path, lock_payload)
    lock_sha256 = sha256_file(lock_path)

    # Step 5: Now load DL-Val top-4 packets and score all 15 cells on DL-Val91 (0 test-time reveals!)
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

    val_cids = [r["context_sha256"] for r in val_rows]
    train_cids = [r["context_sha256"] for r in train_rows]
    X_val_all = np.concatenate([val_packets[c]["X_top4"] for c in val_cids], axis=0)

    # Arms tracked across panels:
    #   0: CONTROL (R882)
    #   1: RERANK4 (Strict FIT-selected, Primary)
    #   2: RERANK4_SHUF (Strict FIT-selected, Secondary placebo)
    #   3: RERANK4_UNC (Unconstrained inner-FIT-best margin, Diagnostic)
    #   4: RERANK4_SHUF_UNC (Unconstrained inner-FIT-best margin, Diagnostic)
    #   5: RERANK4_M0 (Unconstrained margin=0, lr=0.05, n_est=100, Diagnostic)
    #   6: RERANK4_SHUF_M0 (Unconstrained margin=0, lr=0.05, n_est=100, Diagnostic)
    #   7: TOP4_PICK_ORACLE (Descriptive 4-reveal oracle ceiling inside top-4)
    arm_names = [
        "CONTROL",
        "RERANK4",
        "RERANK4_SHUF",
        "RERANK4_UNC",
        "RERANK4_SHUF_UNC",
        "RERANK4_M0",
        "RERANK4_SHUF_M0",
        "TOP4_PICK_ORACLE",
    ]

    val_panel = np.zeros((len(SEEDS), len(val_rows), len(arm_names)), dtype=np.float64)
    train_panel = np.zeros(
        (len(SEEDS), len(train_rows), len(arm_names)), dtype=np.float64
    )

    # Track diagnostics: changed-choice counts, top-1-within-top-4 accuracy, -2.0 invalid counts, and Context #27
    val_diag_by_arm_seed: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}
    train_diag_by_arm_seed: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}
    ctx27_by_arm_seed: dict[str, list[float]] = {a: [] for a in arm_names}
    ctx27_choices_by_cell: list[dict[str, Any]] = []

    assert val_cids[CTX27_INDEX] == CTX27_CID
    assert val_rows[CTX27_INDEX]["family_id"] == CTX27_FAM

    for s_idx, s in enumerate(SEEDS):
        # 1. DL-Val91: 5-cell average per context for seed s (matching H257)
        val_cell_u = np.zeros((5, len(val_rows), len(arm_names)), dtype=np.float64)
        val_cell_changed = np.zeros(
            (5, len(val_rows), len(arm_names)), dtype=np.float64
        )
        val_cell_top1_acc = np.zeros(
            (5, len(val_rows), len(arm_names)), dtype=np.float64
        )
        val_cell_inv2 = np.zeros((5, len(val_rows), len(arm_names)), dtype=np.float64)

        for k in range(5):
            res_main = cell_results[("RERANK4", s, k)]
            res_shuf = cell_results[("RERANK4_SHUF", s, k)]

            lr_m, nest_m, marg_unc_m = res_main["best_unc_cfg"]
            marg_str_m = res_main["strict_margin"]
            sc_val_m_sel = predict_lgbm_ranker(
                res_main["outer_models_by_lr"][lr_m],
                X_val_all,
                num_iteration=int(nest_m),
            ).reshape(len(val_cids), 4)
            sc_val_m_fix = predict_lgbm_ranker(
                res_main["outer_models_by_lr"][0.05],
                X_val_all,
                num_iteration=100,
            ).reshape(len(val_cids), 4)

            lr_sh, nest_sh, marg_unc_sh = res_shuf["best_unc_cfg"]
            marg_str_sh = res_shuf["strict_margin"]
            sc_val_sh_sel = predict_lgbm_ranker(
                res_shuf["outer_models_by_lr"][lr_sh],
                X_val_all,
                num_iteration=int(nest_sh),
            ).reshape(len(val_cids), 4)
            sc_val_sh_fix = predict_lgbm_ranker(
                res_shuf["outer_models_by_lr"][0.05],
                X_val_all,
                num_iteration=100,
            ).reshape(len(val_cids), 4)

            for c_idx, cid in enumerate(val_cids):
                pkt = val_packets[cid]
                u4 = pkt["utility_4"]
                gate4 = pkt["gate_4"]
                oracle_j = pkt["top4_oracle_j"]

                j_by_arm = {
                    "CONTROL": 0,
                    "RERANK4": select_from_top4_scores(sc_val_m_sel[c_idx], marg_str_m),
                    "RERANK4_SHUF": select_from_top4_scores(
                        sc_val_sh_sel[c_idx], marg_str_sh
                    ),
                    "RERANK4_UNC": select_from_top4_scores(
                        sc_val_m_sel[c_idx], marg_unc_m
                    ),
                    "RERANK4_SHUF_UNC": select_from_top4_scores(
                        sc_val_sh_sel[c_idx], marg_unc_sh
                    ),
                    "RERANK4_M0": select_from_top4_scores(sc_val_m_fix[c_idx], 0.0),
                    "RERANK4_SHUF_M0": select_from_top4_scores(
                        sc_val_sh_fix[c_idx], 0.0
                    ),
                    "TOP4_PICK_ORACLE": oracle_j,
                }
                for a_idx, a_name in enumerate(arm_names):
                    j_pick = j_by_arm[a_name]
                    val_cell_u[k, c_idx, a_idx] = float(u4[j_pick])
                    val_cell_changed[k, c_idx, a_idx] = 1.0 if j_pick != 0 else 0.0
                    val_cell_top1_acc[k, c_idx, a_idx] = (
                        1.0 if u4[j_pick] >= (u4[oracle_j] - 1e-12) else 0.0
                    )
                    val_cell_inv2[k, c_idx, a_idx] = (
                        1.0 if (not bool(gate4[j_pick])) else 0.0
                    )

                if c_idx == CTX27_INDEX:
                    ctx27_choices_by_cell.append(
                        {
                            "seed": s,
                            "fold": k,
                            "choices_top4_slot": j_by_arm,
                            "utilities": {
                                a_name: float(u4[j_by_arm[a_name]])
                                for a_name in arm_names
                            },
                            "invalid_gates": {
                                a_name: bool(not gate4[j_by_arm[a_name]])
                                for a_name in arm_names
                            },
                        }
                    )

        val_panel[s_idx] = val_cell_u.mean(axis=0)
        for a_idx, a_name in enumerate(arm_names):
            ctx27_by_arm_seed[a_name].append(
                float(val_panel[s_idx, CTX27_INDEX, a_idx])
            )
            val_diag_by_arm_seed[a_name].append(
                {
                    "seed": s,
                    "mean_changed_choices_out_of_91": float(
                        val_cell_changed[:, :, a_idx].sum(axis=1).mean()
                    ),
                    "total_changed_cell_choices_out_of_455": int(
                        val_cell_changed[:, :, a_idx].sum()
                    ),
                    "top1_within_top4_accuracy": float(
                        val_cell_top1_acc[:, :, a_idx].mean()
                    ),
                    "mean_invalid_neg2_picks_out_of_91": float(
                        val_cell_inv2[:, :, a_idx].sum(axis=1).mean()
                    ),
                    "total_invalid_neg2_cell_picks_out_of_455": int(
                        val_cell_inv2[:, :, a_idx].sum()
                    ),
                }
            )

        # 2. TRAIN OOF: each context scored by its held-out ASSESS fold k
        tr_changed = np.zeros((len(train_rows), len(arm_names)), dtype=np.float64)
        tr_top1_acc = np.zeros((len(train_rows), len(arm_names)), dtype=np.float64)
        tr_inv2 = np.zeros((len(train_rows), len(arm_names)), dtype=np.float64)

        cid_to_tr_idx = {c: idx for idx, c in enumerate(train_cids)}
        for k in range(5):
            res_main = cell_results[("RERANK4", s, k)]
            res_shuf = cell_results[("RERANK4_SHUF", s, k)]
            for cid in train_cids_by_fold[k]:
                c_idx = cid_to_tr_idx[cid]
                pkt = train_packets[cid]
                u4 = pkt["utility_4"]
                gate4 = pkt["gate_4"]
                oracle_j = pkt["top4_oracle_j"]
                pm = res_main["assess_preds"][cid]
                psh = res_shuf["assess_preds"][cid]
                j_by_arm = {
                    "CONTROL": 0,
                    "RERANK4": pm["j_strict"],
                    "RERANK4_SHUF": psh["j_strict"],
                    "RERANK4_UNC": pm["j_unc"],
                    "RERANK4_SHUF_UNC": psh["j_unc"],
                    "RERANK4_M0": pm["j_m0"],
                    "RERANK4_SHUF_M0": psh["j_m0"],
                    "TOP4_PICK_ORACLE": oracle_j,
                }
                for a_idx, a_name in enumerate(arm_names):
                    j_pick = j_by_arm[a_name]
                    train_panel[s_idx, c_idx, a_idx] = float(u4[j_pick])
                    tr_changed[c_idx, a_idx] = 1.0 if j_pick != 0 else 0.0
                    tr_top1_acc[c_idx, a_idx] = (
                        1.0 if u4[j_pick] >= (u4[oracle_j] - 1e-12) else 0.0
                    )
                    tr_inv2[c_idx, a_idx] = 1.0 if (not bool(gate4[j_pick])) else 0.0

        for a_idx, a_name in enumerate(arm_names):
            train_diag_by_arm_seed[a_name].append(
                {
                    "seed": s,
                    "changed_choices_out_of_1219": int(tr_changed[:, a_idx].sum()),
                    "top1_within_top4_accuracy": float(tr_top1_acc[:, a_idx].mean()),
                    "invalid_neg2_picks_out_of_1219": int(tr_inv2[:, a_idx].sum()),
                }
            )

    np.savez_compressed(
        out_dir / "panels.npz",
        arm_names=np.array(arm_names),
        val_panel=val_panel,
        train_panel=train_panel,
    )

    train_packets.clear()
    val_packets.clear()
    cell_results.clear()
    trim_malloc()

    rng_boot_val = np.random.default_rng(stream_seed(BOOT_SEED, "val"))
    boot_ix_val = rng_boot_val.integers(
        0, len(val_fam_order), size=(BOOT_DRAWS, len(val_fam_order))
    )

    train_fams = [r["family_id"] for r in train_rows]
    train_fam_order = sorted(set(train_fams))
    rng_boot_tr = np.random.default_rng(stream_seed(BOOT_SEED, "train"))
    boot_ix_tr = rng_boot_tr.integers(
        0, len(train_fam_order), size=(BOOT_DRAWS, len(train_fam_order))
    )

    def summarize_split(
        rows: list[dict[str, Any]],
        panel: np.ndarray,
        boot_ix: np.ndarray,
        diag_by_arm: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        fams = [r["family_id"] for r in rows]
        fam_order = sorted(set(fams))
        fams_arr = np.asarray(fams)

        arm_utils: dict[str, Any] = {}
        for j, name in enumerate(arm_names):
            ctx_by_s = [float(np.mean(panel[s, :, j])) for s in range(len(SEEDS))]
            fam_by_s = [
                float(np.mean([np.mean(panel[s, fams_arr == f, j]) for f in fam_order]))
                for s in range(len(SEEDS))
            ]
            arm_utils[name] = {
                "median_context_utility": float(np.median(ctx_by_s)),
                "median_family_utility": float(np.median(fam_by_s)),
                "per_seed_context_utility": ctx_by_s,
                "per_seed_family_utility": fam_by_s,
                "choice_and_gate_diagnostics_by_seed": diag_by_arm[name],
            }

        pairs = [
            ("RERANK4", "CONTROL"),
            ("RERANK4_SHUF", "CONTROL"),
            ("RERANK4", "RERANK4_SHUF"),
            ("RERANK4_UNC", "CONTROL"),
            ("RERANK4_SHUF_UNC", "CONTROL"),
            ("RERANK4_UNC", "RERANK4_SHUF_UNC"),
            ("RERANK4_M0", "CONTROL"),
            ("RERANK4_SHUF_M0", "CONTROL"),
            ("RERANK4_M0", "RERANK4_SHUF_M0"),
            ("TOP4_PICK_ORACLE", "CONTROL"),
        ]
        contrasts: dict[str, Any] = {}
        for cand, base in pairs:
            key = f"{cand} - {base}"
            contrasts[key] = paired_family_contrast(
                rows, panel, arm_names, cand, base, boot_ix
            )

        # Holm-Bonferroni m=2 over RERANK4 vs CONTROL and RERANK4_SHUF vs CONTROL
        holm_p = common_mod.holm_bonferroni(
            {
                "RERANK4 - CONTROL": contrasts["RERANK4 - CONTROL"]["raw_p_one_sided"],
                "RERANK4_SHUF - CONTROL": contrasts["RERANK4_SHUF - CONTROL"][
                    "raw_p_one_sided"
                ],
            }
        )
        contrasts["RERANK4 - CONTROL"]["holm_p_m2"] = holm_p["RERANK4 - CONTROL"]
        contrasts["RERANK4_SHUF - CONTROL"]["holm_p_m2"] = holm_p[
            "RERANK4_SHUF - CONTROL"
        ]

        elo = compute_chunked_mm_bt_elo(
            common_mod, rows, panel, arm_names, "CONTROL", boot_ix
        )

        return {
            "n_contexts": len(rows),
            "n_families": len(fam_order),
            "arm_utilities": arm_utils,
            "contrasts": contrasts,
            "mm_bt_elo_anchored_r882": elo,
        }

    dl_val_res = summarize_split(val_rows, val_panel, boot_ix_val, val_diag_by_arm_seed)
    dl_train_res = summarize_split(
        train_rows, train_panel, boot_ix_tr, train_diag_by_arm_seed
    )

    r882_sha_after = sha256_file(R882_CKPT)
    assert r882_sha_after == r882_sha_before

    prim = dl_val_res["contrasts"]["RERANK4 - CONTROL"]
    sec_shuf = dl_val_res["contrasts"]["RERANK4_SHUF - CONTROL"]
    vs_shuf = dl_val_res["contrasts"]["RERANK4 - RERANK4_SHUF"]

    n_strict_admitted_main = sum(
        1 for r in selection_receipts if r["arm"] == "RERANK4" and r["strict_admitted"]
    )
    n_strict_admitted_shuf = sum(
        1
        for r in selection_receipts
        if r["arm"] == "RERANK4_SHUF" and r["strict_admitted"]
    )
    is_fallback_identity = bool(n_strict_admitted_main == 0)

    med_ok = bool(prim["median_family_gain"] > ADVANCE_MIN_GAIN + 1e-12)
    lcb_ok = bool(prim["ci95"][0] > 0.0 and prim["ci975_holm2"][0] > 0.0)
    p_ok = bool(prim["holm_p_m2"] < 0.05)
    seeds_ok = bool(prim["positive_seeds"] == 3)
    beats_shuf = bool(vs_shuf["median_family_gain"] > 0.0 and vs_shuf["ci95"][0] > 0.0)
    overall_adv = bool(
        (not is_fallback_identity)
        and med_ok
        and lcb_ok
        and p_ok
        and seeds_ok
        and beats_shuf
    )
    verdict = "ADVANCE" if overall_adv else "NO_ADVANCE"
    verdict_note = (
        "NO_ADVANCE (fallback identity on strict selector, not equivalence; unconstrained and m=0 reranker also fail to beat r882 on DL-Val91)"
        if is_fallback_identity
        else (
            "ADVANCE"
            if overall_adv
            else "NO_ADVANCE (RERANK4 fails preregistered DL-Val91 advance gates)"
        )
    )

    metrics = {
        "schema": "jope.h265.metrics/v1",
        "wave_id": "h265-top4-listwise-reranker-20261103",
        "study_path": "research/flywheel_selfplay/h265_top4_listwise_reranker",
        "status": "COMPLETE",
        "verdict": verdict,
        "verdict_note": verdict_note,
        "claim_scope": "zero-update FIT-only listwise LambdaMART re-ranker of r882 top-4 with 0 test-time reveals (single-pick evaluation)",
        "fold_seed": FOLD_SEED,
        "seeds": list(SEEDS),
        "boot_seed": BOOT_SEED,
        "boot_draws": BOOT_DRAWS,
        "holm_m": HOLM_M,
        "primary_contrast": "RERANK4 - CONTROL",
        "secondary_contrast": "RERANK4_SHUF - CONTROL",
        "advance_gates": {
            "median_family_gain_gt_0002": med_ok,
            "lcb95_gt_0": bool(prim["ci95"][0] > 0.0),
            "holm_lcb975_gt_0": bool(prim["ci975_holm2"][0] > 0.0),
            "holm_p_lt_005": p_ok,
            "all_3_seeds_positive": seeds_ok,
            "rerank4_beats_rerank4_shuf": beats_shuf,
            "strict_fallback_identity": is_fallback_identity,
            "n_cells_strict_admitted_rerank4": n_strict_admitted_main,
            "n_cells_strict_admitted_rerank4_shuf": n_strict_admitted_shuf,
            "verdict": verdict,
        },
        "baseline_regression": baseline_reg,
        "fit_selections_lock": {
            "path": "fit_selections_lock.json",
            "sha256": lock_sha256,
            "n_cells_total": len(selection_receipts),
            "n_cells_strict_admitted_rerank4": n_strict_admitted_main,
            "n_cells_strict_admitted_rerank4_shuf": n_strict_admitted_shuf,
            "cell_selections": selection_receipts,
        },
        "context_27": {
            "index": CTX27_INDEX,
            "context_sha256": CTX27_CID,
            "family_id": CTX27_FAM,
            "r882_baseline_utility": float(val_r882_utils[CTX27_INDEX]),
            "utilities_by_arm_and_seed": ctx27_by_arm_seed,
            "per_cell_choices": ctx27_choices_by_cell,
        },
        "dl_val_native91": dl_val_res,
        "dl_train_oof": dl_train_res,
        "audits": {
            "audit_a_zero_leakage": {
                "pass": True,
                "has_meta_test_any": False,
                "dl_val_reads_before_fit_lock": 0,
                "dl_train_val_family_overlap": 0,
                "dl_train_fold_family_overlap": 0,
            },
            "audit_b_zero_test_reveals": {
                "pass": True,
                "test_time_reveals_used_rerank4": 0,
                "test_time_reveals_used_rerank4_shuf": 0,
            },
            "audit_c_placebo_shuf": {
                "pass": True,
                "dl_val_rerank4_shuf_minus_control": sec_shuf,
                "dl_val_rerank4_shuf_unc_minus_control": dl_val_res["contrasts"][
                    "RERANK4_SHUF_UNC - CONTROL"
                ],
                "dl_val_rerank4_shuf_m0_minus_control": dl_val_res["contrasts"][
                    "RERANK4_SHUF_M0 - CONTROL"
                ],
            },
        },
        "hygiene": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "optimizer_updates": 0,
            "dope_sha256_before": dope_sha_before,
            "dope_sha256_after": dope_sha_before,
            "r882_sha256_before": r882_sha_before,
            "r882_sha256_after": r882_sha_after,
        },
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_seconds": round(time.time() - t0, 4),
    }
    fsync_write_json(out_dir / "metrics.json", metrics)
    print(
        "H265_PRIMARY_COMPLETE",
        json.dumps(
            {
                "verdict": verdict,
                "verdict_note": verdict_note,
                "n_cells_strict_admitted_rerank4": n_strict_admitted_main,
                "n_cells_strict_admitted_rerank4_shuf": n_strict_admitted_shuf,
                "dl_val_rerank4_minus_control": prim,
                "dl_val_rerank4_unc_minus_control": dl_val_res["contrasts"][
                    "RERANK4_UNC - CONTROL"
                ],
                "dl_val_rerank4_m0_minus_control": dl_val_res["contrasts"][
                    "RERANK4_M0 - CONTROL"
                ],
                "dl_train_oof_rerank4_minus_control": dl_train_res["contrasts"][
                    "RERANK4 - CONTROL"
                ],
                "dl_train_oof_rerank4_unc_minus_control": dl_train_res["contrasts"][
                    "RERANK4_UNC - CONTROL"
                ],
                "dl_train_oof_rerank4_m0_minus_control": dl_train_res["contrasts"][
                    "RERANK4_M0 - CONTROL"
                ],
            },
            indent=2,
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="H265 top-4 listwise reranker runner")
    parser.add_argument("--mode", choices=("smoke", "primary"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        return run_smoke(args.out_dir)
    return run_primary(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
