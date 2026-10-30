#!/usr/bin/env python3
"""
Phase 2: Full-Corpus Regression Dataset Builder, Pre-Computed Multi-Modal Encodings,
Landmark-V2 12-Probe 5-Fold CV Basket, and CV-Test Agreement Gate.

Builds `/mnt/jope-archive/flywheel-20261023/corpus/`:
  1. Extracts the 232 `outer` context candidate pools from `summary.tar.gz` so all non-empty
     contexts across the 1,780-context regression corpus (`train`, `inner`, `outer`) have
     unified 412D candidate pools.
  2. Extracts a Champion GP DSL Bank (`champion_dsl_bank.json`) from `final.json` & `summary.tar.gz`
     containing executable `(pipe X ...)` S-expressions, 412D program features, and 16D projections.
  3. Loads all pre-computed dataset encodings (`A0` 88D, `Dope-112` 112D, `context2048`/`Kumo-S` 2048D,
     `foundation` 2048D, `hyperion_v6` 512D, `tabpfn35` 2048D, `target` 512D) and applies deterministic
     label-free sign-hash projections (`64D` per high-dim modality) to prevent fold leakage.
  4. Computes the 48D `Landmark-V2` 12-probe 5-fold inner-CV basket directly from `teacher.jtt1.gz`
     `(X_train, y_train)` (strictly never touching `X_test, y_test`), plus loads the `H27` 8-probe
     fingerprints (`qualified-fingerprints850.json`).
  5. Evaluates the CV-Test Agreement Gate using `/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod/outer-test/scores/<cid>.json`
     and `final.json`, admitting outer-test signal on `DL-Train` only when CV and test improvements agree.
  6. Preserves both:
     - The exact 850-context 162-component 2-fold partition (`H27`/`H30`/`W1`) + `H30` cached ZERO priors.
     - The full 1,780-context family-disjoint split (`DL-Train` 80%, `DL-Val` 10%, `DL-Holdout` 10%).
"""

import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import os
import struct
import sys
import tarfile
import time
from pathlib import Path

# The isolated metadata mode must not import the legacy numerical builder or follow its paths.
if __name__ == "__main__" and sys.argv[1:2] == ["qualified-manifest"]:
    sys.dont_write_bytecode = True
    from input_manifest import main as manifest_main

    raise SystemExit(manifest_main(sys.argv[2:]))
else:
    import numpy as np

# Add corrected-study to sys.path (with dont_write_bytecode=True to never touch __pycache__)
sys.dont_write_bytecode = True
_CORRECTED_STUDY_DIR = "/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study"
if os.access(_CORRECTED_STUDY_DIR, os.R_OK | os.X_OK):
    sys.path.insert(0, _CORRECTED_STUDY_DIR)
try:
    from rank_prepare import features as dsl_features_dict
except (ModuleNotFoundError, PermissionError, OSError):
    dsl_features_dict = None

CORPUS_OUT_DIR = Path("/mnt/jope-archive/flywheel-20261023/corpus")
OUTER_POOL_DIR = Path("/mnt/jope-archive/flywheel-20261023/outer_pools")
DL03_DIR = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/deep-learning-dl03")
DL03_POOLS_DIR = DL03_DIR / "pools"
ADMITTED_CONTEXTS_PATH = DL03_DIR / "admitted-input-contexts.json"
PROGRAM_FEATURE_NAMES_PATH = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study/program-feature-names.json")
BUILD_V2_ROOT = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/build-v2")
OUTER_TEST_SCORES_DIR = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod/outer-test/scores")

# Benchmark 850-context paths
ROLE_PLAN_850_PATH = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/research-loop/h27-neural-factorial-implementation/preparation-ROLE_PLAN_PRIVATE.json")
UNION_MAP_850_PATH = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/research-loop/h33-explicit-interaction-source-r1/COMPONENT_UNION_MAP_PRIVATE.json")
FINGERPRINTS_850_PATH = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/research-loop/h27-full-cohort-qualified-features/qualified-fingerprints850.json")
FRESH_COV_850_PATH = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/research-loop/h27-full-clean-default-encoder-saved-qa-execution/merge/fresh-covariates.npz")
H30_PRIORS = {
    0: {
        "predictions": Path("/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021/h30/final-oof-r5/fold-0-zero-container/scientific/predictions-OOF.PRIVATE.npz"),
        "preprocessing": Path("/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021/h30/data-fits-r5/fold-0-zero-container/scientific/preprocessing.json"),
    },
    1: {
        "predictions": Path("/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021/h30/final-oof-r5/fold-1-zero-container/scientific/predictions-OOF.PRIVATE.npz"),
        "preprocessing": Path("/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021/h30/data-fits-r5/fold-1-zero-container/scientific/preprocessing.json"),
    },
}

# Fixed W1 / H33 412 -> 16 sign projection
PROJECTION_SHA = "f732f4ac6a8b2e85cb3fea49e9a9055db7c678c7b8444bd9f082c693a53068fc"


def build_w1_sign_matrix() -> np.ndarray:
    cols = []
    for k in range(16):
        col = [
            1 if (hashlib.sha256(b"jope-context-cross-v1\0" + struct.pack("<II", i, k)).digest()[0] & 1) else -1
            for i in range(412)
        ]
        cols.append(col)
    raw = b"".join(struct.pack("b", v) for c in cols for v in c)
    assert hashlib.sha256(raw).hexdigest() == PROJECTION_SHA, "W1 sign matrix SHA mismatch"
    return np.asarray(cols, dtype=np.float64).T  # shape (412, 16)


W1_SIGN_MATRIX = build_w1_sign_matrix()


def build_hash_sign_matrix(in_dim: int, out_dim: int, domain: bytes) -> np.ndarray:
    """Deterministic label-free Johnson-Lindenstrauss sign projection matrix (in_dim, out_dim)."""
    mat = np.empty((in_dim, out_dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(max(1, in_dim))
    for k in range(out_dim):
        digest_bytes = b"".join(
            hashlib.sha256(domain + struct.pack("<II", k, blk)).digest()
            for blk in range((in_dim + 255) // 256)
        )
        bits = np.unpackbits(np.frombuffer(digest_bytes, dtype=np.uint8))[:in_dim]
        mat[:, k] = np.where(bits == 1, scale, -scale).astype(np.float32)
    return mat


def constrained_fitness(r: dict) -> tuple[float, float, bool]:
    ratio = float(r.get("cv_geomean_rmse_ratio") or 1e9)
    mean = float(r.get("cv_mean_log_ratio") or math.log(max(ratio, 1e-12)))
    tail = float(r.get("cv_tail_log_ratio") or mean)
    sd = float(r.get("cv_log_ratio_sd") or 0.0)
    comp = float(r.get("complexity_open_parens") or str(r.get("dsl") or "").count("("))
    gate = bool(r.get("cv_gate_passed")) and math.isfinite(ratio) and ratio <= 0.90
    if not (math.isfinite(mean) and math.isfinite(tail) and math.isfinite(sd)):
        return 1e6, 30.0, False
    f = (0.8 * mean + 0.1 * tail + 0.1 * sd + 0.0005 * comp) if gate else (100.0 + max(mean, 0.0) + max(tail, 0.0) + max(sd, 0.0))
    return float(f), float(math.log(max(ratio, 1e-12))), gate


def extract_outer_pool_and_champions(item: tuple) -> dict:
    """Extract pool NPZ (if missing) and top gate-passing GP DSL strings for a context."""
    row, feat_names = item
    cid = row["context_sha256"]
    split = row["split"]
    final_path = Path(row["final_path"])
    tar_path = final_path.parent / "summary.tar.gz"

    pool_path = DL03_POOLS_DIR / f"{cid}.npz"
    if not pool_path.exists():
        pool_path = OUTER_POOL_DIR / f"{cid}.npz"

    champions = []
    # Read final.json for top winner DSL if available
    final_winner_dsl = None
    final_test_ratio = None
    final_cv_ratio = None
    if final_path.exists():
        try:
            fdata = json.loads(final_path.read_text())
            sel = fdata.get("selection") or fdata.get("winner") or {}
            prog_obj = sel.get("program") or {}
            dsl_str = prog_obj.get("dsl") or sel.get("dsl")
            if dsl_str:
                final_winner_dsl = str(dsl_str).strip()
                final_cv_ratio = sel.get("cv_geomean_rmse_ratio")
                final_test_ratio = sel.get("test_rmse_ratio")
        except Exception:
            pass

    need_pool_write = not pool_path.exists()
    if not need_pool_write and pool_path.parent == OUTER_POOL_DIR:
        try:
            with np.load(pool_path, allow_pickle=False) as pz:
                if pz["fitness"].shape[0] == 0:
                    need_pool_write = True
        except Exception:
            need_pool_write = True
    by_dsl = {}

    if (need_pool_write or final_winner_dsl is None) and tar_path.exists():
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if not (member.name.endswith("/accepts.jsonl") or member.name == "accepts.jsonl"):
                        continue
                    parts = member.name.split("/")
                    gen = next((int(p[3:]) for p in parts if p.startswith("gen") and p[3:].isdigit()), 0)
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    for line in fobj:
                        if not line.strip():
                            continue
                        raw_rec = json.loads(line)
                        rec = raw_rec.get("result") or raw_rec
                        prog_info = rec.get("program") or {}
                        dsl = str(prog_info.get("dsl") or rec.get("dsl") or "").strip()
                        if not dsl:
                            continue
                        f, lr, gate = constrained_fitness(rec)
                        seq = int(prog_info.get("birth_seq") or rec.get("seq") or 0)
                        cur = by_dsl.get(dsl)
                        if cur is None:
                            by_dsl[dsl] = {
                                "dsl": dsl,
                                "fitness": f,
                                "logratio": lr,
                                "gate": gate,
                                "seq_min": seq,
                                "seq_max": seq,
                                "gen_mask": (1 << gen),
                                "count": 1,
                            }
                        else:
                            if f < cur["fitness"]:
                                cur["fitness"] = f
                                cur["logratio"] = lr
                                cur["gate"] = gate
                            cur["seq_min"] = min(cur["seq_min"], seq)
                            cur["seq_max"] = max(cur["seq_max"], seq)
                            cur["gen_mask"] |= (1 << gen)
                            cur["count"] += 1
        except Exception:
            pass

    if need_pool_write:
        items = sorted(by_dsl.values(), key=lambda x: hashlib.sha256(x["dsl"].encode()).hexdigest())
        n = len(items)
        prog = np.zeros((n, len(feat_names)), dtype=np.float32)
        fitness = np.zeros((n,), dtype=np.float64)
        logratio = np.zeros((n,), dtype=np.float64)
        gate = np.zeros((n,), dtype=np.bool_)
        sha = np.zeros((n,), dtype="S64")
        seq_min = np.zeros((n,), dtype=np.int32)
        seq_max = np.zeros((n,), dtype=np.int32)
        gen_mask = np.zeros((n,), dtype=np.int32)
        count = np.zeros((n,), dtype=np.int16)

        for i, it in enumerate(items):
            d = dsl_features_dict(it["dsl"])
            prog[i] = [float(d.get(k, 0.0)) for k in feat_names]
            fitness[i] = it["fitness"]
            logratio[i] = it["logratio"]
            gate[i] = it["gate"]
            sha[i] = hashlib.sha256(it["dsl"].encode()).hexdigest().encode("ascii")
            seq_min[i] = it["seq_min"]
            seq_max[i] = it["seq_max"]
            gen_mask[i] = it["gen_mask"]
            count[i] = min(it["count"], 32767)

        np.savez_compressed(
            pool_path,
            program=prog,
            fitness=fitness,
            logratio=logratio,
            gate=gate,
            program_sha256=sha,
            birth_seq_min=seq_min,
            birth_seq_max=seq_max,
            generation_mask=gen_mask,
            delivery_count=count,
        )

    # Collect up to 5 champion DSLs from this context (only used when context is in train split)
    if final_winner_dsl:
        d = dsl_features_dict(final_winner_dsl)
        fvec = [float(d.get(k, 0.0)) for k in feat_names]
        champions.append({
            "context_sha256": cid,
            "split": split,
            "dsl": final_winner_dsl,
            "program_sha256": hashlib.sha256(final_winner_dsl.encode()).hexdigest(),
            "cv_ratio": final_cv_ratio,
            "test_ratio": final_test_ratio,
            "feat412": fvec,
        })

    # Inspect pool metadata
    n_cands = 0
    n_pass = 0
    best_pool_f = 100.0
    if pool_path.exists():
        try:
            with np.load(pool_path, allow_pickle=False) as pz:
                n_cands = int(pz["fitness"].shape[0])
                if n_cands > 0:
                    n_pass = int(np.sum(pz["gate"]))
                    best_pool_f = float(np.min(pz["fitness"]))
        except Exception:
            pass

    return {
        "context_sha256": cid,
        "pool_path": str(pool_path),
        "n_candidates": n_cands,
        "n_gate_pass": n_pass,
        "best_pool_fitness": best_pool_f,
        "final_cv_ratio": final_cv_ratio,
        "final_test_ratio": final_test_ratio,
        "champions": champions,
    }


def load_jtt1_train_only(jtt1_gz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read ONLY the training matrix (X_train, y_train, folds) from a JTT1.gz file.
    Strictly stops before reading X_test/y_test bytes!
    """
    with gzip.open(jtt1_gz_path, "rb") as f:
        header = f.read(17)
        if len(header) < 17 or header[:4] != b"JTT1":
            raise ValueError(f"Invalid JTT1 header in {jtt1_gz_path}")
        n_train, n_test, p, k = struct.unpack("<IIIB", header[4:17])
        folds = np.frombuffer(f.read(n_train), dtype=np.uint8).copy()
        train_bytes = n_train * (p + 1) * 8
        train_mat = np.frombuffer(f.read(train_bytes), dtype="<f8").reshape(n_train, p + 1).copy()
    X_train = train_mat[:, :p]
    y_train = train_mat[:, p]
    return X_train, y_train, folds


def compute_landmark_v2_48d(item: tuple) -> tuple[str, np.ndarray]:
    """
    Compute the 48D Landmark-V2 basket on (X_train, y_train) using 5-fold inner CV
    and 1-fold (S1 fold) inner CV across 12 orthogonal modeling probes:
      0: ridge_raw (alpha=1.0)
      1: ridge_heavy (alpha=100.0)
      2: ridge_weak (alpha=0.01)
      3: ridge_rank (rank-transformed X and y, monotone Copula probe)
      4: ridge_symlog (sign(X)*log1p(|X|) heavy-tail probe)
      5: ridge_poly2 ([X, clip(X^2, 0, 25)] quadratic basis probe)
      6: svd_pcr (Principal Component Regression on top-k singular vectors)
      7: lasso_soft (Soft-thresholded sparse coordinate step from ridge)
      8: knn_k1 (1-NN local smoother probe)
      9: knn_k16 (16-NN inverse-distance local smoother probe)
      10: stump_boost_d1 (Additive coordinate ridge-residual stumps, depth 1)
      11: tree_inter_d2 (Pairwise interaction residual tree probe, depth 2)
    Returns 48D float32 feature vector:
      [0:12]  5-fold mean log(RMSE_probe / RMSE_ridge_raw)
      [12:24] 1-vs-5 fold stability shift (mean_5fold - mean_1fold)
      [24:36] 5-fold dispersion std(log(RMSE_probe / RMSE_ridge_raw))
      [36:48] 12 mechanistic dataset/routing contrasts
    """
    cid, jtt1_path = item
    out = np.zeros(48, dtype=np.float32)
    try:
        X_raw, y_raw, s1_folds = load_jtt1_train_only(jtt1_path)
    except Exception:
        return cid, out

    n, p = X_raw.shape
    if n < 10 or p < 1:
        return cid, out

    # Cap n at 1024 and p at 96 via deterministic stride if huge, for fast vectorized CV
    if n > 1024:
        idx = np.linspace(0, n - 1, 1024, dtype=np.int64)
        X_raw = X_raw[idx]
        y_raw = y_raw[idx]
        s1_folds = s1_folds[idx]
        n = 1024
    if p > 96:
        col_var = np.nanvar(X_raw, axis=0)
        top_cols = np.argsort(-np.nan_to_num(col_var, nan=0.0))[:96]
        top_cols.sort()
        X_raw = X_raw[:, top_cols]
        p = 96

    X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_clean = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

    def eval_fold_probes(tr_mask: np.ndarray, va_mask: np.ndarray) -> np.ndarray:
        X_tr, y_tr = X_clean[tr_mask], y_clean[tr_mask]
        X_va, y_va = X_clean[va_mask], y_clean[va_mask]
        n_tr, n_va = len(y_tr), len(y_va)
        if n_tr < 4 or n_va < 2:
            return np.zeros(12, dtype=np.float64)

        # Standardize using train fold only
        x_mu = X_tr.mean(axis=0, keepdims=True)
        x_sd = X_tr.std(axis=0, keepdims=True)
        x_sd = np.where(x_sd < 1e-8, 1.0, x_sd)
        Z_tr = np.clip((X_tr - x_mu) / x_sd, -10.0, 10.0)
        Z_va = np.clip((X_va - x_mu) / x_sd, -10.0, 10.0)

        y_mu = float(y_tr.mean())
        y_sd = float(y_tr.std())
        if y_sd < 1e-8:
            return np.zeros(12, dtype=np.float64)
        yc_tr = (y_tr - y_mu) / y_sd
        yc_va = (y_va - y_mu) / y_sd

        def rmse(pred: np.ndarray) -> float:
            err = yc_va - pred
            return float( math.sqrt(max(float(np.mean(err * err)), 1e-12)) )

        # SVD of Z_tr once solves ridge_raw, ridge_heavy, ridge_weak, and svd_pcr simultaneously!
        try:
            U, s, Vt = np.linalg.svd(Z_tr, full_matrices=False)
        except Exception:
            return np.zeros(12, dtype=np.float64)

        uty = U.T @ yc_tr
        s2 = s * s

        def ridge_from_svd(alpha: float) -> tuple[np.ndarray, np.ndarray]:
            w = Vt.T @ ((s / (s2 + alpha)) * uty)
            return Z_tr @ w, Z_va @ w, w

        pred_tr_raw, pred_va_raw, w_raw = ridge_from_svd(1.0)
        _, pred_va_heavy, _ = ridge_from_svd(100.0)
        _, pred_va_weak, _ = ridge_from_svd(0.01)

        base_rmse = rmse(pred_va_raw)
        rmses = np.empty(12, dtype=np.float64)
        rmses[0] = base_rmse
        rmses[1] = rmse(pred_va_heavy)
        rmses[2] = rmse(pred_va_weak)

        # 3: ridge_rank (monotone rank features)
        order_tr = np.argsort(np.argsort(Z_tr, axis=0), axis=0).astype(np.float64) / max(1, n_tr - 1) - 0.5
        # Quantile map validation features against train
        R_tr = order_tr * 2.0
        R_va = np.tanh(Z_va * 0.7)
        try:
            w_rk = np.linalg.solve(R_tr.T @ R_tr + 2.0 * np.eye(p), R_tr.T @ yc_tr)
            rmses[3] = rmse(R_va @ w_rk)
        except Exception:
            rmses[3] = base_rmse

        # 4: ridge_symlog
        L_tr = np.sign(Z_tr) * np.log1p(np.abs(Z_tr))
        L_va = np.sign(Z_va) * np.log1p(np.abs(Z_va))
        try:
            w_sl = np.linalg.solve(L_tr.T @ L_tr + 2.0 * np.eye(p), L_tr.T @ yc_tr)
            rmses[4] = rmse(L_va @ w_sl)
        except Exception:
            rmses[4] = base_rmse

        # 5: ridge_poly2 ([Z, (Z^2 - 1)/sqrt(2)] on top-32 variance features)
        p_poly = min(p, 32)
        P_tr = np.hstack([Z_tr, (Z_tr[:, :p_poly] ** 2 - 1.0) * 0.7071])
        P_va = np.hstack([Z_va, (Z_va[:, :p_poly] ** 2 - 1.0) * 0.7071])
        try:
            w_p2 = np.linalg.solve(P_tr.T @ P_tr + 5.0 * np.eye(P_tr.shape[1]), P_tr.T @ yc_tr)
            rmses[5] = rmse(P_va @ w_p2)
        except Exception:
            rmses[5] = base_rmse

        # 6: svd_pcr (top k = max(1, min(8, len(s)//2)) singular vectors)
        k_pcr = max(1, min(8, len(s) // 2))
        w_pcr = Vt[:k_pcr].T @ (uty[:k_pcr] / np.maximum(s[:k_pcr], 1e-6))
        rmses[6] = rmse(Z_va @ w_pcr)

        # 7: lasso_soft (soft-thresholded sparse weights + refit)
        thresh = float(np.median(np.abs(w_raw))) * 1.25
        w_sparse = np.sign(w_raw) * np.maximum(np.abs(w_raw) - thresh, 0.0)
        rmses[7] = rmse(Z_va @ w_sparse)

        # 8 & 9: knn_k1 and knn_k16 in standardized space
        # Pairwise squared Euclidean distance: ||Z_va||^2 + ||Z_tr||^2 - 2 Z_va Z_tr^T
        va_norm = np.sum(Z_va * Z_va, axis=1, keepdims=True)
        tr_norm = np.sum(Z_tr * Z_tr, axis=1, keepdims=True).T
        d2 = np.maximum(va_norm + tr_norm - 2.0 * (Z_va @ Z_tr.T), 0.0) / max(1, p)
        nn1_idx = np.argmin(d2, axis=1)
        rmses[8] = rmse(yc_tr[nn1_idx])

        k16 = min(16, n_tr)
        part_idx = np.argpartition(d2, k16 - 1, axis=1)[:, :k16]
        part_d2 = np.take_along_axis(d2, part_idx, axis=1)
        wts = 1.0 / (1e-3 + part_d2)
        wts /= np.sum(wts, axis=1, keepdims=True)
        pred_k16 = np.sum(wts * yc_tr[part_idx], axis=1)
        rmses[9] = rmse(pred_k16)

        # 10: stump_boost_d1 (additive nonlinear residual bins on top-8 correlated features)
        res_tr = yc_tr - pred_tr_raw
        corr = np.abs(Z_tr.T @ res_tr)
        top8 = np.argsort(-corr)[:min(8, p)]
        pred_boost1_tr = pred_tr_raw.copy()
        pred_boost1_va = pred_va_raw.copy()
        for c_idx in top8:
            r_cur = yc_tr - pred_boost1_tr
            pos_m = Z_tr[:, c_idx] > 0.0
            neg_m = ~pos_m
            v_pos = float(r_cur[pos_m].mean()) if np.any(pos_m) else 0.0
            v_neg = float(r_cur[neg_m].mean()) if np.any(neg_m) else 0.0
            pred_boost1_tr += 0.35 * np.where(pos_m, v_pos, v_neg)
            pred_boost1_va += 0.35 * np.where(Z_va[:, c_idx] > 0.0, v_pos, v_neg)
        rmses[10] = rmse(pred_boost1_va)

        # 11: tree_inter_d2 (depth-2 quadrant interaction residuals on top pairs)
        pred_boost2_tr = pred_boost1_tr.copy()
        pred_boost2_va = pred_boost1_va.copy()
        top4 = top8[:min(4, len(top8))]
        for i_a in range(len(top4)):
            for i_b in range(i_a + 1, len(top4)):
                ca, cb = top4[i_a], top4[i_b]
                r_cur = yc_tr - pred_boost2_tr
                for sa in (True, False):
                    for sb in (True, False):
                        m_tr = (Z_tr[:, ca] > 0.0) == sa
                        m_tr &= (Z_tr[:, cb] > 0.0) == sb
                        if np.sum(m_tr) >= 3:
                            val = float(r_cur[m_tr].mean()) * 0.25
                            m_va = ((Z_va[:, ca] > 0.0) == sa) & ((Z_va[:, cb] > 0.0) == sb)
                            pred_boost2_tr[m_tr] += val
                            pred_boost2_va[m_va] += val
        rmses[11] = rmse(pred_boost2_va)

        log_ratios = np.log(np.clip(rmses / max(base_rmse, 1e-12), 0.1, 10.0))
        return log_ratios

    # 5-fold deterministic interleaved split
    fold_ids = np.arange(n) % 5
    fold5_mat = np.zeros((5, 12), dtype=np.float64)
    for k in range(5):
        va_m = fold_ids == k
        tr_m = ~va_m
        fold5_mat[k] = eval_fold_probes(tr_m, va_m)

    mean_5 = fold5_mat.mean(axis=0)
    std_5 = fold5_mat.std(axis=0)

    # 1-fold S1 split (using JTT1's s1_folds == 0 as train, == 1 as val)
    va_1 = s1_folds == 1
    tr_1 = ~va_1
    if np.sum(tr_1) >= 4 and np.sum(va_1) >= 2:
        mean_1 = eval_fold_probes(tr_1, va_1)
    else:
        mean_1 = fold5_mat[0]

    shift_1v5 = mean_5 - mean_1

    # 12 mechanistic contrasts
    best_nonlin = min(mean_5[5], mean_5[8], mean_5[9], mean_5[10], mean_5[11])
    nonlin_gap = best_nonlin - mean_5[0]
    inter_gap = mean_5[11] - mean_5[10]
    rank_gap = mean_5[3] - mean_5[0]
    symlog_gap = mean_5[4] - mean_5[0]
    poly_gap = mean_5[5] - mean_5[0]
    pcr_gap = mean_5[6] - mean_5[0]
    sparse_gap = mean_5[7] - mean_5[0]
    smooth_gap = mean_5[9] - mean_5[8]
    shrink_gap = mean_5[1] - mean_5[2]
    log_np_ratio = math.log(max(n, 1) / max(p, 1))

    y_std = float(np.std(y_clean))
    if y_std > 1e-8:
        yz = (y_clean - float(np.mean(y_clean))) / y_std
        y_skew = float(np.clip(np.mean(yz ** 3), -10.0, 10.0))
        y_kurt = float(np.clip(np.log1p(max(0.0, float(np.mean(yz ** 4)) - 3.0)), 0.0, 10.0))
    else:
        y_skew, y_kurt = 0.0, 0.0

    contrasts = np.array([
        nonlin_gap, inter_gap, rank_gap, symlog_gap,
        poly_gap, pcr_gap, sparse_gap, smooth_gap,
        shrink_gap, log_np_ratio, y_skew, y_kurt,
    ], dtype=np.float32)

    out[0:12] = mean_5.astype(np.float32)
    out[12:24] = shift_1v5.astype(np.float32)
    out[24:36] = std_5.astype(np.float32)
    out[36:48] = contrasts
    return cid, np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    t0 = time.monotonic()
    CORPUS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTER_POOL_DIR.mkdir(parents=True, exist_ok=True)

    admitted_rows = json.loads(ADMITTED_CONTEXTS_PATH.read_text())
    feat_names = json.loads(PROGRAM_FEATURE_NAMES_PATH.read_text())
    print(f"[Phase 2] Loaded {len(admitted_rows)} contexts from admitted-input-contexts.json, {len(feat_names)} DSL features.")

    # Step 1: Extract outer pools & inspect all 1,780 pools + champion DSLs in parallel
    print("[Phase 2] Extracting outer pools & champion GP DSL bank across 24 workers...")
    pool_meta_by_cid = {}
    all_champions = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
        tasks = [(r, feat_names) for r in admitted_rows]
        for res in ex.map(extract_outer_pool_and_champions, tasks, chunksize=16):
            pool_meta_by_cid[res["context_sha256"]] = res
            all_champions.extend(res["champions"])

    non_empty_cids = [cid for cid, m in pool_meta_by_cid.items() if m["n_candidates"] > 0]
    print(f"[Phase 2] Candidate pools ready: {len(non_empty_cids)} / {len(admitted_rows)} non-empty contexts, {len(all_champions)} champion DSLs extracted.")

    # Step 2: Load 850-context benchmark metadata & H30 ZERO priors
    role_plan_850 = json.loads(ROLE_PLAN_850_PATH.read_text())
    union_map_850 = json.loads(UNION_MAP_850_PATH.read_text())
    fp850_data = json.loads(FINGERPRINTS_850_PATH.read_text())
    fp850_by_cid = {r["context_sha256"]: r for r in fp850_data["rows"]}

    benchmark_850_by_cid = {}
    by_src_idx = {r["source_row_index"]: r for r in role_plan_850["rows"]}
    fold_roles_by_src = {0: {}, 1: {}}
    for f in role_plan_850["folds"]:
        f_idx = f["fold"]
        for role_name in ("gradient", "monitor", "OOF"):
            for s_idx in f[f"{role_name}_source_row_indices"]:
                fold_roles_by_src[f_idx][s_idx] = role_name

    union_by_comp = union_map_850["component_to_union"]
    for s_idx, r in by_src_idx.items():
        cid = r["context_sha256"]
        benchmark_850_by_cid[cid] = {
            "source_row_index": s_idx,
            "component": r["component"],
            "union_component": union_by_comp[r["component"]],
            "family_id": r["family_id"],
            "fold0_role": fold_roles_by_src[0][s_idx],
            "fold1_role": fold_roles_by_src[1][s_idx],
        }

    # Load H30 ZERO prior scores & target scales for the 850 benchmark contexts
    h30_prior_by_cid = {}
    h30_scales = {}
    for fold_idx in (0, 1):
        prep = json.loads(H30_PRIORS[fold_idx]["preprocessing"].read_text())
        S = float(prep["target_scale"])
        h30_scales[fold_idx] = S
        with np.load(H30_PRIORS[fold_idx]["predictions"], allow_pickle=False) as npz:
            cids_arr = [x.decode("ascii") for x in npz["context_sha256"].tolist()]
            offsets = npz["offsets"].tolist()
            scores = npz["score"]
            pids = npz["program_sha256"]
            for j, cid in enumerate(cids_arr):
                lo, hi = offsets[j], offsets[j + 1]
                h30_prior_by_cid[cid] = {
                    "fold": fold_idx,
                    "scale": S,
                    "program_sha256": [x.decode("ascii") for x in pids[lo:hi].tolist()],
                    "prior_cost": (-scores[lo:hi]).astype(np.float64),
                }
    print(f"[Phase 2] Loaded H30 ZERO priors for {len(h30_prior_by_cid)} benchmark contexts (scales: {h30_scales}).")

    # Step 3: Family-disjoint full-corpus split across all 1,780 contexts
    # Ensure any family in benchmark_850 respects its fold assignment, and partition all 282 families into DL-Train (80%), DL-Val (10%), DL-Holdout (10%)
    families = sorted({r["family_id"] for r in admitted_rows})
    family_split = {}
    for fam in families:
        h = int.from_bytes(hashlib.sha256(b"jope-flywheel-family-split-v1\0" + fam.encode()).digest()[:4], "little") % 100
        if h < 80:
            family_split[fam] = "DL-Train"
        elif h < 90:
            family_split[fam] = "DL-Val"
        else:
            family_split[fam] = "DL-Holdout"

    # Step 4 & 5: Reuse context_features.npz if already computed, else compute
    N_ALL = len(admitted_rows)
    cid_to_idx = {r["context_sha256"]: i for i, r in enumerate(admitted_rows)}
    ctx_feat_path = CORPUS_OUT_DIR / "context_features.npz"
    if ctx_feat_path.exists():
        print("[Phase 2] Reusing cached context_features.npz (A0, Dope-112, MM-Proj, H27-FP16, Landmark-V2-48)...")
        with np.load(ctx_feat_path, allow_pickle=False) as cf:
            a0_mat = cf["a0"]
            dope112_mat = cf["dope112"]
            mm_proj_mat = cf["mm_proj"]
            h27_fp_mat = cf["h27_fp16"]
            landmark_v2_mat = cf["landmark_v2_48"]
    else:
        print("[Phase 2] Loading pre-computed encodings (A0, Dope, Kumo-S/Context2048, Foundation, Hyperion-V6, TabPFN35, Target)...")
        view_dir_map = {}
        for sub in sorted(BUILD_V2_ROOT.glob("*/*")):
            if sub.is_dir():
                view_dir_map[sub.name] = sub

        dl03_cov = np.load(DL03_DIR / "covariates.npz", allow_pickle=True)
        dl03_dope_by_cid = {str(c): dl03_cov["dope"][i] for i, c in enumerate(dl03_cov["context_sha256"])}

        proj_ctx2048 = build_hash_sign_matrix(2048, 64, b"jope-proj-ctx2048-v1\0")
        proj_found2048 = build_hash_sign_matrix(2048, 64, b"jope-proj-found2048-v1\0")
        proj_hyp512 = build_hash_sign_matrix(512, 64, b"jope-proj-hyp512-v1\0")
        proj_tab2048 = build_hash_sign_matrix(2048, 64, b"jope-proj-tab2048-v1\0")
        proj_tgt512 = build_hash_sign_matrix(512, 64, b"jope-proj-tgt512-v1\0")

        a0_mat = np.zeros((N_ALL, 88), dtype=np.float32)
        dope112_mat = np.zeros((N_ALL, 112), dtype=np.float32)
        mm_proj_mat = np.zeros((N_ALL, 320), dtype=np.float32)
        h27_fp_mat = np.zeros((N_ALL, 16), dtype=np.float32)

        for idx, r in enumerate(admitted_rows):
            cid = r["context_sha256"]
            vid = r["view_id"]
            a0_p = Path(r["a0_path"])
            if a0_p.exists():
                a0_vec = np.frombuffer(a0_p.read_bytes()[:352], dtype="<f4").copy()
                a0_mat[idx] = np.nan_to_num(a0_vec, nan=0.0, posinf=0.0, neginf=0.0)

            vdir = view_dir_map.get(vid)
            dope_vec = None
            if vdir and (vdir / "dope.npy").exists():
                arr = np.load(vdir / "dope.npy")
                if arr.shape[0] >= 112:
                    dope_vec = arr[:112].astype(np.float32)
            if dope_vec is None and cid in dl03_dope_by_cid:
                dope_vec = dl03_dope_by_cid[cid][:112].astype(np.float32)
            if dope_vec is not None:
                dope112_mat[idx] = np.nan_to_num(dope_vec, nan=0.0, posinf=0.0, neginf=0.0)

            if vdir is not None:
                for m_idx, (fname, expected_d, pmat) in enumerate([
                    ("context2048.npy", 2048, proj_ctx2048),
                    ("foundation.npy", 2048, proj_found2048),
                    ("hyperion_v6.npy", 512, proj_hyp512),
                    ("tabpfn35.npy", 2048, proj_tab2048),
                    ("target.npy", 512, proj_tgt512),
                ]):
                    fpath = vdir / fname
                    if fpath.exists():
                        try:
                            vec = np.load(fpath).astype(np.float32).reshape(-1)
                            if vec.shape[0] == expected_d:
                                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                                norm = float(np.linalg.norm(vec))
                                if norm > 1e-8:
                                    vec = vec / norm
                                mm_proj_mat[idx, m_idx * 64 : (m_idx + 1) * 64] = vec @ pmat
                        except Exception:
                            pass

            if cid in fp850_by_cid:
                fp_row = fp850_by_cid[cid]
                h27_fp_mat[idx, :8] = np.asarray(fp_row["five"], dtype=np.float32)
                h27_fp_mat[idx, 8:16] = np.asarray(fp_row["one"], dtype=np.float32)

        print("[Phase 2] Computing 48D Landmark-V2 12-probe 5-fold CV basket across 1,780 JTT1 training matrices...")
        landmark_v2_mat = np.zeros((N_ALL, 48), dtype=np.float32)
        with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
            jtt_tasks = [(r["context_sha256"], r["teacher"]["path"]) for r in admitted_rows]
            for cid, lm_vec in ex.map(compute_landmark_v2_48d, jtt_tasks, chunksize=16):
                landmark_v2_mat[cid_to_idx[cid]] = lm_vec

        map8 = [0, 1, 3, 7, 8, 9, 10, 11]
        for idx, r in enumerate(admitted_rows):
            if r["context_sha256"] not in fp850_by_cid:
                five_proxy = landmark_v2_mat[idx, map8]
                one_proxy = five_proxy - landmark_v2_mat[idx, [12 + m for m in map8]]
                h27_fp_mat[idx, :8] = five_proxy
                h27_fp_mat[idx, 8:16] = one_proxy

    # Step 6: Evaluate CV-Test Agreement Gate across contexts with outer-test scores
    print("[Phase 2] Evaluating CV-Test Agreement Gate across outer-test score files and final.json...")
    cv_test_agreement = {}
    agreed_count = 0
    overfit_count = 0
    total_with_test = 0

    for idx, r in enumerate(admitted_rows):
        cid = r["context_sha256"]
        pmeta = pool_meta_by_cid[cid]
        cv_ratio = pmeta["final_cv_ratio"]
        test_ratio = pmeta["final_test_ratio"]

        # Check outer-test/scores/<cid>.json as well
        ot_path = OUTER_TEST_SCORES_DIR / f"{cid}.json"
        ot_candidates = {}
        if ot_path.exists():
            try:
                ot_data = json.loads(ot_path.read_text())
                off_t = ot_data.get("official_test") or {}
                if off_t.get("status") == "completed":
                    m_rmse = (off_t.get("metrics") or {}).get("rmse")
                    b_rmse = (off_t.get("baseline_metrics") or {}).get("rmse")
                    if m_rmse is not None and b_rmse is not None and float(b_rmse) > 1e-12:
                        test_ratio = float(m_rmse) / float(b_rmse)
                        sel_sha = ot_data.get("selected_program_sha256")
                        if sel_sha:
                            ot_candidates[sel_sha] = test_ratio
            except Exception:
                pass

        if cv_ratio is not None and test_ratio is not None and math.isfinite(cv_ratio) and math.isfinite(test_ratio) and cv_ratio > 0 and test_ratio > 0:
            total_with_test += 1
            log_cv = math.log(max(cv_ratio, 1e-6))
            log_test = math.log(max(test_ratio, 1e-6))
            sign_agree = (log_cv <= 0.0 and log_test <= 0.0) or (log_cv > 0.0 and log_test > 0.0)
            mag_agree = abs(log_test - log_cv) <= 0.35
            agreed = bool(sign_agree and mag_agree)
            overfit = bool(log_cv < -0.02 and log_test > 0.02)
            if agreed:
                agreed_count += 1
            if overfit:
                overfit_count += 1
            cv_test_agreement[cid] = {
                "has_test": True,
                "cv_ratio": float(cv_ratio),
                "test_ratio": float(test_ratio),
                "log_cv": float(log_cv),
                "log_test": float(log_test),
                "sign_agree": bool(sign_agree),
                "mag_agree": bool(mag_agree),
                "agreed": agreed,
                "overfit": overfit,
                "candidate_test_ratios": ot_candidates,
            }
        else:
            cv_test_agreement[cid] = {
                "has_test": False,
                "cv_ratio": float(cv_ratio) if cv_ratio is not None else None,
                "test_ratio": None,
                "agreed": False,
                "overfit": False,
                "candidate_test_ratios": ot_candidates,
            }

    print(
        f"[Phase 2] CV-Test Agreement Gate summary: {total_with_test} contexts with outer-test scores, "
        f"{agreed_count} agreed ({100.0 * agreed_count / max(1, total_with_test):.1f}%), "
        f"{overfit_count} CV-overfit flagged ({100.0 * overfit_count / max(1, total_with_test):.1f}%)."
    )

    # Save unified context encodings & metadata
    np.savez_compressed(
        CORPUS_OUT_DIR / "context_features.npz",
        context_sha256=np.array([r["context_sha256"] for r in admitted_rows], dtype="S64"),
        a0=a0_mat,
        dope112=dope112_mat,
        mm_proj=mm_proj_mat,
        h27_fp16=h27_fp_mat,
        landmark_v2_48=landmark_v2_mat,
    )

    # Save H30 priors NPZ for fast loading
    np.savez_compressed(
        CORPUS_OUT_DIR / "h30_zero_priors_850.npz",
        scales=np.array([h30_scales[0], h30_scales[1]], dtype=np.float64),
        **{f"prior_{cid}": h30_prior_by_cid[cid]["prior_cost"] for cid in h30_prior_by_cid},
    )

    # Save Champion DSL bank (filter to DL-Train contexts or store with split tag so fold-hygiene is enforced at retrieval time)
    for ch in all_champions:
        cid = ch["context_sha256"]
        row_idx = cid_to_idx[cid]
        fam = admitted_rows[row_idx]["family_id"]
        ch["family_id"] = fam
        ch["dl_split"] = family_split[fam]
        ch["in_850"] = cid in benchmark_850_by_cid
        if cid in benchmark_850_by_cid:
            ch["fold0_role"] = benchmark_850_by_cid[cid]["fold0_role"]
            ch["fold1_role"] = benchmark_850_by_cid[cid]["fold1_role"]
            ch["union_component"] = benchmark_850_by_cid[cid]["union_component"]
        f412 = np.asarray(ch["feat412"], dtype=np.float64)
        proj16 = (f412 @ W1_SIGN_MATRIX) / math.sqrt(412.0)
        ch["proj16"] = proj16.tolist()
        ch["a0_summary"] = a0_mat[row_idx, :16].tolist()
        ch["landmark_summary"] = landmark_v2_mat[row_idx, :12].tolist()

    (CORPUS_OUT_DIR / "champion_dsl_bank.json").write_text(json.dumps(all_champions, indent=2))

    # Save master corpus index
    corpus_records = []
    for idx, r in enumerate(admitted_rows):
        cid = r["context_sha256"]
        fam = r["family_id"]
        pmeta = pool_meta_by_cid[cid]
        ag = cv_test_agreement[cid]
        b850 = benchmark_850_by_cid.get(cid)
        corpus_records.append({
            "index": idx,
            "context_sha256": cid,
            "dataset_id": r.get("dataset_id") or r.get("dataset_hash", ""),
            "view_id": r["view_id"],
            "family_id": fam,
            "source_split": r["split"],
            "dl_split": family_split[fam],
            "teacher_jtt1_path": r["teacher"]["path"],
            "final_path": r["final_path"],
            "pool_path": pmeta["pool_path"],
            "n_candidates": pmeta["n_candidates"],
            "n_gate_pass": pmeta["n_gate_pass"],
            "best_pool_fitness": pmeta["best_pool_fitness"],
            "cv_test_gate": {
                "has_test": ag["has_test"],
                "cv_ratio": ag["cv_ratio"],
                "test_ratio": ag["test_ratio"],
                "agreed": ag["agreed"],
                "overfit": ag["overfit"],
            },
            "benchmark_850": b850,
        })

    manifest = {
        "schema": "jope.flywheel-corpus/v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "total_contexts": len(admitted_rows),
        "non_empty_contexts": len(non_empty_cids),
        "benchmark_850_contexts": len(benchmark_850_by_cid),
        "total_families": len(families),
        "dl_split_counts": {
            sp: sum(1 for rec in corpus_records if rec["dl_split"] == sp and rec["n_candidates"] > 0)
            for sp in ("DL-Train", "DL-Val", "DL-Holdout")
        },
        "source_split_counts": {
            sp: sum(1 for rec in corpus_records if rec["source_split"] == sp and rec["n_candidates"] > 0)
            for sp in ("train", "inner", "outer")
        },
        "cv_test_gate_summary": {
            "total_with_test": total_with_test,
            "agreed_count": agreed_count,
            "overfit_count": overfit_count,
            "agreement_rate": round(agreed_count / max(1, total_with_test), 4),
        },
        "feature_dimensions": {
            "a0": 88,
            "dope112": 112,
            "mm_proj": 320,
            "h27_fp16": 16,
            "landmark_v2_48": 48,
            "total_context_dim": 88 + 112 + 320 + 16 + 48,
            "program_dsl_dim": 412,
            "w1_proj_dim": 16,
        },
        "records": corpus_records,
    }
    (CORPUS_OUT_DIR / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[Phase 2] Wrote {CORPUS_OUT_DIR / 'corpus_manifest.json'} in {manifest['elapsed_seconds']}s.")
    print("  DL split non-empty counts:", manifest["dl_split_counts"])
    print("  Source split non-empty counts:", manifest["source_split_counts"])


if __name__ == "__main__":
    main()
