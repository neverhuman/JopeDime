"""FIT-row-only conditioning, Ultra research view, and 196D R_packet_v1 extractor.

Implements:
  1. Strict FIT-only JTT1 reader refusing nonzero query rows (`nq != 0`) and
     query-key dictionaries (`FORBIDDEN_QUERY_KEYS`).
  2. Deterministic 5-fold FIT-only native 23-landmark CV panel (`landmarks-native-v1`),
     including the 5 `ultra` fine probes (`gbdt`, `knn1`, `knn16`, `net`, `gbdt_quarter`)
     and per-table fine-probe latency telemetry (`U-S3`).
  3. All Stage-1A + DOPE conditioning paths on frozen `dope_mlc_v3_multitask_best`:
       - Fast GPU-sketch `Stage1ADistilledBridge` (`landmark_bridge` & `nolm_bridge`)
       - Pinned H58/H469 `FusedLandmarkWorkspace` (`hpc_high`, bit-exact parity `DVP-S2`)
       - Native 23-landmark `high` (`native_high`) and `ultra` (`native_ultra`, strictly
         enforcing `conditioning_level == "ultra"` so `high`-masked `vec256` never enters `ultra`).
  4. Scale-invariant, column-permutation-invariant, extrema-free `dope.packet-features/v1`
     (`196D`, with `dope.packet-global/v1` as its `28D` prefix) and `R_packet_notarget`
     (`include_target_bundle=False`, `P-S2` ablation).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import operator
from pathlib import Path
import struct
import sys
import time
from typing import Any

import os

import numpy as np
import torch

MICRO_ROOT = Path(os.environ.get("JOPE_MICRO_ROOT", Path.home() / "JopeDimeMicro"))
DATA_ROOT = Path(os.environ.get("JOPE_DATA_ROOT", Path.home() / "jope-data"))
MICRO_PYTHON = Path(os.environ.get("JOPE_MICRO_PYTHON", MICRO_ROOT / "python"))
if MICRO_PYTHON.is_dir() and str(MICRO_PYTHON) not in sys.path:
    sys.path.insert(0, str(MICRO_PYTHON))

DEFAULT_LANDMARK_NAMES_23: tuple[str, ...] = (
    "linear",
    "asinh",
    "signed_sqrt",
    "poly3",
    "tanh",
    "abs",
    "hinge7",
    "step11",
    "periodic",
    "rbf3",
    "row_summary",
    "pairs8",
    "rff192",
    "missing_only",
    "best1_step",
    "stump",
    "tree_d3",
    "gbdt",
    "knn1",
    "knn16",
    "net",
    "linear_quarter",
    "gbdt_quarter",
)

try:
    from research.levels.building_block_landmarks import build_v3_conditioning_bundle  # noqa: E402
    from research.levels.descriptors import LANDMARK_NAMES  # noqa: E402
    from research.levels.multilevel_model import MultiLevelLandmarkModel  # noqa: E402
    from research.levels.scaling import FitOnlyTableTransform  # noqa: E402
except ModuleNotFoundError:  # Offline CI environment without JopeDimeMicro checkout
    build_v3_conditioning_bundle = None  # type: ignore[assignment]
    LANDMARK_NAMES = DEFAULT_LANDMARK_NAMES_23  # type: ignore[assignment]
    MultiLevelLandmarkModel = Any  # type: ignore[assignment,misc]
    FitOnlyTableTransform = None  # type: ignore[assignment]

RESEARCH_ENCODER_VIEW_RECIPE = "research-encoder-view/v1"
PACKET_GLOBAL_SCHEMA = "dope.packet-global/v1"
PACKET_GLOBAL_WIDTH = 28
PACKET_FEATURES_SCHEMA = "dope.packet-features/v1"
PACKET_FEATURES_WIDTH = 196

CANONICAL_GATE_KEYS_14: tuple[str, ...] = (
    "asinh_atan_w",
    "copula_retention",
    "het_scale_w",
    "hinge_w",
    "monotone_w",
    "neural_w",
    "noise_scale",
    "pair_w",
    "periodic_fract_w",
    "regime_sharpness",
    "rff_w",
    "ridge_alpha",
    "row_summary_w",
    "tree_w",
)

PACKET_GLOBAL_FIELDS: tuple[str, ...] = (
    *CANONICAL_GATE_KEYS_14,
    "residual_std_scaled",
    "neural_w_active",
    "res_net_mean_tanh",
    "is_regression",
    "is_binary",
    "is_multiclass",
    "log2_k_classes_norm",
    "class_entropy_norm",
    "log10_n_train_norm",
    "log2_p_use_norm",
    "aspect_ratio_tanh",
    "n_trees_norm",
    "n_pairs_norm",
    "level_code",
)

LEVEL_CODE = {"high": 0.0, "ultra": 1.0}
FORBIDDEN_QUERY_KEYS: tuple[str, ...] = (
    "X_query",
    "y_query",
    "query_rows",
    "query_features",
    "X_test",
    "y_test",
    "X_qry",
    "y_qry",
    "X_val",
    "y_val",
    "X_tune",
    "y_tune",
    "query_targets",
    "test_rows",
)


def assert_fit_only_mapping(mapping: Mapping[str, object]) -> None:
    for key in FORBIDDEN_QUERY_KEYS:
        if key in mapping:
            raise ValueError(f"FIT-only boundary refuses forbidden query key {key}")
    role = mapping.get("role")
    if role is not None and role != "fit":
        raise ValueError("FIT-only boundary requires role=='fit'")


def read_fit_only_jtt1(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    """Decode a train-only 2-fold JTT1 binary table, strictly refusing nq != 0."""
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("JTT1 training input SHA-256 mismatch")
    if len(raw) < 17 or raw[:4] != b"JTT1":
        raise ValueError("invalid JTT1 magic header")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    n_folds = int(raw[16])
    if nq != 0:
        raise ValueError(f"JTT1 contains {nq} query rows; H470 requires strictly FIT-only nq==0")
    if n_folds != 2:
        raise ValueError(f"JTT1 requires 2-fold train-only header, found {n_folds}")
    expected_bytes = 17 + nt + nt * (width + 1) * 8
    if len(raw) != expected_bytes or nt < 4 or width < 1:
        raise ValueError("JTT1 byte length or dimension mismatch")
    mat = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
    x_fit = mat[:, :-1]
    y_fit = mat[:, -1]
    meta = {
        "sha256": actual_sha,
        "n_fit": int(nt),
        "n_query": int(nq),
        "n_features": int(width),
        "n_folds": n_folds,
    }
    return x_fit, y_fit, meta


def landmark_float32(features: np.ndarray) -> np.ndarray:
    """Exact H58/H469 out-of-f32-range column prescaler before FusedLandmarkWorkspace."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    maxima = np.max(np.where(np.isfinite(values), np.abs(values), 0.0), axis=0)
    columns = maxima > np.finfo(np.float32).max
    if columns.any():
        values = values.copy()
        values[:, columns] /= maxima[columns]
    return np.ascontiguousarray(values, dtype=np.float32)


def pool_col_landmarks_34d(col_lm: np.ndarray) -> np.ndarray:
    """Pool [B, 64, 17] or [64, 17] column landmarks to 34D [mean_active(17), max_active(17)].

    Matches the active-column mask convention (nonzero row indicator or col_lm[..., 16] > 0).
    """
    arr = np.asarray(col_lm, dtype=np.float32)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (64, 17):
        raise ValueError(f"col_lm requires shape [B, 64, 17], got {arr.shape}")
    b = arr.shape[0]
    out = np.zeros((b, 34), dtype=np.float32)
    for i in range(b):
        mat = arr[i]
        active = np.any(np.abs(mat) > 1e-12, axis=1)
        if np.any(active):
            sub = mat[active]
            out[i, :17] = np.mean(sub, axis=0)
            out[i, 17:] = np.max(sub, axis=0)
    return out[0] if single else out


def encode_dope_research_view(
    dope_model: MultiLevelLandmarkModel,
    dataset_features: torch.Tensor,
    col_landmarks: torch.Tensor,
    *,
    level: str,
    conditioning_level: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen DOPE router + col_gate_head at `high` or `ultra` without touching product code.

    Enforces:
      - `level == conditioning_level` (`high` never feeds `ultra` and vice versa)
      - `dope_model.training == False` and all parameters `requires_grad == False`
      - Dense finite FP32 inputs `[B, 256]` and `[B, 64, 17]`.
    Returns `(dope629 [B, 629], hidden256 [B, 256])`.
    """
    if level not in ("high", "ultra"):
        raise ValueError(f"unsupported DOPE research view level: {level!r}")
    if conditioning_level != level:
        raise ValueError(
            f"conditioning_level={conditioning_level!r} must match level={level!r} "
            "(never pass high-masked vec256 into ultra or vice versa)"
        )
    if any(m.training for m in dope_model.modules()):
        raise ValueError("dope_model must be in eval() mode")
    if any(p.requires_grad for p in dope_model.parameters()):
        raise ValueError("dope_model parameters must have requires_grad == False")
    if (
        not isinstance(dataset_features, torch.Tensor)
        or not isinstance(col_landmarks, torch.Tensor)
        or dataset_features.dtype != torch.float32
        or col_landmarks.dtype != torch.float32
        or dataset_features.ndim != 2
        or dataset_features.shape[1] != 256
        or dataset_features.shape[0] < 1
        or col_landmarks.shape != (dataset_features.shape[0], 64, 17)
    ):
        raise ValueError("encode_dope_research_view requires FP32 [B, 256] and [B, 64, 17] tensors")
    if not bool(torch.isfinite(dataset_features).all()) or not bool(torch.isfinite(col_landmarks).all()):
        raise ValueError("encode_dope_research_view refuses non-finite conditioning inputs")

    with torch.no_grad():
        r = dope_model.router
        lvl_idx = 1 if level == "high" else 2
        b = dataset_features.shape[0]
        x140 = r.input_norm(dataset_features[:, :140])
        x_extra = r.v3_inter_norm(dataset_features[:, 140:256])
        x256 = torch.cat([x140, x_extra], dim=-1)
        lvl_t = torch.full((b,), lvl_idx, dtype=torch.long, device=dataset_features.device)
        hidden256 = r.backbone(x140) + r.level_embed(lvl_t) + r.inter_attn(x256)

        routed = r(dataset_features, level=level)
        task = routed["task_latent"]
        columns = torch.cat([task.unsqueeze(1).expand(-1, 64, -1), col_landmarks], dim=-1)
        col_gates = 1.0 + 0.25 * torch.tanh(dope_model.target_net.col_gate_head(columns).squeeze(-1))
        dope629 = torch.cat(
            [
                dataset_features,
                task,
                routed["dist_latent"],
                torch.stack([routed["gates"][k] for k in CANONICAL_GATE_KEYS_14], dim=-1),
                routed["landmark_pred"],
                routed["bb_pred"],
                col_gates,
            ],
            dim=-1,
        )
        dope629 = torch.nan_to_num(dope629, nan=0.0, posinf=0.0, neginf=0.0)
        hidden256 = torch.nan_to_num(hidden256, nan=0.0, posinf=0.0, neginf=0.0)
    return dope629, hidden256


# =============================================================================
# Native 23-Landmark FIT-Only 5-Fold CV (`landmarks-native-v1` faithful panel)
# =============================================================================
def _chatterjee_xi_1d(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 4:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    n = xv.size
    if float(np.std(yv)) <= 1e-12 or float(np.std(xv)) <= 1e-12:
        return 0.0
    y_ranks = np.argsort(np.argsort(yv, kind="mergesort"), kind="mergesort").astype(np.float64)
    order = np.argsort(xv, kind="mergesort")
    diffs = float(np.mean(np.abs(np.diff(y_ranks[order]))))
    return float(np.clip(1.0 - (3.0 * diffs) / float(n + 1), 0.0, 1.0))


def _nested_ridge_fit_predict(
    phi_tr: np.ndarray,
    y_tr: np.ndarray,
    phi_qr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Nested inner-holdout (r % 5 == 0) Ridge over lambdas [1e-4, 1e-2, 1.0], matching ridge.rs."""
    n, d = phi_tr.shape
    lambdas = (1e-4, 1e-2, 1.0)
    row_idx = np.arange(n)
    itr = (row_idx % 5) != 0
    iva = ~itr
    if int(np.sum(iva)) < 2 or int(np.sum(itr)) < 2:
        itr = np.ones(n, dtype=bool)
        iva = np.ones(n, dtype=bool)

    def _build_gram(x_m: np.ndarray, y_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        ones = np.ones((x_m.shape[0], 1), dtype=np.float64)
        aug = np.hstack([ones, x_m])
        g = aug.T @ aug
        b = aug.T @ y_v
        diag_scale = float(np.mean(np.diag(g)[1:])) if d > 0 else 1.0
        return g, b, max(diag_scale, 1e-12)

    g_in, b_in, sc_in = _build_gram(phi_tr[itr], y_tr[itr])
    ones_va = np.ones((int(np.sum(iva)), 1), dtype=np.float64)
    aug_va = np.hstack([ones_va, phi_tr[iva]])
    y_va = y_tr[iva]

    best_loss = float("inf")
    best_idx = 1
    p_aug = d + 1
    for idx, lam in enumerate(lambdas):
        reg = np.full(p_aug, lam * sc_in + 1e-9, dtype=np.float64)
        reg[0] = 1e-12
        try:
            w = np.linalg.solve(g_in + np.diag(reg), b_in)
            pred = aug_va @ w
            loss = float(np.sum((pred - y_va) ** 2))
            if math.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_idx = idx
        except np.linalg.LinAlgError:
            continue

    g_out, b_out, sc_out = _build_gram(phi_tr, y_tr)
    reg_out = np.full(p_aug, lambdas[best_idx] * sc_out + 1e-9, dtype=np.float64)
    reg_out[0] = 1e-12
    try:
        w_out = np.linalg.solve(g_out + np.diag(reg_out), b_out)
    except np.linalg.LinAlgError:
        w_out = np.linalg.lstsq(g_out + np.diag(reg_out), b_out, rcond=None)[0]

    ones_qr = np.ones((phi_qr.shape[0], 1), dtype=np.float64)
    pred_qr = np.hstack([ones_qr, phi_qr]) @ w_out
    return pred_qr, best_idx


def _build_basis_for_landmark(
    lm_id: int,
    z_tr: np.ndarray,
    miss_tr: np.ndarray,
    y_tr: np.ndarray,
    z_qr: np.ndarray,
    miss_qr: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Basis expansions matching dope-core/src/landmark/basis.rs for IDs 1..14 and 22."""
    eff_id = 1 if lm_id == 22 else lm_id
    p = z_tr.shape[1]
    if eff_id == 14:
        if miss_tr.shape[1] == 0:
            return None
        return miss_tr, miss_qr

    cols = list(range(p))
    if len(cols) > 64 or eff_id == 12:
        xis = [(_chatterjee_xi_1d(z_tr[:, j], y_tr), -j) for j in cols]
        cols = [c for _, c in sorted(zip(xis, cols), key=lambda kv: kv[0], reverse=True)]
    cols = cols[: (8 if eff_id == 12 else 64)]

    if eff_id == 13:
        rng = np.random.default_rng(seed)
        m_sub = min(256, z_tr.shape[0])
        ia = rng.integers(0, z_tr.shape[0], size=m_sub)
        ib = rng.integers(0, z_tr.shape[0], size=m_sub)
        d2 = np.sum((z_tr[ia, : len(cols)] - z_tr[ib, : len(cols)]) ** 2, axis=1)
        pos = d2[d2 > 0.0]
        bw = max(math.sqrt(float(np.median(pos))) if pos.size > 0 else 1.0, 1e-6)
        n_rff = 192
        w_rff = rng.standard_normal((len(cols), n_rff)) / bw
        b_rff = rng.uniform(0.0, 2.0 * math.pi, size=(1, n_rff))
        scale = math.sqrt(2.0 / n_rff)
        return (
            np.cos(z_tr[:, cols] @ w_rff + b_rff) * scale,
            np.cos(z_qr[:, cols] @ w_rff + b_rff) * scale,
        )

    tr_parts = [z_tr[:, cols]]
    qr_parts = [z_qr[:, cols]]
    xt = z_tr[:, cols]
    xq = z_qr[:, cols]

    if eff_id == 2:
        tr_parts.append(np.arcsinh(xt))
        qr_parts.append(np.arcsinh(xq))
    elif eff_id == 3:
        tr_parts.append(np.sign(xt) * np.sqrt(np.abs(xt)))
        qr_parts.append(np.sign(xq) * np.sqrt(np.abs(xq)))
    elif eff_id == 4:
        tr_parts.extend([xt ** 2, (xt ** 3) / 3.0])
        qr_parts.extend([xq ** 2, (xq ** 3) / 3.0])
    elif eff_id == 5:
        tr_parts.extend([np.tanh(xt), np.tanh(xt / 3.0)])
        qr_parts.extend([np.tanh(xq), np.tanh(xq / 3.0)])
    elif eff_id == 6:
        tr_parts.append(np.abs(xt))
        qr_parts.append(np.abs(xq))
    elif eff_id in (7, 8):
        n_knots = 7 if eff_id == 7 else 11
        qs = np.quantile(xt, np.linspace(1.0 / (n_knots + 1), n_knots / (n_knots + 1), n_knots), axis=0)
        for k_i in range(n_knots):
            qk = qs[k_i : k_i + 1, :]
            if eff_id == 7:
                tr_parts.append(np.maximum(xt - qk, 0.0))
                qr_parts.append(np.maximum(xq - qk, 0.0))
            else:
                tr_parts.append((xt > qk).astype(np.float64))
                qr_parts.append((xq > qk).astype(np.float64))
    elif eff_id == 9:
        q_lo = np.quantile(xt, 0.02, axis=0, keepdims=True)
        q_hi = np.quantile(xt, 0.98, axis=0, keepdims=True)
        span = np.maximum(q_hi - q_lo, 1e-6)
        ut = np.clip(2.0 * (xt - q_lo) / span - 1.0, -1.5, 1.5)
        uq = np.clip(2.0 * (xq - q_lo) / span - 1.0, -1.5, 1.5)
        for freq in (1.0, 2.0, 3.0):
            tr_parts.extend([np.sin(freq * math.pi * ut), np.cos(freq * math.pi * ut)])
            qr_parts.extend([np.sin(freq * math.pi * uq), np.cos(freq * math.pi * uq)])
    elif eff_id == 10:
        for center in (-1.0, 0.0, 1.0):
            tr_parts.append(np.exp(-0.5 * (xt - center) ** 2))
            qr_parts.append(np.exp(-0.5 * (xq - center) ** 2))
    elif eff_id == 11:
        def _row_s(m: np.ndarray) -> np.ndarray:
            rm = np.mean(m, axis=1, keepdims=True)
            return np.hstack([
                rm,
                np.sum(m, axis=1, keepdims=True),
                np.sum(np.abs(m), axis=1, keepdims=True),
                np.sqrt(np.sum(m ** 2, axis=1, keepdims=True)),
                np.std(m, axis=1, keepdims=True),
                np.min(m, axis=1, keepdims=True),
                np.max(m, axis=1, keepdims=True),
            ])
        tr_parts.append(_row_s(xt))
        qr_parts.append(_row_s(xq))
    elif eff_id == 12:
        k_c = xt.shape[1]
        if k_c >= 2:
            tr_parts.append(np.column_stack([xt[:, a] * xt[:, b] for a in range(k_c) for b in range(a + 1, k_c)]))
            qr_parts.append(np.column_stack([xq[:, a] * xq[:, b] for a in range(k_c) for b in range(a + 1, k_c)]))

    phi_tr = np.hstack(tr_parts)
    phi_qr = np.hstack(qr_parts)
    if phi_tr.shape[1] > 511:
        phi_tr = phi_tr[:, :511]
        phi_qr = phi_qr[:, :511]
    return phi_tr, phi_qr


def _single_oblivious_tree_predict(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_qr: np.ndarray,
    depth: int,
) -> np.ndarray:
    n, p = z_tr.shape
    cols = list(range(min(p, 24)))
    if p > 24:
        corrs = np.abs((z_tr.T @ (y_tr - np.mean(y_tr))) / max(n, 1))
        cols = np.argsort(corrs)[::-1][:24].tolist()
    q_grid = np.linspace(0.15, 0.85, 7)
    col_qs = {c: np.unique(np.quantile(z_tr[:, c], q_grid)) for c in cols}
    y_c = y_tr - float(np.mean(y_tr))
    leaf_tr = np.zeros(n, dtype=np.int64)
    leaf_qr = np.zeros(z_qr.shape[0], dtype=np.int64)
    for d in range(depth):
        best_gain = -1.0
        best_f = cols[0]
        best_t = 0.0
        n_leaves = 1 << (d + 1)
        for c in cols:
            for th in col_qs[c]:
                cand = (leaf_tr << 1) | (z_tr[:, c] >= th).astype(np.int64)
                cnt = np.bincount(cand, minlength=n_leaves).astype(np.float64) + 5.0
                sums = np.bincount(cand, weights=y_c, minlength=n_leaves)
                gain = float(np.sum((sums ** 2) / cnt))
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, c, float(th)
        leaf_tr = (leaf_tr << 1) | (z_tr[:, best_f] >= best_t).astype(np.int64)
        leaf_qr = (leaf_qr << 1) | (z_qr[:, best_f] >= best_t).astype(np.int64)
    n_final = 1 << depth
    cnt_f = np.bincount(leaf_tr, minlength=n_final).astype(np.float64) + 5.0
    sums_f = np.bincount(leaf_tr, weights=y_tr, minlength=n_final)
    vals = sums_f / cnt_f
    return vals[leaf_qr]


def _gbdt_fit_predict(z_tr: np.ndarray, y_tr: np.ndarray, z_qr: np.ndarray, seed: int) -> np.ndarray:
    import lightgbm as lgb

    if z_tr.shape[0] < 12 or float(np.std(y_tr)) <= 1e-12:
        return np.full(z_qr.shape[0], float(np.mean(y_tr)), dtype=np.float64)
    params = {
        "objective": "regression",
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.15,
        "max_bin": 32,
        "min_data_in_leaf": max(3, min(10, z_tr.shape[0] // 8)),
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "seed": int(seed % 2147483647),
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    dset = lgb.Dataset(z_tr, label=y_tr, params={"max_bin": 32, "verbose": -1}, free_raw_data=True)
    booster = lgb.train(params, dset, num_boost_round=64)
    return np.asarray(booster.predict(z_qr), dtype=np.float64)


def _knn_fit_predict(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_qr: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    n_ref = min(z_tr.shape[0], 1024)
    x_ref = z_tr[:n_ref]
    y_ref = y_tr[:n_ref]
    p = x_ref.shape[1]
    if p > 32:
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((p, 32)) / math.sqrt(32.0)
        x_ref = x_ref @ proj
        q_mat = z_qr @ proj
    else:
        q_mat = z_qr
    x_sq = np.sum(x_ref ** 2, axis=1)[None, :]
    q_sq = np.sum(q_mat ** 2, axis=1)[:, None]
    d2 = np.maximum(q_sq + x_sq - 2.0 * (q_mat @ x_ref.T), 0.0)
    k_eff = max(1, min(k, n_ref))
    if k_eff == 1:
        idx = np.argmin(d2, axis=1)
        return y_ref[idx]
    part = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]
    return np.mean(y_ref[part], axis=1)


def _net_mlp_fit_predict(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_qr: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Deterministic 1-hidden-layer (h=16) MLP in pure NumPy Adam (matching learners.rs id 21)."""
    cols = min(z_tr.shape[1], 32)
    xt = z_tr[:, :cols]
    xq = z_qr[:, :cols]
    n = xt.shape[0]
    h_dim = 16
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((cols, h_dim)) * (1.0 / math.sqrt(max(cols, 1)))
    b1 = np.zeros(h_dim, dtype=np.float64)
    w2 = rng.standard_normal((h_dim, 1)) * 0.1
    b2 = np.array([float(np.mean(y_tr))], dtype=np.float64)

    m_w1, v_w1 = np.zeros_like(w1), np.zeros_like(w1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_w2, v_w2 = np.zeros_like(w2), np.zeros_like(w2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    lr = 0.025
    wd = 1e-3
    yv = y_tr.reshape(n, 1)
    inv_n = 2.0 / max(n, 1)

    for step in range(1, 121):
        pre = xt @ w1 + b1
        act = np.tanh(pre)
        pred = act @ w2 + b2
        err = (pred - yv) * inv_n
        gw2 = act.T @ err + wd * w2
        gb2 = np.sum(err, axis=0)
        dact = (err @ w2.T) * (1.0 - act ** 2)
        gw1 = xt.T @ dact + wd * w1
        gb1 = np.sum(dact, axis=0)

        b1_c = 1.0 - 0.9 ** step
        b2_c = 1.0 - 0.999 ** step
        m_w1 = 0.9 * m_w1 + 0.1 * gw1
        v_w1 = 0.999 * v_w1 + 0.001 * (gw1 ** 2)
        w1 -= lr * (m_w1 / b1_c) / (np.sqrt(v_w1 / b2_c) + 1e-8)

        m_b1 = 0.9 * m_b1 + 0.1 * gb1
        v_b1 = 0.999 * v_b1 + 0.001 * (gb1 ** 2)
        b1 -= lr * (m_b1 / b1_c) / (np.sqrt(v_b1 / b2_c) + 1e-8)

        m_w2 = 0.9 * m_w2 + 0.1 * gw2
        v_w2 = 0.999 * v_w2 + 0.001 * (gw2 ** 2)
        w2 -= lr * (m_w2 / b1_c) / (np.sqrt(v_w2 / b2_c) + 1e-8)

        m_b2 = 0.9 * m_b2 + 0.1 * gb2
        v_b2 = 0.999 * v_b2 + 0.001 * (gb2 ** 2)
        b2 -= lr * (m_b2 / b1_c) / (np.sqrt(v_b2 / b2_c) + 1e-8)

    return (np.tanh(xq @ w1 + b1) @ w2 + b2).reshape(-1)


def compute_native_23_landmark_report(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 20261023,
) -> dict[str, Any]:
    """Compute a genuine FIT-only 5-fold CV 23-landmark report (`landmarks-native-v1`).

    Never reads query/outer rows. Measures exact added latency of the 5 `ultra` fine
    landmarks (`gbdt`, `knn1`, `knn16`, `net`, `gbdt_quarter`) for `U-S3`.
    """
    t0_total = time.perf_counter()
    x_raw = np.asarray(x_fit, dtype=np.float64)
    if x_raw.ndim == 1:
        x_raw = x_raw.reshape(-1, 1)
    y_raw = np.asarray(y_fit, dtype=np.float64).reshape(-1)
    n, p = x_raw.shape
    if n < 10 or p < 1:
        raise ValueError(f"table too small for 5-fold landmark CV: ({n}, {p})")

    rng_perm = np.random.default_rng(seed)
    perm = rng_perm.permutation(n)
    fold_id = np.empty(n, dtype=np.int64)
    fold_id[perm] = np.arange(n) % n_folds

    ridge_folds = []
    small_folds = []
    q_ridge_folds = []
    q_small_folds = []

    def _prep_fold(tr_idx: np.ndarray, qr_idx: np.ndarray, cap_tr: int, cap_qr: int, quarter: bool):
        tr_use = tr_idx[:cap_tr]
        if quarter:
            tr_use = tr_use[: max(4, len(tr_use) // 4)]
        qr_use = qr_idx[:cap_qr]
        xt_raw = x_raw[tr_use]
        xq_raw = x_raw[qr_use]
        miss_any = np.any(~np.isfinite(xt_raw), axis=0)
        miss_cols = np.where(miss_any)[0]
        mt = (~np.isfinite(xt_raw[:, miss_cols])).astype(np.float64) if miss_cols.size else np.zeros((len(tr_use), 0))
        mq = (~np.isfinite(xq_raw[:, miss_cols])).astype(np.float64) if miss_cols.size else np.zeros((len(qr_use), 0))

        zt = np.zeros_like(xt_raw, dtype=np.float64)
        zq = np.zeros_like(xq_raw, dtype=np.float64)
        for j in range(p):
            col_t = xt_raw[:, j]
            fin_t = np.isfinite(col_t)
            if int(np.sum(fin_t)) >= 2:
                med = float(np.median(col_t[fin_t]))
                ct = np.where(fin_t, col_t, med)
                cq = np.where(np.isfinite(xq_raw[:, j]), xq_raw[:, j], med)
                mu = float(np.mean(ct))
                sd = float(np.std(ct))
                if sd > 1e-12:
                    zt[:, j] = np.clip((ct - mu) / sd, -5.0, 5.0)
                    zq[:, j] = np.clip((cq - mu) / sd, -5.0, 5.0)
        yt_raw = y_raw[tr_use]
        yq_raw = y_raw[qr_use]
        ym = float(np.mean(yt_raw))
        ys = max(float(np.std(yt_raw)), 1e-12)
        yt = (yt_raw - ym) / ys
        yq = (yq_raw - ym) / ys
        return zt, mt, yt, zq, mq, yq

    for k in range(n_folds):
        qr_i = np.where(fold_id == k)[0]
        tr_i = np.where(fold_id != k)[0]
        ridge_folds.append(_prep_fold(tr_i, qr_i, 2048, 2048, False))
        small_folds.append(_prep_fold(tr_i, qr_i, 1024, max(64, 1024 // n_folds), False))
        q_ridge_folds.append(_prep_fold(tr_i, qr_i, 2048, 2048, True))
        q_small_folds.append(_prep_fold(tr_i, qr_i, 1024, max(64, 1024 // n_folds), True))

    def _score_folds(folds_list, runner_fn):
        losses = 0.0
        nulls = 0.0
        fold_gains = []
        lam_indices = []
        for k_idx, (zt, mt, yt, zq, mq, yq) in enumerate(folds_list):
            res = runner_fn(k_idx, zt, mt, yt, zq, mq)
            if res is None:
                return {"state": "unsupported", "gain": None, "se": None, "stable": False, "lambda_indices": []}
            pred, lam_i = res
            if not np.all(np.isfinite(pred)):
                return {"state": "failed", "gain": None, "se": None, "stable": False, "lambda_indices": []}
            pred_loss = float(np.sum((pred - yq) ** 2))
            null_pred = float(np.mean(yt))
            null_loss = float(np.sum((yq - null_pred) ** 2))
            losses += pred_loss
            nulls += null_loss
            if null_loss > 1e-12:
                fold_gains.append(1.0 - pred_loss / null_loss)
            if lam_i is not None:
                lam_indices.append(int(lam_i))
        if nulls <= 1e-12 or len(fold_gains) < 2:
            return {"state": "uninformative", "gain": 0.0, "se": 0.0, "stable": False, "lambda_indices": lam_indices}
        gain = 1.0 - losses / nulls
        se = float(np.std(fold_gains, ddof=1) / math.sqrt(len(fold_gains)))
        stable = bool(se <= max(0.02, 0.25 * abs(gain)))
        state = "complete" if gain > 0.01 else "uninformative"
        return {
            "state": state,
            "gain": round(float(gain), 6),
            "se": round(float(se), 6),
            "stable": stable,
            "lambda_indices": lam_indices,
        }

    cells = []
    fine_ids = {18, 19, 20, 21, 23}
    fine_elapsed_s = 0.0

    for idx_0, lm_name in enumerate(LANDMARK_NAMES):
        lm_id = idx_0 + 1
        t_cell = time.perf_counter()
        if lm_id <= 14 or lm_id == 22:
            f_use = q_ridge_folds if lm_id == 22 else ridge_folds

            def _run_basis(k_i, zt, mt, yt, zq, mq, _id=lm_id):
                built = _build_basis_for_landmark(_id, zt, mt, yt, zq, mq, seed + 1009 * _id + k_i)
                if built is None:
                    return None
                return _nested_ridge_fit_predict(built[0], yt, built[1])

            cell_res = _score_folds(f_use, _run_basis)
        elif lm_id == 15:
            def _run_best1(k_i, zt, mt, yt, zq, mq):
                cand_cols = list(range(min(zt.shape[1], 32)))
                if zt.shape[1] > 32:
                    corrs = np.abs((zt.T @ (yt - np.mean(yt))) / max(len(yt), 1))
                    cand_cols = np.argsort(corrs)[::-1][:32].tolist()
                best = None
                r_idx = np.arange(len(yt))
                itr = (r_idx % 5) != 0
                iva = ~itr
                for cj in cand_cols:
                    b_pair = _build_basis_for_landmark(
                        8, zt[:, cj : cj + 1], mt, yt, zq[:, cj : cj + 1], mq, seed + k_i
                    )
                    if b_pair is None:
                        continue
                    pt_in, _ = _nested_ridge_fit_predict(b_pair[0][itr], yt[itr], b_pair[0][iva])
                    l_in = float(np.sum((pt_in - yt[iva]) ** 2))
                    if best is None or l_in < best[0]:
                        p_out, l_idx = _nested_ridge_fit_predict(b_pair[0], yt, b_pair[1])
                        best = (l_in, p_out, l_idx)
                return None if best is None else (best[1], best[2])

            cell_res = _score_folds(ridge_folds, _run_best1)
        elif lm_id in (16, 17):
            depth = 1 if lm_id == 16 else 3
            cell_res = _score_folds(
                small_folds,
                lambda k_i, zt, mt, yt, zq, mq, _d=depth: (_single_oblivious_tree_predict(zt, yt, zq, _d), None),
            )
        elif lm_id in (18, 23):
            f_use = q_small_folds if lm_id == 23 else small_folds
            cell_res = _score_folds(
                f_use,
                lambda k_i, zt, mt, yt, zq, mq, _id=lm_id: (
                    _gbdt_fit_predict(zt, yt, zq, seed + 97 * _id + k_i),
                    None,
                ),
            )
        elif lm_id in (19, 20):
            k_nn = 1 if lm_id == 19 else 16
            cell_res = _score_folds(
                small_folds,
                lambda k_i, zt, mt, yt, zq, mq, _k=k_nn: (
                    _knn_fit_predict(zt, yt, zq, _k, seed + 131 * _k + k_i),
                    None,
                ),
            )
        elif lm_id == 21:
            cell_res = _score_folds(
                small_folds,
                lambda k_i, zt, mt, yt, zq, mq: (
                    _net_mlp_fit_predict(zt, yt, zq, seed + 733 + k_i),
                    None,
                ),
            )
        else:
            raise AssertionError(f"unhandled landmark id {lm_id}")

        dt = time.perf_counter() - t_cell
        if lm_id in fine_ids:
            fine_elapsed_s += dt
        cells.append({"id": lm_id, "name": lm_name, **cell_res})

    # Per-column univariate spectrum (up to 32 columns, 9 slots: lin, step, hinge, log, sat, periodic, abs, xi, stump)
    spec_cols = min(p, 32)
    spectrum = []
    empty_m = np.zeros((1, 0), dtype=np.float64)
    for slot in range(spec_cols):
        vals: list[float | None] = [None] * 9
        for slot_pos, u_id in ((0, 1), (1, 8), (2, 7), (3, 2), (4, 5), (5, 9), (6, 6)):
            def _uni_runner(k_i, zt, _mt, yt, zq, _mq, _s=slot, _uid=u_id):
                bp = _build_basis_for_landmark(
                    _uid, zt[:, _s : _s + 1], empty_m, yt, zq[:, _s : _s + 1], empty_m, seed + _uid
                )
                return None if bp is None else _nested_ridge_fit_predict(bp[0], yt, bp[1])

            u_res = _score_folds(ridge_folds, _uni_runner)
            if u_res["state"] == "complete" and u_res["stable"] and u_res["gain"] is not None:
                vals[slot_pos] = float(u_res["gain"])
        stump_res = _score_folds(
            small_folds,
            lambda k_i, zt, _mt, yt, zq, _mq, _s=slot: (
                _single_oblivious_tree_predict(zt[:, _s : _s + 1], yt, zq[:, _s : _s + 1], 1),
                None,
            ),
        )
        if stump_res["state"] == "complete" and stump_res["stable"] and stump_res["gain"] is not None:
            vals[8] = float(stump_res["gain"])
        vals[7] = round(_chatterjee_xi_1d(x_raw[:, slot], y_raw), 6)
        spectrum.append({"slot": slot, "values": vals})

    ym_all = float(np.mean(y_raw))
    yv_all = float(np.mean((y_raw - ym_all) ** 2))
    if yv_all > 1e-24:
        y_skew = float(np.mean((y_raw - ym_all) ** 3) / (yv_all ** 1.5))
        y_kurt = float(np.mean((y_raw - ym_all) ** 4) / (yv_all ** 2) - 3.0)
        t_stats = [round(y_skew, 6), round(y_kurt, 6), None]
    else:
        t_stats = [None, None, None]

    total_elapsed_s = time.perf_counter() - t0_total
    return {
        "panel": "landmarks-native-v1",
        "kind": "real",
        "privacy": "private",
        "task": "regression",
        "classes": 0,
        "rows": int(n),
        "features": int(p),
        "cells": cells,
        "target_stats": t_stats,
        "spectrum": spectrum,
        "timing": {
            "total_ms": round(total_elapsed_s * 1000.0, 3),
            "fine_5_landmarks_ms": round(fine_elapsed_s * 1000.0, 3),
        },
    }


def build_high_and_ultra_conditioning_from_fit(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Scale FIT rows via `FitOnlyTableTransform` and assemble `high` and `ultra` DOPE conditioning."""
    assert_fit_only_mapping(report)
    scaler = FitOnlyTableTransform.fit(x_fit, y_fit, task="regression")
    x_scaled, y_scaled = scaler.transform(x_fit, y_fit)
    assert y_scaled is not None

    cond_high = build_v3_conditioning_bundle(dict(report), x_scaled, y_scaled, level="high")
    cond_ultra = build_v3_conditioning_bundle(dict(report), x_scaled, y_scaled, level="ultra")

    def _pad_col_lm(col24: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((64, 17), dtype=np.float32)
        mask = np.zeros(64, dtype=bool)
        n_c = min(col24.shape[0], 64)
        if n_c > 0:
            out[:n_c, :17] = col24[:n_c, :17]
            mask[:n_c] = True
        return out, mask

    col_high, mask_high = _pad_col_lm(cond_high["col_landmarks24"])
    col_ultra, mask_ultra = _pad_col_lm(cond_ultra["col_landmarks24"])

    # Extract true squashed gains & completion flags for all 23 landmarks (especially the 5 fine probes)
    gains23_ultra = np.asarray(cond_ultra["rep_desc"]["gains23"], dtype=np.float32)
    mask23_ultra = np.asarray(cond_ultra["rep_desc"]["mask23"], dtype=bool)

    return {
        "x_scaled": x_scaled,
        "y_scaled": y_scaled,
        "vec256_high": np.asarray(cond_high["vec256"], dtype=np.float32),
        "col_lm_high": col_high,
        "col_mask_high": mask_high,
        "vec256_ultra": np.asarray(cond_ultra["vec256"], dtype=np.float32),
        "col_lm_ultra": col_ultra,
        "col_mask_ultra": mask_ultra,
        "gains23_ultra": gains23_ultra,
        "mask23_ultra": mask23_ultra,
    }


def _canonical_column_permutation(x_scaled: np.ndarray) -> np.ndarray:
    """Target-free, column-equivariant ordering so `encode_table` (`target_net.fc1`, `p > 64` cap,
    and SVD sign orientation) is invariant to arbitrary input column permutations (`P-S3`).
    """
    p = x_scaled.shape[1]
    if p <= 1:
        return np.arange(p, dtype=np.int64)
    keys = []
    for j in range(p):
        col = x_scaled[:, j]
        q10, q25, q50, q75, q90 = np.quantile(col, (0.10, 0.25, 0.50, 0.75, 0.90))
        iqr = float(q75 - q25)
        span = float(q90 - q10)
        mean_abs = float(np.mean(np.abs(col - q50)))
        sd = float(np.std(col))
        m3 = float(np.mean((col - float(np.mean(col))) ** 3))
        m4 = float(np.mean((col - float(np.mean(col))) ** 4))
        keys.append((
            round(iqr, 6),
            round(span, 6),
            round(mean_abs, 6),
            round(sd, 6),
            round(m3, 5),
            round(m4, 5),
            round(float(q50), 6),
        ))
    order = sorted(range(p), key=lambda j: keys[j], reverse=True)
    return np.asarray(order, dtype=np.int64)


def _permute_report_spectrum(report: Mapping[str, Any], col_order: np.ndarray) -> dict[str, Any]:
    inv = {int(orig_j): int(new_pos) for new_pos, orig_j in enumerate(col_order.tolist())}
    remapped = []
    for item in report.get("spectrum") or []:
        old_slot = int(item.get("slot", -1))
        if old_slot in inv:
            remapped.append({"slot": inv[old_slot], "values": list(item.get("values") or [])})
    remapped.sort(key=lambda d: d["slot"])
    out = dict(report)
    out["spectrum"] = remapped
    return out


def encode_fit_table_for_packet_v1(
    dope_model: MultiLevelLandmarkModel,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    report: Mapping[str, Any],
    *,
    level: str = "high",
    include_v3_primitives: bool = True,
    canonicalize_columns: bool = True,
) -> dict[str, Any]:
    """Run frozen `MultiLevelLandmarkModel.encode_table` strictly on scaled FIT rows (`role='fit'`)."""
    assert_fit_only_mapping(report)
    if any(m.training for m in dope_model.modules()):
        raise ValueError("dope_model must remain in eval() mode")
    if canonicalize_columns and x_scaled.shape[1] > 1:
        col_order = _canonical_column_permutation(x_scaled)
        x_in = np.ascontiguousarray(x_scaled[:, col_order], dtype=np.float64)
        rep_in = _permute_report_spectrum(report, col_order)
    else:
        col_order = np.arange(x_scaled.shape[1], dtype=np.int64)
        x_in = np.ascontiguousarray(x_scaled, dtype=np.float64)
        rep_in = dict(report)

    with torch.no_grad():
        enc = dope_model.encode_table(
            x_in,
            np.ascontiguousarray(y_scaled, dtype=np.float64),
            rep_in,
            task="regression",
            level=level,
            include_v3_primitives=include_v3_primitives,
        )
    # Strip target_bundle raw extrema keys from downstream view while keeping exact wire semantics
    enc = dict(enc)
    enc["role"] = "fit"
    enc["_canonical_col_order"] = col_order
    return enc


# =============================================================================
# Exact `dope.packet-features/v1` (196D) & `dope.packet-global/v1` (28D)
# =============================================================================
def extract_packet_column_tokens_v1(
    encoding: Mapping[str, object],
    *,
    include_target_bundle: bool = True,
) -> tuple[list[list[float]], list[float]]:
    """Return `(col_tokens [p_use, 32], activities [p_use])` for equivariance testing (`P-S3`)."""
    assert_fit_only_mapping(encoding)
    p_use = _count("p_use", encoding.get("p_use"))
    if p_use > 64:
        raise ValueError("packet_features_v1 requires p_use <= 64")
    bundle = encoding.get("target_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("packet_features_v1 requires target_bundle")

    task = str(encoding.get("task", "regression"))
    b_kind = str(bundle.get("task_kind", task))
    is_mc = task == "multiclass" or b_kind == "multiclass"

    trees_raw = _sequence("target_bundle.trees", bundle.get("trees", ()))
    mc_trees_raw = _sequence("target_bundle.mc_trees", bundle.get("mc_trees", ()))
    active_trees = mc_trees_raw if len(mc_trees_raw) > 0 else trees_raw
    n_trees = len(active_trees)
    pairs = _sequence("target_bundle.pairs", bundle.get("pairs", ()))

    col_knots = _col_matrix(encoding, "col_knots", p_use)
    chol = _col_matrix(encoding, "chol_factor", p_use)
    diag_res = _col_vec(encoding, "diag_res_std", p_use, default=1.0)
    is_bin_col = [bool(x) for x in _col_list(encoding, "is_binary_col", p_use, default=False)]
    bin_probs = _col_vec(encoding, "binary_probs", p_use, default=0.0)
    disc_map = encoding.get("discrete_support") if isinstance(encoding.get("discrete_support"), Mapping) else {}
    reg_means = np.asarray(encoding.get("regime_means", np.zeros((0, p_use))), dtype=np.float64)
    reg_scales = np.asarray(encoding.get("regime_scales", np.zeros((0, p_use))), dtype=np.float64)

    chol_sq = [float(np.sum(np.square(chol[j]))) for j in range(p_use)]
    chol_top1 = [float(np.max(np.abs(chol[j]))) if len(chol[j]) > 0 else 0.0 for j in range(p_use)]
    basis_rows = _extract_basis_rows(bundle, is_mc) if include_target_bundle else []

    pair_deg = [0] * p_use
    if include_target_bundle:
        for pair in pairs:
            a, c = int(pair[0]), int(pair[1])
            if 0 <= a < p_use:
                pair_deg[a] += 1
            if 0 <= c < p_use:
                pair_deg[c] += 1

    split_cnt, depth_w, root_cnt, col_thresh, _, _, _, _, _ = (
        _scan_trees(active_trees, p_use)
        if include_target_bundle
        else ([0] * p_use, [0.0] * p_use, [0] * p_use, [[] for _ in range(p_use)], [], [], 0, 0.0, 0.0)
    )
    tot_splits = sum(split_cnt)
    tot_dw = sum(depth_w)
    sum_col_b = sum(
        _at(basis_rows, 1 + j) + _at(basis_rows, 1 + p_use + j) + _at(basis_rows, 1 + 2 * p_use + j)
        for j in range(p_use)
    )

    col_tokens = [[0.0] * 32 for _ in range(p_use)]
    activities = [0.0] * p_use
    for j in range(p_use):
        knots9 = _resample_knots_9(col_knots[j])
        tok = [0.0] * 32
        tok[:9] = knots9
        denom = max(abs((knots9[6] - knots9[2]) * 6.0), 1e-4)
        tok[9] = knots9[6] - knots9[2]
        tok[10] = math.tanh(((knots9[6] + knots9[2] - 2.0 * knots9[4]) * 6.0) / denom)
        tok[11] = math.tanh((((knots9[8] - knots9[0]) * 6.0) / denom) / 4.0)
        tok[12] = 1.0 if is_bin_col[j] else 0.0
        if is_bin_col[j]:
            pb = min(max(bin_probs[j], 0.0), 1.0)
            tok[13] = 4.0 * pb * (1.0 - pb)
        if j in disc_map:
            u_vals = disc_map[j][0]
            tok[14] = 1.0
            tok[15] = min(max(len(u_vals) / 32.0, 0.0), 2.0)
        tok[16] = min(max(chol_sq[j], 0.0), 2.0)
        tok[17] = min(max(chol_top1[j], 0.0), 2.0)
        tok[18] = min(max(diag_res[j], 0.0), 2.0)
        if reg_means.ndim == 2 and reg_means.shape[0] > 0 and reg_means.shape[1] > j:
            tok[19] = math.tanh(_mean_std_max(reg_means[:, j].tolist())[1])
        if reg_scales.ndim == 2 and reg_scales.shape[0] > 0 and reg_scales.shape[1] > j:
            tok[20] = min(max(float(np.mean(reg_scales[:, j])), 0.0), 4.0) / 4.0
        if include_target_bundle:
            w_l = _at(basis_rows, 1 + j)
            w_q = _at(basis_rows, 1 + p_use + j)
            w_a = _at(basis_rows, 1 + 2 * p_use + j)
            sp_sh = split_cnt[j] / float(max(tot_splits, 1))
            bs_sh = (w_l + w_q + w_a) / (sum_col_b + 1e-8)
            act = 0.5 * bs_sh + 0.5 * sp_sh
            tok[21] = pair_deg[j] / float(max(len(pairs), 1))
            tok[22] = sp_sh
            tok[23] = depth_w[j] / max(tot_dw, 1e-8)
            tok[24] = root_cnt[j] / float(max(n_trees, 1))
            if col_thresh[j]:
                tm, ts, _ = _mean_std_max(col_thresh[j])
                tok[25] = math.tanh(tm / 4.0)
                tok[26] = math.tanh(ts / 4.0) if len(col_thresh[j]) > 1 else 0.0
            tok[27] = math.tanh(w_l)
            tok[28] = math.tanh(w_q)
            tok[29] = math.tanh(w_a)
            tok[30] = bs_sh
            tok[31] = act
            activities[j] = act
        col_tokens[j] = tok
    return col_tokens, activities


def packet_global_v1(
    encoding: Mapping[str, object],
    *,
    include_target_bundle: bool = True,
) -> tuple[float, ...]:
    """28D `dope.packet-global/v1` prefix (deduplicated `noise_scale`, zero `y_min`/`y_max`)."""
    return packet_features_v1(encoding, include_target_bundle=include_target_bundle)[:PACKET_GLOBAL_WIDTH]


def packet_features_v1(
    encoding: Mapping[str, object],
    *,
    include_target_bundle: bool = True,
) -> tuple[float, ...]:
    """196D `dope.packet-features/v1` scale-invariant, extrema-free fitted packet vector (`WP2`).

    Matches DOPE `research.levels.research_encoder_view.packet_features_v1` coordinate-for-coordinate:
      - `[0:14]`    14 canonical router gates (`noise_scale` once)
      - `[14:28]`   14 normalized task/count/residual globals (NO `y_min`/`y_max`)
      - `[28:44]`   16 regime/copula/`w_basis` norm statistics
      - `[44:140]`  `3 x 32 = 96` `(mean, std, max)` symmetric column-packet pool
      - `[140:172]` `32` `exp(4*(act - max_act))` activity-weighted column pool
      - `[172:196]` `24` tree/pair/latent summary statistics.
    """
    if not isinstance(encoding, Mapping):
        raise ValueError("packet_features_v1 requires an encode_table mapping")
    assert_fit_only_mapping(encoding)

    level = encoding.get("level")
    if level not in LEVEL_CODE:
        raise ValueError("packet_features_v1 requires level high or ultra")
    n_train = _count("n_train", encoding.get("n_train"))
    p_use = _count("p_use", encoding.get("p_use"))
    if p_use > 64:
        raise ValueError("packet_features_v1 requires p_use <= 64")

    gates = encoding.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(CANONICAL_GATE_KEYS_14):
        raise ValueError("packet_features_v1 requires the 14 canonical gates")
    bundle = encoding.get("target_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("packet_features_v1 requires target_bundle")

    task = str(encoding.get("task", "regression"))
    b_kind = str(bundle.get("task_kind", task))
    is_mc = task == "multiclass" or b_kind == "multiclass"
    is_bin = (not is_mc) and task in ("classification", "binary")
    is_reg = (not is_mc) and (not is_bin) and task in ("regression", "")
    if not (is_reg or is_bin or is_mc):
        raise ValueError("packet_features_v1 unsupported task")

    out = [0.0] * PACKET_FEATURES_WIDTH
    for idx, key in enumerate(CANONICAL_GATE_KEYS_14):
        out[idx] = _finite(f"gates.{key}", gates[key])

    trees_raw = _sequence("target_bundle.trees", bundle.get("trees", ()))
    mc_trees_raw = _sequence("target_bundle.mc_trees", bundle.get("mc_trees", ()))
    active_trees = mc_trees_raw if len(mc_trees_raw) > 0 else trees_raw
    n_trees = len(active_trees)
    pairs = _sequence("target_bundle.pairs", bundle.get("pairs", ()))
    priors_seq = bundle.get("class_priors", ())
    k_cls = (
        max(int(bundle.get("k_classes", 3)), len(priors_seq) if hasattr(priors_seq, "__len__") else 0, 3)
        if is_mc
        else (2 if is_bin else 1)
    )

    if include_target_bundle:
        for req in ("residual_std", "neural_w", "res_net_mean", "class_prior"):
            if req not in bundle:
                raise ValueError(f"packet_features_v1 requires target_bundle.{req}")
        res_std = _finite("target_bundle.residual_std", bundle["residual_std"])
        neural_w = _finite("target_bundle.neural_w", bundle["neural_w"])
        res_mean = _finite("target_bundle.res_net_mean", bundle["res_net_mean"])
        cls_prior = _finite("target_bundle.class_prior", bundle["class_prior"])
        if not is_mc:
            out[14] = min(max(res_std, 0.0), 4.0) / 4.0
            out[15] = min(max(neural_w, 0.0), 2.0) / 2.0
            out[16] = math.tanh(res_mean)
        if is_bin:
            p = min(max(cls_prior, 1e-6), 1.0 - 1e-6)
            out[21] = (-p * math.log(p) - (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)
        elif is_mc and priors_seq is not None and len(priors_seq) > 0:
            p_arr = [max(_finite("target_bundle.class_priors", x), 1e-6) for x in priors_seq]
            s_p = sum(p_arr)
            ent = sum(-(x / s_p) * math.log(x / s_p) for x in p_arr)
            out[21] = min(max(ent / math.log(max(len(p_arr), 2)), 0.0), 1.0)
        out[25] = min(max(n_trees / 56.0, 0.0), 2.0)
        out[26] = min(max(len(pairs) / 28.0, 0.0), 2.0)

    out[17] = 1.0 if is_reg else 0.0
    out[18] = 1.0 if is_bin else 0.0
    out[19] = 1.0 if is_mc else 0.0
    out[20] = min(max(math.log2(k_cls) / 8.0, 0.0), 2.0)
    out[22] = min(max(math.log10(n_train) / 6.0, 0.0), 2.0)
    out[23] = min(max(math.log2(p_use) / 6.0, 0.0), 2.0)
    out[24] = math.tanh(p_use / float(n_train))
    out[27] = LEVEL_CODE[level]

    reg_w = np.asarray(encoding.get("regime_weights", ()), dtype=np.float64).reshape(-1)
    if reg_w.size > 0:
        if not np.isfinite(reg_w).all():
            raise ValueError("packet_features_v1 non-finite regime_weights")
        rw = np.maximum(reg_w, 0.0)
        rw = np.sort(rw / max(float(np.sum(rw)), 1e-8))[::-1]
        for k in range(min(8, rw.size)):
            out[28 + k] = float(rw[k])

    chol = _col_matrix(encoding, "chol_factor", p_use)
    diag_res = _col_vec(encoding, "diag_res_std", p_use, default=1.0)
    chol_sq = [float(np.sum(np.square(chol[j]))) for j in range(p_use)]
    out[36], out[37], out[38] = _mean_std_max(chol_sq)
    out[39] = float(sum(diag_res[:p_use])) / float(p_use)

    basis_rows = _extract_basis_rows(bundle, is_mc) if include_target_bundle else []
    if include_target_bundle and basis_rows:
        l1 = sum(basis_rows)
        l2 = math.sqrt(sum(v * v for v in basis_rows))
        top4 = sum(sorted(basis_rows, reverse=True)[:4])
        out[40] = math.tanh(math.log1p(l2))
        out[41] = math.tanh(l1 / float(len(basis_rows)))
        out[42] = max(basis_rows) / (l2 + 1e-8)
        out[43] = top4 / (l1 + 1e-8)

    pair_deg = [0] * p_use
    uniq_pair_cols: set[int] = set()
    if include_target_bundle:
        for pair in pairs:
            a, c = int(pair[0]), int(pair[1])
            if 0 <= a < p_use:
                pair_deg[a] += 1
                uniq_pair_cols.add(a)
            if 0 <= c < p_use:
                pair_deg[c] += 1
                uniq_pair_cols.add(c)

    split_cnt, _, _, _, all_thresh, all_leaves, depth_sum, first_rms, last_rms = (
        _scan_trees(active_trees, p_use)
        if include_target_bundle
        else ([0] * p_use, [0.0] * p_use, [0] * p_use, [[] for _ in range(p_use)], [], [], 0, 0.0, 0.0)
    )

    col_tokens, activities = extract_packet_column_tokens_v1(
        encoding, include_target_bundle=include_target_bundle
    )
    max_act = max(activities) if activities else 0.0
    exp_w = [math.exp(4.0 * (a - max_act)) for a in activities]
    exp_sum = max(sum(exp_w), 1e-8)
    for d in range(32):
        vals = [col_tokens[j][d] for j in range(p_use)]
        m, s, mx = _mean_std_max(vals)
        out[44 + d] = m
        out[76 + d] = s
        out[108 + d] = mx
        out[140 + d] = sum((exp_w[j] / exp_sum) * col_tokens[j][d] for j in range(p_use))

    if include_target_bundle:
        if n_trees > 0:
            out[172] = min(max((depth_sum / float(n_trees)) / 4.0, 0.0), 2.0)
            if all_leaves:
                abs_l = [abs(v) for v in all_leaves]
                lm_abs, _, lmax_abs = _mean_std_max(abs_l)
                out[173] = math.tanh(lm_abs)
                out[174] = math.tanh(_mean_std_max(all_leaves)[1])
                out[175] = math.tanh(lmax_abs)
            out[176] = first_rms / (first_rms + last_rms + 1e-8)
            out[177] = sum(1 for c in split_cnt if c > 0) / float(p_use)
            if all_thresh:
                out[178] = math.tanh(_mean_std_max([abs(v) for v in all_thresh])[0] / 4.0)
                out[179] = math.tanh(_mean_std_max(all_thresh)[1] / 4.0)
        out[180] = len(uniq_pair_cols) / float(p_use)
        out[181] = max(pair_deg) / float(max(len(pairs), 1))
        t_start = min(1 + 3 * p_use, len(basis_rows))
        tail = basis_rows[t_start:]
        out[182] = math.tanh(_rms(tail))
        tot_l1 = sum(basis_rows) + 1e-8
        out[183] = sum(basis_rows[: min(1 + p_use, len(basis_rows))]) / tot_l1
        out[184] = (
            sum(basis_rows[min(1 + p_use, len(basis_rows)) : min(1 + 2 * p_use, len(basis_rows))]) / tot_l1
        )
        out[185] = (
            sum(basis_rows[min(1 + 2 * p_use, len(basis_rows)) : min(1 + 3 * p_use, len(basis_rows))]) / tot_l1
        )
        out[186] = sum(tail) / tot_l1
        out[187] = 1.0 if bool(bundle.get("include_v3_primitives", False)) else 0.0

    tl = [float(x) for x in np.asarray(encoding.get("task_latent", ()), dtype=np.float32).reshape(-1)]
    dl = [float(x) for x in np.asarray(encoding.get("dist_latent", ()), dtype=np.float32).reshape(-1)]
    out[188:192] = _latent_stats(tl)
    out[192:196] = _latent_stats(dl)

    return tuple(float(np.float32(v if math.isfinite(v) else 0.0)) for v in out)


def _extract_basis_rows(bundle: Mapping[str, object], is_mc: bool) -> list[float]:
    if is_mc and "U_basis" in bundle and bundle["U_basis"] is not None:
        u = np.asarray(bundle["U_basis"], dtype=np.float64)
        if not np.isfinite(u).all():
            raise ValueError("packet_features_v1 non-finite U_basis")
        if u.ndim == 2 and u.size > 0:
            return [float(math.sqrt(np.sum(np.square(row)))) for row in u]
    w = np.asarray(bundle.get("w_basis", ()), dtype=np.float64).reshape(-1)
    if not np.isfinite(w).all():
        raise ValueError("packet_features_v1 non-finite w_basis")
    return [float(abs(v)) for v in w]


def _scan_trees(trees: Sequence[object], p_use: int):
    split_cnt = [0] * p_use
    depth_w = [0.0] * p_use
    root_cnt = [0] * p_use
    col_thresh: list[list[float]] = [[] for _ in range(p_use)]
    all_thresh: list[float] = []
    all_leaves: list[float] = []
    depth_sum = 0
    first_rms, last_rms = 0.0, 0.0
    for ti, tree in enumerate(trees):
        if not isinstance(tree, Mapping):
            continue
        feats = [int(f) for f in tree.get("features", ())]
        ths = [_finite("tree.thresholds", th) for th in tree.get("thresholds", ())]
        depth_sum += len(feats)
        for d, (f, th) in enumerate(zip(feats, ths)):
            all_thresh.append(th)
            if 0 <= f < p_use:
                split_cnt[f] += 1
                depth_w[f] += 2.0 ** (-d)
                if d == 0:
                    root_cnt[f] += 1
                col_thresh[f].append(th)
        lv_arr = np.asarray(tree.get("leaf_values", ()), dtype=np.float64).reshape(-1)
        if not np.isfinite(lv_arr).all():
            raise ValueError("packet_features_v1 non-finite leaf_values")
        lv = [float(x) for x in lv_arr]
        rms = _rms(lv)
        if ti == 0:
            first_rms = rms
        if ti + 1 == len(trees):
            last_rms = rms
        all_leaves.extend(lv)
    return split_cnt, depth_w, root_cnt, col_thresh, all_thresh, all_leaves, depth_sum, first_rms, last_rms


def _col_matrix(encoding: Mapping[str, object], key: str, p_use: int) -> list[list[float]]:
    raw = encoding.get(key)
    if raw is None:
        return [[0.0] for _ in range(p_use)]
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < p_use or not np.isfinite(arr[:p_use]).all():
        raise ValueError(f"packet_features_v1 invalid {key}")
    return [arr[j].tolist() for j in range(p_use)]


def _col_vec(encoding: Mapping[str, object], key: str, p_use: int, *, default: float) -> list[float]:
    raw = encoding.get(key)
    if raw is None:
        return [default] * p_use
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < p_use or not np.isfinite(arr[:p_use]).all():
        raise ValueError(f"packet_features_v1 invalid {key}")
    return [float(x) for x in arr[:p_use]]


def _col_list(encoding: Mapping[str, object], key: str, p_use: int, *, default: object) -> list[object]:
    raw = encoding.get(key)
    if raw is None:
        return [default] * p_use
    seq = list(raw)
    if len(seq) < p_use:
        raise ValueError(f"packet_features_v1 invalid {key}")
    return seq[:p_use]


def _resample_knots_9(knots: Sequence[float]) -> list[float]:
    if not knots:
        return [0.0] * 9
    if len(knots) == 1:
        return [min(max(float(knots[0]), -6.0), 6.0) / 6.0] * 9
    last = float(len(knots) - 1)
    res = []
    for m in range(9):
        pos = (m / 8.0) * last
        lo = min(int(math.floor(pos)), len(knots) - 1)
        hi = min(lo + 1, len(knots) - 1)
        frac = pos - lo
        v = float(knots[lo]) * (1.0 - frac) + float(knots[hi]) * frac
        res.append(min(max(v, -6.0), 6.0) / 6.0)
    return res


def _latent_stats(vals: list[float]) -> list[float]:
    if not vals:
        return [0.0, 0.0, 0.0, 0.0]
    abs_v = [abs(v) for v in vals]
    m_abs, _, max_abs = _mean_std_max(abs_v)
    return [_rms(vals), m_abs, max_abs, _mean_std_max(vals)[1]]


def _rms(vals: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vals) / float(len(vals))) if vals else 0.0


def _mean_std_max(vals: Sequence[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    n = float(len(vals))
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / n
    mx = max(vals)
    return m, math.sqrt(max(var, 0.0)), mx if math.isfinite(mx) else 0.0


def _at(vals: Sequence[float], idx: int) -> float:
    return vals[idx] if 0 <= idx < len(vals) else 0.0


def _count(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"packet_features_v1: {name} must be an integer count >= 1")
    try:
        n = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(f"packet_features_v1: {name} must be an integer count >= 1") from None
    if n < 1:
        raise ValueError(f"packet_features_v1: {name} must be an integer count >= 1")
    return n


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"packet_features_v1: {name} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"packet_features_v1: {name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"packet_features_v1: {name} must be a finite number")
    return number


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"packet_features_v1 requires {name}")
    return value


# =============================================================================
# Batch WP01-Authenticated Cache Extractor (FIT-rows only, strict split isolation)
# =============================================================================
RANDROUTER_SEEDS: tuple[int, ...] = (20261101, 20261102, 20261103, 20261104, 20261105)
DOPE_CKPT_FILE_SHA256 = "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952"
DOPE_FUSED_COMPOSITE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
CPU_EXACT_FUSED_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"

_WORKER_CTX: dict[str, Any] = {}


def memavail_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576.0
    return 0.0


def assert_memavail(min_gb: float = 15.0, tag: str = "", max_wait_s: float = 60.0) -> float:
    import gc

    deadline = time.monotonic() + max(0.0, float(max_wait_s))
    while True:
        avail = memavail_gb()
        if avail >= min_gb:
            return avail
        if time.monotonic() >= deadline:
            raise RuntimeError(f"MemAvailable {avail:.2f} GB < {min_gb:.2f} GB guard at {tag}")
        gc.collect()
        time.sleep(2.0)


def _init_extraction_worker(dope_ckpt_str: str) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    from hpc.landmark_workspace import FusedLandmarkWorkspace

    dope_model = MultiLevelLandmarkModel()
    dope_ckpt = torch.load(dope_ckpt_str, map_location="cpu", weights_only=True)
    dope_sd = dope_ckpt["model_state_dict"] if "model_state_dict" in dope_ckpt else dope_ckpt
    dope_model.load_state_dict(dope_sd, strict=True)
    dope_model.eval()
    for p in dope_model.parameters():
        p.requires_grad_(False)

    rand_models = []
    for rseed in RANDROUTER_SEEDS:
        torch.manual_seed(rseed)
        rm = MultiLevelLandmarkModel().eval()
        for p in rm.parameters():
            p.requires_grad_(False)
        rand_models.append(rm)

    ws = FusedLandmarkWorkspace(num_threads=1)
    _WORKER_CTX["dope_model"] = dope_model
    _WORKER_CTX["ws"] = ws
    _WORKER_CTX["rand_models"] = rand_models


def _extract_single_context_worker(task: tuple[int, str, str, str]) -> dict[str, Any]:
    idx, cid, jtt_path, expected_sha = task
    assert_memavail(15.0, f"worker:{idx}")
    dope_model = _WORKER_CTX["dope_model"]
    ws = _WORKER_CTX["ws"]
    rand_models = _WORKER_CTX["rand_models"]

    x_fit, y_fit, meta = read_fit_only_jtt1(jtt_path, expected_sha256=expected_sha)
    n_fit = int(meta["n_fit"])
    p_feat = int(meta["n_features"])
    r_shape = np.array(
        [math.log10(max(n_fit, 1)) / 6.0, math.log2(max(p_feat, 1)) / 12.0, 0.0],
        dtype=np.float32,
    )

    y64 = y_fit.astype(np.float64)
    yn = ((y64 - y64.mean()) / max(float(y64.std()), 1e-8)).astype(np.float32)
    xf32_lm = landmark_float32(x_fit)

    # 1. Pinned H58/H256 `FusedLandmarkWorkspace` (`hpc_high`)
    v_hpc_np, col_hpc_raw, _ = ws.fit_transform_and_extract_bundle_high(xf32_lm, np.ascontiguousarray(yn))
    clm_hpc_np = np.zeros((64, 17), dtype=np.float32)
    nc_hpc = min(len(col_hpc_raw), 64)
    clm_hpc_np[:nc_hpc, : min(col_hpc_raw.shape[1], 17)] = col_hpc_raw[:nc_hpc, :17]
    v_hpc_t = torch.from_numpy(v_hpc_np.astype(np.float32)).reshape(1, 256)
    clm_hpc_t = torch.from_numpy(clm_hpc_np).reshape(1, 64, 17)
    d_hpc_t, h_hpc_t = encode_dope_research_view(
        dope_model, v_hpc_t, clm_hpc_t, level="high", conditioning_level="high"
    )

    # 2. 5-seed Random-Router control (`R_randrouter`) on `(v_hpc_t, clm_hpc_t)`
    rand_d629 = np.zeros((len(rand_models), 629), dtype=np.float32)
    for r_i, r_mod in enumerate(rand_models):
        dr_t, _ = encode_dope_research_view(
            r_mod, v_hpc_t, clm_hpc_t, level="high", conditioning_level="high"
        )
        rand_d629[r_i] = dr_t[0].numpy()

    # 3. Native 23-landmark 5-fold FIT-CV (`native_high` & `native_ultra`)
    rep = compute_native_23_landmark_report(x_fit, y_fit)
    cond = build_high_and_ultra_conditioning_from_fit(x_fit, y_fit, rep)
    v_nhi_t = torch.from_numpy(cond["vec256_high"]).reshape(1, 256)
    clm_nhi_t = torch.from_numpy(cond["col_lm_high"]).reshape(1, 64, 17)
    d_nhi_t, h_nhi_t = encode_dope_research_view(
        dope_model, v_nhi_t, clm_nhi_t, level="high", conditioning_level="high"
    )
    v_ult_t = torch.from_numpy(cond["vec256_ultra"]).reshape(1, 256)
    clm_ult_t = torch.from_numpy(cond["col_lm_ultra"]).reshape(1, 64, 17)
    d_ult_t, h_ult_t = encode_dope_research_view(
        dope_model, v_ult_t, clm_ult_t, level="ultra", conditioning_level="ultra"
    )

    # 4. Scale-invariant, column-permutation-invariant, extrema-free `196D` `R_packet_v1`
    enc_hi = encode_fit_table_for_packet_v1(
        dope_model, cond["x_scaled"], cond["y_scaled"], rep, level="high", include_v3_primitives=True
    )
    pkt_v1 = np.asarray(packet_features_v1(enc_hi, include_target_bundle=True), dtype=np.float32)
    pkt_notarget = np.asarray(packet_features_v1(enc_hi, include_target_bundle=False), dtype=np.float32)

    enc_ul = encode_fit_table_for_packet_v1(
        dope_model, cond["x_scaled"], cond["y_scaled"], rep, level="ultra", include_v3_primitives=True
    )
    pkt_v1_ultra = np.asarray(packet_features_v1(enc_ul, include_target_bundle=True), dtype=np.float32)

    return {
        "idx": idx,
        "cid": cid,
        "n_fit": n_fit,
        "p_features": p_feat,
        "R_shape": r_shape,
        "vec256_hpc": v_hpc_np.astype(np.float32),
        "col_lm_hpc": clm_hpc_np,
        "col_lm_34_hpc": pool_col_landmarks_34d(clm_hpc_np),
        "dope629_hpc": d_hpc_t[0].numpy(),
        "hidden256_hpc": h_hpc_t[0].numpy(),
        "dope629_randrouter_5": rand_d629,
        "gains23_ultra": cond["gains23_ultra"],
        "mask23_ultra": cond["mask23_ultra"].astype(np.uint8),
        "total_landmark_ms": np.float32(rep["timing"]["total_ms"]),
        "fine_5_landmarks_ms": np.float32(rep["timing"]["fine_5_landmarks_ms"]),
        "vec256_native_high": cond["vec256_high"],
        "col_lm_34_native_high": pool_col_landmarks_34d(cond["col_lm_high"]),
        "dope629_native_high": d_nhi_t[0].numpy(),
        "hidden256_native_high": h_nhi_t[0].numpy(),
        "vec256_ultra": cond["vec256_ultra"],
        "col_lm_34_ultra": pool_col_landmarks_34d(cond["col_lm_ultra"]),
        "dope629_ultra": d_ult_t[0].numpy(),
        "hidden256_ultra": h_ult_t[0].numpy(),
        "R_packet_v1": pkt_v1,
        "R_packet_notarget": pkt_notarget,
        "R_packet_v1_ultra": pkt_v1_ultra,
    }


def _extract_splithalf_worker(task: tuple[int, str, str, str]) -> dict[str, Any]:
    idx, cid, jtt_path, expected_sha = task
    assert_memavail(15.0, f"splithalf:{idx}")
    dope_model = _WORKER_CTX["dope_model"]
    ws = _WORKER_CTX["ws"]
    x_fit, y_fit, meta = read_fit_only_jtt1(jtt_path, expected_sha256=expected_sha)
    n_fit = int(meta["n_fit"])
    p_feat = int(meta["n_features"])
    rng = np.random.default_rng(20261118 + idx)
    perm = rng.permutation(n_fit)
    half = n_fit // 2
    halves = [(x_fit[perm[:half]], y_fit[perm[:half]]), (x_fit[perm[half : 2 * half]], y_fit[perm[half : 2 * half]])]
    out: dict[str, Any] = {"idx": idx, "cid": cid, "n_fit": n_fit, "p_features": p_feat}
    for h_i, (xh, yh) in enumerate(halves):
        y64 = yh.astype(np.float64)
        yn = ((y64 - y64.mean()) / max(float(y64.std()), 1e-8)).astype(np.float32)
        xf32 = landmark_float32(xh)
        v_hpc, col_hpc, _ = ws.fit_transform_and_extract_bundle_high(xf32, np.ascontiguousarray(yn))
        clm = np.zeros((64, 17), dtype=np.float32)
        nc = min(len(col_hpc), 64)
        clm[:nc, : min(col_hpc.shape[1], 17)] = col_hpc[:nc, :17]
        d_hpc, _ = encode_dope_research_view(
            dope_model,
            torch.from_numpy(v_hpc.astype(np.float32)).reshape(1, 256),
            torch.from_numpy(clm).reshape(1, 64, 17),
            level="high",
            conditioning_level="high",
        )
        rep = compute_native_23_landmark_report(xh, yh)
        cond = build_high_and_ultra_conditioning_from_fit(xh, yh, rep)
        d_ult, _ = encode_dope_research_view(
            dope_model,
            torch.from_numpy(cond["vec256_ultra"]).reshape(1, 256),
            torch.from_numpy(cond["col_lm_ultra"]).reshape(1, 64, 17),
            level="ultra",
            conditioning_level="ultra",
        )
        enc = encode_fit_table_for_packet_v1(
            dope_model, cond["x_scaled"], cond["y_scaled"], rep, level="high", include_v3_primitives=True
        )
        pkt = np.asarray(packet_features_v1(enc, include_target_bundle=True), dtype=np.float32)
        sfx = f"h{h_i + 1}"
        out[f"vec256_hpc_{sfx}"] = v_hpc.astype(np.float32)
        out[f"dope629_hpc_{sfx}"] = d_hpc[0].numpy()
        out[f"dope629_ultra_{sfx}"] = d_ult[0].numpy()
        out[f"R_packet_v1_{sfx}"] = pkt
    return out


def build_h470_cache_manifests(
    input_dir: str | Path,
    tokens_cache_path: str | Path,
) -> dict[str, Any]:
    """Compute exact WP01 7-field CacheAuthManifest objects for DL-Train, DL-Val, and SplitHalf."""
    try:
        from .cache_auth import (
            CacheAuthManifest,
            compute_context_inventory_sha256,
            compute_fit_rows_sha256,
            sha256_bytes,
            sha256_file,
        )
    except ImportError:
        from cache_auth import (
            CacheAuthManifest,
            compute_context_inventory_sha256,
            compute_fit_rows_sha256,
            sha256_bytes,
            sha256_file,
        )

    import json

    inp = Path(input_dir)
    selected = json.loads((inp / "selected.json").read_text(encoding="utf-8"))
    fused_src = inp / "cpu_exact_fused.py"
    fused_sha = sha256_file(fused_src)
    if fused_sha != CPU_EXACT_FUSED_SHA256:
        raise ValueError(f"cpu_exact_fused.py SHA-256 mismatch: {fused_sha}")

    dope_ckpt_sha = sha256_file(MICRO_ROOT / "assets" / "models" / "dope_mlc_v3_multitask_best.pt")
    if dope_ckpt_sha != DOPE_CKPT_FILE_SHA256:
        raise ValueError(f"dope_mlc_v3_multitask_best.pt SHA-256 mismatch: {dope_ckpt_sha}")

    stage1b_sha = sha256_file(
        MICRO_ROOT / "assets" / "projections" / "frozen_stage1_stage2_inference_params_v3.pt"
    )
    hpc_so_sha = sha256_file(MICRO_ROOT / "bin" / "libjope_hpc.so")
    tc_sha = sha256_file(tokens_cache_path)
    stage1a_bridge_sha = sha256_bytes(
        f"fused={fused_sha}|hpc_so={hpc_so_sha}|tokens_cache={tc_sha}|dope_composite={DOPE_FUSED_COMPOSITE_SHA256}".encode(
            "utf-8"
        )
    )

    out_manifests: dict[str, Any] = {"selected": selected}
    for role, split_tag in (("train", "dl_train1219"), ("dev", "dl_val91")):
        recs = selected[role]
        cids = [str(r["context_sha256"]) for r in recs]
        fit_tables = {}
        q_counts = {}
        n_fits = []
        for r in recs:
            cid = str(r["context_sha256"])
            xf, yf, meta = read_fit_only_jtt1(
                inp / "tables" / f"{cid}.jtt1",
                expected_sha256=str(r["train_only_jtt1_sha256"]),
            )
            fit_tables[cid] = (xf, yf)
            q_counts[cid] = int(meta["n_query"])
            n_fits.append(int(meta["n_fit"]))
        ctx_sha = compute_context_inventory_sha256(recs)
        fit_sha = compute_fit_rows_sha256(cids, fit_tables, q_counts)
        manifest = CacheAuthManifest(
            dope_weights_sha256=DOPE_FUSED_COMPOSITE_SHA256,
            stage1a_mode_and_bridge_sha256=stage1a_bridge_sha,
            prep_recipe=f"h470-all-views-v1:{split_tag}:native23cv5+packet_v1+hpc_high+sketch",
            context_sha256=ctx_sha,
            fit_rows_sha256=fit_sha,
            conditioning_level="ultra",
            stage1b_sha256=stage1b_sha,
        )
        out_manifests[role] = {
            "manifest": manifest,
            "cids": cids,
            "records": recs,
            "n_fits": np.asarray(n_fits, dtype=np.int32),
        }
        del fit_tables

    # Deterministic 128-context stratified DL-Train subset across 4 n_fit quartiles (32 per quartile, n_fit >= 40)
    tr_recs = selected["train"]
    tr_n = out_manifests["train"]["n_fits"]
    eligible = [i for i, nf in enumerate(tr_n.tolist()) if nf >= 40]
    qs = np.quantile(tr_n[eligible], [0.25, 0.50, 0.75])
    bins = [
        [i for i in eligible if tr_n[i] <= qs[0]],
        [i for i in eligible if qs[0] < tr_n[i] <= qs[1]],
        [i for i in eligible if qs[1] < tr_n[i] <= qs[2]],
        [i for i in eligible if tr_n[i] > qs[2]],
    ]
    rng_sh = np.random.default_rng(20261118)
    chosen_idx: list[int] = []
    for b_list in bins:
        pick = sorted(rng_sh.choice(b_list, size=min(32, len(b_list)), replace=False).tolist())
        chosen_idx.extend(pick)
    sh_recs = [tr_recs[i] for i in chosen_idx]
    sh_cids = [str(r["context_sha256"]) for r in sh_recs]
    sh_tables = {}
    for r in sh_recs:
        cid = str(r["context_sha256"])
        xf, yf, _ = read_fit_only_jtt1(
            inp / "tables" / f"{cid}.jtt1", expected_sha256=str(r["train_only_jtt1_sha256"])
        )
        sh_tables[cid] = (xf, yf)
    sh_manifest = CacheAuthManifest(
        dope_weights_sha256=DOPE_FUSED_COMPOSITE_SHA256,
        stage1a_mode_and_bridge_sha256=stage1a_bridge_sha,
        prep_recipe="h470-splithalf128-v1:dl_train_only",
        context_sha256=compute_context_inventory_sha256(sh_recs),
        fit_rows_sha256=compute_fit_rows_sha256(sh_cids, sh_tables),
        conditioning_level="ultra",
        stage1b_sha256=stage1b_sha,
    )
    out_manifests["splithalf"] = {
        "manifest": sh_manifest,
        "cids": sh_cids,
        "records": sh_recs,
        "train_indices": np.asarray(chosen_idx, dtype=np.int32),
    }
    return out_manifests


def build_or_load_h470_feature_caches(
    *,
    input_dir: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    tokens_cache_path: str | Path | None = None,
    h58_cache_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    num_workers: int = 8,
    include_val91: bool = False,
) -> dict[str, Any]:
    """Load or build WP01-authenticated feature caches.

    By default `include_val91=False` so Part A, Part B, and Part C CV never even open
    the `dl_val91` cache archive! Only final `REGCHECK` sets `include_val91=True` (or
    `--prebuild-all` during raw FIT-only feature caching before any outcome labels are read).
    """
    try:
        from .cache_auth import cache_paths_for_manifest, load_authenticated_npz, save_authenticated_npz
    except ImportError:
        from cache_auth import cache_paths_for_manifest, load_authenticated_npz, save_authenticated_npz

    import multiprocessing as mp

    inp = Path(input_dir) if input_dir is not None else (DATA_ROOT / "hwave-h469" / "input")
    ckpt_path = Path(ckpt_path) if ckpt_path is not None else (DATA_ROOT / "hwave-h469" / "round-3175.pt")
    tokens_cache_path = (
        Path(tokens_cache_path) if tokens_cache_path is not None else (DATA_ROOT / "hwave-h469" / "tokens_cache.pt")
    )
    h58_cache_path = (
        Path(h58_cache_path)
        if h58_cache_path is not None
        else (DATA_ROOT / "h58" / "cache" / "context_features.npz")
    )
    cdir = Path(cache_dir) if cache_dir is not None else (DATA_ROOT / "hwave-h470" / "cache")
    cdir.mkdir(parents=True, exist_ok=True)
    manifests = build_h470_cache_manifests(inp, tokens_cache_path)

    roles_to_prepare = ["train"] + (["dev"] if include_val91 else [])
    need_extract_roles = []
    for role in roles_to_prepare:
        tag = "h470_features_dl_train1219" if role == "train" else "h470_features_dl_val91"
        npz_p, meta_p = cache_paths_for_manifest(cdir, tag, manifests[role]["manifest"])
        if not (npz_p.is_file() and meta_p.is_file()):
            need_extract_roles.append(role)

    sh_npz, sh_meta = cache_paths_for_manifest(
        cdir, "h470_splithalf_dl_train128", manifests["splithalf"]["manifest"]
    )
    need_sh = not (sh_npz.is_file() and sh_meta.is_file())

    if need_extract_roles or need_sh:
        assert_memavail(15.0, "before_pool")
        tc = torch.load(tokens_cache_path, map_location="cpu", weights_only=True)
        h58 = np.load(h58_cache_path, allow_pickle=False) if Path(h58_cache_path).is_file() else None
        h58_map = (
            {(c.decode() if isinstance(c, bytes) else str(c)): i for i, c in enumerate(h58["cids"].tolist())}
            if h58 is not None
            else {}
        )
        import gc
        import importlib.util

        ctx = mp.get_context("spawn")
        dope_ckpt_str = str((MICRO_ROOT / "assets" / "models" / "dope_mlc_v3_multitask_best.pt").resolve())
        staged_role_arrays: dict[str, tuple[list[str], dict[str, np.ndarray]]] = {}
        staged_sh_arrays: dict[str, np.ndarray] | None = None

        with ctx.Pool(
            processes=max(1, int(num_workers)),
            initializer=_init_extraction_worker,
            initargs=(dope_ckpt_str,),
            maxtasksperchild=64,
        ) as pool:
            for role in need_extract_roles:
                recs = manifests[role]["records"]
                tasks = [
                    (
                        i,
                        str(r["context_sha256"]),
                        str(inp / "tables" / f"{r['context_sha256']}.jtt1"),
                        str(r["train_only_jtt1_sha256"]),
                    )
                    for i, r in enumerate(recs)
                ]
                t0 = time.perf_counter()
                rows_out: list[dict[str, Any] | None] = [None] * len(tasks)
                done = 0
                for item in pool.imap_unordered(_extract_single_context_worker, tasks, chunksize=4):
                    rows_out[item["idx"]] = item
                    done += 1
                    if done % 100 == 0 or done == len(tasks):
                        el = time.perf_counter() - t0
                        print(
                            f"[extract:{role}] {done}/{len(tasks)} ({el:.1f}s, mem={memavail_gb():.1f}GB)",
                            flush=True,
                        )
                ordered = [r for r in rows_out if r is not None]
                assert len(ordered) == len(tasks)
                arrays: dict[str, np.ndarray] = {}
                skip_keys = {"idx", "cid"}
                for k in ordered[0]:
                    if k in skip_keys:
                        continue
                    if k == "dope629_randrouter_5":
                        arrays[k] = np.stack([r[k] for r in ordered], axis=1)  # [5, N, 629]
                    else:
                        arrays[k] = np.stack([np.asarray(r[k]) for r in ordered], axis=0)
                cids_role = [r["cid"] for r in ordered]
                staged_role_arrays[role] = (cids_role, arrays)

            if need_sh:
                sh_recs = manifests["splithalf"]["records"]
                sh_tasks = [
                    (
                        i,
                        str(r["context_sha256"]),
                        str(inp / "tables" / f"{r['context_sha256']}.jtt1"),
                        str(r["train_only_jtt1_sha256"]),
                    )
                    for i, r in enumerate(sh_recs)
                ]
                t0 = time.perf_counter()
                sh_rows: list[dict[str, Any] | None] = [None] * len(sh_tasks)
                for item in pool.imap_unordered(_extract_splithalf_worker, sh_tasks, chunksize=4):
                    sh_rows[item["idx"]] = item
                sh_ord = [r for r in sh_rows if r is not None]
                staged_sh_arrays = {
                    "train_indices": manifests["splithalf"]["train_indices"],
                }
                for k in sh_ord[0]:
                    if k in ("idx", "cid"):
                        continue
                    staged_sh_arrays[k] = np.stack([np.asarray(r[k]) for r in sh_ord], axis=0)
                print(f"[extract:splithalf128_workers] done ({time.perf_counter() - t0:.1f}s)", flush=True)

        gc.collect()
        assert_memavail(15.0, "before_single_fused_net")
        torch.set_num_threads(4)
        spec_f = importlib.util.spec_from_file_location("cpu_exact_fused", str((inp / "cpu_exact_fused.py").resolve()))
        assert spec_f is not None and spec_f.loader is not None
        cf = importlib.util.module_from_spec(spec_f)
        spec_f.loader.exec_module(cf)
        net = cf.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
        ck = torch.load(str(Path(ckpt_path).resolve()), map_location="cpu", weights_only=True)
        net.load_state_dict(ck["state_dict"], strict=True)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            for role, (cids_role, arrays) in staged_role_arrays.items():
                tag = "h470_features_dl_train1219" if role == "train" else "h470_features_dl_val91"
                recs = manifests[role]["records"]
                n_r = len(recs)
                v_hpc_t = torch.from_numpy(arrays["vec256_hpc"])
                clm_hpc_t = torch.from_numpy(arrays["col_lm_hpc"])
                d_hpc_t = torch.from_numpy(arrays["dope629_hpc"])
                _, lm48_hpc_t = net._run_dope_mlc_v3_and_lm48(v_hpc_t, clm_hpc_t, use_landmarks=True)
                lm80z_hpc_t = net._compute_lm_smart_80d_gpu(lm48_hpc_t)
                tok_hpc_t, msk_hpc_t, lm48z_hpc_t = net._project_stage1b_tokens_batched_gpu(lm48_hpc_t, d_hpc_t)
                arrays["lm48_hpc"] = lm48_hpc_t.numpy()
                arrays["lm48_z_hpc"] = lm48z_hpc_t.numpy()
                arrays["lm80_z_hpc"] = lm80z_hpc_t.numpy()
                arrays["tokens_10x32_hpc"] = tok_hpc_t.numpy()
                arrays["mask_10_hpc"] = msk_hpc_t.numpy()

                # Native high Stage-1B tokens for Part A.1 comparison
                v_nhi_t = torch.from_numpy(arrays["vec256_native_high"])
                v_nhi_norm = torch.clamp((v_nhi_t - net.step2_lm_mean_1x256) / net.step2_lm_std_1x256, -6.0, 6.0)
                lm48_nhi_t = v_nhi_norm @ net.step2_svd_basis_256x48
                tok_nhi_t, msk_nhi_t, _ = net._project_stage1b_tokens_batched_gpu(
                    lm48_nhi_t, torch.from_numpy(arrays["dope629_native_high"])
                )
                arrays["tokens_10x32_native_high"] = tok_nhi_t.numpy()
                arrays["mask_10_native_high"] = msk_nhi_t.numpy()

                v_sk_l, clm34_sk_l, d_sk_l, lm48z_sk_l, tok_sk_l, msk_sk_l = [], [], [], [], [], []
                v_nl_l, d_nl_l, tok_nl_l = [], [], []
                for r in recs:
                    cid = str(r["context_sha256"])
                    xf, yf, _ = read_fit_only_jtt1(
                        inp / "tables" / f"{cid}.jtt1", expected_sha256=str(r["train_only_jtt1_sha256"])
                    )
                    y64 = yf.astype(np.float64)
                    yn = ((y64 - y64.mean()) / max(float(y64.std()), 1e-8)).astype(np.float32)
                    xf32 = landmark_float32(xf)
                    xt = torch.from_numpy(xf32)[None, :, :]
                    yt = torch.from_numpy(yn)[None, :]
                    rm = torch.ones((1, xf.shape[0]), dtype=torch.float32)
                    cm = torch.ones((1, xf.shape[1]), dtype=torch.float32)
                    v_sk, clm_sk = net._extract_stage1a_landmarks_batched_gpu(xt, yt, rm, cm)
                    d_sk, lm_sk = net._run_dope_mlc_v3_and_lm48(v_sk, clm_sk, use_landmarks=True)
                    tok_sk, msk_sk, lm48z_sk = net._project_stage1b_tokens_batched_gpu(lm_sk, d_sk)
                    v_nl, clm_nl = net._extract_stage1a_fast_moments_batched_gpu(xt, yt, rm, cm)
                    d_nl, lm_nl = net._run_dope_mlc_v3_and_lm48(v_nl, clm_nl, use_landmarks=False)
                    tok_nl, _, _ = net._project_stage1b_tokens_batched_gpu(lm_nl, d_nl)
                    v_sk_l.append(v_sk[0].numpy())
                    clm34_sk_l.append(pool_col_landmarks_34d(clm_sk[0].numpy()))
                    d_sk_l.append(d_sk[0].numpy())
                    lm48z_sk_l.append(lm48z_sk[0].numpy())
                    tok_sk_l.append(tok_sk[0].numpy())
                    msk_sk_l.append(msk_sk[0].numpy())
                    v_nl_l.append(v_nl[0].numpy())
                    d_nl_l.append(d_nl[0].numpy())
                    tok_nl_l.append(tok_nl[0].numpy())

                arrays["vec256_sketch"] = np.stack(v_sk_l, axis=0)
                arrays["col_lm_34_sketch"] = np.stack(clm34_sk_l, axis=0)
                arrays["dope629_sketch"] = np.stack(d_sk_l, axis=0)
                arrays["lm48_z_sketch"] = np.stack(lm48z_sk_l, axis=0)
                arrays["tokens_10x32_sketch"] = np.stack(tok_sk_l, axis=0)
                arrays["mask_10_sketch"] = np.stack(msk_sk_l, axis=0)
                arrays["vec256_nolm"] = np.stack(v_nl_l, axis=0)
                arrays["dope629_nolm"] = np.stack(d_nl_l, axis=0)
                arrays["tokens_10x32_nolm"] = np.stack(tok_nl_l, axis=0)
                arrays["tokens_10x32_campaign"] = np.stack(
                    [tc["token"][c].numpy().astype(np.float32) for c in cids_role], axis=0
                )
                arrays["mask_10_campaign"] = np.stack(
                    [tc["mask"][c].numpy().astype(np.float32) for c in cids_role], axis=0
                )
                if h58 is not None and all(c in h58_map for c in cids_role):
                    h_tok = np.stack([h58["tokens_10x32"][h58_map[c]] for c in cids_role], axis=0)
                    h_d629 = np.stack([h58["dope629"][h58_map[c]] for c in cids_role], axis=0)
                    max_dtok = float(np.max(np.abs(arrays["tokens_10x32_hpc"] - h_tok)))
                    max_ddope = float(np.max(np.abs(arrays["dope629_hpc"] - h_d629)))
                    arrays["h58_parity_max_diff"] = np.array([max_dtok, max_ddope], dtype=np.float32)
                    if max(max_dtok, max_ddope) >= 1e-6:
                        raise RuntimeError(
                            f"DVP-S2 parity failure on {role}: max_dtok={max_dtok}, max_ddope={max_ddope}"
                        )
                save_authenticated_npz(cdir, tag, manifests[role]["manifest"], arrays)
                print(f"[extract:{role}] saved authenticated archive ({n_r} contexts)", flush=True)

            if staged_sh_arrays is not None:
                sh_recs = manifests["splithalf"]["records"]
                sk1, sk2 = [], []
                for idx_sh, r in enumerate(sh_recs):
                    cid = str(r["context_sha256"])
                    xf, yf, meta = read_fit_only_jtt1(
                        inp / "tables" / f"{cid}.jtt1", expected_sha256=str(r["train_only_jtt1_sha256"])
                    )
                    nf = int(meta["n_fit"])
                    pf = int(meta["n_features"])
                    perm = np.random.default_rng(20261118 + idx_sh).permutation(nf)
                    half = nf // 2
                    for dst, (xh, yh) in ((sk1, (xf[perm[:half]], yf[perm[:half]])), (sk2, (xf[perm[half : 2 * half]], yf[perm[half : 2 * half]]))):
                        y64 = yh.astype(np.float64)
                        yn = ((y64 - y64.mean()) / max(float(y64.std()), 1e-8)).astype(np.float32)
                        xt = torch.from_numpy(landmark_float32(xh))[None, :, :]
                        yt = torch.from_numpy(yn)[None, :]
                        rm = torch.ones((1, half), dtype=torch.float32)
                        cm = torch.ones((1, pf), dtype=torch.float32)
                        v_sk, _ = net._extract_stage1a_landmarks_batched_gpu(xt, yt, rm, cm)
                        dst.append(v_sk[0].numpy())
                staged_sh_arrays["vec256_sketch_h1"] = np.stack(sk1, axis=0)
                staged_sh_arrays["vec256_sketch_h2"] = np.stack(sk2, axis=0)
                save_authenticated_npz(
                    cdir, "h470_splithalf_dl_train128", manifests["splithalf"]["manifest"], staged_sh_arrays
                )
                print("[extract:splithalf128] saved authenticated archive", flush=True)
        del net
        gc.collect()

    result: dict[str, Any] = {
        "manifests": manifests,
        "train": load_authenticated_npz(
            cdir, "h470_features_dl_train1219", manifests["train"]["manifest"]
        ),
        "splithalf": load_authenticated_npz(
            cdir, "h470_splithalf_dl_train128", manifests["splithalf"]["manifest"]
        ),
    }
    if include_val91:
        result["dev"] = load_authenticated_npz(
            cdir, "h470_features_dl_val91", manifests["dev"]["manifest"]
        )
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", type=Path, default=DATA_ROOT / "hwave-h470" / "cache")
    ap.add_argument("--prebuild-all", action="store_true", help="Prebuild FIT-only train + val91 caches")
    args = ap.parse_args()
    res = build_or_load_h470_feature_caches(
        cache_dir=args.cache_dir,
        num_workers=args.workers,
        include_val91=args.prebuild_all,
    )
    print(
        "Authenticated caches verified:",
        "train_shape=", res["train"]["R_packet_v1"].shape,
        "h58_parity=", res["train"].get("h58_parity_max_diff"),
        "splithalf_shape=", res["splithalf"]["R_packet_v1_h1"].shape,
        "val91_loaded=", "dev" in res,
    )
