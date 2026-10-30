"""Inter-Landmark Attention Modules and Low-Rank Vector-Leaf Multiclass Head (LR-VLOT).

Provides:
1. InterLandmarkAttentionBlock: 8-token cross-landmark self-attention + bilinear mixer
   that extracts high-order interactions between S3 landmarks, building-block probes,
   and inter-landmark summary statistics for the Deep Learning router.
2. Low-Rank Simplex Logit Basis + Shared-Split Vector-Leaf Oblivious Trees (LR-VLOT)
   for multiclass classification (3 <= K <= 32) with guaranteed class-support quotas.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .building_block_landmarks import compute_row_summary_features


class InterLandmarkAttentionBlock(nn.Module):
    """Cross-landmark self-attention and bilinear interaction mixer over 8 semantic landmark tokens."""

    def __init__(self, in_dim: int = 256, num_tokens: int = 8, token_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.token_proj = nn.Linear(in_dim // num_tokens, token_dim)
        self.qkv = nn.Linear(token_dim, 3 * token_dim)
        self.bilinear_u = nn.Linear(in_dim, hidden_dim // 2)
        self.bilinear_v = nn.Linear(in_dim, hidden_dim // 2)
        self.out_proj = nn.Linear(num_tokens * token_dim + hidden_dim // 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x256: torch.Tensor) -> torch.Tensor:
        squeeze = x256.ndim == 1
        if squeeze:
            x256 = x256.unsqueeze(0)
        b = x256.shape[0]
        tok_in = x256.view(b, self.num_tokens, self.in_dim // self.num_tokens)
        tok = F.silu(self.token_proj(tok_in))
        q, k, v = self.qkv(tok).chunk(3, dim=-1)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.token_dim), dim=-1)
        attended = (attn @ v + tok).reshape(b, -1)
        bilin = torch.tanh(self.bilinear_u(x256)) * F.silu(self.bilinear_v(x256))
        out = self.norm(self.out_proj(torch.cat([attended, bilin], dim=-1)))
        return out.squeeze(0) if squeeze else out


def build_enriched_numpy_basis(
    X: np.ndarray,
    gates: Dict[str, float],
    top_pair_indices: Sequence[Tuple[int, int]],
    *,
    level: str,
    include_v3_primitives: bool = False,
) -> np.ndarray:
    """Build basis matrix combining monotone, hinge, harmonic, pair, and optional v3 building blocks."""
    n, mw = X.shape[0], float(gates.get("monotone_w", 0.55))
    parts = [
        np.ones((n, 1), dtype=np.float64),
        X,
        mw * np.sign(X) * np.log1p(np.abs(X)),
        mw * np.tanh(0.75 * X),
        0.5 * mw * (np.abs(X) - 0.8),
    ]
    if level in ("high", "ultra"):
        hw, pw = float(gates.get("hinge_w", 0.5)), float(gates.get("pair_w", 0.5))
        parts.extend([hw * np.maximum(X - 0.5, 0.0), hw * np.maximum(-X - 0.5, 0.0), 0.35 * hw * np.sin(1.5 * X)])
        if top_pair_indices:
            parts.append(np.column_stack([pw * (X[:, i] * X[:, j]) for i, j in top_pair_indices]))
    if level == "ultra":
        tw = float(gates.get("tree_w", 0.8))
        parts.extend([tw * np.tanh(3.0 * (X - 1.0)), tw * np.tanh(3.0 * (X + 1.0)), 0.5 * tw * np.exp(-0.5 * (X ** 2))])

    if include_v3_primitives:
        aw = float(gates.get("asinh_atan_w", 0.45 * mw))
        rw = float(gates.get("row_summary_w", 0.40 * mw))
        row_s = compute_row_summary_features(X)[:, :4]  # [row_mean, row_std, row_l1, row_l2]
        parts.extend([aw * np.arcsinh(X), aw * np.arctan(X), rw * row_s])
        if level in ("high", "ultra"):
            fw = float(gates.get("periodic_fract_w", 0.30 * float(gates.get("hinge_w", 0.4))))
            parts.append(fw * np.cos(1.5 * X))
    return np.column_stack(parts)


def fit_vector_leaf_oblivious_trees(
    X: np.ndarray,
    residual_mat: np.ndarray,
    driver_prior: np.ndarray,
    *,
    n_trees: int,
    depth: int = 3,
    learning_rate: float = 0.18,
) -> List[Dict[str, Any]]:
    """Fit shared-split vector-leaf oblivious trees on (n, K) multiclass pseudo-residuals."""
    n, p = X.shape
    k_classes = residual_mat.shape[1]
    if n_trees <= 0 or n < 12 or p == 0:
        return []
    r = residual_mat.copy()
    trees: List[Dict[str, Any]] = []
    col_quantiles = [np.unique(np.quantile(X[:, f], np.linspace(0.15, 0.85, 7))) for f in range(p)]
    prior_norm = driver_prior / max(float(np.max(driver_prior)), 1e-6)

    for _ in range(n_trees):
        if float(np.std(r)) <= 1e-7:
            break
        r_c = r - np.mean(r, axis=0, keepdims=True)
        # Screen features using both linear residual correlation across all K classes & driver prior
        lin_corr = np.max(np.abs((X.T @ r_c) / float(n)), axis=1)
        score = lin_corr * (0.45 + 0.55 * prior_norm)
        top_feats = np.argsort(score)[::-1][: min(p, 8)]
        chosen_feats, chosen_thresh, leaf_idx = [], [], np.zeros(n, dtype=np.int64)
        for d in range(depth):
            best_gain, best_f, best_t = -1.0, int(top_feats[d % len(top_feats)]), 0.0
            n_leaves = 1 << (d + 1)
            for f in top_feats:
                for q in col_quantiles[f]:
                    new_idx = (leaf_idx << 1) | (X[:, f] >= q).astype(np.int64)
                    counts = np.bincount(new_idx, minlength=n_leaves).astype(np.float64) + 3.0
                    gain = 0.0
                    for c_idx in range(k_classes):
                        sums = np.bincount(new_idx, weights=r_c[:, c_idx], minlength=n_leaves)
                        gain += float(np.sum((sums ** 2) / counts))
                    if gain > best_gain:
                        best_gain, best_f, best_t = gain, int(f), float(q)
            chosen_feats.append(best_f)
            chosen_thresh.append(best_t)
            leaf_idx = (leaf_idx << 1) | (X[:, best_f] >= best_t).astype(np.int64)

        n_final = 1 << depth
        counts = np.bincount(leaf_idx, minlength=n_final).astype(np.float64) + 4.0
        leaf_mat = np.zeros((n_final, k_classes), dtype=np.float64)
        for c_idx in range(k_classes):
            sums = np.bincount(leaf_idx, weights=r[:, c_idx], minlength=n_final)
            leaf_mat[:, c_idx] = learning_rate * (sums / counts)
        r = r - leaf_mat[leaf_idx]
        trees.append({
            "features": chosen_feats,
            "thresholds": [round(t, 4) for t in chosen_thresh],
            "leaf_values": leaf_mat.astype(np.float16),
        })
    return trees


def eval_vector_leaf_oblivious_trees(X: np.ndarray, trees: Sequence[Dict[str, Any]], k_classes: int) -> np.ndarray:
    n = X.shape[0]
    out = np.zeros((n, k_classes), dtype=np.float64)
    for t in trees:
        idx = np.zeros(n, dtype=np.int64)
        for f, th in zip(t["features"], t["thresholds"]):
            idx = (idx << 1) | (X[:, f] >= th).astype(np.int64)
        lv = np.asarray(t["leaf_values"], dtype=np.float64)
        if lv.ndim == 1:
            lv = lv.reshape(-1, k_classes)
        out += lv[idx]
    return out


def fit_multiclass_target_bundle(
    X: np.ndarray,
    y_idx: np.ndarray,
    *,
    gates: Dict[str, float],
    col_prior: np.ndarray,
    emp_drivers: np.ndarray,
    level: str,
    include_v3_primitives: bool = True,
) -> Dict[str, Any]:
    """Fit Low-Rank Simplex Logit Basis + Vector-Leaf Oblivious Trees (LR-VLOT) on multiclass y_idx."""
    n, p = X.shape
    y_int = np.clip(np.rint(y_idx).astype(np.int64), 0, 31)
    u_classes = np.unique(y_int)
    k_classes = max(int(np.max(u_classes)) + 1 if u_classes.size else 3, 3)
    counts = np.bincount(y_int, minlength=k_classes).astype(np.float64)
    priors = np.maximum(counts / float(max(n, 1)), 1e-4)
    priors = priors / np.sum(priors)

    combined_prior = 0.5 * col_prior + 0.5 * emp_drivers
    top_cols = np.argsort(combined_prior)[::-1][: min(p, 8)]
    max_pairs = 0 if level in ("low", "small", "medium") else 8 if level == "high" else 16
    pairs = [(int(top_cols[a]), int(top_cols[b])) for a in range(len(top_cols)) for b in range(a + 1, len(top_cols))][:max_pairs]

    Phi = build_enriched_numpy_basis(X, gates, pairs, level=level, include_v3_primitives=include_v3_primitives)
    gram = (Phi.T @ Phi) / float(max(n, 1))
    eye = np.eye(Phi.shape[1], dtype=np.float64)
    eye[0, 0] = 1e-6
    alpha = max(float(gates.get("ridge_alpha", 0.25)), 0.05)

    # One-hot centered target in probability simplex space
    Y_oh = np.eye(k_classes, dtype=np.float64)[y_int]
    Y_centered = Y_oh - priors[None, :]
    W_full = np.linalg.solve(gram + alpha * eye, (Phi.T @ Y_centered) / float(max(n, 1)))  # (B, K)

    # Low-rank SVD factorization of W_full: W_full ~ U_basis @ V_basis.T
    rank_cap = 2 if level in ("low", "small") else 4 if level == "medium" else 8 if level == "high" else min(k_classes, 16)
    r_use = min(rank_cap, W_full.shape[0], W_full.shape[1])
    u_svd, s_svd, vt_svd = np.linalg.svd(W_full, full_matrices=False)
    U_basis = (u_svd[:, :r_use] * np.sqrt(s_svd[:r_use])[None, :]).astype(np.float16)
    V_basis = (vt_svd[:r_use, :].T * np.sqrt(s_svd[:r_use])[None, :]).astype(np.float16)

    pred_basis = Phi @ (U_basis.astype(np.float64) @ V_basis.astype(np.float64).T)
    residual_mat = Y_centered - pred_basis

    n_trees = 2 if level in ("low", "small") else 4 if level == "medium" else 12 if level == "high" else 24
    depth = 2 if level in ("low", "small", "medium") else 3
    trees = fit_vector_leaf_oblivious_trees(
        X, residual_mat, combined_prior, n_trees=n_trees, depth=depth, learning_rate=0.20
    )
    return {
        "task_kind": "multiclass",
        "k_classes": k_classes,
        "class_priors": priors.astype(np.float16),
        "gates": gates,
        "pairs": pairs,
        "include_v3_primitives": include_v3_primitives,
        "U_basis": U_basis,
        "V_basis": V_basis,
        "w_basis": U_basis.reshape(-1),
        "trees": trees,
        "res_net_mean": 0.0,
        "neural_w": 0.0,
        "residual_std": 0.05,
        "noise_scale": round(float(gates.get("noise_scale", 0.15)), 5),
        "class_prior": round(float(priors[0]), 6),
        "y_min": 0.0,
        "y_max": float(k_classes - 1),
    }


def predict_multiclass_from_bundle(
    X_syn: np.ndarray,
    bundle: Dict[str, Any],
    *,
    level: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Predict and sample synthetic multiclass targets with guaranteed class support preservation."""
    n = X_syn.shape[0]
    k_classes = int(bundle.get("k_classes", 3))
    priors = np.asarray(bundle.get("class_priors", np.full(k_classes, 1.0 / k_classes)), dtype=np.float64)
    priors = np.maximum(priors, 1e-4)
    priors = priors / np.sum(priors)

    Phi = build_enriched_numpy_basis(
        X_syn, bundle.get("gates", {}), bundle.get("pairs", []),
        level=level, include_v3_primitives=bool(bundle.get("include_v3_primitives", True)),
    )
    U = np.asarray(bundle["U_basis"], dtype=np.float64)
    V = np.asarray(bundle["V_basis"], dtype=np.float64)
    m = min(Phi.shape[1], U.shape[0])
    scores = priors[None, :] + Phi[:, :m] @ (U[:m] @ V.T) + eval_vector_leaf_oblivious_trees(X_syn, bundle.get("trees", []), k_classes)

    # Temperature-scaled Gumbel-max / quota-calibrated assignment so all training classes appear
    noise_amp = float(bundle.get("noise_scale", 0.15)) * (0.08 if level in ("low", "small", "medium") else 0.03)
    noisy_scores = scores + rng.normal(0.0, noise_amp, size=scores.shape)
    y_out = np.argmax(noisy_scores, axis=1).astype(np.float64)

    # Guarantee every active class in D_train has at least min_quota rows in D_syn
    if n >= k_classes:
        used_rows = set()
        for c_idx in np.argsort(priors):
            if priors[c_idx] <= 1e-5:
                continue
            present = np.where(y_out == c_idx)[0]
            if present.size == 0:
                order = np.argsort(noisy_scores[:, c_idx])[::-1]
                for cand in order:
                    if int(cand) not in used_rows:
                        y_out[int(cand)] = float(c_idx)
                        used_rows.add(int(cand))
                        break
            else:
                used_rows.add(int(present[0]))
    return y_out


def to_normal_scores(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    for j in range(p):
        u = (np.argsort(np.argsort(X[:, j])).astype(np.float64) + 0.5) / float(max(n, 1))
        out[:, j] = erfinv_approx(2.0 * u - 1.0) * math.sqrt(2.0)
    return out


def erf_approx(x: np.ndarray) -> np.ndarray:
    sign, ax = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-(ax ** 2)))


def erfinv_approx(y: np.ndarray) -> np.ndarray:
    y, a = np.clip(y, -0.999999, 0.999999), 0.147
    ln = np.log(1.0 - y ** 2)
    term1 = 2.0 / (math.pi * a) + 0.5 * ln
    return np.sign(y) * np.sqrt(np.maximum(np.sqrt(term1 ** 2 - ln / a) - term1, 0.0))


def fit_copula_regimes(Z: np.ndarray, *, n_regimes: int, sharpness: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = Z.shape
    if n_regimes <= 1 or n < 16:
        return np.array([1.0], dtype=np.float64), np.zeros((1, p), dtype=np.float64), np.ones((1, p), dtype=np.float64)
    u, s, _ = np.linalg.svd(Z, full_matrices=False)
    score = u[:, 0] * s[0]
    bounds = np.quantile(score, np.linspace(0.0, 1.0, n_regimes + 1))
    weights, means, scales = np.zeros(n_regimes, dtype=np.float64), np.zeros((n_regimes, p), dtype=np.float64), np.ones((n_regimes, p), dtype=np.float64)
    for k in range(n_regimes):
        mask = ((score >= bounds[k]) & (score < bounds[k + 1])) if k < n_regimes - 1 else (score >= bounds[k])
        if np.sum(mask) >= 2:
            weights[k] = float(np.mean(mask))
            means[k] = sharpness * np.mean(Z[mask], axis=0)
            scales[k] = (1.0 - 0.5 * sharpness) + 0.5 * sharpness * np.maximum(np.std(Z[mask], axis=0), 0.25)
        else:
            weights[k] = 1.0 / float(n_regimes)
    return weights / np.sum(weights), means, scales


def fit_oblivious_trees(
    X: np.ndarray, residual: np.ndarray, driver_prior: np.ndarray, *, n_trees: int, depth: int = 3, learning_rate: float = 0.18
) -> List[Dict[str, Any]]:
    n, p = X.shape
    if n_trees <= 0 or n < 12 or p == 0:
        return []
    r, trees = residual.copy(), []
    col_quantiles = [np.unique(np.quantile(X[:, f], np.linspace(0.15, 0.85, 7))) for f in range(p)]
    prior_norm = driver_prior / max(float(np.max(driver_prior)), 1e-6)
    for _ in range(n_trees):
        if float(np.std(r)) <= 1e-7:
            break
        r_centered = r - float(np.mean(r))
        corrs = np.abs((X.T @ r_centered) / float(n)) * (0.5 + 0.5 * prior_norm)
        top_feats = np.argsort(corrs)[::-1][: min(p, 8)]
        chosen_feats, chosen_thresh, leaf_idx = [], [], np.zeros(n, dtype=np.int64)
        for d in range(depth):
            best_gain, best_f, best_t = -1.0, int(top_feats[d % len(top_feats)]), 0.0
            for f in top_feats:
                for q in col_quantiles[f]:
                    new_idx = (leaf_idx << 1) | (X[:, f] >= q).astype(np.int64)
                    counts = np.bincount(new_idx, minlength=1 << (d + 1)).astype(np.float64)
                    sums = np.bincount(new_idx, weights=r_centered, minlength=1 << (d + 1))
                    gain = float(np.sum((sums ** 2) / (counts + 3.0)))
                    if gain > best_gain:
                        best_gain, best_f, best_t = gain, int(f), float(q)
            chosen_feats.append(best_f)
            chosen_thresh.append(best_t)
            leaf_idx = (leaf_idx << 1) | (X[:, best_f] >= best_t).astype(np.int64)
        counts = np.bincount(leaf_idx, minlength=1 << depth).astype(np.float64)
        sums = np.bincount(leaf_idx, weights=r, minlength=1 << depth)
        leaf_vals = learning_rate * (sums / (counts + 4.0))
        r = r - leaf_vals[leaf_idx]
        trees.append({"features": chosen_feats, "thresholds": [round(t, 4) for t in chosen_thresh], "leaf_values": leaf_vals.astype(np.float16)})
    return trees


def eval_oblivious_trees(X: np.ndarray, trees: Sequence[Dict[str, Any]]) -> np.ndarray:
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for t in trees:
        idx = np.zeros(n, dtype=np.int64)
        for f, th in zip(t["features"], t["thresholds"]):
            idx = (idx << 1) | (X[:, f] >= th).astype(np.int64)
        out += np.asarray(t["leaf_values"], dtype=np.float64)[idx]
    return out

