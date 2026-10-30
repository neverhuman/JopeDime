"""Unified 4-Level Landmark- and Inter-Landmark-Conditioned Tabular Model (DOPE-MLC-v3).

Single unified model supporting four nested encoding tiers across Regression, Binary, and Multiclass:
- Small  (1,024 coords / 2 KiB cap): conditioned on 6 Macro landmarks + core building-block probes
- Medium (4,096 coords / 4 KiB cap): conditioned on 9 Macro landmarks + heavy-tail/row building blocks
- High   (8,192 coords / 16 KiB cap): adds 9 Larger landmarks (18 total) + cross-family ratios & 2D spectrum
- Ultra  (10,240 coords / 64 KiB cap): adds 5 Fine landmarks (23 total) + all 16 building blocks & 48 inter-landmark stats
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .budget_planner import TIER_COORD_WIDTHS, plan_tier_envelope
from .building_block_landmarks import build_v3_conditioning_bundle
from .multiclass_head import (
    InterLandmarkAttentionBlock, build_enriched_numpy_basis, erf_approx as _erf_approx,
    erfinv_approx as _erfinv_approx, eval_oblivious_trees as _eval_oblivious_trees,
    fit_copula_regimes as _fit_copula_regimes, fit_multiclass_target_bundle,
    fit_oblivious_trees as _fit_oblivious_trees, predict_multiclass_from_bundle,
    to_normal_scores as _to_normal_scores,
)

LEVEL_WIDTHS: Dict[str, int] = {"medium": 4096, "high": 8192, "ultra": 10240}
ALL_LEVEL_WIDTHS: Dict[str, int] = TIER_COORD_WIDTHS
MAX_COLS = 64
QUANTILE_KNOTS = np.linspace(0.01, 0.99, 15)
_LEVEL_INDEX = {"low": 0, "small": 0, "medium": 0, "high": 1, "ultra": 2}


class LandmarkHyperRouter(nn.Module):
    """Maps level-masked landmark + inter-landmark summary vectors (256 dims) to structural gates and latents."""

    def __init__(self, in_dim: int = 140, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.small_embed = nn.Parameter(torch.zeros(hidden_dim))
        self.input_norm = nn.LayerNorm(140)
        self.v3_inter_norm = nn.LayerNorm(116)
        self.backbone = nn.Sequential(
            nn.Linear(140, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
        )
        self.inter_attn = InterLandmarkAttentionBlock(in_dim=256, num_tokens=8, token_dim=32, hidden_dim=hidden_dim)
        self.gate_head = nn.Linear(hidden_dim, 10)
        self.v3_gate_head = nn.Linear(hidden_dim, 4)
        self.task_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.dist_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.landmark_recon_head = nn.Linear(hidden_dim, 23)
        self.bb_recon_head = nn.Linear(hidden_dim, 16)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.v3_gate_head.weight)
        nn.init.zeros_(self.v3_gate_head.bias)
        with torch.no_grad():
            self.gate_head.bias.copy_(torch.tensor([0.0, 0.4, 0.3, 0.3, 0.2, 0.8, 0.5, 1.2, 0.5, -0.2], dtype=torch.float32))

    def forward(self, desc_vec: torch.Tensor, level: str) -> Dict[str, torch.Tensor]:
        lvl_idx = _LEVEL_INDEX[level]
        squeeze = desc_vec.ndim == 1
        if squeeze:
            desc_vec = desc_vec.unsqueeze(0)
        b, d = desc_vec.shape
        x140 = self.input_norm(desc_vec[:, :140])
        if d >= 256:
            x_extra = self.v3_inter_norm(desc_vec[:, 140:256])
            x256 = torch.cat([x140, x_extra], dim=-1)
        else:
            x256 = torch.cat([x140, torch.zeros((b, 116), device=desc_vec.device, dtype=desc_vec.dtype)], dim=-1)

        lvl_t = torch.full((b,), lvl_idx, dtype=torch.long, device=desc_vec.device)
        lvl_emb = self.level_embed(lvl_t) + (self.small_embed.unsqueeze(0) if level in ("low", "small") else 0.0)
        h = self.backbone(x140) + lvl_emb + self.inter_attn(x256)
        raw_gates = self.gate_head(h)
        raw_v3 = self.v3_gate_head(h)
        is_high = 1.0 if level in ("high", "ultra") else 0.0
        is_small = level in ("low", "small")
        gates = {
            "ridge_alpha": F.softplus(raw_gates[:, 0]) + (0.45 if is_small else 0.35 if level == "medium" else 0.15 if level == "high" else 0.08),
            "monotone_w": torch.sigmoid(raw_gates[:, 1]) * (0.50 if is_small else 0.55 if level == "medium" else 0.65),
            "hinge_w": torch.sigmoid(raw_gates[:, 2]) * 0.55 * is_high,
            "pair_w": torch.sigmoid(raw_gates[:, 3]) * 0.60 * is_high,
            "rff_w": torch.sigmoid(raw_gates[:, 4]) * 0.45 * is_high,
            "tree_w": torch.sigmoid(raw_gates[:, 5]) * (0.15 if is_small else 0.20 if level == "medium" else 0.70 if level == "high" else 1.00),
            "neural_w": torch.sigmoid(raw_gates[:, 6]) * (0.25 if is_small else 0.35 if level == "medium" else 0.65 if level == "high" else 1.00),
            "copula_retention": torch.sigmoid(raw_gates[:, 7]) * (0.60 if is_small else 0.68 if level == "medium" else 0.90 if level == "high" else 0.99),
            "regime_sharpness": torch.sigmoid(raw_gates[:, 8]) * (0.25 if is_small else 0.35 if level == "medium" else 0.75 if level == "high" else 1.00),
            "noise_scale": torch.sigmoid(raw_gates[:, 9]) * (0.35 if is_small else 0.32 if level == "medium" else 0.20 if level == "high" else 0.12),
            "asinh_atan_w": torch.sigmoid(raw_v3[:, 0]) * (0.35 if is_small else 0.55),
            "row_summary_w": torch.sigmoid(raw_v3[:, 1]) * (0.30 if is_small else 0.50),
            "periodic_fract_w": torch.sigmoid(raw_v3[:, 2]) * 0.45 * is_high,
            "het_scale_w": torch.sigmoid(raw_v3[:, 3]) * 0.40 * is_high,
        }
        task_lat, dist_lat = torch.tanh(self.task_latent_head(h)), torch.tanh(self.dist_latent_head(h))
        lm_pred, bb_pred = torch.tanh(self.landmark_recon_head(h)), torch.tanh(self.bb_recon_head(h))
        if squeeze:
            gates = {k: v.squeeze(0) for k, v in gates.items()}
            task_lat, dist_lat, lm_pred, bb_pred = task_lat.squeeze(0), dist_lat.squeeze(0), lm_pred.squeeze(0), bb_pred.squeeze(0)
        return {"gates": gates, "task_latent": task_lat, "dist_latent": dist_lat, "landmark_pred": lm_pred, "bb_pred": bb_pred}


class ConditionalHyperTargetNet(nn.Module):
    """Differentiable conditional target predictor P_phi(Y | X, z_task, col_landmarks)."""

    def __init__(self, max_cols: int = MAX_COLS, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.max_cols = max_cols
        self.film_gamma = nn.Linear(latent_dim, max_cols)
        self.film_beta = nn.Linear(latent_dim, max_cols)
        self.fc1 = nn.Linear(max_cols + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 1)
        self.col_gate_head = nn.Linear(latent_dim + 17, 1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        nn.init.zeros_(self.col_gate_head.weight)
        nn.init.zeros_(self.col_gate_head.bias)

    def compute_column_gates(
        self, task_latent: torch.Tensor, col_landmarks: Optional[torch.Tensor] = None, X_pad: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute per-column attention multiplier from task_latent and per-column landmark spectrum."""
        t_1d = task_latent.squeeze(0) if task_latent.ndim > 1 else task_latent
        c_pad = torch.zeros((self.max_cols, 17), device=t_1d.device, dtype=t_1d.dtype)
        if col_landmarks is not None:
            p_in, d_in = col_landmarks.shape
            p_take, d_take = min(p_in, self.max_cols), min(d_in, 17)
            c_pad[:p_take, :d_take] = col_landmarks[:p_take, :d_take].to(device=t_1d.device, dtype=t_1d.dtype)
        elif X_pad is not None and X_pad.shape[0] > 1:
            # Deterministic 17-dim column-distribution summary from X_pad at receiver decode time
            xm = X_pad.mean(dim=0)
            xs = X_pad.std(dim=0, unbiased=False).clamp(min=1e-6)
            xz = ((X_pad - xm.unsqueeze(0)) / xs.unsqueeze(0)).clamp(-6.0, 6.0)
            feats = torch.stack([
                xm, xs, (xz ** 3).mean(dim=0) / 6.0, ((xz ** 4).mean(dim=0) - 3.0).clamp(-3.0, 20.0) / 10.0,
                xz.abs().mean(dim=0), (xz > 0.0).to(xz.dtype).mean(dim=0),
                torch.asinh(xz).mean(dim=0), torch.atan(xz).mean(dim=0),
                torch.tanh(xz).mean(dim=0), F.relu(xz - 0.5).mean(dim=0),
                F.relu(-xz - 0.5).mean(dim=0), torch.sin(xz * 1.5).mean(dim=0),
                torch.cos(xz * 1.5).mean(dim=0), (xz - torch.floor(xz)).mean(dim=0),
                xz.min(dim=0).values / 6.0, xz.max(dim=0).values / 6.0, (xz.abs() < 0.1).to(xz.dtype).mean(dim=0),
            ], dim=-1)
            c_pad = feats.to(device=t_1d.device, dtype=t_1d.dtype)
        t_rep = t_1d.unsqueeze(0).expand(self.max_cols, -1)
        raw_g = self.col_gate_head(torch.cat([t_rep, c_pad], dim=-1)).squeeze(-1)
        return 1.0 + 0.25 * torch.tanh(raw_g)

    def forward(
        self, X_pad: torch.Tensor, task_latent: torch.Tensor, col_mask: torch.Tensor, col_landmarks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t_exp = (task_latent.unsqueeze(0) if task_latent.ndim == 1 else task_latent).expand(X_pad.shape[0], -1)
        c_gate = self.compute_column_gates(task_latent, col_landmarks, X_pad=X_pad)
        X_mod = (X_pad * c_gate.unsqueeze(0) * (1.0 + 0.3 * torch.tanh(self.film_gamma(t_exp))) + 0.1 * torch.tanh(self.film_beta(t_exp))) * col_mask.unsqueeze(0)
        h = F.silu(self.fc1(torch.cat([X_mod, t_exp], dim=-1)))
        return self.out_head(F.silu(self.fc2(h)) + h).squeeze(-1)


class CovariateRefinementNet(nn.Module):
    """Learned residual denoiser that refines copula-sampled covariates in normal-score space."""

    def __init__(self, max_cols: int = MAX_COLS, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_cols + latent_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, max_cols),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, X_pad: torch.Tensor, dist_latent: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
        d_exp = (dist_latent.unsqueeze(0) if dist_latent.ndim == 1 else dist_latent).expand(X_pad.shape[0], -1)
        return X_pad + 0.15 * torch.tanh(self.net(torch.cat([X_pad, d_exp], dim=-1))) * col_mask.unsqueeze(0)


class MultiLevelLandmarkModel(nn.Module):
    """Unified 4-Level Model (Small, Medium, High, Ultra) across Regression, Binary, and Multiclass."""

    def __init__(self, max_cols: int = MAX_COLS, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.max_cols, self.latent_dim = max_cols, latent_dim
        self.router = LandmarkHyperRouter(in_dim=256, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.target_net = ConditionalHyperTargetNet(max_cols=max_cols, latent_dim=latent_dim, hidden_dim=hidden_dim // 2)
        self.cov_net = CovariateRefinementNet(max_cols=max_cols, latent_dim=latent_dim, hidden_dim=hidden_dim // 2)

    @staticmethod
    def build_conditioning_vector(
        report: Dict[str, Any], X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, level: str, device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        bundle = build_v3_conditioning_bundle(report, X_train_scaled, y_train_scaled, level=level)
        rep_desc, tab_desc = bundle["rep_desc"], dict(bundle["tab_desc"])
        tab_desc["col_landmarks24"] = bundle["col_landmarks24"]
        tab_desc["v3_driver_weights"] = bundle["v3_driver_weights"]
        tab_desc["inter_landmark_vec48"] = bundle["inter_landmark_vec48"]
        tab_desc["building_block_vec68"] = bundle["building_block_vec68"]
        return torch.tensor(bundle["vec256"], dtype=torch.float32, device=device), rep_desc, tab_desc

    def forward_conditioning(
        self, X_train: torch.Tensor, y_train: torch.Tensor, X_query: torch.Tensor, y_query: torch.Tensor,
        desc_vec: torch.Tensor, emp_drivers: torch.Tensor, task: str, level: str,
        *, col_landmarks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = X_train.device
        n_tr, p = X_train.shape
        n_qr, p_use = X_query.shape[0], min(p, self.max_cols)
        X_tr, X_qr = X_train[:, :p_use], X_query[:, :p_use]

        routed = self.router(desc_vec, level=level)
        gates, task_lat, dist_lat = routed["gates"], routed["task_latent"], routed["dist_latent"]
        lm_pred, bb_pred = routed["landmark_pred"], routed["bb_pred"]

        X_tr_pad = torch.zeros((n_tr, self.max_cols), device=device, dtype=X_tr.dtype)
        X_qr_pad = torch.zeros((n_qr, self.max_cols), device=device, dtype=X_qr.dtype)
        col_mask = torch.zeros(self.max_cols, device=device, dtype=X_tr.dtype)
        X_tr_pad[:, :p_use], X_qr_pad[:, :p_use], col_mask[:p_use] = X_tr, X_qr, 1.0

        phi_tr, phi_qr = _build_differentiable_basis(X_tr, gates, level=level), _build_differentiable_basis(X_qr, gates, level=level)
        gram = (phi_tr.T @ phi_tr) / float(max(n_tr, 1))
        eye = torch.eye(phi_tr.shape[1], device=device, dtype=X_tr.dtype)
        eye[0, 0] = 1e-6
        alpha_mat = gram + gates["ridge_alpha"].clamp(1e-3, 50.0) * eye

        if task == "multiclass":
            y_tr_int, y_qr_int = y_train.long().clamp(0, 31), y_query.long().clamp(0, 31)
            k_cls = max(int(max(y_tr_int.max().item(), y_qr_int.max().item())) + 1, 3)
            Y_oh = F.one_hot(y_tr_int, num_classes=k_cls).to(dtype=X_tr.dtype)
            priors = Y_oh.mean(dim=0, keepdim=True).clamp(1e-3, 1.0)
            W_mc = torch.linalg.solve(alpha_mat, (phi_tr.T @ (Y_oh - priors)) / float(max(n_tr, 1)))
            res_qr = self.target_net(X_qr_pad, task_lat, col_mask, col_landmarks=col_landmarks).unsqueeze(-1)
            logits_qr = torch.log(priors) + 3.0 * (phi_qr @ W_mc) + 0.25 * gates["neural_w"] * res_qr
            cond_loss, lin_w = F.cross_entropy(logits_qr, y_qr_int), torch.mean(torch.abs(W_mc[1 : 1 + p_use]), dim=1)
        else:
            w_basis = torch.linalg.solve(alpha_mat, (phi_tr.T @ y_train) / float(max(n_tr, 1)))
            res_tr = self.target_net(X_tr_pad, task_lat, col_mask, col_landmarks=col_landmarks)
            res_qr = self.target_net(X_qr_pad, task_lat, col_mask, col_landmarks=col_landmarks)
            y_hat_qr = (phi_qr @ w_basis) + gates["neural_w"] * (res_qr - res_tr.mean())
            if task in ("classification", "binary"):
                p_bar = y_train.mean().clamp(0.02, 0.98)
                logits_qr = torch.log(p_bar / (1.0 - p_bar)) + 2.8 * (y_hat_qr - p_bar)
                cond_loss = 0.65 * F.binary_cross_entropy_with_logits(logits_qr, y_query) + 0.35 * F.mse_loss(torch.sigmoid(logits_qr), y_query)
            else:
                cond_loss = 0.60 * F.huber_loss(y_hat_qr, y_query, delta=1.0) + 0.40 * F.mse_loss(y_hat_qr, y_query)
            lin_w = torch.abs(w_basis[1 : 1 + p_use])

        X_cov_ref = self.cov_net(X_tr_pad, dist_lat, col_mask)[:, :p_use]
        cov_tr, cov_qr = (X_cov_ref.T @ X_cov_ref) / float(max(n_tr, 1)), (X_qr.T @ X_qr) / float(max(n_qr, 1))
        off_mask = 1.0 - torch.eye(p_use, device=device)
        target_cov = cov_qr * (gates["copula_retention"] * off_mask + torch.eye(p_use, device=device))
        dist_loss = F.mse_loss(cov_tr, target_cov) + 0.25 * F.mse_loss(X_cov_ref.mean(dim=0), X_qr.mean(dim=0))

        c_gate = self.target_net.compute_column_gates(task_lat, col_landmarks, X_pad=X_tr_pad)[:p_use]
        film_g = torch.abs(1.0 + 0.3 * torch.tanh(self.target_net.film_gamma(task_lat.unsqueeze(0)).squeeze(0)[:p_use]))
        pred_imp = lin_w * film_g * torch.abs(c_gate)
        driver_loss = F.l1_loss(pred_imp / (pred_imp.sum() + 1e-8), emp_drivers[:p_use] / (emp_drivers[:p_use].sum() + 1e-8))

        true_gains, true_mask = desc_vec[:23], desc_vec[23:46] > 0.5
        lm_loss = F.huber_loss(lm_pred[true_mask], true_gains[true_mask]) if true_mask.any() else lm_pred.sum() * 0.0
        if desc_vec.shape[-1] >= 204:
            true_bb = desc_vec[188:204]
            lm_loss = lm_loss + 0.5 * F.huber_loss(bb_pred, true_bb)
        total_loss = cond_loss + 0.25 * dist_loss + 0.20 * driver_loss + 0.10 * lm_loss
        return {
            "total": total_loss, "cond_y_loss": cond_loss.detach(), "dist_x_loss": dist_loss.detach(),
            "driver_loss": driver_loss.detach(), "landmark_loss": lm_loss.detach(),
            "gates": {k: float(v.detach().item()) for k, v in gates.items()},
        }

    @torch.no_grad()
    def encode_table(
        self, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, report: Dict[str, Any],
        *, task: str = "regression", level: str = "ultra", adaptive_budget: bool = False,
        include_v3_primitives: bool = False,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        vec256, rep_desc, tab_desc = self.build_conditioning_vector(report, X_train_scaled, y_train_scaled, level=level, device=device)
        routed = self.router(vec256, level=level)
        gates = {k: float(v.item()) for k, v in routed["gates"].items()}
        task_lat, dist_lat = routed["task_latent"].detach().cpu().numpy().astype(np.float16), routed["dist_latent"].detach().cpu().numpy().astype(np.float16)
        X, y = np.asarray(X_train_scaled, dtype=np.float64), np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)
        n, p = X.shape
        p_use, X_use = min(p, self.max_cols), X[:, : min(p, self.max_cols)]
        k_cls = int(len(np.unique(np.rint(y)))) if task == "multiclass" else (2 if task in ("classification", "binary") else 1)
        plan = plan_tier_envelope(p_use, level, k_classes=k_cls, adaptive_budget=adaptive_budget or level in ("low", "small"))
        if plan["latent_keep"] < 128:
            task_lat[plan["latent_keep"] :], dist_lat[plan["latent_keep"] :] = 0.0, 0.0
        n_knots = plan["n_knots"]
        knot_grid = np.linspace(0.005 if level not in ("low", "small", "medium") else 0.01, 0.995 if level not in ("low", "small", "medium") else 0.99, n_knots)
        col_knots = np.quantile(X_use, knot_grid, axis=0).T.astype(np.float16)
        col_means, col_stds = np.mean(X_use, axis=0).astype(np.float16), np.maximum(np.std(X_use, axis=0), 1e-6).astype(np.float16)
        is_binary_col = np.array([len(np.unique(X_use[:, j])) <= 2 for j in range(p_use)], dtype=bool)
        binary_probs = np.array([float(np.mean(X_use[:, j] > np.mean(X_use[:, j]))) if is_binary_col[j] else 0.0 for j in range(p_use)], dtype=np.float16)
        binary_low_high = np.array([[float(np.min(X_use[:, j])), float(np.max(X_use[:, j]))] if is_binary_col[j] else [0.0, 0.0] for j in range(p_use)], dtype=np.float16)
        discrete_support: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        max_disc = 0 if (adaptive_budget and level in ("low", "small", "medium") and p_use > 24) else (8 if level in ("low", "small", "medium") else 16 if level == "high" else 32)
        for j in range(p_use):
            if not is_binary_col[j] and max_disc > 0:
                u_vals, u_counts = np.unique(X_use[:, j], return_counts=True)
                if 2 < len(u_vals) <= max_disc:
                    discrete_support[j] = (u_vals.astype(np.float16), np.cumsum(u_counts.astype(np.float64) / float(max(n, 1))).astype(np.float16))
        Z_norm = _to_normal_scores(X_use)
        corr = np.nan_to_num((Z_norm.T @ Z_norm) / float(max(n, 1)), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        ret = max(float(gates["copula_retention"]), 0.72 if level in ("low", "small", "medium") else 0.92 if level == "high" else 0.99)
        eigvals, eigvecs = np.linalg.eigh(ret * corr + (1.0 - ret) * np.eye(p_use))
        eigvals, eigvecs = np.maximum(eigvals[::-1], 1e-5), eigvecs[:, ::-1]
        rank_use = plan["rank_use"]
        chol_factor = (eigvecs[:, :rank_use] * np.sqrt(eigvals[:rank_use])[None, :]).astype(np.float16)
        diag_res_std = np.sqrt(np.maximum(1.0 - np.sum(chol_factor.astype(np.float64) ** 2, axis=1), 1e-4)).astype(np.float16)
        regime_weights, regime_means, regime_scales = _fit_copula_regimes(Z_norm, n_regimes=plan["n_regimes"], sharpness=gates["regime_sharpness"])
        v3_emp = tab_desc.get("v3_driver_weights", tab_desc["empirical_driver_weights"])[:p_use]
        col_lm_t = torch.tensor(tab_desc["col_landmarks24"][:p_use], dtype=torch.float32, device=device) if "col_landmarks24" in tab_desc else None
        if task == "multiclass":
            target_bundle = fit_multiclass_target_bundle(
                X_use, y, gates=gates, col_prior=rep_desc["column_prior_importance"][:p_use],
                emp_drivers=v3_emp, level=level, include_v3_primitives=include_v3_primitives,
            )
        else:
            target_bundle = _fit_hierarchical_target_bundle(
                X_use, y, gates=gates, col_prior=rep_desc["column_prior_importance"][:p_use],
                emp_drivers=v3_emp, task=task, level=level, target_net=self.target_net,
                task_lat_tensor=routed["task_latent"], plan=plan, include_v3_primitives=include_v3_primitives,
                col_landmarks=col_lm_t,
            )
        width = ALL_LEVEL_WIDTHS[level]
        code_vec = _pack_code_vector(width=width, col_knots=col_knots, chol_factor=chol_factor, target_bundle=target_bundle, task_lat=task_lat, dist_lat=dist_lat)
        return {
            "level": level, "task": task, "code_width": width, "code_fp16": code_vec.astype(np.float16),
            "p_use": p_use, "n_train": n, "gates": gates, "task_latent": task_lat, "dist_latent": dist_lat,
            "knot_grid": knot_grid.astype(np.float16), "col_knots": col_knots, "col_means": col_means,
            "col_stds": col_stds, "is_binary_col": is_binary_col, "binary_probs": binary_probs,
            "binary_low_high": binary_low_high, "discrete_support": discrete_support,
            "chol_factor": chol_factor, "diag_res_std": diag_res_std,
            "regime_weights": regime_weights.astype(np.float16), "regime_means": regime_means.astype(np.float16),
            "regime_scales": regime_scales.astype(np.float16), "target_bundle": target_bundle,
        }

    @torch.no_grad()
    def generate_synthetic(
        self, encoding: Dict[str, Any], n_rows: Optional[int] = None, *, seed: int = 20261023
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, p_use = int(encoding["n_train"] if n_rows is None else n_rows), int(encoding["p_use"])
        level, task = encoding["level"], encoding["task"]
        rng = np.random.default_rng(seed)

        reg_w = np.asarray(encoding["regime_weights"], dtype=np.float64)
        reg_w = reg_w / np.sum(reg_w)
        reg_m, reg_s = np.asarray(encoding["regime_means"], dtype=np.float64), np.asarray(encoding["regime_scales"], dtype=np.float64)
        chol, diag_std = np.asarray(encoding["chol_factor"], dtype=np.float64), np.asarray(encoding["diag_res_std"], dtype=np.float64)

        reg_idx = rng.choice(len(reg_w), size=n, p=reg_w)
        Z_syn = reg_m[reg_idx] + reg_s[reg_idx] * (rng.standard_normal((n, chol.shape[1])) @ chol.T + rng.standard_normal((n, p_use)) * diag_std[None, :])

        device = next(self.parameters()).device
        Z_pad = torch.zeros((n, self.max_cols), dtype=torch.float32, device=device)
        Z_pad[:, :p_use] = torch.tensor(Z_syn, dtype=torch.float32, device=device)
        col_mask = torch.zeros(self.max_cols, dtype=torch.float32, device=device)
        col_mask[:p_use] = 1.0
        dist_lat = torch.tensor(np.asarray(encoding["dist_latent"], dtype=np.float32), device=device)
        Z_ref = self.cov_net(Z_pad, dist_lat, col_mask)[:, :p_use].detach().cpu().numpy().astype(np.float64)
        Z_syn = 0.94 * Z_syn + 0.06 * Z_ref
        Z_syn = (Z_syn - np.mean(Z_syn, axis=0)) / np.maximum(np.std(Z_syn, axis=0), 1e-6)

        U_syn = np.clip(0.5 * (1.0 + _erf_approx(Z_syn / math.sqrt(2.0))), 0.001, 0.999)
        knot_grid, knots = np.asarray(encoding.get("knot_grid", QUANTILE_KNOTS), dtype=np.float64), np.asarray(encoding["col_knots"], dtype=np.float64)
        is_bin, bin_p, bin_lh = np.asarray(encoding["is_binary_col"], dtype=bool), np.asarray(encoding["binary_probs"], dtype=np.float64), np.asarray(encoding["binary_low_high"], dtype=np.float64)
        disc_sup = encoding.get("discrete_support", {})

        X_syn = np.zeros((n, p_use), dtype=np.float64)
        for j in range(p_use):
            if is_bin[j]:
                X_syn[:, j] = np.where(U_syn[:, j] >= (1.0 - bin_p[j]), bin_lh[j, 1], bin_lh[j, 0])
            elif j in disc_sup:
                u_vals, cum_p = disc_sup[j]
                X_syn[:, j] = np.asarray(u_vals, dtype=np.float64)[np.clip(np.searchsorted(np.asarray(cum_p, dtype=np.float64), U_syn[:, j]), 0, len(u_vals) - 1)]
            else:
                X_syn[:, j] = np.interp(U_syn[:, j], knot_grid, knots[j])

        tb = encoding["target_bundle"]
        if task == "multiclass" or tb.get("task_kind") == "multiclass":
            y_syn = predict_multiclass_from_bundle(X_syn, tb, level=level, rng=rng)
        else:
            y_syn = _predict_from_target_bundle(
                X_syn, tb, task=task, level=level, rng=rng,
                target_net=self.target_net, task_lat=torch.tensor(np.asarray(encoding["task_latent"], dtype=np.float32), device=device), device=device,
            )
        return X_syn, y_syn


def _build_differentiable_basis(X: torch.Tensor, gates: Dict[str, torch.Tensor], *, level: str) -> torch.Tensor:
    n, p = X.shape
    mw, aw = gates["monotone_w"], gates.get("asinh_atan_w", 0.0 * gates["monotone_w"])
    parts = [
        torch.ones((n, 1), device=X.device, dtype=X.dtype), X,
        mw * torch.sign(X) * torch.log1p(torch.abs(X)), mw * torch.tanh(X * 0.75),
        0.5 * mw * (torch.abs(X) - 0.8), aw * torch.asinh(X), aw * torch.atan(X),
    ]
    if level in ("high", "ultra"):
        hw, pw = gates["hinge_w"], gates["pair_w"]
        parts.extend([hw * F.relu(X - 0.5), hw * F.relu(-X - 0.5), 0.35 * hw * torch.sin(X * 1.5)])
        pair_cols = [pw * (X[:, i : i + 1] * X[:, j : j + 1]) for i in range(min(p, 8)) for j in range(i + 1, min(p, 8))]
        if pair_cols:
            parts.append(torch.cat(pair_cols, dim=1))
    if level == "ultra":
        tw = gates["tree_w"]
        parts.extend([tw * torch.tanh(3.0 * (X - 1.0)), tw * torch.tanh(3.0 * (X + 1.0)), 0.5 * tw * torch.exp(-0.5 * (X ** 2))])
    return torch.cat(parts, dim=1)


def _fit_hierarchical_target_bundle(
    X: np.ndarray, y: np.ndarray, *, gates: Dict[str, float], col_prior: np.ndarray, emp_drivers: np.ndarray,
    task: str, level: str, target_net: ConditionalHyperTargetNet, task_lat_tensor: torch.Tensor,
    plan: Optional[Dict[str, int]] = None, include_v3_primitives: bool = False,
    col_landmarks: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    n, p = X.shape
    device = task_lat_tensor.device
    X_pad, col_mask = torch.zeros((n, target_net.max_cols), dtype=torch.float32, device=device), torch.zeros(target_net.max_cols, dtype=torch.float32, device=device)
    X_pad[:, :p], col_mask[:p] = torch.tensor(X, dtype=torch.float32, device=device), 1.0
    c_gate_np = target_net.compute_column_gates(task_lat_tensor, col_landmarks, X_pad=X_pad)[:p].detach().cpu().numpy().astype(np.float64)
    combined_prior = (0.5 * col_prior + 0.5 * emp_drivers) * np.abs(c_gate_np)
    top_cols = np.argsort(combined_prior)[::-1][: min(p, 10)]
    max_pairs = plan["max_pairs"] if plan else (0 if level in ("low", "small", "medium") else 16 if level == "high" else 28)
    pairs = [(int(top_cols[a]), int(top_cols[b])) for a in range(len(top_cols)) for b in range(a + 1, len(top_cols))][:max_pairs]

    Phi = build_enriched_numpy_basis(X, gates, pairs, level=level, include_v3_primitives=include_v3_primitives)
    gram = (Phi.T @ Phi) / float(max(n, 1))
    eye = np.eye(Phi.shape[1], dtype=np.float64)
    eye[0, 0] = 1e-6
    w_basis = np.linalg.solve(gram + float(gates["ridge_alpha"]) * eye, (Phi.T @ y) / float(max(n, 1)))
    pred_basis = Phi @ w_basis
    res_net = target_net(X_pad, task_lat_tensor, col_mask, col_landmarks=col_landmarks).detach().cpu().numpy().astype(np.float64)
    res_net_mean = float(np.mean(res_net))
    residual = y - (pred_basis + float(gates["neural_w"]) * (res_net - res_net_mean))

    n_trees = plan["n_trees"] if plan else (2 if level in ("low", "small") else 4 if level == "medium" else 24 if level == "high" else 56)
    depth = plan["tree_depth"] if plan else (2 if level in ("low", "small", "medium") else 3 if level == "high" else 4)
    trees = _fit_oblivious_trees(
        X, residual, combined_prior, n_trees=n_trees, depth=depth,
        learning_rate=0.16 if level in ("low", "small", "medium") else 0.19 if level == "high" else 0.22,
    )
    final_res = residual - _eval_oblivious_trees(X, trees)
    return {
        "gates": gates, "pairs": pairs, "include_v3_primitives": include_v3_primitives,
        "w_basis": w_basis.astype(np.float16), "res_net_mean": round(res_net_mean, 6),
        "neural_w": round(float(gates["neural_w"]), 5), "trees": trees, "residual_std": round(float(np.std(final_res)), 6),
        "noise_scale": round(float(gates["noise_scale"]), 5), "class_prior": round(float(np.mean(y)) if task in ("classification", "binary") else 0.5, 6),
        "y_min": round(float(np.min(y)), 5), "y_max": round(float(np.max(y)), 5),
    }


def _predict_from_target_bundle(
    X_syn: np.ndarray, bundle: Dict[str, Any], *, task: str, level: str,
    rng: np.random.Generator, target_net: ConditionalHyperTargetNet, task_lat: torch.Tensor, device: torch.device,
) -> np.ndarray:
    n, p = X_syn.shape
    gates_for_basis = bundle.get("gates") or {
        "monotone_w": 0.55 if level in ("low", "small", "medium") else 0.65, "hinge_w": 0.55 if level in ("high", "ultra") else 0.0,
        "pair_w": 0.60 if level in ("high", "ultra") else 0.0, "tree_w": 1.00 if level == "ultra" else 0.0,
    }
    Phi = build_enriched_numpy_basis(X_syn, gates_for_basis, bundle["pairs"], level=level, include_v3_primitives=bool(bundle.get("include_v3_primitives", False)))
    w_basis = np.asarray(bundle["w_basis"], dtype=np.float64)
    m = min(Phi.shape[1], w_basis.size)
    pred = Phi[:, :m] @ w_basis[:m]
    X_pad, col_mask = torch.zeros((n, target_net.max_cols), dtype=torch.float32, device=device), torch.zeros(target_net.max_cols, dtype=torch.float32, device=device)
    X_pad[:, :p], col_mask[:p] = torch.tensor(X_syn, dtype=torch.float32, device=device), 1.0
    res_net = target_net(X_pad, task_lat, col_mask).detach().cpu().numpy().astype(np.float64)
    pred = pred + float(bundle["neural_w"]) * (res_net - float(bundle["res_net_mean"])) + _eval_oblivious_trees(X_syn, bundle["trees"])
    if task in ("classification", "binary"):
        prior = float(np.clip(bundle["class_prior"], 0.01, 0.99))
        score = pred + rng.normal(0.0, float(bundle["noise_scale"]) * (0.12 if level in ("low", "small", "medium") else 0.06 if level == "high" else 0.03), size=n)
        y_out = (score >= float(np.quantile(score, 1.0 - prior))).astype(np.float64)
        if n >= 2 and len(np.unique(y_out)) < 2:
            y_out[int(np.argmax(score))], y_out[int(np.argmin(score))] = 1.0, 0.0
        return y_out
    noise_sigma = float(bundle["residual_std"]) * float(bundle["noise_scale"]) * (0.42 if level in ("low", "small", "medium") else 0.22 if level == "high" else 0.10)
    return np.clip(pred + rng.normal(0.0, noise_sigma, size=n), float(bundle["y_min"]) - 0.5, float(bundle["y_max"]) + 0.5)


def _pack_code_vector(
    *, width: int, col_knots: np.ndarray, chol_factor: np.ndarray,
    target_bundle: Dict[str, Any], task_lat: np.ndarray, dist_lat: np.ndarray,
) -> np.ndarray:
    parts = [
        col_knots.reshape(-1).astype(np.float32), chol_factor.reshape(-1).astype(np.float32),
        np.asarray(target_bundle["w_basis"], dtype=np.float32).reshape(-1),
        task_lat.reshape(-1).astype(np.float32), dist_lat.reshape(-1).astype(np.float32),
    ] + [np.asarray(t["leaf_values"], dtype=np.float32).reshape(-1) for t in target_bundle["trees"]]
    flat = np.concatenate(parts)
    out = np.zeros(width, dtype=np.float32)
    take = min(width, flat.size)
    out[:take] = np.tanh(flat[:take] / 4.0) * 4.0
    return out
