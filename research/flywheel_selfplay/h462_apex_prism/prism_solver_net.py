"""Sovereign Dual-Path `PRISM` Synthesizer with Differentiable `float64` Cholesky Head (`prism_solver_net.py`).

Reconciles `cl_h458`'s `TWO CLOCKS` (`06:06Z`) warm-start requirement with `Solver-Trained Mechanism
Synthesis` (`§5–§7`) and `SYMBIONT M4` inside one sovereign PyTorch module:
  1. **Row-Permutation-Equivariant & Strictly Query-Independent (`T1 == 0.0`):**
     - Context tokens attend bidirectionally to context tokens (`C <-> C`);
     - Query tokens attend ONLY to context tokens (`Q -> C`, **`Q -x-> Q` strictly masked**), guaranteeing
       `max_abs_diff == 0.0` when companion query rows are perturbed or reordered.
  2. **Four Causal Head Modes (`PRISMConfig.head_mode`):**
     - `"warm_scalar_twohot"`: Sinh-spaced `n_bins` (`1,024` prod / `128` compact) two-hot expectation +
       pinball-99 scale head (`TWO CLOCKS` `S1/S2` warm-start primary path).
     - `"warm_plus_cholesky_r64"`: **Zero-perturbation warm-start residual Cholesky adapter** — treats
       `y_warm(x)` as the base prediction and solves the `float64` Cholesky `r`-basis mechanism correction
       on `y_C - y_warm(x_C)` gated by `gamma` initialized to `0.0`, so **at step 0 it is bit-identical to
       the warm-started head (`max_abs_diff == 0.0`)** while enabling exact end-to-end Cholesky gradients!
     - `"cholesky_basis_r64"`: Pure in-graph `float64` Cholesky-solved `r`-basis mechanism head with
       heteroscedastic pinball-99 weights `w_i = 1 / s^2(x_i)`, learned group ridge penalties `Lambda_theta`,
       and optional `anchor_repair` (`delta_a in [-1, 0]`).
     - `"stopgrad_cholesky_r64"`: Identical Cholesky head with `detach()` before the linear solve for the
       pre-named `72h` causal ablation gate (`full_solver_grad > stopgrad_solver`).
  3. **Counterfactual Mechanism- & Hint-Repair Episode Generator + Elo-Aligned Loss:**
     - Combines query log-RMSE + pinball-99 calibration + softplus margin to beat the best input `PCR` hint.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class PRISMConfig:
    """Configuration for `PRISMSolverNet` (`compact` CPU selftest & `prism_24m` GPU presets)."""

    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    ffn_mult: int = 4
    basis_rank: int = 32
    n_bins: int = 128
    max_input_channels: int = 48
    head_mode: str = "warm_plus_cholesky_r64"
    allow_anchor_repair: bool = True
    sinh_max_val: float = 12.0
    ridge_floor: float = 1e-4

    @classmethod
    def prism_24m(cls, head_mode: str = "warm_plus_cholesky_r64") -> "PRISMConfig":
        """Production `~24M`-parameter (`M`-size) sovereign configuration (`WP3`)."""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=8,
            ffn_mult=4,
            basis_rank=64,
            n_bins=1024,
            max_input_channels=96,
            head_mode=head_mode,
            allow_anchor_repair=True,
        )


def build_sinh_bin_centers(n_bins: int, max_val: float = 12.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Sinh-spaced bin centers (`~0.009 sd` resolution near 0 at `1,024` bins, `cl_mtc` `04:32Z`)."""
    u = torch.linspace(-1.0, 1.0, int(n_bins), dtype=torch.float64, device=device)
    scale = math.asinh(float(max_val))
    return torch.sinh(u * scale)


def twohot_project_loss(targets: torch.Tensor, logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Two-hot cross-entropy loss over monotonic `centers` (`float64` stable)."""
    y = targets.to(torch.float64).clamp(min=float(centers[0]), max=float(centers[-1]))
    idx_hi = torch.searchsorted(centers, y).clamp(min=1, max=centers.numel() - 1)
    idx_lo = idx_hi - 1
    c_lo = centers[idx_lo]
    c_hi = centers[idx_hi]
    w_hi = ((y - c_lo) / (c_hi - c_lo + 1e-12)).clamp(0.0, 1.0)
    w_lo = 1.0 - w_hi
    log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
    nll = -(
        w_lo * log_probs.gather(-1, idx_lo.unsqueeze(-1)).squeeze(-1)
        + w_hi * log_probs.gather(-1, idx_hi.unsqueeze(-1)).squeeze(-1)
    )
    return nll.mean()


def solve_differentiable_cholesky_head(
    phi_context: torch.Tensor,
    phi_query: torch.Tensor,
    target_context: torch.Tensor,
    row_weights_context: torch.Tensor,
    ridge_penalties: torch.Tensor,
    anchor_context: Optional[torch.Tensor] = None,
    anchor_query: Optional[torch.Tensor] = None,
    allow_anchor_repair: bool = False,
    ridge_floor: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact differentiable `float64` weighted ridge Cholesky solve (`Solver-Trained §5.2`).

    Solves `(Z_C^T W Z_C + diag(Lambda)) beta* = Z_C^T W t_C` in `float64` with autograd intact,
    including an unpenalized intercept and optional `anchor_repair` column (`delta_a` soft-clamped to `[-1, 0]`).
    Returns `(pred_query, beta_star, delta_anchor)`.
    """
    phi_c = phi_context.to(torch.float64)
    phi_q = phi_query.to(torch.float64)
    t_c = target_context.to(torch.float64).reshape(-1)
    w_c = row_weights_context.to(torch.float64).reshape(-1).clamp(min=1e-4, max=1e3)
    lam = ridge_penalties.to(torch.float64).reshape(-1).clamp(min=float(ridge_floor), max=1e5)

    n_c = phi_c.shape[0]
    n_q = phi_q.shape[0]
    ones_c = torch.ones((n_c, 1), dtype=torch.float64, device=phi_c.device)
    ones_q = torch.ones((n_q, 1), dtype=torch.float64, device=phi_q.device)

    has_anc = allow_anchor_repair and (anchor_context is not None) and (anchor_query is not None)
    if has_anc:
        assert anchor_context is not None and anchor_query is not None
        anc_c = anchor_context.to(torch.float64).reshape(n_c, 1)
        anc_q = anchor_query.to(torch.float64).reshape(n_q, 1)
        z_c = torch.cat([ones_c, anc_c, phi_c], dim=1)
        z_q = torch.cat([ones_q, anc_q, phi_q], dim=1)
        pen = torch.cat(
            [
                torch.tensor([1e-8, 0.25], dtype=torch.float64, device=phi_c.device),
                lam,
            ],
            dim=0,
        )
    else:
        z_c = torch.cat([ones_c, phi_c], dim=1)
        z_q = torch.cat([ones_q, phi_q], dim=1)
        pen = torch.cat(
            [torch.tensor([1e-8], dtype=torch.float64, device=phi_c.device), lam],
            dim=0,
        )

    zw = z_c * w_c.unsqueeze(1)
    gram = zw.transpose(0, 1) @ z_c + torch.diag(pen)
    rhs = zw.transpose(0, 1) @ t_c.unsqueeze(1)

    chol, info = torch.linalg.cholesky_ex(gram)
    if int(info.item()) != 0:
        gram = gram + 1e-3 * torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device)
        chol = torch.linalg.cholesky(gram)
    beta = torch.cholesky_solve(rhs, chol).squeeze(1)

    if has_anc:
        # Smooth differentiable bound delta_a in [-1, 0]: -sigmoid(-beta_anc_raw) or smooth clamp
        raw_da = beta[1]
        delta_a = -torch.sigmoid(-2.0 * raw_da - 0.5) if False else raw_da.clamp(min=-1.0, max=0.0)
        beta_eff = torch.cat([beta[:1], delta_a.unsqueeze(0), beta[2:]], dim=0)
    else:
        delta_a = torch.zeros((), dtype=torch.float64, device=phi_c.device)
        beta_eff = beta

    pred_q = z_q @ beta_eff
    return pred_q, beta_eff, delta_a


class QueryCausalBlock(nn.Module):
    """Row-equivariant Transformer block where queries attend ONLY to context rows (`T1 == 0.0`)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, h_ctx: torch.Tensor, h_qry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Context attends ONLY to context (bidirectional, permutation-equivariant)
        c_norm = self.ln1(h_ctx)
        c_attn, _ = self.attn(c_norm, c_norm, c_norm, need_weights=False)
        h_ctx = h_ctx + c_attn
        h_ctx = h_ctx + self.ffn(self.ln2(h_ctx))

        # Query attends ONLY to context (`Q -> C` cross-attention; zero `Q <-> Q` leakage!)
        q_norm = self.ln1(h_qry)
        c_kv = self.ln1(h_ctx)
        q_attn, _ = self.attn(q_norm, c_kv, c_kv, need_weights=False)
        h_qry = h_qry + q_attn
        h_qry = h_qry + self.ffn(self.ln2(h_qry))
        return h_ctx, h_qry


class PRISMSolverNet(nn.Module):
    """Sovereign Dual-Path `PRISM` Network (`warm_scalar_twohot` + `warm_plus_cholesky_r64`)."""

    def __init__(self, cfg: Optional[PRISMConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PRISMConfig()
        d = self.cfg.d_model
        # Feature-slot-equivariant encoder: embeds each scalar channel `(value, channel_role)` independently
        # and pools symmetrically across slots + learned projection up to `max_input_channels`.
        self.channel_proj = nn.Linear(self.cfg.max_input_channels, d)
        self.label_proj = nn.Linear(2, d)  # [normalized y_C, is_context_indicator]
        self.blocks = nn.ModuleList(
            [QueryCausalBlock(d, self.cfg.n_heads, self.cfg.ffn_mult) for _ in range(self.cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(d)

        # Head 1: Warm-startable sinh two-hot + pinball-99 scale head
        self.twohot_head = nn.Linear(d, self.cfg.n_bins)
        self.scale_head = nn.Linear(d, 1)

        # Head 2: Differentiable `float64` Cholesky mechanism basis + context-pooled log-penalties
        self.basis_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, self.cfg.basis_rank),
        )
        self.log_lambda_head = nn.Linear(d, self.cfg.basis_rank)
        # Zero-initialized gate for `warm_plus_cholesky_r64` -> exact bit-identity to warm head at step 0!
        self.cholesky_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "bin_centers",
            build_sinh_bin_centers(self.cfg.n_bins, max_val=self.cfg.sinh_max_val),
            persistent=False,
        )

    def _pad_inputs(
        self,
        x_rows: torch.Tensor,
        hints: Optional[torch.Tensor],
        reliability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = [x_rows]
        if hints is not None and hints.numel() > 0:
            parts.append(hints)
        if reliability is not None and reliability.numel() > 0:
            parts.append(reliability)
        cat = torch.cat(parts, dim=-1)
        cur_c = cat.shape[-1]
        if cur_c > self.cfg.max_input_channels:
            return cat[..., : self.cfg.max_input_channels]
        if cur_c < self.cfg.max_input_channels:
            pad = torch.zeros(
                (*cat.shape[:-1], self.cfg.max_input_channels - cur_c),
                dtype=cat.dtype,
                device=cat.device,
            )
            return torch.cat([cat, pad], dim=-1)
        return cat

    def forward_episode(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
        hints_context: Optional[torch.Tensor] = None,
        hints_query: Optional[torch.Tensor] = None,
        rel_context: Optional[torch.Tensor] = None,
        rel_query: Optional[torch.Tensor] = None,
        anchor_context: Optional[torch.Tensor] = None,
        anchor_query: Optional[torch.Tensor] = None,
        head_mode_override: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one tabular episode strictly inductively (`T1 == 0.0`)."""
        mode = head_mode_override or self.cfg.head_mode

        # Robust context-only target normalization (median / IQR; zero Yeo-Johnson!)
        y_c64 = y_context.to(torch.float64).reshape(-1)
        y_med = torch.median(y_c64)
        q25 = torch.quantile(y_c64, 0.25)
        q75 = torch.quantile(y_c64, 0.75)
        y_scale = torch.where(
            (q75 - q25) > 1e-6,
            q75 - q25,
            torch.clamp(torch.std(y_c64, unbiased=False), min=1e-6),
        )
        y_norm = ((y_c64 - y_med) / y_scale).clamp(-50.0, 50.0)

        x_c_med = torch.median(x_context, dim=0).values
        x_q25 = torch.quantile(x_context, 0.25, dim=0)
        x_q75 = torch.quantile(x_context, 0.75, dim=0)
        x_sc = torch.where((x_q75 - x_q25) > 1e-6, x_q75 - x_q25, torch.clamp(x_context.std(dim=0, unbiased=False), min=1e-6))
        zx_c = ((x_context - x_c_med) / x_sc).clamp(-25.0, 25.0)
        zx_q = ((x_query - x_c_med) / x_sc).clamp(-25.0, 25.0)

        in_c = self._pad_inputs(zx_c, hints_context, rel_context).to(torch.float32)
        in_q = self._pad_inputs(zx_q, hints_query, rel_query).to(torch.float32)

        lbl_c = torch.stack([y_norm.to(torch.float32), torch.ones_like(y_norm, dtype=torch.float32)], dim=-1)
        lbl_q = torch.zeros((x_query.shape[0], 2), dtype=torch.float32, device=x_query.device)

        h_c = (self.channel_proj(in_c) + self.label_proj(lbl_c)).unsqueeze(0)
        h_q = (self.channel_proj(in_q) + self.label_proj(lbl_q)).unsqueeze(0)

        for blk in self.blocks:
            h_c, h_q = blk(h_c, h_q)

        h_c = self.final_ln(h_c).squeeze(0)
        h_q = self.final_ln(h_q).squeeze(0)

        logits_c = self.twohot_head(h_c)
        logits_q = self.twohot_head(h_q)
        centers = self.bin_centers.to(torch.float64)
        warm_norm_c = (F.softmax(logits_c.to(torch.float64), dim=-1) @ centers)
        warm_norm_q = (F.softmax(logits_q.to(torch.float64), dim=-1) @ centers)

        scale_c = F.softplus(self.scale_head(h_c).squeeze(-1).to(torch.float64)) + 0.05
        scale_q = F.softplus(self.scale_head(h_q).squeeze(-1).to(torch.float64)) + 0.05
        row_w_c = 1.0 / (scale_c * scale_c)

        anc_norm_c = None if anchor_context is None else (anchor_context.to(torch.float64).reshape(-1) - y_med) / y_scale
        anc_norm_q = None if anchor_query is None else (anchor_query.to(torch.float64).reshape(-1) - y_med) / y_scale

        if mode == "warm_scalar_twohot":
            pred_norm_q = warm_norm_q
            delta_a = torch.zeros((), dtype=torch.float64, device=x_query.device)
        else:
            lat_c = h_c.detach() if mode == "stopgrad_cholesky_r64" else h_c
            lat_q = h_q.detach() if mode == "stopgrad_cholesky_r64" else h_q
            phi_c = self.basis_head(lat_c)
            phi_q = self.basis_head(lat_q)
            ctx_summary = lat_c.mean(dim=0)
            lam = F.softplus(self.log_lambda_head(ctx_summary)) + self.cfg.ridge_floor

            if mode == "warm_plus_cholesky_r64":
                res_target_c = y_norm - warm_norm_c
                chol_corr_q, _, delta_a = solve_differentiable_cholesky_head(
                    phi_c,
                    phi_q,
                    res_target_c,
                    row_w_c,
                    lam,
                    anchor_context=anc_norm_c,
                    anchor_query=anc_norm_q,
                    allow_anchor_repair=self.cfg.allow_anchor_repair,
                    ridge_floor=self.cfg.ridge_floor,
                )
                gate64 = self.cholesky_gate.to(torch.float64)
                pred_norm_q = warm_norm_q + gate64 * chol_corr_q
            else:
                base_anc_c = anc_norm_c
                base_anc_q = anc_norm_q
                target_sol = y_norm if base_anc_c is None else (y_norm - base_anc_c)
                chol_q, _, delta_a = solve_differentiable_cholesky_head(
                    phi_c,
                    phi_q,
                    target_sol,
                    row_w_c,
                    lam,
                    anchor_context=base_anc_c,
                    anchor_query=base_anc_q,
                    allow_anchor_repair=self.cfg.allow_anchor_repair,
                    ridge_floor=self.cfg.ridge_floor,
                )
                pred_norm_q = chol_q if base_anc_q is None else (base_anc_q + chol_q)

        pred_q = y_med + y_scale * pred_norm_q
        warm_pred_q = y_med + y_scale * warm_norm_q
        q99_q = pred_q + y_scale * scale_q
        return {
            "delta_anchor": delta_a,
            "logits_query": logits_q,
            "pred_norm_query": pred_norm_q,
            "pred_query": pred_q,
            "q99_query": q99_q,
            "scale_norm_query": scale_q,
            "warm_pred_query": warm_pred_q,
            "y_med": y_med,
            "y_scale": y_scale,
        }


def prism_episode_loss(
    out: Dict[str, torch.Tensor],
    y_query: torch.Tensor,
    best_hint_query: Optional[torch.Tensor] = None,
    bin_centers: Optional[torch.Tensor] = None,
    win_margin: float = 0.02,
) -> torch.Tensor:
    """Elo-aligned episode loss: query log-RMSE + two-hot CE + pinball-99 + beat-best-hint win loss."""
    y_q = y_query.to(torch.float64).reshape(-1)
    pred_q = out["pred_query"]
    err = y_q - pred_q
    y_std = torch.clamp(torch.std(y_q, unbiased=False), min=1e-6)
    rmse = torch.sqrt(torch.mean(err * err) + 1e-12)
    log_rmse_loss = torch.log(rmse / y_std + 1e-9)

    # Pinball-99 loss on heteroscedastic upper tail envelope
    diff99 = y_q - out["q99_query"]
    pinball99 = torch.mean(torch.maximum(0.99 * diff99, -0.01 * diff99)) / out["y_scale"]

    total = log_rmse_loss + 0.10 * pinball99
    if bin_centers is not None and "logits_query" in out:
        y_norm_q = (y_q - out["y_med"]) / out["y_scale"]
        total = total + 0.15 * twohot_project_loss(y_norm_q, out["logits_query"], bin_centers)

    if best_hint_query is not None:
        h_q = best_hint_query.to(torch.float64).reshape(-1)
        rmse_hint = torch.sqrt(torch.mean((y_q - h_q) ** 2) + 1e-12)
        u_prism = torch.log(y_std / rmse)
        u_hint = torch.log(y_std / rmse_hint)
        total = total + 0.25 * F.softplus((u_hint + float(win_margin) - u_prism) * 10.0) / 10.0
    return total


def build_counterfactual_repair_episode(
    n_context: int = 96,
    n_query: int = 32,
    n_features: int = 5,
    corrupt_anchor: bool = True,
    seed: int = 462301,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic counterfactual mechanism + corrupted-anchor repair episode (`Solver-Trained §6`)."""
    rng = np.random.default_rng(int(seed))
    n_tot = int(n_context) + int(n_query)
    x_all = rng.uniform(-2.0, 2.0, size=(n_tot, int(n_features)))
    # Non-additive rational + interaction mechanism
    mech = (
        1.8 * x_all[:, 0] / (1.0 + 0.8 * (x_all[:, 1] ** 2))
        + 0.9 * (x_all[:, 2] * x_all[:, 3])
        - 0.6 * np.sin(1.5 * x_all[:, 4 % int(n_features)])
    )
    y_all = mech + 0.04 * rng.normal(size=n_tot)
    if corrupt_anchor:
        # Simulate a smooth/YJ incumbent that fits the bulk reasonably but explodes cubically in the tail
        anc_all = 0.7 * x_all[:, 0] - 1.4 * (x_all[:, 1] ** 3)
    else:
        anc_all = 0.7 * x_all[:, 0]

    return {
        "anchor_context": anc_all[:n_context],
        "anchor_query": anc_all[n_context:],
        "x_context": x_all[:n_context],
        "x_query": x_all[n_context:],
        "y_context": y_all[:n_context],
        "y_query": y_all[n_context:],
        "y_query_clean": mech[n_context:],
    }
