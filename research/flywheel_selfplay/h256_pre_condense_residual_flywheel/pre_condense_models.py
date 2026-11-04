#!/usr/bin/env python3
"""H256 Pre-Condense Residual Bypass Models (`PreCondenseResidualBridge`).

Implements the zero-initialized residual bypass before Stage-1B condensation:
  s_new(p, D) = s_existing(p, T_10x32(D)) + Delta_s_theta(p, D_629, H_256, C_64x17, H_rev)

Supports the 6 Staged Ablation Arms:
  - Arm A (`A_FROZEN_10TOK_BASELINE`): Existing detached 10x32 Stage-1B tokens + Causal 2+2 baseline.
  - Arm B (`B_CAPACITY_CTRL_10TOK`): Extra downstream residual capacity on detached 10x32 tokens only.
  - Arm C (`C_LEARNED_COMPACT_BRIDGE_629`): Learned 629 -> 10x32 compact bridge replacing fixed SVD.
  - Arm D (`D_PRE_CONDENSE_FULL629_20TOK`): Coordinate-preserving Full629 (20x32 semantic block tokens + 11D log-norm scale side-channel).
  - Arm E (`E_FULL629_PLUS_HIDDEN256_28TOK`): Full629 (20x32) + pre-head DOPE router Hidden256 (8x32).
  - Arm F (`F_FULL629_HIDDEN256_COL64X17`): Full629 (20x32) + Hidden256 (8x32) + Masked Per-Column (64x18 -> 64x32) tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARM_NAMES: Tuple[str, ...] = (
    "A_FROZEN_10TOK_BASELINE",
    "B_CAPACITY_CTRL_10TOK",
    "C_LEARNED_COMPACT_BRIDGE_629",
    "D_PRE_CONDENSE_FULL629_20TOK",
    "E_FULL629_PLUS_HIDDEN256_28TOK",
    "F_FULL629_HIDDEN256_COL64X17",
    "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED",
)

# Semantic slices inside `dope629`
BLOCK_SLICES: Dict[str, Tuple[int, int]] = {
    "desc256": (0, 256),
    "task128": (256, 384),
    "dist128": (384, 512),
    "gates14": (512, 526),
    "lm23": (526, 549),
    "bb16": (549, 565),
    "col_gates64": (565, 629),
}


@dataclass
class FoldScaler:
    """Fold-safe standardizer fit strictly on training fold contexts (`fit_cids`)."""

    dope629_mean: np.ndarray
    dope629_std: np.ndarray
    hidden256_mean: np.ndarray
    hidden256_std: np.ndarray
    col18_mean: np.ndarray
    col18_std: np.ndarray

    @classmethod
    def fit(
        cls,
        fit_cids: List[str],
        raw_ctx: Dict[str, Dict[str, np.ndarray]],
    ) -> "FoldScaler":
        dope_mat = np.stack([raw_ctx[c]["dope629"] for c in fit_cids], axis=0).astype(np.float64)
        hid_mat = np.stack([raw_ctx[c]["hidden256"] for c in fit_cids], axis=0).astype(np.float64)

        col_list = []
        for c in fit_cids:
            cm = raw_ctx[c]["col_mask_64"]
            cl = raw_ctx[c]["col_lm_64x17"]
            cg = raw_ctx[c]["dope629"][565:629, None]
            c18 = np.concatenate([cl, cg], axis=-1)
            if np.any(cm):
                col_list.append(c18[cm])
        col_cat = np.concatenate(col_list, axis=0).astype(np.float64)

        return cls(
            dope629_mean=dope_mat.mean(axis=0).astype(np.float32),
            dope629_std=np.maximum(dope_mat.std(axis=0), 1e-4).astype(np.float32),
            hidden256_mean=hid_mat.mean(axis=0).astype(np.float32),
            hidden256_std=np.maximum(hid_mat.std(axis=0), 1e-4).astype(np.float32),
            col18_mean=col_cat.mean(axis=0).astype(np.float32),
            col18_std=np.maximum(col_cat.std(axis=0), 1e-4).astype(np.float32),
        )

    def blend_with_s3_scaler(self, s3_stats: Dict[str, np.ndarray], alpha_s3: float = 0.25) -> "FoldScaler":
        w_f = float(1.0 - alpha_s3)
        w_s = float(alpha_s3)
        return FoldScaler(
            dope629_mean=(w_f * self.dope629_mean + w_s * s3_stats["dope629_mean"]).astype(np.float32),
            dope629_std=np.maximum(w_f * self.dope629_std + w_s * s3_stats["dope629_std"], 1e-4).astype(np.float32),
            hidden256_mean=(w_f * self.hidden256_mean + w_s * s3_stats["hidden256_mean"]).astype(np.float32),
            hidden256_std=np.maximum(w_f * self.hidden256_std + w_s * s3_stats["hidden256_std"], 1e-4).astype(np.float32),
            col18_mean=(w_f * self.col18_mean + w_s * s3_stats["col18_mean"]).astype(np.float32),
            col18_std=np.maximum(w_f * self.col18_std + w_s * s3_stats["col18_std"], 1e-4).astype(np.float32),
        )

    def transform_context(
        self,
        entry: Dict[str, np.ndarray],
        device: torch.device,
        ablate_block: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        dope_raw = entry["dope629"].astype(np.float32)
        dope_z = np.clip((dope_raw - self.dope629_mean) / self.dope629_std, -15.0, 15.0)
        hid_z = np.clip(
            (entry["hidden256"].astype(np.float32) - self.hidden256_mean) / self.hidden256_std,
            -15.0,
            15.0,
        )
        col_lm = entry["col_lm_64x17"].astype(np.float32)
        col_mask = entry["col_mask_64"].astype(bool)
        col_gates = dope_raw[565:629, None]
        col18_raw = np.concatenate([col_lm, col_gates], axis=-1)
        col18_z = np.clip((col18_raw - self.col18_mean[None, :]) / self.col18_std[None, :], -15.0, 15.0)
        col18_z = col18_z * col_mask[:, None].astype(np.float32)

        if ablate_block is not None:
            if ablate_block in BLOCK_SLICES:
                s, e = BLOCK_SLICES[ablate_block]
                dope_z = dope_z.copy()
                dope_z[s:e] = 0.0
            elif ablate_block == "hidden256":
                hid_z = np.zeros_like(hid_z)
            elif ablate_block == "col_lm_64x17":
                col18_z = np.zeros_like(col18_z)

        # Compute 11D scale-preserving log-norms across the 7 semantic blocks of dope_z
        def _rms(x: np.ndarray) -> float:
            return float(np.log1p(np.sqrt(np.mean(x * x) + 1e-8)))

        def _max(x: np.ndarray) -> float:
            return float(np.log1p(np.max(np.abs(x))))

        norms_11 = np.array(
            [
                _rms(dope_z[0:256]),
                _max(dope_z[0:256]),
                _rms(dope_z[256:384]),
                _max(dope_z[256:384]),
                _rms(dope_z[384:512]),
                _max(dope_z[384:512]),
                _rms(dope_z[512:526]),
                _rms(dope_z[526:549]),
                _rms(dope_z[549:565]),
                _rms(dope_z[565:629]),
                _max(dope_z[565:629]),
            ],
            dtype=np.float32,
        )
        if ablate_block == "block_log_norms":
            norms_11 = np.zeros_like(norms_11)

        return {
            "tokens_10x32": torch.as_tensor(entry["tokens_10x32"], dtype=torch.float32, device=device).unsqueeze(0),
            "mask_10": torch.as_tensor(entry["mask_10"].astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0),
            "dope629_z": torch.as_tensor(dope_z, dtype=torch.float32, device=device).unsqueeze(0),
            "block_log_norms": torch.as_tensor(norms_11, dtype=torch.float32, device=device).unsqueeze(0),
            "hidden256_z": torch.as_tensor(hid_z, dtype=torch.float32, device=device).unsqueeze(0),
            "col18_z": torch.as_tensor(col18_z, dtype=torch.float32, device=device).unsqueeze(0),
            "col_mask_64": torch.as_tensor(col_mask.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0),
        }


class SyncFreeCrossAttention(nn.Module):
    """4-head scaled dot-product cross-attention with explicit key validity mask (`1.0 = valid, 0.0 = pad`)."""

    def __init__(self, d_model: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Sq, D = query.shape
        Sk = key_value.shape[1]
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(query).view(B, Sq, H, Dh).transpose(1, 2)
        K = self.k_proj(key_value).view(B, Sk, H, Dh).transpose(1, 2)
        V = self.v_proj(key_value).view(B, Sk, H, Dh).transpose(1, 2)

        attn_mask = None
        if valid_mask is not None:
            attn_mask = torch.where(valid_mask.view(B, 1, 1, Sk) > 0.5, 0.0, -1e4).to(Q.dtype)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, Sq, D)
        return self.out_proj(out)


class PreCondenseResidualBridge(nn.Module):
    """Zero-initialized Pre-Condense Residual Bypass supporting Arms B, C, D, E, and F."""

    def __init__(
        self,
        arm: str,
        d_token: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if arm not in ARM_NAMES:
            raise ValueError(f"Unknown arm: {arm}")
        self.arm = arm
        self.d_token = d_token

        # 1. Arm B: Capacity control operating only on existing 10x32 tokens
        self.tok10_proj = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.SiLU(),
            nn.Linear(d_token, d_token),
        )
        self.tok10_norm = nn.LayerNorm(d_token)
        self.tok10_type_emb = nn.Parameter(torch.zeros(10, d_token))
        self.tok10_gate_logits = nn.Parameter(torch.zeros(10))

        # 2. Arm C: Learned compact bridge 629 -> 10x32
        self.compact_bridge_629 = nn.Sequential(
            nn.Linear(629, 160),
            nn.SiLU(),
            nn.LayerNorm(160),
            nn.Linear(160, 10 * d_token),
        )
        self.compact_norm = nn.LayerNorm(d_token)
        self.compact_type_emb = nn.Parameter(torch.zeros(10, d_token))
        self.compact_gate_logits = nn.Parameter(torch.zeros(10))

        # 3. Arms D, E, F: Coordinate-Preserving Semantic Block Tokenization of Full629 (20 x 32)
        self.desc_proj = nn.Linear(d_token, d_token)
        self.task_proj = nn.Linear(d_token, d_token)
        self.dist_proj = nn.Linear(d_token, d_token)
        self.heads_proj = nn.Linear(d_token, d_token)
        self.colgate_proj = nn.Linear(d_token, d_token)
        for layer in (
            self.desc_proj,
            self.task_proj,
            self.dist_proj,
            self.heads_proj,
            self.colgate_proj,
        ):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.full629_norm = nn.LayerNorm(d_token)
        self.full629_type_emb = nn.Parameter(torch.zeros(20, d_token))
        self.full629_gate_logits = nn.Parameter(torch.zeros(20))

        # 4. Arms E, F: Pre-head DOPE router `hidden256` (8 x 32 tokens)
        self.hidden_proj = nn.Linear(d_token, d_token)
        nn.init.eye_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)
        self.hidden_norm = nn.LayerNorm(d_token)
        self.hidden_type_emb = nn.Parameter(torch.zeros(8, d_token))
        self.hidden_gate_logits = nn.Parameter(torch.zeros(8))

        # 5. Arm F: Masked per-column encoder (`64 x 18 -> 64 x 32`)
        self.col_encoder = nn.Sequential(
            nn.Linear(18, d_token),
            nn.SiLU(),
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
        )
        self.col_type_emb = nn.Parameter(torch.zeros(1, d_token))
        self.col_gate_logit = nn.Parameter(torch.tensor(0.0))

        # 6. Candidate-conditioned Cross-Attention & Bilinear Context-Program Interaction
        self.ctx_cross_attn = SyncFreeCrossAttention(d_model=d_token, n_heads=n_heads)
        self.ctx_bilinear_u = nn.Linear(d_token * 2, 24, bias=False)
        self.prog_bilinear_v = nn.Linear(24, 24, bias=False)
        self.scale_side_proj = nn.Linear(11, 8, bias=True)

        # Stage-1 Pre-Reveal Prior Residual Readout (zero-initialized last layer!)
        self.prior_residual_mlp = nn.Sequential(
            nn.Linear(d_token + 24 + d_token + 8 + 2, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
        )
        self.prior_residual_out = nn.Linear(32, 1, bias=True)
        nn.init.zeros_(self.prior_residual_out.weight)
        nn.init.zeros_(self.prior_residual_out.bias)

        # Stage-2 Causal 2+2 Trajectory Conditioning with Pre-Condense Context
        self.hist_proj = nn.Sequential(
            nn.Linear(24 + d_token + 3, d_token),
            nn.LayerNorm(d_token),
            nn.SiLU(),
        )
        self.null_hist_tok = nn.Parameter(torch.zeros(1, 1, d_token))
        self.traj_cross_attn = SyncFreeCrossAttention(d_model=d_token, n_heads=n_heads)
        self.ctx_kernel_scale_raw = nn.Parameter(torch.tensor(0.5))

        self.traj_residual_mlp = nn.Sequential(
            nn.Linear(d_token + d_token + 24 + 3, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
        )
        self.traj_residual_out = nn.Linear(32, 1, bias=True)
        nn.init.zeros_(self.traj_residual_out.weight)
        nn.init.zeros_(self.traj_residual_out.bias)

    def load_pretrained_s3_context_encoder(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Warm-starts the Pre-Condense context encoder from Stage-1 Full-S3 7-Split Regression
        pretraining while keeping `prior_residual_out` and `traj_residual_out` identically zero.
        """
        own_sd = self.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in own_sd and own_sd[k].shape == v.shape and not k.endswith("_out.weight") and not k.endswith("_out.bias"):
                own_sd[k].copy_(v.to(own_sd[k].device, dtype=own_sd[k].dtype))
                loaded += 1
        nn.init.zeros_(self.prior_residual_out.weight)
        nn.init.zeros_(self.prior_residual_out.bias)
        nn.init.zeros_(self.traj_residual_out.weight)
        nn.init.zeros_(self.traj_residual_out.bias)
        return loaded

    def encode_context_tokens(
        self,
        ctx_tensors: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(ctx_tokens [B, T, 32], ctx_mask [B, T], scale_feat_8 [B, 8])`."""
        B = ctx_tensors["dope629_z"].shape[0]
        device = ctx_tensors["dope629_z"].device

        if self.arm == "B_CAPACITY_CTRL_10TOK":
            raw_10 = ctx_tensors["tokens_10x32"]
            mask_10 = ctx_tensors["mask_10"].clone()
            mask_10[:, 0] = 1.0
            gates = torch.sigmoid(self.tok10_gate_logits).view(1, 10, 1)
            tok = self.tok10_norm(raw_10 + self.tok10_proj(raw_10) + self.tok10_type_emb.unsqueeze(0))
            tok = tok * gates * mask_10.unsqueeze(-1)
            scale_8 = torch.zeros((B, 8), dtype=torch.float32, device=device)
            return tok, mask_10, scale_8

        if self.arm == "C_LEARNED_COMPACT_BRIDGE_629":
            dope_z = ctx_tensors["dope629_z"]
            tok = self.compact_bridge_629(dope_z).view(B, 10, self.d_token)
            gates = torch.sigmoid(self.compact_gate_logits).view(1, 10, 1)
            tok = self.compact_norm(tok + self.compact_type_emb.unsqueeze(0)) * gates
            mask_10 = torch.ones((B, 10), dtype=torch.float32, device=device)
            scale_8 = self.scale_side_proj(ctx_tensors["block_log_norms"])
            return tok, mask_10, scale_8

        # Arms D, E, F start with Coordinate-Preserving Full629 (20 x 32)
        dope_z = ctx_tensors["dope629_z"]
        norms_11 = ctx_tensors["block_log_norms"]

        desc_8x32 = self.desc_proj(dope_z[:, 0:256].view(B, 8, 32))
        task_4x32 = self.task_proj(dope_z[:, 256:384].view(B, 4, 32))
        dist_4x32 = self.dist_proj(dope_z[:, 384:512].view(B, 4, 32))
        heads_64 = torch.cat([dope_z[:, 512:565], norms_11], dim=-1)  # 53 + 11 = 64
        heads_2x32 = self.heads_proj(heads_64.view(B, 2, 32))
        colg_2x32 = self.colgate_proj(dope_z[:, 565:629].view(B, 2, 32))

        raw_20 = torch.cat([desc_8x32, task_4x32, dist_4x32, heads_2x32, colg_2x32], dim=1)
        gates_20 = torch.sigmoid(self.full629_gate_logits).view(1, 20, 1)
        tok_20 = self.full629_norm(raw_20 + self.full629_type_emb.unsqueeze(0)) * gates_20
        mask_20 = torch.ones((B, 20), dtype=torch.float32, device=device)
        scale_8 = self.scale_side_proj(norms_11)

        if self.arm == "D_PRE_CONDENSE_FULL629_20TOK":
            return tok_20, mask_20, scale_8

        # Arms E and F add Pre-Head DOPE router `hidden256` (8 x 32)
        hid_z = ctx_tensors["hidden256_z"]
        raw_h8 = self.hidden_proj(hid_z.view(B, 8, 32))
        gates_h8 = torch.sigmoid(self.hidden_gate_logits).view(1, 8, 1)
        tok_h8 = self.hidden_norm(raw_h8 + self.hidden_type_emb.unsqueeze(0)) * gates_h8
        mask_h8 = torch.ones((B, 8), dtype=torch.float32, device=device)

        if self.arm == "E_FULL629_PLUS_HIDDEN256_28TOK":
            tok_28 = torch.cat([tok_20, tok_h8], dim=1)
            mask_28 = torch.cat([mask_20, mask_h8], dim=1)
            return tok_28, mask_28, scale_8

        # Arm F adds Masked Per-Column Tokens (64 x 18 -> 64 x 32)
        col18_z = ctx_tensors["col18_z"]
        col_mask = ctx_tensors["col_mask_64"]
        col_tok = (self.col_encoder(col18_z) + self.col_type_emb.unsqueeze(0)) * torch.sigmoid(
            self.col_gate_logit
        )
        col_tok = col_tok * col_mask.unsqueeze(-1)

        tok_92 = torch.cat([tok_20, tok_h8, col_tok], dim=1)
        mask_92 = torch.cat([mask_20, mask_h8, col_mask], dim=1)
        return tok_92, mask_92, scale_8

    def forward_prior_and_context(
        self,
        ctx_tensors: Dict[str, torch.Tensor],
        cand_q: torch.Tensor,
        sem24: torch.Tensor,
        gate_risk: torch.Tensor,
        sp_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes Stage-1 pre-reveal residual cost `delta_prior [B, M]` and candidate-context
        features `(attn_ctx [B, M, 32], bilinear_24 [B, M, 24])`.
        """
        B, M, _ = cand_q.shape
        ctx_tokens, ctx_mask, scale_8 = self.encode_context_tokens(ctx_tensors)

        # Mean + max masked pooling for global bilinear summary
        mask_exp = ctx_mask.unsqueeze(-1)
        mask_sum = mask_exp.sum(dim=1).clamp_min(1.0)
        ctx_mean = (ctx_tokens * mask_exp).sum(dim=1) / mask_sum
        ctx_masked_for_max = torch.where(mask_exp > 0.5, ctx_tokens, torch.full_like(ctx_tokens, -1e4))
        ctx_max = ctx_masked_for_max.max(dim=1).values
        ctx_summary_64 = torch.cat([ctx_mean, ctx_max], dim=-1)  # [B, 64]

        attn_ctx = self.ctx_cross_attn(cand_q, ctx_tokens, valid_mask=ctx_mask)  # [B, M, 32]
        u_ctx = self.ctx_bilinear_u(ctx_summary_64).unsqueeze(1)  # [B, 1, 24]
        v_prog = self.prog_bilinear_v(sem24)  # [B, M, 24]
        bilinear_24 = u_ctx * v_prog  # [B, M, 24]

        scale_exp = scale_8.unsqueeze(1).expand(B, M, 8)
        extra_2 = torch.stack([gate_risk, sp_prior], dim=-1)  # [B, M, 2]
        prior_in = torch.cat([attn_ctx, bilinear_24, cand_q, scale_exp, extra_2], dim=-1)
        h_prior = self.prior_residual_mlp(prior_in)
        raw_prior = self.prior_residual_out(h_prior).squeeze(-1)  # [B, M]
        delta_prior = raw_prior - raw_prior.mean(dim=-1, keepdim=True)
        return delta_prior, attn_ctx, bilinear_24

    def forward_causal_trajectory_residual(
        self,
        attn_ctx: torch.Tensor,
        bilinear_24: torch.Tensor,
        cand_q: torch.Tensor,
        sem24: torch.Tensor,
        base_cost_plus_prior: torch.Tensor,
        rev_idx: torch.Tensor,
        rev_y_comb: torch.Tensor,
        rev_y_f: torch.Tensor,
        rev_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Computes Stage-2 Causal 2+2 trajectory residual `delta_traj [B, M]` conditioned on `rev_idx`."""
        B, M, D = attn_ctx.shape
        K_rev = int(rev_idx.shape[1])

        idx_24 = rev_idx.unsqueeze(-1).expand(B, K_rev, 24)
        idx_32 = rev_idx.unsqueeze(-1).expand(B, K_rev, D)
        rev_sem24 = torch.gather(sem24, dim=1, index=idx_24)
        rev_attn_ctx = torch.gather(attn_ctx, dim=1, index=idx_32)
        rev_pred = torch.gather(base_cost_plus_prior, dim=1, index=rev_idx)

        r_rev = rev_y_comb - rev_pred  # [B, K_rev]

        # Context-enriched candidate similarity kernel
        joint_feat = F.normalize(torch.cat([sem24, attn_ctx], dim=-1), p=2, dim=-1, eps=1e-8)
        rev_joint = F.normalize(torch.cat([rev_sem24, rev_attn_ctx], dim=-1), p=2, dim=-1, eps=1e-8)
        cos_sim = torch.bmm(joint_feat, rev_joint.transpose(1, 2)).clamp(-1.0, 1.0)
        d2 = (1.0 - cos_sim) * F.softplus(self.ctx_kernel_scale_raw)
        wk = 0.70 / (0.20 + d2) + 0.30 / (1.50 + d2)
        sem_ctx_adj = torch.sum(wk * r_rev.unsqueeze(1), dim=-1) / (0.25 + torch.sum(wk, dim=-1))

        h_in = torch.cat(
            [
                rev_sem24,
                rev_attn_ctx,
                r_rev.unsqueeze(-1),
                rev_y_f.unsqueeze(-1),
                rev_gate.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, K_rev, 24 + 32 + 3 = 59]
        h_tok = self.hist_proj(h_in)
        null_tok = self.null_hist_tok.expand(B, 1, D)
        kv_seq = torch.cat([null_tok, h_tok], dim=1)

        traj_attn = self.traj_cross_attn(cand_q, kv_seq, valid_mask=None)
        best_rev = rev_y_f.min(dim=-1, keepdim=True).values.unsqueeze(-1).expand(B, M, 1)
        traj_in = torch.cat(
            [
                attn_ctx,
                traj_attn,
                bilinear_24,
                sem_ctx_adj.unsqueeze(-1),
                base_cost_plus_prior.unsqueeze(-1),
                best_rev,
            ],
            dim=-1,
        )
        h_traj = self.traj_residual_mlp(traj_in)
        raw_traj = self.traj_residual_out(h_traj).squeeze(-1)
        delta_traj = raw_traj - raw_traj.mean(dim=-1, keepdim=True)
        return delta_traj
