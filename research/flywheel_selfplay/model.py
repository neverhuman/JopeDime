#!/usr/bin/env python3
"""
Phase 4a: `FlywheelGameNet` — Deep Reinforcement Learning / Self-Play Policy & Value Network
for GP Tree Proposal and Active Sequential Allocation.

Architecture Highlights:
  1. Multi-Modal Context + Landmark-V2 Trunk (`ContextLandmarkEncoder`):
     - Fuses `A0` (88D), `Dope-112` (112D), Pre-Computed Foundation/Kumo-S/Hyperion-V6/TabPFN35/Target
       JL projections (320D), `H27` 8-probe fingerprints (16D), and `Landmark-V2` 12-probe 5-fold CV
       basket (48D) -> 584D total context state.
     - Applies fold-safe robust quantile/IQR clipping and LayerNorm + SwiGLU residual blocks.
  2. Program AST & Bilinear Interaction Tower (`ProgramInteractionEncoder`):
     - Encodes 412D GP AST features + 16D W1 sign projection + Leave-One-Family-Out Empirical-Bayes
       Specificity-over-Popularity (`SoS`) prior + 128D explicit `(Landmark-V2[:8] (x) Proj16)`
       Kronecker interaction features (`H33` exact interaction mechanism).
     - Extracts explicit structural head/stage family masks (e.g., `histgbm`, `ridge`, `ksmooth`,
       `winsor_clip`, `cheb`, `hinge`, `poly_interact`, `rank`) for head-aware kernel conditioning.
  3. In-Context Revealed Trajectory Set-Attention + Head-Aware Multi-Scale Kernel (`TrajectoryConditioner`):
     - Contains the exact `W1` 5-NN inverse-`(1 + d^2)` residual skip connection (`alpha=1.0` anchor).
     - Adds a learned Head-Aware Multi-Scale Residual Kernel (conditioning on whether revealed neighbor `i`
       shares the same terminal estimator head and preprocessing stages as candidate `j`) PLUS a
       neural Cross-Attention Transformer block over revealed `(program_emb, residual, gate_pass)` tokens.
     - Computes an epistemic uncertainty / exploration bonus across unqueried AST clusters during active
       search steps `t < B`.
  4. Calibrated S1 Gate Safety Shield (`GateShieldHead`):
     - Predicts `P(gate_passed | context, landmark, program, revealed_history)` and applies a softplus
       barrier penalty to prevent selecting gate-failing trees (`gate failure <= 0.1%`).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_head_and_stage_signatures(prog_412: torch.Tensor) -> torch.Tensor:
    """
    Extract a compact, L2-normalized 24D structural operator/head signature from the 412D
    DSL feature matrix `prog_412` (`[..., 412]`).
    Used to measure structural AST similarity (head match + transform stage match) alongside
    the 16D W1 random sign projection.
    """
    # Partition the 412D feature vector into 24 deterministic hash buckets via folded projection
    orig_shape = prog_412.shape[:-1]
    x = prog_412.reshape(-1, 412)
    # Pad 412 -> 432 (18 * 24)
    pad = F.pad(x, (0, 20))
    folded = pad.view(-1, 18, 24)
    sig = folded.mean(dim=1) + 0.5 * folded.max(dim=1).values
    sig = F.normalize(sig, p=2, dim=-1, eps=1e-6)
    return sig.view(*orig_shape, 24)


class SwiGLUResBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc_gate = nn.Linear(dim, hidden_dim)
        self.fc_val = nn.Linear(dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        # Small residual init for stability
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc_gate(h)) * self.fc_val(h)
        h = self.drop(self.fc_out(h))
        return x + h


class FlywheelGameNet(nn.Module):
    def __init__(
        self,
        ctx_dim: int = 584,
        prog_dim: int = 412,
        proj_dim: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.prog_dim = prog_dim
        self.proj_dim = proj_dim
        self.d_model = d_model

        # 1. Context + Landmark-V2 Trunk (584D -> d_model)
        self.ctx_in = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, d_model),
            nn.SiLU(),
        )
        self.ctx_block = SwiGLUResBlock(d_model, d_model * 2, dropout=dropout)

        # Dedicated Landmark-V2 + H27-FIVE routing projection (64D -> 16D routing weights over proj16)
        self.landmark_router = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, proj_dim),
        )
        nn.init.normal_(self.landmark_router[-1].weight, std=0.01)
        nn.init.zeros_(self.landmark_router[-1].bias)

        # 2. Program AST + Bilinear Interaction Tower
        # Inputs: prog_412 (412D) + proj16 (16D) + kron_128 (8D landmark * 16D proj16 = 128D) + sos_prior (2D) = 558D
        self.prog_in_dim = prog_dim + proj_dim + 128 + 2
        self.prog_in = nn.Sequential(
            nn.LayerNorm(self.prog_in_dim),
            nn.Linear(self.prog_in_dim, d_model),
            nn.SiLU(),
        )
        self.prog_block = SwiGLUResBlock(d_model, d_model * 2, dropout=dropout)

        # Context-Program Prior Cost Residual Head
        self.prior_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.prior_head[-1].weight)
        nn.init.zeros_(self.prior_head[-1].bias)

        # 3. In-Context Revealed Trajectory Cross-Attention & Multi-Scale Kernel Conditioner
        # Token encoder for revealed outcomes: [prog_emb (d_model), residual, gate_float, prior_cost, step_frac]
        self.history_token_in = nn.Sequential(
            nn.Linear(d_model + 4, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.traj_out = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 4),
            nn.Linear(d_model * 2 + 4, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.traj_out[-1].weight)
        nn.init.zeros_(self.traj_out[-1].bias)

        # Multi-scale kernel parameters (initialized so W1 5-NN is the exact anchor!)
        self.alpha_prior = nn.Parameter(torch.tensor(0.08))
        self.alpha_struct_kernel = nn.Parameter(torch.tensor(0.15))
        self.alpha_attn = nn.Parameter(torch.tensor(0.05))
        self.alpha_ucb = nn.Parameter(torch.tensor(0.02))

        # 4. Calibrated S1 Gate Safety Shield
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 2),
            nn.Linear(d_model * 2 + 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -2.5)
        self.gate_barrier_weight = nn.Parameter(torch.tensor(0.05))

        # 5. Multi-Generation Evolvability Lookahead Head (\hat{v}_{evo,\theta}(c, p))
        # Decouples immediate 1-step cost \hat{c}_\theta(c, p) from G-step evolutionary potential
        self.evolvability_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.evolvability_head[-1].weight)
        nn.init.zeros_(self.evolvability_head[-1].bias)

    def encode_candidates(
        self,
        ctx_feat: torch.Tensor,      # [B, 584]
        prog_412: torch.Tensor,      # [B, N, 412]
        proj_16: torch.Tensor,       # [B, N, 16]
        sos_prior: torch.Tensor,     # [B, N, 2] (mean & freq of program across disjoint train families)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute context embeddings, candidate embeddings, structural AST signatures,
        and prior cost adjustments `[B, N]`.
        """
        bsz, n_cands, _ = prog_412.shape
        ctx_clamped = torch.clamp(ctx_feat, -15.0, 15.0)
        ctx_emb = self.ctx_block(self.ctx_in(ctx_clamped))  # [B, d_model]

        # Extract H27-FP16 (indices 520:536) and Landmark-V2-48 (indices 536:584) -> 64D landmark state
        lm_64 = ctx_clamped[:, 520:584]  # [B, 64]
        lm_top8 = ctx_clamped[:, 520:528]  # [B, 8] (H27 FIVE / Landmark top-8)

        # Explicit H33 Kronecker interaction: outer(lm_top8 [B, 1, 8], proj_16 [B, N, 16]) -> [B, N, 128]
        kron_128 = (lm_top8.unsqueeze(1).unsqueeze(-1) * proj_16.unsqueeze(2)).reshape(bsz, n_cands, 128)

        prog_clamped = torch.clamp(prog_412, -15.0, 15.0)
        prog_in_vec = torch.cat([prog_clamped, proj_16.float(), kron_128.float(), sos_prior.float()], dim=-1)
        prog_emb = self.prog_block(self.prog_in(prog_in_vec))  # [B, N, d_model]

        # Bilinear landmark routing score + neural prior residual
        lm_weights = self.landmark_router(lm_64)  # [B, 16]
        bilinear_delta = torch.einsum("bd,bnd->bn", lm_weights, proj_16.float())  # [B, N]

        ctx_exp = ctx_emb.unsqueeze(1).expand(bsz, n_cands, self.d_model)
        joint = torch.cat([ctx_exp, prog_emb], dim=-1)
        mlp_delta = self.prior_head(joint).squeeze(-1)  # [B, N]

        # SoS empirical-bayes prior term (sos_prior[..., 0] is family-out normalized cost offset)
        sos_delta = sos_prior[..., 0].float() * sos_prior[..., 1].float()

        prior_delta = torch.tanh(bilinear_delta + mlp_delta + 0.35 * sos_delta)  # bounded in [-1, 1]
        struct_sig = extract_head_and_stage_signatures(prog_clamped)  # [B, N, 24]
        return ctx_emb, prog_emb, struct_sig, prior_delta

    def predict_costs(
        self,
        ctx_feat: torch.Tensor,           # [B, 584]
        prog_412: torch.Tensor,           # [B, N, 412]
        proj_16: torch.Tensor,            # [B, N, 16] (float64 for exact W1 distance parity)
        base_prior_cost: torch.Tensor,    # [B, N] (float64 H30 ZERO prior cost or fallback)
        scale: torch.Tensor,              # [B] (float64 target scale S)
        sos_prior: torch.Tensor,          # [B, N, 2]
        revealed_indices: torch.Tensor,   # [B, T] (int64 indices of revealed candidates, -1 if padding)
        revealed_raw_f: torch.Tensor,     # [B, T] (float64 revealed constrained fitness F)
        revealed_gate: torch.Tensor,      # [B, T] (float32 1.0 if gate passed else 0.0)
        enable_prior_correction: bool = True,
        enable_struct_kernel: bool = True,
        enable_cross_attn: bool = True,
        enable_ucb: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute predicted candidate costs `[B, N]` conditioned on revealed trajectory `H_t`.
        Strictly preserves exact `W1` 5-NN residual conditioning as the skip-connection backbone.
        """
        bsz, n_cands, _ = prog_412.shape
        t_steps = revealed_indices.shape[1]

        ctx_emb, prog_emb, struct_sig, prior_delta = self.encode_candidates(
            ctx_feat, prog_412, proj_16, sos_prior
        )

        # Effective prior cost (anchored at base_prior_cost)
        if enable_prior_correction:
            eff_prior = base_prior_cost + self.alpha_prior.double() * prior_delta.double()
        else:
            eff_prior = base_prior_cost

        if t_steps == 0:
            # Zero-shot / pre-reveal step
            ctx_exp = ctx_emb.unsqueeze(1).expand(bsz, n_cands, self.d_model)
            gate_in = torch.cat([ctx_exp, prog_emb, eff_prior.float().unsqueeze(-1), torch.zeros_like(eff_prior.float().unsqueeze(-1))], dim=-1)
            fail_logit = self.gate_head(gate_in).squeeze(-1)
            gate_pen = F.softplus(self.gate_barrier_weight) * F.sigmoid(fail_logit)
            total_cost = eff_prior + gate_pen.double()
            return {
                "total_cost": total_cost,
                "w1_cost": base_prior_cost,
                "eff_prior": eff_prior,
                "fail_logit": fail_logit,
            }

        # Gather revealed candidate representations
        valid_mask = revealed_indices >= 0  # [B, T]
        safe_idx = revealed_indices.clamp(min=0)  # [B, T]

        # 1. Exact W1 5-NN residual conditioning in 16D sign-projection space (FP64)
        idx_16 = safe_idx.unsqueeze(-1).expand(bsz, t_steps, 16)
        rev_proj_16 = torch.gather(proj_16.double(), dim=1, index=idx_16)  # [B, T, 16]
        rev_base_prior = torch.gather(base_prior_cost.double(), dim=1, index=safe_idx)  # [B, T]
        rev_eff_prior = torch.gather(eff_prior.double(), dim=1, index=safe_idx)  # [B, T]

        # Revealed residuals: r_i = F_i / S - p_i
        # Cap extreme failure penalty F=100 when computing smooth structural kernel residuals, while keeping exact W1 residual
        rev_res_w1 = revealed_raw_f.double() / scale.double().unsqueeze(-1) - rev_base_prior  # [B, T]
        clamped_f = torch.where(
            revealed_gate > 0.5,
            revealed_raw_f.double(),
            torch.clamp(revealed_raw_f.double(), max=2.5),
        )
        rev_res_smooth = clamped_f / scale.double().unsqueeze(-1) - rev_eff_prior  # [B, T]

        # Pairwise squared Euclidean distances in 16D W1 projection space: [B, N, T]
        diff_16 = proj_16.double().unsqueeze(2) - rev_proj_16.unsqueeze(1)  # [B, N, T, 16]
        dist2_16 = torch.sum(diff_16 * diff_16, dim=-1)  # [B, N, T]
        dist2_16 = torch.where(valid_mask.unsqueeze(1), dist2_16, torch.full_like(dist2_16, 1e12))

        k_nn = min(5, t_steps)
        top_d2, top_order = torch.topk(dist2_16, k=k_nn, dim=-1, largest=False, sorted=True)  # [B, N, k_nn]
        top_valid = top_d2 < 1e11
        w_w1 = torch.where(top_valid, 1.0 / (1.0 + top_d2), torch.zeros_like(top_d2))
        rev_res_w1_exp = rev_res_w1.unsqueeze(1).expand(bsz, n_cands, t_steps)
        top_res_w1 = torch.gather(rev_res_w1_exp, dim=-1, index=top_order)
        w1_num = torch.sum(w_w1 * top_res_w1, dim=-1)
        w1_den = torch.sum(w_w1, dim=-1).clamp(min=1e-12)
        w1_residual_adj = torch.where(valid_mask.any(dim=-1, keepdim=True), w1_num / w1_den, torch.zeros_like(w1_num))
        w1_cost = base_prior_cost.double() + w1_residual_adj  # Exact W1 predictor!

        # 2. Head-Aware Multi-Scale Structural Kernel Conditioning
        struct_adj = torch.zeros_like(w1_cost)
        min_struct_dist = torch.ones((bsz, n_cands), dtype=torch.float32, device=prog_412.device)
        rev_gate_local = torch.zeros((bsz, n_cands), dtype=torch.float32, device=prog_412.device)

        if enable_struct_kernel:
            idx_24 = safe_idx.unsqueeze(-1).expand(bsz, t_steps, 24)
            rev_sig_24 = torch.gather(struct_sig, dim=1, index=idx_24)  # [B, T, 24]
            # Cosine similarity in 24D AST structural space -> squared distance in [0, 4]
            cos_sim = torch.einsum("bnd,btd->bnt", struct_sig, rev_sig_24)  # [B, N, T]
            dist2_struct = (1.0 - cos_sim).clamp(min=0.0) * 4.0  # [B, N, T]
            joint_d2 = 0.5 * dist2_16.float() + 2.0 * dist2_struct
            joint_d2 = torch.where(valid_mask.unsqueeze(1), joint_d2, torch.full_like(joint_d2, 1e9))

            min_struct_dist = torch.min(joint_d2, dim=-1).values.clamp(max=10.0) / 10.0  # [B, N] in [0, 1]

            k_s = min(8, t_steps)
            s_d2, s_ord = torch.topk(joint_d2.double(), k=k_s, dim=-1, largest=False, sorted=True)
            s_val = s_d2 < 1e8
            # Sharp + broad multi-scale inverse distance kernel
            w_s = torch.where(s_val, 0.7 / (0.25 + s_d2) + 0.3 / (2.0 + s_d2), torch.zeros_like(s_d2))
            rev_res_sm_exp = rev_res_smooth.unsqueeze(1).expand(bsz, n_cands, t_steps)
            top_res_sm = torch.gather(rev_res_sm_exp, dim=-1, index=s_ord)
            s_num = torch.sum(w_s * top_res_sm, dim=-1)
            s_den = torch.sum(w_s, dim=-1).clamp(min=1e-12)
            struct_adj = torch.where(valid_mask.any(dim=-1, keepdim=True), s_num / s_den, torch.zeros_like(s_num))

            # Local gate failure rate among structural neighbors
            rev_fail_exp = (1.0 - revealed_gate).double().unsqueeze(1).expand(bsz, n_cands, t_steps)
            top_fail = torch.gather(rev_fail_exp, dim=-1, index=s_ord)
            rev_gate_local = (torch.sum(w_s * top_fail, dim=-1) / s_den).float()

        # 3. Neural Cross-Attention over Revealed Trajectory Tokens
        attn_delta = torch.zeros((bsz, n_cands), dtype=torch.float32, device=prog_412.device)
        if enable_cross_attn:
            idx_emb = safe_idx.unsqueeze(-1).expand(bsz, t_steps, self.d_model)
            rev_prog_emb = torch.gather(prog_emb, dim=1, index=idx_emb)  # [B, T, d_model]
            step_frac = torch.linspace(0.1, 1.0, steps=t_steps, device=prog_412.device).unsqueeze(0).expand(bsz, t_steps)
            tok_feats = torch.stack([
                torch.clamp(rev_res_smooth.float(), -5.0, 5.0),
                revealed_gate.float(),
                torch.clamp(rev_eff_prior.float(), -5.0, 5.0),
                step_frac,
            ], dim=-1)  # [B, T, 4]
            mem_tokens = self.history_token_in(torch.cat([rev_prog_emb, tok_feats], dim=-1))  # [B, T, d_model]
            key_padding_mask = ~valid_mask  # True where padded
            attn_out, _ = self.cross_attn(
                query=prog_emb,
                key=mem_tokens,
                value=mem_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )  # [B, N, d_model]
            summary_feats = torch.stack([
                w1_residual_adj.float().clamp(-5.0, 5.0),
                struct_adj.float().clamp(-5.0, 5.0),
                min_struct_dist,
                rev_gate_local,
            ], dim=-1)  # [B, N, 4]
            ctx_exp = ctx_emb.unsqueeze(1).expand(bsz, n_cands, self.d_model)
            traj_in = torch.cat([ctx_exp, attn_out, summary_feats], dim=-1)
            attn_delta = torch.tanh(self.traj_out(traj_in).squeeze(-1))  # [B, N] bounded in [-1, 1]

        # 4. Calibrated S1 Gate Safety Shield
        ctx_exp = ctx_emb.unsqueeze(1).expand(bsz, n_cands, self.d_model)
        gate_in = torch.cat([
            ctx_exp,
            prog_emb,
            eff_prior.float().unsqueeze(-1),
            rev_gate_local.unsqueeze(-1),
        ], dim=-1)
        fail_logit = self.gate_head(gate_in).squeeze(-1) + 2.0 * rev_gate_local
        gate_pen = F.softplus(self.gate_barrier_weight) * F.sigmoid(fail_logit)

        # Combine W1 anchor + Prior improvement + Head-aware Structural Kernel + Neural Trajectory Cross-Attention + Gate Shield
        w_struct = torch.sigmoid(self.alpha_struct_kernel).double() * 0.35
        blended_residual = (1.0 - w_struct) * w1_residual_adj + w_struct * struct_adj

        total_cost = (
            eff_prior
            + blended_residual
            + self.alpha_attn.double() * attn_delta.double()
            + gate_pen.double()
        )
        if enable_ucb:
            total_cost = total_cost - torch.abs(self.alpha_ucb).double() * min_struct_dist.double()

        return {
            "total_cost": total_cost,
            "w1_cost": w1_cost,
            "eff_prior": eff_prior,
            "w1_residual_adj": w1_residual_adj,
            "struct_adj": struct_adj,
            "attn_delta": attn_delta,
            "fail_logit": fail_logit,
        }

    def predict_evolvability(
        self,
        ctx_feat: torch.Tensor,      # [B, 584]
        prog_412: torch.Tensor,      # [B, N, 412]
        proj_16: torch.Tensor,       # [B, N, 16]
        sos_prior: torch.Tensor,     # [B, N, 2]
    ) -> torch.Tensor:
        r"""
        Predict non-negative G-step evolutionary improvement potential `\hat{v}_{evo,\theta}(c, p)` (`[B, N]`).
        Decoupled from the immediate 1-step cost head so parent selection for GP evolution
        can favor fertile scaffolds while argmin deployment picks the lowest immediate cost.
        """
        bsz, n_cands, _ = prog_412.shape
        ctx_emb, prog_emb, _, _ = self.encode_candidates(ctx_feat, prog_412, proj_16, sos_prior)
        ctx_exp = ctx_emb.unsqueeze(1).expand(bsz, n_cands, self.d_model)
        joint = torch.cat([ctx_exp, prog_emb], dim=-1)
        return F.softplus(self.evolvability_head(joint).squeeze(-1))


def compute_directional_evo_loss(
    model: FlywheelGameNet,
    evo_batch: dict[str, torch.Tensor] | None,
    device: torch.device,
    telemetry_h2d=None,
    telemetry_step: int = 0,
) -> dict[str, torch.Tensor]:
    r"""
    Compute the **Directional Evolutionary Gradient Attribution Loss** (`DirectionalEvoLoss`)
    from real-time evolutionary GP batch trajectories prepared by `AsyncEvoBatchQueue`:
      1. Exact Child Featurization (`\phi(p_parent)` vs `\phi(p_child)`):
         - Evaluates `model.encode_candidates` on BOTH `parent_feat412` and `child_feat412`
           (never grafting `F(c, p_child)` onto `\phi(p_parent)`!).
      2. Paired Parent -> Child Directional Edit Margin Loss (`L_{evo-dir}`):
         - For every evolutionary transition `(c, p_parent, p_child, \Delta F_300)`:
           Let `y_dir = sign(\Delta F_300)` (`-1` when child improved over parent, `+1` when child degraded).
           Requires `\hat{c}_\theta(c, p_child) - \hat{c}_\theta(c, p_parent)` to match `y_dir` with
           margin `m = clamp(|\Delta F_300| / scale, max=0.50)`, weighted by `w = clamp(|\Delta F_300| / 0.05, max=2.0)`.
         - The gradient wrt `\theta` flows strictly along `\phi(p_child) - \phi(p_parent)` in the
           true direction of fitness improvement!
      3. Multi-Generation Evolvability Lookahead Loss (`L_{evo-val}`):
         - Supervises `model.evolvability_head(c, p_parent)` to predict the observed G-step
           evolutionary fitness gain `g_{evo}(c, p_parent) = max(0, F_300(p_parent) - min_{g<=G} F_300(p_child^{(g)}))`.
    """
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    if evo_batch is None or evo_batch.get("n_transitions", 0) == 0:
        return {"total_evo_loss": zero, "dir_margin_loss": zero, "evo_val_loss": zero}

    h2d_slot = -1
    if telemetry_h2d is not None and telemetry_h2d.will_sample(telemetry_step):
        transfer_bytes = sum(
            evo_batch[key].numel() * evo_batch[key].element_size()
            for key in ("trans_ctx_584", "parent_feat412", "child_feat412", "parent_proj16", "child_proj16", "delta_f300", "scale", "child_gate")
        )
    else:
        transfer_bytes = 0
    if telemetry_h2d is not None:
        h2d_slot = telemetry_h2d.begin(telemetry_step, transfer_bytes)
    t_ctx = evo_batch["trans_ctx_584"].to(device)            # [K, 584]
    p_f412 = evo_batch["parent_feat412"].to(device)          # [K, 1, 412]
    c_f412 = evo_batch["child_feat412"].to(device)           # [K, 1, 412]
    p_pr16 = evo_batch["parent_proj16"].to(device)           # [K, 1, 16]
    c_pr16 = evo_batch["child_proj16"].to(device)            # [K, 1, 16]
    delta_f300 = evo_batch["delta_f300"].to(device)          # [K] (negative = child improved)
    scale_k = evo_batch["scale"].to(device).clamp(min=0.1)   # [K]
    c_gate = evo_batch["child_gate"].to(device)              # [K]
    if telemetry_h2d is not None:
        telemetry_h2d.end(h2d_slot)

    k_trans = t_ctx.shape[0]
    both_f412 = torch.cat([p_f412, c_f412], dim=1)           # [K, 2, 412]
    both_pr16 = torch.cat([p_pr16, c_pr16], dim=1)           # [K, 2, 16]
    both_sos = torch.zeros((k_trans, 2, 2), dtype=torch.float32, device=device)

    ctx_emb, prog_emb, _, prior_delta = model.encode_candidates(t_ctx, both_f412, both_pr16, both_sos)
    # Predicted normalized cost offset for parent (index 0) and child (index 1)
    cost_parent = prior_delta[:, 0]  # [K]
    cost_child = prior_delta[:, 1]   # [K]
    pred_diff = cost_child - cost_parent  # Should be < 0 when delta_f300 < 0!

    norm_delta = delta_f300 / scale_k
    y_dir = torch.where(norm_delta < -1e-5, -torch.ones_like(norm_delta), torch.ones_like(norm_delta))
    margin = torch.clamp(torch.abs(norm_delta), min=0.01, max=0.50)
    weight = torch.clamp(torch.abs(norm_delta) / 0.04, min=0.15, max=2.0)

    # Directional contrastive margin loss along \phi(p_child) - \phi(p_parent)
    dir_margin_loss = torch.mean(weight * F.softplus(4.0 * (margin - y_dir * pred_diff)))

    # Gate supervision on evolved children
    ctx_child = ctx_emb
    prog_child = prog_emb[:, 1, :]
    gate_in = torch.cat([
        ctx_child,
        prog_child,
        cost_child.unsqueeze(-1),
        torch.zeros_like(cost_child.unsqueeze(-1)),
    ], dim=-1)
    child_fail_logit = model.gate_head(gate_in).squeeze(-1)
    child_gate_loss = F.binary_cross_entropy_with_logits(child_fail_logit, 1.0 - c_gate)

    # Multi-generation evolvability lookahead supervision on starting parents
    ctx_parent = ctx_emb
    prog_parent = prog_emb[:, 0, :]
    pred_evo_gain = F.softplus(model.evolvability_head(torch.cat([ctx_parent, prog_parent], dim=-1)).squeeze(-1))
    gain = evo_batch["parent_evo_gain"]
    gain_bytes = gain.numel() * gain.element_size() if telemetry_h2d is not None and telemetry_h2d.will_sample(telemetry_step) else 0
    gain_slot = telemetry_h2d.begin(telemetry_step, gain_bytes) if telemetry_h2d is not None else -1
    true_evo_gain = gain.to(device) / scale_k
    if telemetry_h2d is not None:
        telemetry_h2d.end(gain_slot)
    evo_val_loss = F.smooth_l1_loss(pred_evo_gain, true_evo_gain.clamp(min=0.0, max=1.5), beta=0.10)

    total_evo = 0.25 * dir_margin_loss + 0.15 * evo_val_loss + 0.10 * child_gate_loss
    return {
        "total_evo_loss": total_evo,
        "dir_margin_loss": dir_margin_loss,
        "evo_val_loss": evo_val_loss,
    }
