"""The fusion head — the ONLY trainable part. Experts are frozen upstream.

Per test row: a reliability gate (over generalizable features) weights each expert;
the fused prediction = gated calibrated-expert means + a small residual from the
fused adapter representation. Regression first; classification is the same shape
with logits + temperature calibration.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HID = 512


class Adapter(nn.Module):
    """512-d expert hidden -> d_fuse, low-rank."""
    def __init__(self, d_fuse=256, rank=64, p=0.05):
        super().__init__()
        self.pre = nn.LayerNorm(HID)
        self.down = nn.Linear(HID, rank, bias=False)
        self.up = nn.Linear(rank, d_fuse, bias=False)
        self.post = nn.LayerNorm(d_fuse)
        self.gain = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(p)

    def forward(self, h):
        z = self.up(self.drop(F.gelu(self.down(self.pre(h)))))
        return self.post(z) * self.gain


class RegCalibrator(nn.Module):
    """Affine calibration of an expert's mean (identity at init)."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, mean):
        return mean * self.scale + self.bias


class RegressionFusion(nn.Module):
    """experts: list of names. Per-row inputs come from the expert cache."""
    def __init__(self, n_experts, gate_meta_dim=8, conf_dim=3, d_fuse=256,
                 rank=64, gate_hidden=128):
        super().__init__()
        self.n_experts = n_experts
        self.adapters = nn.ModuleList([Adapter(d_fuse, rank) for _ in range(n_experts)])
        self.cal = nn.ModuleList([RegCalibrator() for _ in range(n_experts)])
        # gate input: meta ⊕ per-expert conf ⊕ pairwise disagreement scalar
        gate_in = gate_meta_dim + n_experts * conf_dim + 1
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden), nn.GELU(), nn.LayerNorm(gate_hidden),
            nn.Dropout(0.05), nn.Linear(gate_hidden, n_experts))
        self.gate_temp = nn.Parameter(torch.zeros(1))       # softmax temp = exp(gate_temp)
        self.head = nn.Sequential(
            nn.Linear(d_fuse, 128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128, 1))
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

    def gate_weights(self, gate_meta, conf):
        """gate_meta (M, Dm), conf (M, E, Cc) -> w (M, E)."""
        M = conf.shape[0]
        disagree = conf[..., -1].std(dim=1, keepdim=True) if self.n_experts > 1 \
            else torch.zeros(M, 1, device=conf.device)      # spread of expert means
        gin = torch.cat([gate_meta, conf.reshape(M, -1), disagree], -1)
        logits = self.gate(gin) / torch.exp(self.gate_temp).clamp(0.2, 5.0)
        return torch.softmax(logits, -1)

    def forward(self, means, hidden, gate_meta, conf, parity_expert=None):
        """means (M,E), hidden (M,E,512), gate_meta (M,Dm), conf (M,E,Cc).
        Returns dict(final, gated, residual, w, cal_means)."""
        M, E = means.shape
        cal_means = torch.stack([self.cal[e](means[:, e]) for e in range(E)], -1)  # (M,E)
        if parity_expert is not None:
            w = torch.zeros(M, E, device=means.device); w[:, parity_expert] = 1.0
            gated = (w * cal_means).sum(-1)
            return dict(final=gated, gated=gated, residual=torch.zeros_like(gated),
                        w=w, cal_means=cal_means)
        w = self.gate_weights(gate_meta, conf)               # (M,E)
        gated = (w * cal_means).sum(-1)
        z = torch.stack([self.adapters[e](hidden[:, e]) for e in range(E)], 1)  # (M,E,d)
        z_fused = (w.unsqueeze(-1) * z).sum(1)               # (M,d)
        residual = self.residual_scale * self.head(z_fused).squeeze(-1)
        return dict(final=gated + residual, gated=gated, residual=residual,
                    w=w, cal_means=cal_means)

    def fusion_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ==========================================================================
# ConcatFusion — DEEP STACKING: treat the frozen experts as a feature backbone
# and decide from ALL their fine features (both 512-d reps + interactions +
# calibrated logits + conf + meta) via a real network. "Acts like one model."
# ==========================================================================
class LogitCalibrator(nn.Module):
    """Temperature + per-class bias (identity at init)."""
    def __init__(self, k):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, logits):
        return logits / torch.exp(self.log_temp).clamp(0.2, 5.0) + self.bias


class ConcatFusion(nn.Module):
    def __init__(self, n_experts, out_dim=2, gate_meta_dim=8, conf_dim=3,
                 d_fuse=256, rank=64, head_hidden=(512, 256), p=0.1,
                 interactions=True, route=True):
        super().__init__()
        self.n_experts = n_experts
        self.out_dim = out_dim
        self.route = route
        self.interactions = interactions and n_experts == 2
        self.adapters = nn.ModuleList([Adapter(d_fuse, rank, p) for _ in range(n_experts)])
        self.cal = nn.ModuleList([LogitCalibrator(out_dim) for _ in range(n_experts)])
        # per-DATASET routing gate over experts (the diagnostic: better expert is a
        # per-dataset property; averaging hurts on 78% of datasets).
        gate_in = gate_meta_dim + 2 * n_experts * conf_dim   # meta ⊕ pooled conf mean/std
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(p),
            nn.Linear(128, n_experts))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)  # start uniform
        feat = n_experts * d_fuse + n_experts * out_dim + n_experts * conf_dim + gate_meta_dim
        if self.interactions:
            feat += 2 * d_fuse                          # z_prod ⊕ z_diff
        layers = []; d = feat
        for h in head_hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(p)]; d = h
        layers += [nn.Linear(d, out_dim)]
        self.head = nn.Sequential(*layers)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        # base = learned linear over calibrated expert logits, init = average
        self.base = nn.Linear(n_experts * out_dim, out_dim, bias=False)
        with torch.no_grad():
            w = torch.zeros(out_dim, n_experts * out_dim)
            for e in range(n_experts):
                w[:, e * out_dim:(e + 1) * out_dim] = torch.eye(out_dim) / n_experts
            self.base.weight.copy_(w)
        self.resid_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, logits, hidden, gate_meta, conf, parity_expert=None):
        """logits (M,E,K), hidden (M,E,512), gate_meta (M,Dm), conf (M,E,Cc)."""
        M, E, K = logits.shape
        cal = torch.stack([self.cal[e](logits[:, e]) for e in range(E)], 1)   # (M,E,K)
        cal_flat = cal.reshape(M, E * K)
        if parity_expert is not None:                    # P0: reproduce one expert
            return dict(final=cal[:, parity_expert], base=cal[:, parity_expert],
                        cal=cal, w=None)
        if self.route:                                   # per-dataset routing gate
            pooled = torch.cat([gate_meta[0], conf.mean(0).reshape(-1),
                                conf.std(0).reshape(-1)], -1)
            w = torch.softmax(self.gate(pooled), -1)     # (E,) one weight per dataset
            base = (w.view(1, E, 1) * cal).sum(1)        # (M,K)
        else:
            base = self.base(cal_flat); w = None
        z = torch.stack([self.adapters[e](hidden[:, e]) for e in range(E)], 1)  # (M,E,d)
        parts = [z.reshape(M, -1), cal_flat, conf.reshape(M, -1), gate_meta]
        if self.interactions:
            parts += [z[:, 0] * z[:, 1], z[:, 0] - z[:, 1]]
        feat = torch.cat(parts, -1)
        final = base + self.resid_scale * self.head(feat)
        return dict(final=final, base=base, cal=cal, w=w)

    def fusion_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ==========================================================================
# AttnFusion — per-ROW routing + cross-expert ATTENTION. Each frozen expert's
# adapted rep is a token; a small transformer models agreement/disagreement;
# per-row gate routes (finer than per-dataset -> can EXCEED the per-dataset
# oracle); residual head re-ranks. Anchored: final = routed_base + resid*head.
# ==========================================================================
class AttnFusion(nn.Module):
    def __init__(self, n_experts, out_dim=2, gate_meta_dim=8, conf_dim=3,
                 d=128, rank=64, d_model=128, n_layers=2, n_heads=4, p=0.1):
        super().__init__()
        self.n_experts = n_experts; self.out_dim = out_dim
        self.adapters = nn.ModuleList([Adapter(d, rank, p) for _ in range(n_experts)])
        self.cal = nn.ModuleList([LogitCalibrator(out_dim) for _ in range(n_experts)])
        self.tok = nn.Linear(d + out_dim + conf_dim, d_model)   # expert token
        self.meta_tok = nn.Linear(gate_meta_dim, d_model)       # CLS/meta token
        self.type_emb = nn.Parameter(torch.randn(n_experts + 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                         dropout=p, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, n_layers)
        self.gate_head = nn.Linear(d_model, 1)                  # per expert-token -> route score
        self.resid_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Dropout(p), nn.Linear(d_model, out_dim))
        nn.init.zeros_(self.resid_head[-1].weight); nn.init.zeros_(self.resid_head[-1].bias)
        nn.init.zeros_(self.gate_head.weight); nn.init.zeros_(self.gate_head.bias)  # start uniform
        self.resid_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, logits, hidden, gate_meta, conf, parity_expert=None):
        M, E, K = logits.shape
        cal = torch.stack([self.cal[e](logits[:, e]) for e in range(E)], 1)   # (M,E,K)
        if parity_expert is not None:
            return dict(final=cal[:, parity_expert], cal=cal, w=None)
        z = torch.stack([self.adapters[e](hidden[:, e]) for e in range(E)], 1)  # (M,E,d)
        etok = self.tok(torch.cat([z, cal, conf], -1))          # (M,E,dm)
        ctok = self.meta_tok(gate_meta).unsqueeze(1)            # (M,1,dm)
        tokens = torch.cat([ctok, etok], 1) + self.type_emb.unsqueeze(0)  # (M,E+1,dm)
        h = self.tf(tokens)                                     # (M,E+1,dm)
        w = torch.softmax(self.gate_head(h[:, 1:]).squeeze(-1), -1)  # (M,E) per-row routing
        routed = (w.unsqueeze(-1) * cal).sum(1)                 # (M,K)
        final = routed + self.resid_scale * self.resid_head(h[:, 0])
        return dict(final=final, cal=cal, w=w)

    def fusion_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ==========================================================================
# AttnFusionReg — the REGRESSION analog of AttnFusion. Per-row cross-expert
# attention over each frozen regressor's (adapted hidden ⊕ mean ⊕ conf ⊕
# quantiles) tokens; per-row routing over expert MEANS + a residual correction
# -> a single fused scalar prediction. Trains the trunk only; experts frozen.
# ==========================================================================
class AttnFusionReg(nn.Module):
    def __init__(self, n_experts=2, gate_meta_dim=8, conf_dim=3, quant_dim=19,
                 d=256, rank=128, d_model=512, n_layers=8, n_heads=8, p=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.adapters = nn.ModuleList([Adapter(d, rank, p) for _ in range(n_experts)])
        self.cal = nn.ModuleList([RegCalibrator() for _ in range(n_experts)])
        self.tok = nn.Linear(d + 1 + conf_dim + quant_dim, d_model)   # expert token
        self.meta_tok = nn.Linear(gate_meta_dim, d_model)
        self.type_emb = nn.Parameter(torch.randn(n_experts + 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                         dropout=p, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, n_layers)
        # per-row LINEAR coefficients over expert means (NOT softmax -> not capped by the
        # best expert; init = average, but can learn >1 on the strong / <0 on the weak).
        self.coef_head = nn.Linear(d_model, n_experts)
        nn.init.zeros_(self.coef_head.weight)
        nn.init.constant_(self.coef_head.bias, 1.0 / n_experts)     # start at the average
        self.resid_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Dropout(p), nn.Linear(d_model, 1))
        nn.init.zeros_(self.resid_head[-1].weight); nn.init.zeros_(self.resid_head[-1].bias)
        self.resid_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, means, hidden, gate_meta, conf, quant, parity_expert=None):
        """means (M,E), hidden (M,E,512), gate_meta (M,Dm), conf (M,E,Cc), quant (M,E,Q)."""
        M, E = means.shape
        cal = torch.stack([self.cal[e](means[:, e]) for e in range(E)], 1)   # (M,E) calibrated means
        if parity_expert is not None:
            return dict(final=cal[:, parity_expert], cal=cal, w=None)
        z = torch.stack([self.adapters[e](hidden[:, e]) for e in range(E)], 1)  # (M,E,d)
        etok = self.tok(torch.cat([z, means.unsqueeze(-1), conf, quant], -1))   # (M,E,dm)
        ctok = self.meta_tok(gate_meta).unsqueeze(1)
        tokens = torch.cat([ctok, etok], 1) + self.type_emb.unsqueeze(0)
        h = self.tf(tokens)
        coef = self.coef_head(h[:, 0])                              # (M,E) per-row linear coeffs
        base = (coef * cal).sum(1)                                  # (M,) learned combo (uncapped)
        final = base + self.resid_scale * self.resid_head(h[:, 0]).squeeze(-1)
        return dict(final=final, cal=cal, w=torch.softmax(coef.detach(), -1))

    def fusion_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ==========================================================================
# StackFusionReg — the anti-plateau regression fusion. A PRECISION-weighted
# base (inverse-variance blend of the two calibrated means; strictly beats
# averaging when errors are imperfectly correlated) + a DEEP stacking head
# over ALL per-row features (both adapted 512-d hiddens, means, quantiles,
# conf, meta, cross-expert interactions, uncertainties) -> an UN-throttled
# correction. Escapes span{mean0,mean1}, so it can add value beyond selecting
# one expert. Per-row (keeps the vectorized cached DDP path).
# ==========================================================================
class StackFusionReg(nn.Module):
    def __init__(self, n_experts=2, gate_meta_dim=8, conf_dim=3, quant_dim=19,
                 d_fuse=256, rank=128, head_hidden=(512, 512, 256), p=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.adapters = nn.ModuleList([Adapter(d_fuse, rank, p) for _ in range(n_experts)])
        self.cal = nn.ModuleList([RegCalibrator() for _ in range(n_experts)])
        inter = (2 * d_fuse + 1) if n_experts == 2 else 0        # z0*z1 ⊕ z0-z1 ⊕ |cal0-cal1|
        feat = (n_experts * d_fuse + n_experts + n_experts * quant_dim
                + n_experts * conf_dim + gate_meta_dim + n_experts + inter)
        layers = []; dd = feat
        for hh in head_hidden:
            layers += [nn.Linear(dd, hh), nn.GELU(), nn.LayerNorm(hh), nn.Dropout(p)]; dd = hh
        layers += [nn.Linear(dd, 1)]
        self.head = nn.Sequential(*layers)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # δ starts at 0

    def forward(self, means, hidden, gate_meta, conf, quant, parity_expert=None):
        """means (M,E), hidden (M,E,512), gate_meta (M,Dm), conf (M,E,Cc), quant (M,E,Q)."""
        M, E = means.shape
        cal = torch.stack([self.cal[e](means[:, e]) for e in range(E)], 1)   # (M,E) calibrated means
        if parity_expert is not None:
            return dict(final=cal[:, parity_expert], cal=cal, w=None)
        unc = conf[..., 0].clamp_min(1e-4)                         # (M,E) per-expert uncertainty
        prec = 1.0 / (unc * unc + 1e-8)
        w = prec / prec.sum(1, keepdim=True)                       # precision (inverse-variance) weights
        base = (w * cal).sum(1)                                    # (M,) principled anchor
        z = torch.stack([self.adapters[e](hidden[:, e]) for e in range(E)], 1)  # (M,E,d)
        parts = [z.reshape(M, -1), cal, quant.reshape(M, -1), conf.reshape(M, -1), gate_meta, unc]
        if E == 2:
            parts += [z[:, 0] * z[:, 1], z[:, 0] - z[:, 1], (cal[:, 0] - cal[:, 1]).abs().unsqueeze(-1)]
        delta = self.head(torch.cat(parts, -1)).squeeze(-1)        # (M,) un-throttled correction
        return dict(final=base + delta, cal=cal, w=w)

    def fusion_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ==========================================================================
# StackFusionCls — classification analog of StackFusionReg. A CONFIDENCE
# (inverse-entropy) weighted base over the calibrated LOGIT vectors + a DEEP
# stacking head over ALL per-row features -> an UN-throttled logit correction.
# Per-row (no cross-row/dataset pooling) so it stays valid in the vectorized
# cached DDP step. Escapes the softmax-convex-blend bottleneck of AttnFusion.
# n_experts-generic (works for 2 experts today, 3+ when a GBDT is added).
# ==========================================================================
class StackFusionCls(nn.Module):
    def __init__(self, n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3,
                 d_fuse=256, rank=128, head_hidden=(512, 512, 256), p=0.1):
        super().__init__()
        self.n_experts = n_experts; self.out_dim = out_dim
        self.adapters = nn.ModuleList([Adapter(d_fuse, rank, p) for _ in range(n_experts)])
        self.cal = nn.ModuleList([LogitCalibrator(out_dim) for _ in range(n_experts)])
        inter = (2 * d_fuse + out_dim) if n_experts == 2 else 0    # z0*z1 ⊕ z0-z1 ⊕ (cal0-cal1)
        feat = (n_experts * d_fuse + n_experts * out_dim + n_experts * conf_dim
                + gate_meta_dim + n_experts + inter)              # +n_experts = per-expert certainty
        layers = []; dd = feat
        for hh in head_hidden:
            layers += [nn.Linear(dd, hh), nn.GELU(), nn.LayerNorm(hh), nn.Dropout(p)]; dd = hh
        layers += [nn.Linear(dd, out_dim)]
        self.head = nn.Sequential(*layers)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # δ starts at 0
        self.resid_scale = nn.Parameter(torch.tensor(0.5))       # learnable: base dominates early, head grows when it helps

    def forward(self, logits, hidden, gate_meta, conf, parity_expert=None):
        """logits (M,E,K), hidden (M,E,512), gate_meta (M,Dm), conf (M,E,Cc)."""
        M, E, K = logits.shape
        cal = torch.stack([self.cal[e](logits[:, e]) for e in range(E)], 1)   # (M,E,K) calibrated logits
        if parity_expert is not None:
            return dict(final=cal[:, parity_expert], cal=cal, w=None)
        ent = conf[..., 1].clamp_min(1e-4)                        # (M,E) per-expert uncertainty (entropy)
        prec = 1.0 / (ent + 1e-4)
        w = prec / prec.sum(1, keepdim=True)                      # confidence (inverse-entropy) weights
        base = (w.unsqueeze(-1) * cal).sum(1)                     # (M,K) principled logit anchor
        z = torch.stack([self.adapters[e](hidden[:, e]) for e in range(E)], 1)  # (M,E,d)
        cert = conf[..., 0]                                       # (M,E) max-prob certainty
        parts = [z.reshape(M, -1), cal.reshape(M, E * K), conf.reshape(M, -1), gate_meta, cert]
        if E == 2:
            parts += [z[:, 0] * z[:, 1], z[:, 0] - z[:, 1], cal[:, 0] - cal[:, 1]]
        delta = self.resid_scale * self.head(torch.cat(parts, -1))    # (M,K) correction
        return dict(final=base + delta, cal=cal, w=w)

    def fusion_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
