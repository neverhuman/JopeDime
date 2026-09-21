"""Candidate evaluation for evolution: CRN per-episode margins, paired
statistics, the Wilcoxon promotion test, and the masked-gradient arm.

The CRN unit is the per-episode mean bounded margin (fitness_v2 semantics):
identical episodes (and zoo fits) are used for every candidate and the
incumbent, so candidate−incumbent deltas cancel episode difficulty exactly.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .episodes import Episode
from .eval_protocol import ProtocolRunner
from .fitness_v2 import M_BOUND, M_STAR, TAU, episode_scores, model_rmse


def candidate_margins(runner: ProtocolRunner, episodes: list[Episode]) -> list[float]:
    """Per-episode mean bounded margin, in episode order (CRN identity)."""
    out = []
    for ep in episodes:
        rmse = model_rmse(runner, ep)
        _, margin, _ = episode_scores(rmse, ep)
        out.append(margin if math.isfinite(margin) else float("nan"))
    return out


def paired_stats(cand: list[float], incumbent: list[float]) -> dict:
    a = np.asarray(cand, dtype=np.float64)
    b = np.asarray(incumbent, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) < 8:
        return {"n": int(len(d)), "delta": float("nan"), "se": float("nan"), "wilcoxon_p": None}
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    try:
        from scipy.stats import wilcoxon

        nonzero = d[d != 0]
        p = float(wilcoxon(nonzero, alternative="greater").pvalue) if len(nonzero) >= 8 else 1.0
    except Exception:
        z = d.mean() / max(se, 1e-12)
        p = float(0.5 * math.erfc(z / math.sqrt(2)))
    return {"n": int(len(d)), "delta": float(d.mean()), "se": se, "wilcoxon_p": p}


# ---- masked-gradient arm ---------------------------------------------------


def _model_rmse_torch(model, ep: Episode, device, centres) -> torch.Tensor:
    """Grad-enabled port of ProtocolRunner._mean_single (single forward:
    evolution episodes are <=956 rows)."""
    cy = ep.ctx_y.astype(np.float32)
    mu, sd = float(cy.mean()), float(cy.std()) + 1e-8
    cx = torch.from_numpy(np.ascontiguousarray(ep.ctx_x)).to(device)
    qx = torch.from_numpy(np.ascontiguousarray(ep.qry_x)).to(device)
    cyz = torch.from_numpy((cy - mu) / sd).to(device)
    x = torch.cat([cx, qx], 0).unsqueeze(1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x, cyz.unsqueeze(1))
    pred_z = (torch.softmax(logits.float(), -1) @ centres).reshape(-1)
    pred = pred_z * sd + mu
    target = torch.from_numpy(ep.qry_y.astype(np.float32)).to(device)
    return torch.sqrt(((pred - target) ** 2).mean() + 1e-12)


def _ipo_loss_torch(rmse_model: torch.Tensor, ep: Episode, device) -> torch.Tensor:
    opp = np.array([v for v in ep.zoo_rmse.values() if np.isfinite(v)])
    s_e = float(np.median(opp))
    if s_e <= 1e-12:
        return rmse_model * 0.0
    opp_t = torch.from_numpy(opp.astype(np.float32)).to(device)
    m = (opp_t - rmse_model) / s_e
    m_t = M_BOUND * torch.tanh(m / M_BOUND)
    p = torch.sigmoid(m_t / TAU)
    w = (p * (1 - p)).detach()  # Elo-impact weights, frozen per step
    if float(w.sum()) <= 1e-9:
        return rmse_model * 0.0
    return (w * (m_t - M_STAR) ** 2).sum() / w.sum()


def grad_arm_delta(
    model,
    engine,
    mask_blocks: np.ndarray,
    inner_episodes: list[Episode],
    device,
    k_steps: int = 8,
    batch: int = 16,
    lr: float = 3e-5,
) -> dict:
    """Freeze 97%, AdamW the masked 3% on fresh disposable episodes; return
    the sparse delta and restore the model bit-exactly."""

    table = engine.table
    touched: dict[str, list] = {}
    for bid in mask_blocks.tolist():
        blk = table.blocks[bid]
        touched.setdefault(blk.tensor, []).append(blk)
    full_snapshot = {name: engine._flat[name].clone() for name in touched}

    centres = model.regression_borders.float()
    centres = (centres[:-1] + (centres[1:] - centres[:-1]) / 2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    masks = {}
    for name, blks in touched.items():
        m = torch.zeros_like(engine._flat[name], dtype=torch.bool)
        for b in blks:
            m[b.start : b.end] = True
        masks[name] = m

    named = dict(model.named_parameters())
    model_was_training = model.training
    model.train(False)
    for step in range(k_steps):
        eps = inner_episodes[step * batch : (step + 1) * batch]
        if not eps:
            break
        opt.zero_grad(set_to_none=True)
        # backward PER EPISODE: grads accumulate, graphs free immediately —
        # summing losses first retains all 16 graphs at once (~>20GB, OOM).
        for ep in eps:
            loss = _ipo_loss_torch(_model_rmse_torch(model, ep, device, centres), ep, device) / len(eps)
            loss.backward()
        with torch.no_grad():
            for name, p in named.items():
                if p.grad is None:
                    continue
                flat_grad = p.grad.view(-1)
                if name in masks:
                    flat_grad.mul_(masks[name].float())
                else:
                    flat_grad.zero_()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    sparse: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for name, blks in touched.items():
            flat = engine._flat[name]
            idx_parts, val_parts = [], []
            for b in blks:
                diff = flat[b.start : b.end] - full_snapshot[name][b.start : b.end]
                idx_parts.append(np.arange(b.start, b.end, dtype=np.int64))
                val_parts.append(diff.cpu().numpy())
            sparse[name] = (np.concatenate(idx_parts), np.concatenate(val_parts))
        # bit-exact restore (AdamW with zeroed grads + wd=0 cannot move
        # untouched tensors: m=v=0 ⇒ update 0)
        for name in touched:
            engine._flat[name].copy_(full_snapshot[name])
    model.train(model_was_training)
    return sparse
