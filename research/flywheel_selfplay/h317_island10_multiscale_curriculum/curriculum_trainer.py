#!/usr/bin/env python3
"""H317 neural trainer: optimization recipes, schedules, SAM, Muon, Lion, EMA/SWA, curriculum annealing, drift logging.

Recipes (plan §2 H0.3, plus the coordinator's Opt1+ control):
  opt1   AdamW(lr 8e-4, wd 2e-3) + CosineAnnealingLR (eta_min = 1% lr). Fixed ranking temperature 0.10, no KL, G = 1 view.
  opt1p  "Opt1+": AdamW(8e-4, 2e-3) + WSD + SWA + curriculum - the shared scaffold of opt2-4 without SAM / Muon / Lion,
         so opt2 - opt1p isolates SAM, opt3 - opt1p isolates Muon, opt4 - opt1p isolates Lion + SAM.
  opt2   AdamW(8e-4, 2e-3) + SAM(rho 0.05) + WSD + SWA + curriculum.
  opt3   Muon (momentum 0.95 Nesterov, 5 Newton-Schulz steps, Moonlight RMS-matched scaling 0.2*sqrt(max(m,n)) so
         AdamW's lr 8e-4 / wd 2e-3 transfer) on the 2-D hidden matrices of the transformer blocks + AdamW(8e-4, 2e-3) on
         embeddings/input projections/heads/norms/biases + WSD + SWA + curriculum. The original Keller scaling at lr 0.02
         without weight decay drifted to 1.7-1.9x the init norm in 300 permuted-target steps (6.6x at 3x lr), so it is
         kept only as `Recipe("opt3k", ..., muon_lr=0.02, muon_wd=0, muon_scaling="keller")` for reference.
  opt4   Lion(lr 1.5e-4 = AdamW/5.3, wd 1e-2, betas 0.9/0.99) + SAM(0.05) + WSD + SWA + curriculum.
  Weight averaging is SHARED by opt1p-opt4 (SWA: uniform mean of 10 snapshots over the WSD decay phase; the plan's
  "EMA/SWA" for opt2 is resolved to SWA so the contrasts stay clean; `avg="ema"` remains available).
  Curriculum: ranking temperature 0.25 -> 0.05 (geometric), multi-view KL weight 0.05 -> 0.25 (linear); G = 2 views/step.
WSD = 5% linear warmup from 1% lr, stable to 80% of steps, linear decay to 1% lr. Every recipe clips the global grad norm
at 1.0. Budgets are in optimizer steps (`TrainSpec.steps`) so arms are compute-matched; every run logs the parameter drift
||theta_final - theta_init||_2 of the raw and of the evaluated (averaged) weights.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

import multiscale_axial_bridge as MB


@dataclass
class Recipe:
    name: str
    opt: str  # adamw | muon | lion
    lr: float
    wd: float
    sched: str  # cosine | wsd
    sam_rho: float = 0.0
    avg: str = "none"  # none | ema | swa
    ema_decay: float = 0.995
    curriculum: bool = False
    tau_fixed: float = 0.10
    muon_lr: float = 8e-4
    muon_wd: float = 2e-3
    muon_scaling: str = "moonlight"  # moonlight: update RMS matched to AdamW (0.2*sqrt(max(m,n))); keller: sqrt(max(1,m/n))
    grad_clip: float = 1.0


SHARED_AVG = "swa"
RECIPES: Dict[str, Recipe] = {
    "opt1": Recipe("opt1", "adamw", 8e-4, 2e-3, "cosine"),
    "opt1p": Recipe(
        "opt1p", "adamw", 8e-4, 2e-3, "wsd", avg=SHARED_AVG, curriculum=True
    ),
    "opt2": Recipe(
        "opt2",
        "adamw",
        8e-4,
        2e-3,
        "wsd",
        sam_rho=0.05,
        avg=SHARED_AVG,
        curriculum=True,
    ),
    "opt3": Recipe("opt3", "muon", 8e-4, 2e-3, "wsd", avg=SHARED_AVG, curriculum=True),
    "opt3k": Recipe(
        "opt3k",
        "muon",
        8e-4,
        2e-3,
        "wsd",
        avg=SHARED_AVG,
        curriculum=True,
        muon_lr=0.02,
        muon_wd=0.0,
        muon_scaling="keller",
    ),
    "opt4": Recipe(
        "opt4",
        "lion",
        1.5e-4,
        1e-2,
        "wsd",
        sam_rho=0.05,
        avg=SHARED_AVG,
        curriculum=True,
    ),
}


# ----------------------------------------------------------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------------------------------------------------------
class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): p -= lr * (sign(b1*m + (1-b1)*g) + wd*p); m = b2*m + (1-b2)*g."""

    def __init__(self, params, lr=1.5e-4, betas=(0.9, 0.99), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                upd = (b1 * m + (1 - b1) * p.grad).sign_()
                p.mul_(1 - g["lr"] * g["weight_decay"]).add_(upd, alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


def newton_schulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization (Keller Jordan's Muon coefficients), fp32."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float() / (g.norm() + eps)
    tr = x.shape[0] > x.shape[1]
    if tr:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return x.T if tr else x


class Muon(torch.optim.Optimizer):
    """Muon for 2-D matrices: Nesterov momentum + Newton-Schulz orthogonalized update.

    scaling "moonlight" (Liu et al. 2025, "Muon is Scalable for LLM Training"): update * 0.2 * sqrt(max(rows, cols)) so
    its RMS matches AdamW's and AdamW's lr / decoupled weight decay transfer (H317 default: lr 8e-4, wd 2e-3).
    scaling "keller" (the original): update * sqrt(max(1, rows/cols)), lr ~0.02, no weight decay."""

    def __init__(
        self,
        params,
        lr=8e-4,
        momentum=0.95,
        ns_steps=5,
        weight_decay=2e-3,
        scaling="moonlight",
    ):
        if scaling not in ("moonlight", "keller"):
            raise ValueError(scaling)
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
                scaling=scaling,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only takes 2-D matrices")
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"])
                o = newton_schulz5(upd, g["ns_steps"]).to(p.dtype)
                if g["scaling"] == "moonlight":
                    o = o * (0.2 * math.sqrt(max(p.shape[0], p.shape[1])))
                else:
                    o = o * math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"])


def build_optimizers(
    model: MB.MultiScaleAxialResidualBridge, r: Recipe, lr_mult: float = 1.0
) -> List[torch.optim.Optimizer]:
    params = [p for p in model.parameters() if p.requires_grad]
    if r.opt == "adamw":
        return [
            torch.optim.AdamW(
                params, lr=r.lr * lr_mult, weight_decay=r.wd, betas=(0.9, 0.999)
            )
        ]
    if r.opt == "lion":
        return [Lion(params, lr=r.lr * lr_mult, weight_decay=r.wd)]
    if r.opt == "muon":
        muon, other = model.param_groups_muon()
        return [
            Muon(
                muon,
                lr=r.muon_lr * lr_mult,
                weight_decay=r.muon_wd,
                scaling=r.muon_scaling,
            ),
            torch.optim.AdamW(other, lr=r.lr * lr_mult, weight_decay=r.wd),
        ]
    raise ValueError(r.opt)


@torch.no_grad()
def param_drift(model: nn.Module, init: List[torch.Tensor]) -> dict:
    d = torch.sqrt(
        sum(
            ((p.float() - q.float()) ** 2).sum()
            for p, q in zip(model.parameters(), init)
        )
    )
    n0 = torch.sqrt(sum((q.float() ** 2).sum() for q in init))
    return {"l2": float(d), "rel": float(d / n0.clamp_min(1e-12))}


def lr_factor(r: Recipe, step: int, total: int) -> float:
    t = step / max(1, total)
    if r.sched == "cosine":
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * t))
    if t < 0.05:
        return 0.01 + 0.99 * t / 0.05
    if t < 0.80:
        return 1.0
    return 0.01 + 0.99 * (1.0 - (t - 0.80) / 0.20)


def curriculum(r: Recipe, step: int, total: int) -> tuple[float, float]:
    """(ranking temperature, multi-view KL weight)."""
    if not r.curriculum:
        return r.tau_fixed, 0.0
    t = step / max(1, total - 1)
    return 0.25 * (0.05 / 0.25) ** t, 0.05 + 0.20 * t


def swa_snapshot_steps(total: int, n: int = 10) -> set:
    start = int(0.80 * total)
    return {
        min(total - 1, start + int(round(k * (total - 1 - start) / max(1, n - 1))))
        for k in range(n)
    }


class WeightAverager:
    def __init__(self, model: nn.Module, mode: str, decay: float):
        self.mode, self.decay, self.n = mode, decay, 0
        self.avg = copy.deepcopy(model).eval() if mode != "none" else None
        if self.avg is not None:
            for p in self.avg.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, take_swa: bool) -> None:
        if self.mode == "ema":
            for a, p in zip(self.avg.parameters(), model.parameters()):
                a.lerp_(p, 1.0 - self.decay)
        elif self.mode == "swa" and take_swa:
            self.n += 1
            for a, p in zip(self.avg.parameters(), model.parameters()):
                if self.n == 1:
                    a.copy_(p)
                else:
                    a.add_((p - a) / self.n)
        if self.avg is not None:
            for a, b in zip(self.avg.buffers(), model.buffers()):
                a.copy_(b)

    def final(self, model: nn.Module) -> nn.Module:
        if self.avg is None or (self.mode == "swa" and self.n == 0):
            return model
        return self.avg


def _clip(model: nn.Module, max_norm: float) -> None:
    if max_norm and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def sam_step(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    opts: List[torch.optim.Optimizer],
    rho: float,
    grad_clip: float,
) -> float:
    """SAM (Foret et al. 2021): ascend to w + rho*g/||g||, take the gradient there, step from w."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_fn()
    loss.backward()
    grads = [p.grad for p in params]
    norm = torch.norm(torch.stack([g.norm() for g in grads if g is not None]))
    scale = rho / (norm + 1e-12)
    eps = []
    with torch.no_grad():
        for p in params:
            e = p.grad * scale if p.grad is not None else None
            if e is not None:
                p.add_(e)
            eps.append(e)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss_fn().backward()
    with torch.no_grad():
        for p, e in zip(params, eps):
            if e is not None:
                p.sub_(e)
    _clip(model, grad_clip)
    for o in opts:
        o.step()
    return float(loss.detach())


def plain_step(model: nn.Module, loss_fn, opts, grad_clip: float) -> float:
    loss = loss_fn()
    loss.backward()
    _clip(model, grad_clip)
    for o in opts:
        o.step()
    return float(loss.detach())


# ----------------------------------------------------------------------------------------------------------------------
# GPU-resident training data and slate sampling
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class TrainTensors:
    """All tensors on one device; index 0..N-1 = contexts in `cids` order."""

    cids: List[str]
    cand: torch.Tensor  # [N, Mmax, 452] float16
    n_valid: torch.Tensor  # [N] long
    u: (
        torch.Tensor
    )  # [N, Mmax] float32 (invalid slots = -inf-like -1e9 so they sort last)
    lr: torch.Tensor
    lr_mask: torch.Tensor
    sd: torch.Tensor
    sd_mask: torch.Tensor
    gate: torch.Tensor
    views: torch.Tensor  # [N, 32, 997] float32 (view 0 = exact float32 encodings)
    order: Optional[torch.Tensor] = None  # [N, Mmax] candidate indices by u desc

    def finalize(self) -> "TrainTensors":
        self.order = torch.argsort(self.u, dim=1, descending=True, stable=True)
        return self


def sample_slate(
    tt: TrainTensors, idx: torch.Tensor, slate: int, top_k: int, gen: torch.Generator
) -> torch.Tensor:
    """[Bc, slate] candidate indices: the top_k by utility + uniform random others among the valid candidates."""
    Bc = idx.shape[0]
    Mmax = tt.order.shape[1]
    order = tt.order[idx]
    pos = torch.arange(Mmax, device=idx.device).unsqueeze(0)
    keys = torch.rand(Bc, Mmax, device=idx.device, generator=gen)
    keys = torch.where(
        (pos >= top_k) & (pos < tt.n_valid[idx].unsqueeze(1)),
        keys,
        torch.full_like(keys, -1.0),
    )
    rest = torch.topk(keys, slate - top_k, dim=1).indices
    picked = torch.cat([pos[:, :top_k].expand(Bc, top_k), rest], dim=1)
    return order.gather(1, picked)


def gather_batch(
    tt: TrainTensors, idx: torch.Tensor, slate_idx: torch.Tensor, view_idx: torch.Tensor
):
    ctx = tt.views[idx.unsqueeze(1), view_idx]  # [Bc, G, 997]
    cand = tt.cand[idx.unsqueeze(1), slate_idx].float()  # [Bc, S, 452]
    tgt = {
        k: getattr(tt, k)[idx.unsqueeze(1), slate_idx].float()
        for k in ("u", "lr", "lr_mask", "sd", "sd_mask", "gate")
    }
    return ctx, cand, tgt


@dataclass
class TrainSpec:
    scale: str
    recipe: str
    view_subset: List[int]
    mixup: bool = False
    epochs: float = 1.0
    batch_contexts: int = 16
    slate: int = 128
    top_k: int = 48
    view_isolate_p: float = (
        0.5  # L only: probability a 2-view step runs with view-isolated attention
    )
    steps: Optional[int] = (
        None  # optimizer steps (overrides epochs; the compute-matched budget unit)
    )
    lr_mult: float = (
        1.0  # stability checks only (permuted targets); 1.0 in every real arm
    )

    def views_per_step(self) -> int:
        return (
            2
            if (self.mixup or RECIPES[self.recipe].curriculum)
            and len(self.view_subset) >= 2
            else 1
        )

    def to_dict(self) -> dict:
        return asdict(self)


def make_step_closure(model, ctx, cand, tgt, lam, iso, tau, kl_w, pairs, device):
    """Loss closure for one optimizer step with ALL randomness fixed: the ranking pairs are drawn once (passed in) and
    the global torch RNG state (DropPath masks) is snapshotted here and restored before every evaluation, so SAM's two
    loss evaluations see identical draws."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    def loss_fn():
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        pred = model(ctx, cand, mix_lambda=lam, view_isolated=iso)
        return MB.ranking_losses(pred, tgt, tau, kl_w, pairs=pairs)["total"]

    return loss_fn


@torch.no_grad()
def fit_normalizer(
    model, tt: TrainTensors, tr: torch.Tensor, vs: torch.Tensor, chunk: int = 64
) -> None:
    """Feature-only z-score statistics (float64 sums, chunked) over the training contexts' allowed views and their
    valid candidates. No target is read."""
    acc = {"ctx": [0.0, 0.0, 0], "cand": [0.0, 0.0, 0]}
    for a in range(0, len(tr), chunk):
        idx = tr[a : a + chunk]
        x = tt.views[idx][:, vs].reshape(-1, MB.CTX_DIM).double()
        acc["ctx"][0] = acc["ctx"][0] + x.sum(0)
        acc["ctx"][1] = acc["ctx"][1] + (x * x).sum(0)
        acc["ctx"][2] += x.shape[0]
        valid = torch.arange(tt.cand.shape[1], device=tr.device).unsqueeze(
            0
        ) < tt.n_valid[idx].unsqueeze(1)
        c = tt.cand[idx][valid].double()
        acc["cand"][0] = acc["cand"][0] + c.sum(0)
        acc["cand"][1] = acc["cand"][1] + (c * c).sum(0)
        acc["cand"][2] += c.shape[0]
    for key, mu, sd in (
        ("ctx", model.normalizer.ctx_mu, model.normalizer.ctx_sd),
        ("cand", model.normalizer.cand_mu, model.normalizer.cand_sd),
    ):
        s1, s2, n = acc[key]
        m = s1 / n
        v = (s2 / n - m * m).clamp_min(0.0).sqrt()
        mu.copy_(m.float())
        sd.copy_(torch.where(v > 1e-6, v, torch.ones_like(v)).float())


def train_one(
    tt: TrainTensors,
    train_idx: np.ndarray,
    spec: TrainSpec,
    seed: int,
    device: torch.device,
    max_steps: Optional[int] = None,
    log_every: int = 0,
) -> tuple[nn.Module, dict]:
    """Train one model on contexts `train_idx` (indices into tt). Returns (eval model, stats). Feature normalizer
    statistics come from the training contexts' allowed views and candidates only (no targets)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    r = RECIPES[spec.recipe]
    model = MB.build_model(spec.scale).to(device)
    tr = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    vs = torch.as_tensor(spec.view_subset, device=device, dtype=torch.long)
    fit_normalizer(model, tt, tr, vs)
    init = [p.detach().clone() for p in model.parameters()]
    opts = build_optimizers(model, r, spec.lr_mult)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    steps_per_epoch = int(math.ceil(len(train_idx) / spec.batch_contexts))
    total = (
        int(spec.steps)
        if spec.steps
        else max(1, int(round(spec.epochs * steps_per_epoch)))
    )
    if max_steps:
        total = min(total, max_steps)
    avg = WeightAverager(model, r.avg, r.ema_decay)
    swa_steps = swa_snapshot_steps(total) if r.avg == "swa" else set()
    G = spec.views_per_step()
    model.train()
    perm = np.array([], dtype=np.int64)
    losses = []
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(total):
        if len(perm) < spec.batch_contexts:
            perm = np.concatenate([perm, rng.permutation(train_idx)])
        bidx, perm = perm[: spec.batch_contexts], perm[spec.batch_contexts :]
        idx = torch.as_tensor(bidx, device=device, dtype=torch.long)
        f = lr_factor(r, step, total)
        for o, lrs in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, lrs):
                g["lr"] = b * f
        tau, kl_w = curriculum(r, step, total)
        slate_idx = sample_slate(tt, idx, spec.slate, spec.top_k, gen)
        if G == 1:
            vsel = vs[
                torch.randint(0, len(vs), (len(bidx), 1), device=device, generator=gen)
            ]
        else:
            vsel = vs[
                torch.rand(len(bidx), len(vs), device=device, generator=gen).argsort(
                    dim=1
                )[:, :2]
            ]
        ctx, cand, tgt = gather_batch(tt, idx, slate_idx, vsel)
        lam = None
        if spec.mixup and G == 2:
            lam = torch.as_tensor(
                rng.beta(0.4, 0.4, size=len(bidx)), device=device, dtype=torch.float32
            )
        iso = spec.scale == "L" and G > 1 and rng.random() < spec.view_isolate_p
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss_fn = make_step_closure(
            model,
            ctx,
            cand,
            tgt,
            lam,
            iso,
            tau,
            kl_w,
            MB.draw_pairs(len(bidx), spec.slate, device=device, generator=gen),
            device,
        )

        if r.sam_rho > 0:
            loss = sam_step(model, loss_fn, opts, r.sam_rho, r.grad_clip)
        else:
            loss = plain_step(model, loss_fn, opts, r.grad_clip)
        avg.update(model, step in swa_steps)
        losses.append(loss)
        if log_every and (step + 1) % log_every == 0:
            print(
                {
                    "step": step + 1,
                    "of": total,
                    "loss": round(float(np.mean(losses[-log_every:])), 4),
                },
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    out = avg.final(model)
    out.eval()
    stats = {
        "steps": total,
        "steps_per_epoch": steps_per_epoch,
        "epoch_equivalents": total / steps_per_epoch,
        "lr_mult": spec.lr_mult,
        "drift_raw": param_drift(model, init),
        "drift_eval": param_drift(out, init),
        "views_per_step": G,
        "train_seconds": elapsed,
        "sec_per_step": elapsed / total,
        "peak_mem_mb": (torch.cuda.max_memory_allocated(device) / 2**20)
        if device.type == "cuda"
        else None,
        "params": MB.param_count(out),
        "avg_mode": r.avg,
        "swa_snapshots": avg.n if r.avg == "swa" else None,
        "loss_first": float(losses[0]),
        "loss_last": float(np.mean(losses[-max(1, total // 10) :])),
        "finite": bool(np.all(np.isfinite(losses))),
    }
    return out, stats


@torch.no_grad()
def predict_contexts(
    model: nn.Module, tt: TrainTensors, idx: np.ndarray, device: torch.device
) -> List[np.ndarray]:
    """View-0 predictions [n_valid, 4] float32 for every candidate of each context in idx (one forward per context)."""
    model.eval()
    res = []
    for i in idx:
        n = int(tt.n_valid[i])
        ctx = tt.views[i, 0].view(1, 1, MB.CTX_DIM)
        cand = tt.cand[i, :n].float().unsqueeze(0)
        res.append(model(ctx, cand)[0, 0].float().cpu().numpy())
    return res
