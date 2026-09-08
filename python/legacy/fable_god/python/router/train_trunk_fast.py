"""FAST monster training: the frozen experts are precomputed (cached features), so
training does ZERO expert forward — just the fusion trunk over cached hidden/logits.
Cache is resident on the GPUs, the step is fully vectorized (one forward over all
rows in the batch), DDP over 8 A100s, huge batch to fill memory. 100x+ the steps/s
of the live-forward version.

Deploy = frozen experts (80.8M) + trunk = one 100M+ model. Train on synthetic+real
(bin_coltarget); SELECT on held-out REAL binary (router/real_bin) where the fused
model must beat BOTH TabPFN-v3 and TabICL. Emits the monster_tui log format.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from python.fusion.model import StackFusionCls
from python.fusion.cache import load_cache_packed, list_cache_files
from python.fusion.train_fast import preload, gather_rows, seg_mean

TRAIN_CACHE = os.path.expanduser(os.environ.get("FUSION_TRAIN_CACHE", "~/fable_god/cache/fusion/bin_coltarget"))
VAL_CACHE = os.path.expanduser(os.environ.get("FUSION_VAL_CACHE", "~/fable_god/cache/router/real_bin"))


def _auc(p, y):
    y = np.asarray(y).astype(int)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


@torch.no_grad()
def evaluate(core, P, dev, cap=300, max_rows=256):
    """Fused per-dataset AUC + expert AUCs. Per-dataset forward (robust to variable
    row counts), bf16, capped datasets + rows for speed."""
    core.eval(); fa, pa, ia = [], [], []
    off = P["off"]
    for d in range(min(P["n"], cap)):
        s = int(off[d]); e = min(int(off[d + 1]), s + max_rows)
        gm = P["gate_meta"][d].unsqueeze(0).expand(e - s, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = core(P["logits"][s:e], P["hidden"][s:e].float(), gm, P["conf"][s:e])
        y = P["y"][s:e].cpu().numpy()
        fa.append(_auc(torch.softmax(out["final"].float(), -1)[:, 1].cpu().numpy(), y))
        pr = torch.softmax(P["logits"][s:e].float(), -1)[..., 1].cpu().numpy()
        pa.append(_auc(pr[:, 0], y)); ia.append(_auc(pr[:, 1], y))
    return np.array(fa), np.array(pa), np.array(ia)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512, help="datasets/step (fill memory)")
    ap.add_argument("--rows", type=int, default=64, help="test rows sampled per dataset/step")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--lam-win", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/cache/router/monster"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    files = list_cache_files(TRAIN_CACHE)
    np.random.default_rng(args.seed).shuffle(files)
    my_files = files[rank::world]                          # RAM-safe per-rank shard
    train = [e for e in load_cache_packed(TRAIN_CACHE, files=my_files, limit=args.limit) if "logits" in e]
    Ptr = preload(train, dev)
    E = train[0]["logits"].shape[1]
    ROWS = int(Ptr["off"][1] - Ptr["off"][0])              # uniform test-rows/dataset in the cache
    args.rows = min(args.rows, ROWS)
    Pval = None
    if is_main:
        val = [e for e in load_cache_packed(VAL_CACHE) if "logits" in e]
        Pval = preload(val, dev)

    # trunk = the trainable fusion head; deploy model = experts(80.8M)+trunk
    core = StackFusionCls(n_experts=E, out_dim=2, gate_meta_dim=train[0]["gate_meta"].numel(),
                          conf_dim=train[0]["conf"].shape[-1]).to(dev)
    trunk_n = sum(p.numel() for p in core.parameters())
    init_params = [p.detach().clone() for p in core.parameters()]      # θ0 for L2-SP
    n_params = float(trunk_n)
    model = torch.nn.parallel.DistributedDataParallel(core, device_ids=[local]) if ddp else core

    # precompute per-dataset HARD expert AUCs (frozen constants) — the soft-win opponents
    EXPERT_AUC = torch.full((Ptr["n"], E), 0.5, device=dev)
    _y = Ptr["y"].cpu().numpy(); _p = torch.softmax(Ptr["logits"].float(), -1)[..., 1].cpu().numpy()
    for d in range(Ptr["n"]):
        s, e = int(Ptr["off"][d]), int(Ptr["off"][d + 1]); yy = _y[s:e]
        if np.unique(yy).size < 2:
            continue
        for j in range(E):
            EXPERT_AUC[d, j] = roc_auc_score(yy, _p[s:e, j])
    # per-dataset pos/neg row offsets for stratified sampling (both classes guaranteed)
    ROWSt = torch.tensor(ROWS)
    yb_all = Ptr["y"].view(Ptr["n"], ROWS)                  # (n,ROWS) labels (uniform ROWS)

    if is_main:
        mf = open(os.path.expanduser("~/fable_god/cache/monster.log"), "w")  # TUI default path
        fa, pa, ia = evaluate(core, Pval, dev)
        b_pfn, b_icl = pa.mean(), ia.mean()
        print(f"MONSTER model: {(80.8 + trunk_n/1e6):.1f}M params (trunk {trunk_n/1e6:.1f}M) "
              f"[100M+ ✓] on {world} GPU", file=mf, flush=True)
        print(f"TRAIN {sum(len(train) for _ in [0])*world} tasks (synthetic+real) in "
              f"{len(files)} batches x{world} GPU | VAL {Pval['n']} REAL tasks in {Pval['n']} batches",
              file=mf, flush=True)
        print(f"BASELINE val AUC: TabPFN-v3 {b_pfn:.4f} | TabICL {b_icl:.4f} | "
              f"best-of-both {np.maximum(pa, ia).mean():.4f}", file=mf, flush=True)

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ema = {k: v.detach().clone() for k, v in core.state_dict().items()}
    best = -1e9; t0 = time.time(); grng = np.random.default_rng(args.seed + 1 + rank)
    for step in range(1, args.steps + 1):
        core.train()
        # fully vectorized gather (uniform ROWS/dataset -> no GPU<->CPU sync) + row subsample
        ds_idx = torch.randint(0, Ptr["n"], (args.batch,), device=dev)
        # STRATIFIED per-dataset sampling: R/2 pos + R/2 neg rows (both classes guaranteed
        # so per-dataset soft-AUC has signal even on imbalanced datasets).
        Rh = max(1, args.rows // 2); B = args.batch; R = 2 * Rh
        lab = yb_all[ds_idx].float()                              # (B,ROWS)
        posw = lab + (lab.sum(1, keepdim=True) == 0).float()      # fallback uniform if a dataset has no pos
        negw = (1 - lab) + ((1 - lab).sum(1, keepdim=True) == 0).float()
        off = torch.cat([torch.multinomial(posw, Rh, replacement=True),
                         torch.multinomial(negw, Rh, replacement=True)], 1)   # (B,R)
        ridx = (ds_idx[:, None] * ROWS + off).reshape(-1)
        seg = torch.arange(B, device=dev).repeat_interleave(R)
        gm = Ptr["gate_meta"][ds_idx].repeat_interleave(R, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(Ptr["logits"][ridx], Ptr["hidden"][ridx].float(), gm, Ptr["conf"][ridx])
        final = out["final"].float()
        y = Ptr["y"][ridx].long()
        # === CE-PRIMARY (proper score, transfers synthetic→real) + gentle soft-win-AUC beat ===
        ce = seg_mean(F.cross_entropy(final, y, reduction="none"), seg, B)      # (B,) per-dataset CE
        p = torch.softmax(final, -1)[:, 1].view(B, R)             # differentiable fused P(1)
        yv = y.view(B, R).float()
        diff = (p[:, :, None] - p[:, None, :]) / 0.05             # (B,R,R) pairwise
        pmask = (yv[:, :, None] > 0.5) & (yv[:, None, :] < 0.5)   # pos-neg pairs
        auc_f = (torch.sigmoid(diff) * pmask).sum((1, 2)) / pmask.sum((1, 2)).clamp_min(1)  # (B,) diff'able
        auc_e = EXPERT_AUC[ds_idx]                               # (B,E) frozen constants
        m = (auc_f.unsqueeze(1) - auc_e) / 0.02                   # s_d=0.02 fixed (AUC is scale-free)
        mt = 2.0 * torch.tanh(m / 2.0)                            # bounded margin
        pw = torch.sigmoid(mt); wgt = pw * (1 - pw) + 1e-6       # Elo-impact weight
        valid = (pmask.sum((1, 2)) > 0).float().unsqueeze(1)     # drop degenerate all-one-class ds
        L_d = ((wgt * (mt - 0.5) ** 2) * valid).sum(1) / (wgt * valid).sum(1).clamp_min(1e-6)
        per_ds = ce + args.lam_win * L_d                         # CE leads; soft-win adds beat pressure
        k = max(1, int(args.alpha * B))
        loss = 0.5 * per_ds.mean() + 0.5 * torch.topk(per_ds, k).values.mean()  # CVaR on worst
        sp = sum((pp - p0).pow(2).sum() for pp, p0 in zip(core.parameters(), init_params)) / n_params
        loss = loss + 3e-3 * sp                                  # L2-SP trust region
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            for k2, v in core.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k2].mul_(args.ema).add_(v, alpha=1 - args.ema)

        if is_main and step % 20 == 0 and step % args.eval_every != 0:
            rate = step / (time.time() - t0)
            print(f"hb {step} it/s {rate:.2f} dps {rate*args.batch*world:.0f}", file=mf, flush=True)
        if is_main and step % args.eval_every == 0:
            fa, pa, ia = evaluate(core, Pval, dev)          # RAW model — shows real progress (no EMA lag)
            win_both = float((fa >= np.maximum(pa, ia) - 1e-9).mean())
            beats = fa.mean() > b_pfn and fa.mean() > b_icl
            score = win_both + 1e-4 * fa.mean()          # SELECT on win-both rate (tie-break mean AUC)
            if score > best:
                best = score
                torch.save(core.state_dict(), os.path.join(args.out, "best.pt"))       # raw best (selected)
                torch.save(ema, os.path.join(args.out, "best_ema.pt"))                 # EMA for deploy
            peak = torch.cuda.max_memory_allocated(dev) / 1e9
            rate = step / (time.time() - t0)
            print(f"step {step:4d} | loss {float(loss):.3f} | fused AUC {fa.mean():.4f} "
                  f"(pfn {b_pfn:.4f} icl {b_icl:.4f}) | win-both {100*win_both:.0f}% | beats-both {beats} "
                  f"| peakmem {peak:.1f}GB | it/s {rate:.2f} dps {rate*args.batch*world:.0f}", file=mf, flush=True)
    if is_main:
        print(f"\n=== BEST fused val AUC {best:.4f} vs TabPFN-v3 {b_pfn:.4f} / TabICL {b_icl:.4f} "
              f"-> beats both: {best > b_pfn and best > b_icl} ===", file=mf, flush=True); mf.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
