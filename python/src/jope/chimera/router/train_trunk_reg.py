"""FAST regression fusion training: frozen regressor features are precomputed (cached),
so training does ZERO expert forward. Trains only the AttnFusionReg trunk over cached
means/hidden/quant to produce a single scalar prediction. Cache resident on the GPUs,
vectorized step, bf16, DDP over 8 A100s.

Metric = R² (1 - normalized MSE), higher is better — mirrors the classifier's AUC, so
'beats both experts' and the TUI work unchanged. Deploy = frozen experts + trunk.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from python.fusion.cache import list_cache_files, load_cache_packed
from python.fusion.model import StackFusionReg

REG_CACHE = os.path.expanduser("~/fable_god/cache/fusion/reg_packed")


def preload(entries, dev):
    ss = [e["y"].shape[0] for e in entries]
    off = np.concatenate([[0], np.cumsum(ss)]).astype(np.int64)
    P = dict(means=torch.cat([e["means"] for e in entries]).float().to(dev),
             hidden=torch.cat([e["hidden"] for e in entries]).half().to(dev),
             quant=torch.cat([e["quant"] for e in entries]).float().to(dev),
             conf=torch.cat([e["conf"] for e in entries]).float().to(dev),
             y=torch.cat([e["y"] for e in entries]).float().to(dev),
             gate_meta=torch.stack([e["gate_meta"] for e in entries]).to(dev),
             off=torch.tensor(off, device=dev), n=len(entries))
    P["row_ds"] = torch.repeat_interleave(torch.arange(len(entries), device=dev),
                                          torch.tensor(ss, device=dev))
    return P


def seg_mean(v, seg, n):
    o = torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, seg, v)
    c = torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, seg, torch.ones_like(v))
    return o / c.clamp_min(1)


@torch.no_grad()
def evaluate(core, P, dev, cap=400):
    """Per-dataset R² of the fused model + of each raw expert (from cached means)."""
    core.eval(); fr, pr, ir = [], [], []
    for d in range(min(P["n"], cap)):
        s, e = int(P["off"][d]), int(P["off"][d + 1])
        gm = P["gate_meta"][d].unsqueeze(0).expand(e - s, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = core(P["means"][s:e], P["hidden"][s:e].float(), gm, P["conf"][s:e], P["quant"][s:e])
        y = P["y"][s:e]; var = y.var().clamp_min(1e-8)
        r2 = lambda p: float(1 - ((p - y) ** 2).mean() / var)
        fr.append(r2(out["final"].float())); pr.append(r2(P["means"][s:e, 0])); ir.append(r2(P["means"][s:e, 1]))
    return np.array(fr), np.array(pr), np.array(ir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--rows", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam-win", type=float, default=0.3); ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=512); ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/cache/router/monster_reg"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}"); is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    files = list_cache_files(REG_CACHE)
    tr_files = [f for f in files if "/train_" in f]; va_files = [f for f in files if "/val_" in f]
    np.random.default_rng(args.seed).shuffle(tr_files)
    train = [e for e in load_cache_packed(REG_CACHE, files=tr_files[rank::world]) if "means" in e]
    Ptr = preload(train, dev)
    ROWS = int(Ptr["off"][1] - Ptr["off"][0]); args.rows = min(args.rows, ROWS)
    Q = train[0]["quant"].shape[-1]
    core = StackFusionReg(n_experts=2, gate_meta_dim=train[0]["gate_meta"].numel(),
                          conf_dim=train[0]["conf"].shape[-1], quant_dim=Q).to(dev)
    trunk_n = sum(p.numel() for p in core.parameters())
    init_params = [p.detach().clone() for p in core.parameters()]        # θ0 for L2-SP
    n_params = float(trunk_n)
    model = torch.nn.parallel.DistributedDataParallel(core, device_ids=[local]) if ddp else core

    Pval = None
    if is_main:
        val = [e for e in load_cache_packed(REG_CACHE, files=va_files) if "means" in e]
        Pval = preload(val, dev)
        mf = open(os.path.expanduser("~/fable_god/cache/monster_reg.log"), "w")
        fr, pr, ir = evaluate(core, Pval, dev)
        b_pfn, b_icl = pr.mean(), ir.mean()
        print(f"MONSTER model: {(80.8 + trunk_n/1e6):.1f}M params (trunk {trunk_n/1e6:.1f}M) [100M+ ✓] on {world} GPU", file=mf, flush=True)
        print(f"TRAIN {len(tr_files)} shard-files x{world} GPU | VAL {Pval['n']} REAL tasks in {Pval['n']} batches", file=mf, flush=True)
        print(f"BASELINE val AUC: TabPFN-v3 {b_pfn:.4f} | TabICL {b_icl:.4f} | best-of-both {np.maximum(pr, ir).mean():.4f}", file=mf, flush=True)

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ema = {k: v.detach().clone() for k, v in core.state_dict().items()}
    best = -1e9; best_r2 = best_win = 0.0; t0 = time.time()
    for step in range(1, args.steps + 1):
        core.train()
        ds_idx = torch.randint(0, Ptr["n"], (args.batch,), device=dev)
        off = torch.randint(0, ROWS, (args.batch, args.rows), device=dev)
        ridx = (ds_idx[:, None] * ROWS + off).flatten()
        seg = torch.arange(args.batch, device=dev).repeat_interleave(args.rows)
        gm = Ptr["gate_meta"][ds_idx].repeat_interleave(args.rows, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(Ptr["means"][ridx], Ptr["hidden"][ridx].float(), gm, Ptr["conf"][ridx], Ptr["quant"][ridx])
        y = Ptr["y"][ridx].float(); final = out["final"].float()
        # === IPO soft-win loss: chase BEATING both experts (proven anti-plateau) ===
        rmse_f = torch.sqrt(seg_mean((final - y) ** 2, seg, args.batch) + 1e-12)          # differentiable
        rmse_e = torch.stack([torch.sqrt(seg_mean((Ptr["means"][ridx][:, e].float() - y) ** 2,
                              seg, args.batch) + 1e-12) for e in range(2)], 1).detach()   # (B,2) constants
        s_d = rmse_e.min(1).values.clamp_min(1e-6)                                        # ← STRONGER expert
        m = (rmse_e - rmse_f.unsqueeze(1)) / s_d.unsqueeze(1)          # (B,2)  >0 = fused beats expert j
        mt = 2.0 * torch.tanh(m / 2.0)                                 # bounded margin M·tanh
        pw = torch.sigmoid(mt); w = pw * (1 - pw) + 1e-6              # Elo-impact weight (redirects to hard side)
        L_d = (w * (mt - 0.5) ** 2).sum(1) / w.sum(1)                  # IPO target m*=0.5, per dataset
        k = max(1, int(args.alpha * len(L_d)))
        loss = 0.75 * L_d.mean() + 0.25 * torch.topk(L_d, k).values.mean()       # group-CVaR on worst
        var = (seg_mean(y * y, seg, args.batch) - seg_mean(y, seg, args.batch) ** 2).clamp_min(1e-6)
        nmse = (seg_mean((final - y) ** 2, seg, args.batch) / var).mean()
        lam_nmse = 0.3 * max(0.1, 1 - step / args.steps)              # scheduled proper-score anchor
        sp = sum((p - p0).pow(2).sum() for p, p0 in zip(core.parameters(), init_params)) / n_params
        loss = loss + lam_nmse * nmse + 3e-3 * sp                     # + L2-SP trust region
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            for kk, v in core.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[kk].mul_(args.ema).add_(v, alpha=1 - args.ema)

        if is_main and step % 20 == 0 and step % args.eval_every != 0:
            rate = step / (time.time() - t0)
            print(f"hb {step} it/s {rate:.2f} dps {rate*args.batch*world:.0f}", file=mf, flush=True)
        if is_main and step % args.eval_every == 0:
            fr, pr, ia = evaluate(core, Pval, dev)
            win = float((fr >= np.maximum(pr, ia) - 1e-9).mean())
            beats = fr.mean() > b_pfn and fr.mean() > b_icl
            score = win + 1e-4 * fr.mean()                            # SELECT on win-both rate (tie-break R²)
            if score > best:
                best = score; best_r2 = float(fr.mean()); best_win = win
                torch.save(core.state_dict(), os.path.join(args.out, "best.pt"))
                torch.save(ema, os.path.join(args.out, "best_ema.pt"))
            peak = torch.cuda.max_memory_allocated(dev) / 1e9; rate = step / (time.time() - t0)
            print(f"step {step:4d} | loss {float(loss):.3f} | fused AUC {fr.mean():.4f} (pfn {b_pfn:.4f} icl {b_icl:.4f}) "
                  f"| win-both {100*win:.0f}% | beats-both {beats} | peakmem {peak:.1f}GB | it/s {rate:.2f} dps {rate*args.batch*world:.0f}",
                  file=mf, flush=True)
    if is_main:
        print(f"\n=== BEST fused val R2 {best_r2:.4f} (win-both {100*best_win:.0f}%) vs TabPFN-v3 {b_pfn:.4f} / TabICL {b_icl:.4f} -> beats both: {best_r2 > b_pfn and best_r2 > b_icl} ===", file=mf, flush=True)
        mf.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
