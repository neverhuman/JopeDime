"""Recover the SHUFFLED chimera: experts stay frozen at max weight-distance; retrain the
AttnFusion trunk on cached scrambled features so it routes around the degraded (concerning-KV)
expert A. Warm-start from the original trunk. Manual 8-GPU DDP (torchrun flakes on this box).
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from python.fusion.model import AttnFusion
from python.fusion.cache import load_cache_packed, list_cache_files
from python.fusion.train_fast import preload, seg_mean

TRAIN = os.path.expanduser(os.environ.get("SHUF_TRAIN", "~/fable_god/cache/fusion/bin_coltarget_shuffled"))
VAL = os.path.expanduser(os.environ.get("SHUF_VAL", "~/fable_god/cache/router/real_bin_shuffled"))
ARCH = dict(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3, d=256, rank=128,
            d_model=512, n_layers=8, n_heads=8, p=0.1)


def evaluate(core, P, dev):
    core.eval(); fa, pa, ia = [], [], []
    with torch.no_grad():
        for d in range(P["n"]):
            s, e = int(P["off"][d]), int(P["off"][d + 1])
            y = P["y"][s:e].cpu().numpy()
            if len(np.unique(y)) < 2:
                continue
            lg = P["logits"][s:e]; hd = P["hidden"][s:e].float(); cf = P["conf"][s:e]
            gm = P["gate_meta"][d].unsqueeze(0).repeat(e - s, 1)
            out = core(lg, hd, gm, cf)
            p = torch.softmax(out["final"].float(), -1)[:, 1].cpu().numpy()
            fa.append(roc_auc_score(y, p))
            pa.append(roc_auc_score(y, torch.softmax(lg[:, 0].float(), -1)[:, 1].cpu().numpy()))
            ia.append(roc_auc_score(y, torch.softmax(lg[:, 1].float(), -1)[:, 1].cpu().numpy()))
    return np.array(fa), np.array(pa), np.array(ia)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default=os.path.expanduser("~/fable_god/chimera_classification/chimera_best.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/weights/shuffled/trunk_recovered.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam-win", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=100)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}"); is_main = rank == 0

    files = list_cache_files(TRAIN); np.random.default_rng(0).shuffle(files)
    train = [e for e in load_cache_packed(TRAIN, files=files[rank::world]) if "logits" in e]
    Ptr = preload(train, dev)
    ROWS = int(Ptr["off"][1] - Ptr["off"][0]); args.rows = min(args.rows, ROWS)
    core = AttnFusion(**ARCH).to(dev)
    core.load_state_dict(torch.load(args.warm, map_location=dev))          # warm-start from trained trunk
    trainable = list(core.parameters())

    Pval = None
    if is_main:
        val = [e for e in load_cache_packed(VAL) if "logits" in e]
        Pval = preload(val, dev)
        mf = open(os.path.expanduser("~/fable_god/cache/retrain_shuf.log"), "w")
        fa, pa, ia = evaluate(core, Pval, dev)
        b_pfn, b_icl = pa.mean(), ia.mean()
        print(f"WARM-START (shuffled experts + original trunk): fused AUC {fa.mean():.4f} "
              f"(scr-pfn {b_pfn:.4f} scr-icl {b_icl:.4f}) on {Pval['n']} val | recovering on {world} GPU", file=mf, flush=True)

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ema = {k: v.detach().clone() for k, v in core.state_dict().items()}
    best = -1e9; best_r = 0.0; t0 = time.time(); grng = np.random.default_rng(1 + rank)

    def allreduce():
        if not ddp:
            return
        for p in trainable:
            if p.grad is not None:
                dist.all_reduce(p.grad); p.grad /= world

    for step in range(1, args.steps + 1):
        core.train()
        ds_idx = torch.randint(0, Ptr["n"], (args.batch,), device=dev)
        off = torch.randint(0, ROWS, (args.batch, args.rows), device=dev)
        ridx = (ds_idx[:, None] * ROWS + off).flatten()
        seg = torch.arange(args.batch, device=dev).repeat_interleave(args.rows)
        gm = Ptr["gate_meta"][ds_idx].repeat_interleave(args.rows, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = core(Ptr["logits"][ridx], Ptr["hidden"][ridx].float(), gm, Ptr["conf"][ridx])
        final = out["final"].float(); cal = out["cal"].float(); y = Ptr["y"][ridx].long()
        ce = seg_mean(F.cross_entropy(final, y, reduction="none"), seg, args.batch)
        cal_ce = torch.stack([seg_mean(F.cross_entropy(cal[:, i], y, reduction="none"), seg, args.batch)
                              for i in range(2)], 1)
        l_win = F.relu(ce - cal_ce.min(1).values.detach() + 0.02)
        per_ds = ce + args.lam_win * l_win
        k = max(1, int(args.alpha * len(per_ds)))
        loss = 0.5 * per_ds.mean() + 0.5 * torch.topk(per_ds, k).values.mean()
        opt.zero_grad(); loss.backward(); allreduce()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            for kk, v in core.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[kk].mul_(args.ema).add_(v, alpha=1 - args.ema)
        if is_main and step % args.eval_every == 0:
            fa, pa, ia = evaluate(core, Pval, dev)
            win = float((fa >= np.maximum(pa, ia) - 1e-9).mean())
            if fa.mean() > best:
                best = fa.mean(); best_r = fa.mean()
                torch.save(core.state_dict(), args.out)
                torch.save(ema, args.out.replace(".pt", "_ema.pt"))
            rate = step / (time.time() - t0)
            print(f"step {step:4d} | loss {float(loss):.3f} | fused AUC {fa.mean():.4f} | win-both {100*win:.0f}% "
                  f"| best {best:.4f} | it/s {rate:.1f} dps {rate*args.batch*world:.0f}", file=mf, flush=True)
    if is_main:
        print(f"\n=== RECOVERED trunk: best fused AUC {best:.4f} -> {args.out} ===", file=mf, flush=True); mf.close()
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
