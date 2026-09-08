"""High-throughput fusion trainer: cache PRELOADED resident on the GPUs (fp16),
per-step forward VECTORIZED over all rows in the minibatch (no per-dataset Python
loop), and DDP across ALL 8 A100s. Each rank holds 1/world of the datasets on its
own GPU; grads all-reduce the ~500K fusion params.

Launch:
  torchrun --nproc-per-node=8 -m python.fusion.train_fast --cache DIR --out DIR ...
Single GPU:
  python3 -m python.fusion.train_fast --cache DIR --out DIR
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .model import AttnFusion, ConcatFusion
from .cache import load_cache_packed


def _auc(p, y):
    y = y.astype(int)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


def preload(entries, dev):
    """Concatenate every dataset's rows into ONE resident GPU tensor per field
    (hidden as fp16 to fit), with per-dataset row segments. Batching a step is
    then pure GPU indexing — zero CPU->GPU transfer, zero Python per-dataset loop."""
    seg_sizes = [e["logits"].shape[0] for e in entries]
    off = np.concatenate([[0], np.cumsum(seg_sizes)]).astype(np.int64)
    logits = torch.cat([e["logits"] for e in entries]).to(dev)            # (R,E,K) f32
    hidden = torch.cat([e["hidden"] for e in entries]).half().to(dev)     # (R,E,512) f16
    conf = torch.cat([e["conf"] for e in entries]).to(dev)               # (R,E,Cc)
    y = torch.cat([e["y"] for e in entries]).to(dev)                     # (R,)
    gm = torch.stack([e["gate_meta"] for e in entries]).to(dev)          # (D_ds, Dm)
    row_ds = torch.repeat_interleave(
        torch.arange(len(entries), device=dev), torch.tensor(seg_sizes, device=dev))
    return dict(logits=logits, hidden=hidden, conf=conf, y=y, gate_meta=gm,
                off=torch.tensor(off, device=dev), row_ds=row_ds, n=len(entries))


def gather_rows(P, ds_idx):
    """Rows for the sampled datasets -> flat tensors + a compact segment id."""
    starts = P["off"][ds_idx]; ends = P["off"][ds_idx + 1]
    ranges = [torch.arange(int(s), int(e), device=P["logits"].device)
              for s, e in zip(starts.tolist(), ends.tolist())]
    ridx = torch.cat(ranges)
    seg = torch.repeat_interleave(
        torch.arange(len(ds_idx), device=ridx.device), (ends - starts))
    gm_rows = P["gate_meta"][ds_idx][seg]                                 # (Nrows, Dm)
    return ridx, seg, gm_rows


def seg_mean(vals, seg, n):
    out = torch.zeros(n, device=vals.device, dtype=vals.dtype)
    out.index_add_(0, seg, vals)
    cnt = torch.zeros(n, device=vals.device, dtype=vals.dtype)
    cnt.index_add_(0, seg, torch.ones_like(vals))
    return out / cnt.clamp_min(1)


def run_step(model, P, ds_idx, E, lam_win, lam_bal, alpha, win_margin):
    ridx, seg, gm = gather_rows(P, ds_idx)
    out = model(P["logits"][ridx], P["hidden"][ridx].float(), gm, P["conf"][ridx])
    y = P["y"][ridx].long()
    ce_row = F.cross_entropy(out["final"], y, reduction="none")           # (Nrows,)
    ce_ds = seg_mean(ce_row, seg, len(ds_idx))                            # (K,)
    cal_ce = torch.stack([seg_mean(F.cross_entropy(out["cal"][:, i], y, reduction="none"),
                                   seg, len(ds_idx)) for i in range(E)], 1)  # (K,E)
    l_win = F.relu(ce_ds - cal_ce.min(1).values.detach() + win_margin)    # beat best expert/ds
    per_ds = ce_ds + 0.2 * cal_ce.mean(1) + lam_win * l_win
    k = max(1, int(alpha * len(per_ds)))
    loss = 0.5 * per_ds.mean() + 0.5 * torch.topk(per_ds, k).values.mean()
    if out.get("w") is not None:                                          # anti-collapse
        use = seg_mean(out["w"][:, 0], seg, len(ds_idx)).mean()
        batch_use = torch.stack([use, 1 - use])
        loss = loss + lam_bal * F.relu(0.15 - batch_use).sum()
    return loss


@torch.no_grad()
def evaluate(model, P, E, dev):
    model.eval(); auc = wins = w0 = ce = 0.0
    core = model.module if hasattr(model, "module") else model
    for d in range(P["n"]):
        s, e = int(P["off"][d]), int(P["off"][d + 1])
        gm = P["gate_meta"][d].unsqueeze(0).expand(e - s, -1)
        out = core(P["logits"][s:e], P["hidden"][s:e].float(), gm, P["conf"][s:e])
        y = P["y"][s:e].cpu().numpy()
        ce += float(F.cross_entropy(out["final"], P["y"][s:e].long()))
        p = torch.softmax(out["final"], -1)[:, 1].cpu().numpy()
        fa = _auc(p, y); auc += fa
        probs = torch.softmax(P["logits"][s:e], -1)[..., 1].cpu().numpy()
        wins += (fa >= max(_auc(probs[:, i], y) for i in range(E)) - 1e-9)
        if out.get("w") is not None:
            w0 += float(out["w"][:, 0].mean())
    n = P["n"]
    return auc / n, ce / n, wins / n, w0 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="attn"); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ckpt-every", type=int, default=500); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.12); ap.add_argument("--val-cap", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=0.25); ap.add_argument("--lam-win", type=float, default=0.5)
    ap.add_argument("--lam-bal", type=float, default=0.2); ap.add_argument("--win-margin", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="fusion checkpoint to resume from")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    is_main = rank == 0

    from .cache import list_cache_files
    files = list_cache_files(args.cache)
    np.random.default_rng(args.seed).shuffle(files)
    nvf = max(1, int(0.05 * len(files)))             # ~5% of shard-files held out for val
    val_files, train_files = files[:nvf], files[nvf:]
    my_files = train_files[rank::world]              # RAM-safe: rank loads only its shard
    train_shard = [e for e in load_cache_packed(args.cache, files=my_files) if "logits" in e]
    val = ([e for e in load_cache_packed(args.cache, files=val_files, limit=args.val_cap)
            if "logits" in e][:args.val_cap]) if is_main else []
    E = train_shard[0]["logits"].shape[1]

    Ptr = preload(train_shard, dev)
    Pval = preload(val, dev) if is_main else None
    if is_main:
        gb = (Ptr["hidden"].element_size() * Ptr["hidden"].nelement()) / 1e9
        os.makedirs(args.out, exist_ok=True)
        mf = open(os.path.join(args.out, "metrics.jsonl"), "w")
        print(f"{len(train_files)} train-files ({len(train_shard)} datasets/rank on {world} GPU) "
              f"/ {len(val)} val | {E} experts | PRELOADED {gb:.1f} GB resident/GPU", flush=True)

    Model = AttnFusion if args.model == "attn" else ConcatFusion
    model = Model(n_experts=E, out_dim=2, gate_meta_dim=train_shard[0]["gate_meta"].numel(),
                  conf_dim=train_shard[0]["conf"].shape[-1], p=0.15).to(dev)
    if args.resume and os.path.exists(args.resume):     # resume from a saved fusion checkpoint
        model.load_state_dict(torch.load(args.resume, map_location=dev))
        if is_main:
            print(f"resumed fusion from {args.resume}", flush=True)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])

    if is_main:
        aucs, avg, orac = _baselines(val)
        print(f"BASELINES: tabpfn {aucs[0]:.4f} tabicl {aucs[1]:.4f} | avg {avg:.4f} | ORACLE {orac:.4f}",
              flush=True)
        mf.write(json.dumps(dict(kind="header", experts=["tabpfn_v3", "tabicl"], expert_auc=aucs,
                 avg_auc=avg, oracle_auc=orac, trainable=sum(p.numel() for p in model.parameters()),
                 parity=0.0, n_train=len(train_shard)*world, n_val=len(val))) + "\n"); mf.flush()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    best = 0.0; t0 = time.time(); best_state = None; last_v = (0., 0., 0., 0.)
    grng = np.random.default_rng(args.seed + rank)
    for step in range(1, args.steps + 1):
        model.train()
        ds_idx = torch.from_numpy(grng.integers(0, Ptr["n"], size=args.batch)).to(dev)
        loss = run_step(model, Ptr, ds_idx, E, args.lam_win, args.lam_bal, args.alpha, args.win_margin)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if is_main and step % 10 == 0 and step % args.eval_every != 0:
            mf.write(json.dumps(dict(kind="step", step=step, train_loss=float(loss),
                     val_auc=last_v[0], val_loss=last_v[1], best_auc=best, win_rate=last_v[2],
                     gate_w0=last_v[3], steps_per_s=step / (time.time() - t0))) + "\n"); mf.flush()
        if is_main and step % args.eval_every == 0:
            va, vce, winr, w0 = evaluate(model, Pval, E, dev); last_v = (va, vce, winr, w0)
            if va > best:
                best = va; best_state = copy.deepcopy((model.module if ddp else model).state_dict())
                torch.save(best_state, os.path.join(args.out, "best.pt"))
            sps = step / (time.time() - t0)
            mf.write(json.dumps(dict(kind="step", step=step, train_loss=float(loss), val_loss=vce,
                     val_auc=va, best_auc=best, win_rate=winr, gate_w0=w0, steps_per_s=sps)) + "\n"); mf.flush()
            print(f"step {step:5d} | train {float(loss):.4f} | val AUC {va:.4f} win% {100*winr:.0f} "
                  f"gate {w0:.2f} | {sps:.0f} it/s (best {best:.4f}, orac {orac:.4f})", flush=True)
        if is_main and step % args.ckpt_every == 0:
            mf.write(json.dumps(dict(kind="ckpt", step=step, path=f"ckpt_{step}.pt")) + "\n"); mf.flush()
    if is_main:
        print(f"\nDONE best val AUC {best:.4f} vs oracle {orac:.4f} -> "
              f"{'BEATS ORACLE' if best > orac else 'below'}", flush=True); mf.close()
    if ddp:
        dist.destroy_process_group()


def _baselines(entries):
    E = entries[0]["logits"].shape[1]; auc = np.zeros(E); avg = orac = 0.0
    for e in entries:
        y = e["y"].numpy(); probs = torch.softmax(e["logits"], -1)[..., 1].numpy()
        a = [_auc(probs[:, i], y) for i in range(E)]
        for i in range(E):
            auc[i] += a[i]
        avg += _auc(probs.mean(1), y); orac += max(a)
    n = len(entries); return (auc / n).tolist(), avg / n, orac / n


if __name__ == "__main__":
    main()
