"""MONSTER fusion: ONE 100M+ model = TabPFN-v3 (53M) + TabICL (28M) frozen backbone
+ a beefy fusion trunk. Fine-tune ONLY the KEY layers (late icl_blocks + heads of
each expert + the whole trunk), staged, tiny LR, EMA 0.999, on REAL binary data from
~/quant/binary. Goal: the single fused model BEATS BOTH TabPFN-v3 and TabICL on
held-out real-binary validation. No routing.

The baseline to beat = each expert's OWN standalone prediction, snapshotted BEFORE
any fine-tuning (so 'beat TabPFN-v3 / TabICL' means the real standalone models).
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

from python.fusion import experts as E
from python.fusion.model import AttnFusion
from python.fusion.data import (load_packed_shard, iter_packed_shards, group_by_shape,
                                column_as_target_tasks)
from python.fusion.finetune_e2e import pfn_forward, icl_forward, build_rows, seg_mean


def pad2(lg):
    """Guarantee a 2-class logit (M,B,2): a degenerate 1-class context (imbalanced
    truncation) yields K=1 from an expert; pad class-1 with a large-negative logit."""
    if lg.shape[-1] >= 2:
        return lg[..., :2]
    return torch.cat([lg, torch.full_like(lg[..., :1], -30.0)], -1)


def soft_auc(p1, y, tau=0.05):
    """Differentiable Wilcoxon-Mann-Whitney AUC surrogate for one dataset."""
    pos, neg = p1[y == 1], p1[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return p1.new_tensor(0.5)
    return torch.sigmoid((pos[:, None] - neg[None, :]) / tau).mean()


def _auc(p, y):
    y = np.asarray(y).astype(int)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


def unlock_key_layers(pfn_m, icl_m, n_pfn_blocks=8, icl_unlock=True):
    """Freeze everything, then unfreeze the KEY layers: last n_pfn_blocks TabPFN
    icl_blocks + output_norm + head; TabICL icl_predictor (late). Returns the two
    param groups (expert-late = tiny LR, plus the trunk trained at the head LR)."""
    for p in pfn_m.parameters():
        p.requires_grad_(False)
    for p in icl_m.parameters():
        p.requires_grad_(False)
    late = []
    blocks = pfn_m.icl_blocks
    for b in list(blocks)[-n_pfn_blocks:]:
        for p in b.parameters():
            p.requires_grad_(True); late.append(p)
    for name in ("output_norm", "many_class_decoder"):
        mod = getattr(pfn_m, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(True); late.append(p)
    if icl_unlock:
        for p in icl_m.icl_predictor.parameters():
            p.requires_grad_(True); late.append(p)
    return late


class Monster(nn.Module):
    """One model: both experts (key layers trainable) + fusion trunk -> fused logits."""
    def __init__(self, pfn, icl, d_model=512, n_layers=8, n_heads=8, d=256, rank=128, p=0.1):
        super().__init__()
        self.pfn, self.icl = pfn, icl                        # wrappers (for forward)
        self.pfn_model = pfn.model                           # register as submodules so it's
        self.icl_model = icl.model                           # ONE model (param count + state_dict)
        self.trunk = AttnFusion(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3,
                                d=d, rank=rank, d_model=d_model, n_layers=n_layers,
                                n_heads=n_heads, p=p)

    def forward(self, grp, dev):
        with torch.autocast("cuda", dtype=torch.bfloat16):   # 2x faster, half memory
            pl, ph = pfn_forward(self.pfn, grp, dev)
            il, ih = icl_forward(self.icl, grp, dev)
            pl, il = pad2(pl), pad2(il)
            lg, hd, gm, cf, y, seg = build_rows(pl, ph, il, ih, grp, dev)
            out = self.trunk(lg, hd, gm, cf)
        return out["final"].float(), out["cal"].float(), y, seg, (lg, ph, ih)


def load_pool(base, split, n_shards, seed, col_target_k=0):
    """Load packed binary tasks (fast, massive). col_target_k>0 adds column-as-target
    variants for extra diversity -> never-repeating stream -> no memorization."""
    shards = list(iter_packed_shards(base, split))
    np.random.default_rng(seed).shuffle(shards)
    tasks = []
    for s in shards[:n_shards]:
        try:
            ts = load_packed_shard(s)
        except Exception:
            continue
        ts = [t for t in ts if t.n_classes == 2]              # binary only (ICL needs uniform classes)
        tasks += ts
        if col_target_k:
            for t in ts:
                tasks += list(column_as_target_tasks(t, k=col_target_k, seed=seed))
    return tasks


def chunk_groups(groups, maxb, feat_budget=12000):
    """Split each same-shape group into sub-batches. Cap by a rows*features budget so
    wide tables shrink the batch (memory-safe) while narrow tables fill it (fast)."""
    out = []
    for g in groups:
        cap = max(2, min(maxb, feat_budget // max(g[0].n_features, 1)))
        for i in range(0, len(g), cap):
            out.append(g[i:i + cap])
    return out


@torch.no_grad()
def baseline_expert_auc(model, groups, dev):
    """Standalone expert AUCs (their OWN predictions) per dataset -> the bar to beat."""
    model.eval(); pfn_a, icl_a = [], []
    for grp in groups:
        with torch.autocast("cuda", dtype=torch.bfloat16):   # same precision as training (fair baseline)
            pl, _ = pfn_forward(model.pfn, grp, dev)
            il, _ = icl_forward(model.icl, grp, dev)
        pl, il = pad2(pl).float(), pad2(il).float()
        for b, t in enumerate(grp):
            y = t.yte
            pfn_a.append(_auc(torch.softmax(pl[:, b], -1)[:, 1].cpu().numpy(), y))
            icl_a.append(_auc(torch.softmax(il[:, b], -1)[:, 1].cpu().numpy(), y))
    return np.array(pfn_a), np.array(icl_a)


@torch.no_grad()
def eval_fused(model, groups, dev):
    model.eval(); fa = []
    for grp in groups:
        final, _, _, seg, _ = model(grp, dev)
        p = torch.softmax(final, -1)[:, 1]
        for b, t in enumerate(grp):
            fa.append(_auc(p[seg == b].cpu().numpy(), t.yte))
    return np.array(fa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-shards", type=int, default=250, help="synthetic_packed train shards (~50 tasks each)")
    ap.add_argument("--real-shards", type=int, default=86)
    ap.add_argument("--val-shards", type=int, default=30, help="held-out REAL binary for validation")
    ap.add_argument("--col-target-k", type=int, default=2, help="column-as-target aug per task (diversity)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup-trunk", type=int, default=600, help="trunk-only steps before unlocking experts")
    ap.add_argument("--batch", type=int, default=48, help="datasets/step (fill GPU memory)")
    ap.add_argument("--feat-budget", type=int, default=12000, help="rows*features cap per batch (memory)")
    ap.add_argument("--lr-trunk", type=float, default=1e-3)
    ap.add_argument("--lr-expert", type=float, default=1e-5)
    ap.add_argument("--pfn-blocks", type=int, default=4, help="last N TabPFN blocks to fine-tune (conservative)")
    ap.add_argument("--icl-unlock", action="store_true", help="also unlock TabICL predictor (risky; off by default)")
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--lam-win", type=float, default=0.3, help="gentle beat-experts nudge (not the objective)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/cache/router/monster"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True            # fast A100 matmuls
    torch.backends.cudnn.allow_tf32 = True
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    pfn = E.TabPFNv3ClsBinary(dev); icl = E.TabICLCls(dev)
    model = Monster(pfn, icl).to(dev)
    tot = sum(p.numel() for p in model.parameters())
    trunk_n = sum(p.numel() for p in model.trunk.parameters())
    if is_main:
        print(f"MONSTER model: {tot/1e6:.1f}M params (trunk {trunk_n/1e6:.1f}M) "
              f"[{'100M+ ✓' if tot > 100e6 else 'under 100M'}] on {world} GPU", flush=True)

    if is_main:
        print("loading packed binary pools (synthetic + real train, held-out real val)...", flush=True)
    train_tasks = (load_pool("synthetic_packed", "train", args.syn_shards, args.seed, args.col_target_k)
                   + load_pool("real_packed", "train", args.real_shards, args.seed, args.col_target_k))
    tr_groups = chunk_groups(group_by_shape(train_tasks), args.batch, args.feat_budget)
    val_groups, b_pfn, b_icl = None, None, None
    if is_main:                                              # only rank 0 evaluates
        val_tasks = load_pool("real_packed", "val", args.val_shards, args.seed + 999)  # REAL, held-out
        val_groups = chunk_groups(group_by_shape(val_tasks), 24, args.feat_budget)
        print(f"TRAIN {sum(len(g) for g in tr_groups)} tasks (synthetic+real) in {len(tr_groups)} batches "
              f"x{world} GPU | VAL {sum(len(g) for g in val_groups)} REAL tasks in {len(val_groups)} batches",
              flush=True)
        b_pfn, b_icl = baseline_expert_auc(model, val_groups, dev)
        print(f"BASELINE val AUC: TabPFN-v3 {b_pfn.mean():.4f} | TabICL {b_icl.mean():.4f} | "
              f"best-of-both {np.maximum(b_pfn, b_icl).mean():.4f}", flush=True)

    # stage 0: trunk only
    late = unlock_key_layers(pfn.model, icl.model, args.pfn_blocks, args.icl_unlock)
    for p in late:
        p.requires_grad_(False)                              # start frozen; unlock after warmup
    opt = torch.optim.AdamW([
        {"params": model.trunk.parameters(), "lr": args.lr_trunk},
        {"params": late, "lr": args.lr_expert},
    ], weight_decay=1e-4)
    opt_params = list(model.trunk.parameters()) + late     # the params we optimize + all-reduce + EMA
    ema = {id(p): p.detach().clone() for p in opt_params}

    best = -1e9; unlocked = False; t0 = time.time()
    grng = np.random.default_rng(args.seed + 1 + rank)      # each rank samples DIFFERENT batches
    for step in range(1, args.steps + 1):
        if step == args.warmup_trunk and not unlocked:      # stage 1: unlock KEY expert layers
            for p in late:
                p.requires_grad_(True)
            unlocked = True
            if is_main:
                print(f"[step {step}] unlocked {sum(p.numel() for p in late)/1e6:.1f}M expert key-layer params",
                      flush=True)
        model.train()
        grp = tr_groups[grng.integers(0, len(tr_groups))]
        final, cal, y, seg, _ = model(grp, dev)
        B = len(grp)
        p1 = torch.softmax(final, -1)[:, 1]
        losses = []
        for b in range(B):
            m = seg == b
            ce = F.cross_entropy(final[m], y[m], label_smoothing=args.label_smooth)  # proper score, TRUE labels
            sa = soft_auc(p1[m], y[m])
            se = torch.stack([soft_auc(torch.softmax(cal[m][:, k], -1)[:, 1], y[m])
                              for k in range(2)]).max().detach()
            losses.append(ce + args.lam_win * F.relu(se - sa + 0.01))    # gentle beat-experts nudge
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward()
        if ddp:                                             # average grads across all 8 GPUs
            for p in opt_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world
        torch.nn.utils.clip_grad_norm_([p for p in opt_params if p.requires_grad], 1.0)
        opt.step()
        with torch.no_grad():
            for p in opt_params:
                ema[id(p)].mul_(args.ema).add_(p.detach(), alpha=1 - args.ema)

        if is_main and step % 20 == 0 and step % args.eval_every != 0:   # throughput heartbeat
            rate = step / (time.time() - t0)
            print(f"hb {step} it/s {rate:.2f} dps {rate*args.batch*world:.0f}", flush=True)
        if is_main and step % args.eval_every == 0:
            backup = {id(p): p.detach().clone() for p in opt_params}   # deploy EMA weights for eval/save
            for p in opt_params:
                p.data.copy_(ema[id(p)])
            fa = eval_fused(model, val_groups, dev)
            win_both = float((fa >= np.maximum(b_pfn, b_icl) - 1e-9).mean())
            beats = fa.mean() > b_pfn.mean() and fa.mean() > b_icl.mean()
            if fa.mean() > best:
                best = fa.mean()
                torch.save(copy.deepcopy(model.state_dict()), os.path.join(args.out, "best.pt"))
            for p in opt_params:
                p.data.copy_(backup[id(p)])                 # restore live (raw) weights
            peak = torch.cuda.max_memory_allocated(dev) / 1e9
            rate = step / (time.time() - t0)
            print(f"step {step:4d} | loss {float(loss):.3f} | fused AUC {fa.mean():.4f} "
                  f"(pfn {b_pfn.mean():.4f} icl {b_icl.mean():.4f}) | win-both {100*win_both:.0f}% | "
                  f"beats-both {beats} | peakmem {peak:.1f}GB | it/s {rate:.2f} dps {rate*args.batch*world:.0f}",
                  flush=True)

    if is_main:
        print(f"\n=== BEST fused val AUC {best:.4f} vs TabPFN-v3 {b_pfn.mean():.4f} / "
              f"TabICL {b_icl.mean():.4f} -> beats both: {best > b_pfn.mean() and best > b_icl.mean()} ===",
              flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
