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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from python.fusion import experts as E
from python.fusion.model import AttnFusion
from python.fusion.data import (load_packed_shard, iter_packed_shards, group_by_shape,
                                column_as_target_tasks)
from python.fusion.finetune_e2e import pfn_forward, icl_forward, build_rows, soft_auc, seg_mean


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
        self.pfn, self.icl = pfn, icl
        self.trunk = AttnFusion(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3,
                                d=d, rank=rank, d_model=d_model, n_layers=n_layers,
                                n_heads=n_heads, p=p)

    def forward(self, grp, dev):
        pl, ph = pfn_forward(self.pfn, grp, dev)
        il, ih = icl_forward(self.icl, grp, dev)
        lg, hd, gm, cf, y, seg = build_rows(pl, ph, il, ih, grp, dev)
        out = self.trunk(lg, hd, gm, cf)
        return out["final"], out["cal"], y, seg, (lg, ph, ih)


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


@torch.no_grad()
def baseline_expert_auc(model, groups, dev):
    """Standalone expert AUCs (their OWN predictions) per dataset -> the bar to beat."""
    model.eval(); pfn_a, icl_a = [], []
    for grp in groups:
        pl, _ = pfn_forward(model.pfn, grp, dev)
        il, _ = icl_forward(model.icl, grp, dev)
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
    ap.add_argument("--batch", type=int, default=6, help="datasets/step (grad memory)")
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
    dev = torch.device("cuda"); os.makedirs(args.out, exist_ok=True)

    pfn = E.TabPFNv3ClsBinary(dev); icl = E.TabICLCls(dev)
    model = Monster(pfn, icl).to(dev)
    tot = sum(p.numel() for p in model.parameters())
    trunk_n = sum(p.numel() for p in model.trunk.parameters())
    print(f"MONSTER model: {tot/1e6:.1f}M params (trunk {trunk_n/1e6:.1f}M) "
          f"[{'100M+ ✓' if tot > 100e6 else 'under 100M'}]", flush=True)

    print("loading packed binary pools (synthetic + real train, held-out real val)...", flush=True)
    train_tasks = (load_pool("synthetic_packed", "train", args.syn_shards, args.seed, args.col_target_k)
                   + load_pool("real_packed", "train", args.real_shards, args.seed, args.col_target_k))
    val_tasks = load_pool("real_packed", "val", args.val_shards, args.seed + 999)  # REAL, held-out
    tr_groups = group_by_shape(train_tasks)
    val_groups = group_by_shape(val_tasks)
    print(f"TRAIN {sum(len(g) for g in tr_groups)} tasks (synthetic+real) in {len(tr_groups)} groups | "
          f"VAL {sum(len(g) for g in val_groups)} REAL tasks in {len(val_groups)} groups", flush=True)

    # standalone baselines (the bar to beat), snapshotted BEFORE fine-tuning
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
    trainable = [p for p in model.parameters() if p.requires_grad] + late
    ema = {id(p): p.detach().clone() for p in trainable}

    best = -1e9; best_state = None; unlocked = False
    grng = np.random.default_rng(args.seed + 1)
    for step in range(1, args.steps + 1):
        if step == args.warmup_trunk and not unlocked:      # stage 1: unlock KEY expert layers
            for p in late:
                p.requires_grad_(True)
            unlocked = True
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
            ce = F.cross_entropy(final[m], y[m], label_smoothing=args.label_smooth)  # proper score on TRUE labels
            sa = soft_auc(p1[m], y[m])
            se = torch.stack([soft_auc(torch.softmax(cal[m][:, k], -1)[:, 1], y[m])
                              for k in range(2)]).max().detach()
            losses.append(ce + args.lam_win * F.relu(se - sa + 0.01))    # gentle beat-experts nudge
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in trainable if p.requires_grad], 1.0)
        opt.step()
        with torch.no_grad():
            for p in trainable:
                ema[id(p)].mul_(args.ema).add_(p.detach(), alpha=1 - args.ema)

        if step % args.eval_every == 0:
            fa = eval_fused(model, val_groups, dev)
            win_both = float((fa >= np.maximum(b_pfn, b_icl) - 1e-9).mean())
            beats = fa.mean() > b_pfn.mean() and fa.mean() > b_icl.mean()
            score = fa.mean()
            if score > best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(args.out, "best.pt"))
            print(f"step {step:4d} | loss {float(loss):.3f} | fused AUC {fa.mean():.4f} "
                  f"(pfn {b_pfn.mean():.4f} icl {b_icl.mean():.4f}) | win-both {100*win_both:.0f}% | "
                  f"beats-both {beats}", flush=True)

    print(f"\n=== BEST fused val AUC {best:.4f} vs TabPFN-v3 {b_pfn.mean():.4f} / "
          f"TabICL {b_icl.mean():.4f} -> beats both: {best > b_pfn.mean() and best > b_icl.mean()} ===",
          flush=True)


if __name__ == "__main__":
    main()
