"""UNLOCK experiment: unfreeze BOTH experts + the fusion head and fine-tune the
whole thing end-to-end (live forward, no cache) at a very low LR. Val-gated and
reversible: the frozen-expert fusion checkpoint is the guaranteed fallback — if
val drops below it, that checkpoint stands.

Honest prior: the evolution study proved TabPFN-v3 is at a local optimum (grad
descends in-sample, p=3.7e-14), and lr<=1e-8 is a numerical no-op. Run to confirm.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from python.fusion import experts as E
from python.fusion.model import AttnFusion
from python.fusion.data import (iter_packed_shards, load_packed_shard, group_by_shape,
                                gate_meta_vector)

HID = 512


def _auc(p, y):
    y = y.astype(int)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


def _zc(X, ref, clip=100.0):
    mu = ref.mean(0, keepdims=True); sd = ref.std(0, keepdims=True) + 1e-6
    return np.clip((X - mu) / sd, -clip, clip).astype("float32")


def pfn_forward(pfn, grp, dev):
    """grad-enabled TabPFN cls forward over B same-shape tasks -> (M,B,2),(M,B,512)."""
    cx = torch.from_numpy(np.stack([t.Xtr for t in grp], 1)).to(dev)     # (C,B,F)
    qx = torch.from_numpy(np.stack([t.Xte for t in grp], 1)).to(dev)     # (M,B,F)
    cy = torch.from_numpy(np.stack([t.ytr.astype("float32") for t in grp], 1)).to(dev)
    out = pfn.model(torch.cat([cx, qx], 0), cy, only_return_standard_out=False)
    return out["standard"][..., :2].float(), out["test_embeddings"].float()


def icl_forward(icl, grp, dev):
    X = torch.from_numpy(np.stack([_zc(np.concatenate([t.Xtr, t.Xte], 0), t.Xtr) for t in grp])).to(dev)
    y = torch.from_numpy(np.stack([t.ytr.astype("float32") for t in grp])).to(dev)
    lg = icl.model(X, y).float()                                          # (B,M,K)
    hid = icl._hid["h"].float()                                           # (B,T,512)
    n = grp[0].Xtr.shape[0]
    return lg[..., :2].transpose(0, 1), hid[:, n:].transpose(0, 1)        # (M,B,2),(M,B,512)


def build_rows(pfn_lg, pfn_h, icl_lg, icl_h, grp, dev):
    """Assemble fusion inputs (per row, both experts) with grad. Returns
    logits(N,E,2), hidden(N,E,512), gate_meta(N,Dm), conf(N,E,3), y(N), seg(N)."""
    B = len(grp)
    logits, hidden, conf, ys, segs, gms = [], [], [], [], [], []
    for b, t in enumerate(grp):
        lg = torch.stack([pfn_lg[:, b], icl_lg[:, b]], 1)                # (M,2,2)
        hd = torch.stack([pfn_h[:, b], icl_h[:, b]], 1)                  # (M,2,512)
        p1 = torch.softmax(lg, -1)[..., 1]                              # (M,2)
        c = torch.stack([p1, -(torch.stack([1 - p1, p1], -1).clamp(1e-6, 1 - 1e-6)
                              * torch.log(torch.stack([1 - p1, p1], -1).clamp(1e-6, 1 - 1e-6))).sum(-1),
                         p1 - (1 - p1)], -1)                             # (M,2,3)
        logits.append(lg); hidden.append(hd); conf.append(c)
        ys.append(torch.from_numpy(t.yte.astype("int64")).to(dev))
        segs.append(torch.full((lg.shape[0],), b, device=dev, dtype=torch.long))
        gms.append(torch.from_numpy(gate_meta_vector(t)).to(dev).unsqueeze(0).expand(lg.shape[0], -1))
    return (torch.cat(logits), torch.cat(hidden), torch.cat(gms), torch.cat(conf),
            torch.cat(ys), torch.cat(segs))


def seg_mean(v, seg, n):
    o = torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, seg, v)
    c = torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, seg, torch.ones_like(v))
    return o / c.clamp_min(1)


@torch.no_grad()
def evaluate(pfn, icl, fusion, groups, dev):
    fusion.eval(); auc = wins = 0.0; nd = 0
    for grp in groups:
        pl, ph = pfn_forward(pfn, grp, dev); il, ih = icl_forward(icl, grp, dev)
        lg, hd, gm, cf, y, seg = build_rows(pl, ph, il, ih, grp, dev)
        out = fusion(lg, hd, gm, cf)
        for b, t in enumerate(grp):
            m = (seg == b); p = torch.softmax(out["final"][m], -1)[:, 1].cpu().numpy()
            yb = y[m].cpu().numpy(); fa = _auc(p, yb); auc += fa; nd += 1
            probs = torch.softmax(lg[m], -1)[..., 1].cpu().numpy()
            wins += (fa >= max(_auc(probs[:, i], yb) for i in range(2)) - 1e-9)
    return auc / max(nd, 1), wins / max(nd, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-ckpt", required=True)
    ap.add_argument("--lr-expert", type=float, default=1e-6)
    ap.add_argument("--lr-fusion", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=6, help="tasks/step (grad memory)")
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda")

    pfn = E.TabPFNv3ClsBinary(dev); icl = E.TabICLCls(dev)
    ep = []
    for m in (pfn.model, icl.model):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(True); ep.append(p)
    fusion = AttnFusion(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3, p=0.0).to(dev)
    fusion.load_state_dict(torch.load(args.fusion_ckpt, map_location=dev))

    shards = list(iter_packed_shards("real_packed", "train"))[:args.shards]
    tasks = []
    for s in shards:
        tasks += load_packed_shard(s)
    groups = [g for g in group_by_shape(tasks) if len(g) >= 2]
    rng = np.random.default_rng(0); rng.shuffle(groups)
    nval = max(4, len(groups) // 6); val_g, train_g = groups[:nval], groups[nval:]
    print(f"e2e: {sum(len(g) for g in train_g)} train / {sum(len(g) for g in val_g)} val tasks "
          f"in {len(train_g)}/{len(val_g)} groups | experts UNLOCKED "
          f"({sum(p.numel() for p in ep):,} params) + fusion", flush=True)

    base_auc, base_win = evaluate(pfn, icl, fusion, val_g, dev)
    print(f"FROZEN BASELINE (val): AUC {base_auc:.4f} win% {100*base_win:.0f}  <- the floor to beat/hold",
          flush=True)

    opt = torch.optim.AdamW([{"params": ep, "lr": args.lr_expert},
                             {"params": fusion.parameters(), "lr": args.lr_fusion}], weight_decay=0)
    best = base_auc; t0 = time.time()
    for step in range(1, args.steps + 1):
        fusion.train()
        grp = train_g[rng.integers(0, len(train_g))]
        pl, ph = pfn_forward(pfn, grp, dev); il, ih = icl_forward(icl, grp, dev)
        lg, hd, gm, cf, y, seg = build_rows(pl, ph, il, ih, grp, dev)
        out = fusion(lg, hd, gm, cf)
        ce = seg_mean(F.cross_entropy(out["final"], y, reduction="none"), seg, len(grp))
        cal = torch.stack([seg_mean(F.cross_entropy(out["cal"][:, i], y, reduction="none"), seg, len(grp))
                           for i in range(2)], 1)
        loss = (ce + 0.2 * cal.mean(1) + 0.5 * F.relu(ce - cal.min(1).values.detach() + 0.02)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ep + list(fusion.parameters()), 1.0); opt.step()
        if step % args.eval_every == 0:
            va, vw = evaluate(pfn, icl, fusion, val_g, dev)
            flag = "IMPROVED" if va > best + 1e-4 else ("<< below floor, REVERT" if va < base_auc - 1e-4 else "flat")
            best = max(best, va)
            print(f"step {step:4d} | loss {float(loss):.4f} | val AUC {va:.4f} win% {100*vw:.0f} "
                  f"(floor {base_auc:.4f}, best {best:.4f}) {flag} | {step/(time.time()-t0):.1f} it/s",
                  flush=True)
    verdict = "UNLOCKING HELPED" if best > base_auc + 1e-4 else "no gain -> KEEP FROZEN FALLBACK"
    print(f"\nVERDICT: unlocked best {best:.4f} vs frozen floor {base_auc:.4f} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
