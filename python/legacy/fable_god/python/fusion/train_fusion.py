"""Unified fusion trainer (packed cache) — the production run.

Features requested: checkpoints every N steps, train AND val loss logged, a
one-time val scoring of BOTH single experts (TabPFN-v3 + TabICL) as hard
baselines, group-DRO CVaR tail, early-stop to best-val, and a streaming metrics
JSONL for the TUI. Minibatches sample DATASETS (held-out val datasets) so the
gate/head can't memorize identity.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss

from .model import ConcatFusion, AttnFusion
from .cache import load_cache_packed


def _auc(p, y):
    y = y.astype(int)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


def _ll(p, y):
    y = y.astype(int)
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]) if len(np.unique(y)) > 1 else 0.0


def baselines(entries):
    """One-time per-dataset AUC for each expert + avg-of-probs + oracle."""
    E = entries[0]["logits"].shape[1]
    auc = np.zeros(E); avg = orac = ll = 0.0
    for e in entries:
        y = e["y"].numpy()
        probs = torch.softmax(e["logits"], -1)[..., 1].numpy()   # (M,E)
        a = [_auc(probs[:, i], y) for i in range(E)]
        for i in range(E):
            auc[i] += a[i]
        pavg = probs.mean(1); avg += _auc(pavg, y); orac += max(a); ll += _ll(pavg, y)
    n = len(entries)
    return (auc / n).tolist(), avg / n, orac / n, ll / n


def evaluate(model, entries, dev):
    """Returns mean AUC, logloss, CE, per-dataset win-rate vs best single expert,
    and mean routing weight to expert 0 (ELO-relevant: win-rate)."""
    model.eval(); auc = ll = ce = wins = w0 = 0.0
    with torch.no_grad():
        for e in entries:
            M = e["logits"].shape[0]
            out = model(e["logits"].to(dev), e["hidden"].to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1), e["conf"].to(dev))
            y = e["y"].numpy()
            ce += float(F.cross_entropy(out["final"], e["y"].to(dev).long()))
            p = torch.softmax(out["final"], -1)[:, 1].cpu().numpy()
            fa = _auc(p, y); auc += fa; ll += _ll(p, y)
            probs = torch.softmax(e["logits"], -1)[..., 1].numpy()
            best = max(_auc(probs[:, i], y) for i in range(probs.shape[1]))
            wins += (fa >= best - 1e-9)
            if out.get("w") is not None:
                w0 += float(out["w"].reshape(-1)[0])
    n = len(entries)
    return auc / n, ll / n, ce / n, wins / n, w0 / n


def parity(model, entries, dev):
    e = entries[0]; M = e["logits"].shape[0]; worst = 0.0
    with torch.no_grad():
        for i in range(model.n_experts):
            out = model(e["logits"].to(dev), e["hidden"].to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1),
                        e["conf"].to(dev), parity_expert=i)
            worst = max(worst, float((out["final"] - e["logits"][:, i].to(dev)).abs().max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="dir for checkpoints + metrics.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16, help="datasets per step")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.25, help="CVaR tail fraction")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="attn", choices=["attn", "concat"])
    ap.add_argument("--lam-win", type=float, default=0.5, help="beat-best-expert hinge")
    ap.add_argument("--lam-bal", type=float, default=0.2, help="anti-collapse floor")
    ap.add_argument("--win-margin", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mfile = open(os.path.join(args.out, "metrics.jsonl"), "w")

    def emit(**kw):
        mfile.write(json.dumps(kw) + "\n"); mfile.flush()

    entries = [e for e in load_cache_packed(args.cache, limit=args.limit) if "logits" in e]
    rng = np.random.default_rng(args.seed); rng.shuffle(entries)
    nval = max(2, int(args.val_frac * len(entries)))
    val, train = entries[:nval], entries[nval:]
    val = val[:1500]                    # cap val: fast, stable metric, frequent TUI updates
    dev = torch.device(args.device)
    E = train[0]["logits"].shape[1]
    print(f"cache {len(entries)} datasets -> {len(train)} train / {len(val)} val, {E} experts", flush=True)

    exp_auc, avg, orac, ll = baselines(val)
    names = ["tabpfn_v3", "tabicl"][:E]
    print(f"BASELINES (val, one-time): {dict(zip(names, np.round(exp_auc,4)))} | "
          f"avg {avg:.4f} | ORACLE {orac:.4f}", flush=True)

    Model = AttnFusion if args.model == "attn" else ConcatFusion
    model = Model(n_experts=E, out_dim=2,
                  gate_meta_dim=train[0]["gate_meta"].numel(),
                  conf_dim=train[0]["conf"].shape[-1], p=0.15).to(dev)
    ntr = sum(p.numel() for p in model.fusion_parameters())
    par = parity(model, entries, dev)
    print(f"fusion trainable {ntr:,} | P0 parity {par:.2e}", flush=True)
    emit(kind="header", experts=names, expert_auc=exp_auc, avg_auc=avg, oracle_auc=orac,
         avg_logloss=ll, trainable=ntr, parity=par, n_train=len(train), n_val=len(val))

    opt = torch.optim.AdamW(model.fusion_parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    best = 0.0; best_state = copy.deepcopy(model.state_dict()); t0 = time.time()
    run_tl = []; last_v = (0.0, 0.0, 0.0, 0.0)      # (val_auc, val_ce, win_rate, gate_w0)
    for step in range(1, args.steps + 1):
        model.train()
        idx = rng.integers(0, len(train), size=args.batch)
        per_ds = []; usage = []
        for j in idx:
            e = train[j]; M = e["logits"].shape[0]
            out = model(e["logits"].to(dev), e["hidden"].to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1), e["conf"].to(dev))
            y = e["y"].to(dev).long()
            l_task = F.cross_entropy(out["final"], y)
            ce_e = torch.stack([F.cross_entropy(out["cal"][:, i], y) for i in range(E)])
            l_cal = ce_e.mean()
            # WIN hinge: fused must beat the per-dataset BEST expert (targets oracle+)
            l_win = F.relu(l_task - ce_e.min().detach() + args.win_margin)
            per_ds.append(l_task + 0.2 * l_cal + args.lam_win * l_win)
            if out.get("w") is not None:
                usage.append(out["w"].mean(0))                  # (E,) this dataset's routing
        per_ds = torch.stack(per_ds)
        k = max(1, int(args.alpha * len(per_ds)))               # group-DRO CVaR tail
        loss = 0.5 * per_ds.mean() + 0.5 * torch.topk(per_ds, k).values.mean()
        if usage:                                               # anti-collapse: keep both experts used
            batch_use = torch.stack(usage).mean(0)              # (E,)
            loss = loss + args.lam_bal * F.relu(0.15 - batch_use).sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.fusion_parameters(), 1.0)
        opt.step(); sched.step(); run_tl.append(float(loss))

        if step % 15 == 0 and step % args.eval_every != 0:      # lightweight heartbeat (TUI motion)
            emit(kind="step", step=step, train_loss=float(loss), val_loss=last_v[1],
                 val_auc=last_v[0], best_auc=best, win_rate=last_v[2], gate_w0=last_v[3],
                 resid=float(model.resid_scale), lr=sched.get_last_lr()[0],
                 steps_per_s=step / (time.time() - t0))
        if step % args.eval_every == 0:
            va, vll, vce, winr, w0 = evaluate(model, val, dev)
            last_v = (va, vce, winr, w0)
            tl = float(np.mean(run_tl)); run_tl = []
            if va > best:
                best = va; best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(args.out, "best.pt"))
            sps = step / (time.time() - t0)
            emit(kind="step", step=step, train_loss=tl, val_loss=vce, val_logloss=vll,
                 val_auc=va, best_auc=best, win_rate=winr, gate_w0=w0,
                 resid=float(model.resid_scale), lr=sched.get_last_lr()[0], steps_per_s=sps)
            print(f"step {step:5d} | train {tl:.4f} | val CE {vce:.4f} AUC {va:.4f} "
                  f"win% {100*winr:.0f} gate_pfn {w0:.2f} "
                  f"(best {best:.4f}; exp {max(exp_auc):.4f}/avg {avg:.4f}/orac {orac:.4f})",
                  flush=True)
        if step % args.ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(args.out, f"ckpt_{step}.pt"))
            emit(kind="ckpt", step=step, path=f"ckpt_{step}.pt")

    emit(kind="final", best_auc=best, beats_oracle=bool(best > orac))
    print(f"\nDONE. best val AUC {best:.4f} vs best-expert {max(exp_auc):.4f} / avg {avg:.4f} "
          f"/ oracle {orac:.4f} -> {'BEATS ORACLE' if best>orac else 'below oracle'}", flush=True)
    mfile.close()


if __name__ == "__main__":
    main()
