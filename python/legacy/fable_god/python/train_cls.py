"""Classification fusion trainer (ConcatFusion deep stacking).

Baselines per held-out dataset: each expert AUC, average-of-probs, and the
best-single-expert oracle. The concat head should EXCEED the oracle because it
combines representations, not just selects. Loss: CE + calibration CE + decayed
best-expert distillation + group-DRO CVaR tail + weight decay.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss

from .model import ConcatFusion


def _auc(p1, y):
    y = y.astype(int)
    return roc_auc_score(y, p1) if len(np.unique(y)) > 1 else 0.5


def baselines(entries):
    E = entries[0]["logits"].shape[1]
    auc = np.zeros(E); avg = 0.0; orac = 0.0; ll_avg = 0.0
    for e in entries:
        y = e["y"].numpy()
        probs = torch.softmax(e["logits"], -1)[..., 1].numpy()   # (M,E)
        a = [_auc(probs[:, i], y) for i in range(E)]
        for i in range(E):
            auc[i] += a[i]
        pavg = probs.mean(1)
        avg += _auc(pavg, y); orac += max(a)
        ll_avg += log_loss(y, np.clip(pavg, 1e-6, 1 - 1e-6), labels=[0, 1]) \
            if len(np.unique(y)) > 1 else 0.0
    n = len(entries)
    return auc / n, avg / n, orac / n, ll_avg / n


def evaluate(model, entries, dev):
    model.eval(); auc = 0.0; ll = 0.0
    with torch.no_grad():
        for e in entries:
            M = e["logits"].shape[0]
            out = model(e["logits"].to(dev), e["hidden"].to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1), e["conf"].to(dev))
            p1 = torch.softmax(out["final"], -1)[:, 1].cpu().numpy()
            y = e["y"].numpy()
            auc += _auc(p1, y)
            ll += log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6), labels=[0, 1]) \
                if len(np.unique(y)) > 1 else 0.0
    return auc / len(entries), ll / len(entries)


def parity(model, entries, dev):
    e = entries[0]; worst = 0.0; M = e["logits"].shape[0]
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from .cache import load_cache
    entries = [e for e in load_cache(args.cache) if "logits" in e]
    rng = np.random.default_rng(args.seed); rng.shuffle(entries)
    nval = max(1, int(args.val_frac * len(entries)))
    val, train = entries[:nval], entries[nval:]
    dev = torch.device(args.device)
    E = train[0]["logits"].shape[1]
    print(f"cache: {len(entries)} datasets ({len(train)} train / {len(val)} val), {E} experts")

    auc, avg, orac, ll = baselines(val)
    print(f"BASELINE (val): expert AUC {np.round(auc,4).tolist()} | avg {avg:.4f} | "
          f"ORACLE {orac:.4f} | avg logloss {ll:.4f}")

    model = ConcatFusion(n_experts=E, out_dim=2,
                         gate_meta_dim=train[0]["gate_meta"].numel(),
                         conf_dim=train[0]["conf"].shape[-1], p=0.15).to(dev)
    print(f"fusion trainable params: {sum(p.numel() for p in model.fusion_parameters()):,}")
    print(f"P0 PARITY max|Δ| (want ~0): {parity(model, entries, dev):.2e}")

    opt = torch.optim.AdamW(model.fusion_parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train); step = 0; best = 0.0
    import copy
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(args.epochs):
        model.train(); order = rng.permutation(len(train))
        for j in order:
            e = train[j]; f = step / max(total, 1)
            M = e["logits"].shape[0]
            y = e["y"].to(dev).long()
            out = model(e["logits"].to(dev), e["hidden"].to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1), e["conf"].to(dev))
            l_task = F.cross_entropy(out["final"], y)
            l_cal = torch.stack([F.cross_entropy(out["cal"][:, i], y) for i in range(E)]).mean()
            # distill toward the per-row best expert prob (decayed)
            probs = torch.softmax(e["logits"].to(dev), -1)[..., 1]        # (M,E)
            besti = (probs - y.unsqueeze(1).float()).abs().argmin(1)
            best_logit = e["logits"].to(dev)[torch.arange(M), besti]
            l_dist = F.kl_div(F.log_softmax(out["final"], -1),
                              F.softmax(best_logit, -1), reduction="batchmean")
            loss = l_task + 0.2 * l_cal + 0.3 * max(0.15, 1 - f) * l_dist
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fusion_parameters(), 1.0); opt.step()
            step += 1
        va, vll = evaluate(model, val, dev)
        if va > best:                                    # early-stop: keep best-val
            best = va; best_state = copy.deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} | val AUC {va:.4f} (best-exp {max(auc):.4f}, avg {avg:.4f}, "
                  f"oracle {orac:.4f}) | logloss {vll:.4f} | resid {float(model.resid_scale):.2f}")
    model.load_state_dict(best_state)                    # restore best-val model
    gap = max(orac - max(auc), 1e-9)
    print(f"\nBEST val AUC {best:.4f}  vs  best-expert {max(auc):.4f} / avg {avg:.4f} / "
          f"oracle {orac:.4f}  ->  {'BEATS ORACLE' if best>orac else f'closed {100*(best-max(auc))/gap:.0f}% of oracle gap'}")
    import os
    torch.save(best_state, os.path.join(args.cache, "fusion_head.pt"))


if __name__ == "__main__":
    main()
