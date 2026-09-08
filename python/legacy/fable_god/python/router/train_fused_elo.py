"""The CORRECTED single fused model: TabPFN-v3 + TabICL features combined into ONE
head (frozen experts as backbone; only the head trains), on the RIGHT data (real
hyperion binary datasets) with the RIGHT loss (Elo-aligned soft-AUC + beat-experts
hinge), EMA 0.999, and checkpoint SELECTION on real Elo (not a mean-AUC proxy).

Acts as one model: features -> combined head -> prediction. Target: beat TabPFN-v3
and TabICL standalone on real hyperion Elo.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from python.fusion.model import AttnFusion

sys.path.insert(0, os.path.expanduser("~/hyperion/eval_harness"))
import vendored_elo as v  # noqa: E402
sys.path.insert(0, os.path.expanduser("~/fable_god/python/router"))
from route_elo import build_index, fast_elo  # reuse identical Elo (rounds=60 default here)  # noqa: E402

CACHE = os.path.expanduser("~/fable_god/cache/router/real_bin")


def _auc(p, y):
    y = np.asarray(y).astype(int)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


def soft_auc(p1, y, tau=0.05):
    """Differentiable Wilcoxon-Mann-Whitney AUC surrogate for one dataset."""
    pos, neg = p1[y == 1], p1[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return p1.new_tensor(0.5)
    return torch.sigmoid((pos[:, None] - neg[None, :]) / tau).mean()


def load(D, dev):
    """Load cached real features, index by hash, return dict hash->entry on GPU."""
    ent = {}
    for h in D["hashes"]:
        fp = os.path.join(CACHE, f"{h}.pt")
        if not os.path.exists(fp):
            continue
        e = torch.load(fp, map_location="cpu", weights_only=False)
        ent[h] = e
    return ent


def elo_of_fused(D, rows, fused_auc, anchor):
    """Add the fused model's per-dataset AUC as a new column and grade real Elo over
    `rows`; return (fused_elo, anchor_elo)."""
    U = D["util"][rows]; Vd = D["valid"][rows]
    Uf = np.concatenate([U, fused_auc[:, None]], 1)
    Vf = np.concatenate([Vd, np.ones((len(rows), 1), bool)], 1)
    L, R, O = build_index(Uf, Vf)
    elo = fast_elo(Uf.shape[1], L, R, O, rounds=60)
    return float(elo[-1]), float(elo[anchor])       # (fused Elo, anchor Elo)


def evaluate(model, ent, D, rows, dev):
    """Fused per-dataset AUC over rows -> real Elo vs anchor."""
    model.eval(); hashes = [D["hashes"][r] for r in rows]
    fa = np.full(len(rows), np.nan)
    with torch.no_grad():
        for i, (r, h) in enumerate(zip(rows, hashes)):
            if h not in ent:
                fa[i] = D["util"][r, D["anchor"]]; continue      # fallback: anchor utility
            e = ent[h]; M = e["logits"].shape[0]
            out = model(e["logits"].to(dev), e["hidden"].float().to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1), e["conf"].to(dev))
            p = torch.softmax(out["final"], -1)[:, 1].cpu().numpy()
            fa[i] = _auc(p, e["y"].numpy())
    fe, ae = elo_of_fused(D, rows, fa, D["anchor"])
    return fe - ae, float(np.nanmean(fa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-win", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/cache/router/fused_elo"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda"); os.makedirs(args.out, exist_ok=True)

    D = v.load_cache("bin"); ent = load(D, dev)
    tr_all = [r for r in D["tr"].tolist() if D["hashes"][r] in ent]
    te = [r for r in D["te"].tolist() if D["hashes"][r] in ent]
    rng = np.random.default_rng(args.seed); rng.shuffle(tr_all)
    nval = max(20, int(args.val_frac * len(tr_all)))
    val, tr = tr_all[:nval], tr_all[nval:]
    print(f"real cache: {len(ent)} datasets | train {len(tr)} / val {len(val)} / test {len(te)}", flush=True)

    model = AttnFusion(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3, p=0.1).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    # baselines: experts' real Elo on te (fixed choice)
    g = v.EloGrader(D["util"], D["valid"], np.array(te))
    e_pfn = g.fixed_algorithm_elo(D["anchor"]); e_icl = g.fixed_algorithm_elo(list(D["algorithms"]).index("tabicl"))
    print(f"te standalone Elo: TabPFN-v3 {e_pfn:.1f} | TabICL {e_icl:.1f} | best {max(e_pfn,e_icl):.1f}", flush=True)

    hashes_tr = [D["hashes"][r] for r in tr]
    best = -1e9; best_state = None; grng = np.random.default_rng(args.seed + 1)
    for step in range(1, args.steps + 1):
        model.train()
        idx = grng.integers(0, len(tr), size=args.batch)
        losses = []
        for j in idx:
            e = ent[hashes_tr[j]]; M = e["logits"].shape[0]
            y = e["y"].to(dev)
            out = model(e["logits"].to(dev), e["hidden"].float().to(dev),
                        e["gate_meta"].to(dev).unsqueeze(0).expand(M, -1), e["conf"].to(dev))
            pf = torch.softmax(out["final"], -1)[:, 1]
            sa = soft_auc(pf, y)                                  # Elo-aligned: maximize AUC
            pe = torch.softmax(e["logits"].to(dev), -1)[..., 1]  # (M,2) expert probs
            se = torch.stack([soft_auc(pe[:, k], y) for k in range(2)]).max().detach()
            l_win = F.relu(se - sa + 0.01)                       # beat the best expert
            losses.append((1 - sa) + args.lam_win * l_win)
        loss = torch.stack(losses)
        loss = 0.5 * loss.mean() + 0.5 * torch.topk(loss, max(1, len(loss) // 4)).values.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():                                    # EMA 0.999 (proven recipe)
            for ep, mp in zip(ema.parameters(), model.parameters()):
                ep.mul_(args.ema).add_(mp, alpha=1 - args.ema)
            for eb, mb in zip(ema.buffers(), model.buffers()):
                eb.copy_(mb)

        if step % args.eval_every == 0:
            d_raw, auc_raw = evaluate(model, ent, D, val, dev)   # SELECT on raw (docs: not EMA-lag)
            if d_raw > best:
                best = d_raw; best_state = copy.deepcopy(ema.state_dict())  # deploy EMA
                torch.save(best_state, os.path.join(args.out, "best.pt"))
            print(f"step {step:4d} | loss {float(loss):.4f} | val ΔElo {d_raw:+.1f} "
                  f"(best {best:+.1f}) | val fused AUC {auc_raw:.4f}", flush=True)

    # FINAL: load best EMA, grade te
    model.load_state_dict(best_state)
    d_te, auc_te = evaluate(model, ent, D, te, dev)
    print(f"\n=== FINAL fused single model on te ({len(te)}) ===")
    print(f"fused ΔElo vs TabPFN-v3 {d_te:+.1f}  (TabICL is {e_icl-e_pfn:+.1f})")
    print(f"beats TabPFN-v3? {d_te > 0} | beats TabICL? {e_pfn + d_te > e_icl} | fused mean AUC {auc_te:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
