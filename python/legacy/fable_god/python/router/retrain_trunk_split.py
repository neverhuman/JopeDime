"""Recover the fully-scrambled chimera WITHOUT a new cache build: retrain the AttnFusion trunk
on a disjoint train/val split of real_bin_shuffled (1244 real-binary datasets' SCRAMBLED
features). Experts frozen at max distance; only the trunk learns to route around degraded A.
Warm-start from the trained trunk. Per-dataset (variable rows). Single-GPU, fast, reliable."""
import argparse, os, glob, time
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from python.fusion.model import AttnFusion

ARCH = dict(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3, d=256, rank=128,
            d_model=512, n_layers=8, n_heads=8, p=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/fable_god/cache/router/real_bin_shuffled"))
    ap.add_argument("--warm", default=os.path.expanduser("~/fable_god/chimera_classification/chimera_best.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/weights/shuffled/trunk_recovered.pt"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8, help="datasets/step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()
    dev = torch.device("cuda")

    ent = []
    for f in sorted(glob.glob(args.cache + "/*.pt")):
        e = torch.load(f, map_location="cpu")
        if "logits" in e and len(np.unique(e["y"].numpy())) >= 2 and e["y"].numel() >= 20:
            ent.append({k: e[k] for k in ("logits", "hidden", "conf", "gate_meta", "y")})
    rng = np.random.default_rng(0); rng.shuffle(ent)
    nval = 250; val, train = ent[:nval], ent[nval:]
    print(f"{len(train)} train / {len(val)} val datasets (scrambled features)", flush=True)

    core = AttnFusion(**ARCH).to(dev)
    core.load_state_dict(torch.load(args.warm, map_location=dev))          # warm-start
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ema = {k: v.detach().clone() for k, v in core.state_dict().items()}

    def fwd(e):
        lg = e["logits"].float().to(dev); hd = e["hidden"].float().to(dev); cf = e["conf"].float().to(dev)
        gm = e["gate_meta"].float().to(dev).unsqueeze(0).repeat(lg.shape[0], 1)
        return core(lg, hd, gm, cf), e["y"].to(dev)

    def evaluate(dss):
        core.eval(); fa, pa, ia = [], [], []
        with torch.no_grad():
            for e in dss:
                y = e["y"].numpy()
                out, _ = fwd(e)
                p = torch.softmax(out["final"].float(), -1)[:, 1].cpu().numpy()
                lg = e["logits"].float()
                fa.append(roc_auc_score(y, p))
                pa.append(roc_auc_score(y, torch.softmax(lg[:, 0], -1)[:, 1].numpy()))
                ia.append(roc_auc_score(y, torch.softmax(lg[:, 1], -1)[:, 1].numpy()))
        return np.mean(fa), np.mean(pa), np.mean(ia), float(np.mean(np.array(fa) >= np.maximum(pa, ia) - 1e-9))

    f0, p0, i0, w0 = evaluate(val)
    print(f"WARM-START (scrambled experts + orig trunk): fused AUC {f0:.4f} (scr-pfn {p0:.4f} scr-icl {i0:.4f}) win {100*w0:.0f}%", flush=True)
    best = f0; t0 = time.time()
    for step in range(1, args.steps + 1):
        core.train(); idx = rng.integers(0, len(train), args.bs)
        opt.zero_grad(); losses = []
        for j in idx:
            e = train[j]; out, y = fwd(e); final = out["final"].float(); cal = out["cal"].float()
            ce = F.cross_entropy(final, y.long())
            cal_ce = torch.stack([F.cross_entropy(cal[:, c], y.long()) for c in range(2)])
            losses.append(ce + 0.3 * F.relu(ce - cal_ce.min().detach() + 0.02))
        loss = torch.stack(losses).mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            for k, v in core.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v, alpha=0.001)
        if step % args.eval_every == 0:
            fa, pa, ia, win = evaluate(val)
            if fa > best:
                best = fa; torch.save(core.state_dict(), args.out); torch.save(ema, args.out.replace(".pt", "_ema.pt"))
            print(f"step {step:4d} | loss {float(loss):.3f} | fused AUC {fa:.4f} | win-both {100*win:.0f}% | best {best:.4f} "
                  f"| {step/(time.time()-t0):.1f} it/s", flush=True)
    print(f"\n=== RECOVERED trunk: best fused AUC {best:.4f} (warm-start {f0:.4f}) -> {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
