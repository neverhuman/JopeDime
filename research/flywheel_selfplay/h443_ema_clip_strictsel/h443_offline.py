#!/usr/bin/env python3
"""H443 offline flywheel-plateau wave (CPU only, scratch copy, never the live run dir).

Arms (all start from incumbent round-3175, DOPE frozen, adapter-only training):
  base   : loop emulation -- loop loss, greedy admission (accept cand >= best, tie accepted), restore on reject
  ema    : loop loss, no restore; Polyak/EMA weights (decay per step) evaluated; greedy admission of EMA weights
  clip   : per-context loss normalization + failure-target soft cap, greedy admission, restore on reject
  strict : loop loss, strict admission (mean paired gain > margin AND family-bootstrap 95% lower bound > 0), restore
Validation: 91 DL-Val regression contexts (family-disjoint from 1219 DL-Train), utility = -fitness if gate else -2.0.
"""
import argparse, json, os, sys, time, random, hashlib, resource
from pathlib import Path
import numpy as np

SCR = Path(os.environ.get("H443_ROOT", "/home/ubuntu/jope-data/hwave-h443"))
LOOP = Path("/home/ubuntu/JopeDime/.agent/worktrees/flywheel-gp-fused/research/flywheel_selfplay")
DOPE_PIN = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
INCUMBENT_U = 1.1389572403748425

def memavail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1048576
    return 0.0

PEAK = {"min_memavail_gb": 1e9, "max_rss_gb": 0.0}
def memcheck(tag=""):
    m = memavail_gb(); PEAK["min_memavail_gb"] = min(PEAK["min_memavail_gb"], m)
    PEAK["max_rss_gb"] = max(PEAK["max_rss_gb"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576)
    if m < 15.0:
        raise SystemExit(f"ABORT MemAvailable {m:.1f} GB < 15 GB at {tag}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--slate", type=int, default=192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arms", default="base,ema,clip,strict")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--ema-decay", type=float, default=0.9)
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--out", type=Path, default=SCR / "results")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads); os.environ["MKL_NUM_THREADS"] = str(a.threads)
    import torch, torch.nn.functional as F
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(LOOP))
    os.chdir(LOOP.parents[1])
    import run_fused_closed_loop as L
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "progress.log", "a")
    def P(*x):
        s = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + " ".join(str(v) for v in x)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    selected, pools, tables = L.load_packet(SCR / "input")
    train = [r["context_sha256"] for r in selected["train"]]
    dev = [r["context_sha256"] for r in selected["dev"]]
    fam = {r["context_sha256"]: r["family_id"] for r in selected["train"] + selected["dev"]}
    tf = {fam[c] for c in train}; df = {fam[c] for c in dev}
    assert not (tf & df), "train/val family leak"
    assert all(r["dl_split"] in ("DL-Train", "DL-Val") for r in selected["train"] + selected["dev"])
    nrep = L.replay({c: pools[c] for c in train}, SCR / "promotions.jsonl")
    P("packet", len(train), len(dev), "train_fams", len(tf), "dev_fams", len(df), "journal_events_applied", nrep)
    memcheck("packet")

    src = SCR / "input/cpu_exact_fused.py"
    assert L.sha(src) == "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
    import importlib.util
    fs = importlib.util.spec_from_file_location("cpu_exact_fused", src)
    fused = importlib.util.module_from_spec(fs); fs.loader.exec_module(fused)
    dev_t = torch.device("cpu")
    net = fused.FusedJopeNetwork(device=dev_t, adapter_variant="h40b")
    for p in net.parameters(): p.requires_grad_(False)
    for n, p in net.adapter_10tok.named_parameters():
        if not n.startswith("anchor."): p.requires_grad_(True)
    L.guard.lock_dope_weights_and_eval_mode(net)
    before = L.guard.snapshot_dope_weights(net)
    assert before.composite_sha256 == DOPE_PIN, before.composite_sha256
    ck = torch.load(SCR / "round-3175.pt", map_location="cpu", weights_only=True)
    assert ck["admission"] == "incumbent" and ck["round"] == 3175
    net.load_state_dict(ck["state_dict"], strict=True)
    L.guard.verify_dope_frozen_unchanged(net, before)
    params = [p for p in net.parameters() if p.requires_grad]
    ntrain = sum(p.numel() for p in params)
    assert ntrain == 29211, ntrain
    P("dope_before", before.composite_sha256, "trainable", ntrain)

    # token prep (CPU), cached
    cache = SCR / "tokens_cache.pt"
    if cache.exists():
        tc = torch.load(cache, weights_only=True); token, mask = tc["token"], tc["mask"]
    else:
        token, mask = {}, {}
        ws = L.FusedLandmarkWorkspace(num_threads=min(4, a.threads))
        t0 = time.time()
        for i, (cid, (x, y)) in enumerate(tables.items()):
            y64 = y.astype(np.float64)
            yn = ((y64 - y64.mean()) / max(float(y64.std()), 1e-8)).astype(np.float32)
            v, col, _ = ws.fit_transform_and_extract_bundle_high(L.landmark_float32(x), np.ascontiguousarray(yn))
            cl = np.zeros((64, 17), np.float32); n = min(len(col), 64)
            cl[:n, :min(col.shape[1], 17)] = col[:n, :17]
            with torch.no_grad():
                d, lm = net._run_dope_mlc_v3_and_lm48(torch.as_tensor(v).reshape(1, 256), torch.as_tensor(cl).reshape(1, 64, 17), use_landmarks=True)
                tok, msk, _ = net._project_stage1b_tokens_batched_gpu(lm, d)
            token[cid] = tok[0].detach().clone(); mask[cid] = msk[0].detach().clone()
            if i % 200 == 0: P("prep", i, f"{time.time()-t0:.0f}s"); memcheck("prep")
        torch.save({"token": token, "mask": mask}, cache)
    del tables

    def predict(cids, sels):
        b = len(cids); m = len(sels[0]); k = min(4, m)
        prog = torch.as_tensor(np.stack([pools[c]["program"][s] for c, s in zip(cids, sels)]))
        proj = torch.as_tensor(np.stack([pools[c]["proj16"][s] for c, s in zip(cids, sels)]))
        hf = torch.as_tensor(np.stack([pools[c]["fitness"][s[:k]] for c, s in zip(cids, sels)]), dtype=torch.float32)
        hg = torch.as_tensor(np.stack([pools[c]["gate"][s[:k]] for c, s in zip(cids, sels)]), dtype=torch.float32)
        return net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((b, 584)), prog_412=prog, proj_16=proj,
            base_prior_cost=torch.zeros((b, m)), scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, m, 2)), mm_tokens_10x32=torch.stack([token[c] for c in cids]),
            mm_mask_10=torch.stack([mask[c] for c in cids]),
            revealed_indices=torch.arange(k).reshape(1, -1).expand(b, -1),
            revealed_raw_f=hf, revealed_gate=hg)["total_cost"].float()

    def utility(cids):
        net.eval(); out = []
        with torch.no_grad():
            for c in cids:
                p = pools[c]
                pred = predict([c], [np.arange(len(p["pids"]))])[0].numpy()
                i = int(np.argmin(pred))
                out.append(-float(p["fitness"][i]) if bool(p["gate"][i]) else -2.0)
        return np.array(out)

    t0 = time.time(); u_inc = utility(dev)
    P("incumbent_val", float(u_inc.mean()), "expected", INCUMBENT_U, f"eval {time.time()-t0:.1f}s")
    parity = abs(float(u_inc.mean()) - INCUMBENT_U) < 1e-9
    P("parity", parity)
    dev_fam = np.array([fam[c] for c in dev]); ufam = sorted(set(dev_fam))
    fam_idx = [np.where(dev_fam == f)[0] for f in ufam]
    def boot_mean_ci(d, B=2000, seed=0):
        r = np.random.default_rng(seed); ms = []
        for _ in range(B):
            pick = r.integers(0, len(fam_idx), len(fam_idx))
            ix = np.concatenate([fam_idx[j] for j in pick]); ms.append(d[ix].mean())
        return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))

    init_state = {k: v.clone() for k, v in net.state_dict().items()}
    init_opt = ck["optimizer"]
    if a.probe:
        a.rounds = 1; a.seeds = "1"; a.arms = "base"

    def loss_fn(pred, target_raw, gates, arm):
        if arm != "clip":
            target = torch.where(gates, target_raw, torch.full_like(target_raw, 2.0))
            return F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(
                F.log_softmax(-pred / 0.2, -1), F.softmax(-target / 0.2, -1), reduction="batchmean")
        # clip arm: failures capped at row max passing fitness + 0.25 (<= 2.0); per-context
        # row loss normalized by its detached magnitude and clipped at 3x batch median.
        passing = torch.where(gates, target_raw, torch.full_like(target_raw, -1e9))
        rowmax = passing.max(-1, keepdim=True).values
        rowmax = torch.where(rowmax < -1e8, torch.full_like(rowmax, 1.75), rowmax)
        cap = torch.clamp(rowmax + 0.25, max=2.0)
        target = torch.where(gates, target_raw, cap.expand_as(target_raw))
        row = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
        med = row.detach().median()
        w = torch.clamp(3.0 * med / row.detach().clamp_min(1e-8), max=1.0)
        kl = F.kl_div(F.log_softmax(-pred / 0.2, -1), F.softmax(-target / 0.2, -1), reduction="none").sum(-1)
        return (w * row).mean() + 0.1 * (w * kl).mean()

    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in a.arms.split(","):
            net.load_state_dict(init_state, strict=True)
            opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4); opt.load_state_dict(init_opt)
            L.guard.assert_optimizer_excludes_dope(opt, net)
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            best_u = u_inc.copy(); best_sd = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
            best_opt = opt.state_dict(); best_round = 0
            ema = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
            traj = []; accepts = []; t_arm = time.time()
            for rnd in range(1, a.rounds + 1):
                net.train()
                losses = []
                for step in range(a.steps):
                    ss = int.from_bytes(hashlib.sha256(f"h443/{seed}/{rnd}/{step}".encode()).digest()[:8], "little")
                    cids = random.Random(ss).choices(train, k=a.batch)
                    sels = [L.training_indices(pools[c], a.slate) for c in cids]
                    opt.zero_grad(set_to_none=True)
                    pred = predict(cids, sels)
                    tr = torch.as_tensor(np.stack([pools[c]["fitness"][s] for c, s in zip(cids, sels)]), dtype=torch.float32)
                    g = torch.as_tensor(np.stack([pools[c]["gate"][s] for c, s in zip(cids, sels)]))
                    loss = loss_fn(pred, tr, g, arm)
                    assert torch.isfinite(loss)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                    losses.append(float(loss.detach()))
                    with torch.no_grad():
                        for n, p in net.named_parameters():
                            if p.requires_grad: ema[n].mul_(a.ema_decay).add_(p.detach(), alpha=1 - a.ema_decay)
                    del pred, tr, g, loss
                    memcheck(f"{arm}/{seed}/{rnd}")
                if arm == "ema":
                    live = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
                    with torch.no_grad():
                        for n, p in net.named_parameters():
                            if p.requires_grad: p.copy_(ema[n])
                u = utility(dev); d = u - best_u
                if arm == "strict":
                    lo, hi = boot_mean_ci(d, seed=rnd)
                    acc = bool(d.mean() > a.margin and lo > 0)
                else:
                    acc = bool(u.mean() >= best_u.mean())
                tie = bool(acc and u.mean() == best_u.mean())
                if acc:
                    best_u = u.copy(); best_round = rnd
                    best_sd = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
                    best_opt = opt.state_dict()
                if arm == "ema":
                    with torch.no_grad():
                        for n, p in net.named_parameters():
                            if p.requires_grad: p.copy_(live[n])
                elif not acc:
                    with torch.no_grad():
                        for n, p in net.named_parameters():
                            if p.requires_grad: p.copy_(best_sd[n])
                    opt.load_state_dict(best_opt)
                traj.append({"round": rnd, "cand_val": float(u.mean()), "accepted": acc, "tie": tie, "best_val": float(best_u.mean()), "loss": losses})
                accepts.append(acc)
                P(arm, seed, rnd, f"cand={u.mean():.6f} best={best_u.mean():.6f} acc={acc} tie={tie} loss={losses[-1]:.4f} t={time.time()-t_arm:.0f}s mem={memavail_gb():.1f}G")
            L.guard.verify_dope_frozen_unchanged(net, before)
            # final adopted = best (selected on val); also last-raw (unselected) for selection-bias red-team
            last_raw = traj[-1]["cand_val"]
            d = best_u - u_inc
            lo, hi = boot_mean_ci(d, seed=seed)
            w = float((d > 0).sum()); t = float((d == 0).sum()); n = len(d)
            s = (w + 0.5 * t) / n
            r = np.random.default_rng(seed); ss_ = []
            for _ in range(2000):
                pick = r.integers(0, len(fam_idx), len(fam_idx)); ix = np.concatenate([fam_idx[j] for j in pick])
                dd = d[ix]; ss_.append(((dd > 0).sum() + 0.5 * (dd == 0).sum()) / len(dd))
            def elo(x):
                x = min(max(x, 1e-4), 1 - 1e-4); return 400 * np.log10(x / (1 - x))
            res = {"arm": arm, "seed": seed, "rounds": a.rounds, "adopted_val": float(best_u.mean()), "adopted_round": best_round,
                   "gain_vs_incumbent": float(d.mean()), "gain_ci95_family_boot": [lo, hi],
                   "wins": w, "ties": t, "losses": float((d < 0).sum()), "elo_vs_incumbent": float(elo(s)),
                   "elo_ci95": [float(elo(np.percentile(ss_, 2.5))), float(elo(np.percentile(ss_, 97.5)))],
                   "last_raw_val": last_raw, "n_accepts": int(sum(accepts)), "n_tie_accepts": int(sum(x["tie"] for x in traj)),
                   "max_cand_val": max(x["cand_val"] for x in traj), "wall_s": time.time() - t_arm, "trajectory": traj}
            results.append(res)
            json.dump({"results": results, "peak": PEAK}, open(a.out / "partial.json", "w"))
    after = L.guard.snapshot_dope_weights(net)
    summary = {"wave": "H443", "dope_before": before.composite_sha256, "dope_after": after.composite_sha256,
               "parity_incumbent_val": float(u_inc.mean()), "parity_ok": parity, "n_train": len(train), "n_dev": len(dev),
               "n_dev_families": len(ufam), "family_overlap": 0, "config": vars(a) | {"out": str(a.out)}, "peak": PEAK, "results": results}
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1, default=str)
    P("DONE dope_after", after.composite_sha256, "peak", PEAK)

if __name__ == "__main__":
    main()
