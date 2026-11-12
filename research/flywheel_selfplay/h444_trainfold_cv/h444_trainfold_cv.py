#!/usr/bin/env python3
"""H444 offline flywheel-plateau wave (CPU only, scratch copy, never the live run dir).

Candidate pool (all start from incumbent round-3175, DOPE frozen, adapter-only training):
  fbal_accum : family-balanced sampling + 4x gradient accumulation (effective batch 8192), lr 1e-4, 4 opt steps
  swa_cos    : uniform sampling, linear warmup + cosine LR (peak 3e-4), 16 steps; SWA mean of last 8 step weights
  hilr_wd    : family-balanced sampling, warmup + cosine LR (peak 1e-3), weight decay 1e-2, 16 steps (basin escape)
Selection: family-grouped 5-fold CV over the 1219 DL-Train contexts only. For each fold the candidate is trained on
the other 4 folds' contexts and scored on the held-out fold; out-of-fold paired gains vs the incumbent are pooled and
CI'd by family bootstrap over 205 train families. Strict: accept only if mean OOF gain > margin AND CI lower bound > 0.
Regression check: the CV-best config (argmax OOF mean, whether or not it passes) is retrained on all train contexts and
scored ONCE on the 91 DL-Val contexts. No other arm ever touches val.
"""
import argparse, json, os, sys, time, random, hashlib, resource
from pathlib import Path
import numpy as np

SCR = Path(os.environ.get("H444_ROOT", "/home/ubuntu/jope-data/hwave-h444"))
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
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--slate", type=int, default=192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arms", default="fbal_accum,swa_cos,hilr_wd")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="1,2,3")
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
        a.seeds = "1"; a.folds = 2; a.micro = 2

    # ---- family-grouped K folds over TRAIN contexts only (deterministic, size-balanced greedy) ----
    tr_fams = sorted({fam[c] for c in train}, key=lambda f: hashlib.sha256(f"h444/{f}".encode()).hexdigest())
    fsize = {f: sum(1 for c in train if fam[c] == f) for f in tr_fams}
    fold_of_fam, load = {}, [0] * a.folds
    for f in sorted(tr_fams, key=lambda f: -fsize[f]):
        j = int(np.argmin(load)); fold_of_fam[f] = j; load[j] += fsize[f]
    fold = {c: fold_of_fam[fam[c]] for c in train}
    for k in range(a.folds):
        fin = {fam[c] for c in train if fold[c] == k}; fout = {fam[c] for c in train if fold[c] != k}
        assert not (fin & fout), "CV fold family leak"
        assert not (fin & df), "CV fold touches val family"
    P("folds", load)
    u_inc_train = utility(train)  # incumbent per-context train utility (paired baseline)
    tr_index = {c: i for i, c in enumerate(train)}
    tr_fam_arr = np.array([fam[c] for c in train]); tr_ufam = sorted(set(tr_fam_arr))
    tr_fam_idx = [np.where(tr_fam_arr == f)[0] for f in tr_ufam]
    def boot_train(d, B=2000, seed=0):
        r = np.random.default_rng(seed); ms = []
        for _ in range(B):
            pick = r.integers(0, len(tr_fam_idx), len(tr_fam_idx))
            ix = np.concatenate([tr_fam_idx[j] for j in pick]); ms.append(d[ix].mean())
        return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    P("incumbent_train_utility", float(u_inc_train.mean()))

    ARMS = {
        "fbal_accum": dict(balanced=True, accum=4, peak=1e-4, wd=1e-4, sched="const", swa=False),
        "swa_cos":    dict(balanced=False, accum=1, peak=3e-4, wd=1e-4, sched="cos", swa=True),
        "hilr_wd":    dict(balanced=True, accum=1, peak=1e-3, wd=1e-2, sched="cos", swa=False),
    }

    def train_candidate(arm, seed, ctxs, tag):
        cfg = ARMS[arm]
        net.load_state_dict(init_state, strict=True)
        opt = torch.optim.AdamW(params, lr=cfg["peak"], weight_decay=cfg["wd"]); opt.load_state_dict(init_opt)
        L.guard.assert_optimizer_excludes_dope(opt, net)
        nsteps = max(1, a.micro // cfg["accum"]); warm = max(1, nsteps // 4)
        by_fam = {}
        for c in ctxs: by_fam.setdefault(fam[c], []).append(c)
        fl = sorted(by_fam)
        swa_acc, swa_n, losses = None, 0, []
        for step in range(nsteps):
            if cfg["sched"] == "cos":
                lr = cfg["peak"] * (step + 1) / warm if step < warm else cfg["peak"] * 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, nsteps - warm)))
            else:
                lr = cfg["peak"]
            for g_ in opt.param_groups: g_["lr"] = float(lr); g_["weight_decay"] = cfg["wd"]
            net.train(); opt.zero_grad(set_to_none=True)
            for mi in range(cfg["accum"]):
                ss = int.from_bytes(hashlib.sha256(f"h444/{tag}/{seed}/{step}/{mi}".encode()).digest()[:8], "little")
                rr = random.Random(ss)
                if cfg["balanced"]:
                    cids = [rr.choice(by_fam[rr.choice(fl)]) for _ in range(a.batch)]
                else:
                    cids = rr.choices(ctxs, k=a.batch)
                sels = [L.training_indices(pools[c], a.slate) for c in cids]
                pred = predict(cids, sels)
                tr = torch.as_tensor(np.stack([pools[c]["fitness"][s] for c, s in zip(cids, sels)]), dtype=torch.float32)
                g = torch.as_tensor(np.stack([pools[c]["gate"][s] for c, s in zip(cids, sels)]))
                target = torch.where(g, tr, torch.full_like(tr, 2.0))
                loss = F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(
                    F.log_softmax(-pred / 0.2, -1), F.softmax(-target / 0.2, -1), reduction="batchmean")
                assert torch.isfinite(loss)
                (loss / cfg["accum"]).backward(); losses.append(float(loss.detach()))
                del pred, tr, g, target, loss
                memcheck(f"{tag}/{arm}/{seed}/{step}")
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if cfg["swa"] and step >= nsteps // 2:
                cur = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
                if swa_acc is None: swa_acc = cur
                else:
                    for n in swa_acc: swa_acc[n].add_(cur[n])
                swa_n += 1
        if cfg["swa"] and swa_n:
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if p.requires_grad: p.copy_(swa_acc[n] / swa_n)
        L.guard.verify_dope_frozen_unchanged(net, before)
        return losses

    def elo(x):
        x = min(max(x, 1e-4), 1 - 1e-4); return 400 * np.log10(x / (1 - x))

    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        cv = {}
        for arm in a.arms.split(","):
            t_arm = time.time(); oof = np.full(len(train), np.nan); fold_gain = []
            for k in range(a.folds if not a.probe else 1):
                tr_c = [c for c in train if fold[c] != k]; ho = [c for c in train if fold[c] == k]
                losses = train_candidate(arm, seed, tr_c, f"cv{k}")
                u = utility(ho); ix = np.array([tr_index[c] for c in ho])
                oof[ix] = u - u_inc_train[ix]; fold_gain.append(float(oof[ix].mean()))
                P(arm, seed, "fold", k, f"gain={fold_gain[-1]:+.6f} loss0={losses[0]:.4f} lossN={losses[-1]:.4f} t={time.time()-t_arm:.0f}s mem={memavail_gb():.1f}G")
            m = ~np.isnan(oof); d = np.where(m, oof, 0.0)
            if a.probe: d = d[m]; lo, hi = (0.0, 0.0)
            else: lo, hi = boot_train(d, seed=seed)
            w = float((d > 0).sum()); t = float((d == 0).sum()); n = len(d); s = (w + 0.5 * t) / n
            cv[arm] = {"oof_gain_mean": float(d.mean()), "oof_gain_ci95_family_boot": [lo, hi], "fold_gains": fold_gain,
                       "wins": w, "ties": t, "losses": float((d < 0).sum()), "elo_vs_incumbent_oof": float(elo(s)),
                       "passes_strict": bool(d.mean() > a.margin and lo > 0), "wall_s": time.time() - t_arm}
            P(arm, seed, "CV", json.dumps(cv[arm]))
        sel = max(cv, key=lambda k_: cv[k_]["oof_gain_mean"])
        # regression check: one val evaluation of the CV-selected config trained on all train contexts
        train_candidate(sel, seed, train, "full")
        uv = utility(dev); dv = uv - u_inc
        vlo, vhi = boot_mean_ci(dv, seed=seed)
        vw = float((dv > 0).sum()); vt = float((dv == 0).sum()); vs = (vw + 0.5 * vt) / len(dv)
        reg = {"arm": sel, "val": float(uv.mean()), "val_gain": float(dv.mean()), "val_gain_ci95_family_boot": [vlo, vhi],
               "wins": vw, "ties": vt, "losses": float((dv < 0).sum()), "elo_vs_incumbent_val": float(elo(vs)),
               "not_worse": bool(dv.mean() >= 0)}
        P("REGCHECK", seed, json.dumps(reg))
        results.append({"seed": seed, "cv": cv, "selected": sel, "selected_passes_strict": cv[sel]["passes_strict"], "regression_check": reg})
        json.dump({"results": results, "peak": PEAK}, open(a.out / "partial.json", "w"), indent=1)
    after = L.guard.snapshot_dope_weights(net)
    summary = {"wave": "H444", "dope_before": before.composite_sha256, "dope_after": after.composite_sha256,
               "parity_incumbent_val": float(u_inc.mean()), "parity_ok": parity, "incumbent_train_utility": float(u_inc_train.mean()),
               "n_train": len(train), "n_train_families": len(tr_ufam), "n_dev": len(dev), "n_dev_families": len(ufam),
               "family_overlap": 0, "fold_sizes": load, "config": vars(a) | {"out": str(a.out)}, "arms_cfg": ARMS, "peak": PEAK, "results": results}
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1, default=str)
    P("DONE dope_after", after.composite_sha256, "peak", PEAK)

if __name__ == "__main__":
    main()
