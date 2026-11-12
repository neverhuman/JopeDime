#!/usr/bin/env python3
"""H446 offline flywheel-plateau wave (CPU only, scratch copy, never the live run dir).

Same instrument as H444/H445: family-grouped 5-fold CV over the 1219 DL-Train contexts, strict = OOF mean > margin
AND family-bootstrap CI lower bound > 0, ties never win; val91 scored ONCE per seed as a regression check.

H445 red-team finding that motivates the arms: the adapter sees the fitness/gate of the first 4 slate programs as
"revealed history". Training slates (live loop `training_indices` and H443-H445) put the TRUE best-48 first, so the
model always trains with the 4 best programs revealed; eval/utility reveals pool indices 0..3, which sit at median
fitness rank ~550/1152. Train/eval mismatch in the revealed channel. Arms change what is revealed / learnable:
  reveal_match: first 4 slate entries = pool idx 0..3 (exactly what eval reveals), then true best-48, random fill;
                lr 1.5e-4, 20 steps (probe: loss under eval-style reveal is ~14.3 vs ~3.8 with top-4 reveal)
  reveal_rand : first 4 slate entries = 4 random programs per row (history-robust), then best-48, random fill;
                lr 1e-4, 12 steps
  fail_hinge  : unchanged reveal, slate adds the incumbent's predicted top-16 (hard negatives) and a margin hinge
                that pushes gate-failing programs above the best passing program (the -2.0 failure contexts)
Seed independence (H445 red-team): weights differed per seed but argmin decisions were coupled (look_ctxnorm fold 4
flipped the same 10 contexts on every seed; seeds 2/3 regcheck identical). H446 keeps the seeded perturbation,
asserts the full-train regcheck adapters differ across seeds, stores per-context val diffs, and the aggregator
reports decision-level diversity so seed spread is not mistaken for a noise estimate.
"""
import argparse, json, os, sys, time, random, hashlib, resource, copy
from pathlib import Path
import numpy as np

SCR = Path(os.environ.get("H446_ROOT", "/home/ubuntu/jope-data/hwave-h446"))
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
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--slate", type=int, default=192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arms", default="reveal_match,reveal_rand,fail_hinge")
    ap.add_argument("--perturb", type=float, default=2e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="1")
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

    # incumbent's predicted top-16 per TRAIN context (hard negatives for fail_hinge) + revealed-rank diagnostic
    inc_top = {}
    net.eval()
    with torch.no_grad():
        for c in train:
            inc_top[c] = np.argsort(predict([c], [np.arange(len(pools[c]["pids"]))])[0].numpy())[:16]
    rk = []
    for c in train:
        o = np.argsort(pools[c]["fitness"]); r_ = np.empty(len(o), int); r_[o] = np.arange(len(o)); rk.append(r_[:4])
    rk = np.array(rk); P("revealed_rank_eval_idx0_3_median", np.median(rk, 0).tolist(), "frac_in_top48", float((rk < 48).mean()))
    ARMS = {
        "reveal_match": dict(lr=1.5e-4, wd=1e-4, steps=20, slate=None, keep=48, batch_div=1, rank_w=0.0, sam=0.0, look=0, ctxnorm=False, reveal="eval", hardneg=0, hinge_w=0.0),
        "reveal_rand":  dict(lr=1e-4, wd=1e-4, steps=12, slate=None, keep=48, batch_div=1, rank_w=0.0, sam=0.0, look=0, ctxnorm=False, reveal="rand", hardneg=0, hinge_w=0.0),
        "fail_hinge":   dict(lr=5e-5, wd=1e-4, steps=8, slate=None, keep=48, batch_div=1, rank_w=0.0, sam=0.0, look=0, ctxnorm=False, reveal="top", hardneg=16, hinge_w=0.5),
    }
    HINGE_M = 0.25
    SIGDIR = a.out.parent / "fold_sig"; SIGDIR.mkdir(exist_ok=True)
    def key(*parts):
        return int.from_bytes(hashlib.sha256("h446/".join(str(p) for p in parts).encode()).digest()[:8], "little")
    def slate_idx(c, size, cfg, rng):
        pool = pools[c]; n = len(pool["pids"]); order = np.argsort(pool["fitness"])
        if cfg["reveal"] == "eval": head = list(range(min(4, n)))
        elif cfg["reveal"] == "rand": head = rng.choice(n, min(4, n), replace=False).tolist()
        else: head = []
        used = set(head); out = list(head)
        for i_ in order[:cfg["keep"]]:
            if len(out) >= size: break
            if int(i_) not in used: out.append(int(i_)); used.add(int(i_))
        for i_ in inc_top[c][:cfg["hardneg"]] if (cfg["hardneg"] and c in inc_top) else []:
            if len(out) >= size: break
            if int(i_) not in used: out.append(int(i_)); used.add(int(i_))
        rest = np.array([i_ for i_ in range(n) if i_ not in used])
        need = size - len(out)
        if need > 0: out.extend(rng.choice(rest, need, replace=len(rest) < need).tolist())
        return np.array(out[:size])
    def adapter_sha():
        h = hashlib.sha256()
        for n, p in net.named_parameters():
            if p.requires_grad: h.update(p.detach().contiguous().numpy().tobytes())
        return h.hexdigest()

    def train_candidate(arm, seed, ctxs, tag):
        cfg = ARMS[arm]
        net.load_state_dict(init_state, strict=True)
        # (1) seeded relative init perturbation of trainable adapter params
        g_ = torch.Generator().manual_seed(key(seed, arm, tag, "init") % (2**63))
        with torch.no_grad():
            for p in params:
                rms = p.detach().pow(2).mean().sqrt().clamp_min(1e-8)
                p.add_(torch.randn(p.shape, generator=g_) * rms * a.perturb)
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
        opt.load_state_dict(copy.deepcopy(init_opt))  # H444 bug: load_state_dict aliases tensors, state leaked across folds
        for st_ in opt.state.values():  # (2) drop the restored first moment; keep second moment scale
            st_["exp_avg"].zero_()
        for gp in opt.param_groups: gp["lr"] = cfg["lr"]; gp["weight_decay"] = cfg["wd"]
        L.guard.assert_optimizer_excludes_dope(opt, net)
        assert float(next(iter(init_opt["state"].values()))["step"]) == 487.0, "init optimizer state mutated"
        torch.manual_seed(key(seed, arm, tag, "dropout") % (2**63))  # (3) dropout stream
        slow = [p.detach().clone() for p in params] if cfg["look"] else None
        bsz = a.batch // cfg["batch_div"]; slate = cfg["slate"] or a.slate
        losses = []
        def batch_loss(cids, sels):
            pred = predict(cids, sels)
            tr = torch.as_tensor(np.stack([pools[c]["fitness"][s_] for c, s_ in zip(cids, sels)]), dtype=torch.float32)
            g = torch.as_tensor(np.stack([pools[c]["gate"][s_] for c, s_ in zip(cids, sels)]))
            target = torch.where(g, tr, torch.full_like(tr, 2.0))
            row = F.smooth_l1_loss(pred, target, reduction="none").mean(-1) + 0.1 * F.kl_div(
                F.log_softmax(-pred / 0.2, -1), F.softmax(-target / 0.2, -1), reduction="none").sum(-1)
            if cfg["rank_w"]:
                row = row + cfg["rank_w"] * F.cross_entropy(-pred / 0.2, target.argmin(-1), reduction="none")
            if cfg["hinge_w"]:  # gate-failing programs must cost more than the best passing one by HINGE_M
                big = torch.full_like(pred, 1e9)
                bp = torch.where(g, pred, big).min(-1).values
                has = g.any(-1); fm = (~g).float()
                h = F.relu(bp.unsqueeze(-1) + HINGE_M - pred) * fm
                hrow = torch.where(has, h.sum(-1) / fm.sum(-1).clamp_min(1.0), torch.zeros_like(bp))
                row = row + cfg["hinge_w"] * hrow
            if cfg["ctxnorm"]:
                row = row / row.detach().clamp_min(1e-3) * row.detach().mean()
            loss = row.mean(); assert torch.isfinite(loss)
            return loss
        for step in range(1 if a.probe else cfg["steps"]):
            rr = random.Random(key(seed, arm, tag, step, "ctx"))
            rng = np.random.default_rng(key(seed, arm, tag, step, "slate"))
            cids = rr.choices(ctxs, k=bsz)
            sels = [slate_idx(c, slate, cfg, rng) for c in cids]
            net.train(); opt.zero_grad(set_to_none=True)
            loss = batch_loss(cids, sels); loss.backward(); losses.append(float(loss.detach())); del loss
            if cfg["sam"]:  # ASAM: ascend to w + e(w), take gradient there, return
                with torch.no_grad():
                    gp = [p for p in params if p.grad is not None]  # skip params unused this batch
                    tw = [p.abs() * p.grad for p in gp]
                    nrm = torch.norm(torch.stack([t.norm() for t in tw])).clamp_min(1e-12)
                    eps = [cfg["sam"] * p.abs() * t / nrm for p, t in zip(gp, tw)]
                    for p, e in zip(gp, eps): p.add_(e)
                opt.zero_grad(set_to_none=True)
                loss2 = batch_loss(cids, sels); loss2.backward(); del loss2
                with torch.no_grad():
                    for p, e in zip(gp, eps): p.sub_(e)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if cfg["look"] and (step + 1) % cfg["look"] == 0:
                with torch.no_grad():
                    for p, sp in zip(params, slow): sp.add_(0.5 * (p - sp)); p.copy_(sp)
            memcheck(f"{tag}/{arm}/{seed}/{step}")
        L.guard.verify_dope_frozen_unchanged(net, before)
        return losses

    def elo(x):
        x = min(max(x, 1e-4), 1 - 1e-4); return 400 * np.log10(x / (1 - x))

    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # pre-flight: perturbation streams differ across seeds for the same arm/fold
        z = [torch.randn(4, generator=torch.Generator().manual_seed(key(sd, "reveal_match", "cv0", "init") % (2**63))) for sd in (1, 2, 3)]
        assert not torch.equal(z[0], z[1]) and not torch.equal(z[1], z[2]), "seed streams collide"
        cv = {}
        for arm in a.arms.split(","):
            t_arm = time.time(); oof = np.full(len(train), np.nan); fold_gain = []
            for k in range(a.folds if not a.probe else 1):
                tr_c = [c for c in train if fold[c] != k]; ho = [c for c in train if fold[c] == k]
                ix = np.array([tr_index[c] for c in ho]); sigf = SIGDIR / f"{arm}_cv{k}_seed{seed}.json"
                if sigf.exists():  # resume after a guard abort
                    oof[ix] = np.array(json.load(open(sigf))["oof"]); fold_gain.append(float(oof[ix].mean()))
                    P(arm, seed, "fold", k, "resumed", f"gain={fold_gain[-1]:+.6f}"); continue
                losses = train_candidate(arm, seed, tr_c, f"cv{k}")
                sha = adapter_sha()
                for other in SIGDIR.glob(f"{arm}_cv{k}_seed*.json"):
                    o = json.load(open(other))
                    assert o["seed"] == seed or o["adapter_sha"] != sha, f"seed {seed} adapter identical to seed {o['seed']} at {arm}/cv{k}"
                u = utility(ho)
                oof[ix] = u - u_inc_train[ix]; fold_gain.append(float(oof[ix].mean()))
                json.dump({"seed": seed, "arm": arm, "fold": k, "adapter_sha": sha, "gain": fold_gain[-1], "oof": oof[ix].tolist()}, open(sigf, "w"))
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
        if a.probe:  # smoke test never reads val91 for a candidate
            P("PROBE_OK", seed, sel); results.append({"seed": seed, "cv": cv, "selected": sel}); continue
        # regression check: one val evaluation of the CV-selected config trained on all train contexts
        train_candidate(sel, seed, train, "full")
        fsha = adapter_sha()
        for other in SIGDIR.glob(f"{sel}_full_seed*.json"):
            o = json.load(open(other))
            assert o["seed"] == seed or o["adapter_sha"] != fsha, f"seed {seed} full adapter identical to seed {o['seed']}"
        uv = utility(dev); dv = uv - u_inc
        json.dump({"seed": seed, "arm": sel, "adapter_sha": fsha, "val_diff": dv.tolist()}, open(SIGDIR / f"{sel}_full_seed{seed}.json", "w"))
        vlo, vhi = boot_mean_ci(dv, seed=seed)
        vw = float((dv > 0).sum()); vt = float((dv == 0).sum()); vs = (vw + 0.5 * vt) / len(dv)
        reg = {"arm": sel, "val": float(uv.mean()), "val_gain": float(dv.mean()), "val_gain_ci95_family_boot": [vlo, vhi],
               "wins": vw, "ties": vt, "losses": float((dv < 0).sum()), "elo_vs_incumbent_val": float(elo(vs)),
               "not_worse": bool(dv.mean() >= 0), "adapter_sha": fsha}
        P("REGCHECK", seed, json.dumps(reg))
        results.append({"seed": seed, "cv": cv, "selected": sel, "selected_passes_strict": cv[sel]["passes_strict"], "regression_check": reg})
        json.dump({"results": results, "peak": PEAK}, open(a.out / "partial.json", "w"), indent=1)
    after = L.guard.snapshot_dope_weights(net)
    summary = {"wave": "H446", "dope_before": before.composite_sha256, "dope_after": after.composite_sha256,
               "parity_incumbent_val": float(u_inc.mean()), "parity_ok": parity, "incumbent_train_utility": float(u_inc_train.mean()),
               "n_train": len(train), "n_train_families": len(tr_ufam), "n_dev": len(dev), "n_dev_families": len(ufam),
               "family_overlap": 0, "fold_sizes": load, "config": vars(a) | {"out": str(a.out)}, "arms_cfg": ARMS, "peak": PEAK, "results": results}
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1, default=str)
    P("DONE dope_after", after.composite_sha256, "peak", PEAK)

if __name__ == "__main__":
    main()
