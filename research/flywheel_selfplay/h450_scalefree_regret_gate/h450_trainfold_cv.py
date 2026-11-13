#!/usr/bin/env python3
"""H450 offline flywheel-plateau wave (CPU only, scratch copy, never the live run dir).

Same instrument as H444-H448: family-grouped 5-fold CV over the 1219 DL-Train contexts (same h444/ fold salt), ties
never win; val91 scored ONCE per seed as a regression check. DOPE frozen (c51a56d0), only the 29,211 adapter params
train. Real S3 packet, DL-Train/DL-Val only.

Motivation (H449 red-team): the regret curriculum (exp(beta * z(regret)), beta -> 1.5) is degenerate. Raw regret is not
scale-free (per-context fitness scales differ by >100x; regret median 0.12, max 15.0, z max 10.4), so at beta = 1.0 the
sampling ESS is 4.3 of ~975 contexts and at beta = 1.5 it is 3.1: the last ~8 of 16 steps train on ~3 heavy-tailed
contexts. That is the 10 -> 42 loss rise (uniform-batch loss stays ~10) and it carries ~0 of its mass on the incumbent's
19 gate-failure contexts. On val91 every trust_regret/regret_ctrl regcheck loses through 2-3 NEW gate failures
(diffs -2.38, -2.30, -3.12) that outweigh ~18 small wins. H450 therefore keeps the trust_big step and fixes the weights:
  big_ctrl        : H447 trust_big (no curriculum) = the reference step
  regret_pct      : curriculum on scale-free regret: regret / std(passing fitness in that context), percentile-ranked
                    over the training-fold contexts, weight exp(beta * q), beta 0 -> 1.5 (max weight ratio e^1.5 ~ 4.5)
  regret_pct_gate : regret_pct + gate hinge (gate-failing programs must cost >= best passing + 0.25, weight 0.5),
                    aimed at the new val gate failures
  regret_pct_avg  : Polyak tail average of the regret_pct iterates over steps 8..16 (no extra training)
New diagnostics: per-fold and val new_gatefails / recovered_gatefails, curriculum ESS at the final step.
Selection rejects ties: strict needs Elo CI low > 0, and a candidate that only ties the incumbent is never selected
over it (val regcheck is reported for the CV-best arm regardless, as a regression check only).
Strict (H448/H449 rule): OOF mean > margin AND family-bootstrap CI lower bound > 0 AND Elo vs incumbent > 0 AND
family-bootstrap Elo CI lower bound > 0. Per-seed arm selection is by OOF Elo (Jepson rule: only Elo lift counts).
"""
import argparse, json, os, sys, time, random, hashlib, resource, copy
from pathlib import Path
import numpy as np

SCR = Path(os.environ.get("H450_ROOT", "/home/ubuntu/jope-data/hwave-h450"))
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
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--arms", default="big_ctrl,regret_pct,regret_pct_avg,regret_pct_gate")
    ap.add_argument("--perturb", type=float, default=5e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--out", type=Path, default=SCR / "results")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    a.out = a.out.resolve()  # H446 slip: relative --out + chdir wrote into the code worktree
    assert str(a.out).startswith(str(SCR.resolve())), f"--out must live under the scratch dir: {a.out}"
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

    GATEFAIL = {}  # (context) -> bool for the LAST utility() call; exact gate flag, not utility <= -2 (passing fitness can exceed 2)
    def utility(cids):
        net.eval(); out = []; GATEFAIL.clear()
        with torch.no_grad():
            for c in cids:
                p = pools[c]
                pred = predict([c], [np.arange(len(p["pids"]))])[0].numpy()
                i = int(np.argmin(pred))
                out.append(-float(p["fitness"][i]) if bool(p["gate"][i]) else -2.0); GATEFAIL[c] = not bool(p["gate"][i])
        return np.array(out)

    t0 = time.time(); u_inc = utility(dev); inc_fail_dev = np.array([GATEFAIL[c] for c in dev])
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
    inc_fail_train = np.array([GATEFAIL[c] for c in train])
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
    BASE = dict(lr=3e-4, steps=16, wd=1e-4, slate=None, keep=48, batch_div=1, rank_w=0.0, sam=0.0, look=0, ctxnorm=False, reveal="top", hardneg=0, hinge_w=0.0, listw=1.0, listT=0.5, curric=0.0, anchor=5.0, rank_target=False, clip_tail=0.0, pair_w=0.0, pairT=0.1, curric_src="z", tail_avg=0)
    ARMS = {"trust_big": dict(BASE), "regret_pct": BASE | dict(curric=1.5, curric_src="pct", tail_avg=8),
            "regret_pct_gate": BASE | dict(curric=1.5, curric_src="pct", hinge_w=0.5)}
    DERIVED = {"big_ctrl": dict(base="trust_big", subs=1, lam=1.0, avg=False), "regret_pct": dict(base="regret_pct", subs=1, lam=1.0, avg=False),
               "regret_pct_avg": dict(base="regret_pct", subs=1, lam=1.0, avg=True), "regret_pct_gate": dict(base="regret_pct_gate", subs=1, lam=1.0, avg=False)}
    # regret of the incumbent per TRAIN context (oracle best passing utility - incumbent utility); used only to weight
    # sampling of training-fold contexts in regret_cur (held-out fold contexts are never sampled)
    oracle = np.array([max([-float(f) for f, g in zip(pools[c]["fitness"], pools[c]["gate"]) if g] or [-2.0]) for c in train])
    regret = oracle - u_inc_train; assert (regret >= -1e-9).all(), "incumbent beats oracle?"
    rz = (regret - regret.mean()) / max(float(regret.std()), 1e-8)
    P("regret_inc_train mean", float(regret.mean()), "frac_zero", float((regret <= 1e-12).mean()))
    # H450 scale-free regret: divide by the spread of passing fitness in the context (heavy-tailed scales cancel)
    spread = np.array([max(float(np.std([float(f) for f, g in zip(pools[c]["fitness"], pools[c]["gate"]) if g])) if any(pools[c]["gate"]) else 1.0, 1e-6) for c in train])
    nreg = regret / spread
    P("nreg quantiles", np.percentile(nreg, [50, 90, 99, 100]).round(3).tolist(), "inc_gatefails_train", int(inc_fail_train.sum()),
      "inc_util_le_-2_train", int((u_inc_train <= -2.0 + 1e-9).sum()), "inc_gatefails_val", int(inc_fail_dev.sum()))
    def curric_weights(ctxs, cfg, beta):
        if cfg["curric_src"] == "pct":  # percentile rank of scale-free regret among the TRAINING-fold contexts only
            v = np.array([nreg[tr_index[c]] for c in ctxs]); q = v.argsort().argsort() / max(len(v) - 1, 1)
            return np.exp(beta * q)
        return np.exp(beta * np.array([rz[tr_index[c]] for c in ctxs]))
    def ess(w):
        p = np.asarray(w, float); p = p / p.sum(); return float(1.0 / (p ** 2).sum())
    trn_names = [n for n, p in net.named_parameters() if p.requires_grad]
    HINGE_M = 0.25
    SIGDIR = a.out.parent / "fold_sig"; SIGDIR.mkdir(exist_ok=True)
    LOSSLOG = open(a.out / "losses.jsonl", "a")
    assert str(SIGDIR).startswith(str(SCR.resolve()))
    def key(*parts):
        return int.from_bytes(hashlib.sha256("h448/".join(str(p) for p in parts).encode()).digest()[:8], "little")
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
        out = out[:size]
        if cfg["pair_w"]:  # incumbent's own argmin must be in the slate, outside the 4 revealed slots
            ic = int(inc_top[c][0])
            if ic in out[:4]:
                out.remove(ic); out.insert(4, ic)
            elif ic not in out:
                out[-1] = ic
        return np.array(out)
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
        losses = []; losses_uniform = []
        anc = [init_state[n].detach().clone() for n in trn_names] if cfg["anchor"] else None
        anc_n = sum(float(t.pow(2).sum()) for t in anc) if anc else 1.0
        AVG.clear()
        def batch_loss(cids, sels):
            pred = predict(cids, sels)
            tr = torch.as_tensor(np.stack([pools[c]["fitness"][s_] for c, s_ in zip(cids, sels)]), dtype=torch.float32)
            g = torch.as_tensor(np.stack([pools[c]["gate"][s_] for c, s_ in zip(cids, sels)]))
            target = torch.where(g, tr, torch.full_like(tr, 2.0))
            l1_target = target
            if cfg["clip_tail"]:  # tails (incl. 2.0 gate fails) can't dominate the pointwise term
                best = target.min(-1, keepdim=True).values
                l1_target = torch.minimum(target, best + cfg["clip_tail"])
            row = F.smooth_l1_loss(pred, l1_target, reduction="none").mean(-1) + 0.1 * F.kl_div(
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
            if cfg["listw"]:  # scale-invariant soft listwise target over true fitness (gate fail = 2.0)
                tt = target.argsort(-1).argsort(-1).float() if cfg["rank_target"] else target  # rank: 0 = best
                tz = (tt - tt.mean(-1, keepdim=True)) / tt.std(-1, keepdim=True).clamp_min(1e-6)
                pz = (pred - pred.mean(-1, keepdim=True)) / pred.std(-1, keepdim=True).clamp_min(1e-4)
                soft = F.softmax(-tz / cfg["listT"], -1)
                row = row + cfg["listw"] * -(soft * F.log_softmax(-pz / cfg["listT"], -1)).sum(-1)
            if cfg["pair_w"]:  # Bradley-Terry vs the incumbent's argmin: every context weighs the same
                ipos = torch.as_tensor([int(np.where(s_ == int(inc_top[c][0]))[0][0]) for c, s_ in zip(cids, sels)])
                ar = torch.arange(len(cids))
                ci = pred[ar, ipos].unsqueeze(-1); ti = target[ar, ipos].unsqueeze(-1)
                better = (target < ti - 1e-9).float(); worse = (target > ti + 1e-9).float()
                lp = better * F.softplus(-(ci - pred) / cfg["pairT"]) + worse * F.softplus(-(pred - ci) / cfg["pairT"])
                row = row + cfg["pair_w"] * lp.sum(-1) / (better + worse).sum(-1).clamp_min(1.0)
            if cfg["ctxnorm"]:
                row = row / row.detach().clamp_min(1e-3) * row.detach().mean()
            loss = row.mean()
            if anc:  # L2-SP trust region around the incumbent adapter
                loss = loss + cfg["anchor"] * sum((p - q).pow(2).sum() for p, q in zip(params, anc)) / anc_n
            assert torch.isfinite(loss)
            return loss
        for step in range(1 if a.probe else cfg["steps"]):
            rr = random.Random(key(seed, arm, tag, step, "ctx"))
            rng = np.random.default_rng(key(seed, arm, tag, step, "slate"))
            if cfg["curric"]:
                nst = 1 if a.probe else cfg["steps"]; beta = cfg["curric"] * step / max(nst - 1, 1)
                wts = curric_weights(ctxs, cfg, beta).tolist()
                if step == nst - 1: ESSLOG[f"{arm}:{tag}"] = {"ess": ess(wts), "n": len(ctxs), "beta": beta}
                cids = rr.choices(ctxs, weights=wts, k=bsz)
            else:
                cids = rr.choices(ctxs, k=bsz)
            sels = [slate_idx(c, slate, cfg, rng) for c in cids]
            net.train(); opt.zero_grad(set_to_none=True)
            loss = batch_loss(cids, sels); loss.backward(); losses.append(float(loss.detach())); del loss
            if cfg["curric"] and step == (0 if a.probe else cfg["steps"] - 1):
                with torch.no_grad():  # same step, uniform context draw: separates reweighting from divergence
                    uc = random.Random(key(seed, arm, tag, step, "uni")).choices(ctxs, k=bsz)
                    us = [slate_idx(c, slate, cfg, rng) for c in uc]
                    losses_uniform.append(float(batch_loss(uc, us)))
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
            nsteps_ = 1 if a.probe else cfg["steps"]
            if cfg["tail_avg"] and step >= nsteps_ - min(cfg["tail_avg"], nsteps_):  # Polyak tail average of iterates
                with torch.no_grad():
                    for n_, p in zip(trn_names, params):
                        AVG[n_] = AVG.get(n_, 0) + p.detach().double()
                    AVG["_n"] = AVG.get("_n", 0) + 1
            memcheck(f"{tag}/{arm}/{seed}/{step}")
        L.guard.verify_dope_frozen_unchanged(net, before)
        LOSSLOG.write(json.dumps({"arm": arm, "seed": seed, "tag": tag, "losses": losses, "uniform_last": losses_uniform, "curric_ess_last": ESSLOG.get(f"{arm}:{tag}")}) + "\n"); LOSSLOG.flush()
        return losses

    SUBCACHE = {}; AVG = {}; ESSLOG = {}
    def build_candidate(arm, seed, ctxs, tag):
        """Train (or reuse) the sub-runs of DERIVED[arm] and load soup/blend weights into net. Returns sub-0 losses."""
        d = DERIVED[arm]; states = []; l0 = None
        for j in range(d["subs"]):
            ck_ = (d["base"], seed, tag, j)  # H450: key includes the trained base arm (several bases now)
            if ck_ not in SUBCACHE:
                ls = train_candidate(d["base"], seed, ctxs, tag if j == 0 else f"{tag}/sub{j}")
                st_last = {k: v.detach().clone() for k, v in net.state_dict().items()}; st_avg = None
                if AVG:
                    st_avg = {k: v.clone() for k, v in st_last.items()}
                    for n_ in trn_names: st_avg[n_] = (AVG[n_] / AVG["_n"]).to(st_last[n_].dtype)
                    P(d["base"], seed, tag, "tail_avg_over", AVG["_n"], "steps")
                SUBCACHE[ck_] = (st_last, ls, st_avg)
            if d.get("avg"): assert SUBCACHE[ck_][2] is not None, "avg requested but base has no tail_avg"
            states.append(SUBCACHE[ck_][2] if d.get("avg") else SUBCACHE[ck_][0]);
            if j == 0: l0 = SUBCACHE[ck_][1]
        new = {}; nchg = 0
        for k, v0 in init_state.items():
            if not torch.is_floating_point(v0):
                new[k] = states[0][k].clone(); continue
            if all(torch.equal(st_[k], v0) for st_ in states):
                new[k] = v0.clone(); continue
            nchg += 1
            soup = torch.stack([st_[k].double() for st_ in states]).mean(0)
            new[k] = (v0.double() + d["lam"] * (soup - v0.double())).to(v0.dtype)
        net.load_state_dict(new, strict=True)
        L.guard.verify_dope_frozen_unchanged(net, before)
        P(arm, seed, tag, "built subs", d["subs"], "lam", d["lam"], "tensors_blended", nchg)
        return l0

    def elo(x):
        x = min(max(x, 1e-4), 1 - 1e-4); return 400 * np.log10(x / (1 - x))
    def boot_train_elo(d, B=2000, seed=0):
        r = np.random.default_rng(seed + 7919); es = []
        for _ in range(B):
            pick = r.integers(0, len(tr_fam_idx), len(tr_fam_idx))
            ix = np.concatenate([tr_fam_idx[j] for j in pick]); x = d[ix]
            es.append(elo(((x > 0).sum() + 0.5 * (x == 0).sum()) / len(x)))
        return float(np.percentile(es, 2.5)), float(np.percentile(es, 97.5))

    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # pre-flight: perturbation streams differ across seeds for the same arm/fold
        z = [torch.randn(4, generator=torch.Generator().manual_seed(key(sd, "soft_list", "cv0", "init") % (2**63))) for sd in (1, 2, 3)]
        assert not torch.equal(z[0], z[1]) and not torch.equal(z[1], z[2]), "seed streams collide"
        cv = {}
        for arm in a.arms.split(","):
            t_arm = time.time(); oof = np.full(len(train), np.nan); fold_gain = []; gsum = {"new_gatefails": 0, "recovered_gatefails": 0, "inc_gatefails": 0}
            for k in range(a.folds if not a.probe else 1):
                tr_c = [c for c in train if fold[c] != k]; ho = [c for c in train if fold[c] == k]
                ix = np.array([tr_index[c] for c in ho]); sigf = SIGDIR / f"{arm}_cv{k}_seed{seed}.json"
                if sigf.exists():  # resume after a guard abort
                    sj = json.load(open(sigf)); oof[ix] = np.array(sj["oof"]); fold_gain.append(float(oof[ix].mean()))
                    for g_k in gsum: gsum[g_k] += sj["gate"][g_k]
                    P(arm, seed, "fold", k, "resumed", f"gain={fold_gain[-1]:+.6f}"); continue
                losses = build_candidate(arm, seed, tr_c, f"cv{k}")
                sha = adapter_sha(); other_oof = {}
                for other in SIGDIR.glob(f"{arm}_cv{k}_seed*.json"):
                    o = json.load(open(other))
                    assert o["seed"] == seed or o["adapter_sha"] != sha, f"seed {seed} adapter identical to seed {o['seed']} at {arm}/cv{k}"
                    if o["seed"] != seed: other_oof[o["seed"]] = o["oof"]
                u = utility(ho)
                oof[ix] = u - u_inc_train[ix]; fold_gain.append(float(oof[ix].mean()))
                cf = np.array([GATEFAIL[c] for c in ho]); inf_ = inc_fail_train[ix]
                gstat = {"new_gatefails": int((cf & ~inf_).sum()), "recovered_gatefails": int((~cf & inf_).sum()), "inc_gatefails": int(inf_.sum())}
                for g_k in gsum: gsum[g_k] += gstat[g_k]
                P(arm, seed, "fold", k, "gate", json.dumps(gstat), "ess_last", json.dumps(ESSLOG.get(f"{DERIVED[arm]['base']}:cv{k}")))
                dis = {sd_: int(sum(1 for p_, q_ in zip(oof[ix].tolist(), oo) if p_ != q_)) for sd_, oo in other_oof.items()}
                P(arm, seed, "fold", k, "changed_vs_inc", int((oof[ix] != 0).sum()), "disagree_vs_seeds", json.dumps(dis))
                json.dump({"seed": seed, "arm": arm, "fold": k, "adapter_sha": sha, "gain": fold_gain[-1], "oof": oof[ix].tolist(), "gate": gstat}, open(sigf, "w"))
                P(arm, seed, "fold", k, f"gain={fold_gain[-1]:+.6f} loss0={losses[0]:.4f} lossN={losses[-1]:.4f} t={time.time()-t_arm:.0f}s mem={memavail_gb():.1f}G")
            m = ~np.isnan(oof); d = np.where(m, oof, 0.0)
            if a.probe: d = d[m]; lo, hi = (0.0, 0.0)
            else: lo, hi = boot_train(d, seed=seed)
            w = float((d > 0).sum()); t = float((d == 0).sum()); n = len(d); s = (w + 0.5 * t) / n
            elo_lo, elo_hi = (0.0, 0.0) if a.probe else boot_train_elo(d, seed=seed)
            cv[arm] = {"oof_gain_mean": float(d.mean()), "oof_gain_ci95_family_boot": [lo, hi], "fold_gains": fold_gain,
                       "wins": w, "ties": t, "losses": float((d < 0).sum()), "elo_vs_incumbent_oof": float(elo(s)),
                       "elo_ci95_family_boot": [elo_lo, elo_hi], "gate": gsum,
                       "passes_utility_strict": bool(d.mean() > a.margin and lo > 0),
                       "passes_strict": bool(d.mean() > a.margin and lo > 0 and elo(s) > 0 and elo_lo > 0), "wall_s": time.time() - t_arm}
            P(arm, seed, "CV", json.dumps(cv[arm]))
        sel = max(cv, key=lambda k_: (cv[k_]["elo_vs_incumbent_oof"], cv[k_]["oof_gain_mean"]))  # Elo-only rule (ties never pass strict)
        if a.probe:  # smoke test never reads val91 for a candidate; exercise the full-train build path only
            build_candidate("regret_pct_avg", seed, train, "full"); P("PROBE_OK", seed, sel, "full_sha", adapter_sha())
            results.append({"seed": seed, "cv": cv, "selected": sel}); continue
        # regression check: one val evaluation of the CV-selected config trained on all train contexts
        build_candidate(sel, seed, train, "full")
        fsha = adapter_sha()
        for other in SIGDIR.glob(f"{sel}_full_seed*.json"):
            o = json.load(open(other))
            assert o["seed"] == seed or o["adapter_sha"] != fsha, f"seed {seed} full adapter identical to seed {o['seed']}"
        uv = utility(dev); dv = uv - u_inc
        json.dump({"seed": seed, "arm": sel, "adapter_sha": fsha, "val_diff": dv.tolist()}, open(SIGDIR / f"{sel}_full_seed{seed}.json", "w"))
        vlo, vhi = boot_mean_ci(dv, seed=seed)
        vw = float((dv > 0).sum()); vt = float((dv == 0).sum()); vs = (vw + 0.5 * vt) / len(dv)
        vcf = np.array([GATEFAIL[c] for c in dev]); vif = inc_fail_dev
        reg = {"arm": sel, "val": float(uv.mean()), "val_gain": float(dv.mean()), "val_gain_ci95_family_boot": [vlo, vhi],
               "wins": vw, "ties": vt, "losses": float((dv < 0).sum()), "elo_vs_incumbent_val": float(elo(vs)),
               "not_worse": bool(dv.mean() >= 0), "adapter_sha": fsha,
               "new_gatefails": int((vcf & ~vif).sum()), "recovered_gatefails": int((~vcf & vif).sum()), "inc_gatefails": int(vif.sum())}
        P("REGCHECK", seed, json.dumps(reg))
        results.append({"seed": seed, "cv": cv, "selected": sel, "selected_passes_strict": cv[sel]["passes_strict"], "regression_check": reg})
        json.dump({"results": results, "peak": PEAK}, open(a.out / "partial.json", "w"), indent=1)
    after = L.guard.snapshot_dope_weights(net)
    summary = {"wave": "H450", "dope_before": before.composite_sha256, "dope_after": after.composite_sha256,
               "parity_incumbent_val": float(u_inc.mean()), "parity_ok": parity, "incumbent_train_utility": float(u_inc_train.mean()),
               "n_train": len(train), "n_train_families": len(tr_ufam), "n_dev": len(dev), "n_dev_families": len(ufam),
               "family_overlap": 0, "fold_sizes": load, "config": vars(a) | {"out": str(a.out)}, "arms_cfg": ARMS, "derived_cfg": DERIVED, "curric_ess": ESSLOG, "peak": PEAK, "results": results}
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1, default=str)
    P("DONE dope_after", after.composite_sha256, "peak", PEAK)

if __name__ == "__main__":
    main()
