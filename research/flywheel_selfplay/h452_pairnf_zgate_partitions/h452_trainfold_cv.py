#!/usr/bin/env python3
"""H452 offline flywheel-plateau wave (CPU only, scratch copy, never the live run dir).

Same model/data instrument as H444-H451 (DOPE c51a56d0 frozen, only the 29,211 adapter params train, real S3 packet,
DL-Train/DL-Val only, ties never win, val91 scored ONCE per seed for the CV-selected arm as a regression check).

Motivation (H451 red-team): pairnf_k25 (trust_big + gate hinge + non-floor Bradley-Terry vs the incumbent argmin, then a
label-free confidence gate keeping only the top 25% of changed picks by z-margin) is the first arm in H447-H451 whose
NON-FLOOR OOF gain is positive on all 5 folds and both seeds (+0.001..+0.005 per fold, non-floor Elo +5.5/+5.2), but
every CI spans 0, it was the max-Elo pick out of 6 arms (winner's curse), and all of it was measured on ONE fixed
family partition (h444/ salt) whose fold 1 holds 7 of the incumbent's 19 TRAIN -2.0 gate failures.
H452 asks: is the pairnf_k25 non-floor Elo real, and does a stronger pairwise objective raise it?
  (1) instrument: TWO fresh family-grouped 5-fold partitions (salts h452a/, h452b/), each STRATIFIED so the families
      holding the incumbent's TRAIN gate failures are spread round-robin over folds (no fold carries the floors);
      every metric is reported per partition and pooled over both (family bootstrap resamples a family in both copies);
  (2) training: bases pairnf (H451 config, unchanged) and pairnf_w3 (pair_w 3.0, listwise weight 0.5: the Elo-shaped
      pairwise term dominates the step);
  (3) decision: label-free gates per base: none, top-25% changed by z (_k25, H451 rule, threshold over the evaluated
      set), absolute z >= 0.40 (_z40, per-context, set-independent; 0.40 ~ the H451 top-15% z cut, label-free).
Selection (pre-registered, Elo only): per seed, max mean NON-FLOOR OOF Elo over the two partitions (tie: pooled gain).
Strict: pooled OOF gain > margin AND gain CI low > 0 AND pooled Elo > 0 AND Elo CI low > 0 AND non-floor Elo CI low > 0
AND Elo > 0 on each partition.
"""
import argparse, json, os, sys, time, random, hashlib, resource, copy
from pathlib import Path
import numpy as np

SCR = Path(os.environ.get("H452_ROOT", "/home/ubuntu/jope-data/hwave-h452"))
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
    ap.add_argument("--arms", default="pairnf,pairnf_k25,pairnf_z40,pairnf_w3,pairnf_w3_k25,pairnf_w3_z40")
    ap.add_argument("--perturb", type=float, default=5e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--partitions", default="h452a/,h452b/")
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
    INC_PICK = {}  # incumbent argmin per context (filled once, before any training)
    ZLOG = {}      # last utility() call: z-margin of every changed decision and whether it was kept
    def utility(cids, keep=None, zmin=None):
        """keep=None: plain argmin. keep=q: among contexts whose pick differs from the incumbent's, keep only the top-q
        fraction by z = (pred[inc] - pred[cand]) / std(pred); the rest revert to the incumbent pick. Label-free."""
        net.eval(); GATEFAIL.clear(); picks = []; zs = []
        with torch.no_grad():
            for c in cids:
                p = pools[c]
                pred = predict([c], [np.arange(len(p["pids"]))])[0].numpy()
                i = int(np.argmin(pred)); j = INC_PICK.get(c, i)
                picks.append([i, j]); zs.append(float((pred[j] - pred[i]) / max(float(pred.std()), 1e-8)) if i != j else 0.0)
        zs = np.array(zs); chg = np.array([i != j for i, j in picks], bool); kept = chg.copy()
        if keep is not None and chg.any():
            thr = np.quantile(zs[chg], 1.0 - keep); kept = chg & (zs >= thr)
        if zmin is not None:  # H452: absolute per-context threshold, independent of the evaluated set
            kept = kept & (zs >= zmin)
        ZLOG.clear(); ZLOG.update({"idx_changed": np.where(chg)[0].tolist(), "z_changed": zs[chg].round(5).tolist(), "kept": kept[chg].tolist(), "n_changed": int(chg.sum()), "n_kept": int(kept.sum())})
        out = []
        for c, (i, j), kp in zip(cids, picks, kept):
            k_ = i if kp or i == j else j; p = pools[c]
            out.append(-float(p["fitness"][k_]) if bool(p["gate"][k_]) else -2.0); GATEFAIL[c] = not bool(p["gate"][k_])
        return np.array(out)
    with torch.no_grad():  # the network currently holds the incumbent (round-3175) weights
        net.eval()
        for c in train + dev:
            INC_PICK[c] = int(np.argmin(predict([c], [np.arange(len(pools[c]["pids"]))])[0].numpy()))

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

    u_inc_train = utility(train)  # incumbent per-context train utility (paired baseline)
    inc_fail_train = np.array([GATEFAIL[c] for c in train])
    # ---- H452: several family-grouped K-fold partitions over TRAIN only, stratified on incumbent gate failures ----
    fail_fams = {}
    for c, f_ in zip(train, inc_fail_train):
        if f_: fail_fams[fam[c]] = fail_fams.get(fam[c], 0) + 1
    SALTS = a.partitions.split(",") if not a.probe else a.partitions.split(",")[:1]
    FOLDS = []
    for salt in SALTS:
        tr_fams = sorted({fam[c] for c in train}, key=lambda f: hashlib.sha256(f"{salt}{f}".encode()).hexdigest())
        fsize = {f: sum(1 for c in train if fam[c] == f) for f in tr_fams}
        fold_of_fam, load, fl = {}, [0] * a.folds, [0] * a.folds
        for f in sorted([f for f in tr_fams if f in fail_fams], key=lambda f: (-fail_fams[f], -fsize[f])):  # stratify floors
            j = min(range(a.folds), key=lambda i: (fl[i], load[i])); fold_of_fam[f] = j; load[j] += fsize[f]; fl[j] += fail_fams[f]
        for f in sorted([f for f in tr_fams if f not in fail_fams], key=lambda f: -fsize[f]):
            j = int(np.argmin(load)); fold_of_fam[f] = j; load[j] += fsize[f]
        fold = {c: fold_of_fam[fam[c]] for c in train}
        for k in range(a.folds):
            fin = {fam[c] for c in train if fold[c] == k}; fout = {fam[c] for c in train if fold[c] != k}
            assert not (fin & fout), "CV fold family leak"
            assert not (fin & df), "CV fold touches val family"
        FOLDS.append(fold)
        P("partition", salt, "fold_sizes", load, "inc_gatefails_per_fold", fl)
    load = [[sum(1 for c in train if fd[c] == k) for k in range(a.folds)] for fd in FOLDS]
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
    assert all(int(inc_top[c][0]) == INC_PICK[c] for c in train), "incumbent pick mismatch"
    rk = []
    for c in train:
        o = np.argsort(pools[c]["fitness"]); r_ = np.empty(len(o), int); r_[o] = np.arange(len(o)); rk.append(r_[:4])
    rk = np.array(rk); P("revealed_rank_eval_idx0_3_median", np.median(rk, 0).tolist(), "frac_in_top48", float((rk < 48).mean()))
    BASE = dict(lr=3e-4, steps=16, wd=1e-4, slate=None, keep=48, batch_div=1, rank_w=0.0, sam=0.0, look=0, ctxnorm=False, reveal="top", hardneg=0, hinge_w=0.0, listw=1.0, listT=0.5, curric=0.0, anchor=5.0, rank_target=False, clip_tail=0.0, pair_w=0.0, pairT=0.1, curric_src="z", tail_avg=0, pair_nf=False)
    ARMS = {"pairnf": BASE | dict(hinge_w=0.5, pair_w=1.0, pair_nf=True),
            "pairnf_w3": BASE | dict(hinge_w=0.5, pair_w=3.0, listw=0.5, pair_nf=True)}
    DERIVED = {}
    for b_ in ("pairnf", "pairnf_w3"):
        DERIVED[b_] = dict(base=b_, subs=1, lam=1.0, avg=False, keep=None, zmin=None)
        DERIVED[f"{b_}_k25"] = dict(base=b_, subs=1, lam=1.0, avg=False, keep=0.25, zmin=None)
        DERIVED[f"{b_}_z40"] = dict(base=b_, subs=1, lam=1.0, avg=False, keep=None, zmin=0.40)
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
                if cfg["pair_nf"]:  # H451: only passing programs, only contexts where the incumbent pick passes (no -2.0 floors)
                    gi = g[ar, ipos].unsqueeze(-1)
                    better = better * g.float() * gi.float(); worse = worse * g.float() * gi.float()
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
                st_last = {k: v for k, v in st_last.items() if not torch.equal(v, init_state[k])}  # H452: diff only (memory)
                SUBCACHE[ck_] = (st_last, ls, st_avg)
            if d.get("avg"): assert SUBCACHE[ck_][2] is not None, "avg requested but base has no tail_avg"
            st_ = SUBCACHE[ck_][2] if d.get("avg") else SUBCACHE[ck_][0]
            states.append({k: st_.get(k, v) for k, v in init_state.items()})
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
        z = [torch.randn(4, generator=torch.Generator().manual_seed(key(sd, "pairnf", "p0cv0", "init") % (2**63))) for sd in (1, 2, 3)]
        assert not torch.equal(z[0], z[1]) and not torch.equal(z[1], z[2]), "seed streams collide"
        cv = {}; arm_list = a.arms.split(",")
        for ai, arm in enumerate(arm_list):
            t_arm = time.time(); D = DERIVED[arm]; per_part = []; OOF = []
            gsum = {"new_gatefails": 0, "recovered_gatefails": 0, "inc_gatefails": 0}
            for pi, fold in enumerate(FOLDS):
                oof = np.full(len(train), np.nan); fold_gain = []
                for k in range(a.folds if not a.probe else 1):
                    tr_c = [c for c in train if fold[c] != k]; ho = [c for c in train if fold[c] == k]
                    ix = np.array([tr_index[c] for c in ho]); tag = f"p{pi}cv{k}"; sigf = SIGDIR / f"{arm}_{tag}_seed{seed}.json"
                    if sigf.exists():  # resume after a guard abort
                        sj = json.load(open(sigf)); oof[ix] = np.array(sj["oof"]); fold_gain.append(float(oof[ix].mean()))
                        for g_k in gsum: gsum[g_k] += sj["gate"][g_k]
                        P(arm, seed, tag, "resumed", f"gain={fold_gain[-1]:+.6f}"); continue
                    losses = build_candidate(arm, seed, tr_c, tag)
                    sha = adapter_sha(); other_oof = {}
                    for other in SIGDIR.glob(f"{arm}_{tag}_seed*.json"):
                        o = json.load(open(other))
                        assert o["seed"] == seed or o["adapter_sha"] != sha, f"seed {seed} adapter identical to seed {o['seed']} at {arm}/{tag}"
                        if o["seed"] != seed: other_oof[o["seed"]] = o["oof"]
                    u = utility(ho, keep=D["keep"], zmin=D["zmin"]); zl = dict(ZLOG)
                    oof[ix] = u - u_inc_train[ix]; fold_gain.append(float(oof[ix].mean()))
                    cf = np.array([GATEFAIL[c] for c in ho]); inf_ = inc_fail_train[ix]
                    gstat = {"new_gatefails": int((cf & ~inf_).sum()), "recovered_gatefails": int((~cf & inf_).sum()), "inc_gatefails": int(inf_.sum())}
                    for g_k in gsum: gsum[g_k] += gstat[g_k]
                    dis = {sd_: int(sum(1 for p_, q_ in zip(oof[ix].tolist(), oo) if p_ != q_)) for sd_, oo in other_oof.items()}
                    json.dump({"seed": seed, "arm": arm, "partition": SALTS[pi], "fold": k, "adapter_sha": sha, "gain": fold_gain[-1], "oof": oof[ix].tolist(), "gate": gstat, "zlog": zl}, open(sigf, "w"))
                    P(arm, seed, tag, "gate", json.dumps(gstat), "changed", zl["n_changed"], "kept", zl["n_kept"], "disagree_vs_seeds", json.dumps(dis))
                    P(arm, seed, tag, f"gain={fold_gain[-1]:+.6f} loss0={losses[0]:.4f} lossN={losses[-1]:.4f} t={time.time()-t_arm:.0f}s mem={memavail_gb():.1f}G")
                m = ~np.isnan(oof); d = np.where(m, oof, 0.0)
                if a.probe: d = d[m]
                w = float((d > 0).sum()); t = float((d == 0).sum()); s_ = (w + 0.5 * t) / len(d)
                nf = ~inc_fail_train if not a.probe else ~inc_fail_train[m]; dn = d[nf]
                per_part.append({"partition": SALTS[pi], "oof_gain_mean": float(d.mean()), "fold_gains": fold_gain, "elo": float(elo(s_)),
                                 "wins": w, "ties": t, "losses": float((d < 0).sum()),
                                 "nonfloor_gain": float(dn.mean()) if len(dn) else 0.0, "nonfloor_elo": float(elo(((dn > 0).sum() + 0.5 * (dn == 0).sum()) / max(len(dn), 1))),
                                 "nonfloor_fold_gains": [float(oof[(np.array([fold[c] for c in train]) == k) & ~inc_fail_train & m].mean()) for k in range(a.folds if not a.probe else 1)]})
                OOF.append(d)
            # pooled over partitions: family bootstrap resamples a family with all of its rows in every partition copy
            if a.probe:
                lo = hi = elo_lo = elo_hi = nlo = nhi = 0.0; dall = np.concatenate(OOF); nfa = np.ones(len(dall), bool)
            else:
                dall = np.concatenate(OOF); nfa = np.concatenate([~inc_fail_train] * len(OOF))
                fidx = [np.concatenate([j * len(train) + tr_fam_idx[q] for j in range(len(OOF))]) for q in range(len(tr_fam_idx))]
                r = np.random.default_rng(seed + 7919); gs, es, ns = [], [], []
                for _ in range(2000):
                    ixb = np.concatenate([fidx[q] for q in r.integers(0, len(fidx), len(fidx))]); x = dall[ixb]; xn = x[nfa[ixb]]
                    gs.append(x.mean()); es.append(elo(((x > 0).sum() + 0.5 * (x == 0).sum()) / len(x)))
                    ns.append(elo(((xn > 0).sum() + 0.5 * (xn == 0).sum()) / max(len(xn), 1)))
                lo, hi = np.percentile(gs, [2.5, 97.5]).tolist(); elo_lo, elo_hi = np.percentile(es, [2.5, 97.5]).tolist(); nlo, nhi = np.percentile(ns, [2.5, 97.5]).tolist()
            dn = dall[nfa]; e_all = elo(((dall > 0).sum() + 0.5 * (dall == 0).sum()) / len(dall))
            e_nf = elo(((dn > 0).sum() + 0.5 * (dn == 0).sum()) / max(len(dn), 1))
            cv[arm] = {"per_partition": per_part, "oof_gain_mean": float(dall.mean()), "oof_gain_ci95_family_boot": [lo, hi],
                       "elo_vs_incumbent_oof": float(e_all), "elo_ci95_family_boot": [elo_lo, elo_hi],
                       "wins": float((dall > 0).sum()), "ties": float((dall == 0).sum()), "losses": float((dall < 0).sum()),
                       "nonfloor": {"n": int(len(dn)), "gain_mean": float(dn.mean()) if len(dn) else 0.0, "wins": int((dn > 0).sum()), "ties": int((dn == 0).sum()),
                                    "losses": int((dn < 0).sum()), "elo": float(e_nf), "elo_ci95_family_boot": [nlo, nhi]},
                       "mean_partition_nonfloor_elo": float(np.mean([pp["nonfloor_elo"] for pp in per_part])), "gate": gsum,
                       "passes_utility_strict": bool(dall.mean() > a.margin and lo > 0),
                       "passes_strict": bool(dall.mean() > a.margin and lo > 0 and e_all > 0 and elo_lo > 0 and nlo > 0 and all(pp["elo"] > 0 for pp in per_part)),
                       "wall_s": time.time() - t_arm}
            P(arm, seed, "CV", json.dumps(cv[arm]))
            # free cached bases once no later arm in the list uses them (memory)
            later = {DERIVED[x]["base"] for x in arm_list[ai + 1:]}
            for ck_ in [c_ for c_ in SUBCACHE if c_[0] == D["base"] and c_[0] not in later and c_[2] != "full"]: del SUBCACHE[ck_]
        sel = max(cv, key=lambda k_: (cv[k_]["mean_partition_nonfloor_elo"], cv[k_]["oof_gain_mean"]))  # H452: non-floor Elo over partitions
        if a.probe:  # smoke test never reads val91 for a candidate; exercise the full-train build path only
            build_candidate("pairnf_w3_z40", seed, train, "full"); u_p = utility(train[:64], zmin=0.40)
            P("PROBE_OK", seed, sel, "full_sha", adapter_sha(), "zlog_train64", json.dumps({k_: ZLOG[k_] for k_ in ("n_changed", "n_kept")}))
            results.append({"seed": seed, "cv": cv, "selected": sel}); continue
        # regression check: one val evaluation of the CV-selected config trained on all train contexts
        build_candidate(sel, seed, train, "full")
        fsha = adapter_sha()
        for other in SIGDIR.glob(f"{sel}_full_seed*.json"):
            o = json.load(open(other))
            assert o["seed"] == seed or o["adapter_sha"] != fsha, f"seed {seed} full adapter identical to seed {o['seed']}"
        uv = utility(dev, keep=DERIVED[sel]["keep"], zmin=DERIVED[sel]["zmin"]); dv = uv - u_inc; vz = dict(ZLOG)
        json.dump({"seed": seed, "arm": sel, "adapter_sha": fsha, "val_diff": dv.tolist()}, open(SIGDIR / f"{sel}_full_seed{seed}.json", "w"))
        vlo, vhi = boot_mean_ci(dv, seed=seed)
        vw = float((dv > 0).sum()); vt = float((dv == 0).sum()); vs = (vw + 0.5 * vt) / len(dv)
        vcf = np.array([GATEFAIL[c] for c in dev]); vif = inc_fail_dev
        reg = {"arm": sel, "val": float(uv.mean()), "val_gain": float(dv.mean()), "val_gain_ci95_family_boot": [vlo, vhi],
               "wins": vw, "ties": vt, "losses": float((dv < 0).sum()), "elo_vs_incumbent_val": float(elo(vs)),
               "not_worse": bool(dv.mean() >= 0), "adapter_sha": fsha,
               "new_gatefails": int((vcf & ~vif).sum()), "recovered_gatefails": int((~vcf & vif).sum()), "inc_gatefails": int(vif.sum()),
               "val_changed": vz["n_changed"], "val_kept": vz["n_kept"]}
        P("REGCHECK", seed, json.dumps(reg))
        results.append({"seed": seed, "cv": cv, "selected": sel, "selected_passes_strict": cv[sel]["passes_strict"], "regression_check": reg})
        json.dump({"results": results, "peak": PEAK}, open(a.out / "partial.json", "w"), indent=1)
    after = L.guard.snapshot_dope_weights(net)
    summary = {"wave": "H452", "partitions": SALTS, "dope_before": before.composite_sha256, "dope_after": after.composite_sha256,
               "parity_incumbent_val": float(u_inc.mean()), "parity_ok": parity, "incumbent_train_utility": float(u_inc_train.mean()),
               "n_train": len(train), "n_train_families": len(tr_ufam), "n_dev": len(dev), "n_dev_families": len(ufam),
               "family_overlap": 0, "fold_sizes": load, "config": vars(a) | {"out": str(a.out)}, "arms_cfg": ARMS, "derived_cfg": DERIVED, "curric_ess": ESSLOG, "peak": PEAK, "results": results}
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1, default=str)
    P("DONE dope_after", after.composite_sha256, "peak", PEAK)

if __name__ == "__main__":
    main()
