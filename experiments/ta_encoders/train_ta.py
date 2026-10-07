#!/usr/bin/env python3
"""Train TA_Hyperion_v8 (--model hv8) or TA_Dope_V2 (--model dv2) on every prepped outer-train context, self-supervised
(masked-y in-context regression + two-view InfoNCE). No GP-HWM label is read here. Every --eval-min minutes (and at step 0)
the 2,689 probe contexts are encoded (support = first <=384 stored rows; plus a within-context y-shuffled control encoding),
a checkpoint is written, and ta_probe.py scores the embeddings against the GP HWM on CPU (subprocess) -> timeseries.jsonl.
"""
import argparse, json, math, os, subprocess, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ta_common as C

EVAL_N = 384


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Store:
    def __init__(self, data, model, pcap):
        self.idx = json.load(open(os.path.join(data, "index.json")))["contexts"]
        self.X = np.load(os.path.join(data, "X.npy")); self.y = np.load(os.path.join(data, "y.npy")); self.miss = np.load(os.path.join(data, "miss.npy"))
        self.model, self.pcap = model, pcap
        self.pos = {r["cid"]: i for i, r in enumerate(self.idx)}
        if model in ("hv8", "hv8raw"):
            hi = json.load(open(os.path.join(data, "hv7rows_index.json")))
            self.R = np.load(os.path.join(data, "hv7rows.npy")); self.hpos = {r["cid"]: r for r in hi["contexts"]}
            self.hrec = hi["recipe"]

    def n(self, i):
        r = self.idx[i]
        if self.model in ("hv8", "hv8raw"): return self.hpos[r["cid"]]["n"]
        if self.model == "hv8ft": return min(r["n"], 512)       # same row pool the v7 cache used (first 512 stored rows)
        return r["n"]

    def get(self, i, rows):
        r = self.idx[i]; y = self.y[r["y_off"]: r["y_off"] + r["n"]][rows]
        X = self.X[r["x_off"]: r["x_off"] + r["n"] * r["p"]].reshape(r["n"], r["p"])
        if self.model in ("hv8", "hv8raw"):
            h = self.hpos[r["cid"]]
            Rr = self.R[h["off"]: h["off"] + h["n"]][rows].astype(np.float32)
            if self.model == "hv8": return Rr, y
            return Rr, y, X[rows][:, : C.PHV].astype(np.float32)
        if self.model == "hv8ft":
            return X[rows][:, : C.PHV].astype(np.float32), y
        return X[rows][:, : self.pcap].astype(np.float32), y, self.miss[r["m_off"]: r["m_off"] + r["p"]][: self.pcap]


def yraw_stats(y, binary):
    """Dope v1 target block on raw support y: mean, var, skew, q10, q50, q90, rare share, binary flag."""
    m, v = y.mean(), y.var(); sd = math.sqrt(v + 1e-12); sk = float(((y - m) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    q10, q25, q50, q75, q90 = np.percentile(y, [10, 25, 50, 75, 90]); iqr = q75 - q25
    rare = min(m, 1 - m) if binary else float(((y < q25 - 1.5 * iqr) | (y > q75 + 1.5 * iqr)).mean())
    z = lambda t: float(np.sign(t) * np.log1p(abs(t)))
    return [z(m), z(v), max(-20, min(20, sk)), z(q10), z(q50), z(q90), rare, float(binary)]


def make_batch(S, ids, rng, torch, dev, N, qfrac, eval_mode=False, yshuf=False):
    """Returns tensors with support/query masks. Train: random N rows, qfrac of them query (y hidden); support split into two
    disjoint halves for the two-view pass. Eval: first min(n, EVAL_N) stored rows, all support (no query)."""
    B = len(ids); P = 1
    items = []
    for i in ids:
        n = S.n(i)
        rows = np.arange(min(n, EVAL_N)) if eval_mode else np.sort(rng.choice(n, min(n, N), replace=False))
        items.append((i, rows, S.get(i, rows)))
    Nm = max(len(it[1]) for it in items)
    if S.model in ("dv2", "hv8ft"): P = max(it[2][0].shape[1] for it in items)
    if S.model == "hv8raw": P = C.PHV
    D = 512 if S.model in ("hv8", "hv8raw") else P
    X = np.zeros((B, Nm, D), np.float32); Y = np.zeros((B, Nm), np.float32); valid = np.zeros((B, Nm), bool)
    q = np.zeros((B, Nm), bool); half = np.zeros((B, Nm), np.int8); fv = np.zeros((B, max(P, 1)), bool); miss = np.zeros((B, max(P, 1)), np.float32)
    yst = np.zeros((B, 8), np.float32); Xr = np.zeros((B, Nm, C.PHV), np.float32) if S.model == "hv8raw" else np.zeros((1, 1, 1), np.float32)
    for b, (i, rows, g) in enumerate(items):
        n = len(rows); xb, yb = g[0], g[1].astype(np.float64)
        if yshuf:
            yb = yb[np.random.default_rng(zlib.crc32(S.idx[i]["cid"].encode())).permutation(n)]
        X[b, :n, : xb.shape[1]] = xb; valid[b, :n] = True
        if S.model == "dv2":
            fv[b, : xb.shape[1]] = True; miss[b, : xb.shape[1]] = g[2]
        if S.model == "hv8ft":
            fv[b, : xb.shape[1]] = True
        if S.model == "hv8raw":
            Xr[b, :n, : g[2].shape[1]] = g[2]; fv[b, : g[2].shape[1]] = True
        if not eval_mode:
            nq = max(4, int(round(qfrac * n))); perm = rng.permutation(n); q[b, perm[:nq]] = True
            half[b, perm[nq::2]] = 1; half[b, perm[nq + 1::2]] = 2
        else:
            half[b, :n] = 3
        sup = valid[b] & ~q[b]
        mu, sd = yb[sup[:n]].mean(), yb[sup[:n]].std() + 1e-8
        Y[b, :n] = np.clip((yb - mu) / sd, -10, 10)
        yst[b] = yraw_stats(yb[sup[:n]], S.idx[i]["binary"])
    T = lambda a, dt=None: torch.from_numpy(a).to(dev)
    return dict(X=T(X), y=T(Y), valid=T(valid), q=T(q), half=T(half), fv=T(fv), miss=T(miss), yst=T(yst), Xr=T(Xr))


def fwd(model, kind, bt, sup, R=None):
    if kind == "hv8":
        return model(bt["X"], bt["y"] * sup, sup, bt["valid"])
    if kind == "hv8raw":
        return model(bt["X"], bt["y"] * sup, sup, bt["valid"], bt["Xr"], bt["fv"])
    if kind == "hv8ft":
        return model(bt["X"], bt["y"] * sup, sup, bt["valid"], bt["fv"], R=R)
    return model(bt["X"], bt["y"] * sup, sup, bt["valid"], bt["fv"], bt["miss"], bt["yst"])


def restandardise(bt, sup, torch):
    """y standardised with the given support's own stats (so each view only ever sees its own support labels)."""
    m = sup.float(); n = m.sum(1).clamp(min=2)
    mu = (bt["y"] * m).sum(1) / n; sd = torch.sqrt((((bt["y"] - mu[:, None]) * m) ** 2).sum(1) / n + 1e-8)
    return ((bt["y"] - mu[:, None]) / sd[:, None]).clamp(-10, 10)


def info_nce(a, b, torch, t=0.1):
    import torch.nn.functional as F
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1); lg = a @ b.T / t
    lab = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hv8", "dv2", "hv8raw", "hv8ft"], required=True); ap.add_argument("--k-train", type=int, default=2); ap.add_argument("--trunk-lr", type=float, default=2e-5); ap.add_argument("--eval-batch", type=int, default=32); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--probe-cids", required=True); ap.add_argument("--probe-data", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--epochs", type=float, default=40); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-min", type=float, default=20); ap.add_argument("--pcap", type=int, default=128); ap.add_argument("--nmax", type=int, default=384)
    ap.add_argument("--max-hours", type=float, default=10); ap.add_argument("--probe-cache", default=""); ap.add_argument("--exclude", default="", help="json {cid: reason}: never sampled (sealed meta-test / sealed_excluded / unknown role)"); ap.add_argument("--probe-threads", type=int, default=6)
    ap.add_argument("--guard-siblings", default="", help="comma list of --model names whose train_ta processes don't count as busy")
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed); torch.set_num_threads(4); dev = "cuda"
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True); os.makedirs(os.path.join(a.out, "emb"), exist_ok=True)
    S = Store(a.data, a.model, a.pcap)
    probe = [c for c in json.load(open(a.probe_cids))]
    missing = [c for c in probe if c not in S.pos]
    if missing: raise SystemExit(f"{len(missing)} probe contexts have no prepped row data")
    pidx = [S.pos[c] for c in probe]
    # SSL validation split (objective monitoring only): 4% of datasets by hash, never trained on
    excl = json.load(open(a.exclude)) if a.exclude else {}
    bad = [c for c in probe if c in excl]
    if bad: raise SystemExit(f"{len(bad)} probe contexts are in the exclusion list")
    vds = {r["dataset_hash"] for r in S.idx if zlib.crc32(r["dataset_hash"].encode()) % 25 == 0}
    train_ids = np.array([i for i, r in enumerate(S.idx) if r["dataset_hash"] not in vds and r["cid"] not in excl])
    val_ids = np.array([i for i, r in enumerate(S.idx) if r["dataset_hash"] in vds and r["cid"] not in excl])
    if a.model == "hv8":
        from ta_hyperion_v8 import TAHyperionV8 as M
        model = M().to(dev)
    elif a.model == "hv8raw":
        from ta_hyperion_v8 import TAHyperionV8Raw as M
        model = M(raw_dim=C.PHV).to(dev)
    elif a.model == "hv8ft":
        sys.path.insert(0, "/home/ubuntu/scratch/jopedime-dl-training-v1/build-v2/HYPERION-V7/tools")
        import hyperion_v7_model as HV7
        V7 = "/home/ubuntu/scratch/jopedime-dl-training-v1/build-v2/HYPERION-V7/weights"
        trunk, v7cfg, v7rep = HV7.load_v7(os.path.join(V7, "model.safetensors"), os.path.join(V7, "manifest.json"), device=dev)
        from ta_hyperion_v8 import TAHyperionV8FT as M
        model = M(trunk, v7cfg, k_train=a.k_train).to(dev)
    else:
        from ta_dope_v2 import TADopeV2 as M
        model = M().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    steps_per_epoch = int(math.ceil(len(train_ids) / a.batch)); total = int(a.epochs * steps_per_epoch)
    if a.model == "hv8ft":
        model.trunk.train()
        groups = [{"params": model.head_params(), "lr": a.lr}, {"params": model.trunk_params(), "lr": a.trunk_lr}]
    else:
        groups = [{"params": list(model.parameters()), "lr": a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, total))))))
    run = {"model": {"hv8": "TA_Hyperion_v8", "dv2": "TA_Dope_V2", "hv8raw": "TA_Hyperion_v8b-raw", "hv8ft": "TA_Hyperion_v8b-ft"}[a.model], "params": nparam, "gpu": C.gpu_info(), "host": os.uname().nodename,
           "torch": torch.__version__, "train_contexts": int(len(train_ids)), "excluded_contexts": len(excl), "ssl_val_contexts": int(len(val_ids)), "probe_contexts": len(pidx),
           "steps_per_epoch": steps_per_epoch, "total_steps": total, "args": vars(a), "amp": "bf16 autocast",
           "hv7_recipe": getattr(S, "hrec", None),
           "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           "ft": ({"trained_blocks": model.trained_blocks, "k_train": a.k_train, "trunk_lr": a.trunk_lr, "head_lr": a.lr, "init": "pinned hyperion v7 (weights sha256 05c17edd...)",
                   "trunk_trainable_params": int(sum(p.numel() for p in model.trunk_params()))} if a.model == "hv8ft" else None), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    log("run", json.dumps({k: v for k, v in run.items() if k != "hv7_recipe"}))
    guard = C.Guard(log, min_free_mib=2500, siblings=tuple(f"--model {m} " for m in a.guard_siblings.split(",") if m))
    rng = np.random.default_rng(1000 + a.seed)
    vrng_seed = 777

    def encode_probe_both():
        model.eval(); out, outs = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(pidx), a.eval_batch):
                bt = make_batch(S, pidx[s:s + a.eval_batch], None, torch, dev, EVAL_N, 0, eval_mode=True, yshuf=False)
                bs = make_batch(S, pidx[s:s + a.eval_batch], None, torch, dev, EVAL_N, 0, eval_mode=True, yshuf=True)
                R = model.rows_embed(bt["X"], bt["valid"], bt["fv"]) if a.model == "hv8ft" else None
                _, ctx, _ = fwd(model, a.model, bt, bt["valid"], R=R); out.append(ctx.float().cpu().numpy())
                _, ctx, _ = fwd(model, a.model, bs, bs["valid"], R=R); outs.append(ctx.float().cpu().numpy())
        model.train(); return np.concatenate(out), np.concatenate(outs)

    def ssl_val():
        model.eval(); vr = np.random.default_rng(vrng_seed); tot, cnt = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(val_ids), a.batch):
                bt = make_batch(S, val_ids[s:s + a.batch], vr, torch, dev, a.nmax, 0.25)
                sup = bt["valid"] & ~bt["q"]; bt["y"] = restandardise(bt, sup, torch)
                yh, _, _ = fwd(model, a.model, bt, sup)
                e = ((yh.float() - bt["y"]) ** 2)[bt["q"]]; tot += float(e.sum()); cnt += int(e.numel())
        model.train(); return tot / max(cnt, 1)

    def checkpoint(step, ep, tr_loss):
        t0 = time.time()
        E, Es = encode_probe_both(); vl = ssl_val()
        tag = f"s{step:07d}"
        np.save(os.path.join(a.out, "emb", f"E_{tag}.npy"), E.astype(np.float32)); np.save(os.path.join(a.out, "emb", f"Eys_{tag}.npy"), Es.astype(np.float32))
        torch.save({"model": model.state_dict(), "step": step, "run": run}, os.path.join(a.out, "ckpt", f"{tag}.pt"))
        meta = {"step": step, "epoch": ep, "train_loss": tr_loss, "ssl_val_mse": vl, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "encode_s": time.time() - t0, "guard_paused_s": guard.paused_s}
        json.dump(meta, open(os.path.join(a.out, "emb", f"meta_{tag}.json"), "w"))
        env = dict(os.environ, OMP_NUM_THREADS=str(a.probe_threads), OPENBLAS_NUM_THREADS=str(a.probe_threads), CUDA_VISIBLE_DEVICES="")
        subprocess.Popen(["nice", "-n", "15", sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ta_probe.py"), "--probe-data", a.probe_data,
                          "--emb", os.path.join(a.out, "emb", f"E_{tag}.npy"), "--emb-yshuf", os.path.join(a.out, "emb", f"Eys_{tag}.npy"),
                          "--meta", os.path.join(a.out, "emb", f"meta_{tag}.json"), "--model", run["model"], "--out", os.path.join(a.out, "timeseries.jsonl"),
                          "--cache", a.probe_cache or os.path.join(a.out, "probe_cache")], env=env, stdout=open(os.path.join(a.out, "probe.log"), "a"), stderr=subprocess.STDOUT)
        log("checkpoint", tag, json.dumps(meta))

    step, t_start, t_last_eval = 0, time.time(), time.time()
    checkpoint(0, 0.0, None)
    ema, ema_c, ema_n = None, None, 0; tok_t = time.time(); seen = 0
    while step < total and time.time() - t_start < a.max_hours * 3600:
        perm = rng.permutation(train_ids)
        for s in range(0, len(perm), a.batch):
            ts = time.time()
            N = int(rng.integers(64, a.nmax + 1))
            bt = make_batch(S, perm[s:s + a.batch], rng, torch, dev, N, 0.25)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, ctxs = [], []
                Rft = model.rows_embed(bt["X"], bt["valid"], bt["fv"]) if a.model == "hv8ft" else None
                for hv in (1, 2):
                    sup = bt["valid"] & (bt["half"] == hv)
                    btv = dict(bt); btv["y"] = restandardise(bt, sup, torch)
                    yh, ctx, _ = fwd(model, a.model, btv, sup, R=Rft)
                    losses.append(((yh.float() - btv["y"]) ** 2)[bt["q"]].mean()); ctxs.append(model.cproj(ctx).float())
                mse = 0.5 * (losses[0] + losses[1]); nce = info_nce(ctxs[0], ctxs[1], torch)
                loss = mse + 0.1 * nce
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1; seen += len(bt["X"])
            lv = float(mse); ema = lv if ema is None else 0.98 * ema + 0.02 * lv; nv = float(nce); ema_c = nv if ema_c is None else 0.98 * ema_c + 0.02 * nv
            guard.step(time.time() - ts)
            if step % 100 == 0:
                rate = seen / (time.time() - tok_t)
                log(f"step {step}/{total} ep {step / steps_per_epoch:.2f} mse_ema {ema:.4f} nce_ema {ema_c:.3f} lr {sched.get_last_lr()[0]:.2e} ctx/s {rate:.1f} "
                    f"epoch_min {len(train_ids) / max(rate, 1e-9) / 60:.1f} mem_mib {torch.cuda.max_memory_allocated() >> 20} paused_s {guard.paused_s:.0f}")
                json.dump({"step": step, "total": total, "epoch": step / steps_per_epoch, "mse_ema": ema, "nce_ema": ema_c, "ctx_per_s": rate,
                           "epoch_min": len(train_ids) / max(rate, 1e-9) / 60, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "eta_s": (total - step) / max(rate / a.batch, 1e-9), "paused_s": guard.paused_s, "gpu": run["gpu"], "host": run["host"]},
                          open(os.path.join(a.out, "progress.json"), "w"))
                seen, tok_t = 0, time.time()
            if time.time() - t_last_eval > a.eval_min * 60:
                checkpoint(step, step / steps_per_epoch, ema); t_last_eval = time.time()
            if step >= total: break
    checkpoint(step, step / steps_per_epoch, ema)
    log("DONE", step)


if __name__ == "__main__":
    main()
