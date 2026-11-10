#!/usr/bin/env python3
"""H419 trainer for APEX-ICL on the synthetic prior (icl_prior.py). Reads no data file.

  icl_train.py train --size S|M --out DIR --hours H [--lr 2e-4] [--seed 1] [--tokens 600000] [--bmax 64]
                     [--max-rows 0] [--resume]
  icl_train.py selftest

The schedule is a function of elapsed training time / --hours (so a run ends on time whatever the throughput):
warmup 2%, stable, cosine decay over the last 15% to 5% of the peak. Row curriculum by the same fraction:
max rows 384 -> 640 (25%) -> 1,024 (50%) -> 1,536 (85%). Checkpoints: ckpt_last.pt every 20 minutes (atomic replace)
and ckpt_hNNN.pt every 2 hours, each with raw and EMA weights, the optimizer and the elapsed time.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

from icl_model import ICL, n_params
import icl_inject as INJ
from icl_prior import KINDS, Prior, draw_features

ROW_BUCKETS = (384, 640, 1024, 1536)
EMA = 0.9995
VAL_SHAPES = ((512, 6), (512, 20), (1024, 10), (1024, 48), (1536, 8), (1536, 80))
VAL_SEED = 419_000_001


def lr_factor(frac):
    if frac < 0.02:
        return frac / 0.02
    if frac < 0.85:
        return 1.0
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - 0.85) / 0.15)))


def max_rows(frac):
    return ROW_BUCKETS[0] if frac < 0.25 else ROW_BUCKETS[1] if frac < 0.5 else ROW_BUCKETS[2] if frac < 0.85 else ROW_BUCKETS[3]


def draw_shape(rng, frac, tokens, bmax, cap_rows=0, row_tokens=0):
    top = max_rows(frac) if not cap_rows else min(cap_rows, max_rows(frac))
    opts = [n for n in ROW_BUCKETS if n <= top]
    N = opts[-1] if rng.random() < 0.5 else rng.choice(opts)
    Fn = draw_features(rng)
    B = max(2, min(bmax, tokens // (N * (Fn + 4)), (row_tokens // N) if row_tokens else bmax))
    hold = max(64, N // 8)
    n_ctx = int(round(math.exp(rng.uniform(math.log(96), math.log(N - hold)))))
    return B, N, Fn, n_ctx


def ridge_u(x, y, n_ctx):
    """Reference: u of a ridge fit on standardised, zero-imputed features (same context/query rows)."""
    x = torch.nan_to_num(x.double())
    xc, xq = x[:, :n_ctx], x[:, n_ctx:]
    mu, sd = xc.mean(1, keepdim=True), xc.std(1, keepdim=True).clamp_min(1e-9)
    xc, xq = (xc - mu) / sd, (xq - mu) / sd
    yc = y[:, :n_ctx].double()
    ym = yc.mean(1, keepdim=True)
    A = xc.transpose(1, 2) @ xc + 1.0 * torch.eye(x.shape[-1], device=x.device, dtype=torch.float64)
    w = torch.linalg.solve(A, xc.transpose(1, 2) @ (yc - ym).unsqueeze(-1))
    pred = (xq @ w).squeeze(-1) + ym
    return u_score(pred, y[:, n_ctx:].double(), ym)


def u_score(pred, yq, ym):
    """Per task: -ln(RMSE / RMSE of the context-mean predictor), clipped to [-2, 6]."""
    r = torch.sqrt(((pred - yq) ** 2).mean(1))
    r0 = torch.sqrt(((ym - yq) ** 2).mean(1)).clamp_min(1e-30)
    return (-torch.log((r / r0).clamp_min(1e-30))).clamp(-2, 6)


@torch.no_grad()
def evaluate(model, dev, n_batches=2, bsz=16, real_val=None):
    model.eval()
    prior = Prior(dev, VAL_SEED)
    rows = []
    for N, Fn in VAL_SHAPES:
        for _ in range(n_batches):
            x, y, kinds = prior.sample(bsz, N, Fn)
            n_ctx = N * 3 // 4
            logits, ym, ys = model(x, y[:, :n_ctx], n_ctx)
            loss = model.loss(logits, y[:, n_ctx:], ym, ys).mean(1)
            u = u_score(model.mean(logits, ym, ys).double(), y[:, n_ctx:].double(), ym.double())
            ur = ridge_u(x, y, n_ctx)
            for i in range(bsz):
                rows.append((KINDS[int(kinds[i])], float(loss[i]), float(u[i]), float(ur[i])))
    # the same shapes with injected columns in 20-60% of the feature slots (robustness to injection)
    prior2, rng2, inj = Prior(dev, VAL_SEED + 1), random.Random(VAL_SEED), []
    for N, Fn in VAL_SHAPES:
        x, y, _k, _n = INJ.sample_synth(prior2, rng2, bsz, N, Fn, p=1.0)
        n_ctx = N * 3 // 4
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx)
        u = u_score(model.mean(logits, ym, ys).double(), y[:, n_ctx:].double(), ym.double())
        inj.extend(zip(u.tolist(), ridge_u(x, y, n_ctx).tolist()))
    # real FIT-SEL SEARCH rows: context = folds 1-4, query = fold 0 (one task per forward, no padding)
    real = []
    for m, n_ctx in (real_val.items if real_val is not None else []):
        x, y = m[None, :, :-1], m[None, :, -1]
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx)
        u = u_score(model.mean(logits, ym, ys).double(), y[:, n_ctx:].double(), ym.double())
        real.append((float(u), float(ridge_u(x, y, n_ctx)), x.shape[-1]))
    model.train()
    out = {"loss": sum(r[1] for r in rows) / len(rows), "u": sum(r[2] for r in rows) / len(rows),
           "u_ridge": sum(r[3] for r in rows) / len(rows), "win_vs_ridge": sum(r[2] > r[3] for r in rows) / len(rows)}
    out.update(u_inj=sum(a for a, _ in inj) / len(inj), u_ridge_inj=sum(b for _, b in inj) / len(inj))
    if real:
        nar = [r for r in real if r[2] <= 100] or real
        out.update(real_n=len(real), real_u=sum(r[0] for r in real) / len(real),
                   real_u_ridge=sum(r[1] for r in real) / len(real),
                   real_win_vs_ridge=sum(r[0] > r[1] for r in real) / len(real),
                   real_u_le100=sum(r[0] for r in nar) / len(nar), real_u_ridge_le100=sum(r[1] for r in nar) / len(nar),
                   real_u_median=sorted(r[0] for r in real)[len(real) // 2])
    for k in KINDS:
        sel = [r for r in rows if r[0] == k]
        if sel:
            out["u_" + k] = sum(r[2] for r in sel) / len(sel)
            out["u_ridge_" + k] = sum(r[3] for r in sel) / len(sel)
    return out


def save(path, model, ema, opt, state):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "state": state,
                "cfg": model.cfg}, tmp)
    os.replace(tmp, path)


def code_sha():
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("icl_model.py", "icl_prior.py", "icl_inject.py", "icl_train.py"):
        h.update(open(os.path.join(here, f), "rb").read())
    return h.hexdigest()


def train(a):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    last = os.path.join(a.out, "ckpt_last.pt")
    torch.manual_seed(a.seed)
    model = ICL(a.size).to(dev)
    ema = ICL(a.size).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    state = {"step": 0, "elapsed": 0.0, "datasets": 0, "next_named": 2.0}
    init_sha = None
    if a.clone and not (a.resume and os.path.exists(last)):   # continue another run: weights, EMA, optimizer, schedule
        ck = torch.load(a.clone, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        state = dict(ck["state"])
        init_sha = hashlib.sha256(open(a.clone, "rb").read()).hexdigest()
    if a.init and not (a.resume and os.path.exists(last)):
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        init_sha = hashlib.sha256(open(a.init, "rb").read()).hexdigest()
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        state = ck["state"]
    elif os.path.exists(os.path.join(a.out, "config.json")):
        sys.exit(f"{a.out} holds a run; pass --resume or a new --out")
    else:
        cfg = dict(vars(a), params=n_params(model), code_sha256=code_sha(), torch=torch.__version__, init_sha256=init_sha,
                   gpu=torch.cuda.get_device_name(0), model=model.cfg, started=time.strftime("%FT%TZ", time.gmtime()))
        cfg.pop("fn", None)
        with open(os.path.join(a.out, "config.json"), "x") as f:
            json.dump(cfg, f, indent=1)
    rng = random.Random(a.seed * 1_000_003 + state["step"])
    prior = Prior(dev, a.seed * 7_919 + state["step"])
    bank = INJ.RealBank(a.real_bank, dev, prior) if a.real_bank else None
    rval = INJ.RealVal(a.val_real, dev) if a.val_real else None
    log = open(os.path.join(a.out, "log.jsonl"), "a", buffering=1)
    total = a.hours * 3600.0
    t_last = time.time()
    t_ck = t_eval = t_log = t_last
    run_loss, run_n, run_ds, run_real, run_inj = 0.0, 0, 0, 0, 0
    model.train()
    sync_k = int(time.time() // a.sync_s) if a.sync_dir else 0
    while state["elapsed"] < total:
        frac = state["elapsed"] / total
        B, N, Fn, n_ctx = draw_shape(rng, frac, a.tokens, a.bmax, a.max_rows, a.row_tokens)
        got = None
        if bank is not None and rng.random() < a.real_frac * min(1.0, state["elapsed"] / max(1.0, a.real_ramp_s)):
            got = bank.sample(rng, min(B, a.real_bmax), N, Fn)
        if got is not None:
            x, y, _modes, _share = got
            B = x.shape[0]
            run_real += 1
        else:
            x, y, _kinds, k_inj = INJ.sample_synth(prior, rng, B, N, Fn, a.inject_p)
            run_inj += k_inj > 0
        slots = torch.randperm(256, device=dev)[:Fn]
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_factor(frac)
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx, slots=slots)
        loss = model.loss(logits, y[:, n_ctx:], ym, ys).mean()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            log.write(json.dumps({"step": state["step"], "skip": "non-finite loss", "shape": [B, N, Fn, n_ctx]}) + "\n")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(EMA, (1 + state["step"]) / (10 + state["step"]))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        now = time.time()
        state["elapsed"] += now - t_last
        t_last = now
        state["step"] += 1
        state["datasets"] += B
        run_loss += float(loss.detach()); run_n += 1; run_ds += B
        if a.sync_dir and int(now // a.sync_s) > sync_k:
            # parameter averaging round (addendum 3): publish my weights, wait for the group mean, load it.
            sync_k = int(now // a.sync_s)
            os.makedirs(a.sync_dir, exist_ok=True)
            po = os.path.join(a.sync_dir, f"out_{a.name}")
            torch.save({"model": model.state_dict()}, po + ".pt.tmp"); os.replace(po + ".pt.tmp", po + ".pt")
            with open(po + ".json.tmp", "w") as f:
                json.dump({"round": sync_k, "step": state["step"], "elapsed": state["elapsed"]}, f)
            os.replace(po + ".json.tmp", po + ".json")
            pi, got, t_w = os.path.join(a.sync_dir, "in"), None, time.time()
            while time.time() - t_w < a.sync_wait:
                try:
                    meta = json.load(open(pi + ".json"))
                    if meta.get("round") == sync_k:
                        model.load_state_dict(torch.load(pi + ".pt", map_location=dev, weights_only=False)["model"])
                        got = meta
                        break
                    if meta.get("round", 0) > sync_k:
                        break
                except (OSError, ValueError, KeyError, RuntimeError):
                    pass
                time.sleep(2.0)
            log.write(json.dumps({"sync": sync_k, "step": state["step"], "h": round(state["elapsed"] / 3600, 4),
                                  "wait_s": round(time.time() - t_w, 1), "loaded": got is not None,
                                  "n": (got or {}).get("n"), "dist": (got or {}).get("dist"),
                                  "loss_before": round(run_loss / max(1, run_n), 5)}) + "\n")
            now = t_last = time.time()
        if now - t_log >= a.log_s:
            log.write(json.dumps({"step": state["step"], "h": round(state["elapsed"] / 3600, 4),
                                  "loss": round(run_loss / run_n, 5), "ds_per_s": round(run_ds / (now - t_log), 2),
                                  "lr": opt.param_groups[0]["lr"], "datasets": state["datasets"],
                                  "real_share": round(run_real / run_n, 3), "inj_share": round(run_inj / run_n, 3),
                                  "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}) + "\n")
            run_loss, run_n, run_ds, run_real, run_inj, t_log = 0.0, 0, 0, 0, 0, now
        if now - t_eval >= a.eval_s:
            ev = evaluate(ema, dev, real_val=rval)
            ev.update(step=state["step"], h=round(state["elapsed"] / 3600, 4), eval="ema")
            log.write(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in ev.items()}) + "\n")
            t_eval = t_last = time.time()
        if now - t_ck >= a.ckpt_s:
            save(last, model, ema, opt, state)
            if state["elapsed"] / 3600 >= state["next_named"]:
                save(os.path.join(a.out, "ckpt_h%03d.pt" % int(state["next_named"])), model, ema, opt, state)
                state["next_named"] += a.named_h
            t_ck = t_last = time.time()
    ev = evaluate(ema, dev, n_batches=6, real_val=rval)
    ev.update(step=state["step"], h=round(state["elapsed"] / 3600, 4), eval="ema_final")
    log.write(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in ev.items()}) + "\n")
    save(last, model, ema, opt, state)
    save(os.path.join(a.out, "ckpt_final.pt"), model, ema, opt, state)
    with open(os.path.join(a.out, "DONE.json"), "x") as f:
        json.dump({"state": state, "final_eval": ev}, f, indent=1)


def selftest(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for size in ("S", "M"):
        print(size, "params", n_params(ICL(size)))
    torch.manual_seed(0)
    m = ICL("S").to(dev).eval()
    p = Prior(dev, 5)
    x, y, k = p.sample(8, 384, 7)
    assert torch.isfinite(y).all() and x.shape == (8, 384, 7), "prior shapes"
    n_ctx = 300
    with torch.no_grad():
        l0, ym, ys = m(x, y[:, :n_ctx], n_ctx, amp=False)
        x2 = x.clone(); x2[:, n_ctx + 3] += 1.0                  # perturb one query row
        l1, _, _ = m(x2, y[:, :n_ctx], n_ctx, amp=False)
        d = (l0 - l1).abs().amax(-1)                             # [B,Nq]
        other = torch.cat([d[:, :3], d[:, 4:]], 1).max().item()
        assert d[:, 3].min().item() > 0 and other < 1e-4, f"query independence: other rows moved {other}"
        x3 = x.clone(); x3[:, 5] += 1.0                          # perturb one context row: every query may move
        l2, _, _ = m(x3, y[:, :n_ctx], n_ctx, amp=False)
        assert (l0 - l2).abs().max().item() > 0
        for kind in range(len(KINDS)):
            xx, yy, _ = p.sample(4, 384, 12, kinds=torch.full((4,), kind, device=dev))
            assert torch.isfinite(yy).all() and yy.std(1).min() > 0, KINDS[kind]
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    first = None
    for i in range(60 if dev.type == "cuda" else 6):
        x, y, _ = p.sample(16, 384, 8)
        lg, ym, ys = m(x, y[:, :288], 288)
        loss = m.loss(lg, y[:, 288:], ym, ys).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else float(loss)
    print("loss first", round(first, 4), "last", round(float(loss), 4))
    assert float(loss) < first, "loss did not fall"
    print("SELFTEST_OK", dev)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--size", default="S"); t.add_argument("--out", required=True)
    t.add_argument("--hours", type=float, required=True); t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--seed", type=int, default=1); t.add_argument("--tokens", type=int, default=600_000)
    t.add_argument("--bmax", type=int, default=64); t.add_argument("--max-rows", type=int, default=0)
    t.add_argument("--row-tokens", type=int, default=0, help="cap on batch x rows (trunk memory); 0 = none")
    t.add_argument("--init", help="checkpoint whose raw and EMA weights start the run (fresh optimizer and schedule)")
    t.add_argument("--real-bank", help="icl_bank.py fit output (FIT-ICL tables)")
    t.add_argument("--real-frac", type=float, default=0.4); t.add_argument("--real-ramp-s", type=float, default=3600)
    t.add_argument("--real-bmax", type=int, default=32); t.add_argument("--inject-p", type=float, default=0.5)
    t.add_argument("--val-real", help="icl_bank.py val output (FIT-SEL SEARCH rows, monitor only)")
    t.add_argument("--clone", help="checkpoint to continue: weights, EMA, optimizer and schedule position")
    t.add_argument("--name", default="w0"); t.add_argument("--sync-dir", help="parameter-averaging exchange directory")
    t.add_argument("--sync-s", type=float, default=300.0); t.add_argument("--sync-wait", type=float, default=240.0)
    t.add_argument("--ckpt-s", type=float, default=1200.0); t.add_argument("--named-h", type=float, default=2.0)
    t.add_argument("--log-s", type=float, default=60); t.add_argument("--eval-s", type=float, default=1800)
    t.add_argument("--resume", action="store_true"); t.set_defaults(fn=train)
    s = sub.add_parser("selftest"); s.set_defaults(fn=selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
