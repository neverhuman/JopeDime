#!/usr/bin/env python3
"""H431 (CW-3 Part B): H419 trainer extended with Factors A-E (all default-off).

Factors (CW3_CURRICULUM.md section 3 Part C):
  A structure:           --prior v3s (uses icl_prior_v3s.PROBS_V3S)
  B noise regularisers:  --aug-cell-drop 0.05, --aug-jitter 0.01, --dropout 0.05
  C hard-batch replay:   --replay 0.3 (512-entry seed buffer of worst model/ridge query RMSE ratio; max 3 replays)
  D stronger base:       --p-resid-strong 0.5 (half of residual batches use strong_base.strong_residual_targets)
  E two-view consistency:--consist 0.1 (symmetric KL between two random column-slot forward passes on half of batches)

With all new flags at their defaults (--prior v1|v2, --aug-cell-drop 0, --aug-jitter 0, --dropout 0, --replay 0,
--p-resid-strong 0, --consist 0), batch generation and training steps are byte-for-byte identical to r9/code/icl_train.py.
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
import torch.nn.functional as F

from icl_model_cw3 import ICL, n_params
import icl_inject as INJ
import icl_resid as RES
from icl_prior_v3s import KINDS, KINDS_V3S, PROBS_SIMPLE, PROBS_V2, PROBS_V3S, Prior, draw_features
import strong_base as SB

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


def draw_shape(rng, frac, tokens, bmax, cap_rows=0, row_tokens=0, row_floor=0):
    top = max(max_rows(frac), row_floor)
    top = top if not cap_rows else min(cap_rows, top)
    opts = [n for n in ROW_BUCKETS if n <= top]
    N = opts[-1] if rng.random() < 0.5 else rng.choice(opts)
    Fn = draw_features(rng)
    B = max(2, min(bmax, tokens // (N * (Fn + 4)), (row_tokens // N) if row_tokens else bmax))
    hold = max(64, N // 8)
    n_ctx = int(round(math.exp(rng.uniform(math.log(96), math.log(N - hold)))))
    return B, N, Fn, n_ctx


def _c(t):
    return t.detach().cpu().double()


def _prior_probs(name: str):
    if name == "v2":
        return PROBS_V2
    if name == "v3s":
        return PROBS_V3S
    return None


def deterministic_synth_batch(device, batch_seed: int, B: int, N: int, Fn: int, inject_p: float, prior_name: str):
    """Pure function of (device, batch_seed, B, N, Fn, inject_p, prior_name) for Factor C hard-batch replay."""
    p = Prior(device, batch_seed * 131 + 17)
    r = random.Random(batch_seed * 997 + 31)
    return INJ.sample_synth(p, r, B, N, Fn, inject_p, _prior_probs(prior_name))


def augment_features(x: torch.Tensor, n_ctx: int, cell_drop: float, jitter: float, gen: torch.Generator) -> torch.Tensor:
    """Factor B input noise regularisers (fitted from context IQR only)."""
    out = x.clone()
    B, N, Fn = out.shape
    if jitter > 0.0:
        miss = ~torch.isfinite(out[:, :n_ctx])
        xc = torch.where(miss, torch.nan, out[:, :n_ctx].float())
        q = torch.nanquantile(xc, torch.tensor([0.25, 0.75], device=out.device), dim=1)
        iqr = torch.nan_to_num(q[1] - q[0], nan=0.0, posinf=0.0, neginf=0.0)
        eps = torch.randn((B, N, Fn), generator=gen, device=out.device, dtype=out.dtype)
        out = out + eps * (jitter * iqr[:, None, :]).to(out.dtype)
    if cell_drop > 0.0:
        drop_mask = torch.rand((B, N, Fn), generator=gen, device=out.device) < cell_drop
        out = torch.where(drop_mask, torch.tensor(float("nan"), dtype=out.dtype, device=out.device), out)
    return out


def symmetric_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """Factor E two-view symmetric KL divergence on bin distributions."""
    lp1, lp2 = F.log_softmax(logits1, dim=-1), F.log_softmax(logits2, dim=-1)
    p1, p2 = lp1.exp(), lp2.exp()
    return 0.5 * ((p1 * (lp1 - lp2)).sum(dim=-1) + (p2 * (lp2 - lp1)).sum(dim=-1)).mean()


def ridge_u(x, y, n_ctx):
    x = torch.nan_to_num(_c(x))
    xc, xq = x[:, :n_ctx], x[:, n_ctx:]
    mu, sd = xc.mean(1, keepdim=True), xc.std(1, keepdim=True).clamp_min(1e-9)
    xc, xq = (xc - mu) / sd, (xq - mu) / sd
    yc = _c(y[:, :n_ctx])
    ym = yc.mean(1, keepdim=True)
    A = xc.transpose(1, 2) @ xc + 1.0 * torch.eye(x.shape[-1], device=x.device, dtype=torch.float64)
    w = torch.linalg.solve(A, xc.transpose(1, 2) @ (yc - ym).unsqueeze(-1))
    pred = (xq @ w).squeeze(-1) + ym
    return u_score(pred, _c(y[:, n_ctx:]), ym)


def u_score(pred, yq, ym):
    r = torch.sqrt(((pred - yq) ** 2).mean(1))
    r0 = torch.sqrt(((ym - yq) ** 2).mean(1)).clamp_min(1e-30)
    return (-torch.log((r / r0).clamp_min(1e-30))).clamp(-2, 6)


@torch.no_grad()
def resid_u(model, x, y, n_ctx, seed):
    yr, used, base_q = RES.residual_targets(x, y, n_ctx, torch.Generator().manual_seed(seed))
    logits, ym, ys = model(x, yr[:, :n_ctx], n_ctx)
    pred = _c(model.mean(logits, ym, ys)) + torch.where(_c(used)[:, None] > 0, base_q.cpu(), torch.zeros_like(base_q.cpu()))
    return u_score(pred, _c(y[:, n_ctx:]), _c(y[:, :n_ctx]).mean(1, keepdim=True))


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
            u = u_score(_c(model.mean(logits, ym, ys)), _c(y[:, n_ctx:]), _c(ym))
            ur = ridge_u(x, y, n_ctx)
            ures = resid_u(model, x, y, n_ctx, VAL_SEED + len(rows))
            for i in range(bsz):
                rows.append((KINDS[int(kinds[i])], float(loss[i]), float(u[i]), float(ur[i]), float(ures[i])))
    prior2, rng2, inj = Prior(dev, VAL_SEED + 1), random.Random(VAL_SEED), []
    for N, Fn in VAL_SHAPES:
        x, y, _k, _n = INJ.sample_synth(prior2, rng2, bsz, N, Fn, p=1.0)
        n_ctx = N * 3 // 4
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx)
        u = u_score(_c(model.mean(logits, ym, ys)), _c(y[:, n_ctx:]), _c(ym))
        inj.extend(zip(u.tolist(), ridge_u(x, y, n_ctx).tolist()))
    prior3, simple = Prior(dev, VAL_SEED + 2), []
    for N, Fn in VAL_SHAPES:
        x, y, kinds = prior3.sample(bsz, N, Fn, probs=PROBS_SIMPLE)
        n_ctx = N * 3 // 4
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx)
        u = u_score(_c(model.mean(logits, ym, ys)), _c(y[:, n_ctx:]), _c(ym))
        ur, ures = ridge_u(x, y, n_ctx), resid_u(model, x, y, n_ctx, VAL_SEED + 7 * N + Fn)
        simple.extend((KINDS[int(kinds[i])], float(u[i]), float(ur[i]), float(ures[i])) for i in range(bsz))
    real = []
    for i, (m, n_ctx) in enumerate(real_val.items if real_val is not None else []):
        x, y = m[None, :, :-1], m[None, :, -1]
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx)
        u = u_score(_c(model.mean(logits, ym, ys)), _c(y[:, n_ctx:]), _c(ym))
        real.append((float(u), float(ridge_u(x, y, n_ctx)), x.shape[-1], float(resid_u(model, x, y, n_ctx, 419_500 + i))))
    model.train()
    out = {"loss": sum(r[1] for r in rows) / len(rows), "u": sum(r[2] for r in rows) / len(rows),
           "u_ridge": sum(r[3] for r in rows) / len(rows), "win_vs_ridge": sum(r[2] > r[3] for r in rows) / len(rows),
           "u_res": sum(r[4] for r in rows) / len(rows), "win_res_vs_ridge": sum(r[4] > r[3] for r in rows) / len(rows)}
    out.update(u_inj=sum(a for a, _ in inj) / len(inj), u_ridge_inj=sum(b for _, b in inj) / len(inj))
    out.update(u_simple=sum(r[1] for r in simple) / len(simple), u_ridge_simple=sum(r[2] for r in simple) / len(simple),
               u_res_simple=sum(r[3] for r in simple) / len(simple),
               win_simple_vs_ridge=sum(r[1] > r[2] for r in simple) / len(simple))
    for k in ("lin", "add", "addint"):
        sel = [r for r in simple if r[0] == k]
        if sel:
            out["u_" + k] = sum(r[1] for r in sel) / len(sel)
            out["u_ridge_" + k] = sum(r[2] for r in sel) / len(sel)
            out["u_res_" + k] = sum(r[3] for r in sel) / len(sel)
    if real:
        nar = [r for r in real if r[2] <= 100] or real
        out.update(real_n=len(real), real_u=sum(r[0] for r in real) / len(real),
                   real_u_ridge=sum(r[1] for r in real) / len(real),
                   real_win_vs_ridge=sum(r[0] > r[1] for r in real) / len(real),
                   real_u_le100=sum(r[0] for r in nar) / len(nar), real_u_ridge_le100=sum(r[1] for r in nar) / len(nar),
                   real_u_median=sorted(r[0] for r in real)[len(real) // 2],
                   real_res_u=sum(r[3] for r in real) / len(real),
                   real_res_win_vs_ridge=sum(r[3] > r[1] for r in real) / len(real),
                   real_res_u_median=sorted(r[3] for r in real)[len(real) // 2],
                   real_res_u_le100=sum(r[3] for r in nar) / len(nar),
                   real_u_all=[round(r[0], 4) for r in real], real_res_u_all=[round(r[3], 4) for r in real])
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
    for f in ("icl_model_cw3.py", "icl_prior_v3s.py", "strong_base.py", "icl_train_cw3.py"):
        p = os.path.join(here, f)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()


def train(a):
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    last = os.path.join(a.out, "ckpt_last.pt")
    torch.manual_seed(a.seed)
    model = ICL(a.size, dropout=a.dropout).to(dev)
    ema = ICL(a.size, dropout=0.0).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    state = {"step": 0, "elapsed": 0.0, "datasets": 0, "next_named": a.named_h}
    init_sha = None
    if a.clone and not (a.resume and os.path.exists(last)):
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
                   gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type), model=model.cfg, started=time.strftime("%FT%TZ", time.gmtime()))
        cfg.pop("fn", None)
        with open(os.path.join(a.out, "config.json"), "x") as f:
            json.dump(cfg, f, indent=1)
    rng = random.Random(a.seed * 1_000_003 + state["step"])
    prior = Prior(dev, a.seed * 7_919 + state["step"])
    bank = INJ.RealBank(a.real_bank, dev, prior) if a.real_bank else None
    rval = INJ.RealVal(a.val_real, dev) if a.val_real else None
    log = open(os.path.join(a.out, "log.jsonl"), "a", buffering=1)
    res_gen = torch.Generator().manual_seed(a.seed * 104_729 + state["step"])
    aug_gen = torch.Generator(device=dev).manual_seed(a.seed * 224_737 + state["step"])
    replay_rng = random.Random(a.seed * 334_211 + state["step"])
    # Replay buffer: list of [ratio_score, batch_seed, B, N, Fn, n_ctx, replays_done]
    replay_buf: list[list] = []

    run_res = 0
    total = a.hours * 3600.0
    t_last = time.time()
    t_ck = t_eval = t_log = t_last
    run_loss, run_n, run_ds, run_real, run_inj = 0.0, 0, 0, 0, 0
    model.train()

    while state["elapsed"] < total and (not a.stop_h or state["elapsed"] < a.stop_h * 3600.0):
        frac = state["elapsed"] / total
        replayed_this_step = False
        cur_bseed = None

        if a.replay > 0.0 and replay_buf and replay_rng.random() < a.replay:
            idx_rep = replay_rng.randrange(len(replay_buf))
            _sc, cur_bseed, B, N, Fn, n_ctx, rcnt = replay_buf[idx_rep]
            x, y, _kinds, k_inj = deterministic_synth_batch(dev, cur_bseed, B, N, Fn, a.inject_p, a.prior)
            run_inj += k_inj > 0
            replayed_this_step = True
            if rcnt + 1 >= 3:
                replay_buf.pop(idx_rep)
            else:
                replay_buf[idx_rep][6] = rcnt + 1
        else:
            B, N, Fn, n_ctx = draw_shape(rng, frac, a.tokens, a.bmax, a.max_rows, a.row_tokens, a.row_floor)
            got = None
            if bank is not None and rng.random() < a.real_frac * min(1.0, state["elapsed"] / max(1.0, a.real_ramp_s)):
                got = bank.sample(rng, min(B, a.real_bmax), N, Fn, fill=not a.no_inject_real)
            if got is not None:
                x, y, _modes, _share = got
                B = x.shape[0]
                run_real += 1
            elif a.replay > 0.0:
                cur_bseed = a.seed * 10_000_019 + state["step"]
                x, y, _kinds, k_inj = deterministic_synth_batch(dev, cur_bseed, B, N, Fn, a.inject_p, a.prior)
                run_inj += k_inj > 0
            else:
                # Exact r9 call path when --replay == 0.0
                x, y, _kinds, k_inj = INJ.sample_synth(prior, rng, B, N, Fn, a.inject_p, _prior_probs(a.prior))
                run_inj += k_inj > 0

        y_orig = y
        if a.p_resid and rng.random() < a.p_resid:
            if a.p_resid_strong > 0.0 and replay_rng.random() < a.p_resid_strong:
                y, _used, _base = SB.strong_residual_targets(x, y, n_ctx, res_gen)
            else:
                y, _used, _base = RES.residual_targets(x, y, n_ctx, res_gen)
            run_res += 1

        x_in = x
        if (a.aug_cell_drop > 0.0 or a.aug_jitter > 0.0) and replay_rng.random() < 0.5:
            x_in = augment_features(x_in, n_ctx, a.aug_cell_drop, a.aug_jitter, aug_gen)

        slots = torch.randperm(256, device=dev)[:Fn]
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_factor(frac)
        logits, ym, ys = model(x_in, y[:, :n_ctx], n_ctx, slots=slots)
        loss = model.loss(logits, y[:, n_ctx:], ym, ys).mean()

        if a.consist > 0.0 and replay_rng.random() < 0.5:
            slots2 = torch.randperm(256, generator=aug_gen, device=dev)[:Fn]
            perm_f = torch.randperm(Fn, generator=aug_gen, device=dev)
            logits2, _, _ = model(x_in[:, :, perm_f], y[:, :n_ctx], n_ctx, slots=slots2)
            loss = loss + a.consist * symmetric_kl(logits, logits2)

        if a.replay > 0.0 and cur_bseed is not None and not replayed_this_step:
            with torch.no_grad():
                pred_q = _c(model.mean(logits, ym, ys))
                rmse_m = float(torch.sqrt(((pred_q - _c(y[:, n_ctx:])) ** 2).mean()).item())
                ur = ridge_u(x, y_orig, n_ctx)
                r0 = float(_c(y_orig[:, n_ctx:] - y_orig[:, :n_ctx].mean(1, keepdim=True)).pow(2).mean().sqrt().clamp_min(1e-9).item())
                rmse_r = max(r0 * math.exp(-float(ur.mean().item())), 1e-9)
                ratio = rmse_m / rmse_r
                if math.isfinite(ratio):
                    if len(replay_buf) < 512:
                        replay_buf.append([ratio, cur_bseed, B, N, Fn, n_ctx, 0])
                    else:
                        min_i = min(range(len(replay_buf)), key=lambda i: replay_buf[i][0])
                        if ratio > replay_buf[min_i][0]:
                            replay_buf[min_i] = [ratio, cur_bseed, B, N, Fn, n_ctx, 0]

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
        if now - t_log >= a.log_s:
            mem = (torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else 0.0
            log.write(json.dumps({"step": state["step"], "h": round(state["elapsed"] / 3600, 4),
                                  "loss": round(run_loss / run_n, 5), "ds_per_s": round(run_ds / max(now - t_log, 1e-6), 2),
                                  "lr": opt.param_groups[0]["lr"], "datasets": state["datasets"],
                                  "real_share": round(run_real / run_n, 3), "inj_share": round(run_inj / run_n, 3),
                                  "res_share": round(run_res / run_n, 3), "rows_top": max(max_rows(frac), a.row_floor),
                                  "mem_gb": round(mem, 2)}) + "\n")
            run_loss, run_n, run_ds, run_real, run_inj, run_res, t_log = 0.0, 0, 0, 0, 0, 0, now
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--size", default="S"); t.add_argument("--out", required=True)
    t.add_argument("--hours", type=float, required=True); t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--seed", type=int, default=1); t.add_argument("--tokens", type=int, default=600_000)
    t.add_argument("--bmax", type=int, default=64); t.add_argument("--max-rows", type=int, default=0)
    t.add_argument("--row-tokens", type=int, default=0)
    t.add_argument("--init"); t.add_argument("--real-bank")
    t.add_argument("--real-frac", type=float, default=0.4); t.add_argument("--real-ramp-s", type=float, default=3600)
    t.add_argument("--real-bmax", type=int, default=32); t.add_argument("--inject-p", type=float, default=0.5)
    t.add_argument("--val-real"); t.add_argument("--clone")
    t.add_argument("--stop-h", type=float, default=0.0)
    t.add_argument("--device", default="cuda"); t.add_argument("--no-inject-real", action="store_true")
    t.add_argument("--name", default="w0"); t.add_argument("--sync-dir")
    t.add_argument("--sync-s", type=float, default=300.0); t.add_argument("--sync-wait", type=float, default=240.0)
    t.add_argument("--ckpt-s", type=float, default=1200.0); t.add_argument("--named-h", type=float, default=2.0)
    t.add_argument("--log-s", type=float, default=60); t.add_argument("--eval-s", type=float, default=1800)
    t.add_argument("--p-resid", type=float, default=0.0)
    t.add_argument("--row-floor", type=int, default=0)
    t.add_argument("--prior", default="v1", choices=("v1", "v2", "v3s"))
    # CW-3 Factors B, C, D, E (all default-off)
    t.add_argument("--aug-cell-drop", type=float, default=0.0)
    t.add_argument("--aug-jitter", type=float, default=0.0)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--replay", type=float, default=0.0)
    t.add_argument("--p-resid-strong", type=float, default=0.0)
    t.add_argument("--consist", type=float, default=0.0)
    t.add_argument("--resume", action="store_true"); t.set_defaults(fn=train)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
