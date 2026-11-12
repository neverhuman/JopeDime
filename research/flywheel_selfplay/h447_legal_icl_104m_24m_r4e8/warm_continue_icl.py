#!/usr/bin/env python3
"""H447 legal plain-ICL GPU warm-continuation trainer (`warm_continue_icl.py`).

Warm-continues plain `icl_model.ICL` (`size in {"S", "M", "L"}`) from an existing checkpoint's `ema` weights
on an H446 legal real-table pretraining bank (`LegalRealBank`, strictly refusing `DEV`, `CONFIRM`, `DL-VAL`,
and `SEALED`), combining:
  - Stream A (35%): Pre-cached inference-aligned `[X, S]` (`struct_cols.build_struct_cols` on `X_ctx, y_ctx -> X_q`,
    `0` query target leakage) + ridge-residual (`RES.residual_targets`) real-table batches, and
  - Stream B (65%): Multi-table `LegalRealBank` + `icl_prior_v3s` (`PROBS_V3S`) synthetic tasks (`p_resid = 0.50`).

Tracks exact integer `cum_cell_tokens`, `cum_row_tokens`, `cum_real_cell_tokens`, and `approx_train_flops = 6 * N * T`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import icl_bank_all_legal as BAL  # noqa: E402
import icl_inject as INJ  # noqa: E402
import icl_resid as RES  # noqa: E402
import struct_cols as SC  # noqa: E402
from icl_model import FID_SLOTS, ICL  # noqa: E402
from icl_prior_v3s import PROBS_V2, PROBS_V3S, Prior, draw_features  # noqa: E402

ROW_BUCKETS = (384, 640, 1024, 1536)
WIDE = 30
SUB = 30


class LegalRealBank(INJ.RealBank):
    """Subclasses `icl_inject.RealBank` while enforcing H446 cohort guards (`0` holdouts)."""

    def __init__(self, path: str, device, prior):
        z = np.load(path, allow_pickle=False)
        cohorts = sorted(set(str(c) for c in z["cohort"].tolist()))
        bad = [c for c in cohorts if c in BAL.FORBIDDEN_COHORTS or c not in BAL.LEGAL_TRAIN_COHORTS]
        if bad:
            raise SystemExit(f"REFUSING bank {path}: illegal/holdout cohort(s) {bad}")
        self.prior, self.dev = prior, torch.device(device)
        data, shape, off = z["data"], z["shape"], z["offset"]
        self.tabs_cpu = [
            np.array(data[o : o + r * c], np.float32, copy=True).reshape(r, c)
            for (r, c), o in zip(shape.tolist(), off.tolist())
        ]
        self.tabs = [torch.from_numpy(t).to(self.dev) for t in self.tabs_cpu]
        self.family = z["family"].tolist()
        self.cohorts = cohorts
        self.n_tables = len(self.tabs)
        self.manifest_sha256 = str(z["manifest_sha256"])


def _build_one_struct_episode(job) -> tuple[np.ndarray, np.ndarray, int, int, int] | None:
    """Worker: builds one `(X_feat [N_ep, W_ep], y_ep [N_ep], n_ctx, f_raw, d_s)` strictly from `(X_ctx, y_ctx) -> X_q`."""
    tab, max_rows, seed = job
    r_tot, c_tot = tab.shape
    if c_tot < 2:
        return None
    fin = np.isfinite(tab[:, -1])
    if int(fin.sum()) < 48:
        return None
    tab_f = tab[fin]
    if float(np.std(tab_f[:, -1])) < 1e-12:
        return None
    rng_np = np.random.default_rng(seed)
    n_rows = min(int(tab_f.shape[0]), max_rows)
    sub = tab_f[rng_np.permutation(int(tab_f.shape[0]))[:n_rows]]
    n_ctx = max(24, min(n_rows - 12, int(round(float(rng_np.uniform(0.70, 0.84)) * n_rows))))
    X_raw = sub[:, :-1].astype(np.float64, copy=False)
    y_vec = sub[:, -1].astype(np.float64, copy=False)
    f_raw = int(X_raw.shape[1])
    if f_raw > 120:
        keep_raw = np.sort(rng_np.choice(f_raw, 120, replace=False))
        X_raw = X_raw[:, keep_raw]
        f_raw = 120
    d_s = 0
    try:
        S_c, S_q, _ = SC.build_struct_cols(X_raw[:n_ctx], y_vec[:n_ctx], X_raw[n_ctx:], seed=int(seed & 0x7FFFFFF))
        d_s = int(S_c.shape[1])
    except Exception:
        d_s = 0
    if d_s > 0:
        X_feat = np.hstack([X_raw, np.vstack([S_c, S_q])]).astype(np.float32)
    else:
        X_feat = X_raw.astype(np.float32)
    return X_feat, y_vec.astype(np.float32), n_ctx, f_raw, d_s


def precache_struct_episodes(
    tabs_cpu: list[np.ndarray], n_episodes: int, max_rows: int, workers: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray, int, int, int]]:
    if n_episodes <= 0 or not tabs_cpu:
        return []
    rng = random.Random(seed)
    jobs = [(tabs_cpu[rng.randrange(len(tabs_cpu))], max_rows, rng.randrange(1 << 30)) for _ in range(n_episodes)]
    if workers > 1 and len(jobs) > 1:
        with Pool(processes=min(workers, len(jobs))) as pool:
            raw = pool.map(_build_one_struct_episode, jobs)
    else:
        raw = [_build_one_struct_episode(j) for j in jobs]
    return [ep for ep in raw if ep is not None]


def sample_batched_struct_r4(
    episodes: list[tuple[np.ndarray, np.ndarray, int, int, int]],
    rng: random.Random,
    res_gen: torch.Generator,
    dev: torch.device,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Forms a uniform `(B_a, N_t, W_t)` batch from pre-cached `[X, S]` episodes (retaining `S` when `W > 30`)."""
    if not episodes:
        return None
    picks = [episodes[rng.randrange(len(episodes))] for _ in range(batch_size)]
    min_ctx = min(p[2] for p in picks)
    min_q = min(p[0].shape[0] - p[2] for p in picks)
    if min_ctx < 16 or min_q < 8:
        return None
    n_ctx = min(min_ctx, 768)
    n_q = min(min_q, 256)
    w_t = min(max(min(p[0].shape[1], SUB) for p in picks), SUB)
    xs, ys = [], []
    for X_ep, y_ep, c_ep, f_raw, d_s in picks:
        g_np = np.random.default_rng(rng.randrange(1 << 30))
        ci = g_np.choice(c_ep, n_ctx, replace=False)
        qi = c_ep + g_np.choice(X_ep.shape[0] - c_ep, n_q, replace=False)
        rows = np.concatenate([ci, qi])
        X_sub = X_ep[rows]
        y_sub = y_ep[rows]
        w_cur = X_sub.shape[1]
        if w_cur > w_t:
            if d_s > 0 and d_s < w_t:
                k_raw = w_t - d_s
                raw_pick = np.sort(g_np.choice(f_raw, min(f_raw, k_raw), replace=False))
                s_cols = np.arange(f_raw, f_raw + d_s, dtype=np.int64)
                keep = np.concatenate([raw_pick, s_cols])[:w_t]
            else:
                keep = np.sort(g_np.choice(w_cur, w_t, replace=False))
            X_sub = X_sub[:, keep]
        elif w_cur < w_t:
            rep = g_np.choice(w_cur, w_t - w_cur, replace=True)
            X_sub = np.hstack([X_sub, X_sub[:, rep]])
        perm_c = g_np.permutation(w_t)
        xs.append(X_sub[:, perm_c])
        ys.append(y_sub)
    x_t = torch.from_numpy(np.stack(xs, axis=0)).to(dev)
    y_t = torch.from_numpy(np.stack(ys, axis=0)).to(dev)
    if rng.random() < 0.85:
        y_t, _u, _b = RES.residual_targets(x_t, y_t, n_ctx, res_gen)
    return x_t, y_t, n_ctx


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cosine_lr_factor(frac: float, min_ratio: float = 0.10, warmup_frac: float = 0.05) -> float:
    frac = max(0.0, min(1.0, frac))
    if frac < warmup_frac:
        return max(min_ratio, frac / max(warmup_frac, 1e-9))
    t = (frac - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def draw_shape(rng: random.Random, tokens: int, bmax: int, row_tokens: int, row_floor: int) -> tuple[int, int, int, int]:
    opts = [n for n in ROW_BUCKETS if n >= row_floor] or [ROW_BUCKETS[-1]]
    N = opts[-1] if rng.random() < 0.5 else rng.choice(opts)
    Fn = draw_features(rng)
    B = max(2, min(bmax, tokens // (N * (Fn + 4)), (row_tokens // N) if row_tokens else bmax))
    hold = max(64, N // 8)
    n_ctx = int(round(math.exp(rng.uniform(math.log(96), math.log(N - hold)))))
    return B, N, Fn, n_ctx


@torch.no_grad()
def eval_real_val(model: ICL, rval: INJ.RealVal) -> dict:
    model.eval()
    u_raw, u_res = [], []
    for i, (m, n_ctx) in enumerate(rval.items):
        x, y = m[None, :, :-1], m[None, :, -1]
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx)
        pred = model.mean(logits, ym, ys).detach().cpu().double()
        yq = y[:, n_ctx:].detach().cpu().double()
        ym_d = y[:, :n_ctx].detach().cpu().double().mean(1, keepdim=True)
        r = float(torch.sqrt(((pred - yq) ** 2).mean()))
        r0 = max(float(torch.sqrt(((ym_d - yq) ** 2).mean())), 1e-30)
        u_raw.append(max(-2.0, min(6.0, -math.log(max(r / r0, 1e-30)))))

        yr, used, bq = RES.residual_targets(x, y, n_ctx, torch.Generator().manual_seed(419_500 + i))
        lg2, ym2, ys2 = model(x, yr[:, :n_ctx], n_ctx)
        p2 = model.mean(lg2, ym2, ys2).detach().cpu().double()
        if bool(used[0].item()):
            p2 = p2 + bq.cpu().double()
        r2 = float(torch.sqrt(((p2 - yq) ** 2).mean()))
        u_res.append(max(-2.0, min(6.0, -math.log(max(r2 / r0, 1e-30)))))
    model.train()
    s_res = sorted(u_res)
    return {
        "real_n": len(u_res),
        "real_u": sum(u_raw) / len(u_raw),
        "real_res_u": sum(u_res) / len(u_res),
        "real_res_u_median": s_res[len(s_res) // 2],
    }


def save_ckpt(path: Path, model: ICL, ema: ICL, opt: torch.optim.Optimizer, state: dict) -> str:
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "state": state,
            "cfg": model.cfg,
        },
        tmp,
    )
    os.replace(tmp, path)
    return sha256_file(path)


def cmd_train(a) -> None:
    init_p = Path(a.init)
    bank_p = Path(a.real_bank)
    out_dir = Path(a.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already non-empty (create-once)")
    init_sha = sha256_file(init_p)
    if init_sha != a.init_sha256:
        raise SystemExit(f"init ckpt sha mismatch: {init_sha} != {a.init_sha256}")
    bank_sha = sha256_file(bank_p)
    if bank_sha != a.bank_sha256:
        raise SystemExit(f"real bank sha mismatch: {bank_sha} != {a.bank_sha256}")

    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    ck = torch.load(init_p, map_location=dev, weights_only=False)
    src_state = ck["ema"] if "ema" in ck else ck["model"]

    model = ICL(a.size).to(dev)
    ema = ICL(a.size).to(dev)
    model.load_state_dict(src_state, strict=True)
    ema.load_state_dict(src_state, strict=True)
    for p in ema.parameters():
        p.requires_grad_(False)
    params = n_params(model)

    with torch.no_grad():
        model.eval()
        ema.eval()
        gx = torch.Generator().manual_seed(447_001)
        tx = torch.randn(2, 64, 8, generator=gx).to(dev)
        ty = torch.randn(2, 48, generator=gx).to(dev)
        lg_m, ym_m, ys_m = model(tx, ty, 48, amp=False)
        lg_e, ym_e, ys_e = ema(tx, ty, 48, amp=False)
        step0_diff = float((model.mean(lg_m, ym_m, ys_m) - ema.mean(lg_e, ym_e, ys_e)).abs().max().item())
        if step0_diff != 0.0:
            raise SystemExit(f"step-0 model vs ema diff {step0_diff} != 0.0")
        model.train()

    decay = [p for _, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
        lr=a.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    prior = Prior(dev, a.seed * 7_919 + 1)
    bank = LegalRealBank(str(bank_p), dev, prior)
    rval = INJ.RealVal(a.val_real, dev) if a.val_real else None
    probs = PROBS_V3S if a.prior == "v3s" else (PROBS_V2 if a.prior == "v2" else None)

    t_pre0 = time.time()
    struct_eps = precache_struct_episodes(
        bank.tabs_cpu, a.precache_episodes, max_rows=1024, workers=a.precache_workers, seed=a.seed + 77
    )
    precache_s = round(time.time() - t_pre0, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    val0 = eval_real_val(ema, rval) if rval is not None else {}

    cfg_rec = {
        "schema": "jope.h447.warm_continue/v2",
        "size": a.size,
        "params": params,
        "device": a.device,
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "init_ckpt": str(init_p),
        "init_sha256": init_sha,
        "real_bank": str(bank_p),
        "bank_sha256": bank_sha,
        "bank_tables": bank.n_tables,
        "bank_cohorts": bank.cohorts,
        "precache_episodes": len(struct_eps),
        "precache_s": precache_s,
        "prior": a.prior,
        "real_frac": a.real_frac,
        "struct_frac": a.struct_frac,
        "p_resid": a.p_resid,
        "lr": a.lr,
        "ema_decay": a.ema_decay,
        "hours": a.hours,
        "max_steps": a.max_steps,
        "target_cell_tokens": a.target_cell_tokens,
        "seed": a.seed,
        "step0_diff": step0_diff,
        "val_step0": val0,
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_rec, indent=2, sort_keys=True) + "\n")
    log_f = open(out_dir / "log.jsonl", "a", buffering=1)

    rng = random.Random(a.seed * 1_000_003 + 11)
    res_gen = torch.Generator().manual_seed(a.seed * 104_729 + 13)
    slot_gen = torch.Generator(device=dev).manual_seed(a.seed * 224_737 + 17)

    t0 = time.time()
    t_eval = t0
    total_s = a.hours * 3600.0
    step = 0
    datasets = 0
    real_batches = 0
    struct_batches = 0
    resid_batches = 0
    cum_cell_tokens = 0
    cum_row_tokens = 0
    cum_real_cell_tokens = 0
    loss_acc = 0.0
    loss_n = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= total_s:
            break
        if a.max_steps > 0 and step >= a.max_steps:
            break
        if a.target_cell_tokens > 0 and cum_cell_tokens >= a.target_cell_tokens:
            break

        frac_t = elapsed / max(total_s, 1e-9)
        frac_s = (step / a.max_steps) if a.max_steps > 0 else 0.0
        frac_tok = (cum_cell_tokens / a.target_cell_tokens) if a.target_cell_tokens > 0 else 0.0
        frac = max(frac_t, frac_s, frac_tok)

        x = y = None
        n_ctx = 0
        is_real_batch = False

        if struct_eps and rng.random() < a.struct_frac:
            sb = sample_batched_struct_r4(struct_eps, rng, res_gen, dev, batch_size=min(6, a.real_bmax))
            if sb is not None:
                x, y, n_ctx = sb
                B = int(x.shape[0])
                Fn = int(x.shape[2])
                is_real_batch = True
                real_batches += 1
                struct_batches += 1
                resid_batches += 1

        if x is None:
            B, N, Fn, n_ctx = draw_shape(rng, a.tokens, a.bmax, a.row_tokens, a.row_floor)
            if a.wide_prob > 0.0 and rng.random() < a.wide_prob:
                Fn = rng.randint(101, 240)
                B = max(2, min(a.real_bmax, a.tokens // (N * (Fn + 4)), (a.row_tokens // N) if a.row_tokens else a.real_bmax))

            got = None
            if rng.random() < a.real_frac:
                got = bank.sample(rng, min(B, a.real_bmax), N, Fn, fill=True)
            if got is not None:
                x, y, _modes, _share = got
                B = int(x.shape[0])
                is_real_batch = True
                real_batches += 1
            else:
                Fn_s = min(Fn, 100)
                x, y, _kinds, _kinj = INJ.sample_synth(prior, rng, B, N, Fn_s, a.inject_p, probs)
                Fn = Fn_s

            if a.p_resid > 0.0 and rng.random() < a.p_resid:
                y, _used, _base = RES.residual_targets(x, y, n_ctx, res_gen)
                resid_batches += 1

        Fn = int(x.shape[2])
        if Fn <= FID_SLOTS:
            slots = torch.randperm(FID_SLOTS, generator=slot_gen, device=dev)[:Fn]
        else:
            slots = torch.randint(0, FID_SLOTS, (Fn,), generator=slot_gen, device=dev)

        lr_now = a.lr * cosine_lr_factor(frac)
        for g in opt.param_groups:
            g["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        logits, ym, ys = model(x, y[:, :n_ctx], n_ctx, slots=slots, amp=(dev.type == "cuda"))
        loss = model.loss(logits, y[:, n_ctx:], ym, ys).mean()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            d_ema = a.ema_decay
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d_ema).add_(pm, alpha=1.0 - d_ema)

        B_cur, N_cur, F_cur = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        row_tok = B_cur * N_cur
        cell_tok = B_cur * N_cur * (F_cur + 1)
        cum_row_tokens += row_tok
        cum_cell_tokens += cell_tok
        if is_real_batch:
            cum_real_cell_tokens += cell_tok

        step += 1
        datasets += B_cur
        loss_acc += float(loss.item())
        loss_n += 1

        now = time.time()
        if step % a.log_every == 0:
            rec = {
                "type": "train",
                "step": step,
                "elapsed_s": round(now - t0, 2),
                "lr": lr_now,
                "loss": round(loss_acc / max(loss_n, 1), 5),
                "datasets": datasets,
                "real_batches": real_batches,
                "struct_batches": struct_batches,
                "resid_batches": resid_batches,
                "cum_row_tokens": cum_row_tokens,
                "cum_cell_tokens": cum_cell_tokens,
                "cum_real_cell_tokens": cum_real_cell_tokens,
            }
            log_f.write(json.dumps(rec, sort_keys=True) + "\n")
            loss_acc, loss_n = 0.0, 0

        if rval is not None and (now - t_eval >= a.eval_s):
            t_eval = now
            vrec = {"type": "eval", "step": step, "elapsed_s": round(now - t0, 2), **eval_real_val(ema, rval)}
            log_f.write(json.dumps(vrec, sort_keys=True) + "\n")

    train_wall_s = time.time() - t0
    val_final = eval_real_val(ema, rval) if rval is not None else {}
    state = {
        "step": step,
        "elapsed": train_wall_s,
        "precache_s": precache_s,
        "datasets": datasets,
        "real_batches": real_batches,
        "struct_batches": struct_batches,
        "resid_batches": resid_batches,
        "cum_row_tokens": cum_row_tokens,
        "cum_cell_tokens": cum_cell_tokens,
        "cum_real_cell_tokens": cum_real_cell_tokens,
        "cum_d91_cell_tokens": 0,
        "params": params,
        "approx_train_flops": 6 * params * cum_cell_tokens,
    }
    final_p = out_dir / "ckpt_wc_final.pt"
    final_sha = save_ckpt(final_p, model, ema, opt, state)
    final_p.chmod(0o444)

    manifest = {
        "schema": "jope.h447.wc_manifest/v2",
        "ckpt": str(final_p),
        "ckpt_sha256": final_sha,
        "size": a.size,
        "params": params,
        "device": a.device,
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "init_sha256": init_sha,
        "bank_sha256": bank_sha,
        "bank_tables": bank.n_tables,
        "state": state,
        "val_step0": val0,
        "val_final": val_final,
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    (out_dir / "wc_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log_f.close()
    print(json.dumps(manifest, sort_keys=True))


def cmd_selftest(_a) -> None:
    os.makedirs("/home/ubuntu/jope-data/h447", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/home/ubuntu/jope-data/h447") as td:
        td_p = Path(td)
        m0 = ICL("S")
        ck_p = td_p / "init_s.pt"
        torch.save({"model": m0.state_dict(), "ema": m0.state_dict(), "cfg": m0.cfg}, ck_p)
        ck_sha = sha256_file(ck_p)

        rng_np = np.random.default_rng(447)
        tab = rng_np.standard_normal((256, 10)).astype(np.float32)
        bank_p = td_p / "legal_smoke.npz"
        np.savez(
            bank_p,
            data=tab.ravel(),
            shape=np.array([[256, 10]], np.int64),
            offset=np.array([0], np.int64),
            cids=np.array(["c0"]),
            family=np.array(["f0"]),
            cohort=np.array(["FIT-ICL"]),
            manifest_sha256=np.array("smoke"),
        )
        bank_sha = sha256_file(bank_p)

        ns = argparse.Namespace(
            size="S",
            init=str(ck_p),
            init_sha256=ck_sha,
            real_bank=str(bank_p),
            bank_sha256=bank_sha,
            val_real="",
            out=str(td_p / "run_out"),
            device="cpu",
            threads=2,
            seed=447,
            lr=3e-5,
            ema_decay=0.99,
            hours=0.05,
            max_steps=3,
            target_cell_tokens=0,
            precache_episodes=4,
            precache_workers=1,
            struct_frac=0.5,
            tokens=2048,
            bmax=2,
            real_bmax=2,
            row_tokens=800,
            row_floor=128,
            real_frac=1.0,
            wide_prob=0.0,
            inject_p=0.0,
            p_resid=0.5,
            prior="v3s",
            log_every=1,
            eval_s=9999,
        )
        global ROW_BUCKETS
        orig_buckets = ROW_BUCKETS
        ROW_BUCKETS = (128,)
        try:
            cmd_train(ns)
        finally:
            ROW_BUCKETS = orig_buckets
        man = json.loads((td_p / "run_out" / "wc_manifest.json").read_text())
        assert man["state"]["step"] == 3 and len(man["ckpt_sha256"]) == 64
        assert man["state"]["cum_cell_tokens"] > 0 and man["state"]["cum_d91_cell_tokens"] == 0

        slot_g = torch.Generator().manual_seed(447)
        slots_wide = torch.randint(0, FID_SLOTS, (264,), generator=slot_g)
        xw = torch.randn(1, 24, 264)
        yw = torch.randn(1, 16)
        with torch.no_grad():
            lg_w, ym_w, ys_w = m0.eval()(xw, yw, 16, slots=slots_wide, amp=False)
            assert torch.isfinite(m0.mean(lg_w, ym_w, ys_w)).all()

        bad_p = td_p / "bad_dev91.npz"
        np.savez(
            bad_p,
            data=tab.ravel(),
            shape=np.array([[256, 10]], np.int64),
            offset=np.array([0], np.int64),
            cids=np.array(["c0"]),
            family=np.array(["f0"]),
            cohort=np.array(["DL-VAL"]),
            manifest_sha256=np.array("smoke"),
        )
        refused = False
        try:
            LegalRealBank(str(bad_p), "cpu", Prior(torch.device("cpu"), 1))
        except SystemExit:
            refused = True
        assert refused, "LegalRealBank failed to refuse DL-VAL cohort"

    print(json.dumps({"all_passed": True, "tests": 3}))
    print("H447_WC_SELFTEST_OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("train")
    p.add_argument("--size", required=True, choices=("S", "M", "L"))
    p.add_argument("--init", required=True)
    p.add_argument("--init-sha256", required=True)
    p.add_argument("--real-bank", required=True)
    p.add_argument("--bank-sha256", required=True)
    p.add_argument("--val-real", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=44701)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--hours", type=float, default=0.75)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--target-cell-tokens", type=int, default=1_000_000_000)
    p.add_argument("--precache-episodes", type=int, default=256)
    p.add_argument("--precache-workers", type=int, default=16)
    p.add_argument("--struct-frac", type=float, default=0.35)
    p.add_argument("--tokens", type=int, default=550_000)
    p.add_argument("--bmax", type=int, default=24)
    p.add_argument("--real-bmax", type=int, default=16)
    p.add_argument("--row-tokens", type=int, default=22_000)
    p.add_argument("--row-floor", type=int, default=1024)
    p.add_argument("--real-frac", type=float, default=0.60)
    p.add_argument("--wide-prob", type=float, default=0.20)
    p.add_argument("--inject-p", type=float, default=0.50)
    p.add_argument("--p-resid", type=float, default=0.50)
    p.add_argument("--prior", default="v3s", choices=("v1", "v2", "v3s"))
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-s", type=int, default=900)
    a = ap.parse_args()
    {"selftest": cmd_selftest, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
