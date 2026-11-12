#!/usr/bin/env python3
"""H449 matched plain-ICL trainer (`train_unfair_icl_h449.py`) for BOTH:
  1. `ALL-NON-DEV91-COMPLETE` (`0` `d91` tables, `1,976+` complete non-`d91` tables, GPU Stage-2), and
  2. `ENTIRE` (`everything_all_d91_bank.npz`, `2,067+` complete tables incl. `100%` of `d91` `[X_search U X_audit, y_all]`,
     strictly inside `--network none` `sandbox_h449_unfair_d91.sh` on `xbabe0`).

Pre-caches `[X, S]` (`struct_cols.build_struct_cols` on `X_ctx, y_ctx -> X_q`, `0` query target leakage) across CPU
workers at startup (including `5` distinct 72/28 folds for every `D91` table when `d91_idx` is non-empty, covering
100% of `d91` rows as both context and query!), then trains with exact integer token & FLOP accounting:
  `cum_cell_tokens`, `cum_row_tokens`, `cum_real_cell_tokens`, `cum_d91_cell_tokens`, `approx_train_flops = 6 * N * T`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import icl_inject as INJ  # noqa: E402
import icl_resid as RES  # noqa: E402
import struct_cols as SC  # noqa: E402
from icl_model import FID_SLOTS, ICL  # noqa: E402
from icl_prior_v3s import PROBS_V3S, Prior, draw_features  # noqa: E402

WIDE = 30
SUB = 30
ROW_BUCKETS = (256, 512, 1024)


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


class QuarantinedEverythingBank(INJ.RealBank):
    """Loads `all_non_d91_complete_bank.npz` or `everything_*_d91_bank.npz` inside `/h449` quarantine."""

    def __init__(self, path: str, device, prior):
        z = np.load(path, allow_pickle=False)
        self.prior, self.dev = prior, torch.device(device)
        data, shape, off = z["data"], z["shape"], z["offset"]
        self.cids = [str(c) for c in z["cids"].tolist()]
        self.cohorts = [str(c) for c in z["cohort"].tolist()]
        self.family = [str(f) for f in z["family"].tolist()]
        self.tabs_cpu = []
        for (r, c), o in zip(shape.tolist(), off.tolist()):
            t = np.array(data[o : o + r * c], np.float32, copy=True).reshape(r, c)
            t[~np.isfinite(t)] = np.nan
            self.tabs_cpu.append(t)
        self.tabs = [torch.from_numpy(t).to(self.dev) for t in self.tabs_cpu]
        self.n_tables = len(self.tabs)
        self.d91_idx = [i for i, coh in enumerate(self.cohorts) if coh.startswith("D91-")]
        self.other_idx = [i for i, coh in enumerate(self.cohorts) if not coh.startswith("D91-")]


def _build_ep_worker(job) -> tuple[np.ndarray, np.ndarray, int, int, int, bool] | None:
    """Builds `(X_feat, y_vec, n_ctx, f_raw, d_s, is_d91)` for a table down to `r >= 16`."""
    tab, is_d91, max_rows, seed = job
    r_tot, c_tot = tab.shape
    if c_tot < 2:
        return None
    fin = np.isfinite(tab[:, -1])
    if int(fin.sum()) < 16:
        return None
    tab_f = tab[fin]
    if float(np.std(tab_f[:, -1])) < 1e-12:
        return None
    rng_np = np.random.default_rng(seed)
    n_rows = min(int(tab_f.shape[0]), max_rows)
    sub = tab_f[rng_np.permutation(int(tab_f.shape[0]))[:n_rows]]
    n_ctx = max(8, min(n_rows - 4, int(round(float(rng_np.uniform(0.68, 0.84)) * n_rows))))
    X_raw = sub[:, :-1].astype(np.float64, copy=False)
    y_vec = sub[:, -1].astype(np.float64, copy=False)
    f_raw = int(X_raw.shape[1])
    if f_raw > 120:
        keep_raw = np.sort(rng_np.choice(f_raw, 120, replace=False))
        X_raw = X_raw[:, keep_raw]
        f_raw = 120
    d_s = 0
    if n_ctx >= 16:
        try:
            S_c, S_q, _ = SC.build_struct_cols(X_raw[:n_ctx], y_vec[:n_ctx], X_raw[n_ctx:], seed=int(seed & 0x7FFFFFF))
            d_s = int(S_c.shape[1])
        except Exception:
            d_s = 0
    if d_s > 0:
        X_feat = np.hstack([X_raw, np.vstack([S_c, S_q])]).astype(np.float32)
    else:
        X_feat = X_raw.astype(np.float32)
    return X_feat, y_vec.astype(np.float32), n_ctx, f_raw, d_s, bool(is_d91)


def precache_episodes(bank: QuarantinedEverythingBank, n_other: int, d91_folds: int, max_rows: int, workers: int, seed: int):
    rng = random.Random(seed)
    jobs = []
    # Every core D91 table gets `d91_folds` distinct splits; D91-S3-COUSIN gets max(1, d91_folds // 2) splits
    for idx in bank.d91_idx:
        n_f = max(1, d91_folds) if bank.cohorts[idx] == "D91-ALL" else max(1, d91_folds // 2)
        for _ in range(n_f):
            jobs.append((bank.tabs_cpu[idx], True, max_rows, rng.randrange(1 << 30)))
    pool_other = bank.other_idx or bank.d91_idx
    for _ in range(max(0, n_other)):
        if pool_other:
            idx = pool_other[rng.randrange(len(pool_other))]
            jobs.append((bank.tabs_cpu[idx], idx in bank.d91_idx, max_rows, rng.randrange(1 << 30)))
    if not jobs:
        return [], []
    if workers > 1 and len(jobs) > 1:
        with Pool(processes=min(workers, len(jobs))) as pool:
            done = pool.map(_build_ep_worker, jobs)
    else:
        done = [_build_ep_worker(j) for j in jobs]
    d91_eps = [ep for ep in done if ep is not None and ep[5]]
    oth_eps = [ep for ep in done if ep is not None and not ep[5]]
    return d91_eps, oth_eps


def sample_from_precache(
    d91_eps: list,
    oth_eps: list,
    rng: random.Random,
    res_gen: torch.Generator,
    dev: torch.device,
    batch_size: int = 1,
    d91_weight: float = 0.65,
) -> tuple[torch.Tensor, torch.Tensor, int, float] | None:
    """Returns `(x_t, y_t, n_ctx, d91_share_in_batch)` from pre-cached `[X, S]` episodes."""
    if not d91_eps and not oth_eps:
        return None
    picks = []
    d91_cnt = 0
    for _ in range(batch_size):
        use_d91 = bool(d91_eps) and (not oth_eps or rng.random() < d91_weight)
        ep = d91_eps[rng.randrange(len(d91_eps))] if use_d91 else oth_eps[rng.randrange(len(oth_eps))]
        picks.append(ep)
        d91_cnt += int(ep[5])

    if batch_size == 1:
        X_ep, y_ep, n_ctx, f_raw, d_s, _ = picks[0]
        g_np = np.random.default_rng(rng.randrange(1 << 30))
        w_cur = X_ep.shape[1]
        X_sub = X_ep
        if w_cur > WIDE:
            if d_s > 0 and d_s < SUB:
                k_raw = max(1, SUB - d_s)
                raw_pick = np.sort(g_np.choice(f_raw, min(f_raw, k_raw), replace=False))
                s_cols = np.arange(f_raw, f_raw + d_s, dtype=np.int64)
                keep = np.concatenate([raw_pick, s_cols])[:SUB]
            else:
                keep = np.sort(g_np.choice(w_cur, SUB, replace=False))
            X_sub = X_sub[:, keep]
            w_cur = X_sub.shape[1]
        perm_c = g_np.permutation(w_cur)
        x_t = torch.from_numpy(X_sub[:, perm_c])[None].to(dev)
        y_t = torch.from_numpy(y_ep)[None].to(dev)
        if rng.random() < 0.85:
            y_t, _u, _b = RES.residual_targets(x_t, y_t, n_ctx, res_gen)
        return x_t, y_t, n_ctx, float(d91_cnt)

    min_ctx = min(p[2] for p in picks)
    min_q = min(p[0].shape[0] - p[2] for p in picks)
    if min_ctx < 8 or min_q < 4:
        # Fall back to single episode so small <48-row D91 tables are NEVER skipped!
        return sample_from_precache(d91_eps, oth_eps, rng, res_gen, dev, batch_size=1, d91_weight=d91_weight)

    n_ctx = min(min_ctx, 768)
    n_q = min(min_q, 256)
    w_t = min(max(min(p[0].shape[1], SUB) for p in picks), SUB)
    xs, ys = [], []
    for X_ep, y_ep, c_ep, f_raw, d_s, _ in picks:
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
    return x_t, y_t, n_ctx, float(d91_cnt) / float(batch_size)


def save_ckpt(path: Path, model: ICL, ema: ICL, state: dict) -> str:
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "state": state, "cfg": model.cfg}, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def cmd_train(a) -> None:
    out_dir = Path(a.out).resolve()
    if "/h449" not in str(out_dir) and "/H449/" not in str(out_dir):
        raise SystemExit(f"QUARANTINE REFUSAL: output {out_dir} must live inside an h449 directory")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already non-empty (create-once)")

    init_p = Path(a.init)
    bank_p = Path(a.real_bank)
    if sha256_file(init_p) != a.init_sha256:
        raise SystemExit("init ckpt sha256 mismatch")
    if sha256_file(bank_p) != a.bank_sha256:
        raise SystemExit("real bank sha256 mismatch")

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
    model.train()
    params = n_params(model)

    decay = [p for _, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
        lr=a.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    prior = Prior(dev, a.seed * 7_919 + 1)
    bank = QuarantinedEverythingBank(str(bank_p), dev, prior)
    d91_frac_in_bank = float(len(bank.d91_idx)) / max(float(bank.n_tables), 1.0)

    t_pre0 = time.time()
    d91_eps, oth_eps = precache_episodes(
        bank,
        n_other=a.precache_other,
        d91_folds=a.d91_folds,
        max_rows=a.max_rows,
        workers=a.precache_workers,
        seed=a.seed + 31,
    )
    precache_s = round(time.time() - t_pre0, 2)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_rec = {
        "schema": "jope.h449.train_cfg/v2",
        "quarantined_diagnostic": True,
        "has_d91_in_bank": len(bank.d91_idx) > 0,
        "size": a.size,
        "params": params,
        "device": a.device,
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "threads": a.threads,
        "init_ckpt": str(init_p),
        "init_sha256": a.init_sha256,
        "real_bank": str(bank_p),
        "bank_sha256": a.bank_sha256,
        "bank_tables": bank.n_tables,
        "d91_tables_in_bank": len(bank.d91_idx),
        "precache_d91_episodes": len(d91_eps),
        "precache_other_episodes": len(oth_eps),
        "precache_s": precache_s,
        "lr": a.lr,
        "ema_decay": a.ema_decay,
        "hours": a.hours,
        "max_steps": a.max_steps,
        "target_cell_tokens": a.target_cell_tokens,
        "bmax": a.bmax,
        "struct_bmax": a.struct_bmax,
        "seed": a.seed,
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_rec, indent=2, sort_keys=True) + "\n")
    log_f = open(out_dir / "log.jsonl", "a", buffering=1)

    rng = random.Random(a.seed * 1_000_003 + 19)
    res_gen = torch.Generator().manual_seed(a.seed * 104_729 + 23)
    slot_gen = torch.Generator(device=dev).manual_seed(a.seed * 224_737 + 29)

    t0 = time.time()
    total_s = a.hours * 3600.0
    step = 0
    datasets = 0
    direct_r4_steps = 0
    stream_b_steps = 0
    cum_row_tokens = 0
    cum_cell_tokens = 0
    cum_real_cell_tokens = 0
    cum_d91_cell_tokens = 0
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

        frac = max(
            elapsed / max(total_s, 1e-9),
            (step / a.max_steps) if a.max_steps > 0 else 0.0,
            (cum_cell_tokens / a.target_cell_tokens) if a.target_cell_tokens > 0 else 0.0,
        )

        x = y = None
        n_ctx = 0
        is_real = False
        d91_share = 0.0

        if (d91_eps or oth_eps) and rng.random() < a.struct_frac:
            # Mix single-episode (captures small <48-row D91 tables at exact row count) and batched episodes
            bs_a = 1 if (d91_eps and rng.random() < 0.35) else a.struct_bmax
            direct = sample_from_precache(d91_eps, oth_eps, rng, res_gen, dev, batch_size=bs_a, d91_weight=0.65)
            if direct is not None:
                x, y, n_ctx, d91_share = direct
                is_real = True
                direct_r4_steps += 1

        if x is None:
            N = rng.choice(ROW_BUCKETS)
            Fn = min(draw_features(rng), 64)
            B = max(2, min(a.bmax, a.tokens // max(N * (Fn + 4), 1)))
            n_ctx = int(round(rng.uniform(0.65, 0.85) * N))
            got = bank.sample(rng, B, N, Fn, fill=True) if rng.random() < a.real_frac else None
            if got is not None:
                x, y, _m, _sh = got
                is_real = True
                d91_share = d91_frac_in_bank
            else:
                x, y, _k, _ki = INJ.sample_synth(prior, rng, B, N, Fn, 0.4, PROBS_V3S)
            if rng.random() < 0.50:
                y, _u, _b = RES.residual_targets(x, y, n_ctx, res_gen)
            stream_b_steps += 1

        Fn_cur = int(x.shape[2])
        if Fn_cur <= FID_SLOTS:
            slots = torch.randperm(FID_SLOTS, generator=slot_gen, device=dev)[:Fn_cur]
        else:
            slots = torch.randint(0, FID_SLOTS, (Fn_cur,), generator=slot_gen, device=dev)

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
        if is_real:
            cum_real_cell_tokens += cell_tok
            cum_d91_cell_tokens += int(round(cell_tok * d91_share))

        step += 1
        datasets += B_cur
        loss_acc += float(loss.item())
        loss_n += 1
        if step % a.log_every == 0:
            log_f.write(json.dumps({
                "step": step,
                "elapsed_s": round(time.time() - t0, 2),
                "lr": lr_now,
                "loss": round(loss_acc / max(loss_n, 1), 5),
                "datasets": datasets,
                "direct_r4_steps": direct_r4_steps,
                "stream_b_steps": stream_b_steps,
                "cum_row_tokens": cum_row_tokens,
                "cum_cell_tokens": cum_cell_tokens,
                "cum_real_cell_tokens": cum_real_cell_tokens,
                "cum_d91_cell_tokens": cum_d91_cell_tokens,
            }, sort_keys=True) + "\n")
            loss_acc, loss_n = 0.0, 0

    state = {
        "step": step,
        "elapsed_s": round(time.time() - t0, 2),
        "precache_s": precache_s,
        "datasets": datasets,
        "direct_r4_steps": direct_r4_steps,
        "stream_b_steps": stream_b_steps,
        "cum_row_tokens": cum_row_tokens,
        "cum_cell_tokens": cum_cell_tokens,
        "cum_real_cell_tokens": cum_real_cell_tokens,
        "cum_d91_cell_tokens": cum_d91_cell_tokens,
        "params": params,
        "approx_train_flops": 6 * params * cum_cell_tokens,
    }
    final_p = out_dir / "ckpt_unfair_final.pt"
    final_sha = save_ckpt(final_p, model, ema, state)
    final_p.chmod(0o444)
    manifest = {
        "schema": "jope.h449.unfair_ckpt_manifest/v2",
        "quarantined_unfair_diagnostic": len(bank.d91_idx) > 0,
        "d91_tables_in_bank": len(bank.d91_idx),
        "bank_tables": bank.n_tables,
        "ckpt": str(final_p),
        "ckpt_sha256": final_sha,
        "size": a.size,
        "params": params,
        "device": a.device,
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "init_sha256": a.init_sha256,
        "bank_sha256": a.bank_sha256,
        "state": state,
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    (out_dir / "unfair_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log_f.close()
    print(json.dumps(manifest, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--size", required=True, choices=("S", "M", "L"))
    p.add_argument("--init", required=True)
    p.add_argument("--init-sha256", required=True)
    p.add_argument("--real-bank", required=True)
    p.add_argument("--bank-sha256", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=449)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ema-decay", type=float, default=0.996)
    p.add_argument("--hours", type=float, default=0.55)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--target-cell-tokens", type=int, default=0)
    p.add_argument("--precache-other", type=int, default=256)
    p.add_argument("--d91-folds", type=int, default=5)
    p.add_argument("--precache-workers", type=int, default=16)
    p.add_argument("--struct-frac", type=float, default=0.50)
    p.add_argument("--struct-bmax", type=int, default=4)
    p.add_argument("--real-frac", type=float, default=0.65)
    p.add_argument("--tokens", type=int, default=550_000)
    p.add_argument("--max-rows", type=int, default=1024)
    p.add_argument("--bmax", type=int, default=16)
    p.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()
    cmd_train(a)


if __name__ == "__main__":
    main()
