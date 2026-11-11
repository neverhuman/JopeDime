#!/usr/bin/env python3
"""H431 (CW-3 Part A): Score a pinned ICL checkpoint on ICB_v1 and on the FIT-SEL SEARCH monitor.

  icb_run.py battery --icb ICB_v1.npz --icb-sha256 HEX --ckpt CKPT --ckpt-sha256 HEX --size M|L --out DIR [--shard k/n] [--threads 1]
  icb_run.py monitor --val fitsel_search_val.npz --val-sha256 HEX --ckpt CKPT --ckpt-sha256 HEX --size M|L --out DIR [--shard k/n] [--threads 1]

CPU inference only (EMA weights). Query targets are replaced with zeros before every call to icl_diag.run_variant.
Every task/context output is create-once so restarts continue without overwriting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import icl_diag as DG  # noqa: E402
from icl_inject import RealVal  # noqa: E402
from icl_model import ICL  # noqa: E402

VARIANTS = ("full", "ridgeres")


def u_clipped(pred: torch.Tensor, yq: torch.Tensor, ym: torch.Tensor, hi: float) -> float:
    r = float(torch.sqrt(((pred - yq) ** 2).mean()))
    r0 = max(float(torch.sqrt(((ym - yq) ** 2).mean())), 1e-30)
    return max(-2.0, min(hi, -math.log(max(r / r0, 1e-30))))


def _load_ckpt(ckpt_path: str, ckpt_sha256: str, size: str) -> tuple[ICL, dict]:
    actual = hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()
    if actual != ckpt_sha256:
        raise SystemExit(f"checkpoint sha256 mismatch: {actual} != {ckpt_sha256}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ICL(size)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, ck


def run_battery(a: argparse.Namespace) -> None:
    if a.threads > 16:
        raise SystemExit("at most 16 threads")
    if torch.cuda.is_available():
        raise SystemExit("CUDA must be hidden (CPU-only wave)")
    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(a.threads)
    random.seed(0)
    torch.manual_seed(0)

    icb_sha = hashlib.sha256(Path(a.icb).read_bytes()).hexdigest()
    if icb_sha != a.icb_sha256:
        raise SystemExit(f"ICB sha256 mismatch: {icb_sha} != {a.icb_sha256}")

    model, ck = _load_ckpt(a.ckpt, a.ckpt_sha256, a.size)
    z = np.load(a.icb, allow_pickle=False)
    metas = json.loads(str(z["meta_json"]))
    shapes = z["shapes"].astype(np.int64)
    x_off = z["x_off"].astype(np.int64)
    y_off = z["y_off"].astype(np.int64)
    r_off = z["r_off"].astype(np.int64)
    x_data = z["x_data"]
    y_data = z["y_data"]
    ref_data = z["ref_q"]

    n_tasks = len(metas)
    # Balance shards by cost (rows * features)
    order = sorted(range(n_tasks), key=lambda j: (int(shapes[j, 0] * shapes[j, 1]), j))
    k, n_sh = (int(v) for v in a.shard.split("/"))
    my_idx = order[k::n_sh]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0

    for i in my_idx:
        p = out / f"task_{i:04d}.json"
        if p.exists():
            skipped += 1
            continue
        t1 = time.time()
        n_rows, fn, n_ctx, n_q = (int(v) for v in shapes[i].tolist())
        xo = int(x_off[i])
        yo = int(y_off[i])
        ro = int(r_off[i])
        x = torch.from_numpy(x_data[xo:xo + n_rows * fn].reshape(n_rows, fn).copy())
        y_true = torch.from_numpy(y_data[yo:yo + n_rows].copy())
        ref_q = torch.from_numpy(ref_data[ro:ro + n_q].copy()).double()

        yq = y_true[n_ctx:].double()
        ym = y_true[:n_ctx].double().mean()
        # Mask query targets strictly to zero before calling run_variant
        y_masked = torch.cat([y_true[:n_ctx], torch.zeros(n_q, dtype=y_true.dtype)])

        rq = DG.ridge_fit_predict(
            torch.nan_to_num(x[:n_ctx].double()),
            y_true[:n_ctx].double(),
            [torch.nan_to_num(x[n_ctx:].double())],
        )[0]
        u8_ridge = u_clipped(rq, yq, ym, 8.0)
        u6_ridge = u_clipped(rq, yq, ym, 6.0)
        u8_ref = u_clipped(ref_q, yq, ym, 8.0)
        u6_ref = u_clipped(ref_q, yq, ym, 6.0)

        u8_m, u6_m, cap_m, ok_m = {}, {}, {}, True
        for v in VARIANTS:
            pred = DG.run_variant(model, v, x, y_masked, n_ctx, 431_500 + i)
            if not bool(torch.isfinite(pred).all()):
                ok_m = False
            u8_v = u_clipped(pred, yq, ym, 8.0)
            u6_v = u_clipped(pred, yq, ym, 6.0)
            u8_m[v] = u8_v
            u6_m[v] = u6_v
            denom = max(u8_ref, 1e-6)
            cap_m[v] = float(np.clip(u8_v / denom, 0.0, 1.2)) if u8_ref > 0.05 else None

        rec = {
            "task_id": i,
            "family": metas[i]["family"],
            "family_name": metas[i]["family_name"],
            "n_ctx": n_ctx,
            "n_q": n_q,
            "F": fn,
            "noise": float(metas[i]["noise"]),
            "rep": int(metas[i]["rep"]),
            "ok": ok_m,
            "u8_ref": u8_ref,
            "u6_ref": u6_ref,
            "u8_ridge": u8_ridge,
            "u6_ridge": u6_ridge,
            "u8": u8_m,
            "u6": u6_m,
            "cap": cap_m,
            "wall_s": round(time.time() - t1, 4),
        }
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        tmp.rename(p)
        done += 1

    st = {
        "schema": "jope.h431.icb-status/v1",
        "mode": "battery",
        "ckpt": a.ckpt,
        "ckpt_sha256": a.ckpt_sha256,
        "ckpt_state": ck.get("state"),
        "icb_sha256": icb_sha,
        "size": a.size,
        "weights": "ema",
        "shard": a.shard,
        "threads": a.threads,
        "device": "cpu",
        "cuda_visible": bool(torch.cuda.is_available()),
        "assigned": len(my_idx),
        "written": done,
        "already_there": skipped,
        "wall_s": round(time.time() - t0, 3),
        "cpu_s": round(time.process_time() - c0, 3),
    }
    st_path = out / f"status_s{k}of{n_sh}.json"
    if not st_path.exists():
        st_path.write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st, sort_keys=True), flush=True)


def run_monitor(a: argparse.Namespace) -> None:
    if a.threads > 16:
        raise SystemExit("at most 16 threads")
    if torch.cuda.is_available():
        raise SystemExit("CUDA must be hidden (CPU-only wave)")
    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(a.threads)
    random.seed(0)
    torch.manual_seed(0)

    val_sha = hashlib.sha256(Path(a.val).read_bytes()).hexdigest()
    if val_sha != a.val_sha256:
        raise SystemExit(f"val bank sha256 mismatch: {val_sha} != {a.val_sha256}")

    model, ck = _load_ckpt(a.ckpt, a.ckpt_sha256, a.size)
    z = np.load(a.val, allow_pickle=False)
    cids = [str(c) for c in z["cids"].tolist()]
    rv = RealVal(a.val, "cpu")
    n_items = len(rv.items)
    order = sorted(range(n_items), key=lambda j: (int(rv.items[j][0].shape[0] * rv.items[j][0].shape[1]), j))
    k, n_sh = (int(v) for v in a.shard.split("/"))
    my_idx = order[k::n_sh]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0

    for i in my_idx:
        p = out / f"ctx_{i:04d}.json"
        if p.exists():
            skipped += 1
            continue
        t1 = time.time()
        m, n_ctx = rv.items[i]
        x, y_true = m[:, :-1], m[:, -1]
        n_q = int(m.shape[0] - n_ctx)
        yq = y_true[n_ctx:].double()
        ym = y_true[:n_ctx].double().mean()
        y_masked = torch.cat([y_true[:n_ctx], torch.zeros(n_q, dtype=y_true.dtype)])

        rq = DG.ridge_fit_predict(
            torch.nan_to_num(x[:n_ctx].double()),
            y_true[:n_ctx].double(),
            [torch.nan_to_num(x[n_ctx:].double())],
        )[0]
        u_ridge = u_clipped(rq, yq, ym, 6.0)
        u_m, ok_m = {}, True
        for v in VARIANTS:
            pred = DG.run_variant(model, v, x, y_masked, n_ctx, 419_500 + i)
            if not bool(torch.isfinite(pred).all()):
                ok_m = False
            u_m[v] = u_clipped(pred, yq, ym, 6.0)

        rec = {
            "i": i,
            "cid": cids[i],
            "family": str(rv.family[i]),
            "n_ctx": int(n_ctx),
            "n_q": n_q,
            "F": int(x.shape[1]),
            "ok": ok_m,
            "u_ridge": u_ridge,
            "u": u_m,
            "wall_s": round(time.time() - t1, 4),
        }
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        tmp.rename(p)
        done += 1

    st = {
        "schema": "jope.h431.mon-status/v1",
        "mode": "monitor",
        "ckpt": a.ckpt,
        "ckpt_sha256": a.ckpt_sha256,
        "ckpt_state": ck.get("state"),
        "val_sha256": val_sha,
        "size": a.size,
        "weights": "ema",
        "shard": a.shard,
        "threads": a.threads,
        "device": "cpu",
        "cuda_visible": bool(torch.cuda.is_available()),
        "assigned": len(my_idx),
        "written": done,
        "already_there": skipped,
        "wall_s": round(time.time() - t0, 3),
        "cpu_s": round(time.process_time() - c0, 3),
    }
    st_path = out / f"status_s{k}of{n_sh}.json"
    if not st_path.exists():
        st_path.write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("battery")
    b.add_argument("--icb", required=True)
    b.add_argument("--icb-sha256", required=True)
    b.add_argument("--ckpt", required=True)
    b.add_argument("--ckpt-sha256", required=True)
    b.add_argument("--size", required=True, choices=("S", "M", "L"))
    b.add_argument("--out", required=True)
    b.add_argument("--shard", default="0/1")
    b.add_argument("--threads", type=int, default=1)
    b.set_defaults(fn=run_battery)

    m = sub.add_parser("monitor")
    m.add_argument("--val", required=True)
    m.add_argument("--val-sha256", required=True)
    m.add_argument("--ckpt", required=True)
    m.add_argument("--ckpt-sha256", required=True)
    m.add_argument("--size", required=True, choices=("S", "M", "L"))
    m.add_argument("--out", required=True)
    m.add_argument("--shard", default="0/1")
    m.add_argument("--threads", type=int, default=1)
    m.set_defaults(fn=run_monitor)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
