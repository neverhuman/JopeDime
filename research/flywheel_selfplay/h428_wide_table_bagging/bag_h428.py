#!/usr/bin/env python3
"""H428: feature bagging for wide tables, CPU inference of the H419 in-context model on the stored FIT-SEL monitor bank.

  bag_h428.py --bank fitsel_search_val.npz --ckpt CKPT --ckpt-sha256 HEX --size M --out DIR [--shard k/n] [--budget-s S]

The monitor's task per context: SEARCH rows only; query = fold 0 of the stored fold labels, context = folds 1-4
(icl_inject.RealVal). For every context with more than WIDE = 30 features, with the checkpoint's EMA weights:
  control  the monitored residual-of-ridge readout on all features (icl_diag.run_variant "ridgeres", the monitor's seed)
  T2       the same readout on each of K = 8 seeded random subsets of 30 features (ridge and model both see the
           subset only), predictions averaged                                         [the queued prompt, literally]
  T1       ridge on ALL features as in the control (icl_resid.residual_targets); the model predicts that ridge's
           residual from each of the same 8 subsets; the 8 residual predictions are averaged and added to the ridge
u = -ln(RMSE / RMSE of the context-mean predictor) on the query rows, clamped to [-2, 6] (icl_diag.u_of).
One create-once OUT/<cid>.json per context (a restart continues); cheapest context first; the run starts no new
context after --budget-s seconds. Narrower tables are not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import icl_diag as DG  # noqa: E402
import icl_resid as RES  # noqa: E402
from icl_inject import RealVal  # noqa: E402
from icl_model import ICL  # noqa: E402

WIDE, SUB, K = 30, 30, 8
SEED = 428_000


def subsets(cid: str, Fn: int):
    out = []
    for k in range(K):
        s = int.from_bytes(hashlib.sha256(f"H428-subset\0{cid}\0{k}".encode()).digest()[:8], "big")
        out.append(np.sort(np.random.default_rng(s).choice(Fn, SUB, replace=False)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-sha256", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budget-s", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    if hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest() != a.ckpt_sha256:
        raise SystemExit("the checkpoint is not the one named by --ckpt-sha256")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ICL(a.size)
    model.load_state_dict(ck["ema"])
    model.eval()
    z = np.load(a.bank, allow_pickle=False)
    if set(z["cohort"].tolist()) != {"FIT-SEL"}:
        raise SystemExit("the bank holds a context outside FIT-SEL")
    cids = [str(c) for c in z["cids"]]
    rv = RealVal(a.bank, "cpu")
    wide = [i for i, (m, _n) in enumerate(rv.items) if m.shape[1] - 1 > WIDE]
    order = sorted(wide, key=lambda i: (rv.items[i][0].shape[0] * (rv.items[i][0].shape[1] - 1), cids[i]))
    k, n = (int(v) for v in a.shard.split("/"))
    order = order[k::n]
    if a.limit:
        order = order[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    stop = False
    for i in order:
        p = out / f"{cids[i]}.json"
        if p.exists():
            skipped += 1
            continue
        if a.budget_s and time.time() - t0 > a.budget_s:
            stop = True
            break
        t1 = time.time()
        m, n_ctx = rv.items[i]
        x, y = m[:, :-1], m[:, -1]
        yq, ym = y[n_ctx:].double(), y[:n_ctx].double().mean()
        seed0 = 419_500 + i                                    # the monitor's seed for this context
        with torch.no_grad():
            pc = DG.run_variant(model, "ridgeres", x, y, n_ctx, seed0)
            yr, used, bq = RES.residual_targets(x[None], y[None], n_ctx, torch.Generator().manual_seed(seed0))
            acc1 = torch.zeros(len(yq), dtype=torch.float64)
            acc2 = torch.zeros(len(yq), dtype=torch.float64)
            for j, S in enumerate(subsets(cids[i], x.shape[1])):
                xs = x[:, torch.from_numpy(S)]
                acc2 += DG.run_variant(model, "ridgeres", xs, y, n_ctx, SEED + 100 * i + j) / K
                acc1 += DG.predict(model, xs, yr[0, :n_ctx], n_ctx) / K
            p1 = acc1 + bq[0].cpu() if bool(used[0]) else acc1
        rq = DG.ridge_fit_predict(torch.nan_to_num(x[:n_ctx].double()), y[:n_ctx].double(), [torch.nan_to_num(x[n_ctx:].double())])[0]
        rec = {"cid": cids[i], "family": rv.family[i], "i": i, "F": int(x.shape[1]), "n_ctx": int(n_ctx), "n_q": int(len(yq)),
               "u_control": DG.u_of(pc, yq, ym), "u_T1": DG.u_of(p1, yq, ym), "u_T2": DG.u_of(acc2, yq, ym),
               "u_ridge": DG.u_of(rq, yq, ym), "residual_used": bool(used[0]),
               "finite": bool(torch.isfinite(pc).all() and torch.isfinite(p1).all() and torch.isfinite(acc2).all()),
               "wall_s": time.time() - t1}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        tmp.rename(p)
        done += 1
        print(json.dumps({"i": i, "cid": cids[i][:12], "F": rec["F"], "s": round(rec["wall_s"], 1)}), flush=True)
    st = {"schema": "jope.h428.bag/v1", "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
          "ckpt_sha256": a.ckpt_sha256, "ckpt_state": ck.get("state"), "size": a.size, "weights": "ema", "threads": 1,
          "device": "cpu", "cuda_visible": bool(torch.cuda.is_available()), "shard": a.shard, "wide_contexts": len(wide),
          "assigned": len(order), "written": done, "already_there": skipped, "budget_stop": stop,
          "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    (out / f"bag_status_{int(t0)}_s{k}of{n}.json").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
