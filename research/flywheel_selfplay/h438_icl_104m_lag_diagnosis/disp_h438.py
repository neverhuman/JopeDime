#!/usr/bin/env python3
"""H438 addendum 2: how far do the weights move? Stored checkpoint copies, CPU, no forward pass.

  disp_h438.py run --mon DIR --out FILE

For each model (24M ICL_M_r3, 104M ICL_L_g1 worker 2) and its stored checkpoint copies of hours 12, 18 and 24, for
the raw weights and the EMA weights: the relative displacement ||theta_b - theta_a|| / ||theta_a|| over 12->18 and
18->24 (all floating-point parameters together, and by top-level module), the optimizer steps between the two
checkpoints, and the cosine between the two successive displacements. Tensors are read and subtracted; no model
is built and nothing is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

HOURS = ("h012", "h018", "h024")
MODELS = {"24M": "M_r3", "104M": "L_w2"}


def load(mon: Path, name: str, h: str):
    p = mon / f"{name}_{h}.pt"
    want = (mon / f"{name}_{h}.pt.sha256").read_text().split()[0]
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{p}: sha256 differs from its stored value")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return {"model": ck["model"], "ema": ck["ema"], "step": int(ck["state"]["step"]), "h": float(ck["state"]["elapsed"]) / 3600.0, "sha256": got}


def moves(a: dict, b: dict, c: dict) -> dict:
    """Displacements a->b and b->c of one weight set (dict name -> tensor)."""
    tot = {"aa": 0.0, "bb": 0.0, "d1": 0.0, "d2": 0.0, "d12": 0.0, "n": 0}
    grp = {}
    for k, ta in a.items():
        if not torch.is_floating_point(ta):
            continue
        x, y, z = ta.double(), b[k].double(), c[k].double()
        d1, d2 = y - x, z - y
        v = {"aa": float((x * x).sum()), "bb": float((y * y).sum()), "d1": float((d1 * d1).sum()), "d2": float((d2 * d2).sum()),
             "d12": float((d1 * d2).sum()), "n": int(x.numel())}
        g = grp.setdefault(k.split(".")[0], {"aa": 0.0, "bb": 0.0, "d1": 0.0, "d2": 0.0, "d12": 0.0, "n": 0})
        for q in v:
            tot[q] += v[q]
            g[q] += v[q]
    f = lambda t: {"parameters": t["n"], "rel_12_to_18": (t["d1"] / t["aa"]) ** 0.5, "rel_18_to_24": (t["d2"] / t["bb"]) ** 0.5,
                   "cos_successive": (t["d12"] / (t["d1"] * t["d2"]) ** 0.5) if t["d1"] > 0 and t["d2"] > 0 else None}
    return {"all": f(tot), "by_module": {k: f(v) for k, v in sorted(grp.items(), key=lambda kv: -kv[1]["n"])[:8]}}


def cmd_run(a) -> None:
    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(2)
    res = {"schema": "jope.h438.displacement/v1", "threads": 2, "models": {}}
    for size, name in MODELS.items():
        ck = [load(Path(a.mon), name, h) for h in HOURS]
        o = {"checkpoints": [{"hour_label": h, "h": round(c["h"], 2), "step": c["step"], "sha256": c["sha256"]} for h, c in zip(HOURS, ck)],
             "steps_12_to_18": ck[1]["step"] - ck[0]["step"], "steps_18_to_24": ck[2]["step"] - ck[1]["step"],
             "hours_12_to_18": ck[1]["h"] - ck[0]["h"], "hours_18_to_24": ck[2]["h"] - ck[1]["h"]}
        for w in ("model", "ema"):
            m = moves(ck[0][w], ck[1][w], ck[2][w])
            al = m["all"]
            al["rel_18_to_24_per_10k_steps"] = al["rel_18_to_24"] / (o["steps_18_to_24"] / 1e4)
            al["rel_12_to_18_per_10k_steps"] = al["rel_12_to_18"] / (o["steps_12_to_18"] / 1e4)
            al["rel_18_to_24_per_sqrt_10k_steps"] = al["rel_18_to_24"] / (o["steps_18_to_24"] / 1e4) ** 0.5
            al["rel_18_to_24_per_hour"] = al["rel_18_to_24"] / o["hours_18_to_24"]
            o["raw" if w == "model" else "ema"] = m
        res["models"][size] = o
        del ck
    L, M = res["models"]["104M"], res["models"]["24M"]
    res["ratio_104M_over_24M"] = {w: {"per_10k_steps_18_to_24": L[w]["all"]["rel_18_to_24_per_10k_steps"] / M[w]["all"]["rel_18_to_24_per_10k_steps"],
                                      "per_sqrt_10k_steps_18_to_24": L[w]["all"]["rel_18_to_24_per_sqrt_10k_steps"] / M[w]["all"]["rel_18_to_24_per_sqrt_10k_steps"],
                                      "per_hour_18_to_24": L[w]["all"]["rel_18_to_24_per_hour"] / M[w]["all"]["rel_18_to_24_per_hour"]} for w in ("raw", "ema")}
    R = res["ratio_104M_over_24M"]["raw"]["per_10k_steps_18_to_24"]
    res["reading"] = "MOVES_LESS_PER_STEP" if R < 0.5 else ("MOVES_MORE_PER_STEP" if R > 2.0 else "SIMILAR_STEP_SIZE")
    res["wall_s"], res["cpu_s"] = time.time() - t0, time.process_time() - c0
    for size, o in res["models"].items():
        for w in ("raw", "ema"):
            al = o[w]["all"]
            print(f"   {size} {w:3s}: steps {o['steps_12_to_18']} / {o['steps_18_to_24']}; rel 12->18 {al['rel_12_to_18']:.4f}, 18->24 {al['rel_18_to_24']:.4f} "
                  f"(per 10k steps {al['rel_18_to_24_per_10k_steps']:.5f}; per hour {al['rel_18_to_24_per_hour']:.5f}); cos of successive moves {al['cos_successive']:.3f}; "
                  f"modules {({k: round(v['rel_18_to_24'], 4) for k, v in o[w]['by_module'].items()})}")
    print(f"   ratio 104M / 24M: {res['ratio_104M_over_24M']}   reading {res['reading']}   ({res['cpu_s']:.0f} CPU-s)")
    with open(a.out, "x") as fo:
        json.dump(res, fo, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mon", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
