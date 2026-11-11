#!/usr/bin/env python3
"""H432 part 2: how good must an added column be to earn stack weight? Stored matrices only; no model pass.

  need_h432.py run     --run-dir DIR --tag TBOTH --apply H424_DEV.apply.json --cols DIR --bank FILE --out DIR [--shard k/n]
  need_h432.py summary --dir DIR --out FILE
  need_h432.py selftest

run, per DEV context: the stored out-of-fold matrix Q (64 candidates) and SEARCH targets y, plus one synthetic 65th
column at a fixed out-of-fold distance d behind the best stored candidate (its RMSE = best stored RMSE x exp(d)), and
the stored weight builder (meta_h416 BCAR, the call H424 made). Recorded: the weight of the added column.
  S  same error pattern: the H424 column's own out-of-fold error, rescaled to the target RMSE (all 115 tables)
  N  independent noise: seeded Gaussian error at the target RMSE, 3 seeds (wide tables only; an optimistic bound,
     because no real model has errors unrelated to every other candidate's)
Replay check: the unscaled H424 column must get exactly the weight stored in H424's value file.
Only SEARCH-side arrays are used. No AUDIT value is computed; nothing here is a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GRID = (-0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
WIDE, SEEDS, NEED = 30, 3, 0.10


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v ** 2)))


def noise(cid: str, j: int, n: int) -> np.ndarray:
    s = int.from_bytes(hashlib.sha256(f"H432-noise\0{cid}\0{j}".encode()).digest()[:8], "big")
    e = np.random.default_rng(s).standard_normal(n)
    return e / rms(e)


def levels(bcar, Q: np.ndarray, y: np.ndarray, e_col: np.ndarray, cid: str, wide: bool) -> dict:
    """Weights of a synthetic 65th column at each grid distance (e_col = the real column's out-of-fold error)."""
    best = float(np.sqrt(((Q - y[:, None]) ** 2).mean(0)).min())
    out = {"rmse_best_stored": best, "rmse_column": rms(e_col), "d_actual": float(np.log(rms(e_col) / best)), "S": [], "N": []}
    for d in GRID:
        target = best * float(np.exp(d))
        w = bcar(np.column_stack([Q, y + e_col * (target / rms(e_col))]), y, cid)
        out["S"].append({"d": d, "w": float(w[-1]), "members": int((w > 0).sum())})
        if wide:
            ws = [float(bcar(np.column_stack([Q, y + noise(cid, j, len(y)) * target]), y, cid)[-1]) for j in range(SEEDS)]
            out["N"].append({"d": d, "w": float(np.mean(ws)), "w_seeds": ws})
    return out


def cmd_run(a) -> None:
    import cache_h422 as CA

    t0, c0 = time.time(), time.process_time()
    per = json.loads(Path(a.apply).read_text())["per_context"]
    z = np.load(a.bank, allow_pickle=False)
    Fn = {str(c): int(s[1]) - 1 for c, s in zip(z["cids"], z["s_shape"])}
    cids = sorted(per)
    if sorted(Fn) != cids:
        raise SystemExit("the bank does not hold exactly the contexts of the value file")
    k, n = (int(v) for v in a.shard.split("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for cid in cids[k::n]:
        p = out / f"{cid}.json"
        if p.exists():
            continue
        t1 = time.process_time()
        _, st = CA.stored(a.run_dir, a.tag, cid)
        q = np.load(Path(a.cols) / f"{cid}.npz", allow_pickle=False)["q_ridgeres"].astype(np.float64)
        Q, y = st["Q"], st["y"]
        if len(q) != len(y) or not per[cid]["column_ok"]:
            raise SystemExit(f"{cid}: the stored column does not fit the stored matrix")
        w_re = float(CA.bcar(np.column_stack([Q, q]), y, cid)[-1])
        r = levels(CA.bcar, Q, y, q - y, cid, Fn[cid] > WIDE)
        r.update({"cid": cid, "features": Fn[cid], "wide": bool(Fn[cid] > WIDE), "n_search": int(len(y)), "w_actual_stored": float(per[cid]["w_added_I"][0]),
                  "w_actual_replayed": w_re, "replay_abs_diff": abs(w_re - float(per[cid]["w_added_I"][0])), "cpu_s": time.process_time() - t1})
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, sort_keys=True))
        tmp.rename(p)
        done += 1
    st = {"schema": "jope.h432.need-run/v1", "shard": a.shard, "written": done, "threads": 1, "torch_loaded": "torch" in sys.modules,
          "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    (out / f"run_status_{int(t0)}_s{k}of{n}.status").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st))


def need_d(rows: list) -> float:
    """Distance at which the weight first falls below NEED, going from the best level outwards (linear interpolation).
    -inf: below NEED even at the best level; +inf: at least NEED on the whole grid."""
    if rows[0]["w"] < NEED:
        return float("-inf")
    for lo, hi in zip(rows, rows[1:]):
        if hi["w"] < NEED:
            return float(lo["d"] + (hi["d"] - lo["d"]) * (lo["w"] - NEED) / (lo["w"] - hi["w"]))
    return float("inf")


def cmd_summary(a) -> None:
    rs = [json.loads(p.read_text()) for p in sorted(Path(a.dir).glob("*.json"))]
    if len(rs) != 115:
        raise SystemExit(f"{len(rs)} contexts, expected 115")

    def curve(sel, model):
        out = []
        for i, d in enumerate(GRID):
            w = np.array([r[model][i]["w"] for r in sel])
            out.append({"d": d, "mean_weight": float(w.mean()), "median_weight": float(np.median(w)), "share_weight_ge_0.10": float((w >= NEED).mean()),
                        "share_weight_gt_0": float((w > 0).mean())})
        nd = np.array([need_d(r[model]) for r in sel])
        q = np.quantile(nd, [0.25, 0.5, 0.75], method="inverted_cdf")
        return {"tables": len(sel), "by_distance": out,
                "need_d": {"quartiles": [None if not np.isfinite(v) else float(v) for v in q], "quartiles_raw": [str(v) for v in q],
                           "below_need_at_best_level": int(np.isneginf(nd).sum()), "need_met_on_whole_grid": int(np.isposinf(nd).sum())}}

    wide, narrow = [r for r in rs if r["wide"]], [r for r in rs if not r["wide"]]

    def actual(sel):
        d, w = np.array([r["d_actual"] for r in sel]), np.array([r["w_actual_stored"] for r in sel])
        bins = []
        for lo, hi in ((-9, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 9)):
            k = (d >= lo) & (d < hi)
            bins.append({"d_from": lo, "d_to": hi, "tables": int(k.sum()), "mean_weight": float(w[k].mean()) if k.any() else None,
                         "share_weight_ge_0.10": float((w[k] >= NEED).mean()) if k.any() else None})
        return {"tables": len(sel), "d_actual_mean": float(d.mean()), "d_actual_median": float(np.median(d)), "d_actual_min": float(d.min()),
                "weight_mean": float(w.mean()), "share_weight_ge_0.10": float((w >= NEED).mean()), "by_actual_distance": bins}

    res = {"schema": "jope.h432.need/v1", "grid": list(GRID), "need_weight": NEED, "noise_seeds": SEEDS,
           "replay_max_abs_diff": float(max(r["replay_abs_diff"] for r in rs)), "cpu_s": float(sum(r["cpu_s"] for r in rs)),
           "S_wide": curve(wide, "S"), "S_narrow": curve(narrow, "S"), "N_wide": curve(wide, "N"),
           "actual_wide": actual(wide), "actual_narrow": actual(narrow), "actual_all": actual(rs)}
    print(f"== H432 part 2: replay max diff {res['replay_max_abs_diff']:.1e}; {len(wide)} wide, {len(narrow)} narrow; {res['cpu_s']:.0f} CPU-s")
    for name in ("S_wide", "S_narrow", "N_wide"):
        c = res[name]
        print(f"   {name}: need_d quartiles {c['need_d']['quartiles_raw']}  below at best level {c['need_d']['below_need_at_best_level']}"
              f"  met on whole grid {c['need_d']['need_met_on_whole_grid']}")
        print("      d      " + " ".join(f"{x['d']:+6.2f}" for x in c["by_distance"]))
        print("      mean w " + " ".join(f"{x['mean_weight']:6.3f}" for x in c["by_distance"]))
        print("      >=0.10 " + " ".join(f"{x['share_weight_ge_0.10']:6.2f}" for x in c["by_distance"]))
    for name in ("actual_wide", "actual_narrow"):
        c = res[name]
        print(f"   {name}: d mean {c['d_actual_mean']:.3f} median {c['d_actual_median']:.3f} min {c['d_actual_min']:.3f}  weight {c['weight_mean']:.3f}"
              f"  share >= 0.10 {c['share_weight_ge_0.10']:.2f}  bins {[(b['d_to'], b['tables'], b['mean_weight']) for b in c['by_actual_distance']]}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


def cmd_selftest(a) -> None:
    """Code path on random numbers only (no stored matrix is opened)."""
    import cache_h422 as CA

    r = np.random.default_rng(1)
    y = r.standard_normal(400)
    Q = y[:, None] + r.standard_normal((400, 8)) * np.linspace(0.3, 1.0, 8)
    out = levels(CA.bcar, Q, y, r.standard_normal(400) * 0.8, "selftest", True)
    print("selftest S", [round(x["w"], 3) for x in out["S"]], "N", [round(x["w"], 3) for x in out["N"]], "need_d", need_d(out["S"]), need_d(out["N"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    for k in ("--run-dir", "--tag", "--apply", "--cols", "--bank", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--shard", default="0/1")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("summary")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
