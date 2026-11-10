#!/usr/bin/env python3
"""H364 PREREG §4.2-§4.3: choose the second-level constants on the two USED TRAIN panels only.

  P201b  H335 Phase T T_AGG (seed 335101, runs T_xbabe0 / T_xbabe2) + H354 Phase D classical-library outputs
  P201c  H357 C_AGG (seed 357101, run H357_C_xbabe0)               + H357 library outputs

Every one of the 144 grid designs is scored on all 395 contexts (AUDIT utilities that H354/H357 already computed and
reported); the design with the largest family LCB is chosen (ties -> conservative). Writes $H364_DATA/dev/ (per-context
records, grid table) and DESIGN.json. Of P201d only the cid list (h364/P201d.json) is read, to assert disjointness;
no P201d view, GP, library or score output is opened. Run from a staged code root (stage_h364.sh).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h364_core as H  # noqa: E402  (installs the H317 guard via h354_core)

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from typing import List  # noqa: E402

import numpy as np  # noqa: E402

K = H.K
OUT = Path(os.environ.get("H364_DATA", "/home/ubuntu/jope-data/h364-2l")) / "dev"
H335_DATA = K.H335_DATA
LIB_B = Path("/home/ubuntu/jope-data/h354-xs/D/lib")
LIB_C = Path("/home/ubuntu/jope-data/h357-xc/lib")
RUN_C = H335_DATA / "runs" / "H357_C_xbabe0"
PANEL_C = H335_DATA / "h357" / "P201c.json"
P201C_CIDS_SHA256 = "36fbf6a92204487a31f2533905a57393e9c317fa7484ab7cbde5026e34468f6d"
P201D_PANEL = H335_DATA / "h364" / "P201d.json"


def dev_contexts() -> List[dict]:
    """(panel, cid, family) for P201b then P201c; asserts the pins and that nothing is a P201d context."""
    fam_b = K.family_map()
    out = [{"panel": "P201b", "cid": c, "family": fam_b[c]} for c in K.p201b_cids()]
    pan = json.loads(PANEL_C.read_text())
    if hashlib.sha256("\n".join(pan["cids"]).encode()).hexdigest() != P201C_CIDS_SHA256:
        raise AssertionError("P201c panel pin")
    out += [{"panel": "P201c", "cid": c, "family": pan["family_of"][c]} for c in pan["cids"]]
    d = set(json.loads(P201D_PANEL.read_text())["cids"])  # the cid LIST only (no P201d data exists on this path)
    if d & {r["cid"] for r in out}:
        raise AssertionError("a dev context is a P201d context")
    return out


def one(row: dict) -> dict:
    designs = H.grid()
    fam = {row["cid"]: row["family"]}
    if row["panel"] == "P201b":
        ctx = K.load_context(row["cid"], fam)
        lib = LIB_B
    else:
        ctx = K.load_context(row["cid"], fam, runs=(RUN_C,), stage="C_AGG", seed=357101)
        lib = LIB_C
    ls, la = H.lib_from_dir(ctx, lib)
    rec = H.evaluate(ctx, ls, la, designs)
    rec["panel"] = row["panel"]
    return rec


def grid_table(recs: List[dict]) -> List[dict]:
    fam_of = {r["cid"]: r["family"] for r in recs}
    rows = []
    for d in H.grid():
        k = d.key()
        gain = {r["cid"]: r["designs"][k]["u"] - r["x0_u"] for r in recs}
        st = H.family_lcb(gain, fam_of)
        g = np.array(list(gain.values()))
        st.update({"design": d, "key": k, "ctx_lt_minus_0p05": int((g < -0.05).sum()), "ctx_min": float(g.min()),
                   "active_share": float(np.mean([r["designs"][k]["active"] for r in recs])),
                   "mean_beta_sum": float(np.mean([r["designs"][k]["beta_sum"] for r in recs])),
                   "by_panel": {p: float(np.mean([gain[r["cid"]] for r in recs if r["panel"] == p]))
                                for p in ("P201b", "P201c")}})
        rows.append(st)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    rows_in = dev_contexts()
    OUT.mkdir(parents=True, exist_ok=True)
    recp = OUT / "dev_contexts.jsonl"
    if recp.exists():
        raise SystemExit(f"{recp} exists (create-once)")
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(one, rows_in, chunksize=4))
    with open(recp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    chk = {"dlr": max(r["x0_check"]["dlr"] for r in recs), "du": max(r["x0_check"]["du"] for r in recs)}
    table = grid_table(recs)
    chosen = H.select(table)
    d = chosen["design"]
    ng = next(r for r in table if r["design"] == H.Design(d.c, d.lam, d.delta, math.inf))

    def slim(r: dict) -> dict:
        return {k: (v.as_json() if k == "design" else v) for k, v in r.items()}

    K.jdump(OUT / "dev_grid.json", {"schema": "jope.h364.dev-grid/v1", "rows": [slim(r) for r in table]})
    gsha = K.sha256_file(OUT / "dev_grid.json")
    design = {"schema": "jope.h364.design/v1", "wave": "H364", "part": "B",
              "rule": "PREREG §4.3: max family LCB = mean - 1.96 sd/sqrt(F) of the unclipped per-family AUDIT gain "
                      "2L - X0 on P201b + P201c; ties within 1e-6 -> smaller c, larger lambda, larger delta, smaller tau",
              "chosen": d.as_json(), "chosen_key": d.key(), "noguard_key": ng["key"],
              "dev": {"contexts": len(recs), "panels": {"P201b": sum(r["panel"] == "P201b" for r in recs),
                                                        "P201c": sum(r["panel"] == "P201c" for r in recs)},
                      "x0_check_worst": chk, "chosen": slim(chosen), "noguard": slim(ng)},
              "dev_grid_sha256": gsha, "dev_contexts_sha256": K.sha256_file(recp)}
    K.jdump(OUT / "DESIGN.json", design)
    print(json.dumps({"chosen": d.as_json(), "lcb": chosen["lcb"], "mean": chosen["mean"],
                      "wtl": [chosen["won"], chosen["tied"], chosen["lost"]], "x0_check": chk}, indent=1))


if __name__ == "__main__":
    main()
