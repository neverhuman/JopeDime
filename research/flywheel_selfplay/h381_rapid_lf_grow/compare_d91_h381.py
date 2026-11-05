#!/usr/bin/env python3
"""H381 part B readout (PREREG §5): the final arms on the 5 selected dev91 contexts vs the STORED board rows.

Inputs: the board root (stored scores only; no entrant is refit, rerun or rescored), cids5.json, and per arm its d91
run dir (unit records) + output key. Board value of an arm = mean of its two search seeds (the h335.* entrant rule).
  per (cid, seed): u, RMSE = ba * exp(-u) for the arm and for the stored h335.H335_AGG per_seed value
  5-context AUDIT-Elo: h335_stats.elo_panel over every complete board entrant + the arms, anchor h335.H257_REPLICA
Direction only (5 families); nothing is written to the board.

  compare_d91_h381.py --board DIR --cids5 F --arm NAME=RUN_DIR:OUTPUT_KEY [...] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

H335_SRC = Path("/home/ubuntu/h335-stage/code")
sys.path.insert(0, str(H335_SRC))
import h335_stats as H  # noqa: E402

SEEDS = (335101, 335102)
REF = "h335.H335_AGG"
ANCHOR = "h335.H257_REPLICA"
BOARD_SHA = "40cb4828cf5c9a0d2d9cc0f32b7900819c992ec9a58c5d18f83ac84b29fa4069"
PRE_READ = "016648dc"  # PREREG §7.4: this context's stored h335.H335_AGG row was printed before the amendment


def board_values(bd: Path):
    board = json.loads((bd / "BOARD.json").read_text())
    if board["board_sha256"] != BOARD_SHA:
        raise SystemExit("not dev91-audit-v1")
    ctx = {c["cid"]: c for c in board["contexts"]}
    vals, recs = {}, {}
    for p in sorted((bd / "scores").glob("*.jsonl")):
        rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        got = {r["cid"]: float(r["utility"]) for r in rs}
        if set(got) == set(ctx):
            vals[p.stem] = got
            recs[p.stem] = {r["cid"]: r for r in rs}
    return ctx, vals, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--cids5", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx, vals, recs = board_values(Path(a.board))
    sel = json.loads(Path(a.cids5).read_text())
    cids = sel["cids"]
    fam_of = {c: ctx[c]["family_id"] for c in cids}
    if len(set(fam_of.values())) != 5:
        raise SystemExit("cids5 must span 5 families")
    base = {c: next(float(r["base_rmse_audit"]) for e, rr in recs.items() if "base_rmse_audit" in rr[c]
                    for r in [rr[c]]) for c in cids}
    ref_seed = {c: [float(x) for x in recs[REF][c]["per_seed"]] for c in cids}
    rep = {"schema": "jope.h381.d91-compare/v1", "cids": cids, "families": [fam_of[c] for c in cids],
           "reference": REF, "arms": {}}
    arm_vals = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        run, key = rest.rsplit(":", 1)
        rows, per_ctx = [], {}
        for c in cids:
            us = []
            for i, s in enumerate(SEEDS):
                p = Path(run) / "stages" / "C_AGG" / f"{c}__{s}.jsonl"
                r = [json.loads(x) for x in p.read_text().splitlines() if x.strip() and '"result"' in x][-1]
                if abs(float(r["ba"]) - base[c]) > 1e-9 * max(1.0, base[c]):
                    raise SystemExit(f"{name} {c[:12]}: ba {r['ba']} != board base_rmse_audit {base[c]}")
                u = float(r["outputs"][key]["value"])
                u0 = ref_seed[c][i]
                rm, rm0 = base[c] * math.exp(-u), base[c] * math.exp(-u0)
                rows.append({"cid": c, "family": fam_of[c], "seed": s, "u_arm": u, "u_ref": u0, "delta_u": u - u0,
                             "rmse_arm": rm, "rmse_ref": rm0, "rmse_change_pct": 100.0 * (rm / rm0 - 1.0),
                             "invalid": bool(r["outputs"][key].get("invalid"))})
                us.append(u)
            per_ctx[c] = sum(us) / len(us)
        arm_vals[f"h381.{name}"] = per_ctx
        d = [r["delta_u"] for r in rows]
        rep["arms"][name] = {"run": run, "output": key, "rows": rows,
                             "contexts_better": sum(1 for c in cids if per_ctx[c] > sum(ref_seed[c]) / 2 + 1e-4),
                             "contexts_worse": sum(1 for c in cids if per_ctx[c] < sum(ref_seed[c]) / 2 - 1e-4),
                             "mean_delta_u": sum(d) / len(d),
                             "mean_rmse_change_pct": sum(r["rmse_change_pct"] for r in rows) / len(rows)}
    v5 = {e: {c: v[c] for c in cids} for e, v in vals.items()}
    v5.update(arm_vals)
    elo = H.elo_panel(v5, fam_of, ANCHOR)
    order = sorted(v5, key=lambda e: -elo["elo"][e]["elo"])
    rep["elo5"] = {e: {"rank": i + 1, "elo": elo["elo"][e]["elo"], "ci95": elo["elo"][e]["ci95"]}
                   for i, e in enumerate(order)}
    rep["note"] = "5 contexts / 5 families: direction only (best sign-flip p = 1/32); not added to the board"
    rep["pre_amendment_board_read"] = {"cid_prefix": PRE_READ, "selected": [c for c in cids if c.startswith(PRE_READ)]}
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    for n, r in rep["arms"].items():
        print(f"{n}: better {r['contexts_better']}/5 worse {r['contexts_worse']}/5 mean du {r['mean_delta_u']:+.4f} "
              f"mean RMSE change {r['mean_rmse_change_pct']:+.2f}%")
    for e in order:
        print(f"{rep['elo5'][e]['rank']:>2} {e:34s} {rep['elo5'][e]['elo']:7.1f}")


if __name__ == "__main__":
    main()
