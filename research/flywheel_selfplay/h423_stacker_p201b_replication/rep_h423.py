#!/usr/bin/env python3
"""H423: the frozen H422 combiner on the stored P201b matrices (a replication; nothing is fitted).

  rep_h423.py apply   --cache DIR --frozen FILE --frozen-sha256 HEX --board P201B_BOARD.json
                      --closed-board FILE [--closed-board FILE ...] --design-families TSV --out FILE
  rep_h423.py readout --apply FILE --out FILE

apply   refuses to run unless the cached contexts are exactly the 201 contexts of the published P201b board and none
        of their context ids, envelope hashes or family ids appears in a closed board definition (dev91). It then
        scores, per context: B = the stored h416.SXB value of the record (not recomputed) and S = the frozen combiner
        (sel_h422.context_stats / idx_of at their published bytes) on the same stored candidates and weights.
readout the test fixed in PREREG section 4: S vs B, h335_stats.primary_test, families as groups.
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
import sel_h422 as S  # noqa: E402

BOOT_SEED, N_BOOT = 423150, 10_000
MIN_INDEP = 5


def cmd_apply(a) -> None:
    t0 = time.process_time()
    raw = Path(a.frozen).read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.frozen_sha256:
        raise SystemExit("the frozen model file is not the one named by --frozen-sha256")
    fr = json.loads(raw)
    if fr["kgrid"] != [str(k) for k in S.KGRID] or fr["features"] != list(S.FEATS):
        raise SystemExit("the frozen model was fitted by other code")
    board = json.loads(Path(a.board).read_text())
    if board.get("board") != "p201b-train-audit-v1":
        raise SystemExit("--board is not the P201b board definition")
    bc = {x["cid"]: x for x in board["contexts"]}
    ctx = S.load_cache([a.cache], {"DL-Train"})
    if sorted(c["cid"] for c in ctx) != sorted(bc):
        raise SystemExit("the cached contexts are not exactly the contexts of the P201b board")
    closed = {"cid": set(), "envelope_sha256": set(), "family_id": set()}
    names = []
    for p in a.closed_board:
        b = json.loads(Path(p).read_text())
        names.append(b["board"])
        for x in b["contexts"]:
            for k in closed:
                closed[k].add(x[k])
    if not any(n.startswith("dev91") for n in names):
        raise SystemExit("no dev91 board definition was given")
    for x in bc.values():
        for k in closed:
            if x[k] in closed[k]:
                raise SystemExit(f"STOP: P201b {k} {x[k]} appears in a closed board ({names})")
    design_fams = {ln.split("\t")[1] for ln in Path(a.design_families).read_text().splitlines() if ln.strip()}
    per, worst, label_free = {}, 0.0, True
    for c in ctx:
        if c["family"] != bc[c["cid"]]["family_id"]:
            raise SystemExit(f"{c['cid']}: family differs from the board definition")
        if not np.isfinite(float(c["u_record"])):
            raise SystemExit(f"{c['cid']}: no stored baseline value")
        worst = max(worst, abs(float(c["u_record"]) - float(c["u_recomputed"])))
        s = S.context_stats(c, c["w_sxb"])
        i = int(S.idx_of(fr["final"], np.array([s["f"]]))[0])
        c2 = dict(c)
        c2["ya"] = np.zeros_like(c["ya"])   # the prediction must not depend on the AUDIT targets
        s2 = S.context_stats(c2, c["w_sxb"], with_audit_targets=False)
        label_free &= bool(np.array_equal(s2["P"], s["P"]) and s2["f"] == s["f"])
        per[c["cid"]] = {"B": float(c["u_record"]), "S": float(s["U"][i]), "k": str(S.KGRID[i]),
                         "stored_path_u": float(s["U"][-1]), "family": c["family"],
                         "family_in_design": bool(c["family"] in design_fams), "members": int((c["w_sxb"] > 0).sum())}
    res = {"schema": "jope.h423.apply/v1", "n": len(ctx), "cohort": "P201b (DL-Train)", "frozen_sha256": a.frozen_sha256,
           "closed_boards_checked": names, "closed_overlap": 0, "replay_max_abs_diff": float(worst),
           "lam0_max_abs_diff": float(max(abs(o["stored_path_u"] - o["B"]) for o in per.values())),
           "label_free": bool(label_free), "cpu_s": time.process_time() - t0, "per_context": per}
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("n", "closed_boards_checked", "replay_max_abs_diff", "lam0_max_abs_diff",
                                          "label_free", "cpu_s")}))


def cmd_readout(a) -> None:
    ap = json.loads(Path(a.apply).read_text())
    per = ap["per_context"]
    cids = sorted(per)
    fam_of = {c: per[c]["family"] for c in cids}
    T16, H = S._stats()
    vals = {x: {c: per[c][x] for c in cids} for x in ("B", "S")}
    prim = T16.pair_stats(np, H, vals["S"], vals["B"], fam_of)
    arms, point, boot, elo = T16.elo_boot(np, H, vals, fam_of, "B", BOOT_SEED, N_BOOT)
    de = T16.delta_elo(np, arms, point, boot, "S", "B")
    g = np.array([per[c]["S"] - per[c]["B"] for c in cids])
    res = {"schema": "jope.h423.readout/v1", "n": len(cids), "families": len(set(fam_of.values())), "cohort": ap["cohort"],
           "baseline": "stored h416.SXB values of the H416 P201b records, not recomputed",
           "family_mean_u": {x: float(H.family_mean(vals[x], fam_of)) for x in vals}, "S_vs_B": prim, "delta_elo": de,
           "k_counts": {k: int(sum(per[c]["k"] == k for c in cids)) for k in sorted({per[c]["k"] for c in cids})},
           "movers": {"up_0.05": int((g > 0.05).sum()), "down_0.05": int((g < -0.05).sum())},
           "reported": {"gain_at_least_0.010": bool(prim["clipped_mean"] >= 0.010),
                        "elo_ci_excludes_0": bool(de["ci95"][0] > 0)}}
    ind = [c for c in cids if not per[c]["family_in_design"]]
    res["families_outside_design"] = {"n": len(ind)}
    if len(ind) >= MIN_INDEP:
        res["families_outside_design"]["S_vs_B"] = T16.pair_stats(
            np, H, {c: per[c]["S"] for c in ind}, {c: per[c]["B"] for c in ind}, {c: fam_of[c] for c in ind})
    if prim["clipped_mean"] < 0 and prim["mirror_signflip_p"] < 0.025:
        res["verdict"] = "REP_HARM"
    elif prim["pass"]:
        res["verdict"] = "REP_CONFIRMED"
    else:
        res["verdict"] = "REP_NULL"
    print(f"== H423 {res['cohort']} n {res['n']} families {res['families']} baseline: {res['baseline']}")
    print(f"   mean u: B {res['family_mean_u']['B']:.4f}  S {res['family_mean_u']['S']:.4f}   k: {res['k_counts']}   movers {res['movers']}")
    for name, s in (("S_vs_B", prim), ("outside design", res["families_outside_design"].get("S_vs_B"))):
        if s is None:
            continue
        print(f"   {name:15s} gain {s['gain']:+.4f} u ({100 * s['pct_rmse']:+.2f}% RMSE) clipped {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} median {s['family_median']:+.4f} W/T/L {s['wtl']} "
              f"pass {s['pass']}")
    print(f"   dElo {de['delta']:+.1f} [{de['ci95'][0]:+.1f}, {de['ci95'][1]:+.1f}]  reported {res['reported']}  verdict {res['verdict']}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--cache", required=True)
    p.add_argument("--frozen", required=True)
    p.add_argument("--frozen-sha256", required=True)
    p.add_argument("--board", required=True)
    p.add_argument("--closed-board", action="append", required=True)
    p.add_argument("--design-families", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_apply)
    p = sub.add_parser("readout")
    p.add_argument("--apply", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_readout)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
