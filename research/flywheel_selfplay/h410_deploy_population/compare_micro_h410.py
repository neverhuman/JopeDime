#!/usr/bin/env python3
"""H410 part-B readout (PREREG §4, §5, §8.4): every H410 output on the 5 dev91 contexts vs the STORED board rows.

Inputs: the stored dev91-audit-v1 board (no entrant is refit, rerun or rescored), cids5.json, the H410 deploy stage
records (stages/H410_<TAG>/<cid>__pool.jsonl), GP arm records (stages/C_AGG/<cid>__<seed>.jsonl) and, optionally, the
A0 headroom extract and the BIG-builder receipt.

Output kinds (deploy records):
  per-seed  "<src>.s<seed>.stock", "own.<src>.s<seed>.<builder>"   -> arm "<TAG>:<src>.stock" / "<TAG>:own.<src>.<b>"
            value per context = mean over the two search seeds (the h335.* entrant rule); rows per seed vs the stored
            h335.H335_AGG per_seed value
  pooled    everything else ("PH.W24", "P2.RUNBAG", "LIB.single", "P2+LIB.BAG_CAR", ...) -> "<TAG>:<key>"
            one value per context, vs the stored H335_AGG seed mean
A missing "BAG_<b>" output (no bag-eligible member) falls back to "<b>" (twin_spec made no change) and is flagged.

Checks: ba == board base_rmse_audit; AB / BIG "H335.s<seed>.stock" == stored h335.H335_AGG per_seed;
"E2L.s<seed>.stock" == the H381 E2L record (--e2l); BIG "C1.s<seed>.stock" == the C1 GP record.

Decisions (PREREG): big_builder (§3 BIG: the best Tier A builder of the AB P2 pool, chosen before BIG is read),
train_primary (§8.4), thesis_on_5 (§5, needs the BIG receipt).

  compare_micro_h410.py --board DIR --cids5 F --deploy TAG=RUN_DIR [...] [--gp NAME=RUN_DIR:KEY ...]
                        [--e2l RUN_DIR:KEY] [--a0 A0.json] [--big-receipt F] [--elo-all 0|1] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

H335_SRC = Path("/home/ubuntu/h335-stage/code")
sys.path.insert(0, str(H335_SRC))
import h335_stats as H  # noqa: E402

SEEDS = (335101, 335102)
REF = "h335.H335_AGG"
ANCHOR = "h335.H257_REPLICA"
BOARD_SHA = "40cb4828cf5c9a0d2d9cc0f32b7900819c992ec9a58c5d18f83ac84b29fa4069"
REFS = {"tabpfn35": "tabpfn_v3p5.v1", "tabfm_ens": "tabfm_ensemble.v1", "kumo": "kumo_tabular_large_e8.v1"}
TIER_A = ("W24", "W48", "CAR", "NNLS")                       # §3 BIG candidates; ties -> earlier
TRAIN_CAND = ("W24", "W48", "CAR", "NNLS", "BAG_W24", "BAG_W48", "BAG_CAR")   # §8.4; ties -> earlier
TIER_A_POOLED = ("RUNBAG",) + TRAIN_CAND                     # §5.2: Tier A pooled outputs (no library)
TOL = 1e-6
PER_SEED = re.compile(r"^(?P<pre>own\.[^.]+|[^.+]+)\.s(?P<seed>\d+)\.(?P<b>.+)$")


def last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


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


def summarize(name, per_ctx, rows, cids, ref_mean, refv, base):
    d = {c: per_ctx[c] - ref_mean[c] for c in cids}
    rm = {c: 100.0 * (math.exp(-(per_ctx[c] - ref_mean[c])) - 1.0) for c in cids}
    gap = {}
    for c in cids:
        den = refv["tabpfn35"][c] - ref_mean[c]
        gap[c] = (per_ctx[c] - ref_mean[c]) / den if den > 0 else None
    g = [v for v in gap.values() if v is not None]
    o = {"per_ctx_u": per_ctx, "delta_u": d, "rmse_change_pct_vs_ref_mean": rm,
         "mean_delta_u": sum(d.values()) / len(cids),
         "contexts_better": sum(1 for c in cids if d[c] > 1e-4), "contexts_worse": sum(1 for c in cids if d[c] < -1e-4),
         "gap_to_tabpfn35_closed": gap, "mean_gap_closed": sum(g) / len(g) if g else None,
         "beats": {k: sum(1 for c in cids if per_ctx[c] > refv[k][c]) for k in refv}}
    if rows is not None:
        o["rows"] = rows
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True)
    ap.add_argument("--cids5", required=True)
    ap.add_argument("--deploy", action="append", default=[])
    ap.add_argument("--gp", action="append", default=[])
    ap.add_argument("--e2l")
    ap.add_argument("--a0")
    ap.add_argument("--big-receipt")
    ap.add_argument("--elo-all", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx, vals, recs = board_values(Path(a.board))
    cids = json.loads(Path(a.cids5).read_text())["cids"]
    fam_of = {c: ctx[c]["family_id"] for c in cids}
    if len(set(fam_of.values())) != 5:
        raise SystemExit("cids5 must span 5 families")
    base = {c: next(float(rr[c]["base_rmse_audit"]) for rr in recs.values() if "base_rmse_audit" in rr[c])
            for c in cids}
    ref_seed = {c: [float(x) for x in recs[REF][c]["per_seed"]] for c in cids}
    ref_mean = {c: sum(v) / len(v) for c, v in ref_seed.items()}
    refv = {k: {c: vals[e][c] for c in cids} for k, e in REFS.items()}
    rep = {"schema": "jope.h410.d91-micro-compare/v1", "board_sha256": BOARD_SHA, "cids": cids,
           "families": [fam_of[c] for c in cids], "reference": REF, "reference_per_seed": ref_seed,
           "reference_seed_mean": ref_mean, "references": {k: {"entrant": REFS[k], "u": refv[k]} for k in REFS},
           "checks": {}, "arms": {}, "fallbacks": {}}

    def check_ba(tag, c, ba):
        if abs(float(ba) - base[c]) > 1e-9 * max(1.0, base[c]):
            raise SystemExit(f"{tag} {c[:12]}: ba {ba} != board base_rmse_audit {base[c]}")

    # GP arms (per-seed C_AGG records)
    gp_seed = {}
    for spec in a.gp:
        name, rest = spec.split("=", 1)
        run, key = rest.rsplit(":", 1)
        per, rows = {}, []
        for c in cids:
            us = []
            for i, s in enumerate(SEEDS):
                r = last_result(Path(run) / "stages" / "C_AGG" / f"{c}__{s}.jsonl")
                check_ba(name, c, r["ba"])
                u = float(r["outputs"][key]["value"])
                gp_seed[(name, c, s)] = u
                rows.append({"cid": c, "seed": s, "u_arm": u, "u_ref": ref_seed[c][i],
                             "delta_u": u - ref_seed[c][i], "invalid": bool(r["outputs"][key].get("invalid"))})
                us.append(u)
            per[c] = sum(us) / len(us)
        rep["arms"][f"GP:{name}"] = summarize(name, per, rows, cids, ref_mean, refv, base)
    e2l_seed = {}
    if a.e2l:
        run, key = a.e2l.rsplit(":", 1)
        for c in cids:
            for s in SEEDS:
                e2l_seed[(c, s)] = float(last_result(Path(run) / "stages" / "C_AGG" / f"{c}__{s}.jsonl")
                                         ["outputs"][key]["value"])

    # deploy arms
    for spec in a.deploy:
        tag, run = spec.split("=", 1)
        dr = {c: last_result(Path(run) / "stages" / f"H410_{tag}" / f"{c}__pool.jsonl") for c in cids}
        for c in cids:
            check_ba(tag, c, dr[c]["ba"])
        keys = sorted(set().union(*[set(r["outputs"]) for r in dr.values()]))
        per_seed_arms, pooled = {}, set()
        for k in keys:
            m = PER_SEED.match(k)
            if m and int(m.group("seed")) in SEEDS:
                per_seed_arms.setdefault(f"{m.group('pre')}.{m.group('b')}", {})[int(m.group("seed"))] = k
            else:
                pooled.add(k)

        def val(k, c):
            o = dr[c]["outputs"].get(k)
            if o is None:
                b = re.sub(r"BAG_", "", k)
                if b != k and b in dr[c]["outputs"]:
                    rep["fallbacks"].setdefault(f"{tag}:{k}", []).append(c[:12])
                    o = dr[c]["outputs"][b]
                else:
                    return None
            return float(o["value"]), bool(o.get("invalid"))

        chk = rep["checks"].setdefault(tag, {})
        for name, by_seed in sorted(per_seed_arms.items()):
            per, rows, ok = {}, [], True
            for c in cids:
                us = []
                for i, s in enumerate(SEEDS):
                    k = by_seed.get(s) or by_seed[SEEDS[1 - i]].replace(f".s{SEEDS[1 - i]}.", f".s{s}.")
                    v = val(k, c)
                    if v is None:
                        ok = False
                        break
                    rows.append({"cid": c, "seed": s, "u_arm": v[0], "u_ref": ref_seed[c][i],
                                 "delta_u": v[0] - ref_seed[c][i], "invalid": v[1]})
                    us.append(v[0])
                if not ok:
                    break
                per[c] = sum(us) / len(us)
            if ok:
                rep["arms"][f"{tag}:{name}"] = summarize(name, per, rows, cids, ref_mean, refv, base)
        for k in sorted(pooled):
            vs = {c: val(k, c) for c in cids}
            if any(v is None for v in vs.values()):
                continue
            o = summarize(k, {c: v[0] for c, v in vs.items()}, None, cids, ref_mean, refv, base)
            o["invalid_contexts"] = sum(1 for v in vs.values() if v[1])
            rep["arms"][f"{tag}:{k}"] = o
        # stock checks
        for c in cids:
            for i, s in enumerate(SEEDS):
                o = dr[c]["outputs"]
                if f"H335.s{s}.stock" in o:
                    chk.setdefault("H335_stock_vs_board_max_abs", 0.0)
                    chk["H335_stock_vs_board_max_abs"] = max(chk["H335_stock_vs_board_max_abs"],
                                                             abs(o[f"H335.s{s}.stock"]["value"] - ref_seed[c][i]))
                if f"E2L.s{s}.stock" in o and e2l_seed:
                    chk.setdefault("E2L_stock_vs_h381_max_abs", 0.0)
                    chk["E2L_stock_vs_h381_max_abs"] = max(chk["E2L_stock_vs_h381_max_abs"],
                                                           abs(o[f"E2L.s{s}.stock"]["value"] - e2l_seed[(c, s)]))
                for (gname, gc, gs), gu in gp_seed.items():
                    if gc == c and gs == s and f"{gname}.s{s}.stock" in o:
                        kk = f"{gname}_stock_vs_gp_max_abs"
                        chk[kk] = max(chk.get(kk, 0.0), abs(o[f"{gname}.s{s}.stock"]["value"] - gu))
        chk["pass"] = all(v <= TOL for k, v in chk.items() if k.endswith("_max_abs"))
        rep.setdefault("pool_sizes", {})[tag] = {c[:12]: dr[c].get("pool_sizes") for c in cids}
        rep.setdefault("library", {})[tag] = {c[:12]: dr[c].get("library") for c in cids}
        rep.setdefault("wall_s", {})[tag] = {c[:12]: dr[c].get("wall_s") for c in cids}

    if a.a0:
        rep["a0_headroom"] = json.loads(Path(a.a0).read_text())

    # decisions
    arms = rep["arms"]
    du = {k: v["mean_delta_u"] for k, v in arms.items()}
    dec = {}
    if all(f"AB:P2.{b}" in du for b in TIER_A):
        best = max(TIER_A, key=lambda b: (du[f"AB:P2.{b}"], -TIER_A.index(b)))
        dec["big_builder"] = {"rule": "PREREG §3 BIG: best Tier A builder (W24/W48/CAR/NNLS, ties -> earlier) by mean "
                                      "du of the AB P2 pool; BIG output = P3+LIB.BAG_<b> (NNLS: P3+LIB.NNLS)",
                              "builder": best, "mean_du": {b: du[f"AB:P2.{b}"] for b in TIER_A},
                              "big_output": f"BIG:P3+LIB.{'NNLS' if best == 'NNLS' else 'BAG_' + best}"}
    if all(f"AB:PH.{b}" in du for b in TRAIN_CAND):
        tb = max(TRAIN_CAND, key=lambda b: (du[f"AB:PH.{b}"], -TRAIN_CAND.index(b)))
        lib = {k: du[k] for k in du if k.startswith("AB:P2+LIB.") or k == "AB:LIB.single"}
        dec["train_primary"] = {"rule": "PREREG §8.4", "builder": tb,
                                "mean_du": {b: du[f"AB:PH.{b}"] for b in TRAIN_CAND},
                                "train_outputs": [f"own.{tb}", f"P2.{tb}"] +
                                                 ([f"P2+LIB.{tb}"] if lib and max(lib.values()) > du[f"AB:PH.{tb}"]
                                                  else []),
                                "library_best": max(lib.items(), key=lambda kv: kv[1]) if lib else None}
    if a.big_receipt:
        br = json.loads(Path(a.big_receipt).read_text())
        bo = br["big_output"]
        if bo in arms:
            ta = {k: du[k] for k in du if re.match(r"^AB:(PH|P2)\.(%s)$" % "|".join(TIER_A_POOLED), k)}
            bta = max(ta.items(), key=lambda kv: kv[1])
            c1 = arms[bo]["contexts_better"] >= 4 and du[bo] >= 0.10
            c2 = du[bo] > 0 and bta[1] >= 0.5 * du[bo]
            dec["thesis_on_5"] = {"rule": "PREREG §5", "big_output": bo, "big_receipt_sha256_file": a.big_receipt,
                                  "big_contexts_better": arms[bo]["contexts_better"], "big_mean_du": du[bo],
                                  "best_tier_a_pooled": bta, "criterion_1": c1, "criterion_2": c2,
                                  "verdict": "THESIS_SUPPORTED_ON_5" if c1 and c2 else "NOT_SUPPORTED"}
    rep["decisions"] = dec

    # Elo: each arm inserted alone into the complete board (rank among entrants), + one headline panel
    v5 = {e: {c: v[c] for c in cids} for e, v in vals.items()}
    if a.elo_all:
        for k, v in arms.items():
            panel = dict(v5)
            panel[f"h410.{k}"] = v["per_ctx_u"]
            el = H.elo_panel(panel, fam_of, ANCHOR, n_boot=2000)["elo"]
            order = sorted(panel, key=lambda e: -el[e]["elo"])
            v["elo5_alone"] = {"elo": el[f"h410.{k}"]["elo"], "rank": order.index(f"h410.{k}") + 1,
                               "of": len(panel)}
    head = [k for k in ("GP:C1", "AB:H335.stock", "AB:E2L.stock", "AB:P2.RUNBAG") if k in arms]
    for d in dec.values():
        if isinstance(d, dict) and d.get("big_output") in arms:
            head.append(d["big_output"])
    if arms:
        best3 = sorted(arms, key=lambda k: -du[k])[:3]
        head += [k for k in best3 if k not in head]
        panel = dict(v5)
        panel.update({f"h410.{k}": arms[k]["per_ctx_u"] for k in head})
        el = H.elo_panel(panel, fam_of, ANCHOR)
        order = sorted(panel, key=lambda e: -el["elo"][e]["elo"])
        rep["elo5_headline"] = {e: {"rank": i + 1, "elo": el["elo"][e]["elo"], "ci95": el["elo"][e]["ci95"]}
                                for i, e in enumerate(order)}
    rep["note"] = ("5 contexts / 5 families, adaptively tuned on (owner waiver 2026-11-06): direction only "
                   "(best sign-flip p = 1/32); not added to the board")
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")

    print("checks", json.dumps(rep["checks"]))
    print(f"{'ctx':12s} " + " ".join(f"{c[:8]:>8s}" for c in cids))
    print(f"{'H335_AGG':12s} " + " ".join(f"{ref_mean[c]:8.3f}" for c in cids))
    for k in REFS:
        print(f"{k:12s} " + " ".join(f"{refv[k][c]:8.3f}" for c in cids))
    for k in sorted(arms, key=lambda k: -du[k]):
        v = arms[k]
        mg = v["mean_gap_closed"]
        el = v.get("elo5_alone", {})
        print(f"{k:28s} du {v['mean_delta_u']:+.4f} B/W {v['contexts_better']}/{v['contexts_worse']} "
              f"gap {'' if mg is None else f'{100 * mg:+.0f}%':>5s} rank {el.get('rank', '-')}/{el.get('of', '-')} "
              f"| " + " ".join(f"{v['per_ctx_u'][c]:7.3f}" for c in cids))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
