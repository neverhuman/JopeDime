#!/usr/bin/env python3
"""H426: the DEV tests of H420, H421, H422, H424 and H425 restated with clusters of related DEV contexts as the unit.
Stored values only: stage records and result files of those waves, and the frozen H425 screen output. No table, no
model, no GP engine; nothing on CONFIRM. A restatement: no verdict is changed.

  restate_h426.py --out FILE [--h424-second APPLY2.json]

Clusters (connected components; two DEV contexts are linked when the H425 screen matched them to the same FIT-ICL
table; the screen keeps one best match per screen and context):
  feature   links through the matches of the feature screens S1, S2, S3, S4 where they flagged      <- pre-declared
  copy      links through S1-copy and S2 matches only
  any       feature links plus the target-only baseline's matches
  shape     "any" plus: same row count and same feature count, for tables with >= 20 features (manifest only; crude and
            pessimistic: it also joins unrelated tables that happen to have the same shape)
Every comparison is the wave's own statistic, h335_stats.primary_test (clipped unit-mean gain, exact sign-flip over
units, leave-one-unit-out minimum, median), once with the original units (families) and once per clustering with
clusters as units. The original must reproduce the wave's stored numbers or the run stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EV = Path("/home/ubuntu/jope_plan/evidence")
IN = {"h420_dev": EV / "H420/pulled/H420_DEV/H420_DEV", "h417_base": EV / "H417/pulled/APEX_BASE/APEX_BASE",
      "h421_dev": EV / "H421/pulled/H421_DEV/H420_DEV", "h420_readout": EV / "H420/results/H420_DEV.readout.json",
      "h421_readout": EV / "H421/results/H421_DEV.readout.json", "h422_apply": EV / "H422/results/H422_DEV.apply.json",
      "h422_readout": EV / "H422/results/H422_DEV.readout.json", "h424_apply": EV / "H424/results/H424_DEV.apply.json",
      "h424_readout": EV / "H424/results/H424_DEV.readout.json", "h425_list": EV / "H425/results/EXCLUSION_DEV.json",
      "h425_screen": EV / "H425/results/SCREEN_DEV.json", "h425_sens": EV / "H425/results/H425_DEV.sensitivity.json"}
VARIANTS = ("feature", "copy", "any", "shape")
BOOT_SEED, N_BOOT, TOL = 426150, 10_000, 1e-12


def components(cids, links) -> dict:
    parent = {c: c for c in cids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in links:
        g = sorted(group)
        for c in g[1:]:
            ra, rb = find(g[0]), find(c)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {c: "cl:" + find(c)[:16] for c in cids}


def clusterings(screen: dict, cids) -> dict:
    per = screen["per_context"]

    def by_fit(which):
        g = {}
        for c in cids:
            for s, need in which:
                r = per[c].get(s)
                if r and r.get(need) and r.get("fit"):
                    g.setdefault(r["fit"], set()).add(c)
        return [v for v in g.values() if len(v) > 1]

    feat = [("s1", "flag"), ("s2", "flag"), ("s3", "flag"), ("s4", "flag")]
    shape = {}
    for c in cids:
        if per[c]["width"] >= 20:
            shape.setdefault((per[c]["rows"], per[c]["width"]), set()).add(c)
    any_links = by_fit(feat + [("b", "flag")])
    return {"feature": components(cids, by_fit(feat)),
            "copy": components(cids, by_fit([("s1", "flag_copy"), ("s2", "flag")])),
            "any": components(cids, any_links),
            "shape": components(cids, any_links + [v for v in shape.values() if len(v) > 1])}


def stat(T16, H, a: dict, b: dict, unit_of: dict) -> dict:
    s = T16.pair_stats(np, H, a, b, unit_of)
    return {"units": len(set(unit_of[c] for c in a)), "contexts": len(a), "gain": s["gain"], "clipped_mean": s["clipped_mean"],
            "signflip_p": s["signflip_p"], "mirror_signflip_p": s["mirror_signflip_p"], "lofo_min": s["lofo_min"],
            "median": s["family_median"], "wtl": s["wtl"], "pass": bool(s["pass"])}


def elo(T16, H, a, b, unit_of) -> dict:
    arms, point, boot, _ = T16.elo_boot(np, H, {"a": a, "b": b}, unit_of, "b", BOOT_SEED, N_BOOT)
    d = T16.delta_elo(np, arms, point, boot, "a", "b")
    return {"delta": d["delta"], "ci95": d["ci95"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h424-second")
    a = ap.parse_args()
    import readout_h420 as R20
    import train_h416 as T16

    _, H = T16._stats_mod()
    lst = json.loads(IN["h425_list"].read_text())
    screen = json.loads(IN["h425_screen"].read_text())
    if hashlib.sha256(IN["h425_screen"].read_bytes()).hexdigest() != lst["screen_sha256"]:
        raise SystemExit("the screen file is not the one the frozen list was cut from")
    cids = sorted(lst["contexts"])
    if sorted(screen["per_context"]) != cids or len(cids) != 115:
        raise SystemExit("the frozen list does not cover the 115 DEV contexts")
    tier = {c: lst["contexts"][c]["tier"] for c in cids}
    fam = {c: screen["per_context"][c]["family_id"] for c in cids}
    cl = clusterings(screen, cids)

    vals = {}
    for v in ("TBOTH", "TLIB", "TGX"):
        vals[f"H420.{v}"] = R20._load(IN["h420_dev"], v)[2]["SXB"]
    vals["H417.B"] = R20._load(IN["h417_base"], "BASE", only=set(cids))[2]["SXB"]
    for v in ("TBOTH", "TPORT", "TRAND"):
        vals[f"H421.{v}"] = R20._load(IN["h421_dev"], v)[2]["SXB"]
    p422 = json.loads(IN["h422_apply"].read_text())["per_context"]
    for x in ("B", "S", "A", "SA"):
        vals[f"H422.{x}"] = {c: p422[c][x] for c in p422}
    p424 = json.loads(IN["h424_apply"].read_text())["per_context"]
    for x in ("B", "I"):
        vals[f"H424.{x}"] = {c: p424[c][x] for c in p424}
    if a.h424_second:
        p2 = json.loads(Path(a.h424_second).read_text())["per_context"]
        vals["H424.I24"] = {c: p2[c]["I"] for c in p2}
    for k, v in vals.items():
        if sorted(v) != cids:
            raise SystemExit(f"{k}: not the 115 DEV contexts")
    base = vals["H420.TBOTH"]
    for k in ("H421.TBOTH", "H422.B", "H424.B"):
        if max(abs(vals[k][c] - base[c]) for c in cids) > 0:
            raise SystemExit(f"{k} is not the stored H420 TBOTH baseline")

    r420 = json.loads(IN["h420_readout"].read_text())["vs_B"]
    r421 = json.loads(IN["h421_readout"].read_text())["pairs"]
    r422 = json.loads(IN["h422_readout"].read_text())["pairs"]
    r424 = json.loads(IN["h424_readout"].read_text())["pairs"]
    r425 = json.loads(IN["h425_sens"].read_text())["reads"]["first_read_h020"]["table"]["without_feature"]
    clean = [c for c in cids if tier[c] not in ("copy", "possible")]
    comps = [  # wave, comparison, role, arm, control, contexts, stored original
        ("H420", "TBOTH vs B", "primary", "H420.TBOTH", "H417.B", cids, r420["TBOTH"]["SXB"]),
        ("H420", "TLIB vs B", "secondary", "H420.TLIB", "H417.B", cids, r420["TLIB"]["SXB"]),
        ("H420", "TGX vs B", "secondary", "H420.TGX", "H417.B", cids, r420["TGX"]["SXB"]),
        ("H421", "TPORT vs B*", "primary", "H421.TPORT", "H421.TBOTH", cids, r421["TPORT_vs_TBOTH"]["SXB"]),
        ("H421", "TPORT vs TRAND", "control", "H421.TPORT", "H421.TRAND", cids, r421["TPORT_vs_TRAND"]["SXB"]),
        ("H422", "S vs B", "primary", "H422.S", "H422.B", cids, r422["S_vs_B"]),
        ("H422", "A vs B", "reported", "H422.A", "H422.B", cids, r422["A_vs_B"]),
        ("H422", "SA vs B", "reported", "H422.SA", "H422.B", cids, r422["SA_vs_B"]),
        ("H424", "I vs B", "primary", "H424.I", "H424.B", cids, r424["I_vs_B"]),
        ("H425", "I vs B without feature flags", "sensitivity read", "H424.I", "H424.B", clean, r425)]
    if a.h424_second:
        comps.append(("H424", "I24 vs B (second read)", "reported", "H424.I24", "H424.B", cids, None))
    rows, worst = [], 0.0
    for wave, name, role, x, y, sub, ref in comps:
        av, bv = {c: vals[x][c] for c in sub}, {c: vals[y][c] for c in sub}
        o = stat(T16, H, av, bv, {c: fam[c] for c in sub})
        if ref is not None:
            d = max(abs(o["clipped_mean"] - ref["clipped_mean"]), abs(o["signflip_p"] - ref["signflip_p"]))
            worst = max(worst, d)
            if d > TOL or o["pass"] != bool(ref["pass"]):
                raise SystemExit(f"{wave} {name}: the stored original is not reproduced (difference {d})")
        row = {"wave": wave, "comparison": name, "role": role, "original": o, "clusters": {}}
        row["original"]["delta_elo"] = elo(T16, H, av, bv, {c: fam[c] for c in sub})
        for v in VARIANTS:
            u = {c: cl[v][c] for c in sub}
            row["clusters"][v] = stat(T16, H, av, bv, u)
            row["clusters"][v]["flip"] = bool(row["clusters"][v]["pass"] != o["pass"])
        row["clusters"]["feature"]["delta_elo"] = elo(T16, H, av, bv, {c: cl["feature"][c] for c in sub})
        rows.append(row)
    s422 = next(r for r in rows if r["wave"] == "H422" and r["role"] == "primary")
    rule422 = {k: bool(d["clipped_mean"] >= 0.010 and d["pass"] and d["delta_elo"]["ci95"][0] > 0)
               for k, d in (("original", s422["original"]), ("feature", s422["clusters"]["feature"]))}
    sizes = {v: sorted((list(cl[v].values()).count(k) for k in set(cl[v].values())), reverse=True) for v in VARIANTS}
    res = {"schema": "jope.h426.restatement/v1", "role": "restatement; no new verdicts", "contexts": len(cids),
           "units": {"families": len(set(fam.values())), **{v: len(set(cl[v].values())) for v in VARIANTS}},
           "cluster_sizes_above_1": {v: [s for s in sizes[v] if s > 1] for v in VARIANTS},
           "originals_reproduced_max_abs_diff": worst, "h422_three_part_rule": rule422,
           "flips_feature": [f"{r['wave']} {r['comparison']}" for r in rows if r["clusters"]["feature"]["flip"]],
           "flips_any_variant": sorted({f"{r['wave']} {r['comparison']} ({v})" for r in rows for v in VARIANTS if r["clusters"][v]["flip"]}),
           "inputs_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in IN.items() if p.is_file()},
           "rows": rows}
    print(f"== H426 units: {res['units']}   cluster sizes > 1: {res['cluster_sizes_above_1']}")
    for r in rows:
        o, f = r["original"], r["clusters"]["feature"]
        print(f"   {r['wave']} {r['comparison']:30s} [{r['role']:16s}] original n {o['units']:3d} clipped {o['clipped_mean']:+.4f} p {o['signflip_p']:.3g} "
              f"pass {o['pass']} | clusters n {f['units']:3d} clipped {f['clipped_mean']:+.4f} p {f['signflip_p']:.3g} pass {f['pass']} flip {f['flip']}"
              + " | " + " ".join(f"{v}: n {r['clusters'][v]['units']} p {r['clusters'][v]['signflip_p']:.3g} pass {r['clusters'][v]['pass']}" for v in VARIANTS[1:]))
    print(f"   flips (pre-declared clustering): {res['flips_feature']}; any variant: {res['flips_any_variant']}; H422 three-part rule: {rule422}")
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


if __name__ == "__main__":
    main()
