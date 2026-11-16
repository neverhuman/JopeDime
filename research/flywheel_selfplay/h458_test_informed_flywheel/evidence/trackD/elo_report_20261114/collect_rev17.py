#!/usr/bin/env python3
"""Revision 17 addendum: the 19 H450 fleet dev91 readouts of 2026-11-13 that the first bundle left out (reconciler finding R16-2).

Reads result files only. The reader is the one the report uses for r4 epoch 48 and is checked here against every r4 epoch record of the bundle.
usage: collect_rev17.py <out json>    (run on xbabe2)
"""
import glob, hashlib, json, os, sys
E = "/home/ubuntu/jope_plan/evidence/"
bundle = json.load(open("/home/ubuntu/jope-data/h458-quarantine/report/elo_report_bundle.json"))
KL = "kumo_tabular_large_e8.v1"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def readout(path: str, wave: str) -> list:
    j = json.load(open(path))
    out = []
    for k, v in j.items():
        if not (isinstance(v, dict) and "entrant" in v and "elo" in v):
            continue
        out.append({"file": path.replace(E, ""), "wave": wave, "arm": k, "entrant": v["entrant"], "elo": v["elo"]["elo"], "ci95": v["elo"]["ci95"],
                    "family_mean_u": v["family_mean_u"], "rank33": v["rank_if_placed_against_frozen_33"], "n_entrants": v["n_entrants_in_calc"],
                    "gap_elo": v["gap_elo"], "wtl": {r: p["wtl"] for r, p in v["pairs"].items()}, "gain": {r: p["gain"] for r, p in v["pairs"].items()},
                    "kumo_l_in_fit": next(t["elo"] for t in v["top_10_order"] if t["entrant"] == KL), "mtime": int(os.path.getmtime(path)), "flags": {}})
    return out


n_chk = 0
have = {a["file"] for a in bundle["arms"] if a["wave"] == "H450"}
for f in sorted(have):
    old = {a["arm"]: a for a in bundle["arms"] if a["file"] == f}
    new = readout(E + f, "H450")
    assert len(new) == len(old), f
    for a in new:
        for k in ("entrant", "elo", "ci95", "family_mean_u", "rank33", "n_entrants", "gap_elo", "wtl", "gain", "kumo_l_in_fit"):
            assert a[k] == old[a["arm"]][k], (f, a["arm"], k)
        n_chk += 1
arms, sources = [], {}
for f in sorted(glob.glob(E + "H450/diag_results/*/H450_DEV91.json")):
    rel = f.replace(E, "")
    if rel in have or "r4_g48_" in rel:
        continue  # in the bundle, or read by the report itself (r4 epoch 48)
    got = readout(f, "H450F")
    for a in got:
        a["label"] = "in-corpus"
    arms += got
    sources[rel] = {"sha256": sha(f), "arms": len(got), "mtime_utc": __import__("time").strftime("%Y-%m-%dT%H:%MZ", __import__("time").gmtime(os.path.getmtime(f)))}
json.dump({"schema": "jope.h458.elo_report_addendum/v1", "bundle_generated_utc": bundle["generated_utc"], "reader_checked_on_h450_arms": n_chk, "sources": sources, "arms": arms},
          open(sys.argv[1], "x"), indent=1)
print("reader reproduces", n_chk, "H450 bundle arms;", len(sources), "files,", len(arms), "new arm entries")
import collections
c = collections.defaultdict(list)
for a in arms: c[a["arm"]].append(a["elo"])
for k, v in sorted(c.items(), key=lambda kv: -max(kv[1])): print(f"  {len(v):2d}  {k:44s} mean {sum(v) / len(v):8.2f}  best {max(v):8.2f}")
