#!/usr/bin/env python3
"""Revision 14 addendum to the Elo report's bundle: every dev91 Elo placement written after the bundle of 2026-11-14 23:14Z.

Reads result files only (H456 Stage 3 matrices; H462's seven dev91 price receipts; the look ledger for lock facts). Scores nothing,
reads no label. The matrix reader is checked against the bundle's own H455 records before it is trusted for H456.
usage: collect_rev14.py <out json>    (run on xbabe2)
"""
import glob, hashlib, json, os, sys
E = "/home/ubuntu/jope_plan/evidence/"
bundle = json.load(open("/home/ubuntu/jope-data/h458-quarantine/report/elo_report_bundle.json"))
KL = "kumo_tabular_large_e8.v1"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def matrix_arms(path: str, wave: str) -> list:
    j = json.load(open(path))
    out = []
    for arm, v in j["arms_full"].items():
        out.append({"file": path.replace(E, ""), "wave": wave, "arm": arm, "entrant": v["entrant"], "elo": v["elo"]["elo"], "ci95": v["elo"]["ci95"],
                    "family_mean_u": v["family_mean_u"], "rank33": v["rank_if_placed_against_frozen_33"], "n_entrants": v["n_entrants_in_calc"],
                    "gap_elo": v["gap_elo"], "wtl": {r: p["wtl"] for r, p in v["pairs"].items()}, "gain": {r: p["gain"] for r, p in v["pairs"].items()},
                    "kumo_l_in_fit": next(t["elo"] for t in v["top_10_order"] if t["entrant"] == KL), "mtime": int(os.path.getmtime(path))})
    return out


# the reader must reproduce the bundle on every H455 matrix
n_chk = 0
for f in sorted({a["file"] for a in bundle["arms"] if a["wave"] == "H455"}):
    old = {a["arm"]: a for a in bundle["arms"] if a["file"] == f}
    new = matrix_arms(E + f, "H455")
    assert len(new) == len(old), f
    for a in new:
        for k in ("entrant", "elo", "ci95", "family_mean_u", "rank33", "n_entrants", "gap_elo", "wtl", "gain", "kumo_l_in_fit"):
            assert a[k] == old[a["arm"]][k], (f, a["arm"], k)
        n_chk += 1
arms, sources = [], {}
for f in sorted(glob.glob(E + "H456/H456_*_MATRIX.json")):
    got = matrix_arms(f, "H456")
    for a in got:
        a.update({"label": "in-corpus", "locked": None, "pass_index": None, "group": None})
    arms += got
    sources[f.replace(E, "")] = {"sha256": sha(f), "arms": len(got)}

led = {}
for line in open(E + "H458/LOOK_LEDGER.jsonl"):
    r = json.loads(line)
    if str(r.get("arm", "")).startswith("h462."):
        led[os.path.basename(r["receipt"])] = r
IN_CORPUS = "in-corpus"
ZERO_G16 = "in-corpus (1.8% dev91 in the ICL pre-training)"
FAIR = "0% dev91"
WITHDRAWN = "withdrawn: not 0% dev91 (GP anchors carried dev91)"


def label_of(group: str, closure: str) -> str:
    if group.startswith("fair_cv7_hyb"):
        return WITHDRAWN
    if "zero_g16" in group:
        return ZERO_G16
    if "in_corpus" in group or closure.startswith("in-corpus"):
        return IN_CORPUS
    assert closure.startswith("fair") or closure.startswith("0.000000% dev91"), (group, closure)
    return FAIR


def place(fname, group, name, p, label, locked):
    r = led[fname]
    arms.append({"file": "H462/" + fname, "wave": "H462", "arm": (group + "." if group else "") + name, "entrant": "h462." + fname.replace(".json", "") + "." + (group + "." if group else "") + name,
                 "elo": p["elo"], "ci95": p["ci95"], "family_mean_u": p["family_mean_u"], "rank33": p.get("rank"), "n_entrants": (p["n_entrants"] + 1) if p.get("n_entrants") else None,
                 "gap_elo": {"vs_" + KL: p.get("vs_kumo_large_elo"), "vs_tabfm_ensemble.v1": p.get("vs_tabfm_ens_elo")}, "wtl": None, "gain": None,
                 "kumo_l_in_fit": (p["elo"] - p["vs_kumo_large_elo"]) if p.get("vs_kumo_large_elo") is not None else None, "mtime": int(os.path.getmtime(E + "H462/" + fname)), "label": label, "locked": bool(locked),
                 "pass_index": r["pass_index"], "group": group or None})


for fname in sorted(led, key=lambda k: led[k]["pass_index"]):
    j = json.load(open(E + "H462/" + fname))
    r = led[fname]
    assert sha(E + "H462/" + fname) == r["receipt_sha256"], fname
    n0 = len(arms)
    if "single_placements" in j:  # CVREP8 V1/V2, TABCV7
        lab = label_of("", j["provenance_class"])
        assert r["locked_entrant"] == j["fitsel_locked_rule"]
        for name, p in j["single_placements"].items():
            place(fname, "", name, p, lab, name == j["fitsel_locked_rule"])
    else:
        for group, v in j["dev91_quarantined_post_h448"].items():
            lab = label_of(group, v["closure"])
            if "single_placements" in v:  # FAIR_R4E8, MULTISEED
                le = r["locked_entrant"]
                locked_name = le.get(group) if isinstance(le, dict) else None
                for name, p in v["single_placements"].items():
                    place(fname, group, name, p, lab, locked_name is not None and name == locked_name)
            else:  # seed bagging V1/V2: no lock
                for key, name in (("single_placement_mean_u_conv", "one_run"), ("single_placement_pred_avg_2", "pred_avg_2"), ("single_placement_precision_avg_2", "precision_avg_2")):
                    if key in v:
                        place(fname, group, name, v[key], lab, False)
    sources["H462/" + fname] = {"sha256": r["receipt_sha256"], "arms": len(arms) - n0, "pass_index": r["pass_index"], "ledger_label": r["label"], "locked_entrant": r["locked_entrant"],
                                "n_entrants_scored_ledger": r["n_entrants_scored"]}
json.dump({"schema": "jope.h458.elo_report_addendum/v1", "bundle_generated_utc": bundle["generated_utc"], "matrix_reader_checked_on_h455_arms": n_chk,
           "ledger_sha256": sha(E + "H458/LOOK_LEDGER.jsonl"), "sources": sources, "arms": arms}, open(sys.argv[1], "x"), indent=1)
print("matrix reader reproduces", n_chk, "H455 bundle arms;", len(arms), "new arm entries")
for s, v in sources.items():
    print(f"  {v['arms']:3d}  {s}  " + (f"pass {v['pass_index']} ledger says {v['n_entrants_scored_ledger']} entrants, locked {v['locked_entrant']}" if "pass_index" in v else ""))
