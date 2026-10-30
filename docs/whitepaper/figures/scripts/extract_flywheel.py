#!/usr/bin/env python3
"""Read-only extractor: flywheel summary/admission logs -> small CSVs.
Usage: nice python3 extract_flywheel.py RUN_ROOT OUTDIR
RUN_ROOT is the flywheel data root (contains run-*/ and custody/). Only reads files."""
import json, sys, os, glob, csv
root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = {}
paths = sorted(glob.glob(f"{root}/custody/*/*/summary.jsonl")) + sorted(glob.glob(f"{root}/run-*/summary.jsonl"))
for p in paths:
    for line in open(p):
        try: d = json.loads(line)
        except Exception: continue
        r = d.get("round"); v = d.get("val_reg"); g = d.get("train_vs_val_gap_reg")
        if r is None or not isinstance(v, (int, float)): continue
        num = lambda k: d.get(k) if isinstance(d.get(k), (int, float)) else ""
        rows[r] = dict(round=r, val=v, train=(v + g) if isinstance(g, (int, float)) else "",
                       datasets_per_s=num("datasets_per_second"), wall_s=num("wall_seconds"),
                       promotions_per_hr=num("promotions_per_hour"), timestamp=num("timestamp"))
with open(f"{out}/rounds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["round","val","train","datasets_per_s","wall_s","promotions_per_hr","timestamp"])
    w.writeheader(); [w.writerow(rows[k]) for k in sorted(rows)]
adm = {}
for p in glob.glob(f"{root}/run-*/adapter_admissions.jsonl"):
    for line in open(p):
        try: d = json.loads(line)
        except Exception: continue
        if "round" in d: adm[d["round"]] = (d.get("candidate_utility"), d.get("best_utility"), int(bool(d.get("accepted"))), d.get("timestamp"))
with open(f"{out}/admissions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["round","candidate_val","best_val","accepted","timestamp"])
    for k in sorted(adm): w.writerow([k, *adm[k]])
print(len(rows), "summary rounds;", len(adm), "admission rounds")
