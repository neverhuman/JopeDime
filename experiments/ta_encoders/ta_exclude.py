#!/usr/bin/env python3
"""Exclusion list for TA training: contexts whose dataset family holdout role is `heldout` (sealed meta-test) or
`sealed_excluded`, or whose role is unknown (not in the manifest context map), are never sampled. Usage: ta_exclude.py INDEX CTXMAP OUT"""
import json, sys, collections
idx = json.load(open(sys.argv[1]))["contexts"]; ctx = json.load(open(sys.argv[2]))
out = {}
for r in idx:
    p3 = ctx.get(r["cid"], {}).get("p3")
    if p3 in ("heldout", "sealed_excluded"): out[r["cid"]] = p3
    elif p3 is None: out[r["cid"]] = "unknown_role"
json.dump(out, open(sys.argv[3], "w")); print(len(out), collections.Counter(out.values()))
