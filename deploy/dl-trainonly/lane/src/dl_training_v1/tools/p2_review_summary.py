"""Summarize RUN/families/review.csv for the owner's DL-D5 review (PLAN §5 P2, §7.2).

Prints, per row kind: the number of review groups, families, datasets and contexts, the agent-generated recommendation counts,
the defaults that apply if the owner leaves a row undecided, and the flags. Nothing here is a decision: the owner's answers
are recorded separately in review-decisions.json with human provenance. Sealed members never appear in clear in review.csv,
and this summary prints no member names at all.

Usage: p2_review_summary.py REVIEW_CSV [--json OUT]
"""
import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

MANDATORY = ("seed_source_mapping", "sealed_adjacent", "rights_note")


def summarize(path):
    raw = Path(path).read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    by_kind = collections.OrderedDict()
    for r in rows:
        k = r.get("kind", "?")
        s = by_kind.setdefault(k, {"groups": 0, "families": set(), "datasets": 0, "contexts": 0, "fold0": 0,
                                   "recommendation": collections.Counter(), "default_if_undecided": collections.Counter(),
                                   "flags": collections.Counter(), "options": set(), "max_contexts": 0})
        s["groups"] += 1
        s["families"].update(f for f in (r.get("family_ids") or "").split(";") if f)
        for col, key in (("n_datasets", "datasets"), ("n_contexts", "contexts"), ("n_fold0", "fold0")):
            try:
                s[key] += int(r.get(col) or 0)
            except ValueError:
                pass
        try:
            s["max_contexts"] = max(s["max_contexts"], int(r.get("n_contexts") or 0))
        except ValueError:
            pass
        s["recommendation"][r.get("recommendation", "")] += 1
        s["default_if_undecided"][r.get("default_if_undecided", "")] += 1
        for f in (r.get("flags") or "").replace(",", ";").split(";"):
            if f.strip():
                s["flags"][f.strip()] += 1
        s["options"].update(o for o in (r.get("decision_options") or "").replace(",", ";").split(";") if o.strip())
    out = {"review_csv": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "rows": len(rows),
           "mandatory_rows": sum(1 for r in rows if r.get("kind") in MANDATORY), "kinds": {}}
    for k, s in by_kind.items():
        out["kinds"][k] = {"groups": s["groups"], "families": len(s["families"]), "datasets_sum": s["datasets"],
                           "contexts_sum": s["contexts"], "fold0_sum": s["fold0"], "max_contexts": s["max_contexts"],
                           "recommendation": dict(s["recommendation"]), "default_if_undecided": dict(s["default_if_undecided"]),
                           "flags": dict(s["flags"]), "decision_options": sorted(s["options"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("review_csv")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    out = summarize(a.review_csv)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"review.csv {out['sha256'][:12]}… rows={out['rows']} mandatory={out['mandatory_rows']}")
    for k, s in out["kinds"].items():
        rec = ", ".join(f"{v}×{x or '∅'}" for x, v in sorted(s["recommendation"].items(), key=lambda kv: -kv[1]))
        print(f"- {k}: {s['groups']} groups / {s['families']} families / {s['contexts_sum']} ctx (max {s['max_contexts']}); rec: {rec}; "
              f"default: {s['default_if_undecided']}; flags: {s['flags']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
