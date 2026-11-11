#!/usr/bin/env python3
"""H425 addendum 1: the part of the CONFIRM screen that needs no table. Reads the cohort manifest only (stdlib; no
sandbox, no table file is opened, nothing is scored).

  confirm_meta_h425.py --out CONFIRM_META.json

Per CONFIRM context (all tiers): identical dataset id / table hash / family id with any FIT-ICL table of the manifest,
and the source-name screen S4 of screen_h425.py (same token rule; the constants are read from that file's source and
compared, so the two cannot drift). The feature screens S1-S3 are NOT run: CONFIRM tables are not in the sandbox view.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apex_core_meta as AM  # noqa: E402

STOP = {"srcfile", "legacy", "rdatasets", "regression", "classification", "runtime", "dataset", "datasets", "data", "tbl",
        "df", "train", "test", "csv", "v1", "v2", "v3", "the", "of", "and", "openml", "uci", "kaggle"}
S4_MIN_SHARED, S4_MIN_J = 2, 0.75


def name_tokens(key: str):
    s = key.lower()
    body = s.split(":", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{24,64}", body):
        return None
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in STOP and not (t.isdigit() and len(t) < 4)]
    return frozenset(toks) if sum(len(t) for t in toks) >= 5 else None


def _same_rule_as_screen() -> bool:
    src = (HERE / "screen_h425.py").read_text()
    tree = ast.parse(src)
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names == ["STOP"]:
                got["STOP"] = ast.literal_eval(node.value)
            for t in node.targets:
                if isinstance(t, ast.Tuple) and [e.id for e in t.elts] == ["S4_MIN_SHARED", "S4_MIN_J"]:
                    got["S4"] = ast.literal_eval(node.value)
    fn = lambda s: re.search(r"def name_tokens\(key: str\):\n(.*?)\n\n\n", s, re.S).group(1)
    return got.get("STOP") == STOP and got.get("S4") == (S4_MIN_SHARED, S4_MIN_J) and fn(src) == fn(Path(__file__).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not _same_rule_as_screen():
        raise SystemExit("the name rule here differs from screen_h425.py")
    rows = AM.rows()
    fit = [r for r in rows.values() if r["cohort"] == "FIT-ICL"]
    conf = {c: r for c, r in rows.items() if r["cohort"] == "CONFIRM"}
    f_ds, f_sha, f_fam = ({r[k] for r in fit} for k in ("dataset_id", "train_only_jtt1_sha256", "family_id"))
    f_names = [(r["family_key"], name_tokens(r["family_key"])) for r in fit]
    per = {}
    for c, r in sorted(conf.items()):
        t = name_tokens(r["family_key"])
        hits = sorted({k for k, ft in f_names if t is not None and ft is not None
                       and (t == ft or (len(t & ft) >= S4_MIN_SHARED and len(t & ft) / len(t | ft) >= S4_MIN_J))})
        per[c] = {"core": bool(AM.is_core(r)), "family_key": r["family_key"], "rows": r["rows"], "width": r["width"],
                  "same_dataset_id": r["dataset_id"] in f_ds, "same_table_sha256": r["train_only_jtt1_sha256"] in f_sha,
                  "same_family_id": r["family_id"] in f_fam, "name_applies": t is not None, "name_flag": bool(hits),
                  "matched_fit_icl_family_keys": hits}
    doc = {"schema": "jope.h425.confirm_meta/v1", "manifest_sha256": AM.MANIFEST_SHA256, "n": len(per),
           "core": sum(v["core"] for v in per.values()), "fit_icl_tables": len(fit),
           "feature_screens": "not run: CONFIRM tables are not in the sandbox view (addendum 1)",
           "counts": {k: sum(v[k] for v in per.values()) for k in ("same_dataset_id", "same_table_sha256", "same_family_id",
                                                                    "name_applies", "name_flag")},
           "name_flag_core": sum(v["name_flag"] and v["core"] for v in per.values()), "contexts": per}
    raw = json.dumps(doc, sort_keys=True)
    with open(a.out, "x") as f:
        f.write(raw)
    print(json.dumps({k: doc[k] for k in ("n", "core", "counts", "name_flag_core")}), hashlib.sha256(raw.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
