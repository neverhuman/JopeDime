#!/usr/bin/env python3
"""H421 step 1 - harvest (PREREG section 3.1). No worker run, no AUDIT value.

Design contexts: the 185 P201E DL-Train contexts (stored H416 matrices) and the 106 core FIT-SEL contexts (stored
H417 deploy). For every design context the programs that carry stack weight in SXB:

  FIT-SEL  the stored confirm record's spec (members, w);
  P201E    the locked design's weight rule applied offline to the stored matrix: meta_h416.meta_weights("BCAR", Q, y,
           cid). The same call on the FIT-SEL matrices must reproduce the stored FIT-SEL weights (checked here).

Program text comes from the stored evaluation memos (sha -> dsl). Near-duplicates are grouped: same text after every
numeric literal is rounded to 2 significant digits; a group is represented by its member that carries weight in the
most contexts. Groups are ranked by the number of design families in which a member carries weight. Candidate pool
C = the top 120 groups plus the library presets of B* (H420 TBOTH: stock library + LIB_EXT).

  harvest_h421.py --fitsel-run DIR --p201e-run DIR --memo-dir DIR [--memo-dir DIR ...] --out candidates.json
                  --hsets-out hsets.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOP_GROUPS = 120
SXB_KEY = "P3+LIB3.SXB"
_NUM = re.compile(r"(?<![A-Za-z_0-9.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def group_key(dsl: str) -> str:
    """The program text with every numeric literal rounded to 2 significant digits (identifiers such as c3, l2 or
    histgbm2 are not numeric literals)."""
    return _NUM.sub(lambda m: "%.2g" % float(m.group(0)), dsl)


def _last(path: Path) -> dict:
    return json.loads(path.read_text().rstrip("\n").rsplit("\n", 1)[-1])


def _literal(path: Path, name: str):
    """A module-level tuple literal read without importing the module (the GP stack stays unloaded)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{name} not found in {path}")


def _dsl_of(memo_dirs, cid: str, shas) -> dict:
    need, out = set(shas), {}
    for d in memo_dirs:
        p = Path(d) / f"{cid}.split.jsonl"
        if not p.is_file():
            continue
        with open(p) as f:
            for line in f:
                i = line.find('"sha": "')
                if i < 0:
                    continue
                s = line[i + 8:i + 72]
                if s in need and s not in out:
                    out[s] = json.loads(line)["dsl"]
                    if len(out) == len(need):
                        return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fitsel-run", required=True, help="pulled H417 run (stages/H416_BASE, h416_BASE_npz)")
    ap.add_argument("--p201e-run", required=True, help="pulled H416 design-panel run (config_P201E.json, npz, stages)")
    ap.add_argument("--memo-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hsets-out", required=True, help="json: cid -> shas of the stored stack candidates (H columns)")
    a = ap.parse_args()
    import apex_core_meta as AM
    import meta_h416 as M

    closed = set(AM.cohort("DEV", "all")) | set(AM.cohort("CONFIRM", "all"))   # every tier of the two closed cohorts
    fit = AM.panel("FIT-SEL")
    cfg = json.loads((Path(a.p201e_run) / "config_P201E.json").read_text())
    p201 = list(cfg["cids"])
    design = fit + p201
    if len(set(design)) != len(design) or set(design) & closed:
        raise SystemExit("a DEV or CONFIRM context, or a duplicate, is among the design contexts")
    fam, members, hsets, max_dw = {}, {}, {}, 0.0
    for cid in fit:
        rec = _last(Path(a.fitsel_run) / "stages" / "H416_BASE" / f"{cid}__confirm.jsonl")
        spec = rec["outputs"][SXB_KEY]["spec"]
        fam[cid] = rec["family_id"]
        members[cid] = dict(zip(spec["members"], (float(x) for x in spec["w"])))
        z = np.load(Path(a.fitsel_run) / "h416_BASE_npz" / f"{cid}.npz", allow_pickle=False)
        w = M.meta_weights("BCAR", z["Q"], z["y"], cid)   # self-test of the offline rule on stored FIT-SEL matrices
        hsets[cid] = [str(x) for x in z["shas"]]
        off = {str(s): float(x) for s, x in zip(z["shas"], w) if x > 0}
        if set(off) != set(members[cid]):
            raise SystemExit(f"{cid}: the offline weight rule does not reproduce the stored members")
        max_dw = max(max_dw, max(abs(off[s] - members[cid][s]) for s in off))
    for cid in p201:
        rec = _last(Path(a.p201e_run) / "stages" / "H416_P201E" / f"{cid}__dev.jsonl")
        fam[cid] = rec["family_id"]
        z = np.load(Path(a.p201e_run) / "h416_P201E_npz" / f"{cid}.npz", allow_pickle=False)
        w = M.meta_weights("BCAR", z["Q"], z["y"], cid)
        members[cid] = {str(s): float(x) for s, x in zip(z["shas"], w) if x > 0}
        hsets[cid] = [str(x) for x in z["shas"]]
    groups, missing = {}, 0
    for cid in design:
        dsl = _dsl_of(a.memo_dir, cid, members[cid])
        missing += len(members[cid]) - len(dsl)
        for s, w in members[cid].items():
            if s not in dsl:
                continue
            g = groups.setdefault(group_key(dsl[s]), {"fams": set(), "ctx": 0, "w": 0.0, "by_dsl": {}})
            g["fams"].add(fam[cid]); g["ctx"] += 1; g["w"] += w
            g["by_dsl"][dsl[s]] = g["by_dsl"].get(dsl[s], 0) + 1
    if missing:
        raise SystemExit(f"{missing} stack members have no program text in the memos")
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]["fams"]), -kv[1]["w"], kv[0]))
    lib = [("lib_stock", n, d) for n, d in _literal(HERE / "deploy_h410.py", "LIB_PRESETS")] \
        + [("lib_ext", n, d) for n, d in _literal(HERE / "lang_h420.py", "LIB_EXT")]
    pool, seen = [], set()
    for src, name, d in lib:
        k = group_key(d)
        g = groups.get(k)
        pool.append({"id": len(pool), "source": src, "name": name, "dsl": d, "key": k,
                     "n_families": len(g["fams"]) if g else 0, "n_contexts": g["ctx"] if g else 0,
                     "weight_sum": g["w"] if g else 0.0})
        seen.add(k)
    n_top_in_lib = 0
    for rank, (k, g) in enumerate(ranked[:TOP_GROUPS]):
        if k in seen:
            n_top_in_lib += 1
            continue
        rep = sorted(g["by_dsl"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        pool.append({"id": len(pool), "source": "harvest", "name": f"h{rank + 1:03d}", "dsl": rep, "key": k,
                     "rank": rank + 1, "n_families": len(g["fams"]), "n_contexts": g["ctx"], "weight_sum": g["w"]})
        seen.add(k)
    fams = sorted(set(fam.values()))
    out = {"schema": "jope.h421.candidates/v1", "design_contexts": len(design), "fit_sel": len(fit), "p201e": len(p201),
           "families": len(fams), "groups_total": len(ranked), "top_groups": TOP_GROUPS,
           "top_groups_already_in_library": n_top_in_lib, "library": len(lib),
           "offline_rule_max_abs_weight_diff_on_fit_sel": max_dw,
           "members_per_context_mean": float(np.mean([len(m) for m in members.values()])),
           "family_count_at_rank": {str(r): len(ranked[r - 1][1]["fams"]) for r in (1, 10, 30, 60, 120) if r <= len(ranked)},
           "no_dev_or_confirm_context": True, "cids": design, "family_of": fam, "pool": pool}
    raw = json.dumps(out, sort_keys=True).encode()
    with open(a.out, "xb") as f:
        f.write(raw)
    hraw = json.dumps(hsets, sort_keys=True).encode()
    with open(a.hsets_out, "xb") as f:
        f.write(hraw)
    print(json.dumps({k: out[k] for k in ("design_contexts", "families", "groups_total", "top_groups_already_in_library",
                                          "library", "offline_rule_max_abs_weight_diff_on_fit_sel",
                                          "members_per_context_mean", "family_count_at_rank")}
                     | {"pool": len(pool), "sha256": hashlib.sha256(raw).hexdigest(),
                        "hsets_sha256": hashlib.sha256(hraw).hexdigest()}))


if __name__ == "__main__":
    main()
