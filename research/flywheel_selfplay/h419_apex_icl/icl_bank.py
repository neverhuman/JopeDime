#!/usr/bin/env python3
"""H419 real-table banks, built inside the H335 sandbox with the frozen readers (guard v2 installed by apex_core).

  icl_bank.py fit --out FILE   FIT-ICL tables for training. Every table of cohort FIT-ICL in the manifest (any other
                               cohort is refused): features + target (last column), float32, non-finite -> NaN,
                               at most 4,096 rows and 100 feature columns (seeded samples).
  icl_bank.py val --out FILE   FIT-SEL core contexts for monitoring: SEARCH rows only (no AUDIT row leaves the
                               sandbox), with the deploy_h415.kfold_labels fold of every row.

Both files are ragged: data (float32, concatenated row-major), shape [T,2] = (rows, columns incl. target), offset [T],
cids, family, cohort, manifest_sha256. The val file adds lab (int8, concatenated) and y64 (float64 targets).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apex_core as A  # noqa: E402  (FIRST: frozen stack, pins and guard)

import numpy as np  # noqa: E402

C = A.C
MAX_ROWS, MAX_FEATS = 4096, 100
SEED = 419301


def _write(out: Path, tabs, cids, fams, cohorts, extra=None) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    shape = np.array([t.shape for t in tabs], np.int64)
    off = np.concatenate([[0], np.cumsum(shape[:, 0] * shape[:, 1])[:-1]]).astype(np.int64)
    data = np.concatenate([t.ravel() for t in tabs]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, data=data, shape=shape, offset=off, cids=np.array(cids), family=np.array(fams),
             cohort=np.array(cohorts), manifest_sha256=np.array(A.MANIFEST_SHA256), **(extra or {}))
    rec = {"file": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "tables": len(tabs),
           "cells": int(data.size), "cohorts": sorted(set(cohorts)), "manifest_sha256": A.MANIFEST_SHA256,
           "guard": C.guard_report()}
    Path(str(out) + ".json").write_text(json.dumps(rec, sort_keys=True))
    return rec


def _clean(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64).copy()
    m[~np.isfinite(m)] = np.nan
    return m


def cmd_fit(a) -> None:
    A.install()
    tabs, cids, fams, cohorts = [], [], [], []
    for cid, r in sorted(A.rows().items()):
        if r["cohort"] != "FIT-ICL":
            continue
        env = C.read_jtt1(C.table_path(cid))
        m = _clean(env.mat)
        if not np.isfinite(m[:, -1]).all():
            m = m[np.isfinite(m[:, -1])]
        rng = np.random.default_rng(C.stream_seed(SEED, "fit", cid))
        if m.shape[0] > MAX_ROWS:
            m = m[np.sort(rng.choice(m.shape[0], MAX_ROWS, replace=False))]
        if m.shape[1] - 1 > MAX_FEATS:
            keep = np.sort(rng.choice(m.shape[1] - 1, MAX_FEATS, replace=False))
            m = np.concatenate([m[:, keep], m[:, -1:]], 1)
        if m.shape[0] < 64 or m.shape[1] < 3:
            continue
        tabs.append(m.astype(np.float32)); cids.append(cid); fams.append(r["family_id"]); cohorts.append(r["cohort"])
    if set(cohorts) != {"FIT-ICL"}:
        raise SystemExit("a non-FIT-ICL table reached the training bank")
    print(json.dumps(_write(Path(a.out), tabs, cids, fams, cohorts)))


def cmd_val(a) -> None:
    A.install()
    DP15 = A.DH.DP15
    rows = A.rows()
    tabs, cids, fams, cohorts, labs, ys = [], [], [], [], [], []
    for cid in A.panel("FIT-SEL"):
        env = C.read_jtt1(C.table_path(cid))
        search, _audit, _m = C.search_and_audit_envelopes(cid, env)
        y = np.asarray(search.y, np.float64)
        lab = DP15.kfold_labels(cid, y)
        tabs.append(_clean(search.mat).astype(np.float32)); cids.append(cid)
        fams.append(rows[cid]["family_id"]); cohorts.append(rows[cid]["cohort"])
        labs.append(lab.astype(np.int8)); ys.append(y)
    if set(cohorts) != {"FIT-SEL"}:
        raise SystemExit("a non-FIT-SEL context reached the monitor bank")
    print(json.dumps(_write(Path(a.out), tabs, cids, fams, cohorts,
                            {"lab": np.concatenate(labs), "y64": np.concatenate(ys)})))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "val"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit": cmd_fit, "val": cmd_val}[a.cmd](a)


if __name__ == "__main__":
    main()
