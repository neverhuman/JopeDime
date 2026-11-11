#!/usr/bin/env python3
"""H424 inference bank, built inside the H335 sandbox with the frozen readers (guard v2 installed by apex_core).

  bank_h424.py --panel DEV|FIT-SEL --out FILE

Per core context of the panel: the SEARCH rows (features + target, float32; float64 targets beside them), the
deploy_h415.kfold_labels fold of every SEARCH row, and the FEATURES of the AUDIT rows in the order of the stored AUDIT
predictions. No AUDIT target leaves the sandbox: only the sha256 of the AUDIT target vector is written, so the stack
step can prove that its stored targets are in the same row order.
Ragged file: search (float32, row-major, concatenated), s_shape [T,2] (rows, columns incl. target), audit (float32,
features only), a_shape [T,2] (rows, feature columns), lab (int8), y64, cids, family, cohort, ya_sha256, y_sha256.
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


def _clean(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64).copy()
    m[~np.isfinite(m)] = np.nan
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, choices=("DEV", "FIT-SEL"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    A.install()
    DP15 = A.DH.DP15
    rows = A.rows()
    S, AU, labs, ys, cids, fams, cohorts, ya_sha, y_sha = [], [], [], [], [], [], [], [], []
    for cid in A.panel(a.panel):
        env = C.read_jtt1(C.table_path(cid))
        search, _audit, m = C.search_and_audit_envelopes(cid, env)
        y = np.asarray(search.y, np.float64)
        ya = np.asarray(env.y, np.float64)[m]
        S.append(_clean(search.mat).astype(np.float32))
        AU.append(_clean(np.asarray(env.mat)[m][:, :-1]).astype(np.float32))
        labs.append(DP15.kfold_labels(cid, y).astype(np.int8))
        ys.append(y)
        cids.append(cid)
        fams.append(rows[cid]["family_id"])
        cohorts.append(rows[cid]["cohort"])
        ya_sha.append(hashlib.sha256(np.ascontiguousarray(ya).tobytes()).hexdigest())
        y_sha.append(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())
    if set(cohorts) != {a.panel}:
        raise SystemExit(f"a context outside {a.panel} reached the bank")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, search=np.concatenate([t.ravel() for t in S]), s_shape=np.array([t.shape for t in S], np.int64),
             audit=np.concatenate([t.ravel() for t in AU]), a_shape=np.array([t.shape for t in AU], np.int64),
             lab=np.concatenate(labs), y64=np.concatenate(ys), cids=np.array(cids), family=np.array(fams),
             cohort=np.array(cohorts), ya_sha256=np.array(ya_sha), y_sha256=np.array(y_sha),
             manifest_sha256=np.array(A.MANIFEST_SHA256))
    rec = {"file": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "panel": a.panel,
           "contexts": len(cids), "cohorts": sorted(set(cohorts)), "audit_targets_written": False,
           "search_rows": int(sum(t.shape[0] for t in S)), "audit_rows": int(sum(t.shape[0] for t in AU)),
           "manifest_sha256": A.MANIFEST_SHA256, "guard": C.guard_report()}
    Path(str(out) + ".json").write_text(json.dumps(rec, sort_keys=True))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
