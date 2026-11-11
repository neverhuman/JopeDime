#!/usr/bin/env python3
"""H430: an inference bank for the TRAIN panel P201b, in the layout of bank_h424.py.

  bank_p201b.py --board-root DIR --run-dir H416_CONFIRM_DIR --out FILE

Source of the rows: the reference-board envelopes of the TRAIN development board p201b-train-audit-v1
(envs/<cid>.jtt1; fold byte 0 = SEARCH row, 1 = AUDIT row; target last). Source of the fold labels: the stored H416
record of the same context (h416_P201B_npz/<cid>.npz, field lab), so the columns are built on exactly the folds of
the stored candidate matrix. The script refuses any other board, checks every envelope sha256 against BOARD.json,
and requires the envelope's SEARCH targets and AUDIT targets to equal the stored record's y and ya element for
element. As in bank_h424.py, no AUDIT target is written: only its sha256, for the alignment check of the stack step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

BOARD = "p201b-train-audit-v1"


def read_jtt1(p: Path):
    raw = p.read_bytes()
    if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise SystemExit(f"bad JTT1 header: {p.name}")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq != 0 or len(raw) != 17 + nt + nt * (width + 1) * 8:
        raise SystemExit(f"not an AUDIT envelope: {p.name}")
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17)
    data = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(nt, width + 1)
    return hashlib.sha256(raw).hexdigest(), folds.copy(), data.copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    br = Path(a.board_root)
    board = json.loads((br / "BOARD.json").read_text())
    if board["board"] != BOARD or br.name != BOARD:
        raise SystemExit(f"refusing: {br} is not the TRAIN board {BOARD}")
    ctx = {c["cid"]: c for c in board["contexts"]}
    cids = json.loads((Path(a.run_dir) / "config_P201B.json").read_text())["cids"]
    if sorted(cids) != sorted(ctx):
        raise SystemExit("the stored P201B panel and the board do not hold the same contexts")
    S, AU, labs, ys, fams, ya_sha, y_sha = [], [], [], [], [], [], []
    for cid in cids:
        sha, folds, data = read_jtt1(br / "envs" / f"{cid}.jtt1")
        if sha != ctx[cid]["envelope_sha256"]:
            raise SystemExit(f"{cid}: envelope sha differs from BOARD.json")
        z = np.load(Path(a.run_dir) / "h416_P201B_npz" / f"{cid}.npz", allow_pickle=False)
        s, au = data[folds == 0], data[folds == 1]
        y, ya = s[:, -1].astype(np.float64), au[:, -1].astype(np.float64)
        if not (np.array_equal(y, z["y"].astype(np.float64)) and np.array_equal(ya, z["ya"].astype(np.float64))):
            raise SystemExit(f"{cid}: envelope rows are not in the stored record's order")
        if len(z["lab"]) != len(y):
            raise SystemExit(f"{cid}: fold labels do not match the SEARCH rows")
        m = s.copy()
        m[~np.isfinite(m)] = np.nan
        x = au[:, :-1].copy()
        x[~np.isfinite(x)] = np.nan
        S.append(m); AU.append(x); labs.append(z["lab"].astype(np.int8)); ys.append(y)
        fams.append(ctx[cid]["family_id"])
        ya_sha.append(hashlib.sha256(np.ascontiguousarray(ya).tobytes()).hexdigest())
        y_sha.append(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, search=np.concatenate([t.ravel() for t in S]), s_shape=np.array([t.shape for t in S], np.int64),
             audit=np.concatenate([t.ravel() for t in AU]), a_shape=np.array([t.shape for t in AU], np.int64),
             lab=np.concatenate(labs), y64=np.concatenate(ys), cids=np.array(cids), family=np.array(fams),
             cohort=np.array(["P201B"] * len(cids)), ya_sha256=np.array(ya_sha), y_sha256=np.array(y_sha))
    rec = {"file": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "panel": "P201B",
           "contexts": len(cids), "audit_targets_written": False, "board_sha256": board["board_sha256"],
           "search_rows": int(sum(t.shape[0] for t in S)), "audit_rows": int(sum(t.shape[0] for t in AU))}
    Path(str(out) + ".json").write_text(json.dumps(rec, sort_keys=True))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
