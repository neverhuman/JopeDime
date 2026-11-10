#!/usr/bin/env python3
"""H414: once-score the frozen frontier regressor kimik3p1_max on the fixed board dev91-audit-v1 (owner, 2026-11-07).

Entrant: the owner's file ~/JopeDime/kimik3p1_max.rs (47,156 B, sha256 0c211694…, received 2026-11-07T16:43:30Z;
crates/jope-automl/src/kimik3p1_max.rs), compiled unchanged into frontier_audit_h414 (the H412 frontier AUDIT harness
with a sixth entrant; the four H335 entrants reproduce their 804 stored TRAIN records and grok4p7_xhigh its 201 H412
TRAIN records bit-for-bit). Protocol = the five stored frontier rows: fit on the SEARCH rows (fold 0) of the board
envelope, predict the AUDIT rows (fold 1), seed 42, one fit, 300 s CPU / 900 s wall per context,
u = -ln(RMSE_audit / RMSE(mean(y_SEARCH))). Nothing is chosen on dev91; no stored entrant is re-run.

Guard: importing h335_core installs the H317 read guard v2 (the board root /home/ubuntu/jope-data/h335-stagev/ is
denied before the unlock). This module points the guard's Stage V unlock at its own LOCK (DATA/stage_v/LOCK.json) and
create-once receipt ns/receipts/H414_LOCK_sha256.txt; `unlock` is one-shot, later commands use resume_stage_v.
The dev91 two-fold run is not here: the guard refuses ~/jope-data/h257/ at every stage, so run_h414.sh does it in
bash under the same LOCK, receipt and unlock marker.

Commands: unlock | score | rows
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("H414_DATA", "/home/ubuntu/jope-data/h414/d91"))
# this wave's own cross-process refusal log (never the shared H335 data-root log); set before the guard is installed
os.environ["H317_GUARD_LOG"] = str(DATA / "guard_refusals.log")
sys.path.insert(0, str(HERE))
os.environ.setdefault("H335_GROUP", "BASE")
sys.path.insert(1, os.environ.get("H414_H335_CODE", "/home/ubuntu/h335-stage/code"))
import h335_core as K  # noqa: E402  (FIRST GP-stack import: H317 pins + read guard v2)

C = K.C
C.DATA_ROOT = DATA

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import frontier_audit_h414 as W  # noqa: E402  (stdlib-only runner; staged outside the evidence tree)

WAVE = "H414"
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H414_LOCK_sha256.txt")
BOARD = Path("/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1")
BOARD_SHA256 = "40cb4828cf5c9a0d2d9cc0f32b7900819c992ec9a58c5d18f83ac84b29fa4069"
N = 91
ENTRANT = "kimik3p1_max"
MODEL_ID = "h414.kimik3p1_max"
KIMI_SRC_SHA256 = "0c211694ec66605b994595fd8f4cf2b8894df6f156caa3b80f1fcc75f5d6ceb3"
KIMI_ORIGIN = "owner file ~/JopeDime/kimik3p1_max.rs (xbabe2, mtime 2026-11-07T16:43:30Z, 47156 B)"
RUN = DATA / "run" / "kimi_audit.jsonl"
ROWS = DATA / f"{MODEL_ID}.jsonl"


def lock() -> dict:
    return json.loads((DATA / "stage_v" / "LOCK.json").read_text())


def board() -> dict:
    b = json.loads((BOARD / "BOARD.json").read_text())
    if b["board"] != "dev91-audit-v1" or b["board_sha256"] != BOARD_SHA256 or len(b["contexts"]) != N:
        raise SystemExit("BOARD.json is not dev91-audit-v1")
    return b


def inputs_ok() -> None:
    """dev91-path inputs are verified after the unlock (the guard refuses them before)."""
    for p, sha in lock()["inputs_sha256"].items():
        if C.sha256_file(Path(p)) != sha:
            raise SystemExit(f"input changed after LOCK: {p}")


def cmd_unlock(_a) -> None:
    C.unlock_stage_v()
    print(json.dumps(C.guard_report()))


def cmd_score(a) -> None:
    C.resume_stage_v()
    inputs_ok()
    b = board()
    cids = [c["cid"] for c in b["contexts"]]
    for c in b["contexts"]:
        p = BOARD / "envs" / f"{c['cid']}.jtt1"
        C.assert_allowed(p)
        if C.sha256_file(p) != c["envelope_sha256"]:
            raise SystemExit(f"{c['cid'][:12]}: envelope sha differs from BOARD.json")
    lk = lock()
    binary = Path(lk["binary"])
    RUN.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summ = W.run(BOARD / "envs", cids, RUN, binary, cpu_limit_s=300, wall_limit_s=900, jobs=a.jobs,
                 binary_sha256=lk["binary_sha256"], entrants=(ENTRANT,))
    st = {"wave": WAVE, "entrant": ENTRANT, "summary": summ, "wall_s": time.time() - t0, "guard": C.guard_report(),
          "host": os.uname().nodename, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (DATA / "run" / f"status_{int(t0)}.json").write_text(json.dumps(st, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: st[k] for k in ("summary", "wall_s", "guard")}))


def cmd_rows(_a) -> None:
    """Board rows (create-once) from the run records; failures are final (board rule 2)."""
    C.resume_stage_v()
    inputs_ok()
    if ROWS.exists():
        raise SystemExit(f"{ROWS} exists (create-once)")
    b = board()
    env = {c["cid"]: c["envelope_sha256"] for c in b["contexts"]}
    recs = [json.loads(x) for x in RUN.read_text().splitlines() if x.strip()]
    by = {}
    for r in recs:
        if r["entrant"] != ENTRANT or r["cid"] in by:
            raise SystemExit(f"unexpected or duplicate record {r['entrant']} {r['cid'][:12]}")
        by[r["cid"]] = r
    if set(by) != set(env):
        raise SystemExit(f"{len(by)}/{N} contexts recorded")
    lk_sha = C.sha256_file(DATA / "stage_v" / "LOCK.json")
    run_sha = C.sha256_file(RUN)
    out = []
    for cid in [c["cid"] for c in b["contexts"]]:
        r = by[cid]
        if r["envelope_sha256"] != env[cid]:
            raise SystemExit(f"{cid[:12]}: run envelope sha != BOARD.json")
        out.append({"board": b["board"], "board_sha256": BOARD_SHA256, "cid": cid, "envelope_sha256": env[cid],
                    "model_id": MODEL_ID, "status": r["status"], "utility": float(r["utility"]),
                    "rmse_audit": r.get("rmse_audit"), "base_rmse_audit": r.get("base_rmse_audit"),
                    "wall_s": r.get("runner_wall_s"), "cpu_s": r.get("cpu_s"),
                    "audit_preds_sha256": r.get("audit_preds_sha256"), "harness_version": "h414-frontier-audit-v1",
                    "source": {"wave": WAVE, "entrant": ENTRANT, "entrant_source_sha256": KIMI_SRC_SHA256,
                               "entrant_origin": KIMI_ORIGIN, "seed": 42, "binary_sha256": r["binary_sha256"],
                               "wrapper_version": r["wrapper_version"], "lock_sha256": lk_sha,
                               "kimi_audit_jsonl_sha256": run_sha}})
    data = "".join(json.dumps(x, sort_keys=True) + "\n" for x in out).encode()
    fd = os.open(ROWS, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(fd, data)
    os.close(fd)
    st = {}
    for x in out:
        st[x["status"]] = st.get(x["status"], 0) + 1
    print(json.dumps({"rows": str(ROWS), "sha256": hashlib.sha256(data).hexdigest(), "status": st,
                      "guard": C.guard_report()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("unlock")
    p = sub.add_parser("score")
    p.add_argument("--jobs", type=int, default=16)
    sub.add_parser("rows")
    a = ap.parse_args()
    {"unlock": cmd_unlock, "score": cmd_score, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
