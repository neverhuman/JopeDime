"""Build a reference board's frozen envelopes and BOARD manifest from the H335 AUDIT split.

A board is a fixed list of contexts, each with ONE AUDIT envelope (JTT1, nq = 0): fold 0 = the SEARCH rows (fit
rows), fold 1 = the AUDIT rows (scored rows). The split is `common_h317.search_and_audit_envelopes`, i.e. exactly the
rows every H335 entrant was fitted on and scored on. The manifest pins each envelope by sha256 so a score can only
ever be joined to the bytes it was computed on.

Run inside the H335 sandboxes (the frozen code copy is bound at the H335 source path):
  p201b-train-audit-v1  ./sandbox_h335.sh env H335_GROUP=BASE PYTHONPATH=$SRC python3 make_board.py p201b-train-audit-v1 OUT
  dev91-audit-v1        ./sandbox_h335_stagev.sh v env H335_GROUP=BASE PYTHONPATH=$SRC python3 make_board.py dev91-audit-v1 OUT
                        (Stage V re-entry: resume_stage_v re-verifies the H335 LOCK, receipt and frozen code)
OUT/envs/<cid>.jtt1 is written (existing files must be byte-identical) and OUT/BOARD.json is printed and written.
Prints counts only, never data.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import h335_core as K  # noqa: F401  (installs the grammar group before any H317 import)
import h335_frontier as HF

C = K.C
BOARDS = {
    "p201b-train-audit-v1": "P201b: the second seeded TRAIN context of every TRAIN family with >= 2 contexts "
                            "(201 contexts / 201 families; h335_core.panel_p201b). TRAIN: H335 chose G* and the stack "
                            "on it, and our screeners were trained on TRAIN folds. Development board, not a test.",
    "dev91-audit-v1": "DL-Val91: the 91 dev contexts / 19 families of selected.json['dev'] (H335 Stage V). The fixed "
                      "TEST board. Reuse cohort (H257 tuned on it); read only for locked candidates and reference models.",
}


def main() -> None:
    board, out = sys.argv[1], Path(sys.argv[2])
    if board == "p201b-train-audit-v1":
        rows = C.train_rows()
        cids = K.panel_p201b(rows)
    elif board == "dev91-audit-v1":
        C.resume_stage_v()
        rows = C.dev_rows_for_stage_v()
        cids = sorted(r["context_sha256"] for r in rows)
    else:
        raise SystemExit(f"unknown board {board}; known: {sorted(BOARDS)}")
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    n_new = HF.write_envelopes(cids, out / "envs")
    ctx = []
    for cid in cids:
        raw = (out / "envs" / f"{cid}.jtt1").read_bytes()
        nt, nq, width = struct.unpack_from("<III", raw, 4)
        folds = raw[17:17 + nt]
        ctx.append({"cid": cid, "family_id": fam[cid], "envelope_sha256": hashlib.sha256(raw).hexdigest(),
                    "n_rows": nt, "n_search": folds.count(0), "n_audit": folds.count(1), "width": width})
    manifest = {"schema": "jope.reference-board/v1", "board": board, "description": BOARDS[board],
                "split": "common_h317.search_and_audit_envelopes (H317/H335 AUDIT split; fold 0 = SEARCH/fit rows, "
                         "fold 1 = AUDIT/scored rows)",
                "n_contexts": len(ctx), "n_families": len({c["family_id"] for c in ctx}),
                "contexts": ctx,
                "board_sha256": hashlib.sha256("".join(c["cid"] + c["envelope_sha256"] for c in ctx).encode()).hexdigest()}
    (out / "BOARD.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"board": board, "contexts": len(ctx), "families": manifest["n_families"], "new_envelopes": n_new,
                      "board_sha256": manifest["board_sha256"]}))


if __name__ == "__main__":
    main()
