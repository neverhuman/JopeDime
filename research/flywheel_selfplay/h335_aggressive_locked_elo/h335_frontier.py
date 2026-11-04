#!/usr/bin/env python3
"""H335 LLM-regressor entrants (PREREG §2): write each context's AUDIT envelope (fold 0 = SEARCH rows, fold 1 = AUDIT
rows, nq = 0; `common_h317.search_and_audit_envelopes`, the same split every GP arm uses) and run the 4 frozen Rust
regressors through frontier_audit_h335 (fit on SEARCH only, predict AUDIT, u = -ln(RMSE_audit / RMSE(SEARCH mean));
failure / timeout -> u = 0). Resumable (append-only JSONL). Prints counts only.

  --phase T   P201b (TRAIN), inside sandbox_h335.sh
  --phase V   the 91 DL-Val contexts, only after the pinned Stage V unlock (resume_stage_v), Stage V sandbox
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h335_core as K  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

import frontier_audit_h335 as FA  # noqa: E402

C = K.C
FRONTIER_BINARY_SHA256 = "b359807c289a7fbd8304218d8c29b31d758d552e0b1f8a2e82a748686cc22d43"


def write_envelopes(cids, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid in cids:
        p = out_dir / f"{cid}.jtt1"
        env = C.read_jtt1(C.table_path(cid))
        _, audit, _ = C.search_and_audit_envelopes(cid, env)
        raw = audit.to_bytes()
        if p.exists():
            if p.read_bytes() != raw:
                raise RuntimeError(f"existing envelope differs: {p}")
            continue
        p.write_bytes(raw)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("T", "V"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--binary", default=str(Path(C.DATA_ROOT) / "bin" / "frontier_audit_h335"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cpu-limit", type=float, default=300.0)
    ap.add_argument("--wall-limit", type=float, default=900.0)
    args = ap.parse_args()
    if args.phase == "T":
        cids = K.panel_p201b(C.train_rows())
    else:
        C.resume_stage_v()
        cids = sorted(r["context_sha256"] for r in C.dev_rows_for_stage_v())
    out = Path(args.out_dir)
    n_new = write_envelopes(cids, out / "envs")
    summ = FA.run(out / "envs", cids, out / "frontier_results.jsonl", Path(args.binary), args.cpu_limit,
                  args.wall_limit, args.jobs, binary_sha256=FRONTIER_BINARY_SHA256)
    keep = {k: v for k, v in summ.items() if not isinstance(v, (list, dict)) or k in ("status_counts",)}
    print(json.dumps({"phase": args.phase, "contexts": len(cids), "new_envelopes": n_new, "summary": keep},
                     default=str))


if __name__ == "__main__":
    main()
