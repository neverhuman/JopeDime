#!/usr/bin/env python3
"""Builds the `AMENDMENT_3` §1 & §3 `--budget` JSON for `s9_p201b_stack.py` from H459 `{cid}.meta.json` files.

Label-free and prediction-free: reads ONLY `{cid}.meta.json` and `SETTINGS.json` inside `--block-dir` (never `.npz`
predictions and never audit labels). Emits one budget row per exported column in `--block-dir`:
  - `a5_{tier}_{anc}_{shrunk,ratchet}` and `a5_{tier}_{anc}_stg1..M`: mapped from `meta["a5"]`
  - `a0_{tier}_{anc}_s0`: mapped from `meta["a0_s0"]`
  - `a0_{tier}_{anc}_s1`: mapped from `meta["a0_s1"]`
  - `a0_{tier}_{anc}_{2x|1p5x}`: mapped from `meta["a0_2x"]` / `meta["a0_1p5x"]`
  - `a6_{tier}_{anc}_comm`: `a0_s0` candidate count + `a0_s0.worker_calls_admitted + a0_s0.a6_comm_posthoc_calls`
  - `a6b_{tier}_{anc}_runbag`: exact per-context sum of `a0_s0` + `a0_s1` (`cl_fwriter` J5)
Also records `cids` (sorted list of covered CIDs, required by `s9_p201b_stack.py` `0e8e64fa` `AMENDMENT_4` §4),
`cid_list_sha256` (comma-joined sorted CIDs), `candidates` (`candidates_evaluated` = non-timeout screen returns, matching
`cl_h458`'s `evaluated_total` per `cl_fwriter` `M3`), and `candidates_admitted` (including timed-out candidates).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


def _stat(vals: Sequence[float]) -> Dict[str, float]:
    if not vals:
        raise ValueError("Cannot compute stats on empty value list")
    n = float(len(vals))
    return {
        "mean": sum(vals) / n,
        "max": max(vals),
        "min": min(vals),
    }


def build_budget_table(
    block_dir: Path,
    keep_cids: Optional[Sequence[str]] = None,
    default_slots: int = 3,
) -> Dict[str, Dict[str, object]]:
    settings_p = block_dir / "SETTINGS.json"
    if not settings_p.is_file():
        raise FileNotFoundError(f"Missing {settings_p}")
    settings = json.loads(settings_p.read_text())
    cols: List[str] = list(settings.get("exported_columns", []))
    if not cols:
        raise RuntimeError(f"No exported_columns in {settings_p}")

    meta_by_cid: Dict[str, Dict] = {}
    for mp in sorted(block_dir.glob("*.meta.json")):
        rec = json.loads(mp.read_text())
        cid = str(rec["cid"])
        meta_by_cid[cid] = rec

    if keep_cids is not None:
        keep_list = [str(c) for c in keep_cids]
        present = [c for c in keep_list if c in meta_by_cid]
        missing_n: Optional[int] = len(keep_list) - len(present)
        recs = [meta_by_cid[c] for c in present]
        cid_order = present
    else:
        cid_order = sorted(meta_by_cid)
        missing_n = 0
        recs = [meta_by_cid[c] for c in cid_order]

    if not recs:
        raise RuntimeError(f"No matching .meta.json records found in {block_dir}")

    sorted_cids = sorted(cid_order)
    cid_sha = hashlib.sha256(",".join(sorted_cids).encode("ascii")).hexdigest()
    effort = str(settings.get("effort", ""))
    anc = "hyb" if settings.get("anchor_dir") else "pure"
    prefix = f"{effort}_{anc}"
    dom_mult = float(settings.get("dom_mult", 2.0))
    dom_tag = "2x" if abs(dom_mult - 2.0) < 1e-6 else ("1p5x" if abs(dom_mult - 1.5) < 1e-6 else "dom")

    def _slot_s(arm_rec: Dict) -> float:
        if "slot_seconds" in arm_rec:
            return float(arm_rec["slot_seconds"])
        slots = int(arm_rec.get("inner_slots", default_slots))
        return float(arm_rec["wall_s"]) * max(1, slots)

    def _row_from_extractor(
        col_name: str,
        arm_label: str,
        fn: Callable[[Dict], Dict[str, float]],
    ) -> Dict[str, object]:
        extracted = [fn(r) for r in recs]
        return {
            "dir": str(block_dir),
            "contexts": len(recs),
            "contexts_missing": missing_n,
            "cids": list(sorted_cids),
            "cid_list_sha256": cid_sha,
            "candidates": _stat([x["cands_eval"] for x in extracted]),
            "candidates_evaluated": _stat([x["cands_eval"] for x in extracted]),
            "candidates_admitted": _stat([x["cands"] for x in extracted]),
            "worker_calls": _stat([x["calls"] for x in extracted]),
            "slot_seconds": _stat([x["slot_s"] for x in extracted]),
            "wall_s": _stat([x["wall_s"] for x in extracted]),
            "timeouts": _stat([x["timeouts"] for x in extracted]),
            "stages_executed": sorted({int(x["stages"]) for x in extracted}),
            "arm": [arm_label],
            "column": col_name,
        }

    out: Dict[str, Dict[str, object]] = {}
    for col in cols:
        if col.startswith(f"a5_{prefix}_"):
            out[col] = _row_from_extractor(
                col,
                "a5_oc_fgb_gp",
                lambda r: {
                    "cands": float(r["a5"]["candidates_admitted"]),
                    "cands_eval": float(r["a5"].get("candidates_evaluated", r["a5"]["candidates_admitted"])),
                    "calls": float(r["a5"]["worker_calls_admitted"]),
                    "slot_s": _slot_s(r["a5"]),
                    "wall_s": float(r["a5"]["wall_s"]),
                    "timeouts": float(r["a5"].get("timeouts", 0)),
                    "stages": float(r["a5"].get("stages_executed", 1)),
                },
            )
        elif col == f"a0_{prefix}_s0":
            out[col] = _row_from_extractor(
                col,
                "a0_s0",
                lambda r: {
                    "cands": float(r["a0_s0"]["candidates_admitted"]),
                    "cands_eval": float(r["a0_s0"].get("candidates_evaluated", r["a0_s0"]["candidates_admitted"])),
                    "calls": float(r["a0_s0"]["worker_calls_admitted"]),
                    "slot_s": _slot_s(r["a0_s0"]),
                    "wall_s": float(r["a0_s0"]["wall_s"]),
                    "timeouts": float(r["a0_s0"].get("timeouts", 0)),
                    "stages": 1.0,
                },
            )
        elif col == f"a0_{prefix}_s1":
            out[col] = _row_from_extractor(
                col,
                "a0_s1",
                lambda r: {
                    "cands": float(r["a0_s1"]["candidates_admitted"]),
                    "cands_eval": float(r["a0_s1"].get("candidates_evaluated", r["a0_s1"]["candidates_admitted"])),
                    "calls": float(r["a0_s1"]["worker_calls_admitted"]),
                    "slot_s": _slot_s(r["a0_s1"]),
                    "wall_s": float(r["a0_s1"]["wall_s"]),
                    "timeouts": float(r["a0_s1"].get("timeouts", 0)),
                    "stages": 1.0,
                },
            )
        elif col == f"a0_{prefix}_{dom_tag}":
            key_d = f"a0_{dom_tag}"
            out[col] = _row_from_extractor(
                col,
                key_d,
                lambda r, kd=key_d: {
                    "cands": float(r[kd]["candidates_admitted"]),
                    "cands_eval": float(r[kd].get("candidates_evaluated", r[kd]["candidates_admitted"])),
                    "calls": float(r[kd]["worker_calls_admitted"]),
                    "slot_s": _slot_s(r[kd]),
                    "wall_s": float(r[kd]["wall_s"]),
                    "timeouts": float(r[kd].get("timeouts", 0)),
                    "stages": 1.0,
                },
            )
        elif col == f"a6_{prefix}_comm":
            out[col] = _row_from_extractor(
                col,
                "a6_comm_archive_backfit",
                lambda r: {
                    "cands": float(r["a0_s0"]["candidates_admitted"]),
                    "cands_eval": float(r["a0_s0"].get("candidates_evaluated", r["a0_s0"]["candidates_admitted"])),
                    "calls": float(r["a0_s0"]["worker_calls_admitted"] + r["a0_s0"].get("a6_comm_posthoc_calls", 0)),
                    "slot_s": _slot_s(r["a0_s0"]),
                    "wall_s": float(r["a0_s0"]["wall_s"]),
                    "timeouts": float(r["a0_s0"].get("timeouts", 0)),
                    "stages": 1.0,
                },
            )
        elif col == f"a6b_{prefix}_runbag":
            out[col] = _row_from_extractor(
                col,
                "a6b_runbag_sum_s0_s1",
                lambda r: {
                    "cands": float(r["a0_s0"]["candidates_admitted"] + r["a0_s1"]["candidates_admitted"]),
                    "cands_eval": float(
                        r["a0_s0"].get("candidates_evaluated", r["a0_s0"]["candidates_admitted"])
                        + r["a0_s1"].get("candidates_evaluated", r["a0_s1"]["candidates_admitted"])
                    ),
                    "calls": float(r["a0_s0"]["worker_calls_admitted"] + r["a0_s1"]["worker_calls_admitted"]),
                    "slot_s": _slot_s(r["a0_s0"]) + _slot_s(r["a0_s1"]),
                    "wall_s": float(r["a0_s0"]["wall_s"] + r["a0_s1"]["wall_s"]),
                    "timeouts": float(r["a0_s0"].get("timeouts", 0) + r["a0_s1"].get("timeouts", 0)),
                    "stages": 1.0,
                },
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-dir", required=True, help="Completed H459 column-block directory")
    ap.add_argument("--contexts-from", default="", help="Optional JSON array or SIDECAR.json listing expected CIDs")
    ap.add_argument("--default-slots", type=int, default=3)
    ap.add_argument("--out", required=True, help="Output budget JSON path")
    a = ap.parse_args()

    keep: Optional[List[str]] = None
    if a.contexts_from:
        raw = json.loads(Path(a.contexts_from).read_text())
        keep = list(raw if isinstance(raw, list) else raw["contexts"])

    tbl = build_budget_table(Path(a.block_dir), keep_cids=keep, default_slots=int(a.default_slots))
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_suffix(".tmp.json")
    tmp_p.write_text(json.dumps(tbl, indent=2, sort_keys=True) + "\n")
    tmp_p.replace(out_p)
    print(
        json.dumps(
            {
                "budget_json": str(out_p),
                "n_columns": len(tbl),
                "contexts": next(iter(tbl.values()))["contexts"] if tbl else 0,
                "contexts_missing": next(iter(tbl.values()))["contexts_missing"] if tbl else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
