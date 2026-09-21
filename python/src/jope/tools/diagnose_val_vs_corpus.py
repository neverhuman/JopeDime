#!/usr/bin/env python3
"""Read-only: is val flat while the corpus grows, or treading water on harder labels?

Does not change training, batch size, or production env vars.
"""

import argparse
import json
from pathlib import Path


def last_metric(path):
    last = None
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("rank") != 0 or row.get("batch_qualification") or row.get("invalidated"):
            continue
        last = row
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runs = args.root / "runs"
    rows = []
    for cycle in sorted(runs.rglob("cycle.json")):
        parent = cycle.parent
        index = parent / "corpus-2" / "index.json"
        metric = last_metric(parent / "train-2" / "metrics.jsonl") or last_metric(
            parent / "train-1" / "metrics.jsonl"
        )
        if not index.is_file() or metric is None:
            continue
        if "campaign64-qualification" not in str(parent):
            continue
        corpus = json.loads(index.read_text())
        val = metric.get("validation") or {}
        rows.append(
            {
                "cycle": str(parent.relative_to(args.root)),
                "records": corpus.get("records"),
                "groups": len(corpus.get("groups") or []),
                "train_nll": metric.get("sequence_nll"),
                "val_nll": val.get("sequence_nll"),
                "val_bt": val.get("ranking_bt"),
                "panel_sha256": val.get("panel_sha256"),
                "corpus_sha256": metric.get("corpus_sha256"),
            }
        )
    nlls = [r["val_nll"] for r in rows if isinstance(r.get("val_nll"), (int, float))]
    records = [r["records"] for r in rows if isinstance(r.get("records"), int)]
    if len(nlls) >= 3 and max(nlls) - min(nlls) < 0.08 and records and max(records) > min(records) * 1.05:
        reading = "val_flat_while_corpus_grows"
    elif (
        len(nlls) >= 3
        and records
        and abs(nlls[-1] - nlls[0]) < 0.08
        and abs(records[-1] - records[0]) < max(records) * 0.05
    ):
        reading = "val_and_records_both_flat"
    elif nlls and nlls[-1] < 1.44:
        reading = "val_below_prior_band"
    else:
        reading = "insufficient_or_mixed"
    report = {
        "schema": "parallel.val_vs_corpus.v1",
        "reading": reading,
        "band": {"val_nll": [1.44, 1.50], "val_bt": [0.58, 0.61]},
        "activation": {
            "JOPE_CONTINUE_OPTIMIZER": "gated; default off",
            "JOPE_OBJECTIVE_WEIGHTS": "gated; default nll1_bt025_list01_rt01_abl05_v1",
            "production_batch": "unchanged until Phase 5 boundary",
        },
        "cycles": rows,
        "final_test_targets_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reading": reading, "cycles": len(rows)}))


if __name__ == "__main__":
    main()
