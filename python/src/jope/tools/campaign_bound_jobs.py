#!/usr/bin/env python3
"""List unused campaign cuts. Never enqueues or touches the live GP queue."""

import argparse
import json
from pathlib import Path


def keys(index):
    return {(row["dataset"], int(row["cut"])): row for row in index.get("encodings", [])}


def gp_cuts(roots):
    found = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dataset in root.iterdir():
            if not dataset.is_dir():
                continue
            for cut in dataset.iterdir():
                name = cut.name
                if name.startswith("cut-") and name[4:].isdigit():
                    found.add((dataset.name, int(name[4:])))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--union", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gp-root", type=Path, action="append", default=[])
    args = p.parse_args()
    campaign = json.loads(args.campaign.read_text())
    union = json.loads(args.union.read_text())
    camp, uni = keys(campaign), keys(union)
    gp = gp_cuts(args.gp_root)
    unused = []
    for key, row in sorted(camp.items()):
        if key in gp:
            continue
        unused.append(
            {
                "dataset": key[0],
                "cut": key[1],
                "prepared": row.get("prepared"),
                "prepared_sha256": row.get("prepared_sha256"),
                "reason": "campaign encoding has no GP cut directory",
            }
        )
    report = {
        "schema": "parallel.campaign_bound_jobs.v1",
        "campaign_encodings": len(camp),
        "union_encodings": len(uni),
        "gp_cuts": len(gp),
        "unused_campaign_cuts": len(unused),
        "wasted_gp": len(gp - set(camp)),
        "jobs": unused,
        "enqueued": False,
        "final_test_targets_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "jobs"}))


if __name__ == "__main__":
    main()
