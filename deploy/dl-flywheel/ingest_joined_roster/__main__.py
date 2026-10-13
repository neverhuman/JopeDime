"""Build an owned joined roster from an exact pinned ingest plan."""

import argparse
import json
from pathlib import Path

from .roster import document, publish, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--plan-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    plan = document(
        read({"path": str(args.plan.absolute()), "sha256": args.plan_sha256, "bytes": args.plan_bytes})
    )
    result = publish(plan, args.output)
    args.receipt.write_text(
        json.dumps(
            {
                "plan_sha256": args.plan_sha256,
                "raw_source_refs": plan["sources"],
                "manifest_sha256": result["manifest_sha256"],
                "non_qualifying": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"manifest_sha256": result["manifest_sha256"], "summary": result["manifest"]["summary"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
