#!/usr/bin/env python3
"""H99 CPU-only preparation CLI; it intentionally has no timing/outcome mode."""
from __future__ import annotations

import argparse
import json

from runner import cpu_dry_run, execution_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-dry-run", action="store_true")
    modes.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    result = cpu_dry_run() if args.cpu_dry_run else execution_plan()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
