#!/usr/bin/env python3
"""H99 preparation and read-only launch-preflight CLI; no cell starts here."""
from __future__ import annotations

import argparse
import json

from runner import cpu_dry_run, execution_plan
from coordinator import launch_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-dry-run", action="store_true")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--launch-preflight", action="store_true")
    args = parser.parse_args()
    if args.cpu_dry_run:
        result = cpu_dry_run()
    elif args.launch_preflight:
        result = {"schema": "jope.h99.launch-preflight/v1", "status": "pass", **launch_preflight()}
    else:
        result = execution_plan()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
