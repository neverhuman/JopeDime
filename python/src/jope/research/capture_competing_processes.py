#!/usr/bin/env python3
"""Capture transient processes matched by the Golden42 safety preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    deadline = time.monotonic() + args.seconds
    seen: dict[str, dict[str, object]] = {}
    while time.monotonic() < deadline:
        output = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        now = time.time_ns() // 1_000_000
        for line in output.splitlines():
            lower = line.lower()
            matched = (
                " cargo " in lower
                or " rustc " in lower
                or "jope-dime-regressor" in lower
                or "/scq " in lower
            )
            excluded = "scqd" in lower or "scq controller serve" in lower
            if matched and not excluded:
                entry = seen.setdefault(
                    line.strip(),
                    {"line": line.strip(), "first_seen_unix_ms": now, "samples": 0},
                )
                entry["last_seen_unix_ms"] = now
                entry["samples"] = int(entry["samples"]) + 1
        time.sleep(args.interval)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(list(seen.values()), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
