#!/usr/bin/env python3
"""Publish measured reports once each immutable campaign cycle completes."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def save(path, row):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(path)


def run_report(command, log, heartbeat, timeout=600, interval=10):
    """Keep telemetry live while supervising the report's owned process group."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    begun = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - begun)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                result = process.wait(timeout=min(interval, remaining))
            except subprocess.TimeoutExpired:
                heartbeat(time.monotonic() - begun)
                continue
            if result:
                raise subprocess.CalledProcessError(result, command)
            return
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    spec = json.loads(args.spec.read_text())
    assert spec["qualification"] is True
    campaign = (root / spec["output"]).resolve()
    contexts = (root / spec["contexts"]).resolve()
    assert campaign.is_relative_to(root) and contexts.is_relative_to(root)
    feed = root / "runs/report-progress.json"
    while True:
        cycles = sorted(p.parent for p in campaign.glob("cycle-*/complete.json"))
        completed = sum((p / "report-v1/receipt.json").exists() for p in cycles)
        for cycle in cycles:
            destination = cycle / "report-v1"
            if (destination / "receipt.json").exists():
                continue
            assert not destination.exists(), "Published report has no completion receipt"
            temporary = cycle / f"report-v1-staging-{time.time_ns()}"
            save(
                feed,
                dict(
                    schema="parallel.report_progress.v1",
                    timestamp=int(time.time()),
                    phase="Calculating matched-budget policy comparisons",
                    completed=completed,
                    total=len(cycles),
                    cycle=str(cycle.relative_to(root)),
                    complete=False,
                ),
            )
            cmd = [
                sys.executable,
                str(root / "tools/report_cycle.py"),
                "--root",
                str(root),
                "--cycle",
                str(cycle),
                "--contexts",
                str(contexts),
                "--output",
                str(temporary),
            ]
            try:

                def heartbeat(elapsed):
                    save(
                        feed,
                        dict(
                            schema="parallel.report_progress.v1",
                            timestamp=int(time.time()),
                            phase="Calculating matched-budget policy comparisons",
                            completed=completed,
                            total=len(cycles),
                            cycle=str(cycle.relative_to(root)),
                            complete=False,
                            elapsed_seconds=elapsed,
                        ),
                    )

                with (cycle / "report-build.log").open("a") as log:
                    run_report(cmd, log, heartbeat)
                receipt = json.loads((temporary / "receipt.json").read_text())
                assert receipt["passed"] is True
                temporary.rename(destination)
                completed += 1
            except Exception as error:
                save(
                    feed,
                    dict(
                        schema="parallel.report_progress.v1",
                        timestamp=int(time.time()),
                        phase="Report needs inspection",
                        completed=completed,
                        total=len(cycles),
                        reason=str(error),
                        cycle=str(cycle.relative_to(root)),
                        complete=False,
                    ),
                )
                raise
        stopped = (campaign / "complete.json").exists()
        save(
            feed,
            dict(
                schema="parallel.report_progress.v1",
                timestamp=int(time.time()),
                phase="Completed-cycle reports current"
                if cycles
                else "Waiting for first completed campaign cycle",
                completed=completed,
                total=len(cycles),
                complete=stopped and completed == len(cycles),
                campaign=str(campaign.relative_to(root)),
                qualification=True,
            ),
        )
        if stopped and completed == len(cycles):
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
