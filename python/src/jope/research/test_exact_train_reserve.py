#!/usr/bin/env python3
"""Temporarily hold this campaign's queued CPU shards so xbabe2 can admit training."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from test_exact_common import atomic_json

SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
RECORD = ROOT / "ops" / "trainer-reservation-holds.json"


def command(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCQ), *args], capture_output=True, text=True, check=False, timeout=120
    )


def jobs() -> dict[str, dict]:
    result = command("--json", "ls")
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return {
        row["id"]: row
        for row in json.loads(result.stdout).get("Jobs", {}).get("jobs", [])
        if row.get("id")
    }


def mutate(action: str, job_ids: list[str]) -> tuple[list[str], list[dict]]:
    failures = []
    succeeded = []
    with ThreadPoolExecutor(max_workers=128) as pool:
        outcomes = zip(job_ids, pool.map(lambda job_id: command(action, job_id), job_ids))
        for job_id, result in outcomes:
            if result.returncode == 0:
                succeeded.append(job_id)
            else:
                failures.append(
                    {
                        "job_id": job_id,
                        "error": (result.stderr or result.stdout).strip()[:500],
                    }
                )
    return succeeded, failures


def hold() -> dict:
    current = jobs()
    already_held = sorted(
        job_id
        for job_id, row in current.items()
        if job_id.startswith("jte-")
        and not job_id.startswith("jte-canary-")
        and row.get("project") == "jope-testonly"
        and row.get("state") == "held"
    )
    job_ids = sorted(
        job_id
        for job_id, row in current.items()
        if job_id.startswith("jte-")
        and not job_id.startswith("jte-canary-")
        and row.get("project") == "jope-testonly"
        and row.get("state") == "queued"
    )
    succeeded, failures = mutate("hold", job_ids)
    result = {
        "schema": "JopeTestOnlyTrainerReservationHolds.v1",
        "held_job_ids": [*already_held, *succeeded],
        "requested": len(job_ids),
        "held": len(already_held) + len(succeeded),
        "failures": failures,
        "released": False,
    }
    atomic_json(RECORD, result)
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(job_ids)} holds failed")
    return result


def release() -> dict:
    record = json.loads(RECORD.read_text())
    current = jobs()
    job_ids = [
        job_id
        for job_id in record["held_job_ids"]
        if current.get(job_id, {}).get("state") == "held"
    ]
    succeeded, failures = mutate("release", job_ids)
    record.update(
        {
            "released": not failures,
            "released_jobs": len(succeeded),
            "release_failures": failures,
        }
    )
    atomic_json(RECORD, record)
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(job_ids)} releases failed")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("hold", "release"))
    args = parser.parse_args()
    result = hold() if args.action == "hold" else release()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
