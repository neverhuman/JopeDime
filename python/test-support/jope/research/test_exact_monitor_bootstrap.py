"""Bootstrap and scheduler helpers for the persistent research monitor."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from test_exact_common import sha256_file, validate_catalog_manifest
from test_exact_monitor_config import (
    CAMPAIGN, EVALUATOR, PROPOSALS, RESERVE_FLOOR, ROOT, SCQ, SCRIPTS,
)

def run(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"/home/ubuntu/jope-prime/src:{SCRIPTS}"
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def scheduler_jobs() -> dict[str, dict]:
    result = run([str(SCQ), "--json", "ls"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return {
        row["id"]: row
        for row in json.loads(result.stdout).get("Jobs", {}).get("jobs", [])
        if row.get("id")
    }


def fetch_proposals() -> bool:
    if PROPOSALS.is_file():
        return True
    pointer = json.loads((ROOT / "ops" / "dl-job.json").read_text())
    proposal_job = pointer["job_id"]
    job = scheduler_jobs().get(proposal_job)
    if not job or job.get("state") != "completed":
        return False
    temporary = ROOT / "proposals" / f".fetch-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    result = run([str(SCQ), "fetch", proposal_job, "--output", str(temporary)])
    if result.returncode:
        shutil.rmtree(temporary)
        raise RuntimeError((result.stderr or result.stdout).strip())
    candidates = list(temporary.rglob("dl-proposals.jsonl"))
    summaries = list(temporary.rglob("dl-proposals.summary.json"))
    if len(candidates) != 1:
        shutil.rmtree(temporary)
        raise RuntimeError("proposal fetch did not contain exactly one JSONL")
    PROPOSALS.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidates[0], PROPOSALS)
    if len(summaries) == 1:
        os.replace(summaries[0], PROPOSALS.with_suffix(".summary.json"))
    shutil.rmtree(temporary)
    return True


def bootstrap() -> dict:
    if not fetch_proposals():
        return {"phase": "generating_dl_proposals"}
    try:
        validate_catalog_manifest(
            ROOT / "catalog" / "current.json", verify_contexts=False
        )
    except ValueError as error:
        return {
            "phase": "health_gate_stopped_new_admissions",
            "health_gate": "catalog-count-authority",
            "error": str(error),
        }
    evaluator_hash = sha256_file(EVALUATOR)
    canaries = (
        ROOT / "ops" / "node-canaries.json",
        ROOT / "ops" / "direct-node-canaries.json",
    )

    def canary_is_current(path: Path) -> bool:
        if not path.is_file():
            return False
        report = json.loads(path.read_text())
        hashes = {
            row.get("evaluator_sha256")
            for row in (report.get("nodes") or {}).values()
        }
        return bool(report.get("semantic_bit_parity") and hashes == {evaluator_hash})

    if not any(canary_is_current(path) for path in canaries):
        result = run(
            [sys.executable, str(SCRIPTS / "test_exact_canary.py"), "--timeout", "900"],
            timeout=1200,
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-4000:])
    if not any(canary_is_current(path) for path in canaries):
        raise RuntimeError("node semantic-parity canary did not pass")
    if not CAMPAIGN.is_file():
        result = run(
            [
                sys.executable,
                str(SCRIPTS / "test_exact_campaign.py"),
                "build",
                "--target-pairs",
                "500000",
                "--max-shard",
                "100",
            ],
            timeout=7200,
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-4000:])
    inventory = run(
        [sys.executable, str(SCRIPTS / "test_exact_backlog.py"), "inventory"],
        timeout=3600,
    )
    if inventory.returncode:
        raise RuntimeError((inventory.stderr or inventory.stdout).strip()[-4000:])
    reserve_before = int(json.loads(inventory.stdout)["ready"])
    generation = None
    if reserve_before < RESERVE_FLOOR:
        generated = run(
            [
                sys.executable,
                str(SCRIPTS / "test_exact_wave.py"),
                "--candidates-per-dataset",
                "192",
                "--max-shard",
                "100",
            ],
            timeout=7200,
        )
        if generated.returncode:
            raise RuntimeError((generated.stderr or generated.stdout).strip()[-4000:])
        generation = json.loads(generated.stdout.splitlines()[-1])
    result = run(
        [sys.executable, str(SCRIPTS / "test_exact_backlog.py"), "refill"],
        timeout=7200,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-4000:])
    refill = json.loads(result.stdout)
    if refill.get("needed") and not refill.get("submitted"):
        raise RuntimeError(
            "strict evaluator queue needs work but refill submitted zero jobs"
        )
    return {
        "phase": "backlog_live",
        "reserve_before": reserve_before,
        "generation": generation,
        "refill": refill,
    }


