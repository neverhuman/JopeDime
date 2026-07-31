#!/usr/bin/env python3
"""Run the same precompiled-evaluator canary once on every SCQ node."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import time
from pathlib import Path

from test_exact_backlog import RUNNER, SCQ, ROOT, input_lines, quoted, stage_evaluator
from test_exact_common import atomic_json, sha256_file, sha256_text

NODES = ("xbabe1", "xbabe2", "xbabe3", "xbabe4")
REPORT = ROOT / "ops" / "node-canaries.json"
DIRECT_REPORT = ROOT / "ops" / "direct-node-canaries.json"
GENOME = "(pipe X (standardize) (ridge))"


def run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )


def manifest() -> Path:
    pointer = json.loads((ROOT / "catalog" / "current.json").read_text())
    catalog = json.loads(Path(pointer["catalog"]).read_text())
    entry = next(row for row in catalog["entries"] if row["task"] == "regression")
    value = {
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "task": "regression",
        "dataset_hash": entry["dataset_hash"],
        "train_csv": entry["train_csv"],
        "test_csv": entry["test_csv"],
        "train_sha256": entry["train_sha256"],
        "test_sha256": entry["test_sha256"],
        "candidates": [
            {
                "semantic_sha256": sha256_text(GENOME),
                "canonical_genome": GENOME,
            }
        ],
    }
    path = ROOT / "canary" / "manifest.json"
    atomic_json(path, value)
    return path


def render(node: str, manifest_path: Path, evaluator: Path) -> tuple[str, Path]:
    value = json.loads(manifest_path.read_text())
    job_id = f"jte-canary-v3-{node}-{sha256_file(evaluator)[:8]}"
    path = ROOT / "canary" / f"{job_id}.toml"
    lines = [
        "[job]",
        f"id = {quoted(job_id)}",
        'project = "jope-testonly"',
        'tier = "normal"',
        'command = "/usr/bin/env"',
        (
            'args = ["{input:runner}", "{input:manifest}", "{input:train}", '
            '"{input:test}", "{input:evaluator}", "{output:receipt}"]'
        ),
        "",
        "[resource]",
        "gpus_per_task = 0",
        "cpus_reserved = 1",
        "omp_threads = 1",
        "ram_mb = 512",
        "gang_size = 1",
        'placement = "single_node"',
        "",
        "[dependencies]",
        "smoke = [{ command = \"/usr/bin/grep\", "
        f"args = [\"-qx\", \"{node}\", \"/etc/hostname\"], "
        "timeout_secs = 5 }]",
        "",
        "[env]",
        'RAYON_NUM_THREADS = "1"',
        'OMP_NUM_THREADS = "1"',
        "",
        "[preempt]",
        "preemptible = true",
        'checkpoint = "none"',
        'resume = "rerun"',
        "",
        "[limits]",
        "max_runtime_hours = 1",
        "max_restarts = 1",
        "",
    ]
    for item in (
        ("runner", RUNNER, "runner.sh", None),
        ("manifest", manifest_path, "manifest.json", sha256_file(manifest_path)),
        ("train", Path(value["train_csv"]), "train.csv", value["train_sha256"]),
        ("test", Path(value["test_csv"]), "test.csv", value["test_sha256"]),
        ("evaluator", evaluator, "corpus_eval", sha256_file(evaluator)),
    ):
        lines.extend(input_lines(*item))
    lines.extend(
        [
            "[[output]]",
            'name = "receipt"',
            'source = "receipt"',
            "directory = true",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return job_id, path


def scheduler_jobs() -> dict[str, dict]:
    result = run([str(SCQ), "--json", "ls"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return {
        row["id"]: row
        for row in json.loads(result.stdout).get("Jobs", {}).get("jobs", [])
        if row.get("id")
    }


def record_direct_receipts(receipt_dir: Path, evaluator: Path) -> dict:
    results = {}
    for node in NODES:
        receipt = receipt_dir / f"{node}.receipts.jsonl"
        rows = [json.loads(line) for line in receipt.read_text().splitlines() if line]
        if len(rows) != 1:
            raise RuntimeError(f"{node}: expected one direct canary receipt")
        row = rows[0]
        results[node] = {
            "mode": "direct_node_probe",
            "score": row["score"],
            "score_bits": (
                struct.pack("!d", row["score"]).hex()
                if row["score"] is not None
                else None
            ),
            "secondary_score": row["secondary_score"],
            "secondary_score_bits": (
                struct.pack("!d", row["secondary_score"]).hex()
                if row["secondary_score"] is not None
                else None
            ),
            "valid": row["valid"],
            "failure": row["failure"],
            "manifest_sha256": row["manifest_sha256"],
            "receipt_sha256": sha256_file(receipt),
            "evaluator_sha256": sha256_file(evaluator),
        }
    identities = {
        (row["score_bits"], row["secondary_score_bits"], row["valid"], row["failure"])
        for row in results.values()
    }
    report = {
        "schema": "JopeTestOnlyDirectNodeCanary.v1",
        "semantic_bit_parity": len(identities) == 1,
        "nodes": results,
    }
    atomic_json(DIRECT_REPORT, report)
    if not report["semantic_bit_parity"]:
        raise RuntimeError("direct corpus_eval node probes lost score parity")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--direct-receipts", type=Path)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    evaluator = args.evaluator or stage_evaluator()
    if not evaluator.is_file():
        raise RuntimeError("candidate evaluator is absent")
    evaluator_hash = sha256_file(evaluator)
    if args.direct_receipts:
        print(
            json.dumps(record_direct_receipts(args.direct_receipts, evaluator), sort_keys=True)
        )
        return
    if args.report.is_file():
        prior = json.loads(args.report.read_text())
        hashes = {
            row.get("evaluator_sha256")
            for row in (prior.get("nodes") or {}).values()
        }
        if prior.get("semantic_bit_parity") and hashes == {evaluator_hash}:
            print(args.report.read_text(), end="")
            return
    manifest_path = manifest()
    submitted = {}
    existing_jobs = scheduler_jobs()
    for node in NODES:
        job_id, spec = render(node, manifest_path, evaluator)
        if job_id in existing_jobs:
            submitted[node] = job_id
            continue
        check = run([str(SCQ), "check", str(spec)])
        if check.returncode:
            raise RuntimeError(f"{node} scq check: {(check.stderr or check.stdout).strip()}")
        submit = run([str(SCQ), "submit", str(spec)])
        if submit.returncode and "already exists" not in (submit.stderr or submit.stdout).lower():
            raise RuntimeError(f"{node} submit: {(submit.stderr or submit.stdout).strip()}")
        submitted[node] = job_id
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        jobs = scheduler_jobs()
        states = {node: jobs.get(job_id, {}).get("state") for node, job_id in submitted.items()}
        if all(state == "completed" for state in states.values()):
            break
        if any(state in {"failed", "cancelled"} for state in states.values()):
            raise RuntimeError(f"node canary failed: {states}")
        time.sleep(2)
    else:
        raise RuntimeError("node canaries did not finish before timeout")
    results = {}
    for node, job_id in submitted.items():
        job = scheduler_jobs()[job_id]
        if job.get("node") != node:
            raise RuntimeError(f"{job_id} ran on {job.get('node')}, expected {node}")
        destination = ROOT / "canary" / "outputs" / job_id
        destination.mkdir(parents=True, exist_ok=True)
        fetched = run([str(SCQ), "fetch", job_id, "--output", str(destination)])
        if fetched.returncode:
            raise RuntimeError((fetched.stderr or fetched.stdout).strip())
        receipt_files = list(destination.rglob("*.receipts.jsonl"))
        if len(receipt_files) != 1:
            raise RuntimeError(f"{job_id} receipt resolution failed")
        row = json.loads(receipt_files[0].read_text().strip())
        results[node] = {
            "job_id": job_id,
            "execution_node": job["node"],
            "score": row["score"],
            "score_bits": struct.pack("!d", row["score"]).hex() if row["score"] is not None else None,
            "secondary_score": row["secondary_score"],
            "secondary_score_bits": (
                struct.pack("!d", row["secondary_score"]).hex()
                if row["secondary_score"] is not None
                else None
            ),
            "valid": row["valid"],
            "failure": row["failure"],
            "manifest_sha256": row["manifest_sha256"],
            "evaluator_sha256": sha256_file(evaluator),
        }
    identities = {
        (row["score_bits"], row["secondary_score_bits"], row["valid"], row["failure"])
        for row in results.values()
    }
    report = {
        "schema": "JopeTestOnlyNodeCanary.v1",
        "semantic_bit_parity": len(identities) == 1,
        "nodes": results,
    }
    atomic_json(args.report, report)
    if not report["semantic_bit_parity"]:
        raise RuntimeError("corpus_eval node canaries lost score parity")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
