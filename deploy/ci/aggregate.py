#!/usr/bin/env python3
"""Fail closed on every required lane; only a declared path exclusion permits skipping."""
import argparse
import datetime
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import subprocess

from receipt import INPUTS, digest, identity, validate_counts
from results import summarize

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ci_plan", ROOT / "deploy/ci/ci-plan.py")
plan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_module)


def load_receipts(root, publications=None):
    receipts = {}
    # download-artifact keeps each artifact in a separate immediate directory.
    # Product pipeline evidence also contains receipt.json; it is data covered by
    # the CI receipt's hashes, not another CI lane to discover recursively.
    for artifact in root.iterdir():
        path = artifact / "receipt.json"
        if artifact.is_symlink() or not artifact.is_dir() or path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact lacks a root CI receipt: {artifact.name}")
        receipt = json.loads(path.read_text())
        receipts.setdefault(receipt["lane"], []).append((path, receipt))
    # A timed-out FinalizeArtifact may have committed remotely before its response
    # was lost. Keep each upload, but count only one execution if explicitly numbered
    # publication attempts contain exactly the same complete bytes.
    for lane, candidates in receipts.items():
        if publications is not None:
            publications[lane] = sorted(path.parent.name for path, _ in candidates)
        for path, receipt in candidates:
            base = f"ci-{receipt['run_id']}-{receipt['attempt']}-{lane}-pub-"
            if '-pub-' in path.parent.name and not re.fullmatch(re.escape(base) + r"[1-3]", path.parent.name):
                raise ValueError(f"{lane}: invalid publication attempt name")
        if len(candidates) < 2:
            continue
        signatures = []
        numbers = set()
        for path, receipt in candidates:
            base = f"ci-{receipt['run_id']}-{receipt['attempt']}-{lane}-pub-"
            name = path.parent.name
            if not re.fullmatch(re.escape(base) + r"[1-3]", name):
                raise ValueError(f"{lane}: ambiguous execution receipts, not publication retries")
            number = name.rsplit('-', 1)[1]
            if number in numbers:
                raise ValueError(f"{lane}: repeated publication number")
            numbers.add(number)
            files = []
            for member in sorted(path.parent.rglob('*')):
                if member.is_symlink():
                    raise ValueError(f"{lane}: symlink in publication")
                if member.is_file():
                    files.append((str(member.relative_to(path.parent)), digest(member)))
                elif not member.is_dir():
                    raise ValueError(f"{lane}: special file in publication")
            signatures.append(files)
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"{lane}: divergent publication attempts")
        receipts[lane] = [min(candidates, key=lambda row: row[0].parent.name)]
    return receipts


def verify(mapping, plan, needs, receipts, expected):
    errors, inventory = [], {}
    def reject(message):
        errors.append(message)
    for field, value in expected.items():
        if field in ("source_sha", "checkout_sha", "run_id", "attempt", "event") and plan.get(field) != value:
            reject(f"plan {field} mismatch")
    if plan.get("schema") != "jope.ci-plan/v1":
        reject("unsupported path plan")
    try:
        planned, _ = plan_module.classify(plan["paths"], plan["event"])
        if planned != plan.get("lanes"):
            reject("path plan differs from declared filters")
    except (ValueError, KeyError, TypeError) as exc:
        reject(f"invalid path plan: {exc}")
        planned = dict.fromkeys(plan_module.LANES, True)
    if set(needs) != set(plan_module.LANES):
        reject("missing or unexpected dependency results")
    if set(receipts) - set(plan_module.LANES):
        reject("unexpected receipt lane")
    for lane in plan_module.LANES:
        result = needs.get(lane, {}).get("result", "missing")
        if not planned[lane]:
            if result != "skipped" or lane in receipts:
                reject(f"{lane}: exclusion requires skipped job and no execution receipt")
            inventory[lane] = {"status": "skipped-by-path", "tests": None}
            continue
        if result != "success":
            reject(f"{lane}: job result {result}")
        candidates = receipts.get(lane, [])
        if len(candidates) != 1:
            reject(f"{lane}: expected one receipt, got {len(candidates)}")
            inventory[lane] = {"status": "missing-or-ambiguous", "tests": None}
            continue
        path, receipt = candidates[0]
        for field in ("source_sha", "checkout_sha", "run_id", "attempt", "event"):
            if receipt.get(field) != expected[field]:
                reject(f"{lane}: {field} mismatch")
        if receipt.get("schema") != "jope.ci-receipt/v1" or receipt.get("lane") != lane:
            reject(f"{lane}: receipt schema/lane mismatch")
        if receipt.get("status") != "passed" or receipt.get("exit_status") != 0 or receipt.get("diagnostic"):
            reject(f"{lane}: receipt status {receipt.get('status')}, exit {receipt.get('exit_status')}")
        if receipt.get("dirty_source") != [] or receipt.get("dirty_source_after") != []:
            reject(f"{lane}: dirty or unrecorded source state")
        if receipt.get("command") != ["bash", "deploy/ci/fast-lane.sh", lane]:
            reject(f"{lane}: unexpected command")
        try:
            start = datetime.datetime.fromisoformat(receipt["started_at"])
            end = datetime.datetime.fromisoformat(receipt["finished_at"])
            if start.tzinfo is None or end.tzinfo is None or end < start:
                raise ValueError("invalid execution times")
        except (KeyError, TypeError, ValueError):
            reject(f"{lane}: incomplete execution times")
        for name in INPUTS:
            if receipt.get("inputs", {}).get(name) != digest(ROOT / name):
                reject(f"{lane}: input mismatch {name}")
        evidence = receipt.get("evidence", {})
        actual_files = {str(p.relative_to(path.parent)) for p in path.parent.rglob("*") if p.is_file() and p != path}
        if actual_files != set(evidence):
            reject(f"{lane}: evidence manifest differs from artifact files")
        required = {"command.log"}
        if lane == "changes":
            required.add("plan.json")
        if lane == "rust-test":
            required.update(("junit.xml", "collection.json", "nextest-started", "nextest.toml", "disk-before.txt", "disk-after.txt", "target-bytes.txt"))
        if lane == "python":
            required.add("junit.xml")
        if not required.issubset(evidence):
            reject(f"{lane}: missing required evidence {sorted(required - set(evidence))}")
        for name, expected_digest in evidence.items():
            file = path.parent / name
            if Path(name).is_absolute() or ".." in Path(name).parts or file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(path.parent.resolve()):
                reject(f"{lane}: unsafe or missing evidence {name}")
            elif digest(file) != expected_digest:
                reject(f"{lane}: evidence digest mismatch {name}")
        counts = receipt.get("counts", {})
        try:
            validate_counts(counts)
            if mapping["lanes"][lane]["requires_nonzero_tests"]:
                if any(value is None for value in counts.values()) or counts["collected"] <= 0 or counts["executed"] <= 0 or counts["failed"] != 0:
                    raise ValueError("zero, unknown or failing tests")
            backend = mapping["lanes"][lane]["result_backend"]
            junit = path.parent / "junit.xml" if lane in ("rust-test", "python") else None
            if summarize((path.parent / "command.log").read_text(), backend, junit) != counts:
                raise ValueError("receipt counts differ from execution evidence")
            if backend == "nextest":
                listing = json.loads((path.parent / "collection.json").read_text())
                listed = sum(len(suite.get("testcases", {})) for suite in listing["rust-suites"].values())
                if listing["test-count"] != listed or listed != counts["collected"]:
                    raise ValueError("collection and execution disagree")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            reject(f"{lane}: invalid test counts: {exc}")
        inventory[lane] = {"status": receipt.get("status", "unknown"), "tests": counts, "job_result": result}
    return {"schema": "jope.ci-aggregate/v1", **expected, "status": "failed" if errors else "passed",
            "lanes": inventory, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--needs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        publications = {}
        receipts = load_receipts(args.receipts, publications)
        changes = receipts.get("changes", [])
        if len(changes) != 1:
            raise ValueError("missing or ambiguous changes receipt")
        plan = json.loads((changes[0][0].parent / "plan.json").read_text())
        mapping = json.loads((ROOT / "deploy/ci/coverage.json").read_text())
        checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise ValueError("aggregate checkout is dirty")
        expected = identity(os.environ, checkout)
        if plan["base_sha"] != os.environ.get("BASE_SHA", ""):
            raise ValueError("path plan base SHA mismatch")
        paths = subprocess.check_output(["git", "diff", "--name-only", "-z", plan["base_sha"], "HEAD"], cwd=ROOT, text=True).split("\0") if plan["base_sha"] else []
        if sorted(set(paths) - {""}) != plan["paths"]:
            raise ValueError("path plan does not match checked-out changes")
        report = verify(mapping, plan, json.loads(args.needs.read_text()), receipts, expected)
        report["publication_artifacts"] = publications
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"status": "failed", "errors": [str(exc)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return report["status"] != "passed"


if __name__ == "__main__":
    raise SystemExit(main())
