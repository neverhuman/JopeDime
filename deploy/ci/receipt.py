#!/usr/bin/env python3
"""Run one declared CI lane and retain a fresh, identity-bound execution receipt."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import results

ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = ("CARGO_BUILD_JOBS", "CARGO_INCREMENTAL", "CARGO_PROFILE_DEV_DEBUG", "CARGO_TARGET_DIR",
            "OMP_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "JOPE_PARITY_STRICT", "JOPE_RUNNER_LANE", "JOPE_CPU_CLASS", "JOPE_RUST_TEST_PARTITION",
            "RUSTC_WRAPPER", "SCCACHE_DIR", "SCCACHE_CONF", "SCCACHE_SERVER_UDS")

INPUTS = ("Cargo.lock", "rust-toolchain.toml", "python/uv.lock", "benchmarks/pyproject.toml", "benchmarks/uv.lock", "deploy/ci/coverage.json", "deploy/ci/fast-lane.sh", "deploy/ci/receipt.py", "deploy/ci/results.py", "deploy/ci/ci-plan.py", "deploy/ci/full-lane.sh", "deploy/ci/full-main.py", "deploy/ci/pytest-evidence.sh")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(env, checkout_sha):
    if env.get("GITHUB_EVENT_NAME") == "pull_request" and not env.get("JOPE_SOURCE_SHA"):
        raise ValueError("PR receipts require JOPE_SOURCE_SHA")
    source = env.get("JOPE_SOURCE_SHA", checkout_sha)
    expected_checkout = env.get("GITHUB_SHA", checkout_sha)
    for sha in (source, checkout_sha, expected_checkout):
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("identity requires full lowercase source and checkout SHAs")
    if checkout_sha != expected_checkout:
        raise ValueError("checkout does not match GITHUB_SHA")
    run_id, attempt = env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"]
    if not run_id.isdecimal() or not attempt.isdecimal() or int(attempt) < 1:
        raise ValueError("run ID and positive attempt are required")
    return {"source_sha": source, "checkout_sha": checkout_sha, "run_id": run_id,
            "attempt": int(attempt), "ref": env.get("GITHUB_REF", ""),
            "event": env.get("GITHUB_EVENT_NAME", "local-proof")}


def validate_counts(counts):
    fields = ("collected", "executed", "passed", "failed", "skipped", "excluded")
    if set(counts) != set(fields):
        raise ValueError("counts must explicitly contain collected/executed/passed/failed/skipped/excluded")
    for key, value in counts.items():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"invalid {key} count")
    if all(counts[k] is not None for k in ("executed", "passed", "failed")):
        if counts["executed"] != counts["passed"] + counts["failed"]:
            raise ValueError("execution counts do not reconcile")
    if all(counts[k] is not None for k in ("collected", "executed", "skipped", "excluded")):
        if counts["collected"] != counts["executed"] + counts["skipped"] + counts["excluded"]:
            raise ValueError("collection counts do not reconcile")


def run_lane(lane, command, output, counts_path=None, evidence=(), backend="none", junit=None):
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_identity = identity(os.environ, checkout)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).splitlines()
    # Exclusive creation makes old artifacts and accidentally repeated steps fail closed.
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "jope.ci-receipt/v1", **run_identity, "lane": lane,
               "command": command, "cwd": str(Path.cwd()), "started_at": utc(), "dirty_source": dirty,
               "environment": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
               "inputs": {str(p): digest(ROOT / p) for p in INPUTS},
               "status": "started", "exit_status": None, "counts": dict.fromkeys(("collected", "executed", "passed", "failed", "skipped", "excluded")),
               "evidence": {}}
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        if counts_path is not None and counts_path.exists():
            raise ValueError(f"counts file already exists before execution: {counts_path}")
        with (output / "command.log").open("w") as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                     env=dict(os.environ, JOPE_LANE_EVIDENCE=str(output.resolve())))
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
            child.stdout.close()
            code = child.wait()
            code = code if code >= 0 else 128 - code
        receipt.update(exit_status=code, status="passed" if code == 0 else "failed")
        if backend != "none":
            receipt["counts"] = results.summarize((output / "command.log").read_text(), backend, junit)
            validate_counts(receipt["counts"])
            if backend == "nextest":
                listing = json.loads((output / "collection.json").read_text())
                listed = sum(len(suite.get("testcases", {})) for suite in listing["rust-suites"].values())
                if listing["test-count"] != listed or listed != receipt["counts"]["collected"]:
                    raise ValueError("nextest collection and execution counts disagree")
        if counts_path is not None:
            counts = json.loads(counts_path.read_text())
            validate_counts(counts)
            receipt["counts"] = counts
            (output / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
        for source in evidence:
            if not source.is_file():
                raise ValueError(f"missing expected evidence: {source}")
            target = output / "evidence" / source.name
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
    except (OSError, ValueError) as exc:
        receipt.update(status="failed" if receipt["exit_status"] not in (None, 0) else "unavailable", diagnostic=str(exc))
        code = receipt["exit_status"] or 1
    receipt["evidence"] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p != path}
    receipt["dirty_source_after"] = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).splitlines()
    receipt["finished_at"] = utc()
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--backend", choices=["none", "nextest", "pytest", "rust", "unittest"], default="none")
    parser.add_argument("--junit", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    mapping = json.loads((ROOT / "deploy/ci/coverage.json").read_text())
    if args.lane not in mapping["lanes"] or not command:
        parser.error("a declared lane and command are required")
    try:
        return run_lane(args.lane, command, args.output, args.counts, args.evidence, args.backend, args.junit)
    except (ValueError, OSError, KeyError) as exc:
        print(f"receipt: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
