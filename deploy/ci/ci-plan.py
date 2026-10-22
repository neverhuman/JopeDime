#!/usr/bin/env python3
"""Declare every fast lane's applicability before execution."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

LANES = ("changes", "hygiene", "rust-check", "rust-test", "emitters", "inputs", "python", "e2e-smoke")
# These preserve the existing path gates, adding shared receipt/plan code to every lane.
IMAGE = r"ops/ci/|tools/security-lane\.sh$|deploy/runner/|deploy/docker/|\.github/actions/|deploy/ci/(cargo-artifact|receipt|results|ci-plan|fast-lane|pytest-evidence|aggregate|coverage|training-proof|test-training-proof)"
COMMON = rf"Cargo\.(toml|lock)$|rust-toolchain\.toml$|{IMAGE}|\.github/workflows/ci\.yml$"
FILTERS = {
    "rust": rf"^(crates/|benchmarks/|schemas/|{COMMON}|\.github/workflows/typescript\.yml$|\.config/nextest\.toml$|deny\.toml$|deploy/ci/|deploy/systemd/|deploy/appliance/|models/bundles/|docs/release/signing\.pub$)",
    "emitters": rf"^(crates/feat-math/|datasets/test/|deploy/ci/emitters|{COMMON})",
    "inputs": rf"^(datasets/synthetic/|crates/(jope-cli|jope-campaigns|jope-arena)/|python/src/jope/research/|{COMMON})",
    "python": rf"^(python/|benchmarks/|crates/jope-bench/adapters/|deploy/systemd/|deploy/dl-flywheel/pipelines/rlfw/|deploy/dl-flywheel/pipelines/standalone_paper_monitor_v1/|deploy/dl-flywheel/dashboard/|{IMAGE}|\.github/workflows/ci\.yml$)",
    "e2e": rf"^(crates/(jope-pipeline|jope-cli|jope-engine|jope-datasets|feat-math|jope-assets|jope-runtime|jope-appliance|jope-dist)/|datasets/test/|schemas/|deploy/ci/(e2e-smoke|emitters)|deploy/appliance/|docs/release/signing\.pub$|{COMMON})",
}
GATES = {"rust-check": "rust", "rust-test": "rust", "emitters": "emitters", "inputs": "inputs", "python": "python", "e2e-smoke": "e2e"}


def classify(paths, event):
    if event not in ("pull_request", "workflow_dispatch"):
        raise ValueError(f"unsupported fast CI event: {event}")
    gates = {name: event == "workflow_dispatch" or any(re.search(pattern, path) for path in paths) for name, pattern in FILTERS.items()}
    return {lane: gates[GATES[lane]] if lane in GATES else True for lane in LANES}, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    event = os.environ["GITHUB_EVENT_NAME"]
    base = os.environ.get("BASE_SHA", "")
    if event == "pull_request" and not re.fullmatch(r"[a-f0-9]{40}", base):
        raise ValueError("PR path classification requires full base SHA")
    paths = subprocess.check_output(["git", "diff", "--name-only", "-z", base, "HEAD"], text=True).split("\0") if base else []
    paths = sorted(set(paths) - {""})
    lanes, gates = classify(paths, event)
    plan = {"schema": "jope.ci-plan/v1", "event": event, "base_sha": base,
            "source_sha": os.environ["JOPE_SOURCE_SHA"], "checkout_sha": os.environ["GITHUB_SHA"],
            "run_id": os.environ["GITHUB_RUN_ID"], "attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
            "paths": paths, "lanes": lanes}
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for name, selected in gates.items():
            stream.write(f"{name}={str(selected).lower()}\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
