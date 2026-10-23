#!/usr/bin/env python3
"""Retain and verify exact-source test and census evidence; compilation is not GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

LOGS = ("contracts.log", "census.log", "golden.log", "ordered-bound.log", "bindings.log", "pipeline.log", "cli.log")


def digest(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect(root, mode):
    if mode not in {"host", "compile", "device"} or (root / "mode.txt").read_text().strip() != mode:
        raise ValueError("qualification mode mismatch")
    source = (root / "source.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source) or source != os.environ.get("JOPE_SOURCE_SHA", source):
        raise ValueError("source identity mismatch")
    totals = {}
    all_text = ""
    for name in LOGS:
        text = (root / name).read_text()
        results = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
        if len(results) != 1 or int(results[0][0]) <= 0 or results[0][1:] != ("0", "0"):
            raise ValueError(f"missing or incomplete test execution: {name}")
        totals[name] = int(results[0][0])
        all_text += text
    censuses = {}
    for raw in re.findall(r"CUDA C\+\+ census: (\{[^\n]+\})", all_text):
        item = json.loads(raw)
        kind = item["kind"]
        if kind in censuses or item["unaccounted"] != 0 or item["total"] <= 0:
            raise ValueError("invalid or duplicate census")
        executed = item["executed"]
        count = len(executed) if isinstance(executed, list) else executed
        if count <= 0 or count + len(item.get("refused", [])) != item["total"]:
            raise ValueError("incomplete census denominator")
        censuses[kind] = item
    if set(censuses) != {"recipe", "bound", "legacy-bound", "v2"}:
        raise ValueError("missing registry or versioned census")
    device_batches = len(re.findall(r"CUDA_LINEAR_DEVICE_PASS rows=\d+", all_text))
    if mode == "device" and (device_batches <= 0 or not (root / "device.txt").is_file()):
        raise ValueError("missing actual device prediction evidence")
    files = {name: digest(root / name) for name in (*LOGS, "mode.txt", "source.txt", "toolchains.txt")}
    if mode == "device":
        files["device.txt"] = digest(root / "device.txt")
    return {"schema": "jope.cuda-qualification/v1", "source_sha": source, "mode": mode,
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
            "executed_tests": totals, "censuses": censuses, "files": files,
            "device_prediction_batches": device_batches,
            "gpu_execution": "qualified-linear-head-fixtures" if mode == "device" else "unknown"}


def main():
    action, folder, mode = sys.argv[1:]
    root = Path(folder)
    receipt = root / "qualification.json"
    actual = inspect(root, mode)
    if action == "create":
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if source != actual["source_sha"]:
            raise ValueError("checkout/source mismatch")
        subprocess.run(["git", "diff", "--quiet", "HEAD"], check=True)
        receipt.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    elif action == "verify":
        if json.loads(receipt.read_text()) != actual:
            raise ValueError("qualification receipt mismatch")
    else:
        raise ValueError("unknown evidence operation")
    print(json.dumps({"mode": mode, "source_sha": actual["source_sha"], "gpu_execution": actual["gpu_execution"]}))


if __name__ == "__main__":
    main()
