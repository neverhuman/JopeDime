#!/usr/bin/env python3
"""Strict counterpart to the task card's status-masking grep pipeline."""

import hashlib
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPIKE = pathlib.Path(__file__).resolve().parent
BANNED = {
    "onig",
    "onig_sys",
    "esaxx",
    "esaxx-rs",
    "tch",
    "torch-sys",
    "rusqlite",
    "libsqlite3",
    "libsqlite3-sys",
    "pyo3",
    "openssl-sys",
    "aws-lc-sys",
    "syntect",
}


def main():
    env = dict(
        os.environ,
        CARGO_BUILD_JOBS="8",
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        CARGO_TARGET_DIR=str(ROOT / "target"),
        TMPDIR=str(ROOT / "target/tmp"),
    )
    results = []
    for package in ("mcp-spike-dispatcher", "mcp-spike-rmcp3", "jope-appliance"):
        command = [
            "cargo",
            "tree",
            "--locked",
            "-p",
            package,
            "-e",
            "normal",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ]
        if package != "jope-appliance":
            command += ["--manifest-path", "research/mcp-spike/Cargo.toml"]
        completed = subprocess.run(
            ["nice", "-n", "10", *command],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pairs = sorted(
            set(re.findall(r"^([A-Za-z0-9_-]+) v([^\s]+)", completed.stdout, re.M))
        )
        banned = sorted({name for name, _ in pairs} & BANNED)
        status = (
            "PASS" if completed.returncode == 0 and pairs and not banned else "FAIL"
        )
        results.append(
            {
                "package": package,
                "command": command,
                "exit_code": completed.returncode,
                "status": status,
                "unique_packages_including_root": len(pairs),
                "packages": pairs,
                "banned_matches": banned,
                "stderr": completed.stderr,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            }
        )
        print(
            package,
            status,
            f"{len(pairs)} unique packages including root",
            "banned:",
            banned,
        )
    (SPIKE / "evidence/dependencies.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    return 0 if all(result["status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
