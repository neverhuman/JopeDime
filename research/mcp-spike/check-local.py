#!/usr/bin/env python3
"""Run the card's exact local checks and preserve failures without changing product code."""

import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPIKE = pathlib.Path(__file__).resolve().parent
COMMANDS = [
    ("runtime-fmt", "cargo fmt -p jope-runtime -- --check"),
    (
        "runtime-clippy",
        "cargo clippy -p jope-runtime --all-targets --locked -- -D warnings",
    ),
    ("runtime-nextest", "cargo nextest run -P pr --locked -p jope-runtime"),
    ("workspace-deny", "cargo deny check"),
    (
        "card-banned-tree",
        'cargo tree -p jope-appliance -e normal | grep -E "onig|esaxx|tch|torch-sys|rusqlite|libsqlite3|pyo3|openssl-sys|aws-lc-sys|syntect" && exit 1 || true',
    ),
    ("hygiene", "just hygiene"),
    ("lint-paths", "just lint-paths"),
    ("audit", "just audit"),
    (
        "spike-fmt",
        "cargo fmt --manifest-path research/mcp-spike/Cargo.toml --all -- --check",
    ),
    (
        "spike-clippy",
        "cargo clippy --manifest-path research/mcp-spike/Cargo.toml --workspace --all-targets --locked -- -D warnings",
    ),
    (
        "spike-deny",
        "cargo deny --manifest-path research/mcp-spike/Cargo.toml --locked --config deny.toml check",
    ),
]


def main():
    logdir = SPIKE / "evidence/local/checks"
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        CARGO_BUILD_JOBS="8",
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        CARGO_TARGET_DIR=str(ROOT / "target"),
        TMPDIR=str(ROOT / "target/tmp"),
    )
    env["PATH"] = str(ROOT / "target/tools") + os.pathsep + env["PATH"]
    env.pop("JOPE_AUDIT_REPORT_ONLY", None)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt = {
        "schema": "jope.mcp-spike-checks/v1",
        "source_head": head,
        "candidate_source": "source-files.json binds uncommitted spike files if present",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            key: env[key]
            for key in (
                "CARGO_BUILD_JOBS",
                "CARGO_INCREMENTAL",
                "CARGO_PROFILE_DEV_DEBUG",
                "CARGO_TARGET_DIR",
                "TMPDIR",
            )
        },
        "nice": 10,
        "checks": [],
    }
    output = SPIKE / "evidence/checks.json"
    for name, command in COMMANDS:
        load = os.getloadavg()[0]
        allowed_cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        if load > 0.85 * allowed_cpus:
            raise SystemExit(f"CPU load {load} exceeds lane budget; stop before {name}")
        log = logdir / f"{name}.log"
        start = time.monotonic()
        utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log.open("wb") as stream:
            completed = subprocess.run(
                ["nice", "-n", "10", "bash", "-o", "pipefail", "-c", command],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        data = log.read_bytes()
        lines = data.decode("utf-8", errors="replace").splitlines()
        item = {
            "name": name,
            "command": command,
            "utc": utc,
            "wall_seconds": time.monotonic() - start,
            "exit_code": completed.returncode,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "log": str(log.relative_to(SPIKE)),
            "log_sha256": hashlib.sha256(data).hexdigest(),
            "output_head": lines[:8],
            "output_tail": lines[-35:],
        }
        if name == "card-banned-tree":
            item["status"] = "UNQUALIFIED"
            item["limitation"] = (
                "The card's && exit 1 || true masks both dependency errors and banned matches; use strict scan evidence separately."
            )
        receipt["checks"].append(item)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            f"{name}: {item['status']} exit={completed.returncode} {item['wall_seconds']:.2f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
