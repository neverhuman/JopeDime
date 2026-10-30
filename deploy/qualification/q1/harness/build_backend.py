#!/usr/bin/env python3
"""Compile the reference adapter using actual locked Cargo artifact messages, one lane target."""

import hashlib
import json
import os
from pathlib import Path
import subprocess


def build():
    root = Path(__file__).resolve().parents[4]
    target = root / "target"
    output_dir = target / "qualification" / "q1"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CARGO_BUILD_JOBS="8",
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
    )
    env.pop("CARGO_TARGET_DIR", None)
    cargo = [
        "cargo",
        "build",
        "--locked",
        "--no-default-features",
        "-p",
        "feat-math",
        "--message-format=json-render-diagnostics",
        "--target-dir",
        str(target),
    ]
    result = subprocess.run(cargo, cwd=root, env=env, capture_output=True, text=True)
    (output_dir / "cargo-artifacts.jsonl").write_text(result.stdout)
    (output_dir / "cargo-stderr.log").write_text(result.stderr)
    result.check_returncode()
    artifacts = {}
    for line in result.stdout.splitlines():
        item = json.loads(line)
        if item.get("reason") == "compiler-artifact" and item["target"]["name"] in (
            "feat_math",
            "serde_json",
        ):
            libs = [f for f in item["filenames"] if f.endswith(".rlib")]
            if len(libs) == 1:
                artifacts[item["target"]["name"]] = libs[0]
    if set(artifacts) != {"feat_math", "serde_json"}:
        raise RuntimeError("missing or ambiguous Cargo library artifacts")
    source = Path(__file__).with_name("native_heuristic.rs")
    output = output_dir / (
        "native-heuristic.exe" if os.name == "nt" else "native-heuristic"
    )
    command = [
        "rustc",
        "--edition=2021",
        "--crate-name",
        "q1_native_heuristic",
        "-C",
        "debuginfo=line-tables-only",
        "-D",
        "warnings",
        "-L",
        "dependency=" + str(Path(artifacts["serde_json"]).parent),
    ]
    for name, artifact in sorted(artifacts.items()):
        command += ["--extern", name + "=" + artifact]
    command += [str(source), "-o", str(output)]
    subprocess.run(command, cwd=root, env=env, check=True)
    receipt = {
        "backend": str(output),
        "backend_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "root_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "lock_sha256": hashlib.sha256((root / "Cargo.lock").read_bytes()).hexdigest(),
        "cargo_command": cargo,
        "rustc_command": command,
        "artifacts": artifacts,
    }
    (output_dir / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
