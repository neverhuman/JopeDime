#!/usr/bin/env python3
"""Reproduce the frozen submission's four errors and annotation-only successor."""
import argparse
import collections
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "crates/jope-automl/submissions/gemini-argon-4"
CORRECTED = ROOT / "crates/jope-automl/src/gemini_argon_4.rs"
PINS = {
    "crates/jope-automl/submissions/gemini-argon-4/original.rs":
        "ddddc2557195830ee7fdb156f14802c63da22c171a31b15c232e45635b74b88c",
    "crates/jope-automl/src/gemini_argon_4.rs":
        "b0aa42d24d0b067e0597f6f538b14cab75dca8b976731668c894d80684d4992d",
    "crates/jope-automl/src/GPT6p1_Sol_Max.rs":
        "fe5bb21ed62edb3aebc1bd141d54834dd3f42b97cb9b0366f2e6b24bc99f75a8",
    "crates/jope-bench/submissions/GPT6_Astra_Max.rs":
        "b51ad3291c6e433388a49ed8338d069a245300d634c9da76be1fb1f1454cd0eb",
    "crates/jope-bench/regressors/Opus5p5_Extra.rs":
        "2e2352f8ac7087076ef60e0408dfee5319a8d7aa8aaac8d657bc67f07f755834",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh retained evidence directory")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.is_relative_to("/tmp"), "evidence must remain outside /tmp")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.monotonic()
    commands = []

    def run(name, argv):
        before = time.monotonic()
        result = subprocess.run(argv, cwd=ROOT, capture_output=True)
        for suffix, data in [("stdout", result.stdout), ("stderr", result.stderr)]:
            (output / f"{name}.{suffix}").write_bytes(data)
        commands.append({"argv": argv, "exit": result.returncode,
                         "wall_seconds": time.monotonic() - before,
                         "stdout_sha256": digest(result.stdout), "stderr_sha256": digest(result.stderr)})
        return result

    receipt = {"schema": "jope.gemini-source-qualification/v1", "started_utc": started,
               "scope": "frozen source and compiler proof; no scientific measurements",
               "commands": commands, "passed": False}
    try:
        receipt["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        receipt["proof_source_sha256"] = digest(Path(__file__).read_bytes())
        receipt["tracked_dirty_paths"] = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).splitlines()
        for name, expected in PINS.items():
            require(digest((ROOT / name).read_bytes()) == expected, f"frozen source changed: {name}")
        original = (RECORD / "original.rs").read_bytes()
        corrected = CORRECTED.read_bytes()
        correction = json.loads((RECORD / "correction.json").read_text())
        require(correction["original_sha256"] == digest(original), "original identity differs")
        require(correction["corrected_sha256"] == digest(corrected), "corrected identity differs")
        changes = [
            (b"let mut qs = Vec::with_capacity(n_q);", b"let mut qs: Vec<f64> = Vec::with_capacity(n_q);"),
            (b"let beta1 = 0.9;", b"let beta1: f64 = 0.9;"),
            (b"let beta2 = 0.999;", b"let beta2: f64 = 0.999;"),
            (b"let mut edges = Vec::new();", b"let mut edges: Vec<f64> = Vec::new();"),
        ]
        rebuilt = original
        for before, after in changes:
            require(rebuilt.count(before) == 1, "annotation site is not unique")
            rebuilt = rebuilt.replace(before, after)
        require(rebuilt == corrected, "changes exceed the four annotations")
        patch = json.loads((RECORD / "annotation-only.patch.json").read_text())["patch"]
        require(patch == "".join(difflib.unified_diff(
            original.decode().splitlines(keepends=True), corrected.decode().splitlines(keepends=True),
            fromfile="original.rs", tofile="gemini_argon_4.rs")), "recorded patch differs")
        require(digest(patch.encode()) == correction["patch_sha256"], "patch identity differs")
        compiler = run("compiler", ["rustc", "-vV"])
        pin = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
        require(compiler.returncode == 0 and f"release: {pin}\n".encode() in compiler.stdout,
                "compiler differs from the repository pin")
        build = ROOT / "target/gemini-source-qualification"
        build.mkdir(parents=True, exist_ok=True)
        for name, source in [("original", RECORD / "original.rs"), ("corrected", CORRECTED)]:
            result = run(name, ["rustc", "--edition=2021", "--crate-name", "gemini_argon_4",
                               "--crate-type", "rlib", "--error-format=json", "-D", "unsafe-code",
                               "-C", "opt-level=1", str(source), "-o", str(build / f"{name}.rlib")])
            if name == "original":
                diagnostics = [json.loads(line) for line in result.stderr.splitlines()]
                codes = [d["code"]["code"] for d in diagnostics if d.get("level") == "error" and d.get("code")]
                require(result.returncode == 1 and collections.Counter(codes) == {"E0282": 2, "E0689": 2},
                        "original no longer fails with its four annotation errors")
                receipt["original_error_codes"] = codes
            else:
                require(result.returncode == 0, "annotation-only successor did not compile")
                receipt["compiled_artifact_sha256"] = digest((build / f"{name}.rlib").read_bytes())
        receipt.update(passed=True, source_sha256=PINS, patch_sha256=digest(patch.encode()),
                       compiler=compiler.stdout.decode(), scientific_fits=0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt["error"] = str(error)
    finally:
        receipt.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       wall_seconds=time.monotonic() - clock)
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
