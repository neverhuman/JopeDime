#!/usr/bin/env python3
"""Build a labeled Darwin D1 dev package and rehearse its native clean-host leg.

DEV-BUILD receipts use the D1 v2 shape for inspection but are explicitly not gate
receipts: their status is not PASS and they contain no signed-candidate claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")
FEATURES = ["embedded", "engine", "mcp", "client"]
PACKAGE_SCHEMA = "jope.rel07.d1-darwin-dev-package/v1"
RECEIPT_SCHEMA = "jope.rel07.d1-platform-leg/v2"


class D1DevError(RuntimeError):
    pass


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D1DevError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise D1DevError(f"expected JSON object: {path}")
    return value


def require_macos_arm64() -> None:
    if platform.system() != "Darwin" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        raise D1DevError("D1 macOS dev rehearsal requires native Darwin arm64")


def repo_source_sha() -> str:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode or dirty.returncode or dirty.stdout.strip():
        raise D1DevError("source checkout must be clean and readable")
    value = head.stdout.strip()
    if not SOURCE_SHA.fullmatch(value):
        raise D1DevError("source checkout does not have a full commit SHA")
    return value


def external_new_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise D1DevError("output path must be absolute and non-symlink")
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise D1DevError("output cannot be under /tmp")
    if path.exists():
        raise D1DevError(f"output already exists: {path}")
    if path.resolve().is_relative_to(ROOT.resolve()):
        raise D1DevError("output must be outside the source checkout")
    return path


def toolchain_path_is_empty(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def validate_build_record(source_sha: str, code: Path, record_path: Path) -> dict:
    if code.is_symlink() or not code.is_file() or not os.access(code, os.X_OK):
        raise D1DevError("base executable must be a regular executable file")
    record = read_json(record_path)
    code_sha = digest(code)
    if (
        record.get("schema") != "jope.rel07.d1-darwin-build/v1"
        or record.get("source_sha") != source_sha
        or record.get("target_triple") != "aarch64-apple-darwin"
        or record.get("profile") != "full"
        or record.get("features") != FEATURES
        or record.get("base_executable_sha256") != code_sha
        or record.get("checks")
        != {
            "native_arm64": True,
            "embedded_source_pin": True,
            "full_features": True,
            "macos_os_libraries_only": True,
        }
    ):
        raise D1DevError("Darwin build receipt does not bind the expected source/code")
    return {"sha256": code_sha, "record": record}


def package_dev(source_sha: str, code: Path, build_record: Path, output: str) -> dict:
    require_macos_arm64()
    if not SOURCE_SHA.fullmatch(source_sha) or repo_source_sha() != source_sha:
        raise D1DevError("package source SHA must match the clean checkout HEAD")
    code = code.resolve(strict=True)
    build_record = build_record.resolve(strict=True)
    validate_build_record(source_sha, code, build_record)
    archs = subprocess.run(
        ["lipo", "-archs", str(code)], text=True, capture_output=True, check=False
    )
    if archs.returncode or archs.stdout.strip() != "arm64":
        raise D1DevError("base executable is not native Darwin arm64")
    version = subprocess.run(
        [str(code), "--version", "--json"], text=True, capture_output=True, check=False
    )
    if version.returncode:
        raise D1DevError("base executable version probe failed")
    try:
        identity = json.loads(version.stdout)
    except json.JSONDecodeError as error:
        raise D1DevError("base executable returned invalid version JSON") from error
    features = identity.get("features", {})
    if (
        identity.get("source_sha40") != source_sha
        or identity.get("target") != "aarch64-apple-darwin"
        or any(features.get(name) is not True for name in FEATURES)
        or features.get("dev_keys") is not False
        or identity.get("trusts_development_key") is not False
    ):
        raise D1DevError("base executable identity/features differ from D1")

    out = external_new_dir(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    executable = out / "jopedime"
    shutil.copyfile(code, executable)
    executable.chmod(0o755)
    package = {
        "schema": PACKAGE_SCHEMA,
        "label": "DEV-BUILD",
        "qualification": "NOT_QUALIFIED",
        "source_sha": source_sha,
        "platform": "macos-arm64",
        "target_triple": "aarch64-apple-darwin",
        "profile": "full",
        "features": FEATURES,
        "artifact": "jopedime",
        "artifact_sha256": digest(executable),
        "signed": False,
        "build_record_sha256": digest(build_record),
    }
    (out / "DEV-BUILD.json").write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return package


def validate_dev_candidate(candidate: Path, source_sha: str) -> tuple[Path, dict]:
    if candidate.is_symlink() or not candidate.is_dir():
        raise D1DevError("--candidate must be a DEV-BUILD package directory")
    metadata_path = candidate / "DEV-BUILD.json"
    executable = candidate / "jopedime"
    if metadata_path.is_symlink() or executable.is_symlink():
        raise D1DevError("candidate package files must not be symlinks")
    package = read_json(metadata_path)
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("label") != "DEV-BUILD"
        or package.get("qualification") != "NOT_QUALIFIED"
        or package.get("signed") is not False
        or package.get("source_sha") != source_sha
        or package.get("platform") != "macos-arm64"
        or package.get("target_triple") != "aarch64-apple-darwin"
        or package.get("profile") != "full"
        or package.get("features") != FEATURES
        or package.get("artifact") != "jopedime"
        or package.get("artifact_sha256") != digest(executable)
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise D1DevError("candidate is not an intact source-bound D1 DEV-BUILD")
    return executable, package


def run_candidate(candidate_value: str, output: str) -> dict:
    require_macos_arm64()
    source_sha = repo_source_sha()
    candidate = Path(candidate_value)
    executable, package = validate_dev_candidate(candidate, source_sha)
    out = external_new_dir(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()

    link_log = out / "link-audit.txt"
    audit = subprocess.run(
        ["bash", str(ROOT / "deploy/ci/link-audit.sh"), str(executable), "fused-cpu"],
        text=True,
        capture_output=True,
        check=False,
    )
    link_log.write_text(audit.stdout + audit.stderr, encoding="utf-8")
    if audit.returncode:
        raise D1DevError("Darwin OS-library link audit failed")

    clean_log = out / "clean-host-driver.log"
    driver = subprocess.run(
        [
            "bash",
            str(ROOT / "deploy/appliance/test-clean-host-macos.sh"),
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    clean_log.write_text(driver.stdout + driver.stderr, encoding="utf-8")
    if driver.returncode:
        raise D1DevError("native macOS clean-host discovery/prediction run failed")
    try:
        generic_receipt = json.JSONDecoder().raw_decode(
            driver.stdout[driver.stdout.rfind("{") :]
        )[0]
    except (ValueError, json.JSONDecodeError) as error:
        raise D1DevError("macOS clean-host driver receipt was not readable") from error
    work = Path(generic_receipt["work"])
    home = Path(generic_receipt["home"])
    tmp = Path(generic_receipt["tmp"])
    profile = work.parent / "no-network.sb"
    tool_path = work.parent / "no-toolchain"
    toolchain_absent = toolchain_path_is_empty(tool_path)
    if not toolchain_absent:
        raise D1DevError("clean-host toolchain PATH is not an empty directory")
    app_env = {
        "HOME": str(home),
        "JOPEDIME_HOME": str(home),
        "TMPDIR": str(tmp),
        "PATH": str(tool_path),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
    }

    def app(*args: str, stdout_path: Path | None = None) -> None:
        output_stream = (
            stdout_path.open("w", encoding="utf-8")
            if stdout_path
            else subprocess.DEVNULL
        )
        try:
            result = subprocess.run(
                [
                    "sandbox-exec",
                    "-f",
                    str(profile),
                    "/usr/bin/env",
                    "-i",
                    *(f"{key}={value}" for key, value in app_env.items()),
                    str(executable),
                    *args,
                ],
                cwd=work,
                text=True,
                stderr=subprocess.PIPE,
                check=False,
                stdout=output_stream,
            )
        finally:
            if stdout_path:
                output_stream.close()
        if result.returncode:
            raise D1DevError(
                f"native D1 command failed ({args[0]}): {result.stderr[-1500:]}"
            )

    train = work / "train.csv"
    model_root = work / "models"
    app(
        "fit",
        "--train",
        str(train),
        "--target",
        "y",
        "--method",
        "auto",
        "--candidates",
        "ridge",
        "--folds",
        "2",
        "--model-root",
        str(model_root),
        "--device",
        "cpu",
        "--threads",
        "1",
        "--json",
        stdout_path=out / "fit.json",
    )
    models = sorted(model_root.glob("sha256-*"))
    if len(models) != 1 or not models[0].is_dir():
        raise D1DevError("native D1 synthetic fit did not produce exactly one model")
    exports: dict[str, bool] = {}
    for language in ("python", "rust", "javascript"):
        target = out / f"export-{language}.bin"
        app(
            "model",
            "export",
            "--model",
            str(models[0]),
            "--format",
            language,
            "--output",
            str(target),
            stdout_path=out / f"export-{language}.json",
        )
        exports[language] = target.is_file() and target.stat().st_size > 0
        if not exports[language]:
            raise D1DevError(f"native D1 {language} export is empty")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "gate_id": "D1",
        "platform": "macos-arm64",
        "status": "DEV-BUILD",
        "build_class": "DEV-BUILD",
        "qualification": "NOT_QUALIFIED",
        "source_sha": source_sha,
        "artifact_sha256": package["artifact_sha256"],
        "target_triple": "aarch64-apple-darwin",
        "profile": "full",
        "features": FEATURES,
        "checks": {
            "network_disabled": generic_receipt.get("network_probe") == "passed",
            "toolchain_absent": toolchain_absent,
            "discovery": True,
            "prediction": True,
            "link_audit_os_libraries_only": True,
            "exports": exports,
        },
        "dev_build_package_sha256": digest(candidate / "DEV-BUILD.json"),
        "evidence": {
            "network_discovery_prediction": str(clean_log),
            "link_audit": str(link_log),
            "exports": {name: str(out / f"export-{name}.bin") for name in exports},
        },
    }
    receipt_path = out / "D1-macos-arm64-DEV-BUILD.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="clean exact-SHA source checkout (defaults to this script's checkout)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser(
        "package", help="wrap a Darwin base executable as DEV-BUILD"
    )
    package.add_argument("--source-sha", required=True)
    package.add_argument("--base-executable", required=True, type=Path)
    package.add_argument("--build-record", required=True, type=Path)
    package.add_argument("--out", required=True)
    run = commands.add_parser("run", help="rehearse D1 on a DEV-BUILD package")
    run.add_argument("--candidate", required=True)
    run.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        ROOT = args.repo_root.resolve(strict=True)
        result = (
            package_dev(
                args.source_sha, args.base_executable, args.build_record, args.out
            )
            if args.command == "package"
            else run_candidate(args.candidate, args.out)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (D1DevError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"D1 DEV-BUILD: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
