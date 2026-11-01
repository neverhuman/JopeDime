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
import socket
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


def version_source(identity: dict) -> str | None:
    source = identity.get("source_sha40")
    if not isinstance(source, str) or not SOURCE_SHA.fullmatch(source):
        source = identity.get("package", {}).get("source_sha40")
    return source


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


def package_fixture(source_sha: str, fixture_package: Path, output: str) -> dict:
    """Wrap the repository's development-key fixture package for a non-gate rehearsal."""
    require_macos_arm64()
    if not SOURCE_SHA.fullmatch(source_sha) or repo_source_sha() != source_sha:
        raise D1DevError("package source SHA must match the clean checkout HEAD")
    fixture_package = fixture_package.resolve(strict=True)
    if (
        fixture_package.is_symlink()
        or not fixture_package.is_file()
        or not os.access(fixture_package, os.X_OK)
    ):
        raise D1DevError("fixture package must be a regular executable file")
    archs = subprocess.run(
        ["lipo", "-archs", str(fixture_package)],
        text=True,
        capture_output=True,
        check=False,
    )
    if archs.returncode or archs.stdout.strip() != "arm64":
        raise D1DevError("fixture package is not native Darwin arm64")
    version = subprocess.run(
        [str(fixture_package), "--version", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode:
        raise D1DevError("fixture package version probe failed")
    try:
        identity = json.loads(version.stdout)
    except json.JSONDecodeError as error:
        raise D1DevError("fixture package returned invalid version JSON") from error
    features = identity.get("features", {})
    if (
        version_source(identity) != source_sha
        or identity.get("target") != "aarch64-apple-darwin"
        or any(features.get(name) is not True for name in FEATURES)
        or features.get("dev_keys") is not True
        or identity.get("trusts_development_key") is not True
    ):
        raise D1DevError("fixture package identity/features differ from D1 DEV-BUILD")

    out = external_new_dir(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    executable = out / "jopedime"
    shutil.copyfile(fixture_package, executable)
    executable.chmod(0o755)
    package = {
        "schema": PACKAGE_SCHEMA,
        "label": "DEV-BUILD",
        "qualification": "NOT_QUALIFIED",
        "source_sha": source_sha,
        "platform": "macos-arm64",
        "target_triple": "aarch64-apple-darwin",
        "profile": "fixture",
        "features": FEATURES,
        "artifact": "jopedime",
        "artifact_sha256": digest(executable),
        "release_signed": False,
        "development_signed": True,
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
        or package.get("release_signed") is not False
        or package.get("source_sha") != source_sha
        or package.get("platform") != "macos-arm64"
        or package.get("target_triple") != "aarch64-apple-darwin"
        or package.get("profile") != "fixture"
        or package.get("features") != FEATURES
        or package.get("artifact") != "jopedime"
        or package.get("artifact_sha256") != digest(executable)
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
        or package.get("development_signed") is not True
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

    work, home, tmp, tool_path = (
        out / name for name in ("work", "home", "tmp", "no-toolchain")
    )
    for directory in (work, home, tmp, tool_path):
        directory.mkdir()
    if not toolchain_path_is_empty(tool_path):
        raise D1DevError("clean-host toolchain PATH is not an empty directory")
    profile = out / "no-network.sb"
    profile.write_text(
        "(version 1)\n(allow default)\n(deny network*)\n", encoding="utf-8"
    )

    network_log = out / "network-probe.log"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        probe = subprocess.run(
            [
                "sandbox-exec",
                "-f",
                str(profile),
                "/usr/bin/curl",
                "-v",
                "--connect-timeout",
                "1",
                "--max-time",
                "2",
                "-sS",
                "-o",
                "/dev/null",
                f"http://127.0.0.1:{port}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    network_output = probe.stdout + probe.stderr
    network_log.write_text(network_output, encoding="utf-8")
    if probe.returncode == 0 or "Operation not permitted" not in network_output:
        raise D1DevError("macOS sandbox did not deny a live loopback connection")

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

    toolchain_absent = toolchain_path_is_empty(tool_path) and all(
        shutil.which(name, path=str(tool_path)) is None
        for name in ("python", "python3", "rustc", "node")
    )
    if not toolchain_absent:
        raise D1DevError("toolchain executable is visible in the clean-host PATH")
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

    def app(*args: str) -> subprocess.CompletedProcess[str]:
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
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stdout + result.stderr)[-3000:]
            raise D1DevError(f"native D1 command failed ({args[0]}): {detail}")
        return result

    version = app("--version", "--json")
    version_path = out / "version.json"
    version_path.write_text(version.stdout, encoding="utf-8")
    try:
        identity = json.loads(version.stdout)
    except json.JSONDecodeError as error:
        raise D1DevError("native D1 version output is not JSON") from error
    if (
        version_source(identity) != source_sha
        or identity.get("target") != "aarch64-apple-darwin"
        or identity.get("features", {}).get("dev_keys")
        is not package["development_signed"]
        or identity.get("trusts_development_key") is not package["development_signed"]
    ):
        raise D1DevError(
            "native D1 executable identity differs from the pinned DEV-BUILD"
        )

    methods = app("methods", "--task", "regression", "--json")
    (out / "methods.json").write_text(methods.stdout, encoding="utf-8")
    try:
        method_inventory = json.loads(methods.stdout)
    except json.JSONDecodeError as error:
        raise D1DevError("native method discovery output is not JSON") from error
    discovered = any(
        item.get("kind") == "native" and item.get("executable") is True
        for item in method_inventory.get("methods", [])
    )
    if not discovered:
        raise D1DevError("native method discovery returned no executable method")

    train = work / "train.csv"
    train.write_text(
        "x,y\n" + "".join(f"{index},{2 * index + 1}\n" for index in range(48)),
        encoding="utf-8",
    )
    rows = work / "rows.csv"
    rows.write_text("x\n3\n17\n31\n", encoding="utf-8")
    model_root = work / "models"
    fit = app(
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
    )
    (out / "fit.json").write_text(fit.stdout, encoding="utf-8")
    models = sorted(model_root.glob("sha256-*"))
    if len(models) != 1 or not models[0].is_dir():
        raise D1DevError("native D1 synthetic fit did not produce exactly one model")

    predict = app(
        "predict",
        "--model",
        str(models[0]),
        "--input",
        str(rows),
        "--output",
        str(out / "predictions.csv"),
        "--device",
        "cpu",
        "--threads",
        "1",
        "--json",
    )
    (out / "predict.json").write_text(predict.stdout, encoding="utf-8")
    predictions = out / "predictions.csv"
    if not predictions.is_file() or len(predictions.read_text().splitlines()) != 4:
        raise D1DevError("native D1 prediction did not return all synthetic rows")

    exports: dict[str, bool] = {}
    for language in ("python", "rust", "javascript"):
        target = out / f"export-{language}.bin"
        export = app(
            "model",
            "export",
            "--model",
            str(models[0]),
            "--format",
            language,
            "--output",
            str(target),
        )
        (out / f"export-{language}.json").write_text(export.stdout, encoding="utf-8")
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
        "profile": package["profile"],
        "features": FEATURES,
        "development_signed": package["development_signed"],
        "checks": {
            "network_disabled": True,
            "toolchain_absent": toolchain_absent,
            "discovery": discovered,
            "prediction": predictions.is_file(),
            "link_audit_os_libraries_only": True,
            "exports": exports,
        },
        "dev_build_package_sha256": digest(candidate / "DEV-BUILD.json"),
        "evidence": {
            "network_probe": str(network_log),
            "version": str(version_path),
            "method_discovery": str(out / "methods.json"),
            "prediction": str(predictions),
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
    fixture = commands.add_parser(
        "package", help="wrap a development-key fixture package as DEV-BUILD"
    )
    fixture.add_argument("--source-sha", required=True)
    fixture.add_argument("--fixture-package", required=True, type=Path)
    fixture.add_argument("--out", required=True)
    run = commands.add_parser("run", help="rehearse D1 on a DEV-BUILD package")
    run.add_argument("--candidate", required=True)
    run.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        ROOT = args.repo_root.resolve(strict=True)
        if args.command == "package":
            result = package_fixture(args.source_sha, args.fixture_package, args.out)
        else:
            result = run_candidate(args.candidate, args.out)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (D1DevError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"D1 DEV-BUILD: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
