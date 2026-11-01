#!/usr/bin/env python3
"""Bind runtime A5 inputs, invoke the A5 provider, and verify its result receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
PROVIDER = ROOT / "deploy/qualification/v1.0/providers/a5.py"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")

# release-gates already supplies a restricted child environment. Preserve its
# build/runtime inputs while keeping credentials and arbitrary caller variables out.
FORWARDED_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "CARGO_TARGET_DIR",
    "CARGO_BUILD_JOBS",
    "CARGO_INCREMENTAL",
    "CARGO_PROFILE_DEV_DEBUG",
    "CARGO_PROFILE_RELEASE_DEBUG",
    "CARGO_ENCODED_RUSTFLAGS",
    "RUSTFLAGS",
    "RUSTC_WRAPPER",
    "CC",
    "CXX",
    "AR",
    "LD_LIBRARY_PATH",
    "PKG_CONFIG_PATH",
    "CMAKE_PREFIX_PATH",
    "LIBTORCH",
    "LIBTORCH_USE_PYTORCH",
    "LIBTORCH_BYPASS_VERSION_CHECK",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "TMPDIR",
    "PYTHONDONTWRITEBYTECODE",
    "AWS_EC2_METADATA_DISABLED",
)


class WrapperError(ValueError):
    """An unsafe or incomplete A5 wrapper input."""


def _path(value: str | None, name: str, *, must_exist: bool) -> Path:
    if not value:
        raise WrapperError(f"A5 wrapper requires {name}")
    path = Path(value)
    if not path.is_absolute():
        raise WrapperError(f"A5 {name} must be absolute")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise WrapperError(f"A5 {name} must not contain symlinks")
    if must_exist and not path.is_file():
        raise WrapperError(f"A5 {name} must be a regular file")
    if must_exist and path.resolve() != path:
        raise WrapperError(f"A5 {name} must use its canonical absolute path")
    return path


def _result_path(value: str | None) -> Path:
    path = _path(value, "result path", must_exist=False)
    if path.exists() or path.is_symlink():
        raise WrapperError("A5 result path must be new")
    if not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise WrapperError("A5 result parent must be an existing regular directory")
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT.resolve()):
        raise WrapperError("A5 result must be outside the source checkout")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(value: str | None) -> str:
    if value is None:
        try:
            value = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise WrapperError(
                "A5 source SHA is required outside a Git checkout"
            ) from error
    if SOURCE_SHA.fullmatch(value) is None:
        raise WrapperError("A5 source SHA must be 40 lowercase hex characters")
    return value


def _write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=os.environ.get("JOPE_REL07_ARTIFACT"))
    parser.add_argument("--result", default=os.environ.get("JOPE_REL07_RESULT"))
    parser.add_argument("--source-sha", default=os.environ.get("JOPE_REL07_SOURCE_SHA"))
    parser.add_argument("--manifest", default=os.environ.get("JOPE_REL07_A5_MANIFEST"))
    parser.add_argument(
        "--manifest-sha256",
        default=os.environ.get("JOPE_REL07_A5_MANIFEST_SHA256"),
    )
    parser.add_argument(
        "--dev-build",
        action="store_true",
        help="write a distinct DEV-BUILD diagnostic, never a gate receipt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        candidate = _path(args.candidate, "candidate path", must_exist=True)
        result = _result_path(args.result)
        source_sha = _source_sha(args.source_sha)
        artifact_sha = _sha256(candidate)

        if args.dev_build:
            receipt = {
                "schema": "jopedime.rel07.a5-dev-build/v1",
                "id": "A5",
                "status": "DEV-BUILD",
                "qualification_status": "NOT_RUN",
                "not_a_gate_receipt": True,
                "source_sha": source_sha,
                "artifact_sha256": artifact_sha,
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_new_json(result, receipt)
            print(json.dumps(receipt, sort_keys=True))
            return 0

        manifest = _path(args.manifest, "transcript manifest", must_exist=True)
        if manifest.resolve().is_relative_to(ROOT.resolve()):
            raise WrapperError(
                "A5 transcript manifest must be outside the source checkout"
            )
        if manifest.parent.resolve() != manifest.parent:
            raise WrapperError("A5 transcript directory must not be a symlink")
        if SHA256.fullmatch(args.manifest_sha256 or "") is None:
            raise WrapperError("A5 manifest SHA-256 must be full lowercase hex")
        if _sha256(manifest) != args.manifest_sha256:
            raise WrapperError("A5 manifest does not match the trusted SHA-256")
        if os.environ.get("JOPE_REL07_GATE_ID", "A5") != "A5":
            raise WrapperError("A5 wrapper requires gate id A5")

        child_environment = {
            key: os.environ[key] for key in FORWARDED_ENV if key in os.environ
        }
        child_environment.update(
            JOPE_REL07_GATE_ID="A5",
            JOPE_REL07_ARTIFACT=str(candidate),
            JOPE_REL07_ARTIFACT_SHA256=artifact_sha,
            JOPE_REL07_SOURCE_SHA=source_sha,
            JOPE_REL07_RESULT=str(result),
            JOPE_REL07_A5_MANIFEST=str(manifest),
            JOPE_REL07_A5_MANIFEST_SHA256=args.manifest_sha256,
        )
        completed = subprocess.run(
            [sys.executable, "-B", str(PROVIDER)],
            cwd=ROOT,
            env=child_environment,
            check=False,
        )
        if completed.returncode:
            return completed.returncode
        if result.is_symlink() or not result.is_file():
            raise WrapperError("A5 provider returned success without a regular result")
        try:
            receipt = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WrapperError("A5 provider result is unreadable JSON") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != "jope.rel07.gate-execution/v1"
            or receipt.get("id") != "A5"
            or receipt.get("status") != "PASS"
            or receipt.get("source_sha") != source_sha
            or receipt.get("artifact_sha256") != artifact_sha
            or receipt.get("evidence_manifest_sha256") != args.manifest_sha256
        ):
            raise WrapperError("A5 provider result is not bound to these inputs")
        print(json.dumps({"id": "A5", "status": "PASS", "result": str(result)}))
        return 0
    except (OSError, WrapperError) as error:
        print(f"A5 wrapper refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
