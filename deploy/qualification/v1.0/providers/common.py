"""Shared candidate binding and result helpers for REL-07 gate providers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")
ARTIFACT_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROOT = Path(__file__).resolve().parents[4]


class ProviderError(ValueError):
    """A candidate or provider input is not safe to qualify."""


@dataclass(frozen=True)
class Context:
    gate_id: str
    artifact: Path
    artifact_sha256: str
    source_sha: str
    result: Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def context(expected_gate: str) -> Context:
    values = {
        "gate_id": os.environ.get("JOPE_REL07_GATE_ID"),
        "artifact": os.environ.get("JOPE_REL07_ARTIFACT"),
        "artifact_sha256": os.environ.get("JOPE_REL07_ARTIFACT_SHA256"),
        "source_sha": os.environ.get("JOPE_REL07_SOURCE_SHA"),
        "result": os.environ.get("JOPE_REL07_RESULT"),
    }
    if values["gate_id"] != expected_gate:
        raise ProviderError(f"provider requires gate {expected_gate}")
    artifact_value = values["artifact"]
    result_value = values["result"]
    artifact_sha = values["artifact_sha256"]
    source_sha = values["source_sha"]
    if not artifact_value or not result_value:
        raise ProviderError("candidate and result paths are required")
    if not artifact_sha or not ARTIFACT_SHA256.fullmatch(artifact_sha):
        raise ProviderError("artifact SHA-256 must be lowercase full hex")
    if not source_sha or not SOURCE_SHA.fullmatch(source_sha):
        raise ProviderError("source SHA must be lowercase full hex")

    artifact = Path(artifact_value)
    result = Path(result_value)
    if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
        raise ProviderError("candidate must be an absolute regular non-symlink file")
    if not result.is_absolute() or result.exists() or result.is_symlink():
        raise ProviderError("result must be a new absolute path")
    if sha256(artifact) != artifact_sha:
        raise ProviderError("candidate bytes differ from the declared artifact SHA")
    return Context(expected_gate, artifact, artifact_sha, source_sha, result)


def verify_candidate(
    ctx: Context, run: Callable[..., subprocess.CompletedProcess] = subprocess.run
) -> None:
    """Verify the release envelope and require its embedded source pin to match."""
    before = sha256(ctx.artifact)
    if before != ctx.artifact_sha256:
        raise ProviderError("candidate changed before verification")

    base = ["cargo", "run", "--locked", "-p", "jope-dist", "--"]
    inspect = run(
        [*base, "inspect", "--file", str(ctx.artifact)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        raise ProviderError("jope-dist inspect rejected the candidate")
    try:
        report = json.loads(inspect.stdout)
        manifest = report["manifest"]
        profile = str(manifest["profile"])
        source = manifest["source"]["sha40"]
        scope = manifest["scope"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ProviderError("candidate inspection omitted release identity") from error
    if Path(profile).stem != "full" or scope == "development":
        raise ProviderError("candidate must use the non-development full profile")
    if source != ctx.source_sha:
        raise ProviderError("candidate source pin differs from the tested checkout")

    verified = run(
        [*base, "verify", "--file", str(ctx.artifact)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode != 0:
        raise ProviderError("jope-dist verify rejected the candidate")
    try:
        verify_report = json.loads(verified.stdout)
    except json.JSONDecodeError as error:
        raise ProviderError("candidate verifier did not emit JSON") from error
    if (
        verify_report.get("schema") != "jope.appliance-verify/v1"
        or verify_report.get("ok") is not True
    ):
        raise ProviderError("candidate verifier did not report a full-byte PASS")
    if sha256(ctx.artifact) != before:
        raise ProviderError("candidate bytes changed during verification")


def junit_counts(path: Path | str) -> dict[str, int]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ProviderError("pytest did not produce valid JUnit evidence") from error
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {
        key: 0
        for key in ("collected", "executed", "passed", "failed", "skipped", "excluded")
    }
    for suite in suites:
        tests = int(suite.attrib.get("tests", "0"))
        failures = int(suite.attrib.get("failures", "0"))
        errors = int(suite.attrib.get("errors", "0"))
        skipped = int(suite.attrib.get("skipped", "0"))
        counts["collected"] += tests
        counts["executed"] += tests - skipped
        counts["passed"] += tests - failures - errors - skipped
        counts["failed"] += failures + errors
        counts["skipped"] += skipped
    return counts


def rust_counts(output: str) -> dict[str, int]:
    match = re.search(
        r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; "
        r"(\d+) ignored; (\d+) measured; \d+ filtered out",
        output,
    )
    if match is None:
        raise ProviderError("cargo test did not report a test result summary")
    passed, failed, skipped, excluded = map(int, match.groups())
    return {
        "collected": passed + failed + skipped + excluded,
        "executed": passed + failed,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "excluded": excluded,
    }


def exact_case_passed(output: str, test: str) -> bool:
    """Whether libtest reported exactly one passing ``test`` case.

    Under ``--nocapture`` a case's own stdout sits between ``test NAME ... ``
    and its final status, so a literal ``test NAME ... ok`` is not reliable.
    The case block runs until the libtest summary; callers still require
    ``rust_counts`` to report exactly one pass.
    """
    marker = f"test {test} ... "
    start = output.find(marker)
    if start < 0 or output.find(marker, start + len(marker)) >= 0:
        return False
    end = output.find("\ntest result:", start)
    block = output[start : end if end >= 0 else len(output)]
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    return (
        bool(lines)
        and lines[-1].endswith("ok")
        and "FAILED" not in block
        and "panicked" not in block
    )


def run_suite(
    ctx: Context,
    command: list[str],
    parse_counts: Callable[[Path | str], dict[str, int]],
    counts_path: Path | None = None,
) -> int:
    actual = ctx.artifact_sha256
    if sha256(ctx.artifact) != actual:
        raise ProviderError("candidate changed before gate assertions")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
        env=os.environ.copy(),
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode
    counts_input = counts_path if counts_path is not None else completed.stdout
    counts = parse_counts(counts_input)
    if sha256(ctx.artifact) != actual:
        raise ProviderError("candidate changed during gate assertions")
    if (
        counts["collected"] != counts["executed"]
        or counts["executed"] != counts["passed"]
        or counts["passed"] <= 0
        or any(counts[key] for key in ("failed", "skipped", "excluded"))
    ):
        status = "FAIL"
    else:
        status = "PASS"
    payload = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": ctx.gate_id,
        "status": status,
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "counts": counts,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = ctx.result.with_name(ctx.result.name + ".new")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(ctx.result)
    return 0 if status == "PASS" else 1
