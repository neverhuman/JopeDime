#!/usr/bin/env python3
"""Emit the minimal 28-gate receipt format consumed by signer_watch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

RELEASE_GATE_IDS = "M1 M2 M3 M4 M5 S1 S2 S3 S4 R1 R2 R3 R4 A1 A2 A3 A4 A5 E1 E2 E3 U1 U2 T1 T2 T3 D1 D2".split()
STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("receipt root must be an object")
    return value


def _rows(value: object) -> dict[str, dict]:
    if not isinstance(value, list):
        raise ValueError("gates must be an array")
    rows: dict[str, dict] = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != {"id", "status", "evidence"}:
            raise ValueError("each gate must contain exactly id, status and evidence")
        gate = row["id"]
        status = row["status"]
        evidence = row["evidence"]
        if not isinstance(gate, str) or gate in rows:
            raise ValueError("gate IDs must be unique strings")
        if gate not in RELEASE_GATE_IDS:
            raise ValueError("unknown or non-release gate: " + gate)
        if not isinstance(status, str) or status.upper() not in STATUSES:
            raise ValueError("unsupported status for " + gate)
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("evidence must be a non-empty string for " + gate)
        rows[gate] = {"id": gate, "status": status.upper(), "evidence": evidence}
    if set(rows) != set(RELEASE_GATE_IDS):
        missing = sorted(set(RELEASE_GATE_IDS) - set(rows))
        extra = sorted(set(rows) - set(RELEASE_GATE_IDS))
        raise ValueError(f"gate inventory mismatch; missing={missing}, extra={extra}")
    return rows


def from_gate_results(results_path: Path, artifact: Path | None) -> dict:
    """Validate explicit gate results and bind PASS results to supplied artifact bytes."""
    source = _json(results_path)
    if source.get("schema") != "jope.rel07.gate-results/v1":
        raise ValueError("unsupported gate-results schema")
    rows = _rows(source.get("gates"))

    if artifact is None:
        if any(row["status"] == "PASS" for row in rows.values()):
            raise ValueError("PASS results require --artifact byte binding")
        return {"gates": [rows[gate] for gate in RELEASE_GATE_IDS]}

    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("artifact must be a regular, non-symlink file")
    with artifact.open("rb") as stream:
        artifact_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if source.get("signed_bytes_sha256") != artifact_sha:
        raise ValueError("gate results are not bound to the supplied signed bytes")
    if any(row["status"] != "PASS" for row in rows.values()):
        raise ValueError("signed-byte receipt requires all 28 gates to PASS")
    return {
        "signed_bytes_sha256": artifact_sha,
        "artifact_sha256": artifact_sha,
        "gates": [
            {**rows[gate], "artifact_sha256": artifact_sha} for gate in RELEASE_GATE_IDS
        ],
    }


def from_cpu_prestage(receipt_path: Path) -> dict:
    """Project CPU prep as NOT_RUN for signer purposes; it is not release qualification."""
    source = _json(receipt_path)
    if source.get("schema") != "jope.qualification-cpu-prestage/v1":
        raise ValueError("input is not a CPU-prestage receipt")
    source_identity = source.get("source")
    source_sha = (
        source_identity.get("main_sha") if isinstance(source_identity, dict) else None
    )
    if not isinstance(source_sha, str) or not HEX_COMMIT.fullmatch(source_sha):
        raise ValueError("CPU-prestage receipt must pin a full main commit")
    records = source.get("release_gates")
    if not isinstance(records, dict) or set(records) != set(RELEASE_GATE_IDS + ["Q1"]):
        raise ValueError("CPU-prestage receipt must inventory all 28 gates and Q1")
    if (
        source.get("release_qualified") is not False
        or source.get("signed_candidate") is not None
    ):
        raise ValueError(
            "CPU-prestage receipt cannot claim signed release qualification"
        )
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    gates = []
    for gate in RELEASE_GATE_IDS:
        record = records[gate]
        if not isinstance(record, dict):
            raise ValueError("invalid release gate record: " + gate)
        check_status = str(record.get("status", "unknown"))
        reason = str(
            record.get("reason", "CPU-prestage is not signed-byte qualification")
        )
        evidence = (
            f"CPU-prestage only; not signed-byte qualification; source={source_sha}; "
            f"receipt_sha256={receipt_sha}; release_gate_status={check_status}; {reason}"
        )
        gates.append({"id": gate, "status": "NOT_RUN", "evidence": evidence})
    return {"gates": gates}


def _assembly_json(path: Path) -> tuple[dict, str]:
    """Read one regular receipt, preserving its exact content identity."""
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError(
            "assembly input must be a regular non-symlink file: " + str(path)
        )

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value

    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("assembly input root must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _assembly_artifact(artifact: Path) -> str:
    if (
        any(part.is_symlink() for part in (artifact, *artifact.parents))
        or not artifact.is_file()
    ):
        raise ValueError("artifact must be a regular non-symlink file")
    with artifact.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def from_scaffold(
    scaffold_path: Path, receipt_paths: list[Path], artifact: Path, source_sha: str
) -> dict:
    """Assemble existing PASS receipts; never execute the scaffold or sign bytes."""
    if not isinstance(source_sha, str) or not HEX_COMMIT.fullmatch(source_sha):
        raise ValueError("assembly requires a full lowercase source SHA")
    digest = _assembly_artifact(artifact)
    scaffold, scaffold_sha = _assembly_json(scaffold_path)
    if (
        scaffold.get("schema") != "jope.rel07.gate-command-scaffold/v1"
        or scaffold.get("qualification_handoff") != "NOT_EXECUTABLE"
        or scaffold.get("candidate_sha256")
        not in ("${JOPE_REL07_ARTIFACT_SHA256}", digest)
    ):
        raise ValueError("unsupported or mismatched command scaffold")
    commands = scaffold.get("gates")
    if not isinstance(commands, list) or len(commands) != len(RELEASE_GATE_IDS):
        raise ValueError("scaffold requires exactly 28 release gates")
    ids = []
    for row in commands:
        if not isinstance(row, dict) or set(row) != {"id", "argv", "timeout_seconds"}:
            raise ValueError("invalid scaffold command row")
        gate, argv, timeout = row["id"], row["argv"], row["timeout_seconds"]
        if not isinstance(gate, str) or gate not in RELEASE_GATE_IDS or gate in ids:
            raise ValueError("unknown or duplicate scaffold gate")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv)
            or type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("invalid scaffold argv or timeout")
        ids.append(gate)
    if len(receipt_paths) != len(RELEASE_GATE_IDS):
        raise ValueError("assembly requires exactly 28 gate receipts")
    rows = {}
    inputs = [(scaffold_path, scaffold_sha)]
    count_keys = {"collected", "executed", "passed", "failed", "skipped", "excluded"}
    for path in receipt_paths:
        result, receipt_sha = _assembly_json(path)
        gate = result.get("id")
        if not isinstance(gate, str) or gate not in ids or gate in rows:
            raise ValueError("unknown or duplicate receipt gate")
        if (
            result.get("schema") != "jope.rel07.gate-execution/v1"
            or result.get("source_sha") != source_sha
            or result.get("artifact_sha256") != digest
        ):
            raise ValueError("receipt schema/source/artifact does not match assembly")
        counts = result.get("counts")
        if (
            result.get("status") != "PASS"
            or not isinstance(counts, dict)
            or set(counts) != count_keys
            or any(type(n) is not int or n < 0 for n in counts.values())
            or not counts["collected"] == counts["executed"] == counts["passed"] > 0
            or any(counts[key] for key in ("failed", "skipped", "excluded"))
        ):
            raise ValueError(
                "receipt must PASS with positive complete execution: " + gate
            )
        rows[gate] = {
            "id": gate,
            "status": "PASS",
            "artifact_sha256": digest,
            "evidence": (
                f"receipt={path.absolute()}; receipt_sha256={receipt_sha}; "
                f"source_sha={source_sha}; scaffold={scaffold_path.absolute()}; "
                f"scaffold_sha256={scaffold_sha}"
            ),
        }
        inputs.append((path, receipt_sha))
    for path, expected in inputs:
        if _assembly_json(path)[1] != expected:
            raise ValueError("assembly input changed: " + str(path))
    if _assembly_artifact(artifact) != digest:
        raise ValueError("artifact changed during assembly")
    return {
        "artifact_sha256": digest,
        "gates": [rows[gate] for gate in RELEASE_GATE_IDS],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--gate-results", type=Path)
    inputs.add_argument("--cpu-prestage", type=Path)
    inputs.add_argument(
        "--scaffold",
        type=Path,
        help="assemble existing receipts, without executing commands",
    )
    parser.add_argument("--gate-receipt", type=Path, action="append", default=[])
    parser.add_argument(
        "--source-sha", help="exact source shared by all 28 gate receipts"
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.scaffold:
            if args.artifact is None or args.source_sha is None:
                raise ValueError("--scaffold requires --artifact and --source-sha")
            receipt = from_scaffold(
                args.scaffold, args.gate_receipt, args.artifact, args.source_sha
            )
        else:
            if args.gate_receipt or args.source_sha is not None:
                raise ValueError("--gate-receipt and --source-sha require --scaffold")
            receipt = (
                from_cpu_prestage(args.cpu_prestage)
                if args.cpu_prestage
                else from_gate_results(args.gate_results, args.artifact)
            )
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output must be a new path")
        if any(part.is_symlink() for part in args.output.parents):
            raise ValueError("output parents must not contain symlinks")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if any(part.is_symlink() for part in args.output.parents):
            raise ValueError("output parents must not contain symlinks")
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "gates": len(receipt["gates"]),
                "all_pass": all(g["status"] == "PASS" for g in receipt["gates"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
