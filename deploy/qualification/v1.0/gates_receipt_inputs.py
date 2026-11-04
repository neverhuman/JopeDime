from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gates_receipt_contract import HEX_COMMIT, RELEASE_GATE_IDS, STATUSES

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
