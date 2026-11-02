from __future__ import annotations

import hashlib
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path

from gates_receipt_contract import (
    HEX_COMMIT,
    HEX_SHA256,
    M3_WAIVER_PROPOSER_SHA256,
    M3_WAIVER_RULING_SHA256,
    RELEASE_GATE_IDS,
)

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


def _source_proposer_sha256(source_sha: str) -> str:
    try:
        raw = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[3]),
                "show",
                source_sha + ":crates/jope-service/src/discover/propose.rs",
            ],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("waiver requires the exact source proposer blob") from error
    return hashlib.sha256(raw).hexdigest()


def _within_operator_mix_limit(
    difference: int, native_n: int, reference_n: int, limit_pp: int
) -> bool:
    """Compare count-derived percentage points without rounding or float drift."""
    return abs(difference) * 100 <= limit_pp * native_n * reference_n


def _m3_waiver(result: dict, ruling: Path, release_version: str) -> dict:
    if release_version != "v1.0.0":
        raise ValueError("M3 waiver expires at v1.0.1; v1.0.0 only")
    if _assembly_artifact(ruling) != M3_WAIVER_RULING_SHA256:
        raise ValueError("M3 waiver owner ruling hash differs")
    if _source_proposer_sha256(result["source_sha"]) != M3_WAIVER_PROPOSER_SHA256:
        raise ValueError("M3 waiver requires the owner-scoped proposer bytes")
    if result.get("status") != "FAIL":
        raise ValueError("M3 waiver requires an explicitly retained raw FAIL")

    # Reuse the maintained full-census/count/pin/tolerance validator. It executes
    # no measurements, and rejects contradictory raw clause flags and counts.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "providers"))
    try:
        from common import Context
        from m3_full import gate_payload
    finally:
        sys.path.pop(0)
    report = dict(
        result, schema="jope.rel07.m3-full-measurement/v1", status="NUMERICAL_FAIL"
    )
    ctx = Context(
        "M3",
        Path("/unused"),
        result["artifact_sha256"],
        result["source_sha"],
        Path("/unused"),
    )
    try:
        validated = gate_payload(ctx, report, result["inputs_sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError("complete raw M3 measurement required for waiver") from error
    clauses = validated["clauses"]
    if validated["counts"] != dict(
        collected=4, executed=4, passed=3, failed=1, skipped=0, excluded=0
    ) or any(
        clauses[key]["passed"] is not True
        for key in ("feat412_proj16", "training_landmarks", "full_train_cv")
    ):
        raise ValueError("M3 waiver covers operator mix only; other clauses must PASS")
    views = []
    for row in clauses["r0_proposer"]["strata"]:
        na, nb = row["reference"], row["native"]
        deltas = {}
        for op, reference in row["reference_counts"].items():
            difference = row["native_counts"][op] * na - reference * nb
            limit = 3 if op == "edit_replace_head" else 2
            if not _within_operator_mix_limit(difference, nb, na, limit):
                raise ValueError("M3 waiver guard exceeded: " + op)
            deltas[op] = 100 * difference / (na * nb)
        views.append({"mode": row["mode"], "deltas_pp": deltas})
    return {
        "id": "M3-WAIVER-V1.0.0-T1955",
        "gate": "M3",
        "disposition": "WAIVED_V1_0_0",
        "expires_at": "v1.0.1",
        "ruling": str(ruling.absolute()),
        "ruling_sha256": M3_WAIVER_RULING_SHA256,
        "source_sha": result["source_sha"],
        "artifact_sha256": result["artifact_sha256"],
        "raw_status": "FAIL",
        "guard_views": views,
    }


def _assembly_controller():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        return runpy.run_path(str(Path(__file__).with_name("release-gates")))
    finally:
        sys.path.pop(0)


def _assembly_candidate(artifact: Path, manifest: Path, source_sha: str):
    """Reuse the maintained controller's same-candidate full/fused admission."""
    value, expected = _assembly_json(manifest)
    controller = _assembly_controller()
    candidate = controller["_candidate_record"](artifact, manifest)
    if (
        candidate["source_sha"] != source_sha
        or candidate["manifest_sha256"] != expected
    ):
        raise ValueError("candidate manifest source/content differs from assembly")
    if candidate["d2_artifact"] is None:
        raise ValueError(candidate["d2_blocked_reason"])
    inputs = [(manifest, expected)]
    for record, identity in [
        (candidate, "full-linux-x86_64"),
        (candidate["d2_artifact"], "fused-cpu-linux-x86_64"),
    ]:
        row = next(
            item
            for item in value["artifacts"]
            if isinstance(item, dict) and item.get("id") == identity
        )
        if row.get("seal_verify_receipt_sha256") != record["seal_receipt_sha256"]:
            raise ValueError("candidate manifest seal receipt pin differs: " + identity)
        path = Path(record["seal_receipt_path"])
        if _assembly_json(path)[1] != record["seal_receipt_sha256"]:
            raise ValueError("candidate seal receipt changed during admission")
        inputs.append((path, record["seal_receipt_sha256"]))
    return candidate, controller["verify_candidate_variant"], inputs


def _controller_provenance(run_dir, provider_inputs, receipt_paths, source, digest):
    """Bind metadata to one completed run without executing or promoting it."""
    execution_path = run_dir / "execution.json"
    execution, execution_sha = _assembly_json(execution_path)
    bindings, inputs_sha = _assembly_json(provider_inputs)
    controller = _assembly_controller()
    admitted = controller["provider_inputs"](provider_inputs)
    if set(admitted) != set(controller["INPUT_FIELDS"]):
        raise ValueError("controller provenance requires all provider input bindings")
    if (
        execution.get("schema") != "jope.rel07.gate-run/v1"
        or execution.get("mode") != "execute"
        or execution.get("source_sha") != source
        or execution.get("artifact_sha256") != digest
        or execution.get("candidate_inputs_sha256") != inputs_sha
        or execution.get("interrupted") is not False
        or any(
            not isinstance(execution.get(key), str) or not execution[key].strip()
            for key in ("machine", "started_utc", "finished_utc")
        )
        or any(
            not isinstance(execution.get(key), str)
            or not HEX_SHA256.fullmatch(execution[key])
            for key in ("runner_sha256", "manifest_sha256")
        )
    ):
        raise ValueError("controller execution identity or completion does not match")
    expected = {run_dir.absolute() / gate / "result.json" for gate in RELEASE_GATE_IDS}
    if len(receipt_paths) != 28 or {p.absolute() for p in receipt_paths} != expected:
        raise ValueError("all receipts must belong to the same controller run")
    checks = execution.get("checks")
    if not isinstance(checks, list) or len(checks) != 28:
        raise ValueError("controller execution requires exactly 28 command records")
    seen = set()
    outputs = []
    pins = [(execution_path, execution_sha), (provider_inputs, inputs_sha)]
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("invalid controller command record")
        gate, argv = check.get("id"), check.get("argv")
        if not isinstance(gate, str) or gate not in RELEASE_GATE_IDS or gate in seen:
            raise ValueError("unknown or duplicate controller command")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv)
            or type(check.get("exit_code")) is not int
            or check["exit_code"] not in ((0, 1) if gate == "M3" else (0,))
            or check.get("timed_out") is not False
            or any(
                not isinstance(check.get(key), str) or not check[key].strip()
                for key in ("started_utc", "finished_utc")
            )
        ):
            raise ValueError("controller command did not complete: " + gate)
        seen.add(gate)
        result = run_dir / gate / "result.json"
        log = run_dir / gate / "command.log"
        raw, result_sha = _assembly_json(result)
        log_sha = _assembly_artifact(log)
        # The frozen controller may reject the raw M3 FAIL before pinning it.
        # Its result still has to satisfy every separate waiver guard below.
        if (
            check.get("log_sha256") != log_sha
            or (check["exit_code"] != 0 and raw.get("status") != "FAIL")
            or (
                check.get("result_sha256") != result_sha
                and not (
                    gate == "M3"
                    and raw.get("status") == "FAIL"
                    and "result_sha256" not in check
                )
            )
        ):
            raise ValueError("controller output hash differs: " + gate)
        outputs.append(
            dict(
                id=gate,
                result=str(result.absolute()),
                result_sha256=result_sha,
                log=str(log.absolute()),
                log_sha256=log_sha,
            )
        )
        pins.extend([(result, result_sha), (log, log_sha)])
    return {
        "schema": "jope.rel07.controller-provenance/v1",
        "execution": {"path": str(execution_path.absolute()), "sha256": execution_sha},
        "inputs": {
            "path": str(provider_inputs.absolute()),
            "sha256": inputs_sha,
            "bindings": bindings,
        },
        "machine": execution["machine"],
        "started_utc": execution["started_utc"],
        "finished_utc": execution["finished_utc"],
        "runner_sha256": execution["runner_sha256"],
        "manifest_sha256": execution["manifest_sha256"],
        "commands": checks,
        "outputs": outputs,
    }, pins


def from_scaffold(
    scaffold_path: Path,
    receipt_paths: list[Path],
    artifact: Path,
    source_sha: str,
    *,
    m3_waiver_ruling: Path | None = None,
    release_version: str | None = None,
    candidate_manifest: Path | None = None,
    controller_run: Path | None = None,
    controller_inputs: Path | None = None,
) -> dict:
    """Assemble existing PASS receipts; never execute the scaffold or sign bytes."""
    if not isinstance(source_sha, str) or not HEX_COMMIT.fullmatch(source_sha):
        raise ValueError("assembly requires a full lowercase source SHA")
    if (m3_waiver_ruling is None) != (release_version is None):
        raise ValueError("M3 waiver requires both ruling and explicit release version")
    digest = _assembly_artifact(artifact)
    if (controller_run is None) != (controller_inputs is None):
        raise ValueError("controller provenance requires both run and inputs")
    provenance, provenance_pins = None, []
    if controller_run is not None:
        provenance, provenance_pins = _controller_provenance(
            controller_run, controller_inputs, receipt_paths, source_sha, digest
        )
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
    waiver = None
    inputs = [(scaffold_path, scaffold_sha)]
    candidate = None
    if candidate_manifest is not None:
        candidate, verify_variant, bindings = _assembly_candidate(
            artifact, candidate_manifest, source_sha
        )
        inputs.extend(bindings)
    count_keys = {"collected", "executed", "passed", "failed", "skipped", "excluded"}
    for path in receipt_paths:
        result, receipt_sha = _assembly_json(path)
        gate = result.get("id")
        if not isinstance(gate, str) or gate not in ids or gate in rows:
            raise ValueError("unknown or duplicate receipt gate")
        gate_digest = (
            candidate["d2_artifact"]["sha256"]
            if gate == "D2" and candidate is not None
            else digest
        )
        if (
            result.get("schema") != "jope.rel07.gate-execution/v1"
            or result.get("source_sha") != source_sha
            or result.get("artifact_sha256") != gate_digest
        ):
            raise ValueError("receipt schema/source/artifact does not match assembly")
        counts = result.get("counts")
        waived = gate == "M3" and m3_waiver_ruling is not None
        if waived:
            waiver = _m3_waiver(result, m3_waiver_ruling, release_version)
        elif (
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
            "status": result["status"],
            "artifact_sha256": gate_digest,
            "evidence": (
                f"receipt={path.absolute()}; receipt_sha256={receipt_sha}; "
                f"source_sha={source_sha}; scaffold={scaffold_path.absolute()}; "
                f"scaffold_sha256={scaffold_sha}"
            ),
        }
        if waived:
            rows[gate].update(disposition="WAIVED_V1_0_0", counts=counts)
        inputs.append((path, receipt_sha))
    for path, expected in inputs:
        if _assembly_json(path)[1] != expected:
            raise ValueError("assembly input changed: " + str(path))
    if _assembly_artifact(artifact) != digest:
        raise ValueError("artifact changed during assembly")
    if candidate is not None:
        verify_variant(candidate, source_sha)
        verify_variant(candidate["d2_artifact"], source_sha)
    if waiver and _assembly_artifact(m3_waiver_ruling) != waiver["ruling_sha256"]:
        raise ValueError("M3 waiver ruling changed during assembly")
    assembled = {
        "artifact_sha256": digest,
        "gates": [rows[gate] for gate in RELEASE_GATE_IDS],
    }
    if candidate is not None:
        d2 = candidate["d2_artifact"]
        assembled["gate_artifacts"] = {
            "D2": {
                "path": d2["path"],
                "sha256": d2["sha256"],
                "source_sha": source_sha,
                "candidate_manifest_sha256": d2["manifest_sha256"],
                "seal_receipt_sha256": d2["seal_receipt_sha256"],
            }
        }
    if waiver:
        assembled.update(release_version=release_version, waivers=[waiver])
    for path, expected in provenance_pins:
        if _assembly_artifact(path) != expected:
            raise ValueError(
                "controller provenance changed during assembly: " + str(path)
            )
    if provenance is not None:
        assembled["provenance"] = provenance
    return assembled
