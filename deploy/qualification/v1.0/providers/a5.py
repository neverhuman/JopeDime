#!/usr/bin/env python3
"""A5: validate candidate-bound full-path captures from three named MCP clients."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from common import ROOT, Context, ProviderError, context, sha256, verify_candidate


CLIENTS = {
    "claude_code": ("claude-code", "claude"),
    "gemini_cli": ("gemini-cli", "gemini"),
    "codex_cli": ("codex-cli", "codex"),
}
IMPLEMENTATIONS = {
    "claude_code": "claude",
    "gemini_cli": "Antigravity Gemini CLI (owner-approved mcp05_t1925_jrb02.md)",
    "codex_cli": "codex",
}
PROTOCOL_IDENTITIES = {
    "claude_code": "claude-code",
    "gemini_cli": "antigravity-client",
    "codex_cli": "codex-mcp-client",
}
TOOL_COUNTS = {
    "system_info": 1,
    "run_status": 5,
    "dataset_register": 2,
    "dataset_inspect": 1,
    "run_start": 3,
    "run_events": 1,
    "model_export": 2,
    "model_predict": 1,
}
REQUIRED_TRUE = (
    "raw_capture_hashes_verified",
    "raw_messages_match_wire",
    "all_rpc_requests_paired",
    "text_matches_structured_results",
    "transport_complete",
    "identical_run_retry_outcome",
    "events_contiguous",
    "identical_export_retry",
    "ag1_named_client_flow_pass",
)
EXIT_FIELDS = (
    "client_exit",
    "engine_exit",
    "engine_stop_exit",
    "server_exit",
    "relay_exit",
)


def check_binding(
    ctx: Context, run: Callable[..., subprocess.CompletedProcess]
) -> None:
    for args, expected in (
        (["git", "rev-parse", "HEAD"], ctx.source_sha),
        (["git", "status", "--porcelain", "--untracked-files=all"], ""),
    ):
        completed = run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode or completed.stdout.strip() != expected:
            raise ProviderError(
                "A5 requires the clean candidate-pinned source checkout"
            )
    if sha256(ctx.artifact) != ctx.artifact_sha256:
        raise ProviderError("candidate bytes changed during A5")


def evidence_manifest() -> Path:
    value = os.environ.get("JOPE_REL07_A5_MANIFEST", "")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProviderError(
            "A5 requires an absolute regular non-symlink transcript manifest"
        )
    if path.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError(
            "A5 transcript evidence must be outside the source checkout"
        )
    if path.parent != path.parent.resolve():
        raise ProviderError("A5 transcript directory must not be a symlink")
    return path


def read_manifest(path: Path, ctx: Context) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderError(
            "A5 transcript manifest is unreadable or invalid JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "jopedime.named-client-transcripts/v1"
    ):
        raise ProviderError("A5 transcript manifest schema is unsupported")
    if payload.get("source") != ctx.source_sha:
        raise ProviderError("A5 transcripts are not bound to the candidate source")
    if payload.get("candidate_artifact_sha256") != ctx.artifact_sha256:
        raise ProviderError("A5 transcripts are not bound to the candidate bytes")
    summary = payload.get("a5")
    if (
        not isinstance(summary, dict)
        or summary.get("full_path_observed_clients") != len(CLIENTS)
        or summary.get("successful_tool_calls") != 48
        or summary.get("failed_tool_calls") != 0
    ):
        raise ProviderError("A5 manifest does not report three complete client flows")
    clients = payload.get("clients")
    if not isinstance(clients, dict) or set(clients) != set(CLIENTS):
        raise ProviderError("A5 requires exactly the three named clients")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ProviderError("A5 manifest has no transcript sidecar inventory")
    for name, metadata in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"", ".", ".."}
            or not isinstance(metadata, dict)
        ):
            raise ProviderError("A5 manifest contains an unsafe sidecar entry")
        expected_sha = metadata.get("sha256")
        expected_size = metadata.get("bytes")
        if (
            not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
            or type(expected_size) is not int
            or expected_size < 0
        ):
            raise ProviderError(f"A5 sidecar metadata is invalid: {name}")
        sidecar = path.parent / name
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ProviderError(f"A5 sidecar is missing or not a regular file: {name}")
        if expected_size != sidecar.stat().st_size or expected_sha != sha256(sidecar):
            raise ProviderError(f"A5 sidecar identity mismatch: {name}")
    return payload, sha256(path)


def check_client_receipts(payload: dict, ctx: Context) -> dict[str, dict]:
    results = {}
    dataset_identity = None
    run_ids = set()
    for key, (slug, client_name) in CLIENTS.items():
        client = payload["clients"][key]
        if (
            not isinstance(client, dict)
            or not isinstance(client.get("version"), str)
            or not client["version"].strip()
            or client.get("implementation") != IMPLEMENTATIONS[key]
        ):
            raise ProviderError(f"A5 client version is missing: {key}")
        executable_sha = client.get("executable_sha256")
        if (
            not isinstance(executable_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", executable_sha) is None
        ):
            raise ProviderError(f"A5 client executable identity is missing: {key}")
        execution_name = f"{slug}.execution.json"
        if execution_name not in payload["files"]:
            raise ProviderError(f"A5 execution receipt is not in the manifest: {key}")
        execution_path = (
            Path(os.environ["JOPE_REL07_A5_MANIFEST"]).parent / execution_name
        )
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderError(
                f"A5 execution receipt is missing or invalid: {key}"
            ) from error
        if (
            not isinstance(execution, dict)
            or execution.get("source") != ctx.source_sha
            or execution.get("binary_sha256") != ctx.artifact_sha256
            or execution.get("client") != key
            or execution.get("client_version") != client["version"]
            or execution.get("client_resolved_path") != client.get("executable")
            or execution.get("client_launcher_sha256") != executable_sha
        ):
            raise ProviderError(f"A5 process receipt identity mismatch: {key}")
        receipt_name = f"{slug}.verification.json"
        if receipt_name not in payload["files"]:
            raise ProviderError(f"A5 semantic receipt is not in the manifest: {key}")
        receipt_path = Path(os.environ["JOPE_REL07_A5_MANIFEST"]).parent / receipt_name
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderError(
                f"A5 semantic receipt is missing or invalid: {key}"
            ) from error
        if not isinstance(receipt, dict):
            raise ProviderError(f"A5 semantic receipt is not an object: {key}")
        lifecycle = receipt.get("lifecycle_identity")
        protocol_identity = None
        if isinstance(lifecycle, list) and lifecycle and isinstance(lifecycle[0], dict):
            first = lifecycle[0]
            client_info = first.get("clientInfo")
            if isinstance(client_info, dict):
                protocol_identity = client_info.get("name")
            if protocol_identity is None:
                metadata = first.get("_meta", {})
                if isinstance(metadata, dict):
                    client_info = metadata.get("io.modelcontextprotocol/clientInfo", {})
                    if isinstance(client_info, dict):
                        protocol_identity = client_info.get("name")
        if (
            receipt.get("schema") != "jope.mcp05-a5-client-review/v1"
            or receipt.get("source") != ctx.source_sha
            or receipt.get("binary_sha256") != ctx.artifact_sha256
            or receipt.get("client") != client_name
            or protocol_identity != PROTOCOL_IDENTITIES[key]
            or receipt.get("client_version") != client["version"]
            or receipt.get("client_sha256") != executable_sha
        ):
            raise ProviderError(f"A5 semantic receipt identity mismatch: {key}")
        if any(receipt.get(field) is not True for field in REQUIRED_TRUE):
            raise ProviderError(f"A5 full-path controls are incomplete: {key}")
        if any(
            type(receipt.get(field)) is not int or receipt[field] != 0
            for field in EXIT_FIELDS
        ):
            raise ProviderError(f"A5 client/server process did not exit cleanly: {key}")
        if (
            receipt.get("tool_calls") != 16
            or receipt.get("successful_calls") != 16
            or receipt.get("failed_calls") != 0
            or receipt.get("tool_counts") != TOOL_COUNTS
            or receipt.get("terminal_state") != "completed"
            or receipt.get("evaluations_used") != "1"
            or receipt.get("event_count") != 6
            or not isinstance(receipt.get("run_retry_contract"), dict)
            or receipt["run_retry_contract"].get("successful_attempts") != 2
            or receipt["run_retry_contract"].get("inventory_counts") != [0, 1, 1]
        ):
            raise ProviderError(f"A5 flow or exact retry evidence is incomplete: {key}")
        run_id = receipt["run_retry_contract"].get("run_id")
        if (
            not isinstance(run_id, str)
            or re.fullmatch(r"run_[0-9A-HJKMNP-TV-Z]{26}", run_id) is None
            or run_id in run_ids
        ):
            raise ProviderError(f"A5 run identity is missing or reused: {key}")
        run_ids.add(run_id)
        placement = receipt.get("plan_placement")
        if (
            not isinstance(placement, dict)
            or placement.get("evaluation") != "cpu"
            or placement.get("workers") != 1
            or placement.get("dope") != "none"
        ):
            raise ProviderError(f"A5 run exceeded the bounded CPU plan: {key}")
        rows, columns = receipt.get("dataset_rows"), receipt.get("dataset_columns")
        if (
            type(rows) is not int
            or rows <= 0
            or type(columns) is not int
            or columns <= 0
        ):
            raise ProviderError(f"A5 TRAIN dataset identity is incomplete: {key}")
        identity = (receipt.get("dataset_id"), rows, columns)
        if not identity[0] or (
            dataset_identity is not None and identity != dataset_identity
        ):
            raise ProviderError("A5 clients did not use the same TRAIN dataset")
        dataset_identity = identity
        predictions = receipt.get("predictions")
        prediction_columns = (
            predictions.get("columns") if isinstance(predictions, dict) else None
        )
        if (
            not isinstance(predictions, dict)
            or type(predictions.get("rows")) is not int
            or predictions.get("rows") != rows
            or not isinstance(prediction_columns, list)
            or "prediction" not in prediction_columns
            or re.fullmatch(r"[0-9a-f]{64}", str(predictions.get("sha256", ""))) is None
        ):
            raise ProviderError(f"A5 model prediction evidence is incomplete: {key}")
        materialized = receipt.get("materialized_files")
        if not isinstance(materialized, dict) or not {
            "manifest.json",
            "src/rust/predictor.rs",
        }.issubset(materialized):
            raise ProviderError(
                f"A5 export materialization evidence is incomplete: {key}"
            )
        for name, identity in materialized.items():
            if (
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or not isinstance(identity, dict)
                or type(identity.get("bytes")) is not int
                or identity["bytes"] < 0
                or not isinstance(identity.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
            ):
                raise ProviderError(
                    f"A5 export materialization identity is invalid: {key}"
                )
        export = receipt.get("export")
        if (
            not isinstance(export, dict)
            or export.get("kind") != "bundle"
            or export.get("retains_training_data") is not False
            or not isinstance(export.get("languages"), list)
            or not any(
                isinstance(row, dict)
                and row.get("language") == "rust"
                and row.get("status") == "exported"
                for row in export["languages"]
            )
        ):
            raise ProviderError(f"A5 bundle export evidence is incomplete: {key}")
        results[key] = {
            "version": client["version"],
            "tool_calls": 16,
            "run_id": run_id,
            "inventory_counts": [0, 1, 1],
        }
    return results


def qualify(
    ctx: Context, run: Callable[..., subprocess.CompletedProcess] = subprocess.run
) -> int:
    if ctx.gate_id != "A5":
        raise ProviderError("A5 provider requires gate A5")
    if ctx.result.exists() or ctx.result.is_symlink():
        raise ProviderError("A5 result must be a new regular file")
    if not ctx.result.parent.is_dir() or ctx.result.parent.is_symlink():
        raise ProviderError("A5 result parent must be an existing regular directory")
    if ctx.result.resolve().is_relative_to(ROOT.resolve()):
        raise ProviderError("A5 result evidence must be outside the source checkout")
    manifest_path = evidence_manifest()
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    payload, manifest_sha = read_manifest(manifest_path, ctx)
    trusted_sha = os.environ.get("JOPE_REL07_A5_MANIFEST_SHA256", "")
    if (
        re.fullmatch(r"[0-9a-f]{64}", trusted_sha) is None
        or trusted_sha != manifest_sha
    ):
        raise ProviderError("A5 manifest does not match the trusted manifest SHA-256")
    clients = check_client_receipts(payload, ctx)
    verifier = (
        ROOT / "deploy/qualification/agents/transcripts/run-prep/verify_capture.py"
    )
    args = [
        sys.executable,
        "-B",
        str(verifier),
        str(manifest_path),
        "--check-run-retry",
        "--manifest-sha256",
        manifest_sha,
    ]
    completed = run(
        args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=300
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode:
        raise ProviderError("A5 raw transcript and retry verification failed")
    try:
        capture = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ProviderError("A5 transcript verifier did not emit JSON") from error
    if (
        not isinstance(capture, dict)
        or capture.get("integrity") != "PASS"
        or capture.get("sidecars_verified") != len(payload["files"])
        or not isinstance(capture.get("run_retry_contract"), dict)
        or set(capture["run_retry_contract"]) != set(CLIENTS)
        or any(
            capture["run_retry_contract"][key]
            != {
                "run_id": clients[key]["run_id"],
                "successful_attempts": 2,
                "inventory_counts": [0, 1, 1],
            }
            for key in CLIENTS
        )
    ):
        raise ProviderError("A5 raw verifier did not confirm all three client captures")
    check_binding(ctx, run)
    final_payload, final_manifest_sha = read_manifest(manifest_path, ctx)
    if final_manifest_sha != manifest_sha:
        raise ProviderError("A5 transcript manifest changed during verification")
    clients = check_client_receipts(final_payload, ctx)
    result = {
        "schema": "jope.rel07.gate-execution/v1",
        "id": "A5",
        "status": "PASS",
        "source_sha": ctx.source_sha,
        "artifact_sha256": ctx.artifact_sha256,
        "evidence_manifest_sha256": manifest_sha,
        "counts": {key: 3 for key in ("collected", "executed", "passed")}
        | {key: 0 for key in ("failed", "skipped", "excluded")},
        "successful_tool_calls": 48,
        "clients": clients,
    }
    ctx.result.parent.mkdir(parents=True, exist_ok=True)
    with ctx.result.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


def main() -> int:
    try:
        return qualify(context("A5"))
    except (OSError, subprocess.TimeoutExpired, ProviderError) as error:
        print(f"A5 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
