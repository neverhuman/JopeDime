#!/usr/bin/env python3
"""REL-07 A5: verify three reviewed named-client captures at the release bytes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

from a4 import check_binding, controlled_runner, new_result, publish
from common import ROOT, Context, ProviderError, context, sha256, verify_candidate


PREP = ROOT / "deploy/qualification/agents/transcripts/run-prep"
SPEC = importlib.util.spec_from_file_location(
    "rel07_a5_capture", PREP / "verify_capture.py"
)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)
CLIENTS = {
    "claude_code": "claude-code",
    "gemini_cli": "gemini-cli",
    "codex_cli": "codex-cli",
}
CLIENT_NAMES = {
    "claude_code": {"claude-code"},
    "gemini_cli": {"gemini-cli", "gemini", "antigravity-client"},
    "codex_cli": {"codex-mcp-client", "codex"},
}
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderError(message)


def read(path: Path) -> bytes:
    require(
        path.is_absolute() and not path.is_symlink() and path.is_file(),
        "A5 evidence must be an absolute regular non-symlink file",
    )
    require(path.stat().st_size <= MAX_FILE, "A5 sidecar exceeds the byte limit")
    return path.read_bytes()


def packet(manifest: Path, digest: str, ctx: Context) -> tuple[dict, dict]:
    require(
        capture._SHA256.fullmatch(digest) is not None,
        "A5 needs an explicit reviewed manifest SHA-256",
    )
    raw = read(manifest)
    require(hashlib.sha256(raw).hexdigest() == digest, "A5 manifest SHA-256 mismatch")
    value = capture.decode_json(raw)
    require(isinstance(value, dict), "A5 manifest must be an object")
    require(value.get("source") == ctx.source_sha, "A5 capture source is stale")
    files = value.get("files")
    require(isinstance(files, dict) and 0 < len(files) <= 256, "A5 sidecars missing")
    required = {"source-and-input.json"} | {
        f"{slug}.{suffix}"
        for slug in CLIENTS.values()
        for suffix in (
            "execution.json",
            "capture.json",
            "command.json",
            "mcp.jsonl",
            "client.raw",
            "server.raw",
            "server.stderr",
        )
    }
    require(required <= files.keys(), "A5 consumed sidecars must all be hash-pinned")
    total = 0
    for name, entry in files.items():
        require(
            isinstance(name, str)
            and Path(name).name == name
            and name not in ("", ".", ".."),
            "unsafe A5 sidecar name",
        )
        require(isinstance(entry, dict), "invalid A5 sidecar entry")
        size = entry.get("bytes")
        require(type(size) is int and 0 <= size <= MAX_FILE, "invalid A5 sidecar size")
        total += size
        path = manifest.parent / name
        require(
            path.is_absolute()
            and not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == size,
            "A5 sidecar is not the declared regular file",
        )
    require(total <= MAX_TOTAL, "A5 packet exceeds the total byte limit")
    verified = capture.verify(
        manifest,
        manifest.parent,
        check_run_retry=True,
        manifest_sha256=digest,
    )
    return value, verified


def client_path(root: Path, slug: str, suffix: str) -> Path:
    return root / f"{slug}.{suffix}"


def full_path(raw: bytes, retry: dict, train_hash: str, train_rows: int) -> None:
    calls = sorted(capture.tool_calls(raw), key=lambda call: call["start"])
    require(all(isinstance(c["data"], dict) for c in calls), "A5 tool failure")
    groups = {}
    for call in calls:
        groups.setdefault(call["name"], []).append(call)
    required = {
        "system_info",
        "dataset_register",
        "dataset_inspect",
        "run_start",
        "run_status",
        "run_events",
        "model_export",
        "model_predict",
    }
    require(required <= groups.keys(), "A5 full golden path is incomplete")
    dataset = "ds_" + train_hash[:24]
    registered = groups["dataset_register"][0]
    require(
        registered["data"].get("dataset_id") == dataset, "A5 input identity mismatch"
    )
    require(
        all(
            c["arguments"].get("dataset_id") == dataset
            and c["data"].get("dataset_id") == dataset
            and c["data"].get("rows") == str(train_rows)
            for c in groups["dataset_inspect"]
        ),
        "A5 dataset inspection does not match the approved TRAIN input",
    )
    plans = [c for c in groups["run_start"] if c["arguments"].get("dry_run") is True]
    starts = [
        c for c in groups["run_start"] if c["arguments"].get("dry_run") is not True
    ]
    require(plans and len(starts) == 2, "A5 plan/create/retry path is missing")
    first, repeated = starts
    require(registered["end"] < first["start"], "A5 run preceded registration")
    require(
        any(c["end"] < first["start"] for c in groups["dataset_inspect"]),
        "A5 inspect ordering",
    )
    require(
        all(c["arguments"].get("dataset") == dataset for c in plans + starts),
        "A5 run input mismatch",
    )
    require(first["data"] == repeated["data"], "A5 run retry outcome changed")
    require(
        any(
            c["end"] < first["start"]
            and c["data"].get("plan_digest") == first["arguments"].get("plan_digest")
            and isinstance(c["data"].get("plan_digest"), str)
            for c in plans
        ),
        "A5 create does not use its returned plan digest",
    )
    run_id = retry["run_id"]
    statuses = [
        c for c in groups["run_status"] if c["arguments"].get("run_id") == run_id
    ]
    require(
        statuses and statuses[-1]["data"].get("state") == "completed",
        "A5 run did not complete",
    )
    terminal = statuses[-1]
    require(
        terminal["data"].get("run_id") == run_id, "A5 snapshot refers to another run"
    )
    evaluations = terminal["data"].get("budget_used", {}).get("evaluations")
    require(
        isinstance(evaluations, str)
        and evaluations.isdecimal()
        and int(evaluations) > 0,
        "A5 requires an actual completed evaluation",
    )
    events = []
    for call in groups["run_events"]:
        require(
            call["arguments"].get("run_id") == run_id, "A5 events refer to another run"
        )
        page = call["data"].get("events")
        require(isinstance(page, list), "A5 missing event page")
        events.extend(page)
    require(
        events
        and [e.get("sequence") for e in events]
        == [str(i + 1) for i in range(len(events))]
        and all(e.get("run_id") == run_id for e in events)
        and any(e.get("kind") == "experiment.finished" for e in events)
        and events[-1].get("kind") == "run.finished"
        and terminal["data"].get("event_watermark") == str(len(events)),
        "A5 events are incomplete, duplicated or out of order",
    )
    exports = groups["model_export"]
    require(
        all(
            c["arguments"].get("run_id") == run_id
            and c["data"].get("kind") == "bundle"
            and isinstance(c["data"].get("languages"), list)
            and any(x.get("status") == "exported" for x in c["data"]["languages"])
            and terminal["end"] < c["start"]
            for c in exports
        ),
        "A5 completed-model export is missing",
    )
    artifacts = {c["data"].get("artifact_id") for c in exports}
    artifacts.add(terminal["data"].get("best", {}).get("model_artifact"))
    artifacts.discard(None)
    require(bool(artifacts), "A5 export identity missing")
    for call in groups["model_predict"]:
        args, result = call["arguments"], call["data"]
        model = args.get("model", {})
        require(
            (model.get("run_id") == run_id or model.get("artifact_id") in artifacts)
            and args.get("input", {}).get("dataset_id") == dataset
            and result.get("rows") == str(train_rows)
            and isinstance(result.get("sha256"), str)
            and capture._SHA256.fullmatch(result["sha256"]) is not None
            and exports[0]["end"] < call["start"],
            "A5 prediction is missing or uses another input/model",
        )


def qualify(ctx: Context, manifest: Path, digest: str, run=None) -> int:
    run = controlled_runner(subprocess.run if run is None else run)
    require(ctx.gate_id == "A5", "provider requires gate A5")
    new_result(ctx)
    check_binding(ctx, run)
    verify_candidate(ctx, run=run)
    value, verified = packet(manifest, digest, ctx)
    root = manifest.parent
    identities = capture.decode_json(read(root / "source-and-input.json"))
    require(isinstance(identities, dict), "A5 source identity must be an object")
    require(
        identities.get("runtime_source") == ctx.source_sha
        and identities.get("binary_sha256") == ctx.artifact_sha256
        and identities.get("version", {}).get("source_sha40") == ctx.source_sha
        and identities["version"].get("features", {}).get("dev_keys") is False
        and identities["version"].get("trusts_development_key") is False,
        "A5 captures must execute these exact production candidate bytes",
    )
    inputs = identities.get("input", {})
    train_hash, rows = inputs.get("header_train_sha256"), inputs.get("rows")
    require(
        isinstance(train_hash, str)
        and capture._SHA256.fullmatch(train_hash) is not None
        and type(rows) is int
        and rows > 0
        and inputs.get("external_test_dataset") is None
        and inputs.get("sealed_or_external_holdout_read") is False,
        "A5 needs the approved TRAIN-only input identity",
    )
    versions = {}
    seen_runs, seen_launchers = set(), set()
    for client, slug in CLIENTS.items():
        record = value["clients"][client]
        execution = capture.decode_json(read(client_path(root, slug, "execution.json")))
        recorded = capture.decode_json(read(client_path(root, slug, "capture.json")))
        command = capture.decode_json(read(client_path(root, slug, "command.json")))
        require(
            isinstance(execution, dict) and isinstance(recorded, dict),
            "A5 missing execution/capture objects",
        )
        require(
            isinstance(command, list)
            and command
            and all(isinstance(a, str) for a in command),
            "A5 client command missing",
        )
        require(
            execution.get("source") == ctx.source_sha
            and execution.get("binary_sha256") == ctx.artifact_sha256
            and execution.get("client_version") == record["version"]
            and re.search(r"\d+\.\d+\.\d+", record["version"]) is not None
            and isinstance(record.get("executable_sha256"), str)
            and capture._SHA256.fullmatch(record["executable_sha256"]) is not None
            and execution.get("client_launcher_sha256") == record["executable_sha256"]
            and execution.get("fixture_sha256") == train_hash
            and all(
                type(execution.get(k)) is int and execution[k] == 0
                for k in ["client_exit", "engine_exit", "engine_stop_exit"]
            )
            and execution.get("forced_engine_signal") is False
            and execution.get("binary_unchanged") is True,
            f"A5 unsuccessful or stale execution: {client}",
        )
        require(
            recorded.get("schema") == "jopedime.mcp-stdio-capture/v1"
            and recorded.get("complete") is True
            and recorded.get("error") is None
            and recorded.get("executable_sha256") == ctx.artifact_sha256
            and recorded.get("producer_sha256") == sha256(PREP / "capture_stdio.py")
            and recorded.get("verifier_sha256") == sha256(PREP / "verify_capture.py")
            and all(
                type(recorded.get(k)) is int and recorded[k] == 0
                for k in ["server_exit_code", "relay_exit_code"]
            ),
            f"A5 incomplete or unbound transport: {client}",
        )
        wire = read(client_path(root, slug, "mcp.jsonl"))
        frames = [capture.decode_json(line) for line in wire.splitlines()]
        infos = []
        for frame in frames:
            message = frame["message"]
            if frame["direction"] == "client_to_server" and message.get("method") in (
                "initialize",
                "server/discover",
            ):
                params = message.get("params", {})
                infos.append(
                    params.get("clientInfo")
                    or params.get("_meta", {}).get(
                        "io.modelcontextprotocol/clientInfo", {}
                    )
                )
        require(
            any(info.get("name") in CLIENT_NAMES[client] for info in infos),
            "A5 named-client lifecycle identity missing",
        )
        for name, suffix, direction in (
            ("wire.jsonl", "mcp.jsonl", None),
            ("client.raw", "client.raw", "client_to_server"),
            ("server.raw", "server.raw", "server_to_client"),
            ("server.stderr", "server.stderr", None),
        ):
            raw = read(client_path(root, slug, suffix))
            require(
                recorded.get("files", {}).get(name)
                == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                "A5 raw capture hash mismatch",
            )
            if direction:
                require(
                    [capture.decode_json(x) for x in raw.splitlines()]
                    == [f["message"] for f in frames if f["direction"] == direction],
                    "A5 raw messages differ from recorded frames",
                )
        retry = verified["run_retry_contract"][client]
        require(
            retry["inventory_counts"] == [0, 1, 1],
            "A5 needs isolated ordered run inventories",
        )
        require(
            retry["run_id"] not in seen_runs
            and record["executable_sha256"] not in seen_launchers,
            "A5 clients cannot reuse one run or client executable identity",
        )
        seen_runs.add(retry["run_id"])
        seen_launchers.add(record["executable_sha256"])
        full_path(wire, retry, train_hash, rows)
        versions[client] = record["version"]
        check_binding(ctx, run)
    # Reverify the pinned inventory after semantic checks, before emitting PASS.
    packet(manifest, digest, ctx)
    return publish(ctx, len(CLIENTS), run, manifest_sha256=digest, clients=versions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        return qualify(context("A5"), args.manifest, args.manifest_sha256)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"A5 provider refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
