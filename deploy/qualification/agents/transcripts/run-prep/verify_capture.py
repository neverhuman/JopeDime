#!/usr/bin/env python3
"""Verify MCP-05 capture integrity and optional run-retry evidence; never qualify A5."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"run_[0-7][0-9A-HJKMNP-TV-Z]{25}\Z")
_CLIENTS = {
    "claude_code": "claude-code",
    "gemini_cli": "gemini-cli",
    "codex_cli": "codex-cli",
}


class CaptureError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def decode_json(raw: bytes) -> object:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value

    def constant(_):
        raise CaptureError("nonfinite JSON constant")

    def number(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON number")
        return value

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=constant,
            parse_float=number,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise CaptureError("invalid JSON or UTF-8") from exc


def tool_calls(raw: bytes) -> list[dict]:
    """Pair lifecycle and tool messages; IDs retain their JSON string/integer type."""
    pending = {}
    calls = []
    for position, line in enumerate(raw.splitlines()):
        frame = decode_json(line)
        require(isinstance(frame, dict), "invalid wire frame")
        direction, message = frame.get("direction"), frame.get("message")
        require(
            direction in ("client_to_server", "server_to_client"),
            "invalid wire direction",
        )
        require(
            isinstance(message, dict) and message.get("jsonrpc") == "2.0",
            "invalid JSON-RPC",
        )
        if "id" not in message:
            require(
                isinstance(message.get("method"), str)
                and "result" not in message
                and "error" not in message,
                "response is missing its RPC ID",
            )
            continue  # A notification, not a response or successful tool call.
        rpc_id = message["id"]
        require(type(rpc_id) in (str, int), "invalid RPC ID type")
        if "method" in message:
            require(
                isinstance(message["method"], str)
                and "result" not in message
                and "error" not in message,
                "invalid RPC request",
            )
            key = (direction, type(rpc_id), rpc_id)
            require(key not in pending, "duplicate in-flight RPC ID")
            pending[key] = (position, message)
            continue
        opposite = (
            "server_to_client"
            if direction == "client_to_server"
            else "client_to_server"
        )
        key = (opposite, type(rpc_id), rpc_id)
        require(key in pending, "response has no matching request")
        start, request = pending.pop(key)
        require(("result" in message) != ("error" in message), "invalid RPC response")
        if opposite != "client_to_server" or request.get("method") != "tools/call":
            continue
        params = request.get("params")
        require(
            isinstance(params, dict) and isinstance(params.get("arguments"), dict),
            "tool arguments must be an object",
        )
        result = message.get("result")
        data = (
            result.get("structuredContent")
            if isinstance(result, dict) and result.get("isError", False) is False
            else None
        )
        calls.append(
            {
                "start": start,
                "end": position,
                "name": params.get("name"),
                "arguments": params["arguments"],
                "data": data,
            }
        )
    require(not pending, "capture ends with unanswered RPC requests")
    return calls


def run_retry(raw: bytes) -> dict:
    calls = tool_calls(raw)
    starts = []
    for call in sorted(calls, key=lambda item: item["start"]):
        if call["name"] == "run_start":
            require(
                type(call["arguments"].get("dry_run", False)) is bool,
                "dry_run must be Boolean",
            )
            if not call["arguments"].get("dry_run", False):
                starts.append(call)
    require(len(starts) == 2, "need exactly two non-dry run_start attempts")
    first, retry = starts
    key = first["arguments"].get("idempotency_key")
    require(
        isinstance(key, str) and bool(key.strip()), "missing run_start idempotency key"
    )
    # Dict equality would equate False with 0 and 1 with 1.0.
    require(
        json.dumps(first["arguments"], sort_keys=True)
        == json.dumps(retry["arguments"], sort_keys=True),
        "retry arguments or idempotency key changed",
    )
    run_ids = []
    for call in starts:
        data = call["data"]
        run_id = data.get("run_id") if isinstance(data, dict) else None
        require(
            isinstance(run_id, str) and _RUN_ID.fullmatch(run_id) is not None,
            "run_start did not return a successful canonical run ID",
        )
        run_ids.append(run_id)
    require(run_ids[0] == run_ids[1], "retry returned a different run ID")

    def inventory(after, before):
        observed = [
            call
            for call in calls
            if call["name"] == "run_status"
            and "run_id" not in call["arguments"]
            and after < call["start"]
            and call["end"] < before
        ]
        require(bool(observed), "missing ordered run-list observation")
        data = max(observed, key=lambda call: call["end"])["data"]
        require(
            isinstance(data, dict)
            and "next_after" in data
            and data["next_after"] is None,
            "run inventory is failed, missing or paginated",
        )
        rows = data.get("runs")
        require(isinstance(rows, list) and len(rows) <= 256, "invalid run inventory")
        ids = []
        for row in rows:
            value = row.get("run_id") if isinstance(row, dict) else None
            require(
                isinstance(value, str) and _RUN_ID.fullmatch(value) is not None,
                "invalid inventory run ID",
            )
            ids.append(value)
        require(len(set(ids)) == len(ids), "duplicate inventory run ID")
        return set(ids)

    before = inventory(-1, first["start"])
    created = inventory(first["end"], retry["start"])
    replayed = inventory(retry["end"], float("inf"))
    require(
        run_ids[0] not in before and created == before | {run_ids[0]},
        "first creation did not add exactly the returned run",
    )
    require(replayed == created, "run inventory changed after retry")
    return {
        "run_id": run_ids[0],
        "successful_attempts": 2,
        "inventory_counts": [len(before), len(created), len(replayed)],
    }


def verify(
    manifest_path: Path, root: Path, *, check_run_retry=False, manifest_sha256=None
) -> dict[str, object]:
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise CaptureError(f"cannot read manifest: {exc}") from exc
    if check_run_retry or manifest_sha256 is not None:
        require(
            isinstance(manifest_sha256, str)
            and _SHA256.fullmatch(manifest_sha256) is not None,
            "--check-run-retry requires an explicit trusted --manifest-sha256",
        )
        require(
            hashlib.sha256(raw_manifest).hexdigest() == manifest_sha256,
            "manifest SHA-256 mismatch",
        )
    manifest = decode_json(raw_manifest)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "jopedime.named-client-transcripts/v1"
    ):
        raise CaptureError("unsupported transcript manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise CaptureError("manifest has no sidecar inventory")

    contents = {}
    verified = 0
    for name, expected in sorted(files.items()):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in ("", ".", "..")
        ):
            raise CaptureError(f"unsafe sidecar name: {name!r}")
        if not isinstance(expected, dict):
            raise CaptureError(f"invalid sidecar entry: {name}")
        digest = expected.get("sha256")
        size = expected.get("bytes")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CaptureError(f"invalid SHA-256 for {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CaptureError(f"invalid byte count for {name}")
        sidecar = root / name
        if sidecar.is_symlink() or not sidecar.is_file():
            raise CaptureError(f"sidecar is missing or not a regular file: {name}")
        actual_size = sidecar.stat().st_size
        if actual_size != size:
            raise CaptureError(
                f"byte count mismatch for {name}: expected {size}, got {actual_size}"
            )
        data = sidecar.read_bytes()
        require(len(data) == size, f"sidecar changed while reading: {name}")
        actual_digest = hashlib.sha256(data).hexdigest()
        if actual_digest != digest:
            raise CaptureError(f"SHA-256 mismatch for {name}")
        if check_run_retry:
            contents[name] = data
        verified += 1

    a5 = manifest.get("a5")
    label = a5.get("status") if isinstance(a5, dict) else None
    result = {
        "integrity": "PASS",
        "sidecars_verified": verified,
        "a5_status": f"UNVERIFIED:{label}"
        if isinstance(label, str)
        else "UNVERIFIED:missing",
    }
    if check_run_retry:
        clients = manifest.get("clients")
        require(isinstance(clients, dict), "missing named-client inventory")
        checks = {}
        for client, slug in _CLIENTS.items():
            record = clients.get(client)
            require(
                isinstance(record, dict)
                and isinstance(record.get("version"), str)
                and bool(record["version"].strip()),
                f"missing pinned version: {client}",
            )
            execution = contents.get(f"{slug}.execution.json")
            wire = contents.get(f"{slug}.mcp.jsonl")
            require(
                execution is not None and wire is not None,
                f"missing execution/wire sidecar: {client}",
            )
            execution = decode_json(execution)
            require(
                isinstance(execution, dict)
                and execution.get("client") == client
                and type(execution.get("exit_code")) is int
                and execution["exit_code"] == 0
                and type(execution.get("engine_exit")) is int
                and execution["engine_exit"] == 0
                and execution.get("timed_out") is False,
                f"unsuccessful execution: {client}",
            )
            checks[client] = run_retry(wire)
        result["run_retry_contract"] = checks
    return result


def self_test() -> int:
    """Synthetic verifier controls only; never launch a client or engine."""
    if __package__:
        from .verify_capture_selftest import run
    else:
        from verify_capture_selftest import run
    return run(globals())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument(
        "--root",
        type=Path,
        help="directory containing sidecars (default: manifest directory)",
    )
    parser.add_argument(
        "--check-run-retry",
        action="store_true",
        help="require the captured retry/inventory subpath for all three clients; does not qualify A5",
    )
    parser.add_argument(
        "--manifest-sha256", help="trusted manifest pin required with --check-run-retry"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run offline verifier controls, never client runs",
    )
    args = parser.parse_args()
    if args.self_test:
        if (
            args.manifest is not None
            or args.root is not None
            or args.check_run_retry
            or args.manifest_sha256 is not None
        ):
            parser.error("--self-test cannot be combined with capture verification")
        return self_test()
    if args.manifest is None:
        parser.error("manifest is required")
    root = args.root if args.root is not None else args.manifest.parent
    try:
        result = verify(
            args.manifest,
            root,
            check_run_retry=args.check_run_retry,
            manifest_sha256=args.manifest_sha256,
        )
    except (CaptureError, OSError) as exc:
        label = "verification" if args.check_run_retry else "integrity"
        print(json.dumps({label: "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
