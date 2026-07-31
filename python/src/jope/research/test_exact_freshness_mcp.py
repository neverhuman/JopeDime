#!/usr/bin/env python3
"""Pinned stdio MCP adapter for the freshness supervisor Unix socket."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from test_exact_freshness import ROOT, socket_request

MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_VERSION = "1.0.0"

TOOLS = {
    "get_freshness_status": {
        "description": "Return checkpoint freshness, queue epochs, and latency health.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "get_active_checkpoint": {
        "description": "Return the admitted checkpoint and immutable epoch.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "list_queue_epochs": {
        "description": "List useful, held, running, and dispatching work by epoch.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "explain_job_provenance": {
        "description": "Join one job to its manifest, claim, epoch, and pair identities.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string", "minLength": 1}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
    "preview_stale_retirement": {
        "description": "Preview stale holds/retirements without changing scheduler state.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "apply_stale_retirement": {
        "description": "Apply a preview after rechecking that every target is held/unstarted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preview_receipt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "idempotency_key": {"type": "string", "minLength": 1},
            },
            "required": ["preview_receipt_sha256", "idempotency_key"],
            "additionalProperties": False,
        },
    },
    "pause_admission": {
        "description": "Pause new pair admission without interrupting active attempts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preview_receipt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "idempotency_key": {"type": "string", "minLength": 1},
            },
            "required": ["preview_receipt_sha256", "idempotency_key"],
            "additionalProperties": False,
        },
    },
    "resume_admission": {
        "description": "Resume new pair admission after a reviewed preview.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preview_receipt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "idempotency_key": {"type": "string", "minLength": 1},
            },
            "required": ["preview_receipt_sha256", "idempotency_key"],
            "additionalProperties": False,
        },
    },
    "request_micro_wave": {
        "description": "Write an idempotent current-checkpoint 128-dataset micro-wave request.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preview_receipt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "idempotency_key": {"type": "string", "minLength": 1},
                "reason": {"type": "string"},
            },
            "required": ["preview_receipt_sha256", "idempotency_key"],
            "additionalProperties": False,
        },
    },
}


def response(identifier: object, *, result: object = None, error: dict | None = None) -> dict:
    payload = {"jsonrpc": "2.0", "id": identifier}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def handle(message: dict, socket_path: Path) -> dict | None:
    identifier = message.get("id")
    method = message.get("method")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return response(
            identifier,
            result={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "jope-selfplay-freshness",
                    "version": SERVER_VERSION,
                },
                "instructions": (
                    "Read-only provenance and receipt-gated freshness mutations. "
                    "The server cannot execute arbitrary SCQ commands or cancel running work."
                ),
            },
        )
    if method == "tools/list":
        return response(
            identifier,
            result={
                "tools": [
                    {"name": name, **definition}
                    for name, definition in TOOLS.items()
                ]
            },
        )
    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name") or "")
        if name not in TOOLS:
            return response(
                identifier,
                error={"code": -32601, "message": f"unknown tool {name}"},
            )
        try:
            result = socket_request(
                name, params.get("arguments") or {}, socket_path=socket_path
            )
            return response(
                identifier,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, sort_keys=True),
                        }
                    ],
                    "structuredContent": result,
                    "isError": False,
                },
            )
        except Exception as error:
            return response(
                identifier,
                result={
                    "content": [{"type": "text", "text": str(error)}],
                    "isError": True,
                },
            )
    return response(
        identifier,
        error={"code": -32601, "message": f"unsupported method {method}"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, default=ROOT / "ops" / "freshness.sock")
    args = parser.parse_args()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            result = handle(json.loads(line), args.socket)
        except Exception as error:
            result = response(
                None,
                error={"code": -32603, "message": f"{type(error).__name__}: {error}"},
            )
        if result is not None:
            sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
