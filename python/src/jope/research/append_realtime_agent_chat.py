#!/usr/bin/env python3
"""Append one normalized, durable event to REALTIME_AGENT_CHAT.md."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path


def normalized_field(value: str, name: str) -> str:
    if "|" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} contains a forbidden field separator or newline")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def append_event(
    chat_path: Path,
    *,
    actor: str,
    hypothesis: str,
    evidence: str,
    next_action: str,
    blocker: str,
    handoff: str,
    timestamp: str | None = None,
) -> str:
    fields = {
        "actor": normalized_field(actor, "actor"),
        "hypothesis": normalized_field(hypothesis, "hypothesis"),
        "evidence": normalized_field(evidence, "evidence"),
        "next_action": normalized_field(next_action, "next_action"),
        "blocker": normalized_field(blocker, "blocker"),
        "handoff": normalized_field(handoff, "handoff"),
    }
    if not fields["hypothesis"].startswith(tuple(f"{event} " for event in (
        "STATE_FREEZE", "CLAIM_REQUEST", "CLAIM_ACK", "CLAIM_CONFLICT",
        "HEARTBEAT", "CHECKPOINT", "ACTION_PROPOSAL", "ACTION_ACK", "HANDOFF",
        "REVIEW_ACK", "LEASE_RELEASE",
    ))):
        raise ValueError("hypothesis must begin with a registered V3 event type")
    if "lease=" not in fields["hypothesis"] or "task=" not in fields["hypothesis"]:
        raise ValueError("hypothesis must include lease=<id> and task=<id>")

    stamp = timestamp or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    stamp = normalized_field(stamp, "timestamp")
    line = " | ".join((stamp, *fields.values())) + "\n"

    chat_path = chat_path.resolve()
    parent_fd = os.open(chat_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with chat_path.open("a", encoding="utf-8") as chat:
            fcntl.flock(chat.fileno(), fcntl.LOCK_EX)
            chat.write(line)
            chat.flush()
            os.fsync(chat.fileno())
            fcntl.flock(chat.fileno(), fcntl.LOCK_UN)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", type=Path, default=Path(__file__).resolve().parents[1] / "REALTIME_AGENT_CHAT.md")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--next-action", required=True)
    parser.add_argument("--blocker", default="none")
    parser.add_argument("--handoff", default="none")
    args = parser.parse_args()
    print(append_event(
        args.chat,
        actor=args.actor,
        hypothesis=args.hypothesis,
        evidence=args.evidence,
        next_action=args.next_action,
        blocker=args.blocker,
        handoff=args.handoff,
    ), end="")


if __name__ == "__main__":
    main()
