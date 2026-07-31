#!/usr/bin/env python3
"""Candidate/active evaluator staging with last-good fallback semantics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from test_exact_common import (
    EVALUATOR_PROTOCOL_VERSION,
    atomic_json,
    evaluation_protocol_sha256,
    sha256_file,
)

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SOURCE = Path("/home/ubuntu/jope-dime/target/release/corpus_eval")
NODES = ("xbabe1", "xbabe2", "xbabe3", "xbabe4")


def pointer_path(root: Path, name: str) -> Path:
    return root / "evaluator" / f"{name}.json"


def stage(source: Path = SOURCE, root: Path = ROOT) -> dict:
    digest = sha256_file(source)
    destination = root / "bin" / digest / "corpus_eval"
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("evaluator changed while staging")
        os.replace(temporary, destination)
        destination.chmod(0o755)
    elif sha256_file(destination) != digest:
        raise RuntimeError("staged evaluator hash drift")
    record = {
        "schema": "JopeEvaluatorArtifact.v1",
        "evaluator": str(destination),
        "evaluator_sha256": digest,
        "source": str(source),
        "compile_policy": "precompiled_broad_eval",
    }
    atomic_json(destination.parent / "stage.json", record)
    return record


def load_pointer(name: str, root: Path = ROOT) -> dict | None:
    path = pointer_path(root, name)
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    evaluator = Path(value["evaluator"])
    if not evaluator.is_file() or sha256_file(evaluator) != value["evaluator_sha256"]:
        raise RuntimeError(f"{name} evaluator pointer failed content verification")
    return value


def active(root: Path = ROOT, source: Path = SOURCE) -> dict:
    current = load_pointer("active", root)
    if current is not None:
        return current
    # One-time bootstrap preserves the evaluator already serving queued work.
    record = {
        **stage(source, root),
        "activated_at_ms": int(time.time() * 1000),
        "activation": "bootstrap-last-good",
    }
    atomic_json(pointer_path(root, "active"), record)
    return record


def candidate(root: Path = ROOT, source: Path = SOURCE) -> dict:
    record = {
        **stage(source, root),
        "staged_at_ms": int(time.time() * 1000),
        "state": "candidate",
    }
    atomic_json(pointer_path(root, "candidate"), record)
    return record


def promote_candidate(canary_report: Path, root: Path = ROOT) -> dict:
    staged = load_pointer("candidate", root)
    if staged is None:
        raise RuntimeError("no candidate evaluator is staged")
    report = json.loads(canary_report.read_text())
    nodes = report.get("nodes") or {}
    if set(nodes) != set(NODES):
        raise RuntimeError("candidate canary did not cover every evaluator node")
    hashes = {row.get("evaluator_sha256") for row in nodes.values()}
    if report.get("semantic_bit_parity") is not True or hashes != {
        staged["evaluator_sha256"]
    }:
        raise RuntimeError("candidate evaluator did not pass node semantic parity")
    promoted = {
        **staged,
        "state": "active",
        "activated_at_ms": int(time.time() * 1000),
        "canary_report": str(canary_report),
        "canary_report_sha256": sha256_file(canary_report),
    }
    atomic_json(pointer_path(root, "active"), promoted)
    return promoted


def activate_protocol(root: Path = ROOT) -> dict:
    """Write an immutable protocol record and atomically select it."""

    # Dataset-specific train/test hashes are folded into each pair hash. This
    # pointer freezes the correctness-changing protocol identity shared by them.
    identity = evaluation_protocol_sha256(
        task="regression",
        metric="r2",
        score_tier="test_exact",
        train_sha256="0" * 64,
        test_sha256="0" * 64,
        version=EVALUATOR_PROTOCOL_VERSION,
    )
    record = {
        "schema": "JopeEvaluationProtocol.v1",
        "version": EVALUATOR_PROTOCOL_VERSION,
        "identity_sha256": identity,
        "task_metrics": {"regression": "r2", "binary": "auc"},
        "score_tier": "test_exact",
        "pair_binding": [
            "task",
            "metric",
            "score_tier",
            "train_sha256",
            "test_sha256",
            "version",
        ],
    }
    immutable = root / "protocols" / f"{identity}.json"
    if not immutable.is_file():
        atomic_json(immutable, record)
    elif json.loads(immutable.read_text()) != record:
        raise RuntimeError("immutable evaluation protocol record drift")
    pointer = {
        "schema": "JopeEvaluationProtocolPointer.v1",
        "protocol": str(immutable),
        "protocol_sha256": sha256_file(immutable),
        "identity_sha256": identity,
        "version": EVALUATOR_PROTOCOL_VERSION,
    }
    atomic_json(root / "protocols" / "active.json", pointer)
    return pointer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("active")
    commands.add_parser("stage-candidate")
    promote = commands.add_parser("promote-candidate")
    promote.add_argument("--canary-report", type=Path, required=True)
    commands.add_parser("activate-protocol")
    args = parser.parse_args()
    if args.command == "active":
        result = active(args.root, args.source)
    elif args.command == "stage-candidate":
        result = candidate(args.root, args.source)
    elif args.command == "promote-candidate":
        result = promote_candidate(args.canary_report, args.root)
    else:
        result = activate_protocol(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
