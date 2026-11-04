"""Checkpoint promotion and command-line entry point for test_exact_accept."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path

from test_exact_common import atomic_json, sha256_file
from test_exact_accept_gate import evaluate_gate, load_accepted_by_task

def promote(
    *,
    evidence_path: Path,
    gate_path: Path,
    accepted_pointer: Path,
    immutable_root: Path,
    enforce_pointer_ancestor: bool = True,
) -> dict:
    evidence = json.loads(evidence_path.read_text())
    # The per-task anchor file lives beside the global accepted pointer; it
    # decides which NEW branches gate (advisory-until-anchor).
    accepted_by_task = load_accepted_by_task(
        accepted_pointer.with_name("accepted-by-task.json")
    )
    result = evaluate_gate(evidence, accepted_by_task)
    atomic_json(gate_path, result)
    if not result["pass"]:
        return result
    checkpoint = Path(evidence["candidate"]["checkpoint_path"])
    if sha256_file(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("accepted checkpoint content hash mismatch")
    immutable_checkpoint = (
        immutable_root
        / "checkpoints"
        / f"{result['checkpoint_sha256']}.model.pt"
    )
    immutable_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not immutable_checkpoint.exists():
        temporary = immutable_checkpoint.with_name(
            f".{immutable_checkpoint.name}.tmp-{os.getpid()}"
        )
        shutil.copy2(checkpoint, temporary)
        if sha256_file(temporary) != result["checkpoint_sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("checkpoint changed while copying to immutable storage")
        os.replace(temporary, immutable_checkpoint)
    elif sha256_file(immutable_checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("immutable checkpoint path contains different bytes")

    evidence_hash = sha256_file(evidence_path)
    immutable_evidence = immutable_root / "gates" / f"{evidence_hash}.evidence.json"
    immutable_evidence.parent.mkdir(parents=True, exist_ok=True)
    if not immutable_evidence.exists():
        shutil.copy2(evidence_path, immutable_evidence)
    if sha256_file(immutable_evidence) != evidence_hash:
        raise RuntimeError("immutable evidence copy hash mismatch")
    gate_hash = sha256_file(gate_path)
    immutable_gate = immutable_root / "gates" / f"{gate_hash}.gate.json"
    if not immutable_gate.exists():
        shutil.copy2(gate_path, immutable_gate)
    if sha256_file(immutable_gate) != gate_hash:
        raise RuntimeError("immutable gate copy hash mismatch")

    lock_path = accepted_pointer.with_suffix(f"{accepted_pointer.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if accepted_pointer.is_file():
            current = json.loads(accepted_pointer.read_text())
            if current["checkpoint_sha256"] == result["checkpoint_sha256"]:
                return result
            if (
                enforce_pointer_ancestor
                and current["checkpoint_sha256"]
                != result["accepted_ancestor_sha256"]
            ):
                raise RuntimeError("accepted ancestor changed before pointer promotion")
        atomic_json(
            accepted_pointer,
            {
                "schema": "JopeSelfplayAcceptedCheckpoint.v2",
                "checkpoint": str(immutable_checkpoint),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "accepted_ancestor_sha256": result["accepted_ancestor_sha256"],
                "gate": str(immutable_gate),
                "gate_sha256": gate_hash,
                "evidence": str(immutable_evidence),
                "evidence_sha256": evidence_hash,
                "test_adaptive": True,
                "benchmark_valid": False,
                "selection_bias": "test-exact-reused",
                "promotion_scope": "internal-selected-test-flywheel-only",
            },
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accepted-pointer",
        type=Path,
        default=Path("/home/ubuntu/jope-flywheel/test_exact_gen0/training/accepted.json"),
    )
    parser.add_argument(
        "--immutable-root",
        type=Path,
        default=Path("/home/ubuntu/jope-flywheel/test_exact_gen0/training/immutable"),
    )
    args = parser.parse_args()
    result = promote(
        evidence_path=args.evidence,
        gate_path=args.output,
        accepted_pointer=args.accepted_pointer,
        immutable_root=args.immutable_root,
    )
    print(json.dumps(result, sort_keys=True))
    # jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
    raise SystemExit(0 if result["pass"] else 2)
