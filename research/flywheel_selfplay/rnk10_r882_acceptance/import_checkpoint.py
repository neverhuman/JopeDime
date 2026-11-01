#!/usr/bin/env python3
"""Explicit research-only importer for the owner-pinned r882 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

CHECKPOINT = Path("/home/ubuntu/jope-data/RNK10-t2115/r882-0084479aa0703667-p30853cad78d86087/round-882.pt")
EXPECTED_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
EXPECTED_POINTER_SHA256 = "30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a"
EXPECTED_DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
OUTPUT_DIR = Path("/home/ubuntu/jope-data/RNK10-t2115-run-20261101")
OUTPUT_WEIGHTS = OUTPUT_DIR / "round-882-state.safetensors"
OUTPUT_RECEIPT = OUTPUT_DIR / "round-882-import.json"
EXPECTED_STATE_TENSORS = 338


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_state(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    if payload.get("round") != 882:
        raise ValueError("checkpoint round is not 882")
    if payload.get("dope_sha256") != EXPECTED_DOPE_SHA256:
        raise ValueError("checkpoint DOPE identity mismatch")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no nonempty state_dict")
    if any(not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("state_dict must contain named tensors only")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-import",
        action="store_true",
        help="required; never enables or changes the production model loader",
    )
    args = parser.parse_args()
    if not args.research_import:
        parser.error("refusing import without explicit --research-import")
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != EXPECTED_SHA256:
        raise SystemExit("pinned r882 checkpoint missing or SHA-256 mismatch")
    pointer = CHECKPOINT.with_name("best_dev_checkpoint.json")
    if sha256(pointer) != EXPECTED_POINTER_SHA256:
        raise SystemExit("pinned r882 pointer missing or SHA-256 mismatch")
    metadata = json.loads(pointer.read_text())
    if metadata.get("best_round") != 882 or metadata.get("evaluation_identity") != "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83":
        raise SystemExit("pinned r882 pointer metadata mismatch")
    if OUTPUT_WEIGHTS.exists() or OUTPUT_RECEIPT.exists():
        raise SystemExit("refusing to overwrite a prior research import")
    if "/flywheel-full-20261030/run-20261030/" in str(OUTPUT_WEIGHTS.resolve()):
        raise SystemExit("refusing output under the live flywheel directory")

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = extract_state(payload)
    if len(state) != EXPECTED_STATE_TENSORS:
        raise SystemExit(f"expected {EXPECTED_STATE_TENSORS} tensors, got {len(state)}")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise SystemExit("r882 state_dict contains a non-F32 tensor")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file({name: value.contiguous() for name, value in state.items()}, str(OUTPUT_WEIGHTS))
    receipt = {
        "schema": "jope.rnk10.research-import/v1",
        "research_only": True,
        "production_bundle": False,
        "source_checkpoint": str(CHECKPOINT),
        "source_checkpoint_sha256": EXPECTED_SHA256,
        "source_pointer_sha256": EXPECTED_POINTER_SHA256,
        "evaluation_identity": metadata["evaluation_identity"],
        "round": 882,
        "dope_sha256": EXPECTED_DOPE_SHA256,
        "state_tensor_count": len(state),
        "imported_safetensors": str(OUTPUT_WEIGHTS),
        "imported_safetensors_sha256": sha256(OUTPUT_WEIGHTS),
        "signing": "none; this is not a VerifiedBundle or production artifact",
    }
    OUTPUT_RECEIPT.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
