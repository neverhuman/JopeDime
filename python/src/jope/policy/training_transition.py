"""Explicit corpus transitions preserve model, optimizer and numerical contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def state_digest(value):
    """Device-independent hash of nested optimizer/model state, including tensor bytes."""
    import torch

    h = hashlib.sha256(b"jope.training-state/v1\0")

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            visit(["tensor", str(item.dtype), list(item.shape)])
            data = item.reshape(-1).view(torch.uint8).numpy().tobytes()
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
        elif isinstance(item, dict):
            h.update(b"dict\0")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
            h.update(b"end\0")
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
            h.update(b"end\0")
        else:
            data = json.dumps([type(item).__name__, item], allow_nan=False).encode()
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)

    visit(value)
    return h.hexdigest()


def invariant(contract):
    value = copy.deepcopy(contract)
    value.pop("input_provenance")
    value["execution"].pop("batch_schedule_sha256", None)
    return digest(value)


def prepare(parent_path, corpus_path, batches):
    import torch

    parent_path, corpus_path = Path(parent_path).resolve(strict=True), Path(corpus_path).resolve(strict=True)
    saved = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (
        saved.get("strict_progress") is not None
        or saved.get("optimizer_state_compatible_with_model") is not True
        or not saved.get("optimizer", {}).get("state")
        or saved.get("last_checkpoint_state") != "current"
    ):
        raise ValueError("transition requires a completed current parent with optimizer state")
    return {
        "schema": "jope.training-corpus-transition/v1",
        "parent": {"path": str(parent_path), "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest()},
        "parent_model_sha256": state_digest(saved["model"]),
        "parent_optimizer_sha256": state_digest(saved["optimizer"]),
        "old_input_provenance_sha256": digest(saved["training_contract"]["input_provenance"]),
        "invariant_sha256": invariant(saved["training_contract"]),
        "new_corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "batch_schedule_sha256": digest(batches),
    }


def validate(transition, saved, expected, parent_path, corpus_path, batches):
    actual = prepare(parent_path, corpus_path, batches)
    if transition != actual or invariant(expected) != transition["invariant_sha256"]:
        raise ValueError("corpus transition changed parent, optimizer, schedule or invariant contract")
    if state_digest(saved["optimizer"]) != transition["parent_optimizer_sha256"]:
        raise ValueError("corpus transition optimizer differs")
    # Only these two fields may change under the explicit transition.
    adapted = dict(saved)
    adapted["training_contract"] = expected
    return adapted
