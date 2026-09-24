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


def row_identity(row):
    value = copy.deepcopy(row.to_json())
    for key in ("oof_delta_to_incumbent", "rank_pct", "routing_class", "seq_weight"):
        value.get("metadata", {}).pop(key, None)
    return digest(value)


def supervision_snapshot(rows):
    return [
        {
            "row_sha256": row_identity(row),
            **{key: row.metadata.get(key) for key in ("oof_delta_to_incumbent", "rank_pct", "routing_class")},
            "seq_weight": row.metadata.get("seq_weight", 1.0),
        }
        for row in rows
    ]


def freeze_supervision(rows, reference_rows, parent):
    """Keep replay targets/routing fixed; new scores compare to frozen incumbents."""
    import math

    from ..split_integrity import partition_training_rows
    from .train_v3 import _recompute_reward_targets, _routing_winners, _score_label

    selected, reference = partition_training_rows(rows), partition_training_rows(reference_rows)
    if (
        selected["report_only"]
        or reference["report_only"]
        or not reference["train"]
        or not reference["development"]
    ):
        raise ValueError("supervision reference must contain only train/development rows")
    for split in ("train", "development"):
        originals = reference[split]
        _recompute_reward_targets(originals)
        frozen = {row_identity(row): row.metadata for row in originals}
        if len(frozen) != len(originals):
            raise ValueError("ambiguous duplicate replay supervision rows")
        groups = {}
        for row in originals:
            md = row.metadata or {}
            identity = md.get("reward_pair_identity") or md.get("dataset_hash") or row.dataset_id
            score = _score_label(row)
            if score is not None:
                groups.setdefault(str(identity), []).append(score)
        _recompute_reward_targets(selected[split])
        for row in selected[split]:
            md = row.metadata or {}
            old = frozen.get(row_identity(row))
            identity = md.get("reward_pair_identity") or md.get("dataset_hash") or row.dataset_id
            scores, score = groups.get(str(identity)), _score_label(row)
            if old is not None:
                for key in ("oof_delta_to_incumbent", "rank_pct"):
                    if key in old:
                        md[key] = old[key]
                    else:
                        md.pop(key, None)
            elif scores and score is not None:
                md["oof_delta_to_incumbent"] = score - max(scores)
                md["rank_pct"] = min(1.0, sum(s < score for s in scores) / max(1, len(scores) - 1))
            row.metadata = md
    winners = _routing_winners(reference["train"])
    families = {name: i for i, name in enumerate(parent.get("routing_families") or [])}
    for row in [*rows, *reference_rows]:
        md = row.metadata or {}
        prov = str(md.get("provenance", "corpus")).split("_g")[0] or "corpus"
        value, rank = md.get("oof_delta_to_incumbent"), md.get("rank_pct")
        finite = isinstance(value, (int, float)) and math.isfinite(float(value))
        if prov not in ("corpus", "quant_report", "seed"):
            winner = finite and (float(value) == 0.0 or (isinstance(rank, (int, float)) and rank >= 0.9))
            md["seq_weight"] = 1.0 if winner else (0.05 if finite else 0.0)
        winner = winners.get(md.get("dataset_hash") or row.dataset_id)
        md["routing_class"] = families.get(winner[1], parent["routing_classes"] - 1) if winner else -1
        row.metadata = md
    return reference
