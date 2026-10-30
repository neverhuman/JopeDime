"""Durable train-pool admission and strict incumbent promotion for closed self-play.

The append-only journal is the recovery source. Replay it before loading a new round.
Fitness is a cost; utility is its negation. Historical admission does not imply an HWM.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def apply_event(pool: dict, event: dict) -> bool:
    """Apply once to every parallel candidate array; also used during restart replay."""
    pid = event["program_sha256"]
    if pid in pool["pids"]:
        return False
    feat = np.asarray(event["feat412"], dtype=np.float32).reshape(1, 412)
    proj = np.asarray(event["proj16"], dtype=np.float64).reshape(1, 16)
    pad = np.pad(feat, ((0, 0), (0, 20))).reshape(1, 18, 24)
    sig = pad.mean(axis=1) + 0.5 * pad.max(axis=1)
    sig /= np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
    for name, row in [("program", feat), ("proj16", proj), ("struct_sig", sig)]:
        pool[name] = np.concatenate([pool[name], row])
    for name, value in [
        ("fitness", event["fitness"]),
        ("logratio", event["logratio"]),
        ("gate", True),
        ("prior_cost", float(np.median(pool["prior_cost"]))),
    ]:
        pool[name] = np.append(pool[name], value)
    pool["pids"].append(pid)
    return True


def promote(
    pool: dict,
    candidate: dict,
    journal: Path,
    round_idx: int,
    *,
    historical_admission: bool = False,
) -> dict | None:
    """Journal before mutation, reject evaluation contexts, gate failures and ties.

    historical_admission admits the owner-requested verified 90 into training support;
    only strict improvements receive kind=promotion. No test outcome enters admission.
    """
    rec = pool["rec"]
    if rec["dl_split"] != "DL-Train" or rec["source_split"] not in ("train", "inner"):
        raise ValueError("promotion is restricted to training contexts")
    if candidate.get("context_sha256", rec["context_sha256"]) != rec["context_sha256"]:
        raise ValueError("context identity mismatch")
    if (
        candidate.get("status") != "completed"
        or candidate.get("cv_gate_passed") is not True
    ):
        return None
    if candidate.get("evaluation_scope") != "full_train_cv":
        raise ValueError("requires full training CV, not a screening estimate")
    pid = hashlib.sha256(candidate["dsl"].encode()).hexdigest()
    if pid != candidate["program_sha256"]:
        raise ValueError("program hash mismatch")
    score = float(candidate["fitness"])
    logratio = float(candidate["logratio"])
    feat = np.asarray(candidate["feat412"]).reshape(412)
    proj = np.asarray(candidate["proj16"]).reshape(16)
    if not (
        math.isfinite(score)
        and math.isfinite(logratio)
        and np.isfinite(feat).all()
        and np.isfinite(proj).all()
    ):
        raise ValueError("non-finite candidate")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        prior = [json.loads(line) for line in stream if line.strip()]
        for event in prior:
            if event["context_sha256"] == rec["context_sha256"]:
                apply_event(pool, event)
        if pid in pool["pids"]:
            return None
        passing = pool["fitness"][pool["gate"]]
        if not len(passing):
            raise ValueError("missing passing incumbent")
        incumbent = float(np.min(passing))
        improves = score < incumbent - 1e-6
        if not improves and not historical_admission:
            return None
        event = {
            "schema": "jope.flywheel-promotion/v1",
            "kind": "promotion" if improves else "historical_admission",
            "context_sha256": rec["context_sha256"],
            "family_id": rec["family_id"],
            "program_sha256": pid,
            "dsl": candidate["dsl"],
            "round": round_idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_utility": -incumbent,
            "new_utility": -score,
            "fitness": score,
            "logratio": logratio,
            "CI": candidate.get("CI"),
            "CI_scope": "unavailable" if candidate.get("CI") is None else "training_cv",
            "evaluation_scope": "full_train_cv",
            "cv_gate_passed": True,
            "input_identity": candidate["input_identity"],
            "feat412": feat.tolist(),
            "proj16": proj.tolist(),
        }
        stream.seek(0, 2)
        stream.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        apply_event(pool, event)
        return event


def replay(pools: dict, journal: Path) -> int:
    count = 0
    if journal.exists():
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            cid = event["context_sha256"]
            if cid not in pools or pools[cid]["rec"]["dl_split"] != "DL-Train":
                raise ValueError("journal contains non-training context")
            count += apply_event(pools[cid], event)
    return count


def training_cv_envelope(raw: bytes) -> bytes:
    """Adapt a train-only JTT1 packet to the pinned S1 parser's nq>=10 envelope.

    The parser checks envelope length but never decodes query cells. Ten all-zero
    query records are structural padding, with no outer rows/features/labels read.
    Training bytes and fold assignments remain identical.
    """
    import struct

    if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError("invalid training packet")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq or len(raw) != 17 + nt + nt * (width + 1) * 8:
        raise ValueError("expected training-only rows")
    out = bytearray(raw)
    struct.pack_into("<I", out, 8, 10)
    out.extend(bytes(10 * (width + 1) * 8))
    return bytes(out)


def training_indices(pool: dict, slate_size: int) -> np.ndarray:
    """Retain the best 48 (including a promoted incumbent) in every training slate."""
    if slate_size < 1 or not len(pool["pids"]):
        raise ValueError("positive slate and nonempty pool required")
    order = np.argsort(pool["fitness"])
    k = min(48, slate_size, len(order))
    rest_pool = order[k:] if len(order) > k else order
    rest = np.random.choice(
        rest_pool, slate_size - k, replace=len(rest_pool) < slate_size - k
    )
    return np.concatenate([order[:k], rest])
