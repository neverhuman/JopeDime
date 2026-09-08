"""Concurrent, immutable broad evaluation receipt store."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    BROAD_RECEIPT_SCHEMA,
    build_hashed,
    canonical_json_bytes,
    content_sha256,
    receipt_dedup_key,
    robust_score,
    validate_broad_receipt,
)


def build_receipt(
    *,
    dataset_hash: str,
    train_sha256: str,
    semantic_sha256: str,
    canonical_genome: str,
    evaluator_release_sha256: str,
    fold_plan_sha256: str,
    vocabulary_sha256: str,
    fidelity: str,
    valid: bool,
    fold_r2: list[float] | None,
    oof_r2: float | None,
    runtime_seconds: float | None,
    failure: Mapping[str, str] | None,
    selection_channel: str,
    recorded_at_ms: int | None = None,
) -> dict[str, Any]:
    protocol = (
        {
            "fidelity": "screening",
            "folds": 3,
            "row_cap": 768,
            "timeout_seconds": 30,
            "omp_threads": 1,
            "train_only": True,
        }
        if fidelity == "screening"
        else {
            "fidelity": "exact",
            "folds": 5,
            "row_cap": None,
            "timeout_seconds": None,
            "omp_threads": 1,
            "train_only": True,
        }
    )
    folds = list(fold_r2 or [])
    receipt = build_hashed(
        BROAD_RECEIPT_SCHEMA,
        "receipt_sha256",
        dataset_hash=dataset_hash,
        train_sha256=train_sha256,
        semantic_sha256=semantic_sha256,
        canonical_genome=canonical_genome,
        evaluator_release_sha256=evaluator_release_sha256,
        fold_plan_sha256=fold_plan_sha256,
        vocabulary_sha256=vocabulary_sha256,
        fidelity=fidelity,
        protocol=protocol,
        valid=valid,
        failure=dict(failure) if failure is not None else None,
        fold_r2=folds,
        oof_r2=oof_r2,
        robust_score=robust_score(oof_r2, folds) if valid and oof_r2 is not None else None,
        runtime_seconds=runtime_seconds,
        label_authority="train_only_oof_r2",
        test_table_accessed=False,
        selection_channel=selection_channel,
        recorded_at_ms=recorded_at_ms or int(time.time() * 1000),
    )
    validate_broad_receipt(receipt)
    return receipt


class ReceiptStore:
    """One content-bound file per scientific dedup key.

    A hard link publishes the fully fsynced temporary file atomically.  Two
    writers racing on the same key either agree byte-for-byte or fail closed.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.receipts = self.root / "receipts"
        self.receipts.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_sha256(receipt: Mapping[str, Any]) -> str:
        return content_sha256(list(receipt_dedup_key(receipt)))

    def append(self, receipt: Mapping[str, Any]) -> tuple[Path, bool]:
        value = dict(receipt)
        validate_broad_receipt(value)
        key = self.key_sha256(value)
        final = self.receipts / f"{key}.json"
        payload = canonical_json_bytes(value) + b"\n"
        if final.exists():
            if final.read_bytes() != payload:
                raise ValueError("receipt dedup key has conflicting immutable evidence")
            return final, False

        temporary = self.receipts / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, final)
                created = True
            except FileExistsError:
                created = False
            if final.read_bytes() != payload:
                raise ValueError("concurrent receipt append conflicts")
            directory = os.open(self.receipts, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return final, created
        finally:
            temporary.unlink(missing_ok=True)

    def import_all(
        self, receipts: Iterable[Mapping[str, Any]]
    ) -> dict[str, int]:
        inserted = duplicates = 0
        for receipt in receipts:
            _, created = self.append(receipt)
            inserted += int(created)
            duplicates += int(not created)
        return {"inserted": inserted, "duplicates": duplicates}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for path in sorted(self.receipts.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            validate_broad_receipt(value)
            if path.stem != self.key_sha256(value):
                raise ValueError(f"receipt filename does not bind dedup key: {path}")
            yield value

    def verify(self) -> dict[str, Any]:
        keys: set[tuple[str, str, str, str]] = set()
        digests: list[str] = []
        valid = failures = 0
        for receipt in self:
            key = receipt_dedup_key(receipt)
            if key in keys:
                raise ValueError("receipt store contains a duplicate scientific key")
            keys.add(key)
            digests.append(receipt["receipt_sha256"])
            valid += int(receipt["valid"])
            failures += int(not receipt["valid"])
        return {
            "schema": "BroadReceiptStoreVerification.v1",
            "receipts": len(keys),
            "valid": valid,
            "failures": failures,
            "store_sha256": content_sha256(digests),
            "duplicate_keys": 0,
        }
