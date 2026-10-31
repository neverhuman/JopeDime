#!/usr/bin/env python3
"""Metadata-only S3 plumbing check for W9 preparation; never fits or scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


BUCKET = "veox-jopedime-datasets-use2"
DATASET_ID = "b3b41e3413953c73"
CANONICAL_SHA256 = "b3b41e3413953c73ee85d6b901dededd1ab905827012fb3126a9d8b454621ad0"
TRAIN_SHA256 = "fc765fe9d309884cebe08fb9e2be0d9a6df219f1407a05684badcfee609d1ccf"
TRAIN_BYTES = 183_601
TRAIN_ROWS = 25
TRAIN_COLUMNS = 840


def s3_stdout(key: str) -> bytes:
    """Fetch one exact object; never echo AWS diagnostics or object contents."""
    result = subprocess.run(
        ["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-", "--only-show-errors"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=45,
    )
    if result.returncode:
        raise RuntimeError(f"S3 read failed for the pinned object (exit {result.returncode})")
    return result.stdout


def dry_run() -> dict[str, object]:
    manifest_key = f"datasets/{DATASET_ID}/manifest.json"
    manifest_bytes = s3_stdout(manifest_key)
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("id") != DATASET_ID
        or manifest.get("dataset_hash") != DATASET_ID
        or manifest.get("canonical_sha256") != CANONICAL_SHA256
        or manifest.get("task") != "regression"
        or manifest.get("origin") != "real"
    ):
        raise RuntimeError("S3 manifest does not match the pinned real regression catalog entry")

    train = manifest.get("files", {}).get("train", {})
    if (
        train.get("sha256") != TRAIN_SHA256
        or train.get("bytes") != TRAIN_BYTES
        or train.get("rows") != TRAIN_ROWS
        or train.get("columns") != TRAIN_COLUMNS
    ):
        raise RuntimeError("manifest train descriptor does not match the pinned catalog entry")

    # Content-addressed train blob only. No test key is constructed or requested.
    train_bytes = s3_stdout(f"blobs/sha256/{TRAIN_SHA256}")
    if len(train_bytes) != TRAIN_BYTES or hashlib.sha256(train_bytes).hexdigest() != TRAIN_SHA256:
        raise RuntimeError("downloaded train blob failed byte-count or SHA256 verification")
    rows = list(csv.reader(io.StringIO(train_bytes.decode("utf-8"), newline="")))
    if len(rows) != TRAIN_ROWS or any(len(row) != TRAIN_COLUMNS for row in rows):
        raise RuntimeError("train blob failed the pinned rectangular CSV shape check")

    return {
        "mode": "dry-run",
        "purpose": "input-plumbing-only; not a W9 row, outcome, fit, or result",
        "dataset_id": DATASET_ID,
        "origin": "real",
        "task": "regression",
        "manifest_key": manifest_key,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "canonical_sha256": CANONICAL_SHA256,
        "train_blob_key": f"blobs/sha256/{TRAIN_SHA256}",
        "train_sha256": TRAIN_SHA256,
        "train_bytes": len(train_bytes),
        "train_rows": len(rows),
        "train_columns": TRAIN_COLUMNS,
        "test_blob_requested": False,
        "model_fit_or_scoring": False,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["dry-run"])
    parser.add_argument(
        "--profile",
        help="optional AWS CLI profile; otherwise use the configured provider chain",
    )
    args = parser.parse_args()
    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile
    try:
        receipt = dry_run()
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"dry-run refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
