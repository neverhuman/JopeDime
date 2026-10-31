#!/usr/bin/env python3
"""Fail-closed CPU preparation for W3; this module does not fit or score models."""

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
from pathlib import Path
from typing import Any


W1_SCHEMA = "jope.w1.fold-manifest/v1"
W3_SCHEMA = "jope.h65.w3-fold-manifest/v1"
PLAN_SCHEMA = "jope.h65.w3-preparation-plan/v1"
PACKET_FILE_SHA256 = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
PACKET_PAYLOAD_SHA256 = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
EXPECTED_CONTEXTS = 1219
EXPECTED_FAMILIES = 205
EXPECTED_FOLDS = 5
EXPECTED_FOLD_SEED = 4900
DOPE_PREFIX = "c51a56d0"
SEEDS = [701, 702, 703]
SMOKE_BUCKET = "veox-jopedime-datasets-use2"
SMOKE_CATALOG_SHA256 = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
SMOKE_DATASET_ID = "b9be38a2d975279c"
SMOKE_CANONICAL_SHA256 = "b9be38a2d975279cf8104003fae130c7282d8262d099c3b08a1a32b235925374"
SMOKE_TRAIN_SHA256 = "b777779435c9e4685735e95ce41a9e28e14e01eb4e2de5cb685219f65414c4d8"
SMOKE_TRAIN_ROWS = 85
SMOKE_TRAIN_COLUMNS = 10


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    try:
        return json.loads(raw), raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid UTF-8 JSON") from exc


def _strings(values: Any, where: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{where} must be a list of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{where} contains duplicates")
    return values


def project_fold_groups(
    folds: Any, expected_contexts: int, expected_families: int
) -> list[dict[str, Any]]:
    """Validate the family partition and retain only its training context/family identities."""
    if not isinstance(folds, list) or not folds:
        raise ValueError("folds must be a nonempty list")
    seen_contexts: set[str] = set()
    seen_families: set[str] = set()
    projected: list[dict[str, Any]] = []
    for expected_index, fold in enumerate(folds):
        if not isinstance(fold, dict) or fold.get("fold_index") != expected_index:
            raise ValueError("W1 fold indices must be ordered and contiguous from zero")
        contexts = _strings(fold.get("held_contexts"), f"fold {expected_index} held_contexts")
        families = _strings(fold.get("held_families"), f"fold {expected_index} held_families")
        if fold.get("held_context_count") != len(contexts):
            raise ValueError(f"fold {expected_index} held context count mismatch")
        if fold.get("held_family_count") != len(families):
            raise ValueError(f"fold {expected_index} held family count mismatch")
        fit_count = fold.get("fit_context_count")
        if not isinstance(fit_count, int) or len(contexts) + fit_count != expected_contexts:
            raise ValueError(f"fold {expected_index} fit/held context total mismatch")
        if seen_contexts.intersection(contexts):
            raise ValueError("a training context is held in more than one fold")
        if seen_families.intersection(families):
            raise ValueError("a family is split across held folds")
        seen_contexts.update(contexts)
        seen_families.update(families)
        projected.append(
            {
                "fold_index": expected_index,
                "held_contexts": contexts,
                "held_families": families,
                "fit_context_count": fit_count,
                "held_context_count": len(contexts),
                "held_family_count": len(families),
            }
        )

    if len(seen_contexts) != expected_contexts:
        raise ValueError("held folds do not partition all DL-Train contexts")
    if len(seen_families) != expected_families:
        raise ValueError("held folds do not partition all DL-Train families")
    return projected


def build_fold_manifest(source: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    """Project the W1 train-only family partition without opening outcomes or row payloads."""
    if source.get("schema") != W1_SCHEMA:
        raise ValueError(f"expected {W1_SCHEMA}")
    expected = {
        "dl_train_contexts": EXPECTED_CONTEXTS,
        "dl_train_families": EXPECTED_FAMILIES,
        "dl_val_contexts_loaded": 0,
        "fold_seed": EXPECTED_FOLD_SEED,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"W1 source {key} must equal {value}")
    if not str(source.get("dope_sha256", "")).startswith(DOPE_PREFIX):
        raise ValueError("W1 source does not bind the frozen DOPE identity")
    if source.get("source_manifest_file_sha256") != PACKET_FILE_SHA256:
        raise ValueError("W1 source manifest file identity mismatch")
    if source.get("source_manifest_sha256") != PACKET_PAYLOAD_SHA256:
        raise ValueError("W1 source packet payload identity mismatch")
    alias_keys = ("cross_family_dataset_aliases", "cross_family_table_aliases")
    for key in alias_keys:
        if not isinstance(source.get(key), int) or source[key] < 0:
            raise ValueError(f"W1 source {key} must be a nonnegative count")
    if len(source.get("folds", [])) != EXPECTED_FOLDS:
        raise ValueError(f"W1 source must contain exactly {EXPECTED_FOLDS} folds")
    projected = project_fold_groups(source["folds"], EXPECTED_CONTEXTS, EXPECTED_FAMILIES)

    return {
        "schema": W3_SCHEMA,
        "wave": "W3",
        "training_role": "DL-Train",
        "source_w1_fold_manifest_sha256": source_sha256,
        "source_manifest_file_sha256": PACKET_FILE_SHA256,
        "source_manifest_sha256": PACKET_PAYLOAD_SHA256,
        "selected_file_sha256": source.get("selected_file_sha256"),
        "champions_file_sha256": source.get("champions_file_sha256"),
        "dope_sha256": source.get("dope_sha256"),
        "fused_source_sha256": source.get("fused_source_sha256"),
        "r882_checkpoint_sha256": source.get("r882_checkpoint_sha256"),
        "cross_family_dataset_aliases": source["cross_family_dataset_aliases"],
        "cross_family_table_aliases": source["cross_family_table_aliases"],
        "context_count": EXPECTED_CONTEXTS,
        "family_count": EXPECTED_FAMILIES,
        "fold_seed": EXPECTED_FOLD_SEED,
        "fold_count": EXPECTED_FOLDS,
        "dl_val_contexts_loaded": 0,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "folds": projected,
    }


def load_winner_config(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("W1 winner config must be a JSON object")
    return value, sha256_bytes(raw)


def make_plan(folds: dict[str, Any], winner_config: dict[str, Any], winner_sha256: str) -> dict[str, Any]:
    if folds.get("schema") != W3_SCHEMA or folds.get("training_role") != "DL-Train":
        raise ValueError("W3 plan requires the verified training-only H65 fold manifest")
    expected = {
        "context_count": EXPECTED_CONTEXTS,
        "family_count": EXPECTED_FAMILIES,
        "fold_seed": EXPECTED_FOLD_SEED,
        "fold_count": EXPECTED_FOLDS,
        "dl_val_contexts_loaded": 0,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "source_manifest_file_sha256": PACKET_FILE_SHA256,
        "source_manifest_sha256": PACKET_PAYLOAD_SHA256,
    }
    for key, value in expected.items():
        if folds.get(key) != value:
            raise ValueError(f"W3 fold manifest {key} must equal {value}")
    project_fold_groups(folds.get("folds"), EXPECTED_CONTEXTS, EXPECTED_FAMILIES)
    return {
        "schema": PLAN_SCHEMA,
        "preparation_only": True,
        "device": "cpu",
        "outcome_values_read": False,
        "winner_config_sha256": winner_sha256,
        "winner_config_schema": winner_config.get("schema"),
        "fold_manifest_sha256": sha256_bytes(
            json.dumps(folds, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "fold_seed": EXPECTED_FOLD_SEED,
        "fold_count": EXPECTED_FOLDS,
        "training_contexts": EXPECTED_CONTEXTS,
        "training_families": EXPECTED_FAMILIES,
        "experiment_seeds": SEEDS,
        "terminal_updates": 40,
        "gpu_or_native_fit_requested": False,
    }


def s3_stdout(key: str) -> bytes:
    """Read one pinned corpus object; never print data or AWS diagnostics."""
    result = subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            f"s3://{SMOKE_BUCKET}/{key}",
            "-",
            "--region",
            "us-east-2",
            "--only-show-errors",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=45,
        env={**os.environ, "AWS_EC2_METADATA_DISABLED": "true"},
    )
    if result.returncode:
        raise RuntimeError(f"anonymous S3 read failed for pinned object (exit {result.returncode})")
    return result.stdout


def catalog_smoke() -> dict[str, Any]:
    """Verify one tiny catalog-listed regression train blob; this is plumbing, not W3 data."""
    manifest_key = f"datasets/{SMOKE_DATASET_ID}/manifest.json"
    manifest_bytes = s3_stdout(manifest_key)
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("pinned public dataset manifest is invalid JSON") from exc
    if (
        manifest.get("id") != SMOKE_DATASET_ID
        or manifest.get("dataset_hash") != SMOKE_DATASET_ID
        or manifest.get("canonical_sha256") != SMOKE_CANONICAL_SHA256
        or manifest.get("task") != "regression"
        or manifest.get("origin") != "real"
    ):
        raise ValueError("S3 manifest does not match the pinned real regression catalog entry")
    train = manifest.get("files", {}).get("train", {})
    if (
        train.get("sha256") != SMOKE_TRAIN_SHA256
        or train.get("rows") != SMOKE_TRAIN_ROWS
        or train.get("columns") != SMOKE_TRAIN_COLUMNS
    ):
        raise ValueError("S3 manifest train descriptor does not match the catalog pin")

    # Only the train blob is requested. No test key is constructed or fetched.
    train_bytes = s3_stdout(f"blobs/sha256/{SMOKE_TRAIN_SHA256}")
    if hashlib.sha256(train_bytes).hexdigest() != SMOKE_TRAIN_SHA256:
        raise ValueError("downloaded train blob failed SHA256 verification")
    rows = list(csv.reader(io.StringIO(train_bytes.decode("utf-8"), newline="")))
    if len(rows) != SMOKE_TRAIN_ROWS or any(len(row) != SMOKE_TRAIN_COLUMNS for row in rows):
        raise ValueError("train blob failed the pinned rectangular CSV shape check")

    return {
        "schema": "jope.h65.s3-plumbing-smoke/v1",
        "mode": "dry-run",
        "purpose": "input-plumbing-only; not a W3 outcome row, fit, score, or result",
        "dataset_id": SMOKE_DATASET_ID,
        "catalog_version": "v246",
        "catalog_source_sha256": SMOKE_CATALOG_SHA256,
        "canonical_sha256": SMOKE_CANONICAL_SHA256,
        "task": "regression",
        "origin": "real",
        "manifest_key": manifest_key,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "train_blob_key": f"blobs/sha256/{SMOKE_TRAIN_SHA256}",
        "train_sha256": SMOKE_TRAIN_SHA256,
        "train_bytes": len(train_bytes),
        "train_rows": len(rows),
        "train_columns": SMOKE_TRAIN_COLUMNS,
        "test_blob_requested": False,
        "outcome_values_emitted": False,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if str(path) == "-":
        sys.stdout.write(encoded)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-folds", help="derive folds from the pinned W1 train-only manifest")
    prepare.add_argument("--w1-fold-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    plan = subparsers.add_parser("plan", help="emit a CPU-only plan bound to the W1 winner config")
    plan.add_argument("--fold-manifest", type=Path, required=True)
    plan.add_argument("--w1-winner-config", type=Path, required=True)
    plan.add_argument("--output", type=Path, default=Path("-"))
    smoke = subparsers.add_parser("smoke-s3", help="verify a tiny catalog-listed train-only S3 input")
    smoke.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare-folds":
            source, raw = read_json(args.w1_fold_manifest)
            if not isinstance(source, dict):
                raise ValueError("W1 fold manifest must be a JSON object")
            manifest = build_fold_manifest(source, sha256_bytes(raw))
            write_json(args.output, manifest)
        elif args.command == "plan":
            folds, _raw = read_json(args.fold_manifest)
            config, config_sha256 = load_winner_config(args.w1_winner_config)
            if not isinstance(folds, dict):
                raise ValueError("W3 fold manifest must be a JSON object")
            write_json(args.output, make_plan(folds, config, config_sha256))
        else:
            receipt = catalog_smoke()
            write_json(args.receipt, receipt)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
