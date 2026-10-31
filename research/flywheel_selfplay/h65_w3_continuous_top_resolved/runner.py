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
COMPARATOR_SCHEMA = "jope.h65.w3-comparator-binding/v1"
ADMISSION_SCHEMA = "jope.h65.w3-launch-admission/v1"
PACKET_FILE_SHA256 = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
PACKET_PAYLOAD_SHA256 = (
    "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
)
EXPECTED_CONTEXTS = 1219
EXPECTED_FAMILIES = 205
EXPECTED_FOLDS = 5
EXPECTED_FOLD_SEED = 4900
DOPE_PREFIX = "c51a56d0"
COMPARATOR_ID = "GBM_PROG412_CTX"
COMPARATOR_METRICS_PATH = "../h108_w1_prior_residual/metrics.json"
COMPARATOR_METRICS_SHA256 = (
    "530e1cd634916c59b84dde9f93fae4988256c09f50c7e970f59491737b5d8903"
)
H59_DISPOSITION_PATH = "../h89_w3_completion_admission/metrics.json"
H59_DISPOSITION_SHA256 = (
    "586e1e1104617dc412e3375ffc9dd69091dde3a0998c0303e4f57a44898aaf32"
)
H113_MANIFEST_PATH = "../h113_w3_continuous/manifest.json"
H113_PREREG_PATH = "../h113_w3_continuous/PREREG.md"
H113_PREREG_SHA256 = "c1fee49929ea039f4c06ede165d24af694e0e310ad07aab4052e26cc64c1c1a1"
COMPARATOR_BINDING_SHA256 = (
    "23d177302d23cdfeb665de4922f2b966284c151e44e1c89c3b3d1c80c36c4f71"
)
SEEDS = [701, 702, 703]
SMOKE_BUCKET = "veox-jopedime-datasets-use2"
SMOKE_CATALOG_SHA256 = (
    "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
)
SMOKE_DATASET_ID = "b9be38a2d975279c"
SMOKE_CANONICAL_SHA256 = (
    "b9be38a2d975279cf8104003fae130c7282d8262d099c3b08a1a32b235925374"
)
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
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
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
        contexts = _strings(
            fold.get("held_contexts"), f"fold {expected_index} held_contexts"
        )
        families = _strings(
            fold.get("held_families"), f"fold {expected_index} held_families"
        )
        if fold.get("held_context_count") != len(contexts):
            raise ValueError(f"fold {expected_index} held context count mismatch")
        if fold.get("held_family_count") != len(families):
            raise ValueError(f"fold {expected_index} held family count mismatch")
        fit_count = fold.get("fit_context_count")
        if (
            not isinstance(fit_count, int)
            or len(contexts) + fit_count != expected_contexts
        ):
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
    projected = project_fold_groups(
        source["folds"], EXPECTED_CONTEXTS, EXPECTED_FAMILIES
    )

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


def validate_comparator_binding(value: Any) -> None:
    if not isinstance(value, dict) or value.get("schema") != COMPARATOR_SCHEMA:
        raise ValueError(f"comparator binding must use {COMPARATOR_SCHEMA}")
    disposition = value.get("w1_disposition")
    if not isinstance(disposition, dict) or disposition.get("status") != "NO_ADVANCE":
        raise ValueError("W1 disposition must remain NO_ADVANCE")
    if (
        disposition.get("accepted_winner") is not None
        or value.get("prior_binding") is not None
    ):
        raise ValueError(
            "NO_ADVANCE cannot be converted into an accepted winner or prior"
        )
    comparator = value.get("strongest_comparator")
    if not isinstance(comparator, dict) or comparator.get("id") != COMPARATOR_ID:
        raise ValueError(f"strongest comparator must be {COMPARATOR_ID}")
    if comparator.get("label") != "context GBM (ungated)":
        raise ValueError(
            "strongest comparator label must match the owner-directed binding"
        )
    metrics = comparator.get("source_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("comparator source metrics must be pinned")
    if (
        metrics.get("path") != COMPARATOR_METRICS_PATH
        or metrics.get("sha256") != COMPARATOR_METRICS_SHA256
    ):
        raise ValueError("comparator source metrics identity mismatch")


def load_comparator_binding(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = read_json(path)
    validate_comparator_binding(value)
    source_path = (path.parent / COMPARATOR_METRICS_PATH).resolve()
    if (
        not source_path.is_file()
        or sha256_bytes(source_path.read_bytes()) != COMPARATOR_METRICS_SHA256
    ):
        raise ValueError("pinned comparator metrics file is missing or changed")
    return value, sha256_bytes(raw)


def load_h59_disposition(base_dir: Path) -> str:
    path = (base_dir / H59_DISPOSITION_PATH).resolve()
    if not path.is_file():
        raise ValueError("pinned H59 disposition metrics are missing")
    value, raw = read_json(path)
    if sha256_bytes(raw) != H59_DISPOSITION_SHA256:
        raise ValueError("pinned H59 disposition metrics identity mismatch")
    if not isinstance(value, dict) or not str(
        value.get("h59_disposition", "")
    ).startswith("COMPLETED_NO_ADMISSION"):
        raise ValueError("H59 disposition must be completed with no admission")
    return H59_DISPOSITION_SHA256


def load_preregistration(base_dir: Path, binding_sha256: str) -> str:
    manifest_path = (base_dir / H113_MANIFEST_PATH).resolve()
    prereg_path = (base_dir / H113_PREREG_PATH).resolve()
    manifest, _raw = read_json(manifest_path)
    if (
        not prereg_path.is_file()
        or sha256_bytes(prereg_path.read_bytes()) != H113_PREREG_SHA256
    ):
        raise ValueError("pinned H113 preregistration is missing or changed")
    if (
        not isinstance(manifest, dict)
        or manifest.get("prereg_sha256") != H113_PREREG_SHA256
    ):
        raise ValueError("H113 manifest does not bind the pinned preregistration")
    if manifest.get("w1_prior_binding") is not None:
        raise ValueError("H113 must keep the W1 prior binding null")
    disposition = manifest.get("w1_disposition")
    if not isinstance(disposition, dict) or disposition.get("status") != "NO_ADVANCE":
        raise ValueError("H113 must bind the W1 NO_ADVANCE disposition")
    comparator = disposition.get("strongest_comparator")
    if (
        not isinstance(comparator, dict)
        or comparator.get("id") != COMPARATOR_ID
        or comparator.get("label") != "context GBM (ungated)"
        or comparator.get("metrics_sha256") != COMPARATOR_METRICS_SHA256
        or comparator.get("binding_sha256") != binding_sha256
        or comparator.get("disposition") != "COMPARATOR_ONLY_NOT_A_WINNER_OR_PRIOR"
        or binding_sha256 != COMPARATOR_BINDING_SHA256
    ):
        raise ValueError("H113 manifest comparator binding identity mismatch")
    prereg_text = prereg_path.read_text(encoding="utf-8")
    if (
        "GBM_PROG412_CTX" not in prereg_text
        or "The W1 prior remains null." not in prereg_text
    ):
        raise ValueError(
            "H113 preregistration must state the comparator and null W1 prior"
        )
    return H113_PREREG_SHA256


def make_plan(
    folds: dict[str, Any], comparator_binding: dict[str, Any], binding_sha256: str
) -> dict[str, Any]:
    validate_comparator_binding(comparator_binding)
    if folds.get("schema") != W3_SCHEMA or folds.get("training_role") != "DL-Train":
        raise ValueError(
            "W3 plan requires the verified training-only H65 fold manifest"
        )
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
    comparator = comparator_binding["strongest_comparator"]
    return {
        "schema": PLAN_SCHEMA,
        "preparation_only": True,
        "device": "cpu",
        "outcome_values_read": False,
        "w1_disposition": "NO_ADVANCE",
        "accepted_w1_winner": None,
        "w1_prior_binding": None,
        "comparator_binding_sha256": binding_sha256,
        "strongest_comparator_id": comparator["id"],
        "strongest_comparator_label": comparator["label"],
        "comparator_metrics_sha256": comparator["source_metrics"]["sha256"],
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
        "launch_authorized": False,
    }


def make_admission_manifest(
    folds: dict[str, Any],
    comparator_binding: dict[str, Any],
    binding_sha256: str,
    h59_sha256: str,
    prereg_sha256: str,
) -> dict[str, Any]:
    """Emit CPU preparation status without manufacturing any launch admission evidence."""
    plan = make_plan(folds, comparator_binding, binding_sha256)
    return {
        "schema": ADMISSION_SCHEMA,
        "status": "BLOCKED",
        "launch_authorized": False,
        "device": "cpu",
        "w1_disposition": "NO_ADVANCE",
        "accepted_w1_winner": None,
        "w1_prior_binding": None,
        "strongest_comparator": {
            "id": plan["strongest_comparator_id"],
            "label": plan["strongest_comparator_label"],
            "metrics_sha256": plan["comparator_metrics_sha256"],
            "binding_sha256": plan["comparator_binding_sha256"],
            "status": "BOUND_FOR_COMPARISON_ONLY",
        },
        "h59_disposition": "COMPLETED_NO_ADMISSION_INCORPORATED",
        "h59_disposition_sha256": h59_sha256,
        "fold_manifest_sha256": plan["fold_manifest_sha256"],
        "preregistration_sha256": prereg_sha256,
        "outcome_export_sha256": None,
        "blockers": [
            {"id": "B1", "status": "OPEN", "evidence_sha256": None},
            {"id": "B2", "status": "OPEN", "evidence_sha256": None},
            {"id": "B3", "status": "OPEN", "evidence_sha256": None},
            {"id": "B4", "status": "OPEN", "evidence_sha256": None},
            {"id": "B5", "status": "OPEN", "evidence_sha256": None},
        ],
        "metrics": {"Elo": None, "val_reg_inner": None, "U4": None},
        "native_fits": 0,
        "optimizer_updates": 0,
        "gpu_seconds": 0,
    }


def validate_admission_manifest(value: dict[str, Any]) -> None:
    if value.get("schema") != ADMISSION_SCHEMA:
        raise ValueError("W3 admission manifest schema mismatch")
    if value.get("launch_authorized") is not False or value.get("status") != "BLOCKED":
        raise ValueError("CPU preparation cannot authorize W3 launch")
    if value.get("w1_disposition") != "NO_ADVANCE":
        raise ValueError("W1 disposition must remain NO_ADVANCE")
    if (
        value.get("accepted_w1_winner") is not None
        or value.get("w1_prior_binding") is not None
    ):
        raise ValueError("W1 NO_ADVANCE must not be represented as a winner or prior")
    if value.get("h59_disposition") != "COMPLETED_NO_ADMISSION_INCORPORATED":
        raise ValueError(
            "H59 disposition must reflect H89's completed no-admission result"
        )
    if value.get("h59_disposition_sha256") != H59_DISPOSITION_SHA256:
        raise ValueError("H59 disposition metrics identity mismatch")
    if value.get("preregistration_sha256") != H113_PREREG_SHA256:
        raise ValueError("H113 preregistration identity mismatch")
    if value.get("outcome_export_sha256") is not None:
        raise ValueError("no outcome export is admitted by CPU preparation")
    comparator = value.get("strongest_comparator")
    if comparator != {
        "id": COMPARATOR_ID,
        "label": "context GBM (ungated)",
        "metrics_sha256": COMPARATOR_METRICS_SHA256,
        "binding_sha256": COMPARATOR_BINDING_SHA256,
        "status": "BOUND_FOR_COMPARISON_ONLY",
    }:
        raise ValueError("strongest comparator binding mismatch")
    blockers = value.get("blockers")
    if (
        not isinstance(blockers, list)
        or len(blockers) != 5
        or any(not isinstance(item, dict) for item in blockers)
    ):
        raise ValueError("W3 admission manifest must retain the ordered B1-B5 census")
    if [item.get("id") for item in blockers] != [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    ]:
        raise ValueError("W3 admission manifest must retain the ordered B1-B5 census")
    expected_statuses = ["OPEN", "OPEN", "OPEN", "OPEN", "OPEN"]
    if [item.get("status") for item in blockers] != expected_statuses:
        raise ValueError(
            "W3 admission blocker state is not the CPU-preparation disposition"
        )
    if any(item.get("evidence_sha256") is not None for item in blockers):
        raise ValueError("open admission blockers must not claim evidence")
    if any(
        type(value.get(key)) is not int or value[key] != 0
        for key in ("native_fits", "optimizer_updates", "gpu_seconds")
    ):
        raise ValueError("CPU admission preparation cannot claim execution")
    if value.get("metrics") != {"Elo": None, "val_reg_inner": None, "U4": None}:
        raise ValueError("CPU admission preparation cannot claim scientific metrics")


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
        raise RuntimeError(
            f"anonymous S3 read failed for pinned object (exit {result.returncode})"
        )
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
        raise ValueError(
            "S3 manifest does not match the pinned real regression catalog entry"
        )
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
    if len(rows) != SMOKE_TRAIN_ROWS or any(
        len(row) != SMOKE_TRAIN_COLUMNS for row in rows
    ):
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
    prepare = subparsers.add_parser(
        "prepare-folds", help="derive folds from the pinned W1 train-only manifest"
    )
    prepare.add_argument("--w1-fold-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    plan = subparsers.add_parser(
        "plan", help="emit a CPU-only plan bound to the NO_ADVANCE comparator"
    )
    plan.add_argument("--fold-manifest", type=Path, required=True)
    plan.add_argument("--comparator-binding", type=Path, required=True)
    plan.add_argument("--output", type=Path, default=Path("-"))
    admission = subparsers.add_parser(
        "admission", help="emit the fail-closed W3 CPU admission manifest"
    )
    admission.add_argument("--fold-manifest", type=Path, required=True)
    admission.add_argument("--comparator-binding", type=Path, required=True)
    admission.add_argument("--output", type=Path, required=True)
    smoke = subparsers.add_parser(
        "smoke-s3", help="verify a tiny catalog-listed train-only S3 input"
    )
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
            binding, binding_sha256 = load_comparator_binding(args.comparator_binding)
            if not isinstance(folds, dict):
                raise ValueError("W3 fold manifest must be a JSON object")
            write_json(args.output, make_plan(folds, binding, binding_sha256))
        elif args.command == "admission":
            folds, _raw = read_json(args.fold_manifest)
            binding, binding_sha256 = load_comparator_binding(args.comparator_binding)
            if not isinstance(folds, dict):
                raise ValueError("W3 fold manifest must be a JSON object")
            h59_sha256 = load_h59_disposition(args.comparator_binding.parent)
            prereg_sha256 = load_preregistration(
                args.comparator_binding.parent, binding_sha256
            )
            manifest = make_admission_manifest(
                folds, binding, binding_sha256, h59_sha256, prereg_sha256
            )
            validate_admission_manifest(manifest)
            write_json(args.output, manifest)
        else:
            receipt = catalog_smoke()
            write_json(args.receipt, receipt)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
