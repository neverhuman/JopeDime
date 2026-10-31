#!/usr/bin/env python3
"""Validate the frozen H-wave candidate inputs without fitting or scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

DEV_CONTEXTS = 91
DEV_FAMILIES = 19
H38_METHOD = "jope-h38-f2smart"
H38_CONFIG = "F2_SMART_LM80_STD_SVD32"
H40B_METHOD = "jope-gov03-h40b-r782"
H40B_SHA256 = "427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_allowlisted_dev(selected_path: Path) -> dict[str, dict[str, Any]]:
    """Return only the exact GOV-03 `dev` roster; reject mixed or altered rosters."""
    selected = json.loads(selected_path.read_text())
    if not isinstance(selected, dict) or set(selected) != {"train", "dev", "protocol"}:
        raise ValueError("selected manifest must have exactly train/dev/protocol keys")
    dev = selected["dev"]
    if not isinstance(dev, list) or len(dev) != DEV_CONTEXTS:
        raise ValueError(f"expected exactly {DEV_CONTEXTS} allowlisted dev contexts")
    indexed: dict[str, dict[str, Any]] = {}
    families: set[str] = set()
    for row in dev:
        if not isinstance(row, dict):
            raise ValueError("dev context entries must be objects")
        context = row.get("context_sha256")
        family = row.get("family_id")
        if not isinstance(context, str) or not SHA256_RE.fullmatch(context):
            raise ValueError("invalid context hash in dev allowlist")
        if context in indexed:
            raise ValueError("duplicate context in dev allowlist")
        if row.get("dl_split") != "DL-Val" or row.get("source_split") not in {
            "train",
            "inner",
        }:
            raise ValueError("only source-train DL-Val contexts are allowed")
        if row.get("task") != "regression":
            raise ValueError("only regression dev contexts are allowed")
        if not isinstance(family, str) or not family:
            raise ValueError("missing family id in dev allowlist")
        table_sha = row.get("train_only_jtt1_sha256")
        if not isinstance(table_sha, str) or not SHA256_RE.fullmatch(table_sha):
            raise ValueError("invalid train-only table hash in dev allowlist")
        indexed[context] = row
        families.add(family)
    if len(families) != DEV_FAMILIES:
        raise ValueError(f"expected {DEV_FAMILIES} dev families, got {len(families)}")
    return indexed


def validate_training_tables(rows: dict[str, dict[str, Any]], table_dir: Path) -> int:
    """Hash-check only allowlisted, prepared training tables; never read outer rows."""
    checked: set[str] = set()
    for row in rows.values():
        expected = row["train_only_jtt1_sha256"]
        context = row["context_sha256"]
        if context in checked:
            continue
        path = table_dir / f"{context}.jtt1"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing allowlisted train-only table: {path.name}"
            )
        if sha256(path) != expected:
            raise ValueError(f"train-only table hash mismatch: {path.name}")
        checked.add(context)
    return len(checked)


def validate_checkpoint(path: Path, expected_sha256: str = H40B_SHA256) -> str:
    """Verify the GOV-03 r782 checkpoint identity without deserializing it."""
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned h40b checkpoint: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"checkpoint hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_prediction_rows(
    path: Path,
    *,
    method: str,
    config: str,
    allowlisted_contexts: set[str],
) -> dict[str, float]:
    """Load finite, per-context prediction/metric rows without extending the roster."""
    if method not in {H38_METHOD, H40B_METHOD}:
        raise ValueError(f"unsupported method: {method}")
    if method == H38_METHOD and config != H38_CONFIG:
        raise ValueError("H38 method must use the pinned F2 configuration")
    if method == H40B_METHOD and config != H40B_SHA256:
        raise ValueError("h40b method must use the pinned r782 checkpoint SHA")
    output: dict[str, float] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("method") != method or record.get("config") != config:
            raise ValueError(f"method/config mismatch on line {line_number}")
        context = record.get("context_sha256")
        if context not in allowlisted_contexts:
            raise ValueError(
                f"context outside GOV-03 dev allowlist on line {line_number}"
            )
        if context in output:
            raise ValueError(f"duplicate context on line {line_number}")
        value = record.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"invalid finite numeric value on line {line_number}")
        output[context] = float(value)
    return output


def run_self_test() -> None:
    """Exercise the row loader with synthetic records, without cohort access."""
    import unittest

    class LoaderTests(unittest.TestCase):
        def write_row(self, directory: str, context: str, value: Any) -> Path:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "method": H38_METHOD,
                        "config": H38_CONFIG,
                        "context_sha256": context,
                        "value": value,
                    }
                )
                + "\n"
            )
            return path

        def test_loads_pinned_h38_row(self):
            with tempfile.TemporaryDirectory() as temp:
                path = self.write_row(temp, "a" * 64, 0.25)
                self.assertEqual(
                    load_prediction_rows(
                        path,
                        method=H38_METHOD,
                        config=H38_CONFIG,
                        allowlisted_contexts={"a" * 64},
                    ),
                    {"a" * 64: 0.25},
                )

        def test_rejects_non_allowlisted_context(self):
            with tempfile.TemporaryDirectory() as temp:
                path = self.write_row(temp, "b" * 64, 0.25)
                with self.assertRaisesRegex(ValueError, "outside GOV-03 dev allowlist"):
                    load_prediction_rows(
                        path,
                        method=H38_METHOD,
                        config=H38_CONFIG,
                        allowlisted_contexts={"a" * 64},
                    )

        def test_rejects_non_finite_value(self):
            with tempfile.TemporaryDirectory() as temp:
                path = self.write_row(temp, "a" * 64, float("nan"))
                with self.assertRaisesRegex(ValueError, "finite numeric"):
                    load_prediction_rows(
                        path,
                        method=H38_METHOD,
                        config=H38_CONFIG,
                        allowlisted_contexts={"a" * 64},
                    )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LoaderTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run", action="store_true", help="validate inputs only; no scoring"
    )
    action.add_argument(
        "--self-test", action="store_true", help="run synthetic row-loader checks"
    )
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--training-tables", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if not (args.selected and args.training_tables and args.checkpoint):
        parser.error(
            "--dry-run requires --selected, --training-tables and --checkpoint"
        )
    if not args.dry_run:
        parser.error(
            "scoring is unavailable until both frozen candidate inference paths are authorized"
        )
    rows = load_allowlisted_dev(args.selected)
    tables = validate_training_tables(rows, args.training_tables)
    checkpoint_sha = validate_checkpoint(args.checkpoint)
    print(
        json.dumps(
            {
                "status": "CPU_PREFLIGHT_PASS",
                "contexts": len(rows),
                "families": len({row["family_id"] for row in rows.values()}),
                "train_only_tables_sha_checked": tables,
                "checkpoint_sha256": checkpoint_sha,
                "h38": {
                    "method": H38_METHOD,
                    "config": H38_CONFIG,
                    "inference_artifact": None,
                },
                "h40b": {
                    "method": H40B_METHOD,
                    "round": 782,
                    "inference": "not executed",
                },
                "predictions_scored": False,
                "sealed_holdout_meta_test_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
