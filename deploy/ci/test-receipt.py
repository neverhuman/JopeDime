#!/usr/bin/env python3
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("receipt", ROOT / "deploy/ci/receipt.py")
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


class ReceiptTests(unittest.TestCase):
    def test_identity_distinguishes_source_and_merge_checkout(self):
        env = dict(GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="2", GITHUB_SHA="b" * 40, JOPE_SOURCE_SHA="a" * 40)
        actual = receipt.identity(env, "b" * 40)
        self.assertEqual("a" * 40, actual["source_sha"])
        self.assertEqual("b" * 40, actual["checkout_sha"])
        with self.assertRaises(ValueError):
            receipt.identity(env, "c" * 40)
        env["GITHUB_RUN_ATTEMPT"] = "0"
        with self.assertRaises(ValueError):
            receipt.identity(env, "b" * 40)

    def test_unknown_skips_are_not_zero(self):
        counts = dict(collected=None, executed=2, passed=2, failed=0, skipped=None, excluded=None)
        receipt.validate_counts(counts)
        self.assertIsNone(counts["skipped"])
        counts.update(collected=3, skipped=0, excluded=0)
        with self.assertRaises(ValueError):
            receipt.validate_counts(counts)

    def test_failure_is_retained_and_cannot_reuse_output(self):
        with tempfile.TemporaryDirectory() as scratch, patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}):
            output = Path(scratch) / "lane"
            code = receipt.run_lane("hygiene", [sys.executable, "-c", "print('failure fixture'); raise SystemExit(23)"], output)
            self.assertEqual(23, code)
            result = json.loads((output / "receipt.json").read_text())
            self.assertEqual("failed", result["status"])
            self.assertEqual(23, result["exit_status"])
            self.assertEqual(receipt.digest(output / "command.log"), result["evidence"]["command.log"])
            with self.assertRaises(FileExistsError):
                receipt.run_lane("hygiene", [sys.executable, "-c", "pass"], output)

    def test_missing_capability_never_passes(self):
        with tempfile.TemporaryDirectory() as scratch, patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}):
            output = Path(scratch) / "lane"
            self.assertEqual(1, receipt.run_lane("hygiene", ["/does-not-exist/jope-proof"], output))
            self.assertEqual("unavailable", json.loads((output / "receipt.json").read_text())["status"])

    def test_stale_counts_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as scratch, patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}):
            root = Path(scratch)
            counts = root / "counts.json"
            counts.write_text('{}')
            output = root / "lane"
            self.assertEqual(1, receipt.run_lane("hygiene", [sys.executable, "-c", "raise SystemExit('must not run')"], output, counts))
            self.assertFalse((output / "command.log").exists())
            self.assertIn("already exists", json.loads((output / "receipt.json").read_text())["diagnostic"])


if __name__ == "__main__":
    unittest.main()
