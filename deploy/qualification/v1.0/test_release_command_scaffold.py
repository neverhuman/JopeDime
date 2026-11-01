"""Controls for the non-executable REL-07 release command scaffold."""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import copy
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCAFFOLD = HERE / "release-command-scaffold.json"
RUNNER_PATH = HERE / "release-gates"


def load_runner():
    loader = SourceFileLoader("release_gates_scaffold_test", str(RUNNER_PATH))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


class ReleaseCommandScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()
        cls.scaffold = json.loads(SCAFFOLD.read_text(encoding="utf-8"))

    def test_every_release_gate_has_exactly_one_provider_command(self):
        self.assertEqual(self.scaffold["schema"], "jope.rel07.gate-command-scaffold/v1")
        self.assertEqual(self.scaffold["qualification_handoff"], "NOT_EXECUTABLE")
        rows = self.scaffold["gates"]
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), 28)
        self.assertEqual(Counter(ids), Counter(self.runner.RELEASE_GATE_IDS))
        for row in rows:
            self.assertEqual(set(row), {"id", "argv", "timeout_seconds"})
            expected = f"deploy/qualification/v1.0/providers/{row['id'].lower()}.py"
            self.assertEqual(row["argv"].count(expected), 1)
            self.assertLessEqual(row["timeout_seconds"], 3600)
            self.assertTrue(row["argv"])
            self.assertNotIn(row["argv"][0], {"sh", "bash", "/bin/sh", "/bin/bash"})

        discovered, missing = self.runner.provider_commands(ROOT)
        self.assertEqual(missing, [])
        expected_argv = {row["id"]: row["argv"] for row in discovered}
        actual_argv = {row["id"]: row["argv"] for row in rows}
        expected_argv["A5"] = [
            "/usr/bin/env",
            "JOPE_REL07_A5_MANIFEST=${JOPE_REL07_A5_MANIFEST}",
            "JOPE_REL07_A5_MANIFEST_SHA256=${JOPE_REL07_A5_MANIFEST_SHA256}",
            *expected_argv["A5"],
        ]
        expected_argv["D1"] += [
            "--macos-receipt",
            "${JOPE_REL07_D1_MACOS_RECEIPT}",
            "--macos-receipt-sha256",
            "${JOPE_REL07_D1_MACOS_RECEIPT_SHA256}",
            "--macos-artifact-sha256",
            "${JOPE_REL07_D1_MACOS_ARTIFACT_SHA256}",
        ]
        expected_argv["M2"] += [
            "--golden-dir",
            "${JOPE_GOLDEN_DIR}",
            "--golden-index-sha256",
            "${JOPE_GOLDEN_INDEX_SHA256}",
            "--parity-bundle-dir",
            "${JOPE_PARITY_BUNDLE_DIR}",
            "--parity-report-dir",
            "${JOPE_PARITY_REPORT_DIR}",
        ]
        expected_argv["M5"] += [
            "--default-off-receipt",
            "${JOPE_REL07_M5_DEFAULT_OFF_RECEIPT}",
            "--default-off-receipt-sha256",
            "${JOPE_REL07_M5_DEFAULT_OFF_RECEIPT_SHA256}",
        ]
        self.assertEqual(actual_argv, expected_argv)

    def test_only_candidate_and_dsc01_inputs_are_placeholders(self):
        allowed = {
            "JOPE_REL07_ARTIFACT_SHA256",
            "JOPE_REL07_A5_MANIFEST",
            "JOPE_REL07_A5_MANIFEST_SHA256",
            "JOPE_REL07_D1_MACOS_RECEIPT",
            "JOPE_REL07_D1_MACOS_RECEIPT_SHA256",
            "JOPE_REL07_D1_MACOS_ARTIFACT_SHA256",
            "JOPE_GOLDEN_DIR",
            "JOPE_GOLDEN_INDEX_SHA256",
            "JOPE_PARITY_BUNDLE_DIR",
            "JOPE_PARITY_REPORT_DIR",
            "JOPE_REL07_M5_DEFAULT_OFF_RECEIPT",
            "JOPE_REL07_M5_DEFAULT_OFF_RECEIPT_SHA256",
        }
        refs = set(re.findall(r"\$\{([A-Z0-9_]+)\}", json.dumps(self.scaffold)))
        self.assertEqual(refs, allowed)
        self.assertEqual(
            self.scaffold["candidate_sha256"], "${JOPE_REL07_ARTIFACT_SHA256}"
        )
        per_gate = {
            row["id"]: json.dumps(row["argv"]) for row in self.scaffold["gates"]
        }
        self.assertIn("JOPE_REL07_A5_MANIFEST", per_gate["A5"])
        self.assertIn("JOPE_REL07_D1_MACOS_RECEIPT", per_gate["D1"])
        self.assertIn("JOPE_REL07_D1_MACOS_ARTIFACT_SHA256", per_gate["D1"])
        self.assertIn("JOPE_GOLDEN_INDEX_SHA256", per_gate["M2"])
        self.assertIn("JOPE_REL07_M5_DEFAULT_OFF_RECEIPT", per_gate["M5"])

    def test_unresolved_scaffold_fails_closed_in_release_runner(self):
        with self.assertRaisesRegex(ValueError, "unsupported gate-command schema"):
            self.runner.manifest(SCAFFOLD)

        # Even if a caller relabels the scaffold as executable, the candidate
        # digest placeholder cannot pass the runner's strict v1 manifest check.
        executable = copy.deepcopy(self.scaffold)
        executable["schema"] = "jope.rel07.gate-commands/v1"
        executable["source_sha"] = "0" * 40
        with self.subTest("candidate digest placeholder"):
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(
                mode="w", suffix=".json", encoding="utf-8"
            ) as stream:
                json.dump(executable, stream)
                stream.flush()
                with self.assertRaisesRegex(ValueError, "invalid artifact_sha256"):
                    self.runner.manifest(Path(stream.name))


if __name__ == "__main__":
    unittest.main()
