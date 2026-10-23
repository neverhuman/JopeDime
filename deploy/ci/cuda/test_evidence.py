"""Missing, stale, skipped, truncated and altered CUDA evidence is never a pass."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cuda_evidence", Path(__file__).with_name("evidence.py"))
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"JOPE_SOURCE_SHA": "a" * 40, "GITHUB_RUN_ID": "17", "GITHUB_RUN_ATTEMPT": "2"})
        self.env.start()
        self.addCleanup(self.env.stop)
        (self.root / "source.txt").write_text("a" * 40 + "\n")
        (self.root / "mode.txt").write_text("compile\n")
        (self.root / "toolchains.txt").write_text("nvcc fixture identity\n")
        (self.root / "environment.txt").write_text("synthetic fixture configuration\n")
        for name in evidence.LOGS:
            (self.root / name).write_text("test result: ok. 5 passed; 0 failed; 0 ignored;\n")
            (self.root / name.replace(".log", ".list")).write_text(
                "".join(f"case_{index}: test\n" for index in range(5)) + "\n5 tests, 0 benchmarks\n")
        with (self.root / "census.log").open("a") as file:
            for kind in ("recipe", "bound", "legacy-bound", "v2"):
                item = {"kind": kind, "total": 2, "executed": ["a"], "refused": ["b"], "unaccounted": 0}
                file.write("CUDA C++ census: " + json.dumps(item) + "\n")

    def test_complete_compiler_evidence_keeps_device_execution_unknown(self):
        result = evidence.inspect(self.root, "compile")
        self.assertEqual(result["gpu_execution"], "unknown")
        self.assertEqual(result["device_prediction_batches"], 0)
        self.assertEqual(result["attempt"], "2")

    def test_missing_and_zero_test_execution_are_refused(self):
        contracts = self.root / "contracts.log"
        contracts.write_text("test result: ok. 1 passed; 0 failed; 0 ignored;\n")
        with self.assertRaisesRegex(ValueError, "selection"):
            evidence.inspect(self.root, "compile")
        contracts.write_text("test result: ok. 5 passed; 0 failed; 0 ignored;\n")
        path = self.root / "golden.log"
        path.write_text("test result: ok. 0 passed; 0 failed; 0 ignored;\n")
        with self.assertRaisesRegex(ValueError, "incomplete test"):
            evidence.inspect(self.root, "compile")
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            evidence.inspect(self.root, "compile")

    def test_ignored_tests_are_not_qualified(self):
        (self.root / "contracts.log").write_text("test result: ok. 5 passed; 0 failed; 1 ignored;\n")
        with self.assertRaisesRegex(ValueError, "incomplete test"):
            evidence.inspect(self.root, "compile")

    def test_stale_source_is_refused(self):
        (self.root / "source.txt").write_text("b" * 40)
        with self.assertRaisesRegex(ValueError, "source identity"):
            evidence.inspect(self.root, "compile")

    def test_missing_census_is_refused(self):
        path = self.root / "census.log"
        original = path.read_text()
        for invalid in (
            {"kind": "recipe", "total": 2, "executed": ["a", "a"], "refused": [], "unaccounted": 0},
            {"kind": "recipe", "total": 2, "executed": ["a"], "refused": ["a"], "unaccounted": 0},
        ):
            lines = original.splitlines()
            lines[1] = "CUDA C++ census: " + json.dumps(invalid)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "census"):
                evidence.inspect(self.root, "compile")
        (self.root / "census.log").write_text("test result: ok. 5 passed; 0 failed; 0 ignored;\n")
        with self.assertRaisesRegex(ValueError, "missing registry"):
            evidence.inspect(self.root, "compile")

    def test_compilation_cannot_be_relabelled_as_device_qualification(self):
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            evidence.inspect(self.root, "device")
        (self.root / "mode.txt").write_text("device")
        with self.assertRaisesRegex(ValueError, "actual device"):
            evidence.inspect(self.root, "device")
        (self.root / "device.txt").write_text("fixture only; not hardware evidence")
        with (self.root / "contracts.log").open("a") as file:
            file.write("CUDA_LINEAR_DEVICE_PASS rows=0\n")
        with self.assertRaisesRegex(ValueError, "actual device"):
            evidence.inspect(self.root, "device")

    def test_changed_logs_change_the_receipt_binding(self):
        before = evidence.inspect(self.root, "compile")
        with (self.root / "bindings.log").open("a") as file:
            file.write("altered bytes\n")
        self.assertNotEqual(before["files"], evidence.inspect(self.root, "compile")["files"])

    def test_duplicate_missing_and_empty_test_selections_are_refused(self):
        path = self.root / "contracts.list"
        for text in ("", "case: test\n" * 5):
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "selection"):
                evidence.inspect(self.root, "compile")
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            evidence.inspect(self.root, "compile")

    def test_device_receipt_binds_its_hardware_description(self):
        (self.root / "mode.txt").write_text("device")
        (self.root / "device.txt").write_text("synthetic unit-test fixture, not actual hardware")
        with (self.root / "contracts.log").open("a") as file:
            file.write("CUDA_LINEAR_DEVICE_PASS rows=3\n")
        before = evidence.inspect(self.root, "device")
        self.assertEqual(before["device_prediction_batches"], 1)
        (self.root / "device.txt").write_text("changed synthetic device identity")
        self.assertNotEqual(before["files"], evidence.inspect(self.root, "device")["files"])


if __name__ == "__main__":
    unittest.main()
