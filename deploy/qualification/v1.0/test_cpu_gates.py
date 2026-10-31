"""CPU qualification controller proofs; no Rust build, GPU, release or dataset jobs."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("cpu_gates_under_test", HERE / "cpu_gates.py")
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)

ONE_TEST = (
    "import os,unittest; "
    "assert os.environ['CUDA_VISIBLE_DEVICES']==''; "
    "assert 'AWS_ACCESS_KEY_ID' not in os.environ and 'GITHUB_TOKEN' not in os.environ; "
    "suite=unittest.TestSuite([unittest.FunctionTestCase(lambda:None)]); "
    "r=unittest.TextTestRunner().run(suite); raise SystemExit(not r.wasSuccessful())"
)


class Controller(unittest.TestCase):
    def setUp(self):
        base = Path(
            os.environ.get("JOPE_QUALIFICATION_TEST_TMP", ROOT / "target/qualification-controller-tests")
        )
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.repo = self.path / "repo"
        self.repo.mkdir()
        self.controls = mock.patch.object(cpu, "CONTROLS", {})
        self.controls.start()
        self.addCleanup(self.controls.stop)

    def source(self, commands):
        (self.repo / "deploy/ci").mkdir(parents=True)
        lanes = {}
        for name, command, backend in commands:
            lanes[name] = dict(
                command=command, result_backend=backend, requires_nonzero_tests=backend != "none"
            )
        for name in cpu.INPUTS:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        (self.repo / "deploy/ci/coverage.json").write_text(json.dumps(dict(lanes=lanes)))
        (self.repo / "deploy/ci/full-main.py").write_text("GROUPS = " + repr({"cpu": tuple(lanes)}) + "\n")
        (self.repo / "deploy/ci/results.py").write_bytes((ROOT / "deploy/ci/results.py").read_bytes())
        subprocess.run(["git", "init", "-q", "--initial-branch=main", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        head = cpu.git(self.repo, "rev-parse", "HEAD")
        subprocess.run(
            ["git", "-C", str(self.repo), "update-ref", "refs/remotes/origin/main", head], check=True
        )
        return head

    def run_checks(self, **kwargs):
        with mock.patch.object(
            cpu, "admission", return_value=dict(cpu_count=8, load1=0, disk_used_percent=1)
        ):
            # execute_gate's default is bound at definition, so inject admission at the call seam.
            original = cpu.execute_gate

            def gate(*args, **options):
                return original(
                    *args, **options, check=lambda _: dict(cpu_count=8, load1=0, disk_used_percent=1)
                )

            with mock.patch.object(cpu, "execute_gate", side_effect=gate):
                return cpu.run(self.repo, self.path / "evidence", execute=True, **kwargs)

    def test_real_successful_child_schema_hashes_and_no_release_promotion(self):
        head = self.source([("full-positive", [sys.executable, "-c", ONE_TEST], "unittest")])
        with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "unit-only", "GITHUB_TOKEN": "unit-only"}):
            report = self.run_checks(expected=head, jobs=32)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checks"][0]["tests"]["executed"], 1)
        self.assertEqual(report["environment"]["CARGO_BUILD_JOBS"], "8")
        self.assertEqual(len(report["release_gates"]), 29)
        self.assertTrue(all(x["status"] == "not-qualified" for x in report["release_gates"].values()))
        self.assertIsNone(report["signed_candidate"])
        self.assertIsNone(report["owner_approval"])
        self.assertFalse(report["release_qualified"])
        log = self.path / "evidence/full-positive/command.log"
        self.assertEqual(report["checks"][0]["outputs"]["command.log"]["sha256"], cpu.digest(log))
        schema = json.loads((HERE / "qualification_receipt.schema.json").read_text())
        jsonschema.Draft202012Validator(schema).validate(report)
        report["release_qualified"] = True
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(report)

    def test_failure_continues_to_later_gate_and_keeps_original_exit(self):
        self.source(
            [
                ("full-a-failure", [sys.executable, "-c", "raise SystemExit(9)"], "none"),
                ("full-b-success", [sys.executable, "-c", ONE_TEST], "unittest"),
            ]
        )
        report = self.run_checks()
        self.assertEqual(report["status"], "failed")
        self.assertEqual([x["exit_status"] for x in report["checks"]], [9, 0])
        self.assertEqual(report["checks"][1]["tests"]["passed"], 1)

    def test_timeout_is_not_pass_and_later_gate_runs(self):
        self.source(
            [
                ("full-a-timeout", [sys.executable, "-c", "import time; time.sleep(30)"], "none"),
                ("full-b-success", [sys.executable, "-c", ONE_TEST], "unittest"),
            ]
        )
        report = self.run_checks(seconds=1.0)
        self.assertEqual(report["checks"][0]["status"], "timed-out")
        self.assertEqual(report["checks"][0]["exit_status"], 124)
        self.assertEqual(report["checks"][1]["status"], "passed")

    def test_zero_tests_missing_summaries_and_skips_are_not_complete_passes(self):
        self.source(
            [
                (
                    "full-a-zero",
                    [
                        sys.executable,
                        "-c",
                        "import unittest; unittest.TextTestRunner().run(unittest.TestSuite())",
                    ],
                    "unittest",
                ),
                ("full-b-missing", [sys.executable, "-c", "print('no execution summary')"], "unittest"),
                (
                    "full-c-skip",
                    [
                        sys.executable,
                        "-c",
                        "import unittest; c=unittest.FunctionTestCase(unittest.skip('fixture')(lambda:None)); unittest.TextTestRunner().run(unittest.TestSuite([c, unittest.FunctionTestCase(lambda:None)]))",
                    ],
                    "unittest",
                ),
            ]
        )
        report = self.run_checks()
        self.assertEqual([x["status"] for x in report["checks"]], ["failed", "failed", "incomplete"])
        self.assertEqual(report["checks"][0]["exit_status"], 0)
        self.assertIn(report["checks"][0]["tests"]["executed"], (None, 0))
        self.assertTrue(report["checks"][0]["diagnostic"])
        self.assertIsNone(report["checks"][1]["tests"]["executed"])
        self.assertEqual(report["checks"][2]["tests"]["skipped"], 1)

    def test_source_mismatch_dirty_source_and_reused_output_refuse_before_execution(self):
        head = self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        with self.assertRaisesRegex(ValueError, "explicitly expected"):
            cpu.run(self.repo, self.path / "wrong", expected="0" * 40, execute=True)
        (self.repo / "Cargo.lock").write_text("modified\n")
        with self.assertRaisesRegex(ValueError, "clean checkout"):
            cpu.run(self.repo, self.path / "dirty", expected=head, execute=True)
        self.assertFalse((self.path / "dirty").exists())
        subprocess.run(["git", "-C", str(self.repo), "restore", "Cargo.lock"], check=True)
        self.run_checks()
        with self.assertRaises(FileExistsError):
            self.run_checks()

    def test_inventory_mismatch_and_symlinked_output_refuse(self):
        self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        (self.repo / "deploy/ci/full-main.py").write_text("GROUPS = {'cpu': ('full-missing',)}\n")
        with self.assertRaisesRegex(ValueError, "partition"):
            cpu.inventory(self.repo)
        link = self.path / "linked"
        link.symlink_to(self.path / "missing", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            cpu.run(self.repo, link)

    def test_missing_executable_is_unavailable_with_original_exit(self):
        self.source([("full-missing", ["jope-rel07-no-such-executable-5f02"], "none")])
        report = self.run_checks()
        self.assertEqual(report["checks"][0]["status"], "unavailable")
        self.assertEqual(report["checks"][0]["exit_status"], 127)
        self.assertIsNone(report["checks"][0]["tests"]["executed"])

    def test_interruption_reaps_only_owned_group_and_retains_unqualified_checkpoint(self):
        process = mock.Mock(pid=314159)
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt, 0]
        with mock.patch.object(cpu.subprocess, "Popen", return_value=process):
            with mock.patch.object(cpu.os, "killpg") as stop:
                with self.assertRaises(KeyboardInterrupt):
                    cpu.invoke(["unused"], self.repo, self.path / "cancel.log", {}, 1)
        stop.assert_called_once_with(process.pid, cpu.signal.SIGTERM)
        self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        with mock.patch.object(cpu, "execute_gate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                cpu.run(self.repo, self.path / "cancelled", execute=True)
        receipt = json.loads((self.path / "cancelled/qualification_receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertTrue(receipt["interrupted"])
        self.assertFalse(receipt["release_qualified"])

    def test_plan_uses_frozen_main_inventory_not_dirty_feature_files(self):
        head = self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        path = self.repo / "deploy/ci/coverage.json"
        original = cpu.digest(path)
        path.write_text("{}")
        report = cpu.run(self.repo, self.path / "frozen-plan", expected=head)
        self.assertEqual([x["id"] for x in report["inventory"]], ["full-only"])
        self.assertEqual(report["inputs"]["deploy/ci/coverage.json"], original)
        self.assertFalse(report["source"]["clean_at_main"])
        self.assertEqual(report["checks"], [])

    def test_plan_has_every_family_and_runs_nothing(self):
        self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        with mock.patch.object(cpu, "invoke", side_effect=AssertionError("plan executed a command")):
            report = cpu.run(self.repo, self.path / "planned")
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["checks"], [])
        self.assertEqual([x["id"] for x in report["inventory"]], ["full-only"])

    def test_coverage_requires_a_backend_for_nonzero_tests(self):
        self.source([("full-only", [sys.executable, "-c", "pass"], "unittest")])
        path = self.repo / "deploy/ci/coverage.json"
        data = json.loads(path.read_text())
        data["lanes"]["full-only"]["result_backend"] = "none"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "requires a result backend"):
            cpu.inventory(self.repo)


if __name__ == "__main__":
    unittest.main()
