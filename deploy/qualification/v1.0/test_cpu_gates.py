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
spec = importlib.util.spec_from_file_location(
    "cpu_gates_under_test", HERE / "cpu_gates.py"
)
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
            os.environ.get(
                "JOPE_QUALIFICATION_TEST_TMP",
                ROOT / "target/qualification-controller-tests",
            )
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

    def plan_metadata(self, status=True):
        root = self.path / "master"
        (root / "source").mkdir(parents=True)
        (root / "tasks").mkdir()
        (root / "status").mkdir()
        (root / "master_plan.md").write_text(
            "All 28 release gates + Q1\n"
            "| P021 | [TEL-01](tasks/T021-TEL-01-cpu.md) | CPU math | done merged #1 |\n"
        )
        (root / "source/spec-v1.md").write_text(
            "## 14. Release gates (28) and Q1\n"
            + "\n".join(
                f"| **{gate}** | Gate {gate} | Exact signed bytes |"
                for gate in cpu.GATE_IDS[:-1]
            )
            + "\n**Capability gate Q1 — learned guidance.** Q1 chooses the default.\n"
        )
        (root / "tasks/T021-TEL-01-cpu.md").write_text("| Gates closed | T1 |\n")
        if status:
            (root / "status/TEL-01.md").write_text(
                "# Historical PASS on a different source\n"
            )
        return root

    def source(self, commands):
        (self.repo / "deploy/ci").mkdir(parents=True)
        (self.repo / ".gitignore").write_text("__pycache__/\n")
        lanes = {}
        for name, command, backend in commands:
            lanes[name] = dict(
                command=command,
                result_backend=backend,
                requires_nonzero_tests=backend != "none",
            )
        for name in cpu.INPUTS:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        (self.repo / "deploy/ci/coverage.json").write_text(
            json.dumps(dict(lanes=lanes))
        )
        (self.repo / "deploy/ci/full-main.py").write_text(
            "GROUPS = " + repr({"cpu": tuple(lanes)}) + "\n"
        )
        (self.repo / "deploy/ci/results.py").write_bytes(
            (ROOT / "deploy/ci/results.py").read_bytes()
        )
        subprocess.run(
            ["git", "init", "-q", "--initial-branch=main", str(self.repo)], check=True
        )
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
            [
                "git",
                "-C",
                str(self.repo),
                "update-ref",
                "refs/remotes/origin/main",
                head,
            ],
            check=True,
        )
        return head

    def run_checks(self, **kwargs):
        with mock.patch.object(
            cpu,
            "admission",
            return_value=dict(cpu_count=8, load1=0, disk_used_percent=1),
        ):
            # execute_gate's default is bound at definition, so inject admission at the call seam.
            original = cpu.execute_gate

            def gate(*args, **options):
                return original(
                    *args,
                    **options,
                    check=lambda _: dict(cpu_count=8, load1=0, disk_used_percent=1),
                )

            with mock.patch.object(cpu, "execute_gate", side_effect=gate):
                return cpu.run(
                    self.repo, self.path / "evidence", execute=True, **kwargs
                )

    def test_real_successful_child_schema_hashes_and_no_release_promotion(self):
        head = self.source(
            [("full-positive", [sys.executable, "-c", ONE_TEST], "unittest")]
        )
        with mock.patch.dict(
            os.environ, {"AWS_ACCESS_KEY_ID": "unit-only", "GITHUB_TOKEN": "unit-only"}
        ):
            report = self.run_checks(expected=head, jobs=32)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checks"][0]["tests"]["executed"], 1)
        self.assertEqual(report["environment"]["CARGO_BUILD_JOBS"], "4")
        self.assertEqual(len(report["release_gates"]), 29)
        self.assertTrue(
            all(
                x["status"] == "not-qualified" for x in report["release_gates"].values()
            )
        )
        self.assertIsNone(report["signed_candidate"])
        self.assertIsNone(report["owner_approval"])
        self.assertFalse(report["release_qualified"])
        log = self.path / "evidence/full-positive/command.log"
        self.assertEqual(
            report["checks"][0]["outputs"]["command.log"]["sha256"], cpu.digest(log)
        )
        schema = json.loads((HERE / "qualification_receipt.schema.json").read_text())
        jsonschema.Draft202012Validator(schema).validate(report)
        report["release_qualified"] = True
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(report)

    def test_source_controls_require_regular_command_logs(self):
        modes = ["regular", "missing", "symlink", "directory"]
        if hasattr(os, "mkfifo"):
            modes.append("fifo")
        for mode in modes:
            with self.subTest(mode=mode):
                folder = self.path / ("log-" + mode)
                command = (
                    "import os,pathlib; "
                    "p=pathlib.Path(os.environ['JOPE_LANE_EVIDENCE'])/'command.log'; "
                )
                if mode != "regular":
                    command += "p.unlink(); "
                if mode == "symlink":
                    command += "p.symlink_to('scratch/sentinel.txt'); "
                elif mode == "directory":
                    command += "p.mkdir(); "
                elif mode == "fifo":
                    command += "os.mkfifo(p); "
                command += (
                    "(p.parent/'scratch'/'sentinel.txt').write_text('fixture only')"
                )
                gate = dict(
                    id="source-log-" + mode,
                    command=[sys.executable, "-c", command],
                    result_backend="none",
                    requires_nonzero_tests=False,
                )
                result = cpu.execute_gate(
                    gate,
                    self.repo,
                    folder,
                    self.path / "target",
                    1,
                    "a" * 40,
                    None,
                    5,
                    check=lambda root: {},
                )
                self.assertEqual(result["exit_status"], 0)
                if mode == "regular":
                    self.assertEqual(result["status"], "passed")
                    self.assertIsNone(result["diagnostic"])
                    self.assertEqual(result["outputs"]["command.log"]["bytes"], 0)
                    self.assertEqual(
                        result["outputs"]["command.log"]["sha256"],
                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    )
                    continue
                self.assertEqual(result["status"], "failed")
                self.assertIn(
                    "command log must be a regular file", result["diagnostic"]
                )
                self.assertNotIn("command.log", result["outputs"])
                if mode == "symlink":
                    self.assertIn("command.log", result["omitted_symlinks"])

    def test_artifact_read_errors_retain_receipts_and_continue(self):
        artifact = (
            "import os,pathlib; "
            "(pathlib.Path(os.environ['JOPE_LANE_EVIDENCE'])/'artifact.txt')"
            ".write_text('controlled fixture'); "
        )
        self.source(
            [
                ("full-a-log", [sys.executable, "-c", "pass"], "none"),
                ("full-b-output", [sys.executable, "-c", artifact], "none"),
                (
                    "full-c-child-failure",
                    [sys.executable, "-c", artifact + "raise SystemExit(9)"],
                    "none",
                ),
                ("full-d-positive", [sys.executable, "-c", ONE_TEST], "unittest"),
            ]
        )
        original_digest = cpu.digest
        failures = {
            "full-a-log": "command.log",
            "full-b-output": "artifact.txt",
            "full-c-child-failure": "artifact.txt",
        }

        def read_failure(path):
            # Inject only the per-file I/O boundary; real children, other hashes
            # and receipt persistence still execute, including when run as root.
            if failures.get(path.parent.name) == path.name:
                if path.parent.name == "full-c-child-failure":
                    raise FileNotFoundError("controlled disappeared evidence")
                raise PermissionError("controlled unreadable evidence")
            return original_digest(path)

        with mock.patch.object(cpu, "digest", side_effect=read_failure):
            report = self.run_checks()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            [check["exit_status"] for check in report["checks"]], [0, 0, 9, 0]
        )
        self.assertEqual(
            [check["status"] for check in report["checks"]],
            ["failed", "failed", "failed", "passed"],
        )
        self.assertEqual(report["checks"][-1]["tests"]["passed"], 1)
        for check in report["checks"][:3]:
            self.assertIn("cannot capture output", check["diagnostic"])
            self.assertIn(failures[check["id"]], check["diagnostic"])
            self.assertNotIn(failures[check["id"]], check["outputs"])
            receipt = self.path / "evidence" / check["id"] / "receipt.json"
            self.assertEqual(json.loads(receipt.read_text()), check)
        self.assertFalse(report["release_qualified"])
        print(
            "REL07_FIXTURE_ARTIFACT_FAILURES "
            + json.dumps(
                [
                    {
                        key: check[key]
                        for key in ("id", "exit_status", "status", "diagnostic")
                    }
                    for check in report["checks"]
                ]
            )
        )

    def test_failure_continues_to_later_gate_and_keeps_original_exit(self):
        self.source(
            [
                (
                    "full-a-failure",
                    [sys.executable, "-c", "raise SystemExit(9)"],
                    "none",
                ),
                ("full-b-success", [sys.executable, "-c", ONE_TEST], "unittest"),
            ]
        )
        report = self.run_checks()
        self.assertEqual(report["status"], "failed")
        self.assertEqual([x["exit_status"] for x in report["checks"]], [9, 0])
        self.assertEqual(report["checks"][1]["tests"]["passed"], 1)

    def test_nonfinite_cli_timeouts_refuse_before_output(self):
        head = self.source(
            [("full-only", [sys.executable, "-c", ONE_TEST], "unittest")]
        )
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                output = self.path / ("invalid-" + value)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "cpu_gates.py"),
                        "plan",
                        "--repo",
                        str(self.repo),
                        "--expected-main-sha",
                        head,
                        "--output",
                        str(output),
                        "--gate-timeout=" + value,
                    ],
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("gate timeout", result.stderr)
                self.assertFalse(output.exists())

    def test_nonfinite_execution_timeout_refuses_before_spawn(self):
        for index, seconds in enumerate((float("nan"), float("inf"), -float("inf"))):
            with self.subTest(seconds=seconds):
                log = self.path / f"invalid-api-{index}.log"
                with mock.patch.object(cpu.subprocess, "Popen") as spawn:
                    with self.assertRaisesRegex(ValueError, "gate timeout"):
                        cpu.invoke(["unused"], self.repo, log, {}, seconds)
                    spawn.assert_not_called()
                self.assertFalse(log.exists())

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "Linux process-state control"
    )
    def test_timeout_removes_descendants_after_leader_exits_and_continues(self):
        state = self.path / "owned-descendant.json"
        child = (
            "import json,os,signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),"
            "pgid=os.getpgrp(),start=Path('/proc/self/stat').read_text().split()[21]))); "
            "time.sleep(60)"
        )
        leader = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]); "
            "time.sleep(60)"
        )
        self.source(
            [
                (
                    "full-a-descendant",
                    [sys.executable, "-c", leader, str(state)],
                    "none",
                ),
                ("full-b-success", [sys.executable, "-c", ONE_TEST], "unittest"),
            ]
        )

        def alive(info):
            try:
                fields = Path(f"/proc/{info['pid']}/stat").read_text().split()
                return fields[21] == info["start"] and fields[2] != "Z"
            except FileNotFoundError:
                return False

        try:
            report = self.run_checks(seconds=2)
            self.assertTrue(state.exists(), "owned descendant did not reach readiness")
            info = json.loads(state.read_text())
            self.assertEqual(report["checks"][0]["exit_status"], 124)
            self.assertEqual(report["checks"][0]["status"], "timed-out")
            self.assertEqual(report["checks"][1]["status"], "passed")
            self.assertFalse(alive(info), "owned descendant survived timeout cleanup")
        finally:
            # On a regression failure, clean only the recorded fixture's own group.
            if state.exists():
                info = json.loads(state.read_text())
                if alive(info) and info["pgid"] > 1 and info["pgid"] != os.getpgrp():
                    if os.getpgid(info["pid"]) == info["pgid"]:
                        os.killpg(info["pgid"], cpu.signal.SIGKILL)

    def test_timeout_is_not_pass_and_later_gate_runs(self):
        self.source(
            [
                (
                    "full-a-timeout",
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    "none",
                ),
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
                (
                    "full-b-missing",
                    [sys.executable, "-c", "print('no execution summary')"],
                    "unittest",
                ),
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
        self.assertEqual(
            [x["status"] for x in report["checks"]], ["failed", "failed", "incomplete"]
        )
        self.assertEqual(report["checks"][0]["exit_status"], 0)
        self.assertIn(report["checks"][0]["tests"]["executed"], (None, 0))
        self.assertTrue(report["checks"][0]["diagnostic"])
        self.assertIsNone(report["checks"][1]["tests"]["executed"])
        self.assertEqual(report["checks"][2]["tests"]["skipped"], 1)

    def test_source_mismatch_dirty_source_and_reused_output_refuse_before_execution(
        self,
    ):
        head = self.source(
            [("full-only", [sys.executable, "-c", ONE_TEST], "unittest")]
        )
        with self.assertRaisesRegex(ValueError, "explicitly expected"):
            cpu.run(self.repo, self.path / "wrong", expected="0" * 40, execute=True)
        (self.repo / "Cargo.lock").write_text("modified\n")
        with self.assertRaisesRegex(ValueError, "clean checkout"):
            cpu.run(self.repo, self.path / "dirty", expected=head, execute=True)
        self.assertFalse((self.path / "dirty").exists())
        subprocess.run(
            ["git", "-C", str(self.repo), "restore", "Cargo.lock"], check=True
        )
        self.run_checks()
        with self.assertRaises(FileExistsError):
            self.run_checks()

    def test_dirty_source_stops_before_the_next_gate(self):
        marker = self.path / "later-ran"
        head = self.source(
            [
                (
                    "full-a-source-change",
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('Cargo.lock').write_text('changed')",
                    ],
                    "none",
                ),
                (
                    "full-b-later",
                    [sys.executable, "-c", f"open({str(marker)!r},'w').close()"],
                    "none",
                ),
            ]
        )
        original = (self.repo / "Cargo.lock").read_bytes()
        original_identity = cpu.identity

        def restore_after_refusal(*args, **kwargs):
            try:
                return original_identity(*args, **kwargs)
            except ValueError:
                # A later restoration must not erase observed source drift.
                (self.repo / "Cargo.lock").write_bytes(original)
                raise

        with mock.patch.object(cpu, "identity", side_effect=restore_after_refusal):
            report = self.run_checks(expected=head)
        self.assertEqual(cpu.git(self.repo, "status", "--porcelain"), "")
        self.assertEqual(report["status"], "failed")
        self.assertIn("clean checkout", report["source_changed"])
        self.assertEqual([x["id"] for x in report["checks"]], ["full-a-source-change"])
        self.assertEqual(report["checks"][0]["exit_status"], 0)
        self.assertEqual(report["source"]["main_sha"], head)
        self.assertFalse(marker.exists())
        self.assertFalse((self.path / "evidence/full-b-later").exists())
        self.assertEqual(
            json.loads((self.path / "evidence/qualification_receipt.json").read_text()),
            report,
        )
        self.assertEqual(
            json.loads(
                (self.path / "evidence/full-a-source-change/receipt.json").read_text()
            ),
            report["checks"][0],
        )
        self.assertFalse(report["release_qualified"])

    def test_changed_main_ref_stops_even_when_checkout_stays_clean(self):
        marker = self.path / "later-ran"
        advance = (
            "import subprocess; "
            "subprocess.run(['git','-c','user.name=Fixture',"
            "'-c','user.email=fixture@example.invalid','commit','--allow-empty',"
            "'-qm','controlled source advance'],check=True); "
            "subprocess.run(['git','update-ref','refs/remotes/origin/main','HEAD'],check=True)"
        )
        head = self.source(
            [
                ("full-a-ref-change", [sys.executable, "-c", advance], "none"),
                (
                    "full-b-later",
                    [sys.executable, "-c", f"open({str(marker)!r},'w').close()"],
                    "none",
                ),
            ]
        )
        report = self.run_checks(expected=head)
        self.assertEqual(cpu.git(self.repo, "status", "--porcelain"), "")
        self.assertNotEqual(cpu.git(self.repo, "rev-parse", "HEAD"), head)
        self.assertEqual(report["status"], "failed")
        self.assertIn("explicitly expected source", report["source_changed"])
        self.assertEqual([x["id"] for x in report["checks"]], ["full-a-ref-change"])
        self.assertEqual(report["checks"][0]["exit_status"], 0)
        self.assertFalse(marker.exists())
        self.assertFalse((self.path / "evidence/full-b-later").exists())
        self.assertFalse(report["release_qualified"])

    def test_inventory_mismatch_and_symlinked_output_refuse(self):
        self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        (self.repo / "deploy/ci/full-main.py").write_text(
            "GROUPS = {'cpu': ('full-missing',)}\n"
        )
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

    def test_interruption_reaps_only_owned_group_and_retains_unqualified_checkpoint(
        self,
    ):
        process = mock.Mock(pid=314159)
        process.wait.return_value = 0
        with mock.patch.object(cpu.subprocess, "Popen", return_value=process):
            with mock.patch.object(
                cpu, "_wait_unreaped", side_effect=KeyboardInterrupt
            ):
                with mock.patch.object(cpu.time, "sleep"):
                    with mock.patch.object(cpu.os, "killpg") as stop:
                        events = mock.Mock()
                        events.attach_mock(stop, "signal")
                        events.attach_mock(process.wait, "reap")
                        with self.assertRaises(KeyboardInterrupt):
                            cpu.invoke(
                                ["unused"], self.repo, self.path / "cancel.log", {}, 1
                            )
                        self.assertEqual(
                            events.mock_calls,
                            [
                                mock.call.signal(process.pid, cpu.signal.SIGTERM),
                                mock.call.signal(process.pid, cpu.signal.SIGKILL),
                                mock.call.reap(),
                            ],
                        )
        self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        with mock.patch.object(cpu, "execute_gate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                cpu.run(self.repo, self.path / "cancelled", execute=True)
        receipt = json.loads(
            (self.path / "cancelled/qualification_receipt.json").read_text()
        )
        self.assertEqual(receipt["status"], "failed")
        self.assertTrue(receipt["interrupted"])
        self.assertFalse(receipt["release_qualified"])

    def test_plan_uses_frozen_main_inventory_not_dirty_feature_files(self):
        head = self.source(
            [("full-only", [sys.executable, "-c", ONE_TEST], "unittest")]
        )
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
        with mock.patch.object(
            cpu, "invoke", side_effect=AssertionError("plan executed a command")
        ):
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

    def test_release_plan_enumerates_28_plus_q1_without_executing_gates(self):
        self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        plan = self.plan_metadata()
        with mock.patch.object(
            cpu, "invoke", side_effect=AssertionError("plan executed gates")
        ):
            report = cpu.release_plan(self.repo, self.path / "release-plan", plan)
        self.assertEqual(list(report["release_gates"]), cpu.GATE_IDS)
        self.assertEqual(report["checks"], [])
        self.assertFalse(report["release_gates"]["Q1"]["release_blocking"])
        self.assertTrue(report["release_gates"]["D1"]["release_blocking"])
        gate = report["release_gates"]["T1"]
        self.assertEqual(gate["status"], "not-qualified")
        self.assertIn(
            "Historical PASS", gate["evidence"][0]["status_note"]["reported_status"]
        )
        self.assertEqual(
            gate["evidence"][0]["status_note"]["sha256"],
            cpu.digest(plan / "status/TEL-01.md"),
        )
        self.assertFalse(gate["evidence"][0]["qualifies_signed_bytes"])
        schema = json.loads((HERE / "qualification_receipt.schema.json").read_text())
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(report)
        gate["status"] = "passed"
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(report)
        self.assertTrue((self.path / "release-plan/gate-status.md").is_file())

    def test_missing_status_note_is_not_fabricated_or_qualified(self):
        report = cpu.release_inventory(self.plan_metadata(status=False))
        note = report["release_gates"]["T1"]["evidence"][0]["status_note"]
        self.assertFalse(note["exists"])
        self.assertIsNone(note["sha256"])
        self.assertIsNone(note["reported_status"])
        self.assertEqual(report["release_gates"]["T1"]["status"], "not-qualified")

    def test_duplicate_or_missing_release_gate_refuses(self):
        root = self.plan_metadata()
        path = root / "source/spec-v1.md"
        original = path.read_text()
        for changed in (
            original + "\n| **M1** | Duplicate | Same |\n",
            original.replace("| **M1** | Gate M1 | Exact signed bytes |", ""),
        ):
            path.write_text(changed)
            with self.assertRaisesRegex(ValueError, "every release gate"):
                cpu.release_inventory(root)

    def test_metadata_escape_is_rejected_before_reading_outside_plan(self):
        root = self.plan_metadata(status=False)
        outside = self.path / "unadmitted"
        outside.write_text("fixture only")
        (root / "status/TEL-01.md").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            cpu.release_inventory(root)

    def test_concurrently_changed_metadata_refuses_snapshot(self):
        root = self.plan_metadata()
        with mock.patch.object(cpu, "digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "changed during inventory"):
                cpu.release_inventory(root)

    def test_normal_exit_stops_descendants_and_preserves_exit_status(self):
        import time

        def alive(info):
            try:
                fields = (
                    Path(f"/proc/{info['pid']}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()
                )
                return fields[19] == info["start"] and fields[0] not in {"Z", "X"}
            except FileNotFoundError:
                return False

        for expected_exit in (0, 7):
            with self.subTest(exit_status=expected_exit):
                state = self.path / f"normal-exit-{expected_exit}.json"
                child = (
                    "import json,os,signal,sys,time; from pathlib import Path; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    "Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),"
                    "start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))); "
                    "time.sleep(60)"
                )
                leader = (
                    "import subprocess,sys,time; from pathlib import Path; "
                    f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]); "
                    "deadline=time.monotonic()+5; "
                    'exec("while not Path(sys.argv[1]).exists():\\n'
                    " if time.monotonic()>deadline: raise RuntimeError('child readiness')\\n"
                    ' time.sleep(0.01)"); '
                    f"raise SystemExit({expected_exit})"
                )
                try:
                    code, timed_out = cpu.invoke(
                        [sys.executable, "-c", leader, str(state)],
                        self.repo,
                        self.path / f"normal-exit-{expected_exit}.log",
                        cpu.child_environment(
                            self.path, self.path / "unused-target", 1, "fixture"
                        ),
                        10,
                    )
                    self.assertTrue(state.exists(), "owned child did not become ready")
                    info = json.loads(state.read_text())
                    deadline = time.monotonic() + 1
                    while alive(info) and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertEqual(code, expected_exit)
                    self.assertFalse(timed_out)
                    self.assertFalse(
                        alive(info), "owned child survived normal gate exit"
                    )
                finally:
                    if state.exists():
                        info = json.loads(state.read_text())
                        if alive(info):
                            descriptor = os.pidfd_open(info["pid"])
                            try:
                                if alive(info):
                                    cpu.signal.pidfd_send_signal(
                                        descriptor, cpu.signal.SIGKILL
                                    )
                            except ProcessLookupError:
                                pass
                            finally:
                                os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
