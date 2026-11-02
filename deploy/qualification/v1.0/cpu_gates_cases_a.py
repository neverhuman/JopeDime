from cpu_gates_test_support import *

class ControllerA(ControllerBase):
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

