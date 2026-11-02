from cpu_gates_test_support import *

class ControllerB(ControllerBase):
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

    def test_gate_status_table_is_complete_and_never_promotes_historical_pass(self):
        self.source([("full-only", [sys.executable, "-c", ONE_TEST], "unittest")])
        plan = self.plan_metadata()
        with mock.patch.object(
            cpu, "invoke", side_effect=AssertionError("plan executed gates")
        ):
            report = cpu.release_plan(self.repo, self.path / "release-plan", plan)

        lines = (self.path / "release-plan/gate-status.md").read_text().splitlines()
        rows = []
        for line in lines:
            if not line.startswith("| "):
                continue
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if cells and cells[0] in cpu.GATE_IDS:
                rows.append(cells)

        self.assertEqual([row[0] for row in rows], cpu.GATE_IDS)
        self.assertTrue(all(row[2] == "not-qualified / not-run" for row in rows))
        evidence = report["release_gates"]["T1"]["evidence"][0]
        self.assertIn("Historical PASS", evidence["status_note"]["reported_status"])
        self.assertFalse(evidence["qualifies_signed_bytes"])
        self.assertEqual(report["checks"], [])

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

