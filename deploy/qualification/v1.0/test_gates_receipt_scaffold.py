"""Mechanical test groups for signer_watch release-gate receipt assembly."""

from __future__ import annotations

import contextlib, copy, hashlib, io, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

import gates_receipt as runner


class ScaffoldAssemblyTests(unittest.TestCase):
    """Synthetic receipts and bytes only; none of these controls qualify a candidate."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rel07-assembly-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "fixture-only.bin"
        self.artifact.write_bytes(b"synthetic fixture only; never release evidence")
        self.digest = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        self.source = "a" * 40
        self.scaffold = self.root / "scaffold.json"
        self.spec = {
            "schema": "jope.rel07.gate-command-scaffold/v1",
            "qualification_handoff": "NOT_EXECUTABLE",
            "candidate_sha256": "${JOPE_REL07_ARTIFACT_SHA256}",
            "gates": [
                {
                    "id": gate,
                    "argv": ["never-execute-this-fixture"],
                    "timeout_seconds": 1,
                }
                for gate in runner.RELEASE_GATE_IDS
            ],
        }
        self.scaffold.write_text(json.dumps(self.spec))
        self.paths = []
        for gate in runner.RELEASE_GATE_IDS:
            path = self.root / (gate + ".json")
            path.write_text(
                json.dumps(
                    {
                        "schema": "jope.rel07.gate-execution/v1",
                        "id": gate,
                        "source_sha": self.source,
                        "artifact_sha256": self.digest,
                        "status": "PASS",
                        "counts": dict(
                            collected=1,
                            executed=1,
                            passed=1,
                            failed=0,
                            skipped=0,
                            excluded=0,
                        ),
                    }
                )
            )
            self.paths.append(path)
        self.output = self.root / "qualification_receipt.json"

    def assemble(self, **kwargs):
        args = dict(
            scaffold_path=self.scaffold,
            receipt_paths=self.paths,
            artifact=self.artifact,
            source_sha=self.source,
        )
        args.update(kwargs)
        return runner.from_scaffold(**args)

    def argv(self):
        args = [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--scaffold",
            str(self.scaffold),
            "--artifact",
            str(self.artifact),
            "--source-sha",
            self.source,
            "--output",
            str(self.output),
        ]
        for path in self.paths:
            args += ["--gate-receipt", str(path)]
        return args

    def test_all_pass_exact_signer_shape_and_content_provenance(self):
        receipt = self.assemble(receipt_paths=list(reversed(self.paths)))
        self.assertEqual(set(receipt), {"artifact_sha256", "gates"})
        self.assertEqual(receipt["artifact_sha256"], self.digest)
        self.assertEqual([r["id"] for r in receipt["gates"]], runner.RELEASE_GATE_IDS)
        for row, path in zip(receipt["gates"], self.paths):
            self.assertEqual(set(row), {"id", "status", "artifact_sha256", "evidence"})
            self.assertEqual(row["status"], "PASS")
            self.assertEqual(row["artifact_sha256"], self.digest)
            self.assertIn(str(path), row["evidence"])
            self.assertIn(
                hashlib.sha256(path.read_bytes()).hexdigest(), row["evidence"]
            )
            self.assertIn(
                hashlib.sha256(self.scaffold.read_bytes()).hexdigest(), row["evidence"]
            )
            self.assertIn(self.source, row["evidence"])
        self.spec["candidate_sha256"] = self.digest
        self.scaffold.write_text(json.dumps(self.spec))
        self.assertEqual(self.assemble()["artifact_sha256"], self.digest)

    def test_missing_extra_duplicate_unknown_or_absent_receipts_refuse(self):
        for paths in [
            self.paths[:-1],
            self.paths + self.paths[:1],
            self.paths[:-1] + self.paths[:1],
            self.paths[:-1] + [self.root / "absent.json"],
        ]:
            with self.subTest(paths=paths), self.assertRaises((ValueError, OSError)):
                self.assemble(receipt_paths=paths)
        value = json.loads(self.paths[0].read_text())
        value["id"] = "Q1"
        self.paths[0].write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.assemble()

    def test_every_non_pass_and_incomplete_count_shape_refuses(self):
        good = json.loads(self.paths[0].read_text())
        changes = [
            {"status": status}
            for status in ["FAIL", "BLOCKED", "NOT_RUN", "SKIP", "pass", None]
        ]
        changes += [
            {"counts": None},
            {"counts": {}},
            {"counts": {**good["counts"], "other": 1}},
        ]
        for key, values in {
            "collected": [0, 2],
            "executed": [0, 2],
            "passed": [0, 2, True, "1", -1],
            "failed": [1],
            "skipped": [1],
            "excluded": [1],
        }.items():
            changes += [{"counts": {**good["counts"], key: value}} for value in values]
        for change in changes:
            self.paths[0].write_text(json.dumps({**good, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.assemble()

    def test_schema_source_artifact_and_explicit_source_pin_are_required(self):
        good = json.loads(self.paths[0].read_text())
        for change in [
            {"schema": "jope.rel07.gate-results/v1"},
            {"source_sha": "b" * 40},
            {"artifact_sha256": "0" * 64},
            {"artifact_sha256": None},
        ]:
            self.paths[0].write_text(json.dumps({**good, **change}))
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "does not match"),
            ):
                self.assemble()
        self.paths[0].write_text(json.dumps(good))
        for source in ["", "a" * 39, "A" * 40, "${SOURCE_SHA}", None]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.assemble(source_sha=source)
        self.artifact.write_bytes(b"other synthetic bytes")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.assemble()

    def test_scaffold_inventory_schema_binding_and_command_validation(self):
        changes = [
            {"gates": self.spec["gates"][:-1]},
            {"gates": self.spec["gates"][:-1] + self.spec["gates"][:1]},
            {"schema": "jope.rel07.gate-commands/v1"},
            {"qualification_handoff": "EXECUTABLE"},
            {"candidate_sha256": "0" * 64},
        ]
        for key, value in [
            ("id", "Q1"),
            ("id", []),
            ("argv", []),
            ("argv", [""]),
            ("argv", [1]),
            ("argv", ["bad\0arg"]),
            ("timeout_seconds", True),
            ("timeout_seconds", 0),
            ("timeout_seconds", float("nan")),
            ("timeout_seconds", float("inf")),
        ]:
            gates = copy.deepcopy(self.spec["gates"])
            gates[0][key] = value
            changes.append({"gates": gates})
        for change in changes:
            self.scaffold.write_text(json.dumps({**self.spec, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.assemble()

    def test_malformed_and_duplicate_key_json_refuses(self):
        for path in [self.scaffold, self.paths[0]]:
            good = path.read_bytes()
            for raw in [
                b"[]",
                b"null",
                b"{",
                b'{"status":"FAIL","status":"PASS"}',
                b"\xff",
            ]:
                path.write_bytes(raw)
                with self.subTest(path=path, raw=raw), self.assertRaises(ValueError):
                    self.assemble()
            path.write_bytes(good)

    def test_symlinked_or_nonregular_inputs_refuse(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for kwargs in [
            {"scaffold_path": alias / self.scaffold.name},
            {"receipt_paths": [alias / p.name for p in self.paths]},
            {"artifact": alias / self.artifact.name},
            {"artifact": self.root},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.assemble(**kwargs)

    def test_inputs_and_artifact_changed_during_assembly_refuse(self):
        original = runner._assembly_json
        saved = self.paths[0].read_bytes()
        artifact = self.artifact.read_bytes()
        for target in [self.paths[0], self.artifact]:

            def child(path):
                result = original(path)
                if path == self.paths[-1]:
                    target.write_bytes(b"changed fixture bytes")
                return result

            with (
                patch.object(runner, "_assembly_json", side_effect=child),
                self.assertRaises(ValueError),
            ):
                self.assemble()
            self.paths[0].write_bytes(saved)
            self.artifact.write_bytes(artifact)

    def test_cli_succeeds_without_executing_scaffold_and_refuses_existing_output(self):
        result = subprocess.run(
            self.argv(), capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"gates": 28, "all_pass": True})
        good = self.output.read_bytes()
        self.assertEqual(json.loads(good), self.assemble())
        again = subprocess.run(self.argv(), capture_output=True, text=True, check=False)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(self.output.read_bytes(), good)

    def test_cli_invalid_receipt_or_missing_arguments_publish_nothing(self):
        invalid = [self.argv()[:-2], [a for a in self.argv() if a != "--scaffold"]]
        for flag in ["--artifact", "--source-sha"]:
            args = self.argv()
            index = args.index(flag)
            invalid.append(args[:index] + args[index + 2 :])
        for argv in invalid:
            result = subprocess.run(argv, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.output.exists())
        bad = json.loads(self.paths[-1].read_text())
        bad["status"] = "FAIL"
        self.paths[-1].write_text(json.dumps(bad))
        result = subprocess.run(
            self.argv(), capture_output=True, text=True, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_cli_symlinked_output_parent_refuses(self):
        alias = self.root / "alias-output"
        alias.symlink_to(self.root, target_is_directory=True)
        args = self.argv()
        args[args.index("--output") + 1] = str(alias / self.output.name)
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def controller_fixture(self):
        run_dir = self.root / "controller"
        checks, paths = [], []
        for gate, old in zip(runner.RELEASE_GATE_IDS, self.paths):
            folder = run_dir / gate
            folder.mkdir(parents=True)
            path = folder / "result.json"
            path.write_bytes(old.read_bytes())
            log = folder / "command.log"
            log.write_text("synthetic command output only\n")
            paths.append(path)
            checks.append(
                dict(
                    id=gate,
                    argv=["never-execute-this-fixture", gate],
                    exit_code=0,
                    timed_out=False,
                    started_utc="2026-11-02T00:00:00+00:00",
                    finished_utc="2026-11-02T00:00:01+00:00",
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                    result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        self.paths = paths
        fields = runner._assembly_controller()["INPUT_FIELDS"]
        inputs = self.root / "provider-inputs.json"
        inputs.write_text(
            json.dumps(
                {
                    "schema": "jope.rel07.provider-inputs/v1",
                    "gates": {
                        gate: {
                            name: "b" * 64
                            if name.endswith("_sha256")
                            else str(self.root)
                            for name in (
                                choices[-1] if isinstance(choices, tuple) else choices
                            )
                        }
                        for gate, choices in fields.items()
                    },
                }
            )
        )
        execution = dict(
            schema="jope.rel07.gate-run/v1",
            mode="execute",
            source_sha=self.source,
            artifact_sha256=self.digest,
            candidate_inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
            machine="synthetic-host",
            interrupted=False,
            started_utc="2026-11-02T00:00:00+00:00",
            finished_utc="2026-11-02T00:00:01+00:00",
            runner_sha256="c" * 64,
            manifest_sha256="d" * 64,
            checks=checks,
        )
        (run_dir / "execution.json").write_text(json.dumps(execution))
        return dict(controller_run=run_dir, controller_inputs=inputs), execution

    def test_controller_provenance_preserves_commands_machine_and_output_pins(self):
        options, execution = self.controller_fixture()
        result = self.assemble(**options)
        provenance = result.pop("provenance")
        self.assertEqual(result, self.assemble())
        self.assertEqual(provenance["commands"], execution["checks"])
        self.assertEqual(provenance["machine"], "synthetic-host")
        self.assertEqual(len(provenance["outputs"]), 28)
        for row, path in zip(provenance["outputs"], self.paths):
            self.assertEqual(row["result"], str(path))
            self.assertEqual(
                row["result_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
        args = self.argv()
        for name, path in options.items():
            args += ["--" + name.replace("_", "-"), str(path)]
        child = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(self.output.read_text())["provenance"], provenance)

    def test_controller_rejects_incomplete_or_mismatched_execution(self):
        options, execution = self.controller_fixture()
        changes = [
            {"source_sha": "b" * 40},
            {"artifact_sha256": "b" * 64},
            {"candidate_inputs_sha256": "b" * 64},
            {"interrupted": True},
            {"machine": ""},
            {"finished_utc": None},
            {"runner_sha256": "bad"},
            {"checks": execution["checks"][:-1]},
            {"checks": execution["checks"][:-1] + execution["checks"][:1]},
        ]
        for key, value in [
            ("exit_code", 1),
            ("exit_code", True),
            ("timed_out", True),
            ("argv", []),
            ("finished_utc", ""),
            ("log_sha256", "b" * 64),
            ("result_sha256", "b" * 64),
        ]:
            checks = copy.deepcopy(execution["checks"])
            checks[0][key] = value
            changes.append({"checks": checks})
        for change in changes:
            (options["controller_run"] / "execution.json").write_text(
                json.dumps({**execution, **change})
            )
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.assemble(**options)

    def test_controller_rejects_borrowed_receipts_and_changed_logs(self):
        options, _ = self.controller_fixture()
        original = self.paths[0]
        borrowed = self.root / "borrowed.json"
        borrowed.write_bytes(original.read_bytes())
        self.paths[0] = borrowed
        with self.assertRaisesRegex(ValueError, "same controller run"):
            self.assemble(**options)
        self.paths[0] = original
        (original.parent / "command.log").write_text("changed output")
        with self.assertRaisesRegex(ValueError, "output hash differs"):
            self.assemble(**options)

    def test_controller_rejects_missing_bindings_and_changes_during_assembly(self):
        options, _ = self.controller_fixture()
        inputs = options["controller_inputs"]
        saved = inputs.read_bytes()
        value = json.loads(saved)
        del value["gates"]["A5"]
        inputs.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "all provider input"):
            self.assemble(**options)
        inputs.write_bytes(saved)
        original = runner._assembly_json

        def change_metadata(path):
            result = original(path)
            if path == self.scaffold:
                inputs.write_text("changed during assembly")
            return result

        with patch.object(runner, "_assembly_json", side_effect=change_metadata):
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                self.assemble(**options)

    def test_controller_pass_requires_successful_exit_and_controller_result_pin(self):
        options, execution = self.controller_fixture()
        m3 = execution["checks"][2]
        original = dict(m3)
        for change in [{"exit_code": 1}, {"result_sha256": None}]:
            execution["checks"][2] = {**original, **change}
            (options["controller_run"] / "execution.json").write_text(
                json.dumps(execution)
            )
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.assemble(**options)
        del execution["checks"][2]["result_sha256"]
        (options["controller_run"] / "execution.json").write_text(json.dumps(execution))
        with self.assertRaises(ValueError):
            self.assemble(**options)

    def test_controller_raw_preflight_failure_is_not_a_waiver(self):
        options, execution = self.controller_fixture()
        path = self.paths[2]
        raw = json.loads(path.read_text())
        raw.update(
            status="FAIL",
            counts=dict(
                collected=0, executed=0, passed=0, failed=0, skipped=0, excluded=0
            ),
        )
        path.write_text(json.dumps(raw))
        execution["checks"][2]["exit_code"] = 1
        del execution["checks"][2]["result_sha256"]
        (options["controller_run"] / "execution.json").write_text(json.dumps(execution))
        with self.assertRaisesRegex(ValueError, "receipt must PASS"):
            self.assemble(**options)
        for name, value in options.items():
            child = subprocess.run(
                self.argv() + ["--" + name.replace("_", "-"), str(value)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(child.returncode, 0)
            self.assertFalse(self.output.exists())

    def test_cli_exclusive_output_creation_preserves_a_racing_receipt(self):
        original = Path.mkdir

        def create_racer(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == self.output.parent:
                self.output.write_text("earlier receipt")
            return result

        with (
            patch.object(sys, "argv", self.argv()[1:]),
            patch.object(Path, "mkdir", create_racer),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                runner.main()
        self.assertEqual(self.output.read_text(), "earlier receipt")


