"""Fail-closed tests for the signer_watch release-gate receipt adapter."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gates_receipt as runner


class GatesReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rel07-gates-receipt-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def gate_rows(self, status: str = "NOT_RUN") -> list[dict[str, str]]:
        return [
            {"id": gate, "status": status, "evidence": f"evidence/{gate}.json"}
            for gate in runner.RELEASE_GATE_IDS
        ]

    def test_release_inventory_is_exactly_28_and_excludes_q1(self) -> None:
        self.assertEqual(len(runner.RELEASE_GATE_IDS), 28)
        self.assertEqual(len(set(runner.RELEASE_GATE_IDS)), 28)
        self.assertNotIn("Q1", runner.RELEASE_GATE_IDS)

    def test_cpu_prestage_projects_all_gates_as_not_run_for_signer(self) -> None:
        path = self.root / "qualification_receipt.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "jope.qualification-cpu-prestage/v1",
                    "source": {"main_sha": "a" * 40},
                    "release_qualified": False,
                    "signed_candidate": None,
                    "release_gates": {
                        gate: {"status": "not-qualified", "reason": "prep only"}
                        for gate in runner.RELEASE_GATE_IDS + ["Q1"]
                    },
                }
            )
        )

        result = runner.from_cpu_prestage(path)

        self.assertEqual(len(result["gates"]), 28)
        self.assertTrue(all(row["status"] == "NOT_RUN" for row in result["gates"]))
        self.assertTrue(
            all(set(row) == {"id", "status", "evidence"} for row in result["gates"])
        )
        self.assertTrue(all("prep only" in row["evidence"] for row in result["gates"]))

    def test_cpu_prestage_cannot_claim_signed_release(self) -> None:
        path = self.root / "qualification_receipt.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "jope.qualification-cpu-prestage/v1",
                    "source": {"main_sha": "a" * 40},
                    "release_qualified": True,
                    "signed_candidate": {"sha256": "b" * 64},
                    "release_gates": {
                        gate: {"status": "PASS"}
                        for gate in runner.RELEASE_GATE_IDS + ["Q1"]
                    },
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "cannot claim signed release"):
            runner.from_cpu_prestage(path)

    def test_pass_results_require_matching_artifact_bytes(self) -> None:
        artifact = self.root / "candidate.bin"
        artifact.write_bytes(b"signed candidate bytes")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        results = self.root / "results.json"
        results.write_text(
            json.dumps(
                {
                    "schema": "jope.rel07.gate-results/v1",
                    "signed_bytes_sha256": digest,
                    "gates": self.gate_rows("PASS"),
                }
            )
        )

        receipt = runner.from_gate_results(results, artifact)

        self.assertEqual(receipt["signed_bytes_sha256"], digest)
        self.assertEqual(receipt["artifact_sha256"], digest)
        self.assertEqual(
            [row["id"] for row in receipt["gates"]], runner.RELEASE_GATE_IDS
        )
        self.assertTrue(
            all(
                set(row) == {"id", "status", "evidence", "artifact_sha256"}
                for row in receipt["gates"]
            )
        )
        self.assertTrue(all(row["status"] == "PASS" for row in receipt["gates"]))
        self.assertTrue(
            all(row["artifact_sha256"] == digest for row in receipt["gates"])
        )

    def test_pass_results_without_artifact_are_refused(self) -> None:
        results = self.root / "results.json"
        results.write_text(
            json.dumps(
                {
                    "schema": "jope.rel07.gate-results/v1",
                    "gates": self.gate_rows("PASS"),
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "require --artifact"):
            runner.from_gate_results(results, None)

    def test_wrong_artifact_hash_is_refused(self) -> None:
        artifact = self.root / "candidate.bin"
        artifact.write_bytes(b"candidate")
        results = self.root / "results.json"
        results.write_text(
            json.dumps(
                {
                    "schema": "jope.rel07.gate-results/v1",
                    "signed_bytes_sha256": "0" * 64,
                    "gates": self.gate_rows("PASS"),
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "not bound"):
            runner.from_gate_results(results, artifact)

    def test_missing_extra_duplicate_or_malformed_gate_fails_closed(self) -> None:
        cases = [
            self.gate_rows()[:-1],
            self.gate_rows() + [{"id": "Q1", "status": "PASS", "evidence": "x"}],
            self.gate_rows()[:-1] + [self.gate_rows()[0]],
            self.gate_rows()[:-1] + [{"id": "D2", "status": "MAYBE", "evidence": "x"}],
        ]
        for gates in cases:
            with self.subTest(gates=len(gates)):
                source = self.root / "results.json"
                source.write_text(
                    json.dumps({"schema": "jope.rel07.gate-results/v1", "gates": gates})
                )
                with self.assertRaises(ValueError):
                    runner.from_gate_results(source, None)


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


if __name__ == "__main__":
    unittest.main()
