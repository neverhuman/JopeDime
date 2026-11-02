"""Waiver and variant assembly test groups for gate receipts."""

from __future__ import annotations

import copy, hashlib, json, subprocess, unittest
from pathlib import Path
from unittest.mock import patch

import gates_receipt as runner
from test_gates_receipt_scaffold import ScaffoldAssemblyTests


class M3WaiverTests(unittest.TestCase):
    """Owner waiver preserves raw failure and never waives CV or another operator."""

    def setUp(self):
        import runpy

        self.fixture = ScaffoldAssemblyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        helpers = runpy.run_path(str(Path(__file__).with_name("test_provider_m3")))
        self.full = helpers["full"]
        self.report = helpers["full_report"](self.fixture.source)
        proposer = self.report["clauses"]["r0_proposer"]
        for row, reference, native in zip(
            proposer["strata"][1:],
            [(64, 64, 64), (278, 277, 277)],
            [(688, 616, 616), (2902, 2589, 2589)],
        ):
            for key, values in [
                ("reference_counts", reference),
                ("native_counts", native),
            ]:
                row[key] = {op: 0 for op in self.full.OPERATOR_FAMILIES}
                for op, value in zip(
                    ["edit_replace_head", "champion_retrieval", "edit_parameter"],
                    values,
                ):
                    row[key][op] = value
            row["passed"] = False
        for key in ["reference_counts", "native_counts"]:
            proposer["strata"][0][key] = {
                op: sum(row[key][op] for row in proposer["strata"][1:])
                for op in self.full.OPERATOR_FAMILIES
            }
        proposer["strata"][0]["passed"] = False
        proposer["passed"] = False
        self.report["status"] = "NUMERICAL_FAIL"
        self.report["counts"].update(passed=3, failed=1)
        ctx = helpers["Context"](
            "M3",
            self.fixture.artifact,
            self.fixture.digest,
            self.fixture.source,
            self.fixture.root / "M3.json",
        )
        self.ctx = ctx
        self.payload = self.full.gate_payload(ctx, self.report, "b" * 64)
        self.path = next(p for p in self.fixture.paths if p.stem == "M3")
        self.path.write_text(json.dumps(self.payload))
        self.ruling = self.fixture.root / "owner-ruling.md"
        self.ruling.write_text("synthetic ruling fixture; not owner authorization")
        self.addCleanup(patch.stopall)
        patch.object(
            runner,
            "M3_WAIVER_RULING_SHA256",
            hashlib.sha256(self.ruling.read_bytes()).hexdigest(),
            create=True,
        ).start()
        patch.object(
            runner,
            "_source_proposer_sha256",
            return_value="48ebbfc081451baf0470ae25e1ae04621823ac4de3dfb0d8e93576206c8dbe39",
            create=True,
        ).start()

    def assemble(self, **changes):
        kwargs = dict(m3_waiver_ruling=self.ruling, release_version="v1.0.0")
        kwargs.update(changes)
        return self.fixture.assemble(**kwargs)

    def test_frozen_rc4_witnesses_keep_the_authorized_waiver(self):
        payload = copy.deepcopy(self.payload)
        payload["source_sha"] = "ccaf14b210b3b55a350218e60f6b49ef76524cf3"
        payload["clauses"]["r0_proposer"]["native_witness_sha256"] = {
            "native_candidates": "94eb53cddd796c98b077d2142dd4b60e679a8d768d38fafa070ba4ee935c0bf2",
            "native_admission": "2ad84b8525a5ecbcf7895e9cb56e417a21f8689e7ac11e7d7fa62c5b65fc3e6d",
        }
        before = copy.deepcopy(payload)
        waiver = runner._m3_waiver(payload, self.ruling, "v1.0.0")
        self.assertEqual(waiver["disposition"], "WAIVED_V1_0_0")
        self.assertEqual(waiver["expires_at"], "v1.0.1")
        self.assertEqual(waiver["raw_status"], "FAIL")
        self.assertEqual(payload, before)

    def test_authorized_waiver_preserves_raw_fail_and_counts(self):
        receipt = self.assemble()
        row = next(x for x in receipt["gates"] if x["id"] == "M3")
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual(row["disposition"], "WAIVED_V1_0_0")
        self.assertEqual(row["counts"], self.payload["counts"])

    def test_same_controller_provenance_retains_waived_raw_failure_exit(self):
        options, execution = self.fixture.controller_fixture()
        execution["checks"][2]["exit_code"] = 1
        del execution["checks"][2]["result_sha256"]
        (options["controller_run"] / "execution.json").write_text(json.dumps(execution))
        receipt = self.assemble(**options)
        row = next(x for x in receipt["gates"] if x["id"] == "M3")
        self.assertEqual((row["status"], row["disposition"]), ("FAIL", "WAIVED_V1_0_0"))
        self.assertEqual(receipt["provenance"]["commands"][2]["exit_code"], 1)
        self.assertEqual(receipt["release_version"], "v1.0.0")
        self.assertEqual(receipt["waivers"][0]["expires_at"], "v1.0.1")
        self.assertEqual(len(receipt["gates"]), 28)
        self.assertEqual(self.payload, json.loads(self.path.read_text()))

    def test_waiver_cannot_carry_into_future_release_or_unknown_ruling(self):
        for version in ["v1.0.1", "v2.0.0", None]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.assemble(release_version=version)
        self.ruling.write_text("different bytes")
        with self.assertRaises(ValueError):
            self.assemble()

    def test_no_waiver_without_current_proposer_and_complete_fresh_m3(self):
        with patch.object(runner, "_source_proposer_sha256", return_value="0" * 64):
            with self.assertRaises(ValueError):
                self.assemble()
        for update in [
            {"counts": {**self.payload["counts"], "skipped": 1}},
            {"clauses": {}},
            {"source_sha": "b" * 40},
            {"artifact_sha256": "0" * 64},
        ]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.path.write_text(json.dumps({**self.payload, **update}))
                self.assemble()

    def test_numerical_failure_cannot_be_waived(self):
        report = copy.deepcopy(self.report)
        report["clauses"]["full_train_cv"].update(passed=False, mismatches=1)
        report["counts"].update(passed=2, failed=2)
        payload = self.full.gate_payload(self.ctx, report, "b" * 64)
        self.path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            self.assemble()

    def test_guard_is_count_derived_for_head_and_every_other_operator(self):
        for op, amount in [("edit_replace_head", 60), ("edit_crossover", 50)]:
            payload = copy.deepcopy(self.payload)
            rows = payload["clauses"]["r0_proposer"]["strata"]
            for row in rows[:2]:
                row["native_counts"]["champion_retrieval"] -= amount
                row["native_counts"][op] += amount
            self.path.write_text(json.dumps(payload))
            with self.subTest(operator=op), self.assertRaises(ValueError):
                self.assemble()

    def test_another_gate_cannot_use_the_m3_waiver(self):
        path = next(p for p in self.fixture.paths if p.stem == "M2")
        value = json.loads(path.read_text())
        value["status"] = "FAIL"
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.assemble()

    def test_cli_keeps_raw_fail_and_does_not_claim_all_pass(self):
        args = self.fixture.argv()[1:] + [
            "--m3-waiver-ruling",
            str(self.ruling),
            "--release-version",
            "v1.0.0",
        ]
        stdout = io.StringIO()
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(stdout):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(
            json.loads(stdout.getvalue()), {"gates": 28, "all_pass": False}
        )
        receipt = json.loads(self.fixture.output.read_text())
        row = next(x for x in receipt["gates"] if x["id"] == "M3")
        self.assertEqual((row["status"], row["disposition"]), ("FAIL", "WAIVED_V1_0_0"))

    def test_ruling_changed_during_assembly_refuses(self):
        original = runner._assembly_json

        def mutate(path):
            result = original(path)
            if path == self.fixture.paths[-1]:
                self.ruling.write_text("changed owner record")
            return result

        with patch.object(runner, "_assembly_json", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "ruling changed"):
                self.assemble()


class CandidateVariantAssemblyTests(unittest.TestCase):
    """Synthetic signed/sealed metadata controls; never candidate qualification."""

    def setUp(self):
        self.fixture = ScaffoldAssemblyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.fused = f.root / "fused-fixture.bin"
        self.fused.write_bytes(b"synthetic fused bytes, distinct from full")
        self.fused_digest = hashlib.sha256(self.fused.read_bytes()).hexdigest()
        rows = []
        for identity, artifact, profile in [
            ("full-linux-x86_64", f.artifact, "full"),
            ("fused-cpu-linux-x86_64", self.fused, "fused-cpu"),
        ]:
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            seal = f.root / (identity + "-seal.json")
            seal.write_text(
                json.dumps(
                    {
                        "schema": "jope.rel07.rc2-sealed-variant/v1",
                        "id": identity,
                        "source_sha": f.source,
                        "sha256": digest,
                        "verify": {"ok": True, "report": {"file_sha256": digest}},
                    }
                )
            )
            rows.append(
                {
                    "id": identity,
                    "profile": profile,
                    "path": str(artifact),
                    "source_sha": f.source,
                    "sha256": digest,
                    "bytes": artifact.stat().st_size,
                    "seal_verify_receipt": str(seal),
                    "seal_verify_receipt_sha256": hashlib.sha256(
                        seal.read_bytes()
                    ).hexdigest(),
                    "signature_receipts": [{"receipt": {"verify": "PASS"}}],
                }
            )
        self.manifest = f.root / "candidate-manifest.json"
        self.candidate = {
            "schema": "jope.rel07.rc2-manifest/v1",
            "source_sha": f.source,
            "release_candidate": True,
            "qualified": False,
            "artifacts": rows,
        }
        self.save_manifest()
        d2 = f.root / "D2.json"
        result = json.loads(d2.read_text())
        result["artifact_sha256"] = self.fused_digest
        d2.write_text(json.dumps(result))

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.candidate))

    def assemble(self):
        return self.fixture.assemble(candidate_manifest=self.manifest)

    def test_same_candidate_preserves_primary_and_raw_fused_digests(self):
        result = self.assemble()
        self.assertEqual(result["artifact_sha256"], self.fixture.digest)
        for row in result["gates"]:
            expected = self.fused_digest if row["id"] == "D2" else self.fixture.digest
            self.assertEqual(row["artifact_sha256"], expected)
        binding = result["gate_artifacts"]["D2"]
        self.assertEqual(binding["sha256"], self.fused_digest)
        self.assertEqual(
            binding["candidate_manifest_sha256"],
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            binding["seal_receipt_sha256"],
            self.candidate["artifacts"][1]["seal_verify_receipt_sha256"],
        )

    def test_fused_digest_without_explicit_candidate_manifest_is_refused(self):
        with self.assertRaises(ValueError):
            self.fixture.assemble()

    def test_primary_digest_cannot_replace_d2_or_another_gate_use_fused(self):
        for gate, digest in [("D2", self.fixture.digest), ("M1", self.fused_digest)]:
            with self.subTest(gate=gate):
                path = self.fixture.root / (gate + ".json")
                before = path.read_bytes()
                result = json.loads(before)
                result["artifact_sha256"] = digest
                path.write_text(json.dumps(result))
                try:
                    with self.assertRaises(ValueError):
                        self.assemble()
                finally:
                    path.write_bytes(before)

    def test_missing_duplicate_wrong_profile_or_source_variant_is_refused(self):
        original = json.loads(json.dumps(self.candidate))
        variants = []
        missing = json.loads(json.dumps(original))
        missing["artifacts"].pop()
        variants.append(missing)
        duplicate = json.loads(json.dumps(original))
        duplicate["artifacts"].append(duplicate["artifacts"][1])
        variants.append(duplicate)
        wrong_profile = json.loads(json.dumps(original))
        wrong_profile["artifacts"][1]["profile"] = "full"
        variants.append(wrong_profile)
        wrong_source = json.loads(json.dumps(original))
        wrong_source["artifacts"][1]["source_sha"] = "b" * 40
        variants.append(wrong_source)
        for value in variants:
            with self.subTest(value=value):
                self.candidate = value
                self.save_manifest()
                with self.assertRaises(ValueError):
                    self.assemble()
        self.candidate = original
        self.save_manifest()

    def test_candidate_source_bytes_size_or_seal_pin_mismatch_is_refused(self):
        changes = [
            (None, "source_sha", "b" * 40),
            (0, "sha256", "0" * 64),
            (1, "bytes", 0),
            (1, "seal_verify_receipt_sha256", "0" * 64),
        ]
        for index, key, value in changes:
            with self.subTest(index=index, key=key):
                target = (
                    self.candidate
                    if index is None
                    else self.candidate["artifacts"][index]
                )
                before = target[key]
                target[key] = value
                self.save_manifest()
                try:
                    with self.assertRaises(ValueError):
                        self.assemble()
                finally:
                    target[key] = before
                    self.save_manifest()

    def test_nonpassing_or_missing_signature_seal_is_refused(self):
        row = self.candidate["artifacts"][1]
        for signatures in [[], [{"receipt": {"verify": "FAIL"}}]]:
            row["signature_receipts"] = signatures
            self.save_manifest()
            with self.assertRaises(ValueError):
                self.assemble()
        row["signature_receipts"] = [{"receipt": {"verify": "PASS"}}]
        self.save_manifest()
        seal = Path(row["seal_verify_receipt"])
        value = json.loads(seal.read_text())
        value["verify"]["ok"] = False
        seal.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.assemble()

    def test_binding_files_and_both_artifacts_cannot_change_during_assembly(self):
        original_read = runner._assembly_json
        for target in [
            self.manifest,
            Path(self.candidate["artifacts"][1]["seal_verify_receipt"]),
            self.fixture.artifact,
            self.fused,
        ]:
            before = target.read_bytes()
            changed = False

            def mutate(path):
                nonlocal changed
                value = original_read(path)
                if path == self.fixture.paths[-1] and not changed:
                    target.write_bytes(before + b" ")
                    changed = True
                return value

            with (
                self.subTest(target=target),
                patch.object(runner, "_assembly_json", side_effect=mutate),
            ):
                try:
                    with self.assertRaises(ValueError):
                        self.assemble()
                finally:
                    target.write_bytes(before)

    def test_cli_accepts_explicit_manifest_and_preserves_fused_hash(self):
        argv = self.fixture.argv() + ["--candidate-manifest", str(self.manifest)]
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(self.fixture.output.read_text())
        self.assertEqual(
            next(row for row in receipt["gates"] if row["id"] == "D2")[
                "artifact_sha256"
            ],
            self.fused_digest,
        )


