"""Synthetic admission controls; no qualification, signing or model execution."""

from pathlib import Path
import copy
import os
import tempfile
import hashlib
import json
import runpy
import sys
import unittest
from unittest.mock import patch

import plan_publisher as publisher
import api01_launch as api_launcher

ROOT = Path(__file__).resolve().parents[4]
V1 = ROOT / "deploy/qualification/v1.0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeAdmission(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(V1))
        self.addCleanup(sys.path.remove, str(V1))
        fixture_type = runpy.run_path(str(V1 / "test_gates_receipt.py"))[
            "M3WaiverTests"
        ]
        self.m3 = fixture_type()
        self.m3.setUp()
        self.addCleanup(self.m3.doCleanups)
        self.root = self.m3.fixture.root
        source = self.m3.fixture.source
        binary = self.m3.fixture.digest
        dataset = {
            "id": "0123456789abcdef",
            "dataset_sha256": "0123456789abcdef" + "0" * 48,
            "cut_sha256": "2" * 64,
            "family": "3" * 64,
        }
        pins = {
            "bundle_sha256": publisher.BUNDLE,
            "dope_sha256": publisher.DOPE,
            "evaluator_sha256": binary,
            "training_family_manifest_sha256": "4" * 64,
        }
        self.plan = {
            "source_commit": source,
            "metadata_source_commit": "b" * 40,
            "backend_sha256": "5" * 64,
            "pins": pins,
            "datasets": [dataset],
        }
        identity = {"source_commit": source, "evaluator_sha256": binary}
        self.data = {
            "native": dict(
                identity,
                status="PASS_PRODUCTION_NATIVE_COMPONENT",
                backend_sha256=self.plan["backend_sha256"],
                version={"source_sha40": source},
                qualification_cases=[{"id": str(i), "passed": True} for i in range(5)],
                observations=[
                    {
                        "policy": p,
                        "charged": 24,
                        "completed": 24,
                        "folds_per_completed": 3,
                        "exact_terminal_replay": True,
                        "adapter_exit": 0,
                    }
                    for p in ["heuristic", "learned"]
                ],
                stop={"engine_exit": 0, "owned_group_processes_remaining": []},
                activity={"learned_forward_count": 4},
            ),
            "native_driver": dict(identity, exit=0),
            "m3_driver": dict(identity, exit=1, status="EXCLUSIVE_RELEASED"),
            "link_inputs": dict(identity),
            "m2_report": {
                "source": source,
                "passed": True,
                "contexts": 50,
                "slates": 100,
                "failed_values": 0,
                "optimizer_updates": 0,
            },
            "m3_raw": copy.deepcopy(self.m3.report),
            "identity_closure": dict(
                identity,
                status="PASS_PINNED_NAMESPACE",
                training_family_manifest_sha256=pins["training_family_manifest_sha256"],
                overlap_counts={"family": 0},
                selected_bindings=[dict(dataset, family="fam-" + dataset["family"])],
            ),
            "reproduction": dict(
                identity,
                verdict="MATCH",
                reused_cargo_units=0,
                compiled_units=288,
                source_clean_after=True,
            ),
        }
        self.refs = {}
        for name, value in self.data.items():
            self.write(name, value)
        self.data["m2"] = dict(
            identity,
            status="PASS_EXACT_ORIGINAL_LINK_INPUTS",
            compile_exit=0,
            replay_exit=0,
            original_link_inputs_sha256=self.refs["link_inputs"]["sha256"],
            report_sha256=self.refs["m2_report"]["sha256"],
        )
        self.write("m2", self.data["m2"])
        self.pin = {
            "source_commit": source,
            "sha256": binary,
            "admission_inputs": {k: v["sha256"] for k, v in self.refs.items()},
        }
        self.receipt = dict(
            identity,
            status="PASS_WITH_WAIVER",
            release_version="v1.0.0",
            backend_sha256=self.plan["backend_sha256"],
            pins=pins,
            fold_count=3,
            bundle_sha256=publisher.BUNDLE,
            dope_sha256=publisher.DOPE,
            charged_admissions={"heuristic": 24, "learned": 24},
            learned_forward_count=4,
            qualified_dataset_cut_identities=[
                {k: dataset[k] for k in ["id", "dataset_sha256", "cut_sha256"]}
            ],
            components=self.refs,
            m3_waiver_ruling=str(self.m3.ruling),
            M3_raw_status="NUMERICAL_FAIL",
            M3_disposition="WAIVED_V1_0_0",
            executed_qualification_checks=10,
            passed_qualification_checks=9,
            failed_qualification_checks=1,
            waived_qualification_checks=1,
            unwaived_failed_qualification_checks=0,
        )
        self.ruling_sha = digest(self.m3.ruling)
        original_module = publisher.module

        def source_module(name, code, filename):
            value = original_module(name, code, filename)
            if name == "_q1_waiver_gates":
                value.M3_WAIVER_RULING_SHA256 = self.ruling_sha
                value._source_proposer_sha256 = lambda source: (
                    "48ebbfc081451baf0470ae25e1ae04621823ac4de3dfb0d8e93576206c8dbe39"
                )
            return value

        self.addCleanup(patch.stopall)
        patch.object(publisher, "EVIDENCE", self.root).start()
        patch.object(publisher, "REPO", ROOT).start()
        patch.object(
            publisher,
            "raw",
            side_effect=lambda source, path: (ROOT / path).read_bytes(),
        ).start()
        patch.object(publisher, "module", side_effect=source_module).start()

    def write(self, name, value, repin=False):
        path = self.root / (name + ".json")
        path.write_text(json.dumps(value))
        self.refs[name] = {"path": str(path), "sha256": digest(path)}
        if repin:
            self.pin["admission_inputs"][name] = self.refs[name]["sha256"]

    def admit(self):
        return publisher.validate_runtime_receipt(
            self.receipt, self.plan, self.pin, "b" * 40
        )

    def test_retains_one_raw_failure_and_returns_only_explicit_waiver(self):
        before = copy.deepcopy(self.receipt)
        result = self.admit()
        self.assertEqual(result["disposition"], "WAIVED_V1_0_0")
        self.assertEqual(self.receipt, before)
        self.assertEqual(self.receipt["failed_qualification_checks"], 1)

    def test_zeroed_failure_missing_waiver_and_future_release_are_refused(self):
        for key, value in [
            ("failed_qualification_checks", 0),
            ("waived_qualification_checks", 0),
            ("unwaived_failed_qualification_checks", 1),
            ("status", "PASS"),
            ("M3_raw_status", "NUMERICAL_PASS"),
            ("release_version", "v1.0.1"),
        ]:
            with self.subTest(key=key):
                before = self.receipt[key]
                self.receipt[key] = value
                try:
                    with self.assertRaises(ValueError):
                        self.admit()
                finally:
                    self.receipt[key] = before

    def test_compiled_metadata_source_and_evaluator_mismatches_are_refused(self):
        for target, key in [
            (self.pin, "source_commit"),
            (self.pin, "sha256"),
            (self.plan, "metadata_source_commit"),
        ]:
            before = target[key]
            target[key] = "c" * 40
            try:
                with self.assertRaises(ValueError):
                    self.admit()
            finally:
                target[key] = before

    def test_changed_missing_and_unreviewed_component_files_are_refused(self):
        path = Path(self.refs["native"]["path"])
        before = path.read_bytes()
        path.write_bytes(before + b" ")
        with self.assertRaises(ValueError):
            self.admit()
        path.write_bytes(before)
        ref = self.refs.pop("native")
        with self.assertRaises(ValueError):
            self.admit()
        self.refs["native"] = ref
        self.pin["admission_inputs"]["native"] = "0" * 64
        with self.assertRaises(ValueError):
            self.admit()

    def test_unwaived_cv_failure_cannot_be_admitted_even_with_new_receipt_hash(self):
        raw = self.data["m3_raw"]
        raw["clauses"]["full_train_cv"]["passed"] = False
        raw["clauses"]["full_train_cv"]["mismatches"] = 1
        raw["counts"].update(passed=2, failed=2)
        self.write("m3_raw", raw, repin=True)
        with self.assertRaises(ValueError):
            self.admit()

    def test_operator_outside_owner_guard_is_refused(self):
        raw = self.data["m3_raw"]
        rows = raw["clauses"]["r0_proposer"]["strata"]
        for row in [rows[0], rows[2]]:
            row["native_counts"]["edit_replace_head"] += 80
            row["native_counts"]["champion_retrieval"] -= 40
            row["native_counts"]["edit_parameter"] -= 40
        self.write("m3_raw", raw, repin=True)
        with self.assertRaises(ValueError):
            self.admit()

    def test_namespace_overlap_and_nonclean_reproduction_are_refused(self):
        self.data["identity_closure"]["overlap_counts"]["family"] = 1
        self.write("identity_closure", self.data["identity_closure"], repin=True)
        with self.assertRaises(ValueError):
            self.admit()
        self.data["identity_closure"]["overlap_counts"]["family"] = 0
        self.write("identity_closure", self.data["identity_closure"], repin=True)
        self.data["reproduction"]["reused_cargo_units"] = 1
        self.write("reproduction", self.data["reproduction"], repin=True)
        with self.assertRaises(ValueError):
            self.admit()

    def test_changed_ruling_is_refused(self):
        self.m3.ruling.write_bytes(self.m3.ruling.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.admit()


class PinSourceBinding(unittest.TestCase):
    def setUp(self):
        self.source = "a" * 40
        self.metadata = "b" * 40
        self.pin = {
            "schema": "jope.q1-evaluator-pin/v1",
            "status": "BUILD_IDENTITY_ONLY_NOT_RUNTIME_QUALIFICATION",
            "source_commit": self.source,
            "source_tree": "1" * 40,
            "sha256": "4e9239a1126589a42c3d80b1742d463d1b2331def59e5a08796169b4f3cec177",
            "target": "x86_64-unknown-linux-gnu",
            "recipe": {
                "path": "evaluator-repro-linux/recipe.json",
                "sha256": publisher.sha(b"{}"),
            },
            "reproductions": [
                {
                    "path": "evaluator-repro-linux/one.json",
                    "sha256": publisher.sha(b"{}"),
                },
                {
                    "path": "evaluator-repro-linux/two.json",
                    "sha256": publisher.sha(b"{}"),
                },
            ],
        }
        self.sources = []

        def raw(source, path):
            self.sources.append(source)
            return (
                json.dumps(self.pin).encode()
                if path.endswith("/evaluator-pin.json")
                else b"{}"
            )

        self.addCleanup(patch.stopall)
        patch.object(publisher, "raw", side_effect=raw).start()
        patch.object(
            publisher,
            "git",
            side_effect=lambda *args: (
                ("1" * 40).encode() if args[0] == "rev-parse" else b""
            ),
        ).start()

    def test_pin_metadata_read_from_explicit_commit_and_compiled_source_retained(self):
        pin, source, files = publisher.pin_materials(
            self.source, self.metadata, self.metadata
        )
        self.assertEqual(source, self.metadata)
        self.assertEqual(pin["source_commit"], self.source)
        self.assertEqual(len(files), 4)
        self.assertEqual(set(self.sources), {self.metadata})

    def test_old_source_wrong_tree_unbound_linux_and_escaped_metadata_refused(self):
        for key, value in [("source_commit", "c" * 40), ("source_tree", "2" * 40)]:
            before = self.pin[key]
            self.pin[key] = value
            with self.assertRaises(ValueError):
                publisher.pin_materials(self.source, self.metadata, self.metadata)
            self.pin[key] = before
        with self.assertRaises(ValueError):
            publisher.pin_materials(self.source, None, self.metadata)
        self.pin["recipe"]["path"] = "../escape.json"
        with self.assertRaises(ValueError):
            publisher.pin_materials(self.source, self.metadata, self.metadata)


class OriginalLedgerIdentity(unittest.TestCase):
    def test_same_path_replacement_ledger_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "api01-t0250"
            state.mkdir()
            ledger = root / "fixture-ledger"
            ledger.mkdir()
            identity = {
                "driver_host": os.uname().nodename.split(".")[0],
                "ledger": str(ledger),
                "resolved": str(ledger.resolve()),
                "device": ledger.stat().st_dev,
                "inode": ledger.stat().st_ino,
            }
            record = root / "api01-t0640"
            record.mkdir()
            (record / "ORIGINAL-LEDGER-IDENTITY.json").write_text(json.dumps(identity))
            with (
                patch.object(api_launcher, "E", state),
                patch.object(api_launcher, "LEDGER", ledger),
            ):
                api_launcher.original_ledger_identity()
                ledger.rename(root / "original-retained")
                ledger.mkdir()
                with self.assertRaisesRegex(ValueError, "physical ledger differs"):
                    api_launcher.original_ledger_identity()
                self.assertTrue((root / "original-retained").is_dir())


if __name__ == "__main__":
    unittest.main()
