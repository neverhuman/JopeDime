"""Protocol/statistical fixtures only; no learned model or sealed payload is executed."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import q1
import q1_bindings


from _q1_test_support import fixture_receipts, qualification_plan

class ProtocolTests(unittest.TestCase):
    def test_source_vocabulary_and_protocol_are_exactly_pinned(self):
        pins = q1_bindings.expected_pins()
        self.assertEqual(set(pins), {"vocabulary_sha256", "protocol_sha256"})
        self.assertTrue(all(len(value) == 64 for value in pins.values()))
        q1_bindings.validate_pins(pins)

    def test_source_binding_rejects_substituted_vocabulary_or_protocol(self):
        for pin_name in ("vocabulary_sha256", "protocol_sha256"):
            with self.subTest(pin_name=pin_name):
                pins = q1_bindings.expected_pins()
                pins[pin_name] = "0" * 64
                with self.assertRaisesRegex(ValueError, pin_name):
                    q1_bindings.validate_pins(pins)

    def test_q1_plan_validator_accepts_bound_artifacts(self):
        q1.validate_plan(qualification_plan())

    def test_d5_rejects_wrong_and_superseded_evaluator_hashes(self):
        import q1_dedup

        for wrong in (
            "0" * 64,
            "40bc4318abdfcc8a178039dcada4328f7144456061b3ae09a77e3583d136260e",
        ):
            plan = qualification_plan()
            plan["pins"]["evaluator_sha256"] = wrong
            with self.subTest(wrong=wrong, consumer="validator"):
                with self.assertRaisesRegex(ValueError, "evaluator_sha256"):
                    q1.validate_plan(plan)
            with self.subTest(wrong=wrong, consumer="builder"):
                with self.assertRaisesRegex(ValueError, "evaluator_sha256"):
                    q1_dedup.make_plan([], ["f" * 40], "a" * 40, wrong)

    def test_d5_rejects_tampered_reproduction_or_recipe_bytes(self):
        pin = q1.load(q1_bindings.EVALUATOR_PIN)
        for reference in (pin["recipe"], *pin["reproductions"]):
            with (
                self.subTest(path=reference["path"]),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                copied_pin = root / "evaluator-pin.json"
                copied_pin.write_bytes(q1_bindings.EVALUATOR_PIN.read_bytes())
                for item in (pin["recipe"], *pin["reproductions"]):
                    target = root / item["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(
                        (q1_bindings.EVALUATOR_PIN.parent / item["path"]).read_bytes()
                    )
                target = root / reference["path"]
                target.write_bytes(target.read_bytes() + b" ")
                with patch.object(q1_bindings, "EVALUATOR_PIN", copied_pin):
                    with self.assertRaisesRegex(ValueError, "reproduction/recipe hash"):
                        q1.validate_plan(qualification_plan())

    def test_q1_plan_validator_rejects_an_unbound_vocabulary(self):
        plan = qualification_plan()
        plan["pins"]["vocabulary_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "vocabulary_sha256"):
            q1.validate_plan(plan)

    def test_strict_json_and_preregistration_binding(self):
        for invalid in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                q1.decode(invalid)
        plan = qualification_plan()
        q1.validate_plan(plan)
        plan["preregistration_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            q1.validate_plan(plan)

    def test_sealed_only_unique_regression_cohort_and_training_closure(self):
        valid = qualification_plan()
        cases = []
        p = copy.deepcopy(valid)
        p["datasets"][0]["holdout_group"] = "dl_holdout"
        cases.append(p)
        p = copy.deepcopy(valid)
        p["datasets"][0]["holdout_group"] = "development_validation"
        cases.append(p)
        p = copy.deepcopy(valid)
        p["datasets"].pop()
        cases.append(p)
        p = copy.deepcopy(valid)
        p["datasets"][1]["family"] = p["datasets"][0]["family"]
        cases.append(p)
        p = copy.deepcopy(valid)
        p["datasets"][0]["family_authoritative"] = False
        cases.append(p)
        p = copy.deepcopy(valid)
        p["datasets"][0]["task"] = "binary"
        cases.append(p)
        p = copy.deepcopy(valid)
        p["training_families"] = [p["datasets"][0]["family"]]
        cases.append(p)
        p = copy.deepcopy(valid)
        p["training_families"] = []
        cases.append(p)
        p = copy.deepcopy(valid)
        p["datasets"][0]["path"] = "forbidden.csv"
        cases.append(p)
        p = copy.deepcopy(valid)
        p["budget"] = 23
        cases.append(p)
        p = copy.deepcopy(valid)
        p["seeds"] = [0, 0, 1]
        cases.append(p)
        p = copy.deepcopy(valid)
        p["pins"].pop("dope_sha256")
        cases.append(p)
        for plan in cases:
            with self.assertRaises(ValueError):
                q1.validate_plan(plan)

    def test_unsigned_protocol_refuses_before_any_backend_or_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = Path(tmp) / "backend"
            backend.write_text("unit fixture never executed")
            plan = qualification_plan()
            plan["backend_sha256"] = q1.sha_file(backend)
            with (
                patch.object(
                    q1,
                    "_invoke_backend",
                    side_effect=AssertionError("backend launched"),
                ),
                patch.object(
                    q1, "smoke_payload", side_effect=AssertionError("payload opened")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "OWNER_SIGNATURE_REQUIRED"):
                    q1.execute(
                        plan,
                        backend,
                        Path(tmp) / "receipt.jsonl",
                        ledger=Path(tmp) / "ledger",
                    )
            self.assertFalse((Path(tmp) / "receipt.jsonl").exists())
            self.assertFalse((Path(tmp) / "ledger").exists())

    def test_real_signature_verification_binds_document_plan_and_backend(self):
        # This temporary fixture key is not an OWNER key or an approval of this study.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = root / "fixture_key"
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
                check=True,
            )
            signers = root / "allowed_signers"
            signers.write_text("jepson " + key.with_suffix(".pub").read_text())
            prereg = root / "fixture_prereg.md"
            prereg.write_text(
                "Owner allowed-signers SHA-256: " + q1.sha_file(signers) + "\n"
            )
            plan = qualification_plan()
            plan["preregistration_sha256"] = q1.sha_file(prereg)
            message = root / "fixture_message"
            message.write_bytes(q1.approval_payload(plan))
            subprocess.run(
                [
                    "ssh-keygen",
                    "-Y",
                    "sign",
                    "-f",
                    str(key),
                    "-n",
                    "jopedime-q1",
                    str(message),
                ],
                check=True,
                capture_output=True,
            )
            q1.verify_owner(plan, message.with_suffix(".sig"), signers, prereg)
            plan["backend_sha256"] = "e" * 64
            with self.assertRaises(ValueError):
                q1.verify_owner(plan, message.with_suffix(".sig"), signers, prereg)
            signers.write_text("changed fixture key")
            with self.assertRaises(ValueError):
                q1.verify_owner(plan, message.with_suffix(".sig"), signers, prereg)

    def test_persistent_consumption_marker_refuses_repeat_before_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend"
            backend.write_text("never executed")
            plan = qualification_plan()
            plan["backend_sha256"] = q1.sha_file(backend)
            ledger = root / "ledger"
            ledger.mkdir()
            plan["ledger_root"] = str(ledger.resolve())
            (ledger / (plan["pins"]["bundle_sha256"] + ".started.json")).write_text(
                "previous unknown outcome"
            )
            with (
                patch.object(q1, "verify_owner"),
                patch.object(q1.os, "getloadavg", return_value=(0, 0, 0)),
                patch.object(
                    q1,
                    "_invoke_backend",
                    side_effect=AssertionError("backend launched"),
                ),
            ):
                with self.assertRaises(FileExistsError):
                    q1.execute(plan, backend, root / "runs.jsonl", ledger=ledger)

    def test_zero_budget_wrong_direction_and_failed_scores_are_refused(self):
        plan = qualification_plan()
        row = fixture_receipts(plan)[1]
        request = {
            "request_sha256": row["request_sha256"],
            "guidance": row["guidance"],
            "budget": 24,
        }
        for field, invalid in [
            ("charged", 0),
            ("completed", 0),
            ("fold_scores_completed", 0),
            ("loss", float("nan")),
            ("loss", -1),
            ("guidance", "learned"),
            ("metric", "auc"),
        ]:
            response = copy.deepcopy(row["response"])
            response[field] = invalid
            with self.assertRaises(ValueError):
                q1.response_valid(response, request)
        response = copy.deepcopy(row["response"])
        response["status"] = "failed"
        with self.assertRaises(ValueError):
            q1.response_valid(response, request)

    def test_failed_backend_preserves_unknown_counts_and_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend"
            backend.write_text("unit fixture")
            plan = q1.make_smoke_plan(backend)
            output = root / "runs.jsonl"
            with (
                patch.object(
                    q1, "smoke_payload", return_value={"folds": [], "x": [], "y": []}
                ),
                patch.object(q1.os, "getloadavg", return_value=(0, 0, 0)),
                patch.object(
                    q1,
                    "_invoke_backend",
                    side_effect=subprocess.TimeoutExpired("fixture", 60),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "incomplete execution panel"):
                    q1.execute(plan, backend, output)
            finished = [
                q1.decode(line)
                for line in output.read_text().splitlines()
                if q1.decode(line)["event"] == "finished"
            ]
            self.assertEqual(len(finished), 6)
            for row in finished:
                self.assertEqual(row["response"]["status"], "failed")
                self.assertIsNone(row["response"]["loss"])
                self.assertIsNone(row["response"]["charged"])
            with (
                patch.object(q1.os, "getloadavg", return_value=(0, 0, 0)),
                self.assertRaisesRegex(ValueError, "overwrite"),
            ):
                q1.execute(plan, backend, output)

    def test_owner_signed_ledger_root_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend"
            backend.write_text("unit fixture")
            plan = qualification_plan()
            plan["backend_sha256"] = q1.sha_file(backend)
            with (
                patch.object(q1, "verify_owner"),
                patch.object(
                    q1,
                    "_invoke_backend",
                    side_effect=AssertionError("backend launched"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "owner-signed plan"):
                    q1.execute(
                        plan,
                        backend,
                        root / "runs.jsonl",
                        ledger=root / "different_ledger",
                    )
            self.assertFalse((root / "different_ledger").exists())

    def test_successful_backend_protocol_survives_group_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend.py"
            backend.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "r=json.load(sys.stdin)\n"
                "json.dump({'schema':'jope.q1-response/v1','request_sha256':r['request_sha256'],"
                "'status':'complete','guidance':r['guidance'],'metric':'rmse','charged':r['budget'],"
                "'completed':r['budget'],'loss':1.0,'fold_scores_completed':2*r['budget'],"
                "'candidate_failures':[],'champion_program':'fixture','adapter':'fixture'},sys.stdout)\n"
            )
            backend.chmod(0o700)
            plan = q1.make_smoke_plan(backend)
            output = root / "receipts.jsonl"
            with (
                patch.object(
                    q1, "smoke_payload", return_value={"folds": [], "x": [], "y": []}
                ),
                patch.object(q1.os, "getloadavg", return_value=(0, 0, 0)),
            ):
                q1.execute(plan, backend, output, backend_cleanup_grace=0.02)
            finished = [
                q1.decode(line)
                for line in output.read_text().splitlines()
                if q1.decode(line)["event"] == "finished"
            ]
            self.assertEqual(len(finished), 6)
            self.assertTrue(
                all(row["response"]["status"] == "complete" for row in finished)
            )

    def test_sequential_timeout_panel_cleans_inherited_and_redirected_children(self):
        for redirect in (False, True):
            with self.subTest(redirect=redirect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                backend = root / "backend.py"
                backend.write_text(
                    "#!/usr/bin/env python3\n"
                    "import json,os,signal,subprocess,sys,time\n"
                    "from pathlib import Path\n"
                    f"root=Path({str(root)!r})\n"
                    "dispatch=root/'dispatches.txt'\n"
                    "prior=[int(p.read_text()) for p in root.glob('child-*.pid')]\n"
                    "def active(pid):\n"
                    " try:\n"
                    "  raw=Path('/proc',str(pid),'stat').read_text(); fields=raw[raw.rfind(')')+2:].split(); return fields[0] not in ('Z','X','x')\n"
                    " except FileNotFoundError: return False\n"
                    "with dispatch.open('a') as f: f.write(str(sum(active(p) for p in prior))+'\\n')\n"
                    "index=len(prior)\n"
                    "pidfile=root/f'child-{index}.pid'\n"
                    "child_code=('import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(30)')\n"
                    + (
                        "redirect=open(root/'redirected-output.log','ab')\n"
                        "subprocess.Popen([sys.executable,'-c',child_code],stdout=redirect,stderr=redirect)\n"
                        "redirect.close()\n"
                        if redirect
                        else "subprocess.Popen([sys.executable,'-c',child_code])\n"
                    )
                    + "time.sleep(30)\n"
                )
                backend.chmod(0o700)
                plan = q1.make_smoke_plan(backend)
                output = root / "receipts.jsonl"
                with (
                    patch.object(
                        q1,
                        "smoke_payload",
                        return_value={"folds": [], "x": [], "y": []},
                    ),
                    patch.object(q1.os, "getloadavg", return_value=(0, 0, 0)),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "incomplete execution panel"
                    ):
                        q1.execute(
                            plan,
                            backend,
                            output,
                            backend_timeout=0.15,
                            backend_cleanup_grace=0.03,
                        )
                dispatches = [
                    int(line)
                    for line in (root / "dispatches.txt").read_text().splitlines()
                ]
                self.assertEqual(dispatches, [0] * 6)
                finished = [
                    q1.decode(line)
                    for line in output.read_text().splitlines()
                    if q1.decode(line)["event"] == "finished"
                ]
                self.assertEqual(len(finished), 6)
                self.assertTrue(
                    all(row["response"]["status"] == "failed" for row in finished)
                )
                self.assertTrue(
                    all(
                        row["response"]["charged"] is None
                        and row["response"]["loss"] is None
                        for row in finished
                    )
                )
                for pidfile in root.glob("child-*.pid"):
                    pid = int(pidfile.read_text())
                    identity = q1._proc_identity(pid)
                    self.assertTrue(
                        identity is None or identity["state"] in ("Z", "X", "x")
                    )

    def test_spawn_or_unproven_cleanup_stops_later_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend"
            backend.write_text("fixture")
            plan = q1.make_smoke_plan(backend)
            output = root / "spawn.jsonl"
            with (
                patch.object(
                    q1, "smoke_payload", return_value={"folds": [], "x": [], "y": []}
                ),
                patch.object(q1.os, "getloadavg", return_value=(0, 0, 0)),
                patch.object(
                    q1.subprocess, "Popen", side_effect=OSError("fixture spawn failure")
                ) as spawn,
            ):
                with self.assertRaisesRegex(
                    q1.BackendSpawnError, "could not be started"
                ):
                    q1.execute(plan, backend, output, backend_timeout=0.1)
            self.assertEqual(spawn.call_count, 1)
            events = [
                q1.decode(line)["event"] for line in output.read_text().splitlines()
            ]
            self.assertEqual(events, ["started"])

            successful_backend = root / "exits.py"
            successful_backend.write_text("#!/usr/bin/env python3\npass\n")
            successful_backend.chmod(0o700)
            plan = q1.make_smoke_plan(successful_backend)
            output = root / "cleanup.jsonl"
            with (
                patch.object(
                    q1, "smoke_payload", return_value={"folds": [], "x": [], "y": []}
                ),
                patch.object(q1.os, "getloadavg", return_value=(0, 0, 0)),
                patch.object(
                    q1,
                    "_terminate_process_group",
                    side_effect=q1.BackendCleanupError("fixture cleanup failure"),
                ) as cleanup,
            ):
                with self.assertRaisesRegex(q1.BackendCleanupError, "no later request"):
                    q1.execute(plan, successful_backend, output, backend_timeout=0.1)
            self.assertEqual(cleanup.call_count, 1)
            events = [
                q1.decode(line)["event"] for line in output.read_text().splitlines()
            ]
            self.assertEqual(events, ["started"])
