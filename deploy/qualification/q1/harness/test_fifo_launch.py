"""Private lock/queue and signing fixtures; no study, real data, GPU or live queue."""

from dataclasses import replace
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fifo_launch as launch
import q1
import q1_cut
from test_q1 import qualification_plan


class FIFOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.lock = self.root / "qualification.lock"
        self.lock.touch()
        self.backend = self.root / "backend"
        self.backend.write_text("#!/bin/sh\nexit 99\n")
        self.backend.chmod(0o700)
        key = self.root / "fixture-key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True
        )
        signers = self.root / "fixture-signers"
        signers.write_text("jepson " + key.with_suffix(".pub").read_text())
        self.prereg = self.root / "fixture-prereg"
        self.prereg.write_text(
            "Owner allowed-signers SHA-256: " + q1.sha_file(signers) + "\n"
        )
        self.driver = self.root / "fixture-driver.py"
        self.driver.write_text(self.fake_driver())
        self.paths = replace(
            launch.Paths(),
            driver=self.driver,
            prereg=self.prereg,
            lock=self.lock,
            queue=self.queue,
        )
        self.plan = qualification_plan()
        self.plan.update(
            preregistration_sha256=q1.sha_file(self.prereg),
            backend_sha256=q1.sha_file(self.backend),
            ledger_root=str(self.root / "ledger"),
        )
        self.plan["pins"]["dope_sha256"] = launch.DOPE
        self.planfile = self.root / "plan.json"
        self.planfile.write_bytes(q1.canonical(self.plan))
        message = self.root / "approval"
        message.write_bytes(q1.approval_payload(self.plan))
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
        self.args = SimpleNamespace(
            plan=self.planfile,
            backend=self.backend,
            signature=message.with_suffix(".sig"),
            allowed_signers=signers,
            ledger=Path(self.plan["ledger_root"]),
            run_dir=self.root / "run",
            wait_seconds=2.0,
            wall_seconds=3.0,
            authorized_launch=True,
        )

    def fake_driver(self):
        return f"""import fcntl, hashlib, json, sys
from pathlib import Path
args = sys.argv[1:]
def value(flag): return Path(args[args.index(flag)+1])
plan = json.loads(value('--plan').read_text())
digest = hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
with Path({str(self.lock)!r}).open('rb') as other:
    try:
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise SystemExit('driver ran without the qualification lease')
out = value('--out')
if args[0]=='run':
    ledger=value('--ledger');ledger.mkdir()
    prefix=plan['pins']['bundle_sha256']
    (ledger/(prefix+'.started.json')).write_text(json.dumps({{'plan_sha256':digest,'status':'started_outcome_unknown'}}))
    out.write_text('{{"fixture_not_numerical":true}}\\n')
    finished={{'plan_sha256':digest,'status':'complete','failed_runs':0,'receipts_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}}
    (ledger/(prefix+'.finished.json')).write_text(json.dumps(finished))
else:
    out.write_text(json.dumps({{'schema':'jope.q1-analysis/v1','plan_sha256':digest,'decision':'FAIL','fixture_not_a_study':True}}))
"""

    def admit(self, **kwargs):
        return launch.admit(
            self.args,
            self.paths,
            source_check=lambda *_: None,
            load=lambda: 0,
            **kwargs,
        )

    def run_launch(self, **kwargs):
        return launch.launch(
            self.args,
            self.paths,
            source_check=lambda *_: None,
            load=lambda: 0,
            lease_options={"poll": 0.01, "backoff": 0},
            grace=0.02,
            **kwargs,
        )

    def resign(self):
        self.planfile.write_bytes(q1.canonical(self.plan))
        message = self.root / "changed-approval"
        message.write_bytes(q1.approval_payload(self.plan))
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(self.root / "fixture-key"),
                "-n",
                "jopedime-q1",
                str(message),
            ],
            check=True,
            capture_output=True,
        )
        self.args.signature = message.with_suffix(".sig")

    def test_preparation_is_not_a_live_ticket_or_lock_claim(self):
        sentinel = self.queue / "w1-owner.json"
        sentinel.write_text('{"owner":"fixture-W1"}')
        before = sentinel.read_bytes()
        with patch.object(fcntl, "flock", side_effect=AssertionError("lock touched")):
            intent = launch.prepare(self.paths, "unit-fixture")
        self.assertEqual(intent["status"], "prepared_not_enqueued")
        self.assertIsNone(intent["active_ticket"])
        self.assertIsNone(intent["planning"]["expected_wall_seconds"])
        self.assertEqual(intent["planning"]["serial_timeout_seconds"], 144000)
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual(list(self.queue.iterdir()), [sentinel])

    def test_synthetic_dry_run_writes_unknown_decision_without_lock_or_payload(self):
        sentinel = self.queue / "w1-owner.json"
        sentinel.write_text('{"owner":"fixture-W1"}')
        before = sentinel.read_bytes()
        output = self.root / "decision-record.json"
        with patch.object(fcntl, "flock", side_effect=AssertionError("lock touched")):
            manifest = launch.load_input_manifest(self.paths.input_manifest, self.paths)
            record = launch.dry_run(self.paths.input_manifest, output, self.paths)
        self.assertTrue(manifest["datasets"])
        self.assertEqual(
            set(manifest["datasets"][0]),
            {"dataset_id", "sha256", "s3_manifest_sha256"},
        )
        self.assertEqual(record["schema"], "jope.q1-decision-record/v1")
        self.assertEqual(record["q1_status"], "NOT_RUN")
        self.assertEqual(record["decision"], "UNKNOWN_NOT_RUN")
        self.assertIsNone(record["primary_effect"])
        self.assertEqual(record["failure_inventory"], "NOT_RUN")
        self.assertFalse(record["controls"]["qualification_lock_acquired"])
        self.assertFalse(record["controls"]["fifo_ticket_created"])
        self.assertFalse(record["controls"]["backend_invoked"])
        self.assertFalse(record["controls"]["dataset_payloads_opened"])
        self.assertFalse(record["controls"]["default_changed"])
        self.assertEqual(q1.load(output), record)
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual(list(self.queue.iterdir()), [sentinel])
        with self.assertRaises(FileExistsError):
            launch.dry_run(self.paths.input_manifest, output, self.paths)

    def test_synthetic_dry_run_consumes_amended_roster_three_fold_cut(self):
        cut_root = self.root / "cuts"
        cut_root.mkdir()
        first_dataset_sha = f"{1:016x}" + "a" * 48
        cut = {
            "schema": q1_cut.SCHEMA,
            "dataset_sha256": first_dataset_sha,
            "train_blob_sha256": "b" * 64,
            "train_rows": 6,
            "target": "target",
            "feature_columns": ["x"],
            "folds": [
                {"repeat": 0, "train": [2, 3, 4, 5], "validation": [0, 1]},
                {"repeat": 0, "train": [0, 1, 4, 5], "validation": [2, 3]},
                {"repeat": 0, "train": [0, 1, 2, 3], "validation": [4, 5]},
            ],
        }
        cut_sha = q1_cut.cut_sha256(cut)
        (cut_root / f"{cut_sha}.json").write_bytes(q1_cut.canonical_bytes(cut))
        datasets = []
        for index in range(40):
            dataset_sha = f"{index + 1:016x}" + "a" * 48
            row = {
                "dataset_id": dataset_sha[:16],
                "sha256": dataset_sha,
                "s3_manifest_sha256": "b" * 64,
            }
            if index == 0:
                row["cut_sha256"] = cut_sha
            datasets.append(row)
        cohort = self.root / "amended-roster.json"
        cohort.write_bytes(
            q1.canonical(
                {
                    "schema": "jope.q1.held-out-cohort/v1",
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                    "source": {"catalog_sha256": "c" * 64},
                }
            )
        )
        paths = replace(
            self.paths,
            cohort=cohort,
            input_manifest=self.root / "inputs-manifest.json",
            cut_definitions=cut_root,
        )
        manifest = launch.build_input_manifest(paths)
        self.assertEqual(manifest["datasets"][0]["cut_sha256"], cut_sha)
        self.assertNotIn("cut_sha256", manifest["datasets"][1])
        paths.input_manifest.write_bytes(q1.canonical(manifest))
        output = self.root / "amended-dry-run.json"
        record = launch.dry_run(paths.input_manifest, output, paths)
        self.assertEqual(record["q1_status"], "NOT_RUN")
        self.assertEqual(record["decision"], "UNKNOWN_NOT_RUN")
        self.assertFalse(record["qualification_eligible"])
        self.assertFalse(record["controls"]["dataset_payloads_opened"])

    def test_input_manifest_rejects_a_non_three_fold_roster_cut(self):
        cut_root = self.root / "cuts"
        cut_root.mkdir()
        first_dataset_sha = f"{1:016x}" + "a" * 48
        definition = {
            "schema": q1_cut.SCHEMA,
            "dataset_sha256": first_dataset_sha,
            "train_blob_sha256": "b" * 64,
            "train_rows": 4,
            "target": "target",
            "feature_columns": ["x"],
            "folds": [
                {"repeat": 0, "train": [0, 1], "validation": [2, 3]},
                {"repeat": 0, "train": [2, 3], "validation": [0, 1]},
            ],
        }
        cut_sha = q1_cut.cut_sha256(definition)
        (cut_root / f"{cut_sha}.json").write_bytes(q1_cut.canonical_bytes(definition))
        datasets = []
        for index in range(40):
            dataset_sha = f"{index + 1:016x}" + "a" * 48
            row = {
                "dataset_id": dataset_sha[:16],
                "sha256": dataset_sha,
                "s3_manifest_sha256": "b" * 64,
            }
            if index == 0:
                row["cut_sha256"] = cut_sha
            datasets.append(row)
        paths = replace(
            self.paths,
            cohort=self.root / "bad-amended-roster.json",
            cut_definitions=cut_root,
        )
        paths.cohort.write_bytes(
            q1.canonical(
                {
                    "schema": "jope.q1.held-out-cohort/v1",
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                    "source": {"catalog_sha256": "c" * 64},
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly three FIT folds"):
            launch.build_input_manifest(paths)

    def test_consumes_real_t0610_amended_roster_cut_packet_and_cuts(self):
        summary = launch.load_amended_roster_packet(self.paths)
        self.assertEqual(summary["schema"], "jope.q1-amended-roster-summary/v1")
        self.assertEqual(summary["amendment_id"], "QLT-04-T0610-R1-R2")
        self.assertEqual(
            summary["roster_cut_packet_sha256"],
            "ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a",
        )
        self.assertEqual(
            summary["fold_manifest_sha256"],
            "fdc243193d9436449cf74aa92b05ff7842c6c88bf4dce0b29ef7edf453b74987",
        )
        self.assertEqual(summary["official_roster_count"], 153)
        self.assertEqual(len(summary["datasets"]), 153)
        self.assertFalse(summary["qualification_eligible"])
        self.assertEqual(summary["q1_status"], "NOT_RUN")
        datasets = [
            {
                "dataset_id": row["dataset_id"],
                "sha256": row["sha256"],
                "s3_manifest_sha256": row["train_blob_sha256"],
                "cut_sha256": row["cut_sha256"],
            }
            for row in summary["datasets"]
        ]
        cohort = self.root / "real-t0610-roster.json"
        cohort.write_bytes(
            q1.canonical(
                {
                    "schema": "jope.q1.held-out-cohort/v1",
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                    "source": {"catalog_sha256": summary["fold_manifest_sha256"]},
                }
            )
        )
        paths = replace(
            self.paths,
            cohort=cohort,
            input_manifest=self.root / "real-t0610-inputs-manifest.json",
        )
        manifest = launch.build_input_manifest(paths)
        self.assertEqual(len(manifest["datasets"]), 153)
        self.assertEqual(
            manifest["datasets"][0]["cut_sha256"],
            "8b686be059e981dab5cf3245d8c7e2f33e7af2cca0a93935b5e7a7083d91645c",
        )
        paths.input_manifest.write_bytes(q1.canonical(manifest))
        output = self.root / "real-t0610-dry-run.json"
        record = launch.dry_run(paths.input_manifest, output, paths)
        self.assertEqual(record["real_s3_metadata_entries"], 153)
        self.assertEqual(record["q1_status"], "NOT_RUN")
        self.assertFalse(record["controls"]["dataset_payloads_opened"])

    def test_synthetic_dry_run_refuses_a_changed_dope_manifest(self):
        source = q1.load(self.paths.input_manifest)
        source["dope_sha256"] = "f" * 64
        manifest = self.root / "bad-input-manifest.json"
        manifest.write_bytes(q1.canonical(source))
        output = self.root / "must-not-exist.json"
        with self.assertRaisesRegex(ValueError, "frozen DOPE differs"):
            launch.dry_run(manifest, output, self.paths)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_unsigned_preregistration_refuses_before_queue_output_or_ledger(self):
        self.prereg.write_text("Owner allowed-signers SHA-256: PENDING\n")
        self.plan["preregistration_sha256"] = q1.sha_file(self.prereg)
        self.planfile.write_bytes(q1.canonical(self.plan))
        with self.assertRaisesRegex(ValueError, "OWNER_SIGNATURE_REQUIRED"):
            self.run_launch()
        self.assertFalse(self.args.run_dir.exists())
        self.assertFalse(self.args.ledger.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_explicit_intent_and_finite_caps_are_required_before_effects(self):
        self.args.authorized_launch = False
        with self.assertRaisesRegex(ValueError, "authorized-launch"):
            self.run_launch()
        self.args.authorized_launch = True
        for field, value in [
            ("wait_seconds", 0),
            ("wait_seconds", 10801),
            ("wait_seconds", math.nan),
            ("wall_seconds", math.inf),
            ("wall_seconds", -1),
            ("wall_seconds", 144001),
        ]:
            prior = getattr(self.args, field)
            setattr(self.args, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.run_launch()
            setattr(self.args, field, prior)
        with self.assertRaises(ValueError):
            launch.limits(10**1000, 10800, "test")
        self.assertFalse(self.args.run_dir.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_bad_signature_backend_and_dope_never_enqueue(self):
        self.args.signature.write_text("not a signature")
        with self.assertRaisesRegex(ValueError, "signature"):
            self.admit()
        self.backend.write_text("changed")
        with self.assertRaisesRegex(ValueError, "backend hash"):
            self.admit()
        self.plan["pins"]["dope_sha256"] = "f" * 64
        self.planfile.write_bytes(q1.canonical(self.plan))
        with self.assertRaisesRegex(ValueError, "DOPE"):
            self.admit()
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_source_must_be_clean_exact_and_merged(self):
        with patch.object(subprocess, "check_output", return_value=b"b" * 40):
            with self.assertRaisesRegex(ValueError, "signed source differs"):
                launch.source_identity(self.paths, "a" * 40)
        with patch.object(
            subprocess, "check_output", side_effect=[b"a" * 40, b" M file"]
        ):
            with self.assertRaisesRegex(ValueError, "clean checkout"):
                launch.source_identity(self.paths, "a" * 40)
        with patch.object(
            subprocess,
            "check_output",
            side_effect=[
                b"a" * 40,
                b"",
                subprocess.CalledProcessError(1, ["git", "merge-base"]),
            ],
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                launch.source_identity(self.paths, "a" * 40)

    def test_consumed_or_aliased_ledger_refuses_without_new_run(self):
        self.args.ledger.mkdir()
        for suffix in ("started", "finished"):
            marker = self.args.ledger / (
                self.plan["pins"]["bundle_sha256"] + "." + suffix + ".json"
            )
            marker.write_text("existing evidence")
            with self.assertRaisesRegex(ValueError, "consumed"):
                self.run_launch()
            self.assertEqual(marker.read_text(), "existing evidence")
            marker.unlink()
        alias = self.root / "alias"
        alias.symlink_to(self.args.ledger, target_is_directory=True)
        self.args.ledger = alias
        with self.assertRaisesRegex(ValueError, "canonical owner-signed"):
            self.admit()
        self.assertFalse(self.args.run_dir.exists())

    def test_fifo_waits_for_earlier_owner_without_pruning_any_other_ticket(self):
        fifo = launch.fifo_module(self.paths)
        earlier = fifo.enqueue_ticket(
            self.queue, pid=os.getpid(), wave="W1fixture", epoch_ns=1
        )
        dead = self.queue / "2-999999999-deadfixture"
        dead.write_text("retained dead ticket")
        notes = []

        def sleep(_seconds):
            notes.append("waited")
            if len(notes) == 2:
                # First callback is the zero courtesy delay. The next callback
                # proves that the earlier live owner prevented admission.
                self.assertEqual(launch.oldest_readonly(fifo, self.queue), earlier)
                earlier.unlink()  # Fixture owner releases its own earlier ticket.

        with launch.lease(
            self.paths, fifo, 1, lambda: 0, sleep=sleep, poll=0.01, backoff=0
        ):
            self.assertEqual(len(notes), 2)
            self.assertTrue(dead.exists())
        self.assertEqual(list(self.queue.iterdir()), [dead])

    def test_real_busy_lock_and_wait_cap_cannot_be_preempted(self):
        fifo = launch.fifo_module(self.paths)
        with self.lock.open("rb") as w1:
            fcntl.flock(w1, fcntl.LOCK_EX)
            with self.assertRaisesRegex(TimeoutError, "wait cap"):
                with launch.lease(
                    self.paths, fifo, 0.04, lambda: 0, poll=0.01, backoff=0
                ):
                    self.fail("preempted W1")
            with self.lock.open("rb") as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_missing_or_symlinked_lock_refuses_without_replacing_it(self):
        fifo = launch.fifo_module(self.paths)
        self.lock.unlink()
        with self.assertRaises(FileNotFoundError):
            with launch.lease(self.paths, fifo, 1, lambda: 0):
                self.fail("missing lock admitted")
        real = self.root / "real-lock"
        real.touch()
        self.lock.symlink_to(real)
        with self.assertRaises(OSError):
            with launch.lease(self.paths, fifo, 1, lambda: 0):
                self.fail("symlinked lock admitted")
        self.assertTrue(self.lock.is_symlink())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_changed_inputs_after_wait_refuse_and_keep_failure_receipt(self):
        checks = []

        def source_check(*_):
            checks.append(1)
            if len(checks) == 2:
                self.backend.write_text("changed while waiting")

        with self.assertRaisesRegex(ValueError, "changed while waiting"):
            launch.launch(
                self.args,
                self.paths,
                source_check=source_check,
                load=lambda: 0,
                lease_options={"poll": 0.01, "backoff": 0},
                grace=0.02,
            )
        receipt = q1.load(self.args.run_dir / "launch-receipt.json")
        self.assertEqual(receipt["last_phase"], "locked")
        self.assertFalse((self.args.run_dir / "run.log").exists())
        self.assertFalse(self.args.ledger.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_positive_run_and_analysis_hold_lease_and_do_not_promote_fail(self):
        result = self.run_launch()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["q1_decision"], "FAIL")
        self.assertFalse(result["default_changed"])
        self.assertTrue((self.args.run_dir / "receipts.jsonl").exists())
        self.assertTrue((self.args.run_dir / "analysis.json").exists())
        self.assertEqual(list(self.queue.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.run_launch()

    def test_source_or_receipt_mutation_blocks_analysis(self):
        original = launch.run_owned

        def mutate(command, *args):
            code = original(command, *args)
            if "run" in command:
                (self.args.run_dir / "receipts.jsonl").write_text(
                    "modified after producer finish"
                )
            return code

        with patch.object(launch, "run_owned", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "ledger/receipt binding"):
                self.run_launch()
        self.assertFalse((self.args.run_dir / "analysis.json").exists())
        self.assertTrue(self.args.ledger.exists())

    def test_owned_process_deadline_and_inherited_lock_descriptor(self):
        with self.lock.open("rb") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            fd = lease.fileno()
            marker = self.root / "descriptor-was-inherited"
            code = f"import os;from pathlib import Path;os.fstat({fd});Path({str(marker)!r}).touch()"
            self.assertEqual(
                launch.run_owned(
                    [sys.executable, "-c", code],
                    self.root / "fd.log",
                    dict(os.environ),
                    1,
                    fd,
                    lambda: 0,
                    0.02,
                ),
                0,
            )
            self.assertTrue(marker.exists())
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "wall cap"):
                launch.run_owned(
                    [sys.executable, "-c", "import time;time.sleep(2)"],
                    self.root / "timeout.log",
                    dict(os.environ),
                    0.05,
                    fd,
                    lambda: 0,
                    0.02,
                )
            self.assertLess(time.monotonic() - started, 1)

    def test_leader_exit_still_kills_its_term_ignoring_descendant(self):
        pidfile = self.root / "descendant.pid"
        script = self.root / "leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(2)"
        script.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\n"
        )
        with self.lock.open("rb") as handle:
            self.assertEqual(
                launch.run_owned(
                    [sys.executable, str(script)],
                    self.root / "descendant.log",
                    dict(os.environ),
                    1,
                    handle.fileno(),
                    lambda: 0,
                    0.02,
                ),
                0,
            )
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 0.5
        while self.pid_is_active(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.pid_is_active(pid))

    @staticmethod
    def pid_is_active(pid):
        identity = q1._proc_identity(pid)
        return identity is not None and identity["state"] not in ("Z", "X", "x")

    def test_wall_timeout_cleans_an_isolated_session_descendant(self):
        pidfile = self.root / "timeout-descendant.pid"
        script = self.root / "timeout-leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(30)"
        script.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\ntime.sleep(30)\n"
        )
        with self.lock.open("rb") as handle:
            with self.assertRaisesRegex(TimeoutError, "wall cap"):
                launch.run_owned(
                    [sys.executable, str(script)],
                    self.root / "timeout-descendant.log",
                    dict(os.environ),
                    0.2,
                    handle.fileno(),
                    lambda: 0,
                    0.02,
                )
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 0.5
        while self.pid_is_active(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.pid_is_active(pid))

    def test_session_cleanup_retries_identity_error_before_returning(self):
        pidfile = self.root / "retry-descendant.pid"
        script = self.root / "retry-leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(30)"
        script.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\n"
        )
        original = q1._signal_scope
        attempts = 0

        def fail_once(session_id, process_group, signum):
            nonlocal attempts
            if attempts == 0:
                attempts += 1
                raise q1.BackendCleanupError("synthetic first cleanup failure")
            return original(session_id, process_group, signum)

        with (
            self.lock.open("rb") as handle,
            patch.object(q1, "_signal_scope", side_effect=fail_once),
        ):
            self.assertEqual(
                launch.run_owned(
                    [sys.executable, str(script)],
                    self.root / "retry-descendant.log",
                    dict(os.environ),
                    1,
                    handle.fileno(),
                    lambda: 0,
                    0.02,
                ),
                0,
            )
        self.assertGreaterEqual(attempts, 1)
        pid = int(pidfile.read_text())
        self.assertFalse(self.pid_is_active(pid))

    def test_sigterm_interrupt_cleans_a_term_ignoring_isolated_descendant(self):
        pidfile = self.root / "interrupt-descendant.pid"
        leader = self.root / "interrupt-leader.py"
        child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(30)"
        leader.write_text(
            f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c',{child!r}],process_group=0)\nwhile not Path({str(pidfile)!r}).exists():time.sleep(0.01)\ntime.sleep(30)\n"
        )
        runner = self.root / "interrupt-runner.py"
        runner.write_text(
            "import os,sys\n"
            f"sys.path.insert(0,{str(Path(launch.__file__).parent)!r})\n"
            "import fifo_launch\n"
            "from pathlib import Path\n"
            f"with open({str(self.lock)!r},'rb') as handle:\n"
            " with fifo_launch.termination_handler():\n"
            f"  fifo_launch.run_owned([sys.executable,{str(leader)!r}],Path({str(self.root / 'interrupt.log')!r}),dict(os.environ),30,handle.fileno(),lambda:0,0.02)\n"
        )
        runner_stderr = (self.root / "interrupt-runner.stderr").open("wb")
        runner_process = subprocess.Popen(
            [sys.executable, str(runner)],
            stdout=subprocess.DEVNULL,
            stderr=runner_stderr,
        )
        try:
            deadline = time.monotonic() + 2
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(
                pidfile.exists(),
                "fixture driver did not launch: "
                + (self.root / "interrupt-runner.stderr").read_text(),
            )
            os.kill(runner_process.pid, signal.SIGTERM)
            self.assertNotEqual(runner_process.wait(timeout=2), 0)
        finally:
            if runner_process.poll() is None:
                runner_process.kill()
                runner_process.wait(timeout=2)
            runner_stderr.close()
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 0.5
        while self.pid_is_active(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.pid_is_active(pid))

    def test_high_load_and_termination_refuse_or_cleanup_without_other_jobs(self):
        with self.assertRaisesRegex(ValueError, "host load"):
            launch.admit(
                self.args, self.paths, source_check=lambda *_: None, load=lambda: 86
            )
        with self.lock.open("rb") as handle:
            with self.assertRaisesRegex(ValueError, "host load"):
                launch.run_owned(
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    self.root / "must-not-start.log",
                    dict(os.environ),
                    1,
                    handle.fileno(),
                    lambda: 86,
                    0.02,
                )
        self.assertFalse((self.root / "must-not-start.log").exists())
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(InterruptedError):
            with launch.termination_handler():
                signal.raise_signal(signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_prepare_output_and_run_directory_never_overwrite(self):
        out = self.root / "intent.json"
        launch.write_new(out, {"fixture": True})
        with self.assertRaises(FileExistsError):
            launch.write_new(out, {"overwrite": True})
        self.assertEqual(q1.load(out), {"fixture": True})
        self.args.run_dir.mkdir()
        sentinel = self.args.run_dir / "prior-evidence"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "new canonical run"):
            self.run_launch()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(self.queue.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
