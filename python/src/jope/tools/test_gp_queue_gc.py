import contextlib
import fcntl
import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import gp_queue_gc as gc
from resume_expired_cycle import read, sha

FOREIGN = "f" * 64
PREPARED = "data/prepared/views-expanded-20/55e9dfc7a86361d7"
RELEASE = Path(__file__).resolve().parents[1] / "recovery/eight-a100-20260912"
SEED_DISPATCH_ROOT = RELEASE / "experiments/seed-dispatch-service-qualification/private-root"
SEED_DISPATCH_EVALUATOR = "b86502912e93cbfbbc9e6697eb5c10c0c006678012c56ff3a84ff3d83f276232"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def job(index, generation=16, **extra):
    base = dict(
        prepared=PREPARED,
        prepared_sha256="1b" * 32,
        cut=index % 2,
        output=f"runs/eight-a100-recovery-20260912/learning/generation-{generation:06d}/neural/ctx{index}/cut-{index % 2}/search",
        seeds=f"runs/eight-a100-recovery-20260912/learning/generation-{generation:06d}/neural/ctx{index}/cut-{index % 2}/seeds.json",
        seeds_sha256=hashlib.sha256(f"seeds{index}".encode()).hexdigest(),
        rng_seed=8899393899179896416 + index,
        generation=generation,
        parent_checkpoint="runs/x/checkpoints/step-3000",
        parent_checkpoint_sha256="c2" * 32,
    )
    base.update(extra)
    return base


def observation(evaluator, **identity):
    row = dict(
        schema="parallel.gp_observation.v1",
        evaluation_index=0,
        evaluator_sha256=evaluator,
        prepared_sha256="1b" * 32,
        cut=0,
        genome="(pipe X)",
        failure=None,
        metrics=dict(rmse=0.7, semantic_sha256="ab" * 32),
    )
    row.update(identity)
    return row


def write_ledger(root, output, evaluator, name="observations.jsonl", progress=True, **identity):
    directory = root / output
    directory.mkdir(parents=True, exist_ok=True)
    row = observation(evaluator, **identity)
    if name.endswith(".jsonl"):
        (directory / name).write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    else:
        (directory / name).write_text(json.dumps([row]))
    if progress:
        receipt = dict(prepared_sha256="1b" * 32, cut=0, evaluations=1)
        receipt.update(progress if isinstance(progress, dict) else {})
        (directory / "progress.json").write_text(json.dumps(receipt))


def write_failure(root, name, output, attempt, timestamp, **extra):
    directory = root / "runs/gp-failures"
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(timestamp=timestamp, output=output, attempt=attempt, exit_code=1, retry_after_seconds=320)
    row.update(extra)
    (directory / name).write_text(json.dumps(row))


class Fixture:
    """Priority file: completed / poisoned / pending-with-ledger / pending-no-ledger / malformed x2 /
    identity-mismatch receipt / wrong-schema search.json, plus gp-failures receipts."""

    def __init__(self, root, bind=True):
        self.root = root
        (root / "runs").mkdir()
        (root / "bin").mkdir()
        (root / "bin/gp-native-v5").write_bytes(b"frozen evaluator bytes v5")
        self.evaluator = sha(root / "bin/gp-native-v5")
        self.manifest = root / "recovery-manifest-v4.json"
        manifest = dict(schema="parallel.recovery_manifest.v1", cpu_executable="bin/gp-native-v5")
        if bind:
            manifest["bindings"] = {"bin/gp-native-v5": self.evaluator, "bin/other": "e" * 64}
        self.manifest.write_text(json.dumps(manifest, indent=2))
        self.jobs = [
            job(0),
            job(1),
            job(2),
            job(3),
            dict(prepared=PREPARED, cut=0),
            job(5, output="../escape/search"),
            job(6),
            job(7),
        ]
        self.completed_output = self.jobs[0]["output"]
        self.poisoned_output = self.jobs[1]["output"]
        self.identity_output = self.jobs[6]["output"]
        self.wrong_schema_output = self.jobs[7]["output"]
        # 0: completed
        directory = root / self.completed_output
        directory.mkdir(parents=True)
        (directory / "search.json").write_text(
            json.dumps(dict(schema="parallel.search.v1", completed_evaluations=2))
        )
        # 1: poisoned by a foreign-evaluator ledger, two failure receipts
        write_ledger(root, self.poisoned_output, FOREIGN)
        write_failure(root, "1789300000-100-1.json", self.poisoned_output, 1, 1789300000)
        write_failure(root, "1789300320-100-2.json", self.poisoned_output, 2, 1789300320)
        # 2: pending, ledger written by the frozen evaluator (committed observations.json + progress)
        write_ledger(root, self.jobs[2]["output"], self.evaluator, name="observations.json")
        # 3: pending, nothing on disk yet
        # 4: malformed (no output); 5: malformed (escapes the workspace)
        # 6: no ledger but a failure receipt diagnosing an identity mismatch
        write_failure(
            root,
            "1789300500-100-3.json",
            self.identity_output,
            1,
            1789300500,
            diagnosis="observation ledger identity mismatch",
        )
        # 7: search.json present but with the wrong schema
        directory = root / self.wrong_schema_output
        directory.mkdir(parents=True)
        (directory / "search.json").write_text(json.dumps(dict(schema="parallel.other.v1")))
        self.document = dict(
            schema="parallel.gp_jobs.v1",
            qualification=True,
            jobs=self.jobs,
            vocabulary="runs/corpus-2/vocabulary.json",
        )
        self.priority = root / "runs/gp-priority-jobs.json"
        self.priority.write_bytes(gc.pretty(self.document))
        self.dead_letter = root / "runs/gp-dead-letter.json"
        self.receipt = root / "runs/gp-queue-gc-audit.json"

    def audit(self, **kw):
        return gc.audit(self.root, self.manifest, self.priority, self.dead_letter, self.receipt, **kw)

    def apply(self, **kw):
        return gc.apply(
            self.root,
            self.manifest,
            self.priority,
            self.dead_letter,
            self.root / "runs/gp-queue-gc-apply.json",
            **kw,
        )


class AuditTests(unittest.TestCase):
    def test_audit_classifies_every_job_and_writes_dead_letter(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            receipt = f.audit()
            self.assertEqual(receipt["schema"], gc.RECEIPT_SCHEMA)
            self.assertEqual(receipt["mode"], "audit")
            self.assertFalse(receipt["applied"])
            self.assertFalse(receipt["lock_taken"])
            self.assertTrue(receipt["binding_checked"])
            self.assertFalse(receipt["final_tests_opened"])
            self.assertEqual(receipt["manifest_sha256"], sha(f.manifest))
            self.assertEqual(receipt["evaluator_sha256"], f.evaluator)
            self.assertEqual(receipt["lock_path"], str(f.root / "runs/gp-priority-jobs.lock"))
            self.assertEqual(
                (
                    receipt["before"],
                    receipt["removed_completed"],
                    receipt["removed_dead"],
                    receipt["after"],
                    receipt["kept_pending"],
                    receipt["kept_malformed"],
                ),
                (8, 1, 2, 5, 2, 3),
            )
            self.assertEqual(
                receipt["before"] - receipt["removed_completed"] - receipt["removed_dead"], receipt["after"]
            )
            dead = read(f.dead_letter)
            self.assertEqual(dead["schema"], gc.DEAD_LETTER_SCHEMA)
            self.assertFalse(dead["final_tests_opened"])
            self.assertEqual(dead["priority_sha256"], sha(f.priority))
            self.assertEqual(dead["expected_evaluator_sha256"], f.evaluator)
            self.assertEqual(dead["completed"], [f.completed_output])
            self.assertEqual(dead["counts"], dict(completed=1, dead=2, pending=2, malformed=3))
            by_output = {e["output"]: e for e in dead["entries"]}
            self.assertEqual(set(by_output), {f.poisoned_output, f.identity_output})
            poisoned = by_output[f.poisoned_output]
            self.assertEqual(poisoned["reason"], "ledger_evaluator_mismatch")
            self.assertEqual(poisoned["ledger_evaluator_sha256"], FOREIGN)
            self.assertEqual(poisoned["attempts"], 2)
            self.assertEqual((poisoned["first_seen"], poisoned["last_seen"]), (1789300000, 1789300320))
            self.assertEqual(poisoned["job_sha256"], hashlib.sha256(canonical(f.jobs[1])).hexdigest())
            self.assertEqual(poisoned["ledger"], "observations.jsonl")
            identity = by_output[f.identity_output]
            self.assertEqual(identity["reason"], "identity_mismatch")
            self.assertEqual(identity["mismatched_fields"], ["gp-failures receipt"])
            self.assertIsNone(identity["ledger_evaluator_sha256"])
            self.assertEqual(identity["attempts"], 1)
            for entry in dead["entries"]:
                self.assertIn(entry["reason"], gc.REASONS)
                self.assertEqual(
                    set(entry)
                    >= {
                        "output",
                        "job_sha256",
                        "reason",
                        "attempts",
                        "first_seen",
                        "last_seen",
                        "ledger_evaluator_sha256",
                    },
                    True,
                )
            malformed = {m["index"]: m["reason"] for m in dead["malformed"]}
            self.assertEqual(set(malformed), {4, 5, 7})
            self.assertIn("output missing or unsafe", malformed[4])
            self.assertIn("output missing or unsafe", malformed[5])
            self.assertIn("schema", malformed[7])
            self.assertEqual(sha(f.dead_letter), receipt["dead_letter_sha256"])

    def test_matching_ledger_is_pending_and_empty_ledger_is_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "x").mkdir()
            (root / "x/observations.jsonl").write_text("")
            self.assertEqual(gc.first_ledger_row(root / "x"), ("observations.jsonl", None))
            self.assertEqual(gc.first_ledger_row(root / "missing"), (None, None))
            (root / "x/observations.jsonl").write_text("{not json\n")
            with self.assertRaises(ValueError):
                gc.first_ledger_row(root / "x")
            failures = {}
            result = gc.classify_job(root, dict(output="x"), "e" * 64, failures)
            self.assertEqual(result["state"], "malformed")
            self.assertIn("ledger unreadable", result["reason"])
            (root / "x/observations.jsonl").write_text(json.dumps(observation("e" * 64)) + "\n")
            self.assertEqual(gc.classify_job(root, dict(output="x"), "e" * 64, failures)["state"], "pending")
            self.assertEqual(
                gc.classify_job(root, dict(output="x"), "d" * 64, failures)["reason"],
                "ledger_evaluator_mismatch",
            )
            (root / "x/progress.json").write_text("{broken")
            result = gc.classify_job(root, dict(output="x"), "e" * 64, failures)
            self.assertEqual(result["state"], "malformed")
            self.assertIn("progress.json", result["reason"])

    def test_committed_ledger_without_progress_is_ignored_like_the_service_does(self):
        # gp.rs:649-653: observations.json is only read when progress.json marks a resume.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_ledger(root, "y", FOREIGN, name="observations.json", progress=False)
            result = gc.classify_job(root, dict(output="y"), "e" * 64, {})
            self.assertEqual((result["state"], result["ledger_ignored"]), ("pending", True))
            self.assertEqual(result["ledger_evaluator_sha256"], FOREIGN)
            (root / "y/progress.json").write_text(json.dumps(dict(prepared_sha256="1b" * 32, cut=0)))
            self.assertEqual(
                gc.classify_job(root, dict(output="y"), "e" * 64, {})["reason"], "ledger_evaluator_mismatch"
            )

    def test_identity_mismatch_mirrors_the_deployed_resume_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = "e" * 64
            base = job(0, output="z")
            # Identity carried by the ledger row disagrees with the job (gp.rs:744-755).
            write_ledger(
                root,
                "z",
                expected,
                seeds_sha256="00" * 32,
                rng_seed=base["rng_seed"],
                progress=dict(seeds_sha256=base["seeds_sha256"], rng_seed=base["rng_seed"]),
            )
            result = gc.classify_job(root, base, expected, {})
            self.assertEqual(
                (result["state"], result["reason"], result["mismatched_fields"]),
                ("dead", "identity_mismatch", ["ledger.seeds_sha256"]),
            )
            # Identity carried by progress.json disagrees (gp.rs:705-714).
            write_ledger(
                root,
                "z",
                expected,
                seeds_sha256=base["seeds_sha256"],
                rng_seed=base["rng_seed"],
                progress=dict(seeds_sha256=base["seeds_sha256"], rng_seed=base["rng_seed"] + 1),
            )
            result = gc.classify_job(root, base, expected, {})
            self.assertEqual(result["mismatched_fields"], ["progress.json.rng_seed"])
            # A first row that is not evaluation 0 cannot be resumed either.
            write_ledger(
                root,
                "z",
                expected,
                evaluation_index=3,
                seeds_sha256=base["seeds_sha256"],
                rng_seed=base["rng_seed"],
                progress=dict(seeds_sha256=base["seeds_sha256"], rng_seed=base["rng_seed"]),
            )
            self.assertEqual(
                gc.classify_job(root, base, expected, {})["mismatched_fields"], ["ledger.evaluation_index"]
            )
            # Absent fields are never inferred: a row without seeds_sha256/rng_seed stays pending.
            write_ledger(root, "z", expected)
            self.assertEqual(gc.classify_job(root, base, expected, {})["state"], "pending")
            self.assertEqual(gc.identity_mismatches(base, None, None), [])

    def test_no_resume_identity_requires_a_failed_attempt(self):
        # gp.rs:656-659: a journal without progress.json is permanent, but a first attempt may be
        # between its first journal append and its first progress receipt, so require attempts >= 1.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = "e" * 64
            write_ledger(root, "w", expected, progress=False)
            self.assertEqual(gc.classify_job(root, dict(output="w"), expected, {})["state"], "pending")
            failures = {"w": dict(gc.no_failures(), attempts=1, first_seen=5, last_seen=5)}
            result = gc.classify_job(root, dict(output="w"), expected, failures)
            self.assertEqual(
                (result["state"], result["reason"], result["mismatched_fields"]),
                ("dead", "identity_mismatch", [gc.NO_RESUME_IDENTITY]),
            )

    def test_attempts_exhausted_is_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "y").mkdir()
            failures = {
                "y": dict(attempts=3, first_seen=1, last_seen=3, max_attempt=3, identity_mismatch=False)
            }
            self.assertEqual(gc.classify_job(root, dict(output="y"), "e" * 64, failures)["state"], "pending")
            exhausted = gc.classify_job(root, dict(output="y"), "e" * 64, failures, max_attempts=3)
            self.assertEqual((exhausted["state"], exhausted["reason"]), ("dead", "attempts_exhausted"))
            self.assertEqual(
                gc.classify_job(root, dict(output="y"), "e" * 64, failures, max_attempts=4)["state"],
                "pending",
            )

    def test_safe_relative_mirrors_manifest_relative(self):
        self.assertTrue(gc.safe_relative("runs/a/search"))
        self.assertTrue(gc.safe_relative("runs//a/"))
        for value in ("", "/abs", "./runs/a", "runs/../a", "runs/./a", None, 3):
            self.assertFalse(gc.safe_relative(value), value)

    def test_audit_fails_closed_on_missing_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            with self.assertRaisesRegex(RuntimeError, "priority file absent"):
                gc.audit(f.root, f.manifest, f.root / "runs/nope.json", f.dead_letter, f.receipt)
            f.priority.write_bytes(b"{not json")
            with self.assertRaisesRegex(RuntimeError, "priority file unreadable"):
                f.audit()
            os.remove(f.root / "bin/gp-native-v5")
            with self.assertRaisesRegex(RuntimeError, "cpu_executable absent"):
                f.audit()
            self.assertFalse(f.dead_letter.exists())

    def test_audit_fails_closed_when_the_binary_differs_from_the_manifest_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            (f.root / "bin/gp-native-v5").write_bytes(b"rotated evaluator")
            with self.assertRaisesRegex(RuntimeError, "differs from the manifest binding"):
                f.audit()
            self.assertFalse(f.dead_letter.exists())
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder), bind=False)
            self.assertFalse(f.audit()["binding_checked"])


class ApplyTests(unittest.TestCase):
    def test_apply_drops_completed_and_dead_preserving_header_and_untouched_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            before_bytes = f.priority.read_bytes()
            audit = f.audit()
            receipt = f.apply()
            self.assertEqual(receipt["mode"], "apply")
            self.assertTrue(receipt["applied"])
            self.assertTrue(receipt["lock_taken"])
            self.assertEqual(
                (receipt["before"], receipt["removed_completed"], receipt["removed_dead"], receipt["after"]),
                (8, 1, 2, 5),
            )
            self.assertEqual(
                receipt["before"] - receipt["removed_completed"] - receipt["removed_dead"], receipt["after"]
            )
            self.assertEqual(
                (receipt["kept_pending"], receipt["kept_malformed"], receipt["kept_dead_unaudited"]),
                (2, 3, 0),
            )
            self.assertEqual(receipt["priority_sha256_before"], audit["priority_sha256"])
            self.assertEqual(receipt["priority_sha256_after"], sha(f.priority))
            self.assertEqual(receipt["lock_path"], str(f.root / "runs/gp-priority-jobs.lock"))
            self.assertTrue((f.root / "runs/gp-priority-jobs.lock").exists())
            after_bytes = f.priority.read_bytes()
            self.assertFalse(after_bytes.endswith(b"\n"))
            rewritten = json.loads(after_bytes)
            self.assertEqual(
                {k: v for k, v in rewritten.items() if k != "jobs"},
                {k: v for k, v in f.document.items() if k != "jobs"},
            )
            self.assertEqual(list(rewritten), list(f.document))
            expected = [f.jobs[i] for i in (2, 3, 4, 5, 7)]
            self.assertEqual([canonical(j) for j in rewritten["jobs"]], [canonical(j) for j in expected])
            self.assertEqual(len(rewritten["jobs"]), receipt["after"])
            # Untouched job objects are byte-identical in the pretty layout.
            self.assertEqual(gc.pretty(dict(f.document, jobs=expected)), after_bytes)
            self.assertNotEqual(before_bytes, after_bytes)
            self.assertFalse(list(f.root.glob("runs/.gp-priority-jobs.json.*")))

    def test_apply_refuses_without_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            with self.assertRaisesRegex(RuntimeError, "run --audit first"):
                f.apply()
            self.assertEqual(json.loads(f.priority.read_bytes())["jobs"], f.jobs)

    def test_apply_refuses_stale_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            f.audit()
            later = time.time() + 30
            os.utime(f.priority, (later, later))
            with self.assertRaisesRegex(RuntimeError, "stale audit.*older than the priority file"):
                f.apply()
            os.utime(f.dead_letter, (later + 1, later + 1))
            document = dict(f.document, jobs=f.jobs + [job(9)])
            f.priority.write_bytes(gc.pretty(document))
            os.utime(f.dead_letter, (later + 60, later + 60))
            with self.assertRaisesRegex(RuntimeError, "stale audit.*changed since --audit"):
                f.apply()
            self.assertEqual(len(json.loads(f.priority.read_bytes())["jobs"]), 9)
            f.audit()
            self.assertEqual(f.apply()["after"], 6)

    def test_apply_refuses_audit_bound_to_another_evaluator(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder), bind=False)
            f.audit()
            (f.root / "bin/gp-native-v5").write_bytes(b"rotated evaluator")
            with self.assertRaisesRegex(RuntimeError, "different manifest or evaluator"):
                f.apply()

    def test_apply_blocks_on_the_queue_lock_and_times_out(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            f.audit()
            lock = f.root / "runs/gp-priority-jobs.lock"
            held = threading.Event()
            release = threading.Event()

            def holder():
                with open(lock, "a+b") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    held.set()
                    release.wait()
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            thread = threading.Thread(target=holder)
            thread.start()
            self.assertTrue(held.wait(5))
            try:
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "queue lock busy"):
                    f.apply(lock_timeout=0.3)
                self.assertGreaterEqual(time.monotonic() - started, 0.3)
                self.assertEqual(len(json.loads(f.priority.read_bytes())["jobs"]), 8)
            finally:
                release.set()
                thread.join(5)
            self.assertEqual(f.apply(lock_timeout=1.0)["after"], 5)
            # The lock is released afterwards: a fresh LOCK_EX|LOCK_NB succeeds.
            with open(lock, "a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def test_apply_drops_searches_completed_after_the_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            f.audit()
            directory = f.root / f.jobs[3]["output"]
            directory.mkdir(parents=True)
            (directory / "search.json").write_text(
                json.dumps(dict(schema="parallel.search.v1", completed_evaluations=1))
            )
            receipt = f.apply()
            self.assertEqual(
                (receipt["removed_completed"], receipt["removed_dead"], receipt["after"]), (2, 2, 4)
            )

    def test_apply_keeps_jobs_that_died_only_after_the_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            f.audit()
            write_ledger(f.root, f.jobs[3]["output"], FOREIGN)
            receipt = f.apply()
            self.assertEqual(
                (receipt["removed_dead"], receipt["kept_dead_unaudited"], receipt["after"]), (2, 1, 5)
            )

    def test_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            f.audit()
            first = f.apply()
            f.audit()
            second = f.apply()
            self.assertEqual(
                (second["before"], second["removed_completed"], second["removed_dead"], second["after"]),
                (first["after"], 0, 0, first["after"]),
            )


class CliTests(unittest.TestCase):
    def test_main_audit_then_apply_with_default_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(Path(folder))
            base = ["--root", str(f.root), "--manifest", str(f.manifest)]
            with contextlib.redirect_stdout(io.StringIO()):
                audit = gc.main(base + ["--audit"])
                self.assertTrue((f.root / "runs/gp-queue-gc-audit.json").exists())
                self.assertEqual(read(f.root / "runs/gp-queue-gc-audit.json")["before"], audit["before"])
                applied = gc.main(base + ["--apply", "--lock-timeout", "2"])
            self.assertEqual(read(f.root / "runs/gp-queue-gc-apply.json")["after"], applied["after"])
            self.assertEqual(applied["after"], 5)

    def test_pretty_layout_matches_the_service_writer(self):
        original = RELEASE / "original/gp-priority-jobs.json"
        if not original.is_file():
            self.skipTest("release mirror not present")
        raw = original.read_bytes()
        self.assertEqual(gc.pretty(json.loads(raw)), raw)


class RealLedgerTests(unittest.TestCase):
    """Real seed-dispatch ledgers (evaluator b865...) against a different bound evaluator."""

    def setUp(self):
        if not (SEED_DISPATCH_ROOT / "runs/gp-priority-jobs.json").is_file():
            self.skipTest("seed-dispatch private-root mirror not present")

    def stage(self, root):
        (root / "runs").mkdir()
        (root / "bin").mkdir()
        (root / "bin/gp-native-v5").write_bytes(b"frozen evaluator bytes v5")
        evaluator = sha(root / "bin/gp-native-v5")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(dict(cpu_executable="bin/gp-native-v5", bindings={"bin/gp-native-v5": evaluator}))
        )
        shutil.copy(SEED_DISPATCH_ROOT / "runs/gp-priority-jobs.json", root / "runs/gp-priority-jobs.json")
        document = json.loads((root / "runs/gp-priority-jobs.json").read_bytes())
        for entry in document["jobs"]:
            shutil.copytree(SEED_DISPATCH_ROOT / entry["output"], root / entry["output"])
        return manifest, document

    def test_real_rows_carry_matching_job_identity_and_a_foreign_evaluator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, document = self.stage(root)
            self.assertEqual(len(document["jobs"]), 4)
            for entry in document["jobs"]:
                directory = root / entry["output"]
                ledger, row = gc.first_ledger_row(directory)
                progress = gc.read_progress(directory)
                self.assertEqual(ledger, "observations.json")
                self.assertEqual(row["evaluator_sha256"], SEED_DISPATCH_EVALUATOR)
                self.assertEqual(gc.identity_mismatches(entry, row, progress), [])
                self.assertIn("continuation", entry)
            # With search.json present every job is completed regardless of the ledger evaluator.
            receipt = gc.audit(
                root,
                manifest,
                root / "runs/gp-priority-jobs.json",
                root / "runs/gp-dead-letter.json",
                root / "runs/audit.json",
            )
            self.assertEqual(
                (receipt["before"], receipt["removed_completed"], receipt["removed_dead"]), (4, 4, 0)
            )
            # Without search.json the foreign-evaluator ledgers are dead-lettered; one job whose
            # ledger is removed entirely stays pending and survives --apply byte-for-byte.
            for entry in document["jobs"]:
                os.remove(root / entry["output"] / "search.json")
            survivor = document["jobs"][-1]
            for name in ("observations.json", "progress.json"):
                os.remove(root / survivor["output"] / name)
            receipt = gc.audit(
                root,
                manifest,
                root / "runs/gp-priority-jobs.json",
                root / "runs/gp-dead-letter.json",
                root / "runs/audit.json",
            )
            self.assertEqual(
                (
                    receipt["before"],
                    receipt["removed_completed"],
                    receipt["removed_dead"],
                    receipt["kept_pending"],
                    receipt["after"],
                ),
                (4, 0, 3, 1, 1),
            )
            dead = read(root / "runs/gp-dead-letter.json")
            self.assertEqual({e["reason"] for e in dead["entries"]}, {"ledger_evaluator_mismatch"})
            self.assertEqual(
                {e["ledger_evaluator_sha256"] for e in dead["entries"]}, {SEED_DISPATCH_EVALUATOR}
            )
            self.assertEqual([e["mismatched_fields"] for e in dead["entries"]], [[], [], []])
            applied = gc.apply(
                root,
                manifest,
                root / "runs/gp-priority-jobs.json",
                root / "runs/gp-dead-letter.json",
                root / "runs/apply.json",
            )
            self.assertEqual((applied["removed_dead"], applied["after"]), (3, 1))
            rewritten = json.loads((root / "runs/gp-priority-jobs.json").read_bytes())
            self.assertEqual(gc.header_of(rewritten), gc.header_of(document))
            self.assertEqual(rewritten["jobs"], [survivor])
            self.assertEqual(
                gc.pretty(dict(document, jobs=[survivor])), (root / "runs/gp-priority-jobs.json").read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
