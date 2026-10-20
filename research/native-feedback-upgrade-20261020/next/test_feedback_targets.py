import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from feedback_targets import Journal, canonical, digest, manifest


class TargetTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ["JOPE_FEEDBACK_TEST_ROOT"])
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.path = Path(self.temp.name) / "journal.sqlite"
        self.cid = "a" * 64
        self.context = {"context_id": self.cid, "parent_lineage_sha256": "b" * 64, "partition": "train",
            "identity": {"task": "regression", "dataset_hash": "123456789abcdef0", "cut_id": "c" * 64,
                "evaluator_sha256": "d" * 64, "vocabulary_sha256": "e" * 64, "protocol": "fixture-inner-v1",
                "comparison": {"numerical_policy": "fixture-f64-v1", "stage": "inner", "aggregation": "mean",
                    "label_role": "train", "sealed": False, "benchmark_valid": True, "recipe_encoding": "fixture-v1",
                    "class_order": [], "checkpoint_sha256": None}}, "metric": "rmse",
            "baseline": {"program_sha256": "f" * 64, "value": 1.}}
        self.doc = {"schema": "jope.measured-feedback-journal/v1", "experiment_sha256": "1"*64,
            "source_sha256": "2"*64, "runtime_sha256": "3"*64, "target_policy_sha256": "4"*64,
            "data_manifest_sha256": "5"*64, "max_logical_attempts": 2, "contexts": [self.context]}
        self.raw = canonical(self.doc)
        self.dsl = "(pipe X (ridge_gcv))"
        self.program = hashlib.sha256(self.dsl.encode()).hexdigest()
        self.journal = Journal(self.path, self.raw)

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def outcome(self, aid, *, status="completed", value=.8):
        return {"schema": "jope.measured-feedback-outcome/v1", "attempt_id": aid,
            "context_id": self.cid, "parent_lineage_sha256": self.context["parent_lineage_sha256"],
            "partition": "train", "identity": copy.deepcopy(self.context["identity"]), "program_sha256": self.program,
            "dsl": self.dsl, "status": status,
            "score": {"metric": "rmse", "direction": "lower", "units": "original_target_units", "value": value} if status == "completed" else None,
            "runtime_seconds": .5, "reported_fresh_fit": True}

    def finish(self, aid, outcome, **kw):
        raw = canonical(outcome)
        ref = {"s3_uri": "s3://fixture-bucket/blobs/"+hashlib.sha256(raw).hexdigest(),
               "version_id": "fixture-version-1", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return self.journal.finish(aid, ref, raw, **kw)

    def test_reproduce_original_failed_duplicate_charge_then_prevent_durable_replay(self):
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from feedback import FeedbackLedger
        legacy = FeedbackLedger({"train": {"partition": "train"}}, [{"context_id": "train", "parent_id": "p",
            "protocol_sha256": "x", "members": [{"program_sha256": "y", "status": "timeout", "q": None}]}])
        legacy.acquire("train", 0, wave=1, propensity=1.)
        legacy.acquire("train", 0, wave=1, propensity=1.)
        self.assertEqual(len(legacy.events), 2)  # preserved original reproducer
        aid = self.journal.reserve(self.cid, self.program)
        row = self.outcome(aid, status="timeout")
        self.assertEqual(self.finish(aid, row), "timeout")
        self.journal.close()
        self.journal = Journal(self.path, self.raw)
        self.assertEqual(self.journal.reserve(self.cid, self.program), aid)
        self.assertEqual(self.finish(aid, row), "timeout")
        snapshot = self.journal.snapshot()
        self.assertEqual(snapshot["charged_logical_attempts"], 1)
        self.assertEqual(snapshot["targets"], [])

    def test_changed_target_requires_strict_measured_improvement(self):
        aid = self.journal.reserve(self.cid, self.program)
        self.assertEqual(self.finish(aid, self.outcome(aid)), "measured")
        target = self.journal.snapshot()["targets"][0]
        self.assertTrue(target["ce_candidate"])
        self.assertEqual(target["prospective_hwm_transition"], "improved")
        self.assertFalse(target["training_admitted"])
        self.assertFalse(target["hwm_published"])

    def test_tie_and_worse_keep_ranking_evidence_without_ce(self):
        aid = self.journal.reserve(self.cid, self.program)
        self.finish(aid, self.outcome(aid, value=1.))
        target = self.journal.snapshot()["targets"][0]
        self.assertEqual(target["prospective_hwm_transition"], "tie")
        self.assertFalse(target["ce_candidate"])
        self.assertTrue(target["ranking_candidate"])

    def test_worse_result_is_retained_without_hwm_lift(self):
        aid = self.journal.reserve(self.cid, self.program)
        self.finish(aid, self.outcome(aid, value=1.1))
        target = self.journal.snapshot()["targets"][0]
        self.assertEqual(target["prospective_hwm_transition"], "not-improved")
        self.assertFalse(target["ce_candidate"])

    def test_direction_comes_from_native_metric_table(self):
        self.journal.close()
        self.path = Path(self.temp.name)/"higher.sqlite"
        self.doc["contexts"][0]["metric"] = "r2"
        self.journal = Journal(self.path, canonical(self.doc))
        aid = self.journal.reserve(self.cid, self.program)
        row = self.outcome(aid, value=1.2)
        row["score"].update(metric="r2", direction="higher", units="dimensionless")
        self.finish(aid, row)
        self.assertTrue(self.journal.snapshot()["targets"][0]["ce_candidate"])

    def test_noncomparable_evaluator_retains_charge_and_refuses_target(self):
        aid = self.journal.reserve(self.cid, self.program)
        row = self.outcome(aid)
        row["identity"]["evaluator_sha256"] = "9"*64
        self.assertEqual(self.finish(aid, row), "rejected-metadata")
        self.assertEqual(self.journal.snapshot()["targets"], [])
        self.assertEqual(self.journal.snapshot()["charged_logical_attempts"], 1)

    def test_wrong_program_role_and_direction_refused(self):
        for field, value in [("partition", "validation"), ("dsl", "(pipe X (ridge))")]:
            with self.subTest(field=field):
                row = self.outcome("0"*64)
                row[field] = value
                # Pure compiler has the same checks used by durable publication.
                from feedback_targets import compile_target
                with self.assertRaises(ValueError):
                    compile_target(self.context, "0"*64, self.program, canonical(row))
        aid = self.journal.reserve(self.cid, self.program)
        row = self.outcome(aid)
        row["score"]["direction"] = "higher"
        self.assertEqual(self.finish(aid, row), "rejected-metadata")

    def test_crash_before_result_commit_preserves_reserved_liability(self):
        aid = self.journal.reserve(self.cid, self.program)
        def fail():
            raise RuntimeError("synthetic interrupt before commit")
        with self.assertRaises(RuntimeError):
            self.finish(aid, self.outcome(aid), before_commit=fail)
        self.journal.close()
        self.journal = Journal(self.path, self.raw)
        snap = self.journal.snapshot()
        self.assertEqual(snap["unresolved_attempt_ids"], [aid])
        self.assertEqual(snap["charged_logical_attempts"], 1)
        self.assertEqual(snap["targets"], [])
        self.finish(aid, self.outcome(aid))
        self.assertEqual(len(self.journal.snapshot()["targets"]), 1)

    def test_conflicting_second_result_cannot_overwrite_first(self):
        aid = self.journal.reserve(self.cid, self.program)
        self.finish(aid, self.outcome(aid))
        first = self.journal.snapshot()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.finish(aid, self.outcome(aid, value=.1))
        self.assertEqual(first, self.journal.snapshot())

    def test_failed_outcome_cannot_supply_numerical_reward(self):
        aid = self.journal.reserve(self.cid, self.program)
        row = self.outcome(aid)
        row["status"] = "failed"
        self.assertEqual(self.finish(aid, row), "rejected-metadata")
        self.assertEqual(self.journal.snapshot()["targets"], [])

    def test_result_version_and_hash_are_mandatory(self):
        aid = self.journal.reserve(self.cid, self.program)
        raw = canonical(self.outcome(aid))
        for version, h in [("null", hashlib.sha256(raw).hexdigest()), ("v1", "0"*64)]:
            with self.assertRaises(ValueError):
                self.journal.finish(aid, {"s3_uri": "s3://fixture/b", "version_id": version, "sha256": h, "bytes": len(raw)}, raw)
        self.assertEqual(self.journal.snapshot()["unresolved_attempt_ids"], [aid])

    def test_manifest_drift_unknown_fields_duplicate_json_and_validation_refused(self):
        changed = copy.deepcopy(self.doc)
        changed["max_logical_attempts"] += 1
        with self.assertRaisesRegex(ValueError, "manifest drift"):
            Journal(self.path, canonical(changed))
        for modification in [lambda d: d.update(extra=True), lambda d: d["contexts"][0].update(partition="validation")]:
            d = copy.deepcopy(self.doc)
            modification(d)
            with self.assertRaises(ValueError):
                manifest(canonical(d))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            manifest(b'{"schema":1,"schema":2}')

    def test_budget_never_refills_after_failure(self):
        self.doc["max_logical_attempts"] = 1
        self.journal.close()
        self.path = Path(self.temp.name)/"budget.sqlite"
        self.journal = Journal(self.path, canonical(self.doc))
        aid = self.journal.reserve(self.cid, self.program)
        self.finish(aid, self.outcome(aid, status="failed"))
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.journal.reserve(self.cid, "7"*64)

    def test_snapshot_rederives_target_and_detects_corruption(self):
        aid = self.journal.reserve(self.cid, self.program)
        self.finish(aid, self.outcome(aid))
        self.journal.db.execute("UPDATE attempts SET target=? WHERE id=?", (canonical({"fabricated": True}), aid))
        with self.assertRaisesRegex(ValueError, "target compilation drift"):
            self.journal.snapshot()

    def test_manifest_loss_cannot_rebind_existing_attempts(self):
        self.journal.reserve(self.cid, self.program)
        self.journal.db.execute("DELETE FROM metadata WHERE key='manifest'")
        self.journal.close()
        with self.assertRaisesRegex(ValueError, "liabilities have no bound manifest"):
            Journal(self.path, self.raw)

    def test_nonfinite_reward_cannot_finish_reserved_liability(self):
        aid = self.journal.reserve(self.cid, self.program)
        row = self.outcome(aid)
        row["score"]["value"] = float("inf")
        with self.assertRaises(ValueError):
            self.finish(aid, row)
        self.assertEqual(self.journal.snapshot()["unresolved_attempt_ids"], [aid])

    def test_concurrent_reservations_cannot_overrun_budget(self):
        from concurrent.futures import ThreadPoolExecutor
        doc = copy.deepcopy(self.doc)
        doc["max_logical_attempts"] = 1
        path = Path(self.temp.name)/"concurrent.sqlite"
        def reserve(program):
            journal = Journal(path, canonical(doc))
            try:
                journal.reserve(self.cid, program)
                return "reserved"
            except ValueError:
                return "exhausted"
            finally:
                journal.close()
        # Initialize WAL once, then exercise independent connections.
        journal = Journal(path, canonical(doc))
        journal.close()
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(reserve, ["6"*64, "7"*64])), ["exhausted", "reserved"])


if __name__ == "__main__":
    unittest.main()
