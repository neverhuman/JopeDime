import tempfile
import unittest
from pathlib import Path

from eval_registry import EvalRegistry, evaluation_key


def full_admission(**overrides):
    row = {
        k: f"{k}-x"
        for k in (
            "program_sha256",
            "family",
            "partition",
            "prepared_sha256",
            "cut",
            "training_sha256",
            "validation_sha256",
            "evaluator_sha256",
            "evaluator_protocol",
            "preprocessing_sha256",
            "target_scale",
            "metric_units",
            "vocabulary_sha256",
            "feature_view_sha256",
            "evaluation_seed",
            "resource_protocol",
        )
    }
    row.update(overrides)
    return row


def full_execution(**overrides):
    row = {
        k: f"{k}-x"
        for k in (
            "evaluator_binary_sha256",
            "entry_point",
            "loaded_libs_sha256",
            "omp_threads",
            "rayon_threads",
            "fit_rng",
            "sealed_input_sha256",
            "resource_window",
            "admission_policy_sha256",
            "clean_completion_sha256",
        )
    }
    row.update(overrides)
    return row


class EvalRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = EvalRegistry(Path(self.tmp.name) / "registry.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def test_partial_identity_is_tombstone_not_a_miss(self):
        admission = {"program_sha256": "abc", "family": "iris"}
        looked = self.reg.lookup(admission, {})
        self.assertEqual(looked["action"], "tombstone")
        self.assertFalse(looked["authorize_fit"])
        again = self.reg.lookup(admission, full_execution())
        self.assertEqual(again["action"], "tombstone")
        self.assertFalse(again["authorize_fit"])

    def test_conflicting_declared_hash_does_not_reuse(self):
        a = full_admission()
        e = full_execution()
        key_a, _ = evaluation_key(a, e)
        b = dict(a)
        b["training_sha256"] = "other-cut"
        key_b, _ = evaluation_key(b, e)
        self.assertNotEqual(key_a, key_b)

    def test_zero_primary_is_a_legal_hit(self):
        a, e = full_admission(), full_execution()
        claimed = self.reg.claim(a, e, "owner-1")
        auth = self.reg.authorize_dispatch(claimed["key"], claimed["start_token"])
        self.assertTrue(auth["authorize_fit"])
        running = self.reg.consume_start(claimed["key"], claimed["start_token"])
        done = self.reg.complete(claimed["key"], running["attempt_token"], {"rmse": 0.0}, 0.0)
        self.assertEqual(done["action"], "completed")
        reuse = self.reg.lookup(a, e)
        self.assertEqual(reuse["action"], "reuse")
        self.assertEqual(reuse["primary_score"], 0.0)
        self.assertFalse(reuse["authorize_fit"])

    def test_same_token_cannot_authorize_a_second_fit(self):
        a, e = full_admission(), full_execution()
        claimed = self.reg.claim(a, e, "owner-1")
        token = claimed["start_token"]
        self.reg.authorize_dispatch(claimed["key"], token)
        first = self.reg.consume_start(claimed["key"], token)
        self.assertEqual(first["action"], "running")
        second = self.reg.consume_start(claimed["key"], token)
        self.assertEqual(second["action"], "rejected")
        self.assertFalse(second["authorize_fit"])
        self.assertGreaterEqual(self.reg.counters().get("registry_replay_blocked", 0), 1)

    def test_late_result_without_attempt_token_is_rejected(self):
        a, e = full_admission(), full_execution()
        claimed = self.reg.claim(a, e, "owner-1")
        self.reg.authorize_dispatch(claimed["key"], claimed["start_token"])
        self.reg.consume_start(claimed["key"], claimed["start_token"])
        bad = self.reg.complete(claimed["key"], "stolen-token", {"rmse": 1.0}, 1.0)
        self.assertEqual(bad["action"], "rejected")

    def test_merge_attributes_cross_shard_hits(self):
        totals = self.reg.merge_counts(
            [
                dict(
                    origin="gp",
                    repeats=10,
                    hits_on_exact=4,
                    hits_on_provisional=0,
                    cross_shard_hits=2,
                    modeled_seconds=30,
                    measured_core_seconds=8,
                ),
                dict(
                    origin="teacher",
                    repeats=5,
                    hits_on_exact=1,
                    hits_on_provisional=1,
                    cross_shard_hits=3,
                    modeled_seconds=15,
                    measured_core_seconds=0,
                ),
            ]
        )
        self.assertEqual(totals["repeats"], 15)
        self.assertEqual(totals["hits_on_exact"], 5)
        self.assertEqual(totals["cross_shard_hits"], 5)
        self.assertEqual(totals["per_origin"]["gp"], 6)
        self.assertEqual(totals["per_origin"]["teacher"], 5)
        self.assertEqual(totals["unattributed"], 4)
        self.assertEqual(totals["wall_reclaimed_modeled"], 45)
        self.assertEqual(totals["measured_core_seconds"], 8)
        self.assertNotEqual(totals["wall_reclaimed_modeled"], totals["measured_core_seconds"])


if __name__ == "__main__":
    unittest.main()
