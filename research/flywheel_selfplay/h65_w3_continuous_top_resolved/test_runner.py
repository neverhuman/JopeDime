from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


RUNNER_PATH = Path(__file__).with_name("runner.py")
SPEC = importlib.util.spec_from_file_location("h65_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
h65 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h65)


def valid_fold_manifest() -> dict:
    fold_groups = []
    for fold_index in range(h65.EXPECTED_FOLDS):
        contexts = [f"context-{i}" for i in range(h65.EXPECTED_CONTEXTS) if i % h65.EXPECTED_FOLDS == fold_index]
        families = [f"family-{i}" for i in range(h65.EXPECTED_FAMILIES) if i % h65.EXPECTED_FOLDS == fold_index]
        fold_groups.append(
            {
                "fold_index": fold_index,
                "held_contexts": contexts,
                "held_families": families,
                "fit_context_count": h65.EXPECTED_CONTEXTS - len(contexts),
                "held_context_count": len(contexts),
                "held_family_count": len(families),
            }
        )
    return {
        "schema": h65.W3_SCHEMA,
        "training_role": "DL-Train",
        "context_count": h65.EXPECTED_CONTEXTS,
        "family_count": h65.EXPECTED_FAMILIES,
        "fold_seed": h65.EXPECTED_FOLD_SEED,
        "fold_count": h65.EXPECTED_FOLDS,
        "dl_val_contexts_loaded": 0,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "source_manifest_file_sha256": h65.PACKET_FILE_SHA256,
        "source_manifest_sha256": h65.PACKET_PAYLOAD_SHA256,
        "folds": fold_groups,
    }


class RunnerTests(unittest.TestCase):
    def test_fold_manifest_requires_family_and_context_separation(self) -> None:
        folds = [
                {
                    "fold_index": i,
                    "held_contexts": [f"context-{i}"],
                    "held_families": [f"family-{i}"],
                    "held_context_count": 1,
                    "held_family_count": 1,
                    "fit_context_count": h65.EXPECTED_FOLDS - 1,
                }
            for i in range(h65.EXPECTED_FOLDS)
        ]
        projected = h65.project_fold_groups(folds, h65.EXPECTED_FOLDS, h65.EXPECTED_FOLDS)
        self.assertEqual(len(projected), h65.EXPECTED_FOLDS)
        folds[1]["held_families"] = folds[0]["held_families"]
        with self.assertRaisesRegex(ValueError, "family is split"):
            h65.project_fold_groups(folds, h65.EXPECTED_FOLDS, h65.EXPECTED_FOLDS)
        folds[1]["held_families"] = ["family-1"]
        folds[1]["held_contexts"] = folds[0]["held_contexts"]
        with self.assertRaisesRegex(ValueError, "context is held"):
            h65.project_fold_groups(folds, h65.EXPECTED_FOLDS, h65.EXPECTED_FOLDS)

    def test_fold_manifest_rejects_wrong_source_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "dl_train_contexts"):
            h65.build_fold_manifest({"schema": h65.W1_SCHEMA, "dl_train_contexts": 0}, "a" * 64)

    def test_winner_config_must_be_a_json_object_and_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "winner.json"
            path.write_text('{"schema":"future-w1-config/v1","arm":"winner"}\n', encoding="utf-8")
            config, digest = h65.load_winner_config(path)
            self.assertEqual(config["schema"], "future-w1-config/v1")
            self.assertEqual(digest, h65.sha256_bytes(path.read_bytes()))
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                h65.load_winner_config(path)

    def test_plan_is_preparation_only_and_cpu_bounded(self) -> None:
        plan = h65.make_plan(valid_fold_manifest(), {"schema": "future-w1/v1"}, "b" * 64)
        self.assertTrue(plan["preparation_only"])
        self.assertEqual(plan["device"], "cpu")
        self.assertFalse(plan["outcome_values_read"])
        self.assertFalse(plan["gpu_or_native_fit_requested"])

    def test_plan_rejects_non_training_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "training-only"):
            h65.make_plan({"schema": h65.W3_SCHEMA, "training_role": "DL-Val"}, {}, "c" * 64)


if __name__ == "__main__":
    unittest.main()
