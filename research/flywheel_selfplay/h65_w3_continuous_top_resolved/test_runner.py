from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


RUNNER_PATH = Path(__file__).with_name("runner.py")
SPEC = importlib.util.spec_from_file_location("h65_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
h65 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h65)


def valid_fold_manifest() -> dict:
    fold_groups = []
    for fold_index in range(h65.EXPECTED_FOLDS):
        contexts = [
            f"context-{i}"
            for i in range(h65.EXPECTED_CONTEXTS)
            if i % h65.EXPECTED_FOLDS == fold_index
        ]
        families = [
            f"family-{i}"
            for i in range(h65.EXPECTED_FAMILIES)
            if i % h65.EXPECTED_FOLDS == fold_index
        ]
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


def valid_comparator_binding() -> dict:
    return {
        "schema": h65.COMPARATOR_SCHEMA,
        "w1_disposition": {"status": "NO_ADVANCE", "accepted_winner": None},
        "prior_binding": None,
        "strongest_comparator": {
            "id": h65.COMPARATOR_ID,
            "label": "context GBM (ungated)",
            "source_metrics": {
                "path": h65.COMPARATOR_METRICS_PATH,
                "sha256": h65.COMPARATOR_METRICS_SHA256,
            },
        },
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
        projected = h65.project_fold_groups(
            folds, h65.EXPECTED_FOLDS, h65.EXPECTED_FOLDS
        )
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
            h65.build_fold_manifest(
                {"schema": h65.W1_SCHEMA, "dl_train_contexts": 0}, "a" * 64
            )

    def test_comparator_binding_is_hash_bound_to_h108_metrics(self) -> None:
        path = RUNNER_PATH.with_name("comparator_binding.json")
        binding, digest = h65.load_comparator_binding(path)
        self.assertEqual(binding["strongest_comparator"]["id"], "GBM_PROG412_CTX")
        self.assertEqual(digest, h65.sha256_bytes(path.read_bytes()))
        self.assertEqual(
            h65.load_h59_disposition(path.parent), h65.H59_DISPOSITION_SHA256
        )
        self.assertEqual(
            h65.load_preregistration(path.parent, digest), h65.H113_PREREG_SHA256
        )

    def test_no_advance_cannot_be_promoted_to_winner_or_prior(self) -> None:
        binding = valid_comparator_binding()
        binding["prior_binding"] = {"id": "invented-prior"}
        with self.assertRaisesRegex(ValueError, "cannot be converted"):
            h65.make_plan(valid_fold_manifest(), binding, "b" * 64)

    def test_plan_binds_comparator_without_inventing_w1_prior(self) -> None:
        binding = valid_comparator_binding()
        plan = h65.make_plan(valid_fold_manifest(), binding, "b" * 64)
        self.assertTrue(plan["preparation_only"])
        self.assertEqual(plan["device"], "cpu")
        self.assertEqual(plan["w1_disposition"], "NO_ADVANCE")
        self.assertIsNone(plan["accepted_w1_winner"])
        self.assertIsNone(plan["w1_prior_binding"])
        self.assertEqual(plan["strongest_comparator_id"], "GBM_PROG412_CTX")
        self.assertFalse(plan["outcome_values_read"])
        self.assertFalse(plan["gpu_or_native_fit_requested"])
        self.assertFalse(plan["launch_authorized"])

    def test_admission_manifest_stays_blocked_until_external_gates_close(self) -> None:
        manifest = h65.make_admission_manifest(
            valid_fold_manifest(),
            valid_comparator_binding(),
            h65.COMPARATOR_BINDING_SHA256,
            h65.H59_DISPOSITION_SHA256,
            h65.H113_PREREG_SHA256,
        )
        h65.validate_admission_manifest(manifest)
        self.assertEqual(
            manifest["h59_disposition"], "COMPLETED_NO_ADMISSION_INCORPORATED"
        )
        self.assertEqual(
            [item["status"] for item in manifest["blockers"]],
            ["OPEN", "OPEN", "OPEN", "OPEN", "OPEN"],
        )
        self.assertEqual(
            manifest["strongest_comparator"]["status"], "BOUND_FOR_COMPARISON_ONLY"
        )
        manifest["launch_authorized"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            h65.validate_admission_manifest(manifest)

    def test_plan_is_preparation_only_and_cpu_bounded(self) -> None:
        plan = h65.make_plan(
            valid_fold_manifest(), valid_comparator_binding(), "b" * 64
        )
        self.assertTrue(plan["preparation_only"])
        self.assertEqual(plan["device"], "cpu")
        self.assertFalse(plan["outcome_values_read"])
        self.assertFalse(plan["gpu_or_native_fit_requested"])

    def test_plan_rejects_non_training_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "training-only"):
            h65.make_plan(
                {"schema": h65.W3_SCHEMA, "training_role": "DL-Val"},
                valid_comparator_binding(),
                "c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
