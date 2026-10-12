"""Paired context variance, power and actual scope elapsed-time boundaries."""

import copy
import json
import unittest
import tempfile
import hashlib
from pathlib import Path

from rlfw.analytics.context_costs import (
    IDENTITY,
    SCHEMA,
    comparison_power,
    context_rows,
    contexts_needed,
    roster_hash,
    scope_rows,
    native_context_rows,
    indexed_context_rows,
)


def exported():
    roster = ["a", "b", "c"]
    return {
        "schema": SCHEMA,
        "non_qualifying": True,
        "benchmark_valid": False,
        "promotion_input": False,
        "lane": "CL",
        "run": "run1",
        "run_receipt_sha256": "f" * 64,
        "context_ids": roster,
        "context_set_sha256": roster_hash(roster),
        "comparison_identity": {k: "a" * 64 for k in IDENTITY},
        "observations": [
            {
                "arm": arm,
                "epoch": 1,
                "updates": 142,
                "contexts": [
                    {
                        "context_id": c,
                        "mean_cost": cost,
                        "items": 4,
                        "item_set_sha256": c * 64,
                    }
                    for c, cost in zip(roster, values)
                ],
            }
            for arm, values in [
                ("control", [1.0, 2.0, 3.0]),
                ("actor", [0.9, 1.8, 2.7]),
            ]
        ],
    }


def rows(export):
    return context_rows(
        export,
        {"inputs": {"checkpoint": {"parameter_sha256": "a" * 64}}},
        source="export.json:1",
        receipt_sha="e" * 64,
        consumer_commit="c" * 40,
        as_of_date="2026-10-10",
    )


class ContextCosts(unittest.TestCase):
    def test_indexed_pair_costs_bind_parameter_updates_and_saved_macro(self):
        ids = ["b", "a"]
        coefficients = {
            k: 1.0 if k == "ce" else 0.0
            for k in ("ce", "bt", "list", "q", "feas", "rt")
        }
        index = {
            "schema": "jope.r22-lpi-scratch-paired-index/v1",
            "run": "/run/x",
            "sets": {
                "a": {
                    "contexts": 2,
                    "sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "definition": "validation",
                }
            },
        }
        checkpoints = {
            "schema": "jope.r22-lpi-scratch-checkpoint-index/v1",
            "run": "/run/x",
            "checkpoints": [],
        }
        parent = {
            "schema": "jope.r22-lpi-scratch-train/v6",
            "non_qualifying": True,
            "benchmark_valid": False,
            "promotion_input": False,
            "inputs": {
                "l6_coefficients": coefficients,
                "checkpoint": {"parameter_sha256": "a" * 64},
            },
            "initial_eval": [],
        }
        events = []
        for arm, values in (("control", [1.0, 1.0]), ("actor", [0.9, 0.7])):
            checkpoint = {
                "tag": "epoch_001",
                "arm": arm,
                "epoch": 1,
                "updates": 142,
                "parameter_sha256": arm,
            }
            checkpoints["checkpoints"].append(checkpoint)
            parent["initial_eval"].append(
                checkpoint | {"a_l6": {"loss": sum(values) / 2}}
            )
            for cid, value in zip(ids, values):
                events.append(
                    checkpoint
                    | {
                        "checkpoint": "epoch_001",
                        "context_id": cid,
                        "set_list": "a",
                        "numerators": {
                            k: value if k == "ce" else 0 for k in coefficients
                        },
                        "denominators": {
                            k: 1 if k == "ce" else 0 for k in coefficients
                        },
                    }
                )

        def run():
            return indexed_context_rows(
                index,
                checkpoints,
                parent,
                events,
                {"a": ids},
                source="index:1",
                index_sha="b" * 64,
                parent_sha="c" * 64,
                consumer_commit="d" * 40,
                as_of_date="2026-10-11",
            )

        self.assertAlmostEqual(
            next(r["value"] for r in run() if r["metric"] == "l6_paired_mean_delta"),
            0.2,
        )
        ids.reverse()
        with self.assertRaisesRegex(ValueError, "roster hash/count"):
            run()
        ids.reverse()
        checkpoints["checkpoints"][0]["updates"] += 1
        with self.assertRaisesRegex(ValueError, "update mismatch"):
            run()
        checkpoints["checkpoints"][0]["updates"] -= 1
        parent["initial_eval"][0]["a_l6"]["loss"] = 3
        with self.assertRaisesRegex(ValueError, "reconcile"):
            run()

    def test_r23_named_control_pairs_the_same_fixed_roster(self):
        export = exported()
        export["observations"][0]["arm"] = "r23_control"
        result = rows(export)
        mean = next(r for r in result if r["metric"] == "l6_paired_mean_delta")
        self.assertAlmostEqual(mean["value"], 0.2)
        self.assertEqual(mean["n"], 3)
        self.assertEqual(json.loads(mean["details"])["control_arm"], "r23_control")

    def test_multiple_native_comparators_require_explicit_selection(self):
        export = exported()
        other = copy.deepcopy(export["observations"][0])
        other["arm"] = "r23_control"
        export["observations"].append(other)
        with self.assertRaisesRegex(ValueError, "ambiguous control"):
            rows(export)
        export["control_arm"] = "r23_control"
        self.assertTrue(any(r["value"] is not None for r in rows(export)))

    def test_sparse_supported_terms_reconcile_macro_not_raw_mean(self):
        ids = ["a", "b"]
        keys = ("ce", "bt", "list", "q", "feas", "rt")
        index = {
            "schema": "jope.r22-lpi-scratch-paired-index/v1",
            "label": "R22 scratch-only",
            "eval_path": "coefficients ce 1.0 bt 0.25 list 0.1 q 0.25 feas 0.1 rt 0.05",
            "sweep_run": "/run/x",
            "sets": {"a": {"contexts": 2, "sha256": roster_hash(ids)}},
            "checkpoints": [],
        }
        events = []
        for arm, q in [("control", 1.0), ("actor", 0.5)]:
            entry = {
                "checkpoint": "epoch1",
                "arm": arm,
                "epoch": 1,
                "updates": 142,
                "parameter_sha256": arm,
                "state_sha256": arm,
                "sets_evaluated": ["a"],
                "aggregate": {"a": {"l6_macro_loss": 1 + 0.25 * q}},
            }
            index["checkpoints"].append(entry)
            for context in ids:
                num = {
                    k: 1 if k == "ce" else q if k == "q" and context == "a" else 0
                    for k in keys
                }
                den = {
                    k: 1 if k == "ce" or k == "q" and context == "a" else 0
                    for k in keys
                }
                events.append(
                    entry
                    | {
                        "set_list": "a",
                        "context_id": context,
                        "l6_loss": 1 + 0.25 * q if context == "a" else 1,
                        "numerators": num,
                        "denominators": den,
                    }
                )
        result = native_context_rows(
            index,
            events,
            {"a": ids},
            source="index:1",
            index_sha="e" * 64,
            consumer_commit="c" * 40,
            as_of_date="2026-10-10",
        )
        self.assertAlmostEqual(
            next(r["value"] for r in result if r["metric"] == "l6_paired_mean_delta"),
            0.125,
        )
        self.assertAlmostEqual(
            next(
                r["value"] for r in result if r["metric"] == "l6_raw_paired_mean_delta"
            ),
            0.0625,
        )
        index["checkpoints"][0]["aggregate"]["a"]["l6_macro_loss"] = 999
        with self.assertRaises(ValueError):
            native_context_rows(
                index,
                events,
                {"a": ids},
                source="index:1",
                index_sha="e" * 64,
                consumer_commit="c" * 40,
                as_of_date="2026-10-10",
            )

    def test_native_indexed_cohorts_keep_validation_and_unused_train_separate(self):
        ids = ["a", "b", "c"]
        index = {
            "schema": "jope.r22-lpi-scratch-paired-index/v1",
            "label": "R22 scratch-only, non-qualifying",
            "eval_path": "coefficients ce 1.0 bt 0.25 list 0.1 q 0.25 feas 0.1 rt 0.05",
            "sweep_run": "/run/sweep3",
            "sets": {},
            "checkpoints": [],
        }
        for name in ("a", "b_prefix_1000"):
            index["sets"][name] = {"contexts": 3, "sha256": roster_hash(ids)}
        events = []
        for arm, costs in (("control", [1, 2, 3]), ("actor", [0.9, 1.8, 2.7])):
            entry = {
                "checkpoint": "epoch1",
                "arm": arm,
                "epoch": 1,
                "updates": 142,
                "parameter_sha256": arm,
                "state_sha256": arm,
                "sets_evaluated": ["a", "b_prefix_1000"],
                "aggregate": {
                    name: {"l6_macro_loss": sum(costs) / len(costs)}
                    for name in index["sets"]
                },
            }
            index["checkpoints"].append(entry)
            for name in index["sets"]:
                events.extend(
                    entry
                    | {
                        "set_list": name,
                        "context_id": c,
                        "l6_loss": cost,
                        "numerators": {
                            k: cost if k == "ce" else 0
                            for k in ("ce", "bt", "list", "q", "feas", "rt")
                        },
                        "denominators": {
                            k: 1 if k == "ce" else 0
                            for k in ("ce", "bt", "list", "q", "feas", "rt")
                        },
                    }
                    for c, cost in zip(ids, costs)
                )
        result = native_context_rows(
            index,
            events,
            {n: ids for n in index["sets"]},
            source="index:1",
            index_sha="e" * 64,
            consumer_commit="c" * 40,
            as_of_date="2026-10-10",
        )
        summaries = [
            r
            for r in result
            if r["metric"] == "l6_between_context_paired_delta_variance"
        ]
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(r["n"] == 3 for r in summaries))
        self.assertNotEqual(summaries[0]["scope"], summaries[1]["scope"])
        with self.assertRaises(ValueError):
            native_context_rows(
                index,
                events,
                {n: ["changed"] for n in index["sets"]},
                source="index:1",
                index_sha="e" * 64,
                consumer_commit="c" * 40,
                as_of_date="2026-10-10",
            )

    def test_actual_context_variance_not_variance_of_epoch_means(self):
        result = rows(exported())
        values = {
            r["metric"]: r["value"]
            for r in result
            if r["kind"] == "scratch_context_summary"
        }
        self.assertAlmostEqual(values["l6_paired_mean_delta"], 0.2)
        self.assertAlmostEqual(values["l6_between_context_paired_delta_variance"], 0.01)
        self.assertAlmostEqual(values["l6_control_between_context_cost_variance"], 1)
        self.assertAlmostEqual(values["l6_paired_mean_SE"], 0.1 / 3**0.5)
        self.assertEqual(
            len([r for r in result if r["kind"] == "scratch_context_pair"]), 3
        )
        self.assertTrue(
            all(r["n"] == 1 for r in result if r["kind"] == "scratch_context_pair")
        )

    def test_mismatched_contexts_items_or_updates_leave_full_panel_unmeasured(self):
        for field, value in (("items", 5), ("item_set_sha256", "different")):
            r = exported()
            r["observations"][-1]["contexts"][0][field] = value
            summary = [x for x in rows(r) if x["kind"] == "scratch_context_summary"]
            self.assertTrue(all(x["value"] is None for x in summary))
        r = exported()
        r["observations"][-1]["updates"] = 141
        self.assertTrue(all(x["value"] is None for x in rows(r)))
        r = exported()
        r["observations"][-1]["contexts"].pop()
        self.assertTrue(
            all(
                x["value"] is None
                for x in rows(r)
                if x["kind"] == "scratch_context_summary"
            )
        )

    def test_duplicate_context_or_roster_refused_and_nan_refused(self):
        r = exported()
        r["observations"][0]["contexts"].append(
            copy.deepcopy(r["observations"][0]["contexts"][0])
        )
        with self.assertRaises(ValueError):
            rows(r)
        r = exported()
        r["context_ids"].append("a")
        with self.assertRaises(ValueError):
            rows(r)
        r = exported()
        r["observations"][0]["contexts"][0]["mean_cost"] = float("nan")
        with self.assertRaises(ValueError):
            rows(r)

    def test_epochs_are_separate_cohorts_not_extra_independent_contexts(self):
        r = exported()
        more = copy.deepcopy(r["observations"])
        for block in more:
            block["epoch"] = 2
            block["updates"] = 284
        r["observations"].extend(more)
        summaries = [
            x
            for x in rows(r)
            if x["metric"] == "l6_between_context_paired_delta_variance"
        ]
        self.assertEqual([x["n"] for x in summaries], [3, 3])
        self.assertEqual([json.loads(x["details"])["epoch"] for x in summaries], [1, 2])

    def test_exact_normal_power_boundary_matches_existing_881_standin(self):
        sigma = 0.05296631946624709
        self.assertEqual(contexts_needed(sigma), 881)
        self.assertLess(comparison_power(880, sigma), 0.8)
        self.assertGreaterEqual(comparison_power(881, sigma), 0.8)

    def test_zero_observed_variance_does_not_certify_population_zero_noise(self):
        r = exported()
        for item in r["observations"][-1]["contexts"]:
            item["mean_cost"] = next(
                x["mean_cost"]
                for x in r["observations"][0]["contexts"]
                if x["context_id"] == item["context_id"]
            )
        result = rows(r)
        self.assertEqual(
            next(
                x["value"]
                for x in result
                if x["metric"] == "l6_between_context_paired_delta_variance"
            ),
            0,
        )
        self.assertIsNone(
            next(
                x["value"]
                for x in result
                if x["metric"] == "l6_contexts_needed_delta_0_005_power_0_80"
            )
        )

    def test_scope_cap_is_not_elapsed_and_running_cost_is_partial(self):
        document = {
            "lane": "CL",
            "run": "run1",
            "sha256": "e" * 64,
            "gpu_count": 1,
            "active_enter_unix": 1000,
            "inactive_enter_unix": None,
            "observed_remote_unix": 4600,
            "properties": {
                "ActiveState": "active",
                "CPUUsageNSec": "7200000000000",
                "RuntimeMaxUSec": "2h",
            },
        }
        result = scope_rows(
            document,
            consumer_commit="c" * 40,
            as_of_date="2026-10-10",
            source="scope.json:1",
        )
        self.assertEqual([x["value"] for x in result], [1, 1, 2])
        self.assertTrue(all("PARTIAL" in x["status"] for x in result))
        document["properties"]["ActiveState"] = "inactive"
        document["inactive_enter_unix"] = 2800
        self.assertEqual(
            scope_rows(
                document,
                consumer_commit="c" * 40,
                as_of_date="2026-10-10",
                source="scope.json:1",
            )[0]["value"],
            0.5,
        )
        document["active_enter_unix"] = None
        document["properties"]["CPUUsageNSec"] = "[not set]"
        self.assertTrue(
            all(
                x["value"] is None
                for x in scope_rows(
                    document,
                    consumer_commit="c" * 40,
                    as_of_date="2026-10-10",
                    source="scope.json:1",
                )
            )
        )

    def test_actual_parquet_ingest_binds_parent_and_context_export(self):
        from rlfw.analytics.build_scratch import build
        from rlfw.analytics.report import read_daily

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "receipt.json"
            parent.write_text(
                json.dumps(
                    {
                        "schema": "jope.r22-lpi-scratch-train/v3",
                        "non_qualifying": True,
                        "benchmark_valid": False,
                        "promotion_input": False,
                        "inputs": {"checkpoint": {"parameter_sha256": "a" * 64}},
                    }
                )
            )
            export = exported()
            export["run_receipt_sha256"] = hashlib.sha256(
                parent.read_bytes()
            ).hexdigest()
            context = root / "context-costs.json"
            context.write_text(json.dumps(export))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "jope.cx-met-scratch-input/v1",
                        "providers": [
                            {
                                "lane": "CL",
                                "patterns": [str(parent)],
                                "context_patterns": [str(context)],
                            }
                        ],
                    }
                )
            )
            receipt = build(manifest, root / "owned", "c" * 40)
            observed = read_daily(
                [
                    Path(r["path"])
                    for r in receipt["outputs"]
                    if r["path"].endswith(".parquet")
                ]
            )
            result = next(
                r
                for r in observed
                if r["metric"] == "l6_between_context_paired_delta_variance"
            )
            self.assertAlmostEqual(result["value"], 0.01)
            self.assertTrue(json.loads(result["details"])["parent_receipt_verified"])
            export["run_receipt_sha256"] = "0" * 64
            context.write_text(json.dumps(export))
            refused = build(manifest, root / "refused", "c" * 40)
            self.assertTrue(
                any(
                    "physical parent" in r.get("reason", "")
                    for r in refused["inventory"]
                )
            )
