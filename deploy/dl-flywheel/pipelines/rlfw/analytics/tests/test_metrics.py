import math
import json
import copy
import unittest

from rlfw.analytics.metrics import (
    feedback_latencies,
    macro,
    north_star,
    paired_noise,
    quantiles,
)
from rlfw.analytics.operational import (
    FEEDBACK_STAGES,
    actor_rows,
    native_actor_audit_rows,
    draw_statistics,
    feedback_ages,
    parent_cluster_rows,
    throughput_rows,
    unmeasured_rows,
)


def context(name="a", parent="p", **extra):
    return {
        "context_id": name,
        "parent": parent,
        "hwm_key": "key:" + name,
        "frozen_q_grid": 1000000,
        "resolved": True,
        **extra,
    }


def event(name="a", **extra):
    return {
        "at": 1,
        "context_id": name,
        "hwm_key": "key:" + name,
        "program_sha256": "new",
        "q_grid": 2000000,
        "verified": True,
        "first_seen": True,
        "ledger_event": "improvement",
        **extra,
    }


def folds(errors):
    return [
        {"index": j, "targets": [0.0, 0.0, 0.0, 0.0], "predictions": [error] * 4}
        for j, error in enumerate(errors)
    ]


class NorthStars(unittest.TestCase):
    def test_parent_macro_has_fixed_denominator(self):
        self.assertEqual(
            macro([("large", 1.0)] * 10 + [("small", 3.0)], {"large", "small"}), 2.0
        )
        self.assertIsNone(macro([("large", 1.0)], {"large", "small"}))

    def test_distinct_programs_not_slots_and_first_labels_not_improvements(self):
        events = [
            event(),
            event(),
            event(program_sha256="first", ledger_event="first_label"),
        ]
        r = north_star(
            events,
            [context()],
            start=0,
            end=10,
            cpu_seconds={"p": 3600},
            cpu_source="evaluator_cgroup",
        )
        self.assertEqual(r["U_useful"], 0.2)
        self.assertEqual(r["G_per_core_hour"], 1.0)
        self.assertEqual(r["delta_H"], 1.0)

    def test_wrong_identity_unknowns_and_end_boundary_do_not_score(self):
        es = [
            event(hwm_key="other"),
            event(verified=False, q_grid=None),
            event(at=10),
            event(first_seen=False, ledger_event="tie"),
        ]
        r = north_star(es, [context()], start=0, end=10)
        self.assertEqual(r["useful_verified_programs"], 0)
        self.assertIsNone(r["G_per_core_hour"])

    def test_missing_novelty_and_unresolved_context_stay_unmeasured(self):
        r = north_star([event(first_seen=None)], [context()], start=0, end=10)
        self.assertIsNone(r["U_useful"])
        r = north_star([event()], [context(resolved=False)], start=0, end=10)
        self.assertIsNone(r["delta_H"])
        self.assertEqual(r["delta_H_observed_lower_bound"], 1.0)

    def test_process_cpu_cannot_qualify_core_hour(self):
        self.assertIsNone(
            north_star(
                [event()],
                [context()],
                start=0,
                end=10,
                cpu_seconds={"p": 3600},
                cpu_source="process",
            )["G_per_core_hour"]
        )
        with self.assertRaises(ValueError):
            north_star(
                [event()],
                [context()],
                start=0,
                end=10,
                cpu_seconds={"p": 0},
                cpu_source="evaluator_cgroup",
            )

    def test_bad_window_and_missing_frozen_comparator(self):
        with self.assertRaises(ValueError):
            north_star([], [context()], start=10, end=0)
        self.assertIsNone(
            north_star([event()], [context(frozen_q_grid=None)], start=0, end=10)[
                "delta_H"
            ]
        )


class FoldNoise(unittest.TestCase):
    def test_paired_constant_error_ratio(self):
        r = paired_noise(folds((2.0, 2.0)), folds((1.0, 1.0)), seed=42)
        self.assertAlmostEqual(r["d0"], math.log(2))
        self.assertAlmostEqual(r["SE_pair"], 0.0, places=14)
        self.assertEqual(r["draws"], 2000)
        self.assertEqual(r["tau_c"], 0.01)

    def test_fold_disagreement_is_se_floor(self):
        r = paired_noise(folds((2.0, 1.0)), folds((1.0, 2.0)))
        self.assertAlmostEqual(r["abs_d0_minus_d1"], 2 * math.log(2))
        self.assertAlmostEqual(r["SE_pair"], math.log(2))

    def test_bootstrap_is_paired_and_deterministic(self):
        a = folds((1.0, 1.0))
        a[0]["predictions"] = [0.1, 1.0, 2.0, 4.0]
        self.assertEqual(paired_noise(a, a, seed=3)["SE_pair"], 0.0)
        b = folds((0.5, 0.5))
        self.assertEqual(paired_noise(a, b, seed=3), paired_noise(a, b, seed=3))
        self.assertGreater(paired_noise(a, b)["bootstrap_se"], 0.0)

    def test_unpaired_targets_nonfinite_and_wrong_draws_refused(self):
        a, b = folds((1.0, 1.0)), folds((0.5, 0.5))
        b[0]["targets"][0] = 1.0
        with self.assertRaises(ValueError):
            paired_noise(a, b)
        with self.assertRaises(ValueError):
            paired_noise(a, folds((math.nan, 1.0)))
        with self.assertRaises(ValueError):
            paired_noise(a, a, draws=20)


class Feedback(unittest.TestCase):
    def test_first_exposure_not_admission_and_right_censoring(self):
        events = [
            {"join_key": "a", "stage": s, "at": t}
            for s, t in (
                ("PROPOSAL_COMMITTED", 1),
                ("ADMITTED", 4),
                ("FIRST_EXPOSURE", 7),
                ("FIRST_EXPOSURE", 9),
            )
        ]
        events += [{"join_key": "b", "stage": "PROPOSAL_COMMITTED", "at": 2}]
        a, b = feedback_latencies(events, observed_until=10)
        self.assertEqual(a["T_feedback"], 6)
        self.assertEqual(a["stage_seconds"]["ADMITTED→FIRST_EXPOSURE"], 3)
        self.assertIsNone(a["stage_seconds"]["READY→ADMITTED"])
        self.assertTrue(b["censored"])
        self.assertIsNone(b["T_feedback"])

    def test_out_of_order_stage_receipts_refused(self):
        with self.assertRaises(ValueError):
            feedback_latencies(
                [
                    {"join_key": "a", "stage": "PROPOSAL_COMMITTED", "at": 2},
                    {"join_key": "a", "stage": "FIRST_EXPOSURE", "at": 1},
                ],
                observed_until=3,
            )

    def test_empty_distribution_is_not_zero(self):
        self.assertIsNone(quantiles([])["p50"])
        self.assertEqual(quantiles([2])["p99"], 2)


class Operational(unittest.TestCase):
    def test_native_actor_probe_is_not_relabelled_as_completed_training_update(self):
        audit = {
            "state": "PASS_INITIAL_NON_QUALIFYING",
            "common_completed_updates": 35,
            "diagnostic_source_sha256": "a" * 64,
            "actor_diagnostics": [
                {
                    "kind": "actor_diagnostics",
                    "arm": "archive_lpi",
                    "epoch": 0,
                    "updates": 0,
                    "fixture_contexts": 4,
                    "fixture_sha256": "f" * 64,
                    "parameter_sha256": "p" * 64,
                    "policy": "pi_M",
                    "gradient_norms_by_term": {
                        "L_archive_PI": {
                            "coefficient": 0.01,
                            "raw": {"actor": 3},
                            "applied": {"actor": 0.03},
                        }
                    },
                    "probability_movement": [
                        {
                            "context_id": "a",
                            "B": 8,
                            "fully_measured_actions": 3,
                            "D_B_full_trie": 7.5,
                            "D_B_pi_M": 2,
                        }
                    ],
                }
            ],
            "hwm_effective_support": {
                "train_contexts": 10,
                "comparable_hwm_contexts": 9,
                "null_hwm_contexts": 1,
                "fixed_clipped_utility_positive_contexts": 1,
                "effective_fraction_all_train": 0.1,
            },
        }
        kwargs = {"date": "2026-10-11", "commit": "c" * 40, "source": "audit:1"}
        rows = native_actor_audit_rows(audit, **kwargs)
        norms = [r for r in rows if r["metric"].startswith("actor_probe_grad_norm:")]
        self.assertEqual([r["value"] for r in norms], [3, 0.03])
        self.assertTrue(
            all(json.loads(r["details"])["updates"] == 0 and r["n"] == 4 for r in norms)
        )
        self.assertFalse(
            any(
                r["metric"]
                in (
                    "contexts_receiving_actor_targets",
                    "feedback_to_first_update_seconds",
                )
                for r in rows
            )
        )
        audit["actor_diagnostics"][0]["probability_movement"][0]["D_B_pi_M"] = 4
        with self.assertRaises(ValueError):
            native_actor_audit_rows(audit, **kwargs)

    def test_actor_objectives_targets_and_hwm_support_need_explicit_receipts(self):
        kwargs = {
            "date": "2026-10-11",
            "commit": "c" * 40,
            "source": "actual-update:1",
            "scope": "OP/R10",
        }
        observations = [
            {
                "objective": "RLOO",
                "update": 1,
                "actor_grad_norm": 2,
                "actor_target_context_ids": ["a", "a", "b"],
                "advantage_support": {"total": 4, "nonzero": 1},
                "archive_exposure_by_parent": {"p": 3},
                "exposure_counter_kind": "step",
            },
            {"objective": "L_PI-archive", "update": 1, "actor_grad_norm": 0.5},
        ]
        rows = actor_rows(observations, **kwargs)
        self.assertEqual(
            [r["value"] for r in rows if r["metric"].startswith("actor_grad_norm:")],
            [2, 0.5],
        )
        self.assertEqual(
            [
                r["value"]
                for r in rows
                if r["metric"] == "contexts_receiving_actor_targets"
            ],
            [2, None],
        )
        self.assertTrue(
            all(
                r["value"] is None
                for r in rows
                if r["metric"] == "nonzero_hwm_advantage_support_fraction"
            )
        )
        observations[0]["advantage_support"].update(
            hwm_identity_verified=True, complete=True
        )
        self.assertEqual(
            [
                r["value"]
                for r in actor_rows(observations, **kwargs)
                if r["metric"] == "nonzero_hwm_advantage_support_fraction"
            ],
            [0.25, None],
        )
        with self.assertRaises(ValueError):
            actor_rows(observations + [observations[0]], **kwargs)

    def test_absent_production_and_seven_feedback_clocks_stay_null(self):
        rows = unmeasured_rows(date="2026-10-11", commit="c" * 40, source="receipt:1")
        self.assertTrue(
            all(r["value"] is None and r["status"] == "unmeasured" for r in rows)
        )
        self.assertEqual(
            {
                r["metric"].split(":", 1)[1]
                for r in rows
                if r["metric"].startswith("feedback_timestamp:")
            },
            set(FEEDBACK_STAGES),
        )

    def test_active_cpu_and_allocated_capacity_have_distinct_rates(self):
        receipt = {
            "label": "NON-QUALIFYING",
            "state": "finished",
            "exit_code": 0,
            "sealed_outer_test_scored": False,
            "source_commit": "c" * 40,
            "programs_scored": 60,
            "cpu_seconds": 120,
            "wall_seconds": 100,
            "programs_per_core_hour": 1800,
        }
        export = {
            "schema": "jope.g-s3gp-throughput-denominator/v1",
            "round": 1,
            "label": "NON-QUALIFYING",
            "sealed_outer_test_scored": False,
            "source_commit": "c" * 40,
            "programs_scored": 60,
            "active_cpu": {
                "source": "RUSAGE",
                "cpu_seconds": 120,
                "active_core_hours": 120 / 3600,
                "programs_per_active_core_hour": 1800,
            },
            "allocated": {
                "reserved_cores": 32,
                "cpu_quota_percent": 3200,
                "wall_seconds": 100,
                "allocated_core_hours": 3200 / 3600,
                "programs_per_allocated_core_hour": 67.5,
            },
        }
        kwargs = {"date": "2026-10-11", "commit": "c" * 40, "source": "receipt:1"}
        actual = {r["metric"]: r for r in throughput_rows(export, receipt, **kwargs)}
        self.assertEqual(actual["programs_per_active_cpu_hour"]["value"], 1800)
        self.assertEqual(actual["programs_per_allocated_core_hour"]["value"], 67.5)
        self.assertTrue(
            all(
                r["status"] == "NON-QUALIFYING" or r["value"] is None
                for r in actual.values()
            )
        )
        wrong = copy.deepcopy(export)
        wrong["allocated"]["allocated_core_hours"] *= 2
        with self.assertRaises(ValueError):
            throughput_rows(wrong, receipt, **kwargs)
        receipt["state"] = "running"
        with self.assertRaises(ValueError):
            throughput_rows(export, receipt, **kwargs)

    def test_distinct_expectation_requires_complete_policy_not_topk(self):
        actual = draw_statistics(
            ["a", "a", "b", "a"], [0.5, 0.5], B=2, complete_policy=True
        )
        self.assertEqual(actual["duplicate_draw_rate"], 0.5)
        self.assertEqual(actual["expected_distinct_D_B"], 1.5)
        self.assertAlmostEqual(actual["entropy"], math.log(2))
        self.assertIsNone(draw_statistics(None, [0.5], B=2)["expected_distinct_D_B"])
        with self.assertRaises(ValueError):
            draw_statistics(None, [0.5], B=2, complete_policy=True)

    def test_adoption_and_evaluation_do_not_invent_first_update(self):
        base = {"join_key": "result/snapshot/term", "context_id": "a", "parent_id": "p"}
        events = [
            base | {"stage": s, "at_unix": t}
            for s, t in [("proposed", 1), ("adopted", 4), ("first_evaluated", 9)]
        ]
        actual = feedback_ages(events, observed_until=10)[0]
        self.assertIsNone(actual["feedback_to_first_update_seconds"])
        self.assertTrue(actual["right_censored"])
        events += [base | {"stage": "first_trained", "at_unix": 7}]
        self.assertEqual(
            feedback_ages(events, observed_until=10)[0][
                "feedback_to_first_update_seconds"
            ],
            6,
        )
        events += [
            base | {"stage": "first_trained", "at_unix": 6, "parent_id": "other"}
        ]
        with self.assertRaises(ValueError):
            feedback_ages(events, observed_until=10)

    def test_feedback_clock_reversal_is_refused(self):
        base = {"join_key": "x", "context_id": "a", "parent_id": "p"}
        with self.assertRaises(ValueError):
            feedback_ages(
                [
                    base | {"stage": "proposed", "at_unix": 5},
                    base | {"stage": "first_trained", "at_unix": 4},
                ],
                observed_until=10,
            )

    def test_parent_means_do_not_count_correlated_cuts_or_repeated_epochs(self):
        pairs = [
            {
                "kind": "scratch_context_pair",
                "metric": "l6_context_control_minus_arm",
                "scope": "OP/run/rloo",
                "context_set_sha256": "r" * 64,
                "value": delta,
                "details": json.dumps(
                    {
                        "context_id": context,
                        "epoch": epoch,
                        "updates": epoch * 10,
                        "export_sha256": "e" * 64,
                        "control_arm": "r23_control",
                        "fixed_contexts": 4,
                        "complete_roster": True,
                    }
                ),
            }
            for epoch in (1, 2)
            for context, delta in zip("abcd", (0, 0, 0, 4))
        ]
        parents = [
            {"context_id": c, "parent_id": "large" if c != "d" else "small"}
            for c in "abcd"
        ]
        kwargs = {"date": "2026-10-11", "commit": "c" * 40, "source": "map:1"}
        summaries = [
            r
            for r in parent_cluster_rows(pairs, parents, **kwargs)
            if r["kind"] == "parent_cluster_summary"
        ]
        means = [r for r in summaries if r["metric"] == "parent_l6_paired_mean_delta"]
        self.assertEqual([r["value"] for r in means], [2, 2])
        self.assertEqual([r["n"] for r in means], [2, 2])
        self.assertEqual(
            [
                r["value"]
                for r in summaries
                if r["metric"] == "parent_l6_paired_mean_SE"
            ],
            [2, 2],
        )
        missing = parent_cluster_rows(pairs, parents[:-1], **kwargs)
        self.assertTrue(all(r["value"] is None for r in missing))
        duplicate = parent_cluster_rows(pairs + [pairs[0]], parents, **kwargs)
        self.assertEqual(
            [
                r["value"]
                for r in duplicate
                if r["metric"] == "parent_l6_paired_mean_delta"
                and r["kind"] == "parent_cluster_summary"
            ],
            [2, 2],
        )
        with self.assertRaises(ValueError):
            parent_cluster_rows(pairs + [pairs[0] | {"value": 99}], parents, **kwargs)
