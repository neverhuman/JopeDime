"""Synthetic arithmetic, evidence refusals and actual Parquet/HTML readback."""

import json
from pathlib import Path
import tempfile
import unittest

from rlfw.analytics.baseline import COMPARISON, baseline_rows, duration
from rlfw.analytics.build_baseline import build, read_json
from rlfw.analytics.extract import digest
from rlfw.analytics.report import read_daily

SHA = "a" * 64
START = 1791936000  # 2026-10-14 00:00 UTC
WINDOW = [START, START + 3600]


def clock(seconds, domain="host-a", uncertainty=None):
    return {"at": START + seconds, "domain": domain, "uncertainty_seconds": uncertainty}


def export(domain, data, complete=True):
    return {
        "schema": "jope.rlfw-baseline-export/v1",
        "domain": domain,
        "window": WINDOW,
        "complete": complete,
        "scope": "fixture/train",
        "producer_commit": None,
        "evidence_status": "synthetic",
        "cohort_sha256": SHA,
        "data": data,
    }


def rows(domain=None, data=None, *, complete=True, window=WINDOW):
    exports = {} if domain is None else {domain: export(domain, data, complete)}
    return baseline_rows(
        exports, window=window, date="2026-10-14", commit="f" * 40, sources={}
    )


def values(domain, data, **kwargs):
    return {r["metric"]: r["value"] for r in rows(domain, data, **kwargs)}


def allocation(identity="one", start=0, end=3600, cores=2, **extra):
    return {
        "span_id": identity,
        "allocation_id": "reservation",
        "interval": [START + start, START + end],
        "cores": cores,
        "source_kind": "reservation",
        "owner_receipt_sha256": SHA,
        **extra,
    }


def cpu(**extra):
    return {
        "counter_id": "cpu",
        "scope_id": "scope",
        "ancestors": [],
        "counter_kind": "process_rusage",
        "invocation_id": "invocation",
        "interval": WINDOW,
        "start_seconds": 100,
        "end_seconds": 1000,
        "receipt_sha256": SHA,
        **extra,
    }


def throughput(**extra):
    return {
        "allocations": [allocation(end=1800)],
        "allocation_complete": True,
        "cpu_counters": [cpu()],
        "cpu_complete": True,
        "counts_complete": True,
        "counts": {
            "interval": WINDOW,
            "programs": 100,
            "folds": 200,
            "physical_attempts": 4,
            "missing_outcomes": 1,
        },
        **extra,
    }


def outcome(identity="one", context="c1", program=SHA, at=0, **extra):
    return {
        "outcome_id": identity,
        "context_id": context,
        "parent_id": "parent",
        "role": "TRAIN",
        "canonical_program_sha256": program,
        "comparison_identity": dict.fromkeys(COMPARISON, SHA),
        "qualified": True,
        "qualification_receipt_sha256": SHA,
        "remote_version_readback_sha256": SHA,
        "first_qualified_proof_sha256": SHA,
        "first_qualified_at": START + at,
        "first_program_seen_at": START,
        "first_program_seen_proof_sha256": SHA,
        "catalog_membership_proof_sha256": SHA,
        "in_frozen_catalog": False,
        **extra,
    }


def arrivals(records):
    return {
        "records": records,
        "lifetime_novelty_complete": True,
        "frozen_catalog_sha256": SHA,
    }


def episode(**extra):
    return {
        "episode_id": "episode",
        "context_id": "c",
        "parent_id": "p",
        "outcome_id": "outcome",
        "target_packet_sha256": SHA,
        "support_sha256": SHA,
        "comparison_identity": dict.fromkeys(COMPARISON, SHA),
        "input_snapshot_sha256": SHA,
        "consumption_receipt_sha256": SHA,
        "parameter_change_receipt_sha256": SHA,
        "learner_invocation": "learner",
        "optimizer_update": 1,
        "exposure_receipt_sha256": SHA,
        "clocks": {
            k: clock(v)
            for k, v in [
                ("proposed", 0),
                ("compute_started", 5),
                ("locally_verified", 7),
                ("remotely_qualified", 9),
                ("adopted", 12),
                ("first_trained", 20),
                ("first_evaluated", 25),
                ("first_exposure", 18),
            ]
        },
        **extra,
    }


def feedback(item):
    return {"episodes": [item], "observed_until": clock(3600)}


def cost(identity="a", prefix="root/a/", usd=0.1, category="storage", **extra):
    return {
        "ledger_event_id": identity,
        "bucket": "fixture-bucket",
        "prefix": prefix,
        "interval": WINDOW,
        "window_kind": "report_window",
        "bytes_measure_kind": "added",
        "bytes": 100,
        "usd": usd,
        "currency": "USD",
        "cost_kind": "estimate",
        "pricing_identity": SHA,
        "cost_category": category,
        **extra,
    }


class ThroughputTests(unittest.TestCase):
    def test_allocated_active_and_wall_denominators_are_distinct(self):
        result = values("throughput", throughput())
        self.assertEqual(result["allocated_core_hours"], 1)
        self.assertEqual(result["active_cpu_core_hours"], 0.25)
        self.assertEqual(result["programs_per_allocated_core_hour"], 100)
        self.assertEqual(result["programs_per_active_cpu_hour"], 400)
        self.assertEqual(result["programs_per_wall_hour"], 100)
        self.assertEqual(result["missing_outcome_rate"], 0.25)

    def test_changing_spans_integrate_reserved_idle_capacity(self):
        result = values(
            "throughput",
            throughput(
                allocations=[
                    allocation(end=1800),
                    allocation("two", start=1800, cores=4),
                ]
            ),
        )
        self.assertEqual(result["allocated_core_hours"], 3)

    def test_clip_reservations_but_never_prorate_whole_run_counts(self):
        data = throughput(
            allocations=[
                allocation(start=-1800, end=1800),
                allocation("two", start=1800, end=5400, cores=4),
            ]
        )
        data["counts"]["interval"] = [START - 1800, START + 5400]
        result = values("throughput", data)
        self.assertEqual(result["allocated_core_hours"], 3)
        self.assertIsNone(result["programs_per_allocated_core_hour"])
        self.assertIsNone(result["programs"])

    def test_quota_is_not_reserved_allocation(self):
        result = values(
            "throughput",
            throughput(allocations=[allocation(source_kind="quota_capacity")]),
        )
        self.assertIsNone(result["allocated_core_hours"])
        self.assertEqual(result["quota_capacity_core_hours"], 2)

    def test_missing_reservation_is_null_not_zero(self):
        self.assertIsNone(values("throughput", {})["allocated_core_hours"])

    def test_zero_and_nonfinite_allocations_refuse(self):
        for n in [0, float("nan"), float("inf"), -1, True]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                rows("throughput", throughput(allocations=[allocation(cores=n)]))

    def test_overlapping_same_reservation_refuses(self):
        with self.assertRaisesRegex(ValueError, "overlapping_allocation"):
            rows(
                "throughput",
                throughput(allocations=[allocation(), allocation("two", start=10)]),
            )

    def test_distinct_reservations_can_overlap(self):
        spans = [allocation(), allocation("two", allocation_id="other")]
        self.assertEqual(
            values("throughput", throughput(allocations=spans))["allocated_core_hours"],
            4,
        )

    def test_identical_aliases_do_not_double_cpu_or_capacity(self):
        result = values(
            "throughput",
            throughput(
                allocations=[allocation(), allocation()], cpu_counters=[cpu(), cpu()]
            ),
        )
        self.assertEqual(result["allocated_core_hours"], 2)
        self.assertEqual(result["active_cpu_core_hours"], 0.25)

    def test_conflicting_counter_alias_refuses(self):
        with self.assertRaisesRegex(ValueError, "conflicting counter_id"):
            rows("throughput", throughput(cpu_counters=[cpu(), cpu(end_seconds=1001)]))

    def test_reset_and_parent_child_cpu_double_count_refuse(self):
        with self.assertRaisesRegex(ValueError, "counter_reset"):
            rows("throughput", throughput(cpu_counters=[cpu(end_seconds=90)]))
        with self.assertRaisesRegex(ValueError, "cpu_double_count"):
            rows(
                "throughput",
                throughput(
                    cpu_counters=[
                        cpu(),
                        cpu(counter_id="child", scope_id="child", ancestors=["scope"]),
                    ]
                ),
            )

    def test_process_and_cgroup_counters_cannot_mix(self):
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            rows(
                "throughput",
                throughput(
                    cpu_counters=[
                        cpu(),
                        cpu(
                            counter_id="two",
                            scope_id="two",
                            counter_kind="evaluator_cgroup",
                        ),
                    ]
                ),
            )

    def test_cpu_counter_crossing_window_is_not_prorated(self):
        with self.assertRaisesRegex(ValueError, "no prorating"):
            rows(
                "throughput",
                throughput(cpu_counters=[cpu(interval=[START - 1, START + 3600])]),
            )

    def test_missing_cpu_counter_and_selective_counts_remain_null(self):
        result = values(
            "throughput", throughput(cpu_counters=[], counts_complete=False)
        )
        self.assertIsNone(result["active_cpu_core_hours"])
        self.assertIsNone(result["programs"])

    def test_configured_thread_product_is_not_measured_concurrency(self):
        result = values(
            "throughput",
            {
                "thread_configuration": {
                    "workers": 8,
                    "inner_threads": 4,
                    "declared_cores": 16,
                }
            },
        )
        self.assertEqual(result["configured_thread_product"], 32)
        self.assertEqual(result["configured_oversubscription_ratio"], 2)


class ArrivalTests(unittest.TestCase):
    def test_conflicting_catalog_membership_across_contexts_refuses(self):
        for first, second in [(True, False), (False, True)]:
            with (
                self.subTest(first=first),
                self.assertRaisesRegex(ValueError, "catalog membership"),
            ):
                rows(
                    "arrivals",
                    arrivals(
                        [
                            outcome(in_frozen_catalog=first),
                            outcome("two", context="c2", in_frozen_catalog=second),
                        ]
                    ),
                )

    def test_catalog_membership_conflict_outside_window_refuses(self):
        with self.assertRaisesRegex(ValueError, "catalog membership"):
            rows(
                "arrivals",
                arrivals(
                    [
                        outcome(
                            at=-1,
                            first_program_seen_at=START - 10,
                            in_frozen_catalog=True,
                        ),
                        outcome("two", context="c2", in_frozen_catalog=False),
                    ]
                ),
            )

    def test_distinct_program_catalog_memberships_remain_valid(self):
        a = outcome(in_frozen_catalog=True)
        b = outcome("two", context="c2", program="b" * 64)
        result = values("arrivals", arrivals([a, a, b]))
        self.assertEqual(result["new_qualified_context_program_outcomes"], 2)
        self.assertEqual(result["new_qualified_non_catalog_programs"], 1)

    def test_unknown_catalog_membership_is_not_inferred_from_peer(self):
        for unknown in [
            {"in_frozen_catalog": None},
            {"catalog_membership_proof_sha256": None, "in_frozen_catalog": True},
        ]:
            with self.subTest(unknown=unknown):
                result = values(
                    "arrivals",
                    arrivals([outcome(), outcome("two", context="c2", **unknown)]),
                )
                self.assertEqual(result["new_qualified_context_program_outcomes"], 2)
                self.assertIsNone(result["new_qualified_non_catalog_programs"])
                self.assertIsNone(result["qualified_non_catalog_programs_per_hour"])

    def test_frozen_catalog_identity_scopes_membership(self):
        for catalog, membership, expected in [
            ("a" * 64, False, 1),
            ("b" * 64, True, 0),
        ]:
            with self.subTest(catalog=catalog):
                data = arrivals([outcome(in_frozen_catalog=membership)])
                data["frozen_catalog_sha256"] = catalog
                result = values("arrivals", data)
                self.assertEqual(result["new_qualified_non_catalog_programs"], expected)

    def test_missing_vocabulary_or_canonicalizer_prevents_qualified_join(self):
        for field in ("task", "vocabulary_sha256", "canonicalizer_sha256"):
            event = outcome()
            event["comparison_identity"].pop(field)
            self.assertIsNone(
                values("arrivals", arrivals([event]))[
                    "new_qualified_context_program_outcomes"
                ]
            )

    def test_train_and_eval_are_not_pooled(self):
        with self.assertRaisesRegex(ValueError, "mixed role"):
            rows(
                "arrivals",
                arrivals([outcome(), outcome("two", context="c2", role="VAL")]),
            )

    def test_parent_counts_do_not_weight_by_duplicate_disclosures(self):
        a = outcome()
        records = rows(
            "arrivals",
            arrivals([a, a, outcome("two", context="c2", parent_id="other")]),
        )
        total = next(
            r
            for r in records
            if r["metric"] == "new_qualified_context_program_outcomes"
        )
        self.assertEqual(
            json.loads(total["details"])["parent_counts"], {"parent": 1, "other": 1}
        )

    def test_absent_ledger_is_null_complete_empty_can_prove_zero(self):
        self.assertIsNone(
            values("arrivals", {})["qualified_context_program_outcomes_per_hour"]
        )
        self.assertEqual(
            values("arrivals", arrivals([]))[
                "qualified_context_program_outcomes_per_hour"
            ],
            0,
        )

    def test_two_contexts_one_program_and_alias_dedup(self):
        a, b = outcome(), outcome("two", context="c2")
        result = values("arrivals", arrivals([a, a, b]))
        self.assertEqual(result["new_qualified_context_program_outcomes"], 2)
        self.assertEqual(result["new_qualified_non_catalog_programs"], 1)

    def test_window_is_half_open_and_redisclosure_is_not_new(self):
        data = arrivals(
            [
                outcome(at=-1),
                outcome("two", at=-1),
            ]
        )
        self.assertEqual(
            values("arrivals", data)["new_qualified_context_program_outcomes"], 0
        )

    def test_pair_first_qualification_clocks_cannot_conflict(self):
        with self.assertRaisesRegex(ValueError, "first_qualification"):
            rows("arrivals", arrivals([outcome(at=-1), outcome("two", at=0)]))

    def test_old_program_on_new_context_is_not_a_new_program(self):
        result = values(
            "arrivals", arrivals([outcome(first_program_seen_at=START - 1)])
        )
        self.assertEqual(result["new_qualified_context_program_outcomes"], 1)
        self.assertEqual(result["new_qualified_non_catalog_programs"], 0)

    def test_missing_global_first_seen_proof_keeps_novelty_null(self):
        result = values(
            "arrivals", arrivals([outcome(first_program_seen_proof_sha256=None)])
        )
        self.assertIsNone(result["new_qualified_non_catalog_programs"])

    def test_distinct_pairs_at_half_open_boundary(self):
        data = arrivals(
            [
                outcome(at=0),
                outcome("two", context="c2", at=3599),
                outcome("three", context="c3", at=3600),
            ]
        )
        self.assertEqual(
            values("arrivals", data)["new_qualified_context_program_outcomes"], 2
        )

    def test_missing_catalog_proof_does_not_erase_measured_outcomes(self):
        result = values(
            "arrivals", arrivals([outcome(catalog_membership_proof_sha256=None)])
        )
        self.assertEqual(result["new_qualified_context_program_outcomes"], 1)
        self.assertIsNone(result["new_qualified_non_catalog_programs"])

    def test_scratch_local_verification_and_missing_novelty_stay_null(self):
        for key in [
            "qualification_receipt_sha256",
            "remote_version_readback_sha256",
            "first_qualified_proof_sha256",
        ]:
            with self.subTest(key=key):
                result = values("arrivals", arrivals([outcome(**{key: None})]))
                self.assertIsNone(result["qualified_context_program_outcomes_per_hour"])

    def test_conflicting_outcome_parent_refuses(self):
        with self.assertRaisesRegex(ValueError, "comparison_identity_mismatch"):
            rows("arrivals", arrivals([outcome(), outcome("two", parent_id="other")]))


class FeedbackTests(unittest.TestCase):
    def test_missing_episodes_have_seven_explicit_null_clocks(self):
        result = values("feedback", {})
        self.assertIn("feedback_to_first_update_seconds", result)
        self.assertIsNone(result["feedback_to_first_update_seconds"])
        self.assertIsNone(result["contexts_receiving_actor_targets"])
        self.assertIsNone(result["heldout_gain_per_declared_resource_budget"])
        for stage in (
            "proposed",
            "compute_started",
            "locally_verified",
            "remotely_qualified",
            "adopted",
            "first_trained",
            "first_evaluated",
        ):
            self.assertIn("feedback_timestamp:" + stage, result)
            self.assertIsNone(result["feedback_timestamp:" + stage])

    def test_feedback_asof_is_fixed_to_window_end(self):
        with self.assertRaisesRegex(ValueError, "observation clock"):
            rows("feedback", {"episodes": [episode()], "observed_until": clock(7200)})

    def test_stage_latency_requires_both_observed_endpoints(self):
        e = episode()
        result = values("feedback", feedback(e))
        self.assertEqual(result["feedback_stage_seconds:proposed->compute_started"], 5)
        e["clocks"].pop("compute_started")
        self.assertIsNone(
            values("feedback", feedback(e))[
                "feedback_stage_seconds:proposed->compute_started"
            ]
        )

    def test_distinct_update_and_exposure_endpoints(self):
        result = values("feedback", feedback(episode()))
        self.assertEqual(result["feedback_to_first_update_seconds"], 20)
        self.assertEqual(result["qualified_feedback_to_first_update_seconds"], 11)
        self.assertEqual(result["T_feedback"], 18)

    def test_adopted_without_update_is_censored(self):
        e = episode()
        e["clocks"].pop("first_trained")
        result = values("feedback", feedback(e))
        self.assertIsNone(result["feedback_to_first_update_seconds"])
        self.assertEqual(result["feedback_censored_age_seconds"], 3600)

    def test_future_update_is_not_observed(self):
        e = episode()
        e["clocks"]["first_trained"] = clock(4000)
        result = values("feedback", feedback(e))
        self.assertIsNone(result["feedback_to_first_update_seconds"])
        self.assertEqual(result["feedback_censored_n"], 1)

    def test_unproven_clock_and_update_zero_probe_are_not_training(self):
        for change in [
            {"consumption_receipt_sha256": None},
            {"parameter_change_receipt_sha256": None},
            {"optimizer_update": 0},
        ]:
            result = values("feedback", feedback(episode(**change)))
            self.assertIsNone(result["feedback_timestamp:first_trained"])
            self.assertIsNone(result["feedback_to_first_update_seconds"])

    def test_cross_host_without_bound_is_unmeasured(self):
        self.assertEqual(duration(clock(0), clock(10, "host-b")), (None, None))
        e = episode()
        e["clocks"]["first_trained"]["domain"] = "host-b"
        result = values("feedback", feedback(e))
        self.assertEqual(result["feedback_completed_n"], 0)
        self.assertEqual(result["feedback_censored_n"], 0)
        self.assertEqual(result["feedback_unknown_latency_n"], 1)

    def test_cross_host_bound_is_preserved(self):
        self.assertEqual(
            duration(clock(10, uncertainty=0.1), clock(20, "host-b", 0.1)), (10, 0.2)
        )

    def test_decreasing_common_clock_refuses(self):
        with self.assertRaisesRegex(ValueError, "clock_order"):
            duration(clock(10), clock(5))

    def test_episode_identity_conflict_refuses(self):
        with self.assertRaisesRegex(ValueError, "conflicting episode_id"):
            rows("feedback", {"episodes": [episode(), episode(parent_id="other")]})

    def test_censored_population_is_separate_from_completed_quantile(self):
        incomplete = episode(episode_id="second")
        incomplete["clocks"].pop("first_trained")
        records = rows(
            "feedback",
            {"episodes": [episode(), incomplete], "observed_until": clock(3600)},
        )
        mean = next(
            r
            for r in records
            if r["metric"] == "proposal_to_update:completed_mean_seconds"
        )
        self.assertEqual(mean["value"], 20)
        self.assertEqual(mean["n"], 1)
        self.assertEqual(json.loads(mean["details"])["censored_n"], 1)


class PhaseCacheTests(unittest.TestCase):
    def test_queue_mean_uses_complete_right_continuous_stream(self):
        data = {
            "queue_events": [
                {"at": START, "depth": 0},
                {"at": START + 1200, "depth": 2},
                {"at": START + 2400, "depth": 0},
            ],
            "queue_event_coverage_complete": True,
        }
        self.assertAlmostEqual(
            values("phases", data)["queue_time_weighted_depth"], 2 / 3
        )
        data["queue_event_coverage_complete"] = False
        self.assertIsNone(values("phases", data)["queue_time_weighted_depth"])

    def test_queue_late_first_sample_and_duplicate_clocks_refuse_full_mean(self):
        data = {
            "queue_events": [{"at": START + 1, "depth": 0}],
            "queue_event_coverage_complete": True,
        }
        self.assertIsNone(values("phases", data)["queue_time_weighted_depth"])
        data["queue_events"] *= 2
        with self.assertRaisesRegex(ValueError, "queue event stream"):
            rows("phases", data)

    def test_custody_ends_at_exact_version_readback_not_put_return(self):
        phase = {
            "phase_id": "custody",
            "name": "custody",
            "start": clock(5),
            "end": clock(25),
            "put_return": clock(10),
            "remote_version_id": "v",
            "readback_sha256": SHA,
        }
        self.assertEqual(
            values("phases", {"records": [phase]})["custody:completed_mean_seconds"], 20
        )
        phase.pop("readback_sha256")
        self.assertIsNone(
            values("phases", {"records": [phase]})["custody:completed_mean_seconds"]
        )

    def test_nested_phases_are_not_summed_and_pending_phase_is_censored(self):
        data = {
            "records": [
                {
                    "phase_id": "a",
                    "name": "execution",
                    "start": clock(0),
                    "end": clock(10),
                },
                {
                    "phase_id": "b",
                    "name": "compile",
                    "start": clock(2),
                    "end": clock(6),
                },
                {"phase_id": "c", "name": "compile", "start": clock(10), "end": None},
            ],
            "observed_until": clock(3600),
        }
        result = values("phases", data)
        self.assertEqual(result["execution:completed_mean_seconds"], 10)
        self.assertEqual(result["compile:completed_mean_seconds"], 4)
        self.assertEqual(result["compile:censored_age_mean_seconds"], 3590)
        self.assertNotIn("summed_phase_wall_seconds", result)

    def test_cache_complete_denominator_alias_and_unknowns(self):
        lookups = [
            {
                "lookup_id": str(i),
                "hit": i < 3,
                "cache_kind": "compile",
                "identity_sha256": SHA,
            }
            for i in range(10)
        ]
        data = {
            "lookups": lookups + lookups[:1],
            "cache_kind": "compile",
            "identity_sha256": SHA,
        }
        self.assertEqual(values("cache", data)["cache_hit_rate"], 0.3)
        self.assertIsNone(values("cache", data, complete=False)["cache_hit_rate"])
        self.assertIsNone(values("cache", dict(data, lookups=[]))["cache_hit_rate"])

    def test_changed_cache_identity_refuses(self):
        with self.assertRaisesRegex(ValueError, "cache_identity_mismatch"):
            rows(
                "cache",
                {
                    "identity_sha256": SHA,
                    "cache_kind": "compile",
                    "lookups": [
                        {
                            "lookup_id": "a",
                            "hit": True,
                            "cache_kind": "compile",
                            "identity_sha256": "b" * 64,
                        }
                    ],
                },
            )


class CostTests(unittest.TestCase):
    def test_literal_prefix_overlap_without_directory_delimiter_refuses(self):
        for left, right in [
            ("root/a", "root/ab/"),
            ("root", "root-child"),
            ("root/a", "root/a"),
            ("", "anything"),
            ("root/", "root/child/"),
        ]:
            for first, second in [(left, right), (right, left)]:
                with (
                    self.subTest(first=first, second=second),
                    self.assertRaisesRegex(ValueError, "scope_double_count"),
                ):
                    rows(
                        "cost",
                        {"records": [cost(prefix=first), cost("b", prefix=second)]},
                    )

    def test_literal_disjoint_prefixes_and_aliases_preserve_total(self):
        for first, second in [
            ("root/a/", "root/ab/"),
            ("root/a", "root/b"),
            ("Root/a", "root/a"),
        ]:
            with self.subTest(first=first, second=second):
                a, b = cost(prefix=first), cost("b", prefix=second, usd=0.2)
                result = values(
                    "cost", {"records": [a, a, b], "expected_categories": ["storage"]}
                )
                self.assertAlmostEqual(result["total_usd"], 0.3)

    def test_literal_prefixes_are_separate_across_bucket_or_category(self):
        for difference, categories in [
            ({"bucket": "other"}, ["storage"]),
            ({"category": "requests"}, ["storage", "requests"]),
        ]:
            with self.subTest(difference=difference):
                result = values(
                    "cost",
                    {
                        "records": [
                            cost(prefix="root"),
                            cost("b", prefix="root-child", usd=0.2, **difference),
                        ],
                        "expected_categories": categories,
                    },
                )
                self.assertAlmostEqual(result["total_usd"], 0.3)

    def test_disjoint_times_with_overlapping_prefixes_stay_unmeasured(self):
        result = values(
            "cost",
            {
                "records": [
                    cost(prefix="root", interval=[START, START + 1800]),
                    cost("b", prefix="root-child", interval=[START + 1800, WINDOW[1]]),
                ],
                "expected_categories": ["storage"],
            },
        )
        self.assertIsNone(result["total_usd"])
        self.assertIsNone(result["known_partial_usd"])

    def test_disjoint_unknown_prefix_cost_keeps_total_null(self):
        a, b = cost(prefix="root/a"), cost("b", prefix="root/b", usd=None)
        result = values(
            "cost", {"records": [a, a, b], "expected_categories": ["storage"]}
        )
        self.assertIsNone(result["total_usd"])
        self.assertEqual(result["known_partial_usd"], 0.1)

    def test_physical_requests_aliases_count_once(self):
        a = cost(request_operations=3)
        result = values("cost", {"records": [a, a], "requests_complete": True})
        self.assertEqual(result["request_operations"], 3)

    def test_complete_disjoint_cost_and_aliases(self):
        a, b = cost(), cost("b", prefix="root/b/", usd=0.2)
        result = values(
            "cost", {"records": [a, a, b], "expected_categories": ["storage"]}
        )
        self.assertAlmostEqual(result["total_usd"], 0.3)

    def test_unknown_component_keeps_total_null_and_partial_separate(self):
        result = values(
            "cost",
            {
                "records": [cost(), cost("b", prefix="root/b/", usd=None)],
                "expected_categories": ["storage"],
            },
        )
        self.assertIsNone(result["total_usd"])
        self.assertEqual(result["known_partial_usd"], 0.1)

    def test_billed_estimated_currency_and_pricing_mixes_refuse(self):
        for change in [
            {"cost_kind": "billed"},
            {"currency": "EUR"},
            {"pricing_identity": "other"},
        ]:
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "cost_identity_mismatch"),
            ):
                rows(
                    "cost", {"records": [cost(), cost("b", prefix="root/b/", **change)]}
                )

    def test_overlapping_prefix_rollups_refuse(self):
        with self.assertRaisesRegex(ValueError, "scope_double_count"):
            rows(
                "cost",
                {"records": [cost(prefix="root/"), cost("b", prefix="root/child/")]},
            )

    def test_empty_bucket_prefix_overlaps_every_child(self):
        with self.assertRaisesRegex(ValueError, "scope_double_count"):
            rows("cost", {"records": [cost(prefix=""), cost("b", prefix="child/")]})

    def test_monthly_stock_cannot_fill_daily_bill_or_additions(self):
        result = values(
            "cost",
            {
                "records": [
                    cost(
                        window_kind="month",
                        bytes_measure_kind="stock",
                        stock_as_of=WINDOW[1],
                    )
                ],
                "expected_categories": ["storage"],
            },
        )
        self.assertEqual(result["stock_bytes"], 100)
        self.assertIsNone(result["bytes_added"])
        self.assertIsNone(result["total_usd"])

    def test_trailing_window_cannot_fill_utc_day_and_zero_outcomes_null(self):
        record = cost(interval=[START - 43200, START + 43200])
        result = values(
            "cost", {"records": [record], "expected_categories": ["storage"]}
        )
        self.assertIsNone(result["total_usd"])
        self.assertIsNone(result["usd_per_qualified_outcome"])

    def test_negative_or_nonfinite_cost_refuses(self):
        for amount in [-0.1, float("inf"), float("nan")]:
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                rows("cost", {"records": [cost(usd=amount)]})


class BuilderTests(unittest.TestCase):
    def test_copied_owner_parquet_cost_ledger_and_source_pins(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "owner.parquet"
            pq.write_table(pa.Table.from_pylist([cost()]), ledger)
            owner = root / "owner.json"
            spec = export("cost", {"expected_categories": ["storage"]})
            spec["ledger"] = {
                "path": str(ledger),
                "sha256": digest(ledger),
                "bytes": ledger.stat().st_size,
            }
            owner.write_text(json.dumps(spec))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "jope.rlfw-baseline-manifest/v1",
                        "date": "2026-10-14",
                        "window": WINDOW,
                        "window_kind": "whole_run",
                        "synthetic": True,
                        "exports": {
                            "cost": {
                                "path": str(owner),
                                "sha256": digest(owner),
                                "bytes": owner.stat().st_size,
                            }
                        },
                    }
                )
            )
            before = {
                p: (digest(p), p.stat().st_mtime_ns) for p in (ledger, owner, manifest)
            }
            receipt = build(manifest, root / "output", commit="f" * 40)
            result = {
                r["metric"]: r
                for r in read_daily([root / "output/daily/2026-10-14.parquet"])
            }
            self.assertEqual(result["total_usd"]["value"], 0.1)
            self.assertIn(str(owner) + ":1", result["total_usd"]["sources"])
            self.assertEqual(len(receipt["inputs"]), 3)
            self.assertEqual(
                before, {p: (digest(p), p.stat().st_mtime_ns) for p in before}
            )
            self.assertTrue(receipt["synthetic"])

    def test_missing_exports_roundtrip_nulls_and_html_markdown_from_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "jope.rlfw-baseline-manifest/v1",
                        "date": "2026-10-14",
                        "window": WINDOW,
                        "window_kind": "whole_run",
                        "exports": {},
                    }
                )
            )
            output = root / "output"
            receipt = build(manifest, output, commit="f" * 40)
            records = read_daily([output / "daily/2026-10-14.parquet"])
            self.assertTrue(receipt["passed"])
            self.assertEqual(
                receipt["null_rows"], sum(r["value"] is None for r in records)
            )
            self.assertTrue(
                any(r["metric"] == "total_usd" and r["value"] is None for r in records)
            )
            self.assertIn("unmeasured", (output / "dashboard.html").read_text())
            self.assertIn("unmeasured", (output / "METRICS.md").read_text())
            for ref in receipt["files"]:
                self.assertEqual(digest(output / ref["path"]), ref["sha256"])
            self.assertEqual(
                json.loads((output / "COMPLETED.json").read_text()), receipt
            )
            with self.assertRaises(ValueError):
                build(manifest, output, commit="f" * 40)

    def test_bad_owner_pin_never_produces_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owner = root / "owner.json"
            owner.write_text("{}")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "jope.rlfw-baseline-manifest/v1",
                        "date": "2026-10-14",
                        "window": WINDOW,
                        "window_kind": "whole_run",
                        "exports": {
                            "cost": {"path": str(owner), "sha256": "0" * 64, "bytes": 2}
                        },
                    }
                )
            )
            with self.assertRaises(ValueError):
                build(manifest, root / "output", commit="f" * 40)
            self.assertFalse((root / "output/COMPLETED.json").exists())

    def test_non_midnight_day_and_duplicate_json_keys_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                read_json(manifest)
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "jope.rlfw-baseline-manifest/v1",
                        "date": "2026-10-14",
                        "window": WINDOW,
                        "window_kind": "utc_day",
                        "exports": {},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "midnight"):
                build(manifest, root / "output", commit="f" * 40)


if __name__ == "__main__":
    unittest.main()
