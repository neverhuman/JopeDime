"""Synthetic arena: blind charging, restart, cache and deterministic replay."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rlfw.arena.arms import BaselineArms
from rlfw.arena.game_spec import digest, R5_KPI_VERSION, R5_OUTCOME_CLASSES
from rlfw.arena.kpi import compute_report
from rlfw.arena.oracle import BlindOracle
from rlfw.arena.report import read_run, write_reports
from rlfw.arena.scheduler import ArenaScheduler, context_order
from rlfw.bench_cli import (
    compare,
    parser,
    reuse_incumbents,
    verify_device,
    verify_runtime_scope,
)
from rlfw.tests.test_arena_kpi import SHA, spec_at


def program(text="one", tokens=(1, 2)):
    return {
        "dsl": text,
        "program_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tokens": list(tokens),
    }


class _SchedulerSupport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.p = program()
        self.spec = replace(
            spec_at(self.root),
            cache_snapshot_sha256=digest({}),
            contexts=tuple(
                replace(
                    c,
                    hwm_program_sha256=self.p["program_sha256"],
                    identity_sha256=digest(c.context_id),
                )
                for c in spec_at(self.root).contexts
            ),
        )

    def result(self, request, outcome="completed"):
        p = request["program"]
        return {
            "outcome_class": outcome,
            "qualified": outcome == "completed",
            "q_grid": 500_000 if outcome == "completed" else None,
            "context_identity": request["context_identity"],
            "program_sha256": p["program_sha256"],
            "evaluator_sha256": SHA,
            "canonicalizer_sha256": SHA,
            "protocol": self.spec.protocol,
            "role_map_sha256": SHA,
            "closure_verified": True,
            "parity_verified": True,
            "parity_max_normalized_error": 0.0,
            "receipt_sha256": digest(
                [request["context_identity"], p["program_sha256"]]
            ),
            "wall_seconds": 1.0,
            "cpu_seconds": 2.0,
        }

    def arena(self, spec=None, evaluate=None, cache=None, check=None):
        spec = spec or self.spec
        traces = []

        def sample(cid, arm, seed):
            return {
                "program": self.p,
                "tokens": self.p["tokens"],
                "trie_sha256": SHA,
                "checkpoint_parameter_sha256": SHA,
                "total_logprob": 0.0,
                "per_token_logprob": [0.0],
                "logprob_support": "trie_children",
                "logprob_aggregation": "sum_including_eos",
            }

        arms = BaselineArms(
            [self.p], {c.context_id: self.p for c in spec.contexts}, sample
        )
        oracle = BlindOracle(
            spec,
            evaluate or self.result,
            check or (lambda c: {"role": "development", "role_map_sha256": SHA}),
            cache or {},
        )
        scheduler = ArenaScheduler(spec, arms, oracle, traces.append)
        self.addCleanup(scheduler.close)
        return scheduler, traces


class SchedulerTests(_SchedulerSupport):
    def test_duplicates_and_cache_hits_charge_eight_and_blind_slates_first(self):
        calls = []
        scheduler = None

        def evaluator(request):
            calls.append(request)
            self.assertEqual(
                scheduler.db.execute(
                    "SELECT count(*) FROM slates WHERE context=?",
                    (request["context_id"],),
                ).fetchone()[0],
                3,
            )
            self.assertNotIn("q_grid", request)
            self.assertNotIn("q", request["program"])
            return self.result(request)

        scheduler, traces = self.arena(evaluate=evaluator)
        scheduler.run()
        rows = scheduler.rows()
        self.assertEqual(
            len(calls), 5
        )  # one actual execution per exact context/program
        self.assertEqual(len(rows), 5 * 17)
        for arm in ("uniform_trie", "trie_sample:toy"):
            matching = [r for r in rows if r.arm == arm]
            self.assertTrue(all(r.charged for r in matching))
            self.assertEqual(sum(r.duplicate for r in matching), 35)
        self.assertEqual(len(traces), 5 * 32)
        for ctx in self.spec.contexts:
            values = [v for v in traces if v["context_identity"] == ctx.identity_sha256]
            self.assertEqual(
                [v["kind"] for v in values], ["raw_draw"] * 16 + ["selected_slot"] * 16
            )
        self.assertTrue(all(scheduler.committed(r) for r in rows))

    def test_prewarming_does_not_change_logical_budget_or_outcome(self):
        first, _ = self.arena()
        first.run(max_contexts=1)
        context = context_order(self.spec)[0]
        key = first.oracle.execution_key(context, first.arms.catalog[0])
        cache = {
            key: self.result(
                {"context_identity": context.identity_sha256, "program": self.p}
            )
        }
        second_spec = replace(
            self.spec,
            game_id="warm",
            out_dir=str(self.root / "runs/warm"),
            state_path=str(self.root / "state/warm.sqlite"),
            cache_snapshot_sha256=digest(cache),
        )
        second, _ = self.arena(second_spec, cache=cache)
        second.run(max_contexts=1)
        for arena in (first, second):
            rows = [r for r in arena.rows() if r.arm == "uniform_trie"]
            self.assertEqual(sum(r.charged for r in rows), 8)
            self.assertEqual({r.q_grid for r in rows}, {500_000})

    def test_infra_and_parity_retry_twice_then_censor_without_charge(self):
        for outcome in ("infra_unknown", "parity_custody"):
            s = replace(
                self.spec,
                game_id=outcome,
                out_dir=str(self.root / f"runs/{outcome}"),
                state_path=str(self.root / f"state/{outcome}.sqlite"),
            )
            calls = []

            def evaluator(request):
                calls.append(request)
                return self.result(request, outcome)

            arena, _ = self.arena(s, evaluate=evaluator)
            arena.run(max_contexts=1)
            rows = arena.rows()
            self.assertEqual(len(calls), 17 * 3)
            self.assertTrue(
                all(
                    r.outcome == "censored" and not r.charged and r.attempts == 3
                    for r in rows
                )
            )
            self.assertTrue(all(r.q_grid is None for r in rows))

    def test_program_failure_and_timeout_are_charged_without_score(self):
        for outcome in ("program_failed", "timeout"):
            s = replace(
                self.spec,
                game_id=outcome,
                out_dir=str(self.root / f"runs/{outcome}"),
                state_path=str(self.root / f"state/{outcome}.sqlite"),
            )
            arena, _ = self.arena(
                s, evaluate=lambda request: self.result(request, outcome)
            )
            arena.run(max_contexts=1)
            self.assertTrue(
                all(
                    r.charged and r.q_grid is None and r.attempts == 1
                    for r in arena.rows()
                )
            )

    def test_r5_program_parity_is_terminal_cached_and_contexts_close(self):
        spec = replace(
            self.spec, kpi_version=R5_KPI_VERSION, outcome_classes=R5_OUTCOME_CLASSES
        )
        calls = []

        def evaluate(request):
            calls.append(request)
            return {
                **self.result(request, "parity_unqualified"),
                "parity_verified": False,
                "parity_max_normalized_error": 1e-5,
            }

        arena, _ = self.arena(spec, evaluate=evaluate)
        arena.run()
        rows = arena.rows()
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(rows), 5 * 17)
        self.assertEqual(len(arena.closed_contexts()), 5)
        self.assertTrue(
            all(
                r.outcome == "parity_unqualified"
                and r.charged
                and not r.qualified
                and r.q is None
                and r.q_grid is None
                and r.attempts == 1
                and r.censor_reason is None
                for r in rows
            )
        )
        self.assertTrue(all(arena.committed(r) for r in rows))
        self.assertEqual(sum(r.wall_seconds for r in rows), 5.0)
        self.assertEqual(sum(r.cpu_seconds for r in rows), 10.0)
        report = compute_report(spec, rows, arena.hwm_values(), arena.closed_contexts())
        self.assertEqual(report["no_comparable_hwm"], 5)
        self.assertEqual(
            report["hwm_coverage_categories"], {"hwm_parity_unqualified": 5}
        )
        self.assertIn("L=5", report["label"])
        for sensitivity in report["tau_sensitivity"].values():
            for metrics in sensitivity["arms"].values():
                self.assertEqual(metrics["parity_unqualified_rate"]["mean"], 1.0)
                self.assertIsNone(metrics["best_q@8"]["mean"])
                self.assertIsNone(metrics["win_rate"]["mean"])
        write_reports(spec, rows, arena.hwm_values(), arena.closed_contexts())
        read_run(spec.out_dir)
        restored, _ = self.arena(
            spec, evaluate=lambda r: self.fail("closed R5 contexts must not reexecute")
        )
        restored.run()
        self.assertEqual(
            [r.to_dict() for r in restored.rows()], [r.to_dict() for r in rows]
        )

    def test_r5_failure_keeps_successful_siblings_and_other_contexts_active(self):
        other = program("two", (1, 3, 2))
        spec = replace(
            self.spec, kpi_version=R5_KPI_VERSION, outcome_classes=R5_OUTCOME_CLASSES
        )
        calls = []

        def evaluate(request):
            calls.append(request)
            if request["program"]["program_sha256"] == self.p["program_sha256"]:
                return {
                    **self.result(request, "parity_unqualified"),
                    "parity_verified": False,
                    "parity_max_normalized_error": 1e-5,
                }
            return self.result(request)

        def sample(cid, arm, seed):
            return {
                "program": other,
                "tokens": other["tokens"],
                "trie_sha256": SHA,
                "checkpoint_parameter_sha256": SHA,
                "total_logprob": 0.0,
                "per_token_logprob": [0.0, 0.0],
                "logprob_support": "trie_children",
                "logprob_aggregation": "sum_including_eos",
            }

        arena, _ = self.arena(spec, evaluate=evaluate)
        arena.arms = BaselineArms(
            [self.p, other], {c.context_id: self.p for c in spec.contexts}, sample
        )
        arena.run()
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(arena.closed_contexts()), 5)
        self.assertTrue(all(r.charged and r.attempts == 1 for r in arena.rows()))
        self.assertTrue(any(r.outcome == "parity_unqualified" for r in arena.rows()))
        report = compute_report(
            spec, arena.rows(), arena.hwm_values(), arena.closed_contexts()
        )
        metrics = report["tau_sensitivity"]["0.01"]["arms"]["trie_sample:toy"]
        self.assertEqual(metrics["best_q@8"]["mean"], 0.5)
        self.assertEqual(metrics["valid@8"]["mean"], 1.0)
        self.assertIsNone(metrics["win_rate"]["mean"])



if __name__ == "__main__":
    unittest.main()
