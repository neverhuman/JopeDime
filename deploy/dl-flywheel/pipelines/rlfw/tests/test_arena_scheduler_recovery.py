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


from .test_arena_scheduler import _SchedulerSupport, program

class SchedulerRecoveryTests(_SchedulerSupport):
    def test_r5_interrupted_terminal_attempt_resumes_without_native_or_new_attempt(
        self,
    ):
        calls = []
        spec = replace(
            self.spec, kpi_version=R5_KPI_VERSION, outcome_classes=R5_OUTCOME_CLASSES
        )

        def evaluate(request):
            calls.append(request)
            return {
                **self.result(request, "parity_unqualified"),
                "parity_verified": False,
                "parity_max_normalized_error": 1e-5,
            }

        arena, _ = self.arena(spec, evaluate=evaluate)
        context, arm = spec.contexts[0], spec.arms[1]
        candidate = arena.arms.slate(context, arm, 1, 8)[0]

        def interrupted(*args):
            raise RuntimeError("interruption before logical charge")

        for slot, cached, wall in ((1, False, 1.0), (2, True, 0.0)):
            with self.assertRaisesRegex(
                RuntimeError, "interruption before logical charge"
            ):
                arena.oracle.resolve(
                    context,
                    arm,
                    1,
                    slot,
                    candidate,
                    duplicate=slot > 1,
                    commit=interrupted,
                    attempt_store=arena,
                )
            restored, _ = self.arena(
                spec,
                evaluate=lambda r: self.fail("terminal fate must never execute again"),
            )
            row = restored.oracle.resolve(
                context,
                arm,
                1,
                slot,
                candidate,
                duplicate=slot > 1,
                commit=restored.commit,
                attempt_store=restored,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(row.attempts, 1)
            self.assertEqual(row.cache_hit, cached)
            self.assertEqual(row.wall_seconds, wall)
            self.assertEqual(row.cpu_seconds, 2 * wall)
            self.assertEqual(row.unknown_cost_attempts, 0)
            self.assertTrue(row.charged and row.q is None and restored.committed(row))
            self.assertEqual(
                len(restored.attempt_history((context.context_id, arm.name, 1, slot))),
                1,
            )
            arena = restored

    def test_r5_frozen_cache_and_taxonomy_reward_refusals(self):
        context = self.spec.contexts[0]
        value = {
            **self.result(
                {"context_identity": context.identity_sha256, "program": self.p},
                "parity_unqualified",
            ),
            "parity_verified": False,
            "parity_max_normalized_error": None,
        }
        key = digest(
            {
                "context_identity": context.identity_sha256,
                "program_sha256": self.p["program_sha256"],
                "evaluator_sha256": self.spec.evaluator_sha256,
                "canonicalizer_sha256": self.spec.canonicalizer_sha256,
                "protocol": self.spec.protocol,
            }
        )
        cache = {key: value}
        spec = replace(
            self.spec,
            contexts=(context,),
            kpi_version=R5_KPI_VERSION,
            outcome_classes=R5_OUTCOME_CLASSES,
            cache_snapshot_sha256=digest(cache),
        )
        arena, _ = self.arena(
            spec,
            cache=cache,
            evaluate=lambda r: self.fail("known unqualified cache must not execute"),
        )
        arena.run()
        self.assertTrue(
            all(
                r.charged and r.cache_hit and r.q is None and r.wall_seconds == 0.0
                for r in arena.rows()
            )
        )
        candidate = arena.arms.catalog[0]
        for changed in (
            {"q_grid": 0},
            {"q": 0.0},
            {"q_opt": 0.0},
            {"qualified": True},
            {"parity_verified": True},
            {"parity_max_normalized_error": 1e-10},
            {"outcome_class": "parity_custody"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                arena.oracle.validate({**value, **changed}, context, candidate)
        legacy = BlindOracle(
            self.spec,
            self.result,
            lambda c: {"role": "development", "role_map_sha256": SHA},
            {},
        )
        with self.assertRaises(ValueError):
            legacy.validate(value, context, candidate)

    def test_result_is_committed_before_oracle_returns(self):
        arena, _ = self.arena()
        c, arm = self.spec.contexts[0], self.spec.arms[1]
        candidate = arena.arms.slate(c, arm, 1, 8)[0]
        observed = []

        def commit(row, key, result):
            self.assertFalse(arena.committed(row))
            arena.commit(row, key, result)
            observed.append(arena.committed(row))

        row = arena.oracle.resolve(
            c, arm, 1, 1, candidate, duplicate=False, commit=commit
        )
        self.assertEqual(observed, [True])
        self.assertTrue(arena.committed(row))

    def test_resume_preserves_slates_results_and_closed_contexts(self):
        arena, _ = self.arena()
        arena.run(max_contexts=2)
        before = [r.to_dict() for r in arena.rows()]
        restored, _ = self.arena(
            evaluate=lambda r: self.fail("completed contexts must not execute again")
        )
        # A full rerun would execute the remaining three; restrict resume to already-closed roster.
        self.assertEqual([r.to_dict() for r in restored.rows()], before)
        self.assertEqual(len(restored.closed_contexts()), 2)
        arena.run()
        self.assertEqual(len(arena.closed_contexts()), 5)

    def test_five_context_determinism_slates_traces_and_kpis(self):
        arena, traces = self.arena()
        arena.run()
        one = compute_report(
            self.spec, arena.rows(), arena.hwm_values(), arena.closed_contexts()
        )
        other_root = self.root / "other"
        s = replace(
            self.spec,
            bench_root=str(other_root),
            out_dir=str(other_root / "runs/toy"),
            state_path=str(other_root / "state/arena.sqlite"),
        )
        other, other_traces = self.arena(s)
        other.run()
        two = compute_report(
            s, other.rows(), other.hwm_values(), other.closed_contexts()
        )
        self.assertEqual(one["tau_sensitivity"], two["tau_sensitivity"])
        self.assertEqual(one["per_context"], two["per_context"])

        def without_run_ids(values):
            return [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("game_spec_sha256", "decision_id", "slate_id")
                }
                for r in values
            ]

        self.assertEqual(without_run_ids(traces), without_run_ids(other_traces))
        arena.run()
        self.assertEqual(len(traces), 160)

    def test_uniform_over_whole_programs_on_uneven_trie(self):
        catalog = [
            program("one", (1, 3, 2)),
            program("two", (1, 4, 5, 2)),
            program("three", (1, 4, 6, 2)),
        ]
        arms = BaselineArms(catalog, {}, None)
        for p in arms.catalog:
            per = arms.uniform_logprob(p.tokens)
            self.assertAlmostEqual(sum(per), -__import__("math").log(3), places=12)
        counts = {p["program_sha256"]: 0 for p in catalog}
        for seed in range(1000):
            for candidate in arms.slate(
                self.spec.contexts[0], self.spec.arms[1], seed, 8
            ):
                counts[candidate.program.program_sha256] += 1
        self.assertTrue(all(abs(n / 8000 - 1 / 3) < 0.02 for n in counts.values()))





    def test_fresh_roles_parity_hash_and_stop_guardrails(self):
        arena, _ = self.arena(check=lambda c: {"role": "train", "role_map_sha256": SHA})
        with self.assertRaises(ValueError):
            arena.run()
        guarded = replace(
            self.spec,
            game_id="guarded",
            out_dir=str(self.root / "runs/guarded"),
            state_path=str(self.root / "state/guarded.sqlite"),
        )
        invalid, _ = self.arena(
            guarded,
            evaluate=lambda r: {**self.result(r), "parity_max_normalized_error": 1e-8},
        )
        with self.assertRaises(ValueError):
            invalid.run(max_contexts=1)
        (self.root / "state/STOP.json").write_text("{}")
        with self.assertRaises(RuntimeError):
            invalid.run()

    def test_reports_hash_every_artifact_and_drop_partial_context(self):
        arena, _ = self.arena()
        arena.run(max_contexts=1)
        report = write_reports(
            self.spec, arena.rows(), arena.hwm_values(), arena.closed_contexts()
        )
        self.assertEqual(report["closed_contexts"], 1)
        self.assertTrue(report["partial_panel"])
        for name in ("report.json", "report.md", "per_context.csv", "slots.json"):
            self.assertIn(
                self.spec.sha256, (Path(self.spec.out_dir) / name).read_text()
            )
        with self.assertRaises(ValueError):
            compute_report(
                self.spec,
                arena.rows()[:-1],
                arena.hwm_values(),
                arena.closed_contexts(),
            )

    def test_comparator_score_requires_original_measured_receipt(self):
        arena, _ = self.arena()
        arena.run(max_contexts=1)
        forged = arena.hwm_values()
        forged[arena.closed_contexts()[0]] = 999.0
        with self.assertRaises(ValueError):
            compute_report(self.spec, arena.rows(), forged, arena.closed_contexts())

    def test_manifest_detects_changed_report_bytes(self):
        arena, _ = self.arena()
        arena.run(max_contexts=1)
        write_reports(
            self.spec, arena.rows(), arena.hwm_values(), arena.closed_contexts()
        )
        read_run(self.spec.out_dir)
        path = Path(self.spec.out_dir) / "slots.json"
        path.write_text(path.read_text().replace("500000", "900000"))
        with self.assertRaises(ValueError):
            read_run(self.spec.out_dir)

    def test_micro_q_conversion_and_relative_parity_limit(self):
        arena, _ = self.arena()
        arena.run(max_contexts=1)
        self.assertEqual(arena.rows()[0].q, 0.5)
        p = arena.arms.catalog[0]
        context = self.spec.contexts[0]
        measured = self.result(
            {"context_identity": context.identity_sha256, "program": p.to_dict()}
        )
        # A large reference permits absolute error >1e-9; the pinned normalized
        # criterion stays unchanged and its field is named accurately.
        arena.oracle.validate(
            {
                **measured,
                "parity_max_abs_error": 1e-7,
                "parity_max_normalized_error": 1e-10,
            },
            context,
            p,
        )
        with self.assertRaises(ValueError):
            arena.oracle.validate({**measured, "q_grid": 0.5}, context, p)



if __name__ == "__main__":
    unittest.main()
