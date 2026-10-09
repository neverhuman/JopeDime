"""Synthetic arena: blind charging, restart, cache and deterministic replay."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rlfw.arena.arms import BaselineArms
from rlfw.arena.game_spec import digest
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


class SchedulerTests(unittest.TestCase):
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

    def test_compare_cli_recomputes_paired_math_and_creates_only(self):
        arena, _ = self.arena()
        arena.run(max_contexts=2)
        write_reports(
            self.spec, arena.rows(), arena.hwm_values(), arena.closed_contexts()
        )
        out = self.root / "runs/comparison.json"
        args = parser().parse_args(
            [
                "compare",
                "--a",
                self.spec.out_dir,
                "--a-arm",
                "uniform_trie",
                "--b",
                self.spec.out_dir,
                "--b-arm",
                "trie_sample:toy",
                "--out",
                str(out),
            ]
        )
        result = compare(args)
        self.assertEqual(result["paired"]["score"]["mean"], 0.0)
        self.assertEqual(result["paired"]["score"]["ci95"], [0.0, 0.0])
        self.assertEqual(json.loads(out.read_text())["a_spec_sha256"], self.spec.sha256)
        with self.assertRaises(FileExistsError):
            compare(args)

    def test_live_cli_refuses_missing_health_deadline_and_unbounded_scope(self):
        with self.assertRaises(ValueError):
            verify_runtime_scope(self.spec)
        (self.root / "state").mkdir(exist_ok=True)
        (self.root / "state/health-baseline.json").write_text("{}")
        s = replace(self.spec, deadline_unix=9999999999.0)
        original = Path.read_text

        def source(path, *args, **kwargs):
            return (
                "0::/user.slice/foreign.scope\n"
                if str(path) == "/proc/self/cgroup"
                else original(path, *args, **kwargs)
            )

        with (
            mock.patch.object(Path, "read_text", source),
            self.assertRaises(ValueError),
        ):
            verify_runtime_scope(s)

    def test_missing_hwm_retains_secondary_metrics(self):
        c = self.spec.contexts[0]
        s = replace(self.spec, contexts=(replace(c, hwm_program_sha256=None),))
        arena, _ = self.arena(s)
        arena.arms.hwm_programs = {}
        arena.run()
        report = compute_report(
            s, arena.rows(), arena.hwm_values(), arena.closed_contexts()
        )
        self.assertEqual(report["no_comparable_hwm"], 1)
        metrics = report["tau_sensitivity"]["0.01"]["arms"]["uniform_trie"]
        self.assertIsNone(metrics["win_rate"]["mean"])
        self.assertEqual(metrics["best_q@8"]["mean"], 0.5)

    def test_crash_reservations_preserve_retry_ceiling_and_unknown_cost(self):
        arena, _ = self.arena(
            evaluate=lambda r: self.fail("an exhausted slot must not execute")
        )
        c, arm = self.spec.contexts[0], self.spec.arms[1]
        key = (c.context_id, arm.name, 1, 1)
        for _ in range(3):
            arena.reserve_attempt(key)
        candidate = arena.arms.slate(c, arm, 1, 8)[0]
        row = arena.oracle.resolve(
            c,
            arm,
            1,
            1,
            candidate,
            duplicate=False,
            commit=arena.commit,
            attempt_store=arena,
        )
        self.assertEqual(row.outcome, "censored")
        self.assertEqual(row.attempts, 3)
        self.assertEqual(row.unknown_cost_attempts, 3)
        self.assertFalse(row.charged)
        with self.assertRaises(ValueError):
            arena.reserve_attempt(key)

    def test_resume_uses_only_remaining_attempt_and_keeps_observed_costs(self):
        calls = []

        def evaluate(request):
            calls.append(request)
            return self.result(request)

        arena, _ = self.arena(evaluate=evaluate)
        c, arm = self.spec.contexts[0], self.spec.arms[1]
        key = (c.context_id, arm.name, 1, 1)
        arena.reserve_attempt(key)
        arena.reserve_attempt(key)
        request = {"context_identity": c.identity_sha256, "program": self.p}
        arena.record_unknown_attempt(key, 2, self.result(request, "infra_unknown"))
        candidate = arena.arms.slate(c, arm, 1, 8)[0]
        row = arena.oracle.resolve(
            c,
            arm,
            1,
            1,
            candidate,
            duplicate=False,
            commit=arena.commit,
            attempt_store=arena,
        )
        self.assertEqual([r["attempt"] for r in calls], [3])
        self.assertTrue(row.qualified)
        self.assertEqual(row.wall_seconds, 2.0)
        self.assertEqual(row.unknown_cost_attempts, 1)

    def test_runtime_helper_drift_and_deadline_refuse_before_new_attempt(self):
        helper = self.root / "helper.py"
        helper.write_text("pass\n")
        s = replace(
            self.spec,
            runtime_files=(
                ("helper.py", hashlib.sha256(helper.read_bytes()).hexdigest()),
            ),
        )
        arena, _ = self.arena(s)
        helper.write_text("changed\n")
        with self.assertRaises(ValueError):
            arena.run()
        expired = replace(
            self.spec,
            game_id="expired",
            out_dir=str(self.root / "runs/expired"),
            state_path=str(self.root / "state/expired.sqlite"),
            deadline_unix=1.0,
        )
        dead, _ = self.arena(
            expired, evaluate=lambda r: self.fail("expired game must not execute")
        )
        with self.assertRaises(RuntimeError):
            dead.run(max_contexts=1)

    def test_outbox_sidecar_and_report_temporary_cannot_escape_bench(self):
        bench = self.root / "bench"
        outside = self.root / "outside"
        outside.mkdir()
        untouched = outside / "proof"
        untouched.write_text("preserve")
        s = replace(
            self.spec,
            bench_root=str(bench),
            out_dir=str(bench / "runs/toy"),
            state_path=str(bench / "state/arena.sqlite"),
        )
        (bench / "state").mkdir(parents=True)
        sidecar = bench / "state/arena.sqlite-wal"
        sidecar.symlink_to(untouched)
        with self.assertRaises(ValueError):
            self.arena(s)
        self.assertEqual(untouched.read_text(), "preserve")
        # Remove only this freshly created, known fixture symlink.
        sidecar.unlink()
        arena, _ = self.arena(s)
        arena.run(max_contexts=1)
        (Path(s.out_dir) / "report.json.new").symlink_to(untouched)
        with self.assertRaises(ValueError):
            write_reports(s, arena.rows(), arena.hwm_values(), arena.closed_contexts())
        self.assertEqual(untouched.read_text(), "preserve")

    def test_program_boundaries_and_sampler_support_are_checked(self):
        with self.assertRaises(ValueError):
            BaselineArms([program("one", (3, 2))], {}, None)
        arena, _ = self.arena()
        arena.arms.sample = lambda c, a, s: {
            "program": self.p,
            "tokens": self.p["tokens"],
            "trie_sha256": "b" * 64,
            "checkpoint_parameter_sha256": SHA,
            "total_logprob": 0.0,
            "per_token_logprob": [0.0],
            "logprob_support": "trie_children",
            "logprob_aggregation": "sum_including_eos",
        }
        with self.assertRaises(ValueError):
            arena.run()

    def test_device_defaults_and_cuda_requires_complete_same_pin_parity(self):
        args = parser().parse_args(
            [
                "arena",
                "--spec",
                "spec.json",
                "--runtime-binding",
                "binding.py",
                "--policy-code-root",
                "pin",
                "--policy-qualification",
                "qualified.json",
            ]
        )
        self.assertEqual(args.device, "cpu")
        verify_device(self.spec, args.device)
        with self.assertRaises(ValueError):
            verify_device(self.spec, "cuda")
        receipt = self.root / "parity.json"
        payload = {
            "schema": "jope.rlfw-checkpoint-replay/v1",
            "passed": True,
            "checkpoint_parameter_sha256": SHA,
            "code_manifest_sha256": SHA,
            "trie_sha256": SHA,
            "normalizer_sha256": SHA,
            "tokenizer_sha256": SHA,
            "device_parity": {
                "executed": True,
                "passed": True,
                "rows": 16,
                "reference_device": "cpu",
                "candidate_device": "cuda",
                "tolerance": 1e-6,
                "maximum_absolute_errors": {
                    k: 1e-7
                    for k in (
                        "total_logprob",
                        "token_logprob",
                        "q",
                        "p_complete",
                        "runtime_log1p",
                    )
                },
            },
        }

        def frozen():
            receipt.write_text(json.dumps(payload))
            pin = hashlib.sha256(receipt.read_bytes()).hexdigest()
            return replace(
                self.spec,
                device="cuda",
                policy_code_manifest_sha256=SHA,
                device_parity_sha256=pin,
                runtime_files=(("parity.json", pin),),
            )

        verify_device(frozen(), "cuda", receipt)
        payload["device_parity"]["maximum_absolute_errors"]["q"] = 1e-5
        with self.assertRaises(ValueError):
            verify_device(frozen(), "cuda", receipt)
        payload["device_parity"]["maximum_absolute_errors"]["q"] = 1e-7
        payload["checkpoint_parameter_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            verify_device(frozen(), "cuda", receipt)
        payload["checkpoint_parameter_sha256"] = SHA
        payload["device_parity"]["executed"] = False
        with self.assertRaises(ValueError):
            verify_device(frozen(), "cuda", receipt)

    def test_reported_missing_process_cost_preserves_unknown_costs(self):
        arena, _ = self.arena(
            evaluate=lambda r: {**self.result(r, "infra_unknown"), "cost_unknown": True}
        )
        arena.run(max_contexts=1)
        rows = arena.rows()
        self.assertTrue(all(r.unknown_cost_attempts == 3 for r in rows))
        report = compute_report(
            self.spec, rows, arena.hwm_values(), arena.closed_contexts()
        )
        unit = report["per_context"][0]
        self.assertGreater(unit["unknown_cost_attempts"], 0)
        self.assertIsNone(unit["wall_seconds"])

    def test_reused_incumbent_checks_fresh_role_before_disclosure(self):
        parent, _ = self.arena()
        parent.run(max_contexts=1)
        write_reports(
            self.spec, parent.rows(), parent.hwm_values(), parent.closed_contexts()
        )
        child_spec = replace(
            self.spec,
            game_id="child",
            out_dir=str(self.root / "runs/child"),
            state_path=str(self.root / "state/child.sqlite"),
        )
        child, _ = self.arena(
            child_spec, check=lambda c: {"role": "train", "role_map_sha256": SHA}
        )
        with self.assertRaises(ValueError):
            reuse_incumbents(
                child_spec, self.spec.out_dir, child, child.arms, child.oracle
            )
        self.assertEqual(child.rows(), [])
        child.oracle.role_check = lambda c: {
            "role": "development",
            "role_map_sha256": SHA,
        }
        reuse_incumbents(child_spec, self.spec.out_dir, child, child.arms, child.oracle)
        self.assertEqual(len(child.rows()), 1)
        self.assertTrue(child.rows()[0].qualified and child.rows()[0].cache_hit)
        self.assertEqual(child.rows()[0].q_grid, 500_000)
        self.assertEqual(child.rows()[0].wall_seconds, 0)
        child.oracle.evaluate = lambda r: self.fail(
            "qualified reused program must hit cache"
        )
        child.run(max_contexts=1)
        self.assertEqual(len(child.rows()), 17)
        for arm in ("uniform_trie", "trie_sample:toy"):
            self.assertEqual(sum(r.charged for r in child.rows() if r.arm == arm), 8)


if __name__ == "__main__":
    unittest.main()
