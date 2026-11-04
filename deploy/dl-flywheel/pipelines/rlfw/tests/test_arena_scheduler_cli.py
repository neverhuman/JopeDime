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

class SchedulerCliTests(_SchedulerSupport):
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
