"""Train-only bootstrap fixtures with synthetic closures and no model imports."""

import contextlib
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rlfw.arena.arms import Program
from rlfw.arena.game_spec import ArmSpec, digest
from rlfw.no_winner_bootstrap import (
    BootstrapScheduler,
    BootstrapSpec,
    CoverageArms,
    CoverageSlate,
    TrainingContext,
    TrainingOracle,
    check_training_context,
    coverage_selection,
    main,
    parser,
    select_wave,
)
from rlfw.research_hwm import ARTIFACT_ROLES, HWMKey, ResearchHWM

SHA = "a" * 64
RUNTIME_FIXTURES = {}


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contexts = tuple(self.context(i) for i in range(3))
        self.arm = ArmSpec("rerank_mix:toy", SHA, SHA, SHA, SHA)
        for name in ("binding.py", "calibration.json"):
            (self.root / name).write_text(name)
        pins = tuple(
            (name, hashlib.sha256((self.root / name).read_bytes()).hexdigest())
            for name in ("binding.py", "calibration.json")
        )
        self.spec = BootstrapSpec(
            "wave",
            str(self.root),
            str(self.root / "runs/wave"),
            str(self.root / "state/wave.sqlite"),
            self.contexts,
            (self.arm,),
            SHA,
            SHA,
            SHA,
            pins[1][1],
            pins[0][1],
            SHA,
            SHA,
            pins,
            1791608400.0,
            previously_labeled_parents=("parent0",),
        )
        self.catalog = [
            {
                "dsl": f"leaf{i}",
                "program_sha256": hashlib.sha256(f"leaf{i}".encode()).hexdigest(),
                "tokens": [1, i + 3, 2],
            }
            for i in range(40)
        ]
        self.draws = 0
        self.calls = []
        self.artifacts = {}
        dest = self.root / "runs/fixtures"
        dest.mkdir(parents=True)
        for role in ARTIFACT_ROLES:
            p = dest / (role + ".json")
            p.write_text(role)
            self.artifacts[role] = {
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }

    def context(self, i):
        identity = digest(["original", i])
        key = HWMKey(
            identity,
            "dataset",
            "cut",
            ("f0", "f1"),
            SHA,
            SHA,
            SHA,
            "q",
            "maximize",
            "mean-epsilon",
            "jope.flywheel-s1-inner-rmse/v1",
            SHA,
            SHA,
        )
        return TrainingContext(
            f"native-c{i}",
            f"native-view{i}",
            "dataset",
            "cut",
            0,
            f"parent{i}",
            identity,
            SHA,
            SHA,
            key,
            "g0_not_ready" if i % 2 else "no_published_hwm",
        )

    def role(self, cid):
        return next(c for c in self.contexts if c.context_id == cid).authority_value(
            SHA
        )

    def sample_many(self, cid, arm, seeds):
        rows = []
        for seed in seeds:
            p = self.catalog[self.draws % 40]
            self.draws += 1
            rows.append(
                {
                    "program": p,
                    "tokens": p["tokens"],
                    "decoder_seed": seed,
                    "trie_sha256": SHA,
                    "checkpoint_parameter_sha256": SHA,
                    "total_logprob": -2.0,
                    "per_token_logprob": [-2.0, 0.0],
                    "logprob_support": "trie_children",
                    "logprob_aggregation": "sum_including_eos",
                    "batch_protocol_sha256": SHA,
                }
            )
        return rows

    def heads(self, cid, arm, rows):
        return [{"q": -1.0, "p_complete": 1.0, "runtime_seconds": 1.0} for _ in rows]

    def arms(self, probabilities=None):
        return CoverageArms(
            self.catalog,
            lambda cid, arm, s: self.sample_many(cid, arm, [s])[0],
            self.sample_many,
            self.heads,
            probabilities
            or (lambda cid, arm, rows, pin: [(t[1] - 3) / 40 for t in rows]),
            self.spec.calibration_sha256,
        )

    def result(self, request, outcome="completed", q_grid=None):
        self.calls.append(request)
        self.assertNotIn("predicted", request["program"])
        result = {
            "outcome_class": outcome,
            "qualified": outcome == "completed",
            "q_grid": q_grid if outcome == "completed" else None,
            "context_identity": request["context_identity"],
            "program_sha256": request["program"]["program_sha256"],
            "evaluator_sha256": SHA,
            "canonicalizer_sha256": SHA,
            "protocol": self.spec.protocol,
            "role_map_sha256": SHA,
            "closure_verified": True,
            "parity_verified": True,
            "parity_max_normalized_error": 0.0,
            "receipt_sha256": SHA,
            "wall_seconds": 2.0,
            "cpu_seconds": 3.0,
            "artifacts": self.artifacts,
        }
        if result["q_grid"] is None and outcome == "completed":
            result["q_grid"] = int(request["program"]["dsl"][4:]) * 10000
        return result

    def scheduler(self, evaluate=None, role=None, granted=True):
        @contextlib.contextmanager
        def admission():
            yield granted

        ledger = ResearchHWM(
            self.root / "state/ledger.sqlite", state_root=self.root / "state"
        )
        oracle = TrainingOracle(
            self.spec, evaluate or self.result, role or self.role, {}
        )
        value = BootstrapScheduler(
            self.spec,
            self.arms(),
            oracle,
            lambda r: None,
            ledger,
            admission,
            lambda: granted,
            workers=1,
        )
        self.addCleanup(value.close)
        return value

    def test_protected_roles_status_and_missing_closure_hard_refused(self):
        c = self.contexts[0]
        for change in (
            {"role": "development"},
            {"role": "sealed"},
            {"rights_pending": True},
            {"excluded": True},
            {"sealed": True},
            {"vector_sha256": ""},
            {"prepared_input_sha256": ""},
            {"winner_status": "usable_hwm"},
            {"fold": False},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(c, **change)
        with self.assertRaises(ValueError):
            replace(c, identity_sha256="b" * 64)
        for key, value in (
            ("role", "development"),
            ("inputs_verified", False),
            ("vector_verified", False),
            ("excluded", True),
            ("rights_pending", True),
            ("context_id", "other"),
            ("view_id", "other"),
            ("fold", 1),
            ("fold", False),
            ("inputs_verified", 1),
            ("excluded", 0),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_training_context(
                    c,
                    lambda cid, key=key, value=value: {**self.role(cid), key: value},
                    SHA,
                )

    def test_seeded_wave_one_original_parent_and_exact_native_identity(self):
        contexts = self.contexts + (
            replace(self.contexts[0], context_id="sibling", view_id="sibling-view"),
        )
        role = lambda cid: next(
            c for c in contexts if c.context_id == cid
        ).authority_value(SHA)
        wave = select_wave(contexts, role, SHA)
        self.assertEqual(len(wave), 3)
        self.assertEqual(wave, select_wave(tuple(reversed(contexts)), role, SHA))
        self.assertTrue(all(c in contexts for c in wave))
        self.assertTrue(all(c.fold == 0 for c in wave))
        self.assertEqual(BootstrapSpec.from_dict(self.spec.to_dict()), self.spec)

    def test_d5_fixed_deadline_and_expired_execution_refused(self):
        for deadline in (1791550800.0, 1791568800.0, 1791608399.0, 1791608401.0):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                replace(self.spec, deadline_unix=deadline)
        for now in (1791608400.0, 1791608401.0):
            with (
                self.subTest(now=now),
                mock.patch("rlfw.arena.oracle.time.time", return_value=now),
            ):
                value = self.scheduler()
                with self.assertRaisesRegex(RuntimeError, "deadline"):
                    value.run(max_contexts=1)
                self.assertEqual(self.calls, [])
                self.assertEqual(value.rows(), [])
                value.close()

    def test_coverage_probabilities_drive_five_exploit_two_explore_one_audit(self):
        packet = self.arms().prepare_slate(self.contexts[0], self.arm, 20261009)
        self.assertEqual(len(packet.selected), 8)
        self.assertEqual(
            [c.program.dsl for c in packet.selected[:5]],
            ["leaf31", "leaf30", "leaf29", "leaf28", "leaf27"],
        )
        self.assertTrue(all(c.predicted["q"] == -1.0 for c in packet.slate))
        self.assertTrue(all(c.propensity is None for c in packet.selected[:5]))
        self.assertEqual([c.propensity for c in packet.selected[5:7]], [1 / 27, 1 / 26])
        self.assertEqual(packet.selected[-1].propensity, 1 / 40)
        self.assertEqual(
            packet.selected,
            coverage_selection(
                packet.slate,
                packet.predicted_probabilities,
                packet.raw_draws[-1],
                packet.selection_seed,
            ),
        )
        self.assertEqual(CoverageSlate.from_dict(packet.to_dict()), packet)
        self.assertEqual(len(packet.raw_draws), 33)

    def test_unknown_probability_and_packet_changes_refused(self):
        for bad in (float("nan"), -0.1, 1.1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.arms(
                    probabilities=lambda cid, arm, rows, pin, bad=bad: [
                        bad for _ in rows
                    ]
                ).prepare_slate(self.contexts[0], self.arm, 20261009)
        with self.assertRaises(ValueError):
            self.arms(probabilities=lambda *a: []).prepare_slate(
                self.contexts[0], self.arm, 20261009
            )
        packet = (
            self.arms().prepare_slate(self.contexts[0], self.arm, 20261009).to_dict()
        )
        packet["selected"] = list(reversed(packet["selected"]))
        with self.assertRaises(ValueError):
            CoverageSlate.from_dict(packet)

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
    def test_busy_arena_blocks_all_proposals_and_evaluations(self, clock):
        value = self.scheduler(granted=False)
        with self.assertRaises(RuntimeError):
            value.run()
        self.assertEqual(self.draws, 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(value.coverage_report()["contexts_attempted"], 0)

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
    def test_first_labels_measured_only_and_exact_partial_yield_costs(self, clock):
        value = self.scheduler()
        rows = value.run(max_contexts=1)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r.q_grid is not None and r.q_grid >= 0 for r in rows))
        self.assertTrue(all(r.charged for r in rows))
        report = value.coverage_report()
        self.assertEqual(report["contexts_attempted"], 1)
        self.assertEqual(report["first_label_yield"], 1)
        self.assertEqual(report["exit_usable_yield"], 1)
        self.assertEqual(len(report["unattempted_contexts_unknown"]), 2)
        self.assertEqual(report["evaluator_wall_seconds"], 2 * len(self.calls))
        self.assertEqual(report["evaluator_cpu_seconds"], 3 * len(self.calls))
        c = next(c for c in self.contexts if c.context_id == rows[0].context_id)
        events = value.ledger.events(c.hwm_key)
        self.assertEqual(events[0]["event"], "first_label")
        self.assertTrue(all(e["publication"]["source"] == "measured" for e in events))
        self.assertEqual(
            [e["publication"]["q_grid"] for e in events], [r.q_grid for r in rows]
        )

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
    def test_g0_and_program_failure_never_become_zero_q(self, clock):
        value = self.scheduler(evaluate=lambda r: self.result(r, "program_failed"))
        rows = value.run(max_contexts=1)
        self.assertTrue(all(r.q_grid is None and r.q is None for r in rows))
        report = value.coverage_report()
        self.assertEqual(report["first_label_yield"], 0)
        self.assertEqual(report["exit_usable_yield"], 0)
        c = next(c for c in self.contexts if c.context_id == rows[0].context_id)
        self.assertTrue(
            all(
                e["publication"]["q_grid"] is None
                for e in value.ledger.events(c.hwm_key)
            )
        )

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
    def test_exhausted_censored_attempts_publish_refusals_and_charge_slots(self, clock):
        for outcome in ("parity_custody", "infra_unknown"):
            with self.subTest(outcome=outcome):
                fixture = BootstrapTests("runTest")
                fixture.setUp()
                try:

                    def evaluate(request):
                        return {
                            **fixture.result(request, outcome),
                            "cost_unknown": outcome == "infra_unknown",
                        }

                    value = fixture.scheduler(evaluate=evaluate)
                    rows = value.run(max_contexts=1)
                    self.assertEqual(len(rows), 8)
                    self.assertEqual(len(fixture.calls), 24)
                    self.assertEqual(sum(row.attempts for row in rows), 24)
                    self.assertTrue(all(row.outcome == "censored" for row in rows))
                    self.assertTrue(all(row.censor_reason == outcome for row in rows))
                    self.assertTrue(
                        all(row.q_grid is None and row.q is None for row in rows)
                    )
                    events = [
                        event
                        for context in fixture.contexts
                        for event in value.ledger.events(context.hwm_key)
                    ]
                    self.assertEqual(len(events), 8)
                    self.assertTrue(
                        all(
                            event["event"] == "refused"
                            and event["reason"] == outcome
                            and event["publication"]["q_grid"] is None
                            for event in events
                        )
                    )
                    report = value.coverage_report()
                    self.assertEqual(report["contexts_attempted"], 1)
                    self.assertEqual(report["contexts_with_valid_program"], 0)
                    self.assertEqual(report["contexts_exit_usable"], 0)
                    self.assertEqual(report["censored_logical_slots"], 8)
                    self.assertEqual(len(report["closed_contexts"]), 1)
                    if outcome == "infra_unknown":
                        self.assertEqual(report["unknown_cost_attempts"], 24)
                        self.assertIsNone(report["evaluator_wall_seconds"])
                        self.assertIsNone(report["evaluator_cpu_seconds"])
                    else:
                        self.assertEqual(report["unknown_cost_attempts"], 0)
                        self.assertEqual(report["evaluator_wall_seconds"], 48)
                        self.assertEqual(report["evaluator_cpu_seconds"], 72)
                finally:
                    fixture.doCleanups()

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
    def test_role_change_after_execution_and_missing_artifacts_cannot_publish(
        self, clock
    ):
        changed = False

        def role(cid):
            return (
                {**self.role(cid), "role": "development"} if changed else self.role(cid)
            )

        def evaluate(request):
            nonlocal changed
            result = self.result(request)
            changed = True
            return result

        value = self.scheduler(evaluate=evaluate, role=role)
        with self.assertRaises(ValueError):
            value.run(max_contexts=1)
        self.assertEqual(value.rows(), [])
        self.assertTrue(all(not value.ledger.events(c.hwm_key) for c in self.contexts))
        self.assertGreater(value.coverage_report()["unknown_cost_attempts"], 0)
        self.assertIsNone(value.coverage_report()["evaluator_wall_seconds"])

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
    def test_numeric_closure_refusal_stays_null_for_all_slots(self, clock):
        self.artifacts = {}
        value = self.scheduler()
        with self.assertRaises(ValueError):
            value.run(max_contexts=1)
        self.assertEqual(value.rows(), [])
        events = [e for c in self.contexts for e in value.ledger.events(c.hwm_key)]
        self.assertEqual(len(events), 8)
        self.assertTrue(all(e["reason"] == "missing_closure" for e in events))
        self.assertTrue(all(e["publication"]["q_grid"] is None for e in events))

    def test_mixed_packet_restart_reuses_every_raw_draw_without_inference(self):
        first = self.scheduler()
        selected = first.slate(self.contexts[0], self.arm, 20261009)
        first.close()
        resumed = self.scheduler()
        resumed.arms.sample_many = lambda *args: self.fail("restart must not decode")
        resumed.arms.score_heads = lambda *args: self.fail(
            "restart must not score heads"
        )
        resumed.arms.probabilities = lambda *args: self.fail(
            "restart must not recalibrate"
        )
        self.assertEqual(resumed.slate(self.contexts[0], self.arm, 20261009), selected)

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
    def test_usable_yield_strict_gap_boundary(self, clock):
        values = [0, 10000]

        def evaluate(request):
            return self.result(request, q_grid=values[len(self.calls) % 2])

        value = self.scheduler(evaluate=evaluate)
        value.run(max_contexts=1)
        report = value.coverage_report()
        self.assertEqual(report["contexts_with_valid_program"], 1)
        self.assertEqual(report["contexts_exit_usable"], 0)
        self.assertEqual(
            report["new_parent_families"],
            sorted(
                {value.context_by_id[c].lineage_id for c in value.closed_contexts()}
                - {"parent0"}
            ),
        )

    def test_cli_defaults_off_cpu_and_refuses_before_runtime_import(self):
        args = [
            "--spec",
            "missing.json",
            "--runtime-binding",
            "missing.py",
            "--policy-code-root",
            "missing",
            "--policy-qualification",
            "missing.json",
        ]
        self.assertEqual(parser().parse_args(args).device, "cpu")
        with self.assertRaises(ValueError):
            main(args)
        with self.assertRaises(ValueError):
            Program.from_dict(
                {
                    "dsl": "x",
                    "tokens": [1, 2],
                    "program_sha256": hashlib.sha256(b"x").hexdigest(),
                    "q_grid": 0,
                }
            )

    @mock.patch("rlfw.arena.oracle.time.time", return_value=1791572400.0)
    @mock.patch("rlfw.policy_binding.verify_manifest")
    @mock.patch("rlfw.bench_cli.verify_runtime_scope")
    def test_enabled_cli_executes_synthetic_authorized_wave_and_reports(
        self, scope, manifest, clock
    ):
        binding = self.root / "binding.py"
        binding.write_text(
            "from rlfw.tests.test_no_winner_bootstrap import RUNTIME_FIXTURES\n"
            "def build_bootstrap(spec, *, device='cpu'):\n"
            "    return RUNTIME_FIXTURES[spec.game_id]\n"
        )
        calibration = self.root / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "schema": "jope.rlfw-q-positive-calibration/v1",
                    "passed": True,
                    "partition": "parent-held-train",
                    "checkpoint_parameter_sha256": SHA,
                    "trie_sha256": SHA,
                    "normalizer_sha256": SHA,
                    "tokenizer_sha256": SHA,
                }
            )
        )
        qualification = self.root / "qualification.json"
        qualification.write_text(
            json.dumps(
                {
                    "passed": True,
                    "draws": 1000,
                    "determinism_passed": True,
                    "maximum_logprob_error": 0.0,
                    "checkpoint_parameter_sha256s": [SHA],
                }
            )
        )
        pin = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        spec = replace(
            self.spec,
            runtime_binding_sha256=pin(binding),
            calibration_sha256=pin(calibration),
            policy_qualification_sha256=pin(qualification),
            runtime_files=tuple(
                (p.name, pin(p)) for p in (binding, calibration, qualification)
            ),
        )
        spec_path = self.root / "bootstrap.json"
        spec_path.write_text(json.dumps(spec.to_dict()))
        code_root = self.root / "code_pin"
        code_root.mkdir()
        catalog = [
            {
                "dsl": f"leaf{i}",
                "program_sha256": hashlib.sha256(f"leaf{i}".encode()).hexdigest(),
                "tokens": [1, i + 3, 2],
            }
            for i in range(6640)
        ]

        @contextlib.contextmanager
        def admission():
            yield True

        RUNTIME_FIXTURES[spec.game_id] = {
            "catalog": catalog,
            "sample": lambda *a: None,
            "sample_many": self.sample_many,
            "score_heads": self.heads,
            "coverage_probabilities": lambda cid, arm, rows, pin: [0.7 for _ in rows],
            "evaluate": self.result,
            "role_check": self.role,
            "reserve_empty_arena": admission,
            "arena_queue_empty": lambda: True,
        }
        self.addCleanup(RUNTIME_FIXTURES.pop, spec.game_id, None)
        args = [
            "--enable-no-winner-bootstrap",
            "--spec",
            str(spec_path),
            "--runtime-binding",
            str(binding),
            "--policy-code-root",
            str(code_root),
            "--policy-qualification",
            str(qualification),
            "--max-contexts",
            "1",
            "--pool-workers",
            "1",
        ]
        with mock.patch("builtins.print"):
            self.assertEqual(main(args), 0)
        report = json.loads((Path(spec.out_dir) / "coverage-report.json").read_text())
        self.assertEqual(report["contexts_attempted"], 1)
        self.assertEqual(report["contexts_with_valid_program"], 1)
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(report["teacher_admissions"], 0)
        self.assertTrue((Path(spec.out_dir) / "decisions.jsonl").exists())
        scope.assert_called_once()
        manifest.assert_called_once()


if __name__ == "__main__":
    unittest.main()
