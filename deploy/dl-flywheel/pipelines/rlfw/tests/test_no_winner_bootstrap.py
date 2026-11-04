"""Train-only bootstrap fixtures with synthetic closures and no model imports."""

import ast
import contextlib
import hashlib
import json
import re
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rlfw.arena.arms import Program
from rlfw.arena.game_spec import ArmSpec, digest
from rlfw.no_winner_bootstrap import (
    BootstrapScheduler,
    BootstrapSpec,
    CampaignWindow,
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
# Synthetic fixture window: every test that reaches it sets its wall clock explicitly.
WINDOW_CREATED = 1791536400.0
WINDOW_MONOTONIC = 1000.0
# Each change closes the window, or makes it unverifiable, one second before its deadline.
CLOCK_CHANGES = {
    "wall deadline": lambda f, clock: setattr(
        clock, "return_value", f.window.deadline_unix
    ),
    "monotonic expiry": lambda f, clock: setattr(f, "skew", 2.0),
    "other boot": lambda f, clock: setattr(f, "boot", "other-boot"),
}

class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.window = CampaignWindow(
            WINDOW_CREATED,
            WINDOW_CREATED + 72 * 3600,
            72,
            "fixture-boot",
            WINDOW_MONOTONIC,
        )
        self.boot, self.skew = "fixture-boot", 0.0
        clock = mock.patch(
            "rlfw.no_winner_bootstrap.campaign_clock", side_effect=self.clock
        )
        clock.start()
        self.addCleanup(clock.stop)
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
            self.window,
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

    def clock(self):
        # The window never reads the real clock in fixtures; unmocked use fails.
        if not isinstance(time.time, mock.Mock):
            self.fail("bootstrap fixtures require an explicit wall clock")
        now = time.time()
        return {
            "unix": now,
            "monotonic": WINDOW_MONOTONIC + now - WINDOW_CREATED + self.skew,
            "boot_id": self.boot,
        }

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

    def fresh(self):
        fixture = BootstrapTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def attempt_boundary(self, f, change):
        # Review PR291-01: the clock changes after the first request of one slot.
        with mock.patch(
            "rlfw.arena.oracle.time.time", return_value=f.window.deadline_unix - 1
        ) as clock:

            def evaluate(request):
                result = f.result(request, "infra_unknown")
                CLOCK_CHANGES[change](f, clock)
                return result

            value = f.scheduler(evaluate=evaluate)
            with self.assertRaises(RuntimeError):
                value.run(max_contexts=1)
            report = value.coverage_report()
        self.assertEqual([c["attempt"] for c in f.calls], [1])
        attempts = value.db.execute("SELECT number,result FROM attempts").fetchall()
        # The finished unknown attempt keeps its outcome and cost; nothing new is reserved.
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0][0], 1)
        self.assertEqual(json.loads(attempts[0][1])["outcome_class"], "infra_unknown")
        self.assertEqual(report["observed_wall_seconds"], 2.0)
        self.assertEqual(value.rows(), [])
        self.assertTrue(all(not value.ledger.events(c.hwm_key) for c in f.contexts))

    def reservation_boundary(self, f, change):
        # Review of 939b294c: the clock changes inside the durable SQLite reservation.
        with mock.patch(
            "rlfw.arena.oracle.time.time", return_value=f.window.deadline_unix - 1
        ) as clock:
            value = f.scheduler(evaluate=lambda r: f.result(r, "infra_unknown"))
            reserve = value.reserve_attempt

            def reserve_across_boundary(key):
                number = reserve(key)
                CLOCK_CHANGES[change](f, clock)
                return number

            value.reserve_attempt = reserve_across_boundary
            with self.assertRaises(RuntimeError):
                value.run(max_contexts=1)
            report = value.coverage_report()
        # The consumed reservation stays an unavailable outcome; nothing was requested.
        attempts = value.db.execute("SELECT number,result FROM attempts").fetchall()
        self.assertEqual(attempts, [(1, None)])
        self.assertEqual((f.calls, value.rows()), ([], []))
        self.assertEqual(report["unknown_cost_attempts"], 1)
        self.assertIsNone(report["evaluator_wall_seconds"])
        self.assertTrue(all(not value.ledger.events(c.hwm_key) for c in f.contexts))

    def admission_boundary(self, f, change):
        # Review PR291-02: admission returns a valid grant after the window closed.
        with mock.patch(
            "rlfw.arena.oracle.time.time", return_value=f.window.deadline_unix - 1
        ) as clock:
            value = f.scheduler()

            @contextlib.contextmanager
            def admission():
                CLOCK_CHANGES[change](f, clock)
                yield True

            value.admission = admission
            with self.assertRaises(RuntimeError):
                value.run(max_contexts=1)
            report = value.coverage_report()
        self.assertEqual((f.draws, f.calls), (0, []))
        self.assertEqual(report["contexts_attempted"], 0)

    def dispatch_boundary(self, f, change):
        # The clock changes during one model dispatch; the next dispatch never starts.
        with mock.patch(
            "rlfw.arena.oracle.time.time", return_value=f.window.deadline_unix - 1
        ) as clock:
            decode, heads, readouts = f.sample_many, f.heads, []

            def sample_many(cid, arm, seeds):
                rows = decode(cid, arm, seeds)
                CLOCK_CHANGES[change](f, clock)
                return rows

            def score_heads(cid, arm, rows):
                readouts.append(rows)
                return heads(cid, arm, rows)

            f.sample_many, f.heads = sample_many, score_heads
            value = f.scheduler()
            with self.assertRaises(RuntimeError):
                value.run(max_contexts=1)
        self.assertEqual((f.draws, readouts, f.calls), (32, [], []))
        self.assertEqual(
            value.db.execute("SELECT count(*) FROM prepared_slates").fetchone()[0], 0
        )

    def parity_result(self, request, error=1e-5):
        return {
            **self.result(request, "parity_unqualified"),
            "parity_verified": False,
            "parity_max_normalized_error": error,
        }

from ._bootstrap_window_cases import TEST_METHODS as _WINDOW_TESTS
from ._bootstrap_coverage_cases import TEST_METHODS as _COVERAGE_TESTS
from ._bootstrap_cli_cases import TEST_METHODS as _CLI_TESTS
from . import _bootstrap_window_cases as _window_cases
from . import _bootstrap_coverage_cases as _coverage_cases
from . import _bootstrap_cli_cases as _cli_cases

for _case_module, _case_names in (
    (_window_cases, _WINDOW_TESTS),
    (_coverage_cases, _COVERAGE_TESTS),
    (_cli_cases, _CLI_TESTS),
):
    for _case_name in _case_names:
        setattr(BootstrapTests, _case_name, getattr(_case_module, _case_name))

if __name__ == "__main__":
    unittest.main()
