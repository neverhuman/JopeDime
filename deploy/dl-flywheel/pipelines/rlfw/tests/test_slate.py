"""Budgeted reranking, complete raw custody and restart disclosure fixtures."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rlfw.arena.arms import Candidate, Program, pr1_trace_sink
from rlfw.arena.game_spec import ArmSpec, digest
from rlfw.arena.oracle import BlindOracle
from rlfw.arena.scheduler import ArenaScheduler
from rlfw.decision_trace import DecisionTrace
from rlfw.slate import PreparedSlate, RerankArms, select
from rlfw.tests.test_arena_kpi import SHA, spec_at


class SlateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = [
            {
                "dsl": f"leaf{i}",
                "program_sha256": hashlib.sha256(f"leaf{i}".encode()).hexdigest(),
                "tokens": [1, i + 3, 2],
            }
            for i in range(40)
        ]
        base = spec_at(self.root, 1)
        self.spec = replace(
            base,
            slate_k=32,
            cache_snapshot_sha256=digest({}),
            arms=(
                base.arms[0],
                ArmSpec("rerank32:toy", SHA, SHA, SHA, SHA),
            ),
            contexts=(
                replace(
                    base.contexts[0],
                    hwm_program_sha256=self.catalog[0]["program_sha256"],
                ),
            ),
        )
        self.context, self.arm = self.spec.contexts[0], self.spec.arms[1]
        self.draws = 0
        self.head_calls = 0

    def sample_many(self, cid, arm, seeds):
        rows = []
        for s in seeds:
            p = self.catalog[self.draws % len(self.catalog)]
            self.draws += 1
            rows.append(self.sample_row(p, s))
        return rows

    def sample_row(self, p, seed):
        return {
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

    def heads(self, cid, arm, token_rows):
        self.head_calls += 1
        return [
            {"q": float(t[1]), "p_complete": 0.8, "runtime_seconds": 1.0}
            for t in token_rows
        ]

    def arms(self, sample_many=None, heads=None):
        return RerankArms(
            self.catalog,
            {self.context.context_id: self.catalog[0]},
            lambda cid, arm, seed: self.sample_many(cid, arm, [seed])[0],
            sample_many or self.sample_many,
            heads or self.heads,
        )

    def prepare(self, **kwargs):
        return self.arms(**kwargs).prepare_slate(self.context, self.arm, 1)

    def test_fixed_slate_selection_is_pure_and_members_are_exact_leaves(self):
        packet = self.prepare()
        self.assertEqual(len(packet.raw_draws), 32)
        self.assertEqual(len(packet.slate), 32)
        self.assertEqual(len(packet.selected), 8)
        self.assertEqual(
            [c.predicted["q"] for c in packet.selected],
            list(map(float, range(34, 26, -1))),
        )
        self.assertEqual(select(packet.slate), select(tuple(reversed(packet.slate))))
        self.assertTrue(
            all(c.program in [s.program for s in packet.slate] for c in packet.selected)
        )
        self.assertTrue(all(c.propensity is None for c in packet.selected))
        self.assertTrue(all(c.propensity is not None for c in packet.raw_draws))
        self.assertEqual(PreparedSlate.from_dict(packet.to_dict()), packet)
        self.assertTrue(
            all(
                t >= 0
                for t in (
                    packet.decode_seconds,
                    packet.head_seconds,
                    packet.selection_seconds,
                )
            )
        )

    def test_sha_ties_and_feasibility_fallback(self):
        packet = self.prepare()
        tied = tuple(
            replace(c, predicted={"q": 1.0, "p_complete": 0.8, "runtime_seconds": 1.0})
            for c in packet.slate
        )
        self.assertEqual(
            [c.program.program_sha256 for c in select(tied)],
            sorted(c.program.program_sha256 for c in tied)[:8],
        )
        guarded = tuple(
            replace(c, predicted={**c.predicted, "p_complete": 0.5 if i < 8 else 0.49})
            for i, c in enumerate(packet.slate)
        )
        self.assertEqual(
            {c.program for c in select(guarded)}, {c.program for c in guarded[:8]}
        )
        fallback = tuple(
            replace(c, predicted={**c.predicted, "p_complete": 0.5 if i < 7 else 0.49})
            for i, c in enumerate(packet.slate)
        )
        self.assertEqual(
            [c.program for c in select(fallback)], [c.program for c in packet.selected]
        )

    def test_duplicates_shortfall_and_full_batch_overshoot_are_retained(self):
        def duplicate(cid, arm, seeds):
            rows = []
            for s in seeds:
                i = self.draws // 2
                self.draws += 1
                rows.append(self.sample_row(self.catalog[i % 40], s))
            return rows

        packet = self.prepare(sample_many=duplicate)
        self.assertEqual(len(packet.raw_draws), 64)
        self.assertEqual(len(packet.slate), 32)
        self.draws = 0

        def short(cid, arm, seeds):
            rows = []
            for s in seeds:
                rows.append(self.sample_row(self.catalog[self.draws % 10], s))
                self.draws += 1
            return rows

        packet = self.prepare(sample_many=short)
        self.assertEqual(len(packet.raw_draws), 512)
        self.assertEqual(packet.to_dict()["shortfall"], 22)
        self.assertEqual(len(packet.selected), 8)
        self.draws = 0

        def overshoot(cid, arm, seeds):
            rows = []
            for s in seeds:
                index = max(0, self.draws - 1) % 40
                self.draws += 1
                rows.append(self.sample_row(self.catalog[index], s))
            return rows

        packet = self.prepare(sample_many=overshoot)
        self.assertEqual(len(packet.raw_draws), 64)
        self.assertEqual(len(packet.slate), 32)
        self.assertTrue(
            any(
                c.program not in {s.program for s in packet.slate}
                for c in packet.raw_draws
            )
        )

    def test_invalid_catalog_binding_support_seed_and_head_fields_refused(self):
        for key, value in (
            ("trie_sha256", "b" * 64),
            ("checkpoint_parameter_sha256", "b" * 64),
            ("decoder_seed", -1),
            ("logprob_support", "grammar"),
            ("logprob_aggregation", "mean_nll"),
            ("tokens", [1, 999, 2]),
        ):

            def bad(cid, arm, seeds, key=key, value=value):
                return [
                    {**self.sample_row(self.catalog[0], s), key: value} for s in seeds
                ]

            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(sample_many=bad)
        for p in (
            {"q": float("nan"), "p_complete": 0.8, "runtime_seconds": 1},
            {"q": 1, "p_complete": 1.1, "runtime_seconds": 1},
            {"q": 1, "p_complete": 0.8, "runtime_seconds": -1},
            {"q": 1, "p_complete": 0.8, "runtime_seconds": 1, "q_grid": 123},
        ):
            with self.subTest(predicted=p), self.assertRaises(ValueError):
                self.prepare(heads=lambda cid, arm, rows, p=p: [p for _ in rows])
        with self.assertRaises(ValueError):
            self.prepare(sample_many=lambda cid, arm, seeds: [])
        with self.assertRaises(ValueError):
            self.prepare(heads=lambda cid, arm, rows: [])

    def test_packet_tampering_and_wrong_k_refused(self):
        value = self.prepare().to_dict()
        value["selected"] = list(reversed(value["selected"]))
        with self.assertRaises(ValueError):
            PreparedSlate.from_dict(value)
        with self.assertRaises(ValueError):
            replace(self.spec, slate_k=8)
        with self.assertRaises(ValueError):
            self.arms().prepare_slate(
                self.context, replace(self.arm, temperature=0.7), 1
            )

    def test_actual_v2_trace_seals_all_raws_and_only_predicted_head_fields(self):
        trace = DecisionTrace(self.spec.out_dir, runs_root=self.root / "runs")
        scheduler = self.scheduler(sink=lambda value: pr1_trace_sink(trace, value))
        self.addCleanup(scheduler.close)
        scheduler.run()
        trace.seal()
        rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(r["kind"] == "raw_draw" for r in rows), 32)
        self.assertTrue(
            all(
                set(r["predicted"]) == {"q", "p_complete", "runtime_seconds"}
                for r in rows
            )
        )
        self.assertTrue(all("q" not in r and "q_grid" not in r for r in rows))

    def evaluator(self, request):
        self.assertNotIn("predicted", request["program"])
        self.assertNotIn("q", request["program"])
        return {
            "outcome_class": "completed",
            "qualified": True,
            "q_grid": 250_000,
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
            "wall_seconds": 1.0,
            "cpu_seconds": 1.0,
        }

    def scheduler(self, arms=None, evaluator=None, sink=None):
        oracle = BlindOracle(
            self.spec,
            evaluator or self.evaluator,
            lambda c: {"role": "development", "role_map_sha256": SHA},
            {},
        )
        return ArenaScheduler(
            self.spec,
            arms or self.arms(),
            oracle,
            sink or (lambda value: None),
            workers=1,
        )

    def test_durable_packet_raw_trace_before_evaluation_and_fair_eight_charges(self):
        trace = []

        def evaluate(request):
            self.assertEqual(sum(t["kind"] == "raw_draw" for t in trace), 32)
            self.assertEqual(
                scheduler.db.execute("SELECT count(*) FROM prepared_slates").fetchone()[
                    0
                ],
                1,
            )
            self.assertEqual(
                len(list((self.root / "runs/toy/slates").glob("*.json"))), 1
            )
            return self.evaluator(request)

        scheduler = self.scheduler(evaluator=evaluate, sink=trace.append)
        self.addCleanup(scheduler.close)
        rows = scheduler.run()
        chosen = [r for r in rows if r.arm == self.arm.name]
        self.assertEqual(len(chosen), 8)
        self.assertEqual(sum(r.charged for r in chosen), 8)
        self.assertTrue(
            all(r.q_grid == 250_000 and "predicted" not in r.to_dict() for r in chosen)
        )
        selected_trace = [t for t in trace if t["kind"] == "selected_slot"]
        self.assertEqual(len(selected_trace), 8)
        self.assertTrue(
            all(
                t["predicted"]["q"] > 0.25 and t["propensity"] is None
                for t in selected_trace
            )
        )
        self.assertGreaterEqual(scheduler.decode_costs()[1]["wall_seconds"], 0)
        with self.assertRaises(ValueError):
            Program.from_dict({**self.catalog[0], "predicted": {"q": 1}})
        with self.assertRaises(TypeError):
            Candidate.from_dict({**self.prepare().selected[0].to_dict(), "q_grid": 1})

    def test_restart_reuses_precommitted_raw_packet_without_decode_or_heads(self):
        first_trace = []
        scheduler = self.scheduler(sink=first_trace.append)
        selected = scheduler.slate(self.context, self.arm, 1)
        scheduler.close()
        second_trace = []
        arms = self.arms(
            sample_many=lambda *a: self.fail("restart must not decode"),
            heads=lambda *a: self.fail("restart must not score heads"),
        )
        restarted = self.scheduler(arms=arms, sink=second_trace.append)
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.slate(self.context, self.arm, 1), selected)
        self.assertEqual(first_trace, second_trace)
        rows = restarted.run()
        self.assertEqual(len([r for r in rows if r.arm == self.arm.name]), 8)

    def test_insufficient_distinct_support_preserves_raws_and_authorizes_no_evaluation(
        self,
    ):
        def same(cid, arm, seeds):
            return [self.sample_row(self.catalog[0], s) for s in seeds]

        trace = []
        scheduler = self.scheduler(
            arms=self.arms(sample_many=same),
            evaluator=lambda r: self.fail("shortfall must not evaluate"),
            sink=trace.append,
        )
        self.addCleanup(scheduler.close)
        with self.assertRaises(ValueError):
            scheduler.run()
        self.assertEqual(len(trace), 512)
        self.assertEqual(scheduler.rows(), [])
        value = json.loads(
            scheduler.db.execute("SELECT payload FROM prepared_slates").fetchone()[0]
        )
        self.assertEqual(value["shortfall"], 31)
        self.assertEqual(value["selected"], [])
        self.assertEqual(len(list((self.root / "runs/toy/slates").glob("*.json"))), 1)
        with self.assertRaises(ValueError):
            scheduler.run()
        self.assertEqual(self.head_calls, 1)

    def test_baseline_pairing_retains_roster_seed_budget_cache_and_runtime(self):
        baseline = replace(self.spec, arms=spec_at(self.root).arms, slate_k=8)
        self.spec.require_pair(baseline)
        for changed in (
            replace(baseline, seeds=(2,)),
            replace(baseline, cache_snapshot_sha256="b" * 64),
            replace(baseline, contexts=(replace(self.context, lineage_id="other"),)),
        ):
            with self.assertRaises(ValueError):
                self.spec.require_pair(changed)


if __name__ == "__main__":
    unittest.main()
