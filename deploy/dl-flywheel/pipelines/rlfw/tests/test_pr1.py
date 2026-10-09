import hashlib
import json
import math
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from rlfw.decision_trace import DecisionRecord, DecisionTrace
from rlfw.hwm_adapter import original_hwm_program
from rlfw.research_hwm import ARTIFACT_ROLES, HWMKey, Publication, ResearchHWM
from rlfw.trie_logprob import TrieLogProb, child_logprob, trie_program_logprob

PIN = "a" * 64


class ToyTrie:
    programs: ClassVar[dict] = {(1, 3, 2): {}, (1, 4, 2): {}, (1, 4, 5, 2): {}}

    def get(self, prefix):
        return sorted(
            {
                p[len(prefix)]
                for p in self.programs
                if p[: len(prefix)] == tuple(prefix) and len(p) > len(prefix)
            }
        )


class ToyPolicy:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, value):
        self.training = value

    def next_logits(self, ctx, prefix):
        assert not self.training, "dropout must be off"
        return {2: -0.7, 3: 0.8, 4: -1.2, 5: 0.3, 6: 4.0}


def key():
    return HWMKey(
        "native-context",
        "dataset-version",
        "cut",
        ("f0", "f1"),
        PIN,
        PIN,
        PIN,
        "rmse",
        "lower",
        "fold-mean; epsilon=1e-12",
        "two-fold/v1",
        PIN,
        PIN,
    )


def publication(q=100):
    return Publication(
        key().digest,
        PIN,
        "execution",
        "qualified_complete",
        q,
        {
            role: {
                "sha256": PIN,
                "version_id": "exact-version",
                "uri": "research/" + role,
            }
            for role in ARTIFACT_ROLES
        },
    )


def decision():
    return DecisionRecord(
        "draw-1",
        "raw_draw",
        "native-context",
        PIN,
        PIN,
        PIN,
        PIN,
        1.0,
        7,
        (1, 3, 2),
        TrieLogProb(-0.5, (-0.5, 0.0)),
        "program-sha",
        "slate-1",
        "iid",
        predicted={"q": 2.0},
    )


class TrieTests(unittest.TestCase):
    def test_enumeration_mass_and_bruteforce_per_token(self):
        for temperature in (1.0, 0.7):
            mass = 0.0
            for tokens in ToyTrie.programs:
                policy = ToyPolicy()
                score = trie_program_logprob(
                    policy, None, tokens, ToyTrie(), temperature
                )
                mass += math.exp(score.total)
                self.assertTrue(policy.training)
                for position, actual in enumerate(score.per_token, 1):
                    choices = ToyTrie().get(tokens[:position])
                    logits = {2: -0.7, 3: 0.8, 4: -1.2, 5: 0.3}
                    weights = {t: math.exp(logits[t] / temperature) for t in choices}
                    expected = math.log(
                        weights[tokens[position]] / sum(weights.values())
                    )
                    self.assertAlmostEqual(actual, expected, delta=1e-12)
                    if len(choices) == 1:
                        self.assertEqual(actual, 0.0)
                if tokens == (1, 4, 2):
                    self.assertLess(
                        score.per_token[-1], 0.0
                    )  # EOS competes with continuation.
            self.assertAlmostEqual(mass, 1.0, delta=1e-12)

    def test_grammar_counterexample_and_refusal(self):
        policy = ToyPolicy()
        trie = trie_program_logprob(policy, None, (1, 3, 2), ToyTrie())
        grammar = child_logprob({3: 0.8, 4: -1.2, 6: 4.0}, [3, 4, 6], 3, 1.0)
        self.assertGreater(trie.total, grammar)
        with self.assertRaisesRegex(ValueError, "grammar"):
            trie_program_logprob(policy, None, (1, 3, 2), ToyTrie(), support="grammar")

    def test_mean_nll_rejected(self):
        with self.assertRaises(TypeError):
            trie_program_logprob(0.5, None, (1, 3, 2), ToyTrie())
        with self.assertRaises(ValueError):
            TrieLogProb(-0.5, (-0.5,), aggregation="mean_token_nll")

    def test_extreme_logits_remain_stable(self):
        self.assertEqual(child_logprob({2: 1e300}, [2], 2, 0.7), 0.0)
        value = child_logprob({3: 1000, 4: 999}, [3, 4], 3, 1.0)
        self.assertAlmostEqual(value, -math.log1p(math.exp(-1)), delta=1e-12)

    def test_incomplete_and_invalid_temperature_refused(self):
        with self.assertRaises(ValueError):
            trie_program_logprob(ToyPolicy(), None, (1, 3), ToyTrie())
        for temperature in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                trie_program_logprob(
                    ToyPolicy(), None, (1, 3, 2), ToyTrie(), temperature
                )


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "state"
        self.ledger = ResearchHWM(self.state / "ledger.sqlite", state_root=self.state)

    def publish(self, result, check=lambda _: True):
        return self.ledger.publish(key(), result, role_check=check)

    def test_identical_tie_loss_and_strict_improvement(self):
        results = [self.publish(publication(q)) for q in (100, 100, 99, 101)]
        self.assertEqual(
            [r["event"] for r in results],
            ["first_label", "tie", "no_improvement", "improvement"],
        )
        self.assertEqual([r["q_grid"] for r in results], [100, 100, 100, 101])

    def test_refusal_reasons_never_create_score(self):
        cases = [
            (replace(publication(), source="fitness_only"), "fitness_only"),
            (
                replace(publication(), outcome="parity_custody", q_grid=None),
                "parity_custody",
            ),
            (
                replace(publication(), outcome="infra_unknown", q_grid=None),
                "infra_unknown",
            ),
            (replace(publication(), hwm_key="b" * 64), "incompatible_key"),
            (replace(publication(), artifacts={}), "missing_closure"),
        ]
        for result, reason in cases:
            event = self.publish(result)
            self.assertEqual(
                (event["event"], event["reason"], event["q_grid"]),
                ("refused", reason, None),
            )
        self.assertTrue(
            all(e["publication"]["q_grid"] is None for e in self.ledger.events(key()))
        )

    def test_fresh_role_check_and_no_missing_artifact(self):
        called = []
        event = self.publish(publication(), check=lambda k: called.append(k) or False)
        self.assertEqual(event["reason"], "role_changed")
        self.assertEqual(called, [key()])
        for role in ARTIFACT_ROLES:
            refs = publication().artifacts.copy()
            del refs[role]
            self.assertEqual(
                self.publish(replace(publication(), artifacts=refs))["reason"],
                "missing_closure",
            )

    def test_fifty_concurrent_writers_monotone_no_lost_event(self):
        barrier = threading.Barrier(50)

        def writer(q):
            barrier.wait(timeout=20)
            return self.publish(publication(q))

        with ThreadPoolExecutor(max_workers=50) as pool:
            results = list(pool.map(writer, range(50)))
        events = self.ledger.events(key())
        self.assertEqual(len(results), 50)
        self.assertEqual([e["version"] for e in events], list(range(1, 51)))
        best = [e["best_q_grid"] for e in events]
        self.assertEqual(best, sorted(best))
        self.assertEqual(best[-1], 49)
        self.assertEqual(sum(e["event"] == "first_label" for e in events), 1)

    def test_sqlite_wal_and_events_are_append_only(self):
        self.publish(publication())
        with sqlite3.connect(self.ledger.path) as db:
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            for sql in ("DELETE FROM events", "UPDATE events SET event='improvement'"):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)

    def test_each_identity_field_changes_key(self):
        for field in key().__dataclass_fields__:
            old = getattr(key(), field)
            new = (
                ("f2", "f3")
                if field == "inner_folds"
                else ("b" * 64 if len(old) == 64 else old + "-v2")
            )
            self.assertNotEqual(key().digest, replace(key(), **{field: new}).digest)

    def test_predicted_float_grid_and_failure_scores_refused(self):
        for result in (
            replace(publication(), source="predicted"),
            replace(publication(), q_grid=0.1),
            replace(publication(), outcome="timeout"),
        ):
            with self.assertRaises((ValueError, TypeError)):
                self.publish(result)

    def test_paths_refuse_service_and_outside_state(self):
        with self.assertRaises(ValueError):
            ResearchHWM(Path(self.tmp.name) / "other.sqlite", state_root=self.state)
        service = (
            Path(self.tmp.name) / "flywheel_neural_continuous_20261009_v1" / "state"
        )
        with self.assertRaises(ValueError):
            ResearchHWM(service / "ledger.sqlite", state_root=service)


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = Path(self.tmp.name) / "runs"
        self.trace = DecisionTrace(self.runs / "toy", runs_root=self.runs)

    def test_reopen_replay_is_idempotent_and_manifest_binds_bytes(self):
        self.trace.append(decision())
        reopened = DecisionTrace(self.trace.directory, runs_root=self.runs)
        reopened.append(decision())
        manifest = reopened.seal()
        self.assertEqual(manifest["records"], 1)
        self.assertEqual(
            manifest["sha256"], hashlib.sha256(self.trace.path.read_bytes()).hexdigest()
        )
        self.assertEqual(self.trace.seal(), manifest)
        row = json.loads(self.trace.path.read_text())
        self.assertEqual(row["predicted"]["q"], 2.0)
        self.assertNotIn("q_grid", row)
        self.assertIsNone(row["propensity"])

    def test_conflicting_replay_or_append_after_seal_refused(self):
        self.trace.append(decision())
        with self.assertRaises(ValueError):
            self.trace.append(replace(decision(), seed=8))
        self.trace.seal()
        with self.assertRaises(ValueError):
            self.trace.append(replace(decision(), decision_id="draw-2"))

    def test_mean_nll_bad_prediction_and_missing_eos_likelihood_refused(self):
        for record in (
            replace(decision(), logprob=-0.5),
            replace(decision(), predicted={"q_grid": 2}),
            replace(decision(), batch_protocol_sha256="short-pin"),
            replace(decision(), game_spec_sha256="not-a-sha"),
            replace(decision(), tokens=(99, 3, 100)),
            replace(decision(), logprob=TrieLogProb(-0.5, (-0.5,))),
        ):
            with self.assertRaises((ValueError, TypeError)):
                self.trace.append(record)

    def test_partial_tail_and_tampered_seal_refused(self):
        interrupted = b'{"partial":'
        self.trace.path.write_bytes(interrupted)
        self.trace.append(decision())
        archives = list(self.trace.directory.glob("decisions.interrupted.*.jsonl"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), interrupted)
        self.trace.append(decision())
        self.assertEqual(len(self.trace.path.read_bytes().splitlines()), 1)
        self.trace.seal()
        self.trace.path.write_bytes(self.trace.path.read_bytes() + b"{}\n")
        with self.assertRaises(ValueError):
            self.trace.seal()
        with self.assertRaises(ValueError):
            self.trace.append(decision())


class IncumbentTests(unittest.TestCase):
    def test_deployed_descriptor_and_keyed_selection_keep_fold_zero(self):
        context = {
            "context_id": PIN,
            "view_id": "native-view",
            "dataset_hash": "d" * 16,
            "cut_id": "native-cut",
            "fold": 0,
            "parent_lineage_sha256": "b" * 64,
            "source_kind": "native_s1_context",
        }
        dsl = "original program"
        selection = {
            PIN: {
                "dsl": dsl,
                "program_sha256": hashlib.sha256(dsl.encode()).hexdigest(),
                "tokens": [1, 3, 2],
            }
        }
        result = original_hwm_program(context, selection)
        self.assertEqual(result.context_identity, PIN)
        self.assertEqual(result.outer_fold, 0)
        self.assertEqual(result.native_descriptor, context)
        self.assertEqual(result.tokens, (1, 3, 2))

    def test_native_identity_and_program_preserved_without_historical_score(self):
        context = {
            f: "original-" + f
            for f in (
                "context_identity",
                "native_view_id",
                "dataset_version",
                "cut",
                "outer_fold",
                "parent_lineage",
            )
        }
        dsl = "original canonical program"
        selection = {
            "context_identity": context["context_identity"],
            "dsl": dsl,
            "program_sha256": hashlib.sha256(dsl.encode()).hexdigest(),
            "fitness": 999,
        }
        program = original_hwm_program(context, selection)
        self.assertEqual(program.context_identity, context["context_identity"])
        self.assertEqual(program.dsl, dsl)
        self.assertFalse(hasattr(program, "fitness"))
        with self.assertRaises(ValueError):
            original_hwm_program(context, {**selection, "context_identity": "wrong"})
        with self.assertRaises(ValueError):
            original_hwm_program(context, {**selection, "dsl": "tampered"})


if __name__ == "__main__":
    unittest.main()
