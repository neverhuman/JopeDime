import hashlib
import unittest
from unittest.mock import patch

from rlfw import successor_support as S
from rlfw.compact_support import CompactSupportTrie, encode
from rlfw.probability_contract import (
    START_PARAMETER_SHA256,
    require_same_support_reference,
)


def program(text):
    return {"dsl": text, "program_sha256": hashlib.sha256(text.encode()).hexdigest()}


class SuccessorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [program(f"fixture-{i}") for i in range(6640)]
        self.pin = S.digest(
            {
                "schema": "jope.flywheel-full-catalog-trie/v1",
                "vocabulary_sha256": "v" * 64,
                "max_seq": 640,
                "programs": self.catalog,
            }
        )
        # This is a synthetic builder fixture; the native audit separately checks
        # the unmodified production v1 116154767... pin and actual tokenizer.
        p = patch.object(S, "V1_TRIE_SHA256", self.pin)
        p.start()
        self.addCleanup(p.stop)
        self.builder = self.build(self.catalog)

    def build(self, catalog):
        return S.SuccessorBuilder(
            catalog,
            v1_trie_sha256=self.pin,
            vocabulary_sha256="v" * 64,
            max_seq=640,
            role_map={"train-c": "train", "dev-c": "validation"},
            validate_program=lambda p: None,
        )

    def add(self, text, context="train-c", outcome="completed"):
        return self.builder.add_measured(
            context,
            program(text),
            shard_sha256="a" * 64,
            outcome=outcome,
        )

    def freeze(self):
        return self.builder.freeze(
            vocabulary_sha256="v" * 64,
            max_seq=640,
            source_pins={"fixture": "a" * 64},
            operator_names=lambda text: [text.split("-")[0]],
        )

    def test_nontrain_unknown_roles_refused(self):
        for role in ["dev-c", "sealed", "missing"]:
            with self.assertRaisesRegex(ValueError, "training ROLE"):
                self.add("new", role)
        self.assertEqual(len(self.builder.programs), 6640)

    def test_catalog_bytes_not_just_claimed_pin(self):
        wrong = list(self.catalog)
        wrong[0] = program("changed")
        with self.assertRaisesRegex(ValueError, "exact v1"):
            self.build(wrong)

    def test_availability_separate_from_unknown_failed_context_targets(self):
        ph = self.add("new", outcome="timeout")
        programs, manifest = self.freeze()
        self.assertEqual(len(programs), 6641)
        self.assertEqual(manifest["added_program_sha256"], [ph])
        self.assertFalse(manifest["availability_grants_context_outcome_eligibility"])
        for outcome, q, baseline, role in [
            ("timeout", None, True, "train"),
            ("completed", None, True, "train"),
            ("completed", 1.0, False, "train"),
            ("completed", 1.0, True, "validation"),
            ("completed", float("nan"), True, "train"),
        ]:
            row = S.ContextOutcomeEligibility("train-c", ph, outcome, q, baseline, role)
            self.assertFalse(row.optimizer_eligible)
        self.assertTrue(
            S.ContextOutcomeEligibility(
                "train-c", ph, "completed", 0.7, True, "train"
            ).optimizer_eligible
        )

    def test_order_duplicates_and_version_are_stable(self):
        self.add("new-b")
        self.add("new-a")
        self.add("new-a")
        programs, first = self.freeze()
        second = self.freeze()[1]
        self.assertEqual(first, second)
        self.assertEqual(first["added_size"], 2)
        self.assertEqual(first["size"], 6642)
        self.assertEqual(first["qualified_use"], "BLOCKED_JEPSON")
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.add("after-freeze")
        self.assertEqual(first["added_operator_program_coverage"], {"new": 2})
        self.assertEqual(
            [p["program_sha256"] for p in programs], sorted(self.builder.programs)
        )

    def test_reference_evaluated_under_successor_mask_only(self):
        require_same_support_reference("new", "new", START_PARAMETER_SHA256)
        with self.assertRaisesRegex(ValueError, "cross-support"):
            require_same_support_reference("new", "old", START_PARAMETER_SHA256)
        with self.assertRaisesRegex(ValueError, "start checkpoint"):
            require_same_support_reference("new", "new", "0" * 64)

    def test_reused_support_requires_complete_current_train_provenance(self):
        legitimate, unseen = program("prepared-train"), program("prepared-extra")
        # Mirror the audit's pinned-cache preload, then current TRAIN rescan.
        for p in [legitimate, unseen]:
            self.builder.programs[p["program_sha256"]] = dict(p)
        self.add(legitimate["dsl"])
        with self.assertRaisesRegex(ValueError, "1 .*current TRAIN provenance"):
            self.freeze()
        self.assertFalse(self.builder.frozen)
        # A current measured TRAIN row completes custody; no cache-only leaf
        # can enter the positive frozen support, including after a failed freeze.
        self.add(unseen["dsl"])
        programs, manifest = self.freeze()
        self.assertEqual(len(programs), 6642)
        self.assertEqual(
            manifest["added_program_sha256"],
            sorted(p["program_sha256"] for p in [legitimate, unseen]),
        )
        self.assertEqual(manifest["counts"]["measured_rows"], 2)
        self.assertEqual(
            set(manifest["added_program_sha256"]), set(self.builder.sources)
        )

    def test_compact_prefix_index_keeps_eos_and_all_children(self):
        programs = [program("short"), program("long"), program("other")]
        rows = [(1, 3, 2), (1, 3, 4, 2), (1, 5, 2)]
        trie = CompactSupportTrie(
            programs,
            encoded_by_sha={
                p["program_sha256"]: encode(t) for p, t in zip(programs, rows)
            },
            vocabulary_sha256="v" * 64,
            max_seq=8,
        )
        self.assertEqual(set(trie.get((1,))), {3, 5})
        self.assertEqual(set(trie.get((1, 3))), {2, 4})
        self.assertEqual(trie.get((1, 3, 2)), {})
        self.assertEqual(trie.programs[(1, 3, 4, 2)], programs[1])
        self.assertNotIn((1, 3, 4), trie.programs)
        self.assertEqual(trie.get((0, 3)), ())

    def test_compact_duplicate_incomplete_leaves_refused(self):
        programs = [program("a"), program("b")]
        for rows in [[(1, 3, 2), (1, 3, 2)], [(1, 3), (1, 4, 2)]]:
            with self.assertRaises(ValueError):
                CompactSupportTrie(
                    programs,
                    encoded_by_sha={
                        p["program_sha256"]: encode(t) for p, t in zip(programs, rows)
                    },
                    vocabulary_sha256="v" * 64,
                    max_seq=8,
                )


if __name__ == "__main__":
    unittest.main()
