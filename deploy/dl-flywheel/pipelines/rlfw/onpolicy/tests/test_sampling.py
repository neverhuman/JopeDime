"""Exact quota blocks, parent balancing, family cap and role isolation."""

import unittest
from collections import Counter
from dataclasses import replace

from rlfw.onpolicy.sampling import (
    REPLAY_BUCKETS,
    ContextScheduler,
    ReplayItem,
    ReplaySampler,
    SamplingUnavailable,
    ScheduledContext,
)
from rlfw.onpolicy.tests.fixtures import q_for


def items():
    return tuple(
        ReplayItem(q_for(context=f"c{p}", parent=f"p{p}"), REPLAY_BUCKETS)
        for p in range(4)
    )


def contexts():
    return (
        ScheduledContext("p", "c", "comparable_hwm", "train", True, incumbent=q_for()),
        ScheduledContext(
            "p1", "c1", "no_winner_original", "train", True, original=True
        ),
        ScheduledContext("p2", "c2", "augmented", "train", True),
        ScheduledContext("dev", "dc", "development", "development", True),
    )


class SamplingTests(unittest.TestCase):
    def test_replay_exact_four_quarters_and_one_parent_per_batch(self):
        sampler = ReplaySampler(items(), seed=4, rolling_window=8, max_family_batches=4)
        batches = [sampler.batch(8, role_check=lambda q: True) for _ in range(80)]
        self.assertEqual(
            Counter(x.bucket for x in batches), dict.fromkeys(REPLAY_BUCKETS, 20)
        )
        for i in range(0, 80, 4):
            self.assertEqual(
                set(x.bucket for x in batches[i : i + 4]), set(REPLAY_BUCKETS)
            )
        for b in batches:
            self.assertEqual({x.parent for x in b.items}, {b.parent})
            self.assertEqual(len(b.items), 8)

    def test_seed_replay_is_deterministic(self):
        def draw():
            s = ReplaySampler(items(), seed=19, rolling_window=8, max_family_batches=4)
            return tuple(s.batch(3, role_check=lambda q: True) for _ in range(24))

        self.assertEqual(draw(), draw())
        a, b = (
            ContextScheduler(contexts(), seed=11),
            ContextScheduler(tuple(reversed(contexts())), seed=11),
        )
        self.assertEqual(
            [a.next(role_check=lambda c: True) for _ in range(30)],
            [b.next(role_check=lambda c: True) for _ in range(30)],
        )

    def test_parent_uniformity_is_not_proportional_to_1000_siblings(self):
        small = ReplayItem(q_for(context="small", parent="small"), REPLAY_BUCKETS)
        siblings = tuple(
            ReplayItem(q_for(context=f"s{i}", parent="large"), REPLAY_BUCKETS)
            for i in range(1000)
        )
        s = ReplaySampler(
            (small, *siblings), seed=123, rolling_window=1, max_family_batches=1
        )
        counts = Counter(
            s.batch(1, role_check=lambda q: True).parent for _ in range(2000)
        )
        self.assertGreater(counts["small"], 850)
        self.assertLess(counts["small"], 1150)

    def test_family_cap_holds_in_every_rolling_window(self):
        s = ReplaySampler(items(), seed=5, rolling_window=8, max_family_batches=3)
        sequence = [s.batch(4, role_check=lambda q: True).parent for _ in range(60)]
        for i in range(len(sequence) - 7):
            self.assertLessEqual(max(Counter(sequence[i : i + 8]).values()), 3)

    def test_missing_bucket_or_cap_keeps_quota_pending_and_role_rechecked(self):
        q = q_for()
        s = ReplaySampler(
            (ReplayItem(q, REPLAY_BUCKETS),),
            seed=4,
            rolling_window=4,
            max_family_batches=1,
        )
        first = s.batch(1, role_check=lambda q: True)
        pending = s._quota.peek()
        for _ in range(2):
            with self.assertRaises(SamplingUnavailable):
                s.batch(1, role_check=lambda q: True)
            self.assertEqual(s._quota.peek(), pending)
        self.assertNotEqual(first.bucket, pending)
        s = ReplaySampler(items(), seed=4, rolling_window=4, max_family_batches=4)
        pending = s._quota.peek()
        with self.assertRaises(SamplingUnavailable):
            s.batch(1, role_check=lambda q: False)
        self.assertEqual(s._quota.peek(), pending)
        self.assertEqual(s.batch(1, role_check=lambda q: True).bucket, pending)
        missing = ReplaySampler(
            (ReplayItem(q, ("measured",)),),
            seed=4,
            rolling_window=1,
            max_family_batches=1,
        )
        for _ in range(4):
            try:
                missing.batch(1, role_check=lambda q: True)
            except SamplingUnavailable:
                self.assertNotEqual(missing._quota.peek(), "measured")
                break
        else:
            self.fail("missing quota was silently redistributed")

    def test_context_scheduler_exact_shares_and_development_eval_only(self):
        scheduler = ContextScheduler(contexts(), seed=42)
        seen = [scheduler.next(role_check=lambda c: True) for _ in range(100)]
        self.assertEqual(
            Counter(x.pool for x in seen),
            {
                "comparable_hwm": 50,
                "no_winner_original": 30,
                "augmented": 10,
                "development": 10,
            },
        )
        for x in seen:
            self.assertEqual(x.optimizer_allowed, x.pool != "development")
            if x.pool == "no_winner_original":
                self.assertIsNone(x.incumbent)
        for i in range(0, 100, 10):
            self.assertEqual(
                Counter(x.pool for x in seen[i : i + 10]),
                {
                    "comparable_hwm": 5,
                    "no_winner_original": 3,
                    "augmented": 1,
                    "development": 1,
                },
            )

    def test_context_roles_rights_parent_overlap_missing_incumbent_refused(self):
        c = contexts()[0]
        for changes in (
            {"role": "sealed"},
            {"rights_cleared": False},
            {"incumbent": None},
            {"incumbent": q_for(context="wrong")},
            {"pool": "unknown"},
        ):
            with self.assertRaises(ValueError):
                replace(c, **changes)
        with self.assertRaises(ValueError):
            replace(contexts()[1], original=False)
        with self.assertRaises(ValueError):
            ContextScheduler((c, replace(contexts()[3], parent="p")), seed=0)
        with self.assertRaises(ValueError):
            ContextScheduler((c, c), seed=0)

    def test_scheduler_role_flip_blocks_requested_pool_without_redistribution(self):
        s = ContextScheduler(contexts(), seed=4)
        pending = s._quota.peek()
        with self.assertRaises(SamplingUnavailable):
            s.next(role_check=lambda c: False)
        self.assertEqual(s.next(role_check=lambda c: True).pool, pending)

    def test_replay_rejects_unverified_scores_duplicate_rows_and_invalid_caps(self):
        with self.assertRaises(ValueError):
            ReplayItem(0.4, REPLAY_BUCKETS)
        with self.assertRaises(ValueError):
            ReplaySampler(
                (items()[0],) * 2, seed=0, rolling_window=4, max_family_batches=2
            )
        for cap in (0, 5, True):
            with self.assertRaises(ValueError):
                ReplaySampler(items(), seed=0, rolling_window=4, max_family_batches=cap)


if __name__ == "__main__":
    unittest.main()
