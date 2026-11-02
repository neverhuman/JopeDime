#!/usr/bin/env python3
"""Determinism and byte-isolation gates for red-team DL context jitter."""

from __future__ import annotations

import hashlib
import struct
import unittest

from test_exact_dl_generate import deterministic_normal


class TrainOnlyRedteamGenerationTests(unittest.TestCase):
    def test_normal_stream_is_seeded_and_batch_order_independent(self) -> None:
        dataset = "ee56a1015e344c11"
        first = [deterministic_normal(101, dataset, index) for index in range(1536)]
        replay = [
            deterministic_normal(101, dataset, index)
            for index in reversed(range(1536))
        ]
        self.assertEqual(first, list(reversed(replay)))
        second = [deterministic_normal(211, dataset, index) for index in range(1536)]
        self.assertNotEqual(first, second)

    def test_jitter_changes_only_embedding_bytes(self) -> None:
        dataset = "019de5d76a718163"
        embedding = [index / 1536.0 for index in range(1536)]
        target = [index / 512.0 for index in range(512)]
        standard_deviation = [0.25 + (index % 7) / 10.0 for index in range(1536)]
        jittered = [
            value
            + 0.05
            * standard_deviation[index]
            * deterministic_normal(307, dataset, index)
            for index, value in enumerate(embedding)
        ]
        exact = struct.pack("<2048f", *embedding, *target)
        perturbed = struct.pack("<2048f", *jittered, *target)
        self.assertNotEqual(
            hashlib.sha256(exact[: 1536 * 4]).digest(),
            hashlib.sha256(perturbed[: 1536 * 4]).digest(),
        )
        self.assertEqual(exact[1536 * 4 :], perturbed[1536 * 4 :])


if __name__ == "__main__":
    unittest.main()
