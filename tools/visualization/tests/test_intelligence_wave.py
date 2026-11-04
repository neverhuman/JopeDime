"""Regression checks for complete sampled source scroll and wave confinement."""
import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intelligence_wave import CASE_FRAMES, SCROLL_FIRST, SCROLL_LAST, scroll_offsets
from intelligence_wave_effect import apply_wave


class IntelligenceWaveTests(unittest.TestCase):
    def test_each_wrapped_row_is_fully_visible_in_linear_scroll(self):
        row_count, line_height, view_height = 1_350, 19, 845
        offsets = scroll_offsets(row_count, line_height, view_height)
        self.assertEqual(len(offsets), CASE_FRAMES)
        self.assertEqual(offsets[:SCROLL_FIRST + 1], [0] * (SCROLL_FIRST + 1))
        self.assertEqual(offsets[SCROLL_LAST:], [row_count * line_height - view_height]
                         * (CASE_FRAMES - SCROLL_LAST))
        self.assertEqual(offsets, sorted(offsets))
        steps = [offsets[i + 1] - offsets[i] for i in range(SCROLL_FIRST, SCROLL_LAST)]
        self.assertLessEqual(max(steps) - min(steps), 1)
        for row in range(row_count):
            self.assertTrue(any(top <= row * line_height
                                and top + view_height >= (row + 1) * line_height
                                for top in offsets), row)
        with self.assertRaisesRegex(ValueError, "skip"):
            scroll_offsets(10_000, line_height, view_height)

    def test_wave_only_lifts_existing_signal_and_moves_right(self):
        roi = (20, 20, 620, 220)
        base = np.zeros((240, 640, 3), dtype=np.uint8)
        base[100:115, 30:610] = (28, 76, 80)
        base[10:15] = 100
        frame = Image.fromarray(base)
        centroids = []
        for phase in (.2, .5, .8):
            result = np.asarray(apply_wave(frame, phase, roi=roi))
            np.testing.assert_array_equal(result[:20], base[:20])
            np.testing.assert_array_equal(result[base.max(axis=2) == 0], 0)
            gains = (result.astype(int) - base).sum(axis=(0, 2))
            self.assertGreater(gains.sum(), 0)
            centroids.append(np.arange(len(gains)) @ gains / gains.sum())
        self.assertGreater(centroids[1] - centroids[0], 100)
        self.assertGreater(centroids[2] - centroids[1], 100)
        np.testing.assert_array_equal(np.asarray(apply_wave(frame, 0, roi=roi)), base)


if __name__ == "__main__":
    unittest.main()
