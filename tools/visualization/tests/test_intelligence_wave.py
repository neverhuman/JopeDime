"""Regression checks for complete sampled source scroll and wave confinement."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intelligence_wave import (
    CASE_FRAMES, SCROLL_FIRST, SCROLL_LAST, TABLE_SWITCH_FRAMES, code_strip, display_rate,
    scroll_offsets, table_orders,
)
from intelligence_wave_effect import apply_wave
from code_pages import FONT_CANDIDATES
from table_panels import IMPORTANCE_FILL, importance_percentiles, percentile_color


class IntelligenceWaveTests(unittest.TestCase):
    def test_compact_rate_labels(self):
        self.assertEqual(display_rate(2_409_214), "2.4M tokens/sec")
        self.assertEqual(display_rate(299_700), "300k tokens/sec")
        self.assertEqual(display_rate(999_970), "1M tokens/sec")
        self.assertEqual(display_rate(500), "500 tokens/sec")

    def test_table_shuffles_are_reproducible_complete_permutations(self):
        row_ids = range(2, 36, 2)
        orders = table_orders("case-a", row_ids)
        self.assertEqual(orders[0], list(range(17)))
        self.assertEqual(orders, table_orders("case-a", row_ids))
        self.assertEqual(len({tuple(order) for order in orders}), 15)
        for order in orders:
            self.assertEqual(sorted(order), list(range(17)))
        with self.assertRaisesRegex(ValueError, "17 distinct"):
            table_orders("case-a", [0] * 17)

    def test_table_shuffles_track_the_fivefold_cadence_without_drift(self):
        self.assertEqual(TABLE_SWITCH_FRAMES, (0, 8, 17, 25, 34, 42, 50, 59, 67, 76, 84, 92, 101, 109, 118))
        holds = [end - start for start, end in zip(TABLE_SWITCH_FRAMES, (*TABLE_SWITCH_FRAMES[1:], CASE_FRAMES))]
        self.assertEqual(set(holds), {8, 9})
        self.assertEqual(sum(holds), CASE_FRAMES)

    def test_feature_percentiles_average_ties_and_anchor_zeros(self):
        values = [[0, .5, .5], [1, 0, .2]]
        first, second = importance_percentiles(values)
        self.assertEqual(first, [0, .7, .7])
        self.assertEqual(second, [1, 0, .4])
        self.assertEqual(importance_percentiles(values[::-1]), [second, first])
        self.assertEqual(importance_percentiles([[0] * 4] * 17), [[0] * 4] * 17)
        for rank, stop in zip((0, .5, 1), IMPORTANCE_FILL, strict=True):
            self.assertEqual(percentile_color(rank, IMPORTANCE_FILL), (*stop, 255))

    def test_source_spans_follow_their_language_and_file(self):
        font = next((Path(path) for path in FONT_CANDIDATES if Path(path).is_file()), None)
        if font is None:
            self.skipTest("monospace raster font unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            files = []
            for name, language, text, spans in (
                ("model.cpp", "cpp", b"fit();\npredict();\n", [[0, 7], [7, 18]]),
                ("model.js", "javascript", b"const head = 'ridge';\n", [[0, 21]]),
            ):
                (root / name).write_bytes(text)
                files.append({"language": language, "source_path": name,
                              "source_sha256": hashlib.sha256(text).hexdigest(),
                              "operations": [{"label": "ridge", "ordinal": 1, "spans": spans}]})
            _, rows, _, _, cues, bindings = code_strip(files, root, root / "pages", font, 563, 845)
            self.assertEqual([item["filename"] for item in bindings], ["model.cpp", "model.js"])
            self.assertEqual([(cue["file_index"], cue["phase"]) for cue in cues],
                             [(0, "fit"), (0, "predict"), (1, "recipe")])
            self.assertEqual([cue["rows"] for cue in cues], [[0, 0], [1, 1], [4, 4]])
            self.assertEqual(rows[-1], (1, 1))

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
