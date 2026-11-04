"""Geometry controls for canonical JAIN endpoints."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender"))
from jain_spatial_layouts import SPATIAL_LAYOUTS, transverse_position


class SpatialLayoutTests(unittest.TestCase):
    def test_layered_retains_previous_coordinates(self):
        for index in range(92):
            column, row = divmod(index, 32)
            rows = min(32, 92 - column * 32)
            self.assertEqual(transverse_position("layered", .5, index, 92),
                             (-1.1 + 2.2 * column / 2, -4.35 + 8.7 * row / (rows - 1)))

    def test_alternatives_fit_the_stage_and_keep_every_node_distinct(self):
        observed_counts = (92, 140, 111, 103, 183, 331, 379, 155, 204, 77, 395, 60)
        stages = list(enumerate(observed_counts)) + [(5, count) for count in (1, 2, 32, 33, 64, 65)]
        for layout in SPATIAL_LAYOUTS[1:]:
            for stage, count in stages:
                with self.subTest(layout=layout, stage=stage, count=count):
                    points = [transverse_position(layout, stage / 11, index, count) for index in range(count)]
                    self.assertTrue(all(abs(y) <= 2.8 and abs(z) <= 4.1 for y, z in points))
                    self.assertEqual(len({(round(y, 10), round(z, 10)) for y, z in points}), count)

    def test_silhouettes_differ_in_the_reference_projection(self):
        signatures = set()
        for layout in SPATIAL_LAYOUTS:
            signature = []
            for stage in range(12):
                points = [transverse_position(layout, stage / 11, index, 395) for index in range(395)]
                projected = [(round(.488 * y, 1), round(.895 * z + .321 * y, 1)) for y, z in points]
                signature.append(tuple(sorted(set(projected))))
            signatures.add(tuple(signature))
        self.assertEqual(len(signatures), len(SPATIAL_LAYOUTS))

    def test_rejects_invalid_coordinates_and_unknown_layout(self):
        for layout, t, index, count in [("fake", 0, 0, 1), ("iris", math.nan, 0, 1),
                                         ("iris", 1.1, 0, 1), ("iris", .5, 1, 1), ("iris", 0, 0, 0)]:
            with self.subTest(layout=layout, t=t, index=index, count=count):
                with self.assertRaises(ValueError):
                    transverse_position(layout, t, index, count)


if __name__ == "__main__":
    unittest.main()
