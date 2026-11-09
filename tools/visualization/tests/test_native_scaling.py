"""Native rerasterization must preserve the full source and high-bit image values."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from code_pages import make_pages, reconstruct_source
from uhd_pixels import blend_rgba8, blend_rgb16


class NativeScaling(unittest.TestCase):
    def test_double_resolution_preserves_all_utf8_source_chunks(self):
        scratch = Path(__file__).parents[3] / "target" / "visualization-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            source = root / "predictor.rs"
            raw = ('// λ retained\r\n\tpub fn predict(x: f64) -> f64 {\n'
                   '    let y = x * 1.25e-4;\r\n    y\n}\n').encode()
            source.write_bytes(raw)
            hd = make_pages(source, root/"hd", columns=16, rows=3)
            uhd = make_pages(source, root/"uhd", columns=16, rows=3, render_scale=2)
            self.assertEqual(reconstruct_source(hd), raw)
            self.assertEqual(reconstruct_source(uhd), raw)
            self.assertEqual(hd["source_sha256"], uhd["source_sha256"])
            self.assertEqual([p["rows"] for p in hd["pages"]], [p["rows"] for p in uhd["pages"]])
            for key in ("width", "height", "font_size", "line_height"):
                if key in hd:
                    self.assertEqual(uhd[key], hd[key]*2)
            for record in uhd["pages"]:
                with Image.open(root/"uhd"/record["file"]) as image:
                    self.assertEqual(image.size, (hd["width"]*2, hd["height"]*2))
            with self.assertRaisesRegex(ValueError, "render_scale"):
                make_pages(source, root/"bad", render_scale=3)
            self.assertFalse((root/"bad").exists())

    def test_alpha_blending_preserves_uncovered_sixteen_bit_values(self):
        original = np.array([[[1, 258, 65534], [12345, 32001, 65000]]], dtype="<u2")
        target = original.copy()
        overlay = Image.fromarray(np.array([[[255, 1, 2, 0], [12, 34, 56, 127]]], dtype=np.uint8))
        blend_rgba8(target, overlay)
        np.testing.assert_array_equal(target[0,0], original[0,0])
        expected = (original[0,1].astype(np.uint64)*128 + np.array([12,34,56])*257*127 +127)//255
        np.testing.assert_array_equal(target[0,1], expected)
        blend_rgb16(target, original, 1.)
        np.testing.assert_array_equal(target, original)


if __name__ == "__main__":
    unittest.main()
