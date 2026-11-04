"""Protect the supplied mark's separate V components and the O counter."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("veox_endcap", root/"blender"/"veox_endcap.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SuppliedLogo(unittest.TestCase):
    def test_vector_contours_keep_original_hole_and_separate_parts(self):
        raw = (root/"assets"/"veox-logo.svg").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "2fb4e1c587044aad11977f9cbec3f5c5af772153850d6fa226fb4d91f65c1dfb")
        paths = [node.attrib["d"] for node in ET.fromstring(raw).iter() if node.tag.endswith("}path")]
        shapes = [module.contours(path) for path in paths]
        self.assertEqual([len(shape) for shape in shapes], [2,1,1,1,1])
        def area(points):
            return sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1]))/2
        self.assertLess(area(shapes[0][0])*area(shapes[0][1]), 0)
        self.assertTrue(all(.558594 <= x <= 78.415734 and .685547 <= y <= 19.529747
                            for shape in shapes for contour in shape for x,y in contour))

    def test_unsupported_or_unclosed_paths_fail(self):
        for path in ("M0 0 Q1 1 2 2Z", "M0 0L1 1", "Z", "M0 0M1 1Z", ""):
            with self.assertRaises(ValueError):
                module.contours(path)


if __name__ == "__main__":
    unittest.main()
