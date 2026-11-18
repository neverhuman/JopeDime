"""File-scoped contract test. Run this file; the directory is not a package."""
import ast
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "dope_value_probe.py"
ENCODER = ROOT / "crates/jope-encoders/src/dope_encoder.rs"


def load_contract():
    spec = importlib.util.spec_from_file_location("dope_value_probe_contract", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("contract module did not load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DopeValueProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_contract()

    def test_module_stays_off_the_product_path(self):
        tree = ast.parse(SOURCE.read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module != "__future__":
                imported.append(node.module.split(".")[0])
        self.assertEqual(imported, [])
        rust = ENCODER.read_text()
        self.assertIn(f'"{self.mod.DOPE_WEIGHTS_SHA256}"', rust)
        self.assertIn(f'"{self.mod.DOPE512_RECIPE}"', rust)

    def test_layout_and_arm_widths(self):
        mod = self.mod
        ends = [0]
        for name, start, end in mod.DOPE629_SLICES:
            self.assertEqual(start, ends[-1])
            self.assertGreater(end, start)
            ends.append(end)
        self.assertEqual(ends[-1], mod.DOPE629_DIM)
        widths = {name: end - start for name, start, end in mod.DOPE629_SLICES}
        self.assertEqual(widths["vec256"] + widths["task_latent"] + widths["dist_latent"], mod.DOPE512_PREFIX)
        self.assertEqual(mod.LEARNED_TAIL, 373)
        self.assertEqual(mod.ARM_WIDTHS["R_learned"], 373)
        self.assertEqual(mod.ARM_WIDTHS["R_none"], 0)
        self.assertEqual(mod.ARM_WIDTHS["R_shape"], 3)
        self.assertEqual(mod.ARM_WIDTHS["R_desc"], 290)
        self.assertEqual(mod.ARM_WIDTHS["R_dope"], 663)
        self.assertEqual(mod.ARM_WIDTHS["R_noise"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.ARM_WIDTHS["R_randrouter"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.ARM_WIDTHS["R_shuffled"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.ARM_WIDTHS["R_tok"], 320)
        self.assertEqual(mod.ARM_WIDTHS["R_hidden"], 256)
        self.assertEqual(mod.ARM_WIDTHS["R_packet"], 24)
        self.assertEqual(mod.ARM_WIDTHS["R_desc_ultra"], mod.ARM_WIDTHS["R_desc"])
        self.assertEqual(mod.ARM_WIDTHS["R_dope_ultra"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.FINE_LANDMARKS, (
            ("gbdt", 17), ("knn1", 18), ("knn16", 19), ("net", 20), ("gbdt_quarter", 22),
        ))
        self.assertNotIn("profile_oracle", mod.ARM_WIDTHS)
        self.assertEqual(mod.CEILING_ARMS, ("profile_oracle",))
        self.assertEqual([item[0] for item in mod.CONTRASTS], ["C1", "C2", "C3", "C4", "C4", "C5"])
        placebos = [item for item in mod.CONTRASTS if item[0] in mod.PLACEBO_CONTRASTS]
        self.assertEqual([item[2] for item in placebos], ["R_none", "R_none"])
        self.assertIn("scored_probe", mod.NOT_RUN)

    def test_split_and_family_delta(self):
        mod = self.mod
        values = [float(i) for i in range(629)]
        parts = mod.split_dope629(values)
        self.assertEqual(parts["vec256"], tuple(values[:256]))
        self.assertEqual(parts["task_latent"] + parts["dist_latent"], tuple(values[256:512]))
        self.assertEqual(sum(len(part) for part in parts.values()), 629)
        learned = values[256:]
        self.assertEqual(len(learned), mod.LEARNED_TAIL)
        with self.assertRaises(ValueError):
            mod.split_dope629(values[:-1])
        # Family A mean delta 2, family B mean delta 10, unweighted mean 6.
        delta = mod.paired_family_mean_delta([1.0, 3.0, 10.0], [0.0, 0.0, 0.0], ["a", "a", "b"])
        self.assertEqual(delta, 6.0)
        with self.assertRaises(ValueError):
            mod.paired_family_mean_delta([], [], [])
        with self.assertRaises(ValueError):
            mod.paired_family_mean_delta([1.0], [1.0, 2.0], ["a"])
        self.assertTrue(mod.placebo_interval_includes_zero(-0.1, 0.2))
        self.assertTrue(mod.placebo_interval_includes_zero(0.0, 0.2))
        self.assertFalse(mod.placebo_interval_includes_zero(0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
