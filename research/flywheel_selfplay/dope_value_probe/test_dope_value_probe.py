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

        pkt_ends = [0]
        for _, start, end in mod.PACKET_V1_SLICES:
            self.assertEqual(start, pkt_ends[-1])
            self.assertGreater(end, start)
            pkt_ends.append(end)
        self.assertEqual(pkt_ends[-1], mod.PACKET_V1_DIM)
        self.assertEqual(mod.PACKET_V1_DIM, 196)

        self.assertEqual(mod.ARM_WIDTHS["R_learned"], 373)
        self.assertEqual(mod.ARM_WIDTHS["R_none"], 0)
        self.assertEqual(mod.ARM_WIDTHS["R_shape"], 3)
        self.assertEqual(mod.ARM_WIDTHS["lm48_z"], 48)
        self.assertEqual(mod.ARM_WIDTHS["R_lm80"], 80)
        self.assertEqual(mod.ARM_WIDTHS["R_desc"], 290)
        self.assertEqual(mod.ARM_WIDTHS["R_dope"], 663)
        self.assertEqual(mod.ARM_WIDTHS["R_noise"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.ARM_WIDTHS["R_randrouter"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.ARM_WIDTHS["R_shuffled"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.ARM_WIDTHS["R_tok"], 320)
        self.assertEqual(mod.ARM_WIDTHS["R_hidden"], 256)
        self.assertEqual(mod.ARM_WIDTHS["R_packet"], 24)
        self.assertEqual(mod.ARM_WIDTHS["R_packet_v1"], 196)
        self.assertEqual(mod.ARM_WIDTHS["R_packet_notarget"], 196)
        self.assertEqual(mod.ARM_WIDTHS["R_dope_plus_packet_v1"], 859)
        self.assertEqual(mod.ARM_WIDTHS["B03_svd96"], 96)
        self.assertEqual(mod.ARM_WIDTHS["R_desc_ultra"], mod.ARM_WIDTHS["R_desc"])
        self.assertEqual(mod.ARM_WIDTHS["R_dope_ultra"], mod.ARM_WIDTHS["R_dope"])
        self.assertEqual(mod.FINE_LANDMARKS, (
            ("gbdt", 17), ("knn1", 18), ("knn16", 19), ("net", 20), ("gbdt_quarter", 22),
        ))
        self.assertNotIn("profile_oracle", mod.ARM_WIDTHS)
        self.assertIn("profile_oracle", mod.CEILING_ARMS)
        self.assertIn("R_oracle64", mod.CEILING_ARMS)
        self.assertIn("WP3_pred_reveal", mod.UPGRADE_ARMS)
        self.assertIn("B03_svd96", mod.UPGRADE_ARMS)

        contrast_ids = {item[0] for item in mod.CONTRASTS}
        self.assertEqual(contrast_ids, {"C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"})
        placebos = [item for item in mod.CONTRASTS if item[0] in mod.PLACEBO_CONTRASTS]
        self.assertEqual([item[2] for item in placebos], ["R_none", "R_none"])
        self.assertEqual(mod.SCORED_WAVE, "H470")
        self.assertEqual(mod.NOT_RUN, ("holdout_read", "sealed_rows"))

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
        with self.assertRaises(ValueError):
            mod.split_dope629([float("nan")] * 629)

        pkt_vals = [float(i) * 0.5 for i in range(196)]
        pkt_parts = mod.split_packet_v1(pkt_vals)
        self.assertEqual(len(pkt_parts["gates14"]), 14)
        self.assertEqual(len(pkt_parts["task_residual_globals14"]), 14)
        self.assertEqual(len(pkt_parts["copula_regime_basis16"]), 16)
        self.assertEqual(len(pkt_parts["col_pool_sym96"]), 96)
        self.assertEqual(len(pkt_parts["col_pool_activity32"]), 32)
        self.assertEqual(len(pkt_parts["tree_pair_latent24"]), 24)
        self.assertEqual(sum(len(part) for part in pkt_parts.values()), 196)

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
        with self.assertRaises(ValueError):
            mod.placebo_interval_includes_zero(0.5, -0.5)

    def test_paired_family_bootstrap_ci_determinism_and_invariance(self):
        mod = self.mod
        left = [2.0, 4.0, 3.0, 5.0, 6.0, 8.0, 10.0, 12.0]
        right = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        fams = ["f1", "f1", "f2", "f2", "f3", "f3", "f4", "f4"]

        res1 = mod.paired_family_bootstrap_ci(left, right, fams, n_boot=2000, seed=20261118)
        res2 = mod.paired_family_bootstrap_ci(left, right, fams, n_boot=2000, seed=20261118)
        self.assertEqual(res1, res2)

        # Reordering contexts leaves family means and canonical bootstrap draws identical.
        perm = [7, 5, 3, 1, 6, 4, 2, 0]
        res_perm = mod.paired_family_bootstrap_ci(
            [left[i] for i in perm],
            [right[i] for i in perm],
            [fams[i] for i in perm],
            n_boot=2000,
            seed=20261118,
        )
        self.assertEqual(res1, res_perm)

        # Family deltas: f1=2, f2=3, f3=5, f4=9 -> mean = 4.75, strictly positive.
        self.assertAlmostEqual(res1["mean_delta"], 4.75, places=12)
        self.assertGreaterEqual(res1["ci_low"], 2.0)
        self.assertLessEqual(res1["ci_high"], 9.0)
        self.assertEqual(res1["family_wins"], 4)
        self.assertEqual(res1["family_losses"], 0)
        self.assertEqual(res1["family_harm_rate"], 0.0)
        self.assertLess(res1["p_value"], 0.01)

        # Pure symmetric noise around 0 includes 0 in its 95% CI.
        null_left = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5]
        null_right = [0.0] * 8
        null_fams = [f"fam_{i}" for i in range(8)]
        null_res = mod.paired_family_bootstrap_ci(null_left, null_right, null_fams, n_boot=2000, seed=7)
        self.assertAlmostEqual(null_res["mean_delta"], 0.0, places=12)
        self.assertTrue(mod.placebo_interval_includes_zero(null_res["ci_low"], null_res["ci_high"]))
        self.assertGreater(null_res["p_value"], 0.5)

        with self.assertRaises(ValueError):
            mod.paired_family_bootstrap_ci([1.0], [0.0], ["f"], n_boot=0)
        with self.assertRaises(ValueError):
            mod.paired_family_bootstrap_ci([1.0], [0.0], ["f"], alpha=1.0)

    def test_holm_bonferroni(self):
        mod = self.mod
        self.assertEqual(mod.holm_bonferroni([]), ())
        # Classical textbook example: [0.01, 0.04, 0.03, 0.005]
        # Sorted: 0.005 (*4=0.02), 0.01 (*3=0.03), 0.03 (*2=0.06), 0.04 (*1=0.04 -> max(0.06,0.04)=0.06)
        adj = mod.holm_bonferroni([0.01, 0.04, 0.03, 0.005])
        self.assertAlmostEqual(adj[0], 0.03, places=12)
        self.assertAlmostEqual(adj[1], 0.06, places=12)
        self.assertAlmostEqual(adj[2], 0.06, places=12)
        self.assertAlmostEqual(adj[3], 0.02, places=12)

        # Capping at 1.0 and dict interface
        d_adj = mod.holm_bonferroni({"C1": 0.4, "C2": 0.002, "C3": 0.8})
        self.assertAlmostEqual(d_adj["C2"], 0.006, places=12)
        self.assertAlmostEqual(d_adj["C1"], 0.8, places=12)
        self.assertAlmostEqual(d_adj["C3"], 0.8, places=12)

        with self.assertRaises(ValueError):
            mod.holm_bonferroni([-0.01, 0.5])
        with self.assertRaises(ValueError):
            mod.holm_bonferroni([0.2, 1.2])
        with self.assertRaises(ValueError):
            mod.holm_bonferroni("0.05")


if __name__ == "__main__":
    unittest.main()
