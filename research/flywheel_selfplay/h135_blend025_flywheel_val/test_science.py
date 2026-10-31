"""Data-free native-estimand and fail-closed controls."""

import unittest
import numpy as np
from science import native, sanity, summarize, ARMS
from monitor import reason


class Controls(unittest.TestCase):
    def test_native_does_not_clip_or_use_top_four(self):
        self.assertEqual(native([3, 2, 1], [-9, -8, -4], [True] * 3), (4.0, 2))
        self.assertEqual(
            native([0, 0, 1], [-9, -8, -4], [False, True, True]), (-2.0, 0)
        )

    def test_invalid_cost_refused(self):
        with self.assertRaises(ValueError):
            native([np.nan], [1], [True])

    def test_reference_gate_checks_every_context(self):
        ref = dict(best_utility=1.1377942, utility=dict(per_context=[1.1377942] * 91))
        self.assertEqual(sanity(np.full(91, 1.1377942), ref)["status"], "PASS")
        vals = np.full(91, 1.1377942)
        vals[:2] += [0.1, -0.1]
        with self.assertRaises(RuntimeError):
            sanity(vals, ref)

    def fixture(self):
        rows = [
            dict(context_sha256=f"{i:03}", family_id=f"f{i // 5}") for i in range(91)
        ]
        ref = np.zeros(91)
        values = np.stack([np.full(91, 0.01 + i * 0.001) for i in range(3)])
        return rows, ref, {a: values.copy() for a in ARMS}

    def test_primary_only_admits(self):
        rows, ref, arms = self.fixture()
        m = summarize(rows, ref, arms)
        self.assertEqual(m["verdict"], "CANDIDATE_REVIEW_ONLY")
        self.assertFalse(m["arms"]["BLEND_050"]["candidate"])
        self.assertAlmostEqual(m["arms"]["BLEND_025"]["median_gain"], 0.011)

    def test_identical_seeds_fail(self):
        rows, ref, arms = self.fixture()
        arms["BLEND_025"][:] = arms["BLEND_025"][0]
        m = summarize(rows, ref, arms)["arms"]["BLEND_025"]
        self.assertFalse(m["candidate"])
        self.assertFalse(m["gates"]["seed_independence"])

    def test_context_weighted_bootstrap(self):
        rows, ref, arms = self.fixture()
        # Last family has one context, so unequal family averaging is wrong.
        for a in ARMS:
            arms[a][:, -1] += 1
        m = summarize(rows, ref, arms)["arms"]["BLEND_025"]
        self.assertAlmostEqual(m["median_gain"], 0.011 + 1 / 91)
        self.assertLess(m["median_gain"], 0.011 + 1 / 19)

    def test_harm_and_trim_prevent_outlier_admission(self):
        rows, ref, arms = self.fixture()
        for a in ARMS:
            arms[a][:] = -0.02
            arms[a][:, :10] = 2
            arms[a] += np.arange(3)[:, None] * 0.0001
        m = summarize(rows, ref, arms)["arms"]["BLEND_025"]
        self.assertGreater(m["median_gain"], 0.001)
        self.assertFalse(m["gates"]["trim_gt_zero"])
        self.assertFalse(m["gates"]["harm_le_10pct"])

    def test_imported_fit_excludes_validation(self):
        from unittest.mock import patch
        import runner

        _, h = runner.imported()
        from pathlib import Path

        train = dict(context_sha256="train")
        val = dict(context_sha256="val")
        context = {
            k: dict(lm48_z=np.zeros(48), dope629=np.zeros(629))
            for k in ("train", "val")
        }
        pools = {k: dict(u=np.arange(200, dtype=float)) for k in ("train", "val")}
        captured = {}

        class Model:
            n_iter_ = 1

            def __init__(self, **params):
                captured["params"] = params

            def fit(self, x, y):
                captured["x"] = x.copy()
                captured["y"] = y.copy()
                return self

            def predict(self, x):
                captured["target"] = x.copy()
                return np.zeros(len(x))

            def get_params(self):
                return captured["params"]

        def program(pools, cid):
            return np.full((200, 412), 10 if cid == "train" else 99, dtype=np.float32)

        with (
            patch("sklearn.ensemble.HistGradientBoostingRegressor", Model),
            patch.object(h, "headroom"),
            patch.object(h, "check_rss"),
            patch.object(h, "release_memory"),
            patch.object(h, "program", program),
            patch.object(h, "sha", return_value="synthetic"),
            patch.object(h, "write"),
            patch.object(h.np, "savez_compressed"),
            patch.object(Path, "exists", return_value=False),
        ):
            h.fit_one(
                (135, (4,)),
                Path("/synthetic-no-io"),
                [train, val],
                context,
                pools,
                {"train": np.arange(4, 196)},
                {"train": 0, "val": 4},
            )
        self.assertEqual(captured["x"].shape, (192, 1089))
        self.assertTrue((captured["x"][:, :412] == 10).all())
        self.assertTrue((captured["target"][:, :412] == 99).all())
        self.assertEqual(captured["params"], dict(h.MODEL, random_state=135))
        np.testing.assert_array_equal(
            captured["y"], np.arange(4, 196) - np.arange(200).mean()
        )

    def test_guard_boundaries(self):
        s = dict(
            rss=0,
            threads=4,
            disk_percent=90,
            scratch=0,
            load1=1,
            cpus=10,
            available=32 * 1024**3,
        )
        self.assertIsNone(reason(s))
        for key, v in [
            ("rss", 8_000_000_000),
            ("threads", 5),
            ("disk_percent", 93.8),
            ("scratch", 2_000_000_000),
        ]:
            self.assertIsNotNone(reason(dict(s, **{key: v})))


if __name__ == "__main__":
    unittest.main()
