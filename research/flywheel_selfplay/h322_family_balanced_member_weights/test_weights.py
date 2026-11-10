import unittest, tempfile, json
from pathlib import Path
import numpy as np
import runtime as r
import weights as w


class WeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = r.initialize()
        cls.oldrank = cls.G["topk"].fit_ranker
        cls.oldbag = cls.G["bagged"].fit_ranker_n
        cls.oldgate = cls.G["veto"].fit_gate
        cls.transforms = w.install(cls.G)
        r.durable(r.P / "source/weight_transform_preview.json", cls.transforms)

    def setUp(self):
        w.AUDIT = None
        w.ALLOWED = None

    def test_family_mass_and_slot_repeat(self):
        x = w.family_weights(
            ["a", "b", "c", "d"], dict(a="F", b="F", c="F", d="G"), 8
        ).reshape(4, 8)
        np.testing.assert_allclose(x[:3].sum(), x[3:].sum())
        self.assertEqual(x.mean(), 1)
        np.testing.assert_array_equal(x[:, 0], x[:, 7])

    def test_duplicate_occurrences_counted(self):
        x = w.family_weights(["a", "a", "b"], dict(a="F", b="G"), 2).reshape(3, 2)
        np.testing.assert_allclose(x[:, 0], [0.75, 0.75, 1.5])
        self.assertEqual(x[:2].sum(), x[2:].sum())

    def test_recompute_within_subset(self):
        f = dict(a="F", b="F", c="G")
        np.testing.assert_allclose(w.family_weights(["a", "c"], f, 2), 1)

    def test_forbidden_context_denied(self):
        w.ALLOWED = {"a"}
        with self.assertRaises(AssertionError):
            w.family_weights(["b"], {"b": "F"}, 8)

    def test_nonfinite_or_bad_weight_denied(self):
        for x in [[float("nan")], [0], [2], []]:
            with self.assertRaises(AssertionError):
                w.validate(x, 1, "test")

    def test_exact_fitting_callsite_inventory(self):
        self.assertEqual([x["sites"] for x in self.transforms], [1, 1, 1, 4, 1])
        self.assertEqual(
            sum(x["source"].count("_h322_weights(") for x in self.transforms), 5
        )

    def test_uniform_ranker_weights_preserve_predictions(self):
        X = np.random.default_rng(322).normal(size=(160, 4))
        y = np.tile(np.arange(8), 20)
        a = type(self).oldrank(X, y, 20, 8, 0.05, 5)
        b = self.G["topk"].fit_ranker(X, y, 20, 8, 0.05, 5, sample_weight=np.ones(160))
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_uniform_bag_weights_preserve_predictions(self):
        X = np.random.default_rng(323).normal(size=(160, 4))
        y = np.tile(np.arange(8), 20)
        a = type(self).oldbag(X, y, 20, 8, 0.05, 20, 5)
        b = self.G["bagged"].fit_ranker_n(
            X, y, 20, 8, 0.05, 20, 5, sample_weight=np.ones(160)
        )
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_weights_change_actual_classifier_fit(self):
        X = np.zeros((80, 4))
        y = np.array([False] * 64 + [True] * 16)
        cs = [str(i) for i in range(10)]
        fm = {c: "A" if int(c) < 8 else "B" for c in cs}
        ww = w.family_weights(cs, fm, 8)
        a = type(self).oldgate(X, y, 6)
        b = self.G["veto"].fit_gate(X, y, 6, sample_weight=ww)
        self.assertGreater(
            float(np.max(abs(a.predict_proba(X) - b.predict_proba(X)))), 0.1
        )

    def test_complete_member_routes_every_fit(self):
        with tempfile.TemporaryDirectory(dir=r.P) as td:
            rows = []
            pp = {}
            fm = {}
            for f in range(20):
                for j in range(1 + f % 3):
                    c = f"family{f}-ctx{j}"
                    rows.append(dict(context_sha256=c, family_id=f"family{f}"))
                    fm[c] = f"family{f}"
                    x = np.column_stack(
                        [np.arange(8), np.full(8, f), np.full(8, j)]
                    ).astype(float)
                    x = np.pad(x, ((0, 0), (0, 565)))
                    pp[c] = dict(
                        X=x,
                        rel_k=np.arange(8, dtype=np.int32),
                        utility_k=np.arange(8, dtype=float) / 10,
                        gate_k=np.ones(8, dtype=bool),
                        r882_u=0.0,
                    )
            w.AUDIT = Path(td) / "audit.jsonl"
            w.ALLOWED = set(fm)
            g = self.G
            m = g["r302"].fit_and_freeze_member(
                Path(td),
                0,
                30001,
                0,
                123,
                rows,
                pp,
                fm,
                np.concatenate([pp[c]["X"] for c in sorted(pp)[:4]]),
                g["core"],
                g["veto"],
                g["bagged"],
                g["ens"],
            )
            receipt = w.audit_summary(w.AUDIT)
            self.assertEqual(len(receipt["roles"]), 3)
            self.assertEqual(len(m["bags"]), 8)
            self.assertGreater(receipt["fit_calls"], 12)
            w.AUDIT = None
            w.ALLOWED = None


if __name__ == "__main__":
    unittest.main()
