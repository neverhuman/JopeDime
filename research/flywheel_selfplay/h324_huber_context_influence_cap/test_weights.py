import unittest, tempfile, json
from pathlib import Path
import numpy as np
import runtime as r
import weights as w


def packets(spreads):
    return {str(i): {"utility_k": np.linspace(-2, -2+s, 8)} for i,s in enumerate(spreads)}


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

    def test_exact_cap_formula_and_slot_repeat(self):
        x=w.context_weights(["0","1","2"], packets([1,2,10]),8).reshape(3,8)
        np.testing.assert_allclose(x[:,0],np.array([1,1,.2])/(2.2/3))
        np.testing.assert_array_equal(x[:,0],x[:,7])
        self.assertAlmostEqual(x.mean(),1)

    def test_duplicate_occurrences_change_median(self):
        x=w.context_weights(["0","1","1"],packets([1,10]),8)
        np.testing.assert_array_equal(x,np.ones(24))

    def test_recompute_subset_median(self):
        pp=packets([1,2,10]);a=w.context_weights(["0","1","2"],pp,8);b=w.context_weights(["0","2"],pp,8)
        self.assertNotEqual(a[-1],b[-1])

    def test_zero_spread_context_kept_when_delta_positive(self):
        x=w.context_weights(["0","1","2"],packets([0,1,2]),8).reshape(3,8)
        np.testing.assert_allclose(x[:,0],[1.2,1.2,.6])

    def test_zero_median_fails_closed(self):
        with self.assertRaisesRegex(AssertionError,'ZERO_MEDIAN_SPREAD'):
            w.context_weights(["0","1","2"],packets([0,0,2]),8)

    def test_forbidden_context_denied(self):
        w.ALLOWED={"0"}
        with self.assertRaises(AssertionError):w.context_weights(["1"],packets([1,2]),8)

    def test_nonfinite_utilities_denied(self):
        pp=packets([1]);pp['0']['utility_k'][2]=float('nan')
        with self.assertRaises(AssertionError):w.context_weights(['0'],pp,8)

    def test_nonfinite_or_bad_weight_denied(self):
        for x in [[float("nan")],[0],[2],[]]:
            with self.assertRaises(AssertionError):w.validate(x,1,'test')

    def test_exact_routes_and_unweighted_gate(self):
        self.assertEqual([x['sites'] for x in self.transforms],[1,1,2,1])
        self.assertEqual(sum(x['source'].count('_h324_weights(') for x in self.transforms),3)
        self.assertIs(self.G['veto'].fit_gate,type(self).oldgate)
        self.assertNotIn('fit_gate',[x['function'] for x in self.transforms])

    def test_uniform_ranker_weights_preserve_predictions_and_objective(self):
        X=np.random.default_rng(324).normal(size=(160,4));y=np.tile(np.arange(8),20)
        a=type(self).oldrank(X,y,20,8,.05,5)
        b=self.G['topk'].fit_ranker(X,y,20,8,.05,5,sample_weight=np.ones(160))
        np.testing.assert_array_equal(a.predict(X),b.predict(X))
        self.assertEqual(a.get_params(),b.get_params())
        self.assertEqual(b.get_params()['objective'],'lambdarank')

    def test_uniform_bag_weights_preserve_predictions(self):
        X=np.random.default_rng(325).normal(size=(160,4));y=np.tile(np.arange(8),20)
        a=type(self).oldbag(X,y,20,8,.05,20,5)
        b=self.G['bagged'].fit_ranker_n(X,y,20,8,.05,20,5,sample_weight=np.ones(160))
        np.testing.assert_array_equal(a.predict(X),b.predict(X))

    def test_nonuniform_weights_change_actual_ranker(self):
        X=np.tile(np.arange(8),(20,1)).reshape(-1,1).astype(float)
        y=np.concatenate([np.arange(8) if i<10 else np.arange(7,-1,-1) for i in range(20)])
        pp=packets([1]*10+[10]*10);ww=w.context_weights(list(pp),pp,8)
        a=type(self).oldrank(X,y,20,8,.05,5)
        b=self.G['topk'].fit_ranker(X,y,20,8,.05,5,sample_weight=ww)
        self.assertGreater(float(np.max(np.abs(a.predict(X)-b.predict(X)))),1e-6)

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
            self.assertEqual(len(receipt["roles"]), 2)
            self.assertEqual(len(m["bags"]), 8)
            self.assertEqual(receipt["fit_calls"], 21)
            self.assertIs(g["veto"].fit_gate, type(self).oldgate)
            w.AUDIT = None
            w.ALLOWED = None


if __name__=="__main__":unittest.main()
