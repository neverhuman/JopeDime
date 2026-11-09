import unittest, tempfile, json, inspect, ast
from pathlib import Path
import numpy as np
import runtime as r
import regularization as w

class CosineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G=r.initialize()
        cls.oldrank=cls.G['topk'].fit_ranker
        cls.oldbag=cls.G['bagged'].fit_ranker_n
        cls.oldgate=cls.G['veto'].fit_gate
        cls.oldveto=cls.G['veto'].run_cell_veto
        cls.oldmember=cls.G['r302'].fit_and_freeze_member
        cls.transforms=w.install(cls.G)
        r.durable(r.P/'source/regularization_transform_preview.json',cls.transforms)
    def setUp(self):w.AUDIT=None
    def test_exact_schedule_and_discrete_mean(self):
        for lr in [.03,.05,.10]:
            for t in [50,100,200]:
                values=w.rates(lr,t)
                self.assertEqual(values[0],2*lr)
                self.assertAlmostEqual(values[t//2],lr)
                self.assertAlmostEqual(values[-1],lr*(1-np.cos(np.pi/t)))
                self.assertAlmostEqual(sum(values)/t,lr*(1+1/t))
                self.assertTrue(all(a>b>0 for a,b in zip(values,values[1:])))
    def test_short_prefix_not_renormalized(self):
        self.assertNotEqual(w.rates(.05,200)[:50],w.rates(.05,50))
        self.assertGreater(np.mean(w.rates(.05,200)[:50]),.05*(1+1/50))
    def test_only_two_boundaries_change(self):
        self.assertEqual(len(self.transforms),2)
        self.assertTrue(all(x['restored_AST_exact'] for x in self.transforms))
        self.assertIs(self.G['veto'].fit_gate,type(self).oldgate)
        self.assertIs(self.G['veto'].run_cell_veto,type(self).oldveto)
        self.assertIs(self.G['r302'].fit_and_freeze_member,type(self).oldmember)
    def test_actual_ranker_constructor_unchanged_schedule_effective(self):
        X=np.random.default_rng(327).normal(size=(320,12));y=np.tile(np.arange(8),40)
        a=type(self).oldrank(X,y,40,8,.05,5);b=self.G['topk'].fit_ranker(X,y,40,8,.05,5)
        pa=a.get_params();pb=b.get_params()
        self.assertEqual({k for k in pa if pa[k]!=pb[k]},set())
        self.assertGreater(np.max(np.abs(a.predict(X)-b.predict(X))),1e-6)
        for key,value in [('lambda_l2',1),('feature_fraction',.8),('min_data_in_leaf',20)]:
            self.assertIn('['+key+': ',b.booster_.model_to_string())
        self.assertEqual(pb['objective'],'lambdarank')
        self.assertEqual(b._h331_rate_trace,w.rates(.05,200))
        self.assertEqual(b.booster_.params['learning_rate'],w.rates(.05,200)[-1])
        for t,tree in enumerate(b.booster_.dump_model()['tree_info']):
            if t>0:self.assertAlmostEqual(tree['shrinkage'],w.rates(.05,200)[t],places=7)
    def test_bag_selected_rounds_and_seed_retained(self):
        X=np.random.default_rng(328).normal(size=(320,12));y=np.tile(np.arange(8),40)
        a=type(self).oldbag(X,y,40,8,.1,20,7);b=self.G['bagged'].fit_ranker_n(X,y,40,8,.1,20,7)
        self.assertEqual({k for k in a.get_params() if a.get_params()[k]!=b.get_params()[k]},set())
        self.assertEqual(b.get_params()['n_estimators'],20)
        self.assertEqual(b.get_params()['random_state'],7)
        self.assertEqual(b._h331_rate_trace,w.rates(.1,20))
    def test_weights_rejected(self):
        with self.assertRaises(AssertionError):self.G['topk'].fit_ranker(None,None,1,8,.1,1,sample_weight=np.ones(8))
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



if __name__=="__main__":unittest.main()
