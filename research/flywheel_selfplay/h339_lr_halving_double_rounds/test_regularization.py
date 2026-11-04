import ast
import inspect
import json
import unittest
import numpy as np
import runtime as r
import regularization as w
from forecast import forecast


class LRHalfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = r.initialize()
        cls.rank = cls.G['topk'].fit_ranker
        cls.bag = cls.G['bagged'].fit_ranker_n
        cls.gate = cls.G['veto'].fit_gate
        cls.caller = cls.G['veto'].run_cell_veto
        cls.scorer = cls.G['ens'].score_single_member_from_disk
        cls.transforms = w.install(cls.G)
        r.durable(r.P / 'source/regularization_transform_preview.json', cls.transforms)

    def test_exact_two_keyword_inverse(self):
        for original, receipt in zip([type(self).rank, type(self).bag], self.transforms):
            base = ast.parse(inspect.getsource(original))
            changed = ast.parse(receipt['source'])
            a = {k.arg: ast.dump(k.value) for k in w.constructor(base).keywords}
            b = {k.arg: ast.dump(k.value) for k in w.constructor(changed).keywords}
            self.assertEqual({k for k in a if a[k] != b[k]}, {'learning_rate', 'n_estimators'})
            self.assertTrue(receipt['restored_AST_exact'])

    def test_gate_and_scoring_callers_unchanged(self):
        self.assertIs(self.G['veto'].fit_gate, type(self).gate)
        self.assertIs(self.G['veto'].run_cell_veto, type(self).caller)
        self.assertIs(self.G['ens'].score_single_member_from_disk, type(self).scorer)
        self.assertEqual(self.G['core'].LR_GRID, (.03, .05, .1))
        self.assertEqual(self.G['core'].N_EST_GRID, (50, 100, 200))

    def test_actual_native_topk_and_bag_fit(self):
        x = np.random.default_rng(339).normal(size=(320, 12))
        y = np.tile(np.arange(8), 40)
        for fn, args, expected_n in [(self.G['topk'].fit_ranker, (x,y,40,8,.05,3),400), (self.G['bagged'].fit_ranker_n,(x,y,40,8,.1,50,7),100)]:
            m = fn(*args)
            self.assertEqual(m.get_params()['n_estimators'], expected_n)
            self.assertEqual(m.get_params()['learning_rate'], args[4]/2)
            self.assertEqual(m.booster_.params['num_iterations'], expected_n)
            self.assertEqual(m.booster_.num_trees(), expected_n)
            self.assertNotIn('label_gain', m.get_params())
            self.assertEqual(m.get_params()['reg_lambda'], 1.)
            self.assertTrue(all(abs(t['shrinkage']-args[4]/2)<1e-12 for t in m.booster_.dump_model()['tree_info'][1:]))
            # Unchanged callers deliberately request BASE-grid prefixes, not doubled ones.
            self.assertGreater(np.max(np.abs(m.booster_.predict(x,num_iteration=50)-m.booster_.predict(x,num_iteration=100))),1e-6)

    def test_selected_prefix_calls_stay_literal(self):
        for key in ['veto','ens']:
            text = inspect.getsource(self.G[key])
            self.assertIn('num_iteration=',text)
        text=inspect.getsource(type(self).scorer)
        self.assertIn('num_iteration=n_est',text)
        self.assertNotIn('num_iteration=2',text)

    def test_nominal_full_budget_product(self):
        for lr in [.03,.05,.1]:
            for n in [50,100,200]:
                self.assertEqual(lr*n,(lr/2)*(2*n))
                self.assertEqual((lr/2)*n,lr*n/2)

    def test_weights_rejected(self):
        with self.assertRaises(AssertionError):
            self.G['topk'].fit_ranker(None,None,1,8,.1,1,sample_weight=np.ones(8))

    def test_forecast_admit_strictly_before_deadline(self):
        good=forecast(100,5,0,1406)
        self.assertTrue(good['admit'])
        self.assertEqual(good['remaining_batches'],7)
        self.assertFalse(forecast(100,5,0,1405)['admit'])

    def test_forecast_uses_all_remaining_cells_and_reserve(self):
        v=forecast(100,4,0,100000)
        self.assertEqual(v['remaining_cells'],35)
        self.assertEqual(v['remaining_batches'],9)
        self.assertEqual(v['projected_finish_epoch'],1635)


if __name__=='__main__':
    unittest.main()
