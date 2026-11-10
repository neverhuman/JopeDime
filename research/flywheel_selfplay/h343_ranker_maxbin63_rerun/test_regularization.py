import ast
import inspect
import re
import unittest
import numpy as np
import lightgbm as lgb
import runtime as r
import regularization as w
from forecast import forecast


class MaxBinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G=r.initialize()
        cls.rank=cls.G['topk'].fit_ranker;cls.bag=cls.G['bagged'].fit_ranker_n
        cls.gate=cls.G['veto'].fit_gate;cls.caller=cls.G['veto'].run_cell_veto
        cls.scorer=cls.G['ens'].score_single_member_from_disk
        cls.dataset_class=lgb.sklearn.Dataset
        cls.transforms=w.install(cls.G)
        r.durable(r.P/'source/regularization_transform_preview.json',cls.transforms)
        cls.X=np.random.default_rng(341).normal(size=(640,12));cls.y=np.tile(np.arange(8),80)
        cls.base_dataset=lgb.Dataset(cls.X,label=cls.y,group=np.full(80,8),params={'max_bin':255,'num_threads':1},free_raw_data=False)
        cls.base_dataset.construct();cls.base_handle=cls.base_dataset._handle.value
        cls.models=[cls.G['topk'].fit_ranker(cls.X,cls.y,80,8,.05,3),cls.G['bagged'].fit_ranker_n(cls.X,cls.y,80,8,.1,50,7)]
        cls.gate_model=cls.G['veto'].fit_gate(cls.X,np.tile([False,True],320),9)

    def test_exact_one_keyword_inverse(self):
        for original,receipt in zip([type(self).rank,type(self).bag],self.transforms):
            base=ast.parse(inspect.getsource(original));changed=ast.parse(receipt['source'])
            a={k.arg:ast.dump(k.value) for k in w.constructor(base).keywords};b={k.arg:ast.dump(k.value) for k in w.constructor(changed).keywords}
            self.assertEqual(set(b)-set(a),{'max_bin'})
            self.assertTrue(all(a[k]==b[k] for k in a));self.assertTrue(receipt['restored_AST_exact'])

    def test_actual_native_maxbin_and_one_change(self):
        for m in self.models:
            row=m._h343_fit_receipt;w.check_record(row)
            self.assertEqual(row['changed_constructor_keys'],['max_bin'])
            self.assertEqual(m.booster_.params['max_bin'],63)
            self.assertIn('[max_bin: 63]',m.booster_.model_to_string())
            self.assertEqual(row['params']['reg_lambda'],1.)
            self.assertNotIn('label_gain',row['params'])

    def test_fresh_datasets_and_native_bins(self):
        ds=[m._h343_fit_receipt['dataset'] for m in self.models]
        self.assertNotEqual(ds[0]['serial'],ds[1]['serial'])
        for v in ds:
            self.assertEqual(v['new_objects'],1)
            self.assertTrue(v['reference_is_none'] and v['initial_handle_is_none'])
            self.assertLessEqual(v['max_observed_bins'],63)
        self.assertEqual(self.base_dataset._handle.value,self.base_handle)
        self.assertEqual(self.base_dataset.get_params()['max_bin'],255)
        self.assertGreater(max(self.base_dataset.feature_num_bin(i) for i in range(12)),63)
        self.assertIs(lgb.sklearn.Dataset,self.dataset_class)

    def test_prebinned_dataset_rejected(self):
        with self.assertRaises(AssertionError):self.G['topk'].fit_ranker(self.base_dataset,self.y,80,8,.05,3)

    def test_gate_and_callers_unchanged(self):
        self.assertIs(self.G['veto'].fit_gate,type(self).gate)
        self.assertIs(self.G['veto'].run_cell_veto,type(self).caller)
        self.assertIs(self.G['ens'].score_single_member_from_disk,type(self).scorer)
        self.assertIn('[max_bin: 255]',self.gate_model.booster_.model_to_string())
        self.assertNotIn('max_bin',self.gate_model.get_params())

    def test_learning_rate_rounds_and_seed_unchanged(self):
        for model,lr,n,seed in zip(self.models,[.05,.1],[200,50],[3,7]):
            p=model.get_params()
            self.assertEqual((p['learning_rate'],p['n_estimators'],p['random_state']),(lr,n,seed))
            self.assertEqual(model.booster_.params['num_iterations'],n)
            self.assertTrue(all(abs(t['shrinkage']-lr)<1e-12 for t in model.booster_.dump_model()['tree_info'][1:]))

    def test_grid_and_prediction_prefix_unchanged(self):
        self.assertEqual(self.G['core'].LR_GRID,(.03,.05,.1));self.assertEqual(self.G['core'].N_EST_GRID,(50,100,200))
        self.assertIn('num_iteration=n_est',inspect.getsource(type(self).scorer))
        self.assertNotIn('num_iteration=2',inspect.getsource(type(self).scorer))

    def test_weights_rejected(self):
        with self.assertRaises(AssertionError):self.G['topk'].fit_ranker(None,None,1,8,.1,1,sample_weight=np.ones(8))

    def test_forecast_deadline_strict(self):
        self.assertTrue(forecast(100,4,0,1636)['admit'])
        self.assertFalse(forecast(100,4,0,1635)['admit'])

    def test_forecast_full_remaining_and_minimum_cores(self):
        self.assertEqual(forecast(100,8,0,10000)['remaining_batches'],5)
        self.assertEqual(forecast(100,8,0,10000)['remaining_cells'],35)
        with self.assertRaises(AssertionError):forecast(100,3,0,10000)


if __name__=='__main__':unittest.main()
