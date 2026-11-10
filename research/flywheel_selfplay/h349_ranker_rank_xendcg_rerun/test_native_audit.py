import json,unittest
from pathlib import Path
import native_audit as n
class NativeTests(unittest.TestCase):
 def setUp(self):self.r=json.loads((Path(__file__).parent/'native_baseline_reference.json').read_text())
 def treatment(self):
  p=dict(self.r['constructor_params'],objective='rank_xendcg');p['objective_seed']=p['random_state']
  v=dict(self.r['native_params'],objective='rank_xendcg',objective_seed=str(p['random_state']))
  return p,v
 def test_real_base_dump_all_constructor_params_match(self):
  checked=n.compare_native(self.r['constructor_params'],self.r['native_params'],'LGBMRanker',False)
  self.assertEqual(set(checked),set(self.r['constructor_params'])-n.PYTHON_ONLY)
 def test_treatment_exact_objective_and_original_seed(self):
  p,v=self.treatment();n.compare_native(p,v,'LGBMRanker')
  with self.assertRaises(AssertionError):n.compare_native(p,self.r['native_params'],'LGBMRanker')
 def test_native_objective_seed_must_equal_original_random_state(self):
  p,v=self.treatment()
  with self.assertRaises(AssertionError):n.compare_native(p,dict(v,objective_seed='1'),'LGBMRanker')
  with self.assertRaises(AssertionError):n.compare_native(dict(p,objective_seed=p['random_state']+1),v,'LGBMRanker')
 def test_any_other_constructor_native_drift_rejected(self):
  p,v=self.treatment()
  for k,x in [('seed','1'),('num_iterations','199'),('learning_rate','0.1'),('min_data_in_leaf','21'),('lambda_l2','10'),('feature_fraction','0.7')]:
   with self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')
 def test_sigmoid_gpu_dart_label_gain_clip_rejected(self):
  p,v=self.treatment()
  for k,x in [('sigmoid','.5'),('min_sum_hessian_in_leaf','9.87'),('path_smooth','20'),('objective','lambdarank'),('device_type','gpu'),('boosting','dart'),('label_gain','0,1,2'),('max_delta_step','1')]:
   with self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')
if __name__=='__main__':unittest.main()
