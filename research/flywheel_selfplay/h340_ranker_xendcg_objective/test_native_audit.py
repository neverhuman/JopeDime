import copy,json,unittest
from pathlib import Path
import native_audit as n

class NativeTests(unittest.TestCase):
 def setUp(self):self.r=json.loads((Path(__file__).parent/'native_baseline_reference.json').read_text())
 def test_existing_real_gbdt_dump_all_constructor_params_match(self):
  checked=n.compare_native(self.r['constructor_params'],self.r['native_params'],'LGBMRanker',False)
  self.assertEqual(set(checked),set(self.r['constructor_params'])-n.PYTHON_ONLY)
 def test_treatment_requires_native_xendcg(self):
  p=dict(self.r['constructor_params'],objective='rank_xendcg');v=dict(self.r['native_params'],objective='rank_xendcg')
  n.compare_native(p,v,'LGBMRanker')
  with self.assertRaises(AssertionError):n.compare_native(p,self.r['native_params'],'LGBMRanker')
 def test_any_other_constructor_native_drift_rejected(self):
  p=dict(self.r['constructor_params'],objective='rank_xendcg');v=dict(self.r['native_params'],objective='rank_xendcg')
  for k,x in [('seed','1'),('num_iterations','199'),('learning_rate','0.1'),('min_data_in_leaf','21'),('lambda_l2','10'),('feature_fraction','0.7')]:
   bad=dict(v);bad[k]=x
   with self.assertRaises(AssertionError):n.compare_native(p,bad,'LGBMRanker')
 def test_gpu_or_dart_or_label_gain_or_clip_rejected(self):
  p=dict(self.r['constructor_params'],objective='rank_xendcg');v=dict(self.r['native_params'],objective='rank_xendcg')
  for k,x in [('path_smooth','20'),('device_type','gpu'),('boosting','dart'),('label_gain','0,1,2'),('max_delta_step','1')]:
   with self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')

if __name__=='__main__':unittest.main()
