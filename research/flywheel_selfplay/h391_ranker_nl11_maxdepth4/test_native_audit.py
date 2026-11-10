import json,unittest
from pathlib import Path
import native_audit as n
class NativeTests(unittest.TestCase):
 def setUp(self):
  self.r=json.loads((Path(__file__).parent/'native_baseline_reference.json').read_text());self.r['constructor_params']['num_leaves']=15;self.r['native_params']['num_leaves']='15'
 def treatment(self):return dict(self.r['constructor_params'],num_leaves=11,max_depth=4),dict(self.r['native_params'],num_leaves='11',max_depth='4')
 def test_NL15_all_constructor_fields_match(self):
  keys=n.compare_native(self.r['constructor_params'],self.r['native_params'],'LGBMRanker',False);self.assertEqual(set(keys),set(self.r['constructor_params'])-n.PYTHON_ONLY)
 def test_treatment_requires_both_capacity_parameters(self):
  p,v=self.treatment();n.compare_native(p,v,'LGBMRanker')
  for k,x in [('num_leaves','15'),('num_leaves','12'),('max_depth','3'),('max_depth','5')]:
   with self.subTest(k=k,x=x),self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')
 def test_other_default_or_constructor_drift_rejected(self):
  p,v=self.treatment()
  for k,x in [('seed','1'),('num_iterations','199'),('learning_rate','0.1'),('min_data_in_leaf','40'),('lambda_l2','5'),('lambda_l1','.5'),('feature_fraction','.7')]:
   with self.subTest(k=k),self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')
 def test_GPU_or_extra_interventions_rejected(self):
  p,v=self.treatment()
  for k,x in [('sigmoid','.5'),('path_smooth','20'),('objective','rank_xendcg'),('device_type','gpu'),('boosting','dart'),('label_gain','0,1,2'),('max_delta_step','1'),('max_delta_step','0.5'),('min_gain_to_split','.1'),('min_sum_hessian_in_leaf','9.87'),('bagging_seed','0'),('extra_trees','1'),('lambdarank_norm','0')]:
   with self.subTest(k=k,x=x),self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')
 def test_real_veto_retains_L1one_L2two(self):
  r=json.loads((Path(__file__).parent/'base_veto_native_reference.json').read_text());n.compare_native(r['constructor_params'],r['native_params'],'LGBMClassifier')
  with self.assertRaises(AssertionError):n.compare_native(r['constructor_params'],dict(r['native_params'],lambda_l1='0'),'LGBMClassifier')
 def test_veto_native_drift_rejected(self):
  r=json.loads((Path(__file__).parent/'base_veto_native_reference.json').read_text())
  for k,v in [('extra_trees','1'),('feature_fraction','0.6'),('max_delta_step','1'),('max_depth','4'),('bagging_seed','0')]:
   with self.subTest(k=k),self.assertRaises(AssertionError):n.compare_native(r['constructor_params'],dict(r['native_params'],**{k:v}),'LGBMClassifier')
 def model(self,leaves,configured=11,depth=4):return f'Tree=0\nnum_leaves={leaves}\n[num_leaves: {configured}]\n[max_depth: {depth}]\n'
 def test_all_realized_trees_one_through_eleven_valid(self):
  for i in range(1,12):self.assertEqual(n.observed_leaves(self.model(i),'LGBMRanker')['maximum'],i)
 def test_twelve_or_wrong_depth_rejected(self):
  for text in [self.model(12),self.model(8,11,3)]:
   with self.assertRaises(AssertionError):n.observed_leaves(text,'LGBMRanker')
 def test_NL15_and_BASE_bounds_retained(self):
  for configured,limit in [(15,8),(7,7)]:
   self.assertEqual(n.observed_leaves(self.model(limit,configured,3),'LGBMRanker')['maximum'],limit)
   with self.assertRaises(AssertionError):n.observed_leaves(self.model(limit+1,configured,3),'LGBMRanker')
if __name__=='__main__':unittest.main()
