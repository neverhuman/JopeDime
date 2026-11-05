import json,unittest
from pathlib import Path
import native_audit as n
class NativeTests(unittest.TestCase):
 def setUp(self):self.r=json.loads((Path(__file__).parent/'native_baseline_reference.json').read_text())
 def treatment(self):
  r=json.loads((Path(__file__).parent/'nl15-native-reference.json').read_text());p=dict(r['constructor_params']);v=dict(r['native_params']);p['subsample_freq']=5;v['bagging_freq']='5';return p,v
 def test_existing_BASE_all_constructor_fields_match(self):
  keys=n.compare_native(self.r['constructor_params'],self.r['native_params'],'LGBMRanker',False);self.assertEqual(set(keys),set(self.r['constructor_params'])-n.PYTHON_ONLY)
 def test_treatment_requires_native15(self):
  p,v=self.treatment();n.compare_native(p,v,'LGBMRanker')
  with self.assertRaises(AssertionError):n.compare_native(p,self.r['native_params'],'LGBMRanker')
 def test_other_default_or_constructor_drift_rejected(self):
  p,v=self.treatment()
  for k,x in [('seed','1'),('num_iterations','199'),('learning_rate','0.123'),('min_data_in_leaf','40'),('lambda_l2','5'),('lambda_l1','.5'),('feature_fraction','.7'),('max_depth','4')]:
   with self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')
 def test_gpu_sigmoid_xendcg_or_extra_interventions_rejected(self):
  p,v=self.treatment()
  for k,x in [('sigmoid','.5'),('path_smooth','20'),('bagging_freq','1'),('objective','rank_xendcg'),('device_type','gpu'),('boosting','dart'),('label_gain','0,1,2'),('max_delta_step','1'),('min_gain_to_split','.1'),('min_sum_hessian_in_leaf','9.87')]:
   with self.assertRaises(AssertionError):n.compare_native(p,dict(v,**{k:x}),'LGBMRanker')
 def test_real_veto_retains_L1one_L2two(self):
  r=json.loads((Path(__file__).parent/'base_veto_native_reference.json').read_text());n.compare_native(r['constructor_params'],r['native_params'],'LGBMClassifier')
  with self.assertRaises(AssertionError):n.compare_native(r['constructor_params'],dict(r['native_params'],lambda_l1='0'),'LGBMClassifier')
 def test_full_NL15_header_only_freq_delta(self):
  from unittest.mock import patch
  p,v=self.treatment();base=dict(v,bagging_freq='1');canonical={k:x for k,x in base.items() if k not in ('learning_rate','num_iterations')}
  with patch.object(n,'_NL15_HEADERS',{'LGBMRanker:'+v['seed']:canonical}):
   self.assertEqual(n.compare_NL15_header(v,'LGBMRanker'),{'bagging_freq':['1','5']})
   for key,value in [('bagging_fraction','.7'),('feature_fraction','.7'),('lambda_l2','5'),('bagging_seed','999')]:
    with self.assertRaises(AssertionError):n.compare_NL15_header(dict(v,**{key:value}),'LGBMRanker')
 def test_LRH_rates_and_rounds_are_not_active(self):
  p,v=self.treatment()
  with self.assertRaises(AssertionError):n.compare_native(dict(p,learning_rate=.015),dict(v,learning_rate='.015'),'LGBMRanker')
  with self.assertRaises(AssertionError):n.compare_native(dict(p,n_estimators=400),dict(v,num_iterations='400'),'LGBMRanker')
 def model(self,leaves,configured=15):return f'Tree=0\nnum_leaves={leaves}\n[num_leaves: {configured}]\n[max_depth: 3]\n'
 def test_depth3_limits_actual15_setting_to8(self):
  s=n.observed_leaves(self.model(8),'LGBMRanker');self.assertEqual(s['maximum'],8);self.assertEqual(s['configured_num_leaves'],15)
 def test_9_actual_leaves_rejected_at_depth3(self):
  with self.assertRaises(AssertionError):n.observed_leaves(self.model(9),'LGBMRanker')
 def test_BASE7_actual_bound(self):
  self.assertEqual(n.observed_leaves(self.model(7,7),'LGBMRanker')['maximum'],7)
  with self.assertRaises(AssertionError):n.observed_leaves(self.model(8,7),'LGBMRanker')
if __name__=='__main__':unittest.main()
