import unittest,json,copy
from pathlib import Path
import evaluate_h392 as e
import runtime_h392 as rt
import native_audit as n
from depth_patch import MINHESS,BASE_HESS
class SecondaryTests(unittest.TestCase):
 def primary(self,x=.003,lcb=.001):return dict(median=x,lcb95=lcb,per_cell=[x]*9)
 def run_case(self,p=None,s=None,harm=0):return e.decide_with_secondary(p or self.primary(),s or self.primary(.001,.0001),[1],[1],[harm],[0])
 def test_primary_NL15_required(self):self.assertEqual(self.run_case()[1],'ADVANCE')
 def test_secondary_BASE_zero_LCB_does_not_block(self):self.assertEqual(self.run_case(s=self.primary(.01,0))[1],'ADVANCE')
 def test_secondary_BASE_zero_median_does_not_block(self):self.assertEqual(self.run_case(s=self.primary(0,.001))[1],'ADVANCE')
 def test_secondary_positive_does_not_override_primary_margin(self):
  gates,verdict,over,primary=self.run_case(p=self.primary(.00199));self.assertTrue(over);self.assertEqual(verdict,'MINHESS_NOT_ADVANCED');self.assertEqual(primary,verdict)
 def test_secondary_positive_does_not_override_harm(self):self.assertEqual(self.run_case(harm=.01)[1],'MINHESS_NOT_ADVANCED')
 def test_fit_cannot_read_archived_NL15(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h392-hwv/nl15/NL15_P0.npy','r',0))
 def native(self):
  ref=json.loads((Path(__file__).parent/'h352_native_reference.json').read_text())['roles']['LGBMRanker'];seed=next(iter(ref['seed_fields']));native=dict(ref['constant_native_fields'])
  for key,values in ref['seed_fields'][seed].items():native[key]=values[0]
  native['num_iterations']='200';native['max_depth']='3';native['lambda_l1']='0';native['min_sum_hessian_in_leaf']=str(MINHESS);return {'random_state':int(seed)},native
 def test_full_native_H352_defaults_with_only_Hessian_floor_change(self):
  params,native=self.native();record=n.compare_h352_defaults(params,native,'LGBMRanker');self.assertEqual(record['only_intervention'],{'min_sum_hessian_in_leaf':[BASE_HESS,MINHESS]})
 def test_hidden_native_default_drift_rejected(self):
  params,native=self.native()
  for key,value in [('max_bin','19'),('num_leaves','7'),('max_depth','4'),('min_data_in_bin','9'),('lambda_l1','0.5'),('min_sum_hessian_in_leaf',str(BASE_HESS))]:
   with self.assertRaises(AssertionError):n.compare_h352_defaults(params,dict(native,**{key:value}),'LGBMRanker')
 def test_unpinned_seed_rejected(self):
  params,native=self.native()
  with self.assertRaises(AssertionError):n.compare_h352_defaults({'random_state':-777},native,'LGBMRanker')
if __name__=='__main__':unittest.main()
