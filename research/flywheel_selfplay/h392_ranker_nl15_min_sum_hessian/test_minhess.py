"""No-fit controls for the percentile, native alias, and complete committed inverse."""
from pathlib import Path
import ast,json,subprocess,unittest
import numpy as np
from derive_minhess import pooled_value,leaf_weights
import depth_patch as p
from test_depth import source
class MinHessTests(unittest.TestCase):
 def test_quantile_uses_all_leaves_and_linear_interpolation(self):
  q,v=pooled_value([1,2,4,100]);self.assertEqual(q,1.75);self.assertEqual(v,1.75)
 def test_three_significant_figures_once(self):
  q,v=pooled_value([1.23456]*4);self.assertEqual(q,1.23456);self.assertEqual(v,1.23)
 def test_internal_node_weight_excluded(self):
  node=dict(internal_weight=999,left_child=dict(leaf_weight=2),right_child=dict(internal_weight=888,left_child=dict(leaf_weight=4),right_child=dict(leaf_weight=8)))
  self.assertEqual(list(leaf_weights(node)),[2,4,8])
 def test_invalid_weights_refused(self):
  for values in [[],[-1,2],[np.nan,1],[np.inf,1]]:
   with self.assertRaises(AssertionError):pooled_value(values)
 def test_archived_scope_and_single_value(self):
  r=json.loads((Path(__file__).parent/'hessian_value.json').read_text());self.assertEqual(r['models'],12636);self.assertEqual(r['inner_models'],11664);self.assertEqual(r['final_models'],972);self.assertEqual(r['calibration_fits'],0);self.assertGreater(r['leaves'],12636);self.assertGreater(r['value'],r['baseline']);self.assertEqual(float(format(r['raw_q25'],'.3g')),r['value'])
 def test_native_alias_is_installed_default(self):
  import lightgbm as lgb
  self.assertEqual(lgb.LGBMRanker().get_params()['min_child_weight'],p.BASE_HESS)
 def test_inverse_equals_actual_committed_H352_complete_AST(self):
  r=Path('/home/ubuntu/JopeDime');ref='02dfee2f7e0b8543324494dd4f99d074cd6b93b6';name='research/flywheel_selfplay/h352_ranker_num_leaves_15/leaves_patch.py';raw=subprocess.check_output(['git','show',ref+':'+name],cwd=r,text=True);ns={};exec(compile(raw,name,'exec'),ns)
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   _,expected=ns['transform'](source(file,fun));old,new=p.transform(source(file,fun));self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(old,include_attributes=False));self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(p.strip(new),include_attributes=False))
if __name__=='__main__':unittest.main()
