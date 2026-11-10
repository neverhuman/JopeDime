import ast,subprocess,unittest
from pathlib import Path
import capacity_patch as p
W=Path('/home/ubuntu/JopeDime');PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'
def source(file,name):
 raw=subprocess.check_output(['git','-C',str(W),'show',PIN+':'+PREFIX+file],text=True)
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.get_source_segment(raw,node)
def kw(tree):return {k.arg:ast.dump(k.value,include_attributes=False) for k in p.calls(tree)[0].keywords}
class CapacityTests(unittest.TestCase):
 def test_both_whole_H352_ASTs_restore(self):
  raw=subprocess.check_output(['git','-C',str(W),'show','02dfee2f7e:research/flywheel_selfplay/h352_ranker_num_leaves_15/leaves_patch.py'],text=True);ns={};exec(compile(raw,'<committed-H352>','exec'),ns)
  for file,name in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,name));_,baseline=ns['transform'](source(file,name))
   self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(baseline,include_attributes=False))
   self.assertEqual(ast.dump(p.strip(new),include_attributes=False),ast.dump(baseline,include_attributes=False))
 def test_only_two_existing_keywords_change(self):
  for file,name in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,name));a,b=kw(old),kw(new);self.assertEqual(set(a),set(b));self.assertEqual({k for k in a if a[k]!=b[k]},{'num_leaves','max_depth'})
   self.assertEqual(ast.literal_eval(p.calls(new)[0].keywords[[k.arg for k in p.calls(new)[0].keywords].index('num_leaves')].value),11)
   self.assertEqual(ast.literal_eval(p.calls(new)[0].keywords[[k.arg for k in p.calls(new)[0].keywords].index('max_depth')].value),4)
 def test_other_NL15_hyperparameters_unchanged(self):
  _,new=p.transform(source('topk.py','fit_ranker'));v={k.arg:k.value for k in p.calls(new)[0].keywords}
  for k,w in [('num_leaves',11),('max_depth',4),('min_child_samples',20),('reg_lambda',1.0),('subsample',.8),('subsample_freq',1),('colsample_bytree',.8)]:self.assertEqual(ast.literal_eval(v[k]),w)
  for k in ['extra_trees','extra_seed','feature_fraction_seed','bagging_by_query','reg_alpha','max_delta_step']:self.assertNotIn(k,v)
 def test_wrong_baseline_fraction_rejected(self):
  with self.assertRaises(AssertionError):p.transform(source('topk.py','fit_ranker').replace('colsample_bytree=0.8','colsample_bytree=0.6'))
 def test_wrong_BASE_l2_rejected(self):
  with self.assertRaises(AssertionError):p.transform(source('topk.py','fit_ranker').replace('reg_lambda=1.0','reg_lambda=5.0'))
 def test_second_override_rejected(self):
  _,new=p.transform(source('topk.py','fit_ranker'))
  with self.assertRaises(AssertionError):p.transform(ast.unparse(new))
 def test_native_alias_override_rejected(self):
  for alias in ['max_delta_step','max_tree_output','max_leaf_output']:
   s=source('topk.py','fit_ranker').replace('colsample_bytree=0.8',f'colsample_bytree=0.8,{alias}=2')
   with self.subTest(alias=alias),self.assertRaises(AssertionError):p.transform(s)
 def test_classifier_not_patched(self):
  with self.assertRaises(AssertionError):p.transform('def f(): return lgb.LGBMClassifier(colsample_bytree=.8)')
if __name__=='__main__':unittest.main()
