import ast,subprocess,unittest
from pathlib import Path
import gain_patch as p
W=Path('/home/ubuntu/JopeDime');PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'
def source(file,fun):
 raw=subprocess.check_output(['git','-C',str(W),'show',PIN+':'+PREFIX+file],text=True)
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==fun)
 return ast.get_source_segment(raw,node)
def kw(t):return {k.arg:ast.dump(k.value,include_attributes=False) for k in p.calls(t)[0].keywords}
class GainTests(unittest.TestCase):
 def test_both_full_BASE_asts_restore(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(p.strip(new),include_attributes=False))
 def test_revert_equals_committed_H352_AST(self):
  text=subprocess.check_output(['git','-C',str(W),'show','02dfee2f7e:research/flywheel_selfplay/h352_ranker_num_leaves_15/leaves_patch.py'],text=True);namespace={};exec(compile(text,'<committed-H352>','exec'),namespace)
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   src=source(file,fun);_,want=namespace['transform'](src);_,got=p.transform(src)
   self.assertEqual(ast.dump(want,include_attributes=False),ast.dump(p.strip(got),include_attributes=False))
 def test_only_min_split_gain_keyword_added(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));a,b=kw(old),kw(new)
   self.assertNotIn('min_split_gain',a);self.assertEqual(set(b),set(a)|{'min_split_gain'});self.assertEqual(a,{k:v for k,v in b.items() if k!='min_split_gain'});self.assertEqual(b['min_split_gain'],ast.dump(ast.Constant(.001),include_attributes=False));self.assertNotIn('bagging_by_query',b);self.assertNotIn('bagging_seed',b)
 def test_regularization_depth_and_minchild_BASE(self):
  _,new=p.transform(source('topk.py','fit_ranker'));v={k.arg:k.value for k in p.calls(new)[0].keywords}
  for k,w in [('reg_lambda',1.0),('max_depth',3),('min_child_samples',20),('metric','ndcg')]:self.assertEqual(ast.literal_eval(v[k]),w)
  for k in ['reg_alpha','objective_seed','sigmoid','min_sum_hessian_in_leaf']:self.assertNotIn(k,v)
 def test_second_override_rejected(self):
  _,new=p.transform(source('topk.py','fit_ranker'))
  with self.assertRaises(AssertionError):p.transform(ast.unparse(new))
 def test_wrong_BASE_l2_rejected(self):
  with self.assertRaises(AssertionError):p.transform(source('topk.py','fit_ranker').replace('reg_lambda=1.0','reg_lambda=5.0'))
 def test_classifier_not_patched(self):
  with self.assertRaises(AssertionError):p.transform('def f(): return lgb.LGBMClassifier(num_leaves=7)')
if __name__=='__main__':unittest.main()
