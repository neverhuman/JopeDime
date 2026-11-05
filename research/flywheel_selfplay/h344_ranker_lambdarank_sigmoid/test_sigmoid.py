import ast,subprocess,unittest
from pathlib import Path
import sigmoid_patch as p
W=Path('/home/ubuntu/JopeDime');PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'

def source(file,fun):
 raw=subprocess.check_output(['git','-C',str(W),'show',PIN+':'+PREFIX+file],text=True)
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==fun)
 return ast.get_source_segment(raw,node)

def kw(tree):return {x.arg:ast.dump(x.value,include_attributes=False) for x in p.calls(tree)[0].keywords}

class MinHessTests(unittest.TestCase):
 def test_both_pinned_full_function_asts_restore(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun),p.value());self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(p.strip(new),include_attributes=False))
 def test_every_original_keyword_and_round_stream_unchanged(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun),p.value());before=kw(old);after=kw(new)
   self.assertEqual({k:v for k,v in after.items() if k!='sigmoid'},before)
   self.assertEqual(set(after)-set(before),{'sigmoid'})
   self.assertEqual(after['sigmoid'],ast.dump(ast.Constant(float(p.value())),include_attributes=False))
   for x in ['path_smooth','min_child_weight','min_sum_hessian_in_leaf','label_gain','drop_rate','skip_drop','extra_trees','max_delta_step','callbacks','early_stopping_rounds']:self.assertNotIn(x,after)
 def test_default_gbdt_and_minleaf20_retained(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   _,tree=p.transform(source(file,fun),p.value());values={x.arg:x.value for x in p.calls(tree)[0].keywords}
   self.assertEqual(ast.literal_eval(values['boosting_type']),'gbdt');self.assertEqual(ast.literal_eval(values['min_child_samples']),20)
 def test_duplicate_keyword_rejected(self):
  _,new=p.transform(source('topk.py','fit_ranker'),p.value())
  with self.assertRaises(AssertionError):p.transform(ast.unparse(new),p.value())
 def test_dart_boosting_rejected(self):
  bad=source('topk.py','fit_ranker').replace('"gbdt"','"dart"')
  with self.assertRaises(AssertionError):p.transform(bad,p.value())
 def test_classifier_only_function_rejected(self):
  with self.assertRaises(AssertionError):p.transform('def f(): return lgb.LGBMClassifier(boosting_type="gbdt")',p.value())

if __name__=='__main__':unittest.main()
