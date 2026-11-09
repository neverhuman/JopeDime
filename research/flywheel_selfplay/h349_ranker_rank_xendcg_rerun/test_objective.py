import ast,subprocess,unittest
from pathlib import Path
import objective_patch as p
W=Path('/home/ubuntu/JopeDime');PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'
def source(file,fun):
 raw=subprocess.check_output(['git','-C',str(W),'show',PIN+':'+PREFIX+file],text=True)
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==fun)
 return ast.get_source_segment(raw,node)
def kw(t):return {x.arg:ast.dump(x.value,include_attributes=False) for x in p.calls(t)[0].keywords}
class ObjectiveTests(unittest.TestCase):
 def test_complete_pinned_asts_restore(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(p.strip(new),include_attributes=False))
 def test_only_objective_and_seed_delta(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));a,b=kw(old),kw(new)
   self.assertEqual({k:v for k,v in a.items() if k!='objective'},{k:v for k,v in b.items() if k not in ('objective','objective_seed')})
   self.assertEqual(set(b)-set(a),{'objective_seed'});self.assertEqual(b['objective_seed'],a['random_state'])
   self.assertEqual(b['objective'],ast.dump(ast.Constant('rank_xendcg'),include_attributes=False));self.assertNotIn('sigmoid',b)
 def test_original_gbdt_minleaf_and_metric(self):
  _,new=p.transform(source('topk.py','fit_ranker'));v={x.arg:x.value for x in p.calls(new)[0].keywords}
  self.assertEqual(ast.literal_eval(v['boosting_type']),'gbdt');self.assertEqual(ast.literal_eval(v['metric']),'ndcg');self.assertEqual(ast.literal_eval(v['min_child_samples']),20)
 def test_reapplying_override_rejected(self):
  _,new=p.transform(source('topk.py','fit_ranker'))
  with self.assertRaises(AssertionError):p.transform(ast.unparse(new))
 def test_sigmoid_intervention_rejected(self):
  bad=source('topk.py','fit_ranker').replace('objective="lambdarank",','objective="lambdarank", sigmoid=0.5,')
  with self.assertRaises(AssertionError):p.transform(bad)
 def test_wrong_base_objective_rejected(self):
  with self.assertRaises(AssertionError):p.transform(source('topk.py','fit_ranker').replace('"lambdarank"','"rank_xendcg"'))
 def test_classifier_not_patched(self):
  with self.assertRaises(AssertionError):p.transform('def f(): return lgb.LGBMClassifier(boosting_type="gbdt")')
if __name__=='__main__':unittest.main()
