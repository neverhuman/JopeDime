import ast,subprocess,unittest
from pathlib import Path
import lr_patch as p
W=Path('/home/ubuntu/JopeDime');PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'
def source(file,fun):
 raw=subprocess.check_output(['git','-C',str(W),'show',PIN+':'+PREFIX+file],text=True)
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==fun)
 return ast.get_source_segment(raw,node)
def kw(t):return {k.arg:ast.dump(k.value,include_attributes=False) for k in p.calls(t)[0].keywords}
class LRTests(unittest.TestCase):
 def test_both_full_BASE_asts_restore(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(p.strip(new),include_attributes=False))
 def test_revert_equals_committed_H352_AST(self):
  text=subprocess.check_output(['git','-C',str(W),'show','02dfee2f7e:research/flywheel_selfplay/h352_ranker_num_leaves_15/leaves_patch.py'],text=True);namespace={};exec(compile(text,'<committed-H352>','exec'),namespace)
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   src=source(file,fun);_,want=namespace['transform'](src);_,got=p.transform(src)
   self.assertEqual(ast.dump(want,include_attributes=False),ast.dump(p.strip(got),include_attributes=False))
 def test_only_lr_and_round_expressions_changed(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));a,b=kw(old),kw(new)
   self.assertEqual(set(a),set(b));self.assertEqual({k:v for k,v in a.items() if k not in ['learning_rate','n_estimators']},{k:v for k,v in b.items() if k not in ['learning_rate','n_estimators']})
   for lr in [.01,.03,.05,.1]:
    for n in [25,50,100,200]:
     vals=[]
     for tree in [old,new]:
      vals.append({k.arg:eval(compile(ast.Expression(k.value),'<expression>','eval'),{'lr':lr,'n_est':n}) for k in p.calls(tree)[0].keywords if k.arg in ['learning_rate','n_estimators']})
     self.assertEqual(vals[1]['learning_rate'],vals[0]['learning_rate']*.5);self.assertEqual(vals[1]['n_estimators'],vals[0]['n_estimators']*2)
     self.assertEqual(vals[0]['learning_rate']*vals[0]['n_estimators'],vals[1]['learning_rate']*vals[1]['n_estimators'])
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
class PredictionMappingTests(unittest.TestCase):
 def fake(self,objective,loaded=False):
  from types import SimpleNamespace
  return SimpleNamespace(params={} if loaded else dict(objective=objective),dump_model=lambda **kw:dict(objective=objective))
 def test_ranker_positive_limit_doubles_training_and_reloaded(self):
  for loaded in [False,True]:
   wrapped=p.prediction_wrapper(lambda self,data,start_iteration,num_iteration: (start_iteration,num_iteration))
   self.assertEqual(wrapped(self.fake('lambdarank',loaded),'fixture',num_iteration=25),(0,50))
 def test_veto_limit_and_all_tree_defaults_unchanged(self):
  wrapped=p.prediction_wrapper(lambda self,data,start_iteration,num_iteration:(start_iteration,num_iteration))
  for obj in ['binary','lambdarank']:
   for limit in [None,0,-1]:self.assertEqual(wrapped(self.fake(obj),'fixture',num_iteration=limit),(0,limit))
  self.assertEqual(wrapped(self.fake('binary'),'fixture',num_iteration=70),(0,70))
 def test_ranker_nonzero_start_rejected(self):
  with self.assertRaises(AssertionError):p.prediction_wrapper(lambda *a:None)(self.fake('lambdarank'),'fixture',start_iteration=1,num_iteration=25)
if __name__=='__main__':unittest.main()
