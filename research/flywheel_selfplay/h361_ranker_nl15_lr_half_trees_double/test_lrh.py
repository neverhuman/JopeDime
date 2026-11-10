import ast,inspect,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import lrh_patch as p
W='/home/ubuntu/JopeDime';PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'
def source(file,fun):
 if file=='r302':raw=(Path(__file__).parent.parent/'vendor/h302_ensvote2_trainseed_nested_cv/run_h302.py').read_text()
 else:raw=subprocess.check_output(['git','-C',W,'show',PIN+':'+PREFIX+file],text=True)
 node=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==fun);return ast.get_source_segment(raw,node)
class LRHTests(unittest.TestCase):
 def test_all_four_full_H352_ASTs_restore_exactly(self):
  for file,fun,role in [('topk.py','fit_ranker','constructor'),('bagged.py','fit_ranker_n','constructor'),('veto.py','run_cell_veto','veto_selection'),('r302','fit_and_freeze_member','member_lock')]:
   baseline,new=p.transform(source(file,fun),role);self.assertEqual(ast.dump(baseline,include_attributes=False),ast.dump(p.strip(new,role),include_attributes=False))
 def test_constructor_only_LR_and_iterations_differ_from_NL15(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   baseline,new=p.transform(source(file,fun),'constructor');a={x.arg:ast.dump(x.value) for x in p.leaves_patch.calls(baseline)[0].keywords};b={x.arg:ast.dump(x.value) for x in p.leaves_patch.calls(new)[0].keywords}
   self.assertEqual(set(a),set(b));self.assertEqual({k:v for k,v in a.items() if k not in ['learning_rate','n_estimators']},{k:v for k,v in b.items() if k not in ['learning_rate','n_estimators']})
 def test_actual_constructor_grid_halves_and_doubles_without_real_fit(self):
  class Fake:
   def __init__(self,**kwargs):self.params=kwargs
   def fit(self,*a,**kwargs):return self
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   baseline,new=p.transform(source(file,fun),'constructor');ns={'np':np};exec(compile(baseline,'<baseline>','exec'),ns);b=ns[fun];exec(compile(new,'<LRH>','exec'),ns);c=ns[fun]
   with patch('lightgbm.LGBMRanker',Fake):
    for lr in [.03,.05,.1]:
     for n in [50,100,200]:
      args=(np.zeros((8,2)),np.zeros(8),1,8,lr,*([n] if file=='bagged.py' else []),17);old=b(*args).params;got=c(*args).params
      self.assertEqual(got['learning_rate'],old['learning_rate']*.5);self.assertEqual(got['n_estimators'],old['n_estimators']*2)
      self.assertEqual({k:v for k,v in got.items() if k not in ['learning_rate','n_estimators']},{k:v for k,v in old.items() if k not in ['learning_rate','n_estimators']});self.assertEqual(got['random_state'],17)
 def test_veto_prediction_limits_doubled_but_classifier_and_seeds_unchanged(self):
  baseline,new=p.transform(source('veto.py','run_cell_veto'),'veto_selection');kw=[x for x in ast.walk(new) if isinstance(x,ast.keyword) and x.arg=='num_iteration'];self.assertEqual(len(kw),2);self.assertTrue(all(isinstance(x.value,ast.BinOp) for x in kw));self.assertEqual(ast.dump(baseline),ast.dump(p.strip(new,'veto_selection')))
 def test_member_locks_actual_limits_and_reload_limits_doubled_once(self):
  baseline,new=p.transform(source('r302','fit_and_freeze_member'),'member_lock');kws=[x for x in ast.walk(new) if isinstance(x,ast.keyword) and x.arg=='num_iteration'];self.assertEqual(sum(isinstance(x.value,ast.BinOp) for x in kws),2);self.assertEqual(sum(isinstance(x.value,ast.Constant) and x.value.value is None for x in kws),1);self.assertEqual(ast.dump(baseline),ast.dump(p.strip(new,'member_lock')))
if __name__=='__main__':unittest.main()
