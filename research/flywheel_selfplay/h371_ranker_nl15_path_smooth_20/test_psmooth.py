import ast,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import psmooth_patch as p
W='/home/ubuntu/JopeDime';PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'
def source(file,fun):
 raw=subprocess.check_output(['git','-C',W,'show',PIN+':'+PREFIX+file],text=True)
 node=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==fun);return ast.get_source_segment(raw,node)
class PS20Tests(unittest.TestCase):
 def test_full_H352_AST_inverse_and_only_added_keyword(self):
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));self.assertEqual(ast.dump(old),ast.dump(p.strip(new)))
   a={k.arg:ast.dump(k.value) for k in p.leaves_patch.calls(old)[0].keywords};b={k.arg:ast.dump(k.value) for k in p.leaves_patch.calls(new)[0].keywords}
   self.assertEqual(b.pop('path_smooth'),ast.dump(ast.Constant(20.0)));self.assertEqual(a,b)
 def test_real_constructor_grid_and_seeds_unchanged_without_real_fit(self):
  class Fake:
   def __init__(self,**kwargs):self.params=kwargs
   def fit(self,*a,**k):return self
  for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   old,new=p.transform(source(file,fun));ns={'np':np};exec(compile(old,'<H352>','exec'),ns);before=ns[fun];exec(compile(new,'<PS20>','exec'),ns);after=ns[fun]
   with patch('lightgbm.LGBMRanker',Fake):
    for lr in [.03,.05,.1]:
     for n in [50,100,200]:
      args=(np.zeros((8,2)),np.zeros(8),1,8,lr,*([n] if file=='bagged.py' else []),17)
      a=before(*args).params;b=after(*args).params;self.assertEqual(b.pop('path_smooth'),20.0);self.assertEqual(a,b);self.assertEqual(b['random_state'],17)
 def test_no_LRH_prediction_or_member_lock_patch(self):
  text=(Path(__file__).parent/'run_h371.py').read_text();self.assertNotIn('lrh_patch',text)
  self.assertNotIn('LR_SCALE',(Path(__file__).parent/'psmooth_patch.py').read_text())
if __name__=='__main__':unittest.main()
