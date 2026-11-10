import ast,hashlib,subprocess,unittest
from types import SimpleNamespace
import numpy as np
import bag_patch as p
W='/home/ubuntu/JopeDime';PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe';PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'
def raw(file):return subprocess.check_output(['git','-C',W,'show',PIN+':'+PREFIX+file],text=True)
def function(file,name,ns):
 text=raw(file);node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-'+name+'>','exec'),ns);return ns[name]
class BagTests(unittest.TestCase):
 def test_only_constant_changes_and_reverts(self):
  obj=SimpleNamespace(N_BAGS=8,Q_FIXED=6,Q_GRID=(5,6,7));before=vars(obj).copy();p.count_change(obj,8,16);self.assertEqual(vars(obj),dict(before,N_BAGS=16));obj.N_BAGS=8;self.assertEqual(vars(obj),before)
 def test_treatment_doubles_actual_archived_count(self):
  obj=SimpleNamespace(N_BAGS=10,Q_FIXED=6,Q_GRID=(5,6,7));self.assertEqual(p.count_change(obj,10,20)['treatment'],20)
 def test_second_override_or_wrong_baseline_rejected(self):
  obj=SimpleNamespace(N_BAGS=16,Q_FIXED=6,Q_GRID=(5,6,7))
  with self.assertRaises(AssertionError):p.count_change(obj,8,16)
 def test_threshold_rescaling_rejected(self):
  for q,grid in [(12,(5,6,7)),(6,(10,12,14))]:
   with self.assertRaises(AssertionError):p.count_change(SimpleNamespace(N_BAGS=8,Q_FIXED=q,Q_GRID=grid),8,16)
 def test_seed_prefix_and_extended_index_match_original(self):
  stream=function('h265_core.py','stream_seed',dict(hashlib=hashlib,Any=object));core=SimpleNamespace(stream_seed=stream)
  rows=p.seed_extension(core,[30001,30002,30003,1234],8,16)
  for row in rows:
   for key in ['outer_boot','outer_bag']:
    self.assertEqual(row[key][:8],[stream(row['member_seed'],0,'BAG_SWITCH',key,b) for b in range(8)])
    self.assertEqual(row[key][8:],[stream(row['member_seed'],0,'BAG_SWITCH',key,b) for b in range(8,16)])
 def test_entire_H352_constructor_AST_unchanged(self):
  text=subprocess.check_output(['git','-C',W,'show','02dfee2f7e:research/flywheel_selfplay/h352_ranker_num_leaves_15/leaves_patch.py'],text=True);ns={};exec(compile(text,'<H352>','exec'),ns)
  for file,name in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   text=raw(file);node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name);src=ast.get_source_segment(text,node);_,want=ns['transform'](src);_,got=p.transform(src);self.assertEqual(ast.dump(want,include_attributes=False),ast.dump(got,include_attributes=False))
 def test_original_fit_loop_is_dynamic_and_seed_keywords_unchanged(self):
  text=subprocess.check_output(['git','-C',W,'show','819a45aa5e:research/flywheel_selfplay/h302_ensvote2_trainseed_nested_cv/run_h302.py'],text=True);node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='fit_and_freeze_member');src=ast.get_source_segment(text,node)
  self.assertIn('range(bagged.N_BAGS)',src);self.assertIn('"outer_boot", b',src);self.assertIn('"outer_bag", b',src);self.assertNotIn('bagging_seed=',src)
 def test_original_vote_function_uses_all_bags_same_absolute_threshold(self):
  import math
  ns=dict(np=np,math=math);votes=function('bagged.py','vote_counts',ns);switch=function('bagged.py','switch_by_votes',ns)
  scores=np.zeros((16,1,2));scores[:5,0,1]=1
  self.assertEqual(votes(scores,[1],.1).tolist(),[5]);self.assertEqual(switch([1],votes(scores,[1],.1),5),[1]);self.assertEqual(switch([1],votes(scores,[1],.1),6),[0]);scores[15,0,1]=1;self.assertEqual(switch([1],votes(scores,[1],.1),6),[1])
 def test_all_extended_bags_must_be_fitted_inner_and_final(self):
  counts={}
  for member_seed in [1,2,3]:
   for b in range(16):
    raw=':'.join(map(str,(member_seed,0,'BAG_SWITCH','outer_bag',b))).encode();seed=int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')&0x7fffffff;counts[seed]=13
  proof=p.verify_bag_call_counts(counts,[1,2,3],16);self.assertEqual(len(proof['checked']),48)
  last=proof['checked'][-1]['seed'];counts[last]=12
  with self.assertRaises(AssertionError):p.verify_bag_call_counts(counts,[1,2,3],16)
  counts.pop(last)
  with self.assertRaises(AssertionError):p.verify_bag_call_counts(counts,[1,2,3],16)
if __name__=='__main__':unittest.main()
