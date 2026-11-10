import unittest,tempfile,json,sys,inspect,ast,copy,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_h332 as r
import evaluate_h332 as e
import runtime_h332 as rt
import leaf_clipping as lc
from cap_parser import raw_leaf_magnitudes,fixed_cap
PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe'
def pinned_function(filename,name):
 raw=subprocess.check_output(['git','--no-optional-locks','show',PIN+':research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'+filename],text=True);return next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==name)
def function(node,namespace):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'pinned-function','exec'),namespace);return namespace[node.name]
class H332Tests(unittest.TestCase):
 def setUp(self):r.CURRENT_SEED=None;r.AUDIT=None;r.CAP=1.23456789
 def test_inverse_exact_BASE_ASTS(self):
  for filename,name in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   baseline=pinned_function(filename,name);changed,receipt=lc.clip_function(baseline,r.CAP);restored=copy.deepcopy(changed);lc.ranker_call(restored).keywords=[x for x in lc.ranker_call(restored).keywords if x.arg!='max_delta_step']
   self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(baseline,include_attributes=False));self.assertEqual(receipt['BASE_AST_sha256'],receipt['removed_override_AST_sha256'])
 def test_duplicate_or_gain_override_refused(self):
  changed,_=lc.clip_function(pinned_function('topk.py','fit_ranker'),r.CAP)
  with self.assertRaises(AssertionError):lc.clip_function(changed,r.CAP)
 def test_only_cap_changes_parameters_labels_groups_rounds(self):
  import lightgbm as lgb
  calls=[]
  class Fake:
   def __init__(self,**kw):self.params=kw;calls.append(self)
   def fit(self,X,y,**kw):self.y=np.array(y);self.kwargs=kw
  for file,name,args in [('topk.py','fit_ranker',(np.zeros((8,3)),np.arange(8),1,8,.05,11)),('bagged.py','fit_ranker_n',(np.zeros((8,3)),np.arange(8),1,8,.05,100,22))]:
   original=pinned_function(file,name);changed,_=lc.clip_function(original,r.CAP)
   with patch.object(lgb,'LGBMRanker',Fake):function(original,{'np':np})(*args);base=calls[-1];function(changed,{'np':np})(*args);clip=calls[-1]
   self.assertEqual(clip.params,{**base.params,'max_delta_step':r.CAP});np.testing.assert_array_equal(clip.y,base.y);self.assertEqual(set(clip.kwargs),set(base.kwargs));self.assertNotIn('label_gain',clip.params)
   for key in clip.kwargs:np.testing.assert_array_equal(clip.kwargs[key],base.kwargs[key])
 def test_real_constructor_readonly_observer(self):
  import lightgbm as lgb
  original=lgb.LGBMRanker.__init__;sig=inspect.signature(original)
  with patch.object(lgb.LGBMRanker,'__init__',r.observe_init(original,'LGBMRanker')):
   obj=lgb.LGBMRanker(objective='lambdarank',n_jobs=1,max_delta_step=r.CAP);r.validate_parameters('LGBMRanker',obj.get_params());self.assertEqual(inspect.signature(lgb.LGBMRanker.__init__),sig)
 def test_veto_ctor_labels_unchanged(self):
  import lightgbm as lgb
  calls=[]
  class Fake:
   def __init__(self,**kw):self.params=kw;calls.append(self)
   def get_params(self):return self.params
   def fit(self,X,y,**kw):self.y=np.array(y);self.kwargs=kw
  f=function(pinned_function('veto.py','fit_gate'),{'np':np});args=(np.zeros((8,3)),np.arange(8)%2,41)
  with patch.object(lgb,'LGBMClassifier',Fake):f(*args);base=calls[-1]
  parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d,patch.object(Fake,'__init__',r.observe_init(Fake.__init__,'LGBMClassifier')),patch.object(lgb,'LGBMClassifier',Fake):
   r.CURRENT_SEED=3;r.AUDIT=Path(d)/'audit.jsonl';f(*args);self.assertEqual(calls[-1].params,base.params);np.testing.assert_array_equal(calls[-1].y,base.y)
 def test_other_ablation_overrides_refused(self):
  p=dict(n_jobs=1,objective='lambdarank',max_delta_step=r.CAP)
  for extra in [dict(extra_trees=True),dict(extra_seed=1),dict(label_gain=list(range(8)))]:
   with self.assertRaises(AssertionError):r.validate_parameters('LGBMRanker',{**p,**extra})
 def test_veto_cap_refused(self):
  with self.assertRaises(AssertionError):r.validate_parameters('LGBMClassifier',dict(n_jobs=1,objective='binary',max_delta_step=r.CAP))
 def test_invalid_cap_refused(self):
  for cap in [0,-1,float('nan'),float('inf')]:
   with self.assertRaises(AssertionError):lc.clip_function(pinned_function('topk.py','fit_ranker'),cap)
 def test_leaf_parser_inverse_learning_rate(self):
  blob=b'Tree=0\nnum_leaves=2\nleaf_value=-0.2 0.4\nTree=1\nnum_leaves=1\nleaf_value=0\nparameters:\n[objective: lambdarank]\n[extra_trees: 0]\n[max_delta_step: 0]\n[label_gain: ]\n[learning_rate: 0.1]\n'
  values,lr,trees=raw_leaf_magnitudes(blob);np.testing.assert_allclose(values,[2,4,0]);self.assertEqual(lr,.1);self.assertEqual(trees,2)
 def test_cap_quantile_exact_linear(self):self.assertEqual(fixed_cap(np.arange(1,11,dtype=float)),9.1)
 def test_cap_rejects_empty_nonfinite_zero(self):
  for values in [[],[float('nan')],[float('inf')],[0,0],[-1,2]]:
   with self.assertRaises(AssertionError):fixed_cap(values)
 def test_leaf_cardinality_checked(self):
  blob=b'num_leaves=2\nleaf_value=1\nparameters:\n[objective: lambdarank]\n[extra_trees: 0]\n[learning_rate: 0.1]\n'
  with self.assertRaises(AssertionError):raw_leaf_magnitudes(blob)
 def test_context_keys_intact(self):
  from types import SimpleNamespace
  rows=[dict(context_sha256='train',family_id='F')];scope=SimpleNamespace(denied=0);G=dict(train_rows=rows);runtime=SimpleNamespace(Scope=lambda a,b:scope,packets=lambda selected,got:{x['context_sha256']:'packet' for x in selected})
  with patch.dict(r.R,{'r303':SimpleNamespace(G=G),'rt':runtime},clear=True):r.full_packets()
  self.assertEqual(G['P8'],{'train':'packet'})
 def test_member_scope_cleanup(self):
  def original(*args):self.assertEqual(r.CURRENT_SEED,71);raise ValueError('stop')
  with self.assertRaises(ValueError):r.member_scope(original)(0,0,0,0,71)
  self.assertIsNone(r.CURRENT_SEED)
 def pair(self,x=.003,lcb=.001,per=None):return dict(median=x,lcb95=lcb,per_cell=[x]*9 if per is None else per)
 def test_ties_fail(self):self.assertEqual(e.decide(self.pair(0,0),[1],[1],[0],[0])[1],'CLIP_NOT_ADVANCED')
 def test_below_margin_fails(self):self.assertEqual(e.decide(self.pair(.001999),[1],[1],[0],[0])[1],'CLIP_NOT_ADVANCED')
 def test_exact_margin_passes(self):self.assertEqual(e.decide(self.pair(.002),[1],[1],[0],[0])[1],'ADVANCE')
 def test_lcb_must_be_positive(self):self.assertEqual(e.decide(self.pair(lcb=0),[1],[1],[0],[0])[1],'CLIP_NOT_ADVANCED')
 def test_harm_fails(self):self.assertEqual(e.decide(self.pair(),[1],[1],[.2],[.1])[1],'CLIP_NOT_ADVANCED')
 def test_retention_fails(self):self.assertEqual(e.decide(self.pair(),[.89],[1],[0],[0])[1],'CLIP_NOT_ADVANCED')
 def test_root_consistency(self):
  self.assertEqual(e.decide(self.pair(per=[.01,-.01,-.01]*3),[1],[1],[0],[0])[1],'CLIP_NOT_ADVANCED');self.assertEqual(e.decide(self.pair(per=[.01,.01,-.01]*3),[1],[1],[0],[0])[1],'ADVANCE')
 def test_bootstrap_pairing(self):
  a=np.tile(np.linspace(-1,1,20),(9,1));z=e.paired_drop_top(a,a,draws=100);self.assertEqual(z['median'],0);self.assertEqual(z['ci95'],[0,0])
 def test_jaccard(self):self.assertEqual(e.jaccard([0,1,1],[0,1,0])['jaccard'],.5)
 def test_outer_scope_rejected(self):
  scope=rt.Scope({'train'},{'train','outer'},fit=True)
  with self.assertRaises(PermissionError):scope.hook('open',('/home/ubuntu/jope-data/h50/input/pools/outer.npz','r',0))
 def test_base_scope_rejected(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h332-solhwv/base/BASE_P0.npy','r',0))
 def test_scoped_read_excludes_poison(self):
  parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d:
   p=Path(d)/'cache.npz';np.savez(p,features=np.array([[1.,2.],[float('nan'),float('nan')]]));scope=rt.Scope({'train'},{'train'});scope.expected[(str(p),'features','train')]=(0,1)
   np.testing.assert_array_equal(scope.read(p,'features',0,1,'train'),[[1.,2.]])
   with self.assertRaises(PermissionError):scope.read(p,'features',0,2,'train')
 def test_missing_lock_stops_outer_read(self):
  from types import SimpleNamespace
  class Packets(dict):
   reads=0
   def __getitem__(self,key):self.reads+=1;raise AssertionError('outer read')
  packets=Packets();model=SimpleNamespace(G=dict(train_rows=[dict(context_sha256='x',family_id='F')],fam_by_cid={'x':'F'},a302=SimpleNamespace(outer_fold_of_family=lambda f:0),P8=packets));parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d,patch.object(r,'DATA',Path(d)),patch.dict(r.R,{'r303':model},clear=True):
   with self.assertRaises(FileNotFoundError):r.score('P0')
  self.assertEqual(packets.reads,0)
if __name__=='__main__':unittest.main()
