import unittest,tempfile,json,sys,inspect
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_h325 as r
import evaluate_h325 as e
import runtime_h325 as rt

class H325Tests(unittest.TestCase):
 def setUp(self):r.CURRENT_SEED=None;r.AUDIT=None
 def test_actual_estimator_class_signature_and_params(self):
  import lightgbm as lgb
  for cls in [lgb.LGBMRanker,lgb.LGBMClassifier]:
   original=cls.__init__;before=inspect.signature(original)
   with patch.object(cls,'__init__',r.xt_init(original,cls.__name__)):
    r.CURRENT_SEED=4321;obj=cls(random_state=987,n_jobs=1)
    self.assertIsInstance(obj,cls);self.assertEqual(inspect.signature(cls.__init__),before)
    params=obj.get_params();self.assertEqual(params['extra_seed'],4321);self.assertIs(params['extra_trees'],True);self.assertEqual(params['random_state'],987)
 def test_constructor_requires_member_scope(self):
  with self.assertRaises(AssertionError):r.xt_init(lambda *a,**k:None,'x')(object())
 def test_member_scope_clears_on_failure(self):
  def original(*args):self.assertEqual(r.CURRENT_SEED,71);raise ValueError('stop')
  with self.assertRaises(ValueError):r.member_scope(original)(0,0,0,0,71)
  self.assertIsNone(r.CURRENT_SEED)
 def test_all_pinned_constructors_only_two_extra_params(self):
  # AST-extract original pinned function bodies; no optimizer fit or label change.
  import ast,subprocess,lightgbm as lgb
  commit='213b30ad52e1ce48927bba960ed609bd98ab4dfe';calls=[]
  class Fake:
   def __init__(self,**kw):self.params=kw;calls.append(self)
   def get_params(self):return self.params
   def fit(self,X,y,**kw):self.y=y;self.fit_kwargs=kw
  for filename,function,args in [('topk.py','fit_ranker',(np.zeros((8,3)),np.arange(8),1,8,.05,11)),('bagged.py','fit_ranker_n',(np.zeros((8,3)),np.arange(8),1,8,.05,100,22)),('veto.py','fit_gate',(np.zeros((8,3)),np.arange(8)%2,33))]:
   raw=subprocess.check_output(['git','--no-optional-locks','show',commit+':research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'+filename],text=True)
   node=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==function);ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),filename,'exec'),ns)
   kind='LGBMClassifier' if function=='fit_gate' else 'LGBMRanker'
   with patch.object(lgb,kind,Fake):ns[function](*args)
   base=calls[-1]
   with patch.object(Fake,'__init__',r.xt_init(Fake.__init__,kind)),patch.object(lgb,kind,Fake):
    r.CURRENT_SEED=789;ns[function](*args)
   xt=calls[-1];extra={k:xt.params[k] for k in xt.params if k not in base.params}
   self.assertEqual(extra,{'extra_trees':True,'extra_seed':789});self.assertEqual({k:v for k,v in xt.params.items() if k in base.params},base.params)
   np.testing.assert_array_equal(xt.y,base.y)
   self.assertEqual(set(xt.fit_kwargs),set(base.fit_kwargs))
   for key in xt.fit_kwargs:np.testing.assert_array_equal(xt.fit_kwargs[key],base.fit_kwargs[key])
 def pair(self,x=.003,lcb=.001,per=None):return dict(median=x,lcb95=lcb,per_cell=[x]*9 if per is None else per)
 def test_ties_no_advance(self):self.assertEqual(e.decide(self.pair(0,0),[1],[1],[0],[0])[1],'XT_NOT_ADVANCED')
 def test_small_positive_does_not_advance(self):self.assertEqual(e.decide(self.pair(.001999),[1],[1],[0],[0])[1],'XT_NOT_ADVANCED')
 def test_exact_threshold_can_advance(self):self.assertEqual(e.decide(self.pair(.002),[1],[1],[0],[0])[1],'ADVANCE')
 def test_lcb_must_be_strict(self):self.assertEqual(e.decide(self.pair(lcb=0),[1],[1],[0],[0])[1],'XT_NOT_ADVANCED')
 def test_harm_blocks(self):self.assertEqual(e.decide(self.pair(),[1],[1],[.2],[.1])[1],'XT_NOT_ADVANCED')
 def test_retention_blocks(self):self.assertEqual(e.decide(self.pair(),[.89],[1],[0],[0])[1],'XT_NOT_ADVANCED')
 def test_seed_requirement_partition_order(self):
  per=[.01,-.01,-.01,.01,-.01,-.01,.01,-.01,-.01]
  self.assertEqual(e.decide(self.pair(per=per),[1],[1],[0],[0])[1],'XT_NOT_ADVANCED')
  self.assertEqual(e.decide(self.pair(per=[.01,.01,-.01]*3),[1],[1],[0],[0])[1],'ADVANCE')
 def test_bootstrap_pairing(self):
  a=np.tile(np.linspace(-1,1,20),(9,1));z=e.paired_drop_top(a,a,draws=100);self.assertEqual(z['median'],0);self.assertEqual(z['ci95'],[0,0]);self.assertEqual(z['lcb95'],0)
 def test_jaccard(self):self.assertEqual(e.jaccard([0,1,1],[0,1,0])['jaccard'],.5)
 def test_scope_rejects_outer_before_open(self):
  scope=rt.Scope({'train'},{'train','outer'},fit=True)
  with self.assertRaises(PermissionError):scope.hook('open',('/home/ubuntu/jope-data/h50/input/pools/outer.npz','r',0))
  self.assertEqual(scope.denied,1)
 def test_scope_rejects_base_outcomes_during_fit(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h325-solhwv/base/BASE_P0.npy','r',0))
 def test_scoped_byte_read_excludes_poison_tail(self):
  parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d:
   p=Path(d)/'cache.npz';np.savez(p,features=np.array([[1.,2.],[float('nan'),float('nan')]]))
   scope=rt.Scope({'train'},{'train'});scope.expected[(str(p),'features','train')]=(0,1)
   out=scope.read(p,'features',0,1,'train');np.testing.assert_array_equal(out,[[1.,2.]]);self.assertEqual(scope.bytes,16)
   with self.assertRaises(PermissionError):scope.read(p,'features',0,2,'train')
   with self.assertRaises(PermissionError):scope.read(p,'features',1,2,'outer')
 def test_missing_lock_prevents_outer_read(self):
  class Packets(dict):
   reads=0
   def __getitem__(self,k):self.reads+=1;raise AssertionError('outer access')
  from types import SimpleNamespace
  packets=Packets();model=SimpleNamespace(G=dict(train_rows=[dict(context_sha256='x',family_id='F')],fam_by_cid={'x':'F'},a302=SimpleNamespace(outer_fold_of_family=lambda f:0),P8=packets))
  parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d,patch.object(r,'DATA',Path(d)),patch.dict(r.R,{'r303':model},clear=True):
   with self.assertRaises(FileNotFoundError):r.score('P0')
  self.assertEqual(packets.reads,0)
if __name__=='__main__':unittest.main()
