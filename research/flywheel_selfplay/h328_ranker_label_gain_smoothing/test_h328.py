import unittest,tempfile,json,sys,inspect,ast,copy,subprocess,types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_h328 as r
import evaluate_h328 as e
import runtime_h328 as rt
import gain_smoothing as gs

PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe'
def pinned_function(filename,name):
 raw=subprocess.check_output(['git','--no-optional-locks','show',PIN+':research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'+filename],text=True)
 return next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==name)
def function(node,namespace):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'pinned-function','exec'),namespace)
 return namespace[node.name]
class H328Tests(unittest.TestCase):
 def setUp(self):r.CURRENT_SEED=None;r.AUDIT=None
 def test_exact_gain_lists(self):
  self.assertEqual(gs.BASE_EFFECTIVE_GAINS,[0,1,3,7,15,31,63,127]);self.assertEqual(gs.LINEAR_GAINS,[0,1,2,3,4,5,6,7])
 def test_remove_single_override_restores_both_BASE_ASTS(self):
  for filename,name in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   baseline=pinned_function(filename,name);changed,receipt=gs.smooth_function(baseline)
   keywords=gs.ranker_call(changed).keywords;self.assertEqual(sum(x.arg=='label_gain' for x in keywords),1)
   restored=copy.deepcopy(changed);gs.ranker_call(restored).keywords=[x for x in gs.ranker_call(restored).keywords if x.arg!='label_gain']
   self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(baseline,include_attributes=False));self.assertEqual(receipt['BASE_AST_sha256'],receipt['removed_override_AST_sha256'])
 def test_duplicate_gain_override_refused(self):
  changed,_=gs.smooth_function(pinned_function('topk.py','fit_ranker'))
  with self.assertRaises(AssertionError):gs.smooth_function(changed)
 def test_exact_parameters_labels_groups_eval_arguments(self):
  import lightgbm as lgb
  calls=[]
  class Fake:
   def __init__(self,**kwargs):self.params=kwargs;calls.append(self)
   def fit(self,X,y,**kwargs):self.y=np.array(y);self.kwargs=kwargs
  for filename,name,args in [('topk.py','fit_ranker',(np.zeros((8,3)),np.arange(8),1,8,.05,11)),('bagged.py','fit_ranker_n',(np.zeros((8,3)),np.arange(8),1,8,.05,100,22))]:
   baseline=pinned_function(filename,name);changed,_=gs.smooth_function(baseline)
   with patch.object(lgb,'LGBMRanker',Fake):
    function(baseline,{'np':np})(*args);base=calls[-1];function(changed,{'np':np})(*args);lgs=calls[-1]
   self.assertEqual(lgs.params,{**base.params,'label_gain':list(range(8))});np.testing.assert_array_equal(lgs.y,base.y)
   self.assertEqual(set(lgs.kwargs),set(base.kwargs))
   for key in lgs.kwargs:np.testing.assert_array_equal(lgs.kwargs[key],base.kwargs[key])
 def test_veto_constructor_and_fit_arguments_unchanged(self):
  import lightgbm as lgb
  calls=[]
  class Fake:
   def __init__(self,**kwargs):self.params=kwargs;calls.append(self)
   def get_params(self):return self.params
   def fit(self,X,y,**kwargs):self.y=np.array(y);self.kwargs=kwargs
  basefunction=pinned_function('veto.py','fit_gate');gate=function(basefunction,{'np':np});args=(np.zeros((8,3)),np.arange(8)%2,39)
  with patch.object(lgb,'LGBMClassifier',Fake):gate(*args);base=calls[-1]
  parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d,patch.object(Fake,'__init__',r.observe_init(Fake.__init__,'LGBMClassifier')),patch.object(lgb,'LGBMClassifier',Fake):
   r.CURRENT_SEED=71;r.AUDIT=Path(d)/'audit.jsonl';gate(*args);observed=calls[-1]
   self.assertEqual(observed.params,base.params);np.testing.assert_array_equal(observed.y,base.y);self.assertEqual(observed.kwargs,base.kwargs)
   record=json.loads(r.AUDIT.read_text());self.assertNotIn('label_gain',record['params']);self.assertNotIn('extra_trees',record['params']);self.assertNotIn('extra_seed',record['params'])
 def test_actual_estimator_parameters_and_readonly_observer(self):
  import lightgbm as lgb
  original=lgb.LGBMRanker.__init__;sig=inspect.signature(original)
  with patch.object(lgb.LGBMRanker,'__init__',r.observe_init(original,'LGBMRanker')):
   obj=lgb.LGBMRanker(objective='lambdarank',n_jobs=1,random_state=39,label_gain=list(range(8)))
   self.assertIsInstance(obj,lgb.LGBMRanker);self.assertEqual(inspect.signature(lgb.LGBMRanker.__init__),sig);r.validate_parameters('LGBMRanker',obj.get_params())
 def test_tree_overrides_refused(self):
  p=dict(n_jobs=1,objective='lambdarank',label_gain=list(range(8)))
  for extra in [dict(extra_trees=True),dict(extra_seed=3)]:
   with self.assertRaises(AssertionError):r.validate_parameters('LGBMRanker',{**p,**extra})
 def test_classifier_gain_refused(self):
  with self.assertRaises(AssertionError):r.validate_parameters('LGBMClassifier',dict(n_jobs=1,objective='binary',label_gain=list(range(8))))
 def test_original_ordinal_labels_unchanged(self):
  rel=function(pinned_function('topk.py','topk_relevance'),{'np':np});fitness=np.array([1.,2.,3.,4.,5.,6.,7.,8.]);gates=np.array([True]*7+[False]);old=fitness.copy();y=rel(fitness,gates)
  np.testing.assert_array_equal(y,[7,6,5,4,3,2,1,0]);np.testing.assert_array_equal(fitness,old)
 def test_context_identity_fields_intact(self):
  from types import SimpleNamespace
  rows=[dict(context_sha256='train',family_id='F')];scope=SimpleNamespace(denied=0);G=dict(train_rows=rows)
  runtime=SimpleNamespace(Scope=lambda allowed,allids:scope,packets=lambda selected,got:{x['context_sha256']:'packet' for x in selected})
  with patch.dict(r.R,{'r303':SimpleNamespace(G=G),'rt':runtime},clear=True):r.full_packets()
  self.assertEqual(G['P8'],{'train':'packet'})
 def test_member_scope_cleanup(self):
  def original(*args):self.assertEqual(r.CURRENT_SEED,71);raise ValueError('stop')
  with self.assertRaises(ValueError):r.member_scope(original)(0,0,0,0,71)
  self.assertIsNone(r.CURRENT_SEED)
 def pair(self,x=.003,lcb=.001,per=None):return dict(median=x,lcb95=lcb,per_cell=[x]*9 if per is None else per)
 def test_ties_no_advance(self):self.assertEqual(e.decide(self.pair(0,0),[1],[1],[0],[0])[1],'LGS_NOT_ADVANCED')
 def test_small_positive_does_not_advance(self):self.assertEqual(e.decide(self.pair(.001999),[1],[1],[0],[0])[1],'LGS_NOT_ADVANCED')
 def test_exact_threshold_can_advance(self):self.assertEqual(e.decide(self.pair(.002),[1],[1],[0],[0])[1],'ADVANCE')
 def test_lcb_must_be_strict(self):self.assertEqual(e.decide(self.pair(lcb=0),[1],[1],[0],[0])[1],'LGS_NOT_ADVANCED')
 def test_harm_blocks(self):self.assertEqual(e.decide(self.pair(),[1],[1],[.2],[.1])[1],'LGS_NOT_ADVANCED')
 def test_retention_blocks(self):self.assertEqual(e.decide(self.pair(),[.89],[1],[0],[0])[1],'LGS_NOT_ADVANCED')
 def test_seed_requirement_partition_order(self):
  self.assertEqual(e.decide(self.pair(per=[.01,-.01,-.01]*3),[1],[1],[0],[0])[1],'LGS_NOT_ADVANCED');self.assertEqual(e.decide(self.pair(per=[.01,.01,-.01]*3),[1],[1],[0],[0])[1],'ADVANCE')
 def test_bootstrap_pairing(self):
  a=np.tile(np.linspace(-1,1,20),(9,1));z=e.paired_drop_top(a,a,draws=100);self.assertEqual(z['median'],0);self.assertEqual(z['ci95'],[0,0]);self.assertEqual(z['lcb95'],0)
 def test_jaccard(self):self.assertEqual(e.jaccard([0,1,1],[0,1,0])['jaccard'],.5)
 def test_scope_rejects_outer_before_open(self):
  scope=rt.Scope({'train'},{'train','outer'},fit=True)
  with self.assertRaises(PermissionError):scope.hook('open',('/home/ubuntu/jope-data/h50/input/pools/outer.npz','r',0))
  self.assertEqual(scope.denied,1)
 def test_scope_rejects_base_outcomes_during_fit(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h328-solhwv/base/BASE_P0.npy','r',0))
 def test_scoped_byte_read_excludes_poison_tail(self):
  parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d:
   p=Path(d)/'cache.npz';np.savez(p,features=np.array([[1.,2.],[float('nan'),float('nan')]]));scope=rt.Scope({'train'},{'train'});scope.expected[(str(p),'features','train')]=(0,1)
   out=scope.read(p,'features',0,1,'train');np.testing.assert_array_equal(out,[[1.,2.]]);self.assertEqual(scope.bytes,16)
   with self.assertRaises(PermissionError):scope.read(p,'features',0,2,'train')
   with self.assertRaises(PermissionError):scope.read(p,'features',1,2,'outer')
 def test_missing_lock_prevents_outer_read(self):
  class Packets(dict):
   reads=0
   def __getitem__(self,k):self.reads+=1;raise AssertionError('outer access')
  from types import SimpleNamespace
  packets=Packets();model=SimpleNamespace(G=dict(train_rows=[dict(context_sha256='x',family_id='F')],fam_by_cid={'x':'F'},a302=SimpleNamespace(outer_fold_of_family=lambda f:0),P8=packets));parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as d,patch.object(r,'DATA',Path(d)),patch.dict(r.R,{'r303':model},clear=True):
   with self.assertRaises(FileNotFoundError):r.score('P0')
  self.assertEqual(packets.reads,0)
if __name__=='__main__':unittest.main()
