import unittest,tempfile,json,sys,inspect,ast,copy,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_h345 as r
import evaluate_h345 as e
import runtime_h345 as rt
import row_subsampling as rs
import admission_h345 as ad
import datetime
PIN='213b30ad52e1ce48927bba960ed609bd98ab4dfe'
def pinned_function(filename,name):
 raw=subprocess.check_output(['git','--no-optional-locks','show',PIN+':research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'+filename],text=True);return next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==name)
def function(node,namespace):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'pinned-function','exec'),namespace);return namespace[node.name]
class H345Tests(unittest.TestCase):
 def setUp(self):r.CURRENT_SEED=None;r.AUDIT=None
 def test_inverse_exact_BASE_ASTS(self):
  for filename,name in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   base=pinned_function(filename,name);changed,receipt=rs.rowsub_function(base);restored=copy.deepcopy(changed);next(x for x in rs.ranker_call(restored).keywords if x.arg=='subsample').value=ast.Constant(value=.8)
   self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(base,include_attributes=False));self.assertEqual(receipt['BASE_AST_sha256'],receipt['restored_BASE_AST_sha256']);self.assertEqual(receipt['new_keywords'],[])
 def test_duplicate_override_refused(self):
  changed,_=rs.rowsub_function(pinned_function('topk.py','fit_ranker'))
  with self.assertRaises(AssertionError):rs.rowsub_function(changed)
 def test_only_one_value_changes_labels_groups_rounds(self):
  import lightgbm as lgb
  calls=[]
  class Fake:
   def __init__(self,**kw):self.params=kw;calls.append(self)
   def fit(self,X,y,**kw):self.y=np.array(y);self.kwargs=kw
  for file,name,args in [('topk.py','fit_ranker',(np.zeros((8,3)),np.arange(8),1,8,.05,11)),('bagged.py','fit_ranker_n',(np.zeros((8,3)),np.arange(8),1,8,.05,100,22))]:
   original=pinned_function(file,name);changed,_=rs.rowsub_function(original)
   with patch.object(lgb,'LGBMRanker',Fake):function(original,{'np':np})(*args);base=calls[-1];function(changed,{'np':np})(*args);row=calls[-1]
   self.assertEqual(row.params,{**base.params,'subsample':.6});self.assertEqual(set(row.params),set(base.params));np.testing.assert_array_equal(row.y,base.y);self.assertEqual(set(row.kwargs),set(base.kwargs))
   for key in row.kwargs:np.testing.assert_array_equal(row.kwargs[key],base.kwargs[key])
 def test_fit_seed_expression_unchanged_no_new_seed(self):
  for file,name in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
   base=pinned_function(file,name);changed,_=rs.rowsub_function(base);b={x.arg:x.value for x in rs.ranker_call(base).keywords};c={x.arg:x.value for x in rs.ranker_call(changed).keywords};self.assertEqual(ast.dump(b['random_state']),ast.dump(c['random_state']));self.assertNotIn('bagging_seed',c);self.assertNotIn('feature_fraction_seed',c)
 def test_real_constructor_readonly_observer(self):
  import lightgbm as lgb
  original=lgb.LGBMRanker.__init__;sig=inspect.signature(original)
  with patch.object(lgb.LGBMRanker,'__init__',r.observe_init(original,'LGBMRanker')):
   obj=lgb.LGBMRanker(objective='lambdarank',n_jobs=1,random_state=17,colsample_bytree=.8,subsample=.6,subsample_freq=1);r.validate_parameters('LGBMRanker',obj.get_params());self.assertEqual(inspect.signature(lgb.LGBMRanker.__init__),sig)
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
  p=dict(n_jobs=1,objective='lambdarank',random_state=11,colsample_bytree=.8,subsample=.6,subsample_freq=1)
  for extra in [dict(extra_trees=True),dict(extra_seed=1),dict(label_gain=list(range(8))),dict(max_delta_step=1),dict(feature_fraction_bynode=.8),dict(feature_fraction_seed=11),dict(bagging_seed=11)]:
   with self.assertRaises(AssertionError):r.validate_parameters('LGBMRanker',{**p,**extra})
 def test_wrong_fraction_or_frequency_refused(self):
  p=dict(n_jobs=1,objective='lambdarank',random_state=11,colsample_bytree=.8,subsample=.6,subsample_freq=1)
  for extra in [dict(subsample=.8),dict(subsample_freq=0),dict(colsample_bytree=.7)]:
   with self.assertRaises(AssertionError):r.validate_parameters('LGBMRanker',{**p,**extra})
 def test_veto_treatment_refused(self):
  p=dict(n_jobs=1,objective='binary',random_state=11,colsample_bytree=.8,subsample=.8,subsample_freq=1)
  for extra in [dict(subsample=.6),dict(bagging_seed=11),dict(feature_fraction_seed=11)]:
   with self.assertRaises(AssertionError):r.validate_parameters('LGBMClassifier',{**p,**extra})
 def test_admission_priority_first_eight(self):
  idle={c:100 for c in ad.ORDER+[c+64 for c in ad.ORDER]};self.assertEqual(ad.select(idle),ad.ORDER[:8]);idle[21+64]=84.999;self.assertEqual(ad.select(idle),[15,19,20,22,23,24,25,26])
 def test_admission_requires_core_and_sibling_without_rounding(self):
  idle={c:0 for c in ad.ORDER+[c+64 for c in ad.ORDER]}
  for c in [15,19,20]:idle[c]=idle[c+64]=85
  idle[34]=85;idle[34+64]=84.999;self.assertEqual(ad.select(idle),[15,19,20]);self.assertLess(len(ad.select(idle)),4)
 def test_forecast_efficiency_safety_and_reserve(self):
  now=datetime.datetime(2026,11,5,1,tzinfo=datetime.timezone.utc);f=ad.forecast(26982.458116,4,now);self.assertAlmostEqual(f['forecast_seconds'],26982.458116/(4*.9)*1.15+300);self.assertTrue(f['fits']);self.assertAlmostEqual(ad.forecast(26982.458116,8,now)['forecast_seconds']-300,(f['forecast_seconds']-300)/2)
 def test_forecast_boundary_and_worker_refusal(self):
  now=ad.DEADLINE-datetime.timedelta(seconds=1450);self.assertFalse(ad.forecast(3600,4,now)['fits']);self.assertTrue(ad.forecast(3600,4,now-datetime.timedelta(seconds=1))['fits'])
  for n in [3,9]:
   with self.assertRaises(AssertionError):ad.forecast(3600,n,now)
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
 def test_ties_fail(self):self.assertEqual(e.decide(self.pair(0,0),[1],[1],[0],[0])[1],'ROWSUB_NOT_ADVANCED')
 def test_below_margin_fails(self):self.assertEqual(e.decide(self.pair(.001999),[1],[1],[0],[0])[1],'ROWSUB_NOT_ADVANCED')
 def test_exact_margin_passes(self):self.assertEqual(e.decide(self.pair(.002),[1],[1],[0],[0])[1],'ADVANCE')
 def test_lcb_must_be_positive(self):self.assertEqual(e.decide(self.pair(lcb=0),[1],[1],[0],[0])[1],'ROWSUB_NOT_ADVANCED')
 def test_harm_fails(self):self.assertEqual(e.decide(self.pair(),[1],[1],[.2],[.1])[1],'ROWSUB_NOT_ADVANCED')
 def test_retention_fails(self):self.assertEqual(e.decide(self.pair(),[.89],[1],[0],[0])[1],'ROWSUB_NOT_ADVANCED')
 def test_root_consistency(self):
  self.assertEqual(e.decide(self.pair(per=[.01,-.01,-.01]*3),[1],[1],[0],[0])[1],'ROWSUB_NOT_ADVANCED');self.assertEqual(e.decide(self.pair(per=[.01,.01,-.01]*3),[1],[1],[0],[0])[1],'ADVANCE')
 def test_bootstrap_pairing(self):
  a=np.tile(np.linspace(-1,1,20),(9,1));z=e.paired_drop_top(a,a,draws=100);self.assertEqual(z['median'],0);self.assertEqual(z['ci95'],[0,0])
 def test_jaccard(self):self.assertEqual(e.jaccard([0,1,1],[0,1,0])['jaccard'],.5)
 def test_outer_scope_rejected(self):
  scope=rt.Scope({'train'},{'train','outer'},fit=True)
  with self.assertRaises(PermissionError):scope.hook('open',('/home/ubuntu/jope-data/h50/input/pools/outer.npz','r',0))
 def test_base_scope_rejected(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h345-solhwv/base/BASE_P0.npy','r',0))
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
