import unittest,sys,tempfile,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import run_h321 as r
import evaluate_h321 as e

class H321Tests(unittest.TestCase):
 def test_formula_signed_and_slot0(self):
  u=np.array([2.,0.,4.,1.,3.,1.,3.,2.]);y,mad=r.normalized_target(u)
  np.testing.assert_allclose(y,np.clip((u-u[0])/max(np.median(abs(u-np.median(u))),.001),-3,3));self.assertEqual(y[0],0);self.assertTrue((y<0).any());self.assertTrue((y>0).any())
 def test_floor_and_clipping(self):
  y,mad=r.normalized_target([0,0,0,0,0,0,-10,10]);self.assertEqual(mad,0);self.assertEqual(y[-2],-3);self.assertEqual(y[-1],3)
 def test_context_affine_invariance_above_floor(self):
  u=np.arange(8,dtype=float);a,_=r.normalized_target(u);b,_=r.normalized_target(7*u+13);np.testing.assert_array_equal(a,b)
 def test_raw_labels_not_mutated(self):
  u=np.arange(8,dtype=float);old=u.copy();r.normalized_target(u);np.testing.assert_array_equal(u,old)
 def test_contexts_independent(self):
  a,_=r.normalized_target(np.arange(8));r.normalized_target(np.ones(8)*1e5);b,_=r.normalized_target(np.arange(8));np.testing.assert_array_equal(a,b)
 def test_regression_adapter_keeps_numeric_params(self):
  calls={}
  class Fake:
   def __init__(self,**kw):calls.update(kw)
   def fit(self,X,y,**kw):calls['fit_kwargs']=kw;calls['y']=np.array(y)
  with patch('lightgbm.LGBMRegressor',Fake):r.fit_normalized(np.zeros((8,568)),np.linspace(-3,3,8),1,8,.05,100,123)
  self.assertEqual(calls['objective'],'regression_l2');self.assertEqual(calls['metric'],'l2');self.assertEqual(calls['fit_kwargs'],{});self.assertEqual(calls['max_depth'],3);self.assertEqual(calls['num_leaves'],7);self.assertEqual(calls['n_jobs'],1);self.assertEqual(calls['random_state'],123);self.assertEqual(calls['n_estimators'],100)
 def test_ranker200_wrapper(self):
  with patch.object(r,'fit_normalized',return_value='ok') as f:
   self.assertEqual(r.fit_normalized_200('X','y',1,8,.03,9),'ok');self.assertEqual(f.call_args.args[5],200)
 def test_invalid_target_refused(self):
  with self.assertRaises(AssertionError):r.fit_normalized(np.zeros((8,2)),np.ones(8)*4,1,8,.05,50,1)
 def test_bootstrap_pairing_tie(self):
  a=np.tile(np.linspace(-1,1,20),(9,1));b=e.paired_drop_top(a,a,draws=100);self.assertEqual(b['median'],0);self.assertEqual(b['ci95'],[0,0]);self.assertEqual(b['lcb95'],0)
 def test_ties_never_advance(self):
  _,v=e.decide({'median':0,'lcb95':0},[1],[1],[0],[0]);self.assertEqual(v,'CNORM_NOT_ADVANCED')
 def test_harm_blocks(self):
  _,v=e.decide({'median':1,'lcb95':1},[1],[1],[.2],[.1]);self.assertEqual(v,'CNORM_NOT_ADVANCED')
 def test_retention_blocks(self):
  _,v=e.decide({'median':1,'lcb95':1},[.8],[1],[0],[0]);self.assertEqual(v,'CNORM_NOT_ADVANCED')
 def test_positive_gates(self):
  _,v=e.decide({'median':1,'lcb95':1},[1],[1],[0],[0]);self.assertEqual(v,'CNORM_ADVANCE')
 def test_seed_jaccard(self):
  self.assertEqual(e.jaccard([0,1,1],[0,1,0])['jaccard'],.5);self.assertEqual(e.jaccard([0],[0])['jaccard'],1)
 def test_missing_lock_prevents_outer_read(self):
  class Packets(dict):
   reads=0
   def __getitem__(self,key):self.reads+=1;raise AssertionError('outer access')
  packets=Packets();model=SimpleNamespace(G=dict(train_rows=[dict(context_sha256='x',family_id='F')],fam_by_cid={'x':'F'},a302=SimpleNamespace(outer_fold_of_family=lambda f:0),P8=packets))
  parent=r.P/'test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as t,patch.object(r,'DATA',Path(t)),patch.dict(r.R,{'r303':model},clear=True):
   with self.assertRaises(FileNotFoundError):r.score('P0')
  self.assertEqual(packets.reads,0)
 def test_outer_access_guard(self):
  sys.path.insert(0,str(r.FS/'h308_ensvote2_trainonly_repartition'));from trainonly_loader import GuardedPackets
  p=GuardedPackets({'train':1,'outer':2},{'outer'});self.assertEqual(p['train'],1)
  with self.assertRaises(KeyError):p['outer']
  self.assertEqual(p.denied,1)

if __name__=='__main__':unittest.main()
