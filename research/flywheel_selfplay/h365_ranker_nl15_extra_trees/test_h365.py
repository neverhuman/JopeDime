import unittest,tempfile,json,sys,inspect
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_h365 as r
import evaluate_h365 as e
import runtime_h365 as rt

class H365Tests(unittest.TestCase):
 def setUp(self):r.CURRENT_SEED=None;r.AUDIT=None
 def test_member_scope_clears_on_failure(self):
  def original(*args):self.assertEqual(r.CURRENT_SEED,71);raise ValueError('stop')
  with self.assertRaises(ValueError):r.member_scope(original)(0,0,0,0,71)
  self.assertIsNone(r.CURRENT_SEED)
 def test_ranker_and_veto_audit_keeps_real_classes_and_params(self):
  import lightgbm as lgb
  for cls in [lgb.LGBMRanker,lgb.LGBMClassifier]:
   params=dict(n_jobs=1,random_state=987,objective='lambdarank' if cls is lgb.LGBMRanker else 'binary',boosting_type='gbdt')
   if cls is lgb.LGBMRanker:
    from extra_trees_patch import LEAVES
    params.update(extra_trees=True,num_leaves=LEAVES,max_depth=3,min_child_samples=20,reg_lambda=1.0)
   if cls is lgb.LGBMClassifier:params.update(reg_alpha=1.0,reg_lambda=2.0)
   before=cls(**params).get_params();original=cls.__init__;sig=inspect.signature(original)
   with patch.object(cls,'__init__',r.ctor_audit(original,cls.__name__)):
    r.CURRENT_SEED=4321;obj=cls(**params)
    self.assertIsInstance(obj,cls);self.assertEqual(inspect.signature(cls.__init__),sig);self.assertEqual(obj.get_params(),before)
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
  a=np.tile(np.linspace(-1,1,20),(9,1));z=e.paired_drop_top(a,a,draws=100);self.assertEqual(z['median'],0);self.assertEqual(z['ci95'],[0,0]);self.assertEqual(z['lcb95'],0);self.assertEqual(len(z['per_root']),3);self.assertTrue(all(x['ci95']==[0,0] for x in z['per_root']))
 def test_jaccard(self):self.assertEqual(e.jaccard([0,1,1],[0,1,0])['jaccard'],.5)
 def test_scope_rejects_outer_before_open(self):
  scope=rt.Scope({'train'},{'train','outer'},fit=True)
  with self.assertRaises(PermissionError):scope.hook('open',('/home/ubuntu/jope-data/h50/input/pools/outer.npz','r',0))
  self.assertEqual(scope.denied,1)
 def test_scope_rejects_base_outcomes_during_fit(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h365-api01/base/BASE_P0.npy','r',0))
 def test_scope_rejects_archived_native_models_during_fit(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h365-api01/base_native/P0/refit/ranker.txt','r',0))
 def test_scope_rejects_incumbent_outcomes_during_fit(self):
  with self.assertRaises(PermissionError):rt.Scope({'train'},{'train'},fit=True).hook('open',('/home/ubuntu/jope-data/h365-api01/nl15/NL15_P0.npy','r',0))
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
