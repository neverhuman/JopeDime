import unittest
from dataclasses import replace
import numpy as np
from jope_flywheel import target_noise as tn
from jope_flywheel.augmentation import Parent,roles_sha256
from jope_flywheel.protocol import partition

class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.roles={0:'fit',1:'fit',2:'fit',3:'validation'}
        self.parent=Parent('a'*64,'b'*64,'c'*64,'y','d'*64,'e'*64,partition('d'*64,'e'*64),'cut','f'*64,roles_sha256(self.roles),'0'*64)
        self.y=np.array([1.,2.,4.,1000.],dtype=np.float64)
    def fit(self,**kw):
        args=dict(row_ids=[0,1,2],row_roles=self.roles,parent=self.parent,feature_variant_sha256='1'*64,seed=7,fraction=.05)
        args.update(kw);return tn.fit(self.y[:3],**args)
    def test_scale_only_fit(self):
        state=self.fit();self.assertEqual(state.train_std,float(np.std(self.y[:3])))
        with self.assertRaises(ValueError):tn.fit(self.y,row_ids=range(4),row_roles=self.roles,parent=self.parent,feature_variant_sha256='1'*64,seed=7,fraction=.05)
    def test_order_and_chunk_invariance(self):
        s=self.fit();out,r=tn.apply(self.y,row_ids=range(4),state=s)
        perm=[3,0,2,1];other,_=tn.apply(self.y[perm],row_ids=perm,state=s)
        np.testing.assert_array_equal(other,out[perm]);chunk,_=tn.apply(self.y[2:],row_ids=[2,3],state=s)
        np.testing.assert_array_equal(chunk,out[2:]);np.testing.assert_array_equal(self.y,[1.,2.,4.,1000.])
        self.assertNotEqual(r['target_sha256'],r['input_target_sha256']);self.assertFalse(r['parent_scores_inheritable'])
    def test_fit_order_identity(self):
        s=self.fit();p=[2,0,1]
        t=tn.fit(self.y[p],row_ids=p,row_roles=self.roles,parent=self.parent,feature_variant_sha256='1'*64,seed=7,fraction=.05)
        self.assertEqual(s.identity,t.identity)
    def test_zero_unchanged(self):
        s=self.fit(fraction=0);out,r=tn.apply(self.y,row_ids=range(4),state=s)
        np.testing.assert_array_equal(out,self.y);self.assertFalse(r['targets_changed'])
    def test_bounds_and_roles(self):
        for n in [-.01,.051,float('nan'),True]:
            with self.assertRaises(ValueError):self.fit(fraction=n)
        with self.assertRaises(ValueError):tn.apply(self.y,row_ids=[0,1,2,9],state=self.fit())
    def test_constant_nonzero_refused(self):
        self.y[:3]=1
        with self.assertRaises(ValueError):self.fit()
    def test_new_seed_target_identity(self):
        a,r=tn.apply(self.y,row_ids=range(4),state=self.fit())
        b,s=tn.apply(self.y,row_ids=range(4),state=self.fit(seed=8))
        self.assertNotEqual(r['target_identity'],s['target_identity']);self.assertFalse(np.array_equal(a,b))
    def test_reconstructed_and_tampered_state_refused(self):
        state=self.fit()
        for change in [{'fraction':.5},{'fraction':float('nan')},{'train_std':-1},{'seed':True},{'fit_row_ids':(0,3)},
                       {'row_roles':((0,'fit'),(0,'fit'))},{'dtype':'int32'},{'feature_variant_sha256':'bad'}]:
            with self.assertRaises(ValueError):replace(state,**change)
        object.__setattr__(state,'fraction',.5)
        with self.assertRaises(ValueError):tn.apply(self.y,row_ids=range(4),state=state)
if __name__=='__main__':unittest.main()
