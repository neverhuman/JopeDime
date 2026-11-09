"""Deterministic negative controls for objective, scope, ties and decision."""
import unittest
import numpy as np
from run_h316 import normalized_fold,norm_objective,choose
from evaluate_h316 import paired_drop_top,decide

class H316Tests(unittest.TestCase):
    def test_mad_scale_invariance(self):
        fam={'a':'A','b':'A','c':'B','d':'B'};c=list(fam)
        p={x:{'utility_k':np.array([i, i+.5])} for x,i in zip(c,[1,3,1,5])}
        v,_=normalized_fold([1]*4,c,p,fam)
        q={x:{'utility_k':p[x]['utility_k']*(9 if fam[x]=='A' else 1)} for x in c}
        w,_=normalized_fold([1]*4,c,q,fam)
        np.testing.assert_allclose(v,w,rtol=0,atol=1e-12)
    def test_zero_mad_floor_and_winsorization(self):
        p={str(i):{'utility_k':np.array([1.,1.+g])} for i,g in enumerate([0,0,0,0,0,0,0,0,0,100])}
        c=list(p);f={x:'A' for x in c};v,log=normalized_fold([1]*10,c,p,f)
        self.assertEqual(log['scales']['A'],.001);self.assertLess(v.max(),100000);self.assertTrue(np.isfinite(v).all())
    def test_family_equal_median_resists_replication(self):
        self.assertEqual(norm_objective([('a',9)]*100+[('b',1),('c',2)]),2)
    def test_conservative_tie(self):
        t=[dict(tau=.05,k=5,a=2,inner_obj=1,inner_switches=20),dict(tau=.15,k=7,a=3,inner_obj=.99995,inner_switches=2)]
        self.assertEqual(choose(t,'inner_obj')[0]['k'],7)
    def test_guard_rejects_outer_before_lock(self):
        import sys
        from run_h316 import FS
        sys.path.insert(0,str(FS/'h308_ensvote2_trainonly_repartition'))
        from trainonly_loader import GuardedPackets
        p=GuardedPackets({'outer':3,'train':2},{'outer'})
        self.assertEqual(p['train'],2)
        with self.assertRaises(KeyError):p['outer']
        self.assertEqual(p.denied,1)
    def test_bootstrap_pairing_tie(self):
        d=np.tile(np.linspace(-.1,.1,20),(9,1));r=paired_drop_top(d,d,draws=100)
        self.assertEqual(r['median'],0);self.assertEqual(r['ci95'],[0,0]);self.assertEqual(r['lcb95'],0)
    def test_ties_never_advance(self):
        _,v=decide({'median':0,'lcb95':0},[1],[1],[0],[0]);self.assertEqual(v,'NORM_NOT_ADVANCED')
    def test_harm_blocks_positive_primary(self):
        _,v=decide({'median':1,'lcb95':1},[1],[1],[.2],[.1]);self.assertEqual(v,'NORM_NOT_ADVANCED')
    def test_retention_blocks_positive_primary(self):
        _,v=decide({'median':1,'lcb95':1},[.8],[1],[0],[0]);self.assertEqual(v,'NORM_NOT_ADVANCED')
    def test_all_gates_advance(self):
        _,v=decide({'median':1,'lcb95':1},[1],[1],[0],[0]);self.assertEqual(v,'NORM_ADVANCE')

if __name__=='__main__':unittest.main()
