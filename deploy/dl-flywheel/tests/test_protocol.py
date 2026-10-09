import dataclasses
import hashlib
import math
import unittest
from jope_flywheel import protocol as p


def h(s): return hashlib.sha256(str(s).encode()).hexdigest()

class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.ctx = p.Context('regression', h('dataset'), h('target'), 'fold0', h('eval'), h('vocab'), h('proto'), h('split'), 'rmse', 'target', 'minimize')
        self.a = dict(program_sha256=h('a'), context_id=self.ctx.identity, measurement='executed', status='completed', value=.8, score_receipt_sha256=h('ra'))
        self.b = dict(program_sha256=h('b'), context_id=self.ctx.identity, measurement='executed', status='completed', value=1.0, score_receipt_sha256=h('rb'))
    def test_identity_binds_every_comparison_dimension(self):
        for field in ('dataset_sha256','target_sha256','evaluator_sha256','vocabulary_sha256','protocol_sha256','split_sha256'):
            changed = dataclasses.replace(self.ctx, **{field:h('changed')})
            self.assertNotEqual(changed.identity,self.ctx.identity)
            with self.assertRaises(ValueError):p.compare(changed,self.a,self.b)
        with self.assertRaises(ValueError):dataclasses.replace(self.ctx, direction='unknown')
    def test_lineage_partition_inherited(self):
        parent = dict(dataset_sha256=h('parent'), parent_lineage_sha256=h('family'), split_namespace_sha256=h('split'), partition=p.partition(h('family'),h('split')))
        child = dict(parent, dataset_sha256=h('child'), parent_dataset_sha256=parent['dataset_sha256'])
        self.assertEqual(p.inherit_partition(parent,child),parent['partition'])
        for field in ('partition','parent_lineage_sha256','split_namespace_sha256','parent_dataset_sha256'):
            with self.assertRaises(ValueError):p.inherit_partition(parent,dict(child,**{field:h('wrong')}))
    def test_partition_is_stable_and_near_five_percent(self):
        roles=[p.partition(h(i),h('fixed')) for i in range(10000)]
        self.assertTrue(400<roles.count('validation')<600)
        self.assertEqual(roles[:100], [p.partition(h(i),h('fixed')) for i in range(100)])
    def test_quality_scale_and_degenerate_baseline(self):
        self.assertAlmostEqual(p.normalized_quality(.5,1,1e-12)['q_raw'],math.log(2), places=9)
        self.assertEqual(p.normalized_quality(0,0,1e-9)['q_raw'],0)
        self.assertTrue(p.normalized_quality(0,0,1e-9)['baseline_degenerate'])
        self.assertEqual(p.normalized_quality(1e20,1,1e-9)['q_opt'],-math.log(1000))
        for bad in (None,True,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):p.normalized_quality(bad,1,1e-9)
        with self.assertRaises(ValueError):p.normalized_quality(1,1,0)
    def test_opponents_frozen_order_invariant(self):
        f = p.freeze_opponents(self.ctx,[self.b,self.a],h('new'),4)
        self.assertEqual(f,p.freeze_opponents(self.ctx,[self.a,self.b],h('new'),4))
        self.assertEqual(f['winner']['program_sha256'],self.a['program_sha256'])
        self.assertEqual(p.freeze_opponents(self.ctx,[self.a],h('a'),4)['random'],None)
        with self.assertRaises(ValueError):p.freeze_opponents(self.ctx,[self.a,self.a],h('new'),4)
        with self.assertRaises(ValueError):p.freeze_opponents(self.ctx,[dict(self.a,measurement='predicted')],h('new'),4)
        with self.assertRaises(ValueError):p.freeze_opponents(self.ctx,[dict(self.a,context_id=h('other'))],h('new'),4)
    def test_frozen_random_samples_whole_pool(self):
        pool=[dict(self.a,program_sha256=h(i),value=float(i)) for i in range(10)]
        chosen={p.freeze_opponents(self.ctx,pool,h('candidate'),i)['random']['program_sha256'] for i in range(300)}
        self.assertEqual(chosen,{r['program_sha256'] for r in pool})
    def test_measured_wins_ties_failures_unknown(self):
        self.assertEqual(p.compare(self.ctx,self.a,self.b)['outcome'],'win')
        self.assertEqual(p.compare(self.ctx,self.b,self.a)['outcome'],'loss')
        self.assertEqual(p.compare(self.ctx,dict(self.a,value=1.0005),self.b)['outcome'],'tie')
        for status in ('failed','timeout','invalid'):
            result=p.compare(self.ctx,dict(self.a,status=status,value=None),self.b)
            self.assertEqual(result['outcome'],'loss');self.assertIsNone(result['candidate_value'])
        self.assertEqual(p.compare(self.ctx,dict(self.a,status='pending',value=None),self.b)['outcome'],'pending')
    def test_compare_rejects_prediction_nonfinite_and_self(self):
        for changes in (dict(measurement='predicted'),dict(value=float('nan')),dict(value=True),dict(status='mystery'),dict(program_sha256=self.b['program_sha256'])):
            with self.assertRaises(ValueError):p.compare(self.ctx,dict(self.a,**changes),self.b)
        with self.assertRaises(ValueError):p.compare(self.ctx,self.a,self.b,relative_tolerance=2)
        other=dataclasses.replace(self.ctx,direction='maximize')
        a,b=dict(self.a,context_id=other.identity),dict(self.b,context_id=other.identity)
        self.assertEqual(p.compare(other,a,b)['outcome'],'loss')
    def test_elo_zero_sum_and_rejects_pending(self):
        a,b=p.elo_update(1500,1500,'win');self.assertEqual((a,b),(1508,1492))
        a,b=p.elo_update(900,1900,'tie');self.assertAlmostEqual(a+b,2800)
        with self.assertRaises(ValueError):p.elo_update(1500,1500,'pending')
        with self.assertRaises(ValueError):p.elo_update(1500,1500,'win',0)

if __name__=='__main__': unittest.main()
