import hashlib
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from jope_flywheel.queue import Queue

def h(x):return hashlib.sha256(str(x).encode()).hexdigest()

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)/'queue.db';self.q=Queue(self.p)
    def tearDown(self):self.q.close();self.tmp.cleanup()
    def add(self,name='a',stage='augment',deps=()):return self.q.enqueue(name,stage,h('plan'),{'input_sha256':h(name)},deps)
    def test_idempotent_capacity_collision_and_dependencies(self):
        self.assertTrue(self.add());self.assertFalse(self.add())
        with self.assertRaises(ValueError):self.q.enqueue('a','encode',h('plan'),{})
        with self.assertRaises(ValueError):self.add('b','encode',['missing'])
        with self.assertRaises(ValueError):self.add('b','encode',['b'])
        self.assertTrue(self.add('b','encode',['a']))
        self.assertIsNone(self.q.claim('worker',['encode']))
        self.assertEqual(self.q.verify()['events'],2)
    def test_dag_success_and_duplicate_finish(self):
        self.add();self.add('b','encode',['a'])
        a=self.q.claim('w',['augment']);self.assertEqual(a['id'],'a')
        self.assertIsNone(self.q.claim('w2',['encode']))
        self.assertTrue(self.q.finish('a',a['lease_token'],h('receipt'),success=True))
        self.assertFalse(self.q.finish('a',a['lease_token'],h('receipt'),success=True))
        b=self.q.claim('w2',['encode']);self.assertEqual(b['id'],'b')
        with self.assertRaises(ValueError):self.q.finish('b','wrong',h('r'),success=True)
        self.q.finish('b',b['lease_token'],h('r'),success=False,reason='encoder_failed')
        self.assertEqual(self.q.snapshot()['counts']['encode']['failed'],1);self.q.verify()
    def test_expired_work_never_retries_and_requires_reconciliation(self):
        self.add();self.add('second');a=self.q.claim('w',['augment'],now=100,lease_seconds=10)
        with self.assertRaises(ValueError):self.q.finish('a',a['lease_token'],h('r'),success=True,now=111)
        self.assertEqual(self.q.snapshot()['counts']['augment']['needs_reconcile'],1)
        self.assertIsNone(self.q.claim('another',['augment'],now=112))
        self.q.reconcile('a',h('readback'),completed=True,reason='exact_output_readback_verified')
        self.assertEqual(self.q.snapshot()['counts']['augment']['completed'],1)
        self.assertEqual(self.q.claim('another',['augment'])['id'],'second');self.q.verify()
    def test_failed_dependency_blocks_child(self):
        self.add();self.add('b','encode',['a']);a=self.q.claim('w',['augment'])
        self.q.finish('a',a['lease_token'],h('failed'),success=False)
        self.assertIsNone(self.q.claim('w',['encode']));self.q.verify()
    def test_nan_payload_and_limits(self):
        for payload in ({'n':float('nan')},{'n':float('inf')}):
            with self.assertRaises(ValueError):self.q.enqueue('bad','encode',h('plan'),payload)
        with self.assertRaises(ValueError):self.q.claim('w',['unknown'])
        with Queue(Path(self.tmp.name)/'small.db',max_jobs=1) as q:
            q.enqueue('one','encode',h('p'),{})
            with self.assertRaises(ValueError):q.enqueue('two','encode',h('p'),{})
    def packet(self):
        return [dict(job_id='evaluate',stage='evaluate',plan_sha256=h('plan'),payload={}),
                dict(job_id='publish',stage='publish',plan_sha256=h('plan'),payload={},dependencies=['evaluate'])]
    def test_packet_capacity_rolls_back_evaluation_and_history(self):
        self.q.max_jobs=1
        with self.assertRaisesRegex(ValueError,'capacity'):self.q.enqueue_many(self.packet())
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
        self.assertEqual(self.q.verify()['events'],0)
        self.assertIsNone(self.q.claim('worker',['evaluate']))
    def test_packet_is_idempotent_and_dependency_gates_publication(self):
        self.q.max_jobs=2
        self.assertEqual(self.q.enqueue_many(self.packet()),[True,True])
        self.assertEqual(self.q.enqueue_many(self.packet()),[False,False])
        self.assertIsNone(self.q.claim('worker',['publish']))
        job=self.q.claim('worker',['evaluate'])
        self.q.finish(job['id'],job['lease_token'],h('evaluated'),success=True)
        self.assertEqual(self.q.claim('worker',['publish'])['id'],'publish');self.q.verify()
    def test_packet_rejection_preserves_existing_jobs(self):
        self.add();before=self.q.verify()
        for mutation in ('dependency','identity','plan','duplicate'):
            packet=self.packet()
            if mutation=='dependency':packet[1]['dependencies']=['missing']
            if mutation=='identity':packet[1]['job_id']='a'
            if mutation=='plan':packet[1]['plan_sha256']=h('wrong-plan')
            if mutation=='duplicate':packet[1]['job_id']='evaluate'
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):self.q.enqueue_many(packet)
                self.assertEqual(self.q.snapshot()['total_jobs'],1);self.assertEqual(self.q.verify(),before)
    def test_packet_sql_failure_rolls_back_first_job(self):
        self.q.db.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON jobs WHEN NEW.stage='publish' BEGIN SELECT RAISE(ABORT,'injected storage failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.q.enqueue_many(self.packet())
        self.assertEqual(self.q.snapshot()['total_jobs'],0);self.assertEqual(self.q.verify()['events'],0)
    def test_concurrent_claim_only_one_winner(self):
        self.add();result=[];errors=[];barrier=threading.Barrier(4)
        def worker(i):
            try:
                with Queue(self.p) as q:
                    barrier.wait();result.append(q.claim(str(i),['augment']))
            except BaseException as e:errors.append(e)
        threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
        for t in threads:t.start()
        for t in threads:t.join(timeout=10)
        self.assertFalse(errors);self.assertEqual(sum(r is not None for r in result),1);self.q.verify()
    def test_lease_tampering_is_detected(self):
        self.add();a=self.q.claim('worker',['augment']);self.q.verify()
        self.q.db.execute("UPDATE jobs SET lease_token='unrecorded' WHERE id='a'")
        with self.assertRaises(ValueError):self.q.verify()
    def test_reopen_and_tamper_detection(self):
        self.add();self.q.verify()
        self.q.db.execute("UPDATE jobs SET state='completed' WHERE id='a'")
        with self.assertRaises(ValueError):self.q.verify()
        self.q.db.execute("UPDATE jobs SET state='pending' WHERE id='a'")
        self.q.db.execute("UPDATE transitions SET detail='{}'")
        with self.assertRaises(ValueError):self.q.verify()

if __name__=='__main__':unittest.main()
