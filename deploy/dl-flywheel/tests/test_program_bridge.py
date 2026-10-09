from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
from jope_flywheel import program_io, program_inputs
from jope_flywheel.program_observer import ProgramObserver, require_optimizer_admission
from jope_flywheel.program_worker import ProgramWorker, split_scale, compare_predictions
from jope_flywheel.protocol import Context, canonical, digest
from jope_flywheel.queue import Queue
from jope_flywheel.s3_library import S3Library
from test_s3_library import Fake, Error


class Inputs:
    def __init__(self):
        self.context = Context('regression','a'*64,'b'*64,'cut','c'*64,'d'*64,'e'*64,'f'*64,'rmse','prepared','minimize')
        self.descriptor = {'schema':'fixture','variant_identity':'7'*64,'context':asdict(self.context),'teaching_eligible':False}
        self.denied = False
    def describe(self,identity):
        if identity != '7'*64 or self.denied: raise ValueError('unknown or newly excluded context')
        return self.context,b'',self.descriptor


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        self.q = Queue(self.root/'q.sqlite'); self.inputs = Inputs(); self.client = Fake()
        self.store = S3Library(self.client,deadline=time.time()+60,nativecanonical_json=canonical)
        self.obs = ProgramObserver(self.q,self.inputs,self.store,plan_sha256='1'*64,canonicalize=lambda x:x.strip())
        self.request = {'variant_identity':'7'*64,'descriptor_sha256':digest(self.inputs.descriptor),
            'context':asdict(self.inputs.context),'dsl':'(pipe X (ridge))','phase':'training','checkpoint_sha256':'2'*64,
            'step':0,'producer':'unit-test-not-live-model'}
    def tearDown(self): self.q.close(); self.tmp.cleanup()
    def test_both_origins_one_evaluation_with_four_journal_events(self):
        a = self.obs.observe(self.request); b = self.obs.observe({**self.request,'phase':'validation'})
        self.assertEqual(a['evaluation_key'],b['evaluation_key']); self.assertEqual(self.q.snapshot()['total_jobs'],2)
        self.assertEqual(self.obs.journal.verify()['events'],4)
        self.assertFalse(a['optimizer_reward_allowed'])
    def test_invalid_dsl_is_durable_before_canonicalizer(self):
        def reject(text):
            self.assertEqual(self.obs.journal.verify()['unresolved_observations'].__len__(),1)
            raise ValueError('invalid DSL')
        self.obs.gate.canonicalize = reject
        with self.assertRaisesRegex(ValueError,'invalid DSL'): self.obs.observe(self.request)
        self.assertEqual(self.obs.journal.verify()['events'],2)
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
    def test_auth_error_logged_and_not_dispatched(self):
        self.client.denied = True
        with self.assertRaises(Error): self.obs.observe(self.request)
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
        self.assertEqual(self.obs.journal.verify()['unresolved_observations'],[])
    def test_unknown_context_and_changed_role_both_durable(self):
        with self.assertRaises(ValueError): self.obs.observe({**self.request,'variant_identity':'8'*64})
        self.inputs.denied = True
        with self.assertRaises(ValueError): self.obs.observe(self.request)
        self.assertEqual(self.obs.journal.verify()['events'],4)
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
    def test_wrong_descriptor_or_target_refused(self):
        for change in [{'descriptor_sha256':'0'*64},{'context':{**self.request['context'],'target_sha256':'9'*64}}]:
            with self.assertRaises(ValueError): self.obs.observe({**self.request,**change})
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
    def test_even_bad_phase_and_missing_field_leave_evidence(self):
        with self.assertRaises(ValueError): self.obs.observe({**self.request,'phase':'test'})
        del self.request['producer']
        with self.assertRaises(ValueError): self.obs.observe(self.request)
        self.assertEqual(self.obs.journal.verify()['events'],4)
    def test_journal_capacity_stops_producer_without_starting_work(self):
        self.obs.journal.maximum = 1; self.obs.observe(self.request)
        with self.assertRaisesRegex(ValueError,'capacity'): self.obs.observe(self.request)
        self.assertEqual(self.obs.journal.verify()['events'],2)
    def test_queue_capacity_one_cannot_leave_orphan_evaluation(self):
        self.q.max_jobs = 1
        with self.assertRaisesRegex(ValueError,'capacity'): self.obs.observe(self.request)
        self.assertEqual(self.q.snapshot()['total_jobs'],0)
        self.assertEqual(self.q.verify()['events'],0)
        self.assertEqual(self.obs.journal.verify()['events'],2)
    def test_tampered_journal_detected(self):
        self.obs.observe(self.request); self.q.db.execute("UPDATE program_emissions SET body='{}' WHERE seq=1")
        with self.assertRaisesRegex(ValueError,'integrity'): self.obs.journal.verify()
    def test_no_research_measurement_can_be_optimizer_reward(self):
        with self.assertRaises(RuntimeError): require_optimizer_admission({'state':'ready','measurement':{'rmse':.1}})
    def test_resume_preserves_observations_and_deduplicates_work(self):
        first = self.obs.observe(self.request); self.q.close(); self.q = Queue(self.root/'q.sqlite')
        self.obs = ProgramObserver(self.q,self.inputs,self.store,plan_sha256='1'*64,canonicalize=lambda x:x)
        self.assertEqual(self.obs.observe(self.request)['evaluation_key'],first['evaluation_key'])
        self.assertEqual(self.q.snapshot()['total_jobs'],2); self.assertEqual(self.obs.journal.verify()['events'],4)
    def test_publish_failure_stays_failed_and_does_not_retry(self):
        self.obs.observe(self.request)
        job = self.q.claim('fixture',['evaluate']); self.q.finish(job['id'],job['lease_token'],'3'*64,success=True)
        worker = ProgramWorker(self.q,self.inputs,self.store,None,{},self.root,time.time()+60,'1'*64)
        worker.publish = lambda job: (_ for _ in ()).throw(OSError('partial S3 write'))
        with self.assertRaises(OSError): worker.once()
        self.assertEqual(self.q.snapshot()['counts']['publish']['failed'],1)
        self.assertEqual(worker.once()['state'],'idle'); self.q.verify()
    def test_expired_lease_stays_unknown_not_completed(self):
        self.obs.observe(self.request)
        worker = ProgramWorker(self.q,self.inputs,self.store,None,{},self.root,time.time()+60,'1'*64)
        original = self.q.claim
        def claim(*a,**kw):
            kw['now'] = time.time()-70
            return original(*a,**kw)
        worker.evaluate = lambda job: (_ for _ in ()).throw(TimeoutError('window expired'))
        with mock.patch.object(self.q,'claim',side_effect=claim):
            with self.assertRaises(TimeoutError): worker.once()
        self.assertEqual(self.q.snapshot()['counts']['evaluate']['needs_reconcile'],1)
        self.assertEqual(worker.once()['state'],'idle'); self.q.verify()


class NativeBoundariesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
    def tearDown(self): self.tmp.cleanup()
    def test_scaler_uses_only_complement_and_preserves_row_order(self):
        x = np.arange(102*2,dtype=float).reshape(102,2); y = np.arange(102,dtype=float); folds = np.arange(102)%2
        a = split_scale(x,y,folds,0); x[folds==0] *= 1e6; y[folds==0] += 1e8
        b = split_scale(x,y,folds,0)
        self.assertEqual(a['scaler_sha256'],b['scaler_sha256']); self.assertEqual(a['train'],b['train'])
        self.assertNotEqual(a['targets'],b['targets']); self.assertEqual(a['targets'],list(range(0,102,2)))
    def test_parity_fixed_tolerance_and_actual_bit_exact_count(self):
        a = [1.]*25; b = [1.+1e-10]*25
        got = compare_predictions(a,b); self.assertEqual(got['bit_exact_rows'],0); self.assertTrue(got['passed'])
        with self.assertRaises(ValueError): compare_predictions(a,[1.01]*25)
        with self.assertRaises(ValueError): compare_predictions(a,[float('nan')]*25)
    def test_small_fold_refused(self):
        with self.assertRaises(ValueError): split_scale(np.ones((49,1)),np.arange(49),np.arange(49)%2,0)
    def test_log_bound_kills_and_records_failure(self):
        with self.assertRaises(ValueError):
            program_io.run([sys.executable,'-c','import os; os.write(1,b"x"*(5*1024**2))'],self.root,'logcap',time.time()+10,5)
        self.assertEqual((self.root/'logcap.log').stat().st_size,4*1024**2)
        receipt = json.loads((self.root/'logcap-command.json').read_bytes()); self.assertEqual(receipt['failure'],'log byte cap')
    def test_hanging_child_has_receipt_and_bounded_exit(self):
        start = time.monotonic()
        with self.assertRaises(ValueError):
            program_io.run([sys.executable,'-c','import time; time.sleep(60)'],self.root,'hang',time.time()+10,.1)
        self.assertLess(time.monotonic()-start,3)
        self.assertEqual(json.loads((self.root/'hang-command.json').read_bytes())['failure'],'deadline')
    def test_descendant_pipe_is_closed_without_unbounded_wait(self):
        command = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])'
        start = time.monotonic()
        with self.assertRaises(ValueError): program_io.run([sys.executable,'-c',command],self.root,'pipe',time.time()+10,5)
        self.assertLess(time.monotonic()-start,3)
    def test_campaign_monotonic_cannot_be_extended_by_backwards_clock(self):
        with mock.patch.object(program_io,'CAMPAIGN_MONOTONIC_END',None):
            with mock.patch.object(program_io.time,'time',return_value=100),mock.patch.object(program_io.time,'monotonic',return_value=5):
                program_io.bind_campaign_deadline(110)
            with mock.patch.object(program_io.time,'time',return_value=90),mock.patch.object(program_io.time,'monotonic',return_value=16):
                program_io.bind_campaign_deadline(110)
                with self.assertRaises(TimeoutError): program_io.tick(110)
    def test_symlinked_source_and_parent_refused(self):
        a = self.root/'a'; a.write_bytes(b'abc'); (self.root/'link').symlink_to(a)
        with self.assertRaises(OSError): program_io.read(self.root/'link')
        (self.root/'dir').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(OSError): program_io.read(self.root/'dir/a')


class InputTests(unittest.TestCase):
    def test_unknown_registry_identity_before_any_io(self):
        registry = object.__new__(program_inputs.RegisteredInputs); registry.rows = {}; registry.deadline = time.time()+30
        with self.assertRaisesRegex(ValueError,'unregistered'): registry.describe('a'*64)
    def test_dataset_version_hash_and_namespace(self):
        client = Fake(); raw = b'dataset'; key = program_inputs.PREFIX+'datasets/flywheel/v1/'+'a'*64+'/metadata.json'
        client.objects[key] = (raw,{},'v1')
        ref = {'bucket':program_inputs.BUCKET,'key':key,'version_id':'v1','sha256':program_io.sha(raw),'bytes':len(raw)}
        self.assertEqual(program_inputs.dataset_object(client,ref,variant='a'*64,name='metadata.json',deadline=time.time()+30),raw)
        for changes in ({'sha256':'0'*64},{'version_id':'null'},{'bucket':'foreign'}):
            with self.assertRaises(ValueError): program_inputs.dataset_object(client,{**ref,**changes},variant='a'*64,name='metadata.json',deadline=time.time()+30)
    def test_fresh_current_role_layer_and_new_target_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'gp-hwm-index').mkdir(); (root/'exclusions').mkdir()
            def write(path,obj):
                raw = canonical(obj); path.write_bytes(raw); return {'path':str(path),'sha256':program_io.sha(raw),'bytes':len(raw)}
            layer = write(root/'layer.json',{'schema':'jope.dltrain-training-exclusion-layer/v1','entries':[{'context_sha256':'a'*64,'view_ids':['blocked']}]})
            rp = root/'exclusions/TRAINING-EXCLUSION-LAYERS-v1.json'; reg = write(rp,{'layers':[layer]})
            tp = root/'gp-hwm-index/gp-target-one-holdout-after-layers-v1.json'; target = write(tp,{'registry':reg,'views':['eligible']})
            conf = {'authority_pins':[layer,reg,target]}
            self.assertEqual(program_inputs.current_roles(conf),({'eligible'},{'a'*64},{'blocked'}))
            newer = tp.with_name('gp-target-one-holdout-after-layers-v2.json'); newer.write_text('{}'); os.utime(newer,(time.time()+5,time.time()+5))
            with self.assertRaisesRegex(ValueError,'newer target'): program_inputs.current_roles(conf)


if __name__ == '__main__': unittest.main()
