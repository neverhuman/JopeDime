"""New bridge tests; all measurements here are explicitly offline fixtures."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests'), str(ROOT / 'pipelines/continuous'), str(ROOT / 'pipelines/variants')]
import test_program_gate as fixture_module
from neural_outbox import Outbox
from jope_flywheel.program_observer import ProgramObserver
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.variant_outbox import VariantConsumer, claim_variant, census_queue


class Inputs:
    def __init__(self, ctx, descriptor):
        self.ctx, self.descriptor = ctx, descriptor
    def describe(self, identity):
        if identity != self.descriptor['variant_identity']:
            raise ValueError('unknown variant')
        return self.ctx, b'fixture', self.descriptor


class Worker:
    def __init__(self, queue, store, entry):
        self.queue, self.q, self.store, self.entry = queue, queue, store, entry
        self.calls = 0
        self.fail = None
    def once(self):
        self.calls += 1
        if self.fail:
            raise self.fail
        j = self.q.claim('offline-fixture', ['evaluate', 'publish'], lease_seconds=600)
        if j is None:
            return {'state': 'idle'}
        if j['stage'] == 'publish':
            self.store.entry = self.entry
        self.q.finish(j['id'], j['lease_token'], digest({'fixture': j['id']}), success=True)
        return {'state': 'completed', 'stage': j['stage'], 'job': j['id']}


class VariantTests(unittest.TestCase):
    def setUp(self):
        # Reuse the seven-artifact fixture without rerunning its historical tests.
        self.fixture = fixture_module.GateTests('test_complete_receipt_closure_required')
        self.fixture.setUp()
        self.fixture.fixture()
        f = self.fixture
        self.q, self.store, self.ctx = f.q, f.store, f.ctx
        self.entry = copy.deepcopy(self.store.entry)
        self.tmp = tempfile.TemporaryDirectory()
        self.outbox = Outbox(str(Path(self.tmp.name).resolve() / 'outbox.sqlite'),
                             model_origin_sha256='1' * 64, max_pending=100)
        self.pub = {'bucket': 'fixture', 'key': 'receipt', 'version_id': 'v1',
                    'sha256': '2' * 64, 'bytes': 123}
        self.actual = {'variant_identity': '3' * 64, 'parent_lineage_sha256': '4' * 64,
                       'partition': 'train', 'dataset_s3': {'receipt': self.pub}}
        self.descriptor = {'source_kind': 'flywheel_variant',
            'observation_dispatch': 'qualified_flywheel_worker_adapter',
            'variant_identity': self.actual['variant_identity'], 'context_id': self.ctx.identity,
            'context': asdict(self.ctx), 'parent_lineage_sha256': '4' * 64,
            'partition': 'train', 'dataset_publication': self.pub,
            'encoder_publication': dict(self.pub, key='encoder')}
        self.inputs = Inputs(self.ctx, self.actual)
        self.observer = ProgramObserver(self.q, self.inputs, self.store,
                                       plan_sha256='5' * 64, canonicalize=str.strip)
        self.worker = Worker(self.q, self.store, self.entry)
        self.worker.inputs = self.inputs
        self.store.record_refs = {}
        old_lookup = self.store.lookup
        def lookup(key):
            entry = old_lookup(key)
            if entry is not None:
                raw = canonical(entry)
                self.store.blobs['record'] = raw
                self.store.record_refs[key] = {'bucket': 'fixture', 'key': 'record',
                    'version_id': 'v1', 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest()}
            return entry
        self.store.lookup = lookup
        self.consumer = VariantConsumer(self.outbox, self.inputs, self.observer,
            self.worker, self.store, descriptors={self.ctx.identity: copy.deepcopy(self.descriptor)},
            canonicalize=str.strip, blocked_keys=[], failure_custody=lambda *a: None,
            owner='offline-fixture', deadline=self.outbox.now() + 1800)

    def tearDown(self):
        self.outbox.close()
        self.fixture.tearDown()
        self.tmp.cleanup()

    def record(self, step=1, phase='training', program=None, descriptor=None):
        d = copy.deepcopy(descriptor or self.descriptor)
        raw = self.fixture.dsl if program is None else program
        cid = self.ctx.identity
        train = [cid] if phase == 'training' else ['6' * 64]
        dev = [] if phase == 'training' else [cid]
        batch = {'batch_id': digest({'step': step}), 'campaign_sha256': '7' * 64,
            'checkpoint_sha256': '8' * 64, 'global_step': step,
            'manifest_sha256': '9' * 64, 'train_data_run_ids': train,
            'validation_data_run_ids': dev, 'emitted_train_data_run_ids': train}
        e = {'phase': phase, 'data_run_id': cid, 'context_id': cid,
            'parent_lineage_sha256': '4' * 64, 'input_descriptor': d,
            'raw_program_utf8': raw, 'raw_program_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'decoder_seed': 1, 'decoder_protocol_sha256': 'a' * 64,
            'decode_artifact_sha256': 'b' * 64}
        emissions = [e]
        if phase == 'validation':
            native = dict(e, phase='training', data_run_id='6' * 64,
                          context_id='6' * 64, input_descriptor={'source_kind': 'native_s1_context'})
            emissions.insert(0, native)
        self.outbox.record_batch(batch, emissions)

    def cache_fixture(self):
        import time
        from jope_flywheel.program_inputs import RegisteredInputs, BUCKET, PREFIX
        from test_s3_library import Fake
        class Counted(Fake):
            def __init__(self): super().__init__(); self.gets=0; self.heads=0
            def get_object(self, **kw): self.gets+=1; return super().get_object(**kw)
            def head_object(self, **kw): self.heads+=1; return super().head_object(**kw)
        inputs=object.__new__(RegisteredInputs);inputs.client=Counted()
        inputs.deadline=time.time()+60;inputs.version_cache=None;inputs.enable_version_cache(1024**2)
        variant='a'*64;key=PREFIX+'datasets/flywheel/v1/'+variant+'/metadata.json';raw=b'{"fixture":true}'
        inputs.client.objects[key]=(raw,{},'v1')
        ref={'bucket':BUCKET,'key':key,'version_id':'v1','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        return inputs,variant,ref,raw

    def test_dataset_cache_repeated_exact_read_uses_one_actual_get(self):
        inputs,variant,ref,raw=self.cache_fixture()
        for _ in range(3): self.assertEqual(inputs.dataset_read(ref,variant=variant,name='metadata.json'),raw)
        self.assertEqual(inputs.client.gets,1)
        self.assertEqual(inputs.cache_stats()['hits'],2)
        self.assertEqual(inputs.cache_stats()['backend_calls'],1)

    def test_dataset_cache_new_version_is_not_an_old_version_hit(self):
        inputs,variant,ref,raw=self.cache_fixture();inputs.dataset_read(ref,variant=variant,name='metadata.json')
        inputs.client.objects[ref['key']]=(raw,{},'v2')
        self.assertEqual(inputs.dataset_read(dict(ref,version_id='v2'),variant=variant,name='metadata.json'),raw)
        self.assertEqual(inputs.client.gets,2);self.assertEqual(inputs.cache_stats()['hits'],0)

    def test_dataset_cache_changed_hash_or_size_never_reuses_old_bytes(self):
        inputs,variant,ref,raw=self.cache_fixture();inputs.dataset_read(ref,variant=variant,name='metadata.json')
        for changes in ({'sha256':'f'*64},{'bytes':len(raw)+1}):
            with self.assertRaises(ValueError): inputs.dataset_read(dict(ref,**changes),variant=variant,name='metadata.json')
        self.assertEqual(inputs.cache_stats()['hits'],0)
        self.assertEqual(inputs.cache_stats()['backend_failures'],2)

    def test_dataset_namespace_is_checked_even_when_exact_ref_is_cached(self):
        inputs,variant,ref,raw=self.cache_fixture();inputs.dataset_read(ref,variant=variant,name='metadata.json')
        for change in ({'variant':'b'*64,'name':'metadata.json'},{'variant':variant,'name':'generation.json'}):
            with self.assertRaisesRegex(ValueError,'namespace'): inputs.dataset_read(ref,**change)
        self.assertEqual(inputs.client.gets,1);self.assertEqual(inputs.cache_stats()['hits'],0)

    def test_corrupt_dataset_cache_fails_without_backend_fallback(self):
        inputs,variant,ref,raw=self.cache_fixture();inputs.dataset_read(ref,variant=variant,name='metadata.json')
        cache=inputs.version_cache.cache;key=inputs.version_cache.identity(ref);entry=cache._items[key]
        cache._items[key]=type(entry)(b'corrupt',entry.sha256,entry.charge)
        with self.assertRaisesRegex(ValueError,'corrupted'): inputs.dataset_read(ref,variant=variant,name='metadata.json')
        self.assertEqual(inputs.client.gets,1);self.assertEqual(inputs.cache_stats()['integrity_failures'],1)

    def test_dataset_cache_hit_still_stops_at_deadline(self):
        inputs,variant,ref,raw=self.cache_fixture();inputs.dataset_read(ref,variant=variant,name='metadata.json');inputs.deadline=0
        with self.assertRaises(TimeoutError): inputs.dataset_read(ref,variant=variant,name='metadata.json')
        self.assertEqual(inputs.client.gets,1);self.assertEqual(inputs.cache_stats()['hits'],0)

    def test_dataset_get_failure_is_not_cached(self):
        inputs,variant,ref,raw=self.cache_fixture();del inputs.client.objects[ref['key']]
        with self.assertRaises(KeyError): inputs.dataset_read(ref,variant=variant,name='metadata.json')
        self.assertEqual(inputs.cache_stats()['entries'],0);self.assertEqual(inputs.cache_stats()['backend_failures'],1)
        # Fixture-only second request proves absence was not memoized; no live execution retry.
        inputs.client.objects[ref['key']]=(raw,{},'v1')
        self.assertEqual(inputs.dataset_read(ref,variant=variant,name='metadata.json'),raw)
        self.assertEqual(inputs.client.gets,2)

    def warmed_role_fixture(self):
        from jope_flywheel.program_io import sha
        inputs,variant,ref,raw=self.cache_fixture();inputs.dataset_read(ref,variant=variant,name='metadata.json')
        root=Path(self.tmp.name).resolve();(root/'roles').mkdir(exist_ok=True)
        def write(name,value):
            path=root/'roles'/name;body=canonical(value);path.write_bytes(body)
            return {'path':str(path),'sha256':sha(body),'bytes':len(body)}
        source=write('source.json',{})
        layer=write('layer.json',{'schema':'jope.dltrain-training-exclusion-layer/v1','entries':[]})
        registry=write('TRAINING-EXCLUSION-LAYERS-v1.json',{'layers':[layer]})
        target=write('gp-target-one-holdout-after-layers-v1.json',{'registry':registry,'views':['eligible']})
        inputs.rows={variant:{'parent':'c'*64}}
        inputs.parents={'c'*64:{'view_id':'eligible','retained_lineage_context_sha256':[]}}
        inputs.config={'generator_config':source}
        inputs.generator={'source_pins':[source],'parent_registry':source,'authority_pins':[source,layer,registry,target]}
        return inputs,variant,source,target

    def test_warm_cache_cannot_bypass_changed_local_source_pin(self):
        inputs,variant,source,target=self.warmed_role_fixture();Path(source['path']).write_text('{"changed":true}')
        with self.assertRaises(ValueError): inputs.describe(variant)
        self.assertEqual(inputs.client.gets,1);self.assertEqual(inputs.cache_stats()['hits'],0)

    def test_warm_cache_cannot_bypass_new_exclusion_authority(self):
        import os,time
        inputs,variant,source,target=self.warmed_role_fixture()
        p=Path(target['path']);new=p.with_name('gp-target-one-holdout-after-layers-v2.json');new.write_text('{}');os.utime(new,(time.time()+5,time.time()+5))
        with self.assertRaisesRegex(ValueError,'newer target authority'): inputs.describe(variant)
        self.assertEqual(inputs.client.gets,1);self.assertEqual(inputs.cache_stats()['hits'],0)

    def test_library_cache_keeps_live_heads_and_exact_changed_wrapper_versions(self):
        import time
        from variant_dispatch import Library
        from jope_flywheel.s3_library import PREFIX
        inputs,variant,ref,raw=self.cache_fixture();client=inputs.client;key=PREFIX+'d'*64+'/record.json'
        first=canonical({'fixture':1});client.objects[key]=(first,{'sha256':hashlib.sha256(first).hexdigest()},'v1')
        library=Library(client,deadline=time.time()+60,cache_bytes=1024**2)
        self.assertEqual(library.lookup('d'*64),{'fixture':1});self.assertEqual(library.lookup('d'*64),{'fixture':1})
        second=canonical({'fixture':2});client.objects[key]=(second,{'sha256':hashlib.sha256(second).hexdigest()},'v2')
        self.assertEqual(library.lookup('d'*64),{'fixture':2})
        self.assertEqual(client.heads,3);self.assertEqual(client.gets,2);self.assertEqual(library.cache_stats()['hits'],1)

    def test_library_put_readback_stays_real_then_repeated_closure_uses_verified_bytes(self):
        import time
        from variant_dispatch import Library
        inputs,variant,ref,raw=self.cache_fixture();client=inputs.client
        library=Library(client,deadline=time.time()+60,cache_bytes=1024**2)
        ref=library.put_artifact('program',b'(ridge)')
        self.assertEqual(client.gets,1)
        for _ in range(4):self.assertEqual(library.get_version(ref),(b'(ridge)','v1'))
        self.assertEqual(client.gets,1);self.assertEqual(library.cache_stats()['hits'],4)
        with self.assertRaisesRegex(ValueError,'namespace'):library.get_version(dict(ref,key='foreign'))

    def test_cli_cache_budget_is_explicit_bounded_and_disabled_for_legacy(self):
        from variant_dispatch import cache_budget
        self.assertEqual(cache_budget({}),0)
        self.assertEqual(cache_budget({'verified_cache_bytes':64*1024**2}),32*1024**2)
        for value in (True,-2,1,1024,64*1024**2+2,3*1024**2+1):
            with self.assertRaises(ValueError):cache_budget({'verified_cache_bytes':value})

    def test_phase_preference_skips_older_opposite_phase_and_preserves_native(self):
        self.record(step=1, phase='validation')
        self.record(step=2, phase='training')
        claim = claim_variant(self.outbox,'phase-fixture',lease_seconds=600,
                              deadline=self.outbox.now()+1800,preferred_phase='training')
        self.assertEqual(claim['body']['emission']['phase'],'training')
        self.assertEqual(claim['body']['batch']['global_step'],2)
        states=self.outbox.verify()['states']
        self.assertEqual(states['leased'],1)
        self.assertEqual(states['recorded'],2)

    def test_phase_preference_falls_back_without_reclaiming_previous_lease(self):
        self.record(step=1,phase='training')
        self.record(step=2,phase='training')
        claims=[claim_variant(self.outbox,'phase-fixture',lease_seconds=600,
                              deadline=self.outbox.now()+1800,preferred_phase='validation')for _ in range(2)]
        self.assertEqual([c['body']['batch']['global_step']for c in claims],[1,2])
        self.assertNotEqual(claims[0]['lease']['emission_id'],claims[1]['lease']['emission_id'])
        self.assertEqual(self.outbox.verify()['states']['leased'],2)

    def test_invalid_phase_preference_cannot_mutate_or_execute(self):
        self.record()
        before=self.outbox.verify()
        with self.assertRaisesRegex(ValueError,'qualified phase preference'):
            self.consumer.once(preferred_phase='outer-test')
        self.assertEqual(self.outbox.verify(),before)
        self.assertEqual(self.worker.calls,0)

    def test_cached_success_verifies_all_artifacts_and_outbox_chain(self):
        self.record()
        r = self.consumer.once()
        self.assertEqual(r['state'], 'cached_verified')
        self.assertEqual(self.worker.calls, 0)
        self.assertEqual(self.q.snapshot()['total_jobs'], 0)
        self.assertEqual(self.outbox.verify()['states']['cached_verified'], 1)

    def test_missing_result_runs_evaluate_then_publish_before_resolution(self):
        self.record()
        self.store.entry = None
        r = self.consumer.once()
        self.assertEqual(r['state'], 'published_verified')
        self.assertEqual(self.worker.calls, 2)
        self.assertEqual(self.q.snapshot()['counts']['publish']['completed'], 1)
        self.q.verify()
        self.assertEqual(self.outbox.verify()['states']['published_verified'], 1)

    def test_validation_preserves_role_and_leaves_native_proposals_alone(self):
        self.actual['partition'] = self.descriptor['partition'] = 'validation'
        self.consumer.descriptors[self.ctx.identity] = copy.deepcopy(self.descriptor)
        self.record(phase='validation')
        r = self.consumer.once()
        self.assertEqual(r['phase'], 'validation')
        self.assertFalse(r['optimizer_reward_allowed'])
        self.assertEqual(self.outbox.verify()['states']['recorded'], 1)

    def test_phase_mismatch_is_nonnumeric_and_never_executes(self):
        self.record(phase='validation')
        r = self.consumer.once()
        self.assertEqual(r['state'], 'unknown_attempt')
        self.assertEqual(self.worker.calls, 0)
        self.assertIsNone(json.loads(self.outbox.db.execute(
            "SELECT outcome FROM emissions WHERE state='unknown_attempt'").fetchone()[0])['record_ref'])

    def test_changed_encoder_descriptor_refused(self):
        d = copy.deepcopy(self.descriptor)
        d['encoder_publication']['version_id'] = 'changed'
        self.record(descriptor=d)
        self.assertEqual(self.consumer.once()['state'], 'unknown_attempt')
        self.assertEqual(self.worker.calls, 0)

    def test_changed_executable_target_refused(self):
        self.record()
        self.inputs.ctx = type(self.ctx)(**dict(asdict(self.ctx), target_sha256='c' * 64))
        self.assertEqual(self.consumer.once()['state'], 'unknown_attempt')
        self.assertEqual(self.worker.calls, 0)

    def test_wrong_cached_artifact_is_never_a_score(self):
        self.record()
        self.store.blobs['rust_source'] += b'corruption'
        self.assertEqual(self.consumer.once()['state'], 'unknown_attempt')
        self.assertEqual(self.worker.calls, 0)

    def test_lookup_failure_is_unknown_not_cache_miss(self):
        self.record()
        self.store.fail = True
        self.assertEqual(self.consumer.once()['state'], 'unknown_attempt')
        self.assertEqual(self.q.snapshot()['total_jobs'], 0)
        self.assertEqual(self.worker.calls, 0)

    def test_partial_worker_failure_keeps_queue_and_model_attempt(self):
        self.record()
        self.store.entry = None
        self.worker.fail = OSError('partial external outcome')
        r = self.consumer.once()
        self.assertEqual(r['state'], 'unknown_attempt')
        self.assertFalse(r['automatic_retry'])
        self.assertEqual(self.q.snapshot()['total_jobs'], 2)
        self.assertEqual(self.outbox.verify()['states']['unknown_attempt'], 1)

    def test_canonical_unknown_alias_cannot_be_reexecuted_in_later_batch(self):
        self.record()
        self.store.fail = True
        self.consumer.once()
        self.store.fail = False
        self.record(step=2, program='  ' + self.fixture.dsl + '  ')
        r = self.consumer.once()
        self.assertEqual(r['state'], 'unknown_attempt')
        self.assertIn('canonical alias', r['error'])
        self.assertEqual(self.worker.calls, 0)

    def test_historical_unknown_key_refuses_even_an_available_cache(self):
        from jope_flywheel.program_gate import request_identity
        self.record()
        self.consumer.blocked_keys = {request_identity(self.ctx, self.fixture.dsl)}
        r = self.consumer.once()
        self.assertIn('historical failed/unknown', r['error'])
        self.assertEqual(r['state'], 'unknown_attempt')
        self.assertEqual(self.worker.calls, 0)

    def test_expiring_window_cannot_claim_work(self):
        self.record()
        self.consumer.deadline = self.outbox.now() + 599
        with self.assertRaisesRegex(ValueError, 'reserve complete'):
            self.consumer.once()
        self.assertEqual(self.outbox.verify()['states']['recorded'], 1)

    def test_unrelated_pending_job_refused_before_any_worker_execution(self):
        self.record()
        self.store.entry = None
        foreign = 'gp-evaluate-' + 'd' * 64
        self.q.enqueue(foreign, 'evaluate', '5' * 64, {'unrelated': True})
        r = self.consumer.once()
        self.assertEqual(r['state'], 'unknown_attempt')
        self.assertIn('unrelated or unresolved', r['error'])
        self.assertEqual(self.worker.calls, 0)
        row = self.q.db.execute('SELECT state,attempts FROM jobs WHERE id=?', (foreign,)).fetchone()
        self.assertEqual(tuple(row), ('pending', 0))

    def test_mismatched_worker_inputs_refused_before_outbox_claim(self):
        self.worker.inputs = object()
        with self.assertRaisesRegex(ValueError, 'one exact queue'):
            VariantConsumer(self.outbox, self.inputs, self.observer, self.worker,
                self.store, descriptors={}, canonicalize=str.strip, blocked_keys=[],
                failure_custody=lambda *a: None, owner='fixture', deadline=self.outbox.now()+1800)

    def prepare_missing(self):
        self.store.entry = None
        self.observer.observe({'variant_identity':self.actual['variant_identity'],
            'descriptor_sha256':digest(self.actual),'context':asdict(self.ctx),
            'dsl':self.fixture.dsl,'phase':'training','checkpoint_sha256':'8'*64,
            'step':1,'producer':'fixture'})

    def test_historical_never_attempted_pair_is_not_a_retry(self):
        self.prepare_missing()
        self.assertEqual(census_queue(self.q.db)['blocked_evaluation_keys'], [])

    def test_historical_partial_evaluation_blocks_new_packet(self):
        self.prepare_missing(); self.worker.once()
        self.assertEqual(len(census_queue(self.q.db)['blocked_evaluation_keys']), 1)

    def test_historical_completed_pair_still_requires_future_library_readback(self):
        self.prepare_missing(); self.worker.once(); self.worker.once()
        self.assertEqual(census_queue(self.q.db)['blocked_evaluation_keys'], [])

    def test_historical_failed_pair_is_permanently_blocked(self):
        self.prepare_missing()
        job = self.q.claim('fixture', ['evaluate'])
        self.q.finish(job['id'],job['lease_token'],'e'*64,success=False,reason='fixture_failure')
        self.assertEqual(len(census_queue(self.q.db)['blocked_evaluation_keys']), 1)

    def test_historical_live_attempt_is_never_retried(self):
        self.prepare_missing(); self.q.claim('fixture',['evaluate'])
        self.assertEqual(len(census_queue(self.q.db)['blocked_evaluation_keys']), 1)

    def test_historical_materialized_attempt_tampering_refused(self):
        self.prepare_missing()
        self.q.db.execute('UPDATE jobs SET attempts=1')
        with self.assertRaisesRegex(ValueError,'lease differs'):
            census_queue(self.q.db)

    def test_historically_completed_key_with_missing_library_is_not_reexecuted(self):
        from jope_flywheel.program_gate import request_identity
        self.record(); self.store.entry = None
        self.consumer.completed_keys = {request_identity(self.ctx,self.fixture.dsl)}
        r = self.consumer.once()
        self.assertEqual(r['state'],'unknown_attempt')
        self.assertIn('historical completed',r['error'])
        self.assertEqual(self.worker.calls,0)
        self.assertEqual(self.q.db.execute('SELECT sum(attempts) FROM jobs').fetchone()[0],0)

    def test_historically_completed_key_allows_only_verified_cache(self):
        from jope_flywheel.program_gate import request_identity
        self.record(); self.consumer.completed_keys = {request_identity(self.ctx,self.fixture.dsl)}
        self.assertEqual(self.consumer.once()['state'],'cached_verified')
        self.assertEqual(self.worker.calls,0)

    def test_private_indexed_outbox_snapshot_preserves_exact_frozen_proof(self):
        from variant_dispatch import outbox_snapshot
        self.record()
        expected = self.outbox.verify()
        copydb = outbox_snapshot(Path(self.tmp.name).resolve() / 'outbox.sqlite')
        try:
            verifier = Outbox.__new__(Outbox); verifier.db = copydb
            self.assertEqual(verifier.verify(),expected)
            self.assertTrue(copydb.execute("SELECT name FROM sqlite_master WHERE name='variant_private_batch_lookup'").fetchone())
        finally:
            copydb.close()
        self.assertIsNone(self.outbox.db.execute("SELECT name FROM sqlite_master WHERE name='variant_private_batch_lookup'").fetchone())
        self.assertEqual(self.outbox.verify(),expected)

    def test_private_snapshot_does_not_invent_later_training_observations(self):
        from variant_dispatch import outbox_snapshot
        self.record()
        copydb = outbox_snapshot(Path(self.tmp.name).resolve() / 'outbox.sqlite')
        try:
            verifier = Outbox.__new__(Outbox); verifier.db = copydb
            before = verifier.verify()
            self.record(step=2)
            self.assertEqual(verifier.verify(),before)
            self.assertEqual(self.outbox.verify()['observations'],2)
            self.assertEqual(before['observations'],1)
        finally:
            copydb.close()

    def test_unknown_row_is_never_released_by_new_consumer(self):
        self.record()
        self.store.fail = True
        self.consumer.once()
        self.assertIsNone(claim_variant(self.outbox, 'other', lease_seconds=600,
                                       deadline=self.outbox.now() + 1800))
        self.assertEqual(self.outbox.verify()['states']['unknown_attempt'], 1)


if __name__ == '__main__':
    unittest.main()
