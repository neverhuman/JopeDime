"""CPU-only adversarial fixtures; no real GP, network, S3 or live queues."""
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import sqlite3
import tempfile
import time
import threading
import ast
import inspect
import unittest
from unittest.mock import patch

from neural_outbox import Outbox, canonical, digest, sha
from readonly_snapshot import readonly_snapshot, verified_snapshot
from native_global_guard import GlobalGuard, BlockedKey, CacheRequired, GLOBAL_DEADLINE


def canonicalize(program):
    return ' '.join(program.split())


def raced_execution(config, current, ready, result):
    guard = GlobalGuard(config, canonicalize=canonicalize, packet_sha256='e' * 64,
                        deadline=time.time() + 40)
    ready.wait(10)
    try:
        guard.execute_once('1' * 64, '(ridge)', '9' * 64, current,
                           lambda: result.put('executed'))
    except BlockedKey:
        result.put('blocked')
    finally:
        guard.close()


class SuccessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.boxes = [Outbox(self.root / ('outbox%d.sqlite' % i), model_origin_sha256='a' * 64) for i in range(2)]
        for box in self.boxes:self.addCleanup(box.close)
        self.counter = 0

    def emit(self, box=0, *, program='(ridge)', cid='1' * 64):
        self.counter += 1
        b = dict(batch_id=digest({'counter': self.counter, 'box': box}), campaign_sha256='b' * 64,
                 checkpoint_sha256='c' * 64, global_step=self.counter, manifest_sha256='d' * 64,
                 train_data_run_ids=[cid], validation_data_run_ids=[], emitted_train_data_run_ids=[cid])
        e = dict(phase='training', data_run_id=cid, context_id=cid, parent_lineage_sha256='2' * 64,
                 input_descriptor={'source_kind': 'native_s1_context', 'context_id': cid,
                                   'parent_lineage_sha256': '2' * 64}, raw_program_utf8=program,
                 raw_program_sha256=sha(program.encode()), decoder_seed=4,
                 decoder_protocol_sha256='3' * 64, decode_artifact_sha256='4' * 64)
        return self.boxes[box].record_batch(b, [e])['emission_ids'][0]

    def claim(self, box=0):
        claim = self.boxes[box].claim('fixture', lease_seconds=30, deadline=time.time() + 40)
        return {'path': str(self.root / ('outbox%d.sqlite' % box)), 'lease': claim['lease'],
                'body_sha256': digest(claim['body'])}

    def finish(self, box, state):
        c = self.claim(box)
        success = state in ('cached_verified', 'published_verified')
        ref = dict(bucket='fixture', key='result', version_id='exact-v1', sha256='8' * 64, bytes=3)
        self.boxes[box].finish(c['lease'], dict(state=state, evaluation_key='9' * 64 if success else None,
                              record_ref=ref if success else None,
                              attempt_custody_ref=ref if state == 'failure_custody_verified' else None,
                              error=None if success else 'fixture unknown'), verify_outcome=lambda *_: True)

    def config(self, entries=(), extras=()):
        evidence = self.root / 'closed-writers.json'
        if not evidence.exists():evidence.write_bytes(canonical({'fixture_only': True, 'legacy_writers': 'physically closed'}))
        ref = lambda p: dict(path=str(p), sha256=sha(p.read_bytes()), bytes=p.stat().st_size)
        sources = [dict(path=str(self.root / ('outbox%d.sqlite' % i)), kind='outbox', config=o.config)
                   for i, o in enumerate(self.boxes)] + list(extras)
        sources.sort(key=lambda s: s['path'])
        census = dict(schema='jope.native-global-census/v1', scope_sha256='5' * 64,
                      canonicalizer_sha256='6' * 64, source_paths=[s['path'] for s in sources],
                      coverage_complete=True, all_legacy_writers_stopped=True,
                      source_identities={d['path']:{'device':Path(d['path']).stat().st_dev,'inode':Path(d['path']).stat().st_ino} for d in sources},
                      closure_evidence=[ref(evidence)], entries=list(entries))
        p = self.root / ('census-' + digest(census) + '.json')
        p.write_bytes(canonical(census))
        return dict(schema='jope.native-global-execution-guard/v1', registry_path=str(self.root / 'global.sqlite'),
                    scope_sha256='5' * 64, canonicalizer_sha256='6' * 64,
                    global_deadline_unix=GLOBAL_DEADLINE, sources=sources, census=ref(p), max_snapshot_bytes=16 * 1024**2)

    def guard(self, config=None):
        g = GlobalGuard(config or self.config(), canonicalize=canonicalize,
                        packet_sha256='e' * 64, deadline=time.time() + 40)
        self.addCleanup(g.close)
        return g

    def test_snapshot_private_indexes_equivalent_no_source_mutation(self):
        for i in range(8):self.emit(cid=digest(i))
        box = self.boxes[0]
        proof = box.verify()
        before = {str(p): sha(p.read_bytes()) for p in self.root.glob('outbox0.sqlite*')}
        schema = box.db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
        db, actual = verified_snapshot(self.root / 'outbox0.sqlite', deadline=time.time() + 10,
                                       expected_config=box.config)
        try:
            self.assertEqual(actual, proof)
            self.assertIn('native_private_batch_lookup', [r[0] for r in db.execute('SELECT name FROM sqlite_master')])
        finally:db.close()
        self.assertEqual(schema, box.db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall())
        # SQLite shared-memory locking bookkeeping can vary; database/WAL bytes cannot.
        for name, pin in before.items():
            if not name.endswith('-shm'):self.assertEqual(sha(Path(name).read_bytes()), pin)

    def test_snapshot_fixed_transaction_under_concurrent_append(self):
        self.emit()
        original = sqlite3.Connection.backup
        # Source read transaction is fixed before a new committed source emission.
        class Source(sqlite3.Connection):
            def backup(connection, target, **kwargs):
                self.emit(cid='7' * 64)
                return original(connection, target, **kwargs)
        connect = sqlite3.connect
        def factory(*a, **kw):
            if a[0] != ':memory:':kw['factory'] = Source
            return connect(*a, **kw)
        with patch('readonly_snapshot.sqlite3.connect', side_effect=factory):
            db, proof = verified_snapshot(self.root / 'outbox0.sqlite', deadline=time.time() + 10)
        db.close()
        self.assertEqual(proof['observations'], 1)
        self.assertEqual(self.boxes[0].verify()['observations'], 2)

    def test_snapshot_refuses_symlink_size_expiry_and_corruption(self):
        self.emit()
        alias = self.root / 'alias.sqlite';alias.symlink_to(self.root / 'outbox0.sqlite')
        for path, kwargs in [(alias, {}), (self.root/'outbox0.sqlite', {'max_bytes': 1024}),
                             (self.root/'outbox0.sqlite', {'deadline': time.time()-1}),
                             (self.root/'outbox0.sqlite', {'kind': 'outbox; DROP TABLE config;--'})]:
            with self.assertRaises(ValueError):
                readonly_snapshot(path, **{'deadline': time.time()+10, **kwargs})
        self.boxes[0].db.execute("UPDATE emissions SET state='unknown_attempt'")
        with self.assertRaises(ValueError):verified_snapshot(self.root/'outbox0.sqlite', deadline=time.time()+10)

    def test_different_raw_alias_failed_unknown_blocks_globally(self):
        for state in ('unknown_attempt', 'failure_custody_verified', 'pending_unsupported'):
            with self.subTest(state=state):
                # Separate context for each state, with whitespace-only raw aliases.
                cid=digest(state);self.emit(1, program=' (ridge)  ', cid=cid);self.finish(1,state)
                g=self.guard()
                with self.assertRaises(BlockedKey):g.check(cid,'(ridge)')

    def test_completed_cache_miss_never_executes(self):
        self.emit(1, program=' (ridge) ');self.finish(1,'published_verified')
        self.emit(0);current=self.claim(0);g=self.guard();calls=[]
        self.assertTrue(g.check('1'*64,'(ridge)',current=current))
        with self.assertRaises(CacheRequired):g.execute_once('1'*64,'(ridge)','9'*64,current,lambda:calls.append(1))
        self.assertEqual(calls,[]);self.assertEqual(g.verify()['claims'],0)

    def test_leased_alias_blocks_even_expired_or_future_clock(self):
        self.emit(1);current=self.claim(1);g=self.guard()
        for clock in (current['lease']['expires_at']+1,current['lease']['started_at']-10):
            g.now=lambda:clock
            with self.assertRaises(BlockedKey):g.check('1'*64,'(ridge)')

    def test_postclaim_new_alias_prevents_native_callback(self):
        self.emit(0);current=self.claim(0);g=self.guard();calls=[]
        old=g._postclaim_check
        def insert(*a):
            self.emit(1,program='  (ridge)');self.finish(1,'unknown_attempt')
            return old(*a)
        with patch.object(g,'_postclaim_check',side_effect=insert):
            with self.assertRaises(BlockedKey):g.execute_once('1'*64,'(ridge)','9'*64,current,lambda:calls.append(1))
        self.assertEqual(calls,[]);self.assertEqual(g.verify()['claims'],1)
        self.assertEqual(g.db.execute('SELECT state FROM claims').fetchone()[0],'unknown')

    def test_global_once_callback_failure_and_success_both_nonretryable(self):
        for fail in (False,True):
            cid=digest({'fail':fail});self.emit(cid=cid);current=self.claim();g=self.guard();calls=[]
            def execute():
                calls.append(1)
                if fail:raise OSError('unknown external outcome')
                return 'actual-result'
            if fail:
                with self.assertRaises(OSError):g.execute_once(cid,'(ridge)','9'*64,current,execute)
            else:self.assertEqual(g.execute_once(cid,'(ridge)','9'*64,current,execute),'actual-result')
            with self.assertRaises(BlockedKey):g.execute_once(cid,'(ridge)','9'*64,current,execute)
            self.assertEqual(calls,[1]);g.verify()

    def test_complete_only_returns_cache_requirement(self):
        self.emit();current=self.claim();g=self.guard()
        g.execute_once('1'*64,'(ridge)','9'*64,current,lambda:'result')
        ref=dict(bucket='b',key='k',version_id='v',sha256='8'*64,bytes=1)
        g.complete('1'*64,'(ridge)',ref)
        self.assertTrue(g.check('1'*64,'(ridge)',current=current));g.verify()

    def test_concurrent_processes_exact_one_callback(self):
        self.emit();current=self.claim();config=self.config()
        g=self.guard(config)  # Registry initialized before concurrent constructors.
        ctx=multiprocessing.get_context('spawn');ready=ctx.Event();result=ctx.Queue()
        processes=[ctx.Process(target=raced_execution,args=(config,current,ready,result)) for _ in range(2)]
        for p in processes:p.start()
        ready.set()
        for p in processes:p.join(15);self.assertEqual(p.exitcode,0)
        values=sorted(result.get(timeout=2) for _ in range(2))
        self.assertEqual(values,['blocked','executed']);self.assertEqual(g.verify()['claims'],1)

    def test_global_registry_tamper_and_census_drift_refused(self):
        self.emit();current=self.claim();g=self.guard();g.execute_once('1'*64,'(ridge)','9'*64,current,lambda:None)
        g.db.execute("UPDATE claims SET state='completed'")
        with self.assertRaises(ValueError):g.verify()
        config=self.config();config['canonicalizer_sha256']='0'*64
        with self.assertRaises(ValueError):self.guard(config)

    def test_historical_census_unknown_overrides_completed(self):
        common=dict(context_id='1'*64,canonical_program_sha256=sha(b'(ridge)'),evaluation_key='9'*64,source=str(self.root/'closed-writers.json'))
        g=self.guard(self.config([{**common,'disposition':'completed_cache_only'},{**common,'disposition':'blocked'}]))
        with self.assertRaises(BlockedKey):g.check('1'*64,'(ridge)')

    def test_dynamic_queue_attempt_after_preflight_blocks(self):
        from jope_flywheel.queue import Queue
        p=self.root/'historical.sqlite';q=Queue(p);self.addCleanup(q.close)
        payload={'evaluation_key':'9'*64,'context':{'identity':'1'*64},'program':' (ridge) '}
        q.enqueue('gp-evaluate-'+('9'*64),'evaluate','5'*64,payload)
        g=self.guard(self.config(extras=[dict(path=str(p),kind='queue',config=None)]))
        self.assertFalse(g.check('1'*64,'(ridge)'))
        q.claim('worker',['evaluate'],lease_seconds=1)
        with self.assertRaises(BlockedKey):g.check('1'*64,'(ridge)')

    def test_real_queue_context_shape_and_publish_unknown_are_blocked(self):
        from jope_flywheel.queue import Queue
        from jope_flywheel.protocol import Context
        from native_global_guard import queue_aliases
        context=dict(task='regression',dataset_sha256='a'*64,target_sha256='b'*64,cut_id='actual-cut',
                     evaluator_sha256='c'*64,vocabulary_sha256='d'*64,protocol_sha256='e'*64,
                     split_sha256='f'*64,metric='rmse',units='prepared_synthetic_target_units',direction='minimize')
        cid=Context(**context).identity;p=self.root/'real-queue.sqlite';q=Queue(p);self.addCleanup(q.close)
        key='9'*64;program=' (ridge) '
        q.enqueue('eval','evaluate','5'*64,dict(context=context,evaluation_key=key,program=program,program_sha256=sha(program.encode())))
        q.enqueue('pub','publish','5'*64,dict(evaluation_key=key),dependencies=['eval'])
        g=self.guard(self.config(extras=[dict(path=str(p),kind='queue',config=None)]))
        self.assertFalse(g.check(cid,'(ridge)'))
        q.claim('worker',['evaluate'],lease_seconds=1)
        with self.assertRaises(BlockedKey):g.check(cid,'(ridge)')
        # Read-only extraction includes publication-only states; no invented context.identity.
        self.assertEqual(list(queue_aliases(q.db))[0]['context_id'],cid)
        q.db.execute("UPDATE jobs SET state='completed' WHERE id='eval'")
        q.db.execute("UPDATE jobs SET state='needs_reconcile',attempts=1 WHERE id='pub'")
        with self.assertRaises(BlockedKey):g.check(cid,'(ridge)')

    def test_dispatch_skips_failed_alias_without_leasing_and_claims_fresh(self):
        from native_dispatch import claim_native
        old=self.emit();self.emit(1,program=' (ridge)');self.finish(1,'unknown_attempt')
        fresh=self.emit(cid='7'*64);g=self.guard()
        c=claim_native(self.boxes[0],'worker',lease_seconds=30,deadline=time.time()+40,
                       guard=g,canonicalize=canonicalize)
        self.assertEqual(c['lease']['emission_id'],fresh)
        self.assertEqual(self.boxes[0].db.execute('SELECT state FROM emissions WHERE id=?',(old,)).fetchone()[0],'recorded')
        self.boxes[0].verify()

    def test_producer_commits_while_expensive_guard_is_paused(self):
        from native_dispatch import claim_native
        old=self.emit(); body=json.loads(self.boxes[0].db.execute('SELECT body FROM emissions WHERE id=?',(old,)).fetchone()[0])
        finished=threading.Event(); errors=[]
        def producer():
            try:
                other=Outbox(self.root/'outbox0.sqlite',model_origin_sha256='a'*64)
                try:
                    batch=dict(body['batch']);batch['batch_id']=digest('concurrent-producer');batch['global_step']+=1
                    other.record_batch(batch,[body['emission']])
                finally:other.close()
            except BaseException as e:errors.append(e)
            finally:finished.set()
        class Guard:
            def check(inner,*args):
                thread=threading.Thread(target=producer);thread.start()
                self.assertTrue(finished.wait(2),'producer blocked by history/canonicalization write lock')
                thread.join(1);self.assertEqual(errors,[])
        claim=claim_native(self.boxes[0],'worker',lease_seconds=30,deadline=time.time()+40,guard=Guard(),canonicalize=canonicalize)
        self.assertEqual(claim['lease']['emission_id'],old)
        self.assertEqual(self.boxes[0].verify()['observations'],2)

    def test_claim_cas_rechecks_live_state_after_snapshot(self):
        from native_dispatch import claim_native
        self.emit(); competing=[]
        class Guard:
            def check(inner,*args):
                other=Outbox(self.root/'outbox0.sqlite',model_origin_sha256='a'*64)
                try:competing.append(other.claim('competitor',lease_seconds=30,deadline=time.time()+40))
                finally:other.close()
        claim=claim_native(self.boxes[0],'worker',lease_seconds=30,deadline=time.time()+40,guard=Guard(),canonicalize=canonicalize)
        self.assertIsNone(claim);self.assertIsNotNone(competing[0]);self.boxes[0].verify()

    def test_evaluator_config_contract_and_exact_build_plan(self):
        from native_dispatch import PACKET_FIELDS,validate_build
        from native_evaluate import NativeEvaluator
        tree=ast.parse(inspect.getsource(NativeEvaluator))
        keys={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and
              isinstance(n.value,ast.Attribute) and isinstance(n.value.value,ast.Name) and
              n.value.value.id=='self' and n.value.attr=='config' and isinstance(n.slice,ast.Constant)}
        self.assertLessEqual(keys,PACKET_FIELDS)
        self.assertIn('source_build_plan_sha256',keys)
        from test_native_build import actual_fixture
        from unittest.mock import patch
        c,build,reader=actual_fixture()
        with patch('native_dispatch.bound',side_effect=reader):
            validate_build(c,build)
            for changed in ('0'*64,None,'short'):
                with self.assertRaises(ValueError):validate_build({**c,'source_build_plan_sha256':changed},build)

    def test_current_lease_cannot_authorize_a_different_program(self):
        self.emit();current=self.claim();g=self.guard();calls=[]
        with self.assertRaises(ValueError):
            g.execute_once('1'*64,'(different)','9'*64,current,lambda:calls.append(1))
        self.assertEqual(calls,[]);self.assertEqual(g.verify()['claims'],0)

    def test_guard_restart_retains_attempt_and_rejects_source_removal(self):
        self.emit();current=self.claim();config=self.config();g=self.guard(config)
        g.execute_once('1'*64,'(ridge)','9'*64,current,lambda:None)
        second=self.guard(config)
        with self.assertRaises(BlockedKey):second.check('1'*64,'(ridge)',current=current)
        changed=json.loads(json.dumps(config));changed['sources']=changed['sources'][:1]
        with self.assertRaises(ValueError):self.guard(changed)

    def test_postclaim_clock_expiry_is_durable_hold_not_execution(self):
        self.emit();current=self.claim();g=self.guard();calls=[]
        old=g._postclaim_check
        def expires(*a):
            result=old(*a);g.now=lambda:g.deadline+1;return result
        with patch.object(g,'_postclaim_check',side_effect=expires):
            with self.assertRaises(ValueError):g.execute_once('1'*64,'(ridge)','9'*64,current,lambda:calls.append(1))
        self.assertEqual(calls,[]);g.verify()
        self.assertEqual(g.db.execute('SELECT state FROM claims').fetchone()[0],'unknown')

    def test_lease_expiry_between_postclaim_and_callback_is_refused(self):
        self.emit();current=self.claim();g=self.guard();calls=[]
        original=g._postclaim_check
        def expires(*a):
            result=original(*a);g.now=lambda:current['lease']['expires_at']+0.1;return result
        with patch.object(g,'_postclaim_check',side_effect=expires):
            with self.assertRaisesRegex(ValueError,'lease expired'):
                g.execute_once('1'*64,'(ridge)','9'*64,current,lambda:calls.append(1))
        self.assertEqual(calls,[]);g.verify()
        self.assertEqual(g.db.execute('SELECT state FROM claims').fetchone()[0],'unknown')

    def test_registry_hardlink_and_foreign_db_do_not_mutate_source(self):
        self.emit();config=self.config();source=self.root/'outbox0.sqlite'
        before={r[0] for r in self.boxes[0].db.execute('SELECT name FROM sqlite_master')}
        alias=self.root/'guard-hardlink.sqlite';os.link(source,alias);config['registry_path']=str(alias)
        with self.assertRaises(ValueError):self.guard(config)
        self.assertEqual(before,{r[0] for r in self.boxes[0].db.execute('SELECT name FROM sqlite_master')})
        alias.unlink()
        foreign=self.root/'foreign.sqlite';connection=sqlite3.connect(foreign)
        connection.execute('CREATE TABLE foreign_data(value)');connection.commit();connection.close()
        pin=sha(foreign.read_bytes());config['registry_path']=str(foreign)
        with self.assertRaises((ValueError,sqlite3.Error)):self.guard(config)
        self.assertEqual(sha(foreign.read_bytes()),pin)
        self.boxes[0].verify()

    def test_source_replaced_after_census_before_first_registry_is_refused(self):
        self.emit(1);self.finish(1,'unknown_attempt');config=self.config()
        self.boxes[1].close();source=self.root/'outbox1.sqlite';source.rename(self.root/'old-before-registry.sqlite')
        replacement=Outbox(source,model_origin_sha256='a'*64);self.addCleanup(replacement.close)
        with self.assertRaisesRegex(ValueError,'pinned census before first registry'):self.guard(config)
        self.assertFalse(Path(config['registry_path']).exists())

    def test_source_replacement_refused_live_and_across_restart(self):
        self.emit(1);self.finish(1,'unknown_attempt');config=self.config();g=self.guard(config)
        with self.assertRaises(BlockedKey):g.check('1'*64,'(ridge)')
        source=self.root/'outbox1.sqlite';self.boxes[1].close()
        source.rename(self.root/'old-source.sqlite')
        replacement=Outbox(source,model_origin_sha256='a'*64);self.addCleanup(replacement.close)
        with self.assertRaisesRegex(ValueError,'physical identity'):
            g.check('1'*64,'(ridge)')
        with self.assertRaisesRegex(ValueError,'physical'):
            self.guard(config)

    def test_closed_packet_requires_new_schema_and_original_global_clock(self):
        from native_dispatch import PACKET_FIELDS,validate_packet
        c={key:None for key in PACKET_FIELDS}
        c.update(schema='jope.native-original-successor/v5',created_unix=time.time()-1,
                 deadline_unix=time.time()+1000,global_deadline_unix=GLOBAL_DEADLINE,threads=1,maximum_emissions=128)
        validate_packet(c)
        for change in ({'extra':True},{'threads':2},{'global_deadline_unix':GLOBAL_DEADLINE+1},
                       {'schema':'jope.native-original-drain/v4'},{'created_unix':time.time()-1801},{'created_unix':time.time()+10},{'deadline_unix':time.time()-1}):
            with self.assertRaises(ValueError):validate_packet({**c,**change})


if __name__=='__main__':unittest.main()
