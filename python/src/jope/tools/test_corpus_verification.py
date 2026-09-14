import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

import continuous_learning as controller
import corpus_verification as verification
from promotion_gate import immutable
from resume_expired_cycle import atomic,read,sha
from teaching_manifest import SHARD_FIELDS
import test_continuous_learning as fixtures


class CorpusVerificationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        corpus=self.root/'runs/snapshot/corpus';corpus.mkdir(parents=True)
        shard=self.root/'runs/shard';shard.mkdir()
        (shard/'records.bin').write_bytes(bytes(8*532));(shard/'labels.jsonl').write_bytes(b'{}\n'*8)
        (corpus/'vocabulary.json').write_text('{}')
        row=dict(context=0,path='runs/shard',sha256='',records=8,max_length=1,
            records_sha256=sha(shard/'records.bin'),labels_sha256=sha(shard/'labels.jsonl'),legal_sets=[[0]])
        atomic(shard/'shard.json',row);row['sha256']=sha(shard/'shard.json')
        (corpus/'groups.bin').write_bytes(struct.pack('<QQ',0,8))
        catalogue=[{key:row[key] for key in SHARD_FIELDS}]
        index=dict(schema='parallel.teaching_shards.v1',record_bytes=532,records=8,qualification=True,
            context_index_sha256='contexts',vocabulary_sha256=sha(corpus/'vocabulary.json'),
            groups_sha256=sha(corpus/'groups.bin'),records_sha256=hashlib.sha256(json.dumps(catalogue,separators=(',',':')).encode()).hexdigest(),
            contexts=[dict(partition='meta_train',family='fixture')],shards=[row])
        atomic(corpus/'index.json',index);atomic(corpus/'provenance.json',dict(index_sha256=sha(corpus/'index.json')))
        self.supervisor=fixtures.ContinuousLearningTests().scheduler(self.root)
        self.supervisor.children={}
        self.supervisor.stopping=False
        self.supervisor.m.update(async_corpus_verification=True,contexts_sha256='contexts',
            vocabulary_sha256=index['vocabulary_sha256'],automatic_training=False)
        self.supervisor.s.update(snapshots=[],covered_searches=['old'],last_snapshot_at=1,
                                teaching_sources=['old-source'])
        self.pack=dict(kind='pack',receipt='runs/snapshot/corpus/provenance.json',output='runs/snapshot/corpus',
            command=['packer'],attempts=1,created_at=1,retry_at=0,covered_searches=['old','new'],sources=['new-source'])
        self.supervisor.s['tasks']['pack']=self.pack
        self.supervisor.finish_task('pack')
        self.task=self.supervisor.s['tasks']['pack_verify']

    def run_verifier(self):
        verification.run(self.root,self.root/self.task['request'],self.root/self.task['receipt'])

    def test_completed_pack_remains_pending_and_does_not_block_gpu_decisions(self):
        self.assertEqual(self.supervisor.s['snapshots'],[])
        self.assertEqual(self.supervisor.s['covered_searches'],['old'])
        self.assertEqual(self.supervisor.s['last_snapshot_at'],1)
        self.supervisor.s['generations']=[dict(number=0,path='pending',stage=2,corpus='corpus')]
        self.supervisor.gpu_work()
        self.assertIn('proposal',self.supervisor.s['tasks'])
        with patch.object(controller,'verify_corpus',side_effect=AssertionError('synchronous verifier')):
            self.supervisor.finish_task('pack')
        self.assertEqual(self.supervisor.s['snapshots'],[])

    def test_published_receipt_survives_restart_before_commit_exactly_once(self):
        self.run_verifier()
        restored=read(self.supervisor.state_path)
        self.assertEqual(restored['snapshots'],[])
        self.supervisor.s=restored
        self.supervisor.start_tasks()
        self.assertEqual(len(self.supervisor.s['snapshots']),1)
        self.assertEqual(self.supervisor.s['tasks'],{})
        self.assertEqual(self.supervisor.s['covered_searches'],['old','new'])
        self.supervisor.s=read(self.supervisor.state_path)
        self.supervisor.start_tasks()
        self.assertEqual(len(self.supervisor.s['snapshots']),1)

    def test_stale_receipt_for_another_pack_attempt_is_rejected(self):
        self.run_verifier();self.pack['attempts']+=1
        with self.assertRaisesRegex(RuntimeError,'another pack'):
            self.supervisor.finish_task('pack_verify')
        self.assertEqual(self.supervisor.s['snapshots'],[])

    def test_authorized_controller_migration_preserves_verification_identity(self):
        from operational_release import migrate
        self.run_verifier()
        self.supervisor.s['manifest_sha256']='test'
        self.supervisor.s=migrate(self.supervisor.s,'test','new-controller')
        self.supervisor.manifest_sha='new-controller'
        self.supervisor.start_tasks()
        self.assertEqual(len(self.supervisor.s['snapshots']),1)
        self.assertEqual(self.supervisor.s['tasks'],{})

    def test_migration_after_intent_before_publication_keeps_original_full_request(self):
        from operational_release import migrate
        request_path=self.root/self.task['request']
        original=read(request_path);request_path.unlink()
        del self.supervisor.s['tasks']['pack_verify']
        self.supervisor.s['manifest_sha256']='test'
        self.supervisor.s=migrate(self.supervisor.s,'test','new-controller')
        self.supervisor.manifest_sha='new-controller'
        with patch.object(verification,'prepare',side_effect=AssertionError('must retain old request')):
            self.supervisor.finish_task('pack')
        self.assertEqual(read(request_path),original)
        self.assertEqual(self.supervisor.s['tasks']['pack_verify']['command'][2],original['verifier_path'])

    def test_migration_rejects_changed_contexts_before_consuming_receipt(self):
        self.run_verifier();self.supervisor.m['contexts_sha256']='incompatible'
        with self.assertRaisesRegex(RuntimeError,'incompatible'):
            self.supervisor.start_tasks()
        self.assertEqual(self.supervisor.s['snapshots'],[])

    def test_request_and_success_receipt_tampering_cannot_replace_frozen_request(self):
        self.run_verifier()
        request=self.root/self.task['request'];value=read(request)
        value['contexts_sha256']='changed';atomic(request,value)
        receipt=self.root/self.task['receipt'];value=read(receipt)
        value['request_sha256']=sha(request);atomic(receipt,value)
        with self.assertRaisesRegex(RuntimeError,'frozen verification request changed'):
            self.supervisor.finish_task('pack_verify')
        self.assertEqual(self.supervisor.s['snapshots'],[])

    def test_corruption_creates_failure_and_is_quarantined_after_restart(self):
        (self.root/'runs/shard/records.bin').write_bytes(b'bad')
        with self.assertRaisesRegex(RuntimeError,'payload changed'):self.run_verifier()
        self.assertFalse((self.root/self.task['receipt']).exists())
        self.supervisor.s=read(self.supervisor.state_path)
        with patch.object(controller,'validate_manifest',side_effect=AssertionError('must not relaunch corrupt pack')):
            self.supervisor.start_tasks()
        self.assertTrue(self.supervisor.s['tasks']['pack_verify']['quarantined'])
        self.assertEqual(self.supervisor.s['snapshots'],[])

    def test_nonzero_verifier_exit_with_stray_success_cannot_commit(self):
        self.run_verifier()
        class Failed:
            returncode=1
            def poll(self):return 1
        self.supervisor.children['pack_verify']=(Failed(),None,None,time.time())
        self.supervisor.poll_tasks()
        self.assertTrue(self.task['quarantined']);self.assertEqual(self.supervisor.s['snapshots'],[])

    def test_sighup_exit_is_not_mistaken_for_an_adopted_process(self):
        self.run_verifier()
        class Signaled:
            returncode=-1
            def poll(self):return -1
        self.supervisor.children['pack_verify']=(Signaled(),None,None,time.time())
        self.supervisor.poll_tasks()
        self.assertTrue(self.task['quarantined']);self.assertEqual(self.supervisor.s['snapshots'],[])

    def test_missing_immutable_shard_is_quarantined_instead_of_retried(self):
        (self.root/'runs/shard/records.bin').unlink()
        with self.assertRaises(FileNotFoundError):self.run_verifier()
        self.supervisor.s=read(self.supervisor.state_path)
        self.supervisor.start_tasks()
        self.assertTrue(self.supervisor.s['tasks']['pack_verify']['quarantined'])

    def test_surviving_verifier_is_adopted_without_relaunch(self):
        identity=dict(pid=123,start_ticks='1',boot='fixture')
        self.task.update(process=identity,started_at=time.time())
        with patch.object(controller,'process_identity',return_value=identity), \
             patch.object(controller.os,'getpgid',return_value=123), \
             patch.object(controller.runtime,'popen',side_effect=AssertionError('must adopt')):
            self.supervisor.start_tasks()
        self.assertIsInstance(self.supervisor.children['pack_verify'][0],controller.SurvivingProcess)

    def test_verification_receipt_does_not_replace_payload_checks_at_use(self):
        self.run_verifier();(self.root/'runs/shard/records.bin').write_bytes(b'bad')
        self.supervisor.finish_task('pack_verify')
        with self.assertRaisesRegex(RuntimeError,'payload changed'):
            controller.verify_corpus(self.root,self.pack['output'],'contexts',self.supervisor.m['vocabulary_sha256'])
