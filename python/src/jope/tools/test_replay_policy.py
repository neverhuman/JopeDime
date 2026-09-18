import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import replay_policy as replay
from resume_expired_cycle import atomic, sha
from teaching_manifest import SHARD_FIELDS


class ReplayPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir='/dev/shm');self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)

    def corpus(self, name, changed=()):
        path=self.root/name;path.mkdir()
        records=bytearray()
        labels=[]
        for group in range(3):
            for row in range(8):
                record=bytearray(532)
                struct.pack_into('<H',record,0,row+1)
                struct.pack_into('<H',record,512,1)
                struct.pack_into('<fff',record,520,.5 if group in changed else .4,.5,1)
                records.extend(record)
                labels.append(dict(record=group*8+row,context=group,program='program-'+str(row),
                    rmse_cut_normalized=row+(.5 if group in changed else 1.)))
        (path/'records.bin').write_bytes(records)
        (path/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
        index=dict(record_bytes=532,records=24,records_sha256=sha(path/'records.bin'),
            vocabulary_sha256='v',context_index_sha256='c',contexts=[],groups=[])
        for g in range(3):
            partition='meta_validation' if g==2 else 'meta_train'
            index['contexts'].append(dict(prepared_sha256='p'+str(g),cut=0,family='f'+str(g),partition=partition))
            index['groups'].append(dict(context=g,family='f'+str(g),partition=partition,records=list(range(g*8,g*8+8))))
        atomic(path/'index.json',index)
        atomic(path/'provenance.json',dict(index_sha256=sha(path/'index.json'),labels_sha256=sha(path/'labels.jsonl')))
        return path

    def test_fresh_replay_excludes_changed_validation_groups_and_advances_seed(self):
        before=self.corpus('before'); after=self.corpus('after',(1,2))
        checkpoint=self.root/'checkpoint';checkpoint.mkdir()
        atomic(checkpoint/'state.json',dict(vocabulary_sha256='v'))
        policy=replay.freeze(after,before,checkpoint,'fresh_balanced','reset')
        self.assertEqual(policy['fresh_groups'],[1])
        self.assertEqual(policy['fresh_fraction'],.5)
        self.assertEqual(policy['fresh_records'], {'1':list(range(8,16))})
        self.assertEqual(policy,replay.freeze(after,before,checkpoint,'fresh_balanced','reset'))
        old=replay.freeze(before,before,checkpoint,'fresh_balanced','reset')
        self.assertNotEqual(policy['sampling_seed'],old['sampling_seed'])
        self.assertEqual(old['fresh_fraction'],0)
        self.assertIn('no fresh',old['fallback'])
        next_block=replay.freeze(after,before,checkpoint,'fresh_balanced','reset',block_identity='1')
        self.assertNotEqual(policy['sampling_seed'],next_block['sampling_seed'])

    def test_target_or_format_rebuild_is_not_a_discovery(self):
        before=self.corpus('before');after=self.corpus('after',(0,1))
        (after/'labels.jsonl').write_bytes((before/'labels.jsonl').read_bytes())
        atomic(after/'provenance.json',dict(index_sha256=sha(after/'index.json'),labels_sha256=sha(after/'labels.jsonl')))
        checkpoint=self.root/'checkpoint';checkpoint.mkdir()
        atomic(checkpoint/'state.json',dict(vocabulary_sha256='v'))
        policy=replay.freeze(after,before,checkpoint,'fresh_balanced','reset')
        self.assertEqual(policy['fresh_records'],{})
        self.assertEqual(policy['fresh_fraction'],0)

    def test_carry_cannot_silently_reset_missing_optimizer(self):
        corpus=self.corpus('corpus');checkpoint=self.root/'checkpoint';checkpoint.mkdir()
        atomic(checkpoint/'state.json',dict(vocabulary_sha256='v'))
        reset=replay.freeze(corpus,corpus,checkpoint,'broad','reset')
        carry=replay.freeze(corpus,corpus,checkpoint,'broad','carry')
        self.assertTrue(reset['eligible']);self.assertFalse(carry['eligible'])
        self.assertEqual(reset['sampling_seed'],carry['sampling_seed'])
        self.assertEqual(len(carry['blocked_reasons']),3)

    def test_teacher_improved_keeps_only_programs_that_beat_the_prior_teacher(self):
        before=self.corpus('before'); after=self.corpus('after',(1,))
        checkpoint=self.root/'checkpoint';checkpoint.mkdir()
        atomic(checkpoint/'state.json',dict(vocabulary_sha256='v'))
        default=replay.freeze(after,before,checkpoint,'fresh_balanced','reset')
        improved=replay.freeze(after,before,checkpoint,'fresh_balanced','reset',
                               discoveries='teacher_improved')
        self.assertEqual(default['fresh_records'], {'1':list(range(8,16))})
        self.assertEqual(improved['fresh_records'], {'1':[8]})
        self.assertEqual(improved['discoveries'], 'teacher_improved')
        self.assertEqual(improved['sampling_version'], 'lineage_teacher_improved_v1')
        self.assertNotEqual(default['sampling_seed'], improved['sampling_seed'])

    def test_legacy_replay_cannot_manufacture_scientific_family_names(self):
        before=self.corpus('before'); after=self.corpus('after',(1,))
        checkpoint=self.root/'checkpoint';checkpoint.mkdir()
        atomic(checkpoint/'state.json',dict(vocabulary_sha256='v'))
        policy=replay.freeze(after,before,checkpoint,'fresh_balanced','reset')
        original=(after/'index.json').read_bytes()
        dest=self.root/'view'
        with self.assertRaisesRegex(RuntimeError, 'use native --replay-policy'):
            replay.materialize_view(after, policy, dest)
        self.assertFalse(dest.exists())
        self.assertEqual((after/'index.json').read_bytes(), original)
        self.assertEqual(policy['fresh_fraction'], .5)
        self.assertTrue(policy['fresh_records'])

    def test_tampered_corpus_is_detected(self):
        corpus=self.corpus('corpus')
        with (corpus/'records.bin').open('r+b') as stream: stream.write(b'bad')
        with self.assertRaisesRegex(RuntimeError,'hash changed'):replay.groups(corpus)

    def test_sharded_coordinates_match_legacy_freshness_and_detect_tampering(self):
        legacy=self.corpus('legacy',(1,))
        before,expected=replay.groups(legacy)
        path=self.root/'runs/sharded';path.mkdir(parents=True)
        (path/'vocabulary.json').write_text('{}')
        shards=[];offsets=bytearray();labels=[json.loads(s) for s in (legacy/'labels.jsonl').read_text().splitlines()]
        for group in range(3):
            directory=self.root/'runs'/f'group-{group}';directory.mkdir()
            (directory/'records.bin').write_bytes((legacy/'records.bin').read_bytes()[group*8*532:(group+1)*8*532])
            (directory/'labels.jsonl').write_text(''.join(json.dumps(dict(row,record=i))+'\n'
                for i,row in enumerate(labels[group*8:group*8+8])))
            shard=dict(context=group,path=str(directory.relative_to(self.root)),sha256='',records=8,max_length=1,
                records_sha256=sha(directory/'records.bin'),labels_sha256=sha(directory/'labels.jsonl'),legal_sets=[[0]])
            atomic(directory/'shard.json',shard);shard['sha256']=sha(directory/'shard.json');shards.append(shard)
            offsets.extend(struct.pack('<QQ',group*8,8))
        (path/'groups.bin').write_bytes(offsets)
        identity=hashlib.sha256(json.dumps([{k:s[k] for k in SHARD_FIELDS} for s in shards],
            separators=(',',':')).encode()).hexdigest()
        index=dict(schema='parallel.teaching_shards.v1',record_bytes=532,records=24,shards=shards,
            records_sha256=identity,groups_sha256=sha(path/'groups.bin'),contexts=before['contexts'],
            vocabulary_sha256=sha(path/'vocabulary.json'),context_index_sha256='c')
        atomic(path/'index.json',index);atomic(path/'provenance.json',dict(index_sha256=sha(path/'index.json')))
        _,actual=replay.groups(path)
        self.assertEqual(actual,expected)
        with (self.root/shards[0]['path']/'labels.jsonl').open('a') as stream:stream.write('{}\n')
        with self.assertRaisesRegex(RuntimeError,'payload changed'):replay.groups(path)
