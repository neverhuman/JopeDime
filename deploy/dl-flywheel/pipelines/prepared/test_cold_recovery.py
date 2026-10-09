"""Interrupted CPU input recovers one existing reservation, never an ACKed update."""
import copy, json, time, unittest
from unittest.mock import patch
from pathlib import Path
import torch
from jope_flywheel.protocol import canonical
from jope_flywheel.program_io import sha
from neural_native_stream import BatchStream
from test_resume_transfer import TransferTests
from reconcile import capture
import baseline_runner, continuous_runner, cold_reservation as C


class ColdTests(TransferTests):
    def setup_cold(self):
        moved=self.base/'training';self.root.rename(moved);self.root=moved
        raw=(self.root/'initial-batch-manifest.json').read_bytes()
        planner=BatchStream(self.root/'batches.sqlite',raw,sha(raw),seed=4,batch_size=2,
            initial_checkpoint_sha256=self.ancestry['checkpoint_sha256'])
        batch=planner.reserve(json.loads((self.root/'optimizer-control.json').read_bytes())['checkpoint_parameter_sha256'])
        planner.close()
        self.failed=self.base/'failed';self.failed.mkdir()
        runner=self.failed/'continuous_runner.py';runner.write_bytes(b'# qualified fixture ordering\n')
        campaign=self.base
        config={'owned_sources':{}}
        config_path=campaign/'config.json';config_path.write_bytes(canonical(config))
        ref=lambda p:{'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
        execution={'owned_sources':{'continuous_runner.py':sha(runner.read_bytes())},'campaign_root':str(campaign),
            'campaign_config':ref(config_path),'frozen_owner':{'pid':123,'control_group':'fixture/jopedime-neural-continuous-20261009-v1.service'}}
        path=self.failed/'execution-config.json';path.write_bytes(canonical(execution))
        trace=[{'sequence':1,'config_sha256':sha(path.read_bytes()),'host':'xbabe2','pid':456,'monotonic_ns':1,
            'kind':'start','stage':'input.training_batch','absolute_optimizer_step':432,'batch_id':batch.batch_id}]
        trace_path=self.failed/'stage-timing.jsonl';trace_path.write_bytes(canonical(trace[0])+b'\n')
        owners=[{'unit':'jopedime-neural-continuous-20261009-v1','original_pid':123,'control_group':'fixture/jopedime-neural-continuous-20261009-v1.service'},
                {'unit':'jopedime-neural-continuous-resume3-20261009','original_pid':456,'control_group':'fixture/jopedime-neural-continuous-resume3-20261009.service'}]
        permit={'schema':'jope.exact-unstarted-reservation-permit/v1','execution_config':ref(path),
            'trace':ref(trace_path),'stopped_owners':owners}
        self.constants={'EXECUTION':sha(path.read_bytes()),'TRACE':sha(trace_path.read_bytes()),'RUNNER':sha(runner.read_bytes()),
            'STEP':432,'EVENTS':1,'BATCH':batch.batch_id}
        return batch,permit,campaign/'training'

    def prove(self,permit,root):
        with patch.multiple(C,**self.constants):
            return C.prove(root,permit,owner_check=lambda o:{'MainPID':'0','ActiveState':'inactive'})

    def test_same_reserved_batch_finishes_once_and_is_bit_exact(self):
        batch,permit,alias=self.setup_cold();certificate=self.prove(permit,alias)
        receipt=capture(self.root,campaign_sha256='e'*64,frozen=False,max_updates=4,
            plateau_patience=5,learning_rate_floor=1e-6,require_closed_epoch=False,certified_unstarted=certificate)
        receipt['execution_config_sha256']='f'*64
        original_intent=canonical({k:v for k,v in receipt['certified_batch'].items()if k!='rows'})
        replacement=copy.copy(self.snapshot);replacement.identity='f'*64;replacement.batch_manifest=copy.deepcopy(self.snapshot.batch_manifest)
        replacement.batch_manifest['snapshot_sha256']='f'*64;attempted=[]
        def adopt(s,r):
            if attempted:return None
            attempted.append(True);return replacement
        continuous_runner.run(self.snapshot,self.snapshot,self.config,root=self.root,campaign_sha='e'*64,
            ancestry=self.ancestry,preprocessing=self.pre,deadline=time.time()+60,authority_check=lambda s:None,
            adoption_check=adopt,resume=receipt)
        proof=json.loads((self.root/'batch-000000432.json').read_bytes())
        self.assertEqual(proof['batch']['batch_id'],batch.batch_id)
        self.assertEqual(canonical({k:v for k,v in proof['batch'].items()if k!='rows'}),original_intent)
        db=BatchStream(self.root/'batches.sqlite',canonical(replacement.batch_manifest),sha(canonical(replacement.batch_manifest)),
            seed=4,batch_size=2,initial_checkpoint_sha256=self.ancestry['checkpoint_sha256'])
        self.assertIsNone(db.pending());self.assertEqual(db.verify()['committed_batches'],4)
        self.assertEqual(db.db.execute("SELECT count(*)FROM events WHERE kind='batch.reserved'").fetchone()[0],4);db.close()
        uninterrupted=self.base/'uninterrupted'
        baseline_runner.run(self.snapshot,self.snapshot,self.config,root=uninterrupted,campaign_sha='e'*64,
            ancestry=self.ancestry,preprocessing=self.pre,deadline=time.time()+60,authority_check=lambda s:None,adoption_check=lambda s,r:None)
        from safetensors.torch import load
        left=load((self.root/'rolling/000000433/state.safetensors').read_bytes());right=load((uninterrupted/'rolling/000000433/state.safetensors').read_bytes())
        for k in left:self.assertTrue(torch.equal(left[k],right[k]),k)

    def test_ack_and_changed_trace_refuse(self):
        batch,permit,alias=self.setup_cold()
        trace=Path(permit['trace']['path']);trace.write_bytes(trace.read_bytes()+b'bad tail\n')
        with self.assertRaises(ValueError):self.prove(permit,alias)
        trace.write_bytes(trace.read_bytes().split(b'bad tail')[0])
        import sqlite3
        db=sqlite3.connect(self.root/'outbox.sqlite');db.execute('INSERT INTO batches VALUES(?,?,?)',(batch.batch_id,b'{}',b'{}'));db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'durable ACK'):self.prove(permit,alias)

    def test_later_optimizer_stage_refuses(self):
        batch,permit,alias=self.setup_cold();path=Path(permit['trace']['path']);row=json.loads(path.read_bytes())
        row['stage']='model.forward_backward_optimizer';path.write_bytes(canonical(row)+b'\n')
        permit['trace']['sha256']=sha(path.read_bytes());permit['trace']['bytes']=path.stat().st_size;self.constants['TRACE']=sha(path.read_bytes())
        with self.assertRaisesRegex(ValueError,'uncompleted CPU input'):self.prove(permit,alias)

    def test_changed_membership_or_unrelated_pending_refuses(self):
        batch,permit,alias=self.setup_cold();certificate=self.prove(permit,alias);certificate['batch_id']='f'*64
        with self.assertRaisesRegex(ValueError,'untouched reservation'):
            capture(self.root,campaign_sha256='e'*64,frozen=False,max_updates=4,
                plateau_patience=5,learning_rate_floor=1e-6,require_closed_epoch=False,certified_unstarted=certificate)

    def test_sealed_capture_roundtrip_matches_fresh_cold_custody(self):
        batch,permit,alias=self.setup_cold()
        certified=self.prove(permit,alias)
        first=capture(self.root,campaign_sha256='e'*64,frozen=False,max_updates=4,
            plateau_patience=5,learning_rate_floor=1e-6,require_closed_epoch=False,
            certified_unstarted=certified)
        path=self.base/'sealed-capture.json';path.write_bytes(canonical(first))
        sealed=json.loads(path.read_bytes())
        fresh=capture(self.root,campaign_sha256='e'*64,frozen=False,max_updates=4,
            plateau_patience=5,learning_rate_floor=1e-6,require_closed_epoch=False,
            certified_unstarted=certified)
        self.assertNotEqual(fresh,sealed)
        normalized=json.loads(canonical(fresh))
        self.assertEqual(normalized,sealed)
        for field in ('certified_batch','control','certified_unstarted'):
            changed=copy.deepcopy(sealed)
            changed[field]['tampered']='identity change'
            self.assertNotEqual(normalized,changed)
        changed=copy.deepcopy(sealed)
        changed['certified_batch']['ordered_data_run_ids'].reverse()
        self.assertNotEqual(normalized,changed)
        changed=copy.deepcopy(sealed);changed['checkpoint_state']['sha256']='0'*64
        self.assertNotEqual(normalized,changed)

    def test_rehashed_changed_membership_cannot_reuse_certified_batch_id(self):
        batch,permit,alias=self.setup_cold();certificate=self.prove(permit,alias)
        import sqlite3
        from jope_flywheel.protocol import digest
        db=sqlite3.connect(self.root/'batches.sqlite');db.row_factory=sqlite3.Row
        row=db.execute('SELECT body FROM batches WHERE id=?',(batch.batch_id,)).fetchone();intent=json.loads(row['body'])
        intent['ordered_data_run_ids'].reverse();db.execute('UPDATE batches SET body=?WHERE id=?',(canonical(intent).decode(),batch.batch_id))
        tail=db.execute('SELECT *FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        event={'seq':tail['seq'],'kind':tail['kind'],'body':intent,'previous':tail['previous']}
        db.execute('UPDATE events SET body=?,sha=?WHERE seq=?',(canonical(intent).decode(),digest(event),tail['seq']));db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'ordered membership digest'):
            capture(self.root,campaign_sha256='e'*64,frozen=False,max_updates=4,plateau_patience=5,
                learning_rate_floor=1e-6,require_closed_epoch=False,certified_unstarted=certificate)


if __name__=='__main__':unittest.main()
