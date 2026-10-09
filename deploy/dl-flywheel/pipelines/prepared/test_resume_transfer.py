"""CPU-only physical continuation and refusal probes; no live data, S3 or GPU."""
import copy, json, tempfile, time, types, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from dltrain import model as M
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from neural_native_train import fit_preprocessing, prepare_microbatches, run_microbatches
from neural_storage import checkpoint
from neural_outbox import Outbox
from neural_native_stream import BatchStream
from fixture_continuous import ContinuousTests
import baseline_runner
import continuous_runner
from reconcile import capture
from prepared_admission import adopt


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.snapshot, n = ContinuousTests().fixture()
        # A two-update fixture segment inside a four-update logical test budget.
        n.update(max_updates=getattr(self,'initial_updates',2), evaluation_every_batches=1, max_pending_emissions=100,
                 output_cap_bytes=24*1024**3, plateau_patience=5, learning_rate_floor=1e-6)
        torch.manual_seed(4)
        net = M.PolicyLM(M.PolicyConfig(282,2051,32,1,4,64,128,dropout=.2))
        opt = torch.optim.AdamW(M.param_groups(net,.01), lr=.0003)
        norm, pre = fit_preprocessing(self.snapshot)
        ids = [c for c,r in self.snapshot.contexts.items() if r['partition']=='train']
        packs, total, _ = prepare_microbatches(self.snapshot, ids[:2], epoch=0,
            batch_index=0, normalizer=norm, max_padded_tokens=4096)
        run_microbatches(net,packs,total,optimizer=opt,deadline=time.time()+60)
        m = checkpoint(net,opt,self.base/'parent',step=429,config_sha256='d'*64)
        old = Outbox(self.base/'old.sqlite',model_origin_sha256=m['parameter_sha256']);old.close()
        ref = lambda name: {'path':str(self.base/'parent'/name),
                           'sha256':sha((self.base/'parent'/name).read_bytes())}
        self.config = {'neural':n,'checkpoint_manifest':ref('manifest.json'),
            'checkpoint_state':ref('state.safetensors'),'parameter_sha256':m['parameter_sha256'],
            'parent_outbox':str(self.base/'old.sqlite'),'training_config':{'sha256':'d'*64},
            'global_deadline_unix':time.time()+120}
        self.ancestry = {'planner_ref':'fixture','head':{'events':861,'committed_batches':429,
            'unresolved_batches':0,'head_sha256':'d'*64},'checkpoint_sha256':m['parameter_sha256'],
            'optimizer_steps':429,'completed_epochs':1,
            'fingerprints':[digest({'oldbatch':i})for i in range(429)],
            'exposed_context_ids':ids,'presentations':len(ids)}
        self.pre = canonical(pre); self.root = self.base/'run'
        baseline_runner.run(self.snapshot,self.snapshot,self.config,root=self.root,
            campaign_sha='e'*64,ancestry=self.ancestry,preprocessing=self.pre,
            deadline=time.time()+60,authority_check=lambda s:None,adoption_check=lambda s,r:None)
        self.snapshot.extension_raw = canonical({'tasks':[]})
        self.config['neural']['max_updates'] = getattr(self,'final_updates',4)

    def capture(self, frozen=True, require_closed_epoch=True):
        return capture(self.root,campaign_sha256='e'*64,frozen=frozen,max_updates=self.config['neural']['max_updates'],
                       plateau_patience=5,learning_rate_floor=1e-6,require_closed_epoch=require_closed_epoch)

    def test_exact_physical_transfer_keeps_journals_and_advances_only_new_batches(self):
        receipt = self.capture(); receipt['execution_config_sha256']='f'*64
        self.assertEqual(receipt['absolute_step'],431)
        initial_events = receipt['planner_head']['events']
        original = (self.root/'batch-000000430.json').read_bytes()
        # Same member set, new snapshot identity: isolates boundary adoption and
        # exact checkpoint/control/journal reuse independently of qualified P.load.
        replacement = copy.copy(self.snapshot); replacement.identity='f'*64
        replacement.batch_manifest=copy.deepcopy(self.snapshot.batch_manifest)
        replacement.batch_manifest['snapshot_sha256']=replacement.identity
        attempted=[]
        def adoption(s,r):
            if attempted:return None
            attempted.append(True);return replacement
        result=continuous_runner.run(self.snapshot,self.snapshot,self.config,root=self.root,
            campaign_sha='e'*64,ancestry=self.ancestry,preprocessing=self.pre,
            deadline=time.time()+60,authority_check=lambda s:None,adoption_check=adoption,resume=receipt)
        self.assertEqual(result['absolute_optimizer_steps'],433)
        self.assertEqual((self.root/'batch-000000430.json').read_bytes(),original)
        progress=json.loads((self.root/'progress.json').read_bytes())
        self.assertEqual(progress['unique_train_data_runs_seen'],4)
        self.assertEqual(progress['total_train_presentations'],12)
        self.assertEqual(progress['model_generated_programs'],8)
        self.assertEqual(progress['execution_transfer']['absolute_resume_step'],431)
        self.assertGreater(json.loads((self.root/'result.json').read_bytes())['planner_verification']['events'],initial_events)
        self.assertEqual(json.loads((self.root/'optimizer-control.json').read_bytes())['absolute_optimizer_step'],433)
        # Compare the resumed dropout/AdamW sequence against four uninterrupted
        # updates from the same physical parent, normalization and shuffled seed.
        uninterrupted=self.base/'uninterrupted'
        baseline_runner.run(self.snapshot,self.snapshot,self.config,root=uninterrupted,
            campaign_sha='e'*64,ancestry=self.ancestry,preprocessing=self.pre,
            deadline=time.time()+60,authority_check=lambda s:None,adoption_check=lambda s,r:None)
        from safetensors.torch import load
        left=load((self.root/'rolling/000000433/state.safetensors').read_bytes())
        right=load((uninterrupted/'rolling/000000433/state.safetensors').read_bytes())
        self.assertEqual(set(left),set(right))
        for key in left:self.assertTrue(torch.equal(left[key],right[key]),key)

    def test_unknown_reservation_refuses_without_mutation(self):
        receipt=self.capture();raw=(self.root/'initial-batch-manifest.json').read_bytes()
        p=BatchStream(self.root/'batches.sqlite',raw,sha(raw),seed=4,batch_size=2,
                      initial_checkpoint_sha256=self.ancestry['checkpoint_sha256'])
        p.reserve(receipt['control']['checkpoint_parameter_sha256']);before=p.verify()
        with self.assertRaisesRegex(ValueError,'unknown optimizer'):self.capture()
        self.assertEqual(p.verify(),before);p.close()

    def test_unfrozen_owner_refused(self):
        with self.assertRaisesRegex(ValueError,'frozen'):self.capture(False)

    def test_mismatched_control_and_checkpoint_refused(self):
        p=self.root/'optimizer-control.json';control=json.loads(p.read_bytes())
        control['absolute_optimizer_step']-=1;p.write_bytes(canonical(control))
        with self.assertRaisesRegex(ValueError,'optimizer-control'):self.capture()

    def test_corrupted_planner_history_refuses_adoption_before_any_mutation(self):
        raw=(self.root/'initial-batch-manifest.json').read_bytes()
        p=BatchStream(self.root/'batches.sqlite',raw,sha(raw),seed=4,batch_size=2,
                      initial_checkpoint_sha256=self.ancestry['checkpoint_sha256'])
        p.db.execute("UPDATE events SET sha=?WHERE seq=1",('0'*64,))
        before=p.db.execute('SELECT count(*)FROM events').fetchone()[0]
        with self.assertRaisesRegex(ValueError,'tampered'):
            adopt(p,self.snapshot,self.snapshot,checkpoint_sha256=p._get('checkpoint'),authority_check=lambda e:None)
        self.assertEqual(p.db.execute('SELECT count(*)FROM events').fetchone()[0],before);p.close()


class OpenEpochTests(TransferTests):
    initial_updates=3
    final_updates=5

    def test_open_epoch_transfer_defers_admission_and_keeps_bit_exact_sequence(self):
        with self.assertRaisesRegex(ValueError,'closed current epoch'):self.capture()
        receipt=self.capture(require_closed_epoch=False);receipt['execution_config_sha256']='f'*64
        self.assertEqual(receipt['absolute_step'],432)
        original=(self.root/'batch-000000432.json').read_bytes()
        replacement=copy.copy(self.snapshot);replacement.identity='f'*64
        replacement.batch_manifest=copy.deepcopy(self.snapshot.batch_manifest)
        replacement.batch_manifest['snapshot_sha256']=replacement.identity
        attempts=[]
        def adoption(s,r):
            if attempts:return None
            attempts.append(json.loads((r/'optimizer-control.json').read_bytes())['absolute_optimizer_step'])
            return replacement
        continuous_runner.run(self.snapshot,self.snapshot,self.config,root=self.root,
            campaign_sha='e'*64,ancestry=self.ancestry,preprocessing=self.pre,
            deadline=time.time()+60,authority_check=lambda s:None,adoption_check=adoption,resume=receipt)
        self.assertEqual(attempts,[433]);self.assertFalse((self.root/'adopted-prepared-transfer-000000432.json').exists())
        self.assertTrue((self.root/'adopted-prepared-transfer-000000433.json').exists())
        self.assertEqual((self.root/'batch-000000432.json').read_bytes(),original)
        for step in (433,434):
            proof=json.loads((self.root/f'batch-{step:09d}.json').read_bytes())
            self.assertEqual(proof['batch']['manifest_sha256'],sha(canonical(self.snapshot.batch_manifest if step==433 else replacement.batch_manifest)))
        uninterrupted=self.base/'uninterrupted'
        baseline_runner.run(self.snapshot,self.snapshot,self.config,root=uninterrupted,
            campaign_sha='e'*64,ancestry=self.ancestry,preprocessing=self.pre,
            deadline=time.time()+60,authority_check=lambda s:None,adoption_check=lambda s,r:None)
        from safetensors.torch import load
        left=load((self.root/'rolling/000000434/state.safetensors').read_bytes())
        right=load((uninterrupted/'rolling/000000434/state.safetensors').read_bytes())
        self.assertEqual(set(left),set(right))
        for key in left:self.assertTrue(torch.equal(left[key],right[key]),key)


    def test_open_capture_stale_progress_and_unknown_intent_refused(self):
        path=self.root/'progress.json';raw=path.read_bytes();value=json.loads(raw);value['total_train_presentations']+=1;path.write_bytes(canonical(value))
        with self.assertRaisesRegex(ValueError,'published progress'):self.capture(require_closed_epoch=False)
        path.write_bytes(raw)
        receipt=self.capture(require_closed_epoch=False)
        self.assertEqual(receipt['current_epoch']['state'],'open')
        self.assertEqual(len(receipt['current_epoch']['remaining_context_ids']),2)
        manifest=(self.root/'initial-batch-manifest.json').read_bytes()
        p=BatchStream(self.root/'batches.sqlite',manifest,sha(manifest),seed=4,batch_size=2,initial_checkpoint_sha256=self.ancestry['checkpoint_sha256'])
        p.reserve(receipt['control']['checkpoint_parameter_sha256']);before=p.verify()
        with self.assertRaisesRegex(ValueError,'unknown optimizer'):self.capture(require_closed_epoch=False)
        self.assertEqual(p.verify(),before);p.close()

    def test_changed_authority_at_boundary_refuses_admission_without_replaying_commit(self):
        receipt=self.capture(require_closed_epoch=False);receipt['execution_config_sha256']='f'*64
        replacement=copy.copy(self.snapshot);replacement.identity='f'*64
        replacement.batch_manifest=copy.deepcopy(self.snapshot.batch_manifest);replacement.batch_manifest['snapshot_sha256']=replacement.identity
        def authority(s):
            if s is replacement:raise ValueError('changed lineage authority')
        with self.assertRaisesRegex(ValueError,'changed lineage authority'):
            continuous_runner.run(self.snapshot,self.snapshot,self.config,root=self.root,
                campaign_sha='e'*64,ancestry=self.ancestry,preprocessing=self.pre,
                deadline=time.time()+60,authority_check=authority,adoption_check=lambda s,r:replacement,resume=receipt)
        result=json.loads((self.root/'result.json').read_bytes())
        self.assertEqual(result['absolute_optimizer_steps'],433)
        self.assertEqual(result['planner_verification']['unresolved_batches'],0)
        self.assertFalse(list(self.root.glob('adopted-prepared-transfer-*.json')))
        self.assertFalse((self.root/'batch-000000434.json').exists())
        self.assertEqual(result['snapshot_sha256'],self.snapshot.identity)


if __name__=='__main__':unittest.main()
