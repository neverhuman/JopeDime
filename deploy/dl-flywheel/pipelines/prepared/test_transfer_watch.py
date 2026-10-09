"""Control-plane refusal probes: no real process freeze, launch, kill or CUDA."""
import json, os, tempfile, time, types, unittest
from pathlib import Path
from unittest.mock import patch
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from neural_native_stream import BatchStream
import transfer_watch as W


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.campaign=self.root/'campaign';self.training=self.campaign/'training';self.training.mkdir(parents=True)
        config={'neural':{'max_updates':50000,'plateau_patience':5,'learning_rate_floor':1e-6},'global_deadline_unix':1791592146.380285}
        self.config=canonical(config);(self.campaign/'config.json').write_bytes(self.config)
        self.raw=canonical({'schema':'jope.native-corpus-batches/v1','snapshot_sha256':'a'*64,'role_authority_sha256':'b'*64,
            'contexts':[{'context_id':digest({'id':i}),'context_sha256':digest({'id':i}),'view_id':str(i),'dataset_hash':'a'*16,'fold':i,
             'cut_id':'cut','parent_lineage_sha256':digest({'parent':i}),'partition':'train','source_kind':'native_s1_context',
             'observation_dispatch':'pending_native_worker_adapter','gp_hwm_ref':{},'kumo_source_sha256':'d'*64}for i in range(2)],'pending':[],'refused':[]})
        self.planner=BatchStream(self.training/'batches.sqlite',self.raw,sha(self.raw),seed=7,batch_size=2,initial_checkpoint_sha256='0'*64)
        self.addCleanup(self.planner.close)
        batch=self.planner.reserve('0'*64);self.planner.commit(batch.batch_id,checkpoint_before='0'*64,checkpoint_after='1'*64,
            optimizer_step=1,ordered_data_run_ids=batch.ordered_data_run_ids,evidence={'origin':'native_optimizer','event_sha256':'2'*64})
        self.head=self.planner.verify();self.manifest=self.planner._get('manifest')
        self.progress={'state':'running','completed_epochs':2};(self.training/'progress.json').write_bytes(canonical(self.progress));(self.training/'optimizer-control.json').write_bytes(b'{}')
        sources={}
        for name in ('continuous_cli.py','continuous_runner.py','prepared_admission.py','prepared_snapshot.py','reconcile.py','transfer_watch.py'):
            (self.root/name).write_bytes(b'# mocked execution only\n');sources[name]=sha((self.root/name).read_bytes())
        receipt={'state':'passed','model_and_adamw_and_cpu_cuda_rng_bit_exact':True,'live_optimizer_updates':0,'planner_mutations':0,'outbox_mutations':0}
        r=self.root/'qualification.json';r.write_bytes(canonical(receipt))
        now=time.time();plan={'schema':'jope.boundary-transfer-plan/v1','campaign_root':str(self.campaign),'created_unix':now,
            'deadline_unix':now+600,'global_deadline_unix':1791592146.380285,'sources':sources,'prepared_candidate':{},
            'cuda_qualification':{'path':str(r),'sha256':sha(r.read_bytes()),'bytes':r.stat().st_size},'start_completed_epochs':1}
        body=canonical(plan);(self.root/'transfer-plan.json').write_bytes(body);self.pin=sha(body);self.calls=[];self.started=False
        self.capture={'progress':self.progress,'absolute_step':430,'planner_head':self.head,'manifest_sha256':self.manifest}

    def owner(self,name):
        return {'MainPID':'123','ControlGroup':'test-owned-cgroup','ActiveState':'active'if name==W.OLD_UNIT or self.started else'inactive'}

    def run_mock(self,after_launch):
        real_read=Path.read_text
        def text(p,*args,**kwargs):
            if str(p).endswith('/test-owned-cgroup/cgroup.events'):return 'frozen 1\n'
            return real_read(p,*args,**kwargs)
        def call(*args):
            self.calls.append(args)
            if args[0]=='nvidia-smi':return '23000'
            if args[0]=='systemd-run':
                self.started=True;after_launch();raise RuntimeError('injected unknown launch acknowledgement')
            return ''
        with patch.object(W,'__file__',str(self.root/'transfer_watch.py')),patch.object(W,'CAMPAIGN',sha(self.config)),\
             patch.object(W.sys,'argv',['watch','--plan-sha',self.pin]),patch.object(W,'unit',self.owner),\
             patch.object(W,'call',call),patch.object(W,'capture',return_value=self.capture),\
             patch.object(Path,'read_text',text),patch.object(W.os,'getloadavg',return_value=(1.,1.,1.)):
            with self.assertRaisesRegex(RuntimeError,'unknown launch'):W.main()

    def test_unknown_launch_without_journal_change_thaws_only_own_ancestor(self):
        self.run_mock(lambda:None)
        self.assertIn(('systemctl','--user','thaw',W.OLD_UNIT),self.calls)
        self.assertEqual(self.planner.verify(),self.head)

    def test_adoption_before_launch_error_forbids_ancestor_thaw(self):
        def mutate():
            value=json.loads(self.raw);value['snapshot_sha256']='f'*64;raw=canonical(value)
            self.planner.adopt_manifest(raw,sha(raw),checkpoint_sha256='1'*64)
        self.run_mock(mutate)
        self.assertNotIn(('systemctl','--user','thaw',W.OLD_UNIT),self.calls)
        events=[json.loads(x)for x in(self.root/'transfer-events.jsonl').read_bytes().splitlines()]
        self.assertEqual(events[-1]['stage'],'integrity_stop_no_ancestor_replay')


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.old={'snapshot_sha256':'a'*64};self.prepared={'extension_sha256':'b'*64,'batch_manifest':{'snapshot_sha256':'b'*64}}
        self.captured={'manifest_sha256':'c'*64,'control':{'checkpoint_parameter_sha256':'d'*64}}
        self.proof={'absolute_optimizer_step':430,'local_optimizer_step':1,'execution_config_sha256':'e'*64,'config_sha256':W.CAMPAIGN,
            'checkpoint':{'step':430,'config_sha256':W.CAMPAIGN},'batch':{'manifest_sha256':'c'*64}}
        self.commit={'optimizer_step':1}
    def verify(self,receipt=None):W.verify_transition(self.proof,self.commit,self.captured,'e'*64,self.prepared,self.old,receipt,{'config_sha256':W.CAMPAIGN,'parameter_sha256':'d'*64})
    def test_held_candidate_physical_commit_needs_no_adoption(self):self.verify()
    def test_wrong_execution_identity_refuses(self):
        self.proof['execution_config_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'proof identity'):self.verify()
    def test_changed_manifest_needs_exact_bound_transition(self):
        self.proof['batch']['manifest_sha256']=sha(canonical(self.prepared['batch_manifest']))
        with self.assertRaisesRegex(ValueError,'transition receipt'):self.verify()
        receipt={'schema':'jope.prepared-epoch-adoption/v1','manifest_sha256':self.proof['batch']['manifest_sha256'],
            'previous_snapshot_sha256':'a'*64,'snapshot_sha256':'b'*64,'checkpoint_sha256':'d'*64,'optimizer_updates':0}
        self.verify(receipt);receipt['checkpoint_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'transition receipt'):self.verify(receipt)


class RetirementTests(unittest.TestCase):
    def test_nonzero_ack_with_physically_dead_ancestor_succeeds(self):
        result=types.SimpleNamespace(returncode=1,stdout='',stderr='ack failed after signal')
        events=[]
        with patch.object(W.subprocess,'run',return_value=result),patch.object(W,'unit',return_value={'MainPID':'0','ActiveState':'failed'}):
            self.assertTrue(W.retire_ancestor({'MainPID':'123','ControlGroup':'no-real-test-owner'},lambda stage,**row:events.append((stage,row))))
        self.assertEqual(events[-1][1]['returncode'],1);self.assertTrue(events[-1][1]['ancestor_retired'])
    def test_live_ancestor_is_not_declared_retired(self):
        result=types.SimpleNamespace(returncode=0,stdout='',stderr='')
        with patch.object(W.subprocess,'run',return_value=result),patch.object(W,'unit',return_value={'MainPID':'123','ActiveState':'active'}):
            self.assertFalse(W.retire_ancestor({'MainPID':'123','ControlGroup':'no-real-test-owner'},lambda *a,**k:None))


if __name__=='__main__':unittest.main()
