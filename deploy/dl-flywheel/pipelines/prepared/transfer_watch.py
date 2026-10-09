"""One bounded owned epoch handoff. Never thaw an ancestor after journal change."""
import json, os, sqlite3, subprocess, sys, time
from pathlib import Path
from jope_flywheel.program_io import bound, create, need, read, sha
from jope_flywheel.protocol import canonical
from reconcile import capture

OLD_UNIT='jopedime-neural-continuous-20261009-v1'
NEW_UNIT='jopedime-neural-continuous-resume3-20261009'
CAMPAIGN='fec184217e7c50a3fc3bc1cc7e33edafdbd218d1fafa1bd0d68debf626dcd4ed'


def call(*args):
    return subprocess.check_output(args,text=True,timeout=15).strip()


def unit(name):
    return dict(x.split('=',1)for x in call('systemctl','--user','show',name,
        '--property=MainPID','--property=ControlGroup','--property=ActiveState').splitlines())


def retire_ancestor(old,log):
    command=['systemctl','--user','kill','--kill-whom=all','--signal=SIGKILL',OLD_UNIT]
    log('retirement_intent',command=command,owner=old)
    result=subprocess.run(command,text=True,capture_output=True,timeout=15,check=False)
    after=unit(OLD_UNIT);group=Path('/sys/fs/cgroup')/old['ControlGroup'].lstrip('/')
    empty=not group.exists()or not(group/'cgroup.procs').read_text().strip()
    retired=after['MainPID']=='0'and after['ActiveState']in('failed','inactive')and empty
    log('retirement_physical_readback',returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        after=after,pinned_cgroup_empty=empty,ancestor_retired=retired)
    return retired


def verify_transition(proof, commit, capture_doc, execution_sha, prepared, old_manifest, adoption=None, adoption_checkpoint=None):
    step=proof['absolute_optimizer_step']
    need(proof['execution_config_sha256']==execution_sha and proof['config_sha256']==CAMPAIGN and
         proof['local_optimizer_step']==step-429==commit['optimizer_step'] and
         proof['checkpoint']['step']==step and proof['checkpoint']['config_sha256']==CAMPAIGN,
         'exact executor/campaign/absolute-local checkpoint proof identity')
    manifest=proof['batch']['manifest_sha256']
    if manifest==capture_doc['manifest_sha256']:return
    need(adoption is not None and adoption['schema']=='jope.prepared-epoch-adoption/v1'and
         adoption['manifest_sha256']==manifest==sha(canonical(prepared['batch_manifest'])) and
         adoption['previous_snapshot_sha256']==old_manifest['snapshot_sha256']and
         adoption['snapshot_sha256']==prepared['extension_sha256']and
         adoption_checkpoint is not None and adoption_checkpoint['config_sha256']==CAMPAIGN and
         adoption['checkpoint_sha256']==adoption_checkpoint['parameter_sha256']and
         adoption['optimizer_updates']==0, 'exact prepared boundary transition receipt')


def main():
    root=Path(__file__).resolve().parent
    raw=read(root/'transfer-plan.json');plan=json.loads(raw)
    need(len(sys.argv)==3 and sys.argv[1]=='--plan-sha' and sha(raw)==sys.argv[2],'fixed plan SHA')
    need(set(plan)=={'schema','campaign_root','created_unix','deadline_unix','global_deadline_unix',
                    'sources','prepared_candidate','cuda_qualification','start_completed_epochs'} and
         plan['schema']=='jope.boundary-transfer-plan/v1','closed handoff plan')
    need(plan['created_unix']<=time.time()<plan['deadline_unix']<=plan['global_deadline_unix']==1791592146.380285
         and plan['deadline_unix']-plan['created_unix']<=1800,'fixed bounded watcher window')
    need(set(plan['sources'])=={'continuous_cli.py','continuous_runner.py','prepared_admission.py',
                              'prepared_snapshot.py','reconcile.py','transfer_watch.py'},
         'complete reviewed execution source roster')
    for rel,pin in plan['sources'].items():need(sha(read(root/rel))==pin,'reviewed transfer source')
    qualification=json.loads(bound(plan['cuda_qualification']))
    need(qualification['state']=='passed' and qualification['model_and_adamw_and_cpu_cuda_rng_bit_exact']is True
         and qualification['live_optimizer_updates']==qualification['planner_mutations']==qualification['outbox_mutations']==0,
         'CUDA acceptance; never use a failed qualification')
    campaign=Path(plan['campaign_root']);config_raw=read(campaign/'config.json');c=json.loads(config_raw)
    need(sha(config_raw)==CAMPAIGN,'same logical campaign')
    training=campaign/'training';frozen=False;launched=False;accepted=False;successor_commit_verified=False;capture_doc=None
    old=None;execution_sha=None;original_progress=None;original_control=None
    def log(stage,**data):
        with(root/'transfer-events.jsonl').open('ab')as f:
            f.write(canonical({'stage':stage,'utc_time_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),**data})+b'\n');f.flush();os.fsync(f.fileno())
    log('waiting_for_reconciled_checkpoint')
    try:
        while time.time()<plan['deadline_unix']-240:
            p=json.loads(read(training/'progress.json',32*1024**2))
            if p['state']!='running':raise ValueError('active trainer is not running; no recovery assumption')
            if p['completed_epochs']<plan['start_completed_epochs']:
                time.sleep(.025);continue
            ready=sqlite3.connect((training/'batches.sqlite').as_uri()+'?mode=ro',uri=True)
            try:pending=ready.execute("SELECT count(*)FROM batches WHERE state!='committed'").fetchone()[0]
            finally:ready.close()
            if pending:time.sleep(.025);continue
            memory={k:int(v.split()[0])*1024 for k,v in
                    (s.split(':',1)for s in Path('/proc/meminfo').read_text().splitlines())}
            gpu_free=int(call('nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'))
            if memory['MemAvailable']<32*1024**3 or gpu_free<16000 or os.getloadavg()[0]>=100:
                log('resource_wait',available_bytes=memory['MemAvailable'],gpu_free_mib=gpu_free,load1=os.getloadavg()[0]);time.sleep(5);continue
            old=unit(OLD_UNIT);need(old['ActiveState']=='active'and int(old['MainPID'])>0,'owned active original unit')
            need(unit(NEW_UNIT)['ActiveState']not in('active','activating'),'successor not already launched')
            call('systemctl','--user','freeze',OLD_UNIT);frozen=True
            need('frozen 1'in(Path('/sys/fs/cgroup')/old['ControlGroup'].lstrip('/')/'cgroup.events').read_text().splitlines(),'physical freeze complete')
            original_progress=read(training/'progress.json',32*1024**2);original_control=read(training/'optimizer-control.json')
            try:
                capture_doc=capture(training,campaign_sha256=CAMPAIGN,frozen=True,max_updates=c['neural']['max_updates'],
                                    plateau_patience=c['neural']['plateau_patience'],learning_rate_floor=c['neural']['learning_rate_floor'],require_closed_epoch=False)
            except ValueError as error:
                log('capture_refused_ancestor_thawed',error=str(error)[:500]);call('systemctl','--user','thaw',OLD_UNIT);frozen=False
                time.sleep(.025);continue
            create(root/'transfer-capture.json',canonical(capture_doc));log('physical_checkpoint_captured',absolute_step=capture_doc['absolute_step'],completed_epochs=capture_doc['progress']['completed_epochs'])
            ref=lambda p:{'path':str(p),'sha256':sha(read(p,32*1024**2)),'bytes':p.stat().st_size}
            execution={'schema':'jope.same-campaign-execution-config/v1','campaign_root':str(campaign),
                       'campaign_config':ref(campaign/'config.json'),'owned_sources':plan['sources'],
                       'transfer_capture':ref(root/'transfer-capture.json'),'prepared_candidate':plan['prepared_candidate'],
                       'frozen_owner':{'unit':OLD_UNIT,'pid':int(old['MainPID']),'control_group':old['ControlGroup']}}
            body=canonical(execution);create(root/'execution-config.json',body);execution_sha=sha(body)
            command=['systemd-run','--user','--unit='+NEW_UNIT,'--property=CPUQuota=800%','--property=MemoryMax=8G',
                '--property=MemorySwapMax=0','--property=Nice=10','--property=TasksMax=64','--property=KillMode=control-group',
                '--property=Restart=no','--property=RuntimeMaxSec='+str(int(c['global_deadline_unix']-time.time())),
                '--setenv=CUDA_VISIBLE_DEVICES=0','--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8','--setenv=TMPDIR='+str(campaign/'tmp'),
                '--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=NUMEXPR_NUM_THREADS=1',
                '/usr/bin/python3','-B',str(root/'continuous_cli.py'),'--execution-sha',execution_sha]
            log('launch_intent',command=command,execution_config_sha256=execution_sha);launched=True;call(*command)
            started=time.time()
            while time.time()<min(started+210,plan['deadline_unix']):
                new=unit(NEW_UNIT);progress=json.loads(read(training/'progress.json',32*1024**2))
                if progress.get('execution_config_sha256')==execution_sha and progress['completed_batches']>capture_doc['absolute_step']:
                    step=progress['completed_batches'];proof=json.loads(read(training/f'batch-{step:09d}.json'))
                    cp=proof['checkpoint'];slot=training/'rolling'/f'{step:09d}'
                    try:state=read(slot/'state.safetensors',32*1024**2)
                    except FileNotFoundError:continue
                    need(new['ActiveState']=='active'and cp['step']==step and cp['parameter_sha256']==progress['checkpoint_sha256'] and
                         len(state)==cp['state_bytes']and sha(state)==cp['state_sha256'],'successor physical committed checkpoint')
                    db=sqlite3.connect((training/'batches.sqlite').as_uri()+'?mode=ro',uri=True,isolation_level=None);db.row_factory=sqlite3.Row
                    try:
                        db.execute('BEGIN');row=db.execute('SELECT state,commit_body FROM batches WHERE id=?',(proof['batch']['batch_id'],)).fetchone()
                        commit=json.loads(row['commit_body'])if row is not None else{}
                        need(row is not None and row['state']=='committed'and commit['evidence']['event_sha256']==sha(read(training/f'batch-{step:09d}.json'))and commit['checkpoint_after']==cp['parameter_sha256'], 'immutable proof joins physical planner commit')
                    finally:db.execute('ROLLBACK');db.close()
                    db=sqlite3.connect((training/'outbox.sqlite').as_uri()+'?mode=ro',uri=True)
                    try:
                        ack=db.execute('SELECT ack FROM batches WHERE id=?',(proof['batch']['batch_id'],)).fetchone()
                        need(ack is not None and json.loads(ack[0])==proof['emission_ack'],'committed successor proposal ACK')
                    finally:db.close()
                    old_raw=read(training/'initial-batch-manifest.json',32*1024**2)
                    need(sha(old_raw)==capture_doc['manifest_sha256'],'captured exact initial manifest')
                    prepared={};adoption=None;adoption_checkpoint=None
                    if proof['batch']['manifest_sha256']!=capture_doc['manifest_sha256']:
                        prepared=json.loads(bound(plan['prepared_candidate']['prepared'],32*1024**2))
                        matches=[]
                        for path in training.glob('adopted-prepared-transfer-*.json'):
                            value=json.loads(read(path))
                            if value.get('manifest_sha256')==proof['batch']['manifest_sha256']:matches.append((path,value))
                        need(len(matches)==1,'one exact adopted manifest receipt')
                        path,adoption=matches[0];adopt_step=int(path.stem.removeprefix('adopted-prepared-transfer-'))
                        need(capture_doc['absolute_step']<=adopt_step<step,'preceding boundary transition')
                        transition_raw=read(training/f'batch-{adopt_step:09d}.json');transition=json.loads(transition_raw)
                        adoption_checkpoint=transition['checkpoint']
                        db=sqlite3.connect((training/'batches.sqlite').as_uri()+'?mode=ro',uri=True)
                        try:
                            transition_row=db.execute('SELECT state,commit_body FROM batches WHERE id=?',(transition['batch']['batch_id'],)).fetchone()
                            transition_commit=json.loads(transition_row[1])if transition_row is not None else{}
                            need(transition_row is not None and transition_row[0]=='committed'and transition_commit['evidence']['event_sha256']==sha(transition_raw)and
                                 transition_commit['checkpoint_after']==adoption_checkpoint['parameter_sha256']and
                                 transition['batch']['manifest_sha256']==capture_doc['manifest_sha256']and adoption_checkpoint['step']==adopt_step,
                                 'exact committed closed epoch transition checkpoint')
                            closed=db.execute('SELECT state FROM epochs WHERE id=?',(transition['batch']['epoch'],)).fetchone()
                            need(closed is not None and closed[0]=='complete','transition follows completed epoch')
                        finally:db.close()
                    verify_transition(proof,commit,capture_doc,execution_sha,prepared,json.loads(old_raw),adoption,adoption_checkpoint)
                    successor_commit_verified=True
                    # SIGKILL cannot run the frozen ancestor's finally block against
                    # the successor's live journals. This was a reconciled own unit.
                    ancestor=unit(OLD_UNIT)
                    need(ancestor['MainPID']==old['MainPID'] and ancestor['ControlGroup']==old['ControlGroup'] and
                         'frozen 1'in(Path('/sys/fs/cgroup')/old['ControlGroup'].lstrip('/')/'cgroup.events').read_text().splitlines(),
                         'unchanged owned physically frozen ancestor')
                    need(retire_ancestor(old,log),'owned ancestor retirement not physically confirmed');frozen=False;accepted=True
                    pointer={'schema':'jope.active-continuous-executor/v1','campaign_sha256':CAMPAIGN,
                             'unit':NEW_UNIT,'execution_config':ref(root/'execution-config.json'),'absolute_resume_step':capture_doc['absolute_step']}
                    create(campaign/'active-executor.json',canonical(pointer))
                    log('transfer_committed',absolute_step=step,unique_contexts=progress['unique_train_data_runs_seen'],execution_config_sha256=execution_sha)
                    create(root/'transfer-result.json',canonical({'state':'passed','execution_config_sha256':execution_sha,
                        'captured_absolute_step':capture_doc['absolute_step'],'first_verified_absolute_step':step,
                        'global_deadline_unix':c['global_deadline_unix'],'ancestor_intentionally_retired':True}))
                    return
                if new['ActiveState']not in('active','activating'):raise ValueError('successor exited before acceptance')
                time.sleep(.25)
            raise TimeoutError('successor handoff acceptance window expired; no automatic optimizer replay')
        log('watcher_window_expired_no_transfer')
    finally:
        if frozen and not accepted and successor_commit_verified:
            log('verified_successor_preserved_retirement_or_pointer_incomplete',automatic_retry=False)
        if frozen and not accepted and not successor_commit_verified:
            if not launched:
                need(unit(OLD_UNIT)['MainPID']==old['MainPID'],'same own ancestor before thaw')
                call('systemctl','--user','thaw',OLD_UNIT);log('no_successor_launch_ancestor_thawed')
            else:
                if launched and unit(NEW_UNIT)['ActiveState']in('active','activating'):
                    call('systemctl','--user','stop',NEW_UNIT)
                # Never thaw after adoption, emitted intent, optimizer or journal drift.
                db=sqlite3.connect((training/'batches.sqlite').as_uri()+'?mode=ro',uri=True,isolation_level=None);db.row_factory=sqlite3.Row
                from neural_native_stream import BatchStream
                check=BatchStream.__new__(BatchStream);check.db=db;check.config=check._get('config')
                try:
                    db.execute('BEGIN');same=capture_doc is not None and check._verify()==capture_doc['planner_head']and check._get('manifest')==capture_doc['manifest_sha256'];db.execute('ROLLBACK')
                finally:db.close()
                if same and unit(OLD_UNIT)['MainPID']==old['MainPID']:
                    from neural_native_train import publish
                    publish(training,'progress.json',json.loads(original_progress));publish(training,'optimizer-control.json',json.loads(original_control))
                    call('systemctl','--user','thaw',OLD_UNIT);log('unmutated_ancestor_thawed')
                else:log('integrity_stop_no_ancestor_replay',automatic_retry=False)


if __name__=='__main__':main()
