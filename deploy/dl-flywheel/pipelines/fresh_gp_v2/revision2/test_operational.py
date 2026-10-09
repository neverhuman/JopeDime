"""New cold-operation fixtures: real files/SQLite; no network, fitting or PUT."""
import ast,hashlib,json,sqlite3,sys,tempfile,time,unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent
sys.path[:0]=[str(H),str(H/'runtime')]
import common,broker,epoch,history_collect as hc,native_runtime
from common import ref,put,obj,canonical,digest
from jope_flywheel.protocol import Context
from jope_flywheel.program_gate import request_identity
from jope_flywheel.queue import Queue
from neural_outbox import Outbox
from global_attempts import Attempts
from execution_boundary import ExecutionBoundary

NOW=1791558000.0 # Explicit fixture wall clock only; production GLOBAL unchanged.
class Native:
    def canonical_program(self,s):
        if s=='invalid':raise ValueError('native syntax')
        return ' '.join(s.split())
def context(i=1):return Context('regression',f'{i:064x}','2'*64,'cut'+str(i),'3'*64,'4'*64,'5'*64,'6'*64,'rmse','prepared_synthetic_target_units','minimize')
class Cases(unittest.TestCase):
 def setUp(self):
    self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name).resolve();self.addCleanup(self.tmp.cleanup)
    self.clock=patch('time.time',return_value=NOW);self.clock.start();self.addCleanup(self.clock.stop)
 def write(self,name,value):return put(self.r,name,value)
 def queue(self,name='queue.sqlite',program='(ridge)'):
    q=Queue(self.r/name);self.addCleanup(q.close);ctx=context();key=request_identity(ctx,program)
    q.enqueue('gp-evaluate-'+key,'evaluate','9'*64,{'context':asdict(ctx),'program':program,'evaluation_key':key})
    return q
 def snapshot(self,q,native=None):return hc.snapshot({'path':str(q.path),'kind':'queue'},native or Native(),common.Clock(NOW+60))
 def test_real_queue_snapshot_preserves_source_and_all_pending_aliases(self):
    q=self.queue();before=q.verify();raw=q.path.read_bytes();doc=self.snapshot(q)
    self.assertEqual(doc['counts']['jobs'],1);self.assertEqual(len(doc['aliases']),1)
    self.assertEqual(q.verify(),before);self.assertEqual(q.path.read_bytes(),raw)
    self.assertEqual(doc['aliases'][0]['disposition'],'hold')
 def test_native_unparseable_is_whole_context_hold(self):
    q=self.queue(program='invalid');doc=self.snapshot(q)
    self.assertIsNone(doc['aliases'][0]['canonical_program_sha256']);self.assertEqual(doc['aliases'][0]['context_id'],context().identity)
 def test_real_outbox_recorded_null_key_held_without_lease(self):
    o=Outbox(self.r/'outbox.sqlite',model_origin_sha256='a'*64);self.addCleanup(o.close)
    cid=context().identity;b={'batch_id':'b'*64,'campaign_sha256':'c'*64,'checkpoint_sha256':'d'*64,'global_step':1,'manifest_sha256':'e'*64,'train_data_run_ids':[cid],'validation_data_run_ids':[],'emitted_train_data_run_ids':[cid]}
    e={'phase':'training','data_run_id':cid,'context_id':cid,'parent_lineage_sha256':'f'*64,'input_descriptor':{'source_kind':'native_s1_context'},'raw_program_utf8':'(ridge)','raw_program_sha256':hashlib.sha256(b'(ridge)').hexdigest(),'decoder_seed':1,'decoder_protocol_sha256':'1'*64,'decode_artifact_sha256':'2'*64}
    o.record_batch(b,[e]);before=o.verify()
    d=hc.snapshot({'path':str(self.r/'outbox.sqlite'),'kind':'outbox','config':o.config},Native(),common.Clock(NOW+60))
    self.assertEqual(d['counts']['emissions'],1);self.assertIsNone(d['aliases'][0]['evaluation_key']);self.assertEqual(d['native_context_ids'],[cid]);self.assertEqual(o.verify(),before)
 def test_tampered_queue_fails_full_verifier(self):
    q=self.queue();q.db.execute("UPDATE jobs SET state='succeeded'")
    with self.assertRaises(ValueError):self.snapshot(q)
 def test_physical_closed_unit_rejects_descendant_and_active(self):
    unit='old.service';base=self.r/'cgroups';rel=base/'user.slice/user-1000.slice/user@1000.service/app.slice'/unit;rel.mkdir(parents=True)
    (rel/'cgroup.procs').write_text('');(rel/'cgroup.events').write_text('populated 0\n')
    def cmd(*_):return 'Id=old.service\nLoadState=loaded\nActiveState=inactive\nMainPID=0\nControlGroup=\n'
    self.assertTrue(hc.closed_unit(unit,common.Clock(NOW+60),cmd,base)['cgroups'])
    (rel/'cgroup.procs').write_text('123')
    with self.assertRaises(ValueError):hc.closed_unit(unit,common.Clock(NOW+60),cmd,base)
 def test_controller_starttime_reuse_and_live_refusal(self):
    proc=self.r/'proc';(proc/'123').mkdir(parents=True)
    (proc/'123/stat').write_text('123 (a name) '+ ' '.join(['S']+['0']*18+['777']))
    with self.assertRaises(ValueError):hc.closed_controller({'pid':123,'starttime':'777'},proc)
    self.assertTrue(hc.closed_controller({'pid':123,'starttime':'666'},proc)['closed'])
 def collections(self):
    prior=self.write('prior.json',{'schema':'jope.native-global-census/v1','coverage_complete':True,'entries':[{'context_id':'a'*64,'canonical_program_sha256':None,'disposition':'blocked'}]})
    sources=[{'path':'/fixture/queue%d'%i,'kind':'queue'}for i in range(18)]+[{'path':'/fixture/out%d'%i,'kind':'outbox'}for i in range(2)]
    unit={'unit':'old.service','systemd':{'MainPID':'0','ActiveState':'inactive'},'cgroups':[{'exists':False}]}
    inv=self.write('inventory.json',{'host':'xbabe2','previous_global_census':prior,'canonicalizer':{'sha256':'c'*64},'legacy_units':['old.service'],'legacy_controllers':[{'pid':123,'starttime':'4'}],'sources':sources})
    refs=[self.write('source%d.json'%i,{'source':s,'full_source_verified':True,'verifier':{'head_sha256':'0'*64},'native_context_ids':[],'aliases':[]})for i,s in enumerate(sources)]
    c={'schema':'jope.fresh128-host-history/v2','host':'xbabe2','physical_readonly':True,'source_sql_mutations':0,'checked_at_unix':NOW,'inventory':inv,'previous_global_census':prior,'canonicalizer_sha256':'c'*64,'units_before':[unit],'units_after':[unit],'controllers':[{'pid':123,'starttime':'4','closed':True}],'sources':refs}
    return c
 def test_complete_history_retains_prior_unknown(self):
    c=self.collections();r=self.write('collection.json',c)
    aliases,_,_,_=hc.validate_collections([r],['/fixture/out0','/fixture/out1'],60)
    self.assertIn(('a'*64,None),aliases)
 def test_history_dropped_source_refused(self):
    c=self.collections();c['sources'].pop()
    with self.assertRaises(ValueError):hc.validate_collections([self.write('collection.json',c)],['/fixture/out0','/fixture/out1'])
 def test_history_forged_controller_identity_refused(self):
    c=self.collections();c['controllers'][0]['pid']=999
    with self.assertRaises(ValueError):hc.validate_collections([self.write('collection.json',c)],['/fixture/out0','/fixture/out1'])
 def test_history_rehash_each_execution(self):
    c=self.collections();r=self.write('collection.json',c);h=hc.CapturedHistory([r],['/fixture/out0','/fixture/out1']);h.check()
    Path(c['sources'][0]['path']).write_text('{}')
    with self.assertRaises(ValueError):h.check()
 def test_native_toolchain_stream_hash_and_tamper(self):
    root=self.r/'native';root.mkdir();f=root/'lib';f.write_bytes(b'exact native lib');r=self.write('native.json',{'files':[{'relative':'lib','bytes':f.stat().st_size,'sha256':ref(f)['sha256']}]})
    self.assertEqual(native_runtime.verify(r,str(root),NOW+60)['files'],1)
    f.write_bytes(b'wrong native lib')
    with self.assertRaises(ValueError):native_runtime.verify(r,str(root),NOW+60)
 def test_broker_registry_epoch_context_alias_never_relaunch(self):
    db=sqlite3.connect(self.r/'broker.sqlite',isolation_level=None);self.addCleanup(db.close)
    db.executescript('CREATE TABLE epochs(sha TEXT PRIMARY KEY,body TEXT,started REAL);CREATE TABLE contexts(context_id TEXT PRIMARY KEY,variant TEXT UNIQUE,epoch TEXT,descriptor_sha TEXT,partition TEXT);')
    e={'context_descriptors':[{'context_id':'a'*64,'variant_identity':'b'*64,'descriptor_sha256':'c'*64,'partition':'validation'}]}
    broker.reserve(db,e,'1'*64)
    with self.assertRaises(ValueError):broker.reserve(db,e,'1'*64)
    with self.assertRaises(sqlite3.IntegrityError):broker.reserve(db,e,'2'*64)
    self.assertEqual(db.execute('SELECT count(*)FROM epochs').fetchone()[0],1)
    self.assertEqual(db.execute('SELECT partition FROM contexts').fetchone()[0],'validation')
 def test_fence_missing_cooperative_writer_authority_refused(self):
    req=self.write('request.json',{'candidate_inventory':{'sha256':'a'*64}});p={'request':req}
    h={'schema':'jope.fresh128-variant-writer-fence/v2','candidate_inventory_sha256':'a'*64,'global_deadline_unix':common.GLOBAL,'foreign_variant_writers_released':True,'all_future_variant_launchers_require_broker':False,'broker_root':str(epoch.BROKER_ROOT),'broker_lock':str(epoch.BROKER_ROOT/'epoch.lock'),'history_collection_sha256':[],'authorized_at_unix':NOW,'root_authorization':'explicit fixture root authorization','compatible_native_writers':[]}
    with self.assertRaises(ValueError):epoch.handoff(self.write('fence.json',h),p,[])
 def test_clock_backward_never_extends_monotonic(self):
    with patch('time.monotonic',return_value=100):c=common.Clock(NOW+60)
    with patch('time.time',return_value=NOW-100),patch('time.monotonic',return_value=161):
        with self.assertRaises(ValueError):c.tick()
 def attempts_fixture(self):
    desc={'variant_identity':'b'*64,'partition':'validation','parent_lineage_sha256':'c'*64,'teaching_eligible':False};ctx=context()
    ds=[{'context_id':context(i+1).identity,'variant_identity':f'{i+100:064x}','descriptor_sha256':'d'*64,'partition':'train'}for i in range(128)]
    ds[0]={'context_id':ctx.identity,'variant_identity':desc['variant_identity'],'descriptor_sha256':digest(desc),'partition':'validation'}
    c={'campaign_sha256':'e'*64,'selection_sha256':'f'*64,'ownership_producer_sha256':'a'*64,'history_sources':[],'work_deadline_unix':NOW+600,'global_deadline_unix':common.GLOBAL,'shard_ids':['shard00'],'context_owners':{r['context_id']:'shard00'for r in ds},'context_descriptors':ds,'source_pins':[]}
    proof={'schema':'jope.fresh-gp-context-ownership/v1','passed':True,'host':'xbabe3','campaign_sha256':c['campaign_sha256'],'producer_sha256':c['ownership_producer_sha256'],'history_census_sha256':digest([]),'selection_sha256':c['selection_sha256'],'checked_at_unix':NOW,'lease_expires_unix':NOW+300,'global_deadline_unix':common.GLOBAL,'legacy_variant_workers_closed':True,'legacy_launch_controllers_closed':True,'active_xbabe2_source_kind':'native_s1_context','active_native_context_overlap':[],'other_variant_context_overlap':[],'all_legacy_units_physically_closed':True,'context_descriptors':ds}
    a=Attempts(self.r/'attempts.sqlite',c,create=True);self.addCleanup(a.close);a.reserve_contexts(ds,proof,'shard00');q=self.queue();j=q.claim('fixture',['evaluate'],lease_seconds=300,max_running=1)
    class Inputs:
        parents={'c'*64:{}}
        def describe(self,identity):return ctx,b'raw',deepcopy(desc)
        class Lease:
            def check(self,*_):pass
        lease=Lease()
    class Hist:
        pin='f'*64
        def check(self):return {}
    g=ExecutionBoundary(c,q,Inputs(),a,Hist(),'shard00',lambda:proof)
    return a,q,j,g,ctx,desc,proof
 def test_real_registry_verifier_respects_existing_transaction(self):
    a,*_=self.attempts_fixture();a.db.execute('BEGIN');a.verify();self.assertTrue(a.db.in_transaction);a.db.execute('ROLLBACK');a.verify();self.assertFalse(a.db.in_transaction)
 def test_final_guard_expensive_preparation_then_expired_lease_zero_intent(self):
    a,q,j,g,ctx,d,p=self.attempts_fixture();old=g.inputs.describe
    def costly(v):
        result=old(v);j['lease_deadline']=NOW-1;return result
    g.inputs.describe=costly
    with self.assertRaises(ValueError):g.before_fit(j,ctx,'(ridge)',b'raw',d)
    self.assertEqual(a.verify()['held_attempts'],0)
 def test_final_guard_preserves_historical_whole_context_hold(self):
    a,q,j,g,ctx,d,p=self.attempts_fixture();g.history.check=lambda:{(ctx.identity,None):[{'unknown':True}]}
    with self.assertRaises(ValueError):g.before_fit(j,ctx,'(ridge)',b'raw',d)
    self.assertEqual(a.db.execute('SELECT state FROM attempts').fetchone()[0],'prior_hold')
 def test_final_guard_owned_validation_exec_permitted_once_not_admitted(self):
    a,q,j,g,ctx,d,p=self.attempts_fixture();g.before_fit(j,ctx,'(ridge)',b'raw',d)
    self.assertFalse(d['teaching_eligible']);self.assertEqual(a.verify()['held_attempts'],1)
    with self.assertRaises(ValueError):g.before_fit(j,ctx,'(ridge)',b'raw',d)
 def test_epoch_freeze_actual_files_exact16shards_and128contexts(self):
    c=self.collections();collection=self.write('collection.json',c)
    request=self.write('prepared-request.json',{'candidate_inventory':{'sha256':'a'*64}})
    sources=self.write('source-manifest.json',{'schema':'jope.fresh128-source/v2','files':[ref(H/'broker.py'),ref(H/'transport.py')]})
    rows=[{'context_id':context(i+1).identity,'variant_identity':f'{i+100:064x}','descriptor_sha256':'d'*64,'partition':'validation'if i<10 else'train'}for i in range(128)]
    contexts=self.write('contexts.json',rows)
    prep=self.write('prepared.json',{'request':request,'contexts':contexts,'source_manifest':sources,'programs':[{'program_sha256':f'{i+1:064x}'}for i in range(8)],'selection':[],'selection_sha256':digest([])})
    fence=self.write('fence.json',{'schema':'jope.fresh128-variant-writer-fence/v2','candidate_inventory_sha256':'a'*64,'global_deadline_unix':common.GLOBAL,'foreign_variant_writers_released':True,'all_future_variant_launchers_require_broker':True,'broker_root':str(epoch.BROKER_ROOT),'broker_lock':str(epoch.BROKER_ROOT/'epoch.lock'),'history_collection_sha256':[collection['sha256']],'authorized_at_unix':NOW,'root_authorization':'fixture explicit closed-writer declaration only','compatible_native_writers':[]})
    request=self.write('epoch-request.json',{'schema':'jope.fresh128-epoch-request/v2','prepared':prep,'authorized_concurrency':1,'history_collections':[collection],'required_outboxes':['/fixture/out0','/fixture/out1'],'fencing_handoff':fence,'broker_producer_sha256':ref(H/'broker.py')['sha256'],'physical_producer_sha256':ref(H/'transport.py')['sha256'],'resource_reservations':[]})
    with patch.object(epoch,'host'):
        er=epoch.freeze(request,self.r/'epoch')
    e=obj(er);self.assertEqual(len(e['shards']),16);self.assertEqual(sum(x['planned_slots']for x in e['shards']),1024)
    self.assertEqual(len(e['context_owners']),128);self.assertEqual(e['deadline_unix']-e['created_unix'],1800)
    self.assertEqual(sum(r['partition']=='validation'for r in e['context_descriptors']),10)
 def test_actual_preflight_invocation_contract_no_jobs_no_nativefit(self):
    import executor,numpy as np
    native=self.r/'native';native.mkdir();(native/'lib').write_bytes(b'fixture')
    nr=self.write('native.json',{'files':[{'relative':'lib','bytes':7,'sha256':ref(native/'lib')['sha256']}]})
    c={'maximum_actual_evaluations':64,'schema':'jope.fresh-variant-gp-campaign/v1','native_canonicalizer':{},'variants':[{'identity':f'{i:064x}','parent_lineage_sha256':f'{i:064x}','partition':'validation'if i==0 else'train'}for i in range(8)],'programs':[{'program_sha256':f'{i:064x}'}for i in range(8)]}
    e={'work_deadline_unix':NOW+600,'source_manifest':{'sha256':'a'*64}};p={'runtime_versions':{'numpy':np.__version__},'native_closure':nr,'execution_native_root':str(native)};shard={'owned_root':str(self.r/'futureworker')}
    with patch.object(executor,'cfg',return_value=(e,c,p,shard,{})),patch.object(executor,'check_inputs',return_value=(object(),object())):
        result=executor.preflight({'sha256':'f'*64},'shard00',self.r/'preflight',client=object(),host_check=False,envelope_check=False,native_factory=lambda *_:object())
    self.assertEqual(obj(result)['native_executions'],0);self.assertFalse(Path(shard['owned_root']).exists())
    db=sqlite3.connect(self.r/'preflight/invocation.sqlite');self.addCleanup(db.close);self.assertEqual(db.execute('SELECT count(*)FROM jobs').fetchone()[0],0)
 def test_cli_exact_source_bootstrap_and_foreign_host_refusal(self):
    import shutil,subprocess
    dst=self.r/'cli';dst.mkdir();files=[]
    for f in list(H.glob('*.py'))+list((H/'runtime').rglob('*.py')):
        if f.name.startswith('test_'):continue
        q=dst/f.relative_to(H);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q);files.append(ref(q))
    manifest=self.write('cli-sources.json',{'schema':'jope.fresh128-source/v2','files':files});request=self.write('probe.json',{'host':'definitely-not-this-host'})
    argv=[sys.executable,str(dst/'fresh_gp.py'),'--source-manifest',manifest['path'],'--source-sha',manifest['sha256'],'probe','--ref',request['path'],'--sha',request['sha256']]
    run=__import__('subprocess').run(argv,capture_output=True,text=True,timeout=10)
    self.assertNotEqual(run.returncode,0);self.assertIn('exact host required',run.stderr)
    (dst/'transport.py').write_text("raise RuntimeError('MUTATED_EXECUTED')")
    run=__import__('subprocess').run(argv,capture_output=True,text=True,timeout=10)
    self.assertIn('source hash differs',run.stderr);self.assertNotIn('MUTATED_EXECUTED',run.stderr)
 def test_runtime_math_functions_remain_exact_borrowed_source(self):
    def nodes(p):return {n.name:ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(p.read_text()))if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old=nodes(H/'borrowed/program_worker_original.py');new=nodes(H/'runtime/jope_flywheel/program_worker.py')
    for name in old:
        if name not in('__init__','evaluate','publish'):self.assertEqual(old[name],new[name],name)
if __name__=='__main__':unittest.main()
