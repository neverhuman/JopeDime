"""New prior2 custody/unknown fixtures only. No network or archived-suite replay."""
import copy,io,json,tempfile,time,unittest
from contextlib import ExitStack
import prepare_config as freeze_config
import launch_v3 as launch
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import trace_recovery_v3 as t

class Prior2:
    def __init__(self,root,complete=2,tail=0,tail_verified=False):
        self.root=root;self.logs=root/'logs';self.logs.mkdir();self.old=root/'prior2';self.old.mkdir();(self.old/'receipts').mkdir()
        self.now=1791564000.;self.created=self.now-200;self.deadline=self.now-100;self.complete=complete;self.tail=tail
        self.prefix={'fresh_first_part':529,'verified_segments':528,'verified_objects':1056,'prior_global_put_attempts':1057,'remaining_put_attempts':3943,
          'previous':'a'*64,'sequence':528,'archived_events':527,'verified_bytes':1000,'gap_parts':[153],'gap_records':[{'key':'unknown-original-153'}],
          'attempted_keys':['old-'+str(i)for i in range(750)]}
        self.raws={};previous=self.prefix['previous'];seq=528
        for part in range(529,529+complete+bool(tail)):
            seq+=1;e={'host':'xbabe3','seq':seq,'previous':previous};previous=t.a.digest(t.a.canonical(e));e['sha256']=previous;body=t.a.canonical(e)+b'\n'
            p=self.logs/f'events-{part:05d}.jsonl';p.write_bytes(body);seal={'schema':'jope.hpc-host-log-segment/v1','path':str(p),'sha256':t.a.digest(body),'bytes':len(body),'last_seq':seq,'event_head':previous};s=t.a.canonical(seal);(self.logs/f'events-{part:05d}.sealed.json').write_bytes(s);self.raws[part]=(body,s,previous,seq)
        self.c={'schema':'jope.trace-archive-recovery/v2','out':str(self.old),'logs_root':str(self.logs),'created_unix':self.created,'deadline_unix':self.deadline,
          'global_deadline_unix':t.a.GLOBAL,'maximum_segments':2500,'maximum_global_put_attempts':5000,'maximum_put_attempts':3943,'prior_capture':{'path':'fixture-prior1','sha256':'b'*64,'bytes':1}}
        self.write('config.json',self.c);self.csha=t.a.digest((self.old/'config.json').read_bytes())
        (self.old/'trace_recovery_v2.py').write_bytes(t.p.read_bytes());(self.old/'trace_archive_v1_c8a13033.py').write_bytes(t.v.base_path.read_bytes())
        self.write('cold-prefix.json',self.prefix);self.write('execution-started.json',{'config_sha256':self.csha,'prior_capture':self.c['prior_capture'],'automatic_retry':False,'utc_ns':int((self.created+1)*1e9)})
        for n in range(1,2*complete+tail+1):
            part=529+(n-1)//2;vr=self.cloud(part,n%2==1);att={'schema':'jope.trace-upload-attempt/v1','attempt':n,'automatic_retry':False,'utc_ns':int((self.created+2)*1e9),'monotonic_ns':n,**{k:vr[k]for k in('bucket','key','bytes','sha256')}};self.write(f'put-attempt-{n:05d}.json',att)
            if n<=2*complete or tail_verified:
                self.write(f'put-reply-{n:05d}.json',{'key':vr['key'],'version_id':vr['version_id'],'utc_ns':int((self.created+3)*1e9)});self.write(f'verified-object-{n:05d}.json',vr)
            else:self.write(f'unknown-put-{n:05d}.json',{**att,'state':'unknown_no_retry','error':'synthetic unknown'})
        for part in range(529,529+complete):
            body,sealed,head,end=self.raws[part];self.write(f'receipts/{part:05d}.json',{'schema':'jope.hpc-host-archive-recovery-receipt/v2','host':'xbabe3','part':part,'first_seq':end,'last_seq':end,'event_head':head,'source_sha256':t.V2_SHA,'config_sha256':self.csha,'prior_capture':self.c['prior_capture'],'gap_parts':[153],'global_put_attempts_through':1057+2*(part-528),'segment':self.cloud(part,True),'seal':self.cloud(part,False)})
        last=528+complete;head=self.raws[last][2]if complete else self.prefix['previous'];seq=last
        self.write('progress.json',{'schema':'jope.trace-recovery-progress/v2','state':'stopped_unknown_or_integrity_no_retry'if tail else'completed_bounded_window_with_preserved_gaps','config_sha256':self.csha,'deadline_unix':self.deadline,'unresolved_gap_parts':[153],'cloud_contiguous_verified_through_part':152,
          'new_verified_segments':complete,'new_verified_objects':2*complete,'put_attempts':2*complete+tail,'global_put_attempts':1057+2*complete+tail,'verified_segments':528+complete,'verified_objects':1056+2*complete,'next_fresh_part':last+1,'local_chain_head':head,'local_chain_last_seq':seq,'archived_events':527+complete,'verified_bytes':1000+sum(len(x[0])+len(x[1])for p,x in self.raws.items()if p<=last),'stage_events':1})
        (self.old/'stage-timing.jsonl').write_bytes(t.a.canonical({'sequence':1,'config_sha256':self.csha,'host':'xbabe3','gpu_used':False})+b'\n')
    def write(self,name,value):(self.old/name).write_bytes(t.a.canonical(value))
    def cloud(self,part,events):
        body,sealed,_,_=self.raws[part];raw=body if events else sealed
        return {'bucket':t.a.BUCKET,'key':t.a.PREFIX+f'{part:05d}/'+t.a.digest(body)+('/events.jsonl'if events else'/seal.json'),'version_id':f'v{part}-{events}','bytes':len(raw),'sha256':t.a.digest(raw),'verified_at_unix_ns':int((self.created+4)*1e9)}
    def entries(self):return {p.relative_to(self.old).as_posix():p.read_bytes()for p in self.old.rglob('*')if p.is_file()}
    def verify(self,entries=None):
        with patch.object(t,'V2_CONFIG_SHA',self.csha),patch.object(t,'V2_ROOT',self.old),patch.object(t.a,'LOGS',str(self.logs)),patch.object(time,'time',return_value=self.now):return t.extend_v2(self.prefix,entries or self.entries())

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
    def test_complete_actual_cardinality78_charges156_and_starts607(self):
        f=Prior2(self.root,78);p=f.verify();self.assertEqual((p['fresh_first_part'],p['verified_segments'],p['verified_objects'],p['prior_global_put_attempts'],p['remaining_put_attempts']),(607,606,1212,1213,3787));self.assertEqual(p['gap_parts'],[153])
    def test_single_unknown_attempt_charged_and_entire_part_held(self):
        p=Prior2(self.root,2,1).verify();self.assertEqual((p['fresh_first_part'],p['prior_global_put_attempts']),(532,1062));self.assertEqual(p['gap_parts'],[153,531])
    def test_two_unknown_attempts_never_make_complete_segment(self):
        p=Prior2(self.root,2,2).verify();self.assertEqual((p['fresh_first_part'],p['verified_segments'],p['prior_global_put_attempts']),(532,530,1063))
    def test_both_verified_without_segment_receipt_remain_held(self):
        p=Prior2(self.root,2,2,True).verify();self.assertEqual(p['verified_objects'],1062);self.assertEqual(p['verified_segments'],530);self.assertIn(531,p['gap_parts'])
    def test_missing_attempt_refuses_numbering(self):
        f=Prior2(self.root);(f.old/'put-attempt-00002.json').unlink()
        with self.assertRaisesRegex(ValueError,'numbering'):f.verify()
    def test_receipt_missing_verified_object_refuses(self):
        f=Prior2(self.root);(f.old/'verified-object-00001.json').unlink()
        with self.assertRaisesRegex(ValueError,'two verified'):f.verify()
    def test_changed_receipt_version_refuses(self):
        f=Prior2(self.root);p=f.old/'receipts/00529.json';d=json.loads(p.read_bytes());d['segment']['version_id']='other';p.write_bytes(t.a.canonical(d))
        with self.assertRaisesRegex(ValueError,'receipt/verified'):f.verify()
    def test_changed_prior_source_refuses(self):
        f=Prior2(self.root);(f.old/'trace_recovery_v2.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'source/config'):f.verify()
    def test_changed_prior_prefix_refuses(self):
        f=Prior2(self.root);d=copy.deepcopy(f.prefix);d['prior_global_put_attempts']=0;f.write('cold-prefix.json',d)
        with self.assertRaisesRegex(ValueError,'qualified prefix'):f.verify()
    def test_new_attempt_outside_old_clock_refuses(self):
        f=Prior2(self.root);p=f.old/'put-attempt-00001.json';d=json.loads(p.read_bytes());d['utc_ns']=int((f.deadline+1)*1e9);p.write_bytes(t.a.canonical(d))
        with self.assertRaisesRegex(ValueError,'attempt/window'):f.verify()
    def test_full_completed_report_cannot_hide_unknown_tail(self):
        f=Prior2(self.root,2,1);p=f.old/'progress.json';d=json.loads(p.read_bytes());d['state']='completed_bounded_window_with_preserved_gaps';p.write_bytes(t.a.canonical(d))
        with self.assertRaisesRegex(ValueError,'completed prior2'):f.verify()
    def test_event_chain_corruption_refuses(self):
        f=Prior2(self.root);(f.logs/'events-00529.jsonl').write_text('bad')
        with self.assertRaisesRegex(ValueError,'segment bytes'):f.verify()
    def test_timing_source_mismatch_refuses(self):
        f=Prior2(self.root);(f.old/'stage-timing.jsonl').write_text('{"sequence":1,"config_sha256":"bad"}\n')
        with self.assertRaisesRegex(ValueError,'timing identity'):f.verify()
    def test_repeated_prior_attempted_key_refuses(self):
        f=Prior2(self.root);f.prefix['attempted_keys'].append(f.cloud(529,True)['key']);f.write('cold-prefix.json',f.prefix)
        with self.assertRaisesRegex(ValueError,'repeated attempted'):f.verify()
    def test_complete_owned_inventory_detects_extra_and_symlink(self):
        f=Prior2(self.root);before=t.physical_files(f.old);f.write('unknown-new.json',{});self.assertNotEqual(before,t.physical_files(f.old));(f.old/'badlink').symlink_to(f.old/'config.json')
        with self.assertRaisesRegex(ValueError,'symlink'):t.physical_files(f.old)
    def archive(self,f,p,budget=3787,client=None):
        out=self.root/'new';out.mkdir();(out/'receipts').mkdir();c={'out':str(out),'logs_root':str(f.logs),'deadline_unix':time.time()+120,'maximum_put_attempts':budget,'source':{'sha256':'1'*64},'prior1_capture':{'sha256':'2'*64},'prior2_capture':{'sha256':'3'*64}}
        return t.Archive(c,client,t.a.Timing(out/'timing.jsonl','4'*64),p)
    def test_old_key_refused_before_put_and_no_old_files_changed(self):
        f=Prior2(self.root);p=f.verify();old=f.entries();x=self.archive(f,p)
        with self.assertRaisesRegex(ValueError,'replay'):x.publish(p['attempted_keys'][-1],b'x')
        self.assertEqual(x.puts,0);self.assertEqual(old,f.entries())
    def test_pair_budget_refuses_before_first_object(self):
        f=Prior2(self.root);x=self.archive(f,f.verify(),1)
        with self.assertRaisesRegex(ValueError,'pair attempt'):x.once()
        self.assertEqual(x.puts,0)
    def test_backward_wall_clock_cannot_extend_monotonic_deadline(self):
        f=Prior2(self.root);x=self.archive(f,f.verify());x.monotonic_deadline=time.monotonic()-1
        with patch.object(time,'time',return_value=1):
            with self.assertRaisesRegex(ValueError,'monotonic'):x.tick()
    def test_one_use_namespace_refuses_second_constructor(self):
        f=Prior2(self.root);p=f.verify();x=self.archive(f,p)
        with self.assertRaises(FileExistsError):t.Archive(x.c,None,x.timing,p)
    def test_fresh_success_exact_two_version_gets_and_schema3(self):
        f=Prior2(self.root,2,1);p=f.verify();part=p['fresh_first_part'];seq=p['sequence']+1;e={'host':'xbabe3','seq':seq,'previous':p['previous']};h=t.a.digest(t.a.canonical(e));e['sha256']=h;raw=t.a.canonical(e)+b'\n';path=f.logs/f'events-{part:05d}.jsonl';path.write_bytes(raw);(f.logs/f'events-{part:05d}.sealed.json').write_bytes(t.a.canonical({'schema':'jope.hpc-host-log-segment/v1','path':str(path),'bytes':len(raw),'sha256':t.a.digest(raw),'last_seq':seq,'event_head':h}))
        class Client:
            def __init__(self):self.objects={};self.puts=self.gets=0
            def put_object(self,**kw):
                self.puts+=1;assert kw['IfNoneMatch']=='*';self.objects[kw['Key']]=kw['Body'];return {'VersionId':str(self.puts)}
            def get_object(self,**kw):self.gets+=1;r=self.objects[kw['Key']];return {'VersionId':kw['VersionId'],'ContentLength':len(r),'Body':io.BytesIO(r)}
        client=Client();x=self.archive(f,p,client=client);self.assertTrue(x.once());x.report('running');r=json.loads((x.out/'receipts'/f'{part:05d}.json').read_bytes())
        self.assertEqual(r['schema'],'jope.hpc-host-archive-recovery-receipt/v3');self.assertEqual((client.puts,client.gets),(2,2));self.assertEqual(r['gap_parts'],[153,531]);self.assertEqual(r['global_put_attempts_through'],p['prior_global_put_attempts']+2)

class ColdJoin(unittest.TestCase):
    """Exercise new-layer real-file joining; old v2 verifier has its own frozen proof.

    Only the already-verified prior1 return is injected here. All prior2 artifacts,
    inventories, old-index keys, raw new event chain and hashes are physically read.
    """
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.f=Prior2(self.root,78)
        self.old1=self.root/'prior1';self.old1.mkdir()
        index={'receipts':[{'receipt':{'segment':{'key':f'original-{n}-events'},'seal':{'key':f'original-{n}-seal'}}}for n in range(153)]}
        ip=self.old1/'original-index.json';ip.write_bytes(t.a.canonical(index));cp=self.old1/'config.json';cp.write_bytes(t.a.canonical({'prior_index':self.ref(ip)}))
        self.prior1={'files':{str(p):self.ref(p)for p in self.old1.iterdir()}}
        closed={name:{'returncode':0,'properties':{'MainPID':'0','ControlGroup':'','ActiveState':'inactive'}}for name in t.OLD_UNITS}
        files={str(p):self.ref(p)for root in(self.old1,self.f.old)for p in root.rglob('*')if p.is_file()}
        self.prior2={'schema':'jope.trace-recovery2-closed-readonly-audit/v1','host':'xbabe3','remote_writes':0,'S3_requests':0,'units_before':copy.deepcopy(closed),'units_after':copy.deepcopy(closed),'files':files}
    def ref(self,p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':t.a.digest(p.read_bytes())}
    def verify(self):
        with ExitStack()as stack:
            stack.enter_context(patch.object(t.v,'OLD_ROOT',self.old1));stack.enter_context(patch.object(t,'V2_ROOT',self.f.old));stack.enter_context(patch.object(t,'V2_CONFIG_SHA',self.f.csha));stack.enter_context(patch.object(t.a,'LOGS',str(self.f.logs)));stack.enter_context(patch.object(time,'time',return_value=self.f.now))
            old=stack.enter_context(patch.object(t.v,'cold_prefix',return_value=copy.deepcopy(self.f.prefix)))
            value=t.cold_prefix(self.prior1,self.prior2);old.assert_called_once();self.assertIs(old.call_args.args[0],self.prior1);return value
    def test_all_three_attempt_layers_retained_exact1213_and_unknown153(self):
        before={p:p.read_bytes()for root in(self.old1,self.f.old)for p in root.rglob('*')if p.is_file()};value=self.verify()
        self.assertEqual(len(value['attempted_keys']),1213);self.assertEqual(len(set(value['attempted_keys'])),1213);self.assertIn('unknown-original-153',value['attempted_keys']);self.assertEqual(value['fresh_first_part'],607)
        self.assertTrue(all(p.read_bytes()==raw for p,raw in before.items()))
    def test_active_prior2_in_capture_refuses_before_old_verifier(self):
        self.prior2['units_after'][t.OLD_UNITS[-1]]['properties']['MainPID']='17'
        with self.assertRaisesRegex(ValueError,'physical closure'):self.verify()
    def test_changed_physical_prior1_file_refuses(self):
        (self.old1/'original-index.json').write_text('{}')
        with self.assertRaises(ValueError):self.verify()
    def test_prior2_file_changed_after_capture_refuses(self):
        (self.f.old/'put-attempt-00001.json').write_text('{}')
        with self.assertRaises(ValueError):self.verify()
    def test_uncaptured_file_after_freeze_refuses_full_inventory(self):
        (self.f.old/'new-unknown-attempt.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'inventory changed'):self.verify()
    def test_omitted_prior1_pin_cannot_erase_history(self):
        self.prior2['files'].pop(str(self.old1/'original-index.json'))
        with self.assertRaisesRegex(ValueError,'inventory changed'):self.verify()

class Constructor(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.c={'schema':'jope.trace-archive-recovery/v3','out':str(freeze_config.ROOT),'created_unix':None,'deadline_unix':None,'global_deadline_unix':freeze_config.GLOBAL,'maximum_segments':2500,'maximum_global_put_attempts':5000,'maximum_put_attempts':3787}
        for key in freeze_config.INPUTS:
            raw=(key+' fixture').encode();name=key+'.json';(self.root/name).write_bytes(raw);self.c[key]={'path':str(freeze_config.ROOT/name),'sha256':t.a.digest(raw),'bytes':len(raw)}
        self.pin()
    def pin(self):
        raw=t.a.canonical(self.c);(self.root/'config.template.json').write_bytes(raw);self.template_sha=t.a.digest(raw)
        raw=t.a.canonical({'state':'passed','template_sha256':self.template_sha});self.review=self.root/'independent.json';self.review.write_bytes(raw);self.review_sha=t.a.digest(raw)
    def freeze(self,now=1791564000):return freeze_config.freeze(self.root,self.template_sha,self.review,self.review_sha,now=now)
    def test_one_use_exact_one_hour_preserves_global_counts(self):
        result=self.freeze();c=json.loads(Path(result['path']).read_bytes());self.assertEqual(c['deadline_unix']-c['created_unix'],3600);self.assertEqual((c['maximum_put_attempts'],c['maximum_segments'],c['global_deadline_unix']),(3787,2500,freeze_config.GLOBAL))
        with self.assertRaises(FileExistsError):self.freeze()
    def test_insufficient_global_window_no_config_created(self):
        with self.assertRaisesRegex(ValueError,'original global deadline'):self.freeze(freeze_config.GLOBAL-100)
        self.assertFalse((self.root/'config.frozen.json').exists())
    def test_changed_review_or_source_refuses_before_freeze(self):
        (self.root/'source.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'input pin'):self.freeze()
        self.assertFalse((self.root/'config.frozen.json').exists())
    def test_replenished_attempt_budget_refuses_before_freeze(self):
        self.c['maximum_put_attempts']=5000;self.pin()
        with self.assertRaisesRegex(ValueError,'original limits'):self.freeze()

class Launch(unittest.TestCase):
    """Operational CLI exercised offline; subprocesses never reach the host."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.now=1791564000.;self.calls=[]
        self.c={'schema':'jope.trace-archive-recovery/v3','out':str(self.root),'created_unix':self.now,'deadline_unix':self.now+3600,'global_deadline_unix':t.a.GLOBAL,'maximum_segments':2500,'maximum_global_put_attempts':5000,'maximum_put_attempts':3787}
        for key in freeze_config.INPUTS:
            raw=key.encode();p=self.root/(key+'.json');p.write_bytes(raw);self.c[key]={'path':str(p),'bytes':len(raw),'sha256':t.a.digest(raw)}
        self.raw=t.a.canonical(self.c);(self.root/'config.json').write_bytes(self.raw);self.sha=t.a.digest(self.raw)
        self.proof={'state':'passed','config_sha256':self.sha,**t.EXPECTED,'s3_requests':0,'qualification_timing':{'started_utc_ns':1,'ended_utc_ns':2,'duration_monotonic_ns':1,'cpu_delta':{'user_seconds':0}}}
        self.active_old=False;self.existing_new=False
    def show(self,unit):
        if unit in launch.OLD:return {'MainPID':'1'if self.active_old else'0','ControlGroup':'x'if self.active_old else'','ActiveState':'active'if self.active_old else'inactive','LoadState':'loaded'}
        return {'MainPID':'0','ControlGroup':'','ActiveState':'inactive','LoadState':'loaded'if self.existing_new else'not-found'}
    def fake_run(self,argv,**kwargs):
        self.calls.append(argv)
        return SimpleNamespace(returncode=0,stdout=json.dumps(self.proof)if '--preflight-only'in argv else'created',stderr='')
    def invoke(self):
        original=Path.read_text
        def read(path,*args,**kw):
            if str(path)=='/proc/meminfo':return f'MemAvailable: {30*1024**2} kB\n'
            return original(path,*args,**kw)
        with ExitStack()as st:
            for obj,name,value in((launch,'ROOT',self.root),(launch,'show',self.show),(launch.subprocess,'run',self.fake_run),(launch.os,'uname',lambda:SimpleNamespace(nodename='xbabe3')),(launch.os,'getloadavg',lambda:(0,0,0)),(time,'time',lambda:self.now),(Path,'read_text',read)):
                st.enter_context(patch.object(obj,name,value))
            st.enter_context(patch('sys.argv',['launch_v3.py','--config-sha',self.sha]));st.enter_context(patch('sys.stdout',io.StringIO()));launch.main()
    def test_exact_cold_preflight_then_bounded_single_launch(self):
        self.invoke();self.assertEqual(len(self.calls),2);self.assertIn('--preflight-only',self.calls[0]);self.assertNotIn('--preflight-only',self.calls[1]);self.assertIn('MemoryMax=512M',self.calls[1]);self.assertIn('CPUQuota=100%',self.calls[1]);self.assertIn('Restart=no',self.calls[1]);self.assertTrue((self.root/'launch-receipt.json').is_file())
        with self.assertRaises(FileExistsError):self.invoke()
        self.assertEqual(len(self.calls),2)
    def test_any_old_active_unit_stops_before_preflight(self):
        self.active_old=True
        with self.assertRaises(AssertionError):self.invoke()
        self.assertEqual(self.calls,[])
    def test_existing_new_unit_stops_before_preflight(self):
        self.existing_new=True
        with self.assertRaises(AssertionError):self.invoke()
        self.assertEqual(self.calls,[])
    def test_wrong_cumulative_preflight_count_stops_before_launch(self):
        self.proof['prior_global_put_attempts']=1212
        with self.assertRaises(AssertionError):self.invoke()
        self.assertEqual(len(self.calls),1);self.assertFalse((self.root/'launch-intent.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
