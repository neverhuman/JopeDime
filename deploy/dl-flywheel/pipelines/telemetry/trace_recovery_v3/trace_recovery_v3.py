"""Cold telemetry continuation from all frozen prior1+prior2 custody.

No past upload is retried. Recorded attempts, including unknown153, remain
charged. Numeric event/S3 verification primitives are exact reviewed copies.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
V2_SHA='fe000e7a0bed06f4b339997eaea745a1b0737430da5543f8f61c4d6ce553df22'
V2_CONFIG_SHA='ffc17797cf633444b616172aeccdfd88325b37ae4cfdc09abc93356bef0a9b70'
V2_ROOT=Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v2')
ROOT=Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v3')
UNIT='jopedime-hpc-archive-recovery-20261009-v3'
SCHEMA='jope.trace-archive-recovery/v3'
p=HERE/'trace_recovery_v2.py'
if not p.is_file()or any(x.is_symlink()for x in(p,*p.parents))or hashlib.sha256(p.read_bytes()).hexdigest()!=V2_SHA:
    raise ValueError('exact reviewed v2 before import')
spec=importlib.util.spec_from_file_location('trace_recovery_v2_frozen',p)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);a=v.a
OLD_UNITS=('jopedime-hpc-archive-20261009-v1',v.OLD_UNIT,v.NEW_UNIT)
EXPECTED={'fresh_first_part':607,'verified_segments':606,'verified_objects':1212,'prior_global_put_attempts':1213,'remaining_put_attempts':3787,'gap_parts':[153]}


def physical_files(root):
    """Complete shallow owned inventory; no unknown nested custody omitted."""
    out=set()
    for p in root.iterdir():
        a.need(not p.is_symlink(),'old custody symlink')
        if p.is_dir():
            a.need(p.name=='receipts','unknown old custody directory')
            for q in p.iterdir():
                a.need(q.is_file()and not q.is_symlink()and q.suffix=='.json','unknown nested old custody')
                out.add(str(q))
        elif p.is_file():
            if not p.name.endswith('.lock'):out.add(str(p))
        else:raise ValueError('old custody special file')
    return out


def extend_v2(prefix,entries,*,loader=a.pinned,segment_reader=a.segment):
    """Verify prior2 physically and extend a fully verified prior1 prefix.

Partial prior2 tails remain charged and held. Production separately requires the
exact reviewed complete prefix; unexpected new tails stop cold qualification.
"""
    raw=entries['config.json'];c=json.loads(raw)
    a.need(a.digest(raw)==V2_CONFIG_SHA and a.digest(entries['trace_recovery_v2.py'])==V2_SHA and
           a.digest(entries['trace_archive_v1_c8a13033.py'])==v.BASE_SHA,'exact prior2 source/config')
    a.need(c['schema']=='jope.trace-archive-recovery/v2'and c['out']==str(V2_ROOT)and c['logs_root']==a.LOGS and
           c['global_deadline_unix']==a.GLOBAL and c['maximum_segments']==2500 and
           c['maximum_global_put_attempts']==5000 and c['maximum_put_attempts']==prefix['remaining_put_attempts'],'prior2 immutable envelope')
    a.need(json.loads(entries['cold-prefix.json'])==prefix,'prior2 previously qualified prefix')
    start=json.loads(entries['execution-started.json']);a.need(start['config_sha256']==V2_CONFIG_SHA and
        start['prior_capture']==c['prior_capture']and start['automatic_retry']is False and
        c['created_unix']*1e9<=start['utc_ns']<c['deadline_unix']*1e9,'prior2 one-use/window binding')
    receipts=v.ordered(entries,'receipts/');attempts=v.ordered(entries,'put-attempt-')
    replies=v.ordered(entries,'put-reply-');verified=v.ordered(entries,'verified-object-');unknown=v.ordered(entries,'unknown-put-')
    first=prefix['fresh_first_part'];count=len(receipts);puts=len(attempts)
    a.need(sorted(receipts)==list(range(first,first+count))and sorted(attempts)==list(range(1,puts+1)),'prior2 numbering')
    a.need(2*count<=puts<=2*count+2 and puts<=c['maximum_put_attempts']and
           set(replies)<=set(attempts)and set(verified)<=set(replies)and set(unknown)<=set(attempts),'prior2 attempt custody closure')
    previous=prefix['previous'];seq=prefix['sequence'];archived=prefix['archived_events'];total=prefix['verified_bytes']
    gaps=list(prefix['gap_parts']);states={};keys=list(prefix['attempted_keys']);last=first+count-1+int(puts>2*count)
    a.need(last<2500,'prior2 global part ceiling')
    for part in range(first,last+1):
        body,sealed,head,end=segment_reader(Path(a.LOGS),part,previous,seq)
        states[part]={'events':{'sha256':a.digest(body),'bytes':len(body)},'seal':{'sha256':a.digest(sealed),'bytes':len(sealed)}}
        if part in receipts:
            r=json.loads(receipts[part]);a.need(r['schema']=='jope.hpc-host-archive-recovery-receipt/v2'and
                r['source_sha256']==V2_SHA and r['config_sha256']==V2_CONFIG_SHA and r['prior_capture']==c['prior_capture']and
                r['gap_parts']==prefix['gap_parts']and r['host']=='xbabe3'and r['part']==part and r['first_seq']==seq+1 and
                r['last_seq']==end and r['event_head']==head and
                r['global_put_attempts_through']==prefix['prior_global_put_attempts']+2*(part-first+1),'prior2 receipt/source/chain')
            a.object_ref(r['segment'],part,(body,sealed),'events.jsonl');a.object_ref(r['seal'],part,(body,sealed),'seal.json')
            archived+=end-seq;total+=len(body)+len(sealed)
        else:gaps.append(part)
        previous,seq=head,end
    for number,raw in attempts.items():
        att=json.loads(raw);part=first+(number-1)//2;label='events'if number%2 else'seal';name='events.jsonl'if number%2 else'seal.json'
        value=states[part][label];key=a.PREFIX+f'{part:05d}/'+states[part]['events']['sha256']+'/'+name
        a.need(att['schema']=='jope.trace-upload-attempt/v1'and att['attempt']==number and att['automatic_retry']is False and
            att['bucket']==a.BUCKET and att['key']==key and att['sha256']==value['sha256']and att['bytes']==value['bytes']and
            c['created_unix']*1e9<=att['utc_ns']<c['deadline_unix']*1e9,'prior2 durable attempt/window')
        a.need(key not in keys,'prior2 repeated attempted key');keys.append(key)
        if number in replies:
            reply=json.loads(replies[number]);a.need(reply['key']==key and att['utc_ns']<=reply['utc_ns'],'prior2 reply binding')
        if number in verified:
            vr=json.loads(verified[number]);a.need(vr['bucket']==a.BUCKET and vr['key']==key and
                vr['bytes']==value['bytes']and vr['sha256']==value['sha256']and vr['version_id']==reply['version_id']and
                vr['version_id']not in(None,'','null')and reply['utc_ns']<=vr['verified_at_unix_ns']<c['deadline_unix']*1e9,'prior2 exact-Version closure')
            if part in receipts:a.need(vr==json.loads(receipts[part])['segment'if number%2 else'seal'],'prior2 receipt/verified mismatch')
        if number in unknown:
            error=json.loads(unknown[number]);a.need(error['state']=='unknown_no_retry'and all(error[k]==value for k,value in att.items()),'prior2 unknown custody')
    a.need(set(range(1,2*count+1))<=set(verified),'prior2 receipt without two verified objects')
    p=json.loads(entries['progress.json']);a.need(p['schema']=='jope.trace-recovery-progress/v2'and
        p['config_sha256']==V2_CONFIG_SHA and p['deadline_unix']==c['deadline_unix']and
        p['state']in('completed_bounded_window_with_preserved_gaps','stopped_unknown_or_integrity_no_retry','running')and
        p['unresolved_gap_parts']==prefix['gap_parts']and p['cloud_contiguous_verified_through_part']==152 and
        0<=count-p['new_verified_segments']<=1 and 0<=puts-p['put_attempts']<=2,'prior2 progress cannot overstate custody')
    if p['state']=='running':a.need(time.time()>=c['deadline_unix'],'prior2 still within active window')
    if p['state']=='completed_bounded_window_with_preserved_gaps':
        a.need(puts==2*count==len(verified)and not unknown and p['new_verified_segments']==count and
            p['new_verified_objects']==2*count and p['put_attempts']==puts and
            p['global_put_attempts']==prefix['prior_global_put_attempts']+puts and
            p['verified_segments']==prefix['verified_segments']+count and p['verified_objects']==prefix['verified_objects']+2*count and
            p['next_fresh_part']==last+1 and p['local_chain_head']==previous and p['local_chain_last_seq']==seq and
            p['archived_events']==archived and p['verified_bytes']==total,'completed prior2 report differs from physical closure')
    lines=entries['stage-timing.jsonl'].splitlines()
    for n,line in enumerate(lines,1):
        e=json.loads(line);a.need(e['sequence']==n and e['config_sha256']==V2_CONFIG_SHA and e['host']=='xbabe3'and e['gpu_used']is False,'prior2 detailed timing identity')
        if e.get('kind')=='end':a.need(e['duration_monotonic_ns']>=0 and all(x>=0 for x in e['cpu_delta'].values()),'prior2 measured timing')
    a.need(p['stage_events']<=len(lines),'prior2 timing overstatement')
    return {**prefix,'previous':previous,'sequence':seq,'fresh_first_part':last+1,
        'verified_segments':prefix['verified_segments']+count,'verified_objects':prefix['verified_objects']+len(verified),
        'archived_events':archived,'verified_bytes':total,'prior_global_put_attempts':prefix['prior_global_put_attempts']+puts,
        'remaining_put_attempts':5000-prefix['prior_global_put_attempts']-puts,'gap_parts':gaps,'attempted_keys':keys,
        'prior2_progress':p,'prior2_complete_segments':count,'prior2_put_attempts':puts,'prior2_unknown_parts':gaps[len(prefix['gap_parts']):]}


def cold_prefix(prior1,prior2,*,loader=a.pinned,segment_reader=a.segment,inventory_reader=physical_files):
    a.need(prior2['schema']=='jope.trace-recovery2-closed-readonly-audit/v1'and prior2['host']=='xbabe3'and
           prior2['remote_writes']==prior2['S3_requests']==0,'closed prior2 capture authority')
    for observations in(prior2['units_before'],prior2['units_after']):
        a.need(set(observations)==set(OLD_UNITS),'all retired units captured')
        for u in observations.values():
            p=u['properties'];a.need(u['returncode']==0 and p['MainPID']=='0'and p['ControlGroup']==''and p['ActiveState']in('inactive','failed'),'captured prior2 physical closure')
    prefix=v.cold_prefix(prior1,loader=loader,segment_reader=segment_reader)
    # Recover original306 verified keys plus unknown153, missing from v2's
    # attempted_keys list but always charged. No historical key is forgotten.
    old=json.loads(loader({k:prior1['files'][str(v.OLD_ROOT/'config.json')][k]for k in('path','bytes','sha256')}))
    index=json.loads(loader(old['prior_index']));keys=[x['receipt'][name]['key']for x in index['receipts']for name in('segment','seal')]
    keys.append(prefix['gap_records'][0]['key']);keys.extend(prefix['attempted_keys'])
    a.need(len(keys)==len(set(keys))==prefix['prior_global_put_attempts'],'all original and prior1 attempted keys retained')
    entries={};expected={v.OLD_ROOT:set(),V2_ROOT:set()}
    for path,item in prior2['files'].items():
        p=Path(path);root=next((r for r in expected if r in p.parents),None);a.need(root is not None and p==Path(item['path']),'all prior capture paths confined')
        raw=loader({k:item[k]for k in('path','bytes','sha256')},64*1024**2);a.need(len(raw)==item['bytes']and a.digest(raw)==item['sha256'],'prior2 physical file drift')
        expected[root].add(path)
        if root==V2_ROOT:entries[p.relative_to(root).as_posix()]=raw
    for root,paths in expected.items():a.need(inventory_reader(root)==paths,'complete old owned file inventory changed')
    a.need(all(path in prior2['files']and all(prior2['files'][path][k]==item[k]for k in('bytes','sha256'))for path,item in prior1['files'].items()),'all prior1 capture pins retained')
    # Stored v2 prefix intentionally has its original shorter attempted-key list.
    extended=extend_v2(prefix,entries,loader=loader,segment_reader=segment_reader)
    newkeys=extended['attempted_keys'][len(prefix['attempted_keys']):]
    a.need(not(set(keys)&set(newkeys)),'cross-layer attempted key replay')
    extended['attempted_keys']=keys+newkeys
    a.need(len(extended['attempted_keys'])==extended['prior_global_put_attempts'],'global attempted-key closure')
    return extended


def preflight(c):
    a.need(set(c)=={'schema','host','logs_root','out','created_unix','deadline_unix','global_deadline_unix','maximum_segments','maximum_global_put_attempts','maximum_put_attempts','source','base','previous_source','prior1_capture','prior2_capture','terminal_proof'},'closed v3 config')
    a.need(c['schema']==SCHEMA and c['host']=='xbabe3'and c['logs_root']==a.LOGS and c['out']==str(ROOT),'fixed v3 scope')
    a.need(c['created_unix']<=time.time()<c['deadline_unix']<=c['global_deadline_unix']==a.GLOBAL and
        0<c['deadline_unix']-c['created_unix']<=3600,'new one-hour/original global clock')
    for key,name in(('source','trace_recovery_v3.py'),('base','trace_archive_v1_c8a13033.py'),('previous_source','trace_recovery_v2.py'),('prior1_capture','prior1-capture.json'),('prior2_capture','prior2-capture.json'),('terminal_proof','terminal-proof.json')):
        a.need(c[key]['path']==str(ROOT/name),'owned pinned v3 input');a.pinned(c[key],16*1024**2)
    a.need(c['source']['sha256']==a.digest(Path(__file__).read_bytes())and c['base']['sha256']==v.BASE_SHA and c['previous_source']['sha256']==V2_SHA,'executing source closure')
    for u in OLD_UNITS:v.show_closed(u)
    prefix=cold_prefix(json.loads(a.pinned(c['prior1_capture'],16*1024**2)),json.loads(a.pinned(c['prior2_capture'],16*1024**2)))
    a.need(all(prefix[k]==value for k,value in EXPECTED.items()),'reviewed exact terminal607/1213 closure required')
    proof=json.loads(a.pinned(c['terminal_proof']));a.need(proof['state']=='passed'and proof['config_sha256']==V2_CONFIG_SHA and
        proof['source_sha256']==V2_SHA and proof['global_put_attempts']==1213 and proof['first_unattempted_part']==607 and
        proof['remaining_global_put_attempts']==3787 and proof['full_raw_sealed_event_chain_reverified_through_part']==606,'root terminal proof binding')
    a.need(c['maximum_segments']==2500 and c['maximum_global_put_attempts']==5000 and c['maximum_put_attempts']==3787,'global budget conservation')
    for u in OLD_UNITS:v.show_closed(u)
    return prefix


class Archive(v.Archive):
    def once(self):
        self.tick();a.need(self.part<2500 and self.puts+2<=self.c['maximum_put_attempts'],'global segment/pair attempt ceiling')
        if not(self.logs/f'events-{self.part:05d}.sealed.json').exists():return False
        with self.timing.span('local.sealed_chain',part=self.part):body,sealed,head,end=a.segment(self.logs,self.part,self.previous,self.sequence)
        key=a.PREFIX+f'{self.part:05d}/'+a.digest(body)+'/'
        first=self.publish(key+'events.jsonl',body);second=self.publish(key+'seal.json',sealed);self.tick()
        receipt={'schema':'jope.hpc-host-archive-recovery-receipt/v3','host':'xbabe3','part':self.part,'segment':first,'seal':second,
            'first_seq':self.sequence+1,'last_seq':end,'event_head':head,'gap_parts':self.prefix['gap_parts'],
            'source_sha256':self.c['source']['sha256'],'config_sha256':self.timing.config_sha,
            'prior1_capture':self.c['prior1_capture'],'prior2_capture':self.c['prior2_capture'],
            'global_put_attempts_through':self.prefix['prior_global_put_attempts']+self.puts}
        a.create(self.out/'receipts'/f'{self.part:05d}.json',a.canonical(receipt))
        self.archived_events+=end-self.sequence;self.verified_bytes+=len(body)+len(sealed);self.previous,self.sequence=head,end;self.part+=1;self.new_segments+=1
        return True

    def report(self,state):
        # Exact v2 counter implementation, with an explicit new schema; the
        # mutable progress file is only in this new owned namespace.
        value={'schema':'jope.trace-recovery-progress/v3','state':state,'utc_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
            'config_sha256':self.timing.config_sha,'deadline_unix':self.c['deadline_unix'],
            'verified_segments':self.prefix['verified_segments']+self.new_segments,'verified_objects':self.prefix['verified_objects']+2*self.new_segments,
            'new_verified_segments':self.new_segments,'new_verified_objects':2*self.new_segments,'archived_events':self.archived_events,'verified_bytes':self.verified_bytes,
            'local_chain_last_seq':self.sequence,'local_chain_head':self.previous,'cloud_contiguous_verified_through_part':152,'unresolved_gap_parts':self.prefix['gap_parts'],
            'next_fresh_part':self.part,'put_attempts':self.puts,'global_put_attempts':self.prefix['prior_global_put_attempts']+self.puts,'get_attempts':self.gets,'head_attempts':self.heads,
            'stage_events':self.timing.sequence,'optimizer_updates':0,'gp_executions':0,'gpu_used':False,'cuda_execution_time_ns':None}
        temp=self.out/'progress.new'
        with temp.open('wb')as stream:stream.write(a.canonical(value));stream.flush();os.fsync(stream.fileno())
        os.replace(temp,self.out/'progress.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--config-sha',required=True);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    raw=a.read(args.config,1024**2);a.need(a.digest(raw)==args.config_sha and os.uname().nodename=='xbabe3','config/host');c=json.loads(raw)
    frozen_mono=time.monotonic()+c['deadline_unix']-time.time()
    def tick():a.need(min(c['deadline_unix']-time.time(),frozen_mono-time.monotonic())>0,'whole-worker fixed wall/monotonic deadline')
    begin=time.time_ns();mono=time.monotonic_ns();cpu=a.cpu();prefix=preflight(c);tick();after=a.cpu()
    checked={'started_utc_ns':begin,'ended_utc_ns':time.time_ns(),'duration_monotonic_ns':time.monotonic_ns()-mono,'cpu_delta':{k:after[k]-cpu[k]for k in cpu}}
    if args.preflight_only:
        print(json.dumps({'state':'passed','config_sha256':args.config_sha,**{k:prefix[k]for k in EXPECTED},'s3_requests':0,'qualification_timing':checked}));return
    group=next(line.split(':',2)[2]for line in Path('/proc/self/cgroup').read_text().splitlines()if line.startswith('0::'));cg=Path('/sys/fs/cgroup')/group.lstrip('/')
    quota,period=(cg/'cpu.max').read_text().split();a.need(quota!='max'and int(quota)<=int(period)and int((cg/'memory.max').read_text())<=512*1024**2 and int((cg/'memory.swap.max').read_text())==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU1/512MiB/swap0/noGPU')
    lock=os.open(Path(a.LOGS)/'archive-v1',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        # Preflight again under the globally shared archive lock: no past writer
        # can change a snapshot between cold validation and new publication.
        locked_begin=time.time_ns();locked_mono=time.monotonic_ns();locked_cpu=a.cpu()
        prefix=preflight(c);tick();locked_after=a.cpu()
        locked_checked={'started_utc_ns':locked_begin,'ended_utc_ns':time.time_ns(),'duration_monotonic_ns':time.monotonic_ns()-locked_mono,'cpu_delta':{k:locked_after[k]-locked_cpu[k]for k in locked_cpu}}
        a.create(ROOT/'execution-started.json',a.canonical({'config_sha256':args.config_sha,'utc_ns':time.time_ns(),'automatic_retry':False,'prior1_capture':c['prior1_capture'],'prior2_capture':c['prior2_capture']}))
        (ROOT/'receipts').mkdir();timing=a.Timing(ROOT/'stage-timing.jsonl',args.config_sha);timing.event({'kind':'verified','stage':'local.cold_prefix',**checked});timing.event({'kind':'verified','stage':'local.locked_cold_prefix',**locked_checked})
        import boto3
        from botocore.config import Config
        client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=20,retries={'max_attempts':0}))
        archive=Archive(c,client,timing,prefix);archive.monotonic_deadline=min(archive.monotonic_deadline,frozen_mono);state='running'
        try:
            archive.report(state)
            while min(c['deadline_unix']-time.time(),archive.monotonic_deadline-time.monotonic())>60 and archive.part<2500 and archive.puts+2<=c['maximum_put_attempts']:
                if not archive.once():time.sleep(min(15,max(0,min(c['deadline_unix']-time.time(),archive.monotonic_deadline-time.monotonic())-60)))
                archive.report(state)
            state='completed_bounded_window_with_preserved_gaps'
        except BaseException as error:
            state='stopped_unknown_or_integrity_no_retry';a.create(ROOT/'fatal.json',a.canonical({'utc_ns':time.time_ns(),'error':repr(error),'automatic_retry':False,'unknown_attempts_preserved':True}));raise
        finally:archive.report(state)
    finally:os.close(lock)


if __name__=='__main__':main()
