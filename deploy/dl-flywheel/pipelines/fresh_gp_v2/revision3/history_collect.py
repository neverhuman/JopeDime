"""Host-local physical read-only full queue/outbox canonical census.

Uses the copied unchanged Queue/Outbox full verifiers on private SQLite backup.
Canonical aliases include zero-attempt/pending and null-key rows conservatively.
No history source is opened writable and no registry/lease is mutated.
"""
import json,os,re,signal,socket,subprocess,tempfile,time
from pathlib import Path
from common import *
from readonly_snapshot import verified_snapshot,source_path
from native_global_guard import queue_aliases
from jope_flywheel.program_io import NativeTools

UNIT=re.compile(r'[a-zA-Z0-9_.-]{1,190}\.service\Z')

def command(argv,clock,input_body=None):
    clock.tick()
    with tempfile.TemporaryFile()as out, tempfile.TemporaryFile()as inp:
        if input_body is not None:
            need(type(input_body)is bytes and len(input_body)<=1024*1024,'bounded stdin payload')
            inp.write(input_body);inp.seek(0)
        p=subprocess.Popen(argv,stdin=inp,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll()is None:
                clock.tick();need(out.tell()<=1024*1024,'bounded physical command output');time.sleep(.02)
            out.seek(0);raw=out.read(1024*1024+1)
            need(len(raw)<=1024*1024 and p.returncode in(0,1,3,4),'physical read failed');clock.tick();return raw.decode()
        finally:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=2)

def closed_unit(unit,clock,runner=command,cgroup_root=Path('/sys/fs/cgroup')):
    need(UNIT.fullmatch(unit),'closed exact unit name')
    text=runner(['systemctl','--user','show',unit,'--property=Id,LoadState,ActiveState,SubState,MainPID,ControlGroup'],clock)
    v=dict(l.split('=',1)for l in text.splitlines()if'='in l)
    need(v.get('Id')==unit and v.get('LoadState')in('loaded','not-found')and
         v.get('ActiveState')in('inactive','failed')and v.get('MainPID')=='0','legacy writer still active')
    candidates={Path('user.slice/user-1000.slice/user@1000.service/app.slice')/unit};cg=v.get('ControlGroup','')
    if cg:need(cg.startswith('/')and'..'not in Path(cg).parts,'safe cgroup');candidates.add(Path(cg.lstrip('/')))
    groups=[]
    for rel in sorted(candidates):
        p=cgroup_root/rel;need(all(not q.is_symlink()for q in(p,*p.parents)),'physical cgroup symlink')
        if not p.exists():groups.append({'path':str(p),'exists':False});continue
        procs=(p/'cgroup.procs').read_text().split();ev=dict(x.split()for x in(p/'cgroup.events').read_text().splitlines())
        need(not procs and ev.get('populated')=='0','legacy descendants remain')
        groups.append({'path':str(p),'exists':True,'pids':[],'populated':0})
    return {'unit':unit,'systemd':v,'cgroups':groups,'checked_at_unix':time.time()}

def closed_controller(row,proc=Path('/proc')):
    # Exact historical PID plus starttime prevents PID-reuse false attribution.
    need(type(row['pid'])is int and row['pid']>0 and type(row['starttime'])is str,'controller identity')
    p=proc/str(row['pid'])
    if not p.exists():return {**row,'closed':True,'reason':'pid_absent'}
    text=(p/'stat').read_text();start=text[text.rfind(')')+2:].split()[19]
    need(start!=row['starttime'],'historical launch controller still running')
    return {**row,'closed':True,'reason':'different_process_starttime','observed_starttime':start}

def snapshot(source,native,clock):
    p=source_path(source['path']);before=p.stat()
    db,proof=verified_snapshot(p,deadline=clock.deadline,kind=source['kind'],expected_config=source.get('config'))
    aliases=[];contexts=set();counts={}
    try:
        if source['kind']=='queue':
            for row in queue_aliases(db):
                clock.tick()
                try:program=native.canonical_program(row['program'])
                except ValueError:program=None
                aliases.append({'context_id':row['context_id'],'canonical_program_sha256':hashlib.sha256(program.encode()).hexdigest()if program else None,
                    'evaluation_key':row['evaluation_key'],'states':row['states'],'disposition':'hold'})
            counts={'jobs':db.execute('SELECT count(*)FROM jobs').fetchone()[0],
                    'evaluation_aliases':len(aliases)}
        else:
            for row in db.execute('SELECT *FROM emissions ORDER BY rowid'):
                clock.tick();body=json.loads(row['body']);e=body['emission'];raw=e['raw_program_utf8']
                need(hashlib.sha256(raw.encode()).hexdigest()==e['raw_program_sha256'],'outbox raw DSL hash')
                try:program=native.canonical_program(raw)
                except ValueError:program=None
                outcome=json.loads(row['outcome'])if row['outcome']else{}
                aliases.append({'context_id':e['context_id'],'canonical_program_sha256':hashlib.sha256(program.encode()).hexdigest()if program else None,
                    'evaluation_key':outcome.get('evaluation_key'),'states':[{'state':row['state']}],
                    'source_kind':e['input_descriptor'].get('source_kind'),'emission_id':row['id'],'disposition':'hold'})
                if e['input_descriptor'].get('source_kind')=='native_s1_context':contexts.add(e['context_id'])
            counts={'emissions':len(aliases)}
        for a in aliases:
            need(type(a['context_id'])is str and len(a['context_id'])==64 and all(c in'0123456789abcdef'for c in a['context_id']),'unresolved context alias')
            need(a['canonical_program_sha256']is None or(type(a['canonical_program_sha256'])is str and len(a['canonical_program_sha256'])==64),'canonical alias or whole-context hold')
        after=p.stat();need((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'source inode replaced during read')
        return {'source':source,'physical':{'device':before.st_dev,'inode':before.st_ino},'verifier':proof,
                'counts':counts,'aliases':aliases,'native_context_ids':sorted(contexts),'full_source_verified':True}
    finally:db.close()

def collect(spec_ref,out,*,hostname=None,native_factory=NativeTools,unit_reader=closed_unit):
    spec=obj(spec_ref);need(spec['schema']=='jope.fresh128-history-inventory/v2'and not spec.get('requires_coordination_before_use'),'closed qualified history inventory')
    coordination=obj(spec['coordination']);need(coordination['schema']=='jope.fresh128-cold-coordination/v2'and coordination['host']==spec['host']and coordination['root_declares_complete_writer_inventory']is True and coordination['legacy_units']==spec['legacy_units']and coordination['legacy_controllers']==spec['legacy_controllers'],'explicit complete writer inventory binding')
    need((hostname or socket.gethostname().split('.')[0])==spec['host']and spec['host']in('xbabe1','xbabe2','xbabe3'),'exact inventory host')
    clock=Clock(min(time.time()+spec['maximum_seconds'],GLOBAL),600)
    original=obj(spec['historical_census']);need(original['schema']=='jope.variant-historical-census/v1'and len(original['source_queues'])==18,'authoritative18 queue census')
    expected={r['path']for r in original['source_queues']if r['host']==spec['host']}
    queues=[r for r in spec['sources']if r['kind']=='queue']
    need({r['path']for r in queues}==expected and len(queues)==len(expected),'no historical queue omitted')
    need(len({r['path']for r in spec['sources']})==len(spec['sources']),'unique source inventory')
    need(spec['legacy_units']and len(set(spec['legacy_units']))==len(spec['legacy_units']),'explicit nonempty units')
    for pin in spec['source_pins']:bound(pin,64*MIB)
    before=[unit_reader(u,clock)for u in spec['legacy_units']]
    controllers=[closed_controller(r)for r in spec['legacy_controllers']]
    native=native_factory(spec['canonicalizer'],clock.deadline);root=fresh(out);sources=[]
    for i,s in enumerate(spec['sources']):
        doc=snapshot(s,native,clock)
        if s['kind']=='queue':
            prior=next(q for q in original['source_queues']if q['path']==s['path']and q['host']==spec['host'])
            keys={a['evaluation_key']for a in doc['aliases']}
            need(set(prior['completed_evaluation_keys'])|set(prior['blocked_evaluation_keys'])<=keys,'historical completed/unknown aliases cannot disappear')
        sources.append(put(root,'history-%03d.json'%i,doc))
    after=[unit_reader(u,clock)for u in spec['legacy_units']]
    need([closed_controller(r)for r in spec['legacy_controllers']]==controllers,'controller closure changed')
    result={'schema':'jope.fresh128-host-history/v2','host':spec['host'],'inventory':spec_ref,'sources':sources,
            'units_before':before,'units_after':after,'controllers':controllers,'checked_at_unix':time.time(),
            'physical_readonly':True,'source_sql_mutations':0,'canonicalizer_sha256':spec['canonicalizer']['sha256'],
            'previous_global_census':spec['previous_global_census']}
    return put(root,'collection.json',result)

def validate_collections(refs,required_outboxes,maximum_age=None):
    aliases={};contexts=set();queue_paths=set();outboxes=set();units=[];inventories=[]
    for r in refs:
        c=obj(r);need(c['schema']=='jope.fresh128-host-history/v2'and c['physical_readonly']is True and c['source_sql_mutations']==0,'qualified host history')
        if maximum_age is not None:need(0<=time.time()-c['checked_at_unix']<=maximum_age,'fresh cold capture required')
        inv=obj(c['inventory']);inventories.append(c['inventory'])
        need(c['previous_global_census']==inv['previous_global_census'],'permanent previous census pin')
        previous=obj(c['previous_global_census'])
        need(previous['schema']=='jope.native-global-census/v1'and previous['coverage_complete']is True,'previous canonical unknown holds')
        for a in previous['entries']:
            aliases.setdefault((a['context_id'],a['canonical_program_sha256']),[]).append({**a,'previous_census':c['previous_global_census']})
        need(c['host']==inv['host']and c['canonicalizer_sha256']==inv['canonicalizer']['sha256'],'history producer identity')
        need([r['unit']for r in c['units_before']]==[r['unit']for r in c['units_after']]==inv['legacy_units'],'all unit closure retained')
        for u in c['units_after']:
            need(u['systemd'].get('MainPID')=='0'and u['systemd'].get('ActiveState')in('inactive','failed')and
                 all(g['exists']is False or(g['pids']==[]and g['populated']==0)for g in u['cgroups']),'physical closure incomplete')
        need(len(c['controllers'])==len(inv['legacy_controllers'])and all(r['closed']is True and all(r[k]==old[k]for k in old)for r,old in zip(c['controllers'],inv['legacy_controllers'])),'controller closure incomplete')
        docs=[obj(x)for x in c['sources']]
        need([d['source']for d in docs]==inv['sources'],'all original queues/outboxes captured')
        for d in docs:
            need(d['full_source_verified']is True and 'head_sha256'in d['verifier'],'full source verifier')
            s=d['source'];(queue_paths if s['kind']=='queue'else outboxes).add(s['path'])
            contexts.update(d['native_context_ids'])
            for a in d['aliases']:aliases.setdefault((a['context_id'],a['canonical_program_sha256']),[]).append({**a,'source':s,'capture':r})
        units.extend({'host':c['host'],'unit':u}for u in inv['legacy_units'])
    need(len(queue_paths)==18 and set(required_outboxes)<=outboxes and len(set(required_outboxes))>=2,'full18 queues and original/continuation outboxes')
    return aliases,contexts,units,inventories

class CapturedHistory:
    def __init__(self,refs,required_outboxes,broker_history,bootstrap):
        self.refs=refs;self.required=required_outboxes;self.broker_history=broker_history;self.bootstrap=bootstrap
        self.pin=digest({'legacy':refs,'permanent':broker_history,'bootstrap':bootstrap})
    def check(self):
        aliases,_,_,_=validate_collections(self.refs,self.required)
        from broker_history import validate
        _,held=validate(self.broker_history,self.bootstrap)
        for key,rows in held.items():aliases.setdefault(key,[]).extend(rows)
        return aliases
    def matching(self,context,program):return self.check().get((context.identity,hashlib.sha256(program.encode()).hexdigest()),[])
