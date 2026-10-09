"""Host-local read-only declared census. No queue writes, GP, S3 or service control.

Run discover to obtain ONLY v1..v10 root/config names, then freeze an inventory.
collect verifies physical old-worker closure, private indexed copies/full original
verifiers, and canonical aliases. prepare combines pinned xb2+xb3 receipts into a
new immutable global census. All outputs are create-only in a NEW owned directory.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time

from native_global_guard import (GLOBAL_DEADLINE, BLOCKED, SUCCESS, canonical, digest,
    exact_file, hash64, need, queue_aliases)
from readonly_snapshot import source_path, verified_snapshot
from verified_cache import CachedNativeTools

H = Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')
ROOTS = [str(H / ('native_original_drain_20261009_v%d' % i)) for i in range(1, 11)]
CAP = 4 * 1024**2
UNIT = re.compile(r'jopedime-[a-zA-Z0-9_.-]{1,180}(?:\.service)?\Z')


def pin(path):
    p = source_path(path);need(p.stat().st_size<=CAP,'bounded artifact before read');raw=p.read_bytes()
    need(len(raw)<=CAP,'bounded artifact')
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def read(ref, cap=CAP):
    need(set(ref)=={'path','sha256','bytes'} and hash64(ref['sha256']) and
         type(ref['bytes'])is int and 0<ref['bytes']<=cap,'closed bounded exact ref')
    return json.loads(exact_file(ref))


def put(root,name,value):
    p=root/name;raw=canonical(value);need(len(raw)<=CAP,'bounded census artifact')
    with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(p)


def fresh_root(path):
    p=Path(path);need(p.is_absolute() and '..'not in p.parts and
      all(not x.is_symlink()for x in (p,*p.parents)),'owned absolute nonsymlink root')
    need(p.parent.is_dir(),'existing owned parent');p.mkdir(exist_ok=False);return p


def tick(deadline):
    need(time.time()<deadline<=GLOBAL_DEADLINE,'unchanged global/census deadline')


def run_read(argv,deadline):
    """Bound captured command output; fixed read-only argv assembled below."""
    tick(deadline)
    import tempfile
    with tempfile.TemporaryFile()as output:
        p=subprocess.Popen(argv,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll()is None:
                tick(deadline);need(output.tell()<=65536,'bounded physical read output');time.sleep(.02)
            need(output.tell()<=65536,'bounded physical read output')
            output.seek(0);raw=output.read(65537);need(len(raw)<=65536,'bounded physical read')
            # systemctl can return 4 for a never-loaded unit; LoadState is checked.
            need(p.returncode in (0,1,3,4),'physical read command failed');tick(deadline)
            return raw.decode()
        finally:
            import signal
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=2)


def closed_unit(unit,deadline,runner=run_read,cgroup_root=Path('/sys/fs/cgroup')):
    need(type(unit)is str and UNIT.fullmatch(unit),'exact declared legacy unit')
    name=unit if unit.endswith('.service')else unit+'.service'
    text=runner(['systemctl','--user','show',name,'--property=Id,LoadState,ActiveState,SubState,MainPID,ControlGroup'],deadline)
    v=dict(line.split('=',1)for line in text.splitlines()if '='in line)
    need(v.get('Id')==name and v.get('LoadState')in ('loaded','not-found') and
         v.get('ActiveState')in ('inactive','failed') and v.get('MainPID')=='0','legacy worker must be physically inactive/PID0')
    cg=v.get('ControlGroup','');groups=[]
    expected=Path('user.slice/user-1000.slice/user@1000.service/app.slice')/name
    candidates={expected}
    if cg:
        need(cg.startswith('/')and '..'not in Path(cg).parts,'safe declared cgroup');candidates.add(Path(cg.lstrip('/')))
    for rel in sorted(candidates):
        p=cgroup_root/rel
        need(all(not x.is_symlink()for x in (p,*p.parents)),'no cgroup symlink')
        if not p.exists():groups.append({'path':str(p),'exists':False});continue
        procs=(p/'cgroup.procs').read_text().split();events=dict(line.split()for line in (p/'cgroup.events').read_text().splitlines())
        need(not procs and events.get('populated')=='0','legacy descendants remain in cgroup')
        groups.append({'path':str(p),'exists':True,'pids':procs,'populated':0})
    return {'unit':name,'systemd':v,'cgroups':groups,'checked_at_unix':time.time()}


def discover(roots):
    """No recursive scan: direct config.json + worker0..7/config.json only."""
    rows=[]
    for name in roots:
        p=Path(name);need(str(p)in ROOTS,'bounded declared native root')
        need(all(not x.is_symlink()for x in (p,*p.parents)),'native root symlink')
        allowed=['config.json']+['worker%d/config.json'%i for i in range(8)]
        files=[n for n in allowed if (p/n).is_file()]
        if p.exists():
            # Reject an unexpected worker/config rather than silently omit it.
            actual={str(q.relative_to(p))for q in p.glob('worker*/config.json')}
            if (p/'config.json').is_file():actual.add('config.json')
            need(actual==set(files),'undeclared native worker config')
        rows.append({'path':str(p),'exists':p.exists(),'configs':files})
    return rows


def summarize_source(source,deadline,canonicalize):
    p=source_path(source['path']);before=p.stat()
    db,proof=verified_snapshot(p,deadline=deadline,kind=source['kind'],expected_config=source['config'])
    aliases=[];states={};recorded_native=0
    try:
        if source['kind']=='outbox':
            for row in db.execute('SELECT *FROM emissions'):
                state=row['state'];states[state]=states.get(state,0)+1
                body=json.loads(row['body']);e=body['emission']
                need(state in {'recorded'}|BLOCKED|SUCCESS,'known emission state')
                if state=='recorded':
                    recorded_native+=int(e['input_descriptor'].get('source_kind')=='native_s1_context');continue
                raw=e['raw_program_utf8'];need(hashlib.sha256(raw.encode()).hexdigest()==e['raw_program_sha256'],'raw emission hash')
                program=canonicalize(raw);outcome=json.loads(row['outcome'])if row['outcome']else{}
                aliases.append({'context_id':e['context_id'],'canonical_program_sha256':hashlib.sha256(program.encode()).hexdigest(),
                    'evaluation_key':outcome.get('evaluation_key'),'disposition':'completed_cache_only'if state in SUCCESS else'blocked','source':str(p)})
        else:
            for a in queue_aliases(db):
                for state in a['states']:states[state['state']]=states.get(state['state'],0)+1
                attempted=any(s['state']!='pending'or s['attempts']!=0 for s in a['states'])
                if not attempted:continue
                blocked=any(s['state']!='completed'and(s['state']!='pending'or s['attempts']!=0)for s in a['states'])
                program=canonicalize(a['program'])
                aliases.append({'context_id':a['context_id'],'canonical_program_sha256':hashlib.sha256(program.encode()).hexdigest(),
                    'evaluation_key':a['evaluation_key'],'disposition':'blocked'if blocked else'completed_cache_only','source':str(p)})
    finally:db.close()
    after=p.stat();need((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'physical source replaced during census')
    # Same canonical alias can occur many times; retain most restrictive disposition.
    unique={canonical(a):a for a in aliases}
    return {'source':source,'physical':{'device':before.st_dev,'inode':before.st_ino},'proof':proof,
            'states':states,'recorded_native':recorded_native,'aliases':list(unique.values())}


def collect(spec_ref,out,*,hostname=None,unit_reader=closed_unit,tools_factory=CachedNativeTools):
    s=read(spec_ref);need(set(s)=={'schema','host','native_roots','queues','legacy_units','canonicalizer','historical_census','extra_evidence','maximum_seconds'},'closed census inventory')
    need(s['schema']=='jope.native-v5-census-inventory/v1'and s['host']in ('xbabe2','xbabe3')and
         s['host']==(hostname or socket.gethostname())and type(s['maximum_seconds'])is int and 1<=s['maximum_seconds']<=600,'owned host/bounded census')
    deadline=min(time.time()+s['maximum_seconds'],GLOBAL_DEADLINE);tick(deadline)
    need(type(s['legacy_units'])is list and 1<=len(s['legacy_units'])<=160 and len(set(s['legacy_units']))==len(s['legacy_units']),'closed nonempty legacy unit inventory')
    roots=s['native_roots'];need([r['path']for r in roots]==(ROOTS if s['host']=='xbabe2'else []),'all v1..v10 explicitly declared on xb2 only')
    observed=discover([r['path']for r in roots]);need(observed==roots,'native held/present/absent/config inventory drift')
    hist=read(s['historical_census']);need(hist['schema']=='jope.variant-historical-census/v1'and len(hist['source_queues'])==18,'exact variant18 census')
    expected=sorted(q['path']for q in hist['source_queues']if q['host']==s['host'])
    need(s['queues']==expected,'all declared variant queues included, no silent dropped aliases')
    closure=[unit_reader(u,deadline)for u in s['legacy_units']]
    extra=[{'ref':r,'body_sha256':digest(read(r))}for r in s['extra_evidence']]
    sources={};configs=[]
    for r in roots:
        for relative in r['configs']:
            ref=pin(Path(r['path'])/relative);c=read(ref)
            need(type(c.get('outbox_config'))is dict,'native historical config requires exact outbox authority')
            desc={'path':c['outbox_path'],'kind':'outbox','config':c['outbox_config']}
            need(desc['path']not in sources or sources[desc['path']]==desc,'historical outbox authority conflict')
            sources[desc['path']]=desc;configs.append(ref)
    for p in expected:
        need(p not in sources,'source type collision');sources[p]={'path':p,'kind':'queue','config':None}
    need(sources and len(sources)<=128,'complete bounded source list')
    raw=exact_file(s['canonicalizer']);need(raw,'pinned actual canonicalizer')
    native=tools_factory(s['canonicalizer'],deadline)
    root=fresh_root(out);snapshots=[]
    for n,source in enumerate(sorted(sources.values(),key=lambda x:x['path'])):
        tick(deadline);summary=summarize_source(source,deadline,native.canonical_program)
        if source['kind']=='queue':
            prior=next(q for q in hist['source_queues']if q['host']==s['host']and q['path']==source['path'])
            observed_keys={a['evaluation_key']for a in summary['aliases']}
            blocked_keys={a['evaluation_key']for a in summary['aliases']if a['disposition']=='blocked'}
            need(set(prior['completed_evaluation_keys'])<=observed_keys and
                 set(prior['blocked_evaluation_keys'])<=blocked_keys,'historical completed/unknown keys retained by actual queue')
        snapshots.append(put(root,'source-%03d.json'%n,summary))
    # Close physical observation after all snapshots; never infer from elapsed leases.
    closure_after=[unit_reader(u,deadline)for u in s['legacy_units']];tick(deadline)
    result={'schema':'jope.native-v5-host-census/v1','host':s['host'],'checked_at_unix':time.time(),
      'inventory':spec_ref,'canonicalizer_sha256':s['canonicalizer']['sha256'],'root_inventory':observed,
      'config_refs':configs,'sources':snapshots,'units_before':closure,'units_after':closure_after,
      'historical_census':s['historical_census'],'extra_evidence':extra,'read_only':True,'no_execution_or_retries':True}
    return put(root,'collection.json',result)


def validate_collection(c):
    """Revalidate exact producer receipt joins; a pinned partial list is not complete."""
    fields={'schema','host','checked_at_unix','inventory','canonicalizer_sha256','root_inventory',
            'config_refs','sources','units_before','units_after','historical_census','extra_evidence','read_only','no_execution_or_retries'}
    need(type(c)is dict and set(c)==fields and c['schema']=='jope.native-v5-host-census/v1' and
         c['host']in ('xbabe2','xbabe3') and c['read_only']is True and c['no_execution_or_retries']is True,'closed complete host collection')
    inv=read(c['inventory'])
    need(set(inv)=={'schema','host','native_roots','queues','legacy_units','canonicalizer','historical_census','extra_evidence','maximum_seconds'} and
         inv['schema']=='jope.native-v5-census-inventory/v1'and inv['host']==c['host'],'exact host inventory authority')
    need(c['canonicalizer_sha256']==inv['canonicalizer']['sha256']and hash64(c['canonicalizer_sha256']),'inventory canonicalizer binding')
    need(c['historical_census']==inv['historical_census'],'same historical census reference')
    hist=read(c['historical_census'])
    need(hist['schema']=='jope.variant-historical-census/v1'and len(hist['source_queues'])==18,'complete variant18 source roster')
    queues=sorted(q['path']for q in hist['source_queues']if q['host']==c['host'])
    need(inv['queues']==queues and len(set(queues))==len(queues),'exact per-host queue union')
    roots=inv['native_roots'];need(c['root_inventory']==roots and [r['path']for r in roots]==(ROOTS if c['host']=='xbabe2'else []),'all ten root inventory records bound')
    expectedconfigs=[]
    for r in roots:
        need(set(r)=={'path','exists','configs'}and type(r['exists'])is bool and type(r['configs'])is list and
             (r['exists']or not r['configs']),'explicit root disposition')
        allowed=['config.json']+['worker%d/config.json'%i for i in range(8)]
        need(len(set(r['configs']))==len(r['configs'])and set(r['configs'])<=set(allowed),'bounded config paths')
        expectedconfigs.extend(str(Path(r['path'])/n)for n in r['configs'])
    need([r['path']for r in c['config_refs']]==expectedconfigs,'all declared native configurations retained')
    expected={q:{'path':q,'kind':'queue','config':None}for q in queues}
    for ref in c['config_refs']:
        cfg=read(ref);d={'path':cfg['outbox_path'],'kind':'outbox','config':cfg['outbox_config']}
        need(d['path']not in expected or expected[d['path']]==d,'native source authority consistent');expected[d['path']]=d
    need(type(inv['legacy_units'])is list and inv['legacy_units'] and len(set(inv['legacy_units']))==len(inv['legacy_units']),'explicit nonempty unique writer units')
    units=[u if u.endswith('.service')else u+'.service'for u in inv['legacy_units']]
    for stage in ('units_before','units_after'):
        observations=c[stage];need(type(observations)is list and [o.get('unit')for o in observations]==units,'all physical unit observations retained')
        for o in observations:
            need(set(o)=={'unit','systemd','cgroups','checked_at_unix'},'full physical closure observation')
            v=o['systemd'];need(v.get('Id')==o['unit']and v.get('LoadState')in ('loaded','not-found')and
              v.get('ActiveState')in ('inactive','failed')and v.get('MainPID')=='0','actual inactive/PID0 proof')
            need(type(o['cgroups'])is list and 1<=len(o['cgroups'])<=2,'physical cgroup observations required')
            for g in o['cgroups']:
                need(type(g)is dict and type(g.get('exists'))is bool and type(g.get('path'))is str and Path(g['path']).is_absolute(),'closed cgroup path/existence')
                need((g['exists']is False and set(g)=={'path','exists'})or
                  (g['exists']is True and set(g)=={'path','exists','pids','populated'}and g['pids']==[]and g['populated']==0),'empty physical descendants')
    need(c['extra_evidence']==[{'ref':r,'body_sha256':digest(read(r))}for r in inv['extra_evidence']],'all additional historical pins bound')
    summaries=[read(ref)for ref in c['sources']]
    need(len(summaries)==len(expected)and len({v['source']['path']for v in summaries})==len(summaries)and
         {v['source']['path']:v['source']for v in summaries}==expected,'complete exact per-host source set')
    for v in summaries:
        need(set(v)=={'source','physical','proof','states','recorded_native','aliases'} and set(v['physical'])=={'device','inode'}and
             all(type(x)is int and x>=0 for x in v['physical'].values())and type(v['recorded_native'])is int and v['recorded_native']>=0,'closed source summary and inode proof')
        need(type(v['proof'])is dict and type(v['proof'].get('events'))is int and hash64(v['proof'].get('head_sha256')),'full verifier head required')
        if v['source']['kind']=='outbox':
            need(v['proof'].get('schema')=='jope.flywheel-neural-outbox-status/v1'and v['proof'].get('states')==v['states'] and
                 v['proof'].get('observations')==sum(v['states'].values())and v['recorded_native']<=v['states'].get('recorded',0),'outbox verifier count/state closure')
        for a in v['aliases']:
            need(set(a)=={'context_id','canonical_program_sha256','evaluation_key','disposition','source'}and a['source']==v['source']['path']and
              hash64(a['context_id'])and hash64(a['canonical_program_sha256'])and a['disposition']in ('blocked','completed_cache_only')and
              (hash64(a['evaluation_key'])or(a['evaluation_key']is None and a['disposition']=='blocked'and c['host']=='xbabe2'and v['source']['kind']=='outbox')),'closed alias custody')
        if v['source']['kind']=='queue':
            prior=next(q for q in hist['source_queues']if q['host']==c['host']and q['path']==v['source']['path'])
            need(set(prior['completed_evaluation_keys'])<={a['evaluation_key']for a in v['aliases']}and
              set(prior['blocked_evaluation_keys'])<={a['evaluation_key']for a in v['aliases']if a['disposition']=='blocked'},'prior known aliases retained after transfer')
    return summaries


def prepare(collection_refs,out,*,original_outbox,continuation_outbox):
    """Merge exact host-local proof; remote source queues become frozen aliases.

    No source DB/registry is opened writable here. Unknown local native keys with
    no evaluation_key remain dynamically guarded by their original outbox; the
    census does not fabricate an evaluation key for them.
    """
    need(len(collection_refs)==2,'both host physical collections required');collections=[read(r)for r in collection_refs]
    need({c['host']for c in collections}=={'xbabe2','xbabe3'},'exact hosts')
    for c in collections:validate_collection(c)
    canons={c['canonicalizer_sha256']for c in collections};need(len(canons)==1,'same canonicalizer')
    histpins={c['historical_census']['sha256']for c in collections};need(len(histpins)==1,'same variant18 inventory')
    root=fresh_root(out);sources=[];entries=[];closure_refs=[];legacy_units=[];outboxes={};unknown=[];physical={}
    for c in collections:
        need(c['schema']=='jope.native-v5-host-census/v1'and c['read_only']is True and c['no_execution_or_retries']is True,'qualified collection schema')
        need(0<=time.time()-c['checked_at_unix']<=600,'fresh host collection')
        need([u['unit']for u in c['units_before']]==[u['unit']for u in c['units_after']],'unit set unchanged')
        legacy_units += [{'host':c['host'],'unit':u['unit']}for u in c['units_after']]
        # Preserve complete original collection including physical unit proof as exact bytes.
        closure_refs.append(put(root,c['host']+'-collection.json',c))
        for i,ref in enumerate(c['sources']):
            v=read(ref);src=v['source']
            copied=put(root,c['host']+'-source-%03d.json'%i,v);closure_refs.append(copied)
            if c['host']=='xbabe2':
                sources.append(src);physical[src['path']]=v['physical']
            if src['kind']=='outbox':outboxes[src['path']]=v
            for a in v['aliases']:
                need(hash64(a['context_id'])and hash64(a['canonical_program_sha256']),'actual canonical identity')
                if not hash64(a['evaluation_key']):
                    need(c['host']=='xbabe2'and src['kind']=='outbox'and a['disposition']=='blocked','unknown remote key cannot be silently omitted')
                    unknown.append(a);continue
                entries.append({**a,'source':src['path']if c['host']=='xbabe2'else copied['path']})
    need(original_outbox in outboxes and continuation_outbox in outboxes and original_outbox!=continuation_outbox,'both native outboxes declared')
    need(outboxes[original_outbox]['recorded_native']==0,'original native queue not empty; no unreviewed original workers')
    need(len({s['path']for s in sources})==len(sources),'unique local source paths');sources.sort(key=lambda s:s['path'])
    scope=digest({'schema':'jope.native-v5-permanent-scope/v1','sources':sources,'canonicalizer_sha256':next(iter(canons)),
                  'host_collections':collection_refs,'global_deadline_unix':GLOBAL_DEADLINE})
    census={'schema':'jope.native-global-census/v1','scope_sha256':scope,'canonicalizer_sha256':next(iter(canons)),
      'source_paths':[s['path']for s in sources],'source_identities':physical,'coverage_complete':True,'all_legacy_writers_stopped':True,
      'closure_evidence':closure_refs,'entries':entries}
    census_ref=put(root,'census.json',census)
    put(root,'unknown-native-keys.json',{'schema':'jope.native-v5-unknown-preserved/v1','aliases':unknown,'disposition':'dynamic original outbox guard; never retry'})
    prepared={'schema':'jope.native-v5-prepared/v1','sources':sources,'census':census_ref,'scope_sha256':scope,
      'canonicalizer_sha256':next(iter(canons)),'original_outbox_empty':True,
      'original_outbox':{'path':original_outbox,'config':outboxes[original_outbox]['source']['config']},
      'continuation_outbox':{'path':continuation_outbox,'config':outboxes[continuation_outbox]['source']['config']},
      'legacy_closure_refs':closure_refs,'legacy_units':legacy_units}
    return put(root,'prepared.json',prepared)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['discover','collect','prepare'])
    p.add_argument('--inventory');p.add_argument('--inventory-sha');p.add_argument('--out');p.add_argument('--collection',action='append',default=[])
    p.add_argument('--original-outbox');p.add_argument('--continuation-outbox');a=p.parse_args()
    if a.mode=='discover':print(json.dumps(discover(ROOTS),indent=2));return
    if a.mode=='collect':
        ref=pin(a.inventory);need(ref['sha256']==a.inventory_sha,'inventory expectedSHA');result=collect(ref,a.out)
    else:
        # Each argument is PATH#SHA; no guessing mutable latest names.
        refs=[]
        for value in a.collection:
            path,want=value.rsplit('#',1);ref=pin(path);need(ref['sha256']==want,'collection expectedSHA');refs.append(ref)
        result=prepare(refs,a.out,original_outbox=a.original_outbox,continuation_outbox=a.continuation_outbox)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
