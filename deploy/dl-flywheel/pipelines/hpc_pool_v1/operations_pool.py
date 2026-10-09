"""Create-only additional native pool; parent workers and registry stay unchanged.

API: dry_plan(manifest_ref) validates only; stage(manifest_ref) returns stage.json
exactref; freeze(stage_ref) returns plan.json exactref; launch(plan_ref) performs
live closure/resource gates, bounded systemd preflights, then one native launch
attempt per worker. Never retry an interrupted stage/freeze/launch automatically.

Manifest (closed): schema=jope.native-v5-pool-input/v1, prepared exactref,
worker_template exactref, source_files=[{relative_path,ref}], root (new absolute
directory), registry_path (one permanent shared absolute file), python exactref,
unit_prefix, worker_count (8, 12 or 24), parent_manifest/parent_plan exactrefs,
registry_identity (existing inode/config hash), and other_allocations. Exactrefs
always have path, sha256, bytes. The template is exact parent worker0; input/build pins
are preserved and staged source pins are added. Dry/stage make no subprocess
calls. Freeze performs local resource readback before fixing its new clock.
Launch performs Python3.12 and at most four simultaneous CPU-only preflights.
SSH is restricted to legacy unit
readback on xbabe3. Run this helper on xbabe2 for the actual launch.

CLI: --input PATH --input-sha SHA [--stage | --freeze | --launch]. Default is dry
validation of a launch manifest; other modes expect manifest/stage/plan in order.
Full 1800 seconds must remain before the original global deadline at freeze.
"""
from pathlib import Path, PurePosixPath
from native_build import verify_native_build
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import sqlite3
import subprocess
import time

GLOBAL_DEADLINE = 1791592146.380285
CAP = 64 * 1024**2
OUTPUT_CAP = 64 * 1024
MANIFEST_FIELDS = {'schema','prepared','worker_template','source_files','root',
                   'registry_path','python','unit_prefix','worker_count','parent_plan','parent_manifest','other_allocations','registry_identity'}
PREPARED_FIELDS = {'schema','sources','census','scope_sha256','canonicalizer_sha256',
                   'original_outbox_empty','original_outbox','continuation_outbox','legacy_closure_refs','legacy_units'}
PACKET_FIELDS = {'schema','created_unix','deadline_unix','global_deadline_unix','global_guard',
 'input_lists','maximum_emissions','native_build_receipt','native_canonicalizer','out',
 'outbox_config','outbox_path','snapshot','source_commit','source_build_plan_sha256','source_pins',
 'threads','training_config','worker','emitter','rustc','worker_id'}
DERIVED = {'schema','created_unix','deadline_unix','global_deadline_unix','global_guard',
           'maximum_emissions','out','outbox_config','outbox_path','threads','worker_id'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hash64(value):
    return type(value) is str and bool(re.fullmatch('[0-9a-f]{64}',value))


def absolute(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and str(p)==path and
         all(not x.is_symlink() for x in (p,*p.parents)), 'exact absolute nonsymlink path')
    return p


def read(ref, cap=CAP):
    need(type(ref) is dict and set(ref)=={'path','sha256','bytes'} and hash64(ref['sha256']) and
         type(ref['bytes']) is int and 0 < ref['bytes'] <= cap, 'closed bounded exactref')
    p=absolute(ref['path']); before=p.stat()
    need(p.is_file() and before.st_size==ref['bytes'], 'exact source size')
    with p.open('rb') as stream: raw=stream.read(ref['bytes']+1)
    after=p.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and
         len(raw)==ref['bytes'] and sha(raw)==ref['sha256'], 'unchanged exact source bytes')
    return raw


def reference(path):
    p=absolute(str(path)); raw=p.read_bytes()
    need(0<len(raw)<=CAP,'bounded output reference')
    return {'path':str(p),'sha256':sha(raw),'bytes':len(raw)}


def document(ref, fields, schema):
    value=json.loads(read(ref))
    need(type(value) is dict and set(value)==fields and value['schema']==schema,'closed '+schema)
    return value


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def create(path, raw):
    p=absolute(str(path)); need(p.parent.is_dir(),'owned parent exists')
    fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    sync_dir(p.parent)
    return reference(p)


def create_json(path, value):
    return create(path,canonical(value))


PERMANENT_REGISTRY = '/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-native-global-20261009-v1/registry.sqlite'
GIB = 1024**3


def parent_guard(m, p):
    return {'schema':'jope.native-global-execution-guard/v1','registry_path':m['registry_path'],
        'scope_sha256':p['scope_sha256'],'canonicalizer_sha256':p['canonicalizer_sha256'],
        'global_deadline_unix':GLOBAL_DEADLINE,'sources':p['sources'],'census':p['census'],
        'max_snapshot_bytes':256*1024**2}


def validate_registry(m, guard):
    """Identity read only; unchanged worker performs full chain verification."""
    path=absolute(m['registry_path']);pin=m['registry_identity']
    need(path.is_file() and set(pin)=={'device','inode','config_sha256'} and
         type(pin['device']) is int and type(pin['inode']) is int and hash64(pin['config_sha256']),
         'existing permanent registry physical pin')
    before=path.stat()
    need(before.st_nlink==1 and (before.st_dev,before.st_ino)==(pin['device'],pin['inode']),
         'same permanent registry inode; never create or replace')
    for suffix in ('-wal','-shm'):
        side=Path(str(path)+suffix)
        need(not side.is_symlink() and (not side.exists() or side.stat().st_nlink==1), 'registry sidecar custody')
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,isolation_level=None,timeout=2)
    try:
        until=time.monotonic()+10;db.set_progress_handler(lambda:int(time.monotonic()>=until),1000)
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        rows=db.execute('SELECT body FROM config').fetchall()
        need(len(rows)==1 and sha(bytes(rows[0][0]))==pin['config_sha256'],'exact existing registry config bytes')
        census=json.loads(read(guard['census'],16*1024**2));physical={}
        for source in guard['sources']:
            s=absolute(source['path']).stat();physical[source['path']]={'device':s.st_dev,'inode':s.st_ino}
        expected={'schema':guard['schema'],'scope_sha256':guard['scope_sha256'],
            'canonicalizer_sha256':guard['canonicalizer_sha256'],'sources':guard['sources'],
            'physical_sources':physical,'census_sha256':guard['census']['sha256'],
            'global_deadline_unix':GLOBAL_DEADLINE}
        need(census['source_identities']==physical and bytes(rows[0][0])==canonical(expected),
             'same permanent scope/census and physical historical sources')
        need({r[0]for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}==
             {'config','claims','events'},'exact existing registry tables')
        return {'path':str(path),'device':before.st_dev,'inode':before.st_ino,
                'config_sha256':pin['config_sha256'],'read_only':True}
    finally:
        db.close();after=path.stat()
        need((after.st_dev,after.st_ino)==(before.st_dev,before.st_ino),'registry replaced during read')


def validate_parent(m, p, template):
    need(type(m['worker_count']) is int and m['worker_count'] in (8,12,24),'explicit 8, 12 or 24 added workers')
    need(m['registry_path']==PERMANENT_REGISTRY,'same exact permanent registry path')
    parent=json.loads(read(m['parent_manifest']))
    need(parent['schema']=='jope.native-v5-launch-input/v1' and parent['registry_path']==m['registry_path'] and
         parent['prepared']==m['prepared'] and parent['python']==m['python'] and parent['root']!=m['root'],
         'exact parent manifest/scope/Python; new output root')
    old=json.loads(read(m['parent_plan']))
    need(old['schema']=='jope.native-v5-launch-plan/v1' and len(old['workers'])==4 and
         old['deadline_unix']-old['created_unix']==1800 and old['deadline_unix']<=GLOBAL_DEADLINE,
         'unchanged four-worker parent plan')
    stage=json.loads(read(old['stage']))
    need(stage['manifest']==m['parent_manifest'] and old['stage']['path']==str(Path(parent['root'])/'stage.json') and
         m['parent_plan']['path']==str(Path(parent['root'])/'plan.json'),'parent plan/stage manifest closure')
    original={x['relative_path']:(x['ref']['sha256'],x['ref']['bytes'])for x in parent['source_files']}
    current={x['relative_path']:(x['ref']['sha256'],x['ref']['bytes'])for x in m['source_files']}
    need(original==current and len(original)==len(parent['source_files'])==len(m['source_files'])==34,
         'all 34 parent worker sources byte unchanged')
    guard=parent_guard(m,p);units=[]
    for i,w in enumerate(old['workers']):
        need(set(w)=={'index','unit','config'} and w['index']==i and w['unit']==parent['unit_prefix']+'-worker'+str(i) and
             w['config']['path']==str(Path(parent['root'])/('worker'+str(i))/'config.json'),'exact parent worker identity')
        c=document(w['config'],PACKET_FIELDS,'jope.native-original-successor/v5')
        need(c['global_guard']==guard and c['outbox_path']==p['continuation_outbox']['path'] and
             c['outbox_config']==p['continuation_outbox']['config'] and c['worker_id']==w['unit'] and
             c['out']==str(Path(w['config']['path']).parent) and c['threads']==1 and c['maximum_emissions']==128 and
             c['created_unix']==old['created_unix'] and c['deadline_unix']==old['deadline_unix'] and
             c['global_deadline_unix']==GLOBAL_DEADLINE,'same parent global guard and immutable bounds')
        need(all(c[k]==template[k]for k in PACKET_FIELDS-DERIVED-{'source_pins'}), 'same parent scientific/input identity')
        for ref in c['source_pins']:read(ref)
        units.append(w['unit'])
    need(m['worker_template']==old['workers'][0]['config'],'template must be exact active parent worker0')
    need(m['unit_prefix']!=parent['unit_prefix'] and not any(u['unit'].removesuffix('.service')in units for u in p['legacy_units']),
         'parent workers remain separate from stopped legacy writers')
    allocations=m['other_allocations']
    need(type(allocations)is list and 1<=len(allocations)<=16,'declared current owned allocations including learner')
    names=[]
    for a in allocations:
        need(set(a)=={'unit','kind','memory_cap_bytes','cpu_quota_percent','expected_argv'} and
             a['kind']in('learner','other_owned') and type(a['unit'])is str and
             re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.@:-]{0,159}',a['unit']) and
             type(a['memory_cap_bytes'])is int and 32*1024**2<=a['memory_cap_bytes']<=32*GIB and
             type(a['cpu_quota_percent'])is int and 1<=a['cpu_quota_percent']<=3200 and
             type(a['expected_argv'])is list and 2<=len(a['expected_argv'])<=64 and
             all(type(x)is str and 0<len(x)<=4096 for x in a['expected_argv']), 'closed owned resource allocation')
        names.append(a['unit'].removesuffix('.service'))
    need(len(names)==len(set(names)) and not set(names)&set(units) and
         sum(a['kind']=='learner'for a in allocations)==1 and
         all(a['memory_cap_bytes']>=8*GIB and a['cpu_quota_percent']==800 for a in allocations if a['kind']=='learner'),
         'one protected learner allocation and no duplicate parent reservation')
    new_names={m['unit_prefix']+'-worker'+str(i) for i in range(m['worker_count'])}
    need(not new_names&(set(names)|set(units)) and
         not new_names&{u['unit'].removesuffix('.service')for u in p['legacy_units']}, 'fresh worker unit names')
    validate_registry(m,guard)


PYTHON_PROBE = '''import hashlib,json,sys
from pathlib import Path
p=Path(sys.executable).resolve();b=p.read_bytes()
print(json.dumps({'schema':'jope.native-pool-python/v1','version':list(sys.version_info[:3]),
 'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}))
'''


def python_probe(ref, runner):
    read(ref)
    proof=json.loads(run_checked(runner,[ref['path'],'-I','-c',PYTHON_PROBE],30)['output'])
    need(type(proof)is dict and set(proof)=={'schema','version','path','sha256','bytes'} and
         proof['schema']=='jope.native-pool-python/v1' and type(proof['version'])is list and
         len(proof['version'])==3 and all(type(x)is int for x in proof['version']) and
         proof['version'][:2]==[3,12] and all(proof[k]==ref[k]for k in ('path','sha256','bytes')),
         'actual pinned Python 3.12 required before any native preflight or launch')
    return proof


ALLOCATION_PROBE = '''import json,os,subprocess,sys
from pathlib import Path
unit=sys.argv[1];name=unit if unit.endswith('.service')else unit+'.service'
p=subprocess.run(['systemctl','--user','show',name,'--property=Id,LoadState,ActiveState,SubState,MainPID,ControlPID,ControlGroup'],capture_output=True,timeout=5)
assert p.returncode in(0,3,4)and len(p.stdout)+len(p.stderr)<=65536
f=dict(x.split('=',1)for x in p.stdout.decode().splitlines()if'='in x);assert f.get('Id')==name
active=f.get('ActiveState')=='active';result={'unit':name,'active':active,'systemd':f}
if active:
 assert f.get('SubState')=='running'and int(f['MainPID'])>0 and f.get('ControlPID')=='0'
 g=f['ControlGroup'];assert g.startswith('/')and'..'not in Path(g).parts
 root=Path('/sys/fs/cgroup')/g.lstrip('/');assert all(not x.is_symlink()for x in(root,*root.parents))
 procs=(root/'cgroup.procs').read_text();assert len(procs)<65536 and f['MainPID']in procs.split()
 current=int((root/'memory.current').read_text());cap=int((root/'memory.max').read_text())
 quota,period=(root/'cpu.max').read_text().split();assert quota!='max'
 args=(Path('/proc')/f['MainPID']/'cmdline').read_bytes();assert len(args)<=65536
 result.update(memory_current_bytes=current,memory_cap_bytes=cap,cpu_quota=int(quota),cpu_period=int(period),
  swap_cap_bytes=int((root/'memory.swap.max').read_text()),argv=[x.decode()for x in args.split(b'\\0')if x])
else:
 assert f.get('ActiveState')in('inactive','failed')and f.get('SubState')in('dead','failed')and f.get('MainPID')=='0'and f.get('ControlPID')=='0'
 # A quiet main PID alone does not prove descendants are gone.
 uid=os.getuid();roots={Path('/sys/fs/cgroup/user.slice')/('user-'+str(uid)+'.slice')/('user@'+str(uid)+'.service')/'app.slice'/name}
 if f.get('ControlGroup'):
  g=f['ControlGroup'];assert g.startswith('/')and'..'not in Path(g).parts;roots.add(Path('/sys/fs/cgroup')/g.lstrip('/'))
 for r in roots:
  assert all(not x.is_symlink()for x in(r,*r.parents))
  if r.exists():assert not(r/'cgroup.procs').read_text().split()and dict(x.split()for x in(r/'cgroup.events').read_text().splitlines()).get('populated')=='0'
print(json.dumps(result))
'''


def allocations(m):
    old=json.loads(read(m['parent_plan']));rows=[]
    for w in old['workers']:
        r=w['config'];root=str(Path(r['path']).parent)
        rows.append({'unit':w['unit'],'kind':'native_parent','memory_cap_bytes':GIB,'cpu_quota_percent':100,
            'expected_argv':[m['python']['path'],'-B',root+'/native_dispatch.py','--config',r['path'],'--config-sha',r['sha256']]})
    return rows+m['other_allocations']


def resource_observation(m, runner):
    need(os.uname().nodename=='xbabe2','pool on xbabe2 only')
    rows=[];residual=0;cpu_reserved=0
    for a in allocations(m):
        proof=json.loads(run_checked(runner,['/usr/bin/python3','-I','-c',ALLOCATION_PROBE,a['unit']],15)['output'])
        need(proof['unit']==a['unit'].removesuffix('.service')+'.service' and type(proof['active'])is bool,'exact owned unit probe')
        if proof['active']:
            need(proof['memory_cap_bytes']==a['memory_cap_bytes'] and proof['swap_cap_bytes']==0 and
                 type(proof['memory_current_bytes'])is int and proof['memory_current_bytes']>=0 and
                 proof['cpu_period']>0 and proof['cpu_quota']*100==a['cpu_quota_percent']*proof['cpu_period'] and
                 proof['argv']==a['expected_argv'],'exact active owned command/cgroup bounds')
            # Full-cap reservation is deliberately conservative: /proc/meminfo
            # and cgroup memory.current are not an atomic cross-process sample.
            # Subtracting a prior current value can double-credit released RAM.
            left=a['memory_cap_bytes'];cpu_reserved+=a['cpu_quota_percent']
        else:
            # A temporarily stopped learner retains its full reservation.
            left=a['memory_cap_bytes'] if a['kind']!='native_parent' else 0
            if a['kind']!='native_parent':cpu_reserved+=a['cpu_quota_percent']
        residual+=left;rows.append({'allocation':a,'physical':proof,'reserved_cap_bytes':left})
    memory={line.split(':',1)[0]:line.split(':',1)[1].strip()for line in Path('/proc/meminfo').read_text().splitlines()}
    available=int(memory['MemAvailable'].split()[0])*1024;disk=shutil.disk_usage(m['root']).free
    required=24*GIB+residual+m['worker_count']*GIB
    cpus=os.cpu_count()or 0
    need(available>=required and disk>=40*GIB and cpus*100>=cpu_reserved+m['worker_count']*100,
         'pool resource hold: foreign24GiB + owned full caps + added1GiB each; disk40GiB and CPU budget')
    return {'host':'xbabe2','available_memory_bytes':available,'required_available_bytes':required,
        'foreign_reserve_bytes':24*GIB,'owned_full_cap_reserve_bytes':residual,'new_workers':m['worker_count'],
        'free_disk_bytes':disk,'logical_cpus':cpus,'owned_units':rows,'observed_at_unix':time.time()}


def fresh_resources(m, runner, resources, now):
    proof=resources(m,runner)
    need(type(proof)is dict and type(proof.get('observed_at_unix'))in(int,float) and
         math.isfinite(proof['observed_at_unix']) and 0<=now()-proof['observed_at_unix']<=5,
         'resource observation must be current within five seconds')
    return proof


def inputs(manifest_ref):
    m=document(manifest_ref,MANIFEST_FIELDS,'jope.native-v5-pool-input/v1')
    root=absolute(m['root']); registry=absolute(m['registry_path'])
    need(root.parent.is_dir() and registry.parent.is_dir() and registry!=root and
         not registry.is_relative_to(root), 'permanent registry outside new packet root')
    need(type(m['unit_prefix']) is str and bool(re.fullmatch('jope-native-v5-[a-z0-9-]{1,60}',m['unit_prefix'])),
         'bounded fresh native unit prefix')
    p=document(m['prepared'],PREPARED_FIELDS,'jope.native-v5-prepared/v1')
    need(p['original_outbox_empty'] is True and hash64(p['scope_sha256']) and
         hash64(p['canonicalizer_sha256']), 'prepared original empty and scope identity')
    need(type(p['sources']) is list and 1<=len(p['sources'])<=128,'bounded prepared sources')
    for source in p['sources']:
        need(set(source)=={'path','kind','config'} and source['kind'] in ('outbox','queue'), 'closed guard source')
        absolute(source['path'])
        need(type(source['config']) is dict if source['kind']=='outbox' else source['config'] is None,
             'source config kind')
    paths=[s['path'] for s in p['sources']]
    need(paths==sorted(set(paths)) and str(registry) not in paths,'sorted unique sources / separate registry')
    continuation=p['continuation_outbox']
    need(set(continuation)=={'path','config'} and
         {'path':continuation['path'],'kind':'outbox','config':continuation['config']} in p['sources'],
         'exact declared continuation outbox')
    original=p['original_outbox']
    need(set(original)=={'path','config'} and original['path']!=continuation['path'] and
         {'path':original['path'],'kind':'outbox','config':original['config']} in p['sources'],
         'exact separate declared original outbox')
    read(p['census']);need(type(p['legacy_closure_refs']) is list and 1<=len(p['legacy_closure_refs'])<=256,'closure pins')
    for ref in p['legacy_closure_refs']:read(ref)
    need(type(p['legacy_units']) is list and 1<=len(p['legacy_units'])<=128,'bounded legacy units')
    for unit in p['legacy_units']:
        need(set(unit)=={'host','unit'} and unit['host'] in ('xbabe2','xbabe3') and
             type(unit['unit']) is str and bool(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.@:-]{0,159}',unit['unit'])), 'exact legacy host/unit')
    template=json.loads(read(m['worker_template']))
    need(type(template) is dict and set(template) in (PACKET_FIELDS,PACKET_FIELDS-{'global_guard'}),
         'closed prior worker template with build plan')
    for key in ('source_build_plan_sha256',):need(hash64(template[key]),'exact source build plan')
    for key in ('native_build_receipt','native_canonicalizer','snapshot','training_config','worker','emitter','rustc'):
        read(template[key])
    for key in ('input_lists','source_pins'):
        need(type(template[key]) is list and 1<=len(template[key])<=256,'bounded required input/source pins')
        for ref in template[key]:read(ref)
    need(template['native_canonicalizer']['sha256']==p['canonicalizer_sha256'],'same prepared canonicalizer')
    build=json.loads(read(template['native_build_receipt']))
    verify_native_build(template,build,read)
    read(m['python']);need(os.access(m['python']['path'],os.X_OK),'pinned executable Python')
    files=m['source_files'];need(type(files) is list and 1<=len(files)<=128,'bounded staged file set')
    names=[];total=0
    for item in files:
        need(set(item)=={'relative_path','ref'},'closed staged source')
        name=item['relative_path']; rel=PurePosixPath(name)
        need(type(name) is str and not rel.is_absolute() and str(rel)==name and '..' not in rel.parts and
             name.endswith('.py') and all(re.fullmatch('[A-Za-z0-9_.-]+',part) for part in rel.parts),'safe Python source path')
        total+=len(read(item['ref']));names.append(name)
    need(total<=CAP and names==sorted(set(names)) and
         {'native_dispatch.py','native_build.py','native_global_guard.py','readonly_snapshot.py','neural_outbox.py',
          'native_evaluate.py','native_inputs.py','native_failure.py','verified_cache.py',
          'native_corpus_authority.py','jope_flywheel/__init__.py'}<=set(names),'complete bounded staged source names')
    validate_parent(m,p,template)
    return m,p,template


def dry_plan(manifest_ref):
    m,p,_=inputs(manifest_ref)
    return {'schema':'jope.native-v5-pool-dry/v1','manifest':manifest_ref,'root':m['root'],
            'workers':m['worker_count'],'continuation_outbox':p['continuation_outbox'],'registry_path':m['registry_path'],
            'runtime_seconds':1800,'global_deadline_unix':GLOBAL_DEADLINE,'external_calls':0}


def stage(manifest_ref):
    m,_,_=inputs(manifest_ref);root=absolute(m['root']);root.mkdir(mode=0o700);sync_dir(root.parent)
    create_json(root/'stage-intent.json',{'schema':'jope.native-v5-stage-intent/v1','manifest':manifest_ref})
    workers=[]
    for i in range(m['worker_count']):
        directory=root/('worker'+str(i));directory.mkdir(mode=0o700)
        refs=[]
        for item in m['source_files']:
            target=directory/item['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
            refs.append(create(target,read(item['ref'])))
        workers.append({'index':i,'root':str(directory),'source_pins':refs})
    return create_json(root/'stage.json',{'schema':'jope.native-v5-pool-staged/v1','manifest':manifest_ref,'workers':workers})


def staged(stage_ref):
    s=document(stage_ref,{'schema','manifest','workers'},'jope.native-v5-pool-staged/v1')
    m,p,t=inputs(s['manifest']);need(len(s['workers'])==m['worker_count'],'exact staged worker count')
    for i,w in enumerate(s['workers']):
        need(set(w)=={'index','root','source_pins'} and w['index']==i and w['root']==str(Path(m['root'])/('worker'+str(i))),
             'exact staged worker root')
        expected=[{'path':str(Path(w['root'])/x['relative_path']),'sha256':x['ref']['sha256'],'bytes':x['ref']['bytes']} for x in m['source_files']]
        need(w['source_pins']==expected,'exact staged source inventory')
        for ref in expected:read(ref)
    need(stage_ref['path']==str(Path(m['root'])/'stage.json'),'stage receipt root')
    return s,m,p,t


def original_empty(prepared):
    """Read only the explicit original outbox; resolved history remains intact."""
    source=prepared['original_outbox'];p=absolute(source['path'])
    db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,isolation_level=None,timeout=2)
    try:
        end=time.monotonic()+10
        db.set_progress_handler(lambda:int(time.monotonic()>=end),1000)
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        need(json.loads(db.execute('SELECT body FROM config').fetchone()[0])==source['config'],'exact live original config')
        count=db.execute("SELECT count(*) FROM emissions WHERE state='recorded' AND json_extract(body,'$.emission.input_descriptor.source_kind')='native_s1_context'").fetchone()[0]
        need(count==0,'original native recorded queue no longer empty')
        return {'path':source['path'],'recorded_native':count}
    finally:db.close()


def freeze(stage_ref, *, now=time.time, resources=None):
    s,m,p,t=staged(stage_ref)
    gate={'resources':(resources or resource_gate)(m,bounded_run),'original':original_empty(p)}
    at=now();need(type(at) in (int,float) and math.isfinite(at) and at+1800<=GLOBAL_DEADLINE,'full fresh 1800-second global window')
    root=Path(m['root']);deadline=at+1800
    create_json(root/'freeze-intent.json',{'schema':'jope.native-v5-freeze-intent/v1','stage':stage_ref,'created_unix':at,'deadline_unix':deadline,'gate':gate})
    guard={'schema':'jope.native-global-execution-guard/v1','registry_path':m['registry_path'],
           'scope_sha256':p['scope_sha256'],'canonicalizer_sha256':p['canonicalizer_sha256'],
           'global_deadline_unix':GLOBAL_DEADLINE,'sources':p['sources'],'census':p['census'],'max_snapshot_bytes':256*1024**2}
    workers=[]
    for w in s['workers']:
        config={k:t[k] for k in PACKET_FIELDS-DERIVED}
        config.update(schema='jope.native-original-successor/v5',created_unix=at,deadline_unix=deadline,
                      global_deadline_unix=GLOBAL_DEADLINE,global_guard=guard,maximum_emissions=128,
                      out=w['root'],outbox_path=p['continuation_outbox']['path'],outbox_config=p['continuation_outbox']['config'],
                      threads=1,worker_id=m['unit_prefix']+'-worker'+str(w['index']))
        config['source_pins']=t['source_pins']+w['source_pins']
        ref=create_json(Path(w['root'])/'config.json',config)
        workers.append({'index':w['index'],'unit':config['worker_id'],'config':ref})
    return create_json(root/'plan.json',{'schema':'jope.native-v5-pool-plan/v1','stage':stage_ref,
        'created_unix':at,'deadline_unix':deadline,'workers':workers})


def bounded_run(argv, *, timeout=30, limit=OUTPUT_CAP):
    """No shell; combined stdout/stderr is capped while the child runs."""
    need(type(argv) is list and all(type(x) is str for x in argv) and 0<timeout<=300 and 0<limit<=OUTPUT_CAP,'bounded subprocess request')
    child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    output=bytearray();end=time.monotonic()+timeout;selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
    try:
        while selector.get_map():
            need(time.monotonic()<end,'subprocess deadline')
            for key,_ in selector.select(min(0.1,max(0,end-time.monotonic()))):
                chunk=os.read(key.fd,4096)
                if not chunk:selector.unregister(key.fileobj);continue
                output.extend(chunk);need(len(output)<=limit,'subprocess output ceiling')
        code=child.wait(timeout=max(0.01,end-time.monotonic()))
        return {'returncode':code,'output':output.decode('utf-8',errors='replace')}
    finally:
        selector.close();child.stdout.close()
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=5)


def run_checked(runner, argv, timeout=30, allowed_codes=(0,)):
    result=runner(argv,timeout=timeout,limit=OUTPUT_CAP)
    need(type(result) is dict and set(result)=={'returncode','output'} and result['returncode'] in allowed_codes and
         type(result['output']) is str and len(result['output'].encode())<=OUTPUT_CAP,'bounded successful command')
    return result


# Literal reviewed read-only program. SSH receives one shlex-quoted command;
# neither manifest text nor a unit name can become executable shell syntax.
PHYSICAL_UNIT = '''import json,os,socket,subprocess,sys,time
from pathlib import Path
unit,host=sys.argv[1:]
assert socket.gethostname()==host
name=unit if unit.endswith('.service') else unit+'.service'
result=subprocess.run(['systemctl','--user','show',name,'--property=Id,LoadState,ActiveState,SubState,MainPID,ControlPID,ControlGroup'],capture_output=True,timeout=5)
assert result.returncode in (0,3,4) and len(result.stdout)+len(result.stderr)<=65536
fields=dict(line.split('=',1) for line in result.stdout.decode().splitlines() if '=' in line)
assert fields.get('Id')==name and fields.get('LoadState') in ('loaded','not-found')
assert fields.get('ActiveState') in ('inactive','failed') and fields.get('SubState') in ('dead','failed')
assert fields.get('MainPID')=='0' and fields.get('ControlPID')=='0'
base=Path('/sys/fs/cgroup');uid=os.getuid()
expected=Path('user.slice')/('user-'+str(uid)+'.slice')/('user@'+str(uid)+'.service')/'app.slice'/name
paths={expected};actual=fields.get('ControlGroup','')
if actual:
 assert actual.startswith('/') and '..' not in Path(actual).parts
 paths.add(Path(actual.lstrip('/')))
groups=[]
for relative in sorted(paths):
 path=base/relative
 assert all(not part.is_symlink() for part in (path,*path.parents))
 if not path.exists():
  groups.append({'path':str(path),'exists':False});continue
 with (path/'cgroup.procs').open('r') as stream: procs=stream.read(65537)
 with (path/'cgroup.events').open('r') as stream: events=stream.read(4097)
 assert len(procs)<=65536 and len(events)<=4096
 values=dict(line.split() for line in events.splitlines())
 assert not procs.split() and values.get('populated')=='0'
 groups.append({'path':str(path),'exists':True,'pids':[],'populated':0})
print(json.dumps({'schema':'jope.native-v5-live-unit/v1','host':host,'unit':name,'systemd':fields,'cgroups':groups,'checked_at_unix':time.time()}))
'''


def live_closure(prepared, runner):
    observations=[]
    for item in prepared['legacy_units']:
        command=['/usr/bin/python3','-I','-c',PHYSICAL_UNIT,item['unit'],item['host']]
        if item['host']=='xbabe3':command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','xbabe3',shlex.join(command)]
        result=run_checked(runner,command,15)
        proof=json.loads(result['output'])
        need(type(proof) is dict and set(proof)=={'schema','host','unit','systemd','cgroups','checked_at_unix'} and
             proof['schema']=='jope.native-v5-live-unit/v1' and proof['host']==item['host'] and
             proof['unit']==item['unit'].removesuffix('.service')+'.service','closed exact physical unit proof')
        fields=proof['systemd'];groups=proof['cgroups']
        need(fields.get('ActiveState') in ('inactive','failed') and fields.get('SubState') in ('dead','failed') and
             fields.get('MainPID')=='0' and fields.get('ControlPID')=='0' and type(groups) is list and 1<=len(groups)<=2 and
             all(type(g) is dict and (set(g)=={'path','exists'} and g['exists'] is False or
                 set(g)=={'path','exists','pids','populated'} and g['exists'] is True and g['pids']==[] and g['populated']==0)
                 for g in groups),
             'legacy unit/cgroup must remain physically empty')
        observations.append(proof)
    return observations


def resource_gate(m, runner):
    return resource_observation(m, runner)


def command(python, worker, *, seconds, preflight=False):
    need(type(seconds) is int and 1<=seconds<=1800,'bounded systemd runtime')
    config=worker['config'];directory=str(Path(config['path']).parent)
    argv=['systemd-run','--user','--unit='+worker['unit']+('-preflight' if preflight else ''),
          '--property=CPUQuota=100%','--property=MemoryMax=1G','--property=MemorySwapMax=0',
          '--property=TasksMax=32','--property=Restart=no','--property=KillMode=control-group',
          '--property=RuntimeMaxSec='+str(seconds),'--property=Nice=15',
          '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
          '--setenv=MKL_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1']
    if preflight:argv+=['--wait','--pipe']
    argv+=[python,'-B',directory+'/native_dispatch.py','--config',config['path'],'--config-sha',config['sha256']]
    if preflight:argv+=['--preflight-only']
    return argv


def launch(plan_ref, *, runner=bounded_run, resources=resource_gate, now=time.time):
    plan=document(plan_ref,{'schema','stage','created_unix','deadline_unix','workers'},'jope.native-v5-pool-plan/v1')
    s,m,p,t=staged(plan['stage']);root=Path(m['root'])
    need(plan_ref['path']==str(root/'plan.json') and plan['created_unix']<=now()<plan['deadline_unix'] and
         plan['deadline_unix']-plan['created_unix']==1800 and plan['deadline_unix']<=GLOBAL_DEADLINE,'same fixed plan window')
    need(type(plan['workers']) is list and len(plan['workers'])==m['worker_count'],'exact planned worker count')
    guards=[]
    for i,w in enumerate(plan['workers']):
        need(set(w)=={'index','unit','config'} and w['index']==i and w['unit']==m['unit_prefix']+'-worker'+str(i) and
             w['config']['path']==str(root/('worker'+str(i))/'config.json'),'exact worker plan')
        c=document(w['config'],PACKET_FIELDS,'jope.native-original-successor/v5');guards.append(c['global_guard'])
        need(c['created_unix']==plan['created_unix'] and c['deadline_unix']==plan['deadline_unix'] and
             c['maximum_emissions']==128 and c['threads']==1 and c['out']==str(root/('worker'+str(i))) and
             c['worker_id']==w['unit'] and c['outbox_path']==p['continuation_outbox']['path'] and
             c['outbox_config']==p['continuation_outbox']['config'],'exact continuation worker bounds')
        need(all(c[k]==t[k] for k in PACKET_FIELDS-DERIVED-{'source_pins'}) and
             c['source_pins']==t['source_pins']+s['workers'][i]['source_pins'],'preserved input/build/staged source pins')
    need(all(g==guards[0] for g in guards) and guards[0]=={
        'schema':'jope.native-global-execution-guard/v1','registry_path':m['registry_path'],
        'scope_sha256':p['scope_sha256'],'canonicalizer_sha256':p['canonicalizer_sha256'],
        'global_deadline_unix':GLOBAL_DEADLINE,'sources':p['sources'],'census':p['census'],'max_snapshot_bytes':256*1024**2},'identical exact shared global guard')
    validate_registry(m, guards[0])
    # Burn before any command, including preflight. A crash/partial attempt must
    # receive explicit forensic review; the same plan can never start twice.
    create_json(root/'operations-attempt.json',{'schema':'jope.native-v5-operations-attempt/v1','plan':plan_ref,'at':now(),'automatic_retry':False})
    try:
        gate={'closure':live_closure(p,runner),'original':original_empty(p),'python':python_probe(m['python'],runner),
              'registry':validate_registry(m,guards[0])}
        for w in plan['workers']:
            for name in (w['unit'],w['unit']+'-preflight'):
                r=run_checked(runner,['systemctl','--user','show',name,'--property=LoadState'],15,allowed_codes=(0,1,4))
                need(r['output'].strip()=='LoadState=not-found','fresh systemd unit required')
        gate['resources']=fresh_resources(m,runner,resources,now)
        create_json(root/'gates-before-preflight.json',gate)
        def one_preflight(w):
            resource_proof=fresh_resources(m,runner,resources,now)
            seconds=min(240,int(plan['deadline_unix']-now()-610));need(seconds>0,'preflight must leave native lease window')
            argv=command(m['python']['path'],w,seconds=seconds,preflight=True)
            create_json(root/('preflight-intent'+str(w['index'])+'.json'),{'argv':argv,'at':now(),'resources':resource_proof})
            result=run_checked(runner,argv,seconds+15)
            lines=result['output'].splitlines()
            proof=next((json.loads(line) for line in reversed(lines) if line.startswith('{')),None)
            need(type(proof) is dict and proof.get('state')=='preflight_pass' and type(proof.get('global_guard')) is dict,
                 'actual native dispatcher global preflight proof')
            global_proof=proof['global_guard']
            need(set(global_proof)=={'schema','sources','registry','historical_census_sha256','automatic_retry'} and
                 global_proof['schema']==guards[0]['schema'] and global_proof['historical_census_sha256']==p['census']['sha256'] and
                 global_proof['automatic_retry'] is False and type(global_proof['sources']) is list and
                 [(x['path'],x['kind']) for x in global_proof['sources']]==[(x['path'],x['kind']) for x in p['sources']] and
                 type(global_proof['registry']) is dict and set(global_proof['registry'])=={'claims','head_sha256'} and
                 type(global_proof['registry']['claims']) is int and global_proof['registry']['claims']>=0 and
                 hash64(global_proof['registry']['head_sha256']),'exact preflight census/source/registry proof')
            create_json(root/('preflight-result'+str(w['index'])+'.json'),result)
        # At most four CPU1/RAM1GiB preflights, within the larger frozen pool reservation.
        # All finish (or fail) before any native worker starts. No retries.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(one_preflight,w) for w in plan['workers']]
            try:
                for future in concurrent.futures.as_completed(futures):future.result()
            except BaseException:
                for future in futures:future.cancel()
                raise
        gate={'closure':live_closure(p,runner),'original':original_empty(p),'python':python_probe(m['python'],runner),
              'registry':validate_registry(m,guards[0])}
        gate['resources']=fresh_resources(m,runner,resources,now)
        need(plan['deadline_unix']-now()>610,'fresh native lease window remains')
        create_json(root/'gates-before-launch.json',gate)
        create_json(root/'launch-intent.json',{'schema':'jope.native-v5-launch-intent/v1','plan':plan_ref,'at':now(),
                    'workers':plan['workers'],'deadline_unix':plan['deadline_unix'],'automatic_retry':False})
        receipts=[]
        for w in plan['workers']:
            seconds=int(plan['deadline_unix']-now());need(seconds>610,'native start must precede lease cutoff')
            read(m['python'])
            resource_proof=fresh_resources(m,runner,resources,now)
            registry_proof=validate_registry(m,guards[0])
            need(0<=now()-resource_proof['observed_at_unix']<=5,'resource gate stale at dispatch boundary')
            need(plan['deadline_unix']-now()>610,'lease window expired during final gate')
            argv=command(m['python']['path'],w,seconds=int(plan['deadline_unix']-now()))
            create_json(root/('worker-launch-intent'+str(w['index'])+'.json'),{'argv':argv,'at':now(),
                'resources':resource_proof,'registry':registry_proof})
            result=run_checked(runner,argv,30)
            receipts.append(create_json(root/('worker-launch-result'+str(w['index'])+'.json'),result))
        return create_json(root/'launched.json',{'schema':'jope.native-v5-launched/v1','plan':plan_ref,'receipts':receipts,
                    'native_completion_verified':False,'automatic_retry':False})
    except BaseException as error:
        create_json(root/'operations-failure.json',{'schema':'jope.native-v5-operations-failure/v1','plan':plan_ref,
                    'error':type(error).__name__+': '+str(error)[:2000],'automatic_retry':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--input-sha',required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--stage',action='store_true')
    mode.add_argument('--freeze',action='store_true');mode.add_argument('--launch',action='store_true')
    args=parser.parse_args();ref=reference(absolute(args.input));need(ref['sha256']==args.input_sha,'exact CLI input SHA')
    action=launch if args.launch else freeze if args.freeze else stage if args.stage else dry_plan
    print(json.dumps(action(ref),sort_keys=True))


if __name__=='__main__':main()
