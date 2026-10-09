"""Pinned KumoS-only qualification for distinct evergreen prepared tasks.

The coordinator runs one isolated Python process for NVIDIA KumoS,
sharing one wall/monotonic deadline and external CPU8/12GiB/one-GPU cgroup.
All source tasks replay against their pinned parent/generator before execution.
No training, GP scores, S3 writes, or mutation of source datasets occurs here.
"""
import argparse, gc, gzip, hashlib, importlib, importlib.util, io, json, math, os
from pathlib import Path
import re, signal, socket, sqlite3, stat, struct, subprocess, sys, tempfile, time, types
sys.dont_write_bytecode = True
for k,v in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items():os.environ[k]=v
import numpy as np

SCHEMA='jope.flywheel-evergreen-kumos-rolling-packet/v1'
ROOT=None  # Bound once per host process by the immutable manifest.
DIMS={'kumo_tabular_s':2051}
PHASES={'ofm':['kumo_tabular_s']}
CONFIG_SHA='cd9ee8f882735c42b49d694b99dbbc4187035804099bc49f0ca81f8ec77a9b0a'
REGISTRY_SHA='8fe409848aadd6256644466faca6a0ceab73ab67131d30243701e0d93d9d0593'
HELPERS={'ofm':'bbd3230459a9bea9871c97055a6d2ec5a7655dcff7f41b57c9866a3fc86354d6'}
WEIGHT_SHA='fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423'
ROSTER='kumo-s-only-v1'
RAW_CAP=16*1024**2
OUT_CAP=512*1024**2
FILE_CAP=16*1024**2
PIPELINE_SHA='b4eb3a3aa5e85ff9af0eab29101c21dec74018826e3244ee9b5103caa47d5e78'

def need(ok,msg):
    if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def safe(path):
    p=Path(path);need(p.is_absolute() and '..'not in p.parts and all(not q.is_symlink()for q in (p,*p.parents)),'absolute nonsymlink path required');return p

def checked(ref,cap=FILE_CAP,load=True):
    p=safe(ref['path']);need(p.is_file() and stat.S_ISREG(p.stat().st_mode) and p.stat().st_size<=cap,'regular bounded source required')
    h=hashlib.sha256();chunks=[];n=0
    with p.open('rb')as f:
        while True:
            b=f.read(1024**2)
            if not b:break
            n+=len(b);need(n<=cap,'source grew beyond cap');h.update(b)
            if load:chunks.append(b)
    need(h.hexdigest()==ref['sha256'] and ('bytes'not in ref or n==ref['bytes']),'source SHA/size mismatch: '+str(p))
    return b''.join(chunks)if load else p

def obj(ref,cap=FILE_CAP):return json.loads(checked(ref,cap))
def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(canonical({'dtype':a.dtype.str,'shape':list(a.shape)}));h.update(a.tobytes());return h.hexdigest()
def npy(v):
    s=io.BytesIO();np.save(s,v,allow_pickle=False);return s.getvalue()
def uncompress(b):
    with gzip.GzipFile(fileobj=io.BytesIO(b))as f:r=f.read(RAW_CAP+1)
    need(len(r)<=RAW_CAP,'decompressed cap');return r

def decode(raw,allow_test=False):
    need(len(raw)>=17 and raw[:4]==b'JTT1','JTT1 required')
    n,nt,p,k=struct.unpack('<IIIB',raw[4:17]);need(50<=n<2000 and 1<=p<=512 and k==2,'bounded task dimensions required')
    need(nt==0 or (allow_test and nt==17),'protected test payload refused')
    need(len(raw)==17+n+8*(n+nt)*(p+1),'JTT1 byte closure differs')
    f=np.frombuffer(raw[17:17+n],dtype=np.uint8).copy();need(set(f.tolist())=={0,1} and min(np.bincount(f))>=25,'two inner folds >=25 rows required')
    a=np.frombuffer(raw[17+n:17+n+8*n*(p+1)],dtype='<f8').reshape(n,p+1).copy()
    need(np.isfinite(a).all() and np.ptp(a[:,-1])>1e-12,'finite nondegenerate task required')
    return a[:,:-1],a[:,-1],f

def protected(raw):
    x,y,_=decode(raw)
    return b'JTT1'+struct.pack('<IIIB',len(y),17,x.shape[1],2)+raw[17:]+np.random.default_rng(1729).normal(size=(17,x.shape[1]+1)).astype('<f8').tobytes()
def vector(v,t):
    need(isinstance(v,np.ndarray)and v.shape==(DIMS[t],)and v.dtype==np.dtype('<f4')and np.isfinite(v).all(),'full finite float32 vector required: '+t);return v

def owned_size(out):
    """Count live owned bytes; Triton may atomically rename temporary cache files.

    Only vanished entries under explicitly owned cache directories are tolerated.
    Missing scientific artifacts, symlinks and nonregular files still fail.
    """
    caches={'triton-cache','cuda-cache','torch-cache','hf-cache'};total=0
    def transient(path):
        relative=Path(path).relative_to(out)
        if caches.intersection(relative.parts):return True
        # The reviewed ledger uses .<export>.<32hex>.tmp before atomic rename.
        # Only vanished named exports at the coordinator/phase root qualify.
        return relative.parent in (Path('.'),Path('ofm')) and (relative.name=='progress.json.tmp' or re.fullmatch(r'\.events\.(csv|jsonl)\.[0-9a-f]{32}\.tmp',relative.name)is not None)
    def walk_error(exc):
        if isinstance(exc,FileNotFoundError)and transient(exc.filename):return
        raise exc
    for base,dirs,files in os.walk(out,followlinks=False,onerror=walk_error):
        for name in dirs+files:
            p=Path(base)/name
            try:info=p.lstat()
            except FileNotFoundError:
                if transient(p):continue
                raise
            need(not stat.S_ISLNK(info.st_mode),'output symlink')
            if stat.S_ISDIR(info.st_mode):continue
            need(stat.S_ISREG(info.st_mode),'nonregular output');total+=info.st_size
    need(total<=OUT_CAP,'total output cap');return total

def write_new(out,rel,b):
    r=Path(rel);need(not r.is_absolute()and'..'not in r.parts,'output escape')
    p=safe(out/r);need(len(b)<=FILE_CAP and owned_size(out)+len(b)<=OUT_CAP,'output cap');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return {'path':str(r),'sha256':sha(b),'bytes':len(b)}
def absolute(out,ref):return {**ref,'path':str(out/ref['path'])}
def atomic(out,name,value):
    write_new(out,name+'.tmp',canonical(value));os.replace(out/(name+'.tmp'),safe(out/name))
def module(ref,name):
    p=checked(ref,load=False);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def coverage(m):
    """Exact selected, already S3-published packet; no original128 assumptions."""
    pub=obj(m['dataset_publication']);n=m['total_variants']
    need(type(n)is int and 0<n<=32,'small bounded packet')
    need(pub['schema']=='jope.flywheel-generated-packet-publication-complete/v1'and pub['state']=='complete'
         and pub['new_gp_labels']==0 and pub['teaching_eligible']is False,'completed dataset publication required')
    need(pub['schema']=='jope.flywheel-generated-packet-publication-complete/v1' and pub['pipeline_config_sha256']==m['pipeline_config']['sha256']==PIPELINE_SHA,'declared pipeline publication');pipeline=obj(m['pipeline_config']);need(pipeline['schema']=='jope.flywheel-growth-pipeline/v2' and pipeline['required_encoders']==['kumo_tabular_s'],'Kumo-only growth pipeline');need(pub['publisher_sha256']==pipeline['publisher']['sha256'],'reviewed dataset publisher required')
    allmeta={v['variant_identity']:v for v in pub['metadata']};published={v['variant_identity']:v for v in pub['variants']}
    objects={r['key']:r for r in pub['object_receipts']}
    need(len(allmeta)==len(pub['metadata'])==len(published)==len(pub['variants'])==n and len(objects)==len(pub['object_receipts'])==4*n,'exact published packet closure')
    assignments={(h,v['identity']):v for h,c in m['hosts'].items()for v in c['variants']}
    ids=[v['identity']for c in m['hosts'].values()for v in c['variants']]
    need(len(ids)==len(set(ids))==n and set(ids)==set(allmeta)==set(published),'exact packet assignment')
    support={};prefix='experiments/gp-policy-dl/v2-trainonly/'
    for key,v in assignments.items():
        r=allmeta[v['identity']];p=published[v['identity']]
        need((v['sequence'],v['lineage'],v['partition'],v['n'],v['p'],v['changed'])==(r['sequence'],r['parent_lineage_sha256'],r['partition'],r['rows'],r['features'],r['target_sha256']!=r['source_target_sha256']),'published task semantics differ')
        need(p['sequence']==v['sequence']and objects.get(p['receipt']['key'])==p['receipt'],'dataset receipt binding')
        for name in ('encoder.jtt1.gz','metadata.json','generation.json'):
            ref=objects.get(prefix+'datasets/flywheel/v1/'+v['identity']+'/'+name)
            need(ref is not None and ref['bucket']=='veox-jopedime-learning-use2'and ref['version_id']and ref['sha256']==v['files'][name]['sha256']and ref['bytes']==v['files'][name]['bytes']and ref['verification']=='exact VersionId GET + SHA256','published input binding differs')
        support[key]=v
    need(len({v['lineage']for v in support.values()})==n,'one distinct lineage per selected packet task')
    return support

def manifest_ok(m,host):
    global ROOT
    root=safe(m['owned_root']);need(root.parent==Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')and root.name.startswith('flywheel_kumos_packet_'),'owned packet root');ROOT=root
    need(m['schema']==SCHEMA and m['approved_for_qualification']is True,'approved encoding manifest required')
    need(m['worker_path']==str(ROOT/'evergreen_kumos_packet.py'),'worker path differs')
    need(m['roster_version']==ROSTER,'roster version differs')
    need(m['components']==DIMS and type(m['total_variants'])is int and 0<m['total_variants']<=32 and m['new_gp_labels']==0 and m['teaching_eligible']is False,'scope differs')
    need(m['limits']=={'seconds_per_host':1200,'cpus_per_host':8,'memory_gib_per_host':12,'gpus_per_host':1},'envelope differs')
    need(0<len(m['hosts'])<=2 and set(m['hosts'])=={'xbabe1'}and host in m['hosts'],'known host assignment required')
    need(m['generator_config']['sha256']==CONFIG_SHA and m['parent_registry']['sha256']==REGISTRY_SHA,'source authority differs')
    need(set(m['helpers'])==set(PHASES),'helper closure differs')
    need(type(m['created_unix'])in(int,float)and type(m['deadline_unix'])in(int,float)and math.isfinite(m['deadline_unix'])and 0<m['deadline_unix']-m['created_unix']<=1200 and time.time()<m['deadline_unix'],'absolute shared20min deadline invalid/expired')
    coverage(m);checked(m['telemetry'],load=False)
    for h,c in m['hosts'].items():
        need(c['host']==h and 0<len(c['variants'])<=32 and c['out']==str(ROOT/'runs'/h),'host output/assignment differs')
        for v in c['variants']:
            need(len(v['identity'])==64 and all(ch in '0123456789abcdef'for ch in v['identity']),'variant ID')
            need(set(v['files'])=={'encoder.jtt1.gz','metadata.json','generation.json','parent_encoder.jtt1.gz'},'input closure differs')
            for name,ref in v['files'].items():need(ref['path']==str(ROOT/'inputs'/v['identity']/name),'input path differs')
    for c in m['hosts'].values():
        reservations=c['cpu_reservations'];need(type(reservations)is list and len(reservations)<=8 and len({r['unit']for r in reservations})==len(reservations),'explicit other-worker reservations')
        for r in reservations:need(set(r)=={'unit','memory_max_bytes'}and re.fullmatch(r'jopedime-[a-z0-9-]+',r['unit'])is not None and type(r['memory_max_bytes'])is int and 0<r['memory_max_bytes']<=64*1024**3,'bounded named CPU reservation')
    for phase,r in m['helpers'].items():need(r['sha256']==HELPERS[phase]and r['path']==str(ROOT/'helpers'/('pilot_'+phase+'.py')),'unreviewed helper')
    return m['hosts'][host]

def authority(m,msha,ref,*,fresh=False):
    need(ref['path']==str(ROOT/'authority'/(ref['sha256']+'.json')),'authority content-addressed path')
    proof=obj(ref);need(proof['schema']=='jope.flywheel-evergreen-encoding-authority/v1'and proof['state']=='passed'and proof['host']=='xbabe2','authority proof state')
    need(proof['manifest_sha256']==msha and proof['authority_helper_sha256']==m['authority_helper_sha256']and proof['generator_config_sha256']==CONFIG_SHA and proof['parent_registry_sha256']==REGISTRY_SHA,'authority proof identity')
    age=time.time()-proof['checked_at_unix'];need(0<=age<=(120 if fresh else 1200),'authority proof expired')
    conf=obj(m['generator_config']);need(proof['authority_pins']==conf['authority_pins'],'authority closure differs')
    parents={p['parent_lineage_sha256']:p for p in (json.loads(x)for x in checked(m['parent_registry'],32*1024**2).splitlines())}
    support={(r['host'],r['variant_identity']):r for r in proof['support']};expected=coverage(m);need(set(support)==set(expected)and len(proof['support'])==m['total_variants'],'authority exact packet support')
    for key,v in expected.items():
        r=support[key];p=parents[v['lineage']]
        need(r['parent_lineage_sha256']==v['lineage']and r['partition']==v['partition']and r['parent_record_sha256']==sha(canonical(p))and r['parent_context_sha256']==p['parent_context_sha256']and r['view_id']==p['view_id']and r['retained_contexts_checked']==len(p['retained_lineage_context_sha256'])and r['eligible_view']is True and r['excluded_contexts_or_views']is False,'authority selected parent differs')
    return proof

def load_inputs(m,cfg):
    """Verify immutable authority and exactly replay every transform including y."""
    conf=obj(m['generator_config']);registry=[json.loads(s)for s in checked(m['parent_registry'],32*1024**2).splitlines()]
    need(len(registry)==1068,'registry closure differs');bylineage={p['parent_lineage_sha256']:p for p in registry};need(len(bylineage)==1068,'duplicate lineage')
    for ref in m['source_pins']:checked(ref,load=False)
    source={Path(r['path']).name:r['sha256']for r in m['source_pins']}
    need(all(source.get(Path(r['path']).name)==r['sha256']for r in conf['source_pins']),'generator implementation differs')
    sys.path.insert(0,str(ROOT/'sources'))
    from jope_flywheel import evergreen as gen
    from jope_flywheel import protocol
    need(Path(gen.__file__)==ROOT/'sources/jope_flywheel/evergreen.py','generator import origin differs')
    result=[]
    for v in cfg['variants']:
        files=v['files'];meta=obj(files['metadata.json']);receipt=obj(files['generation.json']);parent=bylineage[v['lineage']]
        need(meta['parent']==parent and meta['schema']=='jope.flywheel-dataset/v1','exact registered parent required')
        need(parent['rights']=='research'and parent['parent_admission']=='prepared-research-transforms-only'and parent['prepared']is True and parent['raw_units']is False,'parent admission differs')
        need(parent['role_authority']['eligible_target_all_views']is True and parent['role_authority']['excluded_context_or_view']is False,'excluded role')
        need(parent['partition']==protocol.partition(parent['parent_lineage_sha256'],parent['split_namespace_sha256']),'lineage split differs')
        need(files['parent_encoder.jtt1.gz']['sha256']==parent['encoder']['sha256'],'copied parent payload differs')
        praw=uncompress(checked(files['parent_encoder.jtt1.gz']));need(sha(praw)==parent['raw_payload_sha256'],'parent raw hash')
        x,y,f=decode(praw);need((len(y),x.shape[1])==(parent['rows'],parent['features']),'parent shape')
        need(not any(np.array_equal(x[:,j],y)for j in range(x.shape[1])),'target-copy feature')
        raw=uncompress(checked(files['encoder.jtt1.gz']));xx,yy,ff=decode(raw)
        rid,replayed,remeta=gen.generate(parent,meta['recipe']['ordinal'],x,y,f)
        need(rid==v['identity']and raw==replayed and canonical(meta)==canonical(remeta),'generator replay differs')
        need(meta['variant_identity']==v['identity']and meta['content_sha256']==sha(raw)and meta['partition']==v['partition'],'task identity differs')
        need((meta['rows'],meta['features'],meta['parent_lineage_sha256'],meta['target_sha256']!=meta['source_target_sha256'])==(v['n'],v['p'],v['lineage'],v['changed']),'selected descriptor differs')
        need(meta['teaching_eligible']is False and meta['gp_measurements']==0 and meta['inherited_gp_rewards']is False,'inherited label refused')
        need(receipt['schema']==gen.SCHEMA and receipt['sequence']==v['sequence']and receipt['variant_identity']==rid and receipt['raw_sha256']==sha(raw)and receipt['encoder_sha256']==files['encoder.jtt1.gz']['sha256']and receipt['metadata_sha256']==files['metadata.json']['sha256']and receipt['config_sha256']==CONFIG_SHA and receipt['parent_lineage_sha256']==v['lineage'],'generation receipt differs')
        row=v['database_row'];need(row['state'] in ('published','admitted')and row['identity']==rid and row['content_sha']==sha(raw)and row['metadata_sha']==files['metadata.json']['sha256']and row['parent']==v['lineage']and row['sequence']==v['sequence']and row['ordinal']==meta['recipe']['ordinal'],'generated DB snapshot differs')
        result.append({'identity':rid,'sequence':v['sequence'],'raw':raw,'metadata':meta,'files':files})
    return result

def tick(budget):
    need(time.time()<budget['deadline_unix']and time.monotonic()<budget['monotonic_deadline'],'shared deadline exceeded')
def budget_ok(b,msha,host,m):
    need(b['schema']==SCHEMA and b['manifest_sha256']==msha and b['host']==host,'budget identity differs')
    need(b['started_unix']<b['deadline_unix']<=b['started_unix']+1200 and b['deadline_unix']<=m['deadline_unix']and b['started_monotonic']<b['monotonic_deadline']<=b['started_monotonic']+1200,'budget expanded');tick(b)

def runtime(m,cfg,phase):
    need(phase=='ofm','only KumoS phase allowed');h=module(m['helpers']['ofm'],'evergreen_helper_ofm');c=cfg['ofm']
    need(sys.executable==c['python'],'phase Python executable differs')
    sys.path[:0]=c['pythonpath'];inv=obj(c['runtime_inventory'])
    need(inv['schema']=='jope.kumos-runtime-inventory/v1'and sys.version.split()[0]==inv['python']and sys.executable==inv['executable'],'runtime identity differs')
    root=str(h.W/'src/structured-data-models/sdm')
    need(inv['source_roots']==[root]and 0<len(inv['files'])<=1000 and sum(r['bytes']for r in inv['files'])<64*1024**2,'Kumo-only runtime scope')
    pins={r['path']:r['sha256']for r in inv['files']};need(len(pins)==len(inv['files'])and all(pins.get(p)==digest for p,digest in h.CORE.items()),'core source closure')
    paths=list(Path(root).rglob('*.py'));need(paths and all(str(p)in pins for p in paths),'unlisted Kumo source')
    need(all(str(Path(p)).startswith(root+'/')or p in h.CORE for p in pins),'unexpected vendor encoder source')
    for ref in inv['files']:checked(ref,load=False)
    actual={}
    for name,r in inv['modules'].items():
        checked({'path':r['origin'],'sha256':r['origin_sha256']},load=False);m0=importlib.import_module(name);v={'version':str(m0.__version__),'origin':str(Path(m0.__file__)),'origin_sha256':r['origin_sha256']}
        if name=='torch':v['cuda']=m0.version.cuda
        need(v==r,'runtime module differs: '+name);actual[name]=v
    forward=importlib.import_module('sdm.models.kumo.tabular.model');need(str(Path(forward.__file__))in pins,'Kumo forward source differs')
    import torch;need(not torch.cuda.is_initialized(),'CPU preflight initialized CUDA')
    need(set(c['weight_pins'])=={'kumo_tabular_s'}and c['weight_pins']['kumo_tabular_s']['sha256']==WEIGHT_SHA,'only exact licensed KumoS weight allowed')
    checked(c['weight_pins']['kumo_tabular_s'],1024**3,False)
    for ref in c['license_pins']:checked(ref,load=False)
    import ofm_core as core
    for name,rel in [('ofm_core','code/ofm_core.py'),('ofm_models','code/ofm_models.py'),('ta_common','code/ta_code/ta_common.py')]:need(str(Path(sys.modules[name].__file__))==str(h.W/rel),'OFM import origin')
    return h,{'runtime':actual,'core':core}

INTAKE_LOCK=Path('/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/tabpfn-gpu.lock')
LANE_GUARD=Path('/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/cpu-admission.json')
def nvidia(query, kind="gpu"):
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.check_output(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def intake_lock_held():
    """True if any /proc/locks entry names the intake lock's device and inode. Read-only: the lock is never opened."""
    if not INTAKE_LOCK.exists():
        return False
    st = INTAKE_LOCK.stat()
    want = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    for line in Path("/proc/locks").read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            maj, mnr, ino = parts[5].split(":")
            if (int(maj, 16), int(mnr, 16), int(ino)) == want:
                return True
        except ValueError:
            continue
    return False


class Admission:
    def __init__(self,reservations):
        import psutil
        self.psutil, self.samples = psutil, []
        self.reservations=reservations

    def memory_reserve(self):
        remaining=0;observed=[]
        for expected in self.reservations:
            result=subprocess.run(['systemctl','--user','show',expected['unit']+'.service','--property=ActiveState,MemoryCurrent,MemoryMax'],capture_output=True,text=True,timeout=5)
            need(len(result.stdout)<4096 and len(result.stderr)<4096,'bounded shared resource observation')
            fields=dict(line.split('=',1)for line in result.stdout.splitlines()if'='in line);current=0
            if result.returncode==0 and fields.get('ActiveState')=='active':
                need(fields.get('MemoryMax')==str(expected['memory_max_bytes']),'active CPU worker memory ceiling changed')
                current=int(fields['MemoryCurrent']);need(0<=current<=expected['memory_max_bytes'],'CPU worker current memory invalid')
            reserve=max(0,expected['memory_max_bytes']-current);remaining+=reserve
            observed.append({**expected,'active_state':fields.get('ActiveState','unknown'),'current_bytes':current,'unused_reservation_bytes':reserve})
        own=0
        try:
            rel=next(l.split(':',2)[2]for l in Path('/proc/self/cgroup').read_text().splitlines()if l.startswith('0::'));cg=Path('/sys/fs/cgroup')/rel.lstrip('/');limit=int((cg/'memory.max').read_text());current=int((cg/'memory.current').read_text())
            if limit==12*1024**3:need(0<=current<=limit,'own encoder cgroup current');own=current
        except(OSError,ValueError,StopIteration):pass
        observed.append({'own_encoder_current_bytes':own,'foreign_floor_bytes':24*1024**3,'remaining_encoder_reservation_bytes':max(0,12*1024**3-own)})
        return 24*1024**3+max(0,12*1024**3-own)+remaining,observed

    def check(self, start=False):
        reasons = []
        available=self.psutil.virtual_memory().available
        required_memory,reservations=self.memory_reserve()
        if available<required_memory:reasons.append("host available memory below reserve+encoder+other-worker budget")
        own = os.getpid()
        foreign = [a for a in nvidia("pid,used_memory", "apps") if a and a[0].isdigit() and int(a[0]) != own]
        if foreign:
            reasons.append(f"foreign CUDA compute apps: {len(foreign)}")
        free = int(nvidia("memory.free")[0][0])
        if start and free < 20 * 1024:
            reasons.append(f"GPU free {free} MiB < 20480 at start")
        if intake_lock_held():
            reasons.append("intake tabpfn-gpu.lock held")
        cpu = self.psutil.cpu_percent(interval=0.5 if start else None)
        now = time.monotonic(); self.samples = [(t, c) for t, c in self.samples if t > now - 60] + [(now, cpu)]
        rolling = sum(c for _, c in self.samples) / len(self.samples)
        if cpu >= 95 or rolling >= 85:
            reasons.append(f"CPU instant {cpu:.0f}% rolling {rolling:.0f}%")
        if LANE_GUARD.exists():
            try:
                g = json.loads(LANE_GUARD.read_text())
                if g.get("admit") is False:
                    reasons.append("lane cpu-admission admit=false")
            except ValueError:
                reasons.append("lane cpu-admission unreadable")
        return not reasons, reasons, {"cpu": cpu, "rolling": rolling, "gpu_free_mib": free, "foreign_apps": len(foreign),"host_available_bytes":available,"required_available_bytes":required_memory,'cpu_reservations':reservations}


def qualification_ofm(helper,core,enc,t,item):
    raw=item['raw'];x,y,f=decode(raw);d=core.prep({'context_sha256':item['identity']},raw=raw);control=protected(raw)
    refused=False
    try:decode(control)
    except ValueError:refused=True
    need(refused,'production protected payload refusal')
    dc=core.prep({'context_sha256':item['identity']},raw=control);need(helper.prep_equal(d,dc),'protected bytes changed OFM preprocessing')
    v,c,s=helper.vector_ok(enc.one(t,d),t);vr,_,sr=helper.vector_ok(enc.one(t,d),t);vc,_,sc=helper.vector_ok(enc.one(t,dc),t)
    dy={**d,'y':d['y'][np.random.default_rng(23).permutation(len(d['y']))].copy()};vy,_,_=helper.vector_ok(enc.one(t,dy),t)
    tests={'production_refuses_protected_payload':refused,'full_vector_repeat':bool(np.array_equal(v,vr)),'protected_full_vector':bool(np.array_equal(v,vc)),'protected_selection':s==sr==sc,'target_changes_full_vector':not bool(np.array_equal(v,vy)),'target_changes_hidden':not bool(np.array_equal(v[:-3],vy[:-3]))}
    need(all(tests.values()),'OFM qualification failed: '+str(tests));return v,{'passed':True,'tests':tests,'component_info':c,'selection':s}

def audit_selected(identity):
    need(len(identity)==64 and all(c in '0123456789abcdef' for c in identity),'audit identity')
    return int(sha(('kumos-production-audit-v1\0'+identity).encode()),16)%100==0

def qualified_baseline(m,cfg):
    b=obj(m['qualification_baseline']);need(b['schema']=='jope.kumos-qualified-production-baseline/v1' and b['variants']==32 and b['weight_sha256']==WEIGHT_SHA and b['runtime_inventory_sha256']==cfg['ofm']['runtime_inventory']['sha256'] and b['audit_namespace']=='kumos-production-audit-v1' and b['audit_modulus']==100,'production baseline identity')
    prior=obj(b['manifest']);need(prior['worker_sha256']==b['worker']['sha256'] and prior['hosts']['xbabe1']['ofm']['runtime_inventory']['sha256']==b['runtime_inventory_sha256'],'baseline runtime/source identity')
    # Verify every old source, particularly evidence.py, BEFORE any old module import.
    for ref in prior['source_pins']:checked(ref,load=False)
    evidence=Path(prior['owned_root'])/'sources/jope_flywheel/evidence.py';need(any(r['path']==str(evidence)for r in prior['source_pins']),'baseline evidence source pin absent')
    old=module(b['worker'],'qualified_baseline_worker');old.ROOT=Path(prior['owned_root']);pc=prior['hosts']['xbabe1'];closure=old.verify_campaign_seal(prior,pc,'xbabe1',b['manifest']['sha256']);phase=old.stage_seal(Path(pc['out'])/'ofm',prior,'xbabe1',b['manifest']['sha256'],'ofm');need(closure['ledger']==b['coordinator_anchor'] and closure['status_sha256']==b['status_sha256'] and phase==b['phase_anchor'],'baseline sealed anchors changed')
    rows=old.join(prior,pc,'xbabe1',b['manifest']['sha256'],Path(pc['out']));need(len(rows)==32,'baseline full support')
    for row in rows:
        rec=old.obj(row['receipt']);q=old.obj(old.absolute(Path(pc['out'])/'ofm',rec['qualification']));need(q['passed']is True and q['tests'] and all(v is True for v in q['tests'].values())and rec['producer']['weight']['sha256']==WEIGHT_SHA,'baseline full four-call qualification')
    return b

def production_ofm(helper,core,enc,t,item,baseline_ref):
    if audit_selected(item['identity']):
        v,q=qualification_ofm(helper,core,enc,t,item);q.update(mode='sampled_full_audit',encoder_calls=4,audit_selected=True,worker_qualification_baseline=baseline_ref);return v,q
    decode(item['raw']);d=core.prep({'context_sha256':item['identity']},raw=item['raw']);v,c,s=helper.vector_ok(enc.one(t,d),t)
    return v,{'passed':True,'mode':'production_single_call','encoder_calls':1,'audit_selected':False,'worker_qualification_baseline':baseline_ref,'tests':{'finite_dimension2051':True,'input_identity_replayed':True,'zero_outer_test_rows':True},'not_repeated_per_variant':['deterministic_repeat','protected_control','target_sensitivity'],'component_info':c,'selection':s}

def progress(out,host,msha,done,state,expected,**extra):
    value={'schema':SCHEMA,'at':time.time(),'host':host,'manifest_sha256':msha,'state':state,'completed_variant_components':len(done),'expected_variant_components':expected*len(DIMS),'completed_variants':sum(all((v,t)in{(r['variant_identity'],r['component'])for r in done}for t in DIMS)for v in{r['variant_identity']for r in done}),'expected_variants':expected,'new_gp_labels':0,'teaching_eligible':False,'roster_version':ROSTER,'required_components':list(DIMS),'completed_audited_variants':sum(r.get('audit_selected',False)for r in done),'encoder_calls':sum(r.get('encoder_calls',0)for r in done),**extra}
    atomic(out,'progress.json',value);return value

def result_record(out,msha,host,phase,item,t,v,q,producer):
    vector(v,t);prefix='variants/'+item['identity']+'/'+t
    vr=write_new(out,prefix+'.npy',npy(v));qr=write_new(out,prefix+'.qualification.json',canonical(q));meta=item['metadata']
    rec={'schema':SCHEMA,'host':host,'manifest_sha256':msha,'phase':phase,'roster_version':ROSTER,'variant_identity':item['identity'],'component':t,'dimension':DIMS[t],'source_input':item['files']['encoder.jtt1.gz'],'source_metadata':item['files']['metadata.json'],'input_raw_sha256':sha(item['raw']),'target_sha256':meta['target_sha256'],'source_target_sha256':meta['source_target_sha256'],'parent_lineage_sha256':meta['parent_lineage_sha256'],'partition':meta['partition'],'rows':meta['rows'],'features':meta['features'],'outer_test_rows':0,'qualification':qr,'output':vr,'producer':producer,'qualification_mode':q['mode'],'encoder_calls':q['encoder_calls'],'new_gp_labels':0,'teaching_eligible':False,'pretraining_overlap':'Unknown; not proof of unseen data','lift_eligible':False,'label_exposure':'Task training rows and targets enter supervised encoders/diagnostics; not GP-inner-fit-only. No outer-test rows. Prepared source acquisition and development partition caveats inherited.'}
    rr=write_new(out,prefix+'.json',canonical(rec));return {'variant_identity':item['identity'],'component':t,'receipt':rr,'output':vr,'audit_selected':q['audit_selected'],'encoder_calls':q['encoder_calls']}

def phase_run(m,cfg,host,msha,phase,authority_ref,budget=None):
    role_proof=authority(m,msha,authority_ref);items=load_inputs(m,cfg);helper,rt=runtime(m,cfg,phase);qualified_baseline(m,cfg)
    proof={'schema':SCHEMA,'host':host,'phase':phase,'manifest_sha256':msha,'source_sha256':m['worker_sha256'],'runtime':rt['runtime'],'inputs':[{'variant_identity':i['identity'],'input_raw_sha256':sha(i['raw']),'metadata_sha256':i['files']['metadata.json']['sha256'],'target_sha256':i['metadata']['target_sha256'],'partition':i['metadata']['partition'],'rows':i['metadata']['rows'],'features':i['metadata']['features']}for i in items],'generator_replay_exact':True,'fresh_role_proof':authority_ref,'role_checked_at_unix':role_proof['checked_at_unix'],'new_gp_labels':0,'teaching_eligible':False}
    from jope_flywheel.evidence import Ledger,canonical_json,MAX_MANIFEST_BYTES
    canonical_json(proof,limit=MAX_MANIFEST_BYTES)
    if budget is None:return proof
    tick(budget);need(socket.gethostname().split('.')[0]==host,'execution host differs')
    out=safe(Path(cfg['out'])/phase);need(not out.exists(),'fresh phase output required');out.mkdir()
    # Caches stay within the owned phase and count toward the coordinator cap.
    for k,n in [('TORCH_HOME','torch-cache'),('TRITON_CACHE_DIR','triton-cache'),('CUDA_CACHE_PATH','cuda-cache'),('HF_HOME','hf-cache')]:os.environ[k]=str(out/n)
    done=[];rid=m['campaign_id']+'-'+host+'-'+phase
    timing=module(m['telemetry'],'packet_timing');log=timing.Logger(out/'timing.jsonl',{'packet_id':m['packet_id'],'manifest_sha256':msha,'source_sha256':m['worker_sha256']})
    with Ledger(out/'evidence.sqlite')as ledger:
        ledger.register_run(rid,proof);write_new(out,'preflight.json',canonical(proof))
        try:
            admission=Admission(cfg['cpu_reservations']);ok,reasons,observation=admission.check(start=True);need(ok,'GPU admission: '+str(reasons));import torch;need(torch.cuda.device_count()==1,'exactly one GPU required')
            ledger.append('admission','resource.admission',observation,rid)
            t='kumo_tabular_s';tick(budget)
            with log.span('model_load_once',torch):enc=rt['core'].Enc([t])
            enc=timing.Encoder(enc,log,torch);need(set(enc.m)==set(enc.pins)=={t}and enc.pins[t]['path']==cfg['ofm']['weight_pins'][t]['path']and enc.pins[t]['sha256']==WEIGHT_SHA,'loaded KumoS-only weight identity')
            for item in items:
                tick(budget);ok,reasons,obs=admission.check();need(ok,str(reasons));ledger.append(t+'-'+item['identity']+'-start','encoder.intent',{'variant_identity':item['identity'],'component':t},rid)
                enc.select(item);v,q=production_ofm(helper,rt['core'],enc,t,item,m['qualification_baseline']);tick(budget);output_event=log.start('output_write',torch,variant_identity=item['identity']);done.append(result_record(out,msha,host,phase,item,t,v,q,{'weight':enc.pins[t],'runtime_inventory':cfg['ofm']['runtime_inventory'],'roster_version':ROSTER,'licenses':cfg['ofm']['license_pins']}));log.end(output_event,torch,state='complete');ledger.append(t+'-'+item['identity'],'component.qualified',done[-1],rid);progress(out,host,msha,done,'encoding',len(cfg['variants']))
            del enc;gc.collect();torch.cuda.empty_cache()
            tick(budget);need(len(done)==len(cfg['variants'])*len(PHASES[phase]),'phase incomplete');summary={'schema':SCHEMA,'state':'complete','host':host,'phase':phase,'manifest_sha256':msha,'outputs':done,'expected':len(cfg['variants'])*len(PHASES[phase]),'completed':len(done),'new_gp_labels':0,'teaching_eligible':False}
            log.close();write_new(out,'status.json',canonical(summary));ledger.export_jsonl('events.jsonl');ledger.export_csv('events.csv');ledger.seal(rid,{'status':'status.json','events':'events.jsonl','preflight':'preflight.json','timing':'timing.jsonl'});tick(budget);return summary
        except BaseException as exc:
            error=type(exc).__name__+': '+str(exc)[:400];ledger.append('failed','phase.failed',{'error':error,'completed':len(done)},rid);progress(out,host,msha,done,'failed',len(cfg['variants']),error=error);raise

def subprocess_phase(command,out,name,budget,env,on_progress=None):
    tick(budget)
    with (out/(name+'.stdout')).open('xb')as stdout,(out/(name+'.stderr')).open('xb')as stderr:
        p=subprocess.Popen(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
        last_progress=0.
        try:
            while p.poll()is None:
                tick(budget);owned_size(out);need((out/(name+'.stdout')).stat().st_size<=FILE_CAP and (out/(name+'.stderr')).stat().st_size<=FILE_CAP,'child log cap')
                if on_progress is not None and time.monotonic()-last_progress>=3:
                    on_progress();last_progress=time.monotonic()
                time.sleep(.1)
            tick(budget);owned_size(out);need((out/(name+'.stdout')).stat().st_size<=FILE_CAP and (out/(name+'.stderr')).stat().st_size<=FILE_CAP,'child log cap');need(p.returncode==0,'phase failed: '+name+'; see bounded stderr')
        finally:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=5)

def stage_seal(root,m,host,msha,phase):
    from jope_flywheel.evidence import Ledger
    p=safe(root/'evidence.sqlite');need(p.stat().st_size<=16*1024**2,'phase ledger cap')
    con=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,timeout=1);con.row_factory=sqlite3.Row
    try:
        con.execute('PRAGMA query_only=ON');con.execute('BEGIN');ledger=object.__new__(Ledger);ledger._connection=con;ledger.root=ledger._input_root=root;ledger._protected={'evidence.sqlite','evidence.sqlite-wal','evidence.sqlite-shm'};ledger.ledger_id=dict(con.execute('SELECT key,value FROM metadata'))['ledger_id']
        anchor=ledger._verify();runs=list(con.execute('SELECT * FROM runs'));need(len(runs)==1,'phase run count');r=runs[0];need(r['run_id']==m['campaign_id']+'-'+host+'-'+phase and r['source_sha256']==m['worker_sha256']and r['seal_event_id']is not None,'phase ledger run identity/seal')
        registered=json.loads(r['manifest_json']);need(registered['manifest_sha256']==msha and registered['host']==host and registered['phase']==phase,'registered phase identity')
        event=ledger._decode(con.execute('SELECT * FROM events WHERE event_id=?',(r['seal_event_id'],)).fetchone());artifacts=event['payload']['artifacts']
        need({a['path']for a in artifacts}=={'status.json','events.jsonl','preflight.json','timing.jsonl'},'phase artifact seal closure')
        for a in artifacts:checked({**a,'path':str(root/a['path'])},16*1024**2,False)
        con.execute('COMMIT');return anchor
    finally:con.close()

def join(m,cfg,host,msha,out):
    done=[];expected={v['identity']:v for v in cfg['variants']}
    for phase,components in PHASES.items():
        root=out/phase;stage_seal(root,m,host,msha,phase);p=safe(root/'status.json');need(p.stat().st_size<=FILE_CAP,'status cap');status=json.loads(p.read_bytes())
        need(status['schema']==SCHEMA and status['state']=='complete'and status['host']==host and status['phase']==phase and status['manifest_sha256']==msha,'stage identity/incomplete')
        rows=status['outputs'];need(len(rows)==status['completed']==status['expected']==len(cfg['variants'])*len(components),'stage count')
        need({(r['variant_identity'],r['component'])for r in rows}=={(i,t)for i in expected for t in components},'stage exact support')
        for r in rows:
            i,t=r['variant_identity'],r['component'];need(r['receipt']['path']=='variants/'+i+'/'+t+'.json'and r['output']['path']=='variants/'+i+'/'+t+'.npy','stage path')
            rec=obj(absolute(root,r['receipt']));need(rec['manifest_sha256']==msha and rec['host']==host and rec['phase']==phase and rec['variant_identity']==i and rec['component']==t and rec['source_input']==expected[i]['files']['encoder.jtt1.gz']and rec['source_metadata']==expected[i]['files']['metadata.json']and rec['output']==r['output']and rec['teaching_eligible']is False and rec['new_gp_labels']==0,'component identity')
            meta=obj(expected[i]['files']['metadata.json'])
            need(rec['input_raw_sha256']==meta['content_sha256'] and rec['target_sha256']==meta['target_sha256'] and rec['source_target_sha256']==meta['source_target_sha256'] and rec['parent_lineage_sha256']==meta['parent_lineage_sha256'] and rec['partition']==meta['partition'] and rec['rows']==meta['rows'] and rec['features']==meta['features'] and rec['outer_test_rows']==0 and rec['dimension']==DIMS[t],'component task semantics')
            q=obj(absolute(root,rec['qualification']));need(q['passed']is True,'unqualified component');vector(np.load(io.BytesIO(checked(absolute(root,r['output']))),allow_pickle=False),t)
            done.append({**r,'receipt':absolute(root,r['receipt']),'output':absolute(root,r['output'])})
    need(len(done)==len(cfg['variants'])*len(DIMS),'KumoS-output join incomplete');return done

def verify_campaign_seal(m,cfg,host,msha):
    """Readonly final closure; status alone is insufficient after interruption."""
    root=safe(cfg['out']);owned_size(root);p=safe(root/'evidence.sqlite');need(p.stat().st_size<=16*1024**2,'coordinator ledger cap')
    for source_ref in m['source_pins']:checked(source_ref,load=False)
    sys.path.insert(0,str(ROOT/'sources'))
    from jope_flywheel.evidence import Ledger
    con=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,timeout=1);con.row_factory=sqlite3.Row
    try:
        con.execute('PRAGMA query_only=ON');con.execute('BEGIN');ledger=object.__new__(Ledger);ledger._connection=con;ledger.root=ledger._input_root=root;ledger._protected={'evidence.sqlite','evidence.sqlite-wal','evidence.sqlite-shm'};ledger.ledger_id=dict(con.execute('SELECT key,value FROM metadata'))['ledger_id'];anchor=ledger._verify()
        runs=list(con.execute('SELECT * FROM runs'));need(len(runs)==1,'coordinator run count');r=runs[0];need(r['run_id']==m['campaign_id']+'-'+host and r['source_sha256']==m['worker_sha256']and r['seal_event_id']is not None,'coordinator is not sealed')
        registered=json.loads(r['manifest_json']);need(registered['manifest_sha256']==msha and registered['host']==host,'coordinator identity')
        need(all(ledger._decode(e)['event_type']not in ('campaign.failed','phase.failed')for e in con.execute('SELECT * FROM events')),'coordinator failure event')
        event=ledger._decode(con.execute('SELECT * FROM events WHERE event_id=?',(r['seal_event_id'],)).fetchone());artifacts=event['payload']['artifacts'];need({a['path']for a in artifacts}=={'status.json','events.jsonl','budget.json'},'coordinator seal closure')
        for a in artifacts:checked({**a,'path':str(root/a['path'])},16*1024**2,False)
        need(safe(root/'progress.json').stat().st_size<=1024**2,'progress cap')
        status=json.loads((root/'status.json').read_bytes());progress=json.loads((root/'progress.json').read_bytes());budget=json.loads((root/'budget.json').read_bytes())
        need(status['state']==progress['state']=='complete'and status['manifest_sha256']==progress['manifest_sha256']==budget['manifest_sha256']==msha and status['host']==progress['host']==budget['host']==host,'coordinator completion/progress differs')
        need(status['at']<=budget['deadline_unix']<=m['deadline_unix'],'completion after shared deadline')
        con.execute('COMMIT');return {'ledger':anchor,'seal':event['payload'],'status_sha256':sha((root/'status.json').read_bytes()),'status':status}
    finally:con.close()

def coordinator(m,cfg,host,msha,args):
    ref={'path':args.authority_path,'sha256':args.authority_sha};role=authority(m,msha,ref,fresh=not args.preflight_only)
    # File-capped isolated preflight processes share the manifest deadline.
    if args.preflight_only:
        result={};now=time.time();mono=time.monotonic();b={'deadline_unix':m['deadline_unix'],'monotonic_deadline':mono+min(1200,m['deadline_unix']-now)}
        with tempfile.TemporaryDirectory(prefix='cpu-preflight-',dir=ROOT)as temp:
            temp=Path(temp)
            for phase in PHASES:
                cmd=[cfg[phase]['python'],'-B',str(Path(__file__).absolute()),'--manifest',args.manifest,'--manifest-sha',msha,'--host',host,'--phase',phase,'--preflight-only','--authority-path',ref['path'],'--authority-sha',ref['sha256']]
                env=dict(os.environ);env['PYTHONPATH']='';env['CUDA_VISIBLE_DEVICES']='' if phase=='dope'else'0'
                subbudget={'deadline_unix':min(b['deadline_unix'],time.time()+180),'monotonic_deadline':min(b['monotonic_deadline'],time.monotonic()+180)}
                subprocess_phase(cmd,temp,phase,subbudget,env);result[phase]=json.loads((temp/(phase+'.stdout')).read_bytes());tick(b)
        tick(b);return {'schema':SCHEMA,'state':'preflight_pass','manifest_sha256':msha,'host':host,'phases':result,'fresh_role_proof':ref,'no_cuda_execution':True}
    started=time.time();mono=time.monotonic();b={'schema':SCHEMA,'host':host,'manifest_sha256':msha,'authority':ref,'started_unix':started,'started_monotonic':mono,'deadline_unix':min(m['deadline_unix'],started+1200),'monotonic_deadline':mono+min(1200,m['deadline_unix']-started)};budget_ok(b,msha,host,m)
    need(socket.gethostname().split('.')[0]==host,'host mismatch');helper=module(m['helpers']['ofm'],'evergreen_envelope');envproof=helper.envelope()
    out=safe(cfg['out']);need(not out.exists(),'fresh campaign output required');out.mkdir(parents=True)
    bref=write_new(out,'budget.json',canonical(b));sys.path.insert(0,str(ROOT/'sources'))
    for source_ref in m['source_pins']:checked(source_ref,load=False)
    from jope_flywheel.evidence import Ledger
    rid=m['campaign_id']+'-'+host;done=[]
    with Ledger(out/'evidence.sqlite')as ledger:
        ledger.register_run(rid,{'source_sha256':m['worker_sha256'],'manifest_sha256':msha,'host':host,'budget':b,'fresh_role_proof':ref,'envelope':envproof})
        try:
            progress(out,host,msha,done,'qualifying',len(cfg['variants']))
            for phase in PHASES:
                ledger.append(phase+'-start','phase.intent',{'phase':phase},rid)
                cmd=[cfg[phase]['python'],'-B',str(Path(__file__).absolute()),'--manifest',args.manifest,'--manifest-sha',msha,'--host',host,'--phase',phase,'--budget-sha',bref['sha256']]
                env=dict(os.environ);env['PYTHONPATH']='';env['CUDA_VISIBLE_DEVICES']='' if phase=='dope'else'0'
                def child_progress():
                    p=safe(out/phase/'progress.json')
                    if p.exists():
                        need(p.stat().st_size<=1024**2,'phase progress cap');q=json.loads(p.read_bytes())
                        need(q['host']==host and q['manifest_sha256']==msha,'progress identity')
                        n=q['completed_variant_components'];need(type(n)is int and 0<=n<=len(cfg['variants'])*len(PHASES[phase]),'phase progress count')
                        progress(out,host,msha,done,'encoding',len(cfg['variants']),active_phase=phase,active_phase_state=q['state'],observed_completed_components=len(done)+n,final_join_verified=False)
                subprocess_phase(cmd,out,phase,b,env,child_progress)
                ledger.append(phase+'-done','phase.completed',{'phase':phase,'status_sha256':sha((out/phase/'status.json').read_bytes())},rid)
                rows=json.loads((out/phase/'status.json').read_bytes())['outputs'];done.extend(rows);progress(out,host,msha,done,'encoding',len(cfg['variants']))
            tick(b);done=join(m,cfg,host,msha,out);ledger.append('joined','encoders.joined',{'components':len(cfg['variants'])*len(DIMS),'variants':len(cfg['variants']),'gp_labels':0,'teaching_eligible':False},rid)
            ledger.export_jsonl('events.jsonl');ledger.export_csv('events.csv');tick(b)
            summary=progress(out,host,msha,done,'complete',len(cfg['variants']),completed_encoders=list(DIMS),missing_encoders=[],role_authority_recheck_required_before_publication_or_admission=True,elapsed_seconds=time.monotonic()-mono)
            write_new(out,'status.json',canonical({**summary,'outputs':done}));ledger.seal(rid,{'status':'status.json','events':'events.jsonl','budget':'budget.json'});tick(b);owned_size(out);return summary
        except BaseException as exc:
            error=type(exc).__name__+': '+str(exc)[:400];progress(out,host,msha,done,'failed',len(cfg['variants']),error=error)
            if not ledger.verify()['runs'][rid]['sealed']:ledger.append('failed','campaign.failed',{'error':error,'completed_components':len(done)},rid)
            raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--host',choices=('xbabe1',),required=True);p.add_argument('--phase',choices=('all',*PHASES),default='all');p.add_argument('--preflight-only',action='store_true');p.add_argument('--budget-sha');p.add_argument('--authority-path');p.add_argument('--authority-sha');a=p.parse_args(argv)
    m=obj({'path':a.manifest,'sha256':a.manifest_sha});need(sha(Path(__file__).read_bytes())==m['worker_sha256'],'worker source drift');cfg=manifest_ok(m,a.host)
    if a.phase=='all':result=coordinator(m,cfg,a.host,a.manifest_sha,a)
    else:
        b=None
        if not a.preflight_only:
            need(a.budget_sha is not None,'coordinator budget required');b=obj({'path':str(Path(cfg['out'])/'budget.json'),'sha256':a.budget_sha});budget_ok(b,a.manifest_sha,a.host,m)
        ref=b['authority']if b is not None else {'path':a.authority_path,'sha256':a.authority_sha}
        result=phase_run(m,cfg,a.host,a.manifest_sha,a.phase,ref,b)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
if __name__=='__main__':main()
