"""Balanced, bounded dataset expansion with immutable provenance and backpressure.

This service generates prepared research variants. Encoding, GP scoring and
training admission are separate measured stages; merely creating data does not
advance those counters. Children always inherit their parent's DL partition.
"""
from dataclasses import asdict
import argparse,gzip,hashlib,io,json,math,os,sqlite3,struct,time
from pathlib import Path
import numpy as np
from .augmentation import Parent,array_sha256,roles_sha256
from .protocol import canonical,digest,partition,sha256
from . import target_noise

SCHEMA='jope.flywheel-evergreen/v1'
TRANSFORM='prepared-drop-nuisance-targetnoise/v1'
MAX_RAW=16*1024**2

def need(ok,message):
    if not ok:raise ValueError(message)
def byte_sha(body):return hashlib.sha256(body).hexdigest()
def safe_path(path):
    p=Path(path)
    need(p.is_absolute() and '..' not in p.parts and all(not q.is_symlink() for q in (p,*p.parents)),'absolute owned path without symlinks required')
    return p
def write_new(path,body):
    p=safe_path(path)
    with p.open('xb')as out:out.write(body);out.flush();os.fsync(out.fileno())

def pinned(ref,cap=MAX_RAW):
    p=Path(ref['path'])
    need(p.is_absolute() and all(not q.is_symlink() for q in (p,*p.parents)) and p.is_file(),'regular absolute source required')
    need(p.stat().st_size<=cap,'source cap')
    with p.open('rb')as f:b=f.read(cap+1)
    need(len(b)<=cap and byte_sha(b)==ref['sha256'],'source hash/cap mismatch')
    if 'bytes'in ref:need(len(b)==ref['bytes'],'source size mismatch')
    return b

def decode(raw):
    need(len(raw)>=17 and raw[:4]==b'JTT1','JTT1 required')
    n,nt,p,k=struct.unpack('<IIIB',raw[4:17])
    need(50<=n<2000 and nt==0 and 1<=p<=512 and k==2 and len(raw)==17+n+8*n*(p+1),'bounded stripped regression input required')
    folds=np.frombuffer(raw[17:17+n],dtype=np.uint8).copy()
    need(set(folds.tolist())=={0,1} and min(np.bincount(folds))>=25,'two inner folds with at least25rows each required')
    v=np.frombuffer(raw[17+n:],dtype='<f8').reshape(n,p+1).copy()
    need(np.isfinite(v).all(),'finite input required')
    return v[:,:-1],v[:,-1],folds

def encode(x,y,folds):
    need(1<=x.shape[1]<=512 and len(x)<2000 and len(x)==len(y)==len(folds),'variant dimensions')
    return b'JTT1'+struct.pack('<IIIB',len(y),0,x.shape[1],2)+np.asarray(folds,dtype=np.uint8).tobytes()+np.column_stack([x,y]).astype('<f8').tobytes()

def source(parent):
    need(parent['schema']=='jope.flywheel-prepared-parent/v1' and parent['parent_admission']=='prepared-research-transforms-only','qualified research parent required')
    need(parent['rights']=='research' and parent['raw_units']is False and parent['prepared']is True,'prepared research declaration required')
    need(parent['role_authority']['eligible_target_all_views']is True and parent['role_authority']['excluded_context_or_view']is False,'current eligible role required')
    need(parent['partition']==partition(parent['parent_lineage_sha256'],parent['split_namespace_sha256']),'lineage partition mismatch')
    policy=parent['allowed_transforms']
    need(policy['feature_gaussian_noise']is False and policy['resample_rows']is False,'unsupported source policy')
    need(policy['feature_drop_max_fraction']>=.1 and policy['nuisance_max_fraction']>=.1 and policy['nuisance_max_columns']>=8
         and policy['max_output_features']>=512 and policy['max_train_rows']>=1999 and policy['numeric_target_noise']is True,'parent transform limits too restrictive')
    pinned(parent['context'])
    if parent.get('preparation_receipt') is not None:pinned(parent['preparation_receipt'])
    zipped=pinned(parent['encoder'])
    with gzip.GzipFile(fileobj=io.BytesIO(zipped))as f:raw=f.read(MAX_RAW+1)
    need(len(raw)<=MAX_RAW and byte_sha(raw)==parent['raw_payload_sha256'],'raw source hash mismatch')
    x,y,folds=decode(raw)
    need((len(y),x.shape[1])==(parent['rows'],parent['features']),'source shape mismatch')
    need(array_sha256(x)==parent['feature_matrix_sha256'] and array_sha256(y)==parent['target_sha256'],'source feature/target mismatch')
    need(byte_sha(folds.tobytes())==parent['inner_labels_sha256'],'source folds mismatch')
    need(not any(np.array_equal(x[:,j],y) for j in range(x.shape[1])),'exact target-copy feature refused')
    return x,y,folds

def recipe(parent,ordinal):
    need(type(ordinal)is int and 0<=ordinal<256,'at most256 descendants per lineage in protocolv1')
    seed=int(digest({'domain':TRANSFORM,'lineage':parent['parent_lineage_sha256'],'ordinal':ordinal})[:16],16)
    p=parent['features'];drop=(.05,.1,0.,.1)[ordinal%4]
    count=min(p-1,math.floor(p*drop));nuisance=min(8,math.floor(p*.1),512-(p-count)) if ordinal%4!=1 else 0
    noise=(0.,.01,.03,.05)[ordinal%4]
    # Small feature sets may yield no legal drop/add. Still require real novelty.
    if count==0 and nuisance==0 and noise==0:noise=.01
    return {'schema':TRANSFORM,'ordinal':ordinal,'seed':seed,'drop_fraction':drop,'drop_count':count,
            'nuisance_count':nuisance,'nuisance_distribution':'independent standard Gaussian per registered row',
            'feature_gaussian_noise_fraction':0.,'target_noise_fraction':noise,
            'target_noise_definition':'Gaussian standard deviation / designated inner-fit target population standard deviation',
            'target_noise_calibration_fold':0,'target_noise_fit_rows':'inner label0 only',
            'inner_folds':'preserved; no new random row cut','row_selection':'all admitted parent training rows in original order'}

def rng(seed,stream,row=None):
    return np.random.Generator(np.random.PCG64(int(digest({'domain':TRANSFORM,'seed':seed,'stream':stream,'row':row}),16)))

def generate(parent,ordinal,x,y,folds):
    """Pure transformation of authenticated parent bytes; no parent GP rewards."""
    need(array_sha256(x)==parent['feature_matrix_sha256'] and array_sha256(y)==parent['target_sha256'] and byte_sha(folds.tobytes())==parent['inner_labels_sha256'],'generator parent binding mismatch')
    spec=recipe(parent,ordinal);p=x.shape[1]
    dropped=sorted(int(i) for i in rng(spec['seed'],'drop').choice(p,size=spec['drop_count'],replace=False))
    kept=[i for i in range(p) if i not in dropped]
    xx=np.empty((len(y),len(kept)+spec['nuisance_count']),dtype='<f8');xx[:,:len(kept)]=x[:,kept]
    for row in range(len(y)):
        if spec['nuisance_count']:xx[row,len(kept):]=rng(spec['seed'],'nuisance',row).standard_normal(spec['nuisance_count'])
    roles={i:'fit' if int(folds[i])==0 else 'validation' for i in range(len(y))}
    par=Parent(parent['raw_payload_sha256'],parent['parent_context_sha256'],parent['target_sha256'],'target',
        parent['parent_lineage_sha256'],parent['split_namespace_sha256'],parent['partition'],parent['parent_cut_id'],
        digest({'features':p,'semantics':'prepared numeric; unknown original types; Gaussian feature noise disabled'}),roles_sha256(roles),digest(parent))
    feature_id=digest({'protocol':TRANSFORM,'parent':digest(parent),'spec':spec,'dropped':dropped,'x_sha256':array_sha256(xx)})
    fit_ids=[i for i,r in roles.items() if r=='fit']
    # Nonzero noise on a constant calibration target fails closed, recorded by the worker.
    noise=target_noise.fit(y[fit_ids],row_ids=fit_ids,row_roles=roles,parent=par,feature_variant_sha256=feature_id,seed=spec['seed'],fraction=spec['target_noise_fraction'])
    yy,noise_receipt=target_noise.apply(y,row_ids=range(len(y)),state=noise)
    raw=encode(xx,yy,folds)
    need(byte_sha(raw)!=parent['raw_payload_sha256'],'no-op variant refused')
    identity=digest({'schema':SCHEMA,'parent_sha256':digest(parent),'recipe':spec,'raw_sha256':byte_sha(raw)})
    mapping=[{'output_index':j,'source_column':i,'kind':'prepared_numeric_unknown_original_type'}for j,i in enumerate(kept)]
    mapping += [{'output_index':len(kept)+j,'source_column':None,'kind':'synthetic_continuous_gaussian'}for j in range(spec['nuisance_count'])]
    meta={'schema':'jope.flywheel-dataset/v1','variant_identity':identity,'content_sha256':byte_sha(raw),
        'parent_lineage_sha256':parent['parent_lineage_sha256'],'parent_context_sha256':parent['parent_context_sha256'],
        'parent':parent,'parent_record_sha256':digest(parent),'recipe':spec,'dropped_columns':dropped,'column_mapping':mapping,
        'target_noise_fit':noise.manifest(),'target_noise_result':noise_receipt,'rows':len(y),'features':xx.shape[1],
        'outer_test_rows':0,'inner_folds':2,'inner_counts':np.bincount(folds).tolist(),'split_sha256':byte_sha(folds.tobytes()),
        'target_sha256':array_sha256(yy),'source_target_sha256':parent['target_sha256'],'partition':parent['partition'],
        'rights':'research','raw_units':False,'prepared':True,'new_independent_family':False,
        'numpy_version':str(np.__version__),'rng':'sha256-PCG64-per-row/v1','fresh_encodings_required':True,
        'fresh_gp_evaluations_required':True,'inherited_gp_rewards':False,'gp_measurements':0,'teaching_eligible':False,
        'admission':'pending8encoders+nativeGP+versionedS3+frozen_manifest','target_noise_uses_designated_fit_only':True,
        'limitation':'Prepared source/row-selection caveats inherited; augmentation is not an independent dataset family. Noise calibration defines a synthetic task and is not a model-fit step.'}
    return identity,raw,meta

class Expansion:
    def __init__(self,root,parents,*,target,backlog,config_sha):
        self.root=safe_path(root);self.root.mkdir(parents=True,exist_ok=True)
        for name in ('variants','expansion.sqlite','expansion.sqlite-wal','expansion.sqlite-shm','progress.new','progress.json'):safe_path(self.root/name)
        need(0<target<=100000 and 1<=backlog<=2048 and len(parents)*256>=target,'growth bounds')
        need(len({p['parent_lineage_sha256']for p in parents})==len(parents),'one representative per unique lineage')
        self.parents=sorted(parents,key=lambda p:digest({'order':TRANSFORM,'lineage':p['parent_lineage_sha256']}))
        self.target,self.backlog,self.config_sha=target,backlog,config_sha
        self.db=sqlite3.connect(self.root/'expansion.sqlite',timeout=20,isolation_level=None);self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY,sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS variants(sequence INTEGER PRIMARY KEY,parent TEXT NOT NULL,ordinal INTEGER NOT NULL,
        state TEXT NOT NULL,identity TEXT UNIQUE,content_sha TEXT UNIQUE,metadata_sha TEXT,error TEXT,created REAL NOT NULL,
        UNIQUE(parent,ordinal));''')
        identity=digest({'config_sha':config_sha,'parents':parents,'target':target,'backlog':backlog})
        self.db.execute('INSERT OR IGNORE INTO config VALUES(1,?)',(identity,))
        if self.db.execute('SELECT sha FROM config WHERE id=1').fetchone()[0]!=identity:
            self.db.close();raise ValueError('immutable controller config mismatch')
    def close(self):self.db.close()
    def snapshot(self):
        states={r[0]:r[1]for r in self.db.execute('SELECT state,count(*) FROM variants GROUP BY state')}
        return {'schema':SCHEMA,'at':time.time(),'target':self.target,'parent_lineages':len(self.parents),
                'states':states,'generated':sum(v for k,v in states.items()if k in ('generated','published','admitted')),
                'published':states.get('published',0)+states.get('admitted',0),'admitted':states.get('admitted',0),
                'backlog_limit':self.backlog,'config_sha256':self.config_sha}
    def reserve(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            rows=self.db.execute("SELECT count(*) FROM variants WHERE state IN ('reserved','generated','published')").fetchone()[0]
            if rows>=self.backlog:self.db.execute('COMMIT');return None
            # Reserved/unknown outcomes remain capacity-occupying until explicit reconciliation.
            seq=self.db.execute('SELECT coalesce(max(sequence)+1,0) FROM variants').fetchone()[0]
            ready=self.db.execute("SELECT count(*) FROM variants WHERE state IN ('reserved','generated','published','admitted')").fetchone()[0]
            if seq>=len(self.parents)*256 or ready>=self.target:self.db.execute('COMMIT');return None
            p=self.parents[seq%len(self.parents)];ordinal=seq//len(self.parents)
            self.db.execute('INSERT INTO variants(sequence,parent,ordinal,state,created) VALUES(?,?,?,?,?)',(seq,p['parent_lineage_sha256'],ordinal,'reserved',time.time()))
            self.db.execute('COMMIT');return seq,p,ordinal
        except BaseException:self.db.execute('ROLLBACK');raise
    def step(self):
        task=self.reserve()
        if task is None:return False
        seq,parent,ordinal=task
        try:
            x,y,f=source(parent);identity,raw,meta=generate(parent,ordinal,x,y,f)
            prior=self.db.execute('SELECT sequence FROM variants WHERE content_sha=?',(byte_sha(raw),)).fetchone()
            need(prior is None,'duplicate dataset content')
            dest=safe_path(self.root/'variants'/identity);dest.mkdir(parents=True,exist_ok=False)
            zipped=gzip.compress(raw,mtime=0);metadata=canonical(meta)
            for name,body in [('encoder.jtt1.gz',zipped),('metadata.json',metadata)]:
                write_new(dest/name,body)
            receipt={'schema':SCHEMA,'sequence':seq,'variant_identity':identity,'raw_sha256':byte_sha(raw),
                     'encoder_sha256':byte_sha(zipped),'metadata_sha256':byte_sha(metadata),'parent_lineage_sha256':parent['parent_lineage_sha256'],
                     'config_sha256':self.config_sha,'at':time.time()}
            write_new(dest/'generation.json',canonical(receipt))
            self.db.execute("UPDATE variants SET state='generated',identity=?,content_sha=?,metadata_sha=? WHERE sequence=? AND state='reserved'",(identity,byte_sha(raw),byte_sha(metadata),seq))
        except BaseException as exc:
            self.db.execute("UPDATE variants SET state='failed',error=? WHERE sequence=? AND state='reserved'",(type(exc).__name__+': '+str(exc)[:400],seq));raise
        return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True);ap.add_argument('--once',action='store_true');a=ap.parse_args()
    conf=json.loads(pinned({'path':a.config,'sha256':a.config_sha},32*1024**2));need(conf['schema']==SCHEMA and conf['approved']is True,'authorized config required')
    need(conf['host']=='xbabe2' and os.uname().nodename.split('.')[0]=='xbabe2','dedicated intake host required')
    need(time.time()<conf['deadline_unix'],'controller deadline expired')
    for ref in conf['source_pins']:pinned(ref)
    body=pinned(conf['parent_registry'],32*1024**2);parents=[json.loads(s)for s in body.splitlines()]
    controller=Expansion(conf['out'],parents,target=conf['target'],backlog=conf['backlog_limit'],config_sha=a.config_sha)
    try:
        while time.time()<conf['deadline_unix']:
            for ref in conf['authority_pins']:pinned(ref)
            load=os.getloadavg()[0];mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.split(':')[0]in('MemAvailable','MemTotal')}
            disk=os.statvfs(conf['out']);free=disk.f_bavail*disk.f_frsize
            held=load>conf['max_host_load'] or mem['MemAvailable']<conf['min_available_memory_bytes'] or free<conf['min_free_disk_bytes']
            advanced=False
            if not held:advanced=controller.step()
            value={**controller.snapshot(),'state':'resource_wait' if held else ('growing' if advanced else 'downstream_backpressure_or_target'),
                   'load1':load,'available_memory':mem['MemAvailable'],'free_disk':free,'deadline_unix':conf['deadline_unix']}
            out=safe_path(conf['out']);tmp=safe_path(out/'progress.new');destination=safe_path(out/'progress.json');write_new(tmp,canonical(value));os.replace(tmp,destination)
            print(json.dumps({k:value[k]for k in ('at','state','generated','published','admitted','target')}),flush=True)
            if a.once:break
            time.sleep(min(1 if advanced else 30,max(0,conf['deadline_unix']-time.time())))
    finally:controller.close()
if __name__=='__main__':main()
