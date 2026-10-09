#!/usr/bin/env python3
"""Publish only the fixed sealed100 prepared variants; no GP or training admission."""
import argparse
import ast
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
import stat
import struct
import sys
import time

sys.dont_write_bytecode=True
W=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_pilot_20261008')
MANIFEST_SHA='800f7c11ef9c2d55c18742de2c7c8ac10c3ac2cae58b737e7e44ec7c09fb7ea3'
WORKER_SHA='e99b75fec6dcbe704ba0297634e07c99b9c77e06fc209d8c6f74091219e0af7a'
EVIDENCE_SHA='e4bd1d1e92455bd9b4a210d4e46cddf392cc565d7bcc7c1b6bea3e48a6a070a9'
PARENT='0b9bd099d48be1afe7e180eab9543062759a34aa37d6daa1fefe010212262af6'
MISSING=['dope','tabdpt13','tabicl2','kumo_tabular_s','kumo_tabular_l','mitra_v2']
BUCKET='veox-jopedime-learning-use2'
PREFIX='experiments/gp-policy-dl/v2-trainonly/'
MIB=1024**2


def need(ok,message):
    if not ok:raise ValueError(message)

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def tick(deadline):need(time.time()<deadline,'fixed publication deadline exhausted')
def digest(x):need(type(x)is str and re.fullmatch('[0-9a-f]{64}',x),'invalid digest');return x


def read(path,cap=2*MIB):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts,'absolute confined file required')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in p.parts[1:-1]:
            other=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=other
        f=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(f,'rb') as stream:
        before=os.fstat(stream.fileno());need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=cap,'unsafe/big file')
        raw=stream.read(cap+1);after=os.fstat(stream.fileno())
    need(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),'file changed')
    return raw


def bound(ref,root=None,cap=2*MIB):
    p=Path(ref['path']);p=root/p if root is not None else p
    if root is not None:need(not Path(ref['path']).is_absolute() and '..' not in Path(ref['path']).parts,'relative source escape')
    raw=read(p,cap);need(sha(raw)==digest(ref['sha256']),'source SHA mismatch')
    if 'bytes' in ref:need(len(raw)==ref['bytes'],'source byte count')
    return raw


def numeric_vector(raw,width):
    need(raw[:8]==b'\x93NUMPY\x01\x00','NumPy1 required')
    n=struct.unpack('<H',raw[8:10])[0];need(n<=1024 and len(raw)==10+n+4*width,'vector byte closure')
    head=ast.literal_eval(raw[10:10+n].decode('ascii'))
    need(head=={'descr':'<f4','fortran_order':False,'shape':(width,)},'float32 vector contract')
    need(all(math.isfinite(v) for v in struct.unpack('<'+'f'*width,raw[10+n:])),'nonfinite encoding')


def array_hash(raw,shape):
    h=hashlib.sha256(canonical({'dtype':'<f8','shape':shape}));h.update(raw);return h.hexdigest()


def prepared(raw):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:data=f.read(MIB+1)
    need(len(data)<=MIB and data[:4]==b'JTT1' and len(data)>=17,'bounded JTT1 required')
    n,nt,p,nf=struct.unpack('<IIIB',data[4:17]);need(n==192 and nt==0 and 100<=p<=132 and nf==2,'pilot prepared shape')
    need(len(data)==17+n+8*n*(p+1),'prepared byte closure')
    inner=data[17:17+n];need(inner.count(0)==inner.count(1)==96,'inner96/96 roles')
    payload=data[17+n:];need(all(math.isfinite(v[0]) for v in struct.iter_unpack('<d',payload)),'nonfinite prepared value')
    x=b''.join(payload[i*(p+1)*8:(i*(p+1)+p)*8] for i in range(n))
    y=b''.join(payload[(i*(p+1)+p)*8:(i+1)*(p+1)*8] for i in range(n))
    return {'shape':[n,p],'x_sha256':array_hash(x,[n,p]),'y_sha256':array_hash(y,[n]),'inner_sha256':sha(inner),'inner':inner,'y_bytes':y}


def verify_sealed(root,pin,evidence):
    # Borrow only the reviewed verifier against a URI read-only SQL transaction.
    # No Ledger constructor or schema-writing method is called.
    p=root/'evidence.sqlite';read(p,16*MIB)
    for suffix in ('-wal','-shm'):
        sidecar=Path(str(p)+suffix)
        if sidecar.exists():read(sidecar,32*MIB)
    c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,timeout=.5);c.row_factory=sqlite3.Row
    try:
        c.execute('PRAGMA query_only=ON');c.execute('BEGIN')
        ledger=object.__new__(evidence.Ledger);ledger._connection=c;ledger.ledger_id=dict(c.execute('SELECT key,value FROM metadata'))['ledger_id']
        ledger.root=ledger._input_root=root;ledger._protected={'evidence.sqlite','evidence.sqlite-wal','evidence.sqlite-shm'}
        proof=ledger._verify(expected_count=pin['event_count'],expected_head=pin['head_sha256'])
        runs=list(c.execute('SELECT * FROM runs'));need(len(runs)==1 and runs[0]['seal_event_id'] is not None,'pilot must be sealed')
        run=runs[0];need(run['source_sha256']==WORKER_SHA,'sealed source worker mismatch')
        registered=json.loads(run['manifest_json']);need(registered['manifest_sha256']==MANIFEST_SHA,'sealed campaign mismatch')
        seal=json.loads(c.execute('SELECT body_json FROM events WHERE event_id=?',(run['seal_event_id'],)).fetchone()[0])
        artifacts=seal['payload']['artifacts'];need({a['path'] for a in artifacts}=={'status.json','qualification.json','events.jsonl'},'sealed artifacts closure')
        for a in artifacts:bound(a,root,cap=8*MIB)
        return proof
    finally:c.close()


def validate_variant(record,item,parent,base):
    identity=digest(item['variant_identity']);need(record['variant_identity']==identity and record['content_sha256']==item['content_sha256'] and record['index']==item['index'],'variant identity')
    need(record['schema']=='jope.flywheel-encoding-pilot/v1' and record['manifest_sha256']==MANIFEST_SHA,'variant campaign')
    need(record['gp_labels'] is None and record['teaching_eligible'] is False and record['raw_units'] is False,'pilot cannot claim labels/admission/raw units')
    need(record['completed_encoders']==['a0','hyperion_v6all'] and record['missing_encoders']==MISSING,'encoder scope')
    state=record['transform']['state'];par=state['parent'];b=record['batch']
    need(record['parent_context_sha256']==PARENT==par['context_sha256']==parent['context_sha256'],'parent context')
    need(record['lineage_sha256']==par['parent_lineage_sha256'] and record['partition']==par['partition']=='train','lineage/partition')
    need(record['transform']['variant_identity']==b['variant_identity']==identity and sha(canonical(state))==identity,'transform identity digest')
    need(state['spec']['seed']==item['index'] and state['spec']['noise_fraction']==0,'fixed unchanged-target pilot seed/noise')
    need(b['y_unchanged'] is True and b['source_row_order_unchanged'] is True and b['row_ids']==list(range(192)),'row/target preservation')
    need(b['y_sha256']==par['target_sha256']==base['y_sha256'],'parent target digest')
    need(b['row_roles']==['fit' if v==1 else 'validation' for v in base['inner']],'parent role preservation')
    need(set(record['files'])=={'encoder.jtt1.gz','a0.npy','hyperion_v6all.npy'},'source file closure')
    for name,ref in record['files'].items():need(ref['path']=='variants/'+identity+'/'+name,'source file identity path')
    return state


def load_evidence(manifest,host):
    ref=next(x for x in manifest['hosts'][host]['source_pins'] if x['path'].endswith('/jope_flywheel/evidence.py'))
    need(ref['sha256']==EVIDENCE_SHA,'evidence source pin');bound(ref)
    s=importlib.util.spec_from_file_location('pinned_pilot_evidence',ref['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def preflight(c,deadline):
    tick(deadline);host=c['host'];root=W/'runs/recovery1'/host
    raw=read(W/'manifest_recovery1.json');need(sha(raw)==MANIFEST_SHA,'campaign changed');manifest=json.loads(raw)
    need(sha(read(W/'pilot_flywheel_encode_recovery1.py'))==WORKER_SHA,'pilot worker changed')
    cfg=manifest['hosts'][host];need(cfg['out']==str(root),'pilot root')
    parent=json.loads(bound(cfg['parent_context']));need(parent['fold']==1 and parent['dataset_hash']=='9fe58a656fdb458a','parent fold identity')
    base=prepared(bound(cfg['parent_encoder']))
    status_raw=read(root/'status.json');need(sha(status_raw)==c['source_status_sha256'],'status changed');status=json.loads(status_raw)
    need(status['state']=='complete' and status['host']==host and status['manifest_sha256']==MANIFEST_SHA,'sealed completed campaign')
    need(status['completed']==status['expected']==status['unique_content_this_host']==50 and status['new_gp_labels']==0 and status['teaching_eligible'] is False,'fixed50 publication')
    items=status['variants'];need(len(items)==50 and sorted(x['index'] for x in items)==list(range(0 if host=='xbabe2' else 50,50 if host=='xbabe2' else 100)),'host index allocation')
    need(len({x['variant_identity'] for x in items})==len({x['content_sha256'] for x in items})==50,'duplicate pilot variants')
    proof=verify_sealed(root,c['ledger_anchor'],load_evidence(manifest,host));plans=[]
    for item in items:
        tick(deadline);identity=item['variant_identity'];need(item['receipt']['path']=='variants/'+identity+'/manifest.json','receipt path')
        rr=bound(item['receipt'],root);record=json.loads(rr);need(canonical(record)==rr,'source canonical receipt encoding')
        validate_variant(record,item,parent,base)
        data={name:bound(ref,root) for name,ref in record['files'].items()}
        actual=prepared(data['encoder.jtt1.gz']);batch=record['batch']
        need(actual['y_bytes']==base['y_bytes'] and actual['inner']==base['inner'],'target/inner bytes changed')
        need(actual['shape']==batch['shape'] and actual['x_sha256']==batch['x_sha256'] and actual['y_sha256']==batch['y_sha256'],'prepared bytes not bound to transform')
        need(sha(canonical({k:actual[k] for k in ('shape','x_sha256','y_sha256','inner_sha256')}))==record['content_sha256'],'content hash')
        numeric_vector(data['a0.npy'],88);numeric_vector(data['hyperion_v6all.npy'],512)
        prefix=PREFIX+'datasets/flywheel/v1/'+identity+'/'
        objects={prefix+'encoder.jtt1.gz':data['encoder.jtt1.gz'],prefix+'encodings/a0.npy':data['a0.npy'],prefix+'encodings/hyperion_v6all.npy':data['hyperion_v6all.npy']}
        metadata={'schema':'jope.flywheel-published-prepared-dataset/v1','variant_identity':identity,'content_sha256':record['content_sha256'],
                  'rights':'research','prepared':True,'raw_units':False,'independent_parent_lineages':1,'source_pilot_record':record,
                  'source_pilot_record_sha256':item['receipt']['sha256'],'source_pilot_manifest_sha256':MANIFEST_SHA,'source_worker_sha256':WORKER_SHA,
                  'publication_helper_sha256':c['helper_sha256'],'parent_context':{'sha256':cfg['parent_context']['sha256'],'context_sha256':PARENT,
                  'dataset_hash':parent['dataset_hash'],'outer_fold':parent['fold'],'outer_cut_id':parent['cut']['cut_id'],'train_rows':192,'outer_test_rows':0,
                  'inner_folds':2,'inner_counts':[96,96],'target_sha256':base['y_sha256'],'lineage_sha256':record['lineage_sha256'],'partition':record['partition']},
                  'gp_labels':None,'new_gp_labels':0,'gp_labels_inherited':False,'teaching_eligible':False,'native_context_admission':'unqualified',
                  'completed_encoders':record['completed_encoders'],'missing_encoders':MISSING,'encoder_input_provenance':cfg['parent_encoder'],
                  'limits':['Prepared T1 units retain upstream all-source scaling/imputation and float32 rounding.','Fresh GP measurements required; parent GP scores must not be inherited.','Hyperion v6all retains inherited arm-B label-exposure caveat.','One parent lineage; variants do not increase independent dataset families.'],
                  'objects':[{'bucket':BUCKET,'key':k,'sha256':sha(v),'bytes':len(v)} for k,v in objects.items()]}
        objects[prefix+'metadata.json']=canonical(metadata)+b'\n'
        plans.append({'variant_identity':identity,'index':item['index'],'content_sha256':record['content_sha256'],'objects':objects})
    need(sum(len(v) for p in plans for v in p['objects'].values())<=64*MIB,'publication size bound')
    return plans,proof


def put_verified(client,key,raw,deadline):
    tick(deadline);parts=key.removeprefix(PREFIX).split('/')
    allowed=(len(parts) in (5,6) and parts[:3]==['datasets','flywheel','v1']) or (len(parts)==5 and parts[:3]==['receipts','flywheel-datasets','v1'])
    need(key.startswith(PREFIX) and allowed and re.fullmatch('[0-9a-f]{64}',parts[3]),'publication namespace escape')
    need(len(raw)<=2*MIB,'bounded publication object')
    expected=sha(raw)
    try:r=client.put_object(Bucket=BUCKET,Key=key,Body=raw,ContentLength=len(raw),IfNoneMatch='*',Metadata={'sha256':expected});state='created'
    except Exception as e:
        if getattr(e,'response',{}).get('Error',{}).get('Code') not in ('PreconditionFailed','ConditionalRequestConflict','412'):raise
        r=client.head_object(Bucket=BUCKET,Key=key);state='existing_verified'
    version=r.get('VersionId');need(type(version)is str and version not in ('','null'),'real S3 VersionId required')
    tick(deadline);g=client.get_object(Bucket=BUCKET,Key=key,VersionId=version)
    need(g.get('VersionId')==version and g.get('ContentLength')==len(raw),'GET metadata mismatch')
    h=hashlib.sha256();n=0
    with g['Body'] as f:
        while True:
            tick(deadline);b=f.read(MIB)
            if not b:break
            n+=len(b);need(n<=len(raw),'GET byte bound');h.update(b)
    need(n==len(raw) and h.hexdigest()==expected,'GET SHA mismatch');tick(deadline)
    return {'bucket':BUCKET,'key':key,'version_id':version,'sha256':expected,'bytes':len(raw),'state':state,'verification':'exact VersionId GET + SHA256'}


def status(out,c,state,done,verified_objects,**extra):
    value={'schema':'jope.flywheel-dataset-publication-progress/v1','host':c['host'],'state':state,'at':time.time(),'expected_variants':50,
           'completed_variants':len(done),'verified_objects':verified_objects,'expected_objects':250,'new_gp_labels':0,'teaching_eligible':False,
           'parent_lineages':1,'variants':done,**extra}
    p=out/'status.tmp'
    with p.open('xb') as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(p,out/'status.json')
    print(json.dumps({k:v for k,v in value.items() if k!='variants'}),flush=True)


def publish(c,plans,proof,client,deadline,out):
    need(not out.exists() and all(not x.is_symlink() for x in out.parents),'new publication output required')
    out.mkdir(mode=0o700,parents=False,exist_ok=False);done=[];count=0
    status(out,c,'publishing',done,count,ledger=proof)
    try:
        with (out/'objects.jsonl').open('xb') as journal:
            def log(obj):
                nonlocal count
                journal.write(canonical(obj)+b'\n');journal.flush();os.fsync(journal.fileno());count+=1
            for plan in plans:
                refs=[]
                for key,raw in plan['objects'].items():
                    obj=put_verified(client,key,raw,deadline);log(obj);refs.append(obj)
                receipt={'schema':'jope.flywheel-dataset-publication/v1','variant_identity':plan['variant_identity'],'source_pilot_manifest_sha256':MANIFEST_SHA,
                         'host':c['host'],'source_ledger':proof,'objects':refs,'rights':'research','raw_units':False,'gp_labels':None,'teaching_eligible':False}
                raw=canonical(receipt)+b'\n';key=PREFIX+'receipts/flywheel-datasets/v1/'+plan['variant_identity']+'/'+sha(raw)+'.json'
                obj=put_verified(client,key,raw,deadline);log(obj)
                done.append({'index':plan['index'],'variant_identity':plan['variant_identity'],'content_sha256':plan['content_sha256'],'receipt':obj})
                status(out,c,'publishing',done,count,ledger=proof)
        need(len(done)==50 and count==250,'publication closure');tick(deadline);status(out,c,'complete',done,count,ledger=proof)
    except BaseException as e:
        status(out,c,'failed',done,count,error=type(e).__name__+': '+str(e));raise


def envelope():
    relative=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    p=Path('/sys/fs/cgroup')/relative.lstrip('/');cpu=(p/'cpu.max').read_text().split();mem=(p/'memory.max').read_text().strip();tasks=(p/'pids.max').read_text().strip()
    need(cpu[0]!='max' and int(cpu[0])/int(cpu[1])<=1 and mem!='max' and int(mem)<=512*MIB and tasks!='max' and int(tasks)<=32,'bounded CPU/memory/tasks required')
    need(os.getpriority(os.PRIO_PROCESS,0)>=19 and not os.environ.get('CUDA_VISIBLE_DEVICES'),'nice19/noGPU required')
    return {'cpu_max':cpu,'memory_max':mem,'pids_max':tasks}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
    raw=read(a.config);need(sha(raw)==a.config_sha,'config SHA');c=json.loads(raw)
    need(c['schema']=='jope.flywheel-dataset-publication-config/v1' and c['host'] in ('xbabe2','xbabe3') and socket.gethostname()==c['host'],'publication host/config')
    need(c['helper_sha256']==sha(read(Path(__file__).absolute())),'publisher source SHA')
    need(c['deadline_unix']-c['created_unix']==600 and c['created_unix']<=time.time()+5,'fixed600sec publication window')
    deadline=c['deadline_unix'];tick(deadline);limits=envelope()
    out=Path(c['out']);need(out==W/'publication_v1'/c['host'],'owned publication path')
    plans,proof=preflight(c,deadline)
    result={'host':c['host'],'state':'preflight_pass','variants':len(plans),'bytes':sum(len(v) for p in plans for v in p['objects'].values()),'ledger':proof,'envelope':limits}
    print(json.dumps(result),flush=True)
    if a.preflight_only:return
    from botocore.config import Config
    import boto3
    client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=1))
    publish(c,plans,proof,client,deadline,out)

if __name__=='__main__':main()
