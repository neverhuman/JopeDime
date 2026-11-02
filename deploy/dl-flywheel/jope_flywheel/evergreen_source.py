"""Input validation and immutable-byte codecs for canonical evergreen."""
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
