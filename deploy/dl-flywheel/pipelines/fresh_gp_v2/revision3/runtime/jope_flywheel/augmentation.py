"""Deterministic, type-aware feature augmentation of admitted regression slices.

No I/O, execution, resampling, target transforms, or dataset admission occurs here.
The caller verifies the parent/admission receipt and passes the authoritative
lineage and row roles. Fit accepts exactly the named fit rows. Apply can handle
any subset/order of registered rows without changing per-row randomness. Missing
values are deliberately unsupported in this first version.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
import numpy as np
from .protocol import canonical, digest, partition, sha256

SCHEMA='jope.flywheel-augmentation/v1'
RNG='sha256-domain-PCG64-per-row/v1'
TRANSFORMS='small-drop-trainstd-noise-gaussian-nuisance/v1'
KINDS=('continuous','integer','binary','categorical')
ROLES=('fit','validation','test')

def _need(ok,message):
    if not ok:raise ValueError(message)

def _name(value):
    _need(isinstance(value,str) and 0<len(value)<=256 and value.strip()==value
          and not any(ord(c)<32 for c in value),'invalid column/identity name')
    return value

@dataclass(frozen=True)
class Feature:
    name:str
    kind:str
    dtype:str
    def __post_init__(self):
        _name(self.name);_need(self.kind in KINDS,'unknown feature kind')
        _need(isinstance(self.dtype,str),'dtype name must be a string')
        dtype=np.dtype(self.dtype)
        _need(dtype.kind in 'biuf' and self.dtype==dtype.name,'canonical real numeric dtype required')
        if self.kind=='continuous':_need(dtype.kind=='f','continuous feature requires floating dtype')
        if self.kind in ('integer','categorical'):_need(dtype.kind in 'iu','integer/category codes require integer dtype')

@dataclass(frozen=True)
class Spec:
    seed:int
    drop_fraction:float=0.0
    nuisance_fraction:float=0.0
    noise_fraction:float=0.0
    nuisance_max:int=8
    version:str=TRANSFORMS
    def __post_init__(self):
        _need(type(self.seed) is int and 0<=self.seed<2**64,'uint64 seed required')
        for value in (self.drop_fraction,self.nuisance_fraction):
            _need(type(value) in (int,float) and math.isfinite(value) and 0<=value<=.1,'feature fractions must be in [0,0.1]')
        _need(type(self.noise_fraction) in (int,float) and self.noise_fraction in (0,.01,.03),'noise fraction must be0,0.01,or0.03')
        _need(type(self.nuisance_max) is int and 0<=self.nuisance_max<=8,'at most8 nuisance columns')
        _need(self.version==TRANSFORMS,'unknown transform version')

@dataclass(frozen=True)
class Parent:
    dataset_sha256:str
    context_sha256:str
    target_sha256:str
    target_name:str
    parent_lineage_sha256:str
    split_namespace_sha256:str
    partition:str
    cut_id:str
    feature_schema_sha256:str
    row_roles_sha256:str
    admission_receipt_sha256:str
    def __post_init__(self):
        for key,value in asdict(self).items():
            if key.endswith('_sha256'):sha256(value)
        _name(self.target_name);_name(self.cut_id)
        _need(self.partition in ('train','validation'),'protected or unresolved parent role refused')
        _need(self.partition==partition(self.parent_lineage_sha256,self.split_namespace_sha256),'parent lineage partition mismatch')

def schema_sha256(features):
    _need(all(isinstance(f,Feature) for f in features),'Feature schema required')
    return digest([asdict(f) for f in features])

def _roles(roles):
    _need(isinstance(roles,dict) and bool(roles),'nonempty authoritative row-role map required')
    for row,role in roles.items():
        _need(type(row) is int and 0<=row<2**64 and role in ROLES,'invalid row identity/role')
    return tuple(sorted(roles.items()))

def roles_sha256(roles):
    return digest(_roles(roles))

def _ids(ids,n):
    ids=tuple(ids)
    _need(len(ids)==n and len(set(ids))==n and all(isinstance(i,(int,np.integer)) and not isinstance(i,(bool,np.bool_)) and 0<=i<2**64 for i in ids),'unique uint64 row IDs required')
    return tuple(int(i)for i in ids)

def array_sha256(array):
    array=np.ascontiguousarray(array)
    _need(not array.dtype.hasobject,'object array is not a reproducible numeric artifact')
    h=hashlib.sha256(canonical({'dtype':array.dtype.str,'shape':list(array.shape)}))
    h.update(array.tobytes(order='C'));return h.hexdigest()

def _matrix(x,features,target_name):
    _need(isinstance(x,np.ndarray) and x.ndim==2 and x.dtype in (np.dtype('float32'),np.dtype('float64')),'X must be a float32/float64 matrix')
    _need(len(features)==x.shape[1] and len(features)>=1,'feature schema width mismatch')
    _need(all(isinstance(f,Feature)for f in features),'Feature schema required')
    names=[f.name for f in features]
    _need(len(set(names))==len(names) and target_name not in names,'duplicate columns or target present in feature inputs')
    _need(np.isfinite(x).all(),'nonfinite feature input; missingness augmentation is not enabled')
    for j,feature in enumerate(features):
        if feature.kind=='binary':_need(np.isin(x[:,j],[0,1]).all(),'binary feature outside0/1')
        if feature.kind in ('integer','categorical'):
            info=np.iinfo(feature.dtype)
            _need(np.equal(x[:,j],np.floor(x[:,j])).all()
                  and all(info.min<=int(v)<=info.max for v in np.unique(x[:,j])),
                  'invalid integer/category code')

def _rng(spec,parent,stream,row=None):
    seed=digest({'schema':RNG,'seed':spec.seed,'parent_context_sha256':parent.context_sha256,
                 'target_sha256':parent.target_sha256,'stream':stream,'row_id':row})
    return np.random.Generator(np.random.PCG64(int(seed,16)))

@dataclass(frozen=True)
class Fitted:
    parent:Parent
    spec:Spec
    features:tuple[Feature,...]
    row_roles:tuple[tuple[int,str],...]
    fit_row_ids:tuple[int,...]
    fit_x_sha256:str
    input_dtype:str
    retained_columns:tuple[int,...]
    dropped_columns:tuple[int,...]
    train_std:tuple[float,...]
    nuisance_names:tuple[str,...]
    numpy_version:str
    schema:str=SCHEMA
    rng:str=RNG
    @property
    def identity(self):return digest(asdict(self))
    @property
    def column_mapping(self):
        return tuple({'output_index':j,'name':self.features[i].name,'kind':self.features[i].kind,
                      'source_index':i,'source_dtype':self.features[i].dtype}
                     for j,i in enumerate(self.retained_columns))+tuple(
               {'output_index':len(self.retained_columns)+j,'name':name,'kind':'continuous',
                'source_index':None,'source_dtype':None}for j,name in enumerate(self.nuisance_names))
    def manifest(self):
        return {'schema':SCHEMA,'variant_identity':self.identity,'state':asdict(self),
                'column_mapping':self.column_mapping,'fit_row_ids_sha256':digest(self.fit_row_ids),
                'reconstruction':'Re-run fit on the exact admitted fit rows and input SHA; unknown executable transforms are unsupported.'}

def fit(x_fit,*,features,row_ids,row_roles,parent,spec):
    """Fit on every and only registered inner-fit row; never consume y or heldout X."""
    _need(isinstance(parent,Parent) and isinstance(spec,Spec),'typed parent/spec required')
    features=tuple(features);_matrix(x_fit,features,parent.target_name)
    _need(schema_sha256(features)==parent.feature_schema_sha256,'parent feature schema mismatch')
    roles=_roles(row_roles);_need(digest(roles)==parent.row_roles_sha256,'parent row roles mismatch')
    ids=_ids(row_ids,len(x_fit));allowed={i for i,role in roles if role=='fit'}
    _need(len(ids)>=2 and set(ids)==allowed,'fit input must contain exactly authoritative fit rows')
    p=x_fit.shape[1]
    drop=min(p-1,math.floor(p*spec.drop_fraction))
    nuisance=min(spec.nuisance_max,math.floor(p*spec.nuisance_fraction))
    dropped=tuple(sorted(int(i)for i in _rng(spec,parent,'drop-columns').choice(p,size=drop,replace=False)))
    retained=tuple(i for i in range(p)if i not in dropped)
    std=tuple(float(np.std(x_fit[:,i],dtype=np.float64)) if f.kind=='continuous' else 0.
              for i,f in enumerate(features))
    _need(all(math.isfinite(s) and s>=0 for s in std),'nonfinite train-only feature scale')
    names=tuple('__jope_nuisance_'+digest({'parent':asdict(parent),'seed':spec.seed})[:12]+'_'+str(i)for i in range(nuisance))
    _need(not set(names)&{f.name for f in features},'synthetic column name collision')
    return Fitted(parent,spec,features,roles,ids,array_sha256(x_fit),x_fit.dtype.name,
                  retained,dropped,std,names,np.__version__)

@dataclass(frozen=True)
class Batch:
    x:np.ndarray
    y:np.ndarray
    row_ids:tuple[int,...]
    row_roles:tuple[str,...]
    column_mapping:tuple[dict,...]
    receipt:dict

def apply(x,y,*,row_ids,state):
    """Apply to registered rows; y and row ordering/roles are copied unchanged."""
    _need(isinstance(state,Fitted) and state.schema==SCHEMA and state.rng==RNG,'unknown fitted state')
    _need(state.numpy_version==np.__version__,'NumPy version differs from fitted RNG environment')
    _matrix(x,state.features,state.parent.target_name)
    _need(x.dtype.name==state.input_dtype,'matrix dtype differs from fit')
    ids=_ids(row_ids,len(x));roles=dict(state.row_roles)
    _need(all(i in roles for i in ids),'row outside registered context')
    _need(isinstance(y,np.ndarray) and y.ndim==1 and len(y)==len(x) and y.dtype.kind in 'iuf'
          and np.isfinite(y).all(),'finite numeric regression target vector required')
    retained=state.retained_columns;out=np.empty((len(x),len(retained)+len(state.nuisance_names)),dtype=x.dtype)
    out[:,:len(retained)]=x[:,retained]
    continuous=[(j,i)for j,i in enumerate(retained)if state.features[i].kind=='continuous' and state.train_std[i]>0]
    with np.errstate(over='ignore',invalid='ignore'):
        for row_index,row_id in enumerate(ids):
            if state.spec.noise_fraction and continuous:
                noise=_rng(state.spec,state.parent,'feature-noise',row_id).standard_normal(len(continuous))
                for z,(j,i)in zip(noise,continuous):out[row_index,j]+=z*state.spec.noise_fraction*state.train_std[i]
            if state.nuisance_names:
                out[row_index,len(retained):]=_rng(state.spec,state.parent,'nuisance-columns',row_id).standard_normal(len(state.nuisance_names))
    _need(np.isfinite(out).all(),'augmentation produced nonfinite values; refuse variant')
    yy=y.copy();mapping=state.column_mapping;output_roles=tuple(roles[i]for i in ids)
    content={'x_sha256':array_sha256(out),'y_sha256':array_sha256(yy),'row_ids':ids,
             'row_roles':output_roles,'column_mapping':mapping}
    receipt={'schema':'jope.flywheel-augmentation-batch/v1','variant_identity':state.identity,
             'content_sha256':digest(content),'input_x_sha256':array_sha256(x),**content,
             'shape':list(out.shape),'dtype':out.dtype.name,'y_unchanged':np.array_equal(y,yy),
             'source_row_order_unchanged':True,'feature_values_changed':bool(state.dropped_columns or state.nuisance_names or not np.array_equal(x,out))}
    return Batch(out,yy,ids,output_roles,mapping,receipt)
