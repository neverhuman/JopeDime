"""Versioned regression target-noise variants; never reuse parent GP scores.

The noise fraction is Gaussian standard deviation divided by inner-fit target
standard deviation, not a maximum per-value perturbation. Every registered row
receives independent deterministic noise. Fitted scale never sees validation y.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .augmentation import Parent, _ids, _roles, array_sha256
from .protocol import digest

SCHEMA='jope.flywheel-target-noise/v1'
RNG='sha256-PCG64-target-noise-per-row/v1'

def need(ok,message):
    if not ok:raise ValueError(message)

def vector(y):
    need(isinstance(y,np.ndarray) and y.ndim==1 and y.dtype in (np.dtype('float32'),np.dtype('float64')) and np.isfinite(y).all(),
         'finite float32/float64 regression target required')

@dataclass(frozen=True)
class TargetNoise:
    parent:Parent
    feature_variant_sha256:str
    seed:int
    fraction:float
    fit_target_sha256:str
    fit_row_ids:tuple[int,...]
    row_roles:tuple[tuple[int,str],...]
    train_std:float
    dtype:str
    numpy_version:str
    schema:str=SCHEMA
    rng:str=RNG
    def __post_init__(self):
        from .protocol import sha256
        need(type(self.parent)is Parent,'typed parent required')
        sha256(self.feature_variant_sha256);sha256(self.fit_target_sha256)
        need(self.schema==SCHEMA and self.rng==RNG,'unknown target transform')
        need(type(self.seed)is int and 0<=self.seed<2**64,'uint64 seed required')
        need(type(self.fraction)in(int,float) and math.isfinite(self.fraction) and 0<=self.fraction<=.05,'target noise fraction must lie in [0,0.05]')
        need(type(self.train_std)in(int,float) and math.isfinite(self.train_std) and self.train_std>=0 and (self.fraction==0 or self.train_std>0),'invalid fitted target scale')
        need(type(self.row_roles)is tuple and len(dict(self.row_roles))==len(self.row_roles),'unique immutable row roles required')
        roles=_roles(dict(self.row_roles))
        need(roles==self.row_roles and digest(roles)==self.parent.row_roles_sha256,'parent roles differ')
        need(type(self.fit_row_ids)is tuple and self.fit_row_ids==tuple(i for i,r in roles if r=='fit') and len(self.fit_row_ids)>=2,'exact sorted inner-fit row identities required')
        need(self.dtype in ('float32','float64') and type(self.numpy_version)is str and bool(self.numpy_version),'invalid runtime/dtype')
    @property
    def identity(self):return digest(asdict(self))
    def manifest(self):
        return {'schema':SCHEMA,'target_transform_identity':self.identity,'state':asdict(self),
            'noise_definition':'Normal(0, fraction * inner-fit target population std); unbounded Gaussian, no clipping',
            'fresh_encodings_required':True,'fresh_gp_evaluations_required':True,'parent_scores_inheritable':False,
            'partition':self.parent.partition,'parent_lineage_sha256':self.parent.parent_lineage_sha256}

def fit(y_fit,*,row_ids,row_roles,parent,feature_variant_sha256,seed,fraction):
    from .protocol import sha256
    need(isinstance(parent,Parent),'verified typed parent required');sha256(feature_variant_sha256)
    need(type(seed)is int and 0<=seed<2**64,'uint64 seed required')
    need(type(fraction)in(int,float) and math.isfinite(fraction) and 0<=fraction<=.05,'target noise fraction must lie in [0,0.05]')
    vector(y_fit);roles=_roles(row_roles);ids=_ids(row_ids,len(y_fit))
    need(digest(roles)==parent.row_roles_sha256,'parent roles differ')
    need(len(ids)>=2 and set(ids)=={i for i,role in roles if role=='fit'},'exact inner-fit targets only')
    # Canonical order makes fitted identity independent of caller batch order.
    order=np.argsort(np.asarray(ids,dtype=np.uint64));yy=y_fit[order];ids=tuple(ids[int(i)]for i in order)
    std=float(np.std(yy,dtype=np.float64));need(math.isfinite(std),'nonfinite target scale')
    need(fraction==0 or std>0,'constant target cannot yield a nonzero-noise variant')
    return TargetNoise(parent,feature_variant_sha256,seed,float(fraction),array_sha256(yy),ids,roles,std,y_fit.dtype.name,str(np.__version__))

def apply(y,*,row_ids,state):
    """Caller must first bind these y bytes/row IDs to the admitted parent receipt.

    Fitting deliberately never reads validation y; per-subset source authentication
    therefore remains the dataset loader's responsibility, not an inferred claim.
    """
    need(isinstance(state,TargetNoise) and state.schema==SCHEMA and state.rng==RNG,'unknown target transform')
    state.__post_init__()
    need(state.numpy_version==str(np.__version__),'NumPy RNG version mismatch')
    vector(y);need(y.dtype.name==state.dtype,'target dtype differs from fit')
    ids=_ids(row_ids,len(y));roles=dict(state.row_roles)
    need(all(i in roles for i in ids),'unknown target row')
    out=y.copy()
    with np.errstate(over='ignore',invalid='ignore'):
        if state.fraction:
            state_identity=state.identity
            for pos,row in enumerate(ids):
                seed=digest({'domain':RNG,'state_sha256':state_identity,'row_id':row})
                z=np.random.Generator(np.random.PCG64(int(seed,16))).standard_normal()
                out[pos]+=z*state.fraction*state.train_std
    need(np.isfinite(out).all(),'target transform overflow')
    content=array_sha256(out)
    return out,{'schema':SCHEMA,'target_transform_identity':state.identity,'input_target_sha256':array_sha256(y),
        'target_sha256':content,'target_identity':digest({'parent_target_sha256':state.parent.target_sha256,
            'transform':state.identity,'target_sha256':content,'row_ids':ids}),
        'row_ids':ids,'row_roles':tuple(roles[i]for i in ids),'targets_changed':not np.array_equal(y,out),
        'parent_scores_inheritable':False,'teaching_eligible':False}
