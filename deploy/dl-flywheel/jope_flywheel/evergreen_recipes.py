"""Deterministic prepared-data recipes and transformations."""
import math
import numpy as np
from .augmentation import Parent, array_sha256, roles_sha256
from .protocol import digest
from . import target_noise
from .evergreen_source import MAX_RAW, SCHEMA, TRANSFORM, byte_sha, encode, need

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
