"""Physical role authority and immutable admitted-roster joins.

No supplied request field establishes authority. Authority is derived from the
pinned generator's current target/layer files, parent registry and RegisteredInputs.
"""
from dataclasses import asdict
import math
from pathlib import Path
from .readonly_io import need, read_bound, document, canonical, sha, hash64, reference

WEIGHT = 'fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423'
CANONICALIZER = '69975947982bcffb0a55fde96f85b68f043e107e0d946eadffcdac7311f026c1'
EVALUATOR = '626d93a5cd11830e1b7cd43bf46f3fbde42c06273e750faa6d64b69a0c037526'
GENERATOR = 'cd9ee8f882735c42b49d694b99dbbc4187035804099bc49f0ca81f8ec77a9b0a'
REGISTRY = 'd3bbcc43af64e58f7944c8dce0290db309bad2718af2e82b5bc0a30d45a617a1'
KUMO_AUDIT = '0b5aaba0080c34d7ea5d20b21bff8eda8e364f0e11fbdb304347305f4ed5e564'
LICENSE = '44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d'
CODE_LICENSE = '4f321b887d0350e672e06c8a8bc85cff17ee2d1f3c360617079041ceee324c81'
NORMALIZER = '4f093f59af6cb4522c8fda77cce357ecd72d0561e3816ac261ec1c91808ca62d'
ROSTER = {'name':'kumo-s-only-v1','dimension':2051,'weight_sha256':WEIGHT}
ROLE_ROOT = Path('/home/ubuntu/scratch/jopedime-dl-training-v1/receipts')


def current_roles(generator, root):
    """Physical latest pointers + full target/role/layer joins; no copied latest claim."""
    pins=generator['authority_pins'];need(type(pins)is list and 3<=len(pins)<=32,'authority pin count')
    docs={};total=0
    for ref in pins:
        p=Path(ref['path']);need(p.is_relative_to(root),'authority outside designated physical root')
        raw=read_bound(ref,4*1024**2);total+=len(raw);need(total<=16*1024**2,'authority aggregate cap')
        need(str(p) not in docs,'duplicate authority path');docs[str(p)]=(ref,document(raw))
    targets=[v for k,v in docs.items() if '/gp-target-one-holdout-after-layers-' in k]
    roles=[v for k,v in docs.items() if v[1].get('schema')=='jope.dltrain-one-holdout-role-map/v1']
    need(len(targets)==len(roles)==1,'single current target/role')
    tref,target=targets[0];rref,role=roles[0]
    need(target.get('schema')=='jope.dltrain-gp-target-after-layers/v1','target schema')
    lref,layer=docs[target['registry']['path']]
    need(target['registry']['sha256']==lref['sha256'] and target['inputs'].get(rref['path'])==rref['sha256'],'target registry/role hash joins')
    for ref,pattern in [(tref,'gp-target-one-holdout-after-layers-v*.json'),(lref,'TRAINING-EXCLUSION-LAYERS-v*.json')]:
        choices=list(Path(ref['path']).parent.glob(pattern));need(0<len(choices)<=1024,'bounded latest authority roster')
        need(all(not x.is_symlink() and x.is_file() for x in choices),'authority latest symlink')
        newest=max(choices,key=lambda p:(p.stat().st_mtime_ns,p.name))
        need(newest==Path(ref['path']),'newer physical authority exists')
    cids=set();views=set()
    for ref in layer['layers']:
        need(ref['path'] in docs and docs[ref['path']][0]['sha256']==ref['sha256'],'layer hash join')
        obj=docs[ref['path']][1]
        if obj.get('schema')=='jope.dltrain-training-exclusion-layer/v1':
            for row in obj['entries']:
                cids.add(row['context_sha256']);views.update(row.get('view_ids',[]))
        else:
            need(obj.get('schema') in {'jope.dltrain-one-holdout-role-map/v1','jope.dl-p3-exclusion-list/v1',
                 'jope.gp-hwm-excluded-by-ruling/v1','jope.dltrain-gp-target/v1'},'unknown exclusion schema')
    need(len(target['views'])==len(set(target['views']))==target['gp_target'],'target count')
    return set(target['views']),cids,views


def license_contract(audit_ref, license_ref, code_ref):
    need(audit_ref['sha256']==KUMO_AUDIT and license_ref['sha256']==LICENSE and code_ref['sha256']==CODE_LICENSE,'fixed reviewed license pins')
    audit=document(read_bound(audit_ref,128*1024));read_bound(license_ref,65536);read_bound(code_ref,65536)
    w=audit['weights'];c=audit['code']
    need(w['sha256']==WEIGHT and w['license']=='OpenMDW-1.1' and w['repo']=='nvidia/Kumo-Tabular'
         and w['revision']=='3c3e10bbdb590ace29e7026847f92db3c603096d' and w['file']=='small/regressor.pt'
         and c['license']=='Apache-2.0' and c['license_sha256']==CODE_LICENSE,'exact approved Kumo provenance')


def publication_identity(ref):
    keys={'bucket','key','version_id','sha256','bytes'}
    need(set(ref) in (keys,keys|{'state','verification'}),'known publication reference shape')
    if set(ref)!=keys:
        need(ref['state'] in ('created','existing_verified') and ref['verification']=='exact VersionId GET + SHA256','known custody annotations')
    value={k:ref[k]for k in keys};reference(value);return value


def admitted_metadata(prepared, adoption, logical, executor, refs, normalizer):
    need(prepared['schema']=='jope.prepared-mixed-snapshot/v1' and
         executor['campaign_config']==refs['training_config'] and executor['prepared_candidate']['prepared']==refs['prepared'],
         'executor/admitted prepared binding')
    need(adoption['schema']=='jope.prepared-epoch-adoption/v1' and adoption['snapshot_sha256']==prepared['extension_sha256']
         and adoption['manifest_sha256']==sha(canonical(prepared['batch_manifest'])),'actual epoch admission proof')
    ext_raw=prepared['extension_utf8'].encode();ext=document(ext_raw)
    need(sha(ext_raw)==prepared['extension_sha256'] and ext['preprocessing_sha256']==refs['normalizer']['sha256']==NORMALIZER
         and prepared['preprocessing_sha256']==NORMALIZER,'original normalizer unchanged')
    need(normalizer['schema']=='jope.native-kumos-preprocessing/v1' and normalizer['validation_used_for_fit'] is False
         and normalizer['clip']==[-10.,10.] and normalizer['snapshot_sha256']==prepared['native_snapshot_sha256'],'native train-only normalization')
    fit=normalizer['fit_context_ids'];need(type(fit)is list and len(fit)==len(set(fit))==6857 and all(hash64(x)for x in fit),'original full fit roster')
    for name in ('mean','scale'):
        xs=normalizer[name];need(type(xs)is list and len(xs)==2051 and all(type(x)in(int,float) and math.isfinite(x) for x in xs),'finite2051normalizer')
    need(all(x>0 for x in normalizer['scale']),'positive scale')
    objective=logical['neural']['objective']
    need(objective['schema']=='jope.flywheel-native-support-objective/v1' and objective['name']=='native-support-mask-v1'
         and objective['coefficients']=={'ce':1.,'bt':.25,'list':.1,'q':.25,'feas':.1,'rt':.05}
         and logical['neural']['input_dimension']==2051,'unchanged supported objective')
    descriptors=prepared['variant_contexts'];tasks={sha(canonical(t['context'])):t for t in ext['tasks']}
    need(len(descriptors)==len(tasks)==128 and set(descriptors)==set(tasks),'complete admitted128 roster')
    need(len({d['variant_identity'] for d in descriptors.values()})==128,'unique variants')
    for cid,d in descriptors.items():
        t=tasks[cid]
        need(d['context_id']==cid==sha(canonical(d['context'])) and d['source_kind']=='flywheel_variant'
             and d['observation_dispatch']=='qualified_flywheel_worker_adapter' and d['partition']in('train','validation'), 'admitted descriptor')
        need(all(d[k]==t[k] for k in ('variant_identity','parent_lineage_sha256','context')) and
             all(d[k]==publication_identity(t[k])for k in('dataset_publication','encoder_publication')),'prepared task/descriptor drift')
    return descriptors,ext,objective,sha(canonical(fit))


def replay_descriptor(inputs, descriptor):
    ctx,raw,actual=inputs.describe(descriptor['variant_identity'])
    need(ctx.identity==descriptor['context_id'] and asdict(ctx)==descriptor['context'],'RegisteredInputs context differs')
    need(actual['parent_lineage_sha256']==descriptor['parent_lineage_sha256'] and actual['partition']==descriptor['partition']
         and actual['dataset_s3']['receipt']==descriptor['dataset_publication'],'registered parent/role/publication differs')
    return ctx,raw,actual
