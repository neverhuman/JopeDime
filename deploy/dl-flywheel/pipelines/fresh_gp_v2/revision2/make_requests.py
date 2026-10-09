#!/usr/bin/env python3
"""Local-only metadata staging/request builder. Does not authorize or launch work.

--workspace points to the captured audit workspace. --destination is a fresh
LOCAL directory; --remote is the exact new owned absolute deployment path.
The generated history inventories retain old units and require an explicit
fresh coordination JSON before they can be used (not guessed empty writers).
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path[:0]=[str(H),str(H/'runtime')]
from common import canonical,need,ROOT_PREFIX

def build(workspace,destination,remote):
    w=Path(workspace).resolve();dest=Path(destination).resolve();r=Path(remote)
    need(str(r).startswith(ROOT_PREFIX)and'..'not in r.parts,'new JopeDime source proposal')
    dest.mkdir(parents=False,exist_ok=False);(dest/'inputs').mkdir();(dest/'source').mkdir()
    def staged(path,name):
        data=Path(path).read_bytes();p=dest/'inputs'/name;p.write_bytes(data)
        return {'path':str(r/'inputs'/name),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    files=[]
    for f in sorted(list(H.glob('*.py'))+list((H/'runtime').rglob('*.py'))):
        if f.name.startswith('test_')or'borrowed'in f.relative_to(H).parts:continue
        rel=f.relative_to(H);q=dest/'source'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q)
        files.append({'path':str(r/'source'/rel),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size})
    body=canonical({'schema':'jope.fresh128-source/v2','files':files});(dest/'source-manifest.json').write_bytes(body)
    source={'path':str(r/'source-manifest.json'),'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
    old=json.loads((w/'outputs/DL_FLYWHEEL_PROGRAM_LABEL16_CONFIG.json').read_bytes())
    xb3=json.loads((w/'work/flywheel-20261008/proposals/program-label112-recovery1/frozen/shard0-config.json').read_bytes())
    candidate=staged(w/'work/heartbeat-20261009-1203/gp-scale-candidate/fresh128-input-candidates.json','candidate128.json')
    previous=staged(w/'work/heartbeat-20261009-1203/native-operation-inputs/actual-prepared/census.json','previous-global-census.json')
    historical=staged(w/'work/heartbeat-20261009-1109/historical-census-after-v4.json','historical18.json')
    native=staged(w/'work/flywheel-20261008/proposals/program-label112-recovery1/native-closure.json','native-closure.json')
    manifests=[];publishers={}
    base=w/'work/heartbeat-20261009-1203/kumo-xbabe3-next128/revision2'
    for n in range(4):
        packet=base/('packet'+str(n));paths=list((packet/'runs').glob('*/manifest.json'));need(len(paths)==1,'one frozen successful encoder manifest')
        mr=staged(paths[0],'encoder%d.json'%n);manifests.append(mr)
        publishers[mr['sha256']]=staged(packet/'kumo_publish.py','publisher%d.txt'%n)
    req={'schema':'jope.fresh128-prepare-request/v2','source_manifest':source,'candidate_inventory':candidate,
        'generator_config':old['generator_config'],'encoding_manifests':manifests,'encoding_publishers':publishers,
        'execution_tools':{k:xb3[k]for k in('worker','emitter','rustc','native_canonicalizer')},
        'execution_native_root':str(Path(xb3['worker']['path']).parent.parent),'native_closure':native,
        **{k:old[k]for k in('worker','emitter','rustc','native_canonicalizer','protocol_descriptor','vocabulary_descriptor','source_build_plan_sha256','source_commit','programs')}}
    (dest/'prepare-request.json').write_bytes(canonical(req))
    # Existing exact20 source identities (18 queues plus2 outboxes). Fresh source
    # capture is mandatory. The old census is only a permanent no-retry floor.
    captures=w/'work/heartbeat-20261009-1203/native-operation-inputs/actual-prepared'
    hist=json.loads((dest/'inputs/historical18.json').read_bytes());templates={}
    for hostname in('xbabe2','xbabe3'):
        sources=[json.loads(f.read_bytes())['source']for f in sorted(captures.glob(hostname+'-source-*.json'))]
        need(len([s for s in sources if s['kind']=='queue'])==len([s for s in hist['source_queues']if s['host']==hostname]),'all historical host queues')
        collection_path=captures/(hostname+'-collection.json')
        if collection_path.exists():units=[x['unit']for x in json.loads(collection_path.read_bytes())['units_after']]
        else:
            c=json.loads((captures/'xbabe2-collection.json').read_bytes())
            # No invented closure for a host lacking local capture. Root supplies
            # the current named unit list in its explicit coordination receipt.
            units=[]
        template={'schema':'jope.fresh128-history-inventory/v2','host':hostname,'historical_census':historical,
            'previous_global_census':previous,'sources':sources,'legacy_units':units,'legacy_controllers':[],
            'canonicalizer':(old if hostname=='xbabe2'else xb3)['native_canonicalizer'],
            'source_pins':files,'maximum_seconds':600,'requires_coordination_before_use':True}
        (dest/('history-'+hostname+'.template.json')).write_bytes(canonical(template));templates[hostname]=template
    metadata={'schema':'jope.fresh128-staging-plan/v2','remote_root':str(r),'source_manifest':source,
      'required_outboxes':[s['path']for s in templates['xbabe2']['sources']if s['kind']=='outbox'],
      'next':'Complete explicit current coordination receipt; cold collect and prepare; freeze only after both pass.',
      'host_mutations':0,'GP_executions':0,'S3_PUTs':0}
    (dest/'staging-plan.json').write_bytes(canonical(metadata));return metadata

def finalize_history(template_ref,coordination_ref,out):
    from common import obj,put
    t=obj(template_ref);c=obj(coordination_ref)
    need(c['schema']=='jope.fresh128-cold-coordination/v2'and c['host']==t['host']and
         c['root_declares_complete_writer_inventory']is True,'explicit complete current coordination required')
    need(set(t['legacy_units'])<=set(c['legacy_units'])and c['legacy_units'],'old unit inventory cannot shrink')
    need(type(c['legacy_controllers'])is list and c['root_authorization'],'actual current controller inventory')
    t.update(legacy_units=c['legacy_units'],legacy_controllers=c['legacy_controllers'],coordination=coordination_ref)
    t.pop('requires_coordination_before_use',None)
    return put(Path(out).parent,Path(out).name,t)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--destination',required=True);p.add_argument('--remote',required=True);a=p.parse_args()
    print(json.dumps(build(a.workspace,a.destination,a.remote),sort_keys=True))
