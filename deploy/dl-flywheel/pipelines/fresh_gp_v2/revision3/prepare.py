"""CPU-only exact128 replay preparation; GET/canonicalization only, never fit/PUT."""
import gzip,io,json,sys,time
from pathlib import Path
import numpy as np
from common import *
from jope_flywheel.program_io import NativeTools
from jope_flywheel.program_inputs import dataset_object
from staged_inputs import StagedInputs,VersionCache
from role_authority import produce

INVENTORY_SHA='71f0d5129f02f95a6a9821caff080cbb9b3426c4eab844db1bbcca3ed8712030'
WORKER_SHA='626d93a5cd11830e1b7cd43bf46f3fbde42c06273e750faa6d64b69a0c037526'
CANON_SHA='69975947982bcffb0a55fde96f85b68f043e107e0d946eadffcdac7311f026c1'

class Blobs:
    def __init__(self,root):self.root=root;self.rows={};self.bytes=0;(root/'blobs').mkdir()
    def copy(self,r):
        data=bound(r,32*MIB);p=self.root/'blobs'/hashlib.sha256(data).hexdigest()
        if not p.exists():create(p,data);self.bytes+=len(data)
        need(self.bytes<=256*MIB,'bounded total cold source staging')
        descriptor={'source':dict(r,bytes=len(data)),'replica':ref(p)}
        need(r['path']not in self.rows or self.rows[r['path']]==descriptor,'source changed during cold staging')
        self.rows[r['path']]=descriptor;return data
    def mapping(self):return {'schema':'jope.flywheel-source-replicas/v1','replica_root':str(self.root),'entries':list(self.rows.values())}

def prepare(request_ref,out,client,*,hostname=None,native_factory=NativeTools):
    need((hostname or __import__('socket').gethostname().split('.')[0])=='xbabe2','derive at original source authority')
    req=obj(request_ref);need(req['schema']=='jope.fresh128-prepare-request/v2','qualified cold request schema')
    source_closure(req['source_manifest'])
    need(req['candidate_inventory']['sha256']==INVENTORY_SHA,'exact captured128 inventory')
    inventory=obj(req['candidate_inventory']);variants=inventory['variants']
    need(len(variants)==128 and [r['row']['sequence']for r in variants]==list(range(938,1066)),'exact candidate sequence')
    clock=Clock(min(time.time()+600,GLOBAL),600);root=fresh(out);(root/'data').mkdir();(root/'proofs').mkdir()
    staging=Blobs(root);generator=json.loads(staging.copy(req['generator_config']))
    parents=[json.loads(x)for x in staging.copy(generator['parent_registry']).splitlines()]
    by={p['parent_lineage_sha256']:p for p in parents};need(len(by)==len(parents)==1068,'complete parent registry')
    for r in generator['source_pins']:staging.copy(r)
    selection=[{'identity':v['row']['identity'],'sequence':v['row']['sequence'],
        'parent_lineage_sha256':v['row']['parent'],'partition':v['raw_dataset_publication']['qualification']['partition']}for v in variants]
    c={'schema':'jope.fresh-variant-gp-campaign/v1','selection':selection,'selected_count':128,'selection_sha256':digest(selection),
       'campaign_sha256':request_ref['sha256'],'generator_config':req['generator_config'],
       'parent_registry_sha256':generator['parent_registry']['sha256'],'authority_pins':generator['authority_pins'],
       'authority_root':str(root),'authority_pointer':str(root/'current.json'),'authority_deadline_unix':clock.deadline,
       'required_encoders':['kumo_tabular_s'],'owned_root':str(root),
       **{k:req[k]for k in('protocol_descriptor','vocabulary_descriptor','worker','emitter','rustc','native_canonicalizer','source_build_plan_sha256','source_commit')}}
    need(c['worker']['sha256']==WORKER_SHA and c['native_canonicalizer']['sha256']==CANON_SHA,'reviewed exact native protocol tools')
    for k in('worker','emitter','rustc','native_canonicalizer'):bound(c[k],64*MIB)
    def roles():
        p=produce(c);r=put(root/'proofs',digest(p)+'.json',p);atom(root/'current.json',canonical(r));return p
    role=roles();cache=VersionCache(root/'data/version_cache',client,clock.deadline)
    manifests={r['sha256']:obj(r)for r in req['encoding_manifests']}
    need(len(manifests)==4,'four exact completed encoder manifests')
    inputs=[]
    for item in variants:
        clock.tick();row=item['row'];identity=row['identity'];parent=by[row['parent']]
        if time.time()-role['checked_at_unix']>40:role=roles()
        need(parent['partition']==item['raw_dataset_publication']['qualification']['partition'],'current inherited role')
        # Current role is checked before accessing or copying any selected source.
        for name in('encoder','context','preparation_receipt'):
            if parent.get(name)is not None:staging.copy(parent[name])
        files={}
        for name in('encoder.jtt1.gz','metadata.json','generation.json'):
            path=Path(generator['out'])/'variants'/identity/name;r=ref(path);staging.copy(r);files[name]=r
        need(files['metadata.json']['sha256']==row['metadata_sha'],'captured metadata identity')
        raw_ref=item['raw_dataset_publication']['s3_receipt'];raw_doc=json.loads(cache.get(raw_ref,identity,'receipt'))
        objects={r['key'].split('/')[-1]:r for r in raw_doc['objects']}
        need(set(objects)==set(files)and len(raw_doc['objects'])==3,'three exact raw objects')
        s3={**objects,'receipt':raw_ref}
        eref=item['encoding_publication']['receipt'];enc=json.loads(cache.get(eref,identity,'receipt'))
        msha=enc['source_proof']['manifest']['sha256'];need(msha in manifests,'Kumo receipt belongs to exact reviewed packet')
        manifest=manifests[msha];need(manifest['worker_sha256']=='0883dd4085e01cc0e1a3353396fd7004854c4ae2e68868f1eb79a053a6cb2a42','reviewed Kumo-only producer')
        ev=next((v for v in manifest['hosts']['xbabe3']['variants']if v['identity']==identity),None)
        need(ev is not None and ev['database_row']==row,'captured selection belongs to completed encoder packet')
        record=enc['encoder_records']['kumo_tabular_s']['record']
        # Receipt source, publisher and runtime bindings originate in independent
        # manifest/config refs; never simply trust the receipt's own declarations.
        pub=req['encoding_publishers'][msha];bound(pub)
        need(enc['publisher_sha256']==pub['sha256']and record['producer']['runtime_inventory']==manifest['hosts']['xbabe3']['ofm']['runtime_inventory'],
             'independently pinned publisher/runtime')
        k={'receipt':eref,'publisher_sha256':pub['sha256'],'source_proof_sha256':digest(enc['source_proof']),
           'manifest_sha256':msha,'runtime_inventory_sha256':digest(record['producer']['runtime_inventory']),'vector':enc['objects'][0]}
        inputs.append({**row,'files':files,'s3':s3,'kumo':k})
        timing(root,'input_receipts_ready',variant=identity,sequence=row['sequence'],metadata_bytes=files['metadata.json']['bytes'])
    registry={'schema':'jope.flywheel-program-inputs/v1','generator_config_sha256':req['generator_config']['sha256'],
              'research_only':True,'teaching_eligible':False,'inputs':inputs}
    regref=put(root,'registry.json',registry);mapping=put(root,'staging-map.json',staging.mapping());c['staging_map']=mapping
    native=native_factory(c['native_canonicalizer'],clock.deadline)
    reader=StagedInputs(c,registry,native,client,clock.deadline,cache=cache)
    contexts=[]
    for row in inputs:
        clock.tick()
        if time.time()-role['checked_at_unix']>40:role=roles()
        start=time.monotonic_ns();ctx,raw,desc=reader.describe(row['identity'])
        contexts.append({'context_id':ctx.identity,'variant_identity':row['identity'],'descriptor_sha256':digest(desc),
            'partition':desc['partition'],'context':__import__('dataclasses').asdict(ctx),'descriptor':desc})
        timing(root,'native_context_derived',variant=row['identity'],context_id=ctx.identity,duration_ns=time.monotonic_ns()-start,bytes=len(raw))
    need(len({r['context_id']for r in contexts})==128,'128 distinct exact native contexts')
    for p in req['programs']:need(native.canonical_program(p['dsl'])==p['dsl']and hashlib.sha256(p['dsl'].encode()).hexdigest()==p['program_sha256'],'exact native eight-program panel')
    role=roles();cref=put(root,'contexts.json',contexts)
    for key in ('worker','emitter','rustc','native_canonicalizer'):
        need(req['execution_tools'][key]['sha256']==c[key]['sha256'],'execution replica binary identity')
        c[key]=req['execution_tools'][key]
    result={'schema':'jope.fresh128-prepared/v2','request':request_ref,'registry':regref,'staging_map':mapping,'contexts':cref,
            'selection':selection,'selection_sha256':digest(selection),'source_manifest':req['source_manifest'],
            'config_base':{k:v for k,v in c.items()if k not in('owned_root','campaign_sha256','authority_deadline_unix','authority_root','authority_pointer')},
            'programs':req['programs'],'native_closure':req['native_closure'],'execution_native_root':req['execution_native_root'],'runtime_versions':{'python':sys.version.split()[0],'numpy':str(np.__version__)},
            'source_replica_bytes':staging.bytes,'current_role_proof':role,'native_fits':0,'S3_PUTs':0,'at':time.time()}
    clock.tick();return put(root,'prepared.json',result)
