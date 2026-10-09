"""Read-only S3 snapshot controller. No training, GP execution, PUT or DB admission.

A reviewed config pins exact GP campaign/result, encoder/data publication and
consumer sources. Run only under the fixed2CPU/2GiB/noGPU/300s systemd envelope.
The output directory must be new. Sparse membership is explicitly accounted.
"""
import argparse,hashlib,importlib.util,io,json,math,os,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from jope_flywheel import training_snapshot_kumos as S
from jope_flywheel.program_io import NativeTools,bound,bind_campaign_deadline,create,remaining,tick
from jope_flywheel.protocol import Context,canonical,digest
from jope_flywheel.program_gate import request_identity

SCHEMA='jope.flywheel-kumos-snapshot-controller/v1'
BUCKET='veox-jopedime-learning-use2'
PREFIX='experiments/gp-policy-dl/v2-trainonly/'
MIB=1024**2

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(body):return hashlib.sha256(body).hexdigest()

def resource_gate():
    need(socket.gethostname().split('.')[0]=='xbabe2','snapshot host must be xbabe2')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be disabled')
    line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'))
    group=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
    quota,period=(group/'cpu.max').read_text().split()
    need(quota!='max'and int(quota)<=2*int(period),'CPU2 cap required')
    need(int((group/'memory.max').read_text())<=2*1024**3 and (group/'memory.swap.max').read_text().strip()=='0','memory2GiB/swap0 cap required')
    return {'cgroup':str(group),'cpu_max':[quota,period],'memory_max':int((group/'memory.max').read_text()),'cuda_visible_devices':''}

class Store:
    def __init__(self,client,native,deadline):
        self.client,self.nativecanonical_json,self.deadline=client,native.canonical_json,deadline
        self.cache={};self.bindings={};self.bytes_read=0
    def get_version(self,ref):
        tick(self.deadline);ref=S.normalized_ref(ref)
        need(ref['bucket']==BUCKET and any(ref['key'].startswith(PREFIX+x)for x in
            ('datasets/flywheel/v1/','receipts/flywheel-datasets/v1/','program-results/flywheel/v1/')),'read-only source namespace')
        key=(ref['bucket'],ref['key'],ref['version_id']);pair=(ref['sha256'],ref['bytes'])
        need(self.bindings.setdefault(key,pair)==pair,'conflicting exact version pins')
        need(type(ref['bytes'])is int and 0<ref['bytes']<=S.MAX_OBJECT,'object byte cap')
        if key in self.cache:return self.cache[key],key[2]
        need(self.bytes_read+ref['bytes']<=S.MAX_READ,'total read cap')
        r=self.client.get_object(Bucket=ref['bucket'],Key=ref['key'],VersionId=ref['version_id'])
        need(r.get('VersionId')==ref['version_id']and r.get('ContentLength')==ref['bytes'],'exact S3 version/size')
        b=bytearray()
        with r['Body']as stream:
            while True:
                tick(self.deadline);piece=stream.read(min(MIB,ref['bytes']-len(b)+1))
                if not piece:break
                b.extend(piece);need(len(b)<=ref['bytes'],'read exceeds pinned bytes')
        tick(self.deadline);body=bytes(b);need(len(body)==ref['bytes']and sha(body)==ref['sha256'],'read SHA mismatch')
        self.cache[key]=body;self.bytes_read+=len(body);return body,key[2]
    def read(self,ref):return self.get_version(ref)[0]
    def lookup_ref(self,key):
        tick(self.deadline);path=PREFIX+'program-results/flywheel/v1/'+key+'/record.json'
        try:r=self.client.head_object(Bucket=BUCKET,Key=path)
        except Exception as e:
            tick(self.deadline)
            if getattr(e,'response',{}).get('Error',{}).get('Code')in('404','NoSuchKey','NotFound'):return None
            raise
        tick(self.deadline)
        ref={'bucket':BUCKET,'key':path,'version_id':r.get('VersionId'),'sha256':r.get('Metadata',{}).get('sha256'),'bytes':r.get('ContentLength')}
        need(type(ref['bytes'])is int and 0<ref['bytes']<=2*MIB,'result wrapper cap')
        self.read(ref);return ref


def frozen_expected(campaign,result):
    """Account all128 slots, including unknown/error/not-started. Never infer success."""
    variants=campaign['variants'];programs=campaign['programs']
    need(len(variants)==16 and len({x['identity']for x in variants})==16,'exact16 distinct variants')
    need(len({x['parent_lineage_sha256']for x in variants})==16,'distinct parent lineages')
    need(len(programs)==8 and len({x['program_sha256']for x in programs})==8,'exact8 programs')
    need(result['schema']in('jope.flywheel-program-label-results/v1','jope.flywheel-program-label-progress/v1'),'GP outcome schema')
    expected={}
    for j,p in enumerate(programs):
        need(sha(p['dsl'].encode())==p['program_sha256'],'canonical program hash')
        for i,v in enumerate(variants):expected[16*j+i]={'variant_identity':v['identity'],'program_sha256':p['program_sha256'],'partition':v['partition']}
    observed={}
    for row in result['outcomes']:
        slot=row['slot'];need(type(slot)is int and slot in expected and slot not in observed,'unknown/duplicate slot')
        need(all(row[k]==v for k,v in expected[slot].items()),'slot membership/partition changed')
        observed[slot]=row
    return expected,observed


def artifact_budget(refs):
    seen={}
    for source in refs:
        ref=S.normalized_ref(source);key=(ref['bucket'],ref['key'],ref['version_id']);bound=(ref['sha256'],ref['bytes'])
        need(type(ref['bytes'])is int and 0<ref['bytes']<=S.MAX_OBJECT,'artifact size cap')
        need(seen.setdefault(key,bound)==bound,'same version has inconsistent size/SHA')
    total=sum(v[1]for v in seen.values());need(total<=S.MAX_READ,'unique artifact inventory exceeds configured read cap; refuse load')
    return {'unique_objects':len(seen),'unique_bytes':total,'cap_bytes':S.MAX_READ,
            'objects':[{'bucket':k[0],'key':k[1],'version_id':k[2],'sha256':v[0],'bytes':v[1]}for k,v in sorted(seen.items())]}


def fresh_authority(c,deadline):
    body=bound(c['authority_helper']);name='snapshot_role_checker'
    spec=importlib.util.spec_from_file_location(name,c['authority_helper']['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mraw=bound(c['encoder_manifest']);proof=module.check(json.loads(mraw),sha(mraw))
    wait=proof['checked_at_unix']+.10-time.time()
    need(wait<=1 and time.time()+max(0,wait)<deadline,'clock boundary/deadline guard')
    if wait>0:time.sleep(wait)
    need(0<=time.time()-proof['checked_at_unix']<30,'fresh role check age')
    tick(deadline);return canonical(proof)


def make_spec(c,campaign,result,store,native):
    expected,observed=frozen_expected(campaign,result)
    variants={x['identity']:x for x in campaign['variants']}
    enc=S.document(bound(c['encoder_publication']));data=S.document(bound(c['dataset_publication']))
    need(enc['state']=='complete'and enc['completed_variants']==16,'qualified16 encoder publication required')
    erefs={x['variant_identity']:x['receipt']for x in enc['variants']};drefs={x['variant_identity']:x['receipt']for x in data['variants']}
    need(set(erefs)==set(variants),'encoder/GP variant membership differs')
    spec=json.loads(json.dumps(c['snapshot_base']));spec['tasks']=[];refs=[];coverage=[]
    contract={'evaluator_sha256':campaign['worker']['sha256'],
        'vocabulary_sha256':sha(native.canonical_json(campaign['vocabulary_descriptor'])),
        'protocol_sha256':sha(native.canonical_json(campaign['protocol_descriptor'])),'units':'prepared_synthetic_target_units'}
    spec['evaluation_contract']=contract
    for identity,v in sorted(variants.items()):
        er,dr=erefs[identity],drefs[identity];ep=S.document(store.read(er));dp=S.document(store.read(dr));refs.extend([er,dr,*[x for x in ep['objects']if x['key'].endswith('/encodings/kumo_tabular_s.npy')],*dp['objects']])
        mr=next(x for x in dp['objects']if x['key'].endswith('/metadata.json'));meta=S.document(store.read(mr))
        need(meta['variant_identity']==identity and meta['partition']==v['partition']and meta['parent_lineage_sha256']==v['parent_lineage_sha256'],'metadata/current campaign binding')
        cut={'schema':'jope.flywheel-prepared-two-fold-cut/v1','variant_identity':identity,'metadata_sha256':mr['sha256'],
            'dataset_sha256':meta['content_sha256'],'target_sha256':meta['target_sha256'],'split_sha256':meta['split_sha256'],
            'protocol_sha256':contract['protocol_sha256']}
        ctx=Context('regression',cut['dataset_sha256'],cut['target_sha256'],sha(native.canonical_json(cut)),contract['evaluator_sha256'],
            contract['vocabulary_sha256'],contract['protocol_sha256'],cut['split_sha256'],'rmse',contract['units'],'minimize')
        task={'variant_identity':identity,'context':ctx.__dict__,'dataset_publication':dr,'encoder_publication':er,'program_results':[]}
        for j,p in enumerate(campaign['programs']):
            slot=next(s for s,x in expected.items()if x['variant_identity']==identity and x['program_sha256']==p['program_sha256'])
            row=observed.get(slot);state='not_started'if row is None else row['state'];key=request_identity(ctx,p['dsl'])
            item={'slot':slot,'variant_identity':identity,'context_id':ctx.identity,'program_sha256':p['program_sha256'],
                'evaluation_key':key,'campaign_state':state,'snapshot_state':'excluded_unmeasured','partition':v['partition']}
            if state=='ready':
                need(row['evaluation_key']==key and row['context_id']==ctx.identity,'ready outcome identity differs')
                ref=(row.get('publication')or{}).get('record')or store.lookup_ref(key)
                need(ref is not None,'ready result absent from S3')
                wrapper=S.document(store.read(ref));need(wrapper['context']==ctx.__dict__ and wrapper['evaluation_key']==key
                    and wrapper['program_sha256']==p['program_sha256'],'wrapper task/program identity')
                need(set(wrapper['artifacts'])==set(S.ARTIFACTS),'seven artifact descriptors required')
                refs.extend([ref,*wrapper['artifacts'].values()]);task['program_results'].append(ref)
                item.update(snapshot_state='included_pending_loader_verification',wrapper=ref)
            coverage.append(item)
        spec['tasks'].append(task)
    need(len(coverage)==128,'complete slot accounting')
    included=sum(x['snapshot_state']=='included_pending_loader_verification'for x in coverage)
    need(included>0,'no measured programs; do not load an empty snapshot')
    need(c['allow_sparse']is True or included==128,'full128 programs required unless sparse explicitly configured')
    budget=artifact_budget(refs)  # before downloading any program binaries
    return spec,{'expected_program_results':128,'included_program_results':included,'sparse':included<128,
        'slots':sorted(coverage,key=lambda x:x['slot']),'artifact_inventory':budget}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);a=p.parse_args()
    c=S.document(bound({'path':a.config,'sha256':a.config_sha}));deadline=c['deadline_unix'];bind_campaign_deadline(deadline)
    need(c['schema']==SCHEMA and c['approved_for_load']is True and 0<deadline-c['created_unix']<=300,'reviewed immutable300s window required')
    tick(deadline);need(sha(Path(__file__).read_bytes())==c['controller_sha256'],'controller source drift')
    for ref in c['source_pins']:bound(ref)
    resources=resource_gate()
    import platform,boto3,numpy
    versions={'python':platform.python_version(),'numpy':str(numpy.__version__),'boto3':str(boto3.__version__)}
    need(versions==c['runtime_versions'],'pinned CPU runtime versions required');resources['runtime_versions']=versions
    out=Path(c['out']);need(not out.exists()and all(not x.is_symlink()for x in(out,*out.parents)),'new owned output required');out.mkdir(parents=False)
    try:
        campaign=S.document(bound(c['gp_campaign']));result=S.document(bound(c['gp_results']))
        need(result['config_sha256']==c['gp_campaign']['sha256'],'GP outcomes/config identity')
        native=NativeTools(c['native_canonicalizer'],deadline)
        import boto3
        from botocore.config import Config
        client=boto3.Session(profile_name=c['aws_profile']).client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=2))
        store=Store(client,native,deadline);spec,coverage=make_spec(c,campaign,result,store,native)
        sraw=canonical(spec);spec_ref=create(out/'snapshot.json',sraw);create(out/'coverage.json',canonical(coverage))
        # Prefetch the exact full object inventory before taking the120s role proof;
        # this counts every byte and pins every version, including shared binaries.
        for ref in coverage['artifact_inventory']['objects']:store.get_version(ref)
        role=fresh_authority(c,deadline);create(out/'role-proof.json',role)
        loaded=S.load_snapshot(sraw,sha(sraw),store,registry_bytes=bound(c['parent_registry']),authority_bytes=role,
            authority_sha256=sha(role),grammar_bytes=bound(c['grammar']))
        tick(deadline)
        receipt_ref=S.save_receipt(loaded,out/'loaded-receipt.json')
        inventory=bound(c['pending_inventory']);batch_manifest=S.batch_manifest(loaded,pending_inventory_bytes=inventory,pending_inventory_sha256=c['pending_inventory']['sha256'])
        batch_ref=create(out/'batch-manifest.json',canonical(batch_manifest))
        # A final independent role reread is recorded; changing support refuses seal.
        final=fresh_authority(c,deadline)
        old=S.document(role);new=S.document(final)
        need(old['support']==new['support']and old['authority_pins']==new['authority_pins'],'role support changed while loading')
        create(out/'final-role-proof.json',final)
        for ref in c['source_pins']:bound(ref)
        tick(deadline)
        status={'schema':SCHEMA,'state':'loaded'if loaded.data else'no_optimizer_universe','config_sha256':a.config_sha,
            'at':time.time(),'deadline_unix':deadline,'snapshot':spec_ref,'loaded_receipt':receipt_ref,'batch_manifest':batch_ref,'resources':resources,
            'coverage':{'expected_program_results':128,'included_program_results':coverage['included_program_results'],'sparse':coverage['sparse']},
            'unique_artifact_bytes':coverage['artifact_inventory']['unique_bytes'],'S3_bytes_read':store.bytes_read,
            'train_data_content_sha256':loaded.receipt['train_data_content_sha256'],'optimizer_updates':0,
            'database_admission_performed':False,'exhaustive_epochs_performed':0,'roster':S.COMPONENTS,
            'declared_data_runs':len(batch_manifest['rows']),'pending_data_runs':sum(x['status']=='pending'for x in batch_manifest['rows']),
            'scope':'First qualified16 of128 flywheel variants; original7120 corpus remains separate and outstanding.'}
        create(out/'status.json',canonical(status));print(json.dumps(status),flush=True)
    except BaseException as error:
        create(out/'failure.json',canonical({'schema':SCHEMA,'config_sha256':a.config_sha,'state':'failed','at':time.time(),
            'error':type(error).__name__+': '+str(error)[:2000],'optimizer_updates':0,'database_admission_performed':False}))
        raise

if __name__=='__main__':main()
