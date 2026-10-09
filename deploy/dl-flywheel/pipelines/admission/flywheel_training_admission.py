"""Owned, bounded proof preparation and separately approved exact-row CAS.

Prepare is read-only against training/generator data. Commit requires a new
immutable execution config naming the exact reviewed plan SHA and accepted IDs.
No candidate list inferred from aggregate counters can release queue capacity.
"""
import argparse,hashlib,importlib.util,json,math,os,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from jope_flywheel import training_admission as A
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.program_io import bound,read,create,bind_campaign_deadline,tick,remaining,NativeTools
SCHEMA='jope.flywheel-training-admission-controller/v1'
MIB=1024**2

def sha(b):return hashlib.sha256(b).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def doc(ref):return A.document(bound(ref,64*MIB))
def describe(path,cap=32*MIB):
    b=read(path,cap);return {'path':str(path),'sha256':sha(b),'bytes':len(b)}
def import_bound(ref,name):
    bound(ref);s=importlib.util.spec_from_file_location(name,ref['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate_config(c):
    need(set(c)=={'schema','mode','owned_root','unit','created_unix','deadline_unix','controller_sha256','source_pins',
        'snapshot_controller','neural_config','neural_unit','generator_config','parent_registry','candidate_registry','candidate_ids',
        'native_canonicalizer','aws_profile','approval','runtime_versions'},'closed execution config')
    need(c['schema']==SCHEMA and c['mode']in('prepare','commit'),'known controller mode')
    need(type(c['created_unix'])in(int,float)and type(c['deadline_unix'])in(int,float)
         and math.isfinite(c['created_unix'])and math.isfinite(c['deadline_unix'])
         and 0<c['deadline_unix']-c['created_unix']<=300,'fixed300s maximum window')
    need(c['unit'].startswith('jopedime-flywheel-admission-')and all(x.isalnum()or x in '-_'for x in c['unit']),'owned unit name')
    need(type(c['candidate_ids'])is list and len(c['candidate_ids'])==len(set(c['candidate_ids']))==16,'exact16candidate identities')
    for identity in c['candidate_ids']:A.sha256(identity)
    if c['mode']=='prepare':need(c['approval']is None,'prepare cannot imply approval')
    else:
        ap=c['approval'];need(type(ap)is dict and set(ap)=={'plan','spec','accepted_identities','authorization'}and
            ap['authorization']=='root_reviewed_exact_training_admission'and type(ap['accepted_identities'])is list
            and 0<len(ap['accepted_identities'])<=16 and len(set(ap['accepted_identities']))==len(ap['accepted_identities']),'explicit exact-row commit approval')
    return c

def resource_gate(c):
    need(socket.gethostname().split('.')[0]=='xbabe2'and os.environ.get('CUDA_VISIBLE_DEVICES')=='','xbabe2 CPU-only job required')
    line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::'))
    need(line.endswith('/'+c['unit']+'.service'),'exact owned admission unit required')
    group=Path('/sys/fs/cgroup')/line[3:].lstrip('/');quota,period=(group/'cpu.max').read_text().split()
    need(quota!='max'and 0<int(quota)<=2*int(period),'CPU2 cap')
    memory=int((group/'memory.max').read_text());need(0<memory<=2*1024**3 and(group/'memory.swap.max').read_text().strip()=='0','2GiB/swap0 cap')
    need(int((group/'pids.max').read_text())<=32,'Tasks32 cap')
    return {'cgroup':str(group),'cpu_max':[quota,period],'memory_max':memory,'cuda_visible_devices':''}

def terminal_neural(c,deadline):
    tick(deadline);r=subprocess.run(['systemctl','--user','show',c['neural_unit']+'.service','--property=ActiveState,SubState'],capture_output=True,text=True,timeout=min(10,remaining(deadline)))
    need(r.returncode==0 and len(r.stdout)<4096 and len(r.stderr)<4096,'bounded native unit state')
    fields=dict(line.split('=',1)for line in r.stdout.splitlines()if'='in line)
    need(fields.get('ActiveState')in('inactive','failed'),'native training unit must be stopped before proof preparation')
    tick(deadline);return fields

def make_spec(c):
    """Hash every actual artifact after completion; no output is synthesized."""
    neural=doc(c['neural_config']);root=A.path_safe(neural['owned_root']);need(root==Path(c['neural_config']['path']).parent,'neural config runtime identity')
    status_ref=describe(root/'status.json');status=doc(status_ref)
    need(status['schema']=='jope.flywheel-neural-status/v1'and status['state']=='complete'
        and status['config_sha256']==c['neural_config']['sha256'],'complete exact neural source required')
    result_ref=describe(root/'training/result.json');need(status['result']==result_ref,'result/status SHA binding')
    result=doc(result_ref);n=result['optimizer_steps'];need(type(n)is int and 0<n<=32 and result['state']=='completed_bounded_qualification','real complete bounded optimizer run')
    acceptance=doc(neural['snapshot_acceptance']);accepted_status=doc(neural['snapshot_status'])
    loadedref=describe(root/'loaded-receipt.json');loaded=doc(loadedref);manifestref=describe(root/'training/batch-manifest.json');manifest=doc(manifestref)
    by={r['data_run']['variant_identity']:r for r in manifest['rows']};registry=doc(c['candidate_registry'])
    need(registry['generator_config_sha256']==c['generator_config']['sha256'],'candidate inventory generator binding')
    candidates={r['identity']:r for r in registry['inputs']};need(len(candidates)==len(registry['inputs'])and set(c['candidate_ids'])=={v['variant_identity']for v in loaded['contexts'].values()},'exact loaded candidate inventory')
    rows=[]
    for identity in sorted(c['candidate_ids']):
        v=candidates[identity];row=by[identity];rows.append({k:v[k]for k in('sequence','parent','ordinal','identity','content_sha','metadata_sha')});rows[-1]['data_run_id']=row['data_run_id']
    generator=doc(c['generator_config']);parents=[A.document(b)for b in bound(c['parent_registry']).splitlines()]
    pins=list(c['source_pins'])+list(neural['source_pins'])
    for name,expected in neural['owned_sources'].items():
        ref=describe(root/name);need(ref['sha256']==expected,'native source drift');pins.append(ref)
    return {'schema':A.SCHEMA,'source_pins':pins,'database':{'path':str(Path(generator['out'])/'expansion.sqlite'),
        'config_sha256':digest({'config_sha':c['generator_config']['sha256'],'parents':parents,'target':generator['target'],'backlog':generator['backlog_limit']})},
        'generator_config':c['generator_config'],'parent_registry':c['parent_registry'],'snapshot_status':neural['snapshot_status'],
        'accepted_loaded_receipt':accepted_status['loaded_receipt'],'accepted_batch_manifest':accepted_status['batch_manifest'],
        'loaded_receipt':loadedref,'batch_manifest':manifestref,'neural_config':c['neural_config'],'neural_status':status_ref,'neural_result':result_ref,
        'neural_role_proof':describe(root/'role-proof.json'),'neural_window':describe(root/'window.json'),'planner_path':str(root/'training/batches.sqlite'),
        'batch_evidence':[describe(root/('training/batch-%04d/optimizer-evidence.json'%i))for i in range(1,n+1)],
        'checkpoints':[{'manifest':describe(root/('training/checkpoint-%04d/manifest.json'%i)),
                       'state':describe(root/('training/checkpoint-%04d/state.safetensors'%i),64*MIB)}for i in range(n+1)],
        'authority_reference':describe(Path(neural['snapshot_status']['path']).parent/'final-role-proof.json'),'rows':rows},acceptance

def prefetch(spec,store,deadline):
    refs=[]
    for ref in spec['batch_evidence']:
        for measurement in doc(ref)['measurements']:
            need(measurement['state']=='ready','all actual emissions must be measured')
            wrapper_ref=measurement['record'];wrapper=A.document(store.read(wrapper_ref));refs.extend([wrapper_ref,*wrapper['artifacts'].values()])
    loaded=doc(spec['loaded_receipt'])
    refs.extend(v['source_publications']['dataset_publication']for v in loaded['contexts'].values())
    closure=doc(spec['neural_status'])['attempt_custody'];refs.extend([closure['snapshot'],closure['receipt']])
    unique={}
    for r in refs:
        key=(r['bucket'],r['key'],r['version_id']);need(key not in unique or unique[key]==r,'artifact identity collision');unique[key]=r
    total=sum(r['bytes']for r in unique.values());need(total<=256*MIB,'admission exact-artifact256MiB cap')
    for r in unique.values():tick(deadline);store.get_version(r)
    return {'unique_objects':len(unique),'unique_bytes':total}

def admission_store(helper,client,native,deadline,custody):
    """Extend frozen snapshot reads only for the two status-pinned custody refs.

    Cache, exact-version conflict checks and byte accounting remain shared.
    No arbitrary attempt namespace access or latest-version lookup is added.
    """
    allowed={}
    for name in ('snapshot','receipt'):
        ref=helper.S.normalized_ref(custody[name])
        need(ref['bucket']==helper.BUCKET and ref['key'].startswith(helper.PREFIX+'program-attempts/flywheel/v1/'),'attempt custody namespace')
        key=(ref['bucket'],ref['key'],ref['version_id'])
        need(key not in allowed,'distinct custody refs');allowed[key]=ref
    class AdmissionStore(helper.Store):
        def get_version(self,ref):
            tick(self.deadline);ref=helper.S.normalized_ref(ref)
            key=(ref['bucket'],ref['key'],ref['version_id'])
            need(type(ref['bytes'])is int and 0<ref['bytes']<=helper.S.MAX_OBJECT,'object byte cap')
            if key not in self.cache:need(self.bytes_read+ref['bytes']<=256*MIB,'admission total read cap')
            if key not in allowed:return super().get_version(ref)
            need(ref==allowed[key],'custody exact reference changed')
            pair=(ref['sha256'],ref['bytes']);need(self.bindings.setdefault(key,pair)==pair,'conflicting exact version pins')
            if key in self.cache:return self.cache[key],key[2]
            response=self.client.get_object(Bucket=ref['bucket'],Key=ref['key'],VersionId=ref['version_id'])
            need(response.get('VersionId')==ref['version_id']and response.get('ContentLength')==ref['bytes'],'exact S3 version/size')
            data=bytearray()
            with response['Body']as stream:
                while True:
                    tick(self.deadline);piece=stream.read(min(MIB,ref['bytes']-len(data)+1))
                    if not piece:break
                    data.extend(piece);need(len(data)<=ref['bytes'],'read exceeds pinned bytes')
            tick(self.deadline);body=bytes(data);need(len(body)==ref['bytes']and sha(body)==ref['sha256'],'read SHA mismatch')
            self.cache[key]=body;self.bytes_read+=len(body);return body,key[2]
    return AdmissionStore(client,native,deadline)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);args=p.parse_args()
    c=validate_config(doc({'path':args.config,'sha256':args.config_sha}));deadline=c['deadline_unix'];bind_campaign_deadline(deadline);tick(deadline)
    need(sha(Path(__file__).read_bytes())==c['controller_sha256'],'controller source drift')
    root=A.path_safe(c['owned_root']);need(Path(__file__).parent==root and Path(args.config)==root/'config.json','owned execution root/config')
    for ref in c['source_pins']:bound(ref)
    resources=resource_gate(c)
    import platform,boto3,numpy
    from botocore.config import Config
    need({'python':platform.python_version(),'boto3':str(boto3.__version__),'numpy':str(numpy.__version__)}==c['runtime_versions'],'CPU runtime versions')
    out=root/'result';need(not out.exists(),'fresh output required');out.mkdir()
    try:
        unit=terminal_neural(c,deadline);native=NativeTools(c['native_canonicalizer'],deadline)
        helper=import_bound(c['snapshot_controller'],'admission_snapshot_helpers')
        client=boto3.Session(profile_name=c['aws_profile']).client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=2))
        if c['mode']=='prepare':spec,acceptance=make_spec(c);specraw=canonical(spec)
        else:
            # Preserve the reviewed spec byte-for-byte across separate owned
            # prepare/commit runtimes; regenerating its source paths would
            # silently change the approved plan identity.
            specraw=bound(c['approval']['spec'],16*MIB);spec=A.document(specraw)
            need(spec['neural_config']==c['neural_config']and spec['generator_config']==c['generator_config']
                 and spec['parent_registry']==c['parent_registry']and sorted(r['identity']for r in spec['rows'])==sorted(c['candidate_ids']),
                 'approved spec source/candidate binding')
            acceptance=doc(doc(c['neural_config'])['snapshot_acceptance'])
        specref=create(out/'spec.json',specraw)
        store=admission_store(helper,client,native,deadline,doc(spec['neural_status'])['attempt_custody'])
        inventory=prefetch(spec,store,deadline);role=helper.fresh_authority(acceptance,deadline);create(out/'role-proof.json',role)
        plan=A.prepare(specraw,sha(specraw),store,authority_bytes=role,authority_sha256=sha(role));tick(deadline)
        planraw=canonical(plan);planref=create(out/'plan.json',planraw)
        if c['mode']=='commit':
            expected=doc(c['approval']['plan']);need(expected==plan,'actual prepared plan differs from reviewed approval')
            ids=[x['row']['identity']for x in plan['accepted']];need(sorted(ids)==sorted(c['approval']['accepted_identities']),'approved exact identities differ')
            role=helper.fresh_authority(acceptance,deadline);terminal_neural(c,deadline);tick(deadline)
            result=A.commit(planraw,sha(planraw),spec_bytes=specraw,store=store,authority_bytes=role,authority_sha256=sha(role))
            # The DB has committed if this write is interrupted. An exact retry
            # re-verifies the same plan and returns its stored atomic evidence.
            commitref=create(out/'commit.json',canonical(result))
        else:commitref=None
        for ref in c['source_pins']:bound(ref)
        tick(deadline);status={'schema':SCHEMA,'state':'prepared'if c['mode']=='prepare'else'committed','config_sha256':args.config_sha,
            'at':time.time(),'spec':specref,'plan':planref,'commit':commitref,'accepted':len(plan['accepted']),'refused':plan['refused'],
            'inventory':inventory,'resources':resources,'native_unit':unit,'database_modified':c['mode']=='commit'}
        create(out/'status.json',canonical(status));print(json.dumps(status),flush=True)
    except BaseException as error:
        create(out/'failure.json',canonical({'schema':SCHEMA,'config_sha256':args.config_sha,'state':'failed','at':time.time(),
            'error':type(error).__name__+': '+str(error)[:2000],'mode':c['mode'],'commit_outcome':'read atomic admission table before retry'if c['mode']=='commit'else'no mutation'}));raise
if __name__=='__main__':main()
