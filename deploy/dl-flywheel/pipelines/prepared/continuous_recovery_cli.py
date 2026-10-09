"""Immutable continuous research runner under an inherited global deadline."""
import hashlib,json,os,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True


def validate_permit(c):
    p=c['research_optimizer_admission']
    expected={'schema':'jope.continuous-research-optimizer-admission/v1','approved_for_bounded_research':True,
      'production_admission':False,'native_hwm_promotion':False,'snapshot_sha256':c['snapshot']['sha256'],
      'initial_extension_sha256':c['extension']['sha256'],'preprocessing_sha256':c['preprocessing']['sha256'],
      'origin_parameter_sha256':c['parameter_sha256'],'objective_sha256':hashlib.sha256(json.dumps(c['neural']['objective'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
      'global_deadline_unix':c['global_deadline_unix'],'maximum_new_updates':50000,'input_roster':{'kumo_tabular_s':2051},
      'generated_predictions_are_rewards':False,'variant_ce_teacher_rule':'disabled_for_unpromoted_variants/v1',
      'future_adoption':'epoch-boundary exact verified append-only measured extensions; same base, normalizer and lineage roles'}
    if p!=expected:raise ValueError('closed explicit continuous research permit')
    return p


def retain_tasks(old,new):
    """Append-only measured custody: no old program/result/input may disappear."""
    rows={t['variant_identity']:t for t in new['tasks']}
    for t in old['tasks']:
        if t['variant_identity']not in rows:raise ValueError('candidate removes established variant')
        r=rows[t['variant_identity']]
        for k in set(t)-{'program_results'}:
            if r[k]!=t[k]:raise ValueError('candidate changes established input/role/custody')
        if not all(v in r['program_results']for v in t['program_results']):raise ValueError('candidate removes established measured program')


def main():
    execution_root=Path(__file__).resolve().parent
    execution_raw=(execution_root/'execution-config.json').read_bytes();execution=json.loads(execution_raw)
    if execution['schema']!='jope.same-campaign-execution-config/v1':raise ValueError('closed execution schema')
    if set(execution)!={'schema','campaign_root','campaign_config','owned_sources','transfer_capture','prepared_candidate','stopped_owners','cold_reservation_permit'}:raise ValueError('closed execution keys')
    root=Path(execution['campaign_root']);sys.path[:0]=[str(execution_root),str(root),str(root/'consumer'),str(root/'runtime')]
    from jope_flywheel.program_io import read,bound,sha,need,create,tick,bind_campaign_deadline
    from jope_flywheel.protocol import canonical
    raw=bound(execution['campaign_config']);c=json.loads(raw);pin=sha(raw);w=json.loads(read(root/'window.json'))
    need(Path(execution['campaign_config']['path'])==root/'config.json' and len(sys.argv)==3 and sys.argv[1]=='--execution-sha' and sha(execution_raw)==sys.argv[2], 'immutable execution configuration')
    for rel,h in execution['owned_sources'].items():
        p=Path(rel);need(not p.is_absolute() and '..'not in p.parts and sha(read(execution_root/p))==h,'execution source closure')
    need(c['schema']=='jope.continuous-neural-campaign/v1'and c['owned_root']==str(root)and c['host']==socket.gethostname().split('.')[0]=='xbabe2','owned resumed campaign')
    original=json.loads(bound(c['global_deadline_source']))
    need(w['config_sha256']==pin and w['deadline_unix']==c['global_deadline_unix']==original['deadline_unix']==1791592146.380285 and w['created_unix']<=time.time()<w['deadline_unix'],'inherited global deadline; no refresh')
    mono=time.monotonic()+max(0,w['deadline_unix']-time.time());deadline=w['deadline_unix']-c['closure_reserve_seconds'];bind_campaign_deadline(deadline)
    bind_campaign_deadline(time.time()+max(0,mono-time.monotonic()-c['closure_reserve_seconds']))
    for rel,h in c['owned_sources'].items():
        p=Path(rel);need(not p.is_absolute()and'..'not in p.parts and sha(read(root/p))==h,'pinned source:'+rel)
    group=Path('/sys/fs/cgroup')/next(x[3:]for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::')).lstrip('/')
    quota,period=(group/'cpu.max').read_text().split()
    need(quota!='max'and int(quota)<=8*int(period)and int((group/'memory.max').read_text())<=8*1024**3 and int((group/'memory.swap.max').read_text())==0 and int((group/'pids.max').read_text())<=64,'CPU8/RAM8GiB/swap0/tasks64')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'and os.environ.get('TMPDIR')==str(root/'tmp')and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'owned GPU/BLAS/temp boundary')
    import numpy as np,torch,boto3,safetensors
    actual={'python':sys.version.split()[0],'numpy':str(np.__version__),'torch':str(torch.__version__),'boto3':str(boto3.__version__),'safetensors':str(safetensors.__version__)}
    need(actual==c['runtime_versions'],'exact numeric runtime');validate_permit(c)
    from jope_flywheel import training_snapshot_native as N,native_variant_snapshot as V,timing
    from jope_flywheel.program_io import NativeTools
    from jope_flywheel.s3_library import client,BUCKET
    import native_corpus_authority as authority
    from continuous_state import parent_ancestry,append
    from continuous_runner import run
    from neural_native_train import publish,OBJECTIVE
    need(c['neural']['objective']==OBJECTIVE and c['neural']['max_updates']==50000 and c['neural']['max_padded_tokens']==8192 and c['neural']['batch_size']==16 and c['neural']['max_seq']==640 and c['neural']['max_pending_emissions']==32768 and c['neural']['output_cap_bytes']==24*1024**3,'reviewed continuation numerical/cap scope')
    import subprocess
    from cold_reservation import prove,stopped
    need(execution['stopped_owners']==execution['cold_reservation_permit']['stopped_owners'],'exact stopped writer identities')
    certified=prove(root/'training',execution['cold_reservation_permit'])
    create(execution_root/'certified-unstarted.json',canonical(certified))
    timing.ACTIVE=timing.Timing(execution_root/'stage-timing.jsonl',group,config_sha256=sha(execution_raw),origin_checkpoint_sha256=c['parameter_sha256'])
    training=json.loads(bound(c['training_config']));status=json.loads(bound(c['checkpoint_proof']['status']));result=json.loads(bound(c['checkpoint_proof']['result']));last=json.loads(bound(c['checkpoint_proof']['batch']))
    need(status['state']==result['state']=='completed'and status['config_sha256']==c['training_config']['sha256']and result['current_parameter_sha256']==c['parameter_sha256']==last['checkpoint']['parameter_sha256']and last['checkpoint']['step']==429,'completed physical parent429')
    need(training['neural']['seed']==c['neural']['seed']and training['neural']['objective']==OBJECTIVE and training['neural']['evaluation_panel_contexts']==c['neural']['evaluation_panel_contexts'],'same native fixed evaluation protocol')
    ancestry=parent_ancestry(c['planner_path'],c['planner_anchor'],c['parameter_sha256']);pre=bound(c['preprocessing']);registry=bound(c['parent_registry']);parents={p['parent_lineage_sha256']:p for p in map(N.document,registry.splitlines())}
    specraw=bound(c['snapshot']);spec=json.loads(specraw)
    def fresh(snapshot=None):
        tick(deadline);p=canonical(authority.check(spec));doc=json.loads(p)
        need(doc['passed']is True and doc['source_pins']==spec['authority_pins']and 0<=time.time()-doc['checked_at_unix']<=60,'fresh exact native/current role proof')
        if snapshot is not None:V.check_authority(json.loads(snapshot.extension_raw),spec,p,parents,time.time())
        publish(root,'latest-role-proof.json',doc);return p
    with timing.span('input.native_snapshot_load'):native=N.load_snapshot(specraw,c['snapshot']['sha256'],N.LocalStore(c['read_roots']),authority_bytes=fresh())
    tool=NativeTools(c['native_canonicalizer'],deadline);s3=client(c['aws_profile'])
    class Store:
        nativecanonical_json=tool.canonical_json
        def get_version(self,ref):
            tick(deadline);need(ref['bucket']==BUCKET and ref['key'].startswith('experiments/gp-policy-dl/v2-trainonly/')and type(ref['bytes'])is int and 0<ref['bytes']<=32*1024**2,'bounded exact input/GP namespace')
            with timing.span('s3.exact_version_GET',key=ref['key'],version_id=ref['version_id']):
                r=s3.get_object(Bucket=BUCKET,Key=ref['key'],VersionId=ref['version_id']);need(r.get('VersionId')==ref['version_id']and r['ContentLength']==ref['bytes'],'exact version size')
                with r['Body']as f:b=f.read(ref['bytes']+1)
            tick(deadline);need(len(b)==ref['bytes']and sha(b)==ref['sha256'],'exact version SHA');return b,ref['version_id']
    store=Store()
    def compose(raw):
        with timing.span('input.measured_variant_composition',extension_sha256=sha(raw)):
            s=V.compose(native,raw,sha(raw),store,native_spec_bytes=specraw,registry_bytes=registry,preprocessing_bytes=pre,authority_callback=fresh,deadline=deadline)
        s.extension_raw=raw;return s
    mixed=compose(bound(c['extension']));need(len(mixed.contexts)==7123 and len(mixed.variant_records)==3,'actual measured3 coverage')
    reviewed=json.loads(bound(c['measured3_acceptance']));need(reviewed['state']=='passed'and reviewed['extension_sha256']==c['extension']['sha256']and reviewed['origin_config_sha256']==c['measured3_acceptance_origin_config_sha256'],'reviewed actual measured3 acceptance')
    from prepared_admission import candidate
    from reconcile import capture
    resume=json.loads(bound(execution['transfer_capture']))
    need(resume['campaign_sha256']==pin and resume['optimizer_mutations']==resume['journal_mutations']==resume['replays']==0,'reviewed readonly transfer capture')
    actual_capture=capture(root/'training',campaign_sha256=pin,frozen=False,max_updates=c['neural']['max_updates'],plateau_patience=c['neural']['plateau_patience'],learning_rate_floor=c['neural']['learning_rate_floor'],require_closed_epoch=False,certified_unstarted=certified)
    # The sealed receipt is JSON: restore the same canonical representation before comparison.
    actual_capture=json.loads(canonical(actual_capture))
    # Outcome consumers may advance; immutable optimizer/input/ACK custody cannot.
    need({k:v for k,v in actual_capture.items()if k!='outbox_head'}=={k:v for k,v in resume.items()if k!='outbox_head'},'unchanged physical/planner/control custody')
    resume['execution_config_sha256']=sha(execution_raw)
    replacement=candidate(execution['prepared_candidate'],mixed,native,campaign_sha256=pin,preprocessing_sha256=c['preprocessing']['sha256'],registry_sha256=c['parent_registry']['sha256'],authority_check=lambda ext:fresh(type('AuthoritySnapshot',(),{'extension_raw':ext})()))
    attempted=False
    def adopt(current,training_root):
        nonlocal attempted
        if attempted:return None
        attempted=True;fresh(replacement);return replacement
    create(execution_root/'runtime-proof.json',canonical({'config_sha256':pin,'versions':actual,'original_planner':ancestry['head'],'cuda_workspace':os.environ['CUBLAS_WORKSPACE_CONFIG'],'torch_origins':{k:v.__file__ for k,v in sys.modules.items()if k.startswith('dltrain.')and hasattr(v,'__file__')}}))
    need(torch.cuda.is_available()and torch.cuda.device_count()==1,'one available CUDA device')
    result=run(native,mixed,c,root=root/'training',campaign_sha=pin,ancestry=ancestry,preprocessing=pre,deadline=deadline,authority_check=fresh,adoption_check=adopt,resume=resume)
    publish(execution_root,'status.json',{'schema':'jope.continuous-neural-campaign-status/v1','state':result['state'],'config_sha256':pin,'at':time.time(),'result':result,'global_deadline_unix':c['global_deadline_unix'],'training_root':str(root/'training')})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        root=Path(__file__).resolve().parent
        try:
            with(root/'fatal.json').open('xb')as f:f.write(json.dumps({'schema':'jope.continuous-neural-fatal/v1','at':time.time(),'error':type(error).__name__+': '+str(error)[:4000],'automatic_retry':False},sort_keys=True).encode());f.flush();os.fsync(f.fileno())
        except BaseException:pass
        raise
