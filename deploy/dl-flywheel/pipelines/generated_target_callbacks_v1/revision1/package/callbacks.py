"""Configure-once production callbacks for the frozen reference exporter.

Only configuration bytes and independently hash-pinned files enter this module.
No function/client injection API. Tests replace transport boundaries, not these
production callbacks. This is reviewed trusted code, not a Python sandbox.
"""
from dataclasses import asdict, dataclass
import importlib
import platform
from pathlib import Path
import sys
import time
from . import bindings as B
from .readonly_io import (ExactReader, NativeCanonical, Window, canonical, document,
                          hash64, need, read_bound, reference, sha)

_STATE = None
_CONFIGURED = False
GLOBAL_DEADLINE = 1791592146.380285
ARTIFACT_CAPS = dict(zip(('input','program','evaluation','rust_source','rust_executable','rust_parity','score_records'),
                        (8*1024**2,65536,4*1024**2,4*1024**2,16*1024**2,4*1024**2,2*1024**2)))
PINNED_INPUTS = {'prepared':'b8518a6e350018e1fa6e9ba3c23d14c3f635096f5fda7ea7af16f453e1edbb53',
 'adoption':'e90980f0be8adc86521e95344d9d809a6309d6fb9b8d86f9986c4cad2b4b32bd',
 'training_config':'fec184217e7c50a3fc3bc1cc7e33edafdbd218d1fafa1bd0d68debf626dcd4ed',
 'normalizer':B.NORMALIZER}


def _package_check(config):
    root=Path(__file__).parent.resolve();pins=config['package_files']
    need(type(pins)is dict and set(pins)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'complete exact package Python closure')
    for relative,want in pins.items():
        need('..'not in Path(relative).parts and not Path(relative).is_absolute() and hash64(want),'relative source closure')
        read_bound({'path':str(root/relative),'sha256':want},2*1024**2)
    vendor_raw=(root/'VENDOR_PINS.json').read_bytes()
    need(sha(vendor_raw)=='c41b3f9a4eaaea998a9e7f60f66e47fc80e62fabaed2aff977fa4849facca697','fixed vendored import manifest')
    vendor=document(vendor_raw)
    need(all(pins.get(p)==h for p,h in vendor.items()),'unchanged reviewed vendor source pins')
    # All imported in-package modules must originate from this checked tree.
    prefix=__package__+'.'
    for name,module in list(sys.modules.items()):
        if name==__package__ or name.startswith(prefix):
            file=getattr(module,'__file__',None)
            need(file is not None and Path(file).resolve().is_relative_to(root),'loaded package origin')
            rel=str(Path(file).resolve().relative_to(root));need(rel in pins,'unlisted package import')


def _runtime_check(config):
    runtime=config['runtime']
    need(platform.python_version()==runtime['python_version'],'qualified Python version')
    need(str(Path(sys.executable).resolve())==runtime['interpreter']['path'],'qualified Python origin')
    read_bound(runtime['interpreter'],64*1024**2)
    need(set(runtime['modules'])=={'numpy','boto3','botocore'},'required external module roster')
    for name,ref in runtime['modules'].items():
        module=importlib.import_module(name)
        need(str(Path(module.__file__).resolve())==ref['file']['path'] and module.__version__==ref['version'],'runtime module origin/version:'+name)
        read_bound(ref['file'],2*1024**2)
    # Distribution inventory is separate from CheckedCallback's <=64 source pins.
    # Every imported nonstdlib origin must appear in this physical closure, including
    # extension modules and lazily imported SDK dependencies after each operation.
    files=runtime['import_files'];need(type(files)is list and 3<=len(files)<=4096,'runtime import inventory bound')
    bypath={r['path']:r for r in files};need(len(bypath)==len(files),'duplicate runtime source path')
    total=0
    for ref in files:
        raw=read_bound(ref,64*1024**2);total+=len(raw);need(total<=256*1024**2,'runtime source aggregate bound')
    _import_origins(config)


def _import_origins(config):
    import sysconfig
    package=Path(__file__).parent.resolve();stdlib=Path(sysconfig.get_path('stdlib')).resolve()
    allowed={r['path'] for r in config['runtime']['import_files']}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file or file.startswith('<'):continue
        p=Path(file).resolve()
        if p.is_relative_to(package):continue
        if p.is_relative_to(stdlib) and 'site-packages'not in p.parts and 'dist-packages'not in p.parts:continue
        need(str(p)in allowed,'unlisted imported external source:'+str(p))


def _config(raw,expected):
    need(type(raw)is bytes and len(raw)<=2*1024**2 and sha(raw)==expected,'exact bounded configuration')
    c=document(raw)
    need(set(c)=={'schema','mode','created_unix','deadline_unix','global_deadline_unix','max_gets','max_read_bytes',
         'refs','worker_config','runtime','package_files','selected','authority_source_pins'},'closed callback configuration')
    need(c['schema']=='jope.generated-target-production-callbacks/v1' and c['mode']in('selected-qualification','full-authority'),'mode/schema')
    now=time.time();need(type(c['created_unix'])in(int,float) and type(c['deadline_unix'])in(int,float)
         and c['created_unix']<=now<c['deadline_unix']<=c['global_deadline_unix']==GLOBAL_DEADLINE,'fixed original clock')
    need(0<c['deadline_unix']-c['created_unix'] <= (180 if c['mode']=='selected-qualification'else 900),'fixed bounded work window')
    if c['mode']=='selected-qualification':
        need(1<=len(c['selected'])<=4 and c['max_gets']<=64 and c['max_read_bytes']<=128*1024**2,'selected qualification envelope')
    else:need(c['selected']==[],'full authority derives complete128 roster')
    refs=c['refs'];need(set(refs)=={'prepared','adoption','training_config','execution_config','normalizer','kumo_audit','weight_license','code_license','candidate_observation','labels_source','compose_source'},'closed input refs')
    need(all(refs[k]['sha256']==h for k,h in PINNED_INPUTS.items()),'frozen admitted128/original model identity')
    need(refs['labels_source']['sha256']=='c2554a33554e69ee36f7c9eeab094989609c212b9e5c954ccf6f712c35b0befd'
         and refs['compose_source']['sha256']=='68c20ee69226e7be9bf92ad40e3889c91fddd4af337c65c9d838d8f51d875258','unchanged numeric-target construction sources')
    w=c['worker_config'];need(w['registry']['sha256']==B.REGISTRY and w['generator_config']['sha256']==B.GENERATOR
         and w['native_canonicalizer']['sha256']==B.CANONICALIZER and w['worker']['sha256']==B.EVALUATOR,'qualified producer pins')
    need(w['source_commit']=='6bf65b12e9f0e5ec2697720a6119f13ea27d4835'
         and w['source_build_plan_sha256']=='4bb10409dde5a686da91fb96501113a0e5dd1efcde6429cab1024d4270302632','qualified evaluator source/plan')
    need(w['protocol_descriptor']['evaluation_source_build_plan_sha256']==w['source_build_plan_sha256']
         and w['vocabulary_descriptor']['canonicalizer_sha256']==B.CANONICALIZER,'protocol/canonicalizer source binding')
    return c


@dataclass(frozen=True)
class Runtime:
    config_bytes: bytes
    config_sha256: str
    window: object
    reader: object
    native: object
    inputs: object
    descriptors_bytes: bytes
    objective_bytes: bytes
    fit_sha256: str
    replayed: frozenset
    parent_roles_bytes: bytes
    local_pins_bytes: bytes
    observations_bytes: bytes

    def config(self):
        return document(self.config_bytes)


def _physical_checks(state):
    c=state.config();state.window.tick();_package_check(c)
    for ref in document(state.local_pins_bytes):read_bound(ref)
    B.license_contract(c['refs']['kumo_audit'],c['refs']['weight_license'],c['refs']['code_license'])
    generator=state.inputs.generator
    eligible,protected,views=B.current_roles(generator,B.ROLE_ROOT)
    roles=document(state.parent_roles_bytes)
    for descriptor in document(state.descriptors_bytes).values():
        parent=state.inputs.parents[descriptor['parent_lineage_sha256']]
        need(parent['partition']==roles[parent['parent_lineage_sha256']]==descriptor['partition']
             and parent['view_id'] in eligible and parent['view_id']not in views
             and not(set(parent['retained_lineage_context_sha256'])&protected),'current parent role/exclusion')
    _import_origins(c);state.window.tick()


def _encoder_binding(reader,descriptor):
    # Previously adopted vector stays frozen; this verifies its immutable published
    # producer identity, not a new encoder execution or pretraining-overlap claim.
    raw,_=reader.get_version(descriptor['encoder_publication']);doc=document(raw)
    need(doc['schema']in('jope.flywheel-evergreen-encoder-publication/v1','jope.flywheel-evergreen-kumos-publication/v1')
         and doc['variant_identity']==descriptor['variant_identity'] and doc['parent_lineage_sha256']==descriptor['parent_lineage_sha256']
         and doc['partition']==descriptor['partition'],'admitted encoder publication identity')
    item=doc['encoder_records']['kumo_tabular_s'];record=item['record'];q=item['qualification'];qraw=canonical(q['body'])
    need(record['component']=='kumo_tabular_s' and record['dimension']==2051 and record['variant_identity']==descriptor['variant_identity']
         and record['input_raw_sha256']==descriptor['context']['dataset_sha256'] and record['target_sha256']==descriptor['context']['target_sha256']
         and record['partition']==descriptor['partition'] and record['outer_test_rows']==0 and record['teaching_eligible']is False,'encoder context/shape')
    weight=record['producer']['weight']
    need(weight['sha256']==B.WEIGHT and weight['repo']=='nvidia/Kumo-Tabular' and weight['file']=='small/regressor.pt'
         and weight['revision']=='3c3e10bbdb590ace29e7026847f92db3c603096d','exact qualified Kumo weight')
    need(sha(qraw)==q['ref']['sha256']==record['qualification']['sha256'] and len(qraw)==q['ref']['bytes']==record['qualification']['bytes']
         and q['body']['passed']is True,'embedded qualification hash')


def configure(raw,expected):
    """One use per process. Never accepts callbacks/client objects from a caller."""
    global _CONFIGURED,_STATE
    need(not _CONFIGURED,'production runtime cannot be rebound');_CONFIGURED=True
    c=_config(raw,expected);_package_check(c);_runtime_check(c)
    from .vendor import program_inputs,program_io
    _package_check(c)
    window=Window(c['deadline_unix'],c['max_gets'],c['max_read_bytes'])
    native=NativeCanonical(c['worker_config']['native_canonicalizer'],window)
    program_io.bind_campaign_deadline(c['deadline_unix'])
    docs={k:document(read_bound(ref))for k,ref in c['refs'].items() if k not in('weight_license','code_license','labels_source','compose_source')}
    B.license_contract(c['refs']['kumo_audit'],c['refs']['weight_license'],c['refs']['code_license'])
    descriptors,extension,objective,fitsha=B.admitted_metadata(docs['prepared'],docs['adoption'],docs['training_config'],docs['execution_config'],c['refs'],docs['normalizer'])
    import boto3
    from botocore.config import Config
    # Single transport, no caller client injection, no retries or proxy endpoint override.
    import os
    need(not any(os.environ.get(k)for k in ('AWS_ENDPOINT_URL','AWS_ENDPOINT_URL_S3')),'custom S3 endpoint forbidden')
    client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=3,read_timeout=5,
            retries={'total_max_attempts':1,'mode':'standard'},max_pool_connections=1))
    reader=ExactReader(client,window)
    registry=document(read_bound(c['worker_config']['registry']))
    inputs=program_inputs.RegisteredInputs(c['worker_config'],registry,native,None,c['deadline_unix'])
    inputs.version_cache=reader  # VerifiedVersion API; no cache and no alternative source.
    need(len(inputs.rows)==128 and {d['variant_identity']for d in descriptors.values()}==set(inputs.rows),'exact registered128 support')
    need(c['authority_source_pins']==inputs.generator['authority_pins'],'configured physical authority pins')
    B.current_roles(inputs.generator,B.ROLE_ROOT)
    rolemap={d['parent_lineage_sha256']:inputs.parents[d['parent_lineage_sha256']]['partition'] for d in descriptors.values()}
    need(all(rolemap[d['parent_lineage_sha256']]==d['partition']for d in descriptors.values()),'admitted parent partitions')
    observations=docs['candidate_observation']
    need(observations['schema']=='jope.generated-target-readonly-candidate-observation/v1'
         and observations['full_cutoff_verified']is False,'candidate observation is not a cutoff')
    selected=c['selected']
    known={x['id']:x for x in observations['candidate_rows']};need(len(known)==len(observations['candidate_rows']),'unique candidate IDs')
    need(len(selected)==len(set(selected)) and all(x in known for x in selected),'explicit selected observed IDs')
    for identity in selected:
        row=known[identity];e=row['body']['emission'];cid=e['context_id']
        need(row['state']in('published_verified','cached_verified') and e['phase']=='training' and cid in descriptors
             and descriptors[cid]['partition']=='train' and e['input_descriptor']==descriptors[cid], 'selected success/train/admitted descriptor before reads')
    replay_ids=(sorted(descriptors)if c['mode']=='full-authority'else sorted({known[x]['body']['emission']['context_id']for x in selected}))
    localpins=list(c['refs'].values())+[c['worker_config']['generator_config'],c['worker_config']['registry'],c['worker_config']['native_canonicalizer']]
    localpins+=inputs.generator['source_pins']+[inputs.generator['parent_registry']]+inputs.generator['authority_pins']
    for cid in replay_ids:
        need(cid in descriptors,'selected context not admitted')
        d=descriptors[cid];B.replay_descriptor(inputs,d)
        _encoder_binding(reader,d)
        row=inputs.rows[d['variant_identity']];parent=inputs.parents[row['parent']]
        localpins+=list(row['files'].values())+[parent[k]for k in('encoder','context','preparation_receipt')if parent.get(k)is not None]
    # Pinned local input bytes permit rechecking replay custody without redoing
    # all transforms; the first full pass is mandatory before authority callback.
    dedup={r['path']:r for r in localpins}
    need(all(dedup[r['path']]==r for r in localpins),'conflicting local input pins')
    state=Runtime(raw,expected,window,reader,native,inputs,canonical(descriptors),canonical(objective),fitsha,
                  frozenset(replay_ids),canonical(rolemap),canonical(list(dedup.values())),canonical(observations))
    _physical_checks(state);_STATE=state
    return {'replayed_contexts':len(replay_ids),'admitted_roster_contexts':128,'full_authority_ready':len(replay_ids)==128}


def _state():
    need(type(_STATE)is Runtime and _CONFIGURED,'configured production runtime required')
    _STATE.window.tick();_package_check(_STATE.config());return _STATE


def read_version(ref):
    state=_state();value=state.reader.get_version(ref);_import_origins(state.config());return value


def canonicalize(text):
    state=_state();value=state.native.canonical_program(text);_import_origins(state.config());return value


class SevenBodies:
    def __init__(self,entry,bodies,native):
        need(set(entry['artifacts'])==set(bodies)==set(ARTIFACT_CAPS),'exact seven-body closure')
        self._bodies={};self.nativecanonical_json=native.canonical_json
        for name,ref in entry['artifacts'].items():
            reference(ref);raw=bodies[name]
            need(type(raw)is bytes and len(raw)==ref['bytes']<=ARTIFACT_CAPS[name] and sha(raw)==ref['sha256'],'seven-body hash/bound')
            key=canonical(ref);need(key not in self._bodies,'duplicate artifact reference');self._bodies[key]=raw
    def get_version(self,ref):
        reference(ref);key=canonical(ref);need(key in self._bodies,'unprovided artifact version')
        return self._bodies[key],ref['version_id']


def verify(ctx,dsl,entry,bodies):
    state=_state();cid=sha(canonical(ctx));descriptors=document(state.descriptors_bytes)
    need(cid in state.replayed and descriptors[cid]['context']==ctx and descriptors[cid]['partition']=='train','replayed training context required')
    need(state.native.canonical_program(dsl)==dsl,'canonical measured program')
    from .vendor.program_gate import verified_measurement
    from .vendor.protocol import Context
    parity=document(bodies['rust_parity']);provenance=parity['native_provenance'];worker=state.config()['worker_config']
    need(provenance.get('source_commit')==worker['source_commit'] and provenance.get('source_build_plan_sha256')==worker['source_build_plan_sha256'],'measured evaluator source/plan provenance')
    proof=verified_measurement(SevenBodies(entry,bodies,state.native),Context(**ctx),dsl,entry)
    _import_origins(state.config());state.window.tick();return proof


def authority(request):
    state=_state();need(len(state.replayed)==128,'full128 registered replay not completed; no export authority')
    _physical_checks(state);c=state.config();descriptors=document(state.descriptors_bytes)
    prepared=document(read_bound(c['refs']['prepared']));extension=document(prepared['extension_utf8'].encode())
    need(all(set(t[k])=={'bucket','key','version_id','sha256','bytes'}for t in extension['tasks']for k in('dataset_publication','encoder_publication')),'frozen exporter cannot consume annotated predecessor refs; separate successor required')
    need(request['admitted_descriptors']==descriptors and request['authority_source_pins']==c['authority_source_pins']
         and request['objective_sha256']==sha(state.objective_bytes) and request['normalizer_sha256']==B.NORMALIZER
         and request['native_train_context_ids_sha256']==state.fit_sha256 and request['roster']==B.ROSTER
         and request['allowed_campaign_sha256']==[c['refs']['training_config']['sha256']]
         and request['construction_sources']=={k:{'path':c['refs'][r]['path'],'sha256':c['refs'][r]['sha256']}for k,r in [('labels','labels_source'),('compose','compose_source')]}, 'request differs from derived production authority')
    return {'schema':'jope.generated-target-current-authority/v1','state':'passed','request_sha256':sha(canonical(request)),
            'checked_at_unix':time.time(),'source_pins':c['authority_source_pins'],'descriptors':descriptors,
            'parent_roles':document(state.parent_roles_bytes),'objective_sha256':sha(state.objective_bytes),
            'normalizer_sha256':B.NORMALIZER,'native_train_context_ids_sha256':state.fit_sha256,'roster':dict(B.ROSTER)}


def qualify_selected():
    state=_state();c=state.config();need(c['mode']=='selected-qualification','selected qualification mode')
    observations=document(state.observations_bytes);rows={x['id']:x for x in observations['candidate_rows']}
    descriptors=document(state.descriptors_bytes);seen=set();results=[]
    for identity in c['selected']:
        _physical_checks(state);row=rows[identity];body=row['body'];e=body['emission'];out=row['outcome'];cid=e['context_id']
        need(row['state']in('published_verified','cached_verified') and out['state']==row['state'] and out['error']is None
             and row['body_sha256']==sha(canonical(body)), 'selected successful observation identity')
        need(e['input_descriptor']==descriptors[cid] and e['phase']=='training' and descriptors[cid]['partition']=='train'
             and e['parent_lineage_sha256']==descriptors[cid]['parent_lineage_sha256']
             and cid in body['batch']['train_data_run_ids'] and cid not in body['batch']['validation_data_run_ids']
             and body['batch']['campaign_sha256']==c['refs']['training_config']['sha256'],'selected train role/campaign')
        need(sha(e['raw_program_utf8'].encode())==e['raw_program_sha256'],'raw program identity')
        need(body['model_origin_sha256']==observations['config']['model_origin_sha256'] and
             identity==sha(canonical({'origin':body['model_origin_sha256'],'batch_id':body['batch']['batch_id'],'ordinal':body['ordinal']})), 'observed emission identity')
        expected=[('training',x)for x in body['batch']['emitted_train_data_run_ids']]+[('validation',x)for x in body['batch']['validation_data_run_ids']]
        need(type(body['ordinal'])is int and 0<=body['ordinal']<len(expected) and expected[body['ordinal']]==(e['phase'],cid),'observed emission ordinal')
        dsl=canonicalize(e['raw_program_utf8']);ref=out['record_ref'];reference(ref)
        from .vendor.program_gate import request_identity
        from .vendor.protocol import Context
        key=request_identity(Context(**descriptors[cid]['context']),dsl)
        need(key==out['evaluation_key'] and ref['key'].endswith('/'+key+'/record.json') and key not in seen,'canonical selected result key/duplicate')
        seen.add(key);raw,_=read_version(ref);entry=document(raw)
        need(entry.get('evaluation_key')==key and entry.get('status')=='completed' and entry.get('measurement')=='executed','successful exact wrapper')
        need(set(entry.get('artifacts',{}))==set(ARTIFACT_CAPS),'seven artifact roles before reads')
        for name,r in entry['artifacts'].items():
            reference(r);need(r['bytes']<=ARTIFACT_CAPS[name],'artifact role cap before GET')
        bodies={name:read_version(r)[0] for name,r in entry['artifacts'].items()}
        proof=verify(descriptors[cid]['context'],dsl,entry,bodies)
        results.append({'emission_id':identity,'result_ref':ref,'context_id':cid,'measurement':proof,'ce_eligible':False})
    _physical_checks(state)
    return {'schema':'jope.generated-target-selected-qualification/v1','state':'passed','config_sha256':state.config_sha256,
            'results':results,'replayed_selected_contexts':len(state.replayed),'admitted_roster_contexts':128,
            'objective_sha256':sha(state.objective_bytes),'normalizer_sha256':B.NORMALIZER,'native_train_context_ids_sha256':state.fit_sha256,
            'roster':dict(B.ROSTER),'timing':state.window.counters(),'full_cutoff_verified':False,'reference_export_completed':False,
            'numeric_targets_materialized':False,'optimizer_admission':False,'gp_executions':0,'s3_writes':0,'database_writes':0,'optimizer_updates':0}
