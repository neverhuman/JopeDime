"""Fresh five-block research encodings for the immutable 100-variant pilot.

CPU preflight precedes an actual full-vector CUDA qualification for each model.
No S3 calls, GP labels, training or old-output mutations. Requires an external
CPU8/Memory12GiB/one-GPU/1200-second service and an owner-pinned manifest.
"""
import argparse
import gc
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct
import sys
import time
sys.dont_write_bytecode=True
for k,v in {'OMP_NUM_THREADS':'3','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items():os.environ[k]=v
import numpy as np

SCHEMA='jope.flywheel-ofm-pilot/v1'
W=Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008')
SOURCE_MANIFEST='800f7c11ef9c2d55c18742de2c7c8ac10c3ac2cae58b737e7e44ec7c09fb7ea3'
PARENT='0b9bd099d48be1afe7e180eab9543062759a34aa37d6daa1fefe010212262af6'
BUNDLE='bb6b706cd7db29c9feb2e83afae9743655fb8a8a5ee39b00be45eb239075f2f4'
ROLE='75cec49392bd19b86f55a7cf127a2f3e99703282a86bd5f0d3da412ac202a538'
DIMS={'kumo_tabular_l':4099,'kumo_tabular_s':2051,'mitra_v2':4099,'tabdpt13':3075,'tabicl2':2051}
CORE={str(W/'code/ofm_core.py'):'fcfd6e80e5f2b376c2304836a6320f946eb32824fc9b74f226996c2c2ca00d70',str(W/'code/ofm_models.py'):'4a877f66231fdfd558a51421f1589511b80a09afda99291d6de8e52a1f8926f8',str(W/'code/ta_code/ta_common.py'):'0907934bde679ab33b7922cd59d398800ca27b23b6fb17dee3b2f6c0efd7b6e5'}
CAP=256*1024**2


def need(ok,msg):
    if not ok:raise ValueError(msg)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()


def regular(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts,'absolute confined path required')
    need(all(not q.is_symlink() for q in (p,*p.parents)),'symlink refused')
    need(stat.S_ISREG(p.stat().st_mode),'regular file required');return p


def checked(ref,cap=4*1024**2,load=True):
    p=regular(ref['path']);need(p.stat().st_size<=cap,'file exceeds read cap')
    h=hashlib.sha256();chunks=[]
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            h.update(b)
            if load:chunks.append(b)
    need(h.hexdigest()==ref['sha256'],'SHA mismatch: '+str(p))
    if 'bytes' in ref:need(p.stat().st_size==ref['bytes'],'byte count mismatch')
    return b''.join(chunks) if load else p


def obj(ref):return json.loads(checked(ref))


def under(root,ref):
    rel=Path(ref['path']);need(not rel.is_absolute() and '..' not in rel.parts,'relative receipt path required')
    return {**ref,'path':str(root/rel)}


def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(canonical({'dtype':a.dtype.str,'shape':list(a.shape)}));h.update(a.tobytes());return h.hexdigest()


def decode(raw,allow_test=False):
    need(raw[:4]==b'JTT1' and len(raw)>=17,'JTT1 required')
    n,nt,p,k=struct.unpack('<IIIB',raw[4:17]);need(n==192 and 1<=p<=256 and k==2,'pilot input shape differs')
    need((allow_test and nt<=17) or nt==0,'protected test payload refused')
    need(len(raw)==17+n+8*(n+nt)*(p+1),'JTT1 length differs')
    inner=np.frombuffer(raw[17:17+n],dtype=np.uint8).copy();need(np.bincount(inner,minlength=2).tolist()==[96,96],'inner roles differ')
    a=np.frombuffer(raw[17+n:17+n+8*n*(p+1)],dtype='<f8').reshape(n,p+1).copy()
    need(np.isfinite(a).all() and np.ptp(a[:,-1])>1e-12,'nonfinite or degenerate input')
    return a[:,:-1],a[:,-1],inner


def uncompress(raw):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:b=f.read(1024**2+1)
    need(len(b)<=1024**2,'decompressed input cap');return b


def load_inputs(cfg,source_manifest):
    need(source_manifest['sha256']==SOURCE_MANIFEST,'wrong source campaign')
    sm=obj(source_manifest);st=obj(cfg['source_status']);q=obj(cfg['source_qualification']);host=cfg['host']
    need(sm['hosts'][host]['out']==str(Path(cfg['source_status']['path']).parent),'wrong source root')
    need(st['state']=='complete' and st['host']==host and st['manifest_sha256']==SOURCE_MANIFEST,'source run incomplete or wrong identity')
    need(st['completed']==st['expected']==50 and st['new_gp_labels']==0 and st['teaching_eligible'] is False,'source scope differs')
    need(st['completed_encoders']==['a0','hyperion_v6all'] and st['unique_content_this_host']==50,'source encoder/content closure differs')
    need(q['host']==host and q['manifest_sha256']==SOURCE_MANIFEST and q['parent_context_sha256']==PARENT and q['bundle_sha256']==BUNDLE and q['role_clearance_sha256']==ROLE,'source qualification identity differs')
    need(q['qualification']['passed'] is True and q['qualification']['exact_repeat'] is True,'source CUDA qualification absent')
    need(q['rows']==192 and q['outer_test_rows']==0 and q['inner_counts']==[96,96] and q['retained_lineages']==1068,'source role shape differs')
    expected=list(range(50)) if host=='xbabe2' else list(range(50,100));need([v['index'] for v in st['variants']]==expected,'source50 index closure differs')
    need(len({v['variant_identity'] for v in st['variants']})==50,'duplicate variant ID')
    need(len({v['content_sha256'] for v in st['variants']})==50,'duplicate variant scientific content')
    root=Path(cfg['source_status']['path']).parent;inputs=[]
    for v in st['variants']:
        identity=v['variant_identity'];need(v['receipt']['path']=='variants/'+identity+'/manifest.json','wrong source receipt location')
        r=obj(under(root,v['receipt']))
        need(r['manifest_sha256']==SOURCE_MANIFEST and r['index']==v['index'] and r['variant_identity']==identity and r['content_sha256']==v['content_sha256'],'variant receipt binding differs')
        need(r['parent_context_sha256']==PARENT and r['lineage_sha256']==q['lineage_sha256'] and r['partition']==q['partition'],'variant parent/lineage changed')
        need(r['teaching_eligible'] is False and r['gp_labels'] is None and r['completed_encoders']==['a0','hyperion_v6all'],'unexpected inherited labels')
        for name,dim in [('a0.npy',88),('hyperion_v6all.npy',512)]:
            need(r['files'][name]['path']=='variants/'+identity+'/'+name,'wrong source vector location')
            a=np.load(io.BytesIO(checked(under(root,r['files'][name]))),allow_pickle=False)
            need(a.shape==(dim,) and a.dtype==np.dtype('<f4') and np.isfinite(a).all(),'source vector invalid')
        f=r['files']['encoder.jtt1.gz'];need(f['path']=='variants/'+identity+'/encoder.jtt1.gz','wrong input location')
        raw=uncompress(checked(under(root,f),cap=1024**2));x,y,inner=decode(raw)
        need(array_sha(x)==r['batch']['x_sha256'] and array_sha(y)==r['batch']['y_sha256'] and r['batch']['y_sha256']==r['transform']['state']['parent']['target_sha256'],'variant X/y hashes differ')
        need(sha(canonical({'shape':list(x.shape),'x_sha256':array_sha(x),'y_sha256':array_sha(y),'inner_sha256':sha(inner.tobytes())}))==v['content_sha256'],'scientific content hash differs')
        inputs.append({'index':v['index'],'identity':identity,'source_receipt':v['receipt'],'receipt':r,'raw':raw})
    need(len({r['receipt']['batch']['y_sha256'] for r in inputs})==1,'pilot target identity changed across variants')
    return inputs,q


def runtime_check(inv):
    need(inv['schema']=='jope.ofm-pilot-runtime-inventory/v1' and sys.version.split()[0]==inv['python'] and sys.executable==inv['executable'],'runtime executable/version differs')
    need(0<len(inv['files'])<=1000 and sum(r['bytes'] for r in inv['files'])<64*1024**2,'runtime inventory bound')
    pins={r['path']:r['sha256'] for r in inv['files']};need(len(pins)==len(inv['files']),'duplicate runtime source')
    need(all(pins.get(p)==h for p,h in CORE.items()),'reviewed core source missing')
    expected_roots=[str(W/'src/TabDPT-inference/src'),str(W/'src/tabicl/src'),str(W/'src/structured-data-models/sdm'),str(W/'src/mitra-finetune/src'),'/home/ubuntu/.local/lib/python3.12/site-packages/autogluon/tabular/models/mitra/_internal']
    need(inv['source_roots']==expected_roots,'vendor inventory roots differ')
    for root in inv['source_roots']:
        paths=list(Path(root).rglob('*.py'));need(paths and all(str(p) in pins for p in paths),'unlisted vendor Python source')
    for ref in inv['files']:checked(ref,load=False)
    actual={}
    for name,r in inv['modules'].items():
        checked({'path':r['origin'],'sha256':r['origin_sha256']},load=False)
        m=importlib.import_module(name);v={'version':str(m.__version__),'origin':str(Path(m.__file__)),'origin_sha256':r['origin_sha256']}
        if name=='torch':v['cuda']=m.version.cuda
        need(v==r,'runtime module differs: '+name);actual[name]=v
    # Resolve all actual forward classes before reporting CPU readiness; no model
    # is instantiated and no weights are loaded by these import statements.
    for name in ('sdm.models.kumo.tabular.model','tabdpt.model','tabicl._model.tabicl','autogluon.tabular.models.mitra._internal.models.tab2d'):
        module=importlib.import_module(name)
        need(str(Path(module.__file__)) in pins,'forward class outside pinned source inventory: '+name)
    import torch
    need(not torch.cuda.is_initialized(),'CPU preflight initialized CUDA')
    return actual


def layout_for(t):
    n=DIMS[t]-3
    return ([['pooled_states_mean_std_p10_p90',0,2048],['thinking_tokens_mean_std',2048,3072]] if t=='tabdpt13' else [['pooled_states_mean_std_p10_p90',0,n]])+[['scalars_query_r2_mean_spread_log_spread',n,n+3]]


def vector_ok(result,t):
    v,comp,sel=result;need(isinstance(v,np.ndarray) and v.shape==(DIMS[t],) and v.dtype==np.dtype('<f4') and np.isfinite(v).all(),'invalid full '+t+' vector')
    need(comp['dimension']==DIMS[t] and comp['layout']==layout_for(t),'production layout differs')
    return v,comp,sel


def prep(core,item,raw=None):
    raw=item['raw'] if raw is None else raw;decode(raw)
    return core.prep({'context_sha256':item['identity']},raw=raw)


def prep_equal(a,b):
    ar=dict(a['rec']);br=dict(b['rec']);ar.pop('n_test',None);br.pop('n_test',None)
    return np.array_equal(a['X'],b['X']) and np.array_equal(a['y'],b['y']) and ar==br and a['binding']==b['binding']


def qualify(core,enc,t,item,deadline):
    """Nonempty actual-variant, whole production vector T0/T1/T2 gate."""
    need(time.time()<deadline,'qualification deadline')
    raw=item['raw'];x,y,inner=decode(raw);d0=prep(core,item)
    # Test-only bypass constructs protected rows locally. Production prep refuses it.
    fake=b'JTT1'+struct.pack('<IIIB',len(y),17,x.shape[1],2)+raw[17:]+np.random.default_rng(1729).normal(size=(17,x.shape[1]+1)).astype('<f8').tobytes()
    refused=False
    try:prep(core,item,fake)
    except ValueError:refused=True
    need(refused,'protected test production refusal failed')
    d1=core.prep({'context_sha256':item['identity']},raw=fake)
    need(prep_equal(d0,d1),'protected test bytes changed preprocessing')
    v0,c0,s0=vector_ok(enc.one(t,d0),t);vb,_,sb=vector_ok(enc.one(t,d0),t);vt,_,st=vector_ok(enc.one(t,d1),t)
    dy={**d0,'y':d0['y'][np.random.default_rng(23).permutation(len(d0['y']))].copy()}
    vy,_,_=vector_ok(enc.one(t,dy),t)
    tests={'T0_full_vector_determinism':bool(np.array_equal(v0,vb)),
           'T1_full_vector_protected_bytes_isolation':bool(np.array_equal(v0,vt)),
           'T1_selection_isolation':s0==sb==st,'T2_target_permutation_changes_full_vector':not bool(np.array_equal(v0,vy)),
           'T2_hidden_states_use_targets':not bool(np.array_equal(v0[:-3],vy[:-3])),
           'T3_production_refuses_protected_payload':refused}
    need(all(tests.values()),'qualification failed: '+str(tests));need(time.time()<deadline,'qualification exceeded deadline')
    return {'passed':True,'cases':1,'variant_identity':item['identity'],'component':t,'tests':tests,
            'layout':c0['layout'],'vector_sha256':sha(npy(v0)),'label_exposure':'Stored task-training y used by feature selection/context and query-R2 diagnostic; no outer-test or GP labels.',
            'scope':'Same-host/runtime full vector identity; no cross-host exactness or unseen-pretraining claim.'}


def npy(v):
    f=io.BytesIO();np.save(f,v,allow_pickle=False);return f.getvalue()


def write_new(out,rel,body):
    p=out/rel;need(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'output escape')
    need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(body)<=CAP,'output cap')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(body);f.flush();os.fsync(f.fileno())
    return {'path':rel,'sha256':sha(body),'bytes':len(body)}


def progress(out,host,msha,completed,state,**extra):
    counts={t:sum(r['component']==t for r in completed) for t in DIMS}
    done=['a0','hyperion_v6all']+[t for t,n in counts.items() if n==50]
    value={'schema':SCHEMA,'at':time.time(),'host':host,'manifest_sha256':msha,'state':state,'component_counts':counts,
           'completed_variant_components':len(completed),'expected_variant_components':250,'source_variants':50,'independent_parent_lineages':1,
           'completed_encoders':done,'missing_encoders':[t for t in [*DIMS,'dope'] if t not in done],'new_gp_labels':0,'teaching_eligible':False,**extra}
    tmp=write_new(out,'progress.tmp',canonical(value));os.replace(out/'progress.tmp',out/'progress.json');return value


def envelope():
    line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));rel=line.split(':',2)[2]
    root=Path('/sys/fs/cgroup')/rel.lstrip('/');q,p=(root/'cpu.max').read_text().split();mem=(root/'memory.max').read_text().strip()
    need(q!='max' and int(q)/int(p)<=8 and mem!='max' and int(mem)<=12*1024**3,'external CPU8/Memory12GiB envelope absent')
    return {'cpu_max':q+' '+p,'memory_max':int(mem),'cgroup':rel}


def load_module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def manifest_ok(m,host):
    need(m['schema']==SCHEMA and m['approved_for_qualification'] is True,'owner-approved manifest required')
    need(m['source_manifest']['sha256']==SOURCE_MANIFEST and m['total_variants']==100 and m['teaching_eligible'] is False,'scope changed')
    need(m['components']==DIMS and m['remaining_component']=='dope','five reviewed components required')
    need(m['limits']=={'seconds_per_host':1200,'cpus_per_host':8,'memory_gib_per_host':12,'gpus_per_host':1},'resource envelope differs')
    need(set(m['hosts'])=={'xbabe2','xbabe3'} and host in m['hosts'],'host set differs')
    need(math.isfinite(m['deadline_unix']) and time.time()<m['deadline_unix'],'deadline expired')
    cfg=m['hosts'][host];need(cfg['host']==host,'host config mismatch')
    out=Path(cfg['out']);need(out.parts[-3:]==('flywheel_ofm_pilot_20261009','runs',host),'new owned OFM output required')
    return cfg


def run(m,host,msha,preflight_only=False):
    start=time.time();deadline=min(start+1200,m['deadline_unix']);cfg=manifest_ok(m,host)
    expected_path=[cfg['flywheel_root'],str(W/'code'),str(W/'code/ta_code'),str(W/'src/structured-data-models'),str(W/'src/TabDPT-inference/src'),str(W/'src/tabicl/src'),str(W/'src/mitra-finetune/src')]
    need(cfg['pythonpath']==expected_path,'explicit import path differs')
    sys.path[:0]=expected_path
    inv=obj(cfg['runtime_inventory']);actual=runtime_check(inv)
    for ref in cfg['asset_pins']:checked(ref,cap=1024**3,load=False)
    # The module pin is checked before import; package roots cannot be inherited from another run.
    for ref in cfg['flywheel_source_pins']:checked(ref,load=False)
    need(cfg['admission_source']['sha256']=='73b5de9263342b678cbad5a3991163309e693e5a33e81c36090f9e1f25bf3fec','unreviewed admission helper')
    checked(cfg['admission_source'],load=False)
    sys.path[:0]=[cfg['flywheel_root'],str(W/'code'),str(W/'code/ta_code')]
    from jope_flywheel.evidence import Ledger,canonical_json,MAX_MANIFEST_BYTES
    need(str(Path(sys.modules['jope_flywheel.evidence'].__file__))==str(Path(cfg['flywheel_root'])/'jope_flywheel/evidence.py'),'evidence import origin differs')
    import ofm_core as core
    need(all(str(Path(sys.modules[n].__file__))==str(W/p) for n,p in [('ofm_core','code/ofm_core.py'),('ofm_models','code/ofm_models.py'),('ta_common','code/ta_code/ta_common.py')]),'core module origin differs')
    inputs,parent_proof=load_inputs(cfg,m['source_manifest']);first=inputs[0];d=prep(core,first)
    pins={r['path']:r['sha256'] for r in cfg['asset_pins']}
    need(len(pins)==len(cfg['asset_pins']),'duplicate asset pin')
    need(pins.get(str(W/'models/autogluon__mitra-regressor-2/config.json'))=='f760c38c8884757f6141c1fd2e841e93a470be052dc5a4b489790097a002f0e6','Mitra config not pinned')
    # Runtime Enc constructor independently reports these exact weight bytes.
    need(set(cfg['weight_pins'])==set(DIMS) and all(pins.get(r['path'])==r['sha256'] for r in cfg['weight_pins'].values()),'weight closure incomplete')
    overlap=obj(cfg['tabdpt_overlap']);aliases=parent_proof['lineage_datasets'];known={k:overlap[k] for k in aliases if k in overlap}
    proof={'source_sha256':m['worker_sha256'],'manifest_sha256':msha,'host':host,'source_manifest_sha256':SOURCE_MANIFEST,
           'source_status_sha256':cfg['source_status']['sha256'],'source_qualification_sha256':cfg['source_qualification']['sha256'],
           'parent_context_sha256':PARENT,'lineage_sha256':parent_proof['lineage_sha256'],'partition':parent_proof['partition'],
           'variants':len(inputs),'runtime':actual,'runtime_inventory_sha256':cfg['runtime_inventory']['sha256'],
           'tabdpt_known_overlap':known,'teaching_eligible':False,'new_gp_labels':0,'fresh_components':DIMS,'remaining_component':'dope',
           'prepared_feature_units':True,'acquisition_caveat':parent_proof['acquisition_caveat']}
    canonical_json(proof,limit=MAX_MANIFEST_BYTES)
    import torch
    need(not torch.cuda.is_initialized(),'CPU preflight initialized CUDA')
    need(time.time()<deadline,'preflight exhausted budget')
    if preflight_only:return proof
    need(socket.gethostname().split('.')[0]==host,'wrong execution host');proof['envelope']=envelope();canonical_json(proof,limit=MAX_MANIFEST_BYTES)
    out=Path(cfg['out']);need(out.is_absolute() and not out.exists() and '..' not in out.parts and all(not p.is_symlink() for p in out.parents),'fresh absolute output required');out.mkdir(parents=True)
    # Any runtime/JIT cache lives only in this owned directory.
    for key,rel in [('TORCH_HOME','torch-cache'),('TRITON_CACHE_DIR','triton-cache'),('CUDA_CACHE_PATH','cuda-cache'),('HF_HOME','hf-cache')]:os.environ[key]=str(out/rel)
    mod=load_module(checked(cfg['admission_source'],load=False),'ofm_pilot_admission');admission=mod.Admission()
    rid=m['campaign_id']+'-'+host;done=[];current=None
    with Ledger(out/'evidence.sqlite') as ledger:
        ledger.register_run(rid,proof)
        try:
            write_new(out,'preflight.json',canonical(proof));progress(out,host,msha,done,'qualifying')
            ok,reasons,obs=admission.check(start=True);need(ok,'GPU not admitted: '+str(reasons));need(torch.cuda.device_count()==1,'exactly one CUDA device required')
            for t in DIMS:
                need(time.time()<deadline,'pilot deadline');current={'component':t,'phase':'qualification'}
                ledger.append('qualify-intent-'+t,'qualification.intent',current,rid)
                enc=core.Enc([t]);need(enc.pins[t]['path']==cfg['weight_pins'][t]['path'] and enc.pins[t]['sha256']==cfg['weight_pins'][t]['sha256'],'loaded weight differs')
                qual=qualify(core,enc,t,first,deadline);canonical_json(qual);write_new(out,'qualification/'+t+'.json',canonical(qual));ledger.append('qualify-pass-'+t,'qualification.result',qual,rid)
                for item in inputs:
                    need(time.time()<deadline,'pilot deadline');ok,reasons,obs=admission.check();need(ok,'admission changed: '+str(reasons))
                    current={'component':t,'variant_identity':item['identity'],'index':item['index'],'phase':'encoding'}
                    event=t+'-'+item['identity'];ledger.append('start-'+event,'encoder.intent',current,rid)
                    ts=time.monotonic();v,comp,sel=vector_ok(enc.one(t,prep(core,item)),t);need(time.time()<deadline,'encoder exceeded deadline')
                    prefix='variants/'+item['identity']+'/'+t
                    body=write_new(out,prefix+'.npy',npy(v))
                    rec={**current,'schema':SCHEMA,'manifest_sha256':msha,'source_manifest_sha256':SOURCE_MANIFEST,'source_receipt':item['source_receipt'],
                         'parent_context_sha256':PARENT,'lineage_sha256':parent_proof['lineage_sha256'],'partition':parent_proof['partition'],
                         'input_sha256':sha(item['raw']),'output':body,'weight':enc.pins[t],'component_info':comp,'selection':sel,
                         'runtime_inventory_sha256':cfg['runtime_inventory']['sha256'],'qualification_sha256':sha(canonical(qual)),
                         'wall_seconds':time.monotonic()-ts,'new_gp_labels':0,'teaching_eligible':False,
                         'label_exposure':qual['label_exposure'],'tabdpt_known_overlap':known if t=='tabdpt13' else None,
                         'pretraining_overlap':'known overlap' if t=='tabdpt13' and known else 'Unknown; not proof of unseen data'}
                    canonical_json(rec);receipt=write_new(out,prefix+'.json',canonical(rec));done.append({**current,'receipt':receipt,'output':body})
                    ledger.append('done-'+event,'encoder.result',done[-1],rid);progress(out,host,msha,done,'encoding')
                    print(json.dumps({'host':host,'completed_variant_components':len(done),'expected':250,**current}),flush=True)
                del enc;gc.collect();torch.cuda.empty_cache()
            need(len(done)==250 and time.time()<deadline,'incomplete or overdue pilot')
            summary=progress(out,host,msha,done,'complete',elapsed_seconds=time.time()-start)
            write_new(out,'status.json',canonical({**summary,'outputs':done}));ledger.export_jsonl('events.jsonl');ledger.export_csv('events.csv')
            ledger.seal(rid,{'status':'status.json','events':'events.jsonl','preflight':'preflight.json'});return summary
        except BaseException as exc:
            err=type(exc).__name__+': '+str(exc)[:400]
            ledger.append('failed','pilot.failed',{'current':current,'completed':len(done),'error':err},rid)
            failure=progress(out,host,msha,done,'failed',error=err)
            if not (out/'status.json').exists():write_new(out,'status.json',canonical(failure))
            raise


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--manifest',required=True);a.add_argument('--manifest-sha',required=True);a.add_argument('--host',choices=('xbabe2','xbabe3'),required=True);a.add_argument('--preflight-only',action='store_true');v=a.parse_args(argv)
    m=obj({'path':v.manifest,'sha256':v.manifest_sha});need(sha(Path(__file__).read_bytes())==m['worker_sha256'],'worker source drift');print(json.dumps(run(m,v.host,v.manifest_sha,v.preflight_only),sort_keys=True))


if __name__=='__main__':main()
