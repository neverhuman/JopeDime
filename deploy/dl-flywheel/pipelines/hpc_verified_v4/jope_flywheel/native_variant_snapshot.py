"""Append verified prepared variants to a native corpus at an epoch boundary.

This module only reads exact S3 versions and builds a new immutable snapshot.
It does not mutate a planner, normalize vectors, train, or release backpressure.
The caller must retain the preceding model/optimizer/RNG and preprocessing.
Original native descriptors stay byte-identical; variants use a discriminated
descriptor containing their actual registered Context, never a fake native ID.
"""
from collections import OrderedDict,Counter
from dataclasses import asdict,replace
import hashlib,importlib,io,json,math,time
from pathlib import Path
import numpy as np
from . import training_snapshot_kumos as K
from . import training_snapshot_native as N
from .program_gate import verified_measurement,ARTIFACTS,object_bytes
from .protocol import Context,canonical,digest,sha256

SCHEMA='jope.native-variant-extension/v1'
MODULES=('jope_flywheel.training_snapshot_native','jope_flywheel.training_snapshot_kumos',
         'jope_flywheel.program_gate','jope_flywheel.score_join','dltrain.labels','dltrain.dsl_tokenizer')
MAX_CACHE=64*1024**2
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()

class Reader:
 """Bounded transfer accounting; no metadata-only artifact verification."""
 def __init__(self,store,limit,deadline):
  need(type(limit)is int and 0<limit<=8*1024**3,'bounded artifact read budget')
  self.store,self.limit,self.deadline=store,limit,deadline;self.cache=OrderedDict();self.cache_bytes=0;self.read_bytes=0;self.unique={}
  self.nativecanonical_json=getattr(store,'nativecanonical_json',None)
  need(callable(self.nativecanonical_json),'pinned native canonicalizer required')
 def get_version(self,ref):
  r=K.normalized_ref(ref);key=tuple(r[k]for k in ('bucket','key','version_id','sha256','bytes'))
  need(time.time()<self.deadline,'extension deadline')
  if key in self.cache:self.cache.move_to_end(key);return self.cache[key],r['version_id']
  need(self.read_bytes+r['bytes']<=self.limit,'extension artifact read budget')
  b=object_bytes(self.store,r);self.read_bytes+=len(b);need(time.time()<self.deadline,'extension deadline after GET')
  self.unique[key]=len(b);self.cache[key]=b;self.cache_bytes+=len(b)
  while self.cache_bytes>MAX_CACHE:
   _,old=self.cache.popitem(last=False);self.cache_bytes-=len(old)
  return b,r['version_id']
 def read(self,r):return self.get_version(r)[0]

def check_authority(spec,native_spec,raw,parents,now):
 p=N.document(raw);need(p.get('schema')=='jope.native-corpus-current-authority/v1'and p.get('passed')is True,'current native role proof')
 need(p.get('helper_sha256')==spec['authority_helper_sha256']and p.get('source_pins')==native_spec['authority_pins'],'exact role helper/source closure')
 need(type(p.get('checked_at_unix'))in(int,float)and 0<=now-p['checked_at_unix']<=60,'fresh role proof')
 excluded=set(p.get('protected_context_ids',[]));views=set(p.get('protected_views',[]))
 for task in spec['tasks']:
  parent=parents[task['parent_lineage_sha256']]
  need(not(set(parent['retained_lineage_context_sha256'])&excluded)and parent['view_id']not in views,'new variant parent excluded')
 return p

def _anchor(a,sealed=True):
 need(a.get('schema')=='jope.flywheel-evidence/v1'and a.get('event_count',0)>0 and a.get('runs'),'encoder ledger anchor')
 sha256(a['head_sha256'])
 if sealed:need(all(r.get('sealed')is True for r in a['runs'].values()),'encoder seal required')

def encoder(reader,task,spec,meta,refs):
 """Read only KumoS; exact reviewed proof hash preserves explicit recovery."""
 pub=N.document(reader.read(task['encoder_publication']));identity=task['variant_identity']
 need(pub.get('schema')in('jope.flywheel-evergreen-encoder-publication/v1','jope.flywheel-evergreen-kumos-publication/v1')and pub['variant_identity']==identity,'versioned variant encoder publication')
 need('kumo_tabular_s'in pub['completed_encoders']and'kumo_tabular_s'not in pub['missing_encoders']and pub['teaching_eligible']is False and pub['lift_eligible']is False,'qualified research KumoS')
 proof=pub['source_proof'];release={'publisher_sha256':pub['publisher_sha256'],'source_sha256':proof['source_sha256'],'manifest_sha256':proof['manifest']['sha256'],'source_proof_sha256':digest(proof)}
 matches=[r for r in spec['encoder_releases']if all(r.get(k)==v for k,v in release.items())];need(len(matches)==1,'exact independently reviewed encoder release/proof')
 release=matches[0];need(set(release)=={'publisher_sha256','source_sha256','manifest_sha256','source_proof_sha256','custody'},'closed encoder release')
 _anchor(proof['coordinator_ledger']);_anchor(proof['phase_ledgers']['ofm'],release['custody']=='original_sealed')
 if release['custody']=='reviewed_finalization':
  f=proof.get('finalization_custody');need(type(f)is dict and proof['producer_phase_sealed']is False and proof['original_coordinator_state']=='failed'and f['original_producer_sealed']is False,'explicit original failed/unsealed custody')
  body=f['status']['body'];need(sha(canonical(body))==f['status']['ref']['sha256']and len(canonical(body))==f['status']['ref']['bytes'],'finalizer exact status body')
  need(body['state']=='complete'and type(body['recomputed_vectors'])is int and body['recomputed_vectors']==0 and body['teaching_eligible']is False,'custody finalizer not recomputation or training')
 else:need(release['custody']=='original_sealed','known custody branch')
 need(K.normalized_ref(pub['prior_dataset_publication']['receipt'])==K.normalized_ref(task['dataset_publication']),'same dataset publication')
 for name,ref in refs.items():need(K.normalized_ref(pub['prior_dataset_publication']['objects'][name])==K.normalized_ref(ref),'dataset object closure')
 for k in('content_sha256','target_sha256','source_target_sha256','parent_lineage_sha256','partition','rows','features','inner_counts','outer_test_rows'):need(pub[k]==meta[k],'encoder dataset binding:'+k)
 entry=pub['encoder_records']['kumo_tabular_s'];record=entry['record'];vector=next((r for r in pub['objects']if r['key'].endswith('/'+identity+'/encodings/kumo_tabular_s.npy')),None)
 need(vector is not None,'KumoS vector ref');record_raw=canonical(record)
 need(sha(record_raw)==entry['record_ref']['sha256']and len(record_raw)==entry['record_ref']['bytes'],'record exact bytes')
 weight=record['producer']['weight'];need(all(weight.get(k)==v for k,v in {'repo':'nvidia/Kumo-Tabular','revision':K.MODEL_REVISION,'file':'small/regressor.pt','sha256':K.WEIGHT_SHA256}.items()),'reviewed KumoS weight only')
 for k,v in {'manifest_sha256':release['manifest_sha256'],'variant_identity':identity,'component':'kumo_tabular_s','dimension':2051,'input_raw_sha256':meta['content_sha256'],'target_sha256':meta['target_sha256'],'parent_lineage_sha256':meta['parent_lineage_sha256'],'partition':meta['partition']}.items():need(record[k]==v,'encoder record binding:'+k)
 need(record['source_metadata']['sha256']==refs['metadata.json']['sha256']and record['source_input']['sha256']==refs['encoder.jtt1.gz']['sha256'],'encoder input objects')
 # Phase-relative output paths are resolved only against this exact pinned release.
 relative=Path(record['output']['path']);out=Path(proof['source_release']['out'])/'ofm'
 need(out.is_absolute()and'..'not in out.parts and proof['source_release']['source_sha256']==proof['source_sha256']and record['phase']=='ofm','pinned source release root/phase')
 need(not relative.is_absolute()and'..'not in relative.parts and relative.as_posix()=='variants/'+identity+'/kumo_tabular_s.npy','phase relative output path')
 need(entry['vector_ref']['path']==str(out/relative),'pinned producer output root')
 for k in ('sha256','bytes'):need(record['output'][k]==entry['vector_ref'][k]==vector[k],'vector content custody')
 q=entry['qualification'];qb=canonical(q['body']);need(q['body'].get('passed')is True and sha(qb)==q['ref']['sha256']==record['qualification']['sha256']and len(qb)==q['ref']['bytes']==record['qualification']['bytes'],'actual qualified encoder body')
 b=reader.read(vector);f=io.BytesIO(b);version=np.lib.format.read_magic(f);need(version in((1,0),(2,0)),'bounded NPY version')
 shape,order,dtype=(np.lib.format.read_array_header_1_0(f)if version==(1,0)else np.lib.format.read_array_header_2_0(f));need(shape==(2051,)and dtype==np.dtype('<f4')and len(b)-f.tell()==8204,'exact KumoS2051 float32')
 z=np.frombuffer(b,dtype='<f4',offset=f.tell());need(np.isfinite(z).all(),'finite full vector');z.setflags(write=False)
 return z,{'object':K.normalized_ref(vector),'record_sha256':sha(record_raw),'custody':release['custody'],'source_proof_sha256':release['source_proof_sha256']}

class MixedLoader:
 def __init__(self,s):self.snapshot=s
 def get(self,context_ids,*,epoch,batch_index):
  s=self.snapshot;ids=list(context_ids);need(ids and len(ids)<=N.MAX_BATCH_CONTEXTS and len(ids)==len(set(ids))and all(c in s.contexts for c in ids),'bounded distinct mixed contexts')
  need(type(epoch)is int and epoch>=0 and type(batch_index)is int and batch_index>=0,'nonnegative mixed cursor')
  old=[c for c in ids if c in s.native.contexts];records={};support={};nested=None
  if old:
   b=s.native.context_loader.get(old,epoch=epoch,batch_index=batch_index);records.update(b.records);support.update(b.support);nested=b.selection_receipt
  for cid in ids:
   if cid in s.variant_records:
    rows=s.variant_records[cid];width=min(s.program_window,len(rows));start=epoch*s.program_window%len(rows);chosen=[rows[(start+j)%len(rows)]for j in range(width)];best=s.best_observed.get(cid)
    if best is not None and not any(r.program_sha256==best.program_sha256 for r in chosen):chosen.append(best)
    records[cid]=tuple(chosen);support[cid]={'programs_declared':len(rows),'programs_selected':len(chosen),'selected_ce':sum(r.ce_eligible for r in chosen),'selected_quality':sum(r.quality_eligible for r in chosen),'quality_baseline_reconstructed':True,'all_programs_exposed':len(rows)==len(chosen),'source_kind':'flywheel_variant'}
  members=[{'context_id':c,'program_ids':[r.program_sha256 for r in records[c]]}for c in ids];receipt={'schema':'jope.mixed-program-window/v1','snapshot_sha256':s.identity,'epoch':epoch,'batch_index':batch_index,'members':members,'membership_sha256':digest(members),'native_window':nested}
  return N.NativeBatch({c:s.contexts[c]for c in ids},{c:s.encoding(c)for c in ids},records,support,receipt)

class MixedSnapshot:
 def encoding(self,c):return self.variant_vectors[c]if c in self.variant_vectors else self.native.encoding(c)
 def legal_mask_fn(self,tokens):return self.native.legal_mask_fn(tokens)

def compose(native,spec_bytes,expected_sha256,store,*,native_spec_bytes,registry_bytes,preprocessing_bytes,authority_callback,deadline):
 """Verify and append; never call planner.adopt_manifest or fit preprocessing."""
 need(sha(spec_bytes)==sha256(expected_sha256),'extension spec SHA');spec=N.document(spec_bytes);ns=N.document(native_spec_bytes)
 keys={'schema','native_snapshot_sha256','parent_registry_sha256','generator_config_sha256','authority_helper_sha256','dependency_sha256','dataset_publishers','encoder_releases','tasks','max_read_bytes','preprocessing_sha256','teacher_rule'}
 need(set(spec)==keys and spec['schema']==SCHEMA and spec['native_snapshot_sha256']==native.identity==sha(native_spec_bytes),'closed exact native parent snapshot')
 need(spec['teacher_rule']=='disabled_for_unpromoted_variants/v1','no first-GP self-imitation CE');sha256(spec['preprocessing_sha256'])
 need(sha(preprocessing_bytes)==spec['preprocessing_sha256'],'preserved normalizer exact bytes');pre=N.document(preprocessing_bytes)
 need(pre.get('schema')=='jope.native-kumos-preprocessing/v1'and pre.get('snapshot_sha256')==native.identity and pre.get('fit_context_ids')==sorted(c for c,r in native.contexts.items()if r['partition']=='train')and pre.get('validation_used_for_fit')is False,'preserved native train-only normalization identity')
 need(all(len(pre[k])==2051 and all(type(x)in(int,float)and math.isfinite(x)for x in pre[k])for k in('mean','scale'))and all(x>0 for x in pre['scale'])and pre['clip']==[-10.,10.],'preserved normalization shape')
 need(set(spec['dependency_sha256'])==set(MODULES),'dependency source closure')
 for name,pin in spec['dependency_sha256'].items():need(sha(Path(importlib.import_module(name).__file__).read_bytes())==pin,'adapter dependency drift:'+name)
 need(sha(registry_bytes)==spec['parent_registry_sha256']==ns['parent_registry']['sha256'],'same full parent registry')
 parents={p['parent_lineage_sha256']:p for p in map(N.document,registry_bytes.splitlines())};native_roles={r['parent_lineage_sha256']:r['partition']for r in native.contexts.values()}
 need(type(spec['tasks'])is list and 0<len(spec['tasks'])<=1000,'bounded variant extension');seen=set()
 for t in spec['tasks']:
  need(set(t)=={'variant_identity','parent_lineage_sha256','context','dataset_publication','encoder_publication','program_results'},'closed variant task')
  identity=sha256(t['variant_identity']);need(identity not in seen,'duplicate variant');seen.add(identity)
  lin=sha256(t['parent_lineage_sha256']);need(lin in parents and lin in native_roles and parents[lin]['partition']==native_roles[lin],'same established parent lineage/role')
  need(type(t['program_results'])is list and len(t['program_results'])<=256,'bounded complete program refs')
 proof=authority_callback();check_authority(spec,ns,proof,parents,time.time());reader=Reader(store,spec['max_read_bytes'],deadline)
 s=MixedSnapshot();s.native=native;s.identity=expected_sha256;s.max_seq=native.max_seq;s.program_window=native.program_window;s.vocab=native.vocab;s.contexts=dict(native.contexts);s.winners=dict(native.winners);s.best_observed={};s.variant_records={};s.variant_vectors={};s.input_ready_contexts={};details={};pending=list(native.receipt['pending']);refused=list(native.receipt['refused']);programs={p['program_sha256']:p for p in native.proposal_programs}
 for index,task in enumerate(spec['tasks']):
  if index%16==0:proof=authority_callback();check_authority(spec,ns,proof,parents,time.time())
  identity=task['variant_identity'];parent=parents[task['parent_lineage_sha256']]
  if task['encoder_publication']is None:
   pending.append({'variant_identity':identity,'parent_lineage_sha256':task['parent_lineage_sha256'],'partition':parent['partition'],'reason':'AWAITING_KUMOS_AND_VERIFIED_GP_CLOSURE'});continue
  pub=N.document(reader.read(task['dataset_publication']));publisher=pub.get('publication_helper_sha256');need(publisher in spec['dataset_publishers'],'reviewed dataset publisher')
  ctx=Context(**task['context']);ks={'generator_config_sha256':spec['generator_config_sha256'],'dataset_publisher_sha256':publisher,'parent_registry_sha256':spec['parent_registry_sha256'],'evaluation_contract':{k:getattr(ctx,k)for k in('evaluator_sha256','vocabulary_sha256','protocol_sha256','units')}}
  ctx,meta,raw,refs=K._dataset(reader,task,ks,parents,{'parent_lineage_sha256':task['parent_lineage_sha256']});z,zproof=encoder(reader,task,spec,meta,refs);cid=ctx.identity;need(cid not in s.contexts,'context collision with native/variant')
  descriptor={'context_id':cid,'source_kind':'flywheel_variant','variant_identity':identity,'parent_lineage_sha256':meta['parent_lineage_sha256'],'partition':meta['partition'],'context':asdict(ctx),'observation_dispatch':'qualified_flywheel_worker_adapter','dataset_publication':K.normalized_ref(task['dataset_publication']),'encoder_publication':K.normalized_ref(task['encoder_publication'])}
  need(cid not in s.input_ready_contexts,'duplicate input-ready context');s.input_ready_contexts[cid]=descriptor;s.variant_vectors[cid]=z
  if not task['program_results']:
   pending.append({'variant_identity':identity,'context_id':cid,'parent_lineage_sha256':meta['parent_lineage_sha256'],'partition':meta['partition'],'reason':'INPUT_READY_AWAITING_NATIVE_MEASURED_PROGRAMS'});continue
  y,_,_=K.decode_input(raw);role='train_fit'if meta['partition']=='train'else'dev';common={'context_sha256':cid,'dataset_hash':identity,'family_id':meta['parent_lineage_sha256'],'split_role':role};rows=[];actual=[];seen_programs=set()
  for ref in task['program_results']:
   entry=N.document(reader.read(ref));need(set(entry['artifacts'])==set(ARTIFACTS),'seven measured program artifacts');dsl=reader.read(entry['artifacts']['program']).decode();measure=verified_measurement(reader,ctx,dsl,entry);need(reader.read(entry['artifacts']['input'])==raw,'same actual variant scored')
   evaluation=N.document(reader.read(entry['artifacts']['evaluation']));baseline=K.mean_baselines(raw,evaluation);ph=measure['program_sha256'];need(ph not in seen_programs,'duplicate measured program');seen_programs.add(ph);tokens=tuple(native.tokenizer.tokenize(dsl,native.vocab));reason=native.token_refusal(tokens)
   if reason:refused.append({'variant_identity':identity,'context_id':cid,'program_sha256':ph,'reason':reason});continue
   walls=[r.get('wall_us')for r in evaluation['inner_cv']];wall=sum(walls)if all(type(w)is int and w>=0 for w in walls)else None
   rows.append({**common,'program_sha256':ph,'skeleton_sha256':native.tokenizer.skeleton_sha256(dsl),'tokens':tokens,'length':len(tokens),'outcome':'completed','rmse_j':measure['inner_fold_rmse'],'baseline_rmse_j':baseline,'wall_us':wall,'backend':'local','shared_seed':False});actual.append({'program_sha256':ph,'result':K.normalized_ref(ref),'evaluation_key':measure['evaluation_key']});programs[ph]={'program_sha256':ph,'tokens':list(tokens),'dsl':dsl}
  _,labs=native.labels.label_context({**common,'y_train':y.tolist()},rows);lm={l.program_sha256:l for l in labs};known=[r for r in rows if lm[r['program_sha256']].q is not None]
  if not rows:pending.append({'variant_identity':identity,'context_id':cid,'reason':'NO_DECODABLE_MEASURED_PROGRAMS'});continue
  best=min(known,key=lambda r:(-lm[r['program_sha256']].q,r['program_sha256']))if known else None;converted=[]
  for r in rows:
   l=lm[r['program_sha256']];qok=l.q is not None;rt=math.expm1(l.rt_target)if l.rt_valid else None
   converted.append(N.NativeRecord(r['program_sha256'],r['tokens'],'completed',l.q if qok else None,l.q_opt if qok else None,l.q_grid if qok else None,l.feas_class if l.feas_valid else None,rt,l.rt_censored,False,qok,qok,l.rt_valid,'flywheel_verified_s1',r['skeleton_sha256']))
  converted.sort(key=lambda r:digest({'domain':'variant-program-rotation/v1','context':cid,'program':r.program_sha256}));s.variant_records[cid]=tuple(converted)
  if best is not None:s.best_observed[cid]=next(r for r in converted if r.program_sha256==best['program_sha256'])
  s.contexts[cid]=descriptor
  details[cid]={'variant_identity':identity,'records':actual,'best_observed_program_sha256':None if best is None else best['program_sha256'],'ce_teacher_qualified':False,'vector':zproof,'parent_lineage_sha256':meta['parent_lineage_sha256'],'partition':meta['partition']}
 final=authority_callback();check_authority(spec,ns,final,parents,time.time());need(time.time()<deadline,'extension deadline before closure')
 s.proposal_programs=[programs[k]for k in sorted(programs)];s.context_loader=MixedLoader(s)
 s.receipt={'schema':'jope.mixed-native-variant-loaded/v1','snapshot_sha256':s.identity,'native_snapshot_sha256':native.identity,'preserved_preprocessing_sha256':spec['preprocessing_sha256'],'counts':{'native_contexts':len(native.contexts),'variants_declared':len(spec['tasks']),'variants_input_ready':len(s.input_ready_contexts),'variants_loaded':len(details),'contexts':len(s.contexts),'partitions':dict(Counter(c['partition']for c in s.contexts.values()))},'variants':details,'input_ready_contexts':s.input_ready_contexts,'pending':pending,'refused':refused,'actual_artifact_read_bytes':reader.read_bytes,'unique_artifact_bytes':sum(reader.unique.values()),'authority_sha256':sha(final),'normalizer_refit':False,'variant_ce_teachers':0,'optimizer_updates':0,'admission_committed':False}
 s.batch_manifest={'schema':'jope.mixed-corpus-batches/v1','snapshot_sha256':s.identity,'role_authority_sha256':sha(final),'contexts':list(s.contexts.values()),'pending':pending,'refused':refused}
 return s
