"""Compact immutable CPU preparation handoff. No optimizer/planner/network I/O.

prepare() accepts the fully verified MixedSnapshot produced by the frozen
native_variant_snapshot.compose; it is not an alternate artifact verifier.
load() requires a separately pinned successful preparation receipt and current
role authority from its caller. The active v1 runner does not import this file.
"""
from dataclasses import asdict
import base64, hashlib, json, math
import numpy as np
from jope_flywheel import native_variant_snapshot as V, training_snapshot_native as N
SCHEMA='jope.prepared-mixed-snapshot/v1'
MAX_BYTES=32*1024**2
KEYS={'schema','extension_sha256','native_snapshot_sha256','preprocessing_sha256','parent_registry_sha256','extension_utf8','receipt','batch_manifest','variant_contexts','input_ready_contexts','vectors_f32le','variant_records','best_observed','variant_programs'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def prepare(s,extension):
 need(type(extension)is bytes and sha(extension)==s.identity,'exact composed extension')
 e=json.loads(extension);need(e['native_snapshot_sha256']==s.native.identity,'same native snapshot')
 ids=set(s.variant_records);need(ids<=set(s.variant_vectors)<=set(s.input_ready_contexts),'complete variant inputs')
 need(set(s.contexts)==set(s.native.contexts)|ids,'only admitted measured contexts')
 programs={r.program_sha256 for rows in s.variant_records.values()for r in rows}
 vectors={}
 for cid,z in s.variant_vectors.items():
  z=np.asarray(z,dtype='<f4');need(z.shape==(2051,)and np.isfinite(z).all(),'finite required vector');vectors[cid]=base64.b64encode(z.tobytes()).decode()
 records={cid:[asdict(r)for r in rows]for cid,rows in s.variant_records.items()}
 need(all(r['ce_eligible']is False for rows in records.values()for r in rows),'no variant imitation teachers')
 doc={'schema':SCHEMA,'extension_sha256':s.identity,'native_snapshot_sha256':s.native.identity,'preprocessing_sha256':e['preprocessing_sha256'],'parent_registry_sha256':e['parent_registry_sha256'],'extension_utf8':extension.decode(),'receipt':s.receipt,'batch_manifest':s.batch_manifest,'variant_contexts':{cid:s.contexts[cid]for cid in sorted(ids)},'input_ready_contexts':s.input_ready_contexts,'vectors_f32le':vectors,'variant_records':records,'best_observed':{cid:r.program_sha256 for cid,r in s.best_observed.items()},'variant_programs':[p for p in s.proposal_programs if p['program_sha256']in programs]}
 raw=canonical(doc);need(len(raw)<=MAX_BYTES,'prepared snapshot cap');return raw

def load(raw,native,*,receipt,expected_extension_sha256,expected_preprocessing_sha256,expected_parent_registry_sha256,authority_check):
 """CPU-only safe hydration of a sealed artifact; no pickle or code execution.

Caller must pin receipt bytes separately, validate its source/campaign authority,
retain old extension tasks and compare all old context descriptors before any
planner.adopt_manifest. Return value is still unadmitted and cannot train itself.
"""
 need(type(raw)is bytes and len(raw)<=MAX_BYTES,'bounded prepared bytes')
 need(receipt['schema']=='jope.prepared-snapshot-receipt/v1'and receipt['state']=='passed'and receipt['optimizer_updates']==receipt['planner_mutations']==receipt['s3_writes']==0,'verified CPU preparation receipt')
 need(receipt['prepared']['sha256']==sha(raw)and receipt['prepared']['bytes']==len(raw),'exact prepared receipt binding')
 d=json.loads(raw);need(type(d)is dict and set(d)==KEYS and d['schema']==SCHEMA,'closed prepared schema')
 need(d['native_snapshot_sha256']==native.identity and d['extension_sha256']==expected_extension_sha256 and d['preprocessing_sha256']==expected_preprocessing_sha256 and d['parent_registry_sha256']==expected_parent_registry_sha256,'unchanged base/normalizer/lineage')
 extension=d['extension_utf8'].encode();need(sha(extension)==expected_extension_sha256,'exact extension bytes');e=json.loads(extension)
 need(e['native_snapshot_sha256']==native.identity and e['preprocessing_sha256']==expected_preprocessing_sha256 and e['parent_registry_sha256']==expected_parent_registry_sha256,'extension identities')
 authority_check(extension)
 s=V.MixedSnapshot();s.native=native;s.identity=d['extension_sha256'];s.max_seq=native.max_seq;s.program_window=native.program_window;s.vocab=native.vocab;s.contexts=dict(native.contexts);s.winners=dict(native.winners);s.best_observed={};s.variant_records={};s.variant_vectors={};s.input_ready_contexts=d['input_ready_contexts']
 ids=set(d['variant_contexts']);need(ids==set(d['variant_records'])and ids<=set(d['vectors_f32le'])and set(d['vectors_f32le'])==set(s.input_ready_contexts)and not(ids&set(native.contexts)),'complete disjoint prepared contexts')
 roles={c['parent_lineage_sha256']:c['partition']for c in native.contexts.values()}
 for cid,c in d['variant_contexts'].items():
  need(c['context_id']==cid and c['source_kind']=='flywheel_variant'and c['parent_lineage_sha256']in roles and c['partition']==roles[c['parent_lineage_sha256']]and c==s.input_ready_contexts[cid],'preserved lineage role/context')
  s.contexts[cid]=c
 for cid,b64 in d['vectors_f32le'].items():
  b=base64.b64decode(b64,validate=True);need(len(b)==2051*4,'exact2051 float32');z=np.frombuffer(b,dtype='<f4');need(np.isfinite(z).all(),'finite vector');z.setflags(write=False);s.variant_vectors[cid]=z
 fields=set(N.NativeRecord.__dataclass_fields__)
 for cid,rows in d['variant_records'].items():
  need(type(rows)is list and 0<len(rows)<=256 and all(set(r)==fields and r['ce_eligible']is False for r in rows),'closed measured record roster')
  parsed=[]
  for r in rows:
   v=dict(r);v['tokens']=tuple(v['tokens']);parsed.append(N.NativeRecord(**v))
  need(len({r.program_sha256 for r in parsed})==len(parsed),'unique measured programs');s.variant_records[cid]=tuple(parsed)
 for cid,ph in d['best_observed'].items():
  need(cid in s.variant_records,'known best context');matches=[r for r in s.variant_records[cid]if r.program_sha256==ph];need(len(matches)==1,'known measured best');s.best_observed[cid]=matches[0]
 programs={p['program_sha256']:p for p in native.proposal_programs}
 for p in d['variant_programs']:
  ph=p['program_sha256'];need(sha(p['dsl'].encode())==ph,'exact GP text')
  if ph in programs:need(programs[ph]==p,'unchanged existing GP vocabulary')
  programs[ph]=p
 need({r.program_sha256 for rows in s.variant_records.values()for r in rows}<=set(programs),'measured program catalog complete');s.proposal_programs=[programs[k]for k in sorted(programs)]
 s.receipt=d['receipt'];s.batch_manifest=d['batch_manifest'];need(s.batch_manifest['snapshot_sha256']==s.identity and {c['context_id']:c for c in s.batch_manifest['contexts']}==s.contexts,'exact batching membership')
 need(s.receipt['snapshot_sha256']==s.identity and s.receipt['native_snapshot_sha256']==native.identity and s.receipt['preserved_preprocessing_sha256']==expected_preprocessing_sha256 and s.receipt['optimizer_updates']==0 and s.receipt['admission_committed']is False,'prepared is not admitted')
 s.extension_raw=extension;s.context_loader=V.MixedLoader(s);authority_check(extension);return s
