"""Original S1 identities and target-masked inputs, never synthetic variants.

Only registered training rows are decoded. Original outer-test bytes are hashed
opaquely; target cells are overwritten without decoding, exactly as S1 admitted.
No original input/context is rewritten and no old score is relabelled.
"""
from dataclasses import dataclass
import gzip,hashlib,io,json,math,struct,time
from pathlib import Path
import numpy as np
MAX_RAW=8*1024**2
WORKER_SHA='e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8'
CANONICALIZER_SHA='69975947982bcffb0a55fde96f85b68f043e107e0d946eadffcdac7311f026c1'
PROTOCOL='jope.s1-two-fold/v1'
class Unsupported(ValueError):pass
def need(ok,message):
 if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(x):return sha(canonical(x))
def hash64(x):return type(x)is str and len(x)==64 and all(c in'0123456789abcdef'for c in x)
def bound(ref,cap=MAX_RAW):
 p=Path(ref['path']);need(p.is_absolute()and '..'not in p.parts and all(not x.is_symlink()for x in(p,*p.parents)),'absolute nonsymlink source')
 before=p.stat();need(0<before.st_size<=cap,'bounded source')
 with p.open('rb')as f:body=f.read(cap+1)
 after=p.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'changed source')
 need(sha(body)==ref['sha256']and('bytes'not in ref or len(body)==ref['bytes']),'source SHA/size');return body
@dataclass(frozen=True)
class OriginalScoreContext:
 identity:str
 dataset_hash:str
 cut_id:str
 split_sha256:str
 input_sha256:str
 target_sha256:str
 evaluator_sha256:str=WORKER_SHA
 @property
 def dataset_sha256(self):
  """Compatibility adapter: native score rows use catalog dataset_hash (16 hex)."""
  return self.dataset_hash

def masked_training(raw,row):
 """Return admitted original bytes and only train arrays; never read testvalues."""
 need(type(raw)is bytes and 17<=len(raw)<=MAX_RAW and raw[:4]==b'JTT1','original JTT1')
 nt,nq,p=struct.unpack_from('<III',raw,4);k=raw[16];shape=row['shape'];stride=(p+1)*8;end=17+nt+nt*stride
 need(k==2 and nt>=50 and nq>=10 and 1<=p<=512 and len(raw)==17+nt+(nt+nq)*stride,'qualified original native shape')
 need((nt,nq,p,k)==(shape['n_train'],shape['n_test'],shape['p'],shape['k']),'registered header mismatch')
 folds=np.frombuffer(raw,dtype=np.uint8,count=nt,offset=17).copy();need(set(folds.tolist())=={0,1},'two original inner folds')
 counts=[int((folds==i).sum())for i in range(2)];need(counts==shape['fold_sizes']and min(counts)>=25,'registered foldcounts')
 train=np.frombuffer(raw,dtype='<f8',count=nt*(p+1),offset=17+nt).reshape(nt,p+1).copy();need(np.isfinite(train).all(),'finite training values')
 # Opaque slices; no conversion/inspection of any test feature or target value.
 masked=bytearray(raw)
 for i in range(nq):
  offset=end+i*stride+p*8;masked[offset:offset+8]=b'\0'*8
 masked=bytes(masked)
 need(masked[:end]==raw[:end]and len(masked)==row['s1_input']['bytes']and sha(masked)==row['s1_input']['sha256'],'original admitted maskedinputSHA')
 for a in(train,folds):a.setflags(write=False)
 return masked,train[:,:p],train[:,p],folds

class NativeInputs:
 def __init__(self,input_refs,contexts,*,reader=bound,authority,authority_pins):
  need(callable(reader)and callable(authority),'pinned source reader/current authority required');self.reader=reader;self.authority=authority;self.authority_pins=authority_pins;self.contexts=contexts;self.rows={};self.sources={}
  need(type(contexts)is dict and contexts and all(k==v['context_id']==v['context_sha256']and hash64(k)for k,v in contexts.items()),'original snapshot context identities')
  for ref in input_refs:
   doc=json.loads(reader(ref));need(doc['schema']in('jope.dl-s1-prod-input-list/v1','jope.dl-s1-prod-input-list-join/v1'),'native input registry schema')
   for row in doc['admitted']:
    cid=row['context_sha256'];need(cid not in self.rows,'duplicate native input context');self.rows[cid]=row;self.sources[cid]=ref
 def check_authority(self,descriptor,phase):
  cid=descriptor['context_id'];need(cid in self.contexts and descriptor==self.contexts[cid],'descriptor differs from frozen native snapshot')
  need(descriptor['source_kind']=='native_s1_context'and descriptor['context_sha256']==cid,'original native identity')
  need(phase in('training','validation')and descriptor['partition']=={'training':'train','validation':'validation'}[phase],'inherited phase/lineage partition')
  proof=self.authority();now=time.time();need(proof['schema']=='jope.native-corpus-current-authority/v1'and proof['passed']is True and proof['source_pins']==self.authority_pins and 0<=now-proof['checked_at_unix']<=60,'fresh exact source role authority')
  need(cid not in set(proof['protected_context_ids'])and descriptor['view_id']not in set(proof['protected_views']),'current protected original context')
  # The pinned target itself is checked on every dispatch; absence never means eligible.
  target=json.loads(self.reader(self.authority_pins['target']));need(descriptor['view_id']in set(target['views']),'outside current exact native target')
  return proof
 def describe(self,descriptor,phase):
  proof=self.check_authority(descriptor,phase);cid=descriptor['context_id']
  if cid not in self.rows:raise Unsupported('original context has no registered native input locator; leave pending')
  row=self.rows[cid];need((row['view_id'],row['dataset_hash'],row['fold'],row['cut_id'])==(descriptor['view_id'],descriptor['dataset_hash'],descriptor['fold'],descriptor['cut_id']),'native context/cut/fold identity mismatch')
  source=self.sources[cid];doc=json.loads(self.reader(source));need(any(r==row for r in doc['admitted']),'immutable admitted input row')
  cr={'path':str(Path(row['context_dir'])/'context.json'),'sha256':row['context_json_sha256']};context=json.loads(self.reader(cr))
  need(context['context_sha256']==cid and context['view_id']==row['view_id']and context['dataset_hash']==row['dataset_hash']and context['fold']==row['fold'],'original context JSON identity')
  tr=row['teacher'];zipped=self.reader({'path':tr['path'],'sha256':tr['gz_sha256'],'bytes':tr['gz_bytes']})
  with gzip.GzipFile(fileobj=io.BytesIO(zipped))as f:raw=f.read(MAX_RAW+1)
  need(len(raw)==tr['raw_bytes']<=MAX_RAW and sha(raw)==tr['raw_sha256'],'original teacher bytes')
  masked,x,y,folds=masked_training(raw,row)
  data_sha=context['dataset_hash'];need(type(data_sha)is str and len(data_sha)==16 and all(ch in'0123456789abcdef'for ch in data_sha),'native catalog dataset identity')
  score=OriginalScoreContext(cid,data_sha,row['cut_id'],sha(folds.tobytes()),sha(masked),sha(y.astype('<f8').tobytes()))
  info={'schema':'jope.native-original-measurement-input/v1','context':descriptor,'input_list':source,'registered_row_sha256':digest(row),'context_json':cr,'teacher':tr,'s1_input':row['s1_input'],'rows':len(y),'features':x.shape[1],'inner_counts':row['shape']['fold_sizes'],'original_outer_test_rows':row['shape']['n_test'],'outer_test_values_decoded':False,'test_targets_zeroed_without_decoding':True,'fit_target_sha256':sha(y.astype('<f8').tobytes()),'fit_feature_sha256':sha(x.astype('<f8').tobytes()),'split_sha256':sha(folds.tobytes()),'original_s1_role':row['p3']['assignment'],'inherited_snapshot_partition':descriptor['partition'],'authority_sha256':digest(proof),'native_worker_sha256':WORKER_SHA,'native_protocol':PROTOCOL,'teaching_eligible':False}
  return score,masked,x,y,folds,info
