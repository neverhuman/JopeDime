import copy,gzip,hashlib,io,json,sys,time,unittest
from pathlib import Path
import numpy as np
from jope_flywheel import training_snapshot_native as N
from jope_flywheel.protocol import canonical,partition,digest
from dltrain import dsl_tokenizer as T,labels,automaton


class Store:
 def __init__(self):self.b={};self.calls=[]
 def put(self,name,b):
  self.b['/fixture/'+name]=b;return {'path':'/fixture/'+name,'bytes':len(b),'sha256':N.byte_sha(b)}
 def doc(self,name,x):return self.put(name,canonical(x))
 def read(self,r):self.calls.append(r['path']);return self.b[r['path']]

def npy(a):
 f=io.BytesIO();np.save(f,a,allow_pickle=False);return f.getvalue()

class NativeTests(unittest.TestCase):
 def fixture(self):
  st=Store();v=T.Vocab.from_names({});cid='c'*64;view='eeeeeeeeeeeeeeee__r0__f0__fixture';y=np.arange(1,9,dtype='<f8');folds=np.array([0,1]*4,dtype=np.uint8)
  baselines=[float(np.sqrt(np.mean((y[folds==j]-np.mean(y[folds!=j]))**2)))for j in [0,1]]
  c={'context_sha256':cid,'view_id':view,'dataset_hash':'e'*16,'fold':0,'cut_id':'fixture-native-cut-v1','family_id':'fam-'+('f'*64),'split_role':'train_fit','y_train':y.tolist(),'n_records':96}
  rows=[]
  for i in range(96):
   dsl=f'(pipe X (ridge alpha={i+1}.0))';ts=T.tokenize(dsl,v)
   rows.append({'context_sha256':cid,'program_sha256':N.byte_sha(dsl.encode()),'skeleton_sha256':T.skeleton_sha256(dsl),'tokens':ts,'length':len(ts),'outcome':'completed','rmse_j':[q*(.5+i/200)for q in baselines],'baseline_rmse_j':baselines,'wall_us':1000+i,'backend':'local','shared_seed':False,'family_id':c['family_id'],'dataset_hash':c['dataset_hash'],'split_role':'train_fit'})
  sh={'schema':'jope.dl-s1-tail-shard/v1','context':c,'records':rows,'source':{'protocol':'jope.s1-two-fold/v1','vocab_sha256':v.sha256}}
  row=rows[0];dsl=T.detokenize(row['tokens'],v);ph=row['program_sha256']
  meta={'n':1,'cids':[cid],'view_id':[view],'dataset':[c['dataset_hash']],'fold':[0],'program_sha256':[ph]}
  refs={'meta.json':st.doc('meta.json',meta),'hwm_programs.json':st.doc('hwm_programs.json',{'programs':[{'cid':cid,'program_sha256':ph,'dsl':dsl}]}),'gid.npy':st.put('gid.npy',npy(np.array([0],dtype=np.int32))),'blocks_kumo_tabular_s.npy':st.put('blocks_kumo_tabular_s.npy',npy(np.ones((1,2051),np.float32))),'encoding_sha256.json':st.doc('encoding_sha256.json',{'kumo_tabular_s':{cid:'d'*64}})}
  refs['MANIFEST.json']=st.doc('MANIFEST.json',{'files':{k:r['sha256']for k,r in refs.items()}})
  lin='a'*64;ns='b'*64;parent={'bundle_manifest_sha256':refs['MANIFEST.json']['sha256'],'gid_sha256':refs['gid.npy']['sha256'],'parent_lineage_sha256':lin,'partition':partition(lin,ns),'split_namespace_sha256':ns,'retained_group':0,'retained_lineage_context_sha256':[cid]}
  inv={'rows':[{'context_sha256':cid,'view_id':c['view_id'],'dataset_hash':c['dataset_hash'],'fold':c['fold'],'cut':c['cut_id'],'gp_index':{'outcome':'selection_reached','main':{'fixture':True}}}]}
  authpins={'target':st.doc('target.json',{'gp_target':1,'views':[c['view_id']]}),'role':st.doc('role.json',{'fixture':True}),'layers':st.doc('layers.json',{'fixture':True})}
  sr=st.doc('shards/'+cid[:2]+'/'+cid+'.json',sh);idx={'context_sha256':cid,'view_id':c['view_id'],'n_records':len(sh['records']),'shard':'shards/'+cid[:2]+'/'+cid+'.json','sha256':sr['sha256'],'bytes':sr['bytes']}
  import struct
  train=np.column_stack([np.arange(8,dtype='<f8'),y]);outer=np.full((2,2),np.nan,dtype='<f8')
  raw=b'JTT1'+struct.pack('<IIIB',8,2,1,2)+folds.tobytes()+train.tobytes()+outer.tobytes();tb=gzip.compress(raw,mtime=0);teacher=st.put('teacher.jtt1.gz',tb)
  context=st.doc('context.json',{'context_sha256':cid,'view_id':view,'files':{'teacher.jtt1.gz':{'sha256':teacher['sha256']}}})
  inp={'context_sha256':cid,'view_id':view,'dataset_hash':c['dataset_hash'],'teacher':{'path':teacher['path'],'gz_sha256':teacher['sha256'],'gz_bytes':teacher['bytes']},'context_dir':'/fixture','context_json_sha256':context['sha256']}
  gd={'mode':'SYNTAX','vocab_sha256':v.sha256,'signatures':{'schema':'jope.dsl-signatures/v1','operations':[]}}
  # Actual pinned native signature schema from the data-independent oracle.
  gd['signatures']=automaton.build_signatures([])
  gd['grammar']=automaton.build_grammar(gd['signatures'],v,640)
  spec={'prior_ancestry':None,'schema':N.SCHEMA,'components':{'kumo_tabular_s':2051},'max_seq':640,'program_window':8,'consumer_source_sha256':{m:N.byte_sha(Path(sys.modules[m].__file__).read_bytes())for m in N.MODULES},'vocab_sha256':v.sha256,'grammar':st.doc('grammar.json',gd),'authority_pins':authpins,'bundle':refs,'parent_registry':st.put('parents.jsonl',canonical(parent)+b'\n'),'inventory':st.doc('inventory.json',inv),'input_lists':[st.doc('inputs.json',{'admitted':[inp]})],'context_json_bytes':{context['sha256']:context['bytes']},'tails':[{'root':'/fixture','header':st.doc('header.json',{'vocab_sha256':v.sha256,'shard_schema':'jope.dl-s1-tail-shard/v1'}),'index':st.put('index.jsonl',canonical(idx)+b'\n')}]}
  authority={'schema':'jope.native-corpus-current-authority/v1','passed':True,'checked_at_unix':100,'source_pins':authpins}
  return st,spec,authority,cid
 def load(self,st,spec,auth):
  b=canonical(spec);return N.load_snapshot(b,N.byte_sha(b),st,authority_bytes=canonical(auth),now=101)
 def test_real_native_detail_targets_and_baseline(self):
  st,sp,a,c=self.fixture();s=self.load(st,sp,a);b=s.context_loader.get([c],epoch=0,batch_index=0)
  self.assertTrue(b.support[c]['quality_baseline_reconstructed']);self.assertTrue(all(r.q is not None for r in b.records[c]));self.assertIsNone(b.data);self.assertEqual(len(b.encodings[c]),2051)
  self.assertEqual(sum(r.ce_eligible for r in b.records[c]),1)
  self.assertTrue(all(type(r.feasibility)is int for r in b.records[c]));self.assertTrue(all(r.runtime_seconds>=0 for r in b.records[c]))
 def test_teacher_target_mutation_refused_even_with_repin(self):
  st,sp,a,c=self.fixture();sr=sp['input_lists'][0];d=N.document(st.read(sr));tr=d['admitted'][0]['teacher'];raw=bytearray(gzip.decompress(st.b[tr['path']]));raw[17+8+8:17+8+16]=np.float64(999).tobytes();b=gzip.compress(raw);tr['gz_sha256']=N.byte_sha(b);tr['gz_bytes']=len(b);st.b[tr['path']]=b
  cp=N.document(st.b['/fixture/context.json']);cp['files']['teacher.jtt1.gz']['sha256']=tr['gz_sha256'];cr=st.doc('context.json',cp);d['admitted'][0]['context_json_sha256']=cr['sha256'];sp['context_json_bytes'][cr['sha256']]=cr['bytes'];sp['input_lists']=[st.doc('inputs.json',d)]
  with self.assertRaisesRegex(ValueError,'targets differ'):self.load(st,sp,a).context_loader.get([c],epoch=0,batch_index=0)
 def test_source_tamper_fail(self):
  st,sp,a,c=self.fixture();st.b['/fixture/meta.json']+=b' '
  with self.assertRaisesRegex(ValueError,'source bytes/SHA'):self.load(st,sp,a)
 def test_stale_authority(self):
  st,sp,a,c=self.fixture();a['checked_at_unix']=0
  with self.assertRaisesRegex(ValueError,'freshness'):self.load(st,sp,a)
 def test_wrong_authority_pins(self):
  st,sp,a,c=self.fixture();a['source_pins']={}
  with self.assertRaisesRegex(ValueError,'source bindings'):self.load(st,sp,a)
 def test_missing_detail_is_ce_only_not_fake_quality(self):
  st,sp,a,c=self.fixture();sp['tails']=[];s=self.load(st,sp,a);b=s.context_loader.get([c],epoch=0,batch_index=0);r=b.records[c][0]
  self.assertTrue(r.ce_eligible);self.assertIsNone(r.q);self.assertIsNone(r.runtime_seconds);self.assertIsNone(b.data)
 def test_missing_teacher_masks_quality(self):
  st,sp,a,c=self.fixture();sp['input_lists']=[];s=self.load(st,sp,a);b=s.context_loader.get([c],epoch=0,batch_index=0);self.assertFalse(b.support[c]['quality_baseline_reconstructed']);self.assertTrue(all(r.q is None for r in b.records[c]))
 def test_rotation_reproducible_and_exhaustive(self):
  st,sp,a,c=self.fixture();s=self.load(st,sp,a);seen=set()
  for e in range(12):
   b=s.context_loader.get([c],epoch=e,batch_index=e);b2=s.context_loader.get([c],epoch=e,batch_index=e);self.assertEqual(b.selection_receipt,b2.selection_receipt);seen.update(r.program_sha256 for r in b.records[c])
  self.assertEqual(len(seen),96)
 def test_other_encoders_never_read(self):
  st,sp,a,c=self.fixture();self.load(st,sp,a).context_loader.get([c],epoch=0,batch_index=0);self.assertFalse(any('tabpfn' in p or 'hyperion' in p or 'dope' in p for p in st.calls))
 def test_legal_masks_native_shape(self):
  st,sp,a,c=self.fixture();s=self.load(st,sp,a);r=s.winners[c];m=s.legal_mask_fn(r.tokens);self.assertEqual(m.shape,(len(r.tokens)-1,282));self.assertTrue(all(m[i,t]for i,t in enumerate(r.tokens[1:])))
 def test_cross_context_shard_refuses(self):
  st,sp,a,c=self.fixture();idx=N.document(st.read(sp['tails'][0]['index']));p='/fixture/'+idx['shard'];x=N.document(st.b[p]);x['context']['view_id']='other';r=st.doc(idx['shard'],x);idx.update(sha256=r['sha256'],bytes=r['bytes']);sp['tails'][0]['index']=st.put('index.jsonl',canonical(idx)+b'\n')
  with self.assertRaisesRegex(ValueError,'shard/context mismatch'):self.load(st,sp,a).context_loader.get([c],epoch=0,batch_index=0)
 def test_duplicate_lineage_refuses(self):
  st,sp,a,c=self.fixture();b=st.read(sp['parent_registry']);sp['parent_registry']=st.put('parents.jsonl',b+b)
  with self.assertRaisesRegex(ValueError,'overlapping'):self.load(st,sp,a)
 def test_long_tokens_explicitly_refused(self):
  st,sp,a,c=self.fixture();sp['max_seq']=16;gd=N.document(st.read(sp['grammar']));gd['grammar']=automaton.build_grammar(gd['signatures'],T.Vocab.from_names({}),16);sp['grammar']=st.doc('grammar.json',gd);s=self.load(st,sp,a);self.assertFalse(s.winners[c].tokens);self.assertEqual(s.receipt['refused'][0]['reason'],'over_model_max_seq')
 def test_explicit_640_grammar_cap(self):
  st,sp,a,c=self.fixture();s=self.load(st,sp,a);self.assertEqual(s.grammar.max_seq,640)

if __name__=='__main__':unittest.main()
