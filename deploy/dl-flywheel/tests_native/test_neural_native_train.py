import copy,json,tempfile,time,types,unittest
from pathlib import Path
import torch,numpy as np
import test_neural_native_batch as fixture
from dltrain import model as M
from jope_flywheel.protocol import canonical,digest
from neural_native_train import run,prepare_microbatches,run_microbatches,fit_preprocessing,OBJECTIVE
from neural_native_stream import BatchStream,Waiting,validate_manifest
from neural_outbox import Outbox

class NativeTrainingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixture.NativeBatchTests.setUpClass()
 def fixture(self):
  f=fixture.NativeBatchTests();native,cid,mask=f.fixture(True);record=native.records[cid][0];source=native.encodings[cid];contexts={};encodings={};records={}
  for i in range(5):
   c=digest({'context':i});lin=digest({'lineage':i if i!=1 else 0});role='validation'if i==4 else'train'
   contexts[c]={'context_id':c,'context_sha256':c,'view_id':'view'+str(i),'dataset_hash':'a'*16,'fold':i,'cut_id':'cut'+str(i),'parent_lineage_sha256':lin,'partition':role,'source_kind':'native_s1_context','observation_dispatch':'pending_native_worker_adapter','gp_hwm_ref':{'fixture':True},'kumo_source_sha256':digest({'kumo':i})};encodings[c]=np.asarray(source+float(i),dtype=np.float32);records[c]=(copy.deepcopy(record),)
  from dltrain import dsl_tokenizer as T
  vocab=T.Vocab.from_names({});tokens=record.tokens;program={'program_sha256':record.program_sha256,'dsl':T.detokenize(tokens,vocab),'tokens':list(tokens)}
  snapshot=types.SimpleNamespace(contexts=contexts,identity='a'*64,max_seq=128,vocab=vocab,legal_mask_fn=mask,proposal_programs=[program],receipt={'fixture':True},encoding=lambda c:encodings[c])
  def get(ids,*,epoch,batch_index):return types.SimpleNamespace(contexts={c:contexts[c]for c in ids},encodings={c:encodings[c]for c in ids},records={c:records[c]for c in ids},support={},selection_receipt={'epoch':epoch,'index':batch_index})
  snapshot.winners={cid:records[cid][0]for cid in contexts};snapshot.context_loader=types.SimpleNamespace(get=get);snapshot.batch_manifest={'schema':'jope.native-corpus-batches/v1','snapshot_sha256':snapshot.identity,'role_authority_sha256':'b'*64,'contexts':list(contexts.values()),'pending':[{'context_id':'c'*64,'reason':'fixture_pending'}],'refused':[]}
  return snapshot
 def config(self):return {'run_id':'fixture-full-native','max_seq':128,'initialization':'fresh','input_dimension':2051,'objective':OBJECTIVE,'seed':4,'device':'cpu','d_model':32,'layers':1,'heads':4,'d_ff':64,'dropout':0.,'learning_rate':.0003,'batch_size':2,'max_updates':2,'max_epochs':1,'max_padded_tokens':4096,'max_pending_emissions':8,'evaluation_panel_contexts':2,'evaluation_every_batches':1,'decode_temperature':1.,'authority_check_every_batches':1}
 def test_actual_optimizer_all_four_train_contexts_durable_two_proposals_per_batch(self):
  snap=self.fixture();checks=[]
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve()/'run';result=run(snap,self.config(),output=root,deadline=time.time()+60,authority_check=lambda:checks.append(1));r=json.loads((root/'result.json').read_text());p=json.loads((root/'progress.json').read_text())
   self.assertEqual(result['optimizer_steps'],2);self.assertEqual(r['planner']['completed_epochs'],1);self.assertEqual(r['planner']['unique_exposed_data_runs'],4);self.assertEqual(r['outbox']['observations'],4);self.assertEqual(r['outbox']['unresolved'],4);self.assertEqual(p['published_results'],0);self.assertEqual(p['winrate_vs_random'],None);self.assertEqual(len(checks),3)
   self.assertNotEqual(r['initial_checkpoint']['parameter_sha256'],r['current_parameter_sha256']);self.assertEqual(len(list((root/'checkpoints').glob('*/state.safetensors'))),3)
   for step in [1,2]:
    b=json.loads((root/f'batch-{step:06d}.json').read_text());self.assertFalse(b['fresh_generated_rewards_used']);self.assertEqual(len(b['emission_ack']['emission_ids']),2)
 def test_backpressure_stops_before_unacknowledged_optimizer(self):
  cfg=self.config();cfg['max_pending_emissions']=2
  with tempfile.TemporaryDirectory()as d:
   run(self.fixture(),cfg,output=Path(d).resolve()/'run',deadline=time.time()+60,authority_check=lambda:None);r=json.loads((Path(d).resolve()/'run/result.json').read_text());self.assertEqual(r['state'],'waiting_for_mandatory_gp_capacity');self.assertEqual(len(r['proofs']),1);self.assertIsNone(r['planner']['pending_batch'])
 def test_current_role_failure_prevents_next_parameter_update(self):
  calls=[]
  def check():
   calls.append(1)
   if len(calls)==3:raise ValueError('role changed')
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve()/'run'
   with self.assertRaisesRegex(ValueError,'role changed'):run(self.fixture(),self.config(),output=root,deadline=time.time()+60,authority_check=check)
   r=json.loads((root/'result.json').read_text());self.assertEqual(len(r['proofs']),1);self.assertIsNone(r['planner']['pending_batch'])
 def test_train_only_normalizer_ignores_validation_perturbation(self):
  s=self.fixture();_,a=fit_preprocessing(s);old=s.encoding;s.encoding=lambda c:old(c)*1000 if s.contexts[c]['partition']=='validation'else old(c);_,b=fit_preprocessing(s);self.assertEqual(a,b)
 def test_microbatch_gradients_equal_full_context_complete_macro_batch(self):
  s=self.fixture();ids=[c for c,r in s.contexts.items()if r['partition']=='train'][:2];norm,_=fit_preprocessing(s);large,total,_=prepare_microbatches(s,ids,epoch=0,batch_index=0,normalizer=norm,max_padded_tokens=4096)
  # Same selected contexts, split only between contexts.
  small=[prepare_microbatches(s,[c],epoch=0,batch_index=0,normalizer=norm,max_padded_tokens=4096)[0][0]for c in ids]
  torch.manual_seed(2);a=M.PolicyLM(M.PolicyConfig(282,2051,32,1,4,64,128,dropout=0.));b=copy.deepcopy(a);oa=torch.optim.SGD(a.parameters(),lr=.001);ob=torch.optim.SGD(b.parameters(),lr=.001)
  ra=run_microbatches(a,large,total,optimizer=oa,deadline=time.time()+30);rb=run_microbatches(b,small,total,optimizer=ob,deadline=time.time()+30)
  self.assertAlmostEqual(ra['loss'],rb['loss'],places=5)
  for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(x,y,atol=1e-7,rtol=1e-6))
 def test_no_truncation_when_single_context_exceeds_budget(self):
  s=self.fixture();cid=next(iter(s.contexts));old=s.context_loader.get
  def get(ids,**kw):
   v=old(ids,**kw);v.records[cid]=v.records[cid]*100;return v
  s.context_loader.get=get
  with self.assertRaisesRegex(ValueError,'one full-context'):prepare_microbatches(s,[cid],epoch=0,batch_index=0,normalizer=None,max_padded_tokens=128)
 def test_native_manifest_uses_actual_context_not_invented_raw_hash(self):
  s=self.fixture();raw=canonical(s.batch_manifest);_,rows=validate_manifest(raw,digest(s.batch_manifest));self.assertEqual(set(rows),set(s.contexts));self.assertNotIn('target_sha256',next(iter(rows.values()))['data_run'])
  s.batch_manifest['contexts'][0]['target_sha256']='d'*64
  with self.assertRaisesRegex(ValueError,'closed native'):validate_manifest(canonical(s.batch_manifest),digest(s.batch_manifest))
 def test_planner_distinct_lineage_exhaustion_no_repeat_and_resume(self):
  s=self.fixture();raw=canonical(s.batch_manifest)
  with tempfile.TemporaryDirectory()as d:
   p=Path(d).resolve()/'p.sqlite';q=BatchStream(p,raw,digest(s.batch_manifest),seed=1,batch_size=2,initial_checkpoint_sha256='0'*64);seen=[];before='0'*64
   for step in range(1,5):
    if q.summary()['completed_epochs']:break
    b=q.reserve(before);self.assertEqual(len({r['parent_lineage_sha256']for r in b.rows}),len(b.rows));after=digest({'checkpoint':step});q.commit(b.batch_id,checkpoint_before=before,checkpoint_after=after,optimizer_step=step,ordered_data_run_ids=b.ordered_data_run_ids,evidence={'origin':'native_optimizer','event_sha256':digest({'event':step})});seen+=list(b.ordered_data_run_ids);before=after
   self.assertEqual(len(seen),len(set(seen)));self.assertEqual(q.summary()['completed_epochs'],1);q.close();q=BatchStream(p,raw,digest(s.batch_manifest),seed=1,batch_size=2,initial_checkpoint_sha256='0'*64);self.assertGreaterEqual(q.verify()['committed_batches'],2);q.close()
if __name__=='__main__':unittest.main()
