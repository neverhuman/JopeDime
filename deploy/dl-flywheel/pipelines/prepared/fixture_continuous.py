import copy,json,tempfile,time,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import torch
from dltrain import model as M
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.program_io import sha
from neural_native_train import fit_preprocessing,prepare_microbatches,run_microbatches
from neural_storage import checkpoint
from neural_outbox import Outbox
from continuous_runner import run,Plateau
from continuous_state import Recovery
from continuous_cli import retain_tasks
import types,numpy as np
from dltrain import dsl_tokenizer as T
from jope_flywheel.training_snapshot_native import NativeRecord
from neural_native_batch import OBJECTIVE


class ContinuousTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def fixture(self):
  vocab=T.Vocab.from_names({});dsl='(pipe X (ridge_gcv))';tokens=tuple(T.tokenize(dsl,vocab));program={'program_sha256':sha(dsl.encode()),'dsl':dsl,'tokens':list(tokens)}
  record=NativeRecord(program['program_sha256'],tokens,'completed',ce_eligible=True);contexts={};encodings={};records={}
  for i in range(5):
   c=digest({'context':i});lin=digest({'lineage':i if i!=1 else 0});role='validation'if i==4 else'train'
   contexts[c]={'context_id':c,'context_sha256':c,'view_id':'view'+str(i),'dataset_hash':'a'*16,'fold':i,'cut_id':'cut'+str(i),'parent_lineage_sha256':lin,'partition':role,'source_kind':'native_s1_context','observation_dispatch':'pending_native_worker_adapter','gp_hwm_ref':{'fixture':True},'kumo_source_sha256':digest({'kumo':i})};encodings[c]=np.arange(2051,dtype=np.float32)/2051+i;records[c]=(record,)
  # Deliberately permissive finite masks isolate update/continuation invariants;
  # syntax acceptance is separately covered by frozen native-loader tests.
  mask=lambda t:np.ones((len(t)-1,282),dtype=np.bool_)
  snapshot=types.SimpleNamespace(contexts=contexts,identity='a'*64,max_seq=128,vocab=vocab,legal_mask_fn=mask,proposal_programs=[program],receipt={'fixture':True},encoding=lambda c:encodings[c])
  def get(ids,*,epoch,batch_index):return types.SimpleNamespace(contexts={c:contexts[c]for c in ids},encodings={c:encodings[c]for c in ids},records={c:records[c]for c in ids},support={},selection_receipt={'epoch':epoch,'index':batch_index})
  snapshot.winners={cid:records[cid][0]for cid in contexts};snapshot.context_loader=types.SimpleNamespace(get=get);snapshot.batch_manifest={'schema':'jope.native-corpus-batches/v1','snapshot_sha256':snapshot.identity,'role_authority_sha256':'b'*64,'contexts':list(contexts.values()),'pending':[],'refused':[]}
  cfg={'run_id':'fixture','max_seq':128,'objective':OBJECTIVE,'seed':4,'device':'cpu','batch_size':2,'max_updates':3,'max_padded_tokens':4096,'max_pending_emissions':100,'evaluation_panel_contexts':2,'evaluation_every_batches':1,'decode_temperature':1.}
  return snapshot,cfg
 def test_three_real_resumed_updates_rolling_state_and_unchanged_parent(self):
  snapshot,n=self.fixture();n.update(max_updates=3,evaluation_every_batches=1,max_pending_emissions=100,output_cap_bytes=24*1024**3,plateau_patience=5,learning_rate_floor=1e-6)
  torch.manual_seed(4);net=M.PolicyLM(M.PolicyConfig(282,2051,32,1,4,64,128,dropout=0.));opt=torch.optim.AdamW(M.param_groups(net,.01),lr=.0003)
  norm,pre=fit_preprocessing(snapshot);ids=[c for c,r in snapshot.contexts.items()if r['partition']=='train'];packs,total,_=prepare_microbatches(snapshot,ids[:2],epoch=0,batch_index=0,normalizer=norm,max_padded_tokens=4096)
  run_microbatches(net,packs,total,optimizer=opt,deadline=time.time()+60)
  with tempfile.TemporaryDirectory()as d:
   base=Path(d).resolve();m=checkpoint(net,opt,base/'parent',step=429,config_sha256='d'*64);old=(base/'parent/state.safetensors').read_bytes();q=Outbox(base/'old.sqlite',model_origin_sha256=m['parameter_sha256']);q.close()
   ref=lambda name:{'path':str(base/'parent'/name),'sha256':sha((base/'parent'/name).read_bytes())}
   config={'neural':n,'checkpoint_manifest':ref('manifest.json'),'checkpoint_state':ref('state.safetensors'),'parameter_sha256':m['parameter_sha256'],'parent_outbox':str(base/'old.sqlite'),'training_config':{'sha256':'d'*64},'global_deadline_unix':time.time()+100}
   ancestry={'planner_ref':'fixture','head':{'events':861,'committed_batches':429,'unresolved_batches':0,'head_sha256':'d'*64},'checkpoint_sha256':m['parameter_sha256'],'optimizer_steps':429,'completed_epochs':1,'fingerprints':[digest({'oldbatch':i})for i in range(429)],'exposed_context_ids':ids,'presentations':len(ids)}
   result=run(snapshot,snapshot,config,root=base/'run',campaign_sha='e'*64,ancestry=ancestry,preprocessing=canonical(pre),deadline=time.time()+60,authority_check=lambda s:None,adoption_check=lambda s,r:None)
   self.assertEqual(result['absolute_optimizer_steps'],432);self.assertEqual(result['new_optimizer_steps'],3);self.assertEqual((base/'parent/state.safetensors').read_bytes(),old)
   p=json.loads((base/'run/progress.json').read_bytes());self.assertEqual(p['unique_train_data_runs_seen'],4);self.assertGreater(p['total_train_presentations'],4);self.assertEqual(p['model_generated_programs'],6);self.assertEqual(p['published_results'],0)
   self.assertEqual(len(list((base/'run/rolling').iterdir())),2);self.assertLessEqual(len(list((base/'run/best').iterdir())),2);self.assertTrue((base/'run/epochs/000002/state.safetensors').exists())
   for step in range(430,433):
    proof=json.loads((base/'run'/f'batch-{step:09d}.json').read_bytes());self.assertEqual(proof['absolute_optimizer_step'],step);self.assertEqual(proof['checkpoint']['step'],step);self.assertFalse(proof['fresh_generated_rewards_used'])
   control=json.loads((base/'run/optimizer-control.json').read_bytes());self.assertEqual(control['absolute_optimizer_step'],432)
   original=Recovery.committed;calls=[]
   def cleanup_failure(r,path):
    calls.append(path)
    if len(calls)==2:raise OSError('injected rolling cleanup error')
    return original(r,path)
   with patch.object(Recovery,'committed',cleanup_failure):
    with self.assertRaisesRegex(OSError,'cleanup'):run(snapshot,snapshot,config,root=base/'cleanup-failure',campaign_sha='e'*64,ancestry=ancestry,preprocessing=canonical(pre),deadline=time.time()+60,authority_check=lambda s:None,adoption_check=lambda s,r:None)
   failed=json.loads((base/'cleanup-failure/result.json').read_bytes());self.assertEqual(failed['absolute_optimizer_steps'],430);self.assertEqual(failed['planner_verification']['committed_batches'],1);self.assertEqual(failed['planner_verification']['unresolved_batches'],0)
   save=Recovery.save;calls=[]
   def save_failure(r,*args):
    calls.append(args)
    if len(calls)==2:raise OSError('injected checkpoint write error')
    return save(r,*args)
   with patch.object(Recovery,'save',save_failure):
    with self.assertRaisesRegex(OSError,'checkpoint'):run(snapshot,snapshot,config,root=base/'save-failure',campaign_sha='e'*64,ancestry=ancestry,preprocessing=canonical(pre),deadline=time.time()+60,authority_check=lambda s:None,adoption_check=lambda s,r:None)
   failed=json.loads((base/'save-failure/result.json').read_bytes());self.assertEqual(failed['absolute_optimizer_steps'],429);self.assertEqual(failed['planner_verification']['unresolved_batches'],1);self.assertTrue((base/'save-failure/uncertain-physical-state/state.safetensors').exists())
 def test_measured_only_batch_works_without_self_imitation_ce(self):
  s,_=self.fixture();cid=next(iter(s.contexts));old=s.context_loader.get
  def get(ids,**kw):
   b=old(ids,**kw);b.records={k:tuple(replace(r,ce_eligible=False,quality_eligible=True,q=.2,q_opt=.2,q_grid=200000)for r in rows)for k,rows in b.records.items()};return b
  s.context_loader.get=get;packs,total,_=prepare_microbatches(s,[cid],epoch=0,batch_index=0,normalizer=None,max_padded_tokens=4096);self.assertEqual(total['ce'],0);self.assertEqual(total['q'],1)
 def test_all_zero_supported_terms_refused(self):
  s,_=self.fixture();cid=next(iter(s.contexts));old=s.context_loader.get
  def get(ids,**kw):
   b=old(ids,**kw);b.records={k:tuple(replace(r,ce_eligible=False,quality_eligible=False,pairwise_eligible=False,feasibility=None,runtime_eligible=False)for r in rows)for k,rows in b.records.items()};return b
  s.context_loader.get=get
  with self.assertRaisesRegex(ValueError,'supported'):prepare_microbatches(s,[cid],epoch=0,batch_index=0,normalizer=None,max_padded_tokens=4096)
 def test_plateau_reduces_lr_and_keeps_fixed_best(self):
  p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.AdamW([p],lr=.0003);v=Plateau(2,1e-6)
  self.assertTrue(v.observe(1.,o)['improved']);v.observe(1.1,o);r=v.observe(1.2,o);self.assertTrue(r['lr_reduced']);self.assertEqual(r['best_since_continuation_loss'],1.);self.assertEqual(o.param_groups[0]['lr'],.00015)
 def test_adoption_cannot_drop_record_or_change_input(self):
  old={'tasks':[{'variant_identity':'a','context':'old','program_results':[{'sha256':'x'}]}]};new=copy.deepcopy(old);retain_tasks(old,new)
  new['tasks'][0]['program_results']=[]
  with self.assertRaises(ValueError):retain_tasks(old,new)
  new=copy.deepcopy(old);new['tasks'][0]['context']='changed'
  with self.assertRaises(ValueError):retain_tasks(old,new)
 def test_recovery_rejects_foreign_directory(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);root=p/'owned';root.mkdir();r=Recovery(root);foreign=p/'foreign';foreign.mkdir()
   with self.assertRaises(ValueError):r.retire(foreign)
 def test_deadline_immediately_before_optimizer_prevents_step(self):
  s,_=self.fixture();cid=next(iter(s.contexts));b,total,_=prepare_microbatches(s,[cid],epoch=0,batch_index=0,normalizer=None,max_padded_tokens=4096)
  net=M.PolicyLM(M.PolicyConfig(282,2051,32,1,4,64,128,dropout=0.));o=torch.optim.AdamW(net.parameters());before=M.parameter_sha256(net)
  with patch('neural_native_train.tick',side_effect=[None,TimeoutError('expired')]):
   with self.assertRaises(TimeoutError):run_microbatches(net,b,total,optimizer=o,deadline=time.time()+60)
  self.assertEqual(before,M.parameter_sha256(net));self.assertEqual(len(o.state),0)

if __name__=='__main__':unittest.main()
