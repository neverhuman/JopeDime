"""Independent portable state/ancestry regression probes; no live files or CUDA."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from safetensors.torch import load,save
from dltrain import model as M
from jope_flywheel.protocol import canonical,digest
from jope_flywheel.program_io import sha
from neural_storage import checkpoint
from neural_native_stream import BatchStream,fingerprint
from continuous_state import restore,parent_ancestry

class StateReviewTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
 def tearDown(self):self.tmp.cleanup()
 def step(self,net,opt):
  net.train();opt.zero_grad(set_to_none=True)
  t=torch.tensor([[1,3,2],[1,4,2]],dtype=torch.long);lengths=torch.tensor([3,3]);z=torch.arange(4102,dtype=torch.float32).reshape(2,2051)/4102
  out=net(t,lengths,z,torch.tensor([0,1]));loss=out.logits.square().mean()+out.q_hat.square().mean()+out.feas_logits.square().mean()+out.rt_hat.square().mean();loss.backward();opt.step();return float(loss.detach())
 def saved(self):
  torch.manual_seed(9001);net=M.PolicyLM(M.PolicyConfig(282,2051,16,1,2,32,8,dropout=.2));opt=torch.optim.AdamW(M.param_groups(net,.01),lr=.0003)
  self.step(net,opt);p=self.root/'checkpoint';m=checkpoint(net,opt,p,step=1,config_sha256='c'*64)
  return net,opt,p,m
 def refs(self,p):
  return tuple({'path':str(p/n),'sha256':sha((p/n).read_bytes()),'bytes':(p/n).stat().st_size}for n in ('manifest.json','state.safetensors'))
 def mutate(self,p,fn):
  state=load((p/'state.safetensors').read_bytes());fn(state);raw=save(state);(p/'state.safetensors').write_bytes(raw);m=json.loads((p/'manifest.json').read_bytes());m.update(state_sha256=sha(raw),state_bytes=len(raw));(p/'manifest.json').write_bytes(canonical(m))
 def restored(self,p,m):return restore(*self.refs(p),expected_parameter_sha256=m['parameter_sha256'],device='cpu')
 def test_exact_next_dropout_update_matches_uninterrupted_parameters_moments_and_rng(self):
  net,opt,p,m=self.saved();wanted=self.step(net,opt);rng=torch.get_rng_state().clone()
  other,oo,_=self.restored(p,m);got=self.step(other,oo);self.assertEqual(got,wanted)
  self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  for (name,a),(other_name,b)in zip(net.named_parameters(),other.named_parameters()):
   self.assertEqual(name,other_name);self.assertTrue(torch.equal(a,b),name)
   self.assertEqual(set(opt.state[a]),set(oo.state[b]))
   for key in opt.state[a]:self.assertTrue(torch.equal(opt.state[a][key],oo.state[b][key]),name+'.'+key)
 def test_trained_checkpoint_cannot_reset_all_optimizer_moments(self):
  _,_,p,m=self.saved();self.mutate(p,lambda s:[s.pop(k)for k in list(s)if k.startswith('optimizer.')])
  with self.assertRaises(ValueError):self.restored(p,m)
 def test_unknown_optimizer_parameter_cannot_be_silently_ignored(self):
  _,_,p,m=self.saved();self.mutate(p,lambda s:s.update({'optimizer.not_a_parameter.step':torch.tensor(1.)}))
  with self.assertRaises(ValueError):self.restored(p,m)
 def test_partial_optimizer_triplet_refused(self):
  _,_,p,m=self.saved();self.mutate(p,lambda s:s.pop(next(k for k in s if k.endswith('.exp_avg_sq'))))
  with self.assertRaises(ValueError):self.restored(p,m)
 def test_fractional_step_and_negative_second_moment_refused(self):
  for kind in ('step','exp_avg_sq'):
   with self.subTest(kind=kind):
    if (self.root/'checkpoint').exists():
     import shutil;shutil.rmtree(self.root/'checkpoint')
    _,_,p,m=self.saved()
    def edit(s):
     k=next(k for k in s if k.startswith('optimizer.')and k.endswith('.'+kind));s[k]=torch.full_like(s[k],.5 if kind=='step'else-1)
    self.mutate(p,edit)
    with self.assertRaises(ValueError):self.restored(p,m)
 def test_unknown_rng_tensor_refused(self):
  _,_,p,m=self.saved();self.mutate(p,lambda s:s.update({'rng.unknown':torch.tensor([1],dtype=torch.uint8)}))
  with self.assertRaises(ValueError):self.restored(p,m)

class AncestryReviewTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
 def tearDown(self):self.tmp.cleanup()
 def manifest(self):
  rows=[]
  for i in range(5):
   cid=digest({'context':i});rows.append({'context_id':cid,'context_sha256':cid,'view_id':'view'+str(i),'dataset_hash':'a'*16,'fold':i,'cut_id':'cut'+str(i),'parent_lineage_sha256':digest({'family':i}),'partition':'validation'if i==4 else'train','source_kind':'native_s1_context','observation_dispatch':'pending_native_worker_adapter','gp_hwm_ref':{'fixture':True},'kumo_source_sha256':digest({'kumo':i})})
  return canonical({'schema':'jope.native-corpus-batches/v1','snapshot_sha256':'a'*64,'role_authority_sha256':'b'*64,'contexts':rows,'pending':[],'refused':[]})
 def planner(self,name='old.sqlite',origin='0'*64):
  raw=self.manifest();return BatchStream(self.root/name,raw,sha(raw),seed=7,batch_size=2,initial_checkpoint_sha256=origin)
 def update(self,p):
  before=p.summary()['checkpoint_sha256'];step=p.summary()['optimizer_steps_reported']+1;b=p.reserve(before);after=digest({'next':before,'step':step})
  p.commit(b.batch_id,checkpoint_before=before,checkpoint_after=after,optimizer_step=step,ordered_data_run_ids=b.ordered_data_run_ids,evidence={'origin':'native_optimizer','event_sha256':digest({'batch':b.batch_id})});return b
 def complete_parent(self):
  p=self.planner();self.update(p);self.update(p);self.assertEqual(p.summary()['completed_epochs'],1)
  proof=p.verify();checkpoint=p.summary()['checkpoint_sha256'];a=parent_ancestry(self.root/'old.sqlite',proof,checkpoint);return p,a
 def test_complete_parent_head_import_prohibits_all_historical_batches(self):
  old,a=self.complete_parent();p=self.planner('new.sqlite',a['checkpoint_sha256']);p.inherit(a)
  b=p.reserve(a['checkpoint_sha256']);self.assertNotIn(b.fingerprint,a['fingerprints']);self.assertEqual(len({r['parent_lineage_sha256']for r in b.rows}),len(b.rows));p.verify();p.close();old.close()
 def test_partial_current_epoch_refused_even_when_an_older_epoch_completed(self):
  old,a=self.complete_parent();self.update(old);self.assertEqual(old.summary()['completed_epochs'],1)
  with self.assertRaises(ValueError):parent_ancestry(self.root/'old.sqlite',old.verify(),old.summary()['checkpoint_sha256'])
  old.close()
 def test_removed_ancestry_state_is_detected_against_event_and_membership_bans(self):
  old,a=self.complete_parent();p=self.planner('new.sqlite',a['checkpoint_sha256']);p.inherit(a);p.db.execute("DELETE FROM state WHERE key='ancestry'")
  with self.assertRaises(ValueError):p.verify()
  p.close();old.close()
 def test_extra_historical_ban_without_ancestry_is_refused(self):
  p=self.planner();p.db.execute('INSERT INTO inherited_fingerprints VALUES(?)',('e'*64,))
  with self.assertRaises(ValueError):p.verify()
  p.close()
if __name__=='__main__':unittest.main()
