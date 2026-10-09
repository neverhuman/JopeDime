import copy,os,sys,time,unittest,types,torch
from pathlib import Path
import native_test_fixture as fixture
from dltrain import model as M
from neural_native_batch import collate,objective

class NativeBatchTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixture.NeuralTests.setUpClass()
 def fixture(self,winner_only=False):
  f=fixture.NeuralTests();loaded=f.fixture();data=loaded.data;cid=next(iter(data.fit.pi));records=[]
  for r in data.corpus.records[cid]:
   lab=data.corpus.label(cid,r['program_sha256'])
   records.append(types.SimpleNamespace(program_sha256=lab.program_sha256,tokens=tuple(data.corpus.tokens(cid,lab.program_sha256)),outcome='completed',q=None if winner_only else lab.q,q_opt=None if winner_only else lab.q_opt,q_grid=None if winner_only else lab.q_grid,feasibility=None if winner_only else lab.feas_class,runtime_seconds=None if winner_only else .02,runtime_censored=False,ce_eligible=True,quality_eligible=not winner_only,pairwise_eligible=not winner_only,runtime_eligible=not winner_only,source_kind='explicit_test_fixture'))
  native=types.SimpleNamespace(contexts={cid:data.corpus.contexts[cid]},encodings={cid:data.corpus.encodings[cid]},records={cid:records[:1]if winner_only else records},selection_receipt={'fixture':True})
  return native,cid,data.legal_mask_fn
 def net(self):torch.manual_seed(3);return M.PolicyLM(M.PolicyConfig(282,2051,32,1,4,64,128,dropout=0.))
 def test_sparse_winner_ce_uses_no_fabricated_quality_runtime_or_targets(self):
  native,cid,mask=self.fixture(True);b=collate(native,[cid],legal_mask_fn=mask);net=self.net();out=net(b.tokens,b.lengths,b.ctx_z,b.seq_ctx);loss,report,_=objective(out,b);loss.backward()
  self.assertEqual(report['denominators'],{'ce':1,'q':0,'feas':0,'rt':0,'bt':0,'list':0});self.assertFalse(b.receipt['synthetic_y_created']);self.assertGreater(float(loss.detach()),0);self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.parameters()))
 def test_real_quality_pairwise_and_list_support_have_finite_gradients(self):
  native,cid,mask=self.fixture();b=collate(native,[cid],legal_mask_fn=mask);net=self.net();loss,r,_=objective(net(b.tokens,b.lengths,b.ctx_z,b.seq_ctx),b);loss.backward()
  self.assertEqual(r['denominators']['ce'],1);self.assertEqual(r['denominators']['q'],1);self.assertEqual(r['denominators']['list'],1);self.assertGreater(r['support']['bt_pairs'],0);self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()for p in net.parameters()))
 def test_two_contexts_one_forward_and_per_context_mean_matches_singletons(self):
  native,cid,mask=self.fixture();other='f'*64;native.contexts[other]=copy.deepcopy(native.contexts[cid]);native.encodings[other]=native.encodings[cid].copy();native.records[other]=copy.deepcopy(native.records[cid][:1]);native.records[other][0].quality_eligible=False;native.records[other][0].pairwise_eligible=False;net=self.net()
  both=collate(native,[cid,other],legal_mask_fn=mask);_,r,_=objective(net(both.tokens,both.lengths,both.ctx_z,both.seq_ctx),both)
  a=collate(native,[cid],legal_mask_fn=mask);_,ra,_=objective(net(a.tokens,a.lengths,a.ctx_z,a.seq_ctx),a)
  z=collate(native,[other],legal_mask_fn=mask);_,rz,_=objective(net(z.tokens,z.lengths,z.ctx_z,z.seq_ctx),z)
  for k in r['terms']:
   denom=ra['denominators'][k]+rz['denominators'][k];want=(ra['numerators'][k]+rz['numerators'][k])/denom if denom else 0
   self.assertAlmostEqual(r['terms'][k],want,places=5)
  self.assertEqual(len(both.context_ids),2);self.assertEqual(both.ctx_z.shape,(2,2051))
 def test_missing_quality_cannot_be_enabled_and_duplicate_program_refused(self):
  native,cid,mask=self.fixture(True);native.records[cid][0].quality_eligible=True
  with self.assertRaisesRegex(ValueError,'quality'):collate(native,[cid],legal_mask_fn=mask)
  native,cid,mask=self.fixture(True);native.records[cid]*=2
  with self.assertRaisesRegex(ValueError,'duplicated'):collate(native,[cid],legal_mask_fn=mask)
 def test_unknown_heads_cannot_pollute_sparse_loss_with_nan(self):
  native,cid,mask=self.fixture(True);b=collate(native,[cid],legal_mask_fn=mask);net=self.net();out=net(b.tokens,b.lengths,b.ctx_z,b.seq_ctx)
  out=types.SimpleNamespace(logits=out.logits,q_hat=torch.full_like(out.q_hat,float('nan')),rt_hat=torch.full_like(out.rt_hat,float('nan')),feas_logits=torch.full_like(out.feas_logits,float('nan')))
  loss,r,_=objective(out,b);self.assertTrue(torch.isfinite(loss));self.assertEqual(r['terms']['q'],0)
 def test_illegal_tokens_wrong_vectors_and_long_sequences_refused(self):
  for kind in ['vector','length','grammar']:
   native,cid,mask=self.fixture(True)
   if kind=='vector':native.encodings[cid]=native.encodings[cid][:10]
   elif kind=='length':native.records[cid][0].tokens=(1,)+(3,)*128+(2,)
   else:
    old=mask
    def mask(t):return old(t)*False
   with self.assertRaises(ValueError):collate(native,[cid],legal_mask_fn=mask)
if __name__=='__main__':unittest.main()
