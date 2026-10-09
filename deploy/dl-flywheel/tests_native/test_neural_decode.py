import sys,time,unittest,torch
from pathlib import Path
from dltrain import model as M,dsl_tokenizer as T
from jope_flywheel.program_io import sha
from neural_core import ProposalTrie
from neural_decode import BatchedTrie

class DecoderTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def setUp(self):
  torch.manual_seed(31);self.vocab=T.Vocab.from_names({});self.net=M.PolicyLM(M.PolicyConfig(self.vocab.size,2051,32,1,4,64,128,dropout=.1));self.z=torch.randn(4,2051)
  texts=['(pipe X (ridge))','(pipe X (standardize) (ridge lambda=1))','(pipe X (rff_cos append=false gamma=0.7 n=16 seed=5) (ridge lambda=0.1))']
  self.programs=[{'dsl':s,'program_sha256':sha(s.encode())}for s in texts];self.trie=BatchedTrie(self.programs,self.vocab,128)
 def test_batch_one_matches_previous_serial_trace_exactly(self):
  old=ProposalTrie(self.programs,self.vocab,128)
  for seed in [1,7,19]:
   expected=old.decode(self.net,self.z[:1],seed=seed,temperature=1,deadline=time.time()+10);rows,meta=self.trie.decode_many(self.net,self.z[:1],seeds=[seed],temperature=1,deadline=time.time()+10)
   for k,v in expected.items():
    if k!='trie_sha256':self.assertEqual(rows[0][k],v)
  self.assertNotEqual(rows[0]['trie_sha256'],expected['trie_sha256']) # explicitly versioned compact catalog
 def test_deterministic_multirow_traces_and_native_program_membership(self):
  a,meta=self.trie.decode_many(self.net,self.z,seeds=[1,2,3,4],temperature=.7,deadline=time.time()+10);b,_=self.trie.decode_many(self.net,self.z,seeds=[1,2,3,4],temperature=.7,deadline=time.time()+10)
  self.assertEqual(a,b);self.assertEqual(len(a),4);self.assertLess(meta['forward_calls'],meta['tokens_generated'])
  for r in a:self.assertIn(r['program'],self.programs);self.assertEqual(T.detokenize(r['tokens'],self.vocab),r['program']['dsl'])
 def test_training_mode_and_global_rng_preserved(self):
  self.net.train();before=torch.get_rng_state().clone();self.trie.decode_many(self.net,self.z,seeds=[1,2,3,4],temperature=1,deadline=time.time()+10)
  self.assertTrue(self.net.training);self.assertTrue(torch.equal(before,torch.get_rng_state()))
 def test_seed_count_shape_temperature_and_deadline_refuse(self):
  for z,seeds,temp in [(self.z,[1],1),(self.z[:,:5],[1,2,3,4],1),(self.z,[1,2,3,4],0)]:
   with self.assertRaises(ValueError):self.trie.decode_many(self.net,z,seeds=seeds,temperature=temp,deadline=time.time()+10)
  with self.assertRaises(Exception):self.trie.decode_many(self.net,self.z,seeds=[1,2,3,4],temperature=1,deadline=time.time()-1)
if __name__=='__main__':unittest.main()
