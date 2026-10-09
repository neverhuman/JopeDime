"""Portable full seven-artifact fixtures; no live credentials or private paths."""
import copy,importlib,json,time,types,unittest
from pathlib import Path
import numpy as np
import test_training_snapshot_kumos as fixtures
from jope_flywheel import native_variant_snapshot as V,training_snapshot_native as N
from jope_flywheel.protocol import canonical,digest
from dltrain import automaton,labels

class VariantTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixtures.SnapshotTests.setUpClass()
 def fixture(self):
  f=fixtures.SnapshotTests();f.setUp();task=f.task(programs=2,program_texts=['(pipe X (ridge alpha=1.0))','(pipe X (ridge alpha=2.0))']);registry,_=f.freeze();task['parent_lineage_sha256']=f.parents[0]['parent_lineage_sha256']
  pub=f.store.doc(task['encoder_publication']);proof=pub['source_proof'];proof['source_release']={'out':'/fixture/producer','source_sha256':proof['source_sha256']};entry=pub['encoder_records']['kumo_tabular_s'];rec=entry['record'];rec['phase']='ofm';rec['output']['path']='variants/'+task['variant_identity']+'/kumo_tabular_s.npy';entry['vector_ref']={**rec['output'],'path':'/fixture/producer/ofm/'+rec['output']['path']};entry['record_ref'].update(sha256=V.sha(canonical(rec)),bytes=len(canonical(rec)));task['encoder_publication']=f.store.replace(task['encoder_publication'],pub)
  ns={'parent_registry':{'sha256':V.sha(registry)},'authority_pins':{'fixture':True}};nsraw=canonical(ns);n=N.NativeSnapshot();n.identity=V.sha(nsraw);n.max_seq=640;n.program_window=8;n.vocab=f.vocab;n.tokenizer=f.T;n.labels=labels;n._mask_cache={};sig=automaton.build_signatures([]);n.automaton=automaton;n.grammar=automaton.Grammar(automaton.build_grammar(sig,n.vocab,640),sig,n.vocab)
  cid='a'*64;descriptor={'context_id':cid,'parent_lineage_sha256':task['parent_lineage_sha256'],'partition':'train','source_kind':'native_s1_context'};n.contexts={cid:descriptor};n.winners={};n.proposal_programs=[];n.receipt={'pending':[{'context_id':'b'*64,'reason':'native_pending'}],'refused':[]};n.encoding=lambda c:np.zeros(2051,np.float32)
  n.context_loader=types.SimpleNamespace(get=lambda ids,**kw:N.NativeBatch({c:n.contexts[c]for c in ids},{c:n.encoding(c)for c in ids},{c:()for c in ids},{c:{}for c in ids},{'fixture':True}))
  pre=canonical({'schema':'jope.native-kumos-preprocessing/v1','snapshot_sha256':n.identity,'fit_context_ids':[cid],'validation_used_for_fit':False,'mean':[0.]*2051,'scale':[1.]*2051,'clip':[-10.,10.]})
  release={'publisher_sha256':pub['publisher_sha256'],'source_sha256':proof['source_sha256'],'manifest_sha256':proof['manifest']['sha256'],'source_proof_sha256':digest(proof),'custody':'original_sealed'}
  spec={'schema':V.SCHEMA,'native_snapshot_sha256':n.identity,'parent_registry_sha256':V.sha(registry),'generator_config_sha256':f.spec['generator_config_sha256'],'authority_helper_sha256':'c'*64,'dependency_sha256':{m:V.sha(Path(importlib.import_module(m).__file__).read_bytes())for m in V.MODULES},'dataset_publishers':[f.spec['dataset_publisher_sha256']],'encoder_releases':[release],'tasks':[task],'max_read_bytes':32*1024**2,'preprocessing_sha256':V.sha(pre),'teacher_rule':'disabled_for_unpromoted_variants/v1'}
  auth=lambda:canonical({'schema':'jope.native-corpus-current-authority/v1','passed':True,'helper_sha256':'c'*64,'source_pins':ns['authority_pins'],'checked_at_unix':time.time(),'protected_context_ids':[],'protected_views':[]})
  return f,n,spec,nsraw,registry,pre,auth
 def load(self,pack):
  f,n,spec,ns,reg,pre,auth=pack;b=canonical(spec);return V.compose(n,b,V.sha(b),f.store,native_spec_bytes=ns,registry_bytes=reg,preprocessing_bytes=pre,authority_callback=auth,deadline=time.time()+20)
 def test_real_gate_baseline_labels_and_same_native_membership(self):
  p=self.fixture();s=self.load(p);self.assertEqual(s.contexts['a'*64],p[1].contexts['a'*64]);self.assertEqual(s.receipt['counts']['variants_loaded'],1);self.assertFalse(s.receipt['normalizer_refit']);cid=next(iter(s.variant_records));r=s.context_loader.get([cid],epoch=0,batch_index=0);self.assertEqual(len(r.records[cid]),2);self.assertEqual(sum(x.ce_eligible for x in r.records[cid]),0);self.assertTrue(all(x.quality_eligible for x in r.records[cid]));self.assertEqual(s.best_observed[cid].program_sha256,r.records[cid][min(range(2),key=lambda i:-r.records[cid][i].q)].program_sha256);self.assertEqual(s.contexts[cid]['source_kind'],'flywheel_variant');self.assertNotIn('dataset_hash',s.contexts[cid]);self.assertEqual(s.receipt['pending'],p[1].receipt['pending'])
 def test_missing_programs_stay_pending(self):
  p=self.fixture();p[2]['tasks'][0]['program_results']=[];s=self.load(p);self.assertEqual(s.receipt['counts']['variants_loaded'],0);self.assertEqual(len(s.receipt['pending']),2);self.assertEqual(s.receipt['counts']['variants_input_ready'],1);cid=next(iter(s.input_ready_contexts));self.assertNotIn(cid,s.contexts);self.assertEqual(len(s.encoding(cid)),2051)
 def test_wrong_s3_version_refused(self):
  p=self.fixture();p[0].store.wrong_version=True
  with self.assertRaisesRegex(ValueError,'version'):self.load(p)
 def test_read_budget_fail_closed(self):
  p=self.fixture();p[2]['max_read_bytes']=1
  with self.assertRaisesRegex(ValueError,'read budget'):self.load(p)
 def test_role_flip_refused(self):
  p=self.fixture();p[1].contexts['a'*64]['partition']='validation'
  with self.assertRaisesRegex(ValueError,'normalization identity|parent lineage/role'):self.load(p)
 def test_normalization_bytes_refused(self):
  p=list(self.fixture());p[5]+=b' '
  with self.assertRaisesRegex(ValueError,'normalizer exact'):self.load(p)
 def test_unknown_parent_refused(self):
  p=self.fixture();p[2]['tasks'][0]['parent_lineage_sha256']='d'*64
  with self.assertRaisesRegex(ValueError,'parent lineage'):self.load(p)
 def test_unreviewed_encoder_proof_refused(self):
  p=self.fixture();p[2]['encoder_releases'][0]['source_proof_sha256']='e'*64
  with self.assertRaisesRegex(ValueError,'reviewed encoder'):self.load(p)
 def test_changed_target_evaluation_refused(self):
  p=self.fixture();t=p[2]['tasks'][0];entry=p[0].store.doc(t['program_results'][0]);entry['context']['target_sha256']='f'*64;t['program_results'][0]=p[0].store.replace(t['program_results'][0],entry)
  with self.assertRaisesRegex(ValueError,'context/evaluator/target'):self.load(p)
 def test_stale_current_role_proof_refused(self):
  p=list(self.fixture());old=p[6];p[6]=lambda:canonical({**json.loads(old()),'checked_at_unix':0})
  with self.assertRaisesRegex(ValueError,'fresh role'):self.load(p)
 def test_duplicate_variant_refused(self):
  p=self.fixture();p[2]['tasks']*=2
  with self.assertRaisesRegex(ValueError,'duplicate variant'):self.load(p)
 def test_deadline_refuses_before_read(self):
  p=self.fixture();f,n,sp,ns,reg,pre,au=p;b=canonical(sp)
  with self.assertRaisesRegex(ValueError,'deadline'):V.compose(n,b,V.sha(b),f.store,native_spec_bytes=ns,registry_bytes=reg,preprocessing_bytes=pre,authority_callback=au,deadline=0)
 def test_output_traversal_even_with_repin_refused(self):
  p=self.fixture();f=p[0];t=p[2]['tasks'][0];pub=f.store.doc(t['encoder_publication']);e=pub['encoder_records']['kumo_tabular_s'];e['record']['output']['path']='../escape';e['record_ref'].update(sha256=V.sha(canonical(e['record'])),bytes=len(canonical(e['record'])));t['encoder_publication']=f.store.replace(t['encoder_publication'],pub)
  with self.assertRaisesRegex(ValueError,'phase relative'):self.load(p)
 def recovery(self):
  p=self.fixture();f=p[0];t=p[2]['tasks'][0];pub=f.store.doc(t['encoder_publication']);pub['schema']='jope.flywheel-evergreen-kumos-publication/v1';proof=pub['source_proof'];proof['producer_phase_sealed']=False;proof['original_coordinator_state']='failed';proof['phase_ledgers']['ofm']['runs']['fixture']['sealed']=False;body={'state':'complete','recomputed_vectors':0,'teaching_eligible':False};proof['finalization_custody']={'original_producer_sealed':False,'status':{'body':body,'ref':{'sha256':V.sha(canonical(body)),'bytes':len(canonical(body))}}};p[2]['encoder_releases'][0].update(source_proof_sha256=digest(proof),custody='reviewed_finalization');t['encoder_publication']=f.store.replace(t['encoder_publication'],pub);return p
 def test_recovery_retains_original_unsealed_failure(self):
  s=self.load(self.recovery());self.assertEqual(next(iter(s.receipt['variants'].values()))['vector']['custody'],'reviewed_finalization')
 def test_recovery_cannot_claim_original_seal(self):
  p=self.recovery();p[2]['encoder_releases'][0]['custody']='original_sealed'
  with self.assertRaisesRegex(ValueError,'seal required'):self.load(p)
 def test_duplicate_program_refused(self):
  p=self.fixture();p[2]['tasks'][0]['program_results']*=2
  with self.assertRaisesRegex(ValueError,'duplicate measured'):self.load(p)
 def test_repeated_exact_reads_are_cached(self):
  p=self.fixture();f=p[0];ref=p[2]['tasks'][0]['dataset_publication'];r=V.Reader(f.store,ref['bytes'],time.time()+10);a=r.read(ref);b=r.read(ref);self.assertEqual(a,b);self.assertEqual(r.read_bytes,ref['bytes']);self.assertEqual(f.store.calls,1)
if __name__=='__main__':unittest.main()
