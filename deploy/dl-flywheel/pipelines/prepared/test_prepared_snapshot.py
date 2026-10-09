import sys,json,unittest,base64,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent
R=HERE.parent/'continuous'
sys.path[:0]=[str(HERE),str(R/'runtime'),str(R/'consumer')]
import prepared_snapshot as P
from jope_flywheel import native_variant_snapshot as V,training_snapshot_native as N
class TestPrepared(unittest.TestCase):
 def setUp(self):
  self.dsl='GP';ph=P.sha(self.dsl.encode());self.native=SimpleNamespace(identity='a'*64,contexts={'native':{'context_id':'native','parent_lineage_sha256':'b'*64,'partition':'train'}},winners={},max_seq=640,program_window=8,vocab={},proposal_programs=[],legal_mask_fn=lambda tokens:None)
  self.e=P.canonical({'native_snapshot_sha256':self.native.identity,'preprocessing_sha256':'c'*64,'parent_registry_sha256':'d'*64})
  s=V.MixedSnapshot();s.native=self.native;s.identity=P.sha(self.e);s.contexts={**self.native.contexts,'variant':{'context_id':'variant','source_kind':'flywheel_variant','parent_lineage_sha256':'b'*64,'partition':'train'}};s.input_ready_contexts={'variant':s.contexts['variant']};s.variant_vectors={'variant':np.arange(2051,dtype='<f4')};r=N.NativeRecord(ph,(1,2,3),'completed',q=.5,quality_eligible=True,source_kind='flywheel_verified_s1');s.variant_records={'variant':(r,)};s.best_observed={'variant':r};s.proposal_programs=[{'program_sha256':ph,'tokens':[1,2,3],'dsl':self.dsl}];s.receipt={'snapshot_sha256':s.identity,'native_snapshot_sha256':self.native.identity,'preserved_preprocessing_sha256':'c'*64,'optimizer_updates':0,'admission_committed':False};s.batch_manifest={'snapshot_sha256':s.identity,'contexts':list(s.contexts.values())};self.s=s;self.raw=P.prepare(s,self.e);self.authority=[]
 def load(self,raw=None,pin=True,**kwargs):
  raw=self.raw if raw is None else raw;r={'schema':'jope.prepared-snapshot-receipt/v1','state':'passed','optimizer_updates':0,'planner_mutations':0,'s3_writes':0,'prepared':{'sha256':P.sha(raw if pin else self.raw),'bytes':len(raw if pin else self.raw)}}
  return P.load(raw,self.native,receipt=r,expected_extension_sha256=self.s.identity,expected_preprocessing_sha256=kwargs.get('preprocessing','c'*64),expected_parent_registry_sha256='d'*64,authority_check=lambda extension:self.authority.append(extension))
 def edit(self,fn):
  d=json.loads(self.raw);fn(d);return P.canonical(d)
 def test_roundtrip(self):
  s=self.load();self.assertEqual(s.contexts,self.s.contexts);self.assertEqual(s.variant_records,self.s.variant_records);self.assertEqual(s.best_observed,self.s.best_observed);np.testing.assert_array_equal(s.encoding('variant'),self.s.encoding('variant'));self.assertFalse(s.encoding('variant').flags.writeable);self.assertEqual(len(self.authority),2);self.assertEqual(s.extension_raw,self.e)
 def test_hash_custody(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d.update(schema='other')),pin=False)
 def test_lineage_leak(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d['variant_contexts']['variant'].update(partition='validation')))
 def test_normalizer_drift(self):
  with self.assertRaises(ValueError):self.load(preprocessing='f'*64)
 def test_variant_ce_refusal(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d['variant_records']['variant'][0].update(ce_eligible=True)))
 def test_nonfinite_vector(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d['vectors_f32le'].update(variant=base64.b64encode(np.full(2051,np.nan,dtype='<f4').tobytes()).decode())))
 def test_short_vector(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d['vectors_f32le'].update(variant='AAAAAA==')))
 def test_manifest_roster(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d['batch_manifest']['contexts'].pop()))
 def test_context_collision(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d['variant_contexts'].update(native=d['variant_contexts']['variant'])))
 def test_unknown_field(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d.update(unreviewed=True)))
 def test_extension_bytes(self):
  with self.assertRaises(ValueError):self.load(self.edit(lambda d:d.update(extension_utf8='{}')))
 def test_current_role_failure(self):
  def refuse(x):raise ValueError('new current-role exclusion')
  receipt={'schema':'jope.prepared-snapshot-receipt/v1','state':'passed','optimizer_updates':0,'planner_mutations':0,'s3_writes':0,'prepared':{'sha256':P.sha(self.raw),'bytes':len(self.raw)}}
  with self.assertRaises(ValueError):P.load(self.raw,self.native,receipt=receipt,expected_extension_sha256=self.s.identity,expected_preprocessing_sha256='c'*64,expected_parent_registry_sha256='d'*64,authority_check=refuse)
if __name__=='__main__':unittest.main()
