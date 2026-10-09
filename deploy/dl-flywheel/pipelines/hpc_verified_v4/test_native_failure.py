import copy,io,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from neural_outbox import Outbox,canonical,digest,sha
from native_failure import ParityRejected,parity_diagnostic,anchored_emission,CustodyStore,publication,verify_publication,finish_failure,PREFIX
from jope_flywheel import program_io

class Conditional(Exception):
 response={'Error':{'Code':'PreconditionFailed'}}
class FakeS3:
 def __init__(self):self.objects={};self.puts=0;self.fail=False;self.corrupt=False;self.auth=False
 def put_object(self,**kw):
  self.puts+=1
  if self.fail:raise OSError('interrupted unknown PUT outcome')
  assert kw['IfNoneMatch']=='*'
  if kw['Key']in self.objects:raise Conditional()
  self.objects[kw['Key']]=(bytes(kw['Body']),'v'+str(self.puts),kw['Metadata']);return {'VersionId':'v'+str(self.puts)}
 def get_object(self,**kw):
  if self.auth:raise PermissionError('AccessDenied')
  b,v,m=self.objects[kw['Key']];assert kw['VersionId']==v
  return {'Body':io.BytesIO(b+b'x'if self.corrupt else b),'VersionId':v,'ContentLength':len(b)}
 def head_object(self,**kw):
  b,v,m=self.objects[kw['Key']];return {'VersionId':v,'ContentLength':len(b),'Metadata':m}

class FailureTests(unittest.TestCase):
 def setUp(self):
  program_io.CAMPAIGN_MONOTONIC_END=None
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
  self.o=Outbox(self.root/'outbox.sqlite',model_origin_sha256='a'*64,max_pending=32);self.addCleanup(self.o.close)
  self.client=FakeS3();self.store=CustodyStore(self.client,time.time()+60)
  self.config={'outbox_path':str(self.root/'outbox.sqlite'),'outbox_config':self.o.config,'training_config':{'sha256':'b'*64},'source_pins':[],'native_build_receipt':{},'worker':{},'emitter':{},'rustc':{}}
 def emit(self,index=0,kind='native_s1_context'):
  cid=digest({'context':index});bid=digest({'batch':index});dsl='(pipe X (ridge_gcv))';lin=digest({'lineage':index})
  b={'batch_id':bid,'campaign_sha256':'b'*64,'checkpoint_sha256':'c'*64,'global_step':index+1,'manifest_sha256':'d'*64,'train_data_run_ids':[cid],'validation_data_run_ids':[],'emitted_train_data_run_ids':[cid]}
  e={'phase':'training','data_run_id':cid,'context_id':cid,'parent_lineage_sha256':lin,'input_descriptor':{'source_kind':kind,'context_id':cid,'context_sha256':cid,'parent_lineage_sha256':lin},'raw_program_utf8':dsl,'raw_program_sha256':sha(dsl.encode()),'decoder_seed':42,'decoder_protocol_sha256':'e'*64,'decode_artifact_sha256':'f'*64}
  return self.o.record_batch(b,[e])['emission_ids'][0]
 def diagnostic(self,claim):
  d=parity_diagnostic([1.01]+[1.]*24,[1.]*25);d.update(program_sha256=claim['body']['emission']['raw_program_sha256'],context={'identity':claim['body']['emission']['context_id']});return d
 def claim(self):return self.o.claim('fixture',lease_seconds=30,deadline=self.o.now()+60)
 def snap(self,id):return anchored_emission(self.config['outbox_path'],id,self.o.config,'b'*64)
 def test_fixed_tolerance_diagnostics_exact(self):
  d=parity_diagnostic([1.01]+[1.]*24,[1.]*25);self.assertEqual(d['failed_rows'],[0]);self.assertEqual(d['bit_exact_rows'],24);self.assertEqual(d['tolerance'],1e-9)
  for a,b in [([1.]*25,[1.]*25),([float('nan')]*25,[1.]*25),([1.]*24,[2.]*24)]:
   with self.assertRaises(ValueError):parity_diagnostic(a,b)
 def test_successful_custody_no_score_exact_versions(self):
  id=self.emit();claim=self.claim();snap=self.snap(id);ref=publication(snap,self.diagnostic(claim),'parity failed',{},self.store,self.root/'archive')
  receipt=verify_publication(self.store,ref,claim['body'],id);self.assertIsNone(receipt['numeric_score']);self.assertEqual(receipt['canonical_score_rows'],[]);self.assertEqual(self.client.puts,2)
  self.assertEqual(self.snap(id)['state'],'leased')
 def test_known_parity_is_terminal_then_next_candidate_claimable(self):
  self.emit();self.emit(1);c=self.claim();row,keep=finish_failure(self.o,c['lease'],c['body'],ParityRejected(self.diagnostic(c)),self.config,'1'*64,self.store,self.root/'archive')
  self.assertTrue(keep);self.assertEqual(row['state'],'failure_custody_verified');self.assertEqual(self.o.verify()['states']['failure_custody_verified'],1);self.assertIsNotNone(self.claim())
 def test_generic_integrity_error_archived_but_pauses(self):
  self.emit();c=self.claim();row,keep=finish_failure(self.o,c['lease'],c['body'],ValueError('source SHA changed'),self.config,'1'*64,self.store,self.root/'archive')
  self.assertFalse(keep);self.assertEqual(row['state'],'unknown_attempt');self.assertEqual(self.o.verify()['states']['unknown_attempt'],1)
 def test_auth_error_no_false_terminal(self):
  self.emit();c=self.claim();self.client.auth=True
  row,keep=finish_failure(self.o,c['lease'],c['body'],ParityRejected(self.diagnostic(c)),self.config,'1'*64,self.store,self.root/'archive')
  self.assertFalse(keep);self.assertEqual(self.o.verify()['states']['unknown_attempt'],1);self.assertIn('custody_error',row)
 def test_interrupted_put_no_retry_and_no_score(self):
  self.emit();c=self.claim();self.client.fail=True
  row,keep=finish_failure(self.o,c['lease'],c['body'],ParityRejected(self.diagnostic(c)),self.config,'1'*64,self.store,self.root/'archive')
  self.assertFalse(keep);self.assertEqual(self.client.puts,1);self.assertEqual(self.snap(c['lease']['emission_id'])['state'],'unknown_attempt')
 def test_expired_lease_preserved_no_automatic_retry(self):
  self.emit();c=self.claim();self.o.now=lambda:c['lease']['expires_at']+1
  row,keep=finish_failure(self.o,c['lease'],c['body'],ParityRejected(self.diagnostic(c)),self.config,'1'*64,self.store,self.root/'archive')
  self.assertFalse(keep);self.assertIn('finish_error',row);self.assertEqual(self.o.verify()['states']['leased'],1);self.assertIsNone(self.claim())
 def test_historical_unknown_remains_unchanged(self):
  id=self.emit();c=self.claim();self.o.finish(c['lease'],{'state':'unknown_attempt','evaluation_key':None,'record_ref':None,'attempt_custody_ref':None,'error':'old parity rejection'},verify_outcome=lambda *a:True)
  before=self.o.verify();snap=self.snap(id);publication(snap,self.diagnostic(c),'old failure',{},self.store,self.root/'archive')
  self.assertEqual(before,self.o.verify());self.assertIsNone(self.claim())
 def test_diagnostic_tamper_and_wrong_emission_refused(self):
  id=self.emit();c=self.claim();d=self.diagnostic(c);d['tolerance']=1.
  with self.assertRaises(ValueError):publication(self.snap(id),d,'error',{},self.store,self.root/'bad')
  d=self.diagnostic(c);d['context']['identity']='0'*64
  with self.assertRaises(ValueError):publication(self.snap(id),d,'error',{},self.store,self.root/'bad2')
  self.assertEqual(self.client.puts,0)
 def test_corrupt_get_fails(self):
  self.client.corrupt=True
  with self.assertRaises(ValueError):self.store.put(PREFIX+'test',b'abc')
 def test_create_only_collision_requires_exact_bytes(self):
  ref=self.store.put(PREFIX+'test',b'abc');self.assertEqual(ref,self.store.put(PREFIX+'test',b'abc'))
  with self.assertRaises(ValueError):self.store.put(PREFIX+'test',b'def')
 def test_role_phase_and_origin_bound_in_snapshot(self):
  id=self.emit()
  with self.assertRaises(ValueError):anchored_emission(self.config['outbox_path'],id,self.o.config,'0'*64)
  self.o.db.execute("UPDATE emissions SET body=?WHERE id=?",(canonical({'bad':True}),id))
  with self.assertRaises(ValueError):self.snap(id)
 def test_deadline_before_any_put(self):
  expired=CustodyStore(self.client,time.time()-1)
  with self.assertRaises(TimeoutError):expired.put(PREFIX+'test',b'abc')
  self.assertEqual(self.client.puts,0)
 def test_namespace_escape_refused(self):
  with self.assertRaises(ValueError):self.store.put('foreign/x',b'abc')
  self.assertEqual(self.client.puts,0)
 def test_native_claim_skips_variant_without_mutating_it(self):
  from native_dispatch import claim_native,unsupported_count
  variant=self.emit(kind='flywheel_variant');native=self.emit(1)
  c=claim_native(self.o,'native',lease_seconds=30,deadline=time.time()+60)
  self.assertEqual(c['lease']['emission_id'],native);self.assertEqual(self.snap(variant)['state'],'recorded')
  self.assertEqual(unsupported_count(self.config['outbox_path']),1);self.o.verify()
 def test_two_outboxes_own_independent_atomic_claims(self):
  from native_dispatch import claim_native
  self.emit();second=Outbox(self.root/'second.sqlite',model_origin_sha256='a'*64);self.addCleanup(second.close)
  original=self.o;self.o=second;self.emit(1);self.o=original
  a=claim_native(original,'first',lease_seconds=30,deadline=time.time()+60);b=claim_native(second,'second',lease_seconds=30,deadline=time.time()+60)
  self.assertNotEqual(a['lease']['emission_id'],b['lease']['emission_id']);original.verify();second.verify()
if __name__=='__main__':unittest.main()
