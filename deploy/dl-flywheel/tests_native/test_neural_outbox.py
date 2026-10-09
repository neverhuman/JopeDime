import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
from neural_outbox import Outbox,sha,digest

class OutboxTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name).resolve()/'outbox.sqlite';self.now=1000.
 def tearDown(self):self.tmp.cleanup()
 def box(self,**kw):return Outbox(self.path,model_origin_sha256='a'*64,now=lambda:self.now,**kw)
 def fixture(self,n=2,step=1):
  ids=[sha(('run%d'%i).encode())for i in range(n)]
  b={'batch_id':sha(('batch%d'%step).encode()),'campaign_sha256':'b'*64,'checkpoint_sha256':'c'*64,'global_step':step,'manifest_sha256':'d'*64,'train_data_run_ids':ids[:-1]or ids,'validation_data_run_ids':ids[-1:]if n>1 else[]}
  b['emitted_train_data_run_ids']=list(b['train_data_run_ids'])
  emissions=[]
  for i,r in enumerate(ids):
   raw='invalid ( DSL'if i==0 else'(pipe X (ridge))'
   emissions.append({'phase':'validation'if n>1 and i==n-1 else'training','data_run_id':r,'context_id':sha(('context%d'%i).encode()),'parent_lineage_sha256':sha(('parent%d'%i).encode()),'input_descriptor':{'schema':'native-context-test/v1','protected_rows':0},'raw_program_utf8':raw,'raw_program_sha256':sha(raw.encode()),'decoder_seed':i,'decoder_protocol_sha256':'e'*64,'decode_artifact_sha256':'f'*64})
  return b,emissions
 def success(self):return {'state':'cached_verified','evaluation_key':'1'*64,'record_ref':{'bucket':'b','key':'k','version_id':'v','sha256':'2'*64,'bytes':100},'attempt_custody_ref':None,'error':None}
 def test_invalid_dsl_and_both_phases_durable_before_any_validation(self):
  box=self.box();b,e=self.fixture();ack=box.record_batch(b,e);self.assertEqual(len(ack['emission_ids']),2);self.assertEqual(box.verify()['observations'],2)
  self.assertEqual(json.loads(box.db.execute('SELECT body FROM emissions ORDER BY ordinal').fetchone()[0])['emission']['raw_program_utf8'],'invalid ( DSL');box.close()
 def test_capacity_rolls_back_entire_batch_and_no_ack(self):
  box=self.box(max_pending=1);b,e=self.fixture()
  with self.assertRaisesRegex(ValueError,'backpressure'):box.record_batch(b,e)
  self.assertEqual(box.verify()['events'],0);self.assertEqual(box.summary()['batches'],0);box.close()
 def test_injected_second_insert_failure_rolls_back_everything(self):
  box=self.box();box.db.execute("CREATE TRIGGER injected BEFORE INSERT ON emissions WHEN NEW.ordinal=1 BEGIN SELECT RAISE(ABORT,'injected');END")
  with self.assertRaises(sqlite3.IntegrityError):box.record_batch(*self.fixture())
  self.assertEqual(box.verify()['observations'],0);self.assertEqual(box.summary()['batches'],0);box.close()
 def test_reopen_idempotence_and_conflicting_payload(self):
  box=self.box(max_pending=2);b,e=self.fixture();ack=box.record_batch(b,e);box.close();box=self.box(max_pending=2);self.assertEqual(box.record_batch(b,e),ack)
  e[0]['raw_program_utf8']='another invalid';e[0]['raw_program_sha256']=sha(e[0]['raw_program_utf8'].encode())
  with self.assertRaisesRegex(ValueError,'conflicting'):box.record_batch(b,e)
  self.assertEqual(box.verify()['events'],1);box.close()
 def test_unique_context_lineage_phase_constraints(self):
  for mutation in ['phase','lineage','missing']:
   box=self.box();b,e=self.fixture(3)
   if mutation=='phase':e[0]['phase']='validation'
   elif mutation=='lineage':e[1]['parent_lineage_sha256']=e[0]['parent_lineage_sha256']
   else:e.pop()
   with self.assertRaises(ValueError):box.record_batch(b,e)
   self.assertEqual(box.verify()['events'],0);box.close()
 def test_claims_are_disjoint_and_expired_never_replayed(self):
  box=self.box();box.record_batch(*self.fixture());second=self.box();a=box.claim('a',lease_seconds=10,deadline=1100);b=second.claim('b',lease_seconds=10,deadline=1100);self.assertNotEqual(a['lease']['emission_id'],b['lease']['emission_id']);self.now=1011
  self.assertIsNone(box.claim('c',lease_seconds=10,deadline=1100))
  with self.assertRaisesRegex(ValueError,'live owned lease'):box.finish(a['lease'],self.success(),verify_outcome=lambda *_:True)
  self.assertEqual(box.verify()['states'],{'leased':2});second.close();box.close()
 def test_verifier_failure_does_not_accept_ready_or_delete_observation(self):
  box=self.box();box.record_batch(*self.fixture());lease=box.claim('a',lease_seconds=10,deadline=1100)['lease']
  with self.assertRaisesRegex(ValueError,'verifier'):box.finish(lease,self.success(),verify_outcome=lambda *_:False)
  self.assertEqual(box.verify()['unresolved'],2);box.close()
 def test_verified_cache_and_nonnumeric_failure_custody_are_separate(self):
  box=self.box();box.record_batch(*self.fixture());a=box.claim('a',lease_seconds=10,deadline=1100);box.finish(a['lease'],self.success(),verify_outcome=lambda *_:True)
  b=box.claim('b',lease_seconds=10,deadline=1100);failure={'state':'failure_custody_verified','evaluation_key':None,'record_ref':None,'attempt_custody_ref':self.success()['record_ref'],'error':'invalid DSL fixture, not production'};box.finish(b['lease'],failure,verify_outcome=lambda *_:True)
  state=box.verify();self.assertEqual(state['unresolved'],0);self.assertEqual(state['states'],{'cached_verified':1,'failure_custody_verified':1});self.assertEqual(state['numeric_scores_synthesized'],0);box.close()
 def test_failure_cannot_carry_score_or_ready_reference(self):
  box=self.box();box.record_batch(*self.fixture());lease=box.claim('a',lease_seconds=10,deadline=1100)['lease'];bad=self.success();bad.update(state='failure_custody_verified',error='timeout')
  with self.assertRaises(ValueError):box.finish(lease,bad,verify_outcome=lambda *_:True)
  bad=self.success();bad['rmse']=0
  with self.assertRaises(ValueError):box.finish(lease,bad,verify_outcome=lambda *_:True)
  self.assertEqual(box.verify()['unresolved'],2);box.close()
 def test_unsupported_route_remains_visible_and_blocks_backlog_capacity(self):
  box=self.box(max_pending=2);box.record_batch(*self.fixture());a=box.claim('a',lease_seconds=10,deadline=1100)
  box.finish(a['lease'],{'state':'pending_unsupported','evaluation_key':None,'record_ref':None,'attempt_custody_ref':None,'error':'native-context adapter not qualified'},verify_outcome=lambda *_:True)
  with self.assertRaisesRegex(ValueError,'backpressure'):box.record_batch(*self.fixture(step=2))
  self.assertEqual(box.verify()['unresolved'],2);box.close()
 def test_tampering_chain_payload_ack_and_outcome_detected(self):
  for table,column,condition in [('events','sha','seq=1'),('emissions','body','ordinal=0'),('batches','ack','1=1')]:
   path=self.path.with_name(table+'.sqlite');box=Outbox(path,model_origin_sha256='a'*64);box.record_batch(*self.fixture());box.db.execute('UPDATE '+table+' SET '+column+'=?WHERE '+condition,(b'{}',))
   with self.assertRaises((ValueError,KeyError)):box.verify()
   box.close()
 def test_symlink_and_config_drift_refused(self):
  box=self.box();box.close()
  with self.assertRaisesRegex(ValueError,'immutable'):self.box(max_pending=1)
  link=self.path.with_name('link');link.symlink_to(self.path)
  with self.assertRaisesRegex(ValueError,'symlink'):Outbox(link,model_origin_sha256='a'*64)
if __name__=='__main__':unittest.main()
