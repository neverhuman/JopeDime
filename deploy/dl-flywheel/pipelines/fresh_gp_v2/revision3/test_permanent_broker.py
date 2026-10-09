"""Revision3 new permanent-state refusals, real SQLite/files; no GP or network."""
import json,shutil,sqlite3,sys,tempfile,time,unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent
sys.path[:0]=[str(H),str(H/'runtime')]
from common import ref,put,obj,canonical,digest,GLOBAL
import permanent_broker as pb
import broker_history as bh
from global_attempts import Attempts
from jope_flywheel.protocol import Context
from jope_flywheel.queue import Queue
from jope_flywheel.program_gate import request_identity
NOW=1791558000.0
class Native:
 def canonical_program(self,s):return s
class Cases(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name).resolve();self.addCleanup(self.t.cleanup)
  c=patch('time.time',return_value=NOW);c.start();self.addCleanup(c.stop)
  src=put(self.p,'sources.json',{'schema':'jope.fresh128-source/v2','files':[ref(H/'permanent_broker.py')]})
  self.request=put(self.p,'request.json',{'schema':'jope.fresh128-broker-bootstrap-request/v3','create_once_authorized':True,'broker_root':str(pb.ROOT),'source_manifest':src})
  self.root=self.p/'broker';self.boot=pb.bootstrap(self.request,hostname='xbabe2',fixture_root=self.root);self.gen=obj(self.boot)['genesis']
 def reg(self,head=None):
  r=pb.Registry(self.boot,head or self.gen,fixture_root=self.root);self.addCleanup(r.close);return r
 def epoch(self,base=1):
  root=self.p/('epoch'+str(base));root.mkdir();contexts=[];self.actual={}
  for i in range(128):
   c=Context('regression',f'{base+i:064x}','2'*64,'cut'+str(i),'3'*64,'4'*64,'5'*64,'6'*64,'rmse','prepared_synthetic_target_units','minimize')
   v=f'{base+i+10000:064x}';self.actual[c.identity]=c
   d={'context_id':c.identity,'variant_identity':v,'descriptor_sha256':'d'*64,'partition':'validation'if i<10 else'train'};contexts.append(d)
  shards=[{'id':'shard%02d'%i,'unit':'old-%d.service'%i,'owned_root':str(root/('shard%02d'%i)),
           'context_ids':[d['context_id']for d in contexts[8*i:8*i+8]],'variant_ids':[d['variant_identity']for d in contexts[8*i:8*i+8]]}for i in range(16)]
  return {'epoch_root':str(root),'context_descriptors':contexts,'shards':shards,'context_owners':{v:i['id']for i in shards for v in i['context_ids']}}
 def reserve(self,r):
  e=self.epoch();r.reserve(e,digest(e),r.verify());return e
 def test_bootstrap_once_refuses_even_when_database_removed(self):
  (self.root/'registry.sqlite').unlink()
  with self.assertRaises(FileExistsError):pb.bootstrap(self.request,hostname='xbabe2',fixture_root=self.root)
 def test_missing_database_after_reservation_refuses_and_never_recreates(self):
  r=self.reg();self.reserve(r);(self.root/'registry.sqlite').unlink()
  with self.assertRaises(FileNotFoundError):r.verify()
  self.assertFalse((self.root/'registry.sqlite').exists())
 def test_existing_only_open_refuses_missing_database(self):
  r=self.reg();self.reserve(r);head=r.head_anchor;r.close();(self.root/'registry.sqlite').unlink()
  with self.assertRaises(FileNotFoundError):pb.Registry(self.boot,head,fixture_root=self.root)
  self.assertFalse((self.root/'registry.sqlite').exists())
 def test_replaced_database_identical_bytes_refused_before_proof(self):
  r=self.reg();self.reserve(r);p=self.root/'registry.sqlite';copy=self.root/'copy';shutil.copyfile(p,copy);p.rename(self.root/'old.sqlite');copy.rename(p)
  with self.assertRaises(ValueError):r.verify()
 def test_replaced_lock_refused_before_proof(self):
  r=self.reg();self.reserve(r);p=self.root/'epoch.lock';p.rename(self.root/'old.lock');p.write_bytes(b'')
  with self.assertRaises(ValueError):r.verify()
 def test_changed_materialized_row_refused(self):
  r=self.reg();self.reserve(r);r.db.execute("UPDATE contexts SET partition='train'")
  with self.assertRaises(ValueError):r.verify()
 def test_changed_epoch_body_refused(self):
  r=self.reg();self.reserve(r);r.db.execute("UPDATE epochs SET body='{}'")
  with self.assertRaises(ValueError):r.verify()
 def test_inplace_rewind_refused_by_external_exact_head(self):
  r=self.reg();original=list(r.db.iterdump());self.reserve(r)
  external=put(self.p,'external-head.json',obj(r.head_anchor));r.head_anchor=external
  for table in('events','contexts','epochs','config'):r.db.execute('DROP TABLE '+table)
  r.db.executescript('\n'.join(original))
  with self.assertRaises(ValueError):r.verify()
 def test_deleted_commit_custody_refused(self):
  r=self.reg();self.reserve(r);Path(r.head_anchor['path']).unlink()
  with self.assertRaises(FileNotFoundError):r.verify()
 def test_unanchored_transaction_is_retained_and_blocks_reopen(self):
  r=self.reg();e=self.epoch();head=r.verify()
  with patch.object(pb,'anchor_event',side_effect=OSError('fixture interrupted custody')):
   with self.assertRaises(OSError):r.reserve(e,digest(e),head)
  self.assertEqual(r.db.execute('SELECT count(*)FROM contexts').fetchone()[0],128)
  with self.assertRaises(FileNotFoundError):r.verify()
 def reanchor_mutated_event(self,r,body):
  row=r.db.execute('SELECT *FROM events ORDER BY seq DESC LIMIT 1').fetchone()
  head=digest({'seq':row['seq'],'body':body,'previous_sha256':row['prev_sha']})
  Path(r.head_anchor['path']).unlink()
  r.db.execute('UPDATE events SET body=?,sha=? WHERE seq=?',(canonical(body).decode(),head,row['seq']))
  r.head_anchor=pb.anchor_event(self.root,{'seq':row['seq'],'body':body,'previous_sha256':row['prev_sha'],'sha256':head})
 def test_self_consistent_materialized_rewrite_cannot_change_event_ownership(self):
  r=self.reg();self.reserve(r);r.db.execute("UPDATE contexts SET partition='train'")
  body=json.loads(r.db.execute('SELECT body FROM events ORDER BY seq DESC LIMIT 1').fetchone()[0])
  body['state_sha256']=digest(pb.state(r.db));self.reanchor_mutated_event(r,body)
  with self.assertRaisesRegex(ValueError,'reconstructed'):r.verify()
 def test_self_consistent_started_rewrite_cannot_change_event_replay(self):
  r=self.reg();self.reserve(r);r.db.execute('UPDATE epochs SET started=started+1')
  body=json.loads(r.db.execute('SELECT body FROM events ORDER BY seq DESC LIMIT 1').fetchone()[0])
  body['state_sha256']=digest(pb.state(r.db));self.reanchor_mutated_event(r,body)
  with self.assertRaisesRegex(ValueError,'reconstructed'):r.verify()
 def test_unknown_event_kind_refused_even_with_self_consistent_chain(self):
  r=self.reg();self.reserve(r);body=json.loads(r.db.execute('SELECT body FROM events ORDER BY seq DESC LIMIT 1').fetchone()[0])
  body['kind']='manual_reset';self.reanchor_mutated_event(r,body)
  with self.assertRaisesRegex(ValueError,'known permanent'):r.verify()
 def test_inplace_rewind_and_deleted_local_commits_still_refused_by_external_head(self):
  r=self.reg();original=list(r.db.iterdump());self.reserve(r)
  external=put(self.p,'sealed-external-head.json',obj(r.head_anchor));Path(r.head_anchor['path']).unlink();r.head_anchor=external
  for table in('events','contexts','epochs','config'):r.db.execute('DROP TABLE '+table)
  r.db.executescript('\n'.join(original))
  with self.assertRaisesRegex(ValueError,'external exact prior head'):r.verify()
 def test_new_epoch_duplicate_context_rolls_back_without_reassignment(self):
  r=self.reg();e=self.reserve(r);before=r.verify();second={**e,'different_attempt':True}
  with self.assertRaises(sqlite3.IntegrityError):r.reserve(second,digest(second),before)
  self.assertEqual(r.verify(),before)
 def test_wrong_journal_or_schema_configuration_refused(self):
  r=self.reg();r.db.execute('PRAGMA user_version=4')
  with self.assertRaises(ValueError):r.verify()
 def test_cold_inventory_requires_exact_retained_head(self):
  r=self.reg();self.reserve(r);current=put(self.p,'external-current.json',obj(r.head_anchor));r.close()
  req=put(self.p,'inventory-stale.json',{'schema':'jope.permanent-broker-inventory-request/v3','bootstrap':self.boot,'known_head':put(self.p,'external-stale.json',obj(self.gen))})
  with self.assertRaises(ValueError):bh.inventory(req,self.p/'bad.json',hostname='xbabe2',fixture_root=self.root)
  req=put(self.p,'inventory-current.json',{'schema':'jope.permanent-broker-inventory-request/v3','bootstrap':self.boot,'known_head':current})
  result=bh.inventory(req,self.p/'inventory.json',hostname='xbabe2',fixture_root=self.root)
  self.assertEqual(len(obj(result)['contexts']),128)
 def test_cold_inventory_rejects_local_commit_as_external_authority(self):
  r=self.reg();self.reserve(r);current=r.head_anchor;r.close()
  req=put(self.p,'local-inventory.json',{'schema':'jope.permanent-broker-inventory-request/v3','bootstrap':self.boot,'known_head':current})
  with self.assertRaisesRegex(ValueError,'externally retained'):bh.inventory(req,self.p/'not-created.json',hostname='xbabe2',fixture_root=self.root)
  self.assertFalse((self.p/'not-created.json').exists())
 def worker_fixture(self,owned):
  r=self.reg();e=self.reserve(r);head=put(self.p,'external-worker-head.json',obj(r.head_anchor));r.close()
  req=put(self.p,'inventory-request.json',{'schema':'jope.permanent-broker-inventory-request/v3','bootstrap':self.boot,'known_head':head})
  inv=bh.inventory(req,self.p/'inventory.json',hostname='xbabe2',fixture_root=self.root)
  config={'campaign_sha256':digest(e),'selection_sha256':'f'*64,'ownership_producer_sha256':'e'*64,'history_sources':[],
    'work_deadline_unix':NOW+600,'global_deadline_unix':GLOBAL,'shard_ids':[s['id']for s in e['shards']],
    'context_owners':e['context_owners'],'context_descriptors':e['context_descriptors'],'broker_bootstrap':self.boot,'broker_history':{'sha256':'b'*64}}
  proof={'schema':'jope.fresh-gp-context-ownership/v1','passed':True,'host':'xbabe3','campaign_sha256':digest(e),'producer_sha256':'e'*64,
    'history_census_sha256':digest([]),'selection_sha256':'f'*64,'broker_bootstrap_sha256':self.boot['sha256'],'broker_history_sha256':'b'*64,
    'checked_at_unix':NOW,'lease_expires_unix':NOW+300,'global_deadline_unix':GLOBAL,'legacy_variant_workers_closed':True,
    'legacy_launch_controllers_closed':True,'active_xbabe2_source_kind':'native_s1_context','active_native_context_overlap':[],
    'other_variant_context_overlap':[],'all_legacy_units_physically_closed':True,'context_descriptors':e['context_descriptors']}
  root=Path(e['epoch_root']);put(root,'whole-context-registry-binding.json',{'epoch_sha256':digest(e),'config':config,'config_sha256':digest(config)})
  a=Attempts(root/'whole-context-attempts.sqlite',config,create=True)
  if owned:a.reserve_contexts(e['context_descriptors'],proof,'shard00')
  a.close();req=put(self.p,'workers-request.json',{'schema':'jope.prior-broker-worker-history-request/v3','inventory':inv,'canonicalizer':{}})
  def closed(u,clock):return {'unit':u,'systemd':{'MainPID':'0','ActiveState':'inactive'},'cgroups':[{'exists':False}]}
  return e,inv,req,closed
 def test_prior_reserved_missing_queue_refuses_all_new_qualification(self):
  e,inv,req,closed=self.worker_fixture(True)
  with self.assertRaisesRegex(ValueError,'queue missing'):bh.workers(req,self.p/'workers',hostname='xbabe3',native_factory=lambda *_:Native(),unit_reader=closed)
 def test_never_started_prior_contexts_are_still_whole_context_holds(self):
  e,inv,req,closed=self.worker_fixture(False);wr=bh.workers(req,self.p/'workers',hostname='xbabe3',native_factory=lambda *_:Native(),unit_reader=closed)
  h=put(self.p,'history.json',{'schema':'jope.permanent-broker-history/v3','bootstrap':self.boot,'inventory':inv,'workers':wr})
  _,aliases=bh.validate(h,self.boot,live=True,fixture_root=self.root)
  self.assertEqual(len(aliases),128);self.assertTrue(all(key[1]is None for key in aliases))
 def test_new_prior_queue_failure_is_included_beyond_original18(self):
  e,inv,req,closed=self.worker_fixture(True);s=e['shards'][0];p=Path(s['owned_root']);(p/'data').mkdir(parents=True)
  ctx=self.actual[s['context_ids'][0]];program='(ridge)';key=request_identity(ctx,program)
  q=Queue(p/'data/queue.sqlite');q.enqueue('gp-evaluate-'+key,'evaluate','a'*64,{'context':asdict(ctx),'program':program,'evaluation_key':key})
  j=q.claim('fixture',['evaluate'],lease_seconds=300,max_running=1);q.finish(j['id'],j['lease_token'],'b'*64,success=False,reason='native_failure');q.close()
  put(p,'status.json',{'state':'partial','numeric_failure_scores':0})
  wr=bh.workers(req,self.p/'workers',hostname='xbabe3',native_factory=lambda *_:Native(),unit_reader=closed)
  h=put(self.p,'history.json',{'schema':'jope.permanent-broker-history/v3','bootstrap':self.boot,'inventory':inv,'workers':wr})
  _,aliases=bh.validate(h,self.boot,live=True,fixture_root=self.root)
  import hashlib
  self.assertIn((ctx.identity,hashlib.sha256(program.encode()).hexdigest()),aliases)
  self.assertEqual(aliases[(ctx.identity,hashlib.sha256(program.encode()).hexdigest())][0]['states'][0]['state'],'failed')
if __name__=='__main__':unittest.main()
