"""Bounded local census fixtures; no systemd/SSH/GP/S3 calls."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import operations_census as C
from neural_outbox import Outbox,canonical,digest,sha
from test_native_successor import canonicalize

class CensusTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
 def test_physical_closed_unit_requires_pid0_and_empty_descendants(self):
  name='jopedime-fixture.service';text='Id='+name+'\nLoadState=loaded\nActiveState=inactive\nMainPID=0\nControlGroup=/fixture\n'
  cg=self.root/'fixture';cg.mkdir();(cg/'cgroup.procs').write_text('');(cg/'cgroup.events').write_text('populated 0\n')
  C.closed_unit(name,time.time()+30,runner=lambda *_:text,cgroup_root=self.root)
  for bad in [text.replace('MainPID=0','MainPID=123'),text.replace('inactive','active')]:
   with self.assertRaises(ValueError):C.closed_unit(name,time.time()+30,runner=lambda *_:bad,cgroup_root=self.root)
  (cg/'cgroup.procs').write_text('123\n')
  with self.assertRaises(ValueError):C.closed_unit(name,time.time()+30,runner=lambda *_:text,cgroup_root=self.root)
 def test_never_loaded_unit_has_no_cgroup_not_active_claim(self):
  text='Id=jopedime-held.service\nLoadState=not-found\nActiveState=inactive\nMainPID=0\nControlGroup=\n'
  result=C.closed_unit('jopedime-held',time.time()+30,runner=lambda *_:text,cgroup_root=self.root)
  self.assertFalse(result['cgroups'][0]['exists'])
 def test_discovery_only_exact_roots_and_rejects_unexpected_worker(self):
  r=self.root/'v1';r.mkdir();(r/'worker0').mkdir();(r/'worker0/config.json').write_text('{}')
  with patch.object(C,'ROOTS',[str(r)]):
   self.assertEqual(C.discover([str(r)])[0]['configs'],['worker0/config.json'])
   (r/'worker8').mkdir();(r/'worker8/config.json').write_text('{}')
   with self.assertRaises(ValueError):C.discover([str(r)])
  with self.assertRaises(ValueError):C.discover([str(self.root)])
 def emit(self,box,cid='1'*64):
  b=dict(batch_id=digest(cid),campaign_sha256='b'*64,checkpoint_sha256='c'*64,global_step=1,manifest_sha256='d'*64,
         train_data_run_ids=[cid],validation_data_run_ids=[],emitted_train_data_run_ids=[cid])
  e=dict(phase='training',data_run_id=cid,context_id=cid,parent_lineage_sha256='2'*64,
   input_descriptor={'source_kind':'native_s1_context','context_id':cid,'parent_lineage_sha256':'2'*64},raw_program_utf8=' (ridge) ',raw_program_sha256=sha(b' (ridge) '),
   decoder_seed=4,decoder_protocol_sha256='3'*64,decode_artifact_sha256='4'*64)
  box.record_batch(b,[e])
 def test_summary_preserves_unknown_without_fabricating_evalkey_and_no_dbwrites(self):
  p=self.root/'box.sqlite';o=Outbox(p,model_origin_sha256='a'*64);self.addCleanup(o.close);self.emit(o)
  cl=o.claim('fixture',lease_seconds=1,deadline=time.time()+20)
  o.finish(cl['lease'],dict(state='unknown_attempt',evaluation_key=None,record_ref=None,attempt_custody_ref=None,error='old unknown'),verify_outcome=lambda *_:True)
  before={str(x):sha(x.read_bytes())for x in self.root.glob('box.sqlite*')if not str(x).endswith('-shm')}
  v=C.summarize_source({'path':str(p),'kind':'outbox','config':o.config},time.time()+30,canonicalize)
  self.assertEqual(v['aliases'][0]['disposition'],'blocked');self.assertIsNone(v['aliases'][0]['evaluation_key'])
  self.assertEqual(before,{str(x):sha(x.read_bytes())for x in self.root.glob('box.sqlite*')if not str(x).endswith('-shm')})
 def collections(self,original_count=0):
  refs=[];roots=[str(self.root/('native-v%d'%i))for i in range(1,11)]
  mock=patch.object(C,'ROOTS',roots);mock.start();self.addCleanup(mock.stop)
  qs=sorted(str(self.root/('queue%d.sqlite'%i))for i in range(18))
  hist={'schema':'jope.variant-historical-census/v1','source_queues':[{'path':p,'host':'xbabe3','completed_evaluation_keys':[],'blocked_evaluation_keys':['3'*64]}for p in qs]}
  histref=C.put(self.root,'historical.json',hist);canon=C.put(self.root,'canonicalizer.json',{'binary_fixture':True})
  for host in ('xbabe2','xbabe3'):
   refsources=[];configs=[];rinv=[]
   if host=='xbabe2':
    for i,r in enumerate(roots):
     names=['config.json']if i<2 else[];rinv.append({'path':r,'exists':bool(names),'configs':names})
     if names:
      Path(r).mkdir();configs.append(C.put(Path(r),'config.json',{'outbox_path':str(self.root/(['original','continuation'][i]+'.sqlite')),'outbox_config':{'schema':'fixture'}}))
   paths=[str(self.root/(n+'.sqlite'))for n in ['original','continuation']]if host=='xbabe2'else qs
   for i,path in enumerate(paths):
    kind='outbox'if host=='xbabe2'else'queue';src={'path':path,'kind':kind,'config':None if kind=='queue'else{'schema':'fixture'}}
    alias={'context_id':'1'*64,'canonical_program_sha256':'2'*64,'evaluation_key':'3'*64 if host=='xbabe3'else None,'disposition':'blocked','source':src['path']}
    count=original_count if path.endswith('original.sqlite')else 1
    states={'unknown_attempt':1,'recorded':count}if kind=='outbox'else{'needs_reconcile':1}
    proof={'events':3,'head_sha256':'7'*64}
    if kind=='outbox':proof.update(schema='jope.flywheel-neural-outbox-status/v1',states=states,observations=sum(states.values()))
    v={'source':src,'physical':{'device':1,'inode':i},'proof':proof,'states':states,'recorded_native':count if kind=='outbox' else 0,'aliases':[alias]}
    refsources.append(C.put(self.root,host+'-source'+str(i)+'.json',v))
   unit='jopedime-'+host+'.service';u={'unit':unit,'systemd':{'Id':unit,'LoadState':'loaded','ActiveState':'inactive','MainPID':'0'},
       'cgroups':[{'path':'/sys/fs/cgroup/fixture/'+unit,'exists':False}],'checked_at_unix':time.time()}
   inv={'schema':'jope.native-v5-census-inventory/v1','host':host,'native_roots':rinv,'queues':qs if host=='xbabe3'else[],
        'legacy_units':[unit],'canonicalizer':canon,'historical_census':histref,'extra_evidence':[],'maximum_seconds':600}
   invref=C.put(self.root,host+'-inventory.json',inv)
   c={'schema':'jope.native-v5-host-census/v1','host':host,'checked_at_unix':time.time(),'inventory':invref,'canonicalizer_sha256':canon['sha256'],
      'root_inventory':rinv,'config_refs':configs,'sources':refsources,'units_before':[u],
      'units_after':[u],'historical_census':histref,'extra_evidence':[],'read_only':True,'no_execution_or_retries':True}
   refs.append(C.put(self.root,host+'.json',c))
  return refs
 def test_prepare_preserves_unknown_local_and_remote_blocked_refs(self):
  refs=self.collections();result=C.prepare(refs,self.root/'out',original_outbox=str(self.root/'original.sqlite'),continuation_outbox=str(self.root/'continuation.sqlite'))
  p=C.read(result);c=C.read(p['census']);self.assertEqual(len(p['sources']),2);self.assertEqual(len(c['entries']),18)
  self.assertIn(c['entries'][0]['source'],[r['path']for r in c['closure_evidence']])
  self.assertEqual(len(json.loads((self.root/'out/unknown-native-keys.json').read_bytes())['aliases']),2)
 def test_prepare_refuses_nonempty_original_before_worker_plan(self):
  refs=self.collections(original_count=1)
  with self.assertRaisesRegex(ValueError,'not empty'):C.prepare(refs,self.root/'out',original_outbox=str(self.root/'original.sqlite'),continuation_outbox=str(self.root/'continuation.sqlite'))
 def test_prepare_rejects_partial_pinned_remote_inventory_and_unit_proof(self):
  refs=self.collections();remote=C.read(refs[1])
  for i,change in enumerate([{'inventory':{}},{'sources':[]},{'units_after':[{'unit':'jopedime-xbabe3.service'}]},
                            {'historical_census':{'sha256':'5'*64}}]):
   bad=C.put(self.root,'bad%d.json'%i,{**remote,**change})
   with self.assertRaises((ValueError,KeyError)):
    C.prepare([refs[0],bad],self.root/('badout%d'%i),original_outbox=str(self.root/'original.sqlite'),continuation_outbox=str(self.root/'continuation.sqlite'))

 def test_real_collect_prepare_guard_chain_preserves_actual_unknown(self):
  from jope_flywheel.queue import Queue
  from native_global_guard import GlobalGuard,BlockedKey,GLOBAL_DEADLINE
  roots=[str(self.root/('real-v%d'%i))for i in range(10)]
  patcher=patch.object(C,'ROOTS',roots);patcher.start();self.addCleanup(patcher.stop)
  original=Outbox(self.root/'original.sqlite',model_origin_sha256='a'*64);self.addCleanup(original.close)
  current=Outbox(self.root/'continuation.sqlite',model_origin_sha256='a'*64);self.addCleanup(current.close);self.emit(current)
  claim=current.claim('old',lease_seconds=1,deadline=time.time()+10)
  current.finish(claim['lease'],dict(state='unknown_attempt',evaluation_key=None,record_ref=None,attempt_custody_ref=None,error='uncertain'),verify_outcome=lambda *_:True)
  rinv=[]
  for i,path in enumerate(roots):
   names=['config.json']if i<2 else[];rinv.append({'path':path,'exists':bool(names),'configs':names})
   if names:
    Path(path).mkdir();box=[original,current][i];C.put(Path(path),'config.json',{'outbox_path':str(self.root/(['original','continuation'][i]+'.sqlite')),'outbox_config':box.config})
  qs=sorted(str(self.root/('real-queue%d.sqlite'%i))for i in range(18))
  for path in qs:
   q=Queue(path);q.close()
  hist=C.put(self.root,'real-hist.json',{'schema':'jope.variant-historical-census/v1','source_queues':[{'host':'xbabe3','path':q,'completed_evaluation_keys':[],'blocked_evaluation_keys':[]}for q in qs]})
  canon=C.put(self.root,'fixture-binary.json',{'fixture_only':True})
  def units(name,deadline):
   name=name if name.endswith('.service')else name+'.service'
   return {'unit':name,'systemd':{'Id':name,'LoadState':'not-found','ActiveState':'inactive','MainPID':'0'},'cgroups':[{'path':'/sys/fs/cgroup/fixture/'+name,'exists':False}],'checked_at_unix':time.time()}
  class Tools:
   def __init__(self,*args):pass
   canonical_program=staticmethod(canonicalize)
  receipts=[]
  for host in ('xbabe2','xbabe3'):
   inv={'schema':'jope.native-v5-census-inventory/v1','host':host,'native_roots':rinv if host=='xbabe2'else[],
        'queues':qs if host=='xbabe3'else[],'legacy_units':['jopedime-fixture'],'canonicalizer':canon,'historical_census':hist,'extra_evidence':[],'maximum_seconds':30}
   ref=C.put(self.root,host+'-real-inv.json',inv)
   receipts.append(C.collect(ref,self.root/(host+'-collected'),hostname=host,unit_reader=units,tools_factory=Tools))
  prep=C.read(C.prepare(receipts,self.root/'prepared',original_outbox=str(self.root/'original.sqlite'),continuation_outbox=str(self.root/'continuation.sqlite')))
  cfg={'schema':'jope.native-global-execution-guard/v1','registry_path':str(self.root/'guard.sqlite'),'scope_sha256':prep['scope_sha256'],
       'canonicalizer_sha256':prep['canonicalizer_sha256'],'global_deadline_unix':GLOBAL_DEADLINE,'sources':prep['sources'],'census':prep['census'],'max_snapshot_bytes':1024**2}
  guard=GlobalGuard(cfg,canonicalize=canonicalize,packet_sha256='e'*64,deadline=time.time()+30);self.addCleanup(guard.close)
  with self.assertRaises(BlockedKey):guard.check('1'*64,'(ridge)')
  self.assertEqual(guard.verify()['claims'],0)

 def test_exactref_rejects_tamper_and_createonly_output(self):
  r=C.put(self.root,'proof.json',{'proof':'x'});Path(r['path']).write_text('{}')
  with self.assertRaises(ValueError):C.read(r)
  with self.assertRaises(FileExistsError):C.put(self.root,'proof.json',{})
 def test_oversized_pin_refused_before_read(self):
  p=self.root/'large';p.write_bytes(b'x')
  with patch.object(C,'CAP',0):
   with self.assertRaises(ValueError):C.pin(p)

if __name__=='__main__':unittest.main()
