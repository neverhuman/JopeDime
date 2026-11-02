"""Local CPU-only contracts. No SSH, S3, CUDA, or production DB access."""
import ast,copy,hashlib,importlib.util,json,sqlite3,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import campaign as c
import kumo_queue as q
import kumo_controller as controller
import kumo_publish as publisher
import kumo_intake as intake
HERE=Path(__file__).resolve().parent

def module(name):
 spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
worker=module('evergreen_kumos_packet')

class QueueCase(unittest.TestCase):
 def setUp(self):
  self.db=sqlite3.connect(':memory:',isolation_level=None);self.db.row_factory=sqlite3.Row
  snapshot=json.loads((HERE/'selection-readonly.json').read_bytes());self.selection=json.loads((HERE/'selection.json').read_bytes());self.rows=snapshot['rows'];self.pubs=snapshot['publications'];q.ensure(self.db)
  for table,rows in [('variants',self.rows),('flywheel_s3_publications_v1',self.pubs)]:
   fields=list(rows[0]);self.db.execute('CREATE TABLE '+table+'('+','.join('"'+f+'" '+('INTEGER'if f in('sequence','ordinal')else 'REAL'if f=='created'else'TEXT')for f in fields)+')')
   self.db.executemany('INSERT INTO '+table+' VALUES('+','.join('?'for _ in fields)+')',[[r[f]for f in fields]for r in rows])
  self.deadline=time.time()+120
  self.db.execute('INSERT INTO '+q.PACKETS+' VALUES(?,?,?,?,?,?,?)',('a'*64,'xbabe1',c.PIPELINE_SHA,1,2,'claimed','held failure untouched'))
  self.db.execute('INSERT INTO '+q.ITEMS+' VALUES(?,?,?,?,?)',('b'*64,'a'*64,874,'old-row','old-publication'))
  self.db.execute('INSERT INTO '+q.STAGES+' VALUES(?,?,?,?,?)',('d'*64,q.STAGE,'old-receipt','e'*64,1))
 def tearDown(self):self.db.close()
 def claim(self,**kw):
  args=dict(db=self.db,pipeline_sha=c.PIPELINE_SHA,packet_id=c.PACKET_ID,count=32,deadline=self.deadline,selection=self.selection,role_check=lambda row:True);args.update(kw);return q.claim(**args)
 def test_fixed32_and_old_held_rows_preserved(self):
  r=self.claim();self.assertEqual(r['state'],'claimed');self.assertEqual(len(r['rows']),32);self.assertEqual(r['observed_packet_rows_before'],1)
  self.assertEqual(self.db.execute('SELECT state,proof_json FROM '+q.PACKETS+' WHERE packet_id=?',('a'*64,)).fetchone()[:],('claimed','held failure untouched'))
  self.assertEqual(self.db.execute('SELECT row_json FROM '+q.ITEMS+' WHERE sequence=874').fetchone()[0],'old-row')
  self.assertEqual(self.db.execute('SELECT receipt_json FROM '+q.STAGES).fetchone()[0],'old-receipt')
 def test_reinvoke_never_resets_campaign(self):
  self.claim();self.assertEqual(self.claim()['state'],'packet_already_exists');self.assertEqual(self.db.execute('SELECT count(*) FROM '+q.ITEMS).fetchone()[0],33)
 def test_other_xb3_claim_explicit_blocked(self):
  self.db.execute('INSERT INTO '+q.PACKETS+' VALUES(?,?,?,?,?,?,NULL)',('f'*64,'xbabe3',c.PIPELINE_SHA,1,2,'claimed'))
  self.assertEqual(self.claim()['state'],'blocked_claim')
 def test_cross_host_item_blocks_no_partial_claim(self):
  r=self.rows[0];self.db.execute('INSERT INTO '+q.ITEMS+' VALUES(?,?,?,?,?)',(r['identity'],'a'*64,r['sequence'],'x','x'))
  self.assertEqual(self.claim()['state'],'blocked_claim');self.assertEqual(self.db.execute('SELECT count(*) FROM '+q.PACKETS).fetchone()[0],1)
 def test_existing_stage_blocks_no_recompute(self):
  self.db.execute('INSERT INTO '+q.STAGES+' VALUES(?,?,?,?,?)',(self.rows[0]['identity'],q.STAGE,'x','e'*64,1));self.assertEqual(self.claim()['state'],'blocked_claim')
 def test_missing_selected_is_empty_not_blocked(self):
  self.db.execute('DELETE FROM variants WHERE identity=?',(self.rows[0]['identity'],));self.assertEqual(self.claim()['state'],'empty')
 def test_protected_role_refuses_before_claim(self):
  with self.assertRaisesRegex(ValueError,'protected'):self.claim(role_check=lambda row:False)
  self.assertEqual(self.db.execute('SELECT count(*) FROM '+q.PACKETS).fetchone()[0],1)
 def test_source_row_drift_refused(self):
  self.db.execute('UPDATE variants SET content_sha=? WHERE identity=?',('0'*64,self.rows[0]['identity']))
  with self.assertRaisesRegex(ValueError,'source row'):self.claim()
 def test_publisher_provenance_refused(self):
  pub=self.pubs[0];b=json.loads(pub['publication_json']);b['publication_helper_sha256']='0'*64;raw=q.canonical(b).decode();self.db.execute('UPDATE '+q.PUB+' SET publication_json=?,publication_sha=? WHERE identity=?',(raw,q.sha(raw.encode()),pub['identity']))
  with self.assertRaisesRegex(ValueError,'qualified pipeline'):self.claim()
 def test_additional_all_history_safety_ceiling_includes_failures(self):
  for i in range(127):self.db.execute('INSERT INTO '+q.PACKETS+' VALUES(?,?,?,?,?,?,NULL)',(hashlib.sha256(str(i).encode()).hexdigest(),'xbabe1',c.PIPELINE_SHA,1,2,'needs_reconcile'))
  with self.assertRaisesRegex(ValueError,'all-history'):self.claim()
 def test_expired_or_extended_window(self):
  for d in (time.time()-1,time.time()+1801,c.GLOBAL_DEADLINE+1):
   with self.subTest(d=d),self.assertRaisesRegex(ValueError,'window'):self.claim(deadline=d)
 def test_hard_count_and_identity(self):
  for override in ({'count':31},{'count':33},{'packet_id':'0'*64},{'pipeline_sha':'0'*64}):
   with self.subTest(override=override),self.assertRaises(ValueError):self.claim(**override)
 def test_selection_duplicate_or_replaced_refused(self):
  changed=copy.deepcopy(self.selection);changed['rows'][0]['parent']=changed['rows'][1]['parent']
  with self.assertRaisesRegex(ValueError,'unique'):self.claim(selection=changed)
 def test_failure_cannot_touch_old_packet(self):
  with self.assertRaisesRegex(ValueError,'another'):q.fail(self.db,'a'*64,'test')
 def test_commit_whole_support_only_and_preserve_prior(self):
  self.claim();rows=[{'variant_identity':r['identity'],'content_sha256':r['content_sha'],'metadata_sha256':r['metadata_sha'],'required_encoders':['kumo_tabular_s'],'new_gp_labels':0,'teaching_eligible':False}for r in self.rows]
  with self.assertRaisesRegex(ValueError,'support'):q.finish(self.db,c.PACKET_ID,rows[:-1],lambda rs:True,self.deadline)
  q.finish(self.db,c.PACKET_ID,rows,lambda rs:True,self.deadline)
  self.assertEqual(self.db.execute('SELECT count(*) FROM '+q.STAGES).fetchone()[0],33)
  self.assertEqual(self.db.execute('SELECT state FROM '+q.PACKETS+' WHERE packet_id=?',(c.PACKET_ID,)).fetchone()[0],'complete')
 def test_unverified_finish_refused(self):
  self.claim()
  with self.assertRaisesRegex(ValueError,'unverified'):q.finish(self.db,c.PACKET_ID,[],lambda rs:False,self.deadline)

class WorkerContracts(unittest.TestCase):
 def test_only_three_operational_functions_changed(self):
  base=(HERE/'generic_worker_dfba7d52.py').read_bytes();self.assertEqual(q.sha(base),c.WORKER_BASE_SHA)
  def nodes(raw):return {getattr(n,'name',str(i)):ast.dump(n,include_attributes=False)for i,n in enumerate(ast.parse(raw).body)}
  a,b=nodes(base),nodes((HERE/'evergreen_kumos_packet.py').read_bytes());self.assertEqual(set(a),set(b));self.assertEqual({k for k in a if a[k]!=b[k]},{'coverage','load_inputs','owned_size'})
 def test_progress_atomic_rename_race_is_tolerated_only_at_roots(self):
  with tempfile.TemporaryDirectory()as td:
   root=Path(td).resolve();temp=root/'progress.json.tmp';temp.write_bytes(b'complete');artifact=root/'vector.npy';artifact.write_bytes(b'scientific')
   def raced_walk(*a,**kw):
    listed=['progress.json.tmp','vector.npy'];temp.rename(root/'progress.json');yield str(root),[],listed
   with patch.object(worker.os,'walk',raced_walk):self.assertEqual(worker.owned_size(root),len(b'scientific'))
 def test_missing_vector_or_nested_progress_is_not_swallowed(self):
  with tempfile.TemporaryDirectory()as td:
   root=Path(td).resolve()
   for directory,name in [(root,'vector.npy'),(root/'variants','progress.json.tmp')]:
    directory.mkdir(exist_ok=True)
    with self.subTest(name=name),patch.object(worker.os,'walk',return_value=[(str(directory),[],[name])]),self.assertRaises(FileNotFoundError):worker.owned_size(root)
 def test_progress_symlink_is_not_swallowed(self):
  with tempfile.TemporaryDirectory()as td:
   root=Path(td).resolve();target=root/'target';target.write_bytes(b'x');(root/'progress.json.tmp').symlink_to(target)
   with self.assertRaisesRegex(ValueError,'symlink'):worker.owned_size(root)
 def test_owned_size_is_exact_previously_qualified_production2a(self):
  old=HERE/'provenance/production2a_worker_cedc828f.py'
  def body(path):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='owned_size'),include_attributes=False)
  self.assertEqual(body(old),body(HERE/'evergreen_kumos_packet.py'))
 def test_runtime_pin_exact(self):self.assertEqual(q.sha((HERE/'runtime_kumos_xbabe3.json').read_bytes()),c.RUNTIME_SHA)
 def test_full_qualification_still_four_calls_six_tests(self):
  import numpy as np
  class Enc:
   def __init__(self):self.calls=0
   def one(self,t,d):
    self.calls+=1;v=np.zeros(2051,dtype='<f4');v[0]=float(d['y'][0]);return v
  class Helper:
   @staticmethod
   def prep_equal(a,b):return True
   @staticmethod
   def vector_ok(v,t):return v,{},[1]
  class Core:
   @staticmethod
   def prep(c,raw):return {'y':np.arange(50,dtype=float)}
  enc=Enc()
  with patch.object(worker,'decode',side_effect=[(np.zeros((50,1)),np.arange(50.),None),ValueError('protected')]),patch.object(worker,'protected',return_value=b'protected'):
   vec,proof=worker.qualification_ofm(Helper,Core,enc,'kumo_tabular_s',{'raw':b'x','identity':'a'*64})
  self.assertEqual(enc.calls,4);self.assertEqual(len(proof['tests']),6);self.assertTrue(all(proof['tests'].values()))
 def test_outer_test_refused_real_decoder(self):
  import struct,numpy as np
  a=np.column_stack([np.arange(50.),np.arange(50.)+1]);fold=np.array([0]*25+[1]*25,dtype='uint8')
  raw=b'JTT1'+struct.pack('<IIIB',50,1,1,2)+fold.tobytes()+a.astype('<f8').tobytes()+np.zeros((1,2),dtype='<f8').tobytes()
  with self.assertRaisesRegex(ValueError,'protected'):worker.decode(raw)
 def test_checked_wrong_sha_symlink_refused(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t).resolve()/'source';p.write_bytes(b'abc')
   with self.assertRaisesRegex(ValueError,'SHA'):worker.checked({'path':str(p),'sha256':'0'*64})
   l=Path(t).resolve()/'link';l.symlink_to(p)
   with self.assertRaisesRegex(ValueError,'nonsymlink'):worker.checked({'path':str(l),'sha256':q.sha(b'abc')})
 def test_manifest_runtime_source_host_window_gate(self):
  m=json.loads((HERE/'template.json').read_bytes());m.update(created_unix=time.time()-1,deadline_unix=time.time()+60)
  with patch.object(worker,'safe',side_effect=lambda x:Path(x)),patch.object(worker,'coverage',return_value={}):
   worker.manifest_ok(m,'xbabe3')
   for key,val in [('deadline_unix',time.time()-1),('components',{'hyperion_v6all':512}),('generator_config',{'sha256':'0'*64}),('parent_registry',{'sha256':'0'*64})]:
    bad=copy.deepcopy(m);bad[key]=val
    with self.subTest(key=key),self.assertRaises(ValueError):worker.manifest_ok(bad,'xbabe3')
 def test_positive_publication_contract_and_wrong_publisher(self):
  # Directly exercise the actual new growth publisher gate before payload checks.
  m={'total_variants':32};pub={'schema':'jope.flywheel-generated-packet-publication-complete/v1','state':'complete','new_gp_labels':0,'teaching_eligible':False,'pipeline_config_sha256':c.PIPELINE_SHA,'publisher_sha256':'0'*64}
  m['dataset_publication']={}
  with patch.object(worker,'obj',return_value=pub),self.assertRaisesRegex(ValueError,'publisher'):worker.coverage(m)

class OperationalContracts(unittest.TestCase):
 def test_template_not_launchable_and_runtime_drift_refused(self):
  conf=json.loads((HERE/'config.template.json').read_bytes())
  with self.assertRaisesRegex(ValueError,'approved'):controller.validate_config(conf,'0'*64)
  conf.update(approved_for_launch=True,created_unix=time.time(),deadline_unix=time.time()+120)
  conf['runtime']['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'runtime'):controller.validate_config(conf,'0'*64)
 def test_controller_excludes_old_max_packets_loop(self):
  tree=ast.parse((HERE/'kumo_controller.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');self.assertFalse(any(isinstance(n,ast.While)for n in ast.walk(main)))
  self.assertNotIn('--max-packets',(HERE/'kumo_controller.py').read_text())
 def test_publication_rejects_wrong_host_source_deadline(self):
  now=time.time();m={'packet_id':c.PACKET_ID,'total_variants':32,'worker_path':c.ROOT+'/evergreen_kumos_packet.py','owned_root':c.ROOT,'worker_sha256':publisher.ENCODER_SHA,'created_unix':now,'deadline_unix':now+1200,'queue_deadline_unix':now+1800,'hosts':{'xbabe3':{'out':c.ROOT+'/runs/xbabe3'}}}
  conf={'host':'xbabe3','encoder_worker':{'path':m['worker_path'],'sha256':publisher.ENCODER_SHA},'encoder_manifest':{'path':c.ROOT+'/manifest.json'},'deadline_unix':time.time()+1600}
  self.assertEqual(publisher.source_release(conf,m)['source_sha256'],publisher.ENCODER_SHA)
  for f,value in [('worker_sha256','0'*64),('packet_id','0'*64),('queue_deadline_unix',c.GLOBAL_DEADLINE+1)]:
   bad=copy.deepcopy(m);bad[f]=value
   with self.subTest(f=f),self.assertRaises(ValueError):publisher.source_release(conf,bad)
 def test_systemd_bounds_no_gpu_for_io(self):
  cmd=controller.bounded('xbabe3','test','/x',[],120,gpu=True);self.assertIn('CPUQuota=800%',cmd);self.assertIn('MemoryMax=12G',cmd);self.assertIn('MemorySwapMax=0',cmd)
  io=controller.bounded('xbabe2','test','/x',[],120);self.assertIn('--setenv=CUDA_VISIBLE_DEVICES=',io)
 def test_intake_source_verification_before_claim(self):
  tree=ast.parse((HERE/'kumo_intake.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='prepare');text=ast.unparse(fn);self.assertLess(text.index('publisher.check_authority'),text.index('queue.claim'));self.assertLess(text.index('queue.claim'),text.index('root.mkdir'))

class RemoteInvocationCases(unittest.TestCase):
 def test_copy_python_c_runs_without_campaign_in_cwd(self):
  import subprocess
  with tempfile.TemporaryDirectory()as td:
   root=Path(td).resolve()/'owned-runtime'
   def offline_ssh(host,args,body=None,timeout=60):
    self.assertEqual(host,'xbabe3');self.assertEqual(args[:3],['python3','-B','-c'])
    result=subprocess.run(args,input=body,capture_output=True,cwd=Path(td).resolve(),timeout=5)
    self.assertEqual(result.returncode,0,result.stderr.decode());return result.stdout
   with patch.object(controller,'ssh',offline_ssh):receipt=controller.copy('xbabe3',str(root),{'campaign.py':b'# owned source\n','nested/source.py':b'value=3\n'})
   self.assertEqual(receipt['nested/source.py']['sha256'],q.sha(b'value=3\n'))
   with patch.object(controller,'ssh',offline_ssh):controller.copy('xbabe3',str(root),{'campaign.py':b'# owned source\n'})
 def test_all_remote_python_c_scripts_compile(self):
  tree=ast.parse((HERE/'kumo_controller.py').read_text());scripts=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='script'for x in n.targets)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str):
    compile(n.value.value,'remote-python-c','exec');self.assertNotIn('from campaign import',n.value.value);scripts.append(n.value.value)
  self.assertEqual(len(scripts),3)

class BoundaryCases(unittest.TestCase):
 def test_actual_database_config_identity_not_generator_file_sha(self):
  with tempfile.TemporaryDirectory()as d:
   path=Path(d).resolve()/'expansion.sqlite';db=sqlite3.connect(path);db.execute('CREATE TABLE config(id INTEGER,sha TEXT)');db.execute('INSERT INTO config VALUES(1,?)',(c.DB_CONFIG_SHA,));db.commit();db.close()
   q.connect(path,c.DB_CONFIG_SHA).close()
   with self.assertRaisesRegex(ValueError,'config drift'):q.connect(path,q.CONFIG)
   Path(str(path)+'-wal').symlink_to(path)
   with self.assertRaisesRegex(ValueError,'sidecar'):q.connect(path,c.DB_CONFIG_SHA)
 def test_exact_version_sha_size_reader(self):
  import io
  class Client:
   def __init__(self,version='v1',body=b'abc'):self.version=version;self.body=body
   def get_object(self,**kw):return {'VersionId':self.version,'ContentLength':len(self.body),'Body':io.BytesIO(self.body)}
  r={'bucket':'veox-jopedime-learning-use2','key':'experiments/gp-policy-dl/v2-trainonly/x','version_id':'v1','verification':'exact VersionId GET + SHA256','bytes':3,'sha256':q.sha(b'abc')}
  self.assertEqual(intake.get(Client(),r,time.time()+2),b'abc')
  for client in (Client('v2'),Client(body=b'abd'),Client(body=b'abcx')):
   with self.assertRaises(ValueError):intake.get(client,r,time.time()+2)
 def test_parent_source_or_target_changed_refused_before_cas(self):
  qcase=QueueCase();qcase.setUp()
  try:
   qcase.claim();qcase.db.execute('UPDATE variants SET metadata_sha=? WHERE identity=?',('0'*64,qcase.rows[0]['identity']))
   rows=[{'variant_identity':r['identity'],'content_sha256':r['content_sha'],'metadata_sha256':r['metadata_sha'],'required_encoders':['kumo_tabular_s'],'new_gp_labels':0,'teaching_eligible':False}for r in qcase.rows]
   with self.assertRaisesRegex(ValueError,'source variant changed'):q.finish(qcase.db,c.PACKET_ID,rows,lambda rs:True,qcase.deadline)
   self.assertEqual(qcase.db.execute('SELECT count(*) FROM '+q.STAGES).fetchone()[0],1)
  finally:qcase.tearDown()

if __name__=='__main__':unittest.main()
