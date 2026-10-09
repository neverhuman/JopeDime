import concurrent.futures,hashlib,io,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from verified_cache import VerifiedVersionCache,CachedS3Library,CachedNativeTools,ByteLRU,Entry
from jope_flywheel.program_io import NativeTools
from jope_flywheel.protocol import canonical
from jope_flywheel.s3_library import BUCKET,PREFIX

def sha(b):return hashlib.sha256(b).hexdigest()
def ref(body,key='a',version='v1'):
 return {'bucket':BUCKET,'key':PREFIX+key,'version_id':version,'bytes':len(body),'sha256':sha(body)}
class FakeClient:
 def __init__(self):self.objects={};self.gets=0;self.heads=0;self.puts=0
 def get_object(self,**kw):
  self.gets+=1;body=self.objects[kw['Key'],kw['VersionId']];return {'VersionId':kw['VersionId'],'ContentLength':len(body),'Body':io.BytesIO(body)}
 def head_object(self,**kw):
  self.heads+=1;raise PermissionError('no permission')
 def put_object(self,**kw):
  self.puts+=1;self.objects[kw['Key'],'v1']=kw['Body'];return {'VersionId':'v1'}
class VersionTests(unittest.TestCase):
 def setUp(self):self.calls=[];self.body=b'verified body';self.r=ref(self.body)
 def get(self,r):self.calls.append(dict(r));return self.body,r['version_id']
 def cache(self,**kw):return VerifiedVersionCache(self.get,check=lambda:None,**kw)
 def test_only_second_read_hits_complete_identity(self):
  c=self.cache();self.assertEqual(c.get_version(self.r),(self.body,'v1'));self.assertEqual(c.get_version(self.r),(self.body,'v1'));self.assertEqual(len(self.calls),1);s=c.snapshot();self.assertEqual((s['hits'],s['misses'],s['backend_bytes']),(1,1,len(self.body)));self.assertGreaterEqual(s['backend_seconds'],0)
 def test_bucket_key_version_sha_size_all_participate(self):
  c=self.cache();c.get_version(self.r)
  for field,value in [('bucket','other'),('key',PREFIX+'other'),('version_id','v2')]:c.get_version({**self.r,field:value})
  self.assertEqual(len(self.calls),4)
  for field,value in [('sha256','0'*64),('bytes',1)]:
   with self.assertRaises(ValueError):c.get_version({**self.r,field:value})
  self.assertEqual(c.snapshot()['entries'],4)
 def test_first_version_or_body_mismatch_never_cached(self):
  for answer in [(self.body,'wrong'),(b'poison','v1'),(bytearray(self.body),'v1')]:
   calls=[]
   def get(r):calls.append(1);return answer
   c=VerifiedVersionCache(get,check=lambda:None)
   for _ in range(2):
    with self.assertRaises(ValueError):c.get_version(self.r)
   self.assertEqual(len(calls),2);self.assertEqual(c.snapshot()['entries'],0)
 def test_absence_and_authorization_errors_not_cached(self):
  for error in [FileNotFoundError,PermissionError]:
   calls=[]
   def get(r):calls.append(1);raise error('fixture')
   c=VerifiedVersionCache(get,check=lambda:None)
   for _ in range(2):
    with self.assertRaises(error):c.get_version(self.r)
   self.assertEqual(len(calls),2);self.assertEqual(c.snapshot()['backend_failures'],2)
 def test_lru_eviction_and_oversize_bounds(self):
  c=self.cache(max_bytes=600,max_entries=1);c.get_version(self.r);c.get_version({**self.r,'key':PREFIX+'b'});self.assertEqual(c.snapshot()['entries'],1);c.get_version(self.r);self.assertEqual(len(self.calls),3);self.assertLessEqual(c.snapshot()['charged_bytes'],600)
  c=self.cache(max_bytes=10);c.get_version(self.r);c.get_version(self.r);self.assertEqual(c.snapshot()['entries'],0);self.assertEqual(c.snapshot()['uncached_oversize'],2)
 def test_corrupted_stored_body_fails_then_requires_new_get(self):
  c=self.cache();c.get_version(self.r);key=c.identity(self.r);old=c.cache._items[key];c.cache._items[key]=Entry(b'corrupt',old.sha256,old.charge)
  with self.assertRaisesRegex(ValueError,'corrupted'):c.get_version(self.r)
  self.assertEqual(c.snapshot()['entries'],0);c.get_version(self.r);self.assertEqual(len(self.calls),2)
 def test_deadline_on_hit_and_after_slow_get_no_insert(self):
  state={'fail':False}
  def check():
   if state['fail']:raise TimeoutError('deadline')
  c=VerifiedVersionCache(self.get,check=check);c.get_version(self.r);state['fail']=True
  with self.assertRaises(TimeoutError):c.get_version(self.r)
  state['fail']=False
  def late(r):state['fail']=True;return self.body,'v1'
  d=VerifiedVersionCache(late,check=check)
  with self.assertRaises(TimeoutError):d.get_version(self.r)
  self.assertEqual(d.snapshot()['entries'],0)
 def test_bad_refs_rejected_before_backend(self):
  for field,value in [('bytes',True),('bytes',0),('bytes',65*1024**2),('version_id','null'),('version_id',''),('sha256','x'*64),('bucket',''),('key','k'*4097)]:
   with self.assertRaises(ValueError):self.cache().get_version({**self.r,field:value})
  self.assertEqual(self.calls,[])
 def test_concurrent_verified_reads_preserve_budget(self):
  c=self.cache(max_bytes=600,max_entries=1)
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as ex:results=list(ex.map(lambda _:c.get_version(self.r),range(32)))
  self.assertTrue(all(r==(self.body,'v1')for r in results));self.assertLessEqual(c.snapshot()['charged_bytes'],600);self.assertEqual(c.snapshot()['entries'],1)
 def test_subclass_internal_put_readback_and_later_get_use_cache(self):
  client=FakeClient();s=CachedS3Library(client,deadline=time.time()+10);r=s.put(PREFIX+'object',self.body);self.assertEqual(client.gets,1);s.get_version(r);self.assertEqual(client.gets,1);self.assertEqual(s.cache_stats()['hits'],1)
 def test_subclass_lookup_still_checks_current_head_every_time(self):
  client=FakeClient();s=CachedS3Library(client,deadline=time.time()+10)
  for _ in range(2):
   with self.assertRaises(PermissionError):s.lookup('a'*64)
  self.assertEqual(client.heads,2)

class NativeTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name).resolve()/'canonicalizer';self.p.write_bytes(b'pinned executable fixture');self.ref={'path':str(self.p),'sha256':sha(self.p.read_bytes()),'bytes':self.p.stat().st_size};self.calls=[]
 def tearDown(self):self.temp.cleanup()
 def tool(self,**kw):return CachedNativeTools(self.ref.copy(),time.time()+30,**kw)
 def fake(self,mode,value):
  self.calls.append((mode,canonical(value)))
  if mode=='--canonical-json':return canonical(value)
  dsl=value['dsl'].strip();return canonical({'dsl':dsl,'program_sha256':sha(dsl.encode())})
 def test_json_and_program_preserve_types_and_exact_inputs(self):
  with patch.object(NativeTools,'call',side_effect=self.fake):
   t=self.tool();self.assertEqual(t.canonical_json({'a':1}),b'{"a":1}');self.assertEqual(t.canonical_json({'a':1}),b'{"a":1}');self.assertEqual(t.canonical_program(' X '),'X');self.assertEqual(t.canonical_program(' X '),'X');self.assertEqual(t.canonical_program('X'),'X');self.assertEqual(len(self.calls),3);self.assertEqual(t.cache_stats()['hits'],2)
 def test_json_changed_value_and_invalid_program_hash_not_cached(self):
  for mode in ['json','program']:
   bad=b'{"x":2}'if mode=='json'else canonical({'dsl':'X','program_sha256':'0'*64})
   with patch.object(NativeTools,'call',return_value=bad)as call:
    t=self.tool()
    for _ in range(2):
     with self.assertRaises(ValueError):t.canonical_json({'x':1})if mode=='json'else t.canonical_program('X')
    self.assertEqual(call.call_count,2);self.assertEqual(t.cache_stats()['entries'],0)
 def test_native_exception_not_cached(self):
  with patch.object(NativeTools,'call',side_effect=RuntimeError('native failed'))as call:
   t=self.tool()
   for _ in range(2):
    with self.assertRaises(RuntimeError):t.canonical_json({'x':1})
   self.assertEqual(call.call_count,2)
 def test_source_drift_refused_even_on_hit(self):
  with patch.object(NativeTools,'call',side_effect=self.fake):
   t=self.tool();t.canonical_program('X');self.p.write_bytes(b'changed')
   with self.assertRaises(ValueError):t.canonical_program('X')
   self.assertEqual(len(self.calls),1)
 def test_ref_repin_cannot_reuse_instance(self):
  t=self.tool();t.ref['sha256']='f'*64
  with self.assertRaisesRegex(ValueError,'identity changed'):t.canonical_json({})
 def test_stored_program_corruption_refused(self):
  with patch.object(NativeTools,'call',side_effect=self.fake):
   t=self.tool();t.canonical_program('X');k=next(iter(t.memo._items));old=t.memo._items[k];t.memo._items[k]=Entry(b'Y',old.sha256,old.charge)
   with self.assertRaisesRegex(ValueError,'corrupted'):t.canonical_program('X')
 def test_eviction_and_no_mode_collision(self):
  with patch.object(NativeTools,'call',side_effect=self.fake):
   t=self.tool(cache_entries=1);t.canonical_json({'dsl':'X'});t.canonical_program('X');t.canonical_json({'dsl':'X'});self.assertEqual(len(self.calls),3);self.assertEqual(t.cache_stats()['entries'],1)
 def test_expired_hit_and_invalid_nonfinite_input(self):
  with patch.object(NativeTools,'call',side_effect=self.fake):
   t=self.tool();t.canonical_json({});t.deadline=time.time()-1
   with self.assertRaises(TimeoutError):t.canonical_json({})
   t=self.tool()
   with self.assertRaises(ValueError):t.canonical_json({'x':float('nan')})
   self.assertEqual(len(self.calls),1)
if __name__=='__main__':unittest.main()
